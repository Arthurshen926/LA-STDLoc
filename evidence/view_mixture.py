"""Mapping-only view-mixture descriptor diagnostics for Projective Anchors.

The routines in this module are deliberately audit-only.  They never alter
Anchor identity or geometry and every Anchor still owns exactly one potential
PnP vote.  A second descriptor prototype is only a different appearance model
for that same identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from evidence.tracks import LeaveOneQueryOutProjectiveAnchorDescriptorBank
from localization.matcher import global_view_mixture_topk


@dataclass(frozen=True)
class ViewMixture:
    prototypes: torch.Tensor
    priors: torch.Tensor
    view_bin_count: int
    observation_count: int
    angle_degrees: float
    loss_improvement: float
    eligible: bool


def _weighted_mean(descriptors: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return F.normalize(
        (descriptors * weights.clamp_min(1e-8)[:, None]).sum(dim=0), dim=0
    )


def build_view_mixture(
    descriptors: torch.Tensor,
    view_bins: torch.Tensor,
    weights: torch.Tensor,
    *,
    minimum_cluster_observations: int = 2,
    minimum_cluster_view_bins: int = 2,
    minimum_angle_degrees: float = 12.0,
    minimum_loss_improvement: float = 0.015,
) -> ViewMixture:
    """Fit one deterministic two-component spherical mixture to view bins.

    View bins, rather than individual observations, are clustered.  This keeps
    a densely sampled camera neighbourhood from manufacturing a second mode.
    The farthest pair initializes the two centers; ties are resolved by sorted
    view-bin order and assignment ties go to component zero.
    """
    descriptors = F.normalize(torch.as_tensor(descriptors).float(), dim=1)
    view_bins = torch.as_tensor(view_bins).long().reshape(-1)
    weights = torch.as_tensor(weights).float().reshape(-1).clamp_min(1e-8)
    if descriptors.ndim != 2 or descriptors.shape[0] == 0:
        raise ValueError("descriptors must be a non-empty matrix")
    if view_bins.numel() != descriptors.shape[0] or weights.numel() != descriptors.shape[0]:
        raise ValueError("view-mixture metadata does not align")

    unique_bins = torch.unique(view_bins, sorted=True)
    bin_prototypes = torch.stack(
        [_weighted_mean(descriptors[view_bins == value], weights[view_bins == value])
         for value in unique_bins.tolist()]
    )
    bin_weights = torch.stack(
        [weights[view_bins == value].sum() for value in unique_bins.tolist()]
    )
    single = _weighted_mean(descriptors, weights)
    if unique_bins.numel() < 2:
        return ViewMixture(single[None], torch.ones(1), int(unique_bins.numel()),
                           int(descriptors.shape[0]), 0.0, 0.0, False)

    similarity = bin_prototypes @ bin_prototypes.T
    upper = torch.triu(torch.ones_like(similarity, dtype=torch.bool), diagonal=1)
    pair_score = torch.where(upper, similarity, torch.full_like(similarity, float("inf")))
    flat = int(pair_score.reshape(-1).argmin())
    first, second = divmod(flat, int(unique_bins.numel()))
    centers = bin_prototypes[torch.tensor([first, second])]
    assignment = (bin_prototypes @ centers.T)[:, 1] > (bin_prototypes @ centers.T)[:, 0]
    # One deterministic Lloyd update is enough for the small, already
    # quantized view-bin set and avoids convergence-dependent schemas.
    for _ in range(2):
        if bool(assignment.all()) or bool((~assignment).all()):
            break
        centers = torch.stack([
            _weighted_mean(bin_prototypes[~assignment], bin_weights[~assignment]),
            _weighted_mean(bin_prototypes[assignment], bin_weights[assignment]),
        ])
        new_assignment = (bin_prototypes @ centers.T)[:, 1] > (bin_prototypes @ centers.T)[:, 0]
        if torch.equal(new_assignment, assignment):
            break
        assignment = new_assignment

    observation_assignment = assignment[torch.searchsorted(unique_bins, view_bins)]
    counts = torch.stack([(~observation_assignment).sum(), observation_assignment.sum()])
    bin_counts = torch.stack([(~assignment).sum(), assignment.sum()])
    component_weight = torch.stack([
        weights[~observation_assignment].sum(), weights[observation_assignment].sum()
    ])
    priors = component_weight / component_weight.sum()
    single_loss = (weights * (1.0 - descriptors @ single)).sum() / weights.sum()
    mixture_similarity = descriptors @ centers.T
    mixture_loss = (
        weights * (1.0 - mixture_similarity.max(dim=1).values)
    ).sum() / weights.sum()
    improvement = float(single_loss - mixture_loss)
    cosine = float((centers[0] @ centers[1]).clamp(-1.0, 1.0))
    angle = math.degrees(math.acos(cosine))
    eligible = bool(
        int(counts.min()) >= int(minimum_cluster_observations)
        and int(bin_counts.min()) >= int(minimum_cluster_view_bins)
        and angle >= float(minimum_angle_degrees)
        and improvement >= float(minimum_loss_improvement)
    )
    return ViewMixture(
        prototypes=centers if eligible else single[None],
        priors=priors if eligible else torch.ones(1),
        view_bin_count=int(unique_bins.numel()),
        observation_count=int(descriptors.shape[0]),
        angle_degrees=angle,
        loss_improvement=improvement,
        eligible=eligible,
    )


def mixture_scores(
    query: torch.Tensor,
    prototypes: torch.Tensor,
    priors: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Aggregate prototype cosine scores into one score per Anchor.

    Inputs have shapes ``[Q,D]``, ``[N,K,D]`` and ``[N,K]``.  Zero priors mark
    padded prototypes and cannot create an extra geometric vote.
    """
    if float(temperature) <= 0:
        raise ValueError("mixture temperature must be positive")
    query = F.normalize(torch.as_tensor(query).float(), dim=1)
    prototypes = F.normalize(torch.as_tensor(prototypes).float(), dim=2)
    priors = torch.as_tensor(priors).float()
    if prototypes.ndim != 3 or priors.shape != prototypes.shape[:2]:
        raise ValueError("prototype and prior shapes do not align")
    logits = torch.einsum("qd,nkd->qnk", query, prototypes) / float(temperature)
    log_prior = torch.where(priors > 0, priors.log(), torch.full_like(priors, -torch.inf))
    output = float(temperature) * torch.logsumexp(logits + log_prior[None], dim=2)
    single = (priors > 0).sum(dim=1) == 1
    if bool(single.any()):
        # K=1 is a compatibility path, not a one-component approximation to
        # it.  Route it through the exact deployed cosine GEMM so adding an
        # unused padded slot cannot move ties at the last floating-point bit.
        output[:, single] = query @ prototypes[single, 0].T
    return output


class LeaveOneQueryOutViewMixtureMatcher:
    """Strict mapping-LOO matcher over a unified Track/surface Anchor map."""

    def __init__(self, *, state: dict, payload: dict, query_cache: dict,
                 device: torch.device, trim_fraction: float = 0.2,
                 minimum_cluster_observations: int = 2,
                 minimum_cluster_view_bins: int = 2,
                 minimum_angle_degrees: float = 12.0,
                 minimum_loss_improvement: float = 0.015,
                 maximum_prototype_ratio: float = 1.2,
                 temperature: float = 0.05) -> None:
        self.device = torch.device(device)
        self.temperature = float(temperature)
        self.trim_fraction = float(trim_fraction)
        self.thresholds = dict(
            minimum_cluster_observations=int(minimum_cluster_observations),
            minimum_cluster_view_bins=int(minimum_cluster_view_bins),
            minimum_angle_degrees=float(minimum_angle_degrees),
            minimum_loss_improvement=float(minimum_loss_improvement),
        )
        features = torch.as_tensor(state["anchor_features"]).float()
        self.replay = LeaveOneQueryOutProjectiveAnchorDescriptorBank(
            state=state, payload=payload, query_cache=query_cache,
            reference_features=features, trim_fraction=self.trim_fraction,
        )
        if self.replay.track_replay is None:
            raise ValueError("view-mixture map has no Track Anchors")
        self.track_replay = self.replay.track_replay
        count, dim = features.shape
        self.budget_extra = int(math.floor((float(maximum_prototype_ratio) - 1.0) * count))
        prototypes = torch.zeros((count, 2, dim), dtype=torch.float32)
        priors = torch.zeros((count, 2), dtype=torch.float32)
        prototypes[:, 0] = F.normalize(features, dim=1)
        priors[:, 0] = 1
        self.observations = {}
        self.base_eligible = set()
        for local_row, global_row in enumerate(self.replay.track_rows.tolist()):
            item = self._observations(local_row)
            self.observations[local_row] = item
            mixture = build_view_mixture(item[0], item[1], item[2], **self.thresholds)
            if mixture.eligible:
                self.base_eligible.add(local_row)
                prototypes[global_row] = mixture.prototypes
                priors[global_row] = mixture.priors
        if len(self.base_eligible) > self.budget_extra:
            raise ValueError("view-mixture prototype budget is saturated")
        self.base_prototypes = prototypes.to(self.device)
        self.base_priors = priors.to(self.device)
        self.prototypes = self.base_prototypes.clone()
        self.priors = self.base_priors.clone()
        self.previous_rows = torch.empty(0, dtype=torch.long, device=self.device)
        self.maximum_query_local_eligible = len(self.base_eligible)

    def _observations(self, local_row: int):
        replay = self.track_replay
        track = int(replay.track_indices[int(local_row)])
        observations = replay.observation_by_track[track]
        queries = torch.as_tensor(replay.tracks["query_index"])[observations].long()
        keypoints = torch.as_tensor(replay.tracks["keypoint_index"])[observations].long()
        valid = torch.tensor([
            replay.cached_validity[replay.query_names[int(q)]] is None
            or bool(replay.cached_validity[replay.query_names[int(q)]][int(k)])
            for q, k in zip(queries.tolist(), keypoints.tolist())
        ])
        keep = torch.tensor([
            replay.cached_descriptor_keep[replay.query_names[int(q)]] is None
            or bool(replay.cached_descriptor_keep[replay.query_names[int(q)]][int(k)])
            for q, k in zip(queries.tolist(), keypoints.tolist())
        ])
        if bool(valid.any()):
            observations, queries, keypoints, keep = observations[valid], queries[valid], keypoints[valid], keep[valid]
        observations, queries, keypoints = observations[keep], queries[keep], keypoints[keep]
        descriptors = F.normalize(torch.stack([
            replay.cached_descriptors[replay.query_names[int(q)]][int(k)].float()
            for q, k in zip(queries.tolist(), keypoints.tolist())
        ]), dim=1)
        confidence = torch.as_tensor(replay.tracks["confidence"])[observations].float()
        reliability = torch.tensor([
            1.0 if replay.cached_reliability[replay.query_names[int(q)]] is None
            else float(replay.cached_reliability[replay.query_names[int(q)]][int(k)])
            for q, k in zip(queries.tolist(), keypoints.tolist())
        ]).clamp(0, 1)
        return descriptors, replay.query_bins[queries], confidence * reliability, queries

    def _update(self, query_index: int) -> None:
        if self.previous_rows.numel():
            self.prototypes[self.previous_rows] = self.base_prototypes[self.previous_rows]
            self.priors[self.previous_rows] = self.base_priors[self.previous_rows]
        rows, features = self.replay.query_update(query_index)
        device_rows = rows.to(self.device)
        if rows.numel():
            self.prototypes[device_rows].zero_(); self.priors[device_rows].zero_()
            self.prototypes[device_rows, 0] = F.normalize(features.float(), dim=1).to(self.device)
            self.priors[device_rows, 0] = 1
        affected = self.track_replay.rows_by_query[int(query_index)]
        eligible = len(self.base_eligible) - sum(int(row in self.base_eligible) for row in affected)
        for local_row in affected:
            descriptors, bins, weights, queries = self.observations[int(local_row)]
            keep = queries != int(query_index)
            if not bool(keep.any()):
                continue
            mixture = build_view_mixture(descriptors[keep], bins[keep], weights[keep], **self.thresholds)
            if mixture.eligible:
                eligible += 1
                global_row = int(self.replay.track_rows[int(local_row)])
                self.prototypes[global_row] = mixture.prototypes.to(self.device)
                self.priors[global_row] = mixture.priors.to(self.device)
        if eligible > self.budget_extra:
            raise ValueError("query-local view-mixture eligibility saturates budget")
        self.maximum_query_local_eligible = max(self.maximum_query_local_eligible, eligible)
        self.previous_rows = device_rows

    @torch.inference_mode()
    def __call__(self, query_index: int, descriptors: torch.Tensor, topk: int):
        self._update(int(query_index))
        return global_view_mixture_topk(
            descriptors, self.prototypes, self.priors, topk=int(topk),
            temperature=self.temperature,
        )
