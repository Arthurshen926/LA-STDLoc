from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree


def _project_points(
    xyz: torch.Tensor, K: torch.Tensor, pose_w2c: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    xyz = torch.as_tensor(xyz, dtype=torch.float64)
    K = torch.as_tensor(K, dtype=torch.float64)
    pose_w2c = torch.as_tensor(pose_w2c, dtype=torch.float64)
    camera = xyz @ pose_w2c[:3, :3].T + pose_w2c[:3, 3]
    projected = camera @ K.T
    depth = camera[:, 2]
    uv = projected[:, :2] / depth[:, None].clamp_min(1e-12)
    return uv.cpu().numpy(), depth.cpu().numpy()


def robust_fuse_track_descriptors(
    descriptors: torch.Tensor,
    query_bins: torch.Tensor,
    confidence: torch.Tensor | None = None,
    *,
    trim_fraction: float = 0.2,
) -> torch.Tensor:
    """Fuse observations by view bin, then trim bins farthest from the medoid."""
    descriptors = F.normalize(torch.as_tensor(descriptors).float(), dim=1)
    query_bins = torch.as_tensor(query_bins, dtype=torch.long).reshape(-1)
    if descriptors.shape[0] != query_bins.numel() or descriptors.shape[0] == 0:
        raise ValueError("descriptor/view-bin observations are inconsistent")
    if confidence is None:
        confidence = torch.ones(descriptors.shape[0], dtype=torch.float32)
    confidence = torch.as_tensor(confidence, dtype=torch.float32).reshape(-1)
    if confidence.numel() != descriptors.shape[0]:
        raise ValueError("descriptor/confidence observations are inconsistent")
    bin_prototypes = []
    for view_bin in torch.unique(query_bins, sorted=True).tolist():
        selected = query_bins == int(view_bin)
        weight = confidence[selected].clamp_min(1e-4)
        prototype = (descriptors[selected] * weight[:, None]).sum(dim=0)
        bin_prototypes.append(F.normalize(prototype, dim=0))
    prototypes = torch.stack(bin_prototypes)
    similarity = prototypes @ prototypes.T
    medoid = int(similarity.mean(dim=1).argmax())
    keep_count = max(
        1,
        int(round(prototypes.shape[0] * (1.0 - float(trim_fraction)))),
    )
    keep = torch.topk(similarity[medoid], k=keep_count).indices
    return F.normalize(prototypes[keep].mean(dim=0), dim=0)


def protected_micro_anchor_descriptor_loss(
    *,
    candidate_features: torch.Tensor,
    positive_descriptors: torch.Tensor,
    positive_targets: torch.Tensor,
    positive_old_best: torch.Tensor,
    guard_descriptors: torch.Tensor,
    guard_old_best: torch.Tensor,
    initial_features: torch.Tensor,
    positive_margin: float = 0.03,
    guard_margin: float = 0.02,
    temperature: float = 0.03,
    guard_weight: float = 2.0,
    trust_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train new anchors while preserving clean decisions of a frozen bank."""
    candidate_features = F.normalize(candidate_features.float(), dim=1)
    positive_descriptors = F.normalize(positive_descriptors.float(), dim=1)
    guard_descriptors = F.normalize(guard_descriptors.float(), dim=1)
    initial_features = F.normalize(initial_features.float(), dim=1)
    positive_targets = positive_targets.long().reshape(-1)
    positive_old_best = positive_old_best.float().reshape(-1)
    guard_old_best = guard_old_best.float().reshape(-1)
    if positive_descriptors.shape[0] != positive_targets.numel():
        raise ValueError("positive descriptor and target counts must match")
    if positive_old_best.numel() != positive_targets.numel():
        raise ValueError("positive old-bank score count must match targets")
    if guard_descriptors.shape[0] != guard_old_best.numel():
        raise ValueError("guard descriptor and score counts must match")
    if candidate_features.shape != initial_features.shape:
        raise ValueError("candidate and initial feature tensors must align")
    if positive_targets.numel() == 0:
        raise ValueError("protected micro-anchor training requires positives")
    if bool((positive_targets < 0).any()) or bool(
        (positive_targets >= candidate_features.shape[0]).any()
    ):
        raise ValueError("positive target index is out of range")
    temperature = max(float(temperature), 1e-6)

    positive_logits = positive_descriptors @ candidate_features.T
    target_score = positive_logits.gather(
        1, positive_targets[:, None]
    ).squeeze(1)
    if candidate_features.shape[0] > 1:
        competing_logits = positive_logits.clone()
        competing_logits.scatter_(1, positive_targets[:, None], -torch.inf)
        new_competitor = competing_logits.max(dim=1).values
        competitor = torch.maximum(positive_old_best, new_competitor)
    else:
        competitor = positive_old_best
    positive_loss = (
        F.softplus(
            (competitor + float(positive_margin) - target_score) / temperature
        ).mean()
        * temperature
    )

    if guard_descriptors.shape[0] > 0:
        guard_new_best = (
            guard_descriptors @ candidate_features.T
        ).max(dim=1).values
        guard_loss = (
            F.softplus(
                (
                    guard_new_best
                    + float(guard_margin)
                    - guard_old_best
                )
                / temperature
            ).mean()
            * temperature
        )
        guard_violation = (
            guard_new_best + float(guard_margin) > guard_old_best
        ).float().mean()
    else:
        guard_new_best = candidate_features.new_zeros((0,))
        guard_loss = candidate_features.sum() * 0.0
        guard_violation = candidate_features.new_zeros(())

    trust_loss = (
        1.0 - (candidate_features * initial_features).sum(dim=1)
    ).mean()
    total = (
        positive_loss
        + float(guard_weight) * guard_loss
        + float(trust_weight) * trust_loss
    )
    diagnostics = {
        "loss": total.detach(),
        "positive_loss": positive_loss.detach(),
        "guard_loss": guard_loss.detach(),
        "trust_loss": trust_loss.detach(),
        "positive_win_rate": (
            target_score >= competitor + float(positive_margin)
        )
        .float()
        .mean()
        .detach(),
        "positive_target_score": target_score.mean().detach(),
        "positive_competitor_score": competitor.mean().detach(),
        "guard_violation_rate": guard_violation.detach(),
        "guard_new_best_score": (
            guard_new_best.mean().detach()
            if guard_new_best.numel()
            else candidate_features.new_zeros(())
        ),
        "guard_old_best_score": (
            guard_old_best.mean().detach()
            if guard_old_best.numel()
            else candidate_features.new_zeros(())
        ),
    }
    return total, diagnostics


def compute_track_coverage_gain(
    *,
    payload: dict,
    query_cache: dict,
    base_xyz: torch.Tensor,
    radius_px: float = 2.0,
    depth_abs_tolerance_m: float = 0.05,
    depth_rel_tolerance: float = 0.02,
) -> dict[str, torch.Tensor]:
    """Measure observations for which the old bank has no legal 2D anchor."""
    tracks = payload["tracks"]
    geometry = payload["track_geometry"]
    query_names = payload["query_names"]
    track_count = int(geometry["triangulated_xyz"].shape[0])
    gain = torch.zeros(track_count, dtype=torch.long)
    represented = torch.zeros(track_count, dtype=torch.long)
    valid_observations = torch.zeros(track_count, dtype=torch.long)
    observations_by_query = defaultdict(list)
    for observation, query in enumerate(tracks["query_index"].tolist()):
        observations_by_query[int(query)].append(observation)

    for query, observations in observations_by_query.items():
        cached = query_cache[query_names[query]]
        projected, projected_depth = _project_points(
            base_xyz, cached["native_K"], cached["pose_w2c"]
        )
        positive_depth = projected_depth > 0
        tree = cKDTree(projected[positive_depth])
        positive_indices = np.nonzero(positive_depth)[0]
        keypoint_indices = tracks["keypoint_index"][observations].long()
        keypoints = (
            torch.as_tensor(cached["native_keypoints"])[keypoint_indices]
            + float(cached.get("pixel_center_offset", 0.5))
        ).float()
        native_xy = torch.as_tensor(cached["native_keypoints"])[
            keypoint_indices
        ].float()
        native_depth = torch.as_tensor(cached["native_depth"]).float()
        x = native_xy[:, 0].round().long().clamp(0, native_depth.shape[1] - 1)
        y = native_xy[:, 1].round().long().clamp(0, native_depth.shape[0] - 1)
        reference_depth = native_depth[y, x].numpy()
        neighbors = tree.query_ball_point(keypoints.numpy(), r=float(radius_px))
        for local_row, observation in enumerate(observations):
            track = int(tracks["track_index"][observation])
            reference = float(reference_depth[local_row])
            if not np.isfinite(reference) or reference <= 0:
                continue
            valid_observations[track] += 1
            candidate_indices = positive_indices[
                np.asarray(neighbors[local_row], dtype=np.int64)
            ]
            if candidate_indices.size:
                tolerance = float(depth_abs_tolerance_m) + float(
                    depth_rel_tolerance
                ) * reference
                depth_clean = np.abs(
                    projected_depth[candidate_indices] - reference
                ) <= tolerance
                has_existing = bool(np.any(depth_clean))
            else:
                has_existing = False
            if has_existing:
                represented[track] += 1
            else:
                gain[track] += 1
    return {
        "coverage_gain": gain,
        "represented_observations": represented,
        "valid_observations": valid_observations,
    }


def build_add_only_materialized_anchor_map(
    *,
    base_state: dict,
    payload: dict,
    query_cache: dict,
    budget: int,
    minimum_coverage_gain: int = 1,
    minimum_distinct_view_bins: int = 2,
    minimum_separation_m: float = 0.005,
    descriptor_trim_fraction: float = 0.2,
    radius_px: float = 2.0,
    coverage: dict[str, torch.Tensor] | None = None,
) -> tuple[dict, dict]:
    """Create a frozen old bank plus Level-A track-derived micro-anchors."""
    if str(payload.get("schema", "")) != "lafgs_track_first_payload":
        raise ValueError("unsupported Track-First payload schema")
    base_features = F.normalize(
        torch.as_tensor(base_state["landmark_features"]).float(), dim=1
    )
    base_xyz = torch.as_tensor(base_state["landmark_xyz"]).float()
    base_source_ids = torch.as_tensor(
        base_state["landmark_indices"], dtype=torch.long
    ).reshape(-1)
    if not (
        base_features.shape[0] == base_xyz.shape[0] == base_source_ids.numel()
    ):
        raise ValueError("base state tensors are not row-aligned")
    cache = query_cache.get("queries", query_cache)
    tracks = payload["tracks"]
    geometry = payload["track_geometry"]
    assignment = payload["assignment"]
    query_names = payload["query_names"]
    track_xyz = torch.as_tensor(geometry["triangulated_xyz"]).float()
    high_confidence = torch.as_tensor(
        geometry["triangulation_high_confidence"], dtype=torch.bool
    )
    level = torch.as_tensor(
        geometry["track_confidence_level"], dtype=torch.int8
    )
    source_rows = torch.as_tensor(
        assignment["track_landmark_index"], dtype=torch.long
    )
    if coverage is None:
        coverage = compute_track_coverage_gain(
            payload=payload,
            query_cache=cache,
            base_xyz=base_xyz,
            radius_px=radius_px,
        )
    candidate = (
        high_confidence
        & (level == 2)
        & (source_rows >= 0)
        & (
            torch.as_tensor(
                geometry["triangulation_distinct_view_bin_count"]
            )
            >= int(minimum_distinct_view_bins)
        )
        & (coverage["coverage_gain"] >= int(minimum_coverage_gain))
    )
    candidate_indices = torch.nonzero(candidate, as_tuple=False).reshape(-1)
    score = (
        coverage["coverage_gain"].float() * 1000.0
        + torch.as_tensor(
            geometry["triangulation_distinct_view_bin_count"]
        ).float()
        * 10.0
        + torch.as_tensor(
            geometry["triangulation_observation_count"]
        ).float()
        - torch.as_tensor(
            geometry["triangulation_reprojection_median_px"]
        ).float()
    )
    order = candidate_indices[
        torch.argsort(score[candidate_indices], descending=True, stable=True)
    ]
    observation_by_track = defaultdict(list)
    for observation, track in enumerate(tracks["track_index"].tolist()):
        observation_by_track[int(track)].append(observation)

    selected = []
    selected_by_source = defaultdict(list)
    for track in order.tolist() if int(budget) > 0 else []:
        source_row = int(source_rows[track])
        xyz = track_xyz[track]
        duplicate = any(
            float(torch.linalg.norm(xyz - track_xyz[other])) <
            float(minimum_separation_m)
            for other in selected_by_source[source_row]
        )
        if duplicate:
            continue
        selected.append(track)
        selected_by_source[source_row].append(track)
        if len(selected) >= max(int(budget), 0):
            break

    new_features = []
    for track in selected:
        observations = observation_by_track[track]
        query_indices = tracks["query_index"][observations].long()
        keypoint_indices = tracks["keypoint_index"][observations].long()
        descriptors = torch.stack(
            [
                torch.as_tensor(
                    cache[query_names[int(query)]]["native_descriptors"]
                )[int(keypoint)]
                for query, keypoint in zip(
                    query_indices.tolist(), keypoint_indices.tolist()
                )
            ]
        )
        new_features.append(
            robust_fuse_track_descriptors(
                descriptors,
                torch.as_tensor(payload["query_bins"])[query_indices],
                torch.as_tensor(tracks["confidence"])[observations],
                trim_fraction=descriptor_trim_fraction,
            )
        )

    selected_tensor = torch.as_tensor(selected, dtype=torch.long)
    if selected:
        feature_extension = torch.stack(new_features)
        xyz_extension = track_xyz[selected_tensor]
        source_extension = base_source_ids[source_rows[selected_tensor]]
    else:
        feature_extension = base_features.new_zeros((0, base_features.shape[1]))
        xyz_extension = base_xyz.new_zeros((0, 3))
        source_extension = base_source_ids.new_zeros((0,))
    total = int(base_source_ids.numel() + selected_tensor.numel())
    anchor_type = torch.cat(
        (
            torch.zeros(base_source_ids.numel(), dtype=torch.int8),
            torch.ones(selected_tensor.numel(), dtype=torch.int8),
        )
    )
    track_cluster_ids = torch.cat(
        (
            torch.full((base_source_ids.numel(),), -1, dtype=torch.long),
            selected_tensor,
        )
    )
    output = {
        "version": 1,
        "schema": "lafgs_materialized_anchor_map",
        "anchor_ids": torch.arange(total, dtype=torch.long),
        "source_primitive_ids": torch.cat(
            (base_source_ids, source_extension)
        ),
        "track_cluster_ids": track_cluster_ids,
        "anchor_xyz": torch.cat((base_xyz, xyz_extension)),
        "anchor_features": torch.cat((base_features, feature_extension)),
        "anchor_type": anchor_type,
        "base_anchor_count": int(base_source_ids.numel()),
        "requested_micro_anchor_budget": int(budget),
        "micro_anchor_count": int(selected_tensor.numel()),
        "config": {
            "level_a_only": True,
            "add_only": True,
            "old_anchor_descriptors_frozen": True,
            "old_anchor_geometry_frozen": True,
            "minimum_coverage_gain": int(minimum_coverage_gain),
            "minimum_distinct_view_bins": int(minimum_distinct_view_bins),
            "minimum_separation_m": float(minimum_separation_m),
            "descriptor_trim_fraction": float(descriptor_trim_fraction),
            "coverage_radius_px": float(radius_px),
        },
        "micro_anchor_quality": {
            "coverage_gain": coverage["coverage_gain"][selected_tensor],
            "valid_observations": coverage["valid_observations"][
                selected_tensor
            ],
            "view_bin_count": torch.as_tensor(
                geometry["triangulation_distinct_view_bin_count"]
            )[selected_tensor],
            "reprojection_median_px": torch.as_tensor(
                geometry["triangulation_reprojection_median_px"]
            )[selected_tensor],
            "covariance_trace_m2": torch.as_tensor(
                geometry["triangulation_covariance_trace"]
            )[selected_tensor],
        },
    }
    diagnostics = {
        "base_anchor_count": int(base_source_ids.numel()),
        "eligible_track_count": int(candidate.sum()),
        "selected_micro_anchor_count": int(selected_tensor.numel()),
        "selected_source_primitive_count": int(
            torch.unique(source_extension).numel()
        ),
        "selected_multi_anchor_source_count": int(
            sum(len(value) > 1 for value in selected_by_source.values())
        ),
        "coverage_gain_sum": int(
            coverage["coverage_gain"][selected_tensor].sum()
        ),
        "coverage_gain_mean": float(
            coverage["coverage_gain"][selected_tensor].float().mean()
            if selected_tensor.numel()
            else 0.0
        ),
    }
    return output, diagnostics
