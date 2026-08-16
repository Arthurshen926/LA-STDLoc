from __future__ import annotations

import heapq
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


def fuse_projective_anchor_observations(
    descriptors: torch.Tensor,
    query_bins: torch.Tensor,
    *,
    detector_weight: torch.Tensor | None = None,
    view_weight: torch.Tensor | None = None,
    visibility_weight: torch.Tensor | None = None,
    sequence_weight: torch.Tensor | None = None,
    trim_fraction: float = 0.2,
) -> torch.Tensor:
    """GWFF-style fusion for a projective Anchor observation equivalence class.

    Compatibility mode is obtained by passing the historical Track confidence
    as ``detector_weight`` and appearance reliability as
    ``visibility_weight``.  Geometry-view and sequence balancing can then be
    enabled independently without creating a parallel Gaussian landmark map.
    """

    descriptors = torch.as_tensor(descriptors)
    count = int(descriptors.shape[0]) if descriptors.ndim else 0
    if count == 0:
        raise ValueError("projective Anchor fusion requires observations")
    combined = torch.ones(count, dtype=torch.float32, device=descriptors.device)
    for name, value in (
        ("detector_weight", detector_weight),
        ("view_weight", view_weight),
        ("visibility_weight", visibility_weight),
        ("sequence_weight", sequence_weight),
    ):
        if value is None:
            continue
        weight = torch.as_tensor(value, dtype=torch.float32, device=descriptors.device)
        if weight.ndim != 1 or weight.shape[0] != count:
            raise ValueError(f"{name} must have exact shape [{count}]")
        if not torch.isfinite(weight).all() or bool((weight < 0).any()):
            raise ValueError(f"{name} must be finite and non-negative")
        combined = combined * weight
    return robust_fuse_track_descriptors(
        descriptors,
        query_bins,
        combined,
        trim_fraction=trim_fraction,
    )


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
    target_score = positive_logits.gather(1, positive_targets[:, None]).squeeze(1)
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
        guard_new_best = (guard_descriptors @ candidate_features.T).max(dim=1).values
        guard_loss = (
            F.softplus(
                (guard_new_best + float(guard_margin) - guard_old_best) / temperature
            ).mean()
            * temperature
        )
        guard_violation = (
            (guard_new_best + float(guard_margin) > guard_old_best).float().mean()
        )
    else:
        guard_new_best = candidate_features.new_zeros((0,))
        guard_loss = candidate_features.sum() * 0.0
        guard_violation = candidate_features.new_zeros(())

    trust_loss = (1.0 - (candidate_features * initial_features).sum(dim=1)).mean()
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
        "positive_win_rate": (target_score >= competitor + float(positive_margin))
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
    visibility_cache: dict | None = None,
    candidate_track_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Measure observations for which the old bank has no visible legal anchor."""
    tracks = payload["tracks"]
    geometry = payload["track_geometry"]
    query_names = payload["query_names"]
    track_count = int(geometry["triangulated_xyz"].shape[0])
    gain = torch.zeros(track_count, dtype=torch.long)
    represented = torch.zeros(track_count, dtype=torch.long)
    valid_observations = torch.zeros(track_count, dtype=torch.long)
    observation_gap = torch.zeros(tracks["track_index"].numel(), dtype=torch.bool)
    gap_view_bins = [set() for _ in range(track_count)]
    gap_sequences = [set() for _ in range(track_count)]
    if candidate_track_mask is None:
        candidate_track_mask = torch.ones(track_count, dtype=torch.bool)
    candidate_track_mask = torch.as_tensor(
        candidate_track_mask, dtype=torch.bool
    ).reshape(-1)
    if candidate_track_mask.numel() != track_count:
        raise ValueError("candidate_track_mask must align with tracks")
    observations_by_query = defaultdict(list)
    for observation, (track, query) in enumerate(
        zip(
            tracks["track_index"].tolist(),
            tracks["query_index"].tolist(),
        )
    ):
        if bool(candidate_track_mask[int(track)]):
            observations_by_query[int(query)].append(observation)

    for query, observations in observations_by_query.items():
        cached = query_cache[query_names[query]]
        visible = np.ones(base_xyz.shape[0], dtype=bool)
        if visibility_cache is not None:
            name = query_names[query]
            if name not in visibility_cache:
                raise KeyError(f"visibility cache is missing query {name}")
            visible = (
                torch.as_tensor(visibility_cache[name], dtype=torch.bool)
                .reshape(-1)
                .numpy()
            )
            if visible.size != base_xyz.shape[0]:
                raise ValueError(
                    f"visibility rows for {name} do not align with base anchors"
                )
        projected, projected_depth = _project_points(
            base_xyz, cached["native_K"], cached["pose_w2c"]
        )
        positive_depth = (projected_depth > 0) & visible
        positive_indices = np.nonzero(positive_depth)[0]
        tree = cKDTree(projected[positive_depth]) if positive_indices.size else None
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
        neighbors = (
            tree.query_ball_point(keypoints.numpy(), r=float(radius_px))
            if tree is not None
            else [[] for _ in observations]
        )
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
                tolerance = (
                    float(depth_abs_tolerance_m)
                    + float(depth_rel_tolerance) * reference
                )
                depth_clean = (
                    np.abs(projected_depth[candidate_indices] - reference) <= tolerance
                )
                has_existing = bool(np.any(depth_clean))
            else:
                has_existing = False
            if has_existing:
                represented[track] += 1
            else:
                gain[track] += 1
                observation_gap[observation] = True
                query_index = int(tracks["query_index"][observation])
                gap_view_bins[track].add(
                    int(torch.as_tensor(payload["query_bins"])[query_index])
                )
                gap_sequences[track].add(str(query_names[query_index]).split("/", 1)[0])
    return {
        "coverage_gain": gain,
        "represented_observations": represented,
        "valid_observations": valid_observations,
        "coverage_gap_observation_mask": observation_gap,
        "coverage_gain_distinct_view_bins": torch.as_tensor(
            [len(value) for value in gap_view_bins], dtype=torch.long
        ),
        "coverage_gain_distinct_sequences": torch.as_tensor(
            [len(value) for value in gap_sequences], dtype=torch.long
        ),
        "raster_visibility_enabled": torch.tensor(visibility_cache is not None),
    }


def _track_observation_lookup(payload: dict) -> dict[int, list[int]]:
    observations = defaultdict(list)
    for observation, track in enumerate(payload["tracks"]["track_index"].tolist()):
        observations[int(track)].append(observation)
    return observations


def _selected_track_observation_lookup(
    payload: dict, track_indices: torch.Tensor
) -> dict[int, torch.Tensor]:
    """Index only requested tracks while preserving observation order."""
    requested = torch.as_tensor(track_indices, dtype=torch.long).reshape(-1)
    track_rows = torch.as_tensor(
        payload["tracks"]["track_index"], dtype=torch.long
    ).reshape(-1)
    if requested.numel() == 0:
        return {}
    if bool((requested < 0).any()):
        raise ValueError("track indices must be non-negative")
    selected_track = torch.zeros(
        max(int(track_rows.max()) + 1, int(requested.max()) + 1),
        dtype=torch.bool,
    )
    selected_track[requested] = True
    observations = torch.nonzero(selected_track[track_rows], as_tuple=False).reshape(-1)
    selected_rows = track_rows[observations]
    order = torch.argsort(selected_rows, stable=True)
    observations = observations[order]
    selected_rows = selected_rows[order]
    unique, counts = torch.unique_consecutive(selected_rows, return_counts=True)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    return {
        int(track): observations[int(offsets[row]) : int(offsets[row + 1])]
        for row, track in enumerate(unique.tolist())
    }


def fuse_track_descriptors(
    *,
    payload: dict,
    query_cache,
    track_indices: torch.Tensor,
    trim_fraction: float = 0.2,
) -> torch.Tensor:
    """Fuse every requested track independently using its native observations."""
    from evidence.observation_provider import ObservationProvider

    cache = (
        query_cache.records
        if isinstance(query_cache, ObservationProvider)
        else query_cache.get("queries", query_cache)
    )
    query_names = payload["query_names"]
    tracks = payload["tracks"]
    query_bins = torch.as_tensor(payload["query_bins"], dtype=torch.long)
    track_indices = torch.as_tensor(track_indices, dtype=torch.long).reshape(-1)
    observation_by_track = _selected_track_observation_lookup(payload, track_indices)
    cached_descriptors = {
        name: torch.as_tensor(cache[name]["native_descriptors"]) for name in query_names
    }
    cached_validity = {
        name: (
            torch.as_tensor(cache[name]["native_valid_keypoint_mask"]).bool()
            if "native_valid_keypoint_mask" in cache[name]
            else None
        )
        for name in query_names
    }
    cached_reliability = {
        name: (
            torch.as_tensor(cache[name]["native_appearance_reliability"]).float()
            if "native_appearance_reliability" in cache[name]
            else None
        )
        for name in query_names
    }
    cached_descriptor_keep = {
        name: (
            torch.as_tensor(cache[name]["native_descriptor_fusion_keep_mask"]).bool()
            if "native_descriptor_fusion_keep_mask" in cache[name]
            else None
        )
        for name in query_names
    }
    descriptor_policy_presence = {
        value is not None for value in cached_descriptor_keep.values()
    }
    if len(descriptor_policy_presence) != 1:
        raise ValueError("descriptor-fusion keep masks must exist for every query")
    has_descriptor_policy = descriptor_policy_presence == {True}
    features = []
    for track in track_indices.tolist():
        observations = observation_by_track[int(track)]
        query_indices = tracks["query_index"][observations].long()
        keypoint_indices = tracks["keypoint_index"][observations].long()
        valid = torch.as_tensor(
            [
                cached_validity[query_names[int(query)]] is None
                or bool(cached_validity[query_names[int(query)]][int(keypoint)])
                for query, keypoint in zip(
                    query_indices.tolist(), keypoint_indices.tolist()
                )
            ],
            dtype=torch.bool,
        )
        descriptor_keep = torch.as_tensor(
            [
                cached_descriptor_keep[query_names[int(query)]] is None
                or bool(cached_descriptor_keep[query_names[int(query)]][int(keypoint)])
                for query, keypoint in zip(
                    query_indices.tolist(), keypoint_indices.tolist()
                )
            ],
            dtype=torch.bool,
        )
        # Keep geometry fixed even when every rendered observation lies outside
        # the conservative alpha-valid region.  Such a Track retains all of its
        # observations and is explicitly down-weighted by its reliability.
        if bool(valid.any()):
            descriptor_keep = descriptor_keep[valid]
            observations = observations[valid]
            query_indices = query_indices[valid]
            keypoint_indices = keypoint_indices[valid]
        if has_descriptor_policy:
            if not bool(descriptor_keep.any()):
                raise ValueError(
                    "descriptor-fusion policy removed every usable Track observation"
                )
            observations = observations[descriptor_keep]
            query_indices = query_indices[descriptor_keep]
            keypoint_indices = keypoint_indices[descriptor_keep]
        descriptors = torch.stack(
            [
                cached_descriptors[query_names[int(query)]][int(keypoint)]
                for query, keypoint in zip(
                    query_indices.tolist(), keypoint_indices.tolist()
                )
            ]
        )
        confidence = torch.as_tensor(tracks["confidence"])[observations].float()
        reliability = torch.as_tensor(
            [
                (
                    1.0
                    if cached_reliability[query_names[int(query)]] is None
                    else float(
                        cached_reliability[query_names[int(query)]][int(keypoint)]
                    )
                )
                for query, keypoint in zip(
                    query_indices.tolist(), keypoint_indices.tolist()
                )
            ]
        ).clamp(0.0, 1.0)
        features.append(
            fuse_projective_anchor_observations(
                descriptors,
                query_bins[query_indices],
                detector_weight=confidence,
                visibility_weight=reliability,
                trim_fraction=trim_fraction,
            )
        )
    if not features:
        descriptor_dim = int(
            torch.as_tensor(next(iter(cache.values()))["native_descriptors"]).shape[1]
        )
        return torch.zeros((0, descriptor_dim), dtype=torch.float32)
    return torch.stack(features)


class LeaveOneQueryOutTrackDescriptorBank:
    """Replay a fused Track bank while excluding one mapping image at a time.

    Track identity and geometry remain the full-mapping artifacts.  Only the
    descriptor observations contributed by the current feedback query are
    removed.  This prevents a mapping descriptor from matching a map vector
    that contains that same descriptor without introducing held-out folds.
    """

    def __init__(
        self,
        *,
        payload: dict,
        query_cache: dict,
        track_indices: torch.Tensor,
        reference_features: torch.Tensor,
        trim_fraction: float = 0.2,
    ) -> None:
        self.payload = payload
        self.cache = query_cache.get("queries", query_cache)
        self.query_names = list(payload["query_names"])
        self.tracks = payload["tracks"]
        self.query_bins = torch.as_tensor(payload["query_bins"], dtype=torch.long)
        self.track_indices = torch.as_tensor(track_indices, dtype=torch.long).reshape(
            -1
        )
        self.reference_features = torch.as_tensor(reference_features).float()
        self.trim_fraction = float(trim_fraction)
        if self.reference_features.ndim != 2 or self.reference_features.shape[0] != (
            self.track_indices.numel()
        ):
            raise ValueError("reference features and selected Track rows differ")
        if self.track_indices.unique().numel() != self.track_indices.numel():
            raise ValueError("selected Track rows are not unique")
        if list(self.cache) != self.query_names:
            raise ValueError("Track payload and query cache order differs")

        self.observation_by_track = _selected_track_observation_lookup(
            payload, self.track_indices
        )
        self.cached_descriptors = {
            name: torch.as_tensor(self.cache[name]["native_descriptors"])
            for name in self.query_names
        }
        self.cached_validity = {
            name: (
                torch.as_tensor(self.cache[name]["native_valid_keypoint_mask"]).bool()
                if "native_valid_keypoint_mask" in self.cache[name]
                else None
            )
            for name in self.query_names
        }
        self.cached_reliability = {
            name: (
                torch.as_tensor(
                    self.cache[name]["native_appearance_reliability"]
                ).float()
                if "native_appearance_reliability" in self.cache[name]
                else None
            )
            for name in self.query_names
        }
        self.cached_descriptor_keep = {
            name: (
                torch.as_tensor(
                    self.cache[name]["native_descriptor_fusion_keep_mask"]
                ).bool()
                if "native_descriptor_fusion_keep_mask" in self.cache[name]
                else None
            )
            for name in self.query_names
        }
        descriptor_policy_presence = {
            value is not None for value in self.cached_descriptor_keep.values()
        }
        if len(descriptor_policy_presence) != 1:
            raise ValueError("descriptor-fusion keep masks must exist for every query")
        self.track_to_row = {
            int(track): row for row, track in enumerate(self.track_indices.tolist())
        }
        self.rows_by_query: list[list[int]] = [[] for _ in self.query_names]
        observation_track = torch.as_tensor(self.tracks["track_index"]).long()
        observation_query = torch.as_tensor(self.tracks["query_index"]).long()
        observation_keypoint = torch.as_tensor(self.tracks["keypoint_index"]).long()
        for track, query, keypoint in zip(
            observation_track.tolist(),
            observation_query.tolist(),
            observation_keypoint.tolist(),
        ):
            row = self.track_to_row.get(int(track))
            keep = self.cached_descriptor_keep[self.query_names[int(query)]]
            if row is not None and (keep is None or bool(keep[int(keypoint)])):
                self.rows_by_query[int(query)].append(row)
        self.rows_by_query = [sorted(set(rows)) for rows in self.rows_by_query]

        replayed = fuse_track_descriptors(
            payload=payload,
            query_cache=query_cache,
            track_indices=self.track_indices,
            trim_fraction=self.trim_fraction,
        )
        if not torch.equal(replayed, self.reference_features):
            maximum = float((replayed - self.reference_features).abs().max())
            raise ValueError(
                "reference map is not the exact full-observation fused Track bank "
                f"(maximum absolute difference {maximum})"
            )

    def _fuse_observations(self, observations: torch.Tensor) -> torch.Tensor:
        query_indices = torch.as_tensor(self.tracks["query_index"])[observations].long()
        keypoint_indices = torch.as_tensor(self.tracks["keypoint_index"])[
            observations
        ].long()
        valid = torch.as_tensor(
            [
                self.cached_validity[self.query_names[int(query)]] is None
                or bool(
                    self.cached_validity[self.query_names[int(query)]][int(keypoint)]
                )
                for query, keypoint in zip(
                    query_indices.tolist(), keypoint_indices.tolist()
                )
            ],
            dtype=torch.bool,
        )
        descriptor_keep = torch.as_tensor(
            [
                self.cached_descriptor_keep[self.query_names[int(query)]] is None
                or bool(
                    self.cached_descriptor_keep[self.query_names[int(query)]][
                        int(keypoint)
                    ]
                )
                for query, keypoint in zip(
                    query_indices.tolist(), keypoint_indices.tolist()
                )
            ],
            dtype=torch.bool,
        )
        if bool(valid.any()):
            descriptor_keep = descriptor_keep[valid]
            observations = observations[valid]
            query_indices = query_indices[valid]
            keypoint_indices = keypoint_indices[valid]
        if not bool(descriptor_keep.any()):
            raise ValueError(
                "descriptor-fusion policy leaves no observation after query exclusion"
            )
        observations = observations[descriptor_keep]
        query_indices = query_indices[descriptor_keep]
        keypoint_indices = keypoint_indices[descriptor_keep]
        descriptors = torch.stack(
            [
                self.cached_descriptors[self.query_names[int(query)]][int(keypoint)]
                for query, keypoint in zip(
                    query_indices.tolist(), keypoint_indices.tolist()
                )
            ]
        )
        confidence = torch.as_tensor(self.tracks["confidence"])[observations].float()
        reliability = torch.as_tensor(
            [
                (
                    1.0
                    if self.cached_reliability[self.query_names[int(query)]] is None
                    else float(
                        self.cached_reliability[self.query_names[int(query)]][
                            int(keypoint)
                        ]
                    )
                )
                for query, keypoint in zip(
                    query_indices.tolist(), keypoint_indices.tolist()
                )
            ]
        ).clamp(0.0, 1.0)
        return robust_fuse_track_descriptors(
            descriptors,
            self.query_bins[query_indices],
            confidence * reliability,
            trim_fraction=self.trim_fraction,
        )

    def query_update(self, query_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return map rows and fused vectors changed by excluding ``query_index``."""
        query_index = int(query_index)
        if not 0 <= query_index < len(self.query_names):
            raise ValueError("leave-one-query-out index is out of range")
        rows = torch.as_tensor(self.rows_by_query[query_index], dtype=torch.long)
        if rows.numel() == 0:
            return rows, self.reference_features.new_empty(
                (0, self.reference_features.shape[1])
            )
        features = []
        observation_query = torch.as_tensor(self.tracks["query_index"]).long()
        for row in rows.tolist():
            track = int(self.track_indices[row])
            observations = self.observation_by_track[track]
            remaining = observations[observation_query[observations] != query_index]
            if remaining.numel() == 0:
                raise ValueError(
                    "mapping query is the sole observation of a selected Track"
                )
            features.append(self._fuse_observations(remaining))
        return rows, torch.stack(features)


class LeaveOneQueryOutProjectiveAnchorDescriptorBank:
    """Leave-one-query-out replay for a unified Track/surface Anchor bank.

    Track rows retain the historical fusion exactly.  Non-Track rows are
    replayed from the explicit projective observation CSR using the same
    rendered detector/alpha GWFF weights used by surface completion.
    """

    def __init__(
        self,
        *,
        state: dict,
        payload: dict,
        query_cache: dict,
        reference_features: torch.Tensor,
        trim_fraction: float = 0.2,
    ) -> None:
        from evidence.observation_provider import GaussianRenderObservationProvider

        self.state = state
        self.payload = payload
        self.reference_features = torch.as_tensor(reference_features).float()
        self.track_ids = torch.as_tensor(state["track_cluster_ids"])
        if self.track_ids.dtype != torch.long or self.track_ids.ndim != 1:
            raise ValueError("unified map Track IDs must be an int64 vector")
        count = int(self.track_ids.numel())
        if (
            self.reference_features.ndim != 2
            or self.reference_features.shape[0] != count
        ):
            raise ValueError("unified reference features do not align with map rows")
        self.query_names = list(payload["query_names"])
        self.query_bins = torch.as_tensor(payload["query_bins"], dtype=torch.long)
        self.provider = GaussianRenderObservationProvider(
            query_cache,
            query_names=self.query_names,
            query_bins=self.query_bins,
        )
        self.views = [
            self.provider.build_view(index) for index in range(len(self.provider))
        ]
        self.trim_fraction = float(trim_fraction)
        self.track_rows = torch.nonzero(self.track_ids >= 0, as_tuple=False).reshape(-1)
        self.surface_rows = torch.nonzero(self.track_ids < 0, as_tuple=False).reshape(
            -1
        )
        self.track_replay = (
            LeaveOneQueryOutTrackDescriptorBank(
                payload=payload,
                query_cache=query_cache,
                track_indices=self.track_ids[self.track_rows],
                reference_features=self.reference_features[self.track_rows],
                trim_fraction=self.trim_fraction,
            )
            if self.track_rows.numel()
            else None
        )

        observations = state.get("projective_anchor_observations")
        if observations is None:
            if self.surface_rows.numel():
                raise ValueError("surface Anchors lack projective observations")
            self.offsets = torch.zeros(count + 1, dtype=torch.long)
            self.observation_query = torch.empty(0, dtype=torch.long)
            self.observation_keypoint = torch.empty(0, dtype=torch.long)
        else:
            if (
                observations.get("schema") != "lafgs_projective_anchor_observations"
                or int(observations.get("version", -1)) != 1
            ):
                raise ValueError("unsupported projective observation schema")
            self.offsets = torch.as_tensor(observations["observation_offsets"])
            self.observation_query = torch.as_tensor(observations["query_indices"])
            self.observation_keypoint = torch.as_tensor(
                observations["keypoint_indices"]
            )
            if self.offsets.dtype != torch.long or self.offsets.shape != (count + 1,):
                raise ValueError("projective observation offsets must be int64 [N+1]")
            edge_count = int(self.offsets[-1])
            if int(self.offsets[0]) != 0 or bool(
                (self.offsets[1:] < self.offsets[:-1]).any()
            ):
                raise ValueError("projective observation offsets are invalid")
            for value in (self.observation_query, self.observation_keypoint):
                if value.dtype != torch.long or value.shape != (edge_count,):
                    raise ValueError("projective observation indices must be int64 [E]")

        self.rows_by_query: list[list[int]] = [[] for _ in self.query_names]
        if self.track_replay is not None:
            for query_index, local_rows in enumerate(self.track_replay.rows_by_query):
                self.rows_by_query[query_index].extend(
                    self.track_rows[
                        torch.as_tensor(local_rows, dtype=torch.long)
                    ].tolist()
                )
        for row in self.surface_rows.tolist():
            start, end = int(self.offsets[row]), int(self.offsets[row + 1])
            if start == end:
                raise ValueError("surface Anchor has no projective observation")
            for query_index in torch.unique(
                self.observation_query[start:end], sorted=True
            ).tolist():
                self.rows_by_query[int(query_index)].append(int(row))
        self.rows_by_query = [sorted(set(rows)) for rows in self.rows_by_query]

        if self.surface_rows.numel():
            replayed = torch.stack(
                [
                    self._fuse_surface_row(int(row), excluded_query=None)
                    for row in self.surface_rows
                ]
            )
            expected = self.reference_features[self.surface_rows]
            if not torch.equal(replayed, expected):
                maximum = float((replayed - expected).abs().max())
                raise ValueError(
                    "surface reference is not the exact full-observation fused bank "
                    f"(maximum absolute difference {maximum})"
                )

    def _fuse_surface_row(
        self, row: int, *, excluded_query: int | None
    ) -> torch.Tensor:
        start, end = int(self.offsets[row]), int(self.offsets[row + 1])
        queries = self.observation_query[start:end]
        keypoints = self.observation_keypoint[start:end]
        if excluded_query is not None:
            keep = queries != int(excluded_query)
            queries = queries[keep]
            keypoints = keypoints[keep]
        if queries.numel() == 0:
            raise ValueError("mapping query is the sole surface Anchor observation")
        descriptors = []
        detector = []
        alpha = []
        for query_index, keypoint_index in zip(queries.tolist(), keypoints.tolist()):
            view = self.views[int(query_index)]
            keypoint_index = int(keypoint_index)
            descriptors.append(view.descriptors[keypoint_index])
            detector.append(view.detector_scores[keypoint_index])
            if view.keypoint_alpha is not None:
                alpha.append(view.keypoint_alpha[keypoint_index])
            elif view.alpha is not None:
                height, width = view.image_hw
                pixel = torch.floor(view.keypoints[keypoint_index]).long()
                x = int(pixel[0].clamp(0, width - 1))
                y = int(pixel[1].clamp(0, height - 1))
                alpha.append(view.alpha[y, x])
            else:
                raise ValueError("surface Anchor replay requires rendered alpha")
        return fuse_projective_anchor_observations(
            F.normalize(torch.stack(descriptors).float(), dim=1),
            self.query_bins[queries],
            detector_weight=torch.stack(detector).float().clamp_min(0),
            visibility_weight=torch.stack(alpha).float().clamp(0, 1),
            trim_fraction=self.trim_fraction,
        )

    def query_update(self, query_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        query_index = int(query_index)
        if not 0 <= query_index < len(self.query_names):
            raise ValueError("leave-one-query-out index is out of range")
        rows = []
        features = []
        if self.track_replay is not None:
            local_rows, track_features = self.track_replay.query_update(query_index)
            rows.extend(self.track_rows[local_rows].tolist())
            features.extend(track_features)
        surface = [
            row
            for row in self.rows_by_query[query_index]
            if bool(self.track_ids[row] < 0)
        ]
        for row in surface:
            rows.append(int(row))
            features.append(
                self._fuse_surface_row(int(row), excluded_query=query_index)
            )
        if not rows:
            return torch.empty(0, dtype=torch.long), self.reference_features.new_empty(
                (0, self.reference_features.shape[1])
            )
        order = torch.argsort(torch.tensor(rows, dtype=torch.long), stable=True)
        return (
            torch.tensor(rows, dtype=torch.long)[order],
            torch.stack(features)[order],
        )


@torch.no_grad()
def compute_track_functional_statistics(
    *,
    payload: dict,
    query_cache: dict,
    base_xyz: torch.Tensor,
    base_features: torch.Tensor,
    track_indices: torch.Tensor,
    track_features: torch.Tensor,
    radius_px: float = 2.0,
    depth_abs_tolerance_m: float = 0.05,
    depth_rel_tolerance: float = 0.02,
    visibility_cache: dict | None = None,
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Measure rank gaps and false-attractor behavior against the frozen bank."""
    cache = query_cache.get("queries", query_cache)
    query_names = payload["query_names"]
    tracks = payload["tracks"]
    track_indices = torch.as_tensor(track_indices, dtype=torch.long).reshape(-1)
    track_features = F.normalize(torch.as_tensor(track_features).float(), dim=1)
    if track_features.shape[0] != track_indices.numel():
        raise ValueError("track indices and fused features must align")
    track_count = int(payload["track_geometry"]["triangulated_xyz"].shape[0])
    track_to_row = torch.full((track_count,), -1, dtype=torch.long)
    track_to_row[track_indices] = torch.arange(track_indices.numel())
    selected = track_to_row[tracks["track_index"].long()] >= 0
    selected_observations = torch.nonzero(selected, as_tuple=False).reshape(-1)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    old_xyz = torch.as_tensor(base_xyz).float().to(device)
    old_features = F.normalize(torch.as_tensor(base_features).float(), dim=1).to(device)
    candidate_features = track_features.to(device)

    observation_gap = torch.zeros(tracks["track_index"].numel(), dtype=torch.bool)
    observation_old_best = torch.full((tracks["track_index"].numel(),), -torch.inf)
    observation_self_score = torch.full_like(observation_old_best, -torch.inf)
    observation_candidate_best = torch.full_like(observation_old_best, -torch.inf)
    observation_candidate_row = torch.full(
        (tracks["track_index"].numel(),), -1, dtype=torch.long
    )
    observations_by_query = defaultdict(list)
    for observation in selected_observations.tolist():
        query = int(tracks["query_index"][observation])
        observations_by_query[query].append(observation)

    for query, observations in observations_by_query.items():
        cached = cache[query_names[query]]
        observation_tensor = torch.as_tensor(observations, dtype=torch.long)
        keypoint_index = tracks["keypoint_index"][observation_tensor].long()
        descriptors = F.normalize(
            torch.as_tensor(cached["native_descriptors"]).float()[keypoint_index],
            dim=1,
        ).to(device)
        old_score, old_index = (descriptors @ old_features.T).max(dim=1)
        projected, depth = _project_points(
            old_xyz[old_index].cpu(),
            cached["native_K"],
            cached["pose_w2c"],
        )
        native_keypoints = torch.as_tensor(cached["native_keypoints"]).float()[
            keypoint_index.cpu()
        ]
        physical = (
            native_keypoints + float(cached.get("pixel_center_offset", 0.5))
        ).numpy()
        native_depth = torch.as_tensor(cached["native_depth"]).float()
        x = native_keypoints[:, 0].round().long().clamp(0, native_depth.shape[1] - 1)
        y = native_keypoints[:, 1].round().long().clamp(0, native_depth.shape[0] - 1)
        reference_depth = native_depth[y, x].numpy()
        tolerance = float(depth_abs_tolerance_m) + (
            float(depth_rel_tolerance) * np.abs(reference_depth)
        )
        visible = np.ones(len(observations), dtype=bool)
        if visibility_cache is not None:
            name = query_names[query]
            if name not in visibility_cache:
                raise KeyError(f"visibility cache is missing query {name}")
            query_visibility = torch.as_tensor(
                visibility_cache[name], dtype=torch.bool
            ).reshape(-1)
            if query_visibility.numel() != old_xyz.shape[0]:
                raise ValueError(
                    f"visibility rows for {name} do not align with base anchors"
                )
            visible = query_visibility[old_index.cpu()].numpy()
        clean = (
            visible
            & (depth > 0)
            & np.isfinite(reference_depth)
            & (reference_depth > 0)
            & (np.abs(depth - reference_depth) <= tolerance)
            & (np.linalg.norm(projected - physical, axis=1) <= float(radius_px))
        )
        target_row = track_to_row[tracks["track_index"][observation_tensor].long()].to(
            device
        )
        candidate_score = descriptors @ candidate_features.T
        best_score, best_row = candidate_score.max(dim=1)
        self_score = candidate_score.gather(1, target_row[:, None]).squeeze(1)
        observation_gap[observation_tensor] = torch.from_numpy(~clean)
        observation_old_best[observation_tensor] = old_score.cpu()
        observation_self_score[observation_tensor] = self_score.cpu()
        observation_candidate_best[observation_tensor] = best_score.cpu()
        observation_candidate_row[observation_tensor] = best_row.cpu()

    functional_gain = torch.zeros(track_count, dtype=torch.long)
    functional_bins = [set() for _ in range(track_count)]
    functional_sequences = [set() for _ in range(track_count)]
    positive_margin_sum = torch.zeros(track_count)
    positive_count = torch.zeros(track_count, dtype=torch.long)
    false_incoming = torch.zeros(track_count, dtype=torch.long)
    candidate_opportunities = torch.zeros(track_count, dtype=torch.long)
    promoted_correct = torch.zeros(track_count, dtype=torch.long)
    query_bins = torch.as_tensor(payload["query_bins"], dtype=torch.long)
    for observation in selected_observations.tolist():
        track = int(tracks["track_index"][observation])
        query = int(tracks["query_index"][observation])
        positive_count[track] += 1
        positive_margin_sum[track] += (
            observation_self_score[observation] - observation_old_best[observation]
        )
        if bool(observation_gap[observation]):
            functional_gain[track] += 1
            functional_bins[track].add(int(query_bins[query]))
            functional_sequences[track].add(str(query_names[query]).split("/", 1)[0])
        predicted_row = int(observation_candidate_row[observation])
        target_row = int(track_to_row[track])
        beats_old = bool(
            observation_candidate_best[observation] > observation_old_best[observation]
        )
        if beats_old and predicted_row >= 0:
            predicted_track = int(track_indices[predicted_row])
            candidate_opportunities[predicted_track] += 1
        if beats_old and predicted_row == target_row:
            promoted_correct[track] += 1
        elif beats_old and predicted_row >= 0:
            false_incoming[predicted_track] += 1
    return {
        "functional_gap": functional_gain,
        "functional_gap_observation_mask": observation_gap,
        "functional_gap_distinct_view_bins": torch.as_tensor(
            [len(value) for value in functional_bins], dtype=torch.long
        ),
        "functional_gap_distinct_sequences": torch.as_tensor(
            [len(value) for value in functional_sequences], dtype=torch.long
        ),
        "positive_hardnegative_margin_mean": positive_margin_sum
        / positive_count.clamp_min(1),
        "false_attractor_incoming_count": false_incoming,
        "candidate_opportunity_count": candidate_opportunities,
        "false_attractor_opportunity_rate": (
            false_incoming.float() / candidate_opportunities.clamp_min(1).float()
        ),
        "promoted_correct_count": promoted_correct,
        "observation_count": positive_count,
        "observation_old_best_score": observation_old_best,
        "observation_self_score": observation_self_score,
    }


def _canonical_base_tensors(
    base_state: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base_features = F.normalize(
        torch.as_tensor(base_state["landmark_features"]).float(), dim=1
    )
    base_xyz = torch.as_tensor(base_state["landmark_xyz"]).float()
    base_source_ids = torch.as_tensor(
        base_state["landmark_indices"], dtype=torch.long
    ).reshape(-1)
    if not (base_features.shape[0] == base_xyz.shape[0] == base_source_ids.numel()):
        raise ValueError("base state tensors are not row-aligned")
    return base_features, base_xyz, base_source_ids


def _empty_micro_anchor_quality() -> dict[str, torch.Tensor]:
    return {
        "coverage_gain": torch.empty(0, dtype=torch.long),
        "valid_observations": torch.empty(0, dtype=torch.long),
        "view_bin_count": torch.empty(0, dtype=torch.long),
        "reprojection_median_px": torch.empty(0, dtype=torch.float32),
        "covariance_trace_m2": torch.empty(0, dtype=torch.float32),
    }


def _build_add_only_anchor_map_schema(
    *,
    base_features: torch.Tensor,
    base_xyz: torch.Tensor,
    base_source_ids: torch.Tensor,
    selected_track_ids: torch.Tensor,
    source_extension: torch.Tensor,
    xyz_extension: torch.Tensor,
    feature_extension: torch.Tensor,
    micro_anchor_quality: dict[str, torch.Tensor],
    requested_budget: int,
    minimum_coverage_gain: int,
    minimum_distinct_view_bins: int,
    minimum_separation_m: float,
    descriptor_trim_fraction: float,
    radius_px: float,
) -> dict:
    """Build the shared, ordered add-only anchor-map schema."""
    base_count = int(base_source_ids.numel())
    selected_count = int(selected_track_ids.numel())
    if not (
        int(source_extension.shape[0])
        == int(xyz_extension.shape[0])
        == int(feature_extension.shape[0])
        == selected_count
    ):
        raise ValueError("materialized anchor extensions are not row-aligned")
    return {
        "version": 1,
        "schema": "lafgs_materialized_anchor_map",
        "anchor_ids": torch.arange(base_count + selected_count, dtype=torch.long),
        "source_primitive_ids": torch.cat((base_source_ids, source_extension)),
        "track_cluster_ids": torch.cat(
            (
                torch.full((base_count,), -1, dtype=torch.long),
                selected_track_ids,
            )
        ),
        "anchor_xyz": torch.cat((base_xyz, xyz_extension)),
        "anchor_features": torch.cat((base_features, feature_extension)),
        "anchor_type": torch.cat(
            (
                torch.zeros(base_count, dtype=torch.int8),
                torch.ones(selected_count, dtype=torch.int8),
            )
        ),
        "base_anchor_count": base_count,
        "requested_micro_anchor_budget": int(requested_budget),
        "micro_anchor_count": selected_count,
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
        "micro_anchor_quality": micro_anchor_quality,
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
    base_features, base_xyz, base_source_ids = _canonical_base_tensors(base_state)
    cache = query_cache.get("queries", query_cache)
    tracks = payload["tracks"]
    geometry = payload["track_geometry"]
    assignment = payload["assignment"]
    query_names = payload["query_names"]
    track_xyz = torch.as_tensor(geometry["triangulated_xyz"]).float()
    high_confidence = torch.as_tensor(
        geometry["triangulation_high_confidence"], dtype=torch.bool
    )
    level = torch.as_tensor(geometry["track_confidence_level"], dtype=torch.int8)
    source_rows = torch.as_tensor(assignment["track_landmark_index"], dtype=torch.long)
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
            torch.as_tensor(geometry["triangulation_distinct_view_bin_count"])
            >= int(minimum_distinct_view_bins)
        )
        & (coverage["coverage_gain"] >= int(minimum_coverage_gain))
    )
    candidate_indices = torch.nonzero(candidate, as_tuple=False).reshape(-1)
    score = (
        coverage["coverage_gain"].float() * 1000.0
        + torch.as_tensor(geometry["triangulation_distinct_view_bin_count"]).float()
        * 10.0
        + torch.as_tensor(geometry["triangulation_observation_count"]).float()
        - torch.as_tensor(geometry["triangulation_reprojection_median_px"]).float()
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
            float(torch.linalg.norm(xyz - track_xyz[other]))
            < float(minimum_separation_m)
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
                torch.as_tensor(cache[query_names[int(query)]]["native_descriptors"])[
                    int(keypoint)
                ]
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
    micro_anchor_quality = {
        "coverage_gain": coverage["coverage_gain"][selected_tensor],
        "valid_observations": coverage["valid_observations"][selected_tensor],
        "view_bin_count": torch.as_tensor(
            geometry["triangulation_distinct_view_bin_count"]
        )[selected_tensor],
        "reprojection_median_px": torch.as_tensor(
            geometry["triangulation_reprojection_median_px"]
        )[selected_tensor],
        "covariance_trace_m2": torch.as_tensor(
            geometry["triangulation_covariance_trace"]
        )[selected_tensor],
    }
    output = _build_add_only_anchor_map_schema(
        base_features=base_features,
        base_xyz=base_xyz,
        base_source_ids=base_source_ids,
        selected_track_ids=selected_tensor,
        source_extension=source_extension,
        xyz_extension=xyz_extension,
        feature_extension=feature_extension,
        micro_anchor_quality=micro_anchor_quality,
        requested_budget=budget,
        minimum_coverage_gain=minimum_coverage_gain,
        minimum_distinct_view_bins=minimum_distinct_view_bins,
        minimum_separation_m=minimum_separation_m,
        descriptor_trim_fraction=descriptor_trim_fraction,
        radius_px=radius_px,
    )
    diagnostics = {
        "base_anchor_count": int(base_source_ids.numel()),
        "eligible_track_count": int(candidate.sum()),
        "selected_micro_anchor_count": int(selected_tensor.numel()),
        "selected_source_primitive_count": int(torch.unique(source_extension).numel()),
        "selected_multi_anchor_source_count": int(
            sum(len(value) > 1 for value in selected_by_source.values())
        ),
        "coverage_gain_sum": int(coverage["coverage_gain"][selected_tensor].sum()),
        "coverage_gain_mean": float(
            coverage["coverage_gain"][selected_tensor].float().mean()
            if selected_tensor.numel()
            else 0.0
        ),
    }
    return output, diagnostics


def build_canonical_base_anchor_map(
    *,
    base_state: dict,
    minimum_coverage_gain: int = 1,
    minimum_distinct_view_bins: int = 2,
    minimum_separation_m: float = 0.005,
    descriptor_trim_fraction: float = 0.2,
    radius_px: float = 2.0,
) -> tuple[dict, dict]:
    """Materialize the canonical base prefix without evaluating Track coverage.

    A zero micro-anchor budget cannot select a Track row.  The canonical map is
    therefore a pure function of ``base_state`` and the recorded configuration;
    loading the query cache and projecting every base point only computes the
    otherwise-unused ``eligible_track_count`` diagnostic.  This constructor
    makes that contract explicit.  It reports eligibility as unevaluated rather
    than incorrectly treating it as zero.
    """
    base_features, base_xyz, base_source_ids = _canonical_base_tensors(base_state)

    base_count = int(base_source_ids.numel())
    selected_track_ids = torch.empty(0, dtype=torch.long)
    feature_extension = base_features.new_zeros((0, base_features.shape[1]))
    xyz_extension = base_xyz.new_zeros((0, 3))
    source_extension = base_source_ids.new_zeros((0,))
    output = _build_add_only_anchor_map_schema(
        base_features=base_features,
        base_xyz=base_xyz,
        base_source_ids=base_source_ids,
        selected_track_ids=selected_track_ids,
        source_extension=source_extension,
        xyz_extension=xyz_extension,
        feature_extension=feature_extension,
        micro_anchor_quality=_empty_micro_anchor_quality(),
        requested_budget=0,
        minimum_coverage_gain=minimum_coverage_gain,
        minimum_distinct_view_bins=minimum_distinct_view_bins,
        minimum_separation_m=minimum_separation_m,
        descriptor_trim_fraction=descriptor_trim_fraction,
        radius_px=radius_px,
    )
    diagnostics = {
        "base_anchor_count": base_count,
        "eligible_track_count": None,
        "eligibility_evaluated": False,
        "selected_micro_anchor_count": 0,
        "selected_source_primitive_count": 0,
        "selected_multi_anchor_source_count": 0,
        "coverage_gain_sum": 0,
        "coverage_gain_mean": 0.0,
    }
    return output, diagnostics


def build_v2_materialized_anchor_map(
    *,
    base_state: dict,
    payload: dict,
    query_cache: dict,
    budget: int,
    visibility_cache: dict | None = None,
    include_identity_split: bool = True,
    score_mode: str = "balanced",
    cluster_radius_m: float = 0.015,
    cluster_min_descriptor_cosine: float = 0.85,
    descriptor_trim_fraction: float = 0.2,
    radius_px: float = 2.0,
    device: str | torch.device | None = None,
) -> tuple[dict, dict]:
    """Build clustered coverage/identity micro-anchors from full G3 groups."""
    if score_mode not in {"coverage_count", "balanced"}:
        raise ValueError("score_mode must be coverage_count or balanced")
    if payload.get("schema") != "lafgs_track_first_payload":
        raise ValueError("unsupported Track-First payload schema")
    assignment = payload["assignment"]
    required_group_fields = {
        "track_landmark_offsets",
        "track_landmark_indices",
        "track_landmark_costs",
    }
    if not required_group_fields.issubset(assignment):
        raise ValueError("Micro-Anchor V2 requires complete G3 group CSR")

    base_features = F.normalize(
        torch.as_tensor(base_state["landmark_features"]).float(), dim=1
    )
    base_xyz = torch.as_tensor(base_state["landmark_xyz"]).float()
    base_source_ids = torch.as_tensor(
        base_state["landmark_indices"], dtype=torch.long
    ).reshape(-1)
    geometry = payload["track_geometry"]
    tracks = payload["tracks"]
    track_xyz = torch.as_tensor(geometry["triangulated_xyz"]).float()
    group_offsets = torch.as_tensor(
        assignment["track_landmark_offsets"], dtype=torch.long
    )
    group_indices = torch.as_tensor(
        assignment["track_landmark_indices"], dtype=torch.long
    )
    group_costs = torch.as_tensor(
        assignment["track_landmark_costs"], dtype=torch.float32
    )
    candidate_mask = (
        torch.as_tensor(geometry["triangulation_high_confidence"], dtype=torch.bool)
        & (torch.as_tensor(geometry["track_confidence_level"], dtype=torch.int8) == 2)
        & (group_offsets[1:] > group_offsets[:-1])
    )
    track_indices = torch.nonzero(candidate_mask, as_tuple=False).reshape(-1)
    track_features = fuse_track_descriptors(
        payload=payload,
        query_cache=query_cache,
        track_indices=track_indices,
        trim_fraction=descriptor_trim_fraction,
    )
    coverage = compute_track_coverage_gain(
        payload=payload,
        query_cache=query_cache,
        base_xyz=base_xyz,
        radius_px=radius_px,
        visibility_cache=visibility_cache,
        candidate_track_mask=candidate_mask,
    )
    functional = compute_track_functional_statistics(
        payload=payload,
        query_cache=query_cache,
        base_xyz=base_xyz,
        base_features=base_features,
        track_indices=track_indices,
        track_features=track_features,
        radius_px=radius_px,
        visibility_cache=visibility_cache,
        device=device,
    )
    feature_by_track = {
        int(track): track_features[row]
        for row, track in enumerate(track_indices.tolist())
    }
    source_groups = {
        int(track): set(
            group_indices[
                int(group_offsets[track]) : int(group_offsets[track + 1])
            ].tolist()
        )
        for track in track_indices.tolist()
    }

    parent = {int(track): int(track) for track in track_indices.tolist()}
    component_members = {int(track): {int(track)} for track in track_indices.tolist()}

    def find(value):
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != value:
            next_value = parent[value]
            parent[value] = root
            value = next_value
        return root

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
            component_members[left_root].update(component_members.pop(right_root))
        else:
            parent[left_root] = right_root
            component_members[right_root].update(component_members.pop(left_root))

    selected_xyz_np = track_xyz[track_indices].numpy()
    spatial_pairs = cKDTree(selected_xyz_np).query_pairs(r=float(cluster_radius_m))
    eligible_pairs = set()
    for left_row, right_row in sorted(spatial_pairs):
        left = int(track_indices[left_row])
        right = int(track_indices[right_row])
        if source_groups[left].isdisjoint(source_groups[right]):
            continue
        cosine = float(torch.dot(feature_by_track[left], feature_by_track[right]))
        if cosine < float(cluster_min_descriptor_cosine):
            continue
        eligible_pairs.add((min(left, right), max(left, right)))
    clustered_pair_count = 0
    rejected_single_linkage_pair_count = 0
    for left, right in sorted(eligible_pairs):
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            continue
        complete_link = all(
            (min(a, b), max(a, b)) in eligible_pairs
            for a in component_members[left_root]
            for b in component_members[right_root]
        )
        if not complete_link:
            rejected_single_linkage_pair_count += 1
            continue
        union(left, right)
        clustered_pair_count += 1
    members_by_root = defaultdict(list)
    for track in track_indices.tolist():
        members_by_root[find(int(track))].append(int(track))

    observation_by_track = _track_observation_lookup(payload)
    query_names = payload["query_names"]
    query_bins = torch.as_tensor(payload["query_bins"], dtype=torch.long)
    cache = query_cache.get("queries", query_cache)
    covariance = torch.as_tensor(geometry["triangulation_covariance_trace"]).float()
    view_bin_count = torch.as_tensor(
        geometry["triangulation_distinct_view_bin_count"]
    ).long()
    reprojection = torch.as_tensor(
        geometry["triangulation_reprojection_median_px"]
    ).float()

    clusters = []
    for root, members in sorted(members_by_root.items()):
        member_tensor = torch.as_tensor(members, dtype=torch.long)
        weight = torch.reciprocal(covariance[member_tensor].clamp_min(1e-8))
        weight /= weight.sum()
        xyz = (track_xyz[member_tensor] * weight[:, None]).sum(dim=0)
        observations = [
            observation
            for member in members
            for observation in observation_by_track[member]
        ]
        observation_tensor = torch.as_tensor(observations, dtype=torch.long)
        observation_queries = tracks["query_index"][observation_tensor].long()
        observation_keypoints = tracks["keypoint_index"][observation_tensor].long()
        descriptors = torch.stack(
            [
                torch.as_tensor(cache[query_names[int(query)]]["native_descriptors"])[
                    int(keypoint)
                ]
                for query, keypoint in zip(
                    observation_queries.tolist(),
                    observation_keypoints.tolist(),
                )
            ]
        )
        feature = robust_fuse_track_descriptors(
            descriptors,
            query_bins[observation_queries],
            torch.as_tensor(tracks["confidence"])[observation_tensor],
            trim_fraction=descriptor_trim_fraction,
        )
        source_group = sorted(
            set().union(*(source_groups[member] for member in members))
        )
        representative_options = []
        for member in members:
            begin = int(group_offsets[member])
            end = int(group_offsets[member + 1])
            representative_options.extend(
                zip(
                    group_costs[begin:end].tolist(),
                    group_indices[begin:end].tolist(),
                )
            )
        representative_row = min(representative_options)[1]
        geo_gain = int(coverage["coverage_gain"][member_tensor].sum())
        func_gain = int(functional["functional_gap"][member_tensor].sum())
        geo_bins = len(
            {
                int(query_bins[int(tracks["query_index"][observation])])
                for observation in observations
                if bool(coverage["coverage_gap_observation_mask"][observation])
            }
        )
        func_bins = len(
            {
                int(query_bins[int(tracks["query_index"][observation])])
                for observation in observations
                if bool(functional["functional_gap_observation_mask"][observation])
            }
        )
        geo_sequences = len(
            {
                str(query_names[int(tracks["query_index"][observation])]).split("/", 1)[
                    0
                ]
                for observation in observations
                if bool(coverage["coverage_gap_observation_mask"][observation])
            }
        )
        func_sequences = len(
            {
                str(query_names[int(tracks["query_index"][observation])]).split("/", 1)[
                    0
                ]
                for observation in observations
                if bool(functional["functional_gap_observation_mask"][observation])
            }
        )
        false_incoming = int(
            functional["false_attractor_incoming_count"][member_tensor].sum()
        )
        candidate_opportunities = int(
            functional["candidate_opportunity_count"][member_tensor].sum()
        )
        promoted = int(functional["promoted_correct_count"][member_tensor].sum())
        observation_count = int(functional["observation_count"][member_tensor].sum())
        margin = float(
            (
                functional["positive_hardnegative_margin_mean"][member_tensor]
                * functional["observation_count"][member_tensor]
            ).sum()
            / functional["observation_count"][member_tensor].sum().clamp_min(1)
        )
        false_rate = false_incoming / max(candidate_opportunities, 1)
        cluster_xyz = track_xyz[member_tensor]
        cluster_diameter = float(
            torch.cdist(cluster_xyz, cluster_xyz).max()
            if member_tensor.numel() > 1
            else 0.0
        )
        cluster_features = torch.stack([feature_by_track[member] for member in members])
        cluster_min_cosine = float(
            (cluster_features @ cluster_features.T).min()
            if member_tensor.numel() > 1
            else 1.0
        )
        member_query_sets = [
            {
                int(tracks["query_index"][observation])
                for observation in observation_by_track[member]
            }
            for member in members
        ]
        same_query_collisions = sum(
            len(member_query_sets[left] & member_query_sets[right])
            for left in range(len(members))
            for right in range(left + 1, len(members))
        )
        if score_mode == "coverage_count":
            score = (
                1000.0 * geo_gain
                + 10.0 * int(view_bin_count[member_tensor].max())
                + len(observations)
                - float(reprojection[member_tensor].mean())
            )
        else:
            score = (
                100.0 * geo_bins
                + 70.0 * func_bins
                + 30.0 * geo_sequences
                + 20.0 * func_sequences
                + 10.0 * np.sqrt(max(geo_gain, 0))
                + 6.0 * np.sqrt(max(func_gain, 0))
                + 25.0 * max(margin, 0.0)
                + 20.0 * promoted / max(observation_count, 1)
                - 60.0 * false_rate
                - float(reprojection[member_tensor].mean())
            )
        anchor_kind = 1 if geo_gain > 0 else 2
        if geo_gain <= 0 and (not include_identity_split or func_gain <= 0):
            continue
        clusters.append(
            {
                "cluster_id": int(root),
                "members": members,
                "xyz": xyz,
                "feature": feature,
                "source_group": source_group,
                "representative_row": int(representative_row),
                "anchor_kind": anchor_kind,
                "score": float(score),
                "geo_gain": geo_gain,
                "func_gain": func_gain,
                "geo_bins": geo_bins,
                "func_bins": func_bins,
                "false_incoming": false_incoming,
                "candidate_opportunities": candidate_opportunities,
                "promoted_correct": promoted,
                "observation_count": observation_count,
                "margin": margin,
                "track_count": len(members),
                "cluster_diameter_m": cluster_diameter,
                "cluster_min_descriptor_cosine": cluster_min_cosine,
                "same_query_collision_count": same_query_collisions,
                "covariance_trace": float(covariance[member_tensor].mean()),
                "reprojection_median_px": float(reprojection[member_tensor].mean()),
            }
        )
    clusters.sort(
        key=lambda value: (
            -value["score"],
            value["cluster_id"],
        )
    )
    selected_clusters = clusters[: max(int(budget), 0)]

    new_xyz = (
        torch.stack([value["xyz"] for value in selected_clusters])
        if selected_clusters
        else base_xyz.new_zeros((0, 3))
    )
    new_features = (
        torch.stack([value["feature"] for value in selected_clusters])
        if selected_clusters
        else base_features.new_zeros((0, base_features.shape[1]))
    )
    representative_rows = torch.as_tensor(
        [value["representative_row"] for value in selected_clusters],
        dtype=torch.long,
    )
    source_extension = (
        base_source_ids[representative_rows]
        if representative_rows.numel()
        else base_source_ids.new_zeros((0,))
    )
    total = int(base_source_ids.numel() + len(selected_clusters))
    anchor_type = torch.cat(
        (
            torch.zeros(base_source_ids.numel(), dtype=torch.int8),
            torch.as_tensor(
                [value["anchor_kind"] for value in selected_clusters],
                dtype=torch.int8,
            ),
        )
    )

    source_group_offsets = list(range(base_source_ids.numel() + 1))
    source_group_ids = base_source_ids.tolist()
    for value in selected_clusters:
        source_group_ids.extend(
            base_source_ids[
                torch.as_tensor(value["source_group"], dtype=torch.long)
            ].tolist()
        )
        source_group_offsets.append(len(source_group_ids))
    member_offsets = [0] * (base_source_ids.numel() + 1)
    member_ids = []
    for value in selected_clusters:
        member_ids.extend(value["members"])
        member_offsets.append(len(member_ids))
    output = {
        "version": 2,
        "schema": "lafgs_materialized_anchor_map",
        "anchor_ids": torch.arange(total, dtype=torch.long),
        "source_primitive_ids": torch.cat((base_source_ids, source_extension)),
        "track_cluster_ids": torch.cat(
            (
                torch.full((base_source_ids.numel(),), -1, dtype=torch.long),
                torch.as_tensor(
                    [value["cluster_id"] for value in selected_clusters],
                    dtype=torch.long,
                ),
            )
        ),
        "track_cluster_member_offsets": torch.as_tensor(
            member_offsets, dtype=torch.long
        ),
        "track_cluster_member_ids": torch.as_tensor(member_ids, dtype=torch.long),
        "source_group_offsets": torch.as_tensor(source_group_offsets, dtype=torch.long),
        "source_group_primitive_ids": torch.as_tensor(
            source_group_ids, dtype=torch.long
        ),
        "anchor_xyz": torch.cat((base_xyz, new_xyz)),
        "anchor_features": torch.cat((base_features, new_features)),
        "anchor_type": anchor_type,
        "base_anchor_count": int(base_source_ids.numel()),
        "requested_micro_anchor_budget": int(budget),
        "micro_anchor_count": len(selected_clusters),
        "config": {
            "method": "micro_anchor_v2_coverage_identity",
            "level_a_only": True,
            "add_only": True,
            "old_anchor_descriptors_frozen": True,
            "old_anchor_geometry_frozen": True,
            "raster_visibility_enabled": visibility_cache is not None,
            "include_identity_split": bool(include_identity_split),
            "score_mode": score_mode,
            "cluster_radius_m": float(cluster_radius_m),
            "cluster_min_descriptor_cosine": float(cluster_min_descriptor_cosine),
            "descriptor_trim_fraction": float(descriptor_trim_fraction),
            "coverage_radius_px": float(radius_px),
        },
        "micro_anchor_quality": {
            "coverage_gain": torch.as_tensor(
                [value["geo_gain"] for value in selected_clusters]
            ),
            "functional_gain": torch.as_tensor(
                [value["func_gain"] for value in selected_clusters]
            ),
            "coverage_view_bins": torch.as_tensor(
                [value["geo_bins"] for value in selected_clusters]
            ),
            "functional_view_bins": torch.as_tensor(
                [value["func_bins"] for value in selected_clusters]
            ),
            "positive_hardnegative_margin": torch.as_tensor(
                [value["margin"] for value in selected_clusters]
            ),
            "false_attractor_incoming_count": torch.as_tensor(
                [value["false_incoming"] for value in selected_clusters]
            ),
            "candidate_opportunity_count": torch.as_tensor(
                [value["candidate_opportunities"] for value in selected_clusters]
            ),
            "false_attractor_opportunity_rate": torch.as_tensor(
                [
                    value["false_incoming"] / max(value["candidate_opportunities"], 1)
                    for value in selected_clusters
                ]
            ),
            "promoted_correct_count": torch.as_tensor(
                [value["promoted_correct"] for value in selected_clusters]
            ),
            "observation_count": torch.as_tensor(
                [value["observation_count"] for value in selected_clusters]
            ),
            "covariance_trace_m2": torch.as_tensor(
                [value["covariance_trace"] for value in selected_clusters]
            ),
            "reprojection_median_px": torch.as_tensor(
                [value["reprojection_median_px"] for value in selected_clusters]
            ),
            "cluster_track_count": torch.as_tensor(
                [value["track_count"] for value in selected_clusters]
            ),
            "cluster_diameter_m": torch.as_tensor(
                [value["cluster_diameter_m"] for value in selected_clusters]
            ),
            "cluster_min_descriptor_cosine": torch.as_tensor(
                [value["cluster_min_descriptor_cosine"] for value in selected_clusters]
            ),
            "cluster_same_query_collision_count": torch.as_tensor(
                [value["same_query_collision_count"] for value in selected_clusters]
            ),
        },
    }
    diagnostics = {
        "base_anchor_count": int(base_source_ids.numel()),
        "eligible_level_a_track_count": int(track_indices.numel()),
        "cluster_count": len(members_by_root),
        "clustered_pair_count": clustered_pair_count,
        "eligible_cluster_pair_count": len(eligible_pairs),
        "rejected_single_linkage_pair_count": (rejected_single_linkage_pair_count),
        "multi_track_cluster_count": sum(
            len(value) > 1 for value in members_by_root.values()
        ),
        "cluster_diameter_max_m": max(
            (value["cluster_diameter_m"] for value in clusters),
            default=0.0,
        ),
        "cluster_same_query_collision_count": sum(
            value["same_query_collision_count"] for value in clusters
        ),
        "candidate_anchor_count": len(clusters),
        "selected_micro_anchor_count": len(selected_clusters),
        "selected_coverage_anchor_count": sum(
            value["anchor_kind"] == 1 for value in selected_clusters
        ),
        "selected_identity_split_anchor_count": sum(
            value["anchor_kind"] == 2 for value in selected_clusters
        ),
        "selected_source_primitive_count": int(torch.unique(source_extension).numel()),
        "coverage_gain_sum": sum(value["geo_gain"] for value in selected_clusters),
        "functional_gain_sum": sum(value["func_gain"] for value in selected_clusters),
        "raster_visibility_enabled": visibility_cache is not None,
        "score_mode": score_mode,
    }
    return output, diagnostics


def truncate_materialized_anchor_map(state: dict, micro_anchor_budget: int) -> dict:
    """Take a deterministic prefix of a scored materialized anchor map."""
    if state.get("schema") != "lafgs_materialized_anchor_map":
        raise ValueError("unsupported materialized anchor schema")
    base_count = int(state["base_anchor_count"])
    available = int(state["micro_anchor_count"])
    keep_micro = min(max(int(micro_anchor_budget), 0), available)
    keep_rows = base_count + keep_micro
    output = dict(state)
    row_fields = (
        "anchor_ids",
        "source_primitive_ids",
        "track_cluster_ids",
        "anchor_xyz",
        "anchor_features",
        "anchor_type",
    )
    for key in row_fields:
        if key in output:
            output[key] = torch.as_tensor(output[key])[:keep_rows].clone()
    quality = {}
    for key, value in state.get("micro_anchor_quality", {}).items():
        tensor = torch.as_tensor(value)
        quality[key] = tensor[:keep_micro].clone()
    output["micro_anchor_quality"] = quality
    for prefix, value_key in (
        ("track_cluster_member", "track_cluster_member_ids"),
        ("source_group", "source_group_primitive_ids"),
    ):
        offset_key = f"{prefix}_offsets"
        if offset_key not in state or value_key not in state:
            continue
        offsets = torch.as_tensor(state[offset_key], dtype=torch.long)
        end = int(offsets[keep_rows])
        output[offset_key] = offsets[: keep_rows + 1].clone()
        output[value_key] = torch.as_tensor(state[value_key])[:end].clone()
    output["anchor_ids"] = torch.arange(keep_rows, dtype=torch.long)
    output["requested_micro_anchor_budget"] = int(micro_anchor_budget)
    output["micro_anchor_count"] = keep_micro
    output["truncated_from_micro_anchor_count"] = available
    return output


def truncate_materialized_anchor_extension(state: dict, extension_budget: int) -> dict:
    """Keep a frozen canonical prefix and a deterministic extension prefix."""
    if state.get("schema") != "lafgs_materialized_anchor_map":
        raise ValueError("unsupported materialized anchor schema")
    if "canonical_anchor_count" not in state:
        raise ValueError("state does not define a canonical anchor prefix")
    canonical_count = int(state["canonical_anchor_count"])
    total_rows = int(torch.as_tensor(state["anchor_ids"]).numel())
    if canonical_count < 0 or canonical_count > total_rows:
        raise ValueError("canonical anchor count is outside the map")
    available = total_rows - canonical_count
    keep_extension = min(max(int(extension_budget), 0), available)
    keep_rows = canonical_count + keep_extension
    output = dict(state)
    row_fields = (
        "anchor_ids",
        "source_primitive_ids",
        "track_cluster_ids",
        "anchor_xyz",
        "anchor_features",
        "anchor_type",
    )
    for key in row_fields:
        if key in output:
            output[key] = torch.as_tensor(output[key])[:keep_rows].clone()
    output["full_prior_quality"] = {
        key: torch.as_tensor(value)[:keep_extension].clone()
        for key, value in state.get("full_prior_quality", {}).items()
    }
    for prefix, value_key in (
        ("source_group", "source_group_primitive_ids"),
        ("full_prior_source_group", "full_prior_source_group_primitive_ids"),
    ):
        offset_key = f"{prefix}_offsets"
        if offset_key not in state or value_key not in state:
            continue
        offsets = torch.as_tensor(state[offset_key], dtype=torch.long)
        if offsets.numel() == total_rows + 1:
            end = int(offsets[keep_rows])
            output[offset_key] = offsets[: keep_rows + 1].clone()
        elif offsets.numel() == available + 1:
            end = int(offsets[keep_extension])
            output[offset_key] = offsets[: keep_extension + 1].clone()
        else:
            raise ValueError(f"{offset_key} does not align with map rows")
        output[value_key] = torch.as_tensor(state[value_key])[:end].clone()
        for suffix in ("responsibilities", "costs"):
            aligned_key = f"{prefix}_{suffix}"
            if aligned_key in state:
                output[aligned_key] = torch.as_tensor(state[aligned_key])[:end].clone()
    base_count = int(state["base_anchor_count"])
    output["anchor_ids"] = torch.arange(keep_rows, dtype=torch.long)
    output["requested_extension_budget"] = int(extension_budget)
    output["selected_extension_count"] = keep_extension
    output["micro_anchor_count"] = keep_rows - base_count
    output["truncated_from_extension_count"] = available
    return output


def select_function_preserving_base_rows(
    *,
    base_source_primitive_ids: torch.Tensor,
    extension_source_primitive_ids: torch.Tensor,
    landmark_best_track_indices: torch.Tensor,
    visibility_counts: torch.Tensor,
    remove_count: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Retire unsupported base rows while preserving every tracked identity."""
    base_sources = torch.as_tensor(base_source_primitive_ids, dtype=torch.long).reshape(
        -1
    )
    extension_sources = torch.as_tensor(
        extension_source_primitive_ids, dtype=torch.long
    ).reshape(-1)
    best_tracks = torch.as_tensor(
        landmark_best_track_indices, dtype=torch.long
    ).reshape(-1)
    visibility = torch.as_tensor(visibility_counts, dtype=torch.long).reshape(-1)
    if not (base_sources.numel() == best_tracks.numel() == visibility.numel()):
        raise ValueError("base support tensors must be row-aligned")
    requested = max(int(remove_count), 0)
    unsupported = best_tracks < 0
    if requested > int(unsupported.sum()):
        raise ValueError(
            "requested compression would remove Track-First-supported rows"
        )
    child_sources = set(extension_sources.tolist())
    parent_redundant = torch.as_tensor(
        [int(source) in child_sources for source in base_sources.tolist()],
        dtype=torch.bool,
    )
    candidates = torch.nonzero(unsupported, as_tuple=False).reshape(-1)
    ordered = sorted(
        candidates.tolist(),
        key=lambda row: (
            0 if bool(parent_redundant[row]) else 1,
            int(visibility[row]),
            int(row),
        ),
    )
    removed = torch.as_tensor(ordered[:requested], dtype=torch.long)
    keep_mask = torch.ones(base_sources.numel(), dtype=torch.bool)
    keep_mask[removed] = False
    kept = torch.nonzero(keep_mask, as_tuple=False).reshape(-1)
    diagnostics = {
        "requested_remove_count": requested,
        "removed_count": int(removed.numel()),
        "unsupported_candidate_count": int(unsupported.sum()),
        "removed_supported_count": int((best_tracks[removed] >= 0).sum()),
        "removed_parent_redundant_count": int(parent_redundant[removed].sum()),
        "removed_visibility_count_mean": float(
            visibility[removed].float().mean() if removed.numel() else 0.0
        ),
        "removed_visibility_count_max": int(
            visibility[removed].max() if removed.numel() else 0
        ),
    }
    return kept, removed, diagnostics


def select_micro_anchor_set(
    *,
    candidate_gap_observations: list[list[int]],
    candidate_functional_gap_observations: list[list[int]] | None = None,
    observation_query_indices: torch.Tensor,
    query_sequence_indices: torch.Tensor,
    budget: int,
    profile: str,
    false_attractor_rates: torch.Tensor | None = None,
    false_attractor_costs: torch.Tensor | None = None,
    false_attractor_penalty: float = 0.25,
    functional_gap_weight: float = 0.0,
    minimum_marginal_gain: float = float("-inf"),
    initial_selected_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Lazy-greedy selection over geometric and functional gap observations."""
    if profile not in {"unique_gap", "query_saturated", "sequence_tail"}:
        raise ValueError(f"unsupported marginal coverage profile: {profile}")
    candidate_count = len(candidate_gap_observations)
    observation_query_indices = torch.as_tensor(
        observation_query_indices, dtype=torch.long
    ).reshape(-1)
    query_sequence_indices = torch.as_tensor(
        query_sequence_indices, dtype=torch.long
    ).reshape(-1)
    query_count = int(query_sequence_indices.numel())
    if observation_query_indices.numel():
        if int(observation_query_indices.min()) < 0:
            raise ValueError("observation query indices must be non-negative")
        if int(observation_query_indices.max()) >= query_count:
            raise ValueError("observation query index exceeds query metadata")
    if false_attractor_rates is None:
        false_attractor_rates = torch.zeros(candidate_count)
    false_attractor_rates = torch.as_tensor(
        false_attractor_rates, dtype=torch.float32
    ).reshape(-1)
    if false_attractor_rates.numel() != candidate_count:
        raise ValueError("false-attractor rates must align with candidates")
    if false_attractor_costs is not None:
        false_attractor_costs = torch.as_tensor(
            false_attractor_costs, dtype=torch.float32
        ).reshape(-1)
        if false_attractor_costs.numel() != candidate_count:
            raise ValueError("false-attractor costs must align with candidates")
    if candidate_functional_gap_observations is None:
        candidate_functional_gap_observations = [[] for _ in range(candidate_count)]
    if len(candidate_functional_gap_observations) != candidate_count:
        raise ValueError("functional gap lists must align with candidates")

    observation_count = int(observation_query_indices.numel())
    covered = torch.zeros(observation_count, dtype=torch.bool)
    functional_covered = torch.zeros(observation_count, dtype=torch.bool)
    selected_query_count = torch.zeros(query_count, dtype=torch.long)
    sequence_count = (
        int(query_sequence_indices.max()) + 1 if query_sequence_indices.numel() else 0
    )
    sequence_event_count = torch.zeros(sequence_count, dtype=torch.float32)
    if observation_count:
        observation_sequences = query_sequence_indices[observation_query_indices]
        sequence_event_count.scatter_add_(
            0,
            observation_sequences,
            torch.ones(observation_count),
        )
    else:
        observation_sequences = torch.zeros(0, dtype=torch.long)
    positive_sequence_counts = sequence_event_count[sequence_event_count > 0]
    sequence_reference = float(
        positive_sequence_counts.mean() if positive_sequence_counts.numel() else 1.0
    )
    sequence_weight = (
        (sequence_reference / sequence_event_count.clamp_min(1.0))
        .sqrt()
        .clamp(0.5, 3.0)
    )
    query_alpha = 0.0 if profile == "unique_gap" else 0.5
    query_first_bonus = 0.0 if profile == "unique_gap" else 1.0

    candidate_observations = []
    for observations in candidate_gap_observations:
        tensor = torch.unique(
            torch.as_tensor(observations, dtype=torch.long), sorted=True
        )
        if tensor.numel() and (
            int(tensor.min()) < 0 or int(tensor.max()) >= observation_count
        ):
            raise ValueError("candidate references an invalid observation")
        candidate_observations.append(tensor)
    candidate_functional_observations = []
    for observations in candidate_functional_gap_observations:
        tensor = torch.unique(
            torch.as_tensor(observations, dtype=torch.long), sorted=True
        )
        if tensor.numel() and (
            int(tensor.min()) < 0 or int(tensor.max()) >= observation_count
        ):
            raise ValueError("candidate references an invalid functional observation")
        candidate_functional_observations.append(tensor)

    def marginal_gain(candidate: int) -> float:
        observations = candidate_observations[candidate]
        observations = observations[~covered[observations]]
        if observations.numel() == 0:
            gain = 0.0
        else:
            queries = observation_query_indices[observations]
            saturation = (selected_query_count[queries].float() + 1.0).pow(-query_alpha)
            weight = saturation
            if profile == "sequence_tail":
                weight = weight * sequence_weight[observation_sequences[observations]]
            gain = float(weight.sum())
            if query_first_bonus:
                unique_queries = torch.unique(queries)
                gain += query_first_bonus * float(
                    (selected_query_count[unique_queries] == 0).sum()
                )
        functional_observations = candidate_functional_observations[candidate]
        functional_observations = functional_observations[
            ~functional_covered[functional_observations]
        ]
        if functional_observations.numel():
            functional_queries = observation_query_indices[functional_observations]
            functional_saturation = (
                selected_query_count[functional_queries].float() + 1.0
            ).pow(-query_alpha)
            functional_gain = float(functional_saturation.sum())
            if profile == "sequence_tail":
                functional_gain = float(
                    (
                        functional_saturation
                        * sequence_weight[
                            observation_sequences[functional_observations]
                        ]
                    ).sum()
                )
            gain += float(functional_gap_weight) * functional_gain
        if false_attractor_costs is None:
            harmful_incoming = float(false_attractor_rates[candidate]) * max(
                int(candidate_observations[candidate].numel()), 1
            )
        else:
            harmful_incoming = float(false_attractor_costs[candidate])
        false_cost = float(false_attractor_penalty) * harmful_incoming
        return gain - false_cost

    def apply_selection(candidate: int) -> None:
        observations = candidate_observations[candidate]
        new_observations = observations[~covered[observations]]
        if new_observations.numel():
            covered[new_observations] = True
            selected_query_count.scatter_add_(
                0,
                observation_query_indices[new_observations],
                torch.ones(new_observations.numel(), dtype=torch.long),
            )
        functional_observations = candidate_functional_observations[candidate]
        new_functional = functional_observations[
            ~functional_covered[functional_observations]
        ]
        if new_functional.numel():
            functional_covered[new_functional] = True

    heap = [
        (-marginal_gain(candidate), candidate, 0)
        for candidate in range(candidate_count)
    ]
    heapq.heapify(heap)
    if initial_selected_indices is None:
        initial_selected_indices = torch.zeros(0, dtype=torch.long)
    initial_selected_indices = torch.as_tensor(
        initial_selected_indices, dtype=torch.long
    ).reshape(-1)
    stable_initial = []
    seen_initial = set()
    for candidate in initial_selected_indices.tolist():
        if candidate not in seen_initial:
            stable_initial.append(candidate)
            seen_initial.add(candidate)
    initial_selected_indices = torch.as_tensor(stable_initial, dtype=torch.long)
    if initial_selected_indices.numel() and (
        int(initial_selected_indices.min()) < 0
        or int(initial_selected_indices.max()) >= candidate_count
    ):
        raise ValueError("initial selected index exceeds candidate count")
    selection_gain = []
    revision = 0
    target = min(max(int(budget), 0), candidate_count)
    initial_selected_indices = initial_selected_indices[:target]
    selected = initial_selected_indices.tolist()
    selected_mask = torch.zeros(candidate_count, dtype=torch.bool)
    if selected:
        selected_mask[initial_selected_indices] = True
        for candidate in selected:
            apply_selection(candidate)
            revision += 1
    first_nonpositive_rank = None
    while len(selected) < target and heap:
        _, candidate, evaluated_revision = heapq.heappop(heap)
        if bool(selected_mask[candidate]):
            continue
        gain = marginal_gain(candidate)
        if evaluated_revision != revision:
            heapq.heappush(heap, (-gain, candidate, revision))
            continue
        if gain <= float(minimum_marginal_gain):
            first_nonpositive_rank = len(selected)
            break
        selected.append(candidate)
        selection_gain.append(gain)
        selected_mask[candidate] = True
        apply_selection(candidate)
        revision += 1

    selected_tensor = torch.as_tensor(selected, dtype=torch.long)
    terminal_gain = max(
        (
            marginal_gain(candidate)
            for candidate in range(candidate_count)
            if not bool(selected_mask[candidate])
        ),
        default=float("-inf"),
    )
    diagnostics = {
        "profile": profile,
        "requested_budget": int(budget),
        "selected_count": len(selected),
        "initial_selected_count": int(initial_selected_indices.numel()),
        "greedy_selected_count": len(selection_gain),
        "covered_gap_observation_count": int(covered.sum()),
        "covered_functional_gap_observation_count": int(functional_covered.sum()),
        "covered_query_count": int((selected_query_count > 0).sum()),
        "selection_gain_sum": float(sum(selection_gain)),
        "selection_gain_min": float(min(selection_gain, default=0.0)),
        "selection_gain_curve": [float(value) for value in selection_gain],
        "minimum_marginal_gain": float(minimum_marginal_gain),
        "automatic_stop_triggered": first_nonpositive_rank is not None,
        "first_nonpositive_gain_rank": first_nonpositive_rank,
        "best_remaining_marginal_gain": float(terminal_gain),
        "positive_gain_candidate_count": int(
            sum(
                marginal_gain(candidate) > float(minimum_marginal_gain)
                for candidate in range(candidate_count)
                if not bool(selected_mask[candidate])
            )
        ),
        "zero_or_negative_gain_candidate_count": int(
            sum(
                marginal_gain(candidate) <= float(minimum_marginal_gain)
                for candidate in range(candidate_count)
                if not bool(selected_mask[candidate])
            )
        ),
        "selected_false_attractor_rate_mean": float(
            false_attractor_rates[selected_tensor].mean()
            if selected_tensor.numel()
            else 0.0
        ),
        "false_attractor_penalty": float(false_attractor_penalty),
        "false_attractor_cost_mode": (
            "expected_harmful_incoming"
            if false_attractor_costs is not None
            else "rate_times_geometric_gap_count"
        ),
        "functional_gap_weight": float(functional_gap_weight),
    }
    return selected_tensor, diagnostics
