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
    observation_gap = torch.zeros(
        tracks["track_index"].numel(), dtype=torch.bool
    )
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
        tree = (
            cKDTree(projected[positive_depth])
            if positive_indices.size
            else None
        )
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
                observation_gap[observation] = True
                query_index = int(tracks["query_index"][observation])
                gap_view_bins[track].add(
                    int(torch.as_tensor(payload["query_bins"])[query_index])
                )
                gap_sequences[track].add(
                    str(query_names[query_index]).split("/", 1)[0]
                )
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
        "raster_visibility_enabled": torch.tensor(
            visibility_cache is not None
        ),
    }


def _track_observation_lookup(payload: dict) -> dict[int, list[int]]:
    observations = defaultdict(list)
    for observation, track in enumerate(
        payload["tracks"]["track_index"].tolist()
    ):
        observations[int(track)].append(observation)
    return observations


def fuse_track_descriptors(
    *,
    payload: dict,
    query_cache: dict,
    track_indices: torch.Tensor,
    trim_fraction: float = 0.2,
) -> torch.Tensor:
    """Fuse every requested track independently using its native observations."""
    cache = query_cache.get("queries", query_cache)
    query_names = payload["query_names"]
    tracks = payload["tracks"]
    query_bins = torch.as_tensor(payload["query_bins"], dtype=torch.long)
    observation_by_track = _track_observation_lookup(payload)
    features = []
    for track in torch.as_tensor(track_indices, dtype=torch.long).tolist():
        observations = observation_by_track[int(track)]
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
        features.append(
            robust_fuse_track_descriptors(
                descriptors,
                query_bins[query_indices],
                torch.as_tensor(tracks["confidence"])[observations],
                trim_fraction=trim_fraction,
            )
        )
    if not features:
        descriptor_dim = int(
            torch.as_tensor(
                next(iter(cache.values()))["native_descriptors"]
            ).shape[1]
        )
        return torch.zeros((0, descriptor_dim), dtype=torch.float32)
    return torch.stack(features)


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
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Measure rank gaps and false-attractor behavior against the frozen bank."""
    cache = query_cache.get("queries", query_cache)
    query_names = payload["query_names"]
    tracks = payload["tracks"]
    track_indices = torch.as_tensor(track_indices, dtype=torch.long).reshape(-1)
    track_features = F.normalize(
        torch.as_tensor(track_features).float(), dim=1
    )
    if track_features.shape[0] != track_indices.numel():
        raise ValueError("track indices and fused features must align")
    track_count = int(payload["track_geometry"]["triangulated_xyz"].shape[0])
    track_to_row = torch.full((track_count,), -1, dtype=torch.long)
    track_to_row[track_indices] = torch.arange(track_indices.numel())
    selected = track_to_row[tracks["track_index"].long()] >= 0
    selected_observations = torch.nonzero(
        selected, as_tuple=False
    ).reshape(-1)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    old_xyz = torch.as_tensor(base_xyz).float().to(device)
    old_features = F.normalize(
        torch.as_tensor(base_features).float(), dim=1
    ).to(device)
    candidate_features = track_features.to(device)

    observation_gap = torch.zeros(
        tracks["track_index"].numel(), dtype=torch.bool
    )
    observation_old_best = torch.full(
        (tracks["track_index"].numel(),), -torch.inf
    )
    observation_self_score = torch.full_like(observation_old_best, -torch.inf)
    observation_candidate_best = torch.full_like(
        observation_old_best, -torch.inf
    )
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
            torch.as_tensor(cached["native_descriptors"]).float()[
                keypoint_index
            ],
            dim=1,
        ).to(device)
        old_score, old_index = (descriptors @ old_features.T).max(dim=1)
        projected, depth = _project_points(
            old_xyz[old_index].cpu(),
            cached["native_K"],
            cached["pose_w2c"],
        )
        physical = (
            torch.as_tensor(cached["native_keypoints"]).float()[keypoint_index]
            + float(cached.get("pixel_center_offset", 0.5))
        ).numpy()
        clean = (depth > 0) & (
            np.linalg.norm(projected - physical, axis=1) <= float(radius_px)
        )
        target_row = track_to_row[
            tracks["track_index"][observation_tensor].long()
        ].to(device)
        candidate_score = descriptors @ candidate_features.T
        best_score, best_row = candidate_score.max(dim=1)
        self_score = candidate_score.gather(
            1, target_row[:, None]
        ).squeeze(1)
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
    promoted_correct = torch.zeros(track_count, dtype=torch.long)
    query_bins = torch.as_tensor(payload["query_bins"], dtype=torch.long)
    for observation in selected_observations.tolist():
        track = int(tracks["track_index"][observation])
        query = int(tracks["query_index"][observation])
        positive_count[track] += 1
        positive_margin_sum[track] += (
            observation_self_score[observation]
            - observation_old_best[observation]
        )
        if bool(observation_gap[observation]):
            functional_gain[track] += 1
            functional_bins[track].add(int(query_bins[query]))
            functional_sequences[track].add(
                str(query_names[query]).split("/", 1)[0]
            )
        predicted_row = int(observation_candidate_row[observation])
        target_row = int(track_to_row[track])
        beats_old = bool(
            observation_candidate_best[observation]
            > observation_old_best[observation]
        )
        if beats_old and predicted_row == target_row:
            promoted_correct[track] += 1
        elif beats_old and predicted_row >= 0:
            predicted_track = int(track_indices[predicted_row])
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
        "promoted_correct_count": promoted_correct,
        "observation_count": positive_count,
        "observation_old_best_score": observation_old_best,
        "observation_self_score": observation_self_score,
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
    track_count = int(track_xyz.shape[0])
    candidate_mask = (
        torch.as_tensor(
            geometry["triangulation_high_confidence"], dtype=torch.bool
        )
        & (
            torch.as_tensor(
                geometry["track_confidence_level"], dtype=torch.int8
            )
            == 2
        )
        & (group_offsets[1:] > group_offsets[:-1])
    )
    track_indices = torch.nonzero(
        candidate_mask, as_tuple=False
    ).reshape(-1)
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
        else:
            parent[left_root] = right_root

    selected_xyz_np = track_xyz[track_indices].numpy()
    spatial_pairs = cKDTree(selected_xyz_np).query_pairs(
        r=float(cluster_radius_m)
    )
    clustered_pair_count = 0
    for left_row, right_row in sorted(spatial_pairs):
        left = int(track_indices[left_row])
        right = int(track_indices[right_row])
        if source_groups[left].isdisjoint(source_groups[right]):
            continue
        cosine = float(
            torch.dot(feature_by_track[left], feature_by_track[right])
        )
        if cosine < float(cluster_min_descriptor_cosine):
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
    covariance = torch.as_tensor(
        geometry["triangulation_covariance_trace"]
    ).float()
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
        xyz = (
            track_xyz[member_tensor] * weight[:, None]
        ).sum(dim=0)
        observations = [
            observation
            for member in members
            for observation in observation_by_track[member]
        ]
        observation_tensor = torch.as_tensor(observations, dtype=torch.long)
        observation_queries = tracks["query_index"][
            observation_tensor
        ].long()
        observation_keypoints = tracks["keypoint_index"][
            observation_tensor
        ].long()
        descriptors = torch.stack(
            [
                torch.as_tensor(
                    cache[query_names[int(query)]]["native_descriptors"]
                )[int(keypoint)]
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
                if bool(
                    coverage["coverage_gap_observation_mask"][observation]
                )
            }
        )
        func_bins = len(
            {
                int(query_bins[int(tracks["query_index"][observation])])
                for observation in observations
                if bool(
                    functional["functional_gap_observation_mask"][observation]
                )
            }
        )
        geo_sequences = len(
            {
                str(
                    query_names[
                        int(tracks["query_index"][observation])
                    ]
                ).split("/", 1)[0]
                for observation in observations
                if bool(
                    coverage["coverage_gap_observation_mask"][observation]
                )
            }
        )
        func_sequences = len(
            {
                str(
                    query_names[
                        int(tracks["query_index"][observation])
                    ]
                ).split("/", 1)[0]
                for observation in observations
                if bool(
                    functional["functional_gap_observation_mask"][observation]
                )
            }
        )
        false_incoming = int(
            functional["false_attractor_incoming_count"][member_tensor].sum()
        )
        promoted = int(
            functional["promoted_correct_count"][member_tensor].sum()
        )
        observation_count = int(
            functional["observation_count"][member_tensor].sum()
        )
        margin = float(
            (
                functional["positive_hardnegative_margin_mean"][
                    member_tensor
                ]
                * functional["observation_count"][member_tensor]
            ).sum()
            / functional["observation_count"][member_tensor].sum().clamp_min(1)
        )
        false_rate = false_incoming / max(observation_count, 1)
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
        if geo_gain <= 0 and (
            not include_identity_split or func_gain <= 0
        ):
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
                "promoted_correct": promoted,
                "observation_count": observation_count,
                "margin": margin,
                "covariance_trace": float(
                    covariance[member_tensor].mean()
                ),
                "reprojection_median_px": float(
                    reprojection[member_tensor].mean()
                ),
            }
        )
    clusters.sort(
        key=lambda value: (
            -value["score"],
            value["cluster_id"],
        )
    )
    selected_clusters = clusters[: max(int(budget), 0)]

    new_xyz = torch.stack(
        [value["xyz"] for value in selected_clusters]
    ) if selected_clusters else base_xyz.new_zeros((0, 3))
    new_features = torch.stack(
        [value["feature"] for value in selected_clusters]
    ) if selected_clusters else base_features.new_zeros(
        (0, base_features.shape[1])
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
        "source_primitive_ids": torch.cat(
            (base_source_ids, source_extension)
        ),
        "track_cluster_ids": torch.cat(
            (
                torch.full(
                    (base_source_ids.numel(),), -1, dtype=torch.long
                ),
                torch.as_tensor(
                    [value["cluster_id"] for value in selected_clusters],
                    dtype=torch.long,
                ),
            )
        ),
        "track_cluster_member_offsets": torch.as_tensor(
            member_offsets, dtype=torch.long
        ),
        "track_cluster_member_ids": torch.as_tensor(
            member_ids, dtype=torch.long
        ),
        "source_group_offsets": torch.as_tensor(
            source_group_offsets, dtype=torch.long
        ),
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
            "cluster_min_descriptor_cosine": float(
                cluster_min_descriptor_cosine
            ),
            "descriptor_trim_fraction": float(
                descriptor_trim_fraction
            ),
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
                [
                    value["reprojection_median_px"]
                    for value in selected_clusters
                ]
            ),
        },
    }
    diagnostics = {
        "base_anchor_count": int(base_source_ids.numel()),
        "eligible_level_a_track_count": int(track_indices.numel()),
        "cluster_count": len(members_by_root),
        "clustered_pair_count": clustered_pair_count,
        "multi_track_cluster_count": sum(
            len(value) > 1 for value in members_by_root.values()
        ),
        "candidate_anchor_count": len(clusters),
        "selected_micro_anchor_count": len(selected_clusters),
        "selected_coverage_anchor_count": sum(
            value["anchor_kind"] == 1 for value in selected_clusters
        ),
        "selected_identity_split_anchor_count": sum(
            value["anchor_kind"] == 2 for value in selected_clusters
        ),
        "selected_source_primitive_count": int(
            torch.unique(source_extension).numel()
        ),
        "coverage_gain_sum": sum(
            value["geo_gain"] for value in selected_clusters
        ),
        "functional_gain_sum": sum(
            value["func_gain"] for value in selected_clusters
        ),
        "raster_visibility_enabled": visibility_cache is not None,
        "score_mode": score_mode,
    }
    return output, diagnostics


def truncate_materialized_anchor_map(
    state: dict, micro_anchor_budget: int
) -> dict:
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
