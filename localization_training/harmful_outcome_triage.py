"""Diagnose why selected harmful correspondences lack a safe replacement."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

import torch


RANK_FAILURE = 0
TEACHER_MISS = 1
COVERAGE_FAILURE = 2
ACTIVE_TRACK_INCONSISTENCY = 3
UNRESOLVED_NO_VERIFIED_TARGET = 4

# Backward-compatible aliases for V26 artifacts and downstream readers.
ACTIVE_GEOMETRY_FAILURE = ACTIVE_TRACK_INCONSISTENCY
INTRINSICALLY_UNMATCHABLE = UNRESOLVED_NO_VERIFIED_TARGET

CATEGORY_NAMES = (
    "active_map_rank_failure",
    "teacher_miss",
    "coverage_failure",
    "active_track_geometry_identity_inconsistency",
    "unresolved_no_verified_target",
)

REPRESENTATION_REPAIR = 0
STRUCTURE_REPAIR = 1
GEOMETRY_REPAIR = 2
SELECTOR_REJECT = 3

ACTION_NAMES = (
    "representation_repair",
    "structure_repair",
    "geometry_repair",
    "selector_reject",
)

UNRESOLVED_NO_STABLE_TRACK = 0
UNRESOLVED_SURFACE_UNCERTAIN = 1
UNRESOLVED_TRACK_IDENTITY_OR_VISIBILITY = 2
UNRESOLVED_NO_MAP_SUPPORT = 3

UNRESOLVED_REASON_NAMES = (
    "no_stable_3d_track",
    "depth_or_provenance_uncertain",
    "track_identity_or_visibility_inconsistent",
    "no_verified_map_support",
)


@dataclass(frozen=True)
class HarmfulTriageConfig:
    strict_radius_px: float = 2.0
    ambiguous_radius_px: float = 6.0
    depth_abs_tolerance_m: float = 0.05
    depth_rel_tolerance: float = 0.02
    alpha_minimum: float = 0.01
    maximum_candidates_per_row: int = 8
    minimum_track_views: int = 3
    minimum_track_view_bins: int = 2
    maximum_track_reprojection_p90_px: float = 4.0
    minimum_contribution_mass: float = 0.02
    maximum_depth_std_abs_m: float = 0.05
    maximum_depth_std_relative: float = 0.02
    geometry_xyz_threshold_m: float = 0.02
    geometry_reprojection_improvement_px: float = 1.0


def _positive_lookup(record: dict) -> dict[int, torch.Tensor]:
    rows = torch.as_tensor(record["query_rows"]).long().reshape(-1)
    offsets = torch.as_tensor(record["positive_offsets"]).long().reshape(-1)
    indices = torch.as_tensor(record["positive_indices"]).long().reshape(-1)
    if len(offsets) != len(rows) + 1 or int(offsets[-1]) != len(indices):
        raise ValueError("positive teacher CSR is malformed")
    return {
        int(row): indices[int(offsets[index]) : int(offsets[index + 1])]
        for index, row in enumerate(rows.tolist())
    }


def _pack_ragged(
    values: list[torch.Tensor],
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    counts = torch.as_tensor([len(value) for value in values], dtype=torch.long)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    if not values or int(offsets[-1]) == 0:
        return offsets, torch.empty(0, dtype=dtype)
    return offsets, torch.cat(
        [torch.as_tensor(value, dtype=dtype).reshape(-1) for value in values]
    )


def _bilinear_samples(
    image: torch.Tensor,
    keypoints: torch.Tensor,
    *,
    positive_only: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a scalar image and expose local interpolation uncertainty."""

    image = torch.as_tensor(image).float().squeeze()
    points = torch.as_tensor(keypoints).float().reshape(-1, 2)
    if image.ndim != 2:
        raise ValueError("surface image must be scalar HxW")
    height, width = image.shape
    x = points[:, 0].clamp(0, max(width - 1, 0))
    y = points[:, 1].clamp(0, max(height - 1, 0))
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = (x0 + 1).clamp(max=width - 1)
    y1 = (y0 + 1).clamp(max=height - 1)
    dx = x - x0.float()
    dy = y - y0.float()
    values = torch.stack(
        (
            image[y0, x0],
            image[y0, x1],
            image[y1, x0],
            image[y1, x1],
        ),
        dim=1,
    )
    weights = torch.stack(
        (
            (1.0 - dx) * (1.0 - dy),
            dx * (1.0 - dy),
            (1.0 - dx) * dy,
            dx * dy,
        ),
        dim=1,
    )
    valid = torch.isfinite(values)
    if positive_only:
        valid &= values > 0
    weights = weights * valid.float()
    weight_sum = weights.sum(dim=1)
    mean = (weights * values.nan_to_num()).sum(dim=1) / weight_sum.clamp_min(
        1e-8
    )
    variance = (
        weights * (values.nan_to_num() - mean[:, None]).square()
    ).sum(dim=1) / weight_sum.clamp_min(1e-8)
    mean[weight_sum <= 0] = float("nan")
    variance[weight_sum <= 0] = float("inf")
    return mean, variance.sqrt(), weight_sum


def _surface_samples(
    cached: dict,
    query_rows: torch.Tensor,
    *,
    raster_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = torch.as_tensor(query_rows).long().reshape(-1)
    raw_keypoints = torch.as_tensor(cached["native_keypoints"]).float()[rows]
    depth, depth_std, _ = _bilinear_samples(
        cached["native_depth"], raw_keypoints, positive_only=True
    )
    if "native_alpha" in cached:
        alpha, _, _ = _bilinear_samples(
            cached["native_alpha"], raw_keypoints, positive_only=False
        )
    elif raster_valid is not None:
        alpha = torch.as_tensor(raster_valid).float().reshape(-1)[rows]
    else:
        raise ValueError(
            "query cache lacks native_alpha and no raster validity was supplied"
        )
    keypoints = raw_keypoints + float(cached.get("pixel_center_offset", 0.5))
    return keypoints, depth, alpha, depth_std


def _padded_anchor_sources(
    offsets: torch.Tensor,
    primitive_ids: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.as_tensor(offsets).long()
    primitive_ids = torch.as_tensor(primitive_ids).long()
    weights = torch.as_tensor(weights).float()
    counts = offsets[1:] - offsets[:-1]
    width = max(int(counts.max()) if counts.numel() else 0, 1)
    padded_ids = torch.full((len(counts), width), -1, dtype=torch.long)
    padded_weights = torch.zeros((len(counts), width))
    for anchor, count in enumerate(counts.tolist()):
        if count:
            start = int(offsets[anchor])
            padded_ids[anchor, :count] = primitive_ids[start : start + count]
            padded_weights[anchor, :count] = weights[start : start + count]
    return padded_ids, padded_weights


def candidate_provenance_mass(
    raster_primitive_ids: torch.Tensor,
    raster_contribution_mass: torch.Tensor,
    source_ids: torch.Tensor,
    source_weights: torch.Tensor,
    *,
    device: torch.device,
    anchor_chunk_size: int = 2048,
) -> torch.Tensor:
    """Compute keypoint-to-anchor source responsibility mass."""

    raster_ids = torch.as_tensor(raster_primitive_ids).long().to(device)
    raster_mass = torch.as_tensor(raster_contribution_mass).float().to(device)
    source_ids = torch.as_tensor(source_ids).long().to(device)
    source_weights = torch.as_tensor(source_weights).float().to(device)
    if raster_ids.shape != raster_mass.shape:
        raise ValueError("raster primitive IDs and mass must align")
    if source_ids.shape != source_weights.shape:
        raise ValueError("anchor source IDs and weights must align")
    output = torch.zeros(
        (raster_ids.shape[0], source_ids.shape[0]),
        dtype=torch.float32,
        device=device,
    )
    for start in range(0, source_ids.shape[0], int(anchor_chunk_size)):
        end = min(start + int(anchor_chunk_size), source_ids.shape[0])
        matches = (
            source_ids[None, start:end, :, None]
            == raster_ids[:, None, None, :]
        )
        output[:, start:end] = (
            matches.float()
            * source_weights[None, start:end, :, None]
            * raster_mass[:, None, None, :]
        ).sum(dim=(-1, -2))
    return output


def project_depth_legal_candidates(
    *,
    xyz: torch.Tensor,
    pose_w2c: torch.Tensor,
    K: torch.Tensor,
    keypoints: torch.Tensor,
    rendered_depth: torch.Tensor,
    rendered_alpha: torch.Tensor,
    rendered_depth_std: torch.Tensor | None = None,
    provenance_mass: torch.Tensor | None = None,
    config: HarmfulTriageConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-row reprojection errors and strict depth-valid masks."""

    xyz = torch.as_tensor(xyz).float().to(device)
    pose = torch.as_tensor(pose_w2c).float().to(device)
    intrinsics = torch.as_tensor(K).float().to(device)
    points = torch.as_tensor(keypoints).float().to(device)
    reference_depth = torch.as_tensor(rendered_depth).float().to(device)
    alpha = torch.as_tensor(rendered_alpha).float().to(device)
    depth_std = (
        torch.zeros_like(reference_depth)
        if rendered_depth_std is None
        else torch.as_tensor(rendered_depth_std).float().to(device)
    )
    camera = xyz @ pose[:3, :3].T + pose[:3, 3]
    depth = camera[:, 2]
    projected = camera @ intrinsics.T
    uv = projected[:, :2] / depth[:, None].clamp_min(1e-8)
    errors = torch.linalg.norm(points[:, None, :] - uv[None, :, :], dim=2)
    tolerance = float(config.depth_abs_tolerance_m) + (
        float(config.depth_rel_tolerance) * reference_depth.abs()
    )
    valid_row = (
        torch.isfinite(reference_depth)
        & (reference_depth > 0)
        & torch.isfinite(alpha)
        & (alpha >= float(config.alpha_minimum))
        & torch.isfinite(depth_std)
        & (
            depth_std
            <= float(config.maximum_depth_std_abs_m)
            + float(config.maximum_depth_std_relative)
            * reference_depth.abs()
        )
    )
    valid_anchor = torch.isfinite(depth) & (depth > 0)
    depth_legal = (
        valid_row[:, None]
        & valid_anchor[None, :]
        & (
            (depth[None, :] - reference_depth[:, None]).abs()
            <= tolerance[:, None]
        )
    )
    finite = torch.isfinite(errors) & depth_legal
    if provenance_mass is not None:
        mass = torch.as_tensor(provenance_mass).float().to(device)
        if mass.shape != finite.shape:
            raise ValueError("candidate provenance mass must align")
        finite &= mass >= float(config.minimum_contribution_mass)
    return errors, finite


def _best_candidates(
    errors: torch.Tensor,
    legal: torch.Tensor,
    *,
    radius_px: float,
    maximum: int,
    excluded: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    masked = torch.where(
        legal & (errors <= float(radius_px)),
        errors,
        torch.full_like(errors, float("inf")),
    )
    if excluded is not None and masked.numel():
        rows = torch.arange(masked.shape[0], device=masked.device)
        excluded = torch.as_tensor(excluded, device=masked.device).long()
        valid = (excluded >= 0) & (excluded < masked.shape[1])
        masked[rows[valid], excluded[valid]] = float("inf")
    candidate_values: list[torch.Tensor] = []
    error_values: list[torch.Tensor] = []
    maximum = max(int(maximum), 0)
    for row in range(masked.shape[0]):
        indices = torch.where(torch.isfinite(masked[row]))[0]
        if len(indices):
            order = torch.argsort(masked[row, indices], stable=True)
            indices = indices[order[:maximum]]
        candidate_values.append(indices.cpu())
        error_values.append(masked[row, indices].cpu())
    return candidate_values, error_values


def build_track_evidence(
    track_payload: dict,
    *,
    target_query_names: list[str],
    active_track_ids: torch.Tensor,
    config: HarmfulTriageConfig,
) -> dict:
    """Build exact observation and cross-view stability lookups."""

    source_names = list(track_payload["query_names"])
    source_by_name = {name: index for index, name in enumerate(source_names)}
    target_to_source = torch.as_tensor(
        [source_by_name[name] for name in target_query_names], dtype=torch.long
    )
    source_to_target = torch.full(
        (len(source_names),), -1, dtype=torch.long
    )
    source_to_target[target_to_source] = torch.arange(len(target_query_names))
    tracks = track_payload["tracks"]
    geometry = track_payload["track_geometry"]
    track_count = len(geometry["triangulated"])
    def geometry_value(name: str, default, dtype=None):
        if name in geometry:
            return torch.as_tensor(geometry[name], dtype=dtype)
        return torch.full((track_count,), default, dtype=dtype)

    stable = (
        torch.as_tensor(geometry["triangulated"]).bool()
        & (
            torch.as_tensor(geometry["triangulation_distinct_view_count"]).long()
            >= int(config.minimum_track_views)
        )
        & (
            torch.as_tensor(
                geometry["triangulation_distinct_view_bin_count"]
            ).long()
            >= int(config.minimum_track_view_bins)
        )
        & torch.isfinite(
            torch.as_tensor(geometry["triangulation_reprojection_p90_px"]).float()
        )
        & (
            torch.as_tensor(
                geometry["triangulation_reprojection_p90_px"]
            ).float()
            <= float(config.maximum_track_reprojection_p90_px)
        )
    )
    observation: dict[tuple[int, int], list[int]] = defaultdict(list)
    for track, query, keypoint in zip(
        torch.as_tensor(tracks["track_index"]).long().tolist(),
        torch.as_tensor(tracks["query_index"]).long().tolist(),
        torch.as_tensor(tracks["keypoint_index"]).long().tolist(),
    ):
        target_query = int(source_to_target[int(query)])
        if target_query >= 0:
            observation[(target_query, int(keypoint))].append(int(track))
    active_by_track: dict[int, list[int]] = defaultdict(list)
    for anchor, track in enumerate(
        torch.as_tensor(active_track_ids).long().tolist()
    ):
        if int(track) >= 0:
            active_by_track[int(track)].append(int(anchor))
    assigned = torch.as_tensor(
        track_payload["assignment"]["track_landmark_index"]
    ).long()
    if len(assigned) != track_count:
        raise ValueError("track assignment and geometry registries differ")
    return {
        "observation": observation,
        "stable": stable,
        "active_by_track": active_by_track,
        "canonical_by_track": assigned,
        "triangulated_xyz": (
            torch.as_tensor(
                geometry.get(
                    "triangulated_xyz",
                    torch.full((track_count, 3), float("nan")),
                )
            )
            .float()
            .reshape(track_count, 3)
        ),
        "high_confidence": geometry_value(
            "triangulation_high_confidence", False, torch.bool
        ),
        "view_count": geometry_value(
            "triangulation_distinct_view_count", 0, torch.long
        ),
        "view_bin_count": geometry_value(
            "triangulation_distinct_view_bin_count", 0, torch.long
        ),
        "reprojection_p90": geometry_value(
            "triangulation_reprojection_p90_px", float("inf"), torch.float32
        ),
        "parallax": geometry_value(
            "triangulation_parallax_deg", 0.0, torch.float32
        ),
        "covariance_trace": geometry_value(
            "triangulation_covariance_trace", float("inf"), torch.float32
        ),
        "query_bins": torch.as_tensor(track_payload["query_bins"]).long()[
            target_to_source
        ],
    }


def _track_quality_order(
    track_indices: list[int],
    track: dict,
) -> list[int]:
    """Order tracks by geometric support without depending on payload order."""

    return sorted(
        (int(value) for value in track_indices),
        key=lambda value: (
            -int(track["high_confidence"][value]),
            -int(track["view_bin_count"][value]),
            -int(track["view_count"][value]),
            float(track["reprojection_p90"][value]),
            float(track["covariance_trace"][value]),
            -float(track["parallax"][value]),
            value,
        ),
    )


def _project_one(
    xyz: torch.Tensor,
    pose_w2c: torch.Tensor,
    K: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xyz = torch.as_tensor(xyz).float().reshape(-1, 3)
    pose = torch.as_tensor(pose_w2c).float()
    intrinsic = torch.as_tensor(K).float()
    camera = xyz @ pose[:3, :3].T + pose[:3, 3]
    projected = camera @ intrinsic.T
    uv = projected[:, :2] / camera[:, 2:3].clamp_min(1e-8)
    return uv, camera[:, 2]


def _verified_geometry_track(
    *,
    stable_tracks: list[int],
    track: dict,
    active_xyz: torch.Tensor,
    keypoint: torch.Tensor,
    pose_w2c: torch.Tensor,
    K: torch.Tensor,
    config: HarmfulTriageConfig,
) -> tuple[int, int, float, float] | None:
    """Return a track/anchor pair only when geometry discrepancy is explicit."""

    best = None
    for track_index in _track_quality_order(stable_tracks, track):
        triangulated = track["triangulated_xyz"][track_index]
        if not bool(torch.isfinite(triangulated).all()):
            continue
        tri_uv, tri_depth = _project_one(triangulated[None], pose_w2c, K)
        if float(tri_depth[0]) <= 0:
            continue
        tri_error = float(torch.linalg.norm(tri_uv[0] - keypoint))
        for anchor in track["active_by_track"].get(track_index, ()):
            anchor_xyz = active_xyz[int(anchor)]
            anchor_uv, anchor_depth = _project_one(
                anchor_xyz[None], pose_w2c, K
            )
            if float(anchor_depth[0]) <= 0:
                continue
            distance = float(torch.linalg.norm(anchor_xyz - triangulated))
            improvement = float(
                torch.linalg.norm(anchor_uv[0] - keypoint) - tri_error
            )
            if (
                distance >= float(config.geometry_xyz_threshold_m)
                and improvement
                >= float(config.geometry_reprojection_improvement_px)
            ):
                candidate = (
                    int(track_index),
                    int(anchor),
                    distance,
                    improvement,
                )
                if best is None or (improvement, distance) > (
                    best[3],
                    best[2],
                ):
                    best = candidate
    return best


def _verified_teacher_candidates(
    *,
    query_row: int,
    top1: int,
    lookup: dict[int, torch.Tensor],
    errors: torch.Tensor,
    legal: torch.Tensor,
    radius_px: float,
) -> tuple[torch.Tensor, int]:
    candidates = lookup.get(int(query_row), torch.empty(0, dtype=torch.long))
    candidates = torch.as_tensor(candidates).long()
    candidates = candidates[
        (candidates >= 0)
        & (candidates < errors.shape[0])
        & (candidates != int(top1))
    ]
    if not len(candidates):
        return candidates, 0
    device_candidates = candidates.to(errors.device)
    valid = (
        legal[device_candidates]
        & torch.isfinite(errors[device_candidates])
        & (errors[device_candidates] <= float(radius_px))
    )
    return candidates[valid.cpu()], int((~valid).sum())


def triage_harmful_outcomes(
    *,
    active_map: dict,
    canonical_map: dict,
    selected_outcomes: dict,
    dynamic_outcomes: dict,
    active_positive_teacher: dict,
    query_cache: dict,
    track_payload: dict,
    raster_provenance: dict | None = None,
    config: HarmfulTriageConfig,
    device: torch.device,
    progress=None,
) -> tuple[dict, dict]:
    """Classify harmful rows and rebuild their verified active positives."""

    names = list(selected_outcomes["query_names"])
    if names != list(dynamic_outcomes["query_names"]) or names != list(
        active_positive_teacher["query_names"]
    ):
        raise ValueError("harmful triage query registries differ")
    active_xyz = torch.as_tensor(active_map["anchor_xyz"]).float()
    canonical_xyz = torch.as_tensor(canonical_map["anchor_xyz"]).float()
    active_count = len(active_xyz)
    canonical_count = len(canonical_xyz)
    if int(selected_outcomes["anchor_count"]) != active_count:
        raise ValueError("selected outcomes do not align with the active map")
    if int(active_positive_teacher["anchor_count"]) != active_count:
        raise ValueError("positive teacher does not align with the active map")
    cache = query_cache.get("queries", query_cache)
    raster_records = None
    active_source_ids = active_source_weights = None
    canonical_source_ids = canonical_source_weights = None
    if raster_provenance is not None:
        if names != list(raster_provenance["query_names"]):
            raise ValueError("raster provenance query registry differs")
        raster_records = raster_provenance["records"]
        if len(raster_records) != len(names):
            raise ValueError("raster provenance record count differs")
        source_offsets = torch.as_tensor(
            raster_provenance["anchor_source_offsets"]
        ).long()
        if len(source_offsets) != active_count + 1:
            raise ValueError("raster provenance anchor registry differs")
        active_source_ids, active_source_weights = _padded_anchor_sources(
            source_offsets,
            raster_provenance["anchor_source_primitive_ids"],
            raster_provenance["anchor_source_weights"],
        )
        active_primary = torch.as_tensor(
            active_map["source_primitive_ids"]
        ).long()
        if not bool(
            (
                active_source_ids
                == active_primary[:, None]
            ).any(dim=1).all()
        ):
            raise ValueError(
                "raster provenance source lineage differs from active map"
            )
        canonical_source_ids = torch.as_tensor(
            canonical_map["source_primitive_ids"]
        ).long()[:, None]
        canonical_source_weights = torch.ones_like(
            canonical_source_ids, dtype=torch.float32
        )
    track = build_track_evidence(
        track_payload,
        target_query_names=names,
        active_track_ids=active_map["track_cluster_ids"],
        config=config,
    )
    active_xyz_device = active_xyz.to(device)
    canonical_xyz_device = canonical_xyz.to(device)
    totals = Counter()
    by_sequence: dict[str, Counter] = defaultdict(Counter)
    by_view_bin: dict[int, Counter] = defaultdict(Counter)
    output_records = []
    completed_records = []

    for query_index, name in enumerate(names):
        selected_record = selected_outcomes["records"][query_index]
        dynamic_record = dynamic_outcomes["records"][query_index]
        positive_record = active_positive_teacher["records"][query_index]
        rows = torch.as_tensor(selected_record["query_rows"]).long()
        if not torch.equal(rows, torch.as_tensor(dynamic_record["query_rows"]).long()):
            raise ValueError("harmful triage dynamic rows differ")
        if not torch.equal(rows, torch.as_tensor(positive_record["query_rows"]).long()):
            raise ValueError("harmful triage positive rows differ")
        selected = torch.as_tensor(selected_record["selected_row_mask"]).bool()
        harmful = torch.as_tensor(dynamic_record["harmful_inlier_mask"]).bool()
        harmful_positions = torch.where(selected & harmful)[0]
        top1_all = torch.as_tensor(
            selected_record["topk_anchor_indices"]
        ).long()[:, 0]
        top1 = top1_all[harmful_positions]
        harmful_rows = rows[harmful_positions]
        cached = cache[name]
        raster_valid = None
        active_provenance_mass = None
        canonical_provenance_mass = None
        if raster_records is not None:
            raster_record = raster_records[query_index]
            raster_rows = torch.as_tensor(raster_record["query_rows"]).long()
            if int(raster_record["query_index"]) != query_index:
                raise ValueError("raster provenance query order differs")
            raster_valid = torch.zeros(
                len(cached["native_keypoints"]), dtype=torch.bool
            )
            raster_valid[raster_rows] = torch.as_tensor(
                raster_record["valid"]
            ).bool()
            raster_lookup = torch.full(
                (len(cached["native_keypoints"]),), -1, dtype=torch.long
            )
            raster_lookup[raster_rows] = torch.arange(len(raster_rows))
            raster_positions = raster_lookup[harmful_rows]
            if bool((raster_positions < 0).any()):
                raise ValueError(
                    "harmful rows are absent from raster provenance"
                )
            primitive_ids = torch.as_tensor(
                raster_record["primitive_ids"]
            ).long()[raster_positions]
            contribution_mass = torch.as_tensor(
                raster_record["contribution_mass"]
            ).float()[raster_positions]
            active_provenance_mass = candidate_provenance_mass(
                primitive_ids,
                contribution_mass,
                active_source_ids,
                active_source_weights,
                device=device,
            )
            canonical_provenance_mass = candidate_provenance_mass(
                primitive_ids,
                contribution_mass,
                canonical_source_ids,
                canonical_source_weights,
                device=device,
            )
        keypoints, rendered_depth, rendered_alpha, rendered_depth_std = (
            _surface_samples(
            cached,
            harmful_rows,
            raster_valid=raster_valid,
        )
        )
        depth_uncertain = (
            ~torch.isfinite(rendered_depth_std)
            | (
                rendered_depth_std
                > float(config.maximum_depth_std_abs_m)
                + float(config.maximum_depth_std_relative)
                * rendered_depth.abs()
            )
        )
        if len(harmful_rows):
            active_errors, active_legal = project_depth_legal_candidates(
                xyz=active_xyz_device,
                pose_w2c=cached["pose_w2c"],
                K=cached["native_K"],
                keypoints=keypoints,
                rendered_depth=rendered_depth,
                rendered_alpha=rendered_alpha,
                rendered_depth_std=rendered_depth_std,
                provenance_mass=active_provenance_mass,
                config=config,
                device=device,
            )
            active_candidates, active_candidate_errors = _best_candidates(
                active_errors,
                active_legal,
                radius_px=config.strict_radius_px,
                maximum=config.maximum_candidates_per_row,
                excluded=top1,
            )
        else:
            active_errors = torch.empty((0, active_count), device=device)
            active_legal = torch.empty(
                (0, active_count), dtype=torch.bool, device=device
            )
            active_candidates, active_candidate_errors = [], []
        teacher_lookup = _positive_lookup(positive_record)
        categories = []
        actions = []
        original_counts = []
        invalid_teacher_counts = []
        track_indices = []
        track_stable = []
        track_observation_counts = []
        stable_track_values: list[list[int]] = []
        geometry_anchor_indices = []
        geometry_xyz_residuals = []
        geometry_reprojection_improvements = []
        unresolved_reasons = []
        canonical_candidates: list[torch.Tensor] = [
            torch.empty(0, dtype=torch.long) for _ in harmful_rows
        ]
        canonical_errors: list[torch.Tensor] = [
            torch.empty(0) for _ in harmful_rows
        ]
        unresolved = []

        for local, (row, wrong) in enumerate(
            zip(harmful_rows.tolist(), top1.tolist())
        ):
            original = torch.as_tensor(
                teacher_lookup.get(int(row), torch.empty(0, dtype=torch.long))
            ).long()
            original_counts.append(int((original != int(wrong)).sum()))
            verified, invalid_count = _verified_teacher_candidates(
                query_row=int(row),
                top1=int(wrong),
                lookup=teacher_lookup,
                errors=active_errors[local],
                legal=active_legal[local],
                radius_px=config.strict_radius_px,
            )
            invalid_teacher_counts.append(invalid_count)
            observations = track["observation"].get(
                (query_index, int(row)), []
            )
            track_observation_counts.append(len(observations))
            stable_observations = [
                int(value)
                for value in observations
                if bool(track["stable"][int(value)])
            ]
            stable_observations = _track_quality_order(
                stable_observations, track
            )
            best_track = (
                stable_observations[0]
                if stable_observations
                else (int(observations[0]) if observations else -1)
            )
            stable_track_values.append(stable_observations)
            track_indices.append(best_track)
            track_stable.append(bool(stable_observations))
            geometry_anchor_indices.append(-1)
            geometry_xyz_residuals.append(float("nan"))
            geometry_reprojection_improvements.append(float("nan"))
            unresolved_reasons.append(UNRESOLVED_NO_MAP_SUPPORT)
            if len(verified):
                category = RANK_FAILURE
                action = REPRESENTATION_REPAIR
            elif len(active_candidates[local]):
                category = TEACHER_MISS
                action = REPRESENTATION_REPAIR
            else:
                unresolved.append(local)
                category = UNRESOLVED_NO_VERIFIED_TARGET
                action = SELECTOR_REJECT
            categories.append(category)
            actions.append(action)

        if unresolved:
            unresolved_tensor = torch.as_tensor(unresolved).long()
            canonical_full_errors, canonical_legal = (
                project_depth_legal_candidates(
                    xyz=canonical_xyz_device,
                    pose_w2c=cached["pose_w2c"],
                    K=cached["native_K"],
                    keypoints=keypoints[unresolved_tensor],
                    rendered_depth=rendered_depth[unresolved_tensor],
                    rendered_alpha=rendered_alpha[unresolved_tensor],
                    rendered_depth_std=rendered_depth_std[
                        unresolved_tensor
                    ],
                    provenance_mass=(
                        canonical_provenance_mass[unresolved_tensor]
                        if canonical_provenance_mass is not None
                        else None
                    ),
                    config=config,
                    device=device,
                )
            )
            pool_candidates, pool_errors = _best_candidates(
                canonical_full_errors,
                canonical_legal,
                radius_px=config.strict_radius_px,
                maximum=config.maximum_candidates_per_row,
            )
            for offset, local in enumerate(unresolved):
                canonical_candidates[local] = pool_candidates[offset]
                canonical_errors[local] = pool_errors[offset]
                stable_tracks = stable_track_values[local]
                if bool(depth_uncertain[local]) or not bool(
                    raster_valid[int(harmful_rows[local])]
                    if raster_valid is not None
                    else True
                ):
                    unresolved_reasons[local] = (
                        UNRESOLVED_SURFACE_UNCERTAIN
                    )
                    continue
                geometry = _verified_geometry_track(
                    stable_tracks=stable_tracks,
                    track=track,
                    active_xyz=active_xyz,
                    keypoint=keypoints[local],
                    pose_w2c=cached["pose_w2c"],
                    K=cached["native_K"],
                    config=config,
                )
                if geometry is not None:
                    track_index, anchor, distance, improvement = geometry
                    track_indices[local] = track_index
                    geometry_anchor_indices[local] = anchor
                    geometry_xyz_residuals[local] = distance
                    geometry_reprojection_improvements[local] = improvement
                    categories[local] = ACTIVE_TRACK_INCONSISTENCY
                    actions[local] = GEOMETRY_REPAIR
                    continue
                active_track_available = any(
                    track["active_by_track"].get(track_index, ())
                    for track_index in stable_tracks
                )
                canonical_track_available = any(
                    0
                    <= int(track["canonical_by_track"][track_index])
                    < canonical_count
                    for track_index in stable_tracks
                )
                if (
                    len(pool_candidates[offset])
                    or (
                        bool(stable_tracks)
                        and not active_track_available
                        and canonical_track_available
                    )
                ):
                    categories[local] = COVERAGE_FAILURE
                    actions[local] = STRUCTURE_REPAIR
                elif bool(stable_tracks) and active_track_available:
                    unresolved_reasons[local] = (
                        UNRESOLVED_TRACK_IDENTITY_OR_VISIBILITY
                    )
                elif bool(stable_tracks):
                    categories[local] = COVERAGE_FAILURE
                    actions[local] = STRUCTURE_REPAIR
                elif track_observation_counts[local] > 0:
                    unresolved_reasons[local] = UNRESOLVED_NO_STABLE_TRACK
                else:
                    unresolved_reasons[local] = UNRESOLVED_NO_MAP_SUPPORT

        active_candidate_mass = [
            (
                active_provenance_mass[local, candidates].detach().cpu()
                if active_provenance_mass is not None and len(candidates)
                else torch.ones(len(candidates))
            )
            for local, candidates in enumerate(active_candidates)
        ]
        canonical_candidate_mass = [
            (
                canonical_provenance_mass[local, candidates].detach().cpu()
                if canonical_provenance_mass is not None and len(candidates)
                else torch.ones(len(candidates))
            )
            for local, candidates in enumerate(canonical_candidates)
        ]
        active_offsets, active_indices = _pack_ragged(
            active_candidates, dtype=torch.long
        )
        _, active_errors_packed = _pack_ragged(
            active_candidate_errors, dtype=torch.float32
        )
        _, active_mass_packed = _pack_ragged(
            active_candidate_mass, dtype=torch.float32
        )
        canonical_offsets, canonical_indices = _pack_ragged(
            canonical_candidates, dtype=torch.long
        )
        _, canonical_errors_packed = _pack_ragged(
            canonical_errors, dtype=torch.float32
        )
        _, canonical_mass_packed = _pack_ragged(
            canonical_candidate_mass, dtype=torch.float32
        )
        category_tensor = torch.as_tensor(categories, dtype=torch.int8)
        action_tensor = torch.as_tensor(actions, dtype=torch.int8)
        for category in categories:
            label = CATEGORY_NAMES[int(category)]
            totals[label] += 1
            by_sequence[name.split("/", 1)[0]][label] += 1
            by_view_bin[int(track["query_bins"][query_index])][label] += 1
        surface_support_valid = (
            torch.isfinite(rendered_depth)
            & (rendered_depth > 0)
            & torch.isfinite(rendered_alpha)
            & (rendered_alpha >= float(config.alpha_minimum))
            & ~depth_uncertain
        )
        unresolved_mask = (
            category_tensor == UNRESOLVED_NO_VERIFIED_TARGET
        )
        totals["unresolved_invalid_surface_support"] += int(
            (unresolved_mask & ~surface_support_valid).sum()
        )
        totals["unresolved_valid_surface_support"] += int(
            (unresolved_mask & surface_support_valid).sum()
        )
        observation_count_tensor = torch.as_tensor(
            track_observation_counts, dtype=torch.int16
        )
        totals["unresolved_with_track_observation"] += int(
            (unresolved_mask & (observation_count_tensor > 0)).sum()
        )
        unresolved_reason_tensor = torch.as_tensor(
            unresolved_reasons, dtype=torch.int8
        )
        for reason, label in enumerate(UNRESOLVED_REASON_NAMES):
            totals[f"unresolved_reason_{label}"] += int(
                (
                    unresolved_mask
                    & (unresolved_reason_tensor == reason)
                ).sum()
            )
        totals["selected_harmful"] += len(categories)
        totals["invalid_original_alternative"] += sum(invalid_teacher_counts)
        output_records.append(
            {
                "query_index": int(query_index),
                "query_name": name,
                "query_rows": harmful_rows,
                "selected_row_positions": harmful_positions,
                "top1_anchor_indices": top1,
                "category": category_tensor,
                "action": action_tensor,
                "original_alternative_count": torch.as_tensor(
                    original_counts, dtype=torch.int16
                ),
                "invalid_original_alternative_count": torch.as_tensor(
                    invalid_teacher_counts, dtype=torch.int16
                ),
                "active_positive_offsets": active_offsets,
                "active_positive_indices": active_indices,
                "active_positive_reprojection_errors_px": active_errors_packed,
                "active_positive_contribution_mass": active_mass_packed,
                "canonical_positive_offsets": canonical_offsets,
                "canonical_positive_indices": canonical_indices,
                "canonical_positive_reprojection_errors_px": canonical_errors_packed,
                "canonical_positive_contribution_mass": canonical_mass_packed,
                "track_indices": torch.as_tensor(track_indices, dtype=torch.long),
                "track_stable": torch.as_tensor(track_stable, dtype=torch.bool),
                "track_observation_count": observation_count_tensor,
                "surface_support_valid": surface_support_valid.cpu(),
                "depth_uncertainty_std_m": rendered_depth_std.cpu(),
                "unresolved_reason": unresolved_reason_tensor,
                "geometry_anchor_indices": torch.as_tensor(
                    geometry_anchor_indices, dtype=torch.long
                ),
                "geometry_xyz_residual_m": torch.as_tensor(
                    geometry_xyz_residuals, dtype=torch.float32
                ),
                "geometry_reprojection_improvement_px": torch.as_tensor(
                    geometry_reprojection_improvements,
                    dtype=torch.float32,
                ),
            }
        )

        completed_positive_values = []
        positive_lookup = _positive_lookup(positive_record)
        harmful_by_row = {
            int(row): local for local, row in enumerate(harmful_rows.tolist())
        }
        for row in rows.tolist():
            local = harmful_by_row.get(int(row))
            if local is None:
                completed_positive_values.append(
                    torch.as_tensor(
                        positive_lookup.get(
                            int(row), torch.empty(0, dtype=torch.long)
                        )
                    ).long().unique(sorted=True)
                )
            else:
                completed_positive_values.append(
                    torch.as_tensor(active_candidates[local])
                    .long()
                    .unique(sorted=True)
                )
        completed_offsets, completed_indices = _pack_ragged(
            completed_positive_values, dtype=torch.long
        )
        totals["completed_positive_rows"] += int(
            ((completed_offsets[1:] - completed_offsets[:-1]) > 0).sum()
        )
        totals["completed_positive_pairs"] += int(len(completed_indices))
        completed_records.append(
            {
                **positive_record,
                "positive_offsets": completed_offsets,
                "positive_indices": completed_indices,
            }
        )
        if progress is not None:
            progress(query_index + 1, len(names), dict(totals))

    category_counts = {
        name: int(totals[name]) for name in CATEGORY_NAMES
    }
    total_harmful = max(int(totals["selected_harmful"]), 1)
    triage = {
        "schema": "lafgs_harmful_outcome_triage_v2",
        "version": 2,
        "query_names": names,
        "active_anchor_count": active_count,
        "canonical_anchor_count": canonical_count,
        "category_names": list(CATEGORY_NAMES),
        "action_names": list(ACTION_NAMES),
        "records": output_records,
        "summary": {
            "selected_harmful": int(totals["selected_harmful"]),
            "category_counts": category_counts,
            "category_percent": {
                name: 100.0 * value / total_harmful
                for name, value in category_counts.items()
            },
            "invalid_original_alternative_count": int(
                totals["invalid_original_alternative"]
            ),
            "unresolved_breakdown": {
                "invalid_surface_support": int(
                    totals["unresolved_invalid_surface_support"]
                ),
                "valid_surface_support": int(
                    totals["unresolved_valid_surface_support"]
                ),
                "with_track_observation": int(
                    totals["unresolved_with_track_observation"]
                ),
                "reasons": {
                    label: int(totals[f"unresolved_reason_{label}"])
                    for label in UNRESOLVED_REASON_NAMES
                },
            },
            "by_sequence": {
                key: dict(value) for key, value in sorted(by_sequence.items())
            },
            "by_view_bin": {
                str(key): dict(value)
                for key, value in sorted(by_view_bin.items())
            },
        },
        "config": asdict(config),
        "unresolved_reason_names": list(UNRESOLVED_REASON_NAMES),
        "surface_support_source": (
            "bilinear_depth_alpha_plus_anchor_raster_contribution"
            if raster_provenance is not None
            else "bilinear_native_depth_alpha"
        ),
    }
    completed = {
        **active_positive_teacher,
        "schema": "lafgs_verified_completed_active_positive_teacher",
        "version": 2,
        "records": completed_records,
        "diagnostics": {
            **dict(active_positive_teacher.get("diagnostics", {})),
            "positive_rows": int(totals["completed_positive_rows"]),
            "strong_pair_count": int(totals["completed_positive_pairs"]),
            "source_diagnostics": dict(
                active_positive_teacher.get("diagnostics", {})
            ),
            "harmful_completion": {
                "selected_harmful": int(totals["selected_harmful"]),
                "rank_or_teacher_miss_rows": int(
                    totals["active_map_rank_failure"]
                    + totals["teacher_miss"]
                ),
            },
        },
        "triage_summary": triage["summary"],
        "config": {
            **dict(active_positive_teacher.get("config", {})),
            "harmful_completion": asdict(config),
        },
    }
    return triage, completed
