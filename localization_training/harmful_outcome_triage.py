"""Diagnose why selected harmful correspondences lack a safe replacement."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

import torch


RANK_FAILURE = 0
TEACHER_MISS = 1
COVERAGE_FAILURE = 2
ACTIVE_GEOMETRY_FAILURE = 3
INTRINSICALLY_UNMATCHABLE = 4

CATEGORY_NAMES = (
    "active_map_rank_failure",
    "teacher_miss",
    "coverage_failure",
    "active_geometry_failure",
    "intrinsically_unmatchable",
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


def _surface_samples(
    cached: dict,
    query_rows: torch.Tensor,
    *,
    raster_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = torch.as_tensor(query_rows).long().reshape(-1)
    keypoints = torch.as_tensor(cached["native_keypoints"]).float()[rows]
    height, width = map(int, cached["native_input_hw"])
    pixels = torch.floor(keypoints).long()
    x = pixels[:, 0].clamp(0, width - 1)
    y = pixels[:, 1].clamp(0, height - 1)
    depth = torch.as_tensor(cached["native_depth"]).float()[y, x]
    if "native_alpha" in cached:
        alpha = torch.as_tensor(cached["native_alpha"]).float()[y, x]
    elif raster_valid is not None:
        alpha = torch.as_tensor(raster_valid).float().reshape(-1)[rows]
    else:
        raise ValueError(
            "query cache lacks native_alpha and no raster validity was supplied"
        )
    keypoints = keypoints + float(cached.get("pixel_center_offset", 0.5))
    return keypoints, depth, alpha


def project_depth_legal_candidates(
    *,
    xyz: torch.Tensor,
    pose_w2c: torch.Tensor,
    K: torch.Tensor,
    keypoints: torch.Tensor,
    rendered_depth: torch.Tensor,
    rendered_alpha: torch.Tensor,
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
        "query_bins": torch.as_tensor(track_payload["query_bins"]).long()[
            target_to_source
        ],
    }


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
    if raster_provenance is not None:
        if names != list(raster_provenance["query_names"]):
            raise ValueError("raster provenance query registry differs")
        raster_records = raster_provenance["records"]
        if len(raster_records) != len(names):
            raise ValueError("raster provenance record count differs")
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
        if raster_records is not None:
            raster_record = raster_records[query_index]
            raster_rows = torch.as_tensor(raster_record["query_rows"]).long()
            raster_valid = torch.zeros(
                len(cached["native_keypoints"]), dtype=torch.bool
            )
            raster_valid[raster_rows] = torch.as_tensor(
                raster_record["valid"]
            ).bool()
        keypoints, rendered_depth, rendered_alpha = _surface_samples(
            cached,
            harmful_rows,
            raster_valid=raster_valid,
        )
        if len(harmful_rows):
            active_errors, active_legal = project_depth_legal_candidates(
                xyz=active_xyz_device,
                pose_w2c=cached["pose_w2c"],
                K=cached["native_K"],
                keypoints=keypoints,
                rendered_depth=rendered_depth,
                rendered_alpha=rendered_alpha,
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
            best_track = stable_observations[0] if stable_observations else (
                int(observations[0]) if observations else -1
            )
            track_indices.append(best_track)
            track_stable.append(bool(stable_observations))
            if len(verified):
                category = RANK_FAILURE
                action = REPRESENTATION_REPAIR
            elif len(active_candidates[local]):
                category = TEACHER_MISS
                action = REPRESENTATION_REPAIR
            else:
                unresolved.append(local)
                category = INTRINSICALLY_UNMATCHABLE
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
                track_index = track_indices[local]
                stable = bool(track_stable[local])
                active_track_anchor = (
                    track["active_by_track"].get(int(track_index), ())
                    if track_index >= 0
                    else ()
                )
                assigned = (
                    int(track["canonical_by_track"][track_index])
                    if track_index >= 0
                    else -1
                )
                canonical_track_available = 0 <= assigned < canonical_count
                if stable and active_track_anchor:
                    categories[local] = ACTIVE_GEOMETRY_FAILURE
                    actions[local] = GEOMETRY_REPAIR
                elif (
                    len(pool_candidates[offset])
                    or (stable and canonical_track_available)
                ):
                    categories[local] = COVERAGE_FAILURE
                    actions[local] = STRUCTURE_REPAIR

        active_offsets, active_indices = _pack_ragged(
            active_candidates, dtype=torch.long
        )
        _, active_errors_packed = _pack_ragged(
            active_candidate_errors, dtype=torch.float32
        )
        canonical_offsets, canonical_indices = _pack_ragged(
            canonical_candidates, dtype=torch.long
        )
        _, canonical_errors_packed = _pack_ragged(
            canonical_errors, dtype=torch.float32
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
        )
        intrinsic = category_tensor == INTRINSICALLY_UNMATCHABLE
        totals["intrinsic_invalid_surface_support"] += int(
            (intrinsic & ~surface_support_valid).sum()
        )
        totals["intrinsic_valid_surface_support"] += int(
            (intrinsic & surface_support_valid).sum()
        )
        observation_count_tensor = torch.as_tensor(
            track_observation_counts, dtype=torch.int16
        )
        totals["intrinsic_with_track_observation"] += int(
            (intrinsic & (observation_count_tensor > 0)).sum()
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
                "canonical_positive_offsets": canonical_offsets,
                "canonical_positive_indices": canonical_indices,
                "canonical_positive_reprojection_errors_px": canonical_errors_packed,
                "track_indices": torch.as_tensor(track_indices, dtype=torch.long),
                "track_stable": torch.as_tensor(track_stable, dtype=torch.bool),
                "track_observation_count": observation_count_tensor,
                "surface_support_valid": surface_support_valid.cpu(),
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
        "schema": "lafgs_harmful_outcome_triage",
        "version": 1,
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
            "intrinsically_unmatchable_breakdown": {
                "invalid_surface_support": int(
                    totals["intrinsic_invalid_surface_support"]
                ),
                "valid_surface_support": int(
                    totals["intrinsic_valid_surface_support"]
                ),
                "with_track_observation": int(
                    totals["intrinsic_with_track_observation"]
                ),
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
        "surface_support_source": (
            "native_alpha"
            if all(
                "native_alpha" in cache[name]
                for name in names
            )
            else "raster_provenance_valid"
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
