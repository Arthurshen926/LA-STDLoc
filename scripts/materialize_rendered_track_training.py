#!/usr/bin/env python3
"""Materialize train-only labels for a rendered-RGB Track map.

The teacher never reads source mapping RGB or test queries.  Exact multi-view
Track observations provide identity supervision independently of Gaussian
depth.  Rendered alpha/depth are optional support evidence for projection-only
nearby anchors: compatible projections are ignored as negatives, but they are
not promoted to identity positives.  Track xyz always remains ray-triangulated.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
        if reloaded.get("schema") != payload.get("schema"):
            raise RuntimeError("temporary artifact schema did not reload")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sequence_name(image_name: str) -> str:
    return str(image_name).split("/", maxsplit=1)[0]


def _mapping_groups(
    names: list[str], single_trajectory_pose_cells: int
) -> tuple[list[str], list[str], str]:
    """Assign all mapping views to sequence-balanced descriptor groups."""
    if not names:
        raise ValueError("mapping query registry is empty")
    sequences = [_sequence_name(name) for name in names]
    unique = sorted(set(sequences))
    if len(unique) >= 2:
        return sequences, unique, "mapping_trajectory"
    cell_count = int(single_trajectory_pose_cells)
    if cell_count < 1:
        raise ValueError("single-trajectory pose-cell count must be positive")
    groups = [
        f"pose_cell_{min(index * cell_count // len(names), cell_count - 1):02d}"
        for index in range(len(names))
    ]
    return groups, sorted(set(groups)), "contiguous_mapping_pose_cells"


def _project(xyz: torch.Tensor, intrinsic: torch.Tensor, pose: torch.Tensor):
    camera = xyz @ pose[:3, :3].T + pose[:3, 3]
    depth = camera[:, 2]
    uvw = camera @ intrinsic.T
    uv = uvw[:, :2] / depth[:, None].clamp_min(1e-8)
    return uv, depth


def _spatial_candidates(
    projected: torch.Tensor,
    valid: torch.Tensor,
    keypoints: torch.Tensor,
    radius: float,
) -> list[list[int]]:
    cell_size = float(radius)
    valid_rows = torch.nonzero(valid, as_tuple=False).reshape(-1)
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    projected_cell = torch.floor(projected[valid_rows] / cell_size).long()
    for anchor, cell in zip(valid_rows.tolist(), projected_cell.tolist()):
        cells[(int(cell[0]), int(cell[1]))].append(int(anchor))
    result = []
    query_cells = torch.floor(keypoints / cell_size).long()
    radius_squared = float(radius) ** 2
    for keypoint, cell in zip(keypoints, query_cells.tolist()):
        candidates = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                candidates.extend(cells.get((cell[0] + dx, cell[1] + dy), ()))
        if candidates:
            rows = torch.as_tensor(candidates, dtype=torch.long)
            distance_squared = ((projected[rows] - keypoint) ** 2).sum(dim=1)
            rows = rows[distance_squared <= radius_squared]
            result.append(sorted(set(int(value) for value in rows.tolist())))
        else:
            result.append([])
    return result


def _csr(rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = [0]
    values = []
    for row in rows:
        values.extend(sorted(set(int(value) for value in row)))
        offsets.append(len(values))
    return torch.as_tensor(offsets).long(), torch.as_tensor(values).long()


def materialize(
    *,
    anchor_map_path: Path,
    track_payload_path: Path,
    query_cache_path: Path,
    output_dir: Path,
    strong_radius_px: float,
    ambiguous_radius_px: float,
    single_trajectory_pose_cells: int = 3,
    alpha_minimum: float = 0.05,
    depth_abs_tolerance_m: float = 0.05,
    depth_relative_tolerance: float = 0.02,
    scene_calibration_path: Path | None = None,
    expected_scene_calibration_sha256: str | None = None,
) -> dict:
    if float(strong_radius_px) <= 0 or float(ambiguous_radius_px) <= float(
        strong_radius_px
    ):
        raise ValueError("teacher radii must satisfy 0 < strong < ambiguous")
    state = torch.load(anchor_map_path, map_location="cpu", weights_only=False)
    payload = torch.load(track_payload_path, map_location="cpu", weights_only=False)
    cache_payload = torch.load(query_cache_path, map_location="cpu", weights_only=False)
    if cache_payload.get("uses_source_mapping_rgb") is not False:
        raise ValueError("training cache is not rendered-RGB-only")
    if cache_payload.get("uses_test_queries") is not False:
        raise ValueError("training cache contains test queries")
    cache = cache_payload["queries"]
    query_cache_sha256 = sha256_file(query_cache_path)
    names = list(payload["query_names"])
    if names != list(cache):
        raise ValueError("Track payload and rendered cache query order differs")

    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    anchor_count = int(xyz.shape[0])
    all_track_ids = torch.as_tensor(state["track_cluster_ids"]).long()
    if all_track_ids.shape != (anchor_count,):
        raise ValueError("map Track IDs do not align with Anchor rows")
    track_anchor_rows = torch.nonzero(all_track_ids >= 0, as_tuple=False).reshape(-1)
    selected_tracks = all_track_ids[track_anchor_rows]
    if selected_tracks.unique().numel() != selected_tracks.numel():
        raise ValueError("selected Track IDs are not unique")
    track_to_anchor = {
        int(track): int(anchor)
        for anchor, track in zip(track_anchor_rows.tolist(), selected_tracks.tolist())
    }
    tracks = payload["tracks"]
    exact: list[dict[int, set[int]]] = [defaultdict(set) for _ in names]
    strong_exact: list[dict[int, set[int]]] = [defaultdict(set) for _ in names]
    observation_count = int(torch.as_tensor(tracks["track_index"]).numel())
    certified_value = tracks.get("identity_positive_certified")
    if certified_value is None:
        if payload.get("support_repair", {}).get("schema") == (
            "lafgs_rendered_track_support_repair"
        ):
            raise ValueError(
                "support-repaired payload lacks observation-level identity certification"
            )
        certified = torch.ones(observation_count, dtype=torch.bool)
    else:
        certified = torch.as_tensor(certified_value)
        if certified.dtype != torch.bool or certified.shape != (observation_count,):
            raise ValueError("identity-positive certification must be exact bool rows")
    for observation, (track, query, keypoint) in enumerate(
        zip(
            torch.as_tensor(tracks["track_index"]).long().tolist(),
            torch.as_tensor(tracks["query_index"]).long().tolist(),
            torch.as_tensor(tracks["keypoint_index"]).long().tolist(),
        )
    ):
        anchor = track_to_anchor.get(int(track))
        if anchor is not None:
            exact[int(query)][int(keypoint)].add(int(anchor))
            if bool(certified[observation]):
                strong_exact[int(query)][int(keypoint)].add(int(anchor))

    projective_observations = state.get("projective_anchor_observations")
    surface_exact_observation_count = 0
    if projective_observations is not None:
        if (
            projective_observations.get("schema")
            != "lafgs_projective_anchor_observations"
            or int(projective_observations.get("version", -1)) != 1
        ):
            raise ValueError("projective Anchor observation schema differs")
        offsets = torch.as_tensor(projective_observations["observation_offsets"])
        observation_queries = torch.as_tensor(projective_observations["query_indices"])
        observation_keypoints = torch.as_tensor(
            projective_observations["keypoint_indices"]
        )
        if offsets.dtype != torch.long or offsets.shape != (anchor_count + 1,):
            raise ValueError("projective observation offsets must be int64 [N+1]")
        if int(offsets[0]) != 0 or bool((offsets[1:] < offsets[:-1]).any()):
            raise ValueError("projective observation offsets are not valid CSR")
        observation_count = int(offsets[-1])
        for field, value in (
            ("query", observation_queries),
            ("keypoint", observation_keypoints),
        ):
            if value.dtype != torch.long or value.shape != (observation_count,):
                raise ValueError(
                    f"projective observation {field} indices must be int64 [E]"
                )
            if value.numel() and int(value.min()) < 0:
                raise ValueError(
                    f"projective observation {field} indices cannot be negative"
                )
        if observation_queries.numel() and int(observation_queries.max()) >= len(names):
            raise ValueError("projective observation query index is outside registry")
        surface_rows = torch.nonzero(all_track_ids < 0, as_tuple=False).reshape(-1)
        for anchor in surface_rows.tolist():
            start, end = int(offsets[anchor]), int(offsets[anchor + 1])
            for query, keypoint in zip(
                observation_queries[start:end].tolist(),
                observation_keypoints[start:end].tolist(),
            ):
                if int(keypoint) >= int(
                    torch.as_tensor(cache[names[int(query)]]["native_keypoints"]).shape[
                        0
                    ]
                ):
                    raise ValueError(
                        "projective observation keypoint is outside rendered cache"
                    )
                exact[int(query)][int(keypoint)].add(int(anchor))
                strong_exact[int(query)][int(keypoint)].add(int(anchor))
                surface_exact_observation_count += 1

    if not 0.0 <= float(alpha_minimum) <= 1.0:
        raise ValueError("alpha minimum must lie in [0, 1]")
    if float(depth_abs_tolerance_m) < 0 or float(depth_relative_tolerance) < 0:
        raise ValueError("depth tolerances must be non-negative")
    if (scene_calibration_path is None) != (expected_scene_calibration_sha256 is None):
        raise ValueError("scene calibration path and expected SHA must be paired")
    scene_calibration_sha256 = None
    if scene_calibration_path is not None:
        scene_calibration_path = scene_calibration_path.resolve()
        scene_calibration_sha256 = sha256_file(scene_calibration_path)
        if scene_calibration_sha256 != str(expected_scene_calibration_sha256):
            raise ValueError("scene calibration SHA differs")
        calibration = json.loads(scene_calibration_path.read_text())
        calibration_sources = calibration.get("sources", {})
        if (
            calibration.get("schema") != "lafgs_mapping_only_scene_calibration"
            or calibration_sources.get("uses_test_queries") is not False
            or calibration_sources.get("uses_source_mapping_rgb") is not False
            or calibration_sources.get("mapping_source") != "gaussian_render"
        ):
            raise ValueError(
                "scene calibration is not source-image-free mapping-only evidence"
            )
        parameters = calibration.get("parameters", {})
        expected_parameters = {
            "positive_radius_px": float(strong_radius_px),
            "negative_radius_px": float(ambiguous_radius_px),
            "evidence_depth_abs_tolerance_m": float(depth_abs_tolerance_m),
        }
        for key, expected in expected_parameters.items():
            if float(parameters.get(key, float("nan"))) != expected:
                raise ValueError(f"teacher {key} differs from scene calibration")
    records = []
    positive_rows = strong_count = ambiguous_count = exact_count = 0
    compatible_count = exact_depth_disagreement_count = weak_exact_count = 0
    masked_query_row_count = depth_visibility_rejected_anchor_count = 0
    for query_index, name in enumerate(names):
        cached = cache[name]
        all_keypoints = torch.as_tensor(cached["native_keypoints"]).float()
        if "native_valid_keypoint_mask" in cached:
            keypoint_valid = torch.as_tensor(
                cached["native_valid_keypoint_mask"]
            ).bool()
            if keypoint_valid.shape != (all_keypoints.shape[0],):
                raise ValueError("render validity mask and keypoint rows differ")
        else:
            keypoint_valid = torch.ones(all_keypoints.shape[0], dtype=torch.bool)
        exact_rows = torch.zeros(all_keypoints.shape[0], dtype=torch.bool)
        for source_row in exact[query_index]:
            exact_rows[int(source_row)] = True
        retained_rows = keypoint_valid | exact_rows
        query_rows = torch.nonzero(retained_rows, as_tuple=False).reshape(-1)
        masked_query_row_count += int((~retained_rows).sum())
        keypoints = all_keypoints[query_rows]
        keypoints = keypoints + float(cached.get("pixel_center_offset", 0.5))
        intrinsic = torch.as_tensor(cached["native_K"]).float()
        pose = torch.as_tensor(cached["pose_w2c"]).float()
        height, width = torch.as_tensor(cached["native_input_hw"]).long().tolist()
        projected, depth = _project(xyz, intrinsic, pose)
        valid = (
            torch.isfinite(projected).all(dim=1)
            & torch.isfinite(depth)
            & (depth > 1e-5)
            & (projected[:, 0] >= 0.0)
            & (projected[:, 0] < float(width))
            & (projected[:, 1] >= 0.0)
            & (projected[:, 1] < float(height))
        )
        if "native_rendered_alpha" in cached and "native_rendered_depth" in cached:
            alpha = torch.as_tensor(cached["native_rendered_alpha"]).float()
            rendered_depth = torch.as_tensor(cached["native_rendered_depth"]).float()
            if alpha.shape != (height, width) or rendered_depth.shape != (
                height,
                width,
            ):
                raise ValueError("rendered alpha/depth and native image differ")
            x = projected[:, 0].round().long().clamp(0, width - 1)
            y = projected[:, 1].round().long().clamp(0, height - 1)
            reference_depth = rendered_depth[y, x]
            reference_alpha = alpha[y, x]
            tolerance = (
                float(depth_abs_tolerance_m)
                + float(depth_relative_tolerance) * reference_depth.abs()
            )
            visible = (
                torch.isfinite(reference_depth)
                & (reference_depth > 1e-5)
                & (reference_alpha >= float(alpha_minimum))
                & ((depth - reference_depth).abs() <= tolerance)
            )
            depth_visibility_rejected_anchor_count += int((valid & ~visible).sum())
            valid &= visible
        nearby = _spatial_candidates(
            projected, valid, keypoints, float(ambiguous_radius_px)
        )
        positives = []
        exact_positives = []
        support_compatible = []
        ambiguous = []
        for row, candidates in enumerate(nearby):
            candidate_tensor = torch.as_tensor(candidates, dtype=torch.long)
            if candidate_tensor.numel():
                distance = torch.linalg.norm(
                    projected[candidate_tensor] - keypoints[row], dim=1
                )
                strong = candidate_tensor[distance <= float(strong_radius_px)].tolist()
                weak = candidate_tensor[
                    (distance > float(strong_radius_px))
                    & (distance <= float(ambiguous_radius_px))
                ].tolist()
            else:
                strong, weak = [], []
            source_row = int(query_rows[row])
            exact_values = sorted(exact[query_index].get(source_row, ()))
            strong_exact_values = sorted(strong_exact[query_index].get(source_row, ()))
            weak_exact_values = sorted(set(exact_values) - set(strong_exact_values))
            compatible_values = sorted(set(strong) - set(exact_values))
            positives.append(strong_exact_values)
            exact_positives.append(exact_values)
            support_compatible.append(compatible_values)
            ambiguous.append(
                sorted(
                    (set(weak) | set(compatible_values) | set(weak_exact_values))
                    - set(strong_exact_values)
                )
            )
            exact_count += len(exact_values)
            weak_exact_count += len(weak_exact_values)
            compatible_count += len(compatible_values)
            exact_depth_disagreement_count += sum(
                not bool(valid[anchor]) for anchor in exact_values
            )
        positive_offsets, positive_indices = _csr(positives)
        exact_offsets, exact_indices = _csr(exact_positives)
        compatible_offsets, compatible_indices = _csr(support_compatible)
        ambiguous_offsets, ambiguous_indices = _csr(ambiguous)
        positive_rows += int(((positive_offsets[1:] - positive_offsets[:-1]) > 0).sum())
        strong_count += int(positive_indices.numel())
        ambiguous_count += int(ambiguous_indices.numel())
        records.append(
            {
                "query_index": query_index,
                "query_name": name,
                "query_rows": query_rows,
                "positive_offsets": positive_offsets,
                "positive_indices": positive_indices,
                "exact_identity_offsets": exact_offsets,
                "exact_identity_indices": exact_indices,
                "support_compatible_offsets": compatible_offsets,
                "support_compatible_indices": compatible_indices,
                "ambiguous_offsets": ambiguous_offsets,
                "ambiguous_indices": ambiguous_indices,
            }
        )
        if (query_index + 1) % 100 == 0 or query_index + 1 == len(names):
            print(
                json.dumps(
                    {
                        "completed_queries": query_index + 1,
                        "positive_rows": positive_rows,
                        "strong_pairs": strong_count,
                    }
                ),
                flush=True,
            )

    group_labels, sequence_names, mapping_grouping = _mapping_groups(
        names, single_trajectory_pose_cells
    )
    sequence_ids = torch.as_tensor(
        [sequence_names.index(group) for group in group_labels]
    ).long()
    teacher = {
        "schema": "lafgs_v9_active_map_complete_positive_teacher",
        "version": 1,
        "query_cache": str(query_cache_path.resolve()),
        "query_cache_sha256": query_cache_sha256,
        "scene_calibration": (
            str(scene_calibration_path) if scene_calibration_path is not None else None
        ),
        "scene_calibration_sha256": scene_calibration_sha256,
        "anchor_count": int(xyz.shape[0]),
        "query_names": names,
        "records": records,
        "diagnostics": {
            "query_count": len(names),
            "positive_rows": positive_rows,
            "strong_pair_count": strong_count,
            "ambiguous_pair_count": ambiguous_count,
            "exact_track_positive_count": exact_count,
            "strong_certified_exact_positive_count": strong_count,
            "weak_exact_ambiguous_count": weak_exact_count,
            "support_compatible_pair_count": compatible_count,
            "exact_depth_disagreement_audit_count": (exact_depth_disagreement_count),
            "masked_query_row_count": masked_query_row_count,
            "depth_visibility_rejected_anchor_count": (
                depth_visibility_rejected_anchor_count
            ),
            "surface_completion_exact_observation_count": (
                surface_exact_observation_count
            ),
        },
        "config": {
            "strong_radius_px": float(strong_radius_px),
            "ambiguous_radius_px": float(ambiguous_radius_px),
            "alpha_minimum": float(alpha_minimum),
            "depth_abs_tolerance_m": float(depth_abs_tolerance_m),
            "depth_relative_tolerance": float(depth_relative_tolerance),
            "threshold_lineage": (
                "mapping_scene_calibration"
                if scene_calibration_path is not None
                else "explicit_unbound_compatibility"
            ),
            "identity_positive_policy": (
                "cycle_seeded_observation_reprojection_certified_exact_only"
            ),
            "weak_exact_policy": "ambiguous_ignore_not_positive",
            "projection_compatible_policy": "ambiguous_ignore_not_positive",
            "exact_depth_policy": "audit_only_never_hard_reject",
            "geometry_source": (
                "ray_triangulated_track_xyz_plus_rendered_depth_projective_completion"
                if bool((all_track_ids < 0).any())
                else "ray_triangulated_track_xyz_and_mapping_pose"
            ),
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
            "uses_rendered_depth": any(
                "native_rendered_depth" in cache[name] for name in names
            ),
            "uses_rendered_alpha": any(
                "native_rendered_alpha" in cache[name] for name in names
            ),
        },
    }
    graph = {
        "schema": "lafgs_rendered_track_training_graph",
        "version": 1,
        "query_names": names,
        "records": [
            {
                "query_index": record["query_index"],
                "query_rows": record["query_rows"].clone(),
                "ambiguous_training_policy": "ignore",
            }
            for record in records
        ],
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
    }
    enriched_map = dict(state)
    track_reconstruction = dict(state.get("track_centric_reconstruction", {}))
    track_reconstruction["track_indices"] = selected_tracks.clone()
    if "base_canonical_rows" not in track_reconstruction:
        track_reconstruction["base_canonical_rows"] = torch.nonzero(
            all_track_ids < 0, as_tuple=False
        ).reshape(-1)
    enriched_map["track_centric_reconstruction"] = track_reconstruction
    enriched_map["v7_metric_raw_features"] = torch.as_tensor(
        state["anchor_features"]
    ).float()
    enriched_map["descriptor_transform_contract"] = {
        "schema": "lafgs_rendered_track_descriptor_transform_contract",
        "version": 1,
        "mode": "none_identity_only",
        "learned_query_transform": False,
        "learned_anchor_transform": False,
        "learned_anchor_residual": False,
    }
    enriched_payload = dict(payload)
    enriched_payload["pose_view_bins"] = torch.as_tensor(payload["query_bins"]).clone()
    enriched_payload["query_bins"] = sequence_ids
    enriched_payload["training_sequence_names"] = sequence_names

    outputs = {
        "teacher": output_dir / "rendered_track_positive_teacher.pt",
        "graph": output_dir / "rendered_track_training_graph.pt",
        "map": output_dir / "rendered_track_training_map.pt",
        "track_payload": output_dir / "rendered_track_training_payload.pt",
    }
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(path)
    _atomic_torch_save(teacher, outputs["teacher"])
    _atomic_torch_save(graph, outputs["graph"])
    _atomic_torch_save(enriched_map, outputs["map"])
    _atomic_torch_save(enriched_payload, outputs["track_payload"])
    report = {
        "schema": "lafgs_rendered_track_train_only_materialization",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "mapping_sequence_names": sequence_names,
        "mapping_sequence_query_counts": {
            sequence: sum(group == sequence for group in group_labels)
            for sequence in sequence_names
        },
        "mapping_grouping": mapping_grouping,
        "formal_method_uses_crossfit": False,
        "descriptor_transform": "none_identity_only",
        "teacher_diagnostics": teacher["diagnostics"],
        "inputs": {
            "anchor_map": str(anchor_map_path.resolve()),
            "track_payload": str(track_payload_path.resolve()),
            "query_cache": str(query_cache_path.resolve()),
            "scene_calibration": (
                str(scene_calibration_path)
                if scene_calibration_path is not None
                else None
            ),
        },
        "input_sha256": {
            "anchor_map": sha256_file(anchor_map_path),
            "track_payload": sha256_file(track_payload_path),
            "query_cache": query_cache_sha256,
            "scene_calibration": scene_calibration_sha256,
        },
        "outputs": {key: str(value.resolve()) for key, value in outputs.items()},
        "output_sha256": {key: sha256_file(value) for key, value in outputs.items()},
    }
    report_path = output_dir / "rendered_track_training_materialization.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-map", type=Path, required=True)
    parser.add_argument("--track-payload", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strong-radius-px", type=float, default=2.0)
    parser.add_argument("--ambiguous-radius-px", type=float, default=8.0)
    parser.add_argument("--single-trajectory-pose-cells", type=int, default=3)
    parser.add_argument("--alpha-minimum", type=float, default=0.05)
    parser.add_argument("--depth-abs-tolerance-m", type=float, default=0.05)
    parser.add_argument("--depth-relative-tolerance", type=float, default=0.02)
    parser.add_argument("--scene-calibration", type=Path)
    parser.add_argument("--expected-scene-calibration-sha256")
    args = parser.parse_args()
    report = materialize(
        anchor_map_path=args.anchor_map.resolve(),
        track_payload_path=args.track_payload.resolve(),
        query_cache_path=args.query_cache.resolve(),
        output_dir=args.output_dir.resolve(),
        strong_radius_px=args.strong_radius_px,
        ambiguous_radius_px=args.ambiguous_radius_px,
        single_trajectory_pose_cells=args.single_trajectory_pose_cells,
        alpha_minimum=args.alpha_minimum,
        depth_abs_tolerance_m=args.depth_abs_tolerance_m,
        depth_relative_tolerance=args.depth_relative_tolerance,
        scene_calibration_path=args.scene_calibration,
        expected_scene_calibration_sha256=(args.expected_scene_calibration_sha256),
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
