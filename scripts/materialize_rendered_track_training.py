#!/usr/bin/env python3
"""Materialize train-only geometry labels for a rendered-RGB Track map.

The teacher uses frozen mapping poses and the ray-triangulated Track map. It
never reads source mapping RGB, test queries, rendered depth, or Gaussian
primitive geometry. Exact Track observations are positives; additional
projected anchors within the strong/ambiguous radii provide complete local
geometric supervision for mapping replay and A1 metric training.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from scripts.evaluate_rendered_track_crossfit import _crossfit_groups


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
    blocked_folds: int = 3,
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
    names = list(payload["query_names"])
    if names != list(cache):
        raise ValueError("Track payload and rendered cache query order differs")

    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    selected_tracks = torch.as_tensor(state["track_cluster_ids"]).long()
    if selected_tracks.unique().numel() != selected_tracks.numel():
        raise ValueError("selected Track IDs are not unique")
    track_to_anchor = {
        int(track): anchor for anchor, track in enumerate(selected_tracks.tolist())
    }
    tracks = payload["tracks"]
    exact: list[dict[int, set[int]]] = [defaultdict(set) for _ in names]
    for track, query, keypoint in zip(
        torch.as_tensor(tracks["track_index"]).long().tolist(),
        torch.as_tensor(tracks["query_index"]).long().tolist(),
        torch.as_tensor(tracks["keypoint_index"]).long().tolist(),
    ):
        anchor = track_to_anchor.get(int(track))
        if anchor is not None:
            exact[int(query)][int(keypoint)].add(int(anchor))

    records = []
    positive_rows = strong_count = ambiguous_count = exact_count = 0
    for query_index, name in enumerate(names):
        cached = cache[name]
        keypoints = torch.as_tensor(cached["native_keypoints"]).float()
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
        nearby = _spatial_candidates(
            projected, valid, keypoints, float(ambiguous_radius_px)
        )
        positives = []
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
            exact_values = sorted(exact[query_index].get(row, ()))
            positives.append(sorted(set(strong) | set(exact_values)))
            ambiguous.append(sorted(set(weak) - set(positives[-1])))
            exact_count += len(exact_values)
        positive_offsets, positive_indices = _csr(positives)
        ambiguous_offsets, ambiguous_indices = _csr(ambiguous)
        positive_rows += int(((positive_offsets[1:] - positive_offsets[:-1]) > 0).sum())
        strong_count += int(positive_indices.numel())
        ambiguous_count += int(ambiguous_indices.numel())
        records.append(
            {
                "query_index": query_index,
                "query_name": name,
                "query_rows": torch.arange(keypoints.shape[0], dtype=torch.long),
                "positive_offsets": positive_offsets,
                "positive_indices": positive_indices,
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

    group_labels, sequence_names = _crossfit_groups(names, blocked_folds)
    sequence_ids = torch.as_tensor(
        [sequence_names.index(group) for group in group_labels]
    ).long()
    teacher = {
        "schema": "lafgs_v9_active_map_complete_positive_teacher",
        "version": 1,
        "anchor_count": int(xyz.shape[0]),
        "query_names": names,
        "records": records,
        "diagnostics": {
            "query_count": len(names),
            "positive_rows": positive_rows,
            "strong_pair_count": strong_count,
            "ambiguous_pair_count": ambiguous_count,
            "exact_track_positive_count": exact_count,
        },
        "config": {
            "strong_radius_px": float(strong_radius_px),
            "ambiguous_radius_px": float(ambiguous_radius_px),
            "geometry_source": "ray_triangulated_track_xyz_and_mapping_pose",
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
            "uses_rendered_depth": False,
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
    enriched_map["track_centric_reconstruction"] = {
        "track_indices": selected_tracks.clone(),
        "base_canonical_rows": torch.empty(0, dtype=torch.long),
    }
    enriched_map["v7_metric_raw_features"] = torch.as_tensor(
        state["anchor_features"]
    ).float()
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
        "mapping_grouping": (
            "mapping_trajectory"
            if len(set(_sequence_name(name) for name in names)) >= 2
            else "contiguous_mapping_blocks"
        ),
        "teacher_diagnostics": teacher["diagnostics"],
        "inputs": {
            "anchor_map": str(anchor_map_path.resolve()),
            "track_payload": str(track_payload_path.resolve()),
            "query_cache": str(query_cache_path.resolve()),
        },
        "input_sha256": {
            "anchor_map": sha256_file(anchor_map_path),
            "track_payload": sha256_file(track_payload_path),
            "query_cache": sha256_file(query_cache_path),
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
    parser.add_argument("--blocked-folds", type=int, default=3)
    args = parser.parse_args()
    report = materialize(
        anchor_map_path=args.anchor_map.resolve(),
        track_payload_path=args.track_payload.resolve(),
        query_cache_path=args.query_cache.resolve(),
        output_dir=args.output_dir.resolve(),
        strong_radius_px=args.strong_radius_px,
        ambiguous_radius_px=args.ambiguous_radius_px,
        blocked_folds=args.blocked_folds,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
