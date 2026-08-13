#!/usr/bin/env python3
"""Train-only multi-prototype Track descriptor ceiling.

Each Track keeps up to K view-bin prototypes learned without the held mapping
sequence. Prototypes repeat the same ray-triangulated xyz, so deployment remains
one global descriptor bank, one GEMM/Top-1, and one PoseLib call per query.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import torch

from evidence.tracks import robust_fuse_track_descriptors
from map_learning.metric import SharedLowRankMetric
from scripts.evaluate_rendered_track_crossfit import (
    _combined_summary,
    _sequence_name,
)
from topology.deployment_revision import collect_deployment_statistics


def _atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _expand_csr(
    offsets: torch.Tensor,
    indices: torch.Tensor,
    prototype_rows: list[list[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.as_tensor(offsets).long()
    indices = torch.as_tensor(indices).long()
    revised_offsets = [0]
    revised_indices = []
    for row in range(offsets.numel() - 1):
        values = []
        for anchor in indices[offsets[row] : offsets[row + 1]].tolist():
            values.extend(prototype_rows[int(anchor)])
        revised_indices.extend(sorted(set(values)))
        revised_offsets.append(len(revised_indices))
    return torch.as_tensor(revised_offsets).long(), torch.as_tensor(
        revised_indices
    ).long()


def expand_teacher(teacher: dict, prototype_rows: list[list[int]]) -> dict:
    if len(prototype_rows) != int(teacher["anchor_count"]):
        raise ValueError("prototype registry and teacher anchor count differ")
    records = []
    positive_rows = strong_pairs = ambiguous_pairs = 0
    for source in teacher["records"]:
        record = dict(source)
        for prefix in ("positive", "ambiguous"):
            offsets, indices = _expand_csr(
                source[f"{prefix}_offsets"],
                source[f"{prefix}_indices"],
                prototype_rows,
            )
            record[f"{prefix}_offsets"] = offsets
            record[f"{prefix}_indices"] = indices
            if prefix == "positive":
                positive_rows += int(((offsets[1:] - offsets[:-1]) > 0).sum())
                strong_pairs += int(indices.numel())
            else:
                ambiguous_pairs += int(indices.numel())
        records.append(record)
    return {
        **teacher,
        "anchor_count": sum(len(rows) for rows in prototype_rows),
        "records": records,
        "diagnostics": {
            **teacher["diagnostics"],
            "positive_rows": positive_rows,
            "strong_pair_count": strong_pairs,
            "ambiguous_pair_count": ambiguous_pairs,
        },
        "multi_prototype": {
            "prototype_count": sum(len(rows) for rows in prototype_rows),
            "uses_test_queries": False,
        },
    }


def _fold_map(
    *,
    state: dict,
    payload: dict,
    cache_payload: dict,
    held_sequence: str,
    maximum_prototypes: int,
    trim_fraction: float,
) -> tuple[dict, list[list[int]]]:
    names = list(payload["query_names"])
    cache = cache_payload.get("queries", cache_payload)
    tracks = payload["tracks"]
    selected_tracks = torch.as_tensor(state["track_cluster_ids"]).long()
    track_to_anchor = {
        int(track): row for row, track in enumerate(selected_tracks.tolist())
    }
    pose_bins = torch.as_tensor(
        payload.get("pose_view_bins", payload["query_bins"])
    ).long()
    observations: list[dict[int, list[int]]] = [
        defaultdict(list) for _ in selected_tracks
    ]
    for observation, (track, query) in enumerate(
        zip(
            torch.as_tensor(tracks["track_index"]).long().tolist(),
            torch.as_tensor(tracks["query_index"]).long().tolist(),
        )
    ):
        anchor = track_to_anchor.get(int(track))
        if anchor is None or _sequence_name(names[int(query)]) == held_sequence:
            continue
        observations[anchor][int(pose_bins[int(query)])].append(observation)

    prototype_features = []
    prototype_source_anchor = []
    prototype_view_bin = []
    prototype_rows: list[list[int]] = [[] for _ in selected_tracks]
    query_index = torch.as_tensor(tracks["query_index"]).long()
    keypoint_index = torch.as_tensor(tracks["keypoint_index"]).long()
    confidence = torch.as_tensor(tracks["confidence"]).float()
    for anchor, groups in enumerate(observations):
        chosen = sorted(groups, key=lambda value: (-len(groups[value]), int(value)))[
            : int(maximum_prototypes)
        ]
        for view_bin in chosen:
            rows = torch.as_tensor(groups[view_bin], dtype=torch.long)
            queries = query_index[rows]
            keypoints = keypoint_index[rows]
            descriptors = torch.stack(
                [
                    torch.as_tensor(cache[names[int(query)]]["native_descriptors"])[
                        int(keypoint)
                    ]
                    for query, keypoint in zip(queries.tolist(), keypoints.tolist())
                ]
            )
            feature = robust_fuse_track_descriptors(
                descriptors,
                pose_bins[queries],
                confidence[rows],
                trim_fraction=float(trim_fraction),
            )
            prototype_rows[anchor].append(len(prototype_features))
            prototype_features.append(feature)
            prototype_source_anchor.append(anchor)
            prototype_view_bin.append(view_bin)
    if not prototype_features:
        raise RuntimeError("multi-prototype fold produced an empty map")
    source = torch.as_tensor(prototype_source_anchor).long()
    output = dict(state)
    count = int(selected_tracks.numel())
    for key, value in state.items():
        if torch.is_tensor(value) and value.ndim and value.shape[0] == count:
            output[key] = value[source]
    output["anchor_ids"] = torch.arange(len(prototype_features), dtype=torch.long)
    output["anchor_features"] = torch.stack(prototype_features).float()
    output["v7_metric_raw_features"] = output["anchor_features"].clone()
    output["prototype_source_anchor"] = source
    output["prototype_view_bin"] = torch.as_tensor(prototype_view_bin).long()
    output["track_centric_reconstruction"] = {
        "track_indices": torch.as_tensor(output["track_cluster_ids"]).long(),
        "base_canonical_rows": torch.empty(0, dtype=torch.long),
        "track_anchor_count": len(prototype_features),
        "base_reserve_count": 0,
    }
    output["base_anchor_count"] = 0
    output["micro_anchor_count"] = len(prototype_features)
    output["canonical_anchor_count"] = len(prototype_features)
    output["provenance"] = {
        **state.get("provenance", {}),
        "multi_prototype_crossfit": {
            "held_mapping_sequence": held_sequence,
            "maximum_prototypes_per_track": int(maximum_prototypes),
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
        },
    }
    return output, prototype_rows


def _identity_metric(state: dict, map_path: Path) -> dict:
    count, dim = torch.as_tensor(state["anchor_features"]).shape
    metric = SharedLowRankMetric(descriptor_dim=dim, rank=1, max_residual_norm=0.0)
    with torch.no_grad():
        for parameter in metric.parameters():
            parameter.zero_()
    return {
        "schema": "lafgs_shared_metric_state",
        "version": 1,
        "landmark_indices": torch.arange(count, dtype=torch.long),
        "metric_config": metric.export_config(),
        "metric_state_dict": {
            name: value.detach().cpu() for name, value in metric.state_dict().items()
        },
        "map_path": str(map_path.resolve()),
        "step": 0,
        "protocol": "rendered_track_mapping_sequence_crossfit_multi_prototype",
    }


def run(args) -> dict:
    state = torch.load(args.anchor_map, map_location="cpu", weights_only=False)
    payload = torch.load(args.track_payload, map_location="cpu", weights_only=False)
    teacher = torch.load(args.teacher, map_location="cpu", weights_only=False)
    cache = torch.load(args.query_cache, map_location="cpu", weights_only=False)
    if (
        cache.get("uses_source_mapping_rgb") is not False
        or cache.get("uses_test_queries") is not False
    ):
        raise ValueError(
            "multi-prototype crossfit requires rendered mapping-only cache"
        )
    names = list(teacher["query_names"])
    sequences = sorted({_sequence_name(name) for name in names})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    all_rows = []
    folds = []
    for fold_index, held_sequence in enumerate(sequences):
        fold_dir = args.output_dir / held_sequence
        fold_dir.mkdir()
        fold_map, prototype_rows = _fold_map(
            state=state,
            payload=payload,
            cache_payload=cache,
            held_sequence=held_sequence,
            maximum_prototypes=args.maximum_prototypes,
            trim_fraction=args.descriptor_trim_fraction,
        )
        fold_teacher = expand_teacher(teacher, prototype_rows)
        map_path = fold_dir / "anchor_map.pt"
        teacher_path = fold_dir / "positive_teacher.pt"
        metric_path = fold_dir / "metric_state.pt"
        _atomic_save(fold_map, map_path)
        _atomic_save(fold_teacher, teacher_path)
        _atomic_save(_identity_metric(fold_map, map_path), metric_path)
        query_indices = [
            index
            for index, name in enumerate(names)
            if _sequence_name(name) == held_sequence
        ]
        statistics = collect_deployment_statistics(
            state=fold_map,
            metric_state_path=metric_path,
            teacher=fold_teacher,
            query_cache=cache,
            device=torch.device(args.device),
            ransac_reprojection_px=args.ransac_reprojection_px,
            clean_reprojection_px=args.clean_reprojection_px,
            task_translation_m=0.05,
            task_rotation_deg=5.0,
            seed=args.seed,
            query_indices=query_indices,
            progress_label=f"multiprototype_held_{held_sequence}",
            collect_anchor_statistics=False,
        )
        statistics_path = fold_dir / "statistics.pt"
        _atomic_save(statistics, statistics_path)
        all_rows.extend(statistics["queries"])
        folds.append(
            {
                "held_sequence": held_sequence,
                "source_track_count": int(state["anchor_ids"].numel()),
                "prototype_count": int(fold_map["anchor_ids"].numel()),
                "summary": statistics["summary"],
                "statistics": str(statistics_path.resolve()),
            }
        )
        print(json.dumps(folds[-1], sort_keys=True), flush=True)
    # A combined precision is not meaningful without aggregating prototype
    # counters across different fold registries; pose metrics remain exact.
    pose_summary = _combined_summary(
        all_rows,
        {
            "winner_count": torch.zeros(1),
            "correct_winner_count": torch.zeros(1),
            "clean_inlier_count": torch.zeros(1),
            "harmful_inlier_count": torch.zeros(1),
        },
    )
    for key in ("raw_gt_precision_percent", "inlier_gt_precision_percent"):
        pose_summary.pop(key, None)
    report = {
        "schema": "lafgs_rendered_track_multi_prototype_mapping_sequence_crossfit",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "one_global_bank": True,
        "one_top1": True,
        "one_poselib_call_per_query": True,
        "maximum_prototypes_per_track": int(args.maximum_prototypes),
        "folds": folds,
        "combined_pose_summary": pose_summary,
    }
    (args.output_dir / "crossfit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-map", type=Path, required=True)
    parser.add_argument("--track-payload", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-prototypes", type=int, default=2)
    parser.add_argument("--descriptor-trim-fraction", type=float, default=0.2)
    parser.add_argument("--ransac-reprojection-px", type=float, default=12.0)
    parser.add_argument("--clean-reprojection-px", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for field in ("anchor_map", "track_payload", "teacher", "query_cache"):
        setattr(args, field, getattr(args, field).resolve())
    args.output_dir = args.output_dir.resolve()
    if args.maximum_prototypes < 1:
        raise ValueError("maximum_prototypes must be positive")
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
