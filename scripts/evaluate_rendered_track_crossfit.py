#!/usr/bin/env python3
"""Leave-one-mapping-sequence-out replay for rendered-RGB Track maps."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from evidence.tracks import robust_fuse_track_descriptors
from map_learning.metric import SharedLowRankMetric
from topology.deployment_revision import collect_deployment_statistics, subset_teacher


COUNTER_NAMES = (
    "winner_count",
    "correct_winner_count",
    "false_attractor_count",
    "ambiguous_winner_count",
    "clean_inlier_count",
    "harmful_inlier_count",
    "counterfactual_clean_gain",
    "information_deletion_loss",
)


def _combined_summary(query_rows: list[dict], counters: dict) -> dict:
    import numpy as np

    te = np.asarray([row["te_cm"] for row in query_rows], dtype=np.float64)
    ae = np.asarray([row["ae_deg"] for row in query_rows], dtype=np.float64)
    tail = max(int(np.ceil(0.05 * te.size)), 1)
    raw = int(counters["winner_count"].sum())
    correct = int(counters["correct_winner_count"].sum())
    clean = int(counters["clean_inlier_count"].sum())
    harmful = int(counters["harmful_inlier_count"].sum())
    return {
        "query_count": int(te.size),
        "median_te_cm": float(np.median(te)),
        "mean_te_cm": float(np.mean(te)),
        "p90_te_cm": float(np.percentile(te, 90)),
        "cvar95_te_cm": float(np.sort(te)[-tail:].mean()),
        "median_ae_deg": float(np.median(ae)),
        "mean_ae_deg": float(np.mean(ae)),
        "p90_ae_deg": float(np.percentile(ae, 90)),
        "recall_5cm_5deg_percent": float(100.0 * np.mean((te < 5.0) & (ae < 5.0))),
        "catastrophic_100cm_count": int(np.count_nonzero(te >= 100.0)),
        "raw_gt_precision_percent": 100.0 * correct / max(raw, 1),
        "inlier_gt_precision_percent": 100.0 * clean / max(clean + harmful, 1),
        "retained_matches_mean": float(
            np.mean([row["correspondences"] for row in query_rows])
        ),
        "mean_hypotheses": float(np.mean([row["hypotheses"] for row in query_rows])),
    }


def _atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sequence_name(image_name: str) -> str:
    return str(image_name).split("/", maxsplit=1)[0]


def _fold_bank(
    *,
    state: dict,
    payload: dict,
    query_cache: dict,
    held_sequence: str,
    trim_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    names = list(payload["query_names"])
    cache = query_cache.get("queries", query_cache)
    tracks = payload["tracks"]
    query_bins = torch.as_tensor(
        payload.get("pose_view_bins", payload["query_bins"])
    ).long()
    selected_tracks = torch.as_tensor(state["track_cluster_ids"]).long()
    selected_lookup = {
        int(track): row for row, track in enumerate(selected_tracks.tolist())
    }
    observations: dict[int, list[int]] = defaultdict(list)
    for observation, (track, query) in enumerate(
        zip(
            torch.as_tensor(tracks["track_index"]).long().tolist(),
            torch.as_tensor(tracks["query_index"]).long().tolist(),
        )
    ):
        row = selected_lookup.get(int(track))
        if row is not None and _sequence_name(names[int(query)]) != held_sequence:
            observations[row].append(observation)

    eligible = torch.zeros(selected_tracks.numel(), dtype=torch.bool)
    features = []
    for anchor in range(selected_tracks.numel()):
        selected_observations = observations.get(anchor, ())
        if not selected_observations:
            continue
        observation_rows = torch.as_tensor(selected_observations, dtype=torch.long)
        queries = torch.as_tensor(tracks["query_index"]).long()[observation_rows]
        keypoints = torch.as_tensor(tracks["keypoint_index"]).long()[observation_rows]
        descriptor = torch.stack(
            [
                torch.as_tensor(cache[names[int(query)]]["native_descriptors"])[
                    int(keypoint)
                ]
                for query, keypoint in zip(queries.tolist(), keypoints.tolist())
            ]
        )
        features.append(
            robust_fuse_track_descriptors(
                descriptor,
                query_bins[queries],
                torch.as_tensor(tracks["confidence"])[observation_rows],
                trim_fraction=float(trim_fraction),
            )
        )
        eligible[anchor] = True
    if not features:
        raise RuntimeError(f"held fold {held_sequence} has no train-supported anchors")
    return eligible, torch.stack(features)


def _subset_state(state: dict, keep: torch.Tensor, features: torch.Tensor) -> dict:
    keep = torch.as_tensor(keep).bool()
    count = int(keep.numel())
    output = dict(state)
    for key, value in state.items():
        if torch.is_tensor(value) and value.ndim and value.shape[0] == count:
            output[key] = value[keep]
    output["anchor_ids"] = torch.arange(int(keep.sum()), dtype=torch.long)
    output["anchor_features"] = features.float()
    output["v7_metric_raw_features"] = features.float()
    selected_tracks = torch.as_tensor(state["track_cluster_ids"]).long()[keep]
    output["track_cluster_ids"] = selected_tracks
    output["track_centric_reconstruction"] = {
        **state.get("track_centric_reconstruction", {}),
        "track_indices": selected_tracks.clone(),
        "base_canonical_rows": torch.empty(0, dtype=torch.long),
        "track_anchor_count": int(keep.sum()),
        "base_reserve_count": 0,
    }
    output["base_anchor_count"] = 0
    output["micro_anchor_count"] = int(keep.sum())
    output["canonical_anchor_count"] = int(keep.sum())
    return output


def _identity_metric(anchor_count: int, descriptor_dim: int, map_path: Path) -> dict:
    metric = SharedLowRankMetric(
        descriptor_dim=descriptor_dim, rank=1, max_residual_norm=0.0
    )
    with torch.no_grad():
        for parameter in metric.parameters():
            parameter.zero_()
    return {
        "schema": "lafgs_shared_metric_state",
        "version": 1,
        "landmark_indices": torch.arange(anchor_count, dtype=torch.long),
        "metric_config": metric.export_config(),
        "metric_state_dict": {
            name: value.detach().cpu() for name, value in metric.state_dict().items()
        },
        "map_path": str(map_path.resolve()),
        "step": 0,
        "protocol": "rendered_track_leave_one_mapping_sequence_out_identity",
    }


def run(args) -> dict:
    state = torch.load(args.anchor_map, map_location="cpu", weights_only=False)
    payload = torch.load(args.track_payload, map_location="cpu", weights_only=False)
    teacher = torch.load(args.teacher, map_location="cpu", weights_only=False)
    cache = torch.load(args.query_cache, map_location="cpu", weights_only=False)
    if cache.get("uses_source_mapping_rgb") is not False:
        raise ValueError("crossfit cache is not rendered-RGB-only")
    if cache.get("uses_test_queries") is not False:
        raise ValueError("crossfit cache contains test queries")
    names = list(payload["query_names"])
    if names != list(teacher["query_names"]):
        raise ValueError("teacher and Track query order differs")
    sequences = sorted({_sequence_name(name) for name in names})
    if len(sequences) < 2:
        raise ValueError("crossfit requires multiple mapping sequences")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    full_count = int(torch.as_tensor(state["anchor_ids"]).numel())
    aggregate = {
        name: torch.zeros(full_count, dtype=torch.float64) for name in COUNTER_NAMES
    }
    all_rows = []
    folds = []
    for fold_index, held_sequence in enumerate(sequences):
        fold_dir = args.output_dir / held_sequence
        fold_dir.mkdir()
        keep, features = _fold_bank(
            state=state,
            payload=payload,
            query_cache=cache,
            held_sequence=held_sequence,
            trim_fraction=args.descriptor_trim_fraction,
        )
        fold_map = _subset_state(state, keep, features)
        map_path = fold_dir / "anchor_map.pt"
        metric_path = fold_dir / "metric_state.pt"
        teacher_path = fold_dir / "positive_teacher.pt"
        fold_teacher = subset_teacher(teacher, keep, map_path)
        metric = _identity_metric(int(keep.sum()), int(features.shape[1]), map_path)
        _atomic_save(fold_map, map_path)
        _atomic_save(metric, metric_path)
        _atomic_save(fold_teacher, teacher_path)
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
            task_translation_m=args.task_translation_m,
            task_rotation_deg=args.task_rotation_deg,
            seed=args.seed,
            query_indices=query_indices,
            progress_label=f"held_{held_sequence}",
        )
        _atomic_save(statistics, fold_dir / "statistics.pt")
        for name in COUNTER_NAMES:
            aggregate[name][keep] += torch.as_tensor(
                statistics["counters"][name]
            ).double()
        all_rows.extend(statistics["queries"])
        folds.append(
            {
                "held_sequence": held_sequence,
                "query_count": len(query_indices),
                "train_supported_anchor_count": int(keep.sum()),
                "summary": statistics["summary"],
                "statistics": str((fold_dir / "statistics.pt").resolve()),
            }
        )
        print(json.dumps(folds[-1], sort_keys=True), flush=True)

    combined = {
        "schema": "lafgs_rendered_track_mapping_sequence_crossfit_statistics",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "query_rows": all_rows,
        "counters": aggregate,
        "summary": _combined_summary(all_rows, aggregate),
        "folds": folds,
    }
    statistics_path = args.output_dir / "crossfit_statistics.pt"
    _atomic_save(combined, statistics_path)
    report = {
        "schema": "lafgs_rendered_track_mapping_sequence_crossfit_report",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "sequences": sequences,
        "folds": folds,
        "combined_summary": combined["summary"],
        "inputs": {
            "anchor_map": str(args.anchor_map.resolve()),
            "track_payload": str(args.track_payload.resolve()),
            "teacher": str(args.teacher.resolve()),
            "query_cache": str(args.query_cache.resolve()),
        },
        "input_sha256": {
            "anchor_map": sha256_file(args.anchor_map),
            "track_payload": sha256_file(args.track_payload),
            "teacher": sha256_file(args.teacher),
            "query_cache": sha256_file(args.query_cache),
        },
        "statistics": str(statistics_path.resolve()),
        "statistics_sha256": sha256_file(statistics_path),
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
    parser.add_argument("--descriptor-trim-fraction", type=float, default=0.2)
    parser.add_argument("--ransac-reprojection-px", type=float, default=12.0)
    parser.add_argument("--clean-reprojection-px", type=float, default=4.0)
    parser.add_argument("--task-translation-m", type=float, default=0.05)
    parser.add_argument("--task-rotation-deg", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for field in ("anchor_map", "track_payload", "teacher", "query_cache"):
        setattr(args, field, getattr(args, field).resolve())
    args.output_dir = args.output_dir.resolve()
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
