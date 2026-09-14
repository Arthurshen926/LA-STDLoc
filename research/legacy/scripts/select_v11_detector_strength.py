#!/usr/bin/env python3
"""Select one bounded detector strength on render-only validation families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from features.scene_specific_detector import (
    fuse_scene_reliability,
    load_scene_detector_checkpoint,
    protected_scene_candidate_indices,
)
from features.superpoint import SuperPoint
from localization.matcher import global_cosine_top1
from map_learning.v9_causal_feedback import standard_pose_replay


def _summary(rows: list[dict]) -> dict:
    task = np.asarray([row["task_error"] for row in rows])
    te = np.asarray([row["translation_error_cm"] for row in rows])
    ae = np.asarray([row["rotation_error_deg"] for row in rows])
    return {
        "query_count": len(rows),
        "median_task_error": float(np.median(task)),
        "p90_task_error": float(np.percentile(task, 90)),
        "median_translation_cm": float(np.median(te)),
        "p90_translation_cm": float(np.percentile(te, 90)),
        "r5_percent": float(100 * np.mean((te < 5) & (ae < 5))),
        "catastrophic_count": int(np.sum((te >= 100) | (ae >= 30))),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--anchor-map", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--strengths", type=float, nargs="+", default=(0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--keypoints", type=int, default=2048)
    parser.add_argument("--baseline-keypoints", type=int, default=2048)
    parser.add_argument("--candidate-keypoints", type=int, default=4096)
    parser.add_argument("--protected-cores", type=int, nargs="+", default=())
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("invalid shard index/count")
    all_paths = sorted(args.dataset_root.glob(f"{args.split}_*.pt"))
    paths = all_paths[args.shard_index::args.shard_count]
    if not paths:
        raise FileNotFoundError(f"V11 {args.split} split is empty")
    state = torch.load(args.anchor_map, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("lineage", {}).get("pose_contribution_weighting") is not True:
        raise ValueError("V11 selection requires pose-contribution training")
    device = torch.device(args.device)
    head = load_scene_detector_checkpoint(
        checkpoint, map_sha256=sha256_file(args.anchor_map)
    ).to(device).eval()
    encoder = SuperPoint().to(device).eval()
    encoder.nms_radius = 4
    anchors = F.normalize(torch.as_tensor(state["anchor_features"], device=device).float(), dim=1)
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    protected_mode = bool(args.protected_cores)
    arms = {"native": []}
    if protected_mode:
        for core in args.protected_cores:
            arms[f"protected_{int(core)}"] = []
    else:
        for strength in args.strengths:
            if not 0 < float(strength) <= 1:
                raise ValueError("selection strengths must be in (0,1]")
            arms[f"strength_{float(strength):.4f}"] = []
    query_metadata = []
    for index, path in enumerate(paths):
        record = torch.load(path, map_location="cpu", weights_only=False)
        if record.get("uses_test_rgb") is not False or record.get("loo_used") is not False:
            raise ValueError("validation detector data violates V11 lineage")
        source_path = Path(record["source_record"])
        if sha256_file(source_path) != record["source_record_sha256"]:
            raise ValueError("V11 validation source SHA256 differs")
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        image = torch.as_tensor(source["rgb_float16"], device=device).float()[None]
        dense, native_scores = encoder._dense_outputs(image)
        logits = head(dense, output_hw=tuple(native_scores.shape[-2:]))
        sparse_arms = {}
        if protected_mode:
            sparse_candidates = encoder._sparse_from_dense(
                dense, native_scores, top_k=args.candidate_keypoints,
                detection_threshold=0.0,
            )[0]
            native_indices = torch.arange(args.baseline_keypoints, device=device)
            sparse_arms["native"] = {key: value[native_indices] for key, value in sparse_candidates.items()}
            for core in args.protected_cores:
                chosen = protected_scene_candidate_indices(
                    keypoints=sparse_candidates["keypoints"],
                    native_scores=sparse_candidates["keypoint_scores"],
                    detector_logits=logits[0], output_count=args.keypoints,
                    protected_native_count=core,
                )
                sparse_arms[f"protected_{int(core)}"] = {
                    key: value[chosen] for key, value in sparse_candidates.items()
                }
        else:
            score_arms = {"native": native_scores}
            score_arms.update({
                f"strength_{float(strength):.4f}": fuse_scene_reliability(
                    native_scores, logits, strength=float(strength)
                ) for strength in args.strengths
            })
            sparse_arms = {
                name: encoder._sparse_from_dense(
                    dense, scores, top_k=args.keypoints, detection_threshold=0.0
                )[0] for name, scores in score_arms.items()
            }
        for name, sparse in sparse_arms.items():
            matches = global_cosine_top1(
                F.normalize(sparse["descriptors"], dim=1), anchors,
                anchor_descriptors_normalized=True,
            )
            pose = standard_pose_replay(
                keypoints=sparse["keypoints"].cpu() + 0.5,
                anchor_rows=matches.anchor_indices.cpu(), anchor_xyz=xyz,
                intrinsic=source["intrinsics"], ground_truth_w2c=source["pose_w2c"],
            )
            arms[name].append(pose)
        native_points = sparse_arms["native"]["keypoints"].round().long()
        native_points[:, 0].clamp_(0, logits.shape[-1] - 1)
        native_points[:, 1].clamp_(0, logits.shape[-2] - 1)
        native_reliability = torch.sigmoid(
            logits[0, native_points[:, 1], native_points[:, 0]]
        )
        native_keypoint_scores = sparse_arms["native"]["keypoint_scores"].float()
        detector_name = next((name for name in sparse_arms if name != "native"), None)
        overlap = float("nan")
        if detector_name is not None:
            detector_points = sparse_arms[detector_name]["keypoints"].round().long()
            native_linear = native_points[:, 1] * logits.shape[-1] + native_points[:, 0]
            detector_linear = detector_points[:, 1] * logits.shape[-1] + detector_points[:, 0]
            overlap = float(torch.isin(detector_linear, native_linear).float().mean())
        query_metadata.append({
            "query_index": int(record["query_index"]),
            "mean_native_reliability": float(native_reliability.mean()),
            "p10_native_reliability": float(torch.quantile(native_reliability, 0.1)),
            "std_native_reliability": float(native_reliability.std()),
            "fraction_native_reliability_ge_05": float((native_reliability >= 0.5).float().mean()),
            "mean_native_keypoint_score": float(native_keypoint_scores.mean()),
            "std_native_keypoint_score": float(native_keypoint_scores.std()),
            "detector_native_overlap": overlap,
        })
        if (index + 1) % 8 == 0 or index + 1 == len(paths):
            print(f"V11 validation: {index + 1}/{len(paths)}", flush=True)
    summaries = {name: _summary(rows) for name, rows in arms.items()}
    baseline = summaries["native"]
    action_candidates = []
    action_values = args.protected_cores if protected_mode else args.strengths
    for value in action_values:
        name = f"protected_{int(value)}" if protected_mode else f"strength_{float(value):.4f}"
        row = summaries[name]
        row["median_task_gain"] = baseline["median_task_error"] - row["median_task_error"]
        row["r5_delta_percent"] = row["r5_percent"] - baseline["r5_percent"]
        row["eligible"] = bool(
            row["median_task_gain"] > 0
            and row["r5_delta_percent"] >= -0.01
            and row["catastrophic_count"] <= baseline["catastrophic_count"]
        )
        if row["eligible"]:
            # Prefer more protected native rows after the two quality criteria.
            safety_tie = int(value) if protected_mode else -float(value)
            action_candidates.append((row["median_task_gain"], -row["p90_task_error"], safety_tie, value))
    selected = max(action_candidates)[-1] if action_candidates else None
    paired_rows = [
        {
            **query_metadata[index],
            **{
                name: {
                    "task_error": float(rows[index]["task_error"]),
                    "translation_error_cm": float(rows[index]["translation_error_cm"]),
                    "rotation_error_deg": float(rows[index]["rotation_error_deg"]),
                }
                for name, rows in arms.items()
            },
        }
        for index in range(len(paths))
    ]
    output = {
        "schema": "lafgs_v11_detector_validation_selection",
        "version": 1,
        "loo_used": False,
        "uses_test_queries": False,
        "validation_can_select": args.split == "validation",
        "train_can_supervise_action_gate": args.split == "train",
        "split": args.split,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "query_count": len(paths),
        "input": {
            "dataset_manifest": str((args.dataset_root / "manifest.json").resolve()),
            "dataset_manifest_sha256": sha256_file(args.dataset_root / "manifest.json"),
            "anchor_map": str(args.anchor_map.resolve()),
            "anchor_map_sha256": sha256_file(args.anchor_map),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
        },
        "baseline": baseline,
        "arms": summaries,
        "selected_strength": selected,
        "selection_mode": "protected_native_core" if protected_mode else "bounded_strength",
        "candidate_keypoints": args.candidate_keypoints if protected_mode else args.keypoints,
        "paired_rows": paired_rows,
        "decision": "PROPOSE" if selected is not None else "ROLLBACK",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
