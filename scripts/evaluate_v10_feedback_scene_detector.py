#!/usr/bin/env python3
"""Paired native-vs-feedback-detector evaluation on a sealed render batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from features.scene_specific_detector import (
    fuse_scene_reliability,
    load_scene_detector_checkpoint,
    protected_scene_candidate_indices,
    mean_candidate_reliability,
)
from features.superpoint import SuperPoint
from features.scene_action_gate import load_scene_action_gate, query_action_features
from localization.matcher import global_cosine_top1
from map_learning.v9_causal_feedback import standard_pose_replay


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certified-batch", type=Path, required=True)
    parser.add_argument("--anchor-map", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-role", choices=("feedback_query", "confirmation_query"), required=True)
    parser.add_argument("--keypoints", type=int, default=2048)
    parser.add_argument("--baseline-keypoints", type=int, default=2048)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--candidate-keypoints", type=int, default=4096)
    parser.add_argument("--protected-native-count", type=int)
    parser.add_argument("--abstain-threshold", type=float)
    parser.add_argument("--action-gate", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    batch_path = args.certified_batch.resolve()
    batch = json.loads(batch_path.read_text())
    if not (
        batch.get("view_role") == args.expected_role
        and batch.get("uses_test_queries") is False
        and batch.get("map_mutation_count") == 0
    ):
        raise ValueError("detector evaluation requires a sealed non-test batch")
    map_path = args.anchor_map.resolve()
    checkpoint_path = args.checkpoint.resolve()
    state = torch.load(map_path, map_location="cpu", weights_only=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    head = load_scene_detector_checkpoint(
        checkpoint, map_sha256=sha256_file(map_path)
    ).to(device).eval()
    action_gate = None
    if args.action_gate is not None:
        gate_state = torch.load(args.action_gate, map_location="cpu", weights_only=False)
        action_gate = load_scene_action_gate(
            gate_state, map_sha256=sha256_file(map_path),
            detector_sha256=sha256_file(checkpoint_path),
        ).to(device).eval()
    if checkpoint.get("lineage", {}).get("feedback_match_supervision") is not True:
        raise ValueError("checkpoint lacks feedback-match detector lineage")
    encoder = SuperPoint().to(device).eval()
    encoder.nms_radius = 4
    anchors = F.normalize(
        torch.as_tensor(state["anchor_features"], device=device).float(), dim=1
    )
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    selected = [
        row for index, row in enumerate(batch["records"])
        if index % args.shard_count == args.shard_index
    ]
    records = []
    for index, item in enumerate(selected):
        source_path = Path(item["path"])
        if sha256_file(source_path) != item["sha256"]:
            raise ValueError("certified detector record SHA256 differs")
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        if source["certificate"]["decision"] != "ACCEPT":
            continue
        image = torch.as_tensor(source["rgb_float16"], device=device).float()[None]
        dense, native_scores = encoder._dense_outputs(image)
        detector_logits = head(dense, output_hw=tuple(native_scores.shape[-2:]))
        arm_features = {}
        if args.protected_native_count is None:
            detector_scores = fuse_scene_reliability(
                native_scores, detector_logits, strength=args.strength,
            )
            baseline_sparse = encoder._sparse_from_dense(
                dense, native_scores, top_k=args.keypoints, detection_threshold=0.0
            )[0]
            detector_sparse = None
            if action_gate is not None:
                detector_sparse = encoder._sparse_from_dense(
                    dense, detector_scores, top_k=args.keypoints,
                    detection_threshold=0.0,
                )[0]
                gate_features = query_action_features(
                    native_keypoints=baseline_sparse["keypoints"],
                    native_scores=baseline_sparse["keypoint_scores"],
                    detector_keypoints=detector_sparse["keypoints"],
                    detector_logits=detector_logits[0],
                )
                activate = bool(action_gate(gate_features) >= 0)
            else:
                activate = (
                    args.abstain_threshold is None
                    or float(mean_candidate_reliability(
                        baseline_sparse["keypoints"], detector_logits[0]
                    )) >= float(args.abstain_threshold)
                )
            if activate and detector_sparse is None:
                detector_sparse = encoder._sparse_from_dense(
                    dense, detector_scores, top_k=args.keypoints,
                    detection_threshold=0.0,
                )[0]
            proposal_sparse = detector_sparse if activate else baseline_sparse
            sparse_arms = {"baseline": baseline_sparse, "proposal": proposal_sparse}
        else:
            candidates = encoder._sparse_from_dense(
                dense, native_scores, top_k=args.candidate_keypoints,
                detection_threshold=0.0,
            )[0]
            baseline_indices = torch.arange(args.baseline_keypoints, device=device)
            proposal_indices = protected_scene_candidate_indices(
                keypoints=candidates["keypoints"],
                native_scores=candidates["keypoint_scores"],
                detector_logits=detector_logits[0], output_count=args.keypoints,
                protected_native_count=args.protected_native_count,
            )
            sparse_arms = {
                "baseline": {key: value[baseline_indices] for key, value in candidates.items()},
                "proposal": {key: value[proposal_indices] for key, value in candidates.items()},
            }
        for name, sparse in sparse_arms.items():
            query = F.normalize(sparse["descriptors"], dim=1)
            matches = global_cosine_top1(
                query, anchors, anchor_descriptors_normalized=True
            )
            result = standard_pose_replay(
                keypoints=sparse["keypoints"].cpu() + 0.5,
                anchor_rows=matches.anchor_indices.cpu(),
                anchor_xyz=xyz,
                intrinsic=source["intrinsics"],
                ground_truth_w2c=source["pose_w2c"],
            )
            result["pose_w2c"] = result["pose_w2c"].tolist()
            result["selected_keypoint_count"] = int(query.shape[0])
            arm_features[name] = result
        records.append({
            "query_index": int(source["query_index"]),
            "pose_family_id": int(source["pose_family_id"]),
            "baseline": arm_features["baseline"],
            "proposal": arm_features["proposal"],
            "task_gain": arm_features["baseline"]["task_error"] - arm_features["proposal"]["task_error"],
            "detector_activated": bool(activate) if args.protected_native_count is None else True,
        })
        if (index + 1) % 8 == 0 or index + 1 == len(selected):
            print(f"V10 detector shard {args.shard_index}: {index + 1}/{len(selected)}", flush=True)
    output = {
        # Keep the paired schema so the already-frozen decision aggregator is reused.
        "schema": "lafgs_v9_paired_confirmation_shard",
        "version": 1,
        "loo_used": False,
        "uses_test_queries": False,
        "confirmation_can_train_or_select": False,
        "evaluation_role": args.expected_role,
        "action": "single_pass_feedback_scene_detector",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "source_query_count": len(batch["records"]),
        "accepted_query_count": len(records),
        "input": {
            "certified_batch": str(batch_path),
            "certified_batch_sha256": sha256_file(batch_path),
            "anchor_map": str(map_path),
            "anchor_map_sha256": sha256_file(map_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "strength": float(args.strength),
            "candidate_keypoints": int(args.candidate_keypoints),
            "baseline_keypoints": int(args.baseline_keypoints),
            "protected_native_count": args.protected_native_count,
            "abstain_threshold": args.abstain_threshold,
            "action_gate": None if args.action_gate is None else str(args.action_gate.resolve()),
            "action_gate_sha256": None if args.action_gate is None else sha256_file(args.action_gate),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
