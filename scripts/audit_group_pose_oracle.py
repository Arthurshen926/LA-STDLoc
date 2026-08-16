#!/usr/bin/env python3
"""Mapping-only structured-outlier hypothesis oracle for a frozen Track map."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from evidence.tracks import (
    LeaveOneQueryOutProjectiveAnchorDescriptorBank,
    LeaveOneQueryOutTrackDescriptorBank,
)
from localization.group_consensus import (
    build_standard_and_group_diverse_hypotheses,
    classify_hypothesis_oracle,
    correlation_groups_from_map,
)
from localization.localizer import load_shared_metric
from map_learning.trainer import bounded_anchor_bank, track_descriptor_payload_for_loo
from scripts.evaluate_rendered_track_fullmap import _DeviceBankUpdater


_SOURCE_PATHS = (
    "scripts/audit_group_pose_oracle.py",
    "evidence/observation_provider.py",
    "evidence/tracks.py",
    "localization/group_consensus.py",
    "map_learning/trainer.py",
    "scripts/evaluate_rendered_track_fullmap.py",
)


def _producer_identity() -> dict:
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("group oracle producer worktree must be clean")
    return {
        "git_commit": commit,
        "worktree_clean": True,
        "source_sha256": {
            relative: sha256_file(repository / relative) for relative in _SOURCE_PATHS
        },
        "torch_version": torch.__version__,
    }


def _require_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA differs: expected {expected}, got {actual}")
    return actual


def _atomic_json(payload: dict, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        reloaded = json.loads(temporary.read_text())
        if reloaded.get("schema") != payload.get("schema"):
            raise RuntimeError("temporary group oracle did not reload")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict:
    producer = _producer_identity()
    paths = {
        "map": args.map.resolve(),
        "metric": args.metric_state.resolve(),
        "track_payload": args.track_payload.resolve(),
        "teacher": args.teacher.resolve(),
        "query_cache": args.query_cache.resolve(),
    }
    expected = {
        "map": args.expected_map_sha256,
        "metric": args.expected_metric_sha256,
        "track_payload": args.expected_track_payload_sha256,
        "teacher": args.expected_teacher_sha256,
        "query_cache": args.expected_query_cache_sha256,
    }
    input_sha = {
        name: _require_sha(path, expected[name], name) for name, path in paths.items()
    }
    state = torch.load(paths["map"], map_location="cpu", weights_only=False)
    metric_state = torch.load(paths["metric"], map_location="cpu", weights_only=False)
    payload = torch.load(paths["track_payload"], map_location="cpu", weights_only=False)
    teacher = torch.load(paths["teacher"], map_location="cpu", weights_only=False)
    cache_payload = torch.load(
        paths["query_cache"], map_location="cpu", weights_only=False
    )
    if (
        state.get("schema") != "lafgs_materialized_anchor_map"
        or payload.get("rendered_rgb_only") is not True
        or cache_payload.get("uses_source_mapping_rgb") is not False
        or cache_payload.get("uses_test_queries") is not False
    ):
        raise ValueError(
            "group oracle requires frozen source-image-free mapping inputs"
        )
    names = list(payload["query_names"])
    cache = cache_payload.get("queries", cache_payload)
    if names != list(teacher.get("query_names", ())) or names != list(cache):
        raise ValueError("group oracle query registries are not exact and ordered")
    if int(teacher.get("anchor_count", -1)) != int(state["anchor_ids"].numel()):
        raise ValueError("group oracle teacher/map rows differ")
    if metric_state.get("map_path") != str(paths["map"]):
        raise ValueError("group oracle metric is not bound to the selected map")
    if metric_state.get("map_sha256") != input_sha["map"]:
        raise ValueError("group oracle metric map SHA differs")

    device = torch.device(args.device)
    raw_reference_features = torch.as_tensor(
        state.get("v7_metric_raw_features", state["anchor_features"])
    ).float()
    loo_payload = track_descriptor_payload_for_loo(payload)
    if bool((torch.as_tensor(state["track_cluster_ids"]) < 0).any()):
        replay = LeaveOneQueryOutProjectiveAnchorDescriptorBank(
            state=state,
            payload=loo_payload,
            query_cache=cache_payload,
            reference_features=raw_reference_features,
            trim_fraction=float(args.descriptor_trim_fraction),
        )
    else:
        replay = LeaveOneQueryOutTrackDescriptorBank(
            payload=loo_payload,
            query_cache=cache_payload,
            track_indices=state["track_cluster_ids"],
            reference_features=raw_reference_features,
            trim_fraction=float(args.descriptor_trim_fraction),
        )
    online_config = state.get("v7_online_metric", {}).get("config", {})
    updater = _DeviceBankUpdater(
        replay,
        device,
        metric_state=metric_state,
        adapted_reference_features=state["anchor_features"],
        anchor_residual_parameter=state.get("v7_anchor_residual_parameter"),
        anchor_residual_max_norm=float(
            online_config.get("anchor_feature_residual_max_norm", 0.0)
        ),
    )
    metric = load_shared_metric(
        paths["metric"],
        anchor_ids=torch.as_tensor(state["anchor_ids"]).long(),
        device=device,
    )
    bank, _, _ = bounded_anchor_bank(
        metric,
        F.normalize(raw_reference_features, dim=1).to(device),
        (
            None
            if state.get("v7_anchor_residual_parameter") is None
            else torch.as_tensor(state["v7_anchor_residual_parameter"])
            .float()
            .to(device)
        ),
        float(online_config.get("anchor_feature_residual_max_norm", 0.0)),
    )
    expected_bank = F.normalize(
        torch.as_tensor(state["anchor_features"]).float().to(device), dim=1
    )
    if not torch.allclose(bank, expected_bank, atol=1e-6, rtol=1e-6):
        raise ValueError("group oracle deployment bank replay differs")
    bank.copy_(expected_bank)
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    selected_queries = np.arange(len(names), dtype=np.int64)
    if int(args.maximum_queries) > 0 and int(args.maximum_queries) < len(names):
        selected_queries = np.floor(
            np.arange(int(args.maximum_queries))
            * len(names)
            / int(args.maximum_queries)
        ).astype(np.int64)
    group_fields = [
        field.strip() for field in args.group_fields.split(",") if field.strip()
    ]
    if not group_fields:
        raise ValueError("at least one correlation-group field is required")
    rows_by_field = {field: [] for field in group_fields}
    for completed, query_index in enumerate(selected_queries.tolist(), start=1):
        updater(query_index, bank)
        record = teacher["records"][query_index]
        cached = cache[names[query_index]]
        query_rows = torch.as_tensor(record["query_rows"]).long()
        descriptors = F.normalize(
            torch.as_tensor(cached["native_descriptors"]).float()[query_rows], dim=1
        ).to(device)
        adapted, _ = metric(descriptors)
        winners = torch.argmax(adapted @ bank.T, dim=1).cpu()
        points_2d = (
            torch.as_tensor(cached["native_keypoints"]).float()[query_rows]
            + float(cached.get("pixel_center_offset", 0.5))
        ).numpy()
        points_3d = xyz[winners].numpy()
        intrinsic = torch.as_tensor(cached["native_K"]).float().numpy()
        gt_pose = torch.as_tensor(cached["pose_w2c"]).float().numpy()
        for field in group_fields:
            groups = correlation_groups_from_map(state, winners.numpy(), field=field)
            standard, diverse = build_standard_and_group_diverse_hypotheses(
                points_2d,
                points_3d,
                intrinsic,
                groups,
                sample_count=int(args.hypothesis_samples),
                seed=int(args.seed) + query_index,
            )
            row = classify_hypothesis_oracle(
                standard_hypotheses_w2c=standard,
                group_diverse_hypotheses_w2c=diverse,
                points_2d=points_2d,
                points_3d=points_3d,
                intrinsic=intrinsic,
                group_ids=groups,
                ground_truth_w2c=gt_pose,
                reprojection_threshold_px=float(args.reprojection_error_px),
                correct_translation_cm=float(args.correct_translation_cm),
                correct_rotation_deg=float(args.correct_rotation_deg),
            )
            rows_by_field[field].append(
                {"query_index": query_index, "image_name": names[query_index], **row}
            )
        if completed % 25 == 0 or completed == len(selected_queries):
            print(
                json.dumps(
                    {
                        "event": "group_pose_oracle",
                        "queries_complete": completed,
                        "query_count": int(len(selected_queries)),
                    }
                ),
                flush=True,
            )
    summaries = {}
    for field, rows in rows_by_field.items():
        categories = sorted({row["category"] for row in rows})
        summaries[field] = {
            "query_count": len(rows),
            "category_counts": {
                category: sum(row["category"] == category for row in rows)
                for category in categories
            },
            "standard_has_correct_rate": float(
                np.mean([row["standard_has_correct_hypothesis"] for row in rows])
            ),
            "group_diverse_has_correct_rate": float(
                np.mean([row["group_diverse_has_correct_hypothesis"] for row in rows])
            ),
            "group_capped_winner_correct_rate": float(
                np.mean([row["group_capped_winner_correct"] for row in rows])
            ),
            "authorizes_deployment_solver_change": False,
        }
    result = {
        "schema": "lafgs_structured_outlier_group_pose_oracle",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "mapping_leave_one_query_descriptor_out": True,
        "hypothesis_source": "deterministic_opencv_ap3p_offline_oracle",
        "not_exact_poselib_internal_hypotheses": True,
        "inputs": {name: str(path) for name, path in paths.items()},
        "input_sha256": input_sha,
        "configuration": {
            "query_count": int(len(selected_queries)),
            "hypothesis_samples_per_sampler": int(args.hypothesis_samples),
            "seed": int(args.seed),
            "reprojection_error_px": float(args.reprojection_error_px),
            "correct_translation_cm": float(args.correct_translation_cm),
            "correct_rotation_deg": float(args.correct_rotation_deg),
            "group_fields": group_fields,
        },
        "producer": producer,
        "summaries": summaries,
        "queries": rows_by_field,
        "decision": "DIAGNOSTIC_ONLY_DO_NOT_CHANGE_DEPLOYMENT_SOLVER",
        "authorizes_deployment_solver_change": False,
    }
    for name, path in paths.items():
        _require_sha(path, input_sha[name], name)
    _atomic_json(result, args.output.resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--expected-map-sha256", required=True)
    parser.add_argument("--metric-state", type=Path, required=True)
    parser.add_argument("--expected-metric-sha256", required=True)
    parser.add_argument("--track-payload", type=Path, required=True)
    parser.add_argument("--expected-track-payload-sha256", required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--expected-teacher-sha256", required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--expected-query-cache-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--group-fields", default="parent_source_track_ids,coarse_dependency_group_ids"
    )
    parser.add_argument("--hypothesis-samples", type=int, default=32)
    parser.add_argument("--maximum-queries", type=int, default=0)
    parser.add_argument("--descriptor-trim-fraction", type=float, default=0.2)
    parser.add_argument("--reprojection-error-px", type=float, required=True)
    parser.add_argument("--correct-translation-cm", type=float, default=5.0)
    parser.add_argument("--correct-rotation-deg", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args)["summaries"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
