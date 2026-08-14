#!/usr/bin/env python3
"""Evaluate full-mapping rendered Tracks with leave-one-query-out descriptors.

All mapping observations remain part of Track identity and ray geometry.  For
feedback query q only its own descriptor observations are removed from the
affected map vectors before the one global top-1 lookup and one PoseLib solve.
This is a self-match guard, not a held-out construction fold.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from evidence.tracks import LeaveOneQueryOutTrackDescriptorBank
from topology.deployment_revision import collect_deployment_statistics


_SOURCE_PATHS = (
    "scripts/evaluate_rendered_track_fullmap.py",
    "evidence/tracks.py",
    "topology/deployment_revision.py",
    "localization/localizer.py",
    "localization/pose_solver.py",
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
        raise RuntimeError("full-mapping feedback producer worktree must be clean")
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
    if actual != str(expected):
        raise ValueError(f"{label} SHA differs: expected {expected}, got {actual}")
    return actual


def _atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
        if reloaded.get("schema") != payload.get("schema"):
            raise RuntimeError("temporary statistics did not reload")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        reloaded = json.loads(temporary.read_text())
        if reloaded.get("schema") != payload.get("schema"):
            raise RuntimeError("temporary report did not reload")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class _DeviceBankUpdater:
    def __init__(
        self, replay: LeaveOneQueryOutTrackDescriptorBank, device: torch.device
    ) -> None:
        self.replay = replay
        self.base = F.normalize(replay.reference_features, dim=1).to(device)
        self.previous_rows = torch.empty(0, dtype=torch.long, device=device)
        self.affected_anchor_updates = 0

    def __call__(self, query_index: int, bank: torch.Tensor) -> None:
        if bank.shape != self.base.shape or bank.device != self.base.device:
            raise ValueError("deployment bank differs from LOO reference bank")
        if self.previous_rows.numel():
            bank[self.previous_rows] = self.base[self.previous_rows]
        rows, features = self.replay.query_update(query_index)
        device_rows = rows.to(bank.device)
        if device_rows.numel():
            bank[device_rows] = F.normalize(features, dim=1).to(bank.device)
        self.previous_rows = device_rows
        self.affected_anchor_updates += int(rows.numel())


def run(args: argparse.Namespace) -> dict:
    identity = _producer_identity()
    paths = {
        "map": args.map.resolve(),
        "metric": args.metric_state.resolve(),
        "track_payload": args.track_payload.resolve(),
        "teacher": args.teacher.resolve(),
        "query_cache": args.query_cache.resolve(),
        "scene_calibration": args.scene_calibration.resolve(),
    }
    expected = {
        "map": args.expected_map_sha256,
        "metric": args.expected_metric_sha256,
        "track_payload": args.expected_track_payload_sha256,
        "teacher": args.expected_teacher_sha256,
        "query_cache": args.expected_query_cache_sha256,
        "scene_calibration": args.expected_scene_calibration_sha256,
    }
    input_sha256 = {
        label: _require_sha(path, expected[label], label)
        for label, path in paths.items()
    }
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    state = torch.load(paths["map"], map_location="cpu", weights_only=False)
    metric = torch.load(paths["metric"], map_location="cpu", weights_only=False)
    payload = torch.load(paths["track_payload"], map_location="cpu", weights_only=False)
    teacher = torch.load(paths["teacher"], map_location="cpu", weights_only=False)
    cache = torch.load(paths["query_cache"], map_location="cpu", weights_only=False)
    calibration = json.loads(paths["scene_calibration"].read_text())
    if (
        cache.get("uses_source_mapping_rgb") is not False
        or cache.get("uses_test_queries") is not False
        or payload.get("rendered_rgb_only") is not True
    ):
        raise ValueError("full-mapping feedback inputs are not source-image-free")
    names = list(payload["query_names"])
    if names != list(teacher.get("query_names", ())) or names != list(
        cache.get("queries", cache)
    ):
        raise ValueError("mapping query registries are not exact and ordered")
    if int(teacher.get("anchor_count", -1)) != int(state["anchor_ids"].numel()):
        raise ValueError("teacher and map anchor counts differ")
    if (
        Path(str(teacher.get("query_cache", ""))).resolve() != paths["query_cache"]
        or str(teacher.get("query_cache_sha256", "")) != input_sha256["query_cache"]
    ):
        raise ValueError("teacher does not bind the rendered mapping cache")
    if (
        metric.get("map_path") != str(paths["map"])
        or metric.get("map_sha256") != input_sha256["map"]
    ):
        raise ValueError("metric is not bound to the exact evaluated map")
    calibration_sources = calibration.get("sources", {})
    if (
        calibration.get("schema") != "lafgs_mapping_only_scene_calibration"
        or calibration_sources.get("uses_source_mapping_rgb") is not False
        or calibration_sources.get("uses_test_queries") is not False
        or calibration_sources.get("mapping_source") != "gaussian_render"
        or Path(str(calibration_sources.get("query_cache", ""))).resolve()
        != paths["query_cache"]
    ):
        raise ValueError("scene calibration is not bound source-image-free evidence")

    replay = LeaveOneQueryOutTrackDescriptorBank(
        payload=payload,
        query_cache=cache,
        track_indices=state["track_cluster_ids"],
        reference_features=state["anchor_features"],
        trim_fraction=float(args.descriptor_trim_fraction),
    )
    affected = torch.as_tensor([len(rows) for rows in replay.rows_by_query]).long()
    device = torch.device(args.device)
    updater = _DeviceBankUpdater(replay, device)
    parameters = calibration["parameters"]
    statistics = collect_deployment_statistics(
        state=state,
        metric_state_path=paths["metric"],
        teacher=teacher,
        query_cache=cache,
        device=device,
        ransac_reprojection_px=float(parameters["ransac_reprojection_px"]),
        clean_reprojection_px=float(parameters["clean_radius_px"]),
        task_translation_m=float(parameters["task_translation_m"]),
        task_rotation_deg=float(parameters["task_rotation_deg"]),
        seed=int(args.seed),
        deployment_row_limit=int(args.deployment_row_limit),
        collect_anchor_statistics=True,
        progress_label="rendered_track_full_mapping_loo_feedback",
        anchor_bank_updater=updater,
    )
    statistics = {
        "schema": "lafgs_rendered_track_full_mapping_loo_statistics",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "queries": statistics["queries"],
        "counters": statistics["counters"],
        "summary": statistics["summary"],
        "loo": {
            "construction_uses_all_mapping_observations": True,
            "track_identity_and_geometry_remain_full_mapping": True,
            "query_descriptor_excluded_from_affected_anchor_fusion": True,
            "affected_anchor_updates": updater.affected_anchor_updates,
            "minimum_affected_anchors_per_query": int(affected.min()),
            "maximum_affected_anchors_per_query": int(affected.max()),
            "mean_affected_anchors_per_query": float(affected.float().mean()),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    statistics_path = args.output_dir / "full_mapping_loo_statistics.pt"
    _atomic_save(statistics, statistics_path)
    if _producer_identity() != identity:
        raise RuntimeError("full-mapping feedback producer identity changed")
    for label, path in paths.items():
        _require_sha(path, input_sha256[label], label)
    report = {
        "schema": "lafgs_rendered_track_full_mapping_loo_report",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "formal_method_uses_crossfit": False,
        "protocol": (
            "full_mapping_construction_and_feedback_with_"
            "leave_one_query_observation_out_descriptor_fusion"
        ),
        "producer_identity": identity,
        "seed": int(args.seed),
        "configuration": {
            "descriptor_trim_fraction": float(args.descriptor_trim_fraction),
            "deployment_row_limit": int(args.deployment_row_limit),
            "one_global_top1_per_query_row": True,
            "one_poselib_call_per_mapping_query": True,
        },
        "inputs": {label: str(path) for label, path in paths.items()},
        "input_sha256": input_sha256,
        "statistics": str(statistics_path.resolve()),
        "statistics_sha256": sha256_file(statistics_path),
        "loo": statistics["loo"],
        "summary": statistics["summary"],
    }
    report_path = args.output_dir / "full_mapping_loo_report.json"
    _atomic_json(report, report_path)
    return report


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
    parser.add_argument("--scene-calibration", type=Path, required=True)
    parser.add_argument("--expected-scene-calibration-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--descriptor-trim-fraction", type=float, default=0.2)
    parser.add_argument("--deployment-row-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
