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
from evidence.tracks import (
    LeaveOneQueryOutProjectiveAnchorDescriptorBank,
    LeaveOneQueryOutTrackDescriptorBank,
)
from map_learning.metric import SharedLowRankMetric, validate_map_bound_identity_metric
from map_learning.trainer import bounded_anchor_bank, track_descriptor_payload_for_loo
from topology.deployment_revision import collect_deployment_statistics


_SOURCE_PATHS = (
    "scripts/evaluate_rendered_track_fullmap.py",
    "evidence/observation_provider.py",
    "evidence/tracks.py",
    "topology/deployment_revision.py",
    "localization/group_consensus.py",
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
        self,
        replay: (
            LeaveOneQueryOutTrackDescriptorBank
            | LeaveOneQueryOutProjectiveAnchorDescriptorBank
        ),
        device: torch.device,
        *,
        metric_state: dict | None = None,
        adapted_reference_features: torch.Tensor | None = None,
        anchor_residual_parameter: torch.Tensor | None = None,
        anchor_residual_max_norm: float = 0.0,
    ) -> None:
        self.replay = replay
        self.raw_base = F.normalize(replay.reference_features, dim=1).to(device)
        if metric_state is None:
            metric = SharedLowRankMetric(
                descriptor_dim=self.raw_base.shape[1],
                rank=1,
                max_residual_norm=0.0,
            ).to(device)
            with torch.no_grad():
                for parameter in metric.parameters():
                    parameter.zero_()
        else:
            metric = SharedLowRankMetric(**metric_state["metric_config"]).to(device)
            metric.load_state_dict(metric_state["metric_state_dict"], strict=True)
        self.metric = metric.eval()
        self.anchor_residual_parameter = (
            None
            if anchor_residual_parameter is None
            else torch.as_tensor(anchor_residual_parameter).float().to(device)
        )
        self.anchor_residual_max_norm = float(anchor_residual_max_norm)
        with torch.no_grad():
            recomputed, _, _ = bounded_anchor_bank(
                self.metric,
                self.raw_base,
                self.anchor_residual_parameter,
                self.anchor_residual_max_norm,
            )
        self.base = (
            recomputed
            if adapted_reference_features is None
            else F.normalize(
                torch.as_tensor(adapted_reference_features).float().to(device), dim=1
            )
        )
        if self.base.shape != recomputed.shape or not torch.allclose(
            self.base, recomputed, atol=1e-6, rtol=1e-6
        ):
            maximum = float((self.base - recomputed).abs().max())
            raise ValueError(
                "trained map is not the metric transform of its raw Track bank "
                f"(maximum absolute difference {maximum})"
            )
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
            row_residual = (
                None
                if self.anchor_residual_parameter is None
                else self.anchor_residual_parameter[device_rows]
            )
            updates, _, _ = bounded_anchor_bank(
                self.metric,
                features.to(bank.device),
                row_residual,
                self.anchor_residual_max_norm,
            )
            bank[device_rows] = updates
        self.previous_rows = device_rows
        self.affected_anchor_updates += int(rows.numel())


def run(args: argparse.Namespace) -> dict:
    if int(args.cpu_threads) <= 0:
        raise ValueError("CPU thread count must be positive")
    # Track fusion is dominated by thousands of small tensor reductions.  A
    # large global Torch thread pool makes those operations dramatically
    # slower through nested scheduling overhead, especially when two scenes
    # are replayed on separate GPUs.  The operation order and values are
    # unchanged; this only bounds host-side parallelism explicitly.
    torch.set_num_threads(int(args.cpu_threads))
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

    raw_reference_features = torch.as_tensor(
        state.get("v7_metric_raw_features", state["anchor_features"])
    ).float()
    anchor_features = torch.as_tensor(state["anchor_features"])
    if (
        raw_reference_features.dtype != anchor_features.dtype
        or raw_reference_features.shape != anchor_features.shape
        or not torch.equal(raw_reference_features, anchor_features)
        or "v7_anchor_residual_parameter" in state
    ):
        raise ValueError(
            "render-only V4 map must use raw fused descriptors without learned residuals"
        )
    validate_map_bound_identity_metric(
        metric,
        descriptor_dim=int(raw_reference_features.shape[1]),
        anchor_count=int(raw_reference_features.shape[0]),
        map_path=str(paths["map"]),
        map_sha256=input_sha256["map"],
    )
    loo_payload = track_descriptor_payload_for_loo(payload)
    if bool((torch.as_tensor(state["track_cluster_ids"]) < 0).any()):
        replay = LeaveOneQueryOutProjectiveAnchorDescriptorBank(
            state=state,
            payload=loo_payload,
            query_cache=cache,
            reference_features=raw_reference_features,
            trim_fraction=float(args.descriptor_trim_fraction),
        )
    else:
        replay = LeaveOneQueryOutTrackDescriptorBank(
            payload=loo_payload,
            query_cache=cache,
            track_indices=state["track_cluster_ids"],
            reference_features=raw_reference_features,
            trim_fraction=float(args.descriptor_trim_fraction),
        )
    affected = torch.as_tensor([len(rows) for rows in replay.rows_by_query]).long()
    device = torch.device(args.device)
    online_config = state.get("v7_online_metric", {}).get("config", {})
    updater = _DeviceBankUpdater(
        replay,
        device,
        metric_state=metric,
        adapted_reference_features=state["anchor_features"],
        anchor_residual_parameter=state.get("v7_anchor_residual_parameter"),
        anchor_residual_max_norm=float(
            online_config.get("anchor_feature_residual_max_norm", 0.0)
        ),
    )
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
        pose_group_field=(args.group_field if bool(args.group_aware_pose) else None),
        group_hypothesis_samples=int(args.group_hypothesis_samples),
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
            "cpu_threads": int(args.cpu_threads),
            "descriptor_trim_fraction": float(args.descriptor_trim_fraction),
            "descriptor_transform": "none_identity_only",
            "deployment_row_limit": int(args.deployment_row_limit),
            "one_global_top1_per_query_row": True,
            "one_poselib_call_per_mapping_query": not bool(args.group_aware_pose),
            "one_robust_pose_wrapper_per_mapping_query": True,
            "group_aware_pose": bool(args.group_aware_pose),
            "group_field": args.group_field if args.group_aware_pose else None,
            "group_hypothesis_samples": (
                int(args.group_hypothesis_samples) if args.group_aware_pose else 0
            ),
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
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--group-aware-pose", action="store_true")
    parser.add_argument("--group-field", default="parent_source_track_ids")
    parser.add_argument("--group-hypothesis-samples", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
