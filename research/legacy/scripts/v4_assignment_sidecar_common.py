"""Fail-closed loading shared by V4 mapping Top-K sidecar CLIs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from common.hashing import sha256_file
from evidence.tracks import (
    LeaveOneQueryOutProjectiveAnchorDescriptorBank,
    LeaveOneQueryOutTrackDescriptorBank,
)
from map_learning.metric import validate_map_bound_identity_metric
from map_learning.trainer import track_descriptor_payload_for_loo
from scripts.evaluate_rendered_track_fullmap import _DeviceBankUpdater


@dataclass(frozen=True)
class MappingContext:
    paths: dict[str, Path]
    input_sha256: dict[str, str]
    state: dict
    metric: dict
    payload: dict
    teacher: dict
    cache: dict
    calibration: dict
    replay: object
    updater: _DeviceBankUpdater
    affected_anchors_per_query: torch.Tensor


@dataclass(frozen=True)
class ReplayContext:
    paths: dict[str, Path]
    input_sha256: dict[str, str]
    state: dict
    teacher: dict
    calibration: dict


def require_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"{label} SHA differs: expected {expected}, got {actual}")
    return actual


def load_mapping_context(args) -> MappingContext:
    torch.set_num_threads(int(args.cpu_threads))
    if int(args.cpu_threads) <= 0:
        raise ValueError("CPU thread count must be positive")
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
        label: require_sha(path, expected[label], label)
        for label, path in paths.items()
    }
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
        raise ValueError("mapping inputs are not source-image-free")
    names = list(payload["query_names"])
    if names != list(teacher.get("query_names", ())) or names != list(
        cache.get("queries", cache)
    ):
        raise ValueError("mapping query registries are not exact and ordered")
    if int(teacher.get("anchor_count", -1)) != int(state["anchor_ids"].numel()):
        raise ValueError("teacher and map Anchor counts differ")
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
        raise ValueError("calibration is not bound to source-image-free evidence")

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
        raise ValueError("V4 map does not use raw identity-metric descriptors")
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
    return MappingContext(
        paths=paths,
        input_sha256=input_sha256,
        state=state,
        metric=metric,
        payload=payload,
        teacher=teacher,
        cache=cache,
        calibration=calibration,
        replay=replay,
        updater=updater,
        affected_anchors_per_query=affected,
    )


def load_replay_context(args) -> ReplayContext:
    """Load only pose-replay tensors; hash but do not deserialize heavy parents."""
    torch.set_num_threads(int(args.cpu_threads))
    if int(args.cpu_threads) <= 0:
        raise ValueError("CPU thread count must be positive")
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
        label: require_sha(path, expected[label], label)
        for label, path in paths.items()
    }
    state = torch.load(paths["map"], map_location="cpu", weights_only=False)
    teacher = torch.load(paths["teacher"], map_location="cpu", weights_only=False)
    calibration = json.loads(paths["scene_calibration"].read_text())
    if int(teacher.get("anchor_count", -1)) != int(state["anchor_ids"].numel()):
        raise ValueError("teacher and map Anchor counts differ")
    if (
        Path(str(teacher.get("query_cache", ""))).resolve() != paths["query_cache"]
        or str(teacher.get("query_cache_sha256", "")) != input_sha256["query_cache"]
    ):
        raise ValueError("teacher does not bind the rendered mapping cache")
    calibration_sources = calibration.get("sources", {})
    if (
        calibration.get("schema") != "lafgs_mapping_only_scene_calibration"
        or calibration_sources.get("uses_source_mapping_rgb") is not False
        or calibration_sources.get("uses_test_queries") is not False
        or calibration_sources.get("mapping_source") != "gaussian_render"
        or Path(str(calibration_sources.get("query_cache", ""))).resolve()
        != paths["query_cache"]
    ):
        raise ValueError("calibration is not bound to source-image-free evidence")
    return ReplayContext(
        paths=paths,
        input_sha256=input_sha256,
        state=state,
        teacher=teacher,
        calibration=calibration,
    )


def add_mapping_input_arguments(parser) -> None:
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
    parser.add_argument("--descriptor-trim-fraction", type=float, default=0.2)
    parser.add_argument("--deployment-row-limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
