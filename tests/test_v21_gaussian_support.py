from copy import deepcopy

import pytest
import torch

from map_learning.v21_gaussian_support import (
    EVIDENCE_SEMANTICS,
    SCHEMA,
    VERSION,
    build_support_record,
    sample_keypoint_raster_support,
    sha256_json,
    validate_support_payload,
)
from scripts.materialize_v21_gaussian_support import _render_support


HEX = "a" * 64


def _frontend_record() -> dict:
    return {
        "query_index": 7,
        "image_name": "seq/frame00007.png",
        "image_sha256": "b" * 64,
        "sequence_id": "seq",
        "frame_index": 7,
        "block_id": "seq:trajectory_0000",
        "role": "adaptation",
        "source_record_sha256": "c" * 64,
        "pose_w2c_sha256": "d" * 64,
        "keypoints": torch.tensor([[1.0, 1.0], [3.6, 1.0]]),
        "intrinsics": torch.eye(3),
        "image_hw": torch.tensor([3, 4]),
    }


def _source(path: str, digest: str = HEX) -> dict:
    return {"path": path, "sha256": digest, "size_bytes": 10}


def _payload() -> dict:
    frontend = _frontend_record()
    sampled = sample_keypoint_raster_support(
        keypoints=frontend["keypoints"],
        depth=torch.arange(1, 13, dtype=torch.float32).reshape(3, 4),
        alpha=torch.ones(3, 4),
        image_hw=(3, 4),
    )
    record = build_support_record(
        frontend_record=frontend,
        frontend_cache_path="/tmp/cache.pt",
        frontend_cache_sha256="e" * 64,
        frontend_shard_index=0,
        sampled=sampled,
        pixel_center_offset=0.5,
    )
    core = {
        "schema": "lafgs_v21_test_cache_shard_registry",
        "version": 1,
        "role": "adaptation",
        "split_manifest_sha256": "f" * 64,
        "assignment": "ordered_role_query_registry_modulo_shard_count",
        "shard_count": 1,
        "role_query_count": 1,
        "rows": [
            {
                "ordinal": 0,
                "shard_index": 0,
                "query_index": 7,
                "image_name": frontend["image_name"],
                "image_sha256": frontend["image_sha256"],
                "source_record_sha256": frontend["source_record_sha256"],
            }
        ],
    }
    registry = {**core, "registry_sha256": sha256_json(core)}
    render = {
        "gaussian_type": "2dgs",
        "requested_rasterize_mode": "antialiased",
        "effective_rasterize_mode": "omitted_unsupported_by_2dgs_wrapper",
        "rasterize_mode_argument_forwarded": False,
        "pixel_center_offset": 0.5,
        "stored_rasters": False,
    }
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "protocol": "test_adapted",
        "uses_test_queries": True,
        "test_adapted": True,
        "role": "adaptation",
        "training_consumers_allowed": True,
        "ground_truth_pose_is_delayed_feedback_authority": True,
        "control_or_confirmation_forbidden": True,
        "correspondence_truth_claimed": False,
        "negative_labels_created": False,
        "deployment_authority": False,
        "evidence_semantics": dict(EVIDENCE_SEMANTICS),
        "split_manifest_sha256": "f" * 64,
        "stable_map_sha256": "1" * 64,
        "gaussian_ply_sha256": "2" * 64,
        "gaussian_primitive_count": 12,
        "frontend_shard_registry": registry,
        "frontend_shard_registry_sha256": registry["registry_sha256"],
        "source_frontend_shards": [
            {
                **_source("/tmp/cache.pt", "e" * 64),
                "shard_index": 0,
                "shard_count": 1,
                "query_count": 1,
            }
        ],
        "query_count": 1,
        "render_contract": render,
        "render_contract_sha256": sha256_json(render),
        "inputs": {
            "split_manifest": _source("/tmp/split.json", "f" * 64),
            "stable_map": _source("/tmp/map.pt", "1" * 64),
            "gaussian_ply": _source("/tmp/prior.ply", "2" * 64),
            "frontend_caches": [_source("/tmp/cache.pt", "e" * 64)],
            "producer_sources": [_source("/tmp/producer.py", "3" * 64)],
        },
        "records": [record],
    }


def test_sparse_sampling_uses_pixel_center_and_marks_outside_rows_invalid():
    result = sample_keypoint_raster_support(
        keypoints=torch.tensor([[1.0, 1.0], [3.6, 1.0]]),
        depth=torch.arange(1, 13, dtype=torch.float32).reshape(3, 4),
        alpha=torch.ones(3, 4),
        image_hw=(3, 4),
        pixel_center_offset=0.5,
    )
    assert result["sample_pixel_xy"].tolist() == [[1, 1], [4, 1]]
    assert result["gaussian_depth_at_keypoints"][0].item() == 6.0
    assert torch.isnan(result["gaussian_depth_at_keypoints"][1])
    assert result["gaussian_support_valid"].tolist() == [True, False]
    assert result["gaussian_relative_depth_spread_3x3"][0].item() == pytest.approx(
        10.0 / 6.0
    )
    assert torch.isinf(result["gaussian_relative_depth_spread_3x3"][1])


def test_sparse_sampling_does_not_turn_missing_depth_into_support():
    depth = torch.ones(3, 3)
    depth[1, 1] = float("nan")
    result = sample_keypoint_raster_support(
        keypoints=torch.tensor([[1.0, 1.0]]),
        depth=depth,
        alpha=torch.ones(3, 3),
        image_hw=(3, 3),
    )
    assert result["gaussian_support_valid"].tolist() == [False]
    assert result["gaussian_local_valid_fraction_3x3"].item() == pytest.approx(8 / 9)


def test_render_adapter_can_be_exercised_with_a_cpu_synthetic_renderer():
    record = {
        **_frontend_record(),
        "pose_w2c": torch.eye(4),
    }
    calls = []

    def fake_render(model, pose, fov_x, fov_y, width, height, **kwargs):
        calls.append((model, pose.device.type, width, height, kwargs))
        return {
            "depth": torch.full((1, height, width), 2.0),
            "rend_alpha": torch.ones(1, height, width),
        }

    contract = {
        "render_mode": "RGB+ED",
        "rgb_only": True,
        "requested_rasterize_mode": "antialiased",
        "effective_rasterize_mode": "omitted_unsupported_by_2dgs_wrapper",
        "rasterize_mode_argument_forwarded": False,
        "pixel_center_offset": 0.5,
    }
    result = _render_support(
        model="synthetic",
        record=record,
        render_fn=fake_render,
        device=torch.device("cpu"),
        render_contract=contract,
    )
    assert calls[0][:4] == ("synthetic", "cpu", 4, 3)
    assert calls[0][4]["render_mode"] == "RGB+ED"
    assert "rasterize_mode" not in calls[0][4]
    assert result["gaussian_support_valid"].tolist() == [True, False]


def test_payload_is_full_coverage_and_explicitly_non_authorizing():
    payload = _payload()
    validate_support_payload(payload)
    bad = deepcopy(payload)
    bad["deployment_authority"] = True
    with pytest.raises(ValueError, match="unsupported"):
        validate_support_payload(bad)
    misleading = deepcopy(payload)
    misleading["render_contract"]["rasterize_mode"] = "antialiased"
    misleading["render_contract_sha256"] = sha256_json(
        misleading["render_contract"]
    )
    with pytest.raises(ValueError, match="effective 2DGS"):
        validate_support_payload(misleading)


def test_payload_rejects_missing_or_cross_role_shards():
    payload = _payload()
    bad = deepcopy(payload)
    bad["source_frontend_shards"] = []
    with pytest.raises(ValueError, match="coverage"):
        validate_support_payload(bad)
    bad = deepcopy(payload)
    bad["records"][0]["role"] = "control"
    with pytest.raises(ValueError, match="identity"):
        validate_support_payload(bad)
