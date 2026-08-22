from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from common.hashing import sha256_file
from evidence.observation_provider import GaussianRenderObservationProvider
from evidence.virtual_camera_registry import (
    build_virtual_camera_registry,
    resolve_virtual_camera_registry,
)
from scripts.probe_rendered_rgb_track_map import _resolve_render_cameras


def _camera(name: str, x: float, external_id: int):
    pose = torch.eye(4)
    pose[0, 3] = x
    return SimpleNamespace(
        image_name=name,
        colmap_id=external_id,
        image_path=f"/missing/{name}",
        width=80,
        height=48,
        fov_x=1.0,
        fov_y=0.8,
        pose_w2c=pose,
    )


def test_registry_resolves_cross_stage_by_name_not_stale_dataset_indices():
    source = [_camera("z", 1.0, 900), _camera("a", -1.0, 2), _camera("m", 0.0, 77)]
    selected, registry = build_virtual_camera_registry(source, 2)
    assert [camera.image_name for camera in selected] == ["a", "m"]
    # The next stage has a different dataset order. An index-based consumer
    # would select z/a here; the registry must recover a/m exactly.
    reordered = [source[0], source[2], source[1]]
    resolved = resolve_virtual_camera_registry(reordered, registry)
    assert [camera.image_name for camera in resolved] == ["a", "m"]
    assert registry["policy"] == "geometry_pose_intrinsics_v1"
    assert len(registry["registry_sha256"]) == 64


def test_registry_tamper_and_duplicate_geometry_fail_closed():
    source = [_camera("a", -1.0, 1), _camera("b", 1.0, 2)]
    _, registry = build_virtual_camera_registry(source)
    tampered = deepcopy(registry)
    tampered["selected_camera_names"] = list(reversed(tampered["selected_camera_names"]))
    with pytest.raises(ValueError, match="identity"):
        resolve_virtual_camera_registry(source, tampered)
    duplicate = [_camera("a", 0.0, 1), _camera("b", 0.0, 999)]
    with pytest.raises(ValueError, match="duplicate geometry"):
        build_virtual_camera_registry(duplicate)


def test_filename_path_is_not_implicit_sequence_metadata():
    record = {
        "native_keypoints": torch.zeros(1, 2),
        "native_descriptors": torch.ones(1, 2),
        "native_scores": torch.ones(1),
        "native_K": torch.eye(3),
        "pose_w2c": torch.eye(4),
        "native_input_hw": [2, 2],
    }
    provider = GaussianRenderObservationProvider({
        "uses_source_mapping_rgb": False,
        "queries": {"looks/like/sequence/frame.png": record},
    })
    assert provider.build_view(0).sequence_id is None
    assert provider.track_inputs()["query_groups"] == [None]


def test_explicit_registry_controls_render_schedule_and_is_sha_bound(tmp_path):
    source = [_camera("z", 1.0, 900), _camera("a", -1.0, 2), _camera("m", 0.0, 77)]
    _, registry = build_virtual_camera_registry(source, 2)
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry))
    args = SimpleNamespace(
        camera_registry=path,
        expected_camera_registry_sha256=sha256_file(path),
        max_views=0,
    )
    resolved, replayed, actual = _resolve_render_cameras(
        list(reversed(source)), args
    )
    assert [camera.image_name for camera in resolved] == ["a", "m"]
    assert replayed == registry
    assert actual == args.expected_camera_registry_sha256
    args.expected_camera_registry_sha256 = "0" * 64
    with pytest.raises(ValueError, match="SHA differs"):
        _resolve_render_cameras(source, args)
