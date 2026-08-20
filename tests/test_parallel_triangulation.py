from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from evidence.parallel_triangulation import (
    robust_triangulate_associations_fresh_cpu,
)
from evidence.triangulation import robust_triangulate_associations
from map_learning.bootstrap import _track_triangulation_backend
from scripts.materialize_cycle_verified_fisher_coverage_track_factor import (
    _build_arm as build_p8_track_arm,
)
import inspect


def _look_at_pose(center: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    forward = torch.nn.functional.normalize(point - center, dim=0)
    right = torch.nn.functional.normalize(
        torch.cross(forward, forward.new_tensor([0.0, 1.0, 0.0]), dim=0), dim=0
    )
    down = torch.cross(forward, right, dim=0)
    rotation = torch.stack((right, down, forward))
    translation = -(rotation @ center)
    return torch.cat((rotation, translation[:, None]), dim=1)


def _arguments() -> dict:
    points = torch.tensor(
        [[-0.2, 0.0, 3.0], [0.1, 0.1, 3.5], [0.3, -0.1, 4.0], [0.0, 0.2, 4.5]],
        dtype=torch.float64,
    )
    centers = torch.tensor(
        [[-1.0, 0.0, 0.0], [-0.3, 0.1, 0.0], [0.4, -0.1, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    poses = torch.stack([_look_at_pose(center, points.mean(0)) for center in centers])
    camera_K = torch.tensor(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    ).repeat(4, 1, 1)
    uv = []
    rendered_depth = []
    for point in points:
        homogeneous = torch.cat((point, point.new_ones(1)))
        camera = torch.einsum("nij,j->ni", poses, homogeneous)
        pixel = torch.einsum("nij,nj->ni", camera_K, camera)
        uv.append(pixel[:, :2] / pixel[:, 2:])
        rendered_depth.append(camera[:, 2])
    return {
        "landmark_count": 4,
        "landmark_index": torch.arange(4).repeat_interleave(4),
        "query_index": torch.arange(4).repeat(4),
        "uv": torch.cat(uv),
        "rendered_depth": torch.cat(rendered_depth),
        "confidence": torch.ones(16),
        "camera_K": camera_K,
        "pose_w2c": poses,
        "query_bin": torch.arange(4),
        "minimum_views": 3,
        "minimum_view_bins": 2,
        "minimum_parallax_deg": 0.0,
        "maximum_reprojection_px": 0.1,
    }


def _assert_bitwise_equal(left: dict, right: dict) -> None:
    assert tuple(left) == tuple(right)
    assert len(left) == 25
    for field in left:
        expected = torch.as_tensor(left[field]).contiguous()
        actual = torch.as_tensor(right[field]).contiguous()
        assert expected.dtype == actual.dtype
        assert expected.shape == actual.shape
        assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8)), field


def test_fresh_cpu_triangulation_is_25_field_bitwise_exact():
    arguments = _arguments()
    serial = robust_triangulate_associations(**arguments)
    parallel = robust_triangulate_associations_fresh_cpu(
        **arguments, worker_count=2
    )
    _assert_bitwise_equal(serial, parallel)


def test_fresh_cpu_triangulation_recovers_after_partial_worker_failure():
    arguments = _arguments()

    def fail_second(job, shard):
        if shard == 1:
            return ("/bin/false",)
        return (
            sys.executable,
            "-m",
            "evidence.parallel_triangulation",
            "--job",
            str(job),
            "--shard-index",
            str(shard),
        )

    with pytest.raises(RuntimeError, match="worker failure"):
        robust_triangulate_associations_fresh_cpu(
            **arguments, worker_count=2, _command_builder=fail_second
        )
    serial = robust_triangulate_associations(**arguments)
    recovered = robust_triangulate_associations_fresh_cpu(
        **arguments, worker_count=2
    )
    _assert_bitwise_equal(serial, recovered)


def test_fresh_cpu_worker_resolves_defining_checkout_outside_repo(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    arguments = _arguments()
    serial = robust_triangulate_associations(**arguments)
    parallel = robust_triangulate_associations_fresh_cpu(
        **arguments, worker_count=2
    )
    _assert_bitwise_equal(serial, parallel)


def test_v4_bootstrap_uses_two_workers_only_above_threshold():
    args = SimpleNamespace(
        geometry_teacher_triangulation_cpu_workers=2,
        geometry_teacher_parallel_triangulation_min_tracks=5000,
    )
    small_backend, small_extra = _track_triangulation_backend(args, 4999)
    large_backend, large_extra = _track_triangulation_backend(args, 5000)
    assert small_backend is robust_triangulate_associations
    assert small_extra == {}
    assert large_backend is robust_triangulate_associations_fresh_cpu
    assert large_extra == {"worker_count": 2}


def test_frozen_p8_track_factor_keeps_serial_triangulation():
    source = inspect.getsource(build_p8_track_arm)
    assert "triangulation.robust_triangulate_associations(" in source
    assert "fresh_cpu" not in source
