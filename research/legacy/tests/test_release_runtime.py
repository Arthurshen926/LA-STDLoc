"""Release regressions, not benchmark experiments. No GPU or download needed."""

import sys
import subprocess
from unittest.mock import patch
import numpy as np
import pytest
import torch
from anygsloc.artifacts import export_map, sha256, validate
from anygsloc.config import FROZEN
from anygsloc.matching import global_cosine_top1, retain_high_score_matches
from anygsloc.pose import camera_intrinsics, solve_absolute_pose


@pytest.mark.parametrize("count", [0, 3, 256, 257, 1000])
def test_matching_equivalence(count):
    from localization.matcher import (
        global_cosine_top1 as old_match,
        retain_high_score_matches as old_retain,
    )

    torch.manual_seed(19)
    query, anchors = torch.randn(count, 256), torch.randn(35, 256)
    anchors[20] = anchors[0]
    if count:
        query[0] = anchors[0]
    a = global_cosine_top1(query, anchors, chunk_size=9)
    b = old_match(query, anchors, chunk_size=9)
    a = retain_high_score_matches(a, retention_fraction=0.65)
    b = old_retain(b, retention_fraction=0.65)
    for k in ("keypoint_indices", "anchor_indices", "scores"):
        assert torch.equal(getattr(a, k), getattr(b, k))


def test_pose_equivalence():
    from localization.pose_solver import solve_absolute_pose as old_pose

    rng = np.random.default_rng(19)
    xyz = rng.normal(size=(100, 3))
    xyz[:, 2] += 8
    K = camera_intrinsics(1.0, 0.8, 640, 480)
    uv = xyz @ K.T
    uv = uv[:, :2] / uv[:, 2:]
    uv[:15] += 20
    kwargs = dict(
        reprojection_error_px=2.0,
        confidence=0.999,
        min_iterations=250,
        max_iterations=30000,
        seed=2026,
    )
    a, b = solve_absolute_pose(uv, xyz, K, **kwargs), old_pose(uv, xyz, K, **kwargs)
    np.testing.assert_array_equal(a.pose_w2c, b.pose_w2c)
    np.testing.assert_array_equal(a.inliers, b.inliers)
    assert len(a.inliers) > 50


def research_fixture(tmp_path):
    torch.manual_seed(11)
    state = dict(
        schema="lafgs_materialized_anchor_map",
        anchor_ids=torch.arange(300),
        anchor_xyz=torch.randn(300, 3),
        anchor_features=torch.randn(300, 256),
        provenance=dict(uses_test_queries=False, uses_source_mapping_rgb=False),
    )
    source = tmp_path / "research.pt"
    torch.save(state, source)
    metric = dict(
        schema="lafgs_shared_metric_state",
        version=1,
        protocol="v6_identity_shared_metric",
        step=0,
        metric_config=dict(descriptor_dim=256, rank=1, max_residual_norm=0.0),
        landmark_indices=state["anchor_ids"],
        map_sha256=sha256(source),
        metric_state_dict={
            "down.weight": torch.zeros(1, 256),
            "down.bias": torch.zeros(1),
            "up.weight": torch.zeros(256, 1),
        },
    )
    metric_path = tmp_path / "metric.pt"
    torch.save(metric, metric_path)
    return source, metric_path, state


def test_export_exact_and_no_overwrite(tmp_path):
    source, metric, original = research_fixture(tmp_path)
    output = tmp_path / "deployment.pt"
    stats = export_map(source, metric, output)
    result = validate(torch.load(output, weights_only=True))
    for key in ("anchor_ids", "anchor_xyz", "anchor_features"):
        assert torch.equal(result[key], original[key])
    assert stats["tensor_bytes"] == 300 * (8 + 3 * 4 + 256 * 4)
    with pytest.raises(FileExistsError):
        export_map(source, metric, output)


@pytest.mark.parametrize(
    "corrupt", ["learned", "hash", "ids", "config", "test_queries"]
)
def test_export_fails_closed(tmp_path, corrupt):
    source, metric_path, state = research_fixture(tmp_path)
    metric = torch.load(metric_path, weights_only=True)
    if corrupt == "learned":
        metric["metric_state_dict"]["up.weight"][0] = 1
    if corrupt == "hash":
        metric["map_sha256"] = "wrong"
    if corrupt == "ids":
        metric["landmark_indices"] = torch.arange(300).flip(0)
    if corrupt == "config":
        metric["metric_config"]["rank"] = 2
    if corrupt == "test_queries":
        state["provenance"]["uses_test_queries"] = True
        torch.save(state, source)
        metric["map_sha256"] = sha256(source)
    torch.save(metric, metric_path)
    with pytest.raises(ValueError):
        export_map(source, metric_path, tmp_path / "out.pt")


class StubSuperPoint(torch.nn.Module):
    def detectAndCompute(self, image, **kwargs):
        g = torch.Generator().manual_seed(7)
        return [
            dict(
                keypoints=torch.tensor(
                    [[2.0, 2.0], [8.0, 8.0], [15.0, 15.0], [30.0, 30.0], [40.0, 40.0]]
                ),
                descriptors=torch.randn(5, 256, generator=g),
                keypoint_scores=torch.ones(5),
            )
        ]


@pytest.mark.parametrize("masked", [False, True])
def test_frontend_and_matches_equivalent(tmp_path, masked):
    from anygsloc.runtime import Localizer
    from localization.frontend import NativeSuperPointFrontend
    from map_learning.metric import SharedLowRankMetric

    source, metric_path, state = research_fixture(tmp_path)
    output = tmp_path / "deployment.pt"
    export_map(source, metric_path, output)
    mask = torch.ones(64, 64, dtype=torch.bool) if masked else None
    if masked:
        mask[0:5] = False
    image = torch.zeros(3, 64, 64)
    identity = SharedLowRankMetric(rank=1, max_residual_norm=0.0)
    with torch.no_grad():
        for p in identity.parameters():
            p.zero_()
    with (
        patch("anygsloc.runtime.SuperPoint", StubSuperPoint),
        patch("localization.frontend.SuperPoint", StubSuperPoint),
    ):
        engine = Localizer(
            output,
            keypoint_count=1024,
            reprojection_error_px=2.0,
            mask_policy="explicit_valid_mask" if masked else "none",
            device="cpu",
        )
        legacy = NativeSuperPointFrontend(
            device="cpu", keypoint_count=1024, metric=identity
        )
        old = legacy(image, valid_mask=mask)
        new = engine.localize(image, fov_x=1.0, fov_y=0.8, valid_mask=mask)
    assert torch.equal(old.keypoints, new["keypoints"])
    assert torch.equal(old.descriptors, new["descriptors"])
    normalized_map = torch.nn.functional.normalize(
        torch.nn.functional.normalize(state["anchor_features"], dim=1), dim=1
    )
    assert torch.equal(engine.anchor_features, normalized_map)
    expected = retain_high_score_matches(
        global_cosine_top1(
            old.descriptors, normalized_map, anchor_descriptors_normalized=True
        ),
        retention_fraction=0.65,
    )
    assert torch.equal(new["matches"].scores, expected.scores)
    assert new["resolved_config"]["pose_solves"] == 1


def test_runtime_import_isolation():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import anygsloc.runtime,sys; assert not any(k.split('.')[0] in "
            "{'map_learning','priors','topology','gsplat','evaluation','localization'} for k in sys.modules)",
        ],
        check=True,
    )


def test_invalid_scene_parameters():
    with pytest.raises(ValueError):
        FROZEN.resolved(
            keypoint_count=2049, reprojection_error_px=2, mask_policy="none"
        )
    with pytest.raises(ValueError):
        FROZEN.resolved(
            keypoint_count=1024, reprojection_error_px=float("nan"), mask_policy="none"
        )
