import math

import pytest
import torch
import torch.nn as nn
from plyfile import PlyData, PlyElement


def _write_minimal_gaussian_ply(path, vertex_count=3, loc_dim=0):
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{i}" for i in range(3)]
    names += [f"f_rest_{i}" for i in range(45)]
    names += ["opacity", "scale_0", "scale_1", "rot_0", "rot_1", "rot_2", "rot_3"]
    names += [f"loc_{i}" for i in range(loc_dim)]
    dtype = [(name, "f4") for name in names]
    data = torch.zeros(vertex_count, len(names), dtype=torch.float32)
    data[:, 0] = torch.arange(vertex_count, dtype=torch.float32)
    data[:, 2] = 2.0
    data[:, names.index("f_dc_0")] = 0.1
    data[:, names.index("f_dc_1")] = 0.2
    data[:, names.index("f_dc_2")] = 0.3
    data[:, names.index("opacity")] = 0.5
    data[:, names.index("rot_0")] = 1.0
    if loc_dim > 0:
        for i in range(loc_dim):
            data[:, names.index(f"loc_{i}")] = float(i + 1)
    elements = torch.zeros(vertex_count, dtype=torch.float32).numpy().astype(dtype)
    elements[:] = list(map(tuple, data.numpy()))
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GaussianModel loads CUDA tensors")
def test_gaussian_model_loads_rgb_only_ply_with_lafgs_feature_dim(tmp_path):
    from scene.gaussian_model import GaussianModel

    ply_path = tmp_path / "rgb_only.ply"
    _write_minimal_gaussian_ply(ply_path, vertex_count=3, loc_dim=0)

    model = GaussianModel(3)
    model.load_ply(str(ply_path), loc_feature_dim=4)

    assert model.get_xyz.shape[0] == 3
    assert model.get_loc_feature.shape == (3, 1, 4)
    flat = model.get_loc_feature.detach().reshape(3, 4)
    assert torch.isfinite(flat).all()
    assert torch.allclose(torch.linalg.norm(flat, dim=1), torch.ones(3, device=flat.device), atol=1e-5)


def test_build_multiview_init_from_projected_training_views():
    from localization_training.lafgs_reconstruction import build_multiview_initialization

    class DummyGaussians:
        def __init__(self):
            self._xyz = torch.tensor([[-0.5, -0.5, 2.0], [0.5, -0.5, 2.0]])

        @property
        def get_xyz(self):
            return self._xyz

    class DummyCamera:
        FoVx = 2.0 * math.atan(1.25)
        FoVy = FoVx
        image_width = 5
        image_height = 5
        world_view_transform = torch.eye(4)

    feature_map = torch.zeros(2, 5, 5)
    feature_map[:, 2, 2] = torch.tensor([1.0, 0.0])
    feature_map[:, 2, 3] = torch.tensor([0.0, 1.0])

    result = build_multiview_initialization(
        DummyGaussians(),
        [DummyCamera()],
        [feature_map],
        min_observations=1,
    )

    assert result.observation_count.tolist() == [1, 1]
    assert torch.allclose(result.features[0], torch.tensor([1.0, 0.0]), atol=1e-6)
    assert torch.allclose(result.features[1], torch.tensor([0.0, 1.0]), atol=1e-6)


def test_multiview_init_does_not_retain_gradient_graph_from_trainable_geometry():
    from localization_training.lafgs_reconstruction import build_multiview_initialization

    class DummyGaussians:
        def __init__(self):
            self._xyz = torch.tensor([[-0.5, -0.5, 2.0], [0.5, -0.5, 2.0]], requires_grad=True)

        @property
        def get_xyz(self):
            return self._xyz

    class DummyCamera:
        FoVx = 2.0 * math.atan(1.25)
        FoVy = FoVx
        image_width = 5
        image_height = 5
        world_view_transform = torch.eye(4)

    feature_map = torch.zeros(2, 5, 5)
    feature_map[:, 2, 2] = torch.tensor([1.0, 0.0])
    feature_map[:, 2, 3] = torch.tensor([0.0, 1.0])

    result = build_multiview_initialization(
        DummyGaussians(),
        [DummyCamera()],
        [feature_map],
        min_observations=1,
    )

    assert result.features.requires_grad is False
    assert result.reliability.requires_grad is False


def test_chunked_multiview_init_matches_full_aggregation():
    from localization_training.lafgs_reconstruction import (
        MultiViewInitConfig,
        build_multiview_initialization,
    )

    class DummyGaussians:
        def __init__(self):
            self._xyz = torch.tensor(
                [
                    [-0.5, -0.5, 2.0],
                    [0.5, -0.5, 2.0],
                    [-0.5, 0.5, 2.0],
                ]
            )

        @property
        def get_xyz(self):
            return self._xyz

    class DummyCamera:
        FoVx = 2.0 * math.atan(1.25)
        FoVy = FoVx
        image_width = 5
        image_height = 5
        world_view_transform = torch.eye(4)

    feature_map_a = torch.zeros(3, 5, 5)
    feature_map_b = torch.zeros(3, 5, 5)
    for feature_map in (feature_map_a, feature_map_b):
        feature_map[:, 2, 2] = torch.tensor([1.0, 0.0, 0.0])
        feature_map[:, 2, 3] = torch.tensor([0.0, 1.0, 0.0])
    feature_map_a[:, 3, 2] = torch.tensor([0.0, 0.0, 1.0])
    feature_map_b[:, 3, 2] = torch.tensor([1.0, 1.0, 0.0])

    full = build_multiview_initialization(
        DummyGaussians(),
        [DummyCamera(), DummyCamera()],
        [feature_map_a, feature_map_b],
        config=MultiViewInitConfig(min_observations=1, chunk_size=0),
    )
    chunked = build_multiview_initialization(
        DummyGaussians(),
        [DummyCamera(), DummyCamera()],
        [feature_map_a, feature_map_b],
        config=MultiViewInitConfig(min_observations=1, chunk_size=1),
    )

    assert torch.allclose(chunked.features, full.features, atol=1e-6)
    assert torch.allclose(chunked.reliability, full.reliability, atol=1e-6)
    assert torch.equal(chunked.observation_count, full.observation_count)
    assert torch.allclose(chunked.weight_sum, full.weight_sum, atol=1e-6)
    assert chunked.diagnostics["chunk_count"] == 3


def test_chunked_multiview_init_streams_callable_views_without_caching():
    from localization_training.lafgs_reconstruction import (
        MultiViewInitConfig,
        build_multiview_initialization,
    )

    class DummyGaussians:
        def __init__(self):
            self._xyz = torch.tensor([[-0.5, -0.5, 2.0], [0.5, -0.5, 2.0]])

        @property
        def get_xyz(self):
            return self._xyz

    class DummyCamera:
        FoVx = 2.0 * math.atan(1.25)
        FoVy = FoVx
        image_width = 5
        image_height = 5
        world_view_transform = torch.eye(4)

    feature_map = torch.zeros(2, 5, 5)
    feature_map[:, 2, 2] = torch.tensor([1.0, 0.0])
    feature_map[:, 2, 3] = torch.tensor([0.0, 1.0])
    calls = {"count": 0}

    def feature_source(_camera):
        calls["count"] += 1
        return feature_map.clone()

    result = build_multiview_initialization(
        DummyGaussians(),
        [DummyCamera(), DummyCamera()],
        feature_source,
        config=MultiViewInitConfig(min_observations=1, chunk_size=1),
    )

    assert calls["count"] == 4
    assert result.diagnostics["view_count"] == 2
    assert torch.equal(result.observation_count, torch.tensor([2, 2]))


def test_multiview_descriptor_aggregation_weights_visible_observations():
    from localization_training.lafgs_reconstruction import aggregate_multiview_descriptors

    observations = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [-1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ]
    )
    weights = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ]
    )
    valid = torch.tensor(
        [
            [True, True],
            [True, True],
            [False, False],
        ]
    )

    result = aggregate_multiview_descriptors(observations, weights, valid=valid)

    assert torch.allclose(result.features[0], torch.tensor([1.0, 0.0]), atol=1e-6)
    assert torch.allclose(result.features[1], torch.zeros(2), atol=1e-6)
    assert result.observation_count.tolist() == [2, 2]
    assert result.reliability[0] > result.reliability[1]


def test_apply_multiview_initialization_preserves_gaussian_feature_shape():
    from localization_training.lafgs_reconstruction import (
        MultiViewInitResult,
        apply_multiview_initialization,
    )

    class DummyGaussians:
        def __init__(self):
            self._loc_feature = nn.Parameter(torch.zeros(2, 1, 3))
            self.loc_prototype = torch.zeros(2, 3)
            self.loc_prototype_count = torch.zeros(2)
            self.loc_observation_count = torch.zeros(2, dtype=torch.long)
            self.loc_repeatability_ema = torch.zeros(2)

        @property
        def get_loc_feature(self):
            return self._loc_feature

    gaussians = DummyGaussians()
    result = MultiViewInitResult(
        features=torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        reliability=torch.tensor([0.8, 0.2]),
        observation_count=torch.tensor([3, 1]),
        weight_sum=torch.tensor([3.0, 1.0]),
    )

    apply_multiview_initialization(gaussians, result)

    expected = torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]])
    assert torch.allclose(gaussians.get_loc_feature.detach(), expected, atol=1e-6)
    assert torch.equal(gaussians.loc_observation_count, torch.tensor([3, 1]))
    assert torch.allclose(gaussians.loc_repeatability_ema, torch.tensor([0.8, 0.2]))


def test_apply_multiview_initialization_keeps_unobserved_features():
    from localization_training.lafgs_reconstruction import (
        MultiViewInitResult,
        apply_multiview_initialization,
    )

    class DummyGaussians:
        def __init__(self):
            self._loc_feature = nn.Parameter(
                torch.tensor([[[0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]])
            )
            self.loc_prototype = torch.zeros(2, 3)
            self.loc_prototype_count = torch.zeros(2)
            self.loc_observation_count = torch.zeros(2, dtype=torch.long)
            self.loc_repeatability_ema = torch.zeros(2)

        @property
        def get_loc_feature(self):
            return self._loc_feature

    gaussians = DummyGaussians()
    result = MultiViewInitResult(
        features=torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        reliability=torch.tensor([0.9, 0.0]),
        observation_count=torch.tensor([2, 0]),
        weight_sum=torch.tensor([2.0, 0.0]),
    )

    apply_multiview_initialization(gaussians, result)

    expected = torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]]])
    assert torch.allclose(gaussians.get_loc_feature.detach(), expected, atol=1e-6)
    assert torch.equal(gaussians.loc_observation_count, torch.tensor([2, 0]))


def test_soft_3d_to_2d_correspondence_selects_matching_pixel():
    from localization_training.lafgs_reconstruction import soft_3d_to_2d_correspondences

    feature_map = torch.zeros(2, 5, 5)
    feature_map[0, 3, 2] = 1.0
    feature_map[1, :, :] = 0.1
    feature_map[1, 3, 2] = 0.0
    gaussian_features = torch.tensor([[1.0, 0.0]])

    out = soft_3d_to_2d_correspondences(
        gaussian_features,
        feature_map,
        temperature=0.01,
    )

    assert torch.allclose(out.uv[0], torch.tensor([2.0, 3.0]), atol=1e-2)
    assert out.confidence[0] > 0.95


def test_soft_3d_to_2d_correspondence_reports_probability_margin():
    from localization_training.lafgs_reconstruction import soft_3d_to_2d_correspondences

    feature_map = torch.zeros(2, 1, 3)
    feature_map[:, 0, 0] = torch.tensor([1.0, 0.0])
    feature_map[:, 0, 1] = torch.tensor([0.9, 0.1])
    feature_map[:, 0, 2] = torch.tensor([0.0, 1.0])
    gaussian_features = torch.tensor([[1.0, 0.0]])

    out = soft_3d_to_2d_correspondences(
        gaussian_features,
        feature_map,
        temperature=0.2,
    )
    top2 = torch.topk(out.probabilities[0], k=2).values

    assert torch.allclose(out.peak_probability[0], top2[0], atol=1e-6)
    assert torch.allclose(out.margin[0], top2[0] - top2[1], atol=1e-6)


def test_soft_3d_to_2d_local_window_uses_compact_support():
    from localization_training.lafgs_reconstruction import soft_3d_to_2d_correspondences

    feature_map = torch.zeros(2, 9, 9)
    feature_map[0, 4, 4] = 1.0
    feature_map[0, 8, 8] = 4.0
    feature_map[1, :, :] = 0.01
    gaussian_features = torch.tensor([[1.0, 0.0]])

    out = soft_3d_to_2d_correspondences(
        gaussian_features,
        feature_map,
        temperature=0.05,
        projected_uv=torch.tensor([[4.0, 4.0]]),
        local_window_radius=1.5,
    )

    assert out.probabilities.shape[1] < feature_map.shape[1] * feature_map.shape[2]
    assert torch.allclose(out.uv[0], torch.tensor([4.0, 4.0]), atol=0.25)


def test_differentiable_pnp_loss_backpropagates_to_descriptors_and_points():
    from localization_training.lafgs_reconstruction import (
        DifferentiablePnPConfig,
        differentiable_pnp_pose_loss,
    )

    points = torch.tensor(
        [
            [-0.5, -0.5, 2.0],
            [0.5, -0.5, 2.0],
            [-0.5, 0.5, 2.0],
            [0.5, 0.5, 2.0],
        ],
        requires_grad=True,
    )
    descriptors = torch.eye(4, requires_grad=True)
    feature_map = torch.zeros(4, 8, 8)
    shifted_pixels = [(2, 2), (6, 2), (2, 6), (6, 6)]
    for channel, (x, y) in enumerate(shifted_pixels):
        feature_map[channel, y, x] = 1.0
    K = torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])
    pose_gt = torch.eye(4)

    out = differentiable_pnp_pose_loss(
        descriptors,
        feature_map,
        points,
        K,
        pose_gt,
        config=DifferentiablePnPConfig(
            temperature=0.01,
            min_correspondences=4,
            pnp_iterations=0,
            pose_weight=0.0,
            reprojection_weight=0.0,
            gt_reprojection_weight=1.0,
            allow_geometry_grad=True,
        ),
    )
    out.loss.backward()

    assert out.used_correspondences == 4
    assert descriptors.grad is not None
    assert descriptors.grad.abs().sum() > 0
    assert points.grad is not None
    assert points.grad.abs().sum() > 0


def test_differentiable_pnp_can_detach_pose_points_while_training_descriptors():
    from localization_training.lafgs_reconstruction import (
        DifferentiablePnPConfig,
        differentiable_pnp_pose_loss,
    )

    points = torch.tensor(
        [
            [-0.5, -0.5, 2.0],
            [0.5, -0.5, 2.0],
            [-0.5, 0.5, 2.0],
            [0.5, 0.5, 2.0],
        ],
        requires_grad=True,
    )
    descriptors = torch.eye(4, requires_grad=True)
    feature_map = torch.zeros(4, 8, 8)
    shifted_pixels = [(2, 2), (6, 2), (2, 6), (6, 6)]
    for channel, (x, y) in enumerate(shifted_pixels):
        feature_map[channel, y, x] = 1.0
    K = torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])
    pose_gt = torch.eye(4)

    out = differentiable_pnp_pose_loss(
        descriptors,
        feature_map,
        points,
        K,
        pose_gt,
        config=DifferentiablePnPConfig(
            temperature=0.01,
            min_correspondences=4,
            pnp_iterations=0,
            pose_weight=0.0,
            reprojection_weight=0.0,
            gt_reprojection_weight=1.0,
            allow_geometry_grad=True,
            detach_pnp_points=True,
        ),
    )
    out.loss.backward()

    assert out.used_correspondences == 4
    assert out.diagnostics["detach_pnp_points"] == 1.0
    assert descriptors.grad is not None
    assert descriptors.grad.abs().sum() > 0
    assert points.grad is None or points.grad.abs().sum() == pytest.approx(0.0, abs=1e-9)


def test_differentiable_pnp_geometry_reprojection_loss_gates_bad_matches():
    from localization_training.lafgs_reconstruction import (
        DifferentiablePnPConfig,
        differentiable_pnp_pose_loss,
    )

    points = torch.tensor(
        [
            [-0.5, -0.5, 2.0],
            [0.5, -0.5, 2.0],
            [-0.5, 0.5, 2.0],
            [0.5, 0.5, 2.0],
        ],
        requires_grad=True,
    )
    descriptors = torch.eye(4, requires_grad=True)
    feature_map = torch.zeros(4, 8, 8)
    matched_pixels = [(3, 2), (6, 3), (7, 7), (1, 1)]
    for channel, (x, y) in enumerate(matched_pixels):
        feature_map[channel, y, x] = 1.0
    K = torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])
    pose_gt = torch.eye(4)

    out = differentiable_pnp_pose_loss(
        descriptors,
        feature_map,
        points,
        K,
        pose_gt,
        config=DifferentiablePnPConfig(
            temperature=0.01,
            min_correspondences=4,
            pnp_iterations=0,
            pose_weight=0.0,
            reprojection_weight=0.0,
            gt_reprojection_weight=0.0,
            geometry_reprojection_weight=1.0,
            geometry_max_reprojection_error=1.5,
            allow_geometry_grad=True,
        ),
    )
    out.loss.backward()

    grad_norm = points.grad.norm(dim=-1)
    assert out.used_correspondences == 4
    assert out.geometry_reprojection_loss > 0.0
    assert out.diagnostics["geometry_correspondences"] == 2.0
    assert out.diagnostics["geometry_valid_candidate_count"] == 4.0
    assert out.diagnostics["geometry_filter_keep_ratio"] == pytest.approx(0.5)
    assert out.diagnostics["geometry_candidate_reprojection_error_max"] > 1.5
    assert out.diagnostics["geometry_kept_reprojection_error_max"] <= 1.5
    assert torch.all(grad_norm[:2] > 0.0)
    assert torch.allclose(grad_norm[2:], torch.zeros_like(grad_norm[2:]), atol=1e-9)
    assert descriptors.grad is None or descriptors.grad.abs().sum() == 0.0


def test_differentiable_pnp_geometry_reprojection_loss_can_gate_by_match_margin():
    from localization_training.lafgs_reconstruction import (
        DifferentiablePnPConfig,
        differentiable_pnp_pose_loss,
    )

    points = torch.tensor(
        [
            [-0.5, -0.5, 2.0],
            [0.5, -0.5, 2.0],
            [-0.5, 0.5, 2.0],
            [0.5, 0.5, 2.0],
        ],
        requires_grad=True,
    )
    descriptors = torch.eye(4, requires_grad=True)
    feature_map = torch.zeros(4, 8, 8)
    matched_pixels = [(2, 2), (6, 2), (2, 6), (7, 6)]
    for channel, (x, y) in enumerate(matched_pixels):
        feature_map[channel, y, x] = 1.0
    feature_map[:, 2, 2] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    feature_map[:, 2, 3] = torch.tensor([0.99, 0.01, 0.0, 0.0])
    K = torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])
    pose_gt = torch.eye(4)

    out = differentiable_pnp_pose_loss(
        descriptors,
        feature_map,
        points,
        K,
        pose_gt,
        projected_uv=torch.tensor([[2.0, 2.0], [6.0, 2.0], [2.0, 6.0], [6.0, 6.0]]),
        config=DifferentiablePnPConfig(
            temperature=0.2,
            min_correspondences=4,
            pnp_iterations=0,
            pose_weight=0.0,
            reprojection_weight=0.0,
            gt_reprojection_weight=0.0,
            geometry_reprojection_weight=1.0,
            geometry_max_reprojection_error=2.0,
            geometry_margin_threshold=0.4,
            allow_geometry_grad=True,
            local_window_radius=1.5,
        ),
    )
    out.loss.backward()

    grad_norm = points.grad.norm(dim=-1)
    assert out.used_correspondences == 4
    assert out.diagnostics["geometry_correspondences"] == 3.0
    assert out.diagnostics["geometry_margin_threshold"] == pytest.approx(0.4)
    assert grad_norm[0] == pytest.approx(0.0, abs=1e-9)
    assert torch.all(grad_norm[1:] > 0.0)


def test_differentiable_pnp_geometry_reprojection_loss_can_gate_by_peak_probability():
    from localization_training.lafgs_reconstruction import (
        DifferentiablePnPConfig,
        differentiable_pnp_pose_loss,
    )

    points = torch.tensor(
        [
            [-0.5, -0.5, 2.0],
            [0.5, -0.5, 2.0],
            [-0.5, 0.5, 2.0],
            [0.5, 0.5, 2.0],
        ],
        requires_grad=True,
    )
    descriptors = torch.eye(4, requires_grad=True)
    feature_map = torch.zeros(4, 8, 8)
    projected_uv = torch.tensor([[2.0, 2.0], [6.0, 2.0], [2.0, 6.0], [6.0, 6.0]])
    matched_uv = torch.tensor([[2.0, 2.0], [5.0, 2.0], [1.0, 6.0], [5.0, 6.0]])
    for channel, (x, y) in enumerate(matched_uv.tolist()):
        feature_map[channel, int(y), int(x)] = 1.0
    # First landmark has two equally plausible local peaks. It is
    # geometrically close, but not distinctive enough for geometry feedback.
    feature_map[:, 2, 3] = torch.tensor([1.0, 0.0, 0.0, 0.0])

    K = torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])
    pose_gt = torch.eye(4)

    out = differentiable_pnp_pose_loss(
        descriptors,
        feature_map,
        points,
        K,
        pose_gt,
        projected_uv=projected_uv,
        config=DifferentiablePnPConfig(
            temperature=0.05,
            min_correspondences=4,
            pnp_iterations=0,
            pose_weight=0.0,
            reprojection_weight=0.0,
            gt_reprojection_weight=0.0,
            geometry_reprojection_weight=1.0,
            geometry_max_reprojection_error=2.0,
            geometry_peak_probability_threshold=0.8,
            geometry_use_all_correspondences=True,
            geometry_local_window_radius=1.5,
            allow_geometry_grad=True,
        ),
    )
    out.loss.backward()

    grad_norm = points.grad.norm(dim=-1)
    assert out.used_correspondences == 4
    assert out.diagnostics["geometry_correspondences"] == 3.0
    assert out.diagnostics["geometry_peak_probability_threshold"] == pytest.approx(0.8)
    assert out.diagnostics["geometry_candidate_peak_probability_min"] < 0.8
    assert out.diagnostics["geometry_kept_peak_probability_min"] >= 0.8
    assert grad_norm[0] == pytest.approx(0.0, abs=1e-9)
    assert torch.all(grad_norm[1:] > 0.0)


def test_differentiable_pnp_condition_guard_blocks_feedback_and_geometry():
    from localization_training.lafgs_reconstruction import (
        DifferentiablePnPConfig,
        differentiable_pnp_pose_loss,
    )

    points = torch.tensor(
        [
            [-0.5, -0.5, 2.0],
            [0.5, -0.5, 2.0],
            [-0.5, 0.5, 2.0],
            [0.5, 0.5, 2.0],
        ],
        requires_grad=True,
    )
    descriptors = torch.eye(4, requires_grad=True)
    feature_map = torch.zeros(4, 8, 8)
    projected_uv = torch.tensor([[2.0, 2.0], [6.0, 2.0], [2.0, 6.0], [6.0, 6.0]])
    for channel, (x, y) in enumerate(projected_uv.tolist()):
        feature_map[channel, int(y), int(x)] = 1.0
    K = torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])
    pose_gt = torch.eye(4)
    pose_init = torch.eye(4)
    pose_init[0, 3] = 0.25

    out = differentiable_pnp_pose_loss(
        descriptors,
        feature_map,
        points,
        K,
        pose_gt,
        pose_init_w2c=pose_init,
        projected_uv=projected_uv,
        config=DifferentiablePnPConfig(
            temperature=0.05,
            min_correspondences=4,
            pnp_iterations=0,
            pose_weight=1.0,
            reprojection_weight=1.0,
            gt_reprojection_weight=1.0,
            entropy_weight=1.0,
            geometry_reprojection_weight=1.0,
            geometry_max_reprojection_error=2.0,
            geometry_use_all_correspondences=True,
            geometry_local_window_radius=1.5,
            feedback_pose_guard_keep_gt_reprojection=True,
            max_condition_number=100.0,
            allow_geometry_grad=True,
        ),
    )
    out.loss.backward()

    assert out.diagnostics["condition_guard_enabled"] == 1.0
    assert out.diagnostics["condition_guard_passed"] == 0.0
    assert out.diagnostics["condition_guard_scale"] == 0.0
    assert out.diagnostics["condition_guard_max_condition_number"] == pytest.approx(100.0)
    assert out.diagnostics["feedback_gt_reprojection_scale"] == 0.0
    assert out.diagnostics["geometry_correspondences"] == 0.0
    assert descriptors.grad is None or descriptors.grad.abs().sum() == pytest.approx(0.0, abs=1e-9)
    assert points.grad is None or points.grad.abs().sum() == pytest.approx(0.0, abs=1e-9)


def test_differentiable_pnp_geometry_reprojection_loss_can_guard_pose_regressions():
    from localization_training.lafgs_reconstruction import (
        DifferentiablePnPConfig,
        differentiable_pnp_pose_loss,
    )

    points = torch.tensor(
        [
            [-0.6, -0.4, 2.0],
            [0.4, -0.5, 2.2],
            [-0.5, 0.6, 2.4],
            [0.5, 0.4, 1.8],
            [0.0, -0.2, 1.6],
            [0.2, 0.7, 2.7],
        ],
        requires_grad=True,
    )
    descriptors = torch.eye(6, requires_grad=True)
    feature_map = torch.zeros(6, 8, 8)
    mismatched_pixels = [(6, 6), (2, 6), (6, 2), (2, 2), (7, 4), (4, 7)]
    for channel, (x, y) in enumerate(mismatched_pixels):
        feature_map[channel, y, x] = 1.0
    K = torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])
    pose_gt = torch.eye(4)

    out = differentiable_pnp_pose_loss(
        descriptors,
        feature_map,
        points,
        K,
        pose_gt,
        pose_init_w2c=pose_gt,
        config=DifferentiablePnPConfig(
            temperature=0.01,
            min_correspondences=6,
            pnp_iterations=1,
            pose_weight=0.0,
            reprojection_weight=0.0,
            gt_reprojection_weight=0.0,
            geometry_reprojection_weight=1.0,
            geometry_max_reprojection_error=10.0,
            geometry_pose_guard_max_loss_increase=0.0,
            allow_geometry_grad=True,
        ),
    )
    out.loss.backward()

    assert out.used_correspondences == 6
    assert out.diagnostics["geometry_pose_guard_enabled"] == 1.0
    assert out.diagnostics["geometry_pose_guard_passed"] == 0.0
    assert out.diagnostics["geometry_correspondences"] == 0.0
    assert points.grad is None or points.grad.abs().sum() == pytest.approx(0.0, abs=1e-9)


def test_differentiable_pnp_geometry_reprojection_loss_can_guard_absolute_pose_loss():
    from localization_training.lafgs_reconstruction import (
        DifferentiablePnPConfig,
        differentiable_pnp_pose_loss,
    )

    points = torch.tensor(
        [
            [-0.6, -0.4, 2.0],
            [0.4, -0.5, 2.2],
            [-0.5, 0.6, 2.4],
            [0.5, 0.4, 1.8],
            [0.0, -0.2, 1.6],
            [0.2, 0.7, 2.7],
        ],
        requires_grad=True,
    )
    descriptors = torch.eye(6, requires_grad=True)
    feature_map = torch.zeros(6, 8, 8)
    mismatched_pixels = [(6, 6), (2, 6), (6, 2), (2, 2), (7, 4), (4, 7)]
    for channel, (x, y) in enumerate(mismatched_pixels):
        feature_map[channel, y, x] = 1.0
    K = torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])
    pose_gt = torch.eye(4)

    out = differentiable_pnp_pose_loss(
        descriptors,
        feature_map,
        points,
        K,
        pose_gt,
        pose_init_w2c=pose_gt,
        config=DifferentiablePnPConfig(
            temperature=0.01,
            min_correspondences=6,
            pnp_iterations=1,
            pose_weight=0.0,
            reprojection_weight=0.0,
            gt_reprojection_weight=0.0,
            geometry_reprojection_weight=1.0,
            geometry_max_reprojection_error=10.0,
            geometry_pose_guard_max_loss_increase=30.0,
            geometry_pose_guard_max_loss=0.1,
            allow_geometry_grad=True,
        ),
    )
    out.loss.backward()

    assert out.used_correspondences == 6
    assert out.diagnostics["geometry_pose_guard_enabled"] == 1.0
    assert out.diagnostics["geometry_pose_guard_passed"] == 0.0
    assert out.diagnostics["geometry_pose_guard_max_loss"] == pytest.approx(0.1)
    assert out.diagnostics["pose_loss"] > 0.1
    assert out.diagnostics["geometry_correspondences"] == 0.0
    assert points.grad is None or points.grad.abs().sum() == pytest.approx(0.0, abs=1e-9)


def test_differentiable_pnp_feedback_pose_guard_blocks_descriptor_updates():
    from localization_training.lafgs_reconstruction import (
        DifferentiablePnPConfig,
        differentiable_pnp_pose_loss,
    )

    points = torch.tensor(
        [
            [-0.6, -0.4, 2.0],
            [0.4, -0.5, 2.2],
            [-0.5, 0.6, 2.4],
            [0.5, 0.4, 1.8],
            [0.0, -0.2, 1.6],
            [0.2, 0.7, 2.7],
        ]
    )
    descriptors = torch.eye(6, requires_grad=True)
    feature_map = torch.zeros(6, 8, 8)
    mismatched_pixels = [(6, 6), (2, 6), (6, 2), (2, 2), (7, 4), (4, 7)]
    for channel, (x, y) in enumerate(mismatched_pixels):
        feature_map[channel, y, x] = 1.0
    K = torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])
    pose_gt = torch.eye(4)

    out = differentiable_pnp_pose_loss(
        descriptors,
        feature_map,
        points,
        K,
        pose_gt,
        pose_init_w2c=pose_gt,
        config=DifferentiablePnPConfig(
            temperature=0.01,
            min_correspondences=6,
            pnp_iterations=1,
            pose_weight=1.0,
            reprojection_weight=0.1,
            gt_reprojection_weight=1.0,
            feedback_pose_guard_max_loss_increase=0.0,
        ),
    )
    out.loss.backward()

    assert out.used_correspondences == 6
    assert out.diagnostics["feedback_pose_guard_enabled"] == 1.0
    assert out.diagnostics["feedback_pose_guard_passed"] == 0.0
    assert out.diagnostics["pose_loss_delta"] > 0.5
    assert out.loss.detach().item() == pytest.approx(0.0, abs=1e-9)
    assert descriptors.grad is None or descriptors.grad.abs().sum() == pytest.approx(0.0, abs=1e-9)


def test_differentiable_pnp_feedback_pose_guard_can_cap_absolute_pose_loss():
    from localization_training.lafgs_reconstruction import (
        DifferentiablePnPConfig,
        differentiable_pnp_pose_loss,
    )

    points = torch.tensor(
        [
            [-0.6, -0.4, 2.0],
            [0.4, -0.5, 2.2],
            [-0.5, 0.6, 2.4],
            [0.5, 0.4, 1.8],
            [0.0, -0.2, 1.6],
            [0.2, 0.7, 2.7],
        ]
    )
    descriptors = torch.eye(6, requires_grad=True)
    feature_map = torch.zeros(6, 8, 8)
    mismatched_pixels = [(6, 6), (2, 6), (6, 2), (2, 2), (7, 4), (4, 7)]
    for channel, (x, y) in enumerate(mismatched_pixels):
        feature_map[channel, y, x] = 1.0
    K = torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])
    pose_gt = torch.eye(4)

    out = differentiable_pnp_pose_loss(
        descriptors,
        feature_map,
        points,
        K,
        pose_gt,
        pose_init_w2c=pose_gt,
        config=DifferentiablePnPConfig(
            temperature=0.01,
            min_correspondences=6,
            pnp_iterations=1,
            pose_weight=1.0,
            reprojection_weight=0.1,
            gt_reprojection_weight=1.0,
            feedback_pose_guard_max_loss_increase=30.0,
            feedback_pose_guard_max_loss=0.1,
        ),
    )
    out.loss.backward()

    assert out.used_correspondences == 6
    assert out.diagnostics["feedback_pose_guard_enabled"] == 1.0
    assert out.diagnostics["feedback_pose_guard_passed"] == 0.0
    assert out.diagnostics["feedback_pose_guard_max_loss"] == pytest.approx(0.1)
    assert out.diagnostics["pose_loss"] > 0.1
    assert out.loss.detach().item() == pytest.approx(0.0, abs=1e-9)
    assert descriptors.grad is None or descriptors.grad.abs().sum() == pytest.approx(0.0, abs=1e-9)


def test_differentiable_pnp_feedback_pose_guard_can_keep_gt_reprojection_bootstrap():
    from localization_training.lafgs_reconstruction import (
        DifferentiablePnPConfig,
        differentiable_pnp_pose_loss,
    )

    points = torch.tensor(
        [
            [-0.6, -0.4, 2.0],
            [0.4, -0.5, 2.2],
            [-0.5, 0.6, 2.4],
            [0.5, 0.4, 1.8],
            [0.0, -0.2, 1.6],
            [0.2, 0.7, 2.7],
        ]
    )
    descriptors = torch.eye(6, requires_grad=True)
    feature_map = torch.zeros(6, 8, 8)
    mismatched_pixels = [(6, 6), (2, 6), (6, 2), (2, 2), (7, 4), (4, 7)]
    for channel, (x, y) in enumerate(mismatched_pixels):
        feature_map[channel, y, x] = 1.0
    K = torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])
    pose_gt = torch.eye(4)

    out = differentiable_pnp_pose_loss(
        descriptors,
        feature_map,
        points,
        K,
        pose_gt,
        pose_init_w2c=pose_gt,
        config=DifferentiablePnPConfig(
            temperature=0.01,
            min_correspondences=6,
            pnp_iterations=1,
            pose_weight=1.0,
            reprojection_weight=0.1,
            gt_reprojection_weight=1.0,
            feedback_pose_guard_max_loss_increase=0.0,
            feedback_pose_guard_keep_gt_reprojection=True,
        ),
    )
    out.loss.backward()

    assert out.used_correspondences == 6
    assert out.diagnostics["feedback_pose_guard_passed"] == 0.0
    assert out.diagnostics["feedback_pose_guard_keep_gt_reprojection"] == 1.0
    assert out.loss.detach().item() == pytest.approx(out.gt_reprojection_loss.detach().item())
    assert descriptors.grad is not None
    assert descriptors.grad.abs().sum() > 0.0


def test_differentiable_pnp_feedback_pose_guard_can_softly_downweight_descriptor_updates():
    from localization_training.lafgs_reconstruction import (
        DifferentiablePnPConfig,
        differentiable_pnp_pose_loss,
    )

    points = torch.tensor(
        [
            [-0.6, -0.4, 2.0],
            [0.4, -0.5, 2.2],
            [-0.5, 0.6, 2.4],
            [0.5, 0.4, 1.8],
            [0.0, -0.2, 1.6],
            [0.2, 0.7, 2.7],
        ]
    )
    descriptors = torch.eye(6, requires_grad=True)
    feature_map = torch.zeros(6, 8, 8)
    mismatched_pixels = [(6, 6), (2, 6), (6, 2), (2, 2), (7, 4), (4, 7)]
    for channel, (x, y) in enumerate(mismatched_pixels):
        feature_map[channel, y, x] = 1.0
    K = torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])
    pose_gt = torch.eye(4)

    out = differentiable_pnp_pose_loss(
        descriptors,
        feature_map,
        points,
        K,
        pose_gt,
        pose_init_w2c=pose_gt,
        config=DifferentiablePnPConfig(
            temperature=0.01,
            min_correspondences=6,
            pnp_iterations=1,
            pose_weight=1.0,
            reprojection_weight=0.1,
            gt_reprojection_weight=1.0,
            feedback_pose_guard_max_loss_increase=0.0,
            feedback_pose_guard_softness=10.0,
        ),
    )
    out.loss.backward()

    assert out.used_correspondences == 6
    assert out.diagnostics["feedback_pose_guard_enabled"] == 1.0
    assert out.diagnostics["feedback_pose_guard_passed"] == 0.0
    assert 0.0 < out.diagnostics["feedback_pose_guard_scale"] < 1.0
    assert out.diagnostics["feedback_pose_guard_softness"] == pytest.approx(10.0)
    assert out.loss.detach().item() > 0.0
    assert descriptors.grad is not None
    assert descriptors.grad.abs().sum() > 0.0


def test_differentiable_pnp_geometry_reprojection_ignores_feedback_guard_when_geometry_guard_allows():
    from localization_training.lafgs_reconstruction import (
        DifferentiablePnPConfig,
        differentiable_pnp_pose_loss,
    )

    points = torch.tensor(
        [
            [-0.5, -0.5, 2.0],
            [0.5, -0.5, 2.0],
            [-0.5, 0.5, 2.0],
            [0.5, 0.5, 2.0],
        ],
        requires_grad=True,
    )
    descriptors = torch.eye(4, requires_grad=True)
    feature_map = torch.zeros(4, 8, 8)
    near_pixels = [(3, 2), (6, 3), (2, 6), (7, 6)]
    for channel, (x, y) in enumerate(near_pixels):
        feature_map[channel, y, x] = 1.0
    K = torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])
    pose_gt = torch.eye(4)
    pose_init = torch.eye(4)
    pose_init[0, 3] = 1.0

    out = differentiable_pnp_pose_loss(
        descriptors,
        feature_map,
        points,
        K,
        pose_gt,
        pose_init_w2c=pose_init,
        config=DifferentiablePnPConfig(
            temperature=0.01,
            min_correspondences=4,
            pnp_iterations=0,
            pose_weight=0.0,
            reprojection_weight=0.0,
            gt_reprojection_weight=0.0,
            geometry_reprojection_weight=1.0,
            geometry_max_reprojection_error=1.5,
            feedback_pose_guard_max_loss=0.1,
            allow_geometry_grad=True,
        ),
    )
    out.loss.backward()

    assert out.used_correspondences == 4
    assert out.diagnostics["feedback_pose_guard_enabled"] == 1.0
    assert out.diagnostics["feedback_pose_guard_passed"] == 0.0
    assert out.diagnostics["geometry_pose_guard_enabled"] == 0.0
    assert out.diagnostics["geometry_correspondences"] == 4.0
    assert points.grad is not None
    assert points.grad.abs().sum() > 0.0


def test_differentiable_pnp_geometry_feedback_can_use_all_local_candidates_beyond_pnp_cap():
    from localization_training.lafgs_reconstruction import (
        DifferentiablePnPConfig,
        differentiable_pnp_pose_loss,
    )

    points = torch.tensor(
        [
            [-0.75, -0.75, 2.0],
            [-0.25, -0.75, 2.0],
            [0.25, -0.75, 2.0],
            [0.75, -0.75, 2.0],
            [-0.75, 0.75, 2.0],
            [-0.25, 0.75, 2.0],
            [0.25, 0.75, 2.0],
            [0.75, 0.75, 2.0],
        ],
        requires_grad=True,
    )
    descriptors = torch.eye(8, requires_grad=True)
    feature_map = torch.zeros(8, 16, 16)
    projected_uv = torch.tensor(
        [
            [2.0, 2.0],
            [6.0, 2.0],
            [10.0, 2.0],
            [14.0, 2.0],
            [2.0, 14.0],
            [6.0, 14.0],
            [10.0, 14.0],
            [14.0, 14.0],
        ]
    )

    # The first four descriptors have very confident but geometrically wrong
    # global matches, so the PnP cap selects them. The last four have lower
    # global confidence but correct local matches around the GT projection.
    far_pixels = [(14, 14), (10, 14), (6, 14), (2, 14)]
    for channel, (x, y) in enumerate(far_pixels):
        feature_map[channel, y, x] = 1.0
    for channel in range(4, 8):
        x, y = projected_uv[channel].tolist()
        feature_map[channel, int(y), int(x)] = 1.0
        feature_map[channel, int(y), max(int(x) - 1, 0)] = 0.85

    K = torch.tensor([[16.0, 0.0, 8.0], [0.0, 16.0, 8.0], [0.0, 0.0, 1.0]])
    pose_gt = torch.eye(4)

    out = differentiable_pnp_pose_loss(
        descriptors,
        feature_map,
        points,
        K,
        pose_gt,
        projected_uv=projected_uv,
        config=DifferentiablePnPConfig(
            temperature=0.05,
            min_correspondences=4,
            pnp_iterations=0,
            pose_weight=0.0,
            reprojection_weight=0.0,
            gt_reprojection_weight=0.0,
            max_correspondences=4,
            geometry_reprojection_weight=1.0,
            geometry_max_reprojection_error=1.5,
            geometry_confidence_threshold=0.1,
            geometry_use_all_correspondences=True,
            geometry_local_window_radius=1.5,
            allow_geometry_grad=True,
        ),
    )
    out.loss.backward()

    grad_norm = points.grad.norm(dim=-1)
    assert out.used_correspondences == 4
    assert out.diagnostics["geometry_candidate_count"] == 8.0
    assert out.diagnostics["geometry_correspondences"] == 4.0
    assert out.diagnostics["geometry_use_all_correspondences"] == 1.0
    assert out.diagnostics["geometry_local_window_radius"] == pytest.approx(1.5)
    assert torch.allclose(grad_norm[:4], torch.zeros_like(grad_norm[:4]), atol=1e-9)
    assert torch.all(grad_norm[4:] > 0.0)


def test_differentiable_pnp_geometry_guard_can_soft_scale_instead_of_zeroing():
    from localization_training.lafgs_reconstruction import (
        DifferentiablePnPConfig,
        differentiable_pnp_pose_loss,
    )

    points = torch.tensor(
        [
            [-0.5, -0.5, 2.0],
            [0.5, -0.5, 2.0],
            [-0.5, 0.5, 2.0],
            [0.5, 0.5, 2.0],
        ],
        requires_grad=True,
    )
    descriptors = torch.eye(4)
    feature_map = torch.zeros(4, 8, 8)
    projected_uv = torch.tensor([[2.0, 2.0], [6.0, 2.0], [2.0, 6.0], [6.0, 6.0]])
    for channel, (x, y) in enumerate(projected_uv.tolist()):
        feature_map[channel, int(y), int(x)] = 1.0
        feature_map[channel, int(y), max(int(x) - 1, 0)] = 0.85

    K = torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])
    pose_gt = torch.eye(4)
    pose_init = torch.eye(4)
    pose_init[0, 3] = 100.0

    out = differentiable_pnp_pose_loss(
        descriptors,
        feature_map,
        points,
        K,
        pose_gt,
        pose_init_w2c=pose_init,
        projected_uv=projected_uv,
        config=DifferentiablePnPConfig(
            temperature=0.05,
            min_correspondences=4,
            pnp_iterations=0,
            pose_weight=0.0,
            reprojection_weight=0.0,
            gt_reprojection_weight=0.0,
            geometry_reprojection_weight=1.0,
            geometry_max_reprojection_error=1.5,
            geometry_confidence_threshold=0.1,
            geometry_use_all_correspondences=True,
            geometry_local_window_radius=1.5,
            geometry_pose_guard_max_loss=0.01,
            geometry_pose_guard_softness=10.0,
            geometry_pose_guard_min_scale=0.25,
            allow_geometry_grad=True,
        ),
    )
    out.loss.backward()

    assert out.diagnostics["geometry_pose_guard_passed"] == 0.0
    assert out.diagnostics["geometry_pose_guard_scale"] == pytest.approx(0.25)
    assert out.diagnostics["geometry_correspondences"] == 4.0
    assert points.grad.norm() > 0.0


def test_differentiable_pnp_geometry_match_loss_updates_descriptors_without_moving_xyz():
    from localization_training.lafgs_reconstruction import (
        DifferentiablePnPConfig,
        differentiable_pnp_pose_loss,
    )

    points = torch.tensor(
        [
            [-0.5, -0.5, 2.0],
            [0.5, -0.5, 2.0],
            [-0.5, 0.5, 2.0],
            [0.5, 0.5, 2.0],
        ],
        requires_grad=True,
    )
    descriptors = torch.eye(4, requires_grad=True)
    feature_map = torch.zeros(4, 8, 8)
    projected_uv = torch.tensor([[2.0, 2.0], [6.0, 2.0], [2.0, 6.0], [6.0, 6.0]])
    shifted_uv = torch.tensor([[3.0, 2.0], [5.0, 2.0], [2.0, 5.0], [6.0, 5.0]])
    for channel, (x, y) in enumerate(shifted_uv.tolist()):
        feature_map[channel, int(y), int(x)] = 1.0
        gt_x, gt_y = projected_uv[channel].tolist()
        feature_map[channel, int(gt_y), int(gt_x)] = 0.85
        feature_map[(channel + 1) % 4, int(gt_y), int(gt_x)] = 0.45

    K = torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])
    pose_gt = torch.eye(4)

    out = differentiable_pnp_pose_loss(
        descriptors,
        feature_map,
        points,
        K,
        pose_gt,
        projected_uv=projected_uv,
        config=DifferentiablePnPConfig(
            temperature=0.1,
            min_correspondences=4,
            pnp_iterations=0,
            pose_weight=0.0,
            reprojection_weight=0.0,
            gt_reprojection_weight=0.0,
            geometry_reprojection_weight=0.0,
            geometry_match_reprojection_weight=1.0,
            geometry_max_reprojection_error=2.0,
            geometry_confidence_threshold=0.1,
            geometry_local_window_radius=1.5,
            allow_geometry_grad=True,
        ),
    )
    out.loss.backward()

    assert out.used_correspondences == 4
    assert out.geometry_match_reprojection_loss > 0.0
    assert out.diagnostics["geometry_match_correspondences"] == 4.0
    assert descriptors.grad is not None
    assert descriptors.grad.abs().sum() > 0.0
    assert points.grad is None or points.grad.abs().sum() == pytest.approx(0.0, abs=1e-9)


def test_differentiable_pnp_geometry_match_loss_obeys_geometry_pose_guard():
    from localization_training.lafgs_reconstruction import (
        DifferentiablePnPConfig,
        differentiable_pnp_pose_loss,
    )

    points = torch.tensor(
        [
            [-0.6, -0.4, 2.0],
            [0.4, -0.5, 2.2],
            [-0.5, 0.6, 2.4],
            [0.5, 0.4, 1.8],
            [0.0, -0.2, 1.6],
            [0.2, 0.7, 2.7],
        ]
    )
    descriptors = torch.eye(6, requires_grad=True)
    feature_map = torch.zeros(6, 8, 8)
    mismatched_pixels = [(6, 6), (2, 6), (6, 2), (2, 2), (7, 4), (4, 7)]
    for channel, (x, y) in enumerate(mismatched_pixels):
        feature_map[channel, y, x] = 1.0
    K = torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])
    pose_gt = torch.eye(4)

    out = differentiable_pnp_pose_loss(
        descriptors,
        feature_map,
        points,
        K,
        pose_gt,
        pose_init_w2c=pose_gt,
        config=DifferentiablePnPConfig(
            temperature=0.01,
            min_correspondences=6,
            pnp_iterations=1,
            pose_weight=0.0,
            reprojection_weight=0.0,
            gt_reprojection_weight=0.0,
            geometry_match_reprojection_weight=1.0,
            geometry_max_reprojection_error=20.0,
            geometry_pose_guard_max_loss_increase=0.0,
            allow_geometry_grad=True,
        ),
    )
    out.loss.backward()

    assert out.used_correspondences == 6
    assert out.diagnostics["geometry_pose_guard_enabled"] == 1.0
    assert out.diagnostics["geometry_pose_guard_passed"] == 0.0
    assert out.diagnostics["geometry_match_correspondences"] == 0.0
    assert out.geometry_match_reprojection_loss.detach().item() == pytest.approx(0.0, abs=1e-9)
    assert descriptors.grad is None or descriptors.grad.abs().sum() == pytest.approx(0.0, abs=1e-9)


def test_differentiable_pnp_depth_anchor_geometry_loss_updates_xyz_without_descriptor_grad():
    from localization_training.lafgs_reconstruction import (
        DifferentiablePnPConfig,
        differentiable_pnp_pose_loss,
    )

    source_points = torch.tensor(
        [
            [-0.5, -0.5, 2.0],
            [0.5, -0.5, 2.0],
            [-0.5, 0.5, 2.0],
            [0.5, 0.5, 2.0],
        ]
    )
    points = source_points.clone().requires_grad_(True)
    descriptors = torch.eye(4, requires_grad=True)
    feature_map = torch.zeros(4, 8, 8)
    projected_uv = torch.tensor([[2.0, 2.0], [6.0, 2.0], [2.0, 6.0], [6.0, 6.0]])
    shifted_uv = torch.tensor([[3.0, 2.0], [5.0, 2.0], [2.0, 5.0], [6.0, 5.0]])
    for channel, (x, y) in enumerate(shifted_uv.tolist()):
        feature_map[channel, int(y), int(x)] = 1.0
        gt_x, gt_y = projected_uv[channel].tolist()
        feature_map[channel, int(gt_y), int(gt_x)] = 0.85

    K = torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])
    pose_gt = torch.eye(4)

    out = differentiable_pnp_pose_loss(
        descriptors,
        feature_map,
        points,
        K,
        pose_gt,
        projected_uv=projected_uv,
        geometry_anchor_points_world=source_points,
        config=DifferentiablePnPConfig(
            temperature=0.1,
            min_correspondences=4,
            pnp_iterations=0,
            pose_weight=0.0,
            reprojection_weight=0.0,
            gt_reprojection_weight=0.0,
            geometry_reprojection_weight=0.0,
            geometry_depth_anchor_weight=1.0,
            geometry_max_reprojection_error=2.0,
            geometry_confidence_threshold=0.1,
            geometry_local_window_radius=1.5,
            allow_geometry_grad=True,
        ),
    )
    out.loss.backward()

    assert out.geometry_depth_anchor_loss > 0.0
    assert out.diagnostics["geometry_depth_anchor_correspondences"] == 4.0
    assert points.grad is not None
    assert points.grad.abs().sum() > 0.0
    assert descriptors.grad is None or descriptors.grad.abs().sum() == pytest.approx(0.0, abs=1e-9)


def test_differentiable_pnp_geometry_match_loss_has_separate_peak_gate():
    from localization_training.lafgs_reconstruction import (
        DifferentiablePnPConfig,
        differentiable_pnp_pose_loss,
    )

    points = torch.tensor(
        [
            [-0.5, -0.5, 2.0],
            [0.5, -0.5, 2.0],
            [-0.5, 0.5, 2.0],
            [0.5, 0.5, 2.0],
        ],
        requires_grad=True,
    )
    descriptors = torch.eye(4, requires_grad=True)
    feature_map = torch.zeros(4, 8, 8)
    projected_uv = torch.tensor([[2.0, 2.0], [6.0, 2.0], [2.0, 6.0], [6.0, 6.0]])
    shifted_uv = torch.tensor([[3.0, 2.0], [5.0, 2.0], [2.0, 5.0], [6.0, 5.0]])
    for channel, (x, y) in enumerate(shifted_uv.tolist()):
        feature_map[channel, int(y), int(x)] = 1.0
        gt_x, gt_y = projected_uv[channel].tolist()
        feature_map[channel, int(gt_y), int(gt_x)] = 0.85
        feature_map[(channel + 1) % 4, int(gt_y), int(gt_x)] = 0.45
    # First landmark has two equally plausible local peaks, so it should be
    # excluded by the match-only peak gate while the other landmarks remain.
    feature_map[:, 2, 2] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    feature_map[:, 2, 3] = torch.tensor([1.0, 0.0, 0.0, 0.0])

    K = torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])
    pose_gt = torch.eye(4)

    out = differentiable_pnp_pose_loss(
        descriptors,
        feature_map,
        points,
        K,
        pose_gt,
        projected_uv=projected_uv,
        config=DifferentiablePnPConfig(
            temperature=0.05,
            min_correspondences=4,
            pnp_iterations=0,
            pose_weight=0.0,
            reprojection_weight=0.0,
            gt_reprojection_weight=0.0,
            geometry_reprojection_weight=0.0,
            geometry_match_reprojection_weight=1.0,
            geometry_peak_probability_threshold=0.0,
            geometry_match_peak_probability_threshold=0.8,
            geometry_max_reprojection_error=2.0,
            geometry_local_window_radius=1.5,
            allow_geometry_grad=True,
        ),
    )
    out.loss.backward()

    grad_norm = descriptors.grad.norm(dim=-1)
    assert out.used_correspondences == 4
    assert out.diagnostics["geometry_match_peak_probability_threshold"] == pytest.approx(0.8)
    assert out.diagnostics["geometry_peak_probability_threshold"] == pytest.approx(0.0)
    assert out.diagnostics["geometry_match_correspondences"] == 3.0
    assert grad_norm[0] == pytest.approx(0.0, abs=1e-9)
    assert torch.all(grad_norm[1:] > 0.0)
    assert points.grad is None or points.grad.abs().sum() == pytest.approx(0.0, abs=1e-9)


def test_differentiable_pnp_loss_backpropagates_to_landmark_weights():
    from localization_training.lafgs_reconstruction import (
        DifferentiablePnPConfig,
        differentiable_pnp_pose_loss,
    )

    points = torch.tensor(
        [
            [-0.5, -0.5, 2.0],
            [0.5, -0.5, 2.0],
            [-0.5, 0.5, 2.0],
            [0.5, 0.5, 2.0],
        ]
    )
    descriptors = torch.eye(4, requires_grad=True)
    point_weights = torch.tensor([0.2, 0.4, 0.6, 0.8], requires_grad=True)
    feature_map = torch.zeros(4, 8, 8)
    shifted_pixels = [(1, 2), (6, 2), (2, 6), (7, 5)]
    for channel, (x, y) in enumerate(shifted_pixels):
        feature_map[channel, y, x] = 1.0
    K = torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])
    pose_gt = torch.eye(4)

    out = differentiable_pnp_pose_loss(
        descriptors,
        feature_map,
        points,
        K,
        pose_gt,
        point_weights=point_weights,
        config=DifferentiablePnPConfig(
            temperature=0.01,
            min_correspondences=4,
            pnp_iterations=0,
            pose_weight=0.0,
            reprojection_weight=0.0,
            gt_reprojection_weight=1.0,
        ),
    )
    out.loss.backward()

    assert out.used_correspondences == 4
    assert point_weights.grad is not None
    assert point_weights.grad.abs().sum() > 0


def test_differentiable_pnp_landmark_weights_can_use_floor_without_cutting_gradient():
    from localization_training.lafgs_reconstruction import (
        DifferentiablePnPConfig,
        differentiable_pnp_pose_loss,
    )

    points = torch.tensor(
        [
            [-0.5, -0.5, 2.0],
            [0.5, -0.5, 2.0],
            [-0.5, 0.5, 2.0],
            [0.5, 0.5, 2.0],
        ]
    )
    descriptors = torch.eye(4, requires_grad=True)
    point_weights = torch.tensor([0.0, 0.1, 0.2, 0.3], requires_grad=True)
    feature_map = torch.zeros(4, 8, 8)
    shifted_pixels = [(1, 2), (6, 2), (2, 6), (7, 5)]
    for channel, (x, y) in enumerate(shifted_pixels):
        feature_map[channel, y, x] = 1.0
    K = torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])
    pose_gt = torch.eye(4)

    out = differentiable_pnp_pose_loss(
        descriptors,
        feature_map,
        points,
        K,
        pose_gt,
        point_weights=point_weights,
        config=DifferentiablePnPConfig(
            temperature=0.01,
            min_correspondences=4,
            pnp_iterations=0,
            pose_weight=0.0,
            reprojection_weight=0.0,
            gt_reprojection_weight=1.0,
            point_weight_floor=0.75,
        ),
    )
    out.loss.backward()

    assert out.diagnostics["point_weight_min"] >= 0.75 - 1e-6
    assert point_weights.grad is not None
    assert point_weights.grad.abs().sum() > 0


def test_weighted_point_loss_cauchy_limits_outlier_gradient():
    from localization_training.lafgs_reconstruction import _weighted_point_loss

    target_uv = torch.zeros(3, 2)
    weights = torch.ones(3)
    default_uv = torch.tensor([[0.0, 0.0], [1.0, 0.0], [30.0, 0.0]], requires_grad=True)
    robust_uv = default_uv.detach().clone().requires_grad_(True)

    default_loss = _weighted_point_loss(default_uv, target_uv, weights)
    robust_loss = _weighted_point_loss(
        robust_uv,
        target_uv,
        weights,
        loss_type="cauchy",
        robust_delta=2.0,
    )

    default_loss.backward()
    robust_loss.backward()

    assert robust_loss < default_loss
    assert robust_uv.grad[1, 0] > 0.0
    assert robust_uv.grad[-1, 0].abs() < default_uv.grad[-1, 0].abs() * 0.25


def test_diff_pnp_grid_selection_preserves_spatial_coverage_over_confidence_only_topk():
    from localization_training.lafgs_reconstruction import _select_pnp_correspondence_indices

    valid_idx = torch.arange(8)
    confidence = torch.tensor([0.99, 0.98, 0.97, 0.96, 0.50, 0.49, 0.48, 0.47])
    reference_uv = torch.tensor(
        [
            [1.0, 1.0],
            [1.2, 1.0],
            [1.0, 1.2],
            [1.2, 1.2],
            [7.0, 1.0],
            [1.0, 7.0],
            [7.0, 7.0],
            [6.0, 6.0],
        ]
    )

    selected, diagnostics = _select_pnp_correspondence_indices(
        valid_idx,
        confidence,
        reference_uv=reference_uv,
        image_size=(8, 8),
        max_correspondences=4,
        spatial_grid_size=2,
    )

    assert selected.tolist() == [0, 4, 5, 6]
    assert diagnostics["selected_spatial_cells"] == 4
    assert diagnostics["selection_mode"] == "grid"


def test_diff_pnp_rejects_spatially_degenerate_correspondence_sets():
    from localization_training.lafgs_reconstruction import (
        DifferentiablePnPConfig,
        differentiable_pnp_pose_loss,
    )

    points = torch.tensor(
        [
            [-0.5, -0.5, 2.0],
            [0.5, -0.5, 2.0],
            [-0.5, 0.5, 2.0],
            [0.5, 0.5, 2.0],
        ]
    )
    descriptors = torch.eye(4, requires_grad=True)
    feature_map = torch.zeros(4, 8, 8)
    for channel in range(4):
        feature_map[channel, 1, 1] = 1.0
    projected_uv = torch.tensor([[1.0, 1.0], [1.1, 1.0], [1.0, 1.1], [1.1, 1.1]])
    K = torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])
    pose_gt = torch.eye(4)

    out = differentiable_pnp_pose_loss(
        descriptors,
        feature_map,
        points,
        K,
        pose_gt,
        projected_uv=projected_uv,
        config=DifferentiablePnPConfig(
            temperature=0.01,
            min_correspondences=4,
            local_window_radius=1.5,
            min_spatial_span=0.25,
        ),
    )

    assert out.used_correspondences == 0
    assert out.diagnostics["skipped_reason"] == "spatial_span"


def test_pnp_output_can_update_pose_aware_topology_stats():
    from localization_training.lafgs_reconstruction import (
        DifferentiablePnPConfig,
        differentiable_pnp_pose_loss,
        pnp_output_to_landmark_stats,
    )

    points = torch.tensor(
        [
            [-0.5, -0.5, 2.0],
            [0.5, -0.5, 2.0],
            [-0.5, 0.5, 2.0],
            [0.5, 0.5, 2.0],
        ]
    )
    descriptors = torch.eye(4, requires_grad=True)
    feature_map = torch.zeros(4, 8, 8)
    shifted_pixels = [(2, 2), (6, 2), (2, 6), (6, 6)]
    for channel, (x, y) in enumerate(shifted_pixels):
        feature_map[channel, y, x] = 1.0
    K = torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])
    pose_gt = torch.eye(4)

    out = differentiable_pnp_pose_loss(
        descriptors,
        feature_map,
        points,
        K,
        pose_gt,
        config=DifferentiablePnPConfig(
            temperature=0.01,
            min_correspondences=4,
            pnp_iterations=0,
            pose_weight=0.0,
            reprojection_weight=0.0,
            gt_reprojection_weight=1.0,
        ),
    )
    stats = pnp_output_to_landmark_stats(out, points, K, pose_gt)

    assert set(["positive_prob", "entropy", "reproj_error", "information", "repeatability"]).issubset(stats)
    assert stats["positive_prob"].shape == (4,)
    assert stats["reproj_error"][0] > 0.0
    assert torch.all(stats["information"] <= 1.0)
    assert torch.all(stats["information"] >= 0.0)


def test_pnp_output_localization_utility_uses_full_bank_pose_and_reprojection_quality():
    from localization_training.lafgs_reconstruction import (
        DifferentiablePnPOutput,
        SoftCorrespondenceOutput,
        pnp_output_to_landmark_stats,
    )

    points = torch.tensor(
        [
            [-0.5, -0.5, 2.0],
            [0.5, -0.5, 2.0],
            [-0.5, 0.5, 2.0],
        ]
    )
    K = torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])
    pose_gt = torch.eye(4)
    good_uv = torch.tensor([[2.0, 2.0], [6.0, 2.0], [2.0, 6.0]])
    mixed_uv = good_uv.clone()
    mixed_uv[1] = torch.tensor([14.0, 2.0])
    correspondences = SoftCorrespondenceOutput(
        uv=mixed_uv,
        confidence=torch.tensor([0.95, 0.95, 0.95]),
        entropy=torch.tensor([0.05, 0.05, 0.05]),
        peak_probability=torch.tensor([0.95, 0.95, 0.95]),
        margin=torch.tensor([0.9, 0.9, 0.9]),
        probabilities=torch.empty(3, 0),
        valid=torch.tensor([True, True, True]),
    )
    pnp_out = DifferentiablePnPOutput(
        loss=torch.tensor(0.0),
        pose_loss=torch.tensor(0.1),
        reprojection_loss=torch.tensor(0.0),
        gt_reprojection_loss=torch.tensor(0.0),
        geometry_reprojection_loss=torch.tensor(0.0),
        entropy_loss=torch.tensor(0.0),
        pose_w2c=pose_gt,
        correspondences=correspondences,
        used_correspondences=3,
    )

    stats = pnp_output_to_landmark_stats(
        pnp_out,
        points,
        K,
        pose_gt,
        full_bank_positive_prob=torch.tensor([0.95, 0.95, 0.15]),
        full_bank_margin=torch.tensor([0.8, 0.8, -0.8]),
        pose_loss_scale=1.0,
        reprojection_error_scale=4.0,
    )
    degraded_pose_stats = pnp_output_to_landmark_stats(
        DifferentiablePnPOutput(
            loss=pnp_out.loss,
            pose_loss=torch.tensor(10.0),
            reprojection_loss=pnp_out.reprojection_loss,
            gt_reprojection_loss=pnp_out.gt_reprojection_loss,
            geometry_reprojection_loss=pnp_out.geometry_reprojection_loss,
            entropy_loss=pnp_out.entropy_loss,
            pose_w2c=pnp_out.pose_w2c,
            correspondences=correspondences,
            used_correspondences=3,
        ),
        points,
        K,
        pose_gt,
        full_bank_positive_prob=torch.tensor([0.95, 0.95, 0.15]),
        full_bank_margin=torch.tensor([0.8, 0.8, -0.8]),
        pose_loss_scale=1.0,
        reprojection_error_scale=4.0,
    )

    assert "loc_utility" in stats
    assert torch.allclose(stats["information"], stats["loc_utility"])
    assert stats["loc_utility"][0] > stats["loc_utility"][1]
    assert stats["loc_utility"][0] > stats["loc_utility"][2]
    assert stats["outlier"][0] < stats["outlier"][1]
    assert degraded_pose_stats["loc_utility"][0] < stats["loc_utility"][0]


def test_pnp_output_full_bank_probability_falls_back_to_confidence_without_external_stats():
    from localization_training.lafgs_reconstruction import (
        DifferentiablePnPOutput,
        SoftCorrespondenceOutput,
        pnp_output_to_landmark_stats,
    )

    points = torch.tensor([[-0.5, -0.5, 2.0], [0.5, -0.5, 2.0]])
    K = torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])
    pose_gt = torch.eye(4)
    correspondences = SoftCorrespondenceOutput(
        uv=torch.tensor([[2.0, 2.0], [6.0, 2.0]]),
        confidence=torch.tensor([0.25, 0.75]),
        entropy=torch.zeros(2),
        peak_probability=torch.tensor([0.25, 0.75]),
        margin=torch.ones(2),
        probabilities=torch.empty(2, 0),
        valid=torch.tensor([True, True]),
    )
    pnp_out = DifferentiablePnPOutput(
        loss=torch.tensor(0.0),
        pose_loss=torch.tensor(0.0),
        reprojection_loss=torch.tensor(0.0),
        gt_reprojection_loss=torch.tensor(0.0),
        geometry_reprojection_loss=torch.tensor(0.0),
        entropy_loss=torch.tensor(0.0),
        pose_w2c=pose_gt,
        correspondences=correspondences,
        used_correspondences=2,
    )

    stats = pnp_output_to_landmark_stats(pnp_out, points, K, pose_gt)

    assert torch.allclose(stats["full_bank_positive_prob"], stats["positive_prob"])
    assert torch.all(stats["loc_utility"] <= stats["positive_prob"])


def test_diff_pnp_summary_records_usage_loss_and_geometry_grad():
    from localization_training.lafgs_reconstruction import update_diff_pnp_training_summary

    class DummyPnPOutput:
        used_correspondences = 7
        diagnostics = {
            "detach_pnp_points": 1.0,
            "pose_loss_delta": 2.0,
            "geometry_valid_candidate_count": 5.0,
            "geometry_filter_keep_ratio": 0.4,
            "geometry_candidate_confidence_mean": 0.03,
            "geometry_kept_confidence_mean": 0.05,
            "geometry_candidate_margin_mean": 0.002,
            "geometry_kept_margin_mean": 0.006,
            "geometry_candidate_peak_probability_mean": 0.45,
            "geometry_kept_peak_probability_mean": 0.82,
            "geometry_candidate_entropy_mean": 0.65,
            "geometry_kept_entropy_mean": 0.24,
            "geometry_candidate_reprojection_error_mean": 3.0,
            "geometry_kept_reprojection_error_mean": 1.2,
            "geometry_peak_probability_threshold": 0.8,
            "geometry_max_entropy": 0.35,
            "condition_guard_max_condition_number": 100000.0,
            "condition_guard_scale": 1.0,
            "condition_guard_passed": 1.0,
        }

    summary = {
        "diff_pnp_episodes": 0,
        "diff_pnp_used_correspondences_total": 0,
    }

    update_diff_pnp_training_summary(
        summary,
        DummyPnPOutput(),
        torch.tensor(0.25),
        allow_geometry_grad=True,
    )

    assert summary["diff_pnp_episodes"] == 1
    assert summary["diff_pnp_nonzero_loss_episodes"] == 1
    assert summary["diff_pnp_used_correspondences_total"] == 7
    assert summary["diff_pnp_allow_geometry_grad_episodes"] == 1
    assert summary["diff_pnp_detach_pnp_points_total"] == pytest.approx(1.0)
    assert summary["diff_pnp_detach_pnp_points_max"] == pytest.approx(1.0)
    assert summary["diff_pnp_pose_loss_delta_total"] == pytest.approx(2.0)
    assert summary["diff_pnp_pose_loss_delta_max"] == pytest.approx(2.0)
    assert summary["diff_pnp_geometry_valid_candidate_count_total"] == pytest.approx(5.0)
    assert summary["diff_pnp_geometry_filter_keep_ratio_total"] == pytest.approx(0.4)
    assert summary["diff_pnp_geometry_candidate_confidence_mean_total"] == pytest.approx(0.03)
    assert summary["diff_pnp_geometry_kept_confidence_mean_total"] == pytest.approx(0.05)
    assert summary["diff_pnp_geometry_candidate_margin_mean_total"] == pytest.approx(0.002)
    assert summary["diff_pnp_geometry_kept_margin_mean_total"] == pytest.approx(0.006)
    assert summary["diff_pnp_geometry_candidate_peak_probability_mean_total"] == pytest.approx(0.45)
    assert summary["diff_pnp_geometry_kept_peak_probability_mean_total"] == pytest.approx(0.82)
    assert summary["diff_pnp_geometry_candidate_entropy_mean_total"] == pytest.approx(0.65)
    assert summary["diff_pnp_geometry_kept_entropy_mean_total"] == pytest.approx(0.24)
    assert summary["diff_pnp_geometry_candidate_reprojection_error_mean_total"] == pytest.approx(3.0)
    assert summary["diff_pnp_geometry_kept_reprojection_error_mean_total"] == pytest.approx(1.2)
    assert summary["diff_pnp_geometry_peak_probability_threshold_total"] == pytest.approx(0.8)
    assert summary["diff_pnp_geometry_max_entropy_total"] == pytest.approx(0.35)
    assert summary["diff_pnp_condition_guard_max_condition_number_total"] == pytest.approx(100000.0)
    assert summary["diff_pnp_condition_guard_scale_total"] == pytest.approx(1.0)
    assert summary["diff_pnp_condition_guard_passed_total"] == pytest.approx(1.0)
    assert summary["diff_pnp_loss_total"] == pytest.approx(0.25)


def test_pose_aware_split_score_requires_residual_ambiguity_repeatability_and_footprint():
    from localization_training.lafgs_reconstruction import pose_aware_split_score

    score = pose_aware_split_score(
        footprint=torch.tensor([12.0, 12.0, 12.0, 2.0]),
        ambiguity=torch.tensor([0.2, 0.8, 0.8, 0.8]),
        pnp_residual=torch.tensor([5.0, 5.0, 0.0, 5.0]),
        repeatability=torch.tensor([0.9, 0.9, 0.9, 0.9]),
        positive_prob=torch.tensor([0.9, 0.9, 0.9, 0.9]),
        min_footprint=4.0,
        min_repeatability=0.25,
    )

    assert score[1] > score[0]
    assert score[2] == 0.0
    assert score[3] == 0.0


def test_geometry_residual_anchor_penalizes_motion_beyond_rgb_scale_radius():
    from localization_training.lafgs_reconstruction import bounded_geometry_residual_loss

    current = torch.tensor([[0.05, 0.0, 0.0], [0.40, 0.0, 0.0]])
    source = torch.zeros_like(current)
    scale = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    loss, stats = bounded_geometry_residual_loss(
        current,
        source,
        scale,
        max_scale_ratio=0.1,
    )

    assert loss > 0.0
    assert stats["over_limit_count"] == 1
    assert stats["max_residual_norm"] > stats["max_allowed_norm"]


def test_lafgs_synthetic_view_policy_includes_direct_locrec_teacher():
    from localization_training.lafgs_reconstruction import lafgs_should_sample_synthetic_view

    assert lafgs_should_sample_synthetic_view("direct", 0.1, query_camera_count=2, random_value=0.05)
    assert lafgs_should_sample_synthetic_view("dense", 0.1, query_camera_count=2, random_value=0.05)
    assert not lafgs_should_sample_synthetic_view("direct", 0.1, query_camera_count=1, random_value=0.05)
    assert not lafgs_should_sample_synthetic_view("direct", 0.1, query_camera_count=2, random_value=0.2)


def test_lafgs_curriculum_unlocks_pose_and_geometry_in_order():
    from localization_training.lafgs_reconstruction import (
        LaFGSCurriculumConfig,
        lafgs_curriculum_step,
        lafgs_phase_from_starts,
        lafgs_phase_for_iteration,
        lafgs_trainable_param_names,
    )

    config = LaFGSCurriculumConfig(
        mv_init_until=100,
        locrec_until=200,
        diff_pnp_until=300,
        geometry_until=400,
    )

    assert lafgs_phase_for_iteration(50, config) == "mv_init"
    assert lafgs_phase_for_iteration(150, config) == "locrec"
    assert lafgs_phase_for_iteration(250, config) == "diff_pnp"
    assert lafgs_phase_for_iteration(350, config) == "geometry"
    assert lafgs_phase_for_iteration(450, config) == "topology"
    assert lafgs_trainable_param_names("locrec") == {"loc_feature", "loc_opacity"}
    assert "xyz" not in lafgs_trainable_param_names("diff_pnp")
    assert "xyz" in lafgs_trainable_param_names("geometry")
    assert lafgs_phase_from_starts(1, 1, 10, 20, 30) == "locrec"
    assert lafgs_phase_from_starts(10, 1, 10, 20, 30) == "diff_pnp"
    assert lafgs_phase_from_starts(20, 1, 10, 20, 30) == "geometry"
    assert lafgs_phase_from_starts(30, 1, 10, 20, 30) == "topology"
    assert lafgs_curriculum_step(iteration=30001, base_iteration=30000) == 1
    assert lafgs_phase_from_starts(
        lafgs_curriculum_step(iteration=30001, base_iteration=30000),
        1,
        10,
        20,
        30,
    ) == "locrec"


def test_train_lafgs_defaults_enable_direct_lafgs_mainline():
    import train_lafgs

    parser = train_lafgs.build_parser()
    args = parser.parse_args([])
    defaults = train_lafgs.lafgs_defaults(args)

    assert defaults.loc_teacher == "direct"
    assert defaults.localization_enabled is True
    assert defaults.use_loc_opacity is True
    assert defaults.geometry_anchor_weight > 0.0
    assert defaults.loc_full_bank_weight > 0.0
    assert defaults.lafgs_diff_pnp_weight > 0.0
    assert defaults.lafgs_mvinit_enabled is True
    assert defaults.lafgs_curriculum is True
    assert defaults.lafgs_synthetic_feature_source == "rgb"
    assert defaults.lafgs_mvinit_chunk_size > 0
    assert defaults.lafgs_mvinit_view_selection == "uniform"
    assert defaults.lafgs_diff_pnp_start_iter > defaults.lafgs_locrec_start_iter


def test_select_multiview_init_cameras_uniformly_spreads_support_views():
    from localization_training.lafgs_reconstruction import select_multiview_init_cameras

    cameras = [f"cam{i}" for i in range(10)]

    assert select_multiview_init_cameras(cameras, max_views=4, mode="uniform") == [
        "cam0",
        "cam3",
        "cam6",
        "cam9",
    ]
    assert select_multiview_init_cameras(cameras, max_views=4, mode="first") == [
        "cam0",
        "cam1",
        "cam2",
        "cam3",
    ]
    assert select_multiview_init_cameras(cameras, max_views=-1, mode="uniform") == cameras
    assert select_multiview_init_cameras(cameras, max_views=0, mode="uniform") == []


def test_train_lafgs_preserves_explicit_synthetic_feature_source_override():
    import train_lafgs

    parser = train_lafgs.build_parser()
    args = parser.parse_args(["--lafgs_synthetic_feature_source", "loc_feature"])
    defaults = train_lafgs.lafgs_defaults(
        args,
        explicit_overrides={"lafgs_synthetic_feature_source"},
    )

    assert defaults.lafgs_synthetic_feature_source == "loc_feature"


def test_train_lafgs_preserves_explicit_zero_mvinit_views_as_disabled():
    import train_lafgs

    parser = train_lafgs.build_parser()
    args = parser.parse_args(["--lafgs_mvinit_max_views", "0"])
    defaults = train_lafgs.lafgs_defaults(
        args,
        explicit_overrides={"lafgs_mvinit_max_views"},
    )

    assert defaults.lafgs_mvinit_max_views == 0


def test_train_lafgs_exposes_mvinit_feature_scale_override():
    import train_lafgs

    parser = train_lafgs.build_parser()
    defaults = train_lafgs.lafgs_defaults(parser.parse_args([]))
    assert defaults.lafgs_mvinit_feature_scale == pytest.approx(1.0)

    argv = ["--lafgs_mvinit_feature_scale", "0.5"]
    configured = train_lafgs.lafgs_defaults(
        parser.parse_args(argv),
        explicit_overrides=train_lafgs._explicit_lafgs_overrides(argv),
    )

    assert configured.lafgs_mvinit_feature_scale == pytest.approx(0.5)


def test_train_lafgs_preserves_explicit_zero_diff_pnp_weight_as_disabled():
    import train_lafgs

    parser = train_lafgs.build_parser()
    args = parser.parse_args(["--lafgs_diff_pnp_weight", "0"])
    defaults = train_lafgs.lafgs_defaults(
        args,
        explicit_overrides={"lafgs_diff_pnp_weight"},
    )

    assert defaults.lafgs_diff_pnp_weight == 0.0


def test_train_lafgs_keeps_diff_pnp_loc_opacity_weight_optional_with_safe_floor():
    import train_lafgs

    parser = train_lafgs.build_parser()
    defaults = train_lafgs.lafgs_defaults(parser.parse_args([]))
    assert defaults.lafgs_diff_pnp_use_loc_opacity_weight is False
    assert defaults.lafgs_diff_pnp_point_weight_floor == pytest.approx(0.75)

    enabled_argv = ["--lafgs_diff_pnp_use_loc_opacity_weight"]
    enabled = train_lafgs.lafgs_defaults(
        parser.parse_args(enabled_argv),
        explicit_overrides=train_lafgs._explicit_lafgs_overrides(enabled_argv),
    )
    assert enabled.lafgs_diff_pnp_use_loc_opacity_weight is True

    argv = ["--no-lafgs_diff_pnp_use_loc_opacity_weight"]
    disabled = train_lafgs.lafgs_defaults(
        parser.parse_args(argv),
        explicit_overrides=train_lafgs._explicit_lafgs_overrides(argv),
    )
    assert disabled.lafgs_diff_pnp_use_loc_opacity_weight is False

    floor_zero_argv = ["--lafgs_diff_pnp_point_weight_floor", "0"]
    floor_zero = train_lafgs.lafgs_defaults(
        parser.parse_args(floor_zero_argv),
        explicit_overrides=train_lafgs._explicit_lafgs_overrides(floor_zero_argv),
    )
    assert floor_zero.lafgs_diff_pnp_point_weight_floor == 0.0


def test_train_lafgs_exposes_robust_diff_pnp_reprojection_loss_override():
    import train_lafgs

    parser = train_lafgs.build_parser()
    defaults = train_lafgs.lafgs_defaults(parser.parse_args([]))
    assert defaults.lafgs_diff_pnp_reprojection_loss_type == "smooth_l1"
    assert defaults.lafgs_diff_pnp_reprojection_loss_delta == pytest.approx(1.0)

    argv = [
        "--lafgs_diff_pnp_reprojection_loss_type",
        "cauchy",
        "--lafgs_diff_pnp_reprojection_loss_delta",
        "2.5",
    ]
    configured = train_lafgs.lafgs_defaults(
        parser.parse_args(argv),
        explicit_overrides=train_lafgs._explicit_lafgs_overrides(argv),
    )

    assert configured.lafgs_diff_pnp_reprojection_loss_type == "cauchy"
    assert configured.lafgs_diff_pnp_reprojection_loss_delta == pytest.approx(2.5)


def test_train_lafgs_exposes_diff_pnp_geometry_xyz_lr_override():
    import train_lafgs

    parser = train_lafgs.build_parser()
    defaults = train_lafgs.lafgs_defaults(parser.parse_args([]))
    assert defaults.lafgs_diff_pnp_geometry_xyz_lr == pytest.approx(0.0)

    argv = ["--lafgs_diff_pnp_geometry_xyz_lr", "0.00002"]
    configured = train_lafgs.lafgs_defaults(
        parser.parse_args(argv),
        explicit_overrides=train_lafgs._explicit_lafgs_overrides(argv),
    )

    assert configured.lafgs_diff_pnp_geometry_xyz_lr == pytest.approx(2.0e-5)


def test_train_lafgs_exposes_diff_pnp_geometry_depth_anchor_weight_override():
    import train_lafgs

    parser = train_lafgs.build_parser()
    defaults = train_lafgs.lafgs_defaults(parser.parse_args([]))
    assert defaults.lafgs_diff_pnp_geometry_depth_anchor_weight == pytest.approx(0.0)

    argv = ["--lafgs_diff_pnp_geometry_depth_anchor_weight", "0.35"]
    configured = train_lafgs.lafgs_defaults(
        parser.parse_args(argv),
        explicit_overrides=train_lafgs._explicit_lafgs_overrides(argv),
    )

    assert configured.lafgs_diff_pnp_geometry_depth_anchor_weight == pytest.approx(0.35)


def test_train_lafgs_exposes_diff_pnp_localization_utility_scales():
    import train_lafgs

    parser = train_lafgs.build_parser()
    defaults = train_lafgs.lafgs_defaults(parser.parse_args([]))
    assert defaults.lafgs_diff_pnp_utility_pose_loss_scale == pytest.approx(1.0)
    assert defaults.lafgs_diff_pnp_utility_reprojection_error_scale == pytest.approx(4.0)

    argv = [
        "--lafgs_diff_pnp_utility_pose_loss_scale",
        "2.5",
        "--lafgs_diff_pnp_utility_reprojection_error_scale",
        "8.0",
    ]
    configured = train_lafgs.lafgs_defaults(
        parser.parse_args(argv),
        explicit_overrides=train_lafgs._explicit_lafgs_overrides(argv),
    )

    assert configured.lafgs_diff_pnp_utility_pose_loss_scale == pytest.approx(2.5)
    assert configured.lafgs_diff_pnp_utility_reprojection_error_scale == pytest.approx(8.0)


def test_train_lafgs_exposes_diff_pnp_isolated_geometry_grad():
    import train_lafgs

    parser = train_lafgs.build_parser()
    defaults = train_lafgs.lafgs_defaults(parser.parse_args([]))
    assert defaults.lafgs_diff_pnp_isolate_geometry_grad is False

    argv = ["--lafgs_diff_pnp_isolate_geometry_grad"]
    configured = train_lafgs.lafgs_defaults(
        parser.parse_args(argv),
        explicit_overrides=train_lafgs._explicit_lafgs_overrides(argv),
    )

    assert configured.lafgs_diff_pnp_isolate_geometry_grad is True


def test_train_lafgs_exposes_gated_diff_pnp_geometry_reprojection_controls():
    import train_lafgs

    parser = train_lafgs.build_parser()
    defaults = train_lafgs.lafgs_defaults(parser.parse_args([]))
    assert defaults.lafgs_diff_pnp_geometry_reproj_weight == pytest.approx(0.0)
    assert defaults.lafgs_diff_pnp_geometry_max_reproj_error == pytest.approx(0.0)
    assert defaults.lafgs_diff_pnp_geometry_confidence_threshold == pytest.approx(0.0)
    assert defaults.lafgs_diff_pnp_geometry_margin_threshold == pytest.approx(0.0)
    assert defaults.lafgs_diff_pnp_geometry_peak_probability_threshold == pytest.approx(0.0)
    assert defaults.lafgs_diff_pnp_geometry_max_entropy == pytest.approx(0.0)
    assert defaults.lafgs_diff_pnp_geometry_use_all_correspondences is False
    assert defaults.lafgs_diff_pnp_local_window_radius == pytest.approx(0.0)
    assert defaults.lafgs_diff_pnp_geometry_local_window_radius == pytest.approx(0.0)
    assert defaults.lafgs_diff_pnp_max_condition_number == pytest.approx(-1.0)
    assert defaults.lafgs_diff_pnp_geometry_pose_guard_max_loss_increase == pytest.approx(-1.0)
    assert defaults.lafgs_diff_pnp_geometry_pose_guard_max_loss == pytest.approx(-1.0)
    assert defaults.lafgs_diff_pnp_geometry_pose_guard_softness == pytest.approx(0.0)
    assert defaults.lafgs_diff_pnp_geometry_pose_guard_min_scale == pytest.approx(0.0)
    assert defaults.lafgs_diff_pnp_feedback_pose_guard_max_loss_increase == pytest.approx(-1.0)
    assert defaults.lafgs_diff_pnp_feedback_pose_guard_max_loss == pytest.approx(-1.0)
    assert defaults.lafgs_diff_pnp_feedback_pose_guard_softness == pytest.approx(0.0)
    assert defaults.lafgs_diff_pnp_feedback_pose_guard_min_scale == pytest.approx(0.0)
    assert defaults.lafgs_diff_pnp_feedback_pose_guard_keep_gt_reprojection is False
    assert defaults.lafgs_diff_pnp_detach_gt_reprojection_points is False
    assert defaults.lafgs_diff_pnp_detach_pnp_points is False

    argv = [
        "--lafgs_diff_pnp_geometry_reproj_weight",
        "1.5",
        "--lafgs_diff_pnp_geometry_max_reproj_error",
        "3.0",
        "--lafgs_diff_pnp_geometry_confidence_threshold",
        "0.2",
        "--lafgs_diff_pnp_geometry_margin_threshold",
        "0.15",
        "--lafgs_diff_pnp_geometry_peak_probability_threshold",
        "0.8",
        "--lafgs_diff_pnp_geometry_max_entropy",
        "0.35",
        "--lafgs_diff_pnp_geometry_use_all_correspondences",
        "--lafgs_diff_pnp_local_window_radius",
        "1.25",
        "--lafgs_diff_pnp_geometry_local_window_radius",
        "1.5",
        "--lafgs_diff_pnp_max_condition_number",
        "100000",
        "--lafgs_diff_pnp_geometry_pose_guard_max_loss_increase",
        "0.0",
        "--lafgs_diff_pnp_geometry_pose_guard_max_loss",
        "2.0",
        "--lafgs_diff_pnp_geometry_pose_guard_softness",
        "10.0",
        "--lafgs_diff_pnp_geometry_pose_guard_min_scale",
        "0.05",
        "--lafgs_diff_pnp_feedback_pose_guard_max_loss_increase",
        "0.25",
        "--lafgs_diff_pnp_feedback_pose_guard_max_loss",
        "3.0",
        "--lafgs_diff_pnp_feedback_pose_guard_softness",
        "10.0",
        "--lafgs_diff_pnp_feedback_pose_guard_min_scale",
        "0.05",
        "--lafgs_diff_pnp_feedback_pose_guard_keep_gt_reprojection",
        "--lafgs_diff_pnp_detach_gt_reprojection_points",
        "--lafgs_diff_pnp_detach_pnp_points",
    ]
    configured = train_lafgs.lafgs_defaults(
        parser.parse_args(argv),
        explicit_overrides=train_lafgs._explicit_lafgs_overrides(argv),
    )

    assert configured.lafgs_diff_pnp_geometry_reproj_weight == pytest.approx(1.5)
    assert configured.lafgs_diff_pnp_geometry_max_reproj_error == pytest.approx(3.0)
    assert configured.lafgs_diff_pnp_geometry_confidence_threshold == pytest.approx(0.2)
    assert configured.lafgs_diff_pnp_geometry_margin_threshold == pytest.approx(0.15)
    assert configured.lafgs_diff_pnp_geometry_peak_probability_threshold == pytest.approx(0.8)
    assert configured.lafgs_diff_pnp_geometry_max_entropy == pytest.approx(0.35)
    assert configured.lafgs_diff_pnp_geometry_use_all_correspondences is True
    assert configured.lafgs_diff_pnp_local_window_radius == pytest.approx(1.25)
    assert configured.lafgs_diff_pnp_geometry_local_window_radius == pytest.approx(1.5)
    assert configured.lafgs_diff_pnp_max_condition_number == pytest.approx(100000.0)
    assert configured.lafgs_diff_pnp_geometry_pose_guard_max_loss_increase == pytest.approx(0.0)
    assert configured.lafgs_diff_pnp_geometry_pose_guard_max_loss == pytest.approx(2.0)
    assert configured.lafgs_diff_pnp_geometry_pose_guard_softness == pytest.approx(10.0)
    assert configured.lafgs_diff_pnp_geometry_pose_guard_min_scale == pytest.approx(0.05)
    assert configured.lafgs_diff_pnp_feedback_pose_guard_max_loss_increase == pytest.approx(0.25)
    assert configured.lafgs_diff_pnp_feedback_pose_guard_max_loss == pytest.approx(3.0)
    assert configured.lafgs_diff_pnp_feedback_pose_guard_softness == pytest.approx(10.0)
    assert configured.lafgs_diff_pnp_feedback_pose_guard_min_scale == pytest.approx(0.05)
    assert configured.lafgs_diff_pnp_feedback_pose_guard_keep_gt_reprojection is True
    assert configured.lafgs_diff_pnp_detach_gt_reprojection_points is True
    assert configured.lafgs_diff_pnp_detach_pnp_points is True


def test_train_lafgs_preserves_explicit_zero_synthetic_view_ratio_as_disabled():
    import train_lafgs

    parser = train_lafgs.build_parser()
    args = parser.parse_args(["--synthetic_view_ratio", "0"])
    explicit = train_lafgs._explicit_lafgs_overrides(["--synthetic_view_ratio", "0"])
    defaults = train_lafgs.lafgs_defaults(args, explicit_overrides=explicit)

    assert defaults.synthetic_view_ratio == 0.0


def test_train_lafgs_preserves_explicit_zero_full_bank_weight_as_disabled():
    import train_lafgs

    parser = train_lafgs.build_parser()
    args = parser.parse_args(["--loc_full_bank_weight", "0"])
    explicit = train_lafgs._explicit_lafgs_overrides(["--loc_full_bank_weight", "0"])
    defaults = train_lafgs.lafgs_defaults(args, explicit_overrides=explicit)

    assert defaults.loc_full_bank_weight == 0.0
