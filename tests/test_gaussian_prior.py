import torch

from localization_training.gaussian_prior import (
    GaussianPriorGeometry,
    validate_gaussian_anchor_resume,
)


def test_3dgs_anchor_is_covariance_bounded_and_raw_center_is_unchanged():
    prior = GaussianPriorGeometry(
        "3dgs",
        xyz=torch.zeros(1, 3),
        rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        scaling=torch.tensor([[0.20, 0.10, 0.02]]),
    )
    anchor = prior.materialize_anchor(
        torch.full((1, 3), 100.0),
        tangent_bound_m=0.01,
        normal_bound_m=0.005,
        covariance_scale=0.5,
        absolute_bound_m=0.03,
    )

    torch.testing.assert_close(
        anchor,
        torch.tensor([[0.03, 0.03, 0.01]]),
        atol=1e-5,
        rtol=0,
    )
    torch.testing.assert_close(prior.xyz, torch.zeros(1, 3))


def test_2dgs_anchor_preserves_surface_bound_contract():
    prior = GaussianPriorGeometry(
        "2dgs",
        xyz=torch.zeros(1, 3),
        rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        scaling=torch.ones(1, 2),
    )
    bounds = prior.anchor_axis_bounds(
        tangent_bound_m=0.02,
        normal_bound_m=0.004,
        covariance_scale=1.0,
        absolute_bound_m=1.0,
    )
    torch.testing.assert_close(bounds, torch.tensor([[0.02, 0.02, 0.004]]))


def test_2dgs_tangent_displacement_uses_radial_bound():
    prior = GaussianPriorGeometry(
        "2dgs",
        xyz=torch.zeros(1, 3),
        rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        scaling=torch.ones(1, 2),
    )
    anchor = prior.materialize_anchor(
        torch.tensor([[100.0, 100.0, 0.0]]),
        tangent_bound_m=0.02,
        normal_bound_m=0.004,
        covariance_scale=1.0,
        absolute_bound_m=1.0,
    )
    torch.testing.assert_close(
        torch.linalg.norm(anchor[:, :2], dim=1),
        torch.tensor([0.02]),
        atol=1e-6,
        rtol=0,
    )


def test_2dgs_project_and_encode_anchor_obey_surface_bounds():
    prior = GaussianPriorGeometry(
        "2dgs",
        xyz=torch.tensor([[1.0, 2.0, 3.0]]),
        rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        scaling=torch.ones(1, 2),
    )
    projected = prior.project_anchor_target(
        torch.tensor([[1.03, 2.04, 3.02]]),
        tangent_bound_m=0.01,
        normal_bound_m=0.002,
        covariance_scale=1.0,
        absolute_bound_m=1.0,
    )
    local = prior.anchor_local_coordinates(projected)
    torch.testing.assert_close(
        torch.linalg.norm(local[:, :2], dim=1),
        torch.tensor([0.01]),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        local[:, 2], torch.tensor([0.002]), atol=1e-6, rtol=0
    )
    raw = prior.encode_anchor(
        projected,
        tangent_bound_m=0.01,
        normal_bound_m=0.002,
        covariance_scale=1.0,
        absolute_bound_m=1.0,
    )
    reconstructed = prior.materialize_anchor(
        raw,
        tangent_bound_m=0.01,
        normal_bound_m=0.002,
        covariance_scale=1.0,
        absolute_bound_m=1.0,
    )
    torch.testing.assert_close(
        reconstructed, projected, atol=2e-6, rtol=0
    )


def test_3dgs_project_and_encode_anchor_obey_covariance_bounds():
    prior = GaussianPriorGeometry(
        "3dgs",
        xyz=torch.zeros(1, 3),
        rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        scaling=torch.tensor([[0.20, 0.10, 0.02]]),
    )
    projected = prior.project_anchor_target(
        torch.full((1, 3), 1.0),
        tangent_bound_m=0.0,
        normal_bound_m=0.0,
        covariance_scale=0.5,
        absolute_bound_m=0.03,
    )
    torch.testing.assert_close(
        projected,
        torch.tensor([[0.03, 0.03, 0.01]]),
        atol=1e-6,
        rtol=0,
    )
    raw = prior.encode_anchor(
        projected,
        tangent_bound_m=0.0,
        normal_bound_m=0.0,
        covariance_scale=0.5,
        absolute_bound_m=0.03,
    )
    reconstructed = prior.materialize_anchor(
        raw,
        tangent_bound_m=0.0,
        normal_bound_m=0.0,
        covariance_scale=0.5,
        absolute_bound_m=0.03,
    )
    torch.testing.assert_close(
        reconstructed, projected, atol=2e-6, rtol=0
    )


def test_planarity_uses_smallest_to_middle_covariance_scale():
    prior = GaussianPriorGeometry(
        "3dgs",
        xyz=torch.zeros(2, 3),
        rotation=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        ),
        scaling=torch.tensor([[0.1, 0.01, 0.001], [0.1, 0.08, 0.07]]),
    )
    torch.testing.assert_close(prior.planarity, torch.tensor([0.1, 0.875]))


def test_3dgs_proxy_normal_uses_smallest_covariance_axis():
    prior = GaussianPriorGeometry(
        "3dgs",
        xyz=torch.zeros(1, 3),
        rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        scaling=torch.tensor([[0.01, 0.20, 0.10]]),
    )
    torch.testing.assert_close(
        prior.proxy_normals,
        torch.tensor([[1.0, 0.0, 0.0]]),
    )


def test_3dgs_resume_rejects_changed_covariance_bound():
    state = {
        "raw_anchor_offset": torch.ones(1, 3),
        "config": {
            "surface_anchor_parameterization": "covariance_bounded_tanh_v1",
            "covariance_anchor_scale": 0.5,
            "covariance_anchor_absolute_bound_m": 0.03,
        },
    }
    try:
        validate_gaussian_anchor_resume(
            state,
            gaussian_type="3dgs",
            tangent_bound_m=0.0,
            normal_bound_m=0.0,
            covariance_scale=0.5,
            absolute_bound_m=0.02,
        )
    except ValueError as exc:
        assert "absolute_bound_m" in str(exc)
    else:
        raise AssertionError("Changed 3DGS anchor bounds must be rejected")
