import torch

from priors.mapping_supersampling import (
    downsample_mapping_render_package,
    mapping_render_sampling_contract,
)


def test_supersampling_area_filters_rgb_alpha_and_alpha_weighted_depth() -> None:
    package = {
        "render": torch.tensor(
            [
                [[0.0, 0.0], [1.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0]],
                [[1.0, 1.0], [1.0, 1.0]],
            ]
        ),
        "rend_alpha": torch.tensor([[[[1.0], [1.0]], [[0.0], [0.0]]]]),
        "depth": torch.tensor([[[2.0, 4.0], [100.0, 100.0]]]),
        "rend_dist": torch.tensor([[[[0.0], [2.0]], [[2.0], [0.0]]]]),
    }
    output = downsample_mapping_render_package(
        package,
        target_height=1,
        target_width=1,
        supersampling_factor=2,
    )
    assert output["render"].shape == (3, 1, 1)
    assert torch.allclose(output["render"][:, 0, 0], torch.tensor([0.5, 0.5, 1.0]))
    assert torch.allclose(output["alphas"], torch.tensor([[[0.5]]]))
    assert torch.allclose(output["depth"], torch.tensor([[[3.0]]]))
    assert torch.allclose(output["rend_dist"], torch.tensor([[[1.0]]]))


def test_native_factor_is_shape_preserving() -> None:
    package = {
        "render": torch.rand(3, 2, 3),
        "alphas": torch.rand(1, 2, 3),
        "depth": torch.rand(1, 2, 3),
    }
    output = downsample_mapping_render_package(
        package,
        target_height=2,
        target_width=3,
        supersampling_factor=1,
    )
    assert torch.equal(output["render"], package["render"])
    assert torch.equal(output["alphas"], package["alphas"])
    assert torch.equal(output["depth"], package["depth"])


def test_invalid_subpixel_depth_does_not_contaminate_valid_surface() -> None:
    package = {
        "render": torch.ones(3, 2, 4),
        "alphas": torch.tensor([[[1., 0., 1., 1.], [1., 1., 0., 0.]]]),
        "depth": torch.tensor([[[2., float("nan"), -1., float("inf")],
                                [4., float("inf"), 0., float("nan")]]]),
    }
    output = downsample_mapping_render_package(
        package, target_height=1, target_width=2, supersampling_factor=2
    )
    assert torch.equal(output["depth"], torch.tensor([[[3., 0.]]]))
    assert torch.equal(output["alphas"], torch.tensor([[[0.75, 0.5]]]))


def test_2dgs_contract_does_not_claim_unsupported_antialias_argument() -> None:
    contract = mapping_render_sampling_contract(
        gaussian_type="2dgs", supersampling_factor=2
    )
    assert contract["effective_rasterize_mode"] == (
        "omitted_unsupported_by_2dgs_wrapper"
    )
    assert contract["rasterize_mode_argument_forwarded"] is False
    assert contract["rgb_downsampling"] == "area_box"
    assert contract["depth_downsampling"] == "alpha_weighted_area_box"
