import pytest
import torch

from features.photometric import (
    canonicalize_image,
    percentile_grayscale_contract,
    validate_photometric_contract,
)


def test_percentile_grayscale_is_channel_symmetric_and_bounded():
    image = torch.linspace(0, 1, 300).reshape(3, 10, 10)
    output = canonicalize_image(image, percentile_grayscale_contract())
    assert output.shape == image.shape
    assert torch.equal(output[0], output[1])
    assert torch.equal(output[1], output[2])
    assert float(output.min()) == 0.0
    assert float(output.max()) == 1.0


def test_offline_online_synthetic_parity_uses_same_per_image_function():
    first = torch.rand(3, 11, 13, generator=torch.Generator().manual_seed(4))
    second = torch.rand(3, 11, 13, generator=torch.Generator().manual_seed(5))
    batched = canonicalize_image(
        torch.stack((first, second)), percentile_grayscale_contract()
    )
    assert torch.equal(
        batched,
        torch.stack(
            (
                canonicalize_image(first, percentile_grayscale_contract()),
                canonicalize_image(second, percentile_grayscale_contract()),
            )
        ),
    )


def test_constant_image_has_finite_zero_policy():
    output = canonicalize_image(
        torch.full((2, 3, 8, 9), 0.42), percentile_grayscale_contract()
    )
    assert torch.equal(output, torch.zeros_like(output))
    assert torch.isfinite(output).all()


def test_percentile_clips_one_percent_tails():
    image = torch.linspace(0, 1, 100).reshape(1, 1, 10, 10).expand(1, 3, 10, 10)
    output = canonicalize_image(image, percentile_grayscale_contract())[0, 0]
    assert float(output.flatten()[0]) == 0.0
    assert float(output.flatten()[-1]) == 1.0
    assert 0.49 < float(output.flatten()[50]) < 0.52


def test_contract_is_fail_closed():
    modified = percentile_grayscale_contract()
    modified["upper_percentile"] = 0.98
    with pytest.raises(ValueError, match="unsupported or modified"):
        validate_photometric_contract(modified)
