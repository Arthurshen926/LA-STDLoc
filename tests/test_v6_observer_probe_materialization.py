import torch

from scripts.materialize_v6_fixed_map_observer_probes import _sensor_variant


def test_sensor_variants_are_deterministic_bounded_observer_inputs() -> None:
    image = torch.full((3, 20, 30), 0.5)
    variants = (
        "clean",
        "exposure_down",
        "exposure_up",
        "gamma_low",
        "gamma_high",
        "motion_blur_mild",
        "sensor_noise_mild",
        "resize_compression_mild",
        "local_occlusion_mild",
    )
    outputs = {
        name: _sensor_variant(image, name, seed=2026) for name in variants
    }
    assert torch.equal(outputs["clean"], image)
    assert torch.equal(
        outputs["sensor_noise_mild"],
        _sensor_variant(image, "sensor_noise_mild", seed=2026),
    )
    assert all(value.shape == image.shape for value in outputs.values())
    assert all(
        float(value.min()) >= 0.0 and float(value.max()) <= 1.0
        for value in outputs.values()
    )
    assert not torch.equal(outputs["local_occlusion_mild"], image)
