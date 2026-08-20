import torch

from scripts.audit_render_perturbation_stability import (
    decile_diagnostics,
    descriptor_stability,
    perturb_rgb,
)


def test_identity_photometric_perturbation_is_exact():
    image = torch.rand(3, 8, 9, generator=torch.Generator().manual_seed(3))
    assert torch.equal(
        perturb_rgb(image, exposure=1.0, contrast=1.0, gamma=1.0), image
    )


def test_descriptor_stability_detects_one_outlier_variant():
    identity = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    variants = torch.stack((identity, identity, torch.tensor([[0.0, 1.0], [0.0, 1.0]])))
    variance, worst = descriptor_stability(variants)
    assert variance[0] > variance[1]
    assert worst.tolist() == [0.0, 1.0]


def test_decile_diagnostics_recovers_monotonic_relation():
    instability = torch.linspace(0, 1, 1000)
    report = decile_diagnostics(instability, 0.1 + instability)
    assert report["spearman_rho"] > 0.999
    assert report["monotonic_violation_count"] == 0
    assert report["highest_over_lowest_decile_gap_ratio"] > 1.25
