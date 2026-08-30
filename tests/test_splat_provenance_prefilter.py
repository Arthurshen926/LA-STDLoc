import torch

from priors.rasterizer import bank_splat_provenance_2dgs


def test_2dgs_footprint_prefilter_preserves_exact_composition() -> None:
    means = torch.stack(
        [torch.tensor([float(index * 20) + 0.5, 0.5]) for index in range(20)]
    )
    transforms = torch.zeros(20, 3, 3)
    transforms[:, 0, 0] = 1.0
    transforms[:, 1, 1] = 1.0
    transforms[:, 2, 2] = 1.0
    transforms[:, 0, 2] = means[:, 0]
    transforms[:, 1, 2] = means[:, 1]
    metadata = {
        "means2d": means,
        "depths": torch.arange(20).float() + 5.0,
        "ray_transforms": transforms,
        "opacities": torch.full((20,), 0.8),
        "radii": torch.full((20,), 8.0),
    }
    keypoints = torch.tensor([[0.0, 0.0], [20.0, 0.0]])
    universe = torch.arange(20)
    exact = bank_splat_provenance_2dgs(
        keypoints,
        universe,
        metadata,
        topk=2,
        candidate_topk=4,
    )
    screened = bank_splat_provenance_2dgs(
        keypoints,
        universe,
        metadata,
        topk=2,
        candidate_topk=4,
        prefilter_topk=5,
    )
    assert torch.equal(
        screened[0][screened[1] > 1e-8], exact[0][exact[1] > 1e-8]
    )
    assert torch.allclose(
        screened[1][screened[1] > 1e-8], exact[1][exact[1] > 1e-8]
    )
    assert torch.equal(screened[2], exact[2])


def test_full_composition_accounts_for_cumulative_front_occlusion() -> None:
    means = torch.full((4, 2), 0.5)
    transforms = torch.zeros(4, 3, 3)
    transforms[:, 0, 0] = 1.0
    transforms[:, 1, 1] = 1.0
    transforms[:, 2, 2] = 1.0
    transforms[:, 0, 2] = 0.5
    transforms[:, 1, 2] = 0.5
    metadata = {
        "means2d": means,
        "depths": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "ray_transforms": transforms,
        "opacities": torch.tensor([0.4, 0.4, 0.4, 0.9]),
        "radii": torch.full((4,), 8.0),
    }
    truncated = bank_splat_provenance_2dgs(
        torch.tensor([[0.0, 0.0]]),
        torch.arange(4),
        metadata,
        topk=1,
        candidate_topk=1,
    )
    complete = bank_splat_provenance_2dgs(
        torch.tensor([[0.0, 0.0]]),
        torch.arange(4),
        metadata,
        topk=1,
        candidate_topk=0,
        return_diagnostics=True,
    )
    assert int(truncated[0][0, 0]) == 3
    assert int(complete[0][0, 0]) == 0
    assert 0.0 < float(complete[3]["retained_composition_fraction"][0]) < 1.0
