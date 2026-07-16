import torch


def test_compress_2dgs_rgb_meta_keeps_only_landmark_bank():
    from localization_training.splat_provenance import compress_2dgs_rgb_meta_to_bank

    meta = {
        "means2d": torch.arange(10, dtype=torch.float32).reshape(1, 5, 2),
        "depths": torch.arange(5, dtype=torch.float32).reshape(1, 5),
        "ray_transforms": torch.arange(45, dtype=torch.float32).reshape(1, 5, 3, 3),
        "opacities": torch.linspace(0.1, 0.5, 5).reshape(1, 5),
        "radii": torch.arange(5, dtype=torch.float32).reshape(1, 5),
        "rendered_depth": torch.ones(4, 4),
    }
    compressed = compress_2dgs_rgb_meta_to_bank(meta, torch.tensor([4, 1]))

    assert compressed["means2d"].shape == (2, 2)
    assert compressed["depths"].tolist() == [4.0, 1.0]
    assert compressed["ray_transforms"].shape == (2, 3, 3)
    assert compressed["opacities"].shape == (2,)
    assert compressed["radii"].tolist() == [4.0, 1.0]
    assert compressed["rendered_depth"] is meta["rendered_depth"]

from localization_training.splat_provenance import bank_splat_provenance_2dgs


def _centered_transform(x, y):
    return torch.tensor(
        [[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def test_2dgs_provenance_respects_front_to_back_transmittance():
    meta = {
        "means2d": torch.tensor([[[2.5, 2.5], [2.5, 2.5]]]),
        "ray_transforms": torch.stack(
            [_centered_transform(2.5, 2.5), _centered_transform(2.5, 2.5)]
        )[None],
        "depths": torch.tensor([[1.0, 2.0]]),
        "opacities": torch.tensor([[0.8, 0.2]]),
        "radii": torch.tensor([[[3, 3], [3, 3]]]),
    }
    idx, weight, valid = bank_splat_provenance_2dgs(
        torch.tensor([[2.0, 2.0]]),
        torch.tensor([0, 1]),
        meta,
        topk=2,
        depth_abs_tolerance=10.0,
    )
    assert valid.tolist() == [True]
    assert idx.tolist() == [[0, 1]]
    assert torch.allclose(weight.sum(dim=1), torch.ones(1))
    assert weight[0, 0] > weight[0, 1]


def test_2dgs_provenance_depth_guard_removes_occluded_landmark():
    meta = {
        "means2d": torch.tensor([[[1.5, 1.5], [1.5, 1.5]]]),
        "ray_transforms": torch.stack(
            [_centered_transform(1.5, 1.5), _centered_transform(1.5, 1.5)]
        )[None],
        "depths": torch.tensor([[1.0, 3.0]]),
        "opacities": torch.tensor([[0.5, 0.9]]),
        "radii": torch.tensor([[[2, 2], [2, 2]]]),
    }
    depth = torch.ones(1, 4, 4)
    idx, weight, valid = bank_splat_provenance_2dgs(
        torch.tensor([[1.0, 1.0]]),
        torch.tensor([0, 1]),
        meta,
        rendered_depth=depth,
        topk=2,
        depth_abs_tolerance=0.1,
        depth_rel_tolerance=0.0,
    )
    assert valid.tolist() == [True]
    assert idx[0, 0].item() == 0
    assert weight[0, 1].item() == 0.0
