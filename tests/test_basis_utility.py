import torch

from localization_training.basis_utility import (
    deterministic_triplets,
    group_independent_triplets,
    image_triangle_area_fraction,
    triangle_shape_quality,
)


def test_triangle_quality_rejects_collinear_geometry():
    triangles = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        ]
    )
    quality = triangle_shape_quality(triangles)
    assert quality[0] > 0.4
    assert quality[1] == 0
    area = image_triangle_area_fraction(triangles, (10, 10))
    assert torch.allclose(area, torch.tensor([0.005, 0.0]))


def test_triplet_sampling_and_group_independence_are_deterministic():
    selected = torch.arange(20)
    first = deterministic_triplets(
        selected, count=12, seed=7, query_name="seq/frame.png"
    )
    second = deterministic_triplets(
        selected, count=12, seed=7, query_name="seq/frame.png"
    )
    assert torch.equal(first, second)
    assert len(torch.unique(first, dim=0)) == 12
    dependency = torch.arange(20)
    source = torch.arange(20)
    assert group_independent_triplets(
        first, dependency, source
    ).all()

    dependency[first[0, 1]] = dependency[first[0, 0]]
    independent = group_independent_triplets(
        first, dependency, source
    )
    assert not independent[0]
