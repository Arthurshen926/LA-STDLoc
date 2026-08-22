import pytest

from evidence.projective_completion import build_projective_completion


def test_projective_completion_rejects_invalid_voxel_size() -> None:
    with pytest.raises(ValueError):
        build_projective_completion(
            None,
            {},
            voxel_size_m=0.0,
            alpha_minimum=0.05,
            minimum_similarity=0.65,
        )
