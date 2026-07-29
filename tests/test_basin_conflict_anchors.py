import torch

from scripts.build_lafgs_basin_conflict_anchors import (
    append_conflict_anchors,
    verify_teacher_anchor_prefix,
)


def test_conflict_anchor_inherits_geometry_but_gets_new_descriptor_identity():
    state = {
        "anchor_xyz": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "anchor_features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "anchor_ids": torch.tensor([10, 11]),
        "fine_identity_ids": torch.tensor([20, 21]),
        "dependency_group_ids": torch.tensor([3, 4]),
        "source_primitive_ids": torch.tensor([7, 8]),
    }
    output = append_conflict_anchors(
        state,
        [
            {
                "source_anchor": 0,
                "query_group": 2,
                "observation_count": 3,
                "weighted_margin_gain": 0.2,
                "prototype": torch.tensor([0.0, 1.0]),
            }
        ],
        maximum_additions=1,
    )
    torch.testing.assert_close(output["anchor_xyz"][2], state["anchor_xyz"][0])
    torch.testing.assert_close(
        output["anchor_features"][2], torch.tensor([0.0, 1.0])
    )
    assert output["dependency_group_ids"][2] == 3
    assert output["source_primitive_ids"][2] == 7
    assert output["anchor_ids"][2] not in state["anchor_ids"]


def test_teacher_prefix_check_rejects_reordered_superset(tmp_path):
    source = {
        "anchor_ids": torch.tensor([10, 11]),
        "anchor_xyz": torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        "source_primitive_ids": torch.tensor([3, 4]),
    }
    source_path = tmp_path / "source.pt"
    torch.save(source, source_path)
    teacher = {
        "anchor_count": 2,
        "artifacts": {"map": {"path": str(source_path)}},
    }
    superset = {
        key: torch.cat((value.flip(0), value[:1]), dim=0)
        for key, value in source.items()
    }
    import pytest

    with pytest.raises(ValueError, match="prefix"):
        verify_teacher_anchor_prefix(superset, teacher)
