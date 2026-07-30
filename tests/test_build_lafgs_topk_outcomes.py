import pytest
import torch

from scripts.build_lafgs_topk_outcomes import _load_family_prototypes


def _family_state(anchor_ids):
    return {
        "schema": "lafgs_basin_family_prototypes",
        "landmark_indices": anchor_ids,
        "prototype_features": torch.tensor([[1.0, 0.0]]),
        "prototype_anchor_indices": torch.tensor([1]),
        "prototype_bias": torch.tensor([-0.1]),
        "prototype_temperature": torch.tensor([0.8]),
    }


def test_family_prototype_loader_enforces_candidate_graph_alignment(
    tmp_path,
):
    anchor_ids = torch.tensor([4, 9])
    path = tmp_path / "family.pt"
    torch.save(_family_state(anchor_ids), path)
    values = _load_family_prototypes(
        str(path),
        state={
            "anchor_ids": anchor_ids,
            "anchor_features": torch.eye(2),
        },
        device=torch.device("cpu"),
    )
    assert [tuple(value.shape) for value in values] == [
        (1, 2),
        (1,),
        (1,),
        (1,),
    ]

    invalid = _family_state(torch.tensor([9, 4]))
    torch.save(invalid, path)
    with pytest.raises(ValueError, match="does not align"):
        _load_family_prototypes(
            str(path),
            state={
                "anchor_ids": anchor_ids,
                "anchor_features": torch.eye(2),
            },
            device=torch.device("cpu"),
        )
