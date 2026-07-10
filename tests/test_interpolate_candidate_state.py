import pytest
import torch


def _state(indices, features):
    return {
        "landmark_indices": torch.tensor(indices),
        "landmark_features": torch.tensor(features, dtype=torch.float32),
        "pair_scorer_state_dict": {"weight": torch.tensor([3.0])},
        "diagnostics": {"existing": 1},
    }


def test_interpolate_candidate_state_preserves_lineage_and_scorer():
    from scripts.interpolate_candidate_state import interpolate_candidate_states

    base = _state([2, 4], [[1.0, 0.0], [0.0, 1.0]])
    tuned = _state([2, 4], [[0.0, 1.0], [1.0, 0.0]])
    output = interpolate_candidate_states(base, tuned, 0.5)

    expected = torch.full((2, 2), 2.0 ** -0.5)
    assert torch.equal(output["landmark_indices"], base["landmark_indices"])
    assert torch.allclose(output["landmark_features"], expected)
    assert output["pair_scorer_state_dict"]["weight"].item() == 3.0
    assert output["diagnostics"]["existing"] == 1
    assert output["diagnostics"]["feature_interpolation"]["alpha"] == 0.5


def test_interpolate_candidate_state_rejects_mismatched_lineage():
    from scripts.interpolate_candidate_state import interpolate_candidate_states

    base = _state([1], [[1.0, 0.0]])
    tuned = _state([2], [[0.0, 1.0]])

    with pytest.raises(ValueError, match="different landmark indices"):
        interpolate_candidate_states(base, tuned, 0.5)


def test_interpolate_candidate_state_rejects_invalid_alpha():
    from scripts.interpolate_candidate_state import interpolate_candidate_states

    state = _state([1], [[1.0, 0.0]])
    with pytest.raises(ValueError, match="alpha"):
        interpolate_candidate_states(state, state, 1.1)
