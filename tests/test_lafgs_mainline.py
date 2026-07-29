import pytest
import torch

from localization_training.appearance_calibration import (
    validate_dynamic_baseline_binding,
)

from localization_training.mainline import LocalizationRoundState


def test_dynamic_baseline_binding_rejects_unbound_or_wrong_family(tmp_path):
    expected = tmp_path / "family.pt"
    other = tmp_path / "other.pt"
    with pytest.raises(ValueError, match="do not bind"):
        validate_dynamic_baseline_binding(
            {}, base_family_path=str(expected)
        )
    validate_dynamic_baseline_binding(
        {},
        base_family_path=str(expected),
        allow_unbound=True,
    )
    with pytest.raises(ValueError, match="does not match"):
        validate_dynamic_baseline_binding(
            {"family_prototype_state": str(other)},
            base_family_path=str(expected),
        )


class _Metric:
    pass


def _state():
    return LocalizationRoundState(
        anchor_map={"anchor_xyz": torch.zeros((2, 3))},
        metric=_Metric(),
        query_cache={"q": {}},
        complete_positive_teacher={
            "query_names": ["q"],
            "anchor_count": 2,
        },
        dynamic_outcomes={"query_names": ["q"], "anchor_count": 2},
        query_bins={"q": 0},
    )


def test_mainline_state_validates_all_registries():
    _state().validate()


def test_mainline_state_rejects_anchor_identity_mismatch():
    state = _state()
    state.dynamic_outcomes["anchor_count"] = 3
    with pytest.raises(ValueError, match="dynamic outcomes"):
        state.validate()
