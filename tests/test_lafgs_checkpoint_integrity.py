from pathlib import Path
from types import SimpleNamespace

import pytest

from train_lafgs_map import (
    _checkpoint_integrity,
    _history_windows,
    _native_auxiliary_contract,
    _validate_native_objective_semantics,
)


def test_checkpoint_integrity_reports_missing_intermediate_checkpoint(tmp_path):
    for step in (0, 500, 1500):
        (Path(tmp_path) / f"{step}_lafgs_map_state.pt").touch()

    report = _checkpoint_integrity(tmp_path, [0, 500, 1000, 1500, 1500])

    assert report == {
        "requested_steps": [0, 500, 1000, 1500],
        "saved_steps": [0, 500, 1500],
        "missing_steps": [1000],
        "complete": False,
    }


def test_stable_gradient_clipping_recovers_float32_norm_overflow():
    import torch

    from train_lafgs_map import _stable_clip_grad_norm

    parameter = torch.nn.Parameter(torch.zeros(2))
    # The individual entries are finite, but their float32 squared sum is not.
    parameter.grad = torch.tensor([1e30, -1e30], dtype=torch.float32)

    norm, clipped = _stable_clip_grad_norm([parameter], max_norm=10.0)

    assert torch.isfinite(norm)
    assert clipped
    assert torch.linalg.vector_norm(parameter.grad).item() <= 10.001


def test_history_windows_cover_full_course_with_compact_contiguous_means():
    records = [
        {"step": 10, "loss": 9.0, "retrieval_native_keep_count": 1},
        {"step": 20, "loss": 7.0, "retrieval_native_keep_count": 2},
        {"step": 30, "loss": 5.0, "retrieval_native_keep_count": 3},
        {"step": 40, "loss": 3.0, "retrieval_native_keep_count": 4},
        {"step": 50, "loss": 1.0, "retrieval_native_keep_count": 5},
    ]

    windows = _history_windows(records, window_count=3)

    assert [(item["start_step"], item["end_step"], item["record_count"]) for item in windows] == [
        (10, 10, 1),
        (20, 30, 2),
        (40, 50, 2),
    ]
    assert windows[0]["diagnostics"]["loss"] == 9.0
    assert windows[1]["diagnostics"]["loss"] == 6.0
    assert windows[2]["diagnostics"]["retrieval_native_keep_count"] == 4.5


def _native_args(**overrides):
    values = {
        "observation_source": "native",
        "native_anchor_aux_weight": 0.0,
        "mv_weight": 0.0,
        "local_weight": 0.0,
        "dustbin_weight": 0.0,
        "retrieval_weight": 1.0,
        "trust_weight": 0.02,
        "native_outcome_mode": True,
        "objective": "hard",
        "native_sampling_mode": "detector_grid",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_pure_native_contract_records_effective_zero_anchor_losses():
    args = _native_args()

    _validate_native_objective_semantics(args)
    contract = _native_auxiliary_contract(args)

    assert contract["pure_native"] is True
    assert contract["effective_anchor_weights"] == {
        "mv": 0.0,
        "local": 0.0,
        "dustbin": 0.0,
    }
    assert contract["effective_trust_weight"] == 0.02


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"native_anchor_aux_weight": 0.1}, "native_anchor_aux_weight"),
        ({"mv_weight": 0.1}, "mv_weight"),
        ({"local_weight": 0.1}, "local_weight"),
        ({"dustbin_weight": 0.1}, "dustbin_weight"),
        ({"objective": "random"}, "requires --objective hard"),
        ({"native_sampling_mode": "label_balanced"}, "detector_grid"),
        (
            {"native_outcome_mode": False, "retrieval_weight": 1.0},
            "no deployment-aligned descriptor objective",
        ),
    ],
)
def test_native_objective_contract_rejects_inert_or_nondeployment_settings(
    override, message
):
    with pytest.raises(ValueError, match=message):
        _validate_native_objective_semantics(_native_args(**override))


def test_native_plus_anchor_requires_an_effective_anchor_loss():
    args = _native_args(
        observation_source="native_plus_anchor",
        native_anchor_aux_weight=0.05,
        mv_weight=0.0,
        local_weight=0.0,
        dustbin_weight=0.0,
    )

    with pytest.raises(ValueError, match="requires at least one nonzero anchor loss"):
        _validate_native_objective_semantics(args)


def test_formal_ulf_runner_defaults_to_joint_safety_checkpoint_selection():
    runner = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_lafgs_v2_ulfparity_alternating.sh"
    ).read_text()

    assert 'SELECTION_MODE="${LAFGS_ULF_SELECTION_MODE:-safety}"' in runner
    assert '--selection_mode "$SELECTION_MODE"' in runner
    assert '"joint_clean_pose_gate_required": selection_mode == "safety"' in runner
