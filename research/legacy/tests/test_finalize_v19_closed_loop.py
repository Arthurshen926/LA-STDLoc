from argparse import Namespace
import json

import pytest
import torch

from scripts.finalize_v19_closed_loop import finalize_closed_loop


def _save_torch(path, payload):
    torch.save(payload, path)
    return path


def _arguments(tmp_path, *, classification="ANALYSIS_ONLY", candidate=False):
    teacher = _save_torch(
        tmp_path / "teacher.pt",
        {
            "schema": "lafgs_v19_track_extension_teacher_validation",
            "selection_uses_validation": False,
            "selected_tiers": {
                "tier_a": {"authorized_actions": []},
                "tier_b": {"authorized_actions": []},
                "tier_c": {"authorized_actions": ["soft_diagnostic"]},
            },
        },
    )
    full = _save_torch(
        tmp_path / "full.pt",
        {
            "schema": "lafgs_v19_full_pool_sufficiency_audit",
            "uses_test_queries": False,
            "candidate_pool_deficit_authorized": False,
            "totals": {"certified_candidate_pool_deficit": 0},
        },
    )
    compressed = _save_torch(
        tmp_path / "compressed.pt",
        {
            "schema": "lafgs_v19_full_pool_sufficiency_audit",
            "uses_test_queries": False,
            "totals": {"active_map_selection_loss_query_count": 4},
        },
    )
    control = tmp_path / "control.json"
    control.write_text(
        json.dumps(
            {
                "phase": "control",
                "uses_test_queries": False,
                "selected_arm": "alpha_0p25",
            }
        )
    )
    confirmation = tmp_path / "confirmation.json"
    confirmation.write_text(
        json.dumps(
            {
                "phase": "confirmation",
                "uses_test_queries": False,
                "selected_arm": "alpha_0p25",
                "decisions": {
                    "alpha_0p25": {
                        "classification": classification,
                        "hard_safety": {"passed": True},
                        "paired_effect": {"net_gain": 0.5},
                        "bootstrap": {
                            "probability_candidate_lower_risk": 0.99
                        },
                    }
                },
            }
        )
    )
    stable = tmp_path / "stable.pt"
    stable.write_bytes(b"stable-map")
    candidate_path = None
    if candidate:
        candidate_path = _save_torch(
            tmp_path / "candidate.pt",
            {
                "schema": "lafgs_shared_metric_state",
                "loo_used": False,
                "deployment_arm": "alpha_0p25",
            },
        )
    return Namespace(
        teacher_validation=teacher,
        full_pool_audit=full,
        compressed_pool_audit=compressed,
        metric_control=control,
        metric_confirmation=confirmation,
        stable_map=stable,
        candidate_metric=candidate_path,
        output=tmp_path / "decision.json",
    )


def test_v19_finalizer_retains_identity_when_confirmation_rejects(tmp_path) -> None:
    result = finalize_closed_loop(_arguments(tmp_path))
    assert result["deployment"]["metric"] == "identity"
    assert result["deployment"]["metric_sha256"] is None
    assert result["controller"]["mapping_identity_metric_authorized"] is False


def test_v19_finalizer_deploys_exact_confirmed_metric(tmp_path) -> None:
    args = _arguments(
        tmp_path, classification="DEFAULT_CANDIDATE", candidate=True
    )
    result = finalize_closed_loop(args)
    assert result["deployment"]["metric"] == str(args.candidate_metric.resolve())
    assert result["deployment"]["metric_sha256"]
    assert result["deployment"]["reason"] == "confirmed_metric_deployed"


def test_v19_finalizer_rejects_confirmed_metric_without_artifact(tmp_path) -> None:
    args = _arguments(tmp_path, classification="DEFAULT_CANDIDATE")
    with pytest.raises(ValueError, match="exact candidate metric artifact"):
        finalize_closed_loop(args)
