import pytest
import torch
import torch.nn.functional as F

from map_learning.v20_feedback import build_topk_competition_evidence
from map_learning.v20_sparse_descriptor import (
    audit_materialized_sparse_action,
    train_sparse_anchor_descriptors,
)
from tests.test_v20_feedback import _record


def _inputs():
    anchors = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.8, 0.6, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ),
        dim=1,
    )
    left = _record(0, 10)
    right = _record(1, 20)
    for record in (left, right):
        record["query_descriptors"][0] = torch.tensor([0.75, 0.66, 0.0, 0.0])
        record["query_descriptors"][1] = anchors[2]
        scores = F.normalize(record["query_descriptors"], dim=1) @ anchors.T
        record["candidate_scores"] = torch.gather(
            scores, 1, record["candidate_anchor_rows"]
        )
    evidence = build_topk_competition_evidence(
        records=[left, right],
        anchor_count=3,
        equivalence_class_ids=torch.arange(3),
    )
    # An untouched non-unit row must remain byte/numerically exact in the map;
    # normalization belongs to scoring, not global map rewriting.
    anchors[2] *= 2.0
    return anchors, evidence


def test_sparse_positive_only_changes_only_truth_anchors_with_angle_bound() -> None:
    anchors, evidence = _inputs()
    updated, report = train_sparse_anchor_descriptors(
        anchor_features=anchors,
        evidence=evidence,
        mode="positive_only",
        maximum_angle_deg=20.0,
        steps=160,
        learning_rate=0.08,
        device="cpu",
    )
    assert report["selected_anchor_rows"].tolist() == [0]
    assert torch.equal(updated[1:], anchors[1:])
    assert report["maximum_observed_angle_deg"] <= 20.0 + 1e-4
    assert report["repair_margin_after_mean"] > report["repair_margin_before_mean"]
    assert report["query_descriptor_action"] == "native_unchanged"
    assert report["deployment_status"] == "ANALYSIS_ONLY_TEACHER_NOT_AUTHORIZED"
    assert report["post_training_action_scale"] == float(
        report["per_anchor_action_scales"].min()
    )
    assert (
        report["global_seed_action_scale"]
        <= report["post_training_action_scale"]
    )
    assert report["positive_win_nonregression_passed"] is True


def test_sparse_positive_negative_mode_remains_local_and_requires_pose_control() -> None:
    anchors, evidence = _inputs()
    evidence["strong_feedback_authorized"] = True
    updated, report = train_sparse_anchor_descriptors(
        anchor_features=anchors,
        evidence=evidence,
        mode="positive_and_repeated_negative",
        maximum_angle_deg=10.0,
        steps=120,
        learning_rate=0.05,
        device="cpu",
        strong_feedback_authorized=True,
    )
    assert report["selected_anchor_rows"].tolist() == [0, 1]
    assert torch.equal(updated[2], anchors[2])
    assert report["maximum_observed_angle_deg"] <= 10.0 + 1e-4
    assert report["deployment_status"] == "REQUIRES_EXACT_POSE_CONTROL"
    assert report["requires_exact_pose_control"] is True
    assert report["negative_action_anchor_count"] == 1


def test_sparse_negative_action_rejects_winner_without_clean_support() -> None:
    anchors, evidence = _inputs()
    evidence["strong_feedback_authorized"] = True
    evidence["repair_wrong_winner_clean_support_family_counts"] = torch.zeros_like(
        evidence["repair_wrong_winner_clean_support_family_counts"]
    )
    evidence["negative_action_anchor_rows"] = torch.empty(0, dtype=torch.long)
    with pytest.raises(ValueError, match="clean-positive-supported"):
        train_sparse_anchor_descriptors(
            anchor_features=anchors,
            evidence=evidence,
            mode="positive_and_repeated_negative",
            steps=1,
            device="cpu",
            strong_feedback_authorized=True,
        )


def test_multi_positive_objective_moves_the_weak_positive_too() -> None:
    anchors = F.normalize(
        torch.tensor(
            [
                [0.80, 0.60, 0.0],
                [0.60, 0.80, 0.0],
                [0.90, -0.4358899, 0.0],
            ]
        ),
        dim=1,
    )
    clean_query = anchors[0].reshape(1, -1)
    evidence = {
        "schema": "lafgs_v20_topk_competition_evidence",
        "version": 2,
        "uses_test_queries": False,
        "loo_used": False,
        "strong_feedback_authorized": False,
        "repair_query_descriptors": torch.tensor([[1.0, 0.0, 0.0]]),
        "repair_positive_offsets": torch.tensor([0, 2]),
        "repair_positive_anchor_rows": torch.tensor([0, 1]),
        "repair_negative_offsets": torch.tensor([0, 1]),
        "repair_negative_anchor_rows": torch.tensor([2]),
        "repair_wrong_winner_anchor_rows": torch.tensor([2]),
        "repair_wrong_winner_clean_support_family_counts": torch.tensor([0]),
        "negative_action_anchor_rows": torch.empty(0, dtype=torch.long),
        "minimum_negative_action_clean_pose_families": 2,
        "repair_sample_weights": torch.ones(1),
        "protection_query_descriptors": clean_query,
        "protection_positive_offsets": torch.tensor([0, 1]),
        "protection_positive_anchor_rows": torch.tensor([0]),
        "protection_negative_offsets": torch.tensor([0, 1]),
        "protection_negative_anchor_rows": torch.tensor([2]),
        "protection_initial_margin": torch.tensor(
            [float(clean_query @ anchors[0] - clean_query @ anchors[2])]
        ),
    }
    before = anchors[:2, 0].clone()
    updated, report = train_sparse_anchor_descriptors(
        anchor_features=anchors,
        evidence=evidence,
        mode="positive_only",
        maximum_angle_deg=25.0,
        steps=160,
        learning_rate=0.08,
        device="cpu",
    )
    assert torch.all(updated[:2, 0] > before)
    assert (
        report["repair_worst_positive_margin_after_mean"]
        > report["repair_worst_positive_margin_before_mean"]
    )
    assert report["positive_objective"] == "per_positive_listwise_mean"


def test_sparse_repair_rejects_negative_csr_anchor_rows() -> None:
    anchors, evidence = _inputs()
    evidence["repair_negative_anchor_rows"] = evidence[
        "repair_negative_anchor_rows"
    ].clone()
    evidence["repair_negative_anchor_rows"][0] = -1
    with pytest.raises(ValueError, match="invalid Anchor row"):
        train_sparse_anchor_descriptors(
            anchor_features=anchors,
            evidence=evidence,
            steps=1,
            device="cpu",
        )


def test_sparse_repair_rejects_zero_native_descriptor() -> None:
    anchors, evidence = _inputs()
    anchors[0].zero_()
    with pytest.raises(ValueError, match="finite rows"):
        train_sparse_anchor_descriptors(
            anchor_features=anchors,
            evidence=evidence,
            steps=1,
            device="cpu",
        )


def test_sparse_repair_rejects_unsealed_authorization_and_zero_temperature() -> None:
    anchors, evidence = _inputs()
    evidence["strong_feedback_authorized"] = True
    with pytest.raises(ValueError, match="authorization is not evidence-bound"):
        train_sparse_anchor_descriptors(
            anchor_features=anchors,
            evidence=evidence,
            steps=1,
            device="cpu",
        )
    evidence["strong_feedback_authorized"] = False
    with pytest.raises(ValueError, match="parameters must be positive"):
        train_sparse_anchor_descriptors(
            anchor_features=anchors,
            evidence=evidence,
            temperature=0.0,
            steps=1,
            device="cpu",
        )


def test_materialized_audit_rejects_exact_saved_dtype_clean_break() -> None:
    baseline = F.normalize(
        torch.tensor([[1.0, 0.0], [0.9999, 0.014]], dtype=torch.float16).float(),
        dim=1,
    ).half()
    candidate = baseline.clone()
    candidate[0] = torch.tensor([0.99, 0.10], dtype=torch.float16)
    evidence = {
        "schema": "lafgs_v20_topk_competition_evidence",
        "version": 2,
        "uses_test_queries": False,
        "loo_used": False,
        "protection_query_descriptors": torch.tensor([[1.0, 0.0]]),
        "protection_positive_offsets": torch.tensor([0, 1]),
        "protection_positive_anchor_rows": torch.tensor([0]),
        "protection_negative_offsets": torch.tensor([0, 1]),
        "protection_negative_anchor_rows": torch.tensor([1]),
    }

    audit = audit_materialized_sparse_action(
        baseline_anchor_features=baseline,
        candidate_anchor_features=candidate,
        selected_anchor_rows=torch.tensor([0]),
        evidence=evidence,
        clean_margin_slack=0.002,
        maximum_angle_deg=10.0,
    )

    assert audit["passed"] is False
    assert audit["broken_protection_row_count"] == 1


def test_sparse_action_preserves_float64_unselected_rows_bit_exact() -> None:
    anchors, evidence = _inputs()
    anchors = anchors.double()
    anchors[2, 0] = torch.nextafter(
        torch.tensor(0.1, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    )
    updated, _ = train_sparse_anchor_descriptors(
        anchor_features=anchors,
        evidence=evidence,
        mode="positive_only",
        maximum_angle_deg=20.0,
        steps=4,
        device="cpu",
    )

    assert updated.dtype == torch.float64
    assert torch.equal(updated[1:], anchors[1:])
