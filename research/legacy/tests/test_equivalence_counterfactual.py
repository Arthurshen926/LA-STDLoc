import pytest

from topology.equivalence_counterfactual import (
    aggregate_identity_folding_summaries,
    summarize_identity_folding,
)


def _variant(*, correspondences, correct, inliers, clean, te, ae):
    return {
        "correspondence_count": correspondences,
        "unique_anchor_count": correspondences,
        "unique_entity_count": correspondences,
        "labeled_row_count": correspondences,
        "correct_winner_count": correct,
        "ambiguous_winner_count": 0,
        "false_winner_count": correspondences - correct,
        "inlier_count": inliers,
        "clean_inlier_count": clean,
        "harmful_inlier_count": inliers - clean,
        "te_cm": te,
        "ae_deg": ae,
        "failed": inliers < 4,
        "hypotheses": 100,
    }


def test_counterfactual_summary_routes_safe_harm_reduction_forward():
    rows = [
        {
            "baseline": _variant(
                correspondences=10, correct=8, inliers=6, clean=4, te=3.0, ae=2.0
            ),
            "anchor_unique_control": _variant(
                correspondences=9, correct=8, inliers=5, clean=4, te=2.5, ae=1.8
            ),
            "entity_folded": _variant(
                correspondences=8, correct=8, inliers=4, clean=4, te=2.0, ae=1.5
            ),
            "duplicate_anchor_correspondence_removed_count": 1,
            "component_correspondence_removed_count": 1,
        },
        {
            "baseline": _variant(
                correspondences=10, correct=8, inliers=6, clean=5, te=4.0, ae=2.0
            ),
            "anchor_unique_control": _variant(
                correspondences=10, correct=8, inliers=6, clean=5, te=5.0, ae=2.0
            ),
            "entity_folded": _variant(
                correspondences=10, correct=8, inliers=6, clean=5, te=4.0, ae=2.0
            ),
            "duplicate_anchor_correspondence_removed_count": 0,
            "component_correspondence_removed_count": 0,
        },
    ]
    summary = summarize_identity_folding(rows)
    assert summary["paired"]["correspondence_removed_count"] == 1
    assert summary["paired"]["harmful_inlier_delta"] == -1
    assert summary["paired"]["raw_gt_precision_delta_pp"] > 0
    assert summary["routing"] == "go_to_evidence_transfer_prototype"
    assert not summary["physical_map_mutated"]


def test_counterfactual_summary_stops_when_precision_declines():
    rows = [
        {
            "baseline": _variant(
                correspondences=10, correct=9, inliers=6, clean=5, te=2.0, ae=1.0
            ),
            "anchor_unique_control": _variant(
                correspondences=10, correct=9, inliers=6, clean=5, te=2.0, ae=1.0
            ),
            "entity_folded": _variant(
                correspondences=9, correct=7, inliers=5, clean=4, te=2.0, ae=1.0
            ),
            "duplicate_anchor_correspondence_removed_count": 0,
            "component_correspondence_removed_count": 1,
        }
    ]
    summary = summarize_identity_folding(rows)
    assert summary["paired"]["raw_gt_precision_delta_pp"] < 0
    assert summary["routing"] == "stop_physical_dedup_keep_semantic_components"
    assert summary["baseline"]["raw_gt_precision_percent"] == pytest.approx(90.0)


def test_counterfactual_summary_stops_when_recall_declines():
    rows = [
        {
            "baseline": _variant(
                correspondences=10, correct=9, inliers=6, clean=5, te=2.0, ae=1.0
            ),
            "anchor_unique_control": _variant(
                correspondences=10, correct=9, inliers=6, clean=5, te=4.0, ae=1.0
            ),
            "entity_folded": _variant(
                correspondences=9, correct=9, inliers=5, clean=5, te=6.0, ae=1.0
            ),
            "duplicate_anchor_correspondence_removed_count": 0,
            "component_correspondence_removed_count": 1,
        }
    ]
    summary = summarize_identity_folding(rows)
    assert summary["paired"]["recall_5cm_5deg_delta_pp"] < 0
    assert summary["routing"] == "stop_physical_dedup_keep_semantic_components"


def _report(seed, *, cvar_delta, recall_delta=0.0):
    paired = {
        "correspondence_removed_count": 1,
        "correspondence_removed_percent": 0.1,
        "raw_gt_precision_delta_pp": 0.01,
        "inlier_gt_precision_delta_pp": 0.01,
        "harmful_inlier_delta": -1,
        "clean_inlier_delta": 0,
        "median_te_delta_cm": 0.0,
        "mean_te_delta_cm": 0.01,
        "p90_te_delta_cm": 0.0,
        "cvar95_te_delta_cm": cvar_delta,
        "recall_5cm_5deg_delta_pp": recall_delta,
        "catastrophic_count_delta": 0,
        "failure_count_delta": 0,
    }
    return {
        "map_sha256": "same-map",
        "query_count": 96,
        "query_selection": "uniform_mapping_gate",
        "seed": seed,
        "summary": {"paired": paired, "routing": "inconclusive"},
    }


def test_seed_aggregate_stops_consistent_tail_regression():
    result = aggregate_identity_folding_summaries(
        [_report(1, cvar_delta=0.1), _report(2, cvar_delta=0.2)]
    )
    assert result["pose_safe_seed_count"] == 0
    assert result["consistent_tail_regression"]
    assert result["routing"] == "stop_physical_dedup_keep_semantic_components"


def test_seed_aggregate_rejects_different_query_subset():
    first = _report(1, cvar_delta=0.1)
    second = _report(2, cvar_delta=0.2)
    second["query_count"] = 95
    with pytest.raises(ValueError, match="query_count"):
        aggregate_identity_folding_summaries([first, second])
