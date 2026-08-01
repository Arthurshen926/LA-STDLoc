import json

from scripts.summarize_joint_assignment_production_profile import summarize


def _row(name, *, matching, ransac, total, hypotheses, te, rerank=0.0, selector=0.0):
    return {
        "image_name": name,
        "sparse_TE": te,
        "sparse": {
            "sparse_diag_runtime_frontend_ms": 20.0,
            "sparse_diag_runtime_matching_ms": matching,
            "sparse_diag_runtime_ransac_ms": ransac,
            "sparse_diag_runtime_total_ms": total,
            "sparse_diag_ransac_actual_hypotheses": hypotheses,
            "sparse_diag_native_rerank_runtime_ms": rerank,
            "sparse_diag_joint_assignment_fixed_selector_runtime_ms": selector,
        },
    }


def test_production_profile_accounts_for_widened_retrieval(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(
        json.dumps([_row("q.png", matching=3, ransac=30, total=60, hypotheses=1000, te=4)])
    )
    candidate.write_text(
        json.dumps(
            [
                _row(
                    "q.png",
                    matching=9,
                    ransac=15,
                    total=51,
                    hypotheses=400,
                    te=3,
                    rerank=2,
                    selector=1,
                )
            ]
        )
    )
    report = summarize("Scene", "S512-PoseSufficient", baseline, candidate)
    assert report["paired_candidate_minus_baseline_ms"]["matching"]["p90"] == 6
    assert report["explicit_assignment_selection_ms"]["p90"] == 3
    assert report["hypotheses"]["reduction_fraction"] == 0.6
    assert all(report["checks"].values())
