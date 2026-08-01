import json

from scripts.summarize_joint_assignment_p1 import SCENES, summarize


def _row(name, te, hypotheses, assignment_ms=0.0, selector_ms=0.0):
    return {
        "image_name": name,
        "sparse_TE": te,
        "sparse_AE": 0.1,
        "sparse": {
            "sparse_diag_ransac_actual_hypotheses": hypotheses,
            "sparse_diag_native_rerank_runtime_ms": assignment_ms,
            "sparse_diag_joint_assignment_fixed_selector_runtime_ms": selector_ms,
        },
    }


def test_joint_assignment_gate_accepts_five_scene_joint_improvement(tmp_path):
    baseline = {}
    candidate = {}
    for scene in SCENES:
        baseline[scene] = tmp_path / f"{scene}_baseline.json"
        candidate[scene] = tmp_path / f"{scene}_candidate.json"
        baseline[scene].write_text(
            json.dumps([_row("query.png", 4.0, 1000)])
        )
        candidate[scene].write_text(
            json.dumps([_row("query.png", 3.0, 400, 2.0, 1.0)])
        )
    report = summarize(baseline, candidate, "S512-PoseSufficient")
    assert report["status"] == "PASS"
    assert report["macro"]["hypothesis_reduction_fraction"] == 0.6
    json.dumps(report)
