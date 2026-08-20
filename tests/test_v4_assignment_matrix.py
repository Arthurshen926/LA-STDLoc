import pytest

from scripts.run_v4_assignment_mapping_matrix import VARIANTS, _jobs
from scripts.summarize_v4_assignment_mapping_matrix import _summary


def test_assignment_matrix_expands_every_scene_with_shared_variants():
    scenes = {
        f"family/scene{index:02d}": {
            "status": "done",
            "output": f"/tmp/scene{index:02d}",
        }
        for index in range(24)
    }
    jobs = _jobs({"scenes": scenes}, None)
    assert len(jobs) == 24 * len(VARIANTS)
    assert {job["variant"] for job in jobs} == set(VARIANTS)
    assert all(job["scene_key"].startswith("family/") for job in jobs)


def test_assignment_matrix_refuses_partial_source_matrix():
    with pytest.raises(ValueError, match="24 completed"):
        _jobs(
            {"scenes": {"family/scene": {"status": "done", "output": "/tmp"}}},
            None,
        )


def test_assignment_summary_uses_query_weighted_tail_and_diagnostics():
    rows = [
        {
            "te_cm": 1.0,
            "ae_deg": 1.0,
            "correspondences": 8,
            "assignment_unmatched_queries": 2,
            "assignment_reassigned_queries": 3,
            "assignment_top1_collisions": 4,
        },
        {
            "te_cm": 101.0,
            "ae_deg": 1.0,
            "correspondences": 6,
            "assignment_unmatched_queries": 1,
            "assignment_reassigned_queries": 2,
            "assignment_top1_collisions": 3,
        },
    ]
    summary = _summary(rows)
    assert summary["mean_te_cm"] == 51.0
    assert summary["cvar95_te_cm"] == 101.0
    assert summary["recall_5cm_5deg_percent"] == 50.0
    assert summary["catastrophic_100cm_count"] == 1
    assert summary["mean_correspondences"] == 7.0
    assert summary["assignment_unmatched_query_rows"] == 3
    assert summary["assignment_reassigned_query_rows"] == 5
    assert summary["assignment_top1_collisions"] == 7
