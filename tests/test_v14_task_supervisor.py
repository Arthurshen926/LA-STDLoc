from map_learning.v14_task_supervisor import summarize, supervise


def _row(index, before, after, *, before_ok=True, after_ok=True):
    def arm(task, ok):
        return {
            "task_error": task,
            "translation_error_cm": 1.0 if ok else 200.0,
            "rotation_error_deg": 1.0 if ok else 40.0,
        }
    return {
        "query_index": index,
        "pose_family_id": index,
        "baseline": arm(before, before_ok),
        "candidate": arm(after, after_ok),
    }


def test_task_risk_caps_catastrophic_magnitude():
    small = [_row(0, 100.0, 10.0, before_ok=False, after_ok=False)]
    huge = [_row(0, 10000.0, 10.0, before_ok=False, after_ok=False)]
    assert summarize(small, "baseline")["total_risk"] == summarize(huge, "baseline")["total_risk"]


def test_supervisor_advances_clear_safe_gain():
    records = [_row(index, 1.0, 0.5) for index in range(20)]
    decision = supervise(records, "candidate", samples=100, seed=3)
    assert decision["classification"] == "DEFAULT_CANDIDATE"


def test_supervisor_rejects_lost_success_even_with_tail_gain():
    records = [_row(index, 1.0, 0.5) for index in range(20)]
    records[0] = _row(0, 4.0, 0.1, before_ok=True, after_ok=False)
    decision = supervise(records, "candidate", samples=100, seed=3)
    assert decision["classification"] == "NO_ACTION"
