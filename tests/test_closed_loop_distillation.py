from map_learning.closed_loop_distillation import accept_candidate


def _summary(**updates):
    value = dict(
        catastrophic_100cm_count=2,
        cvar95_te_cm=50.0,
        recall_5cm_5deg_percent=90.0,
        mean_te_cm=5.0,
        median_te_cm=1.0,
        anchor_count=100,
        online_latency_ms=10.0,
    )
    value.update(updates)
    return value


def test_recall_is_a_hard_guard_even_when_cvar_improves() -> None:
    decision = accept_candidate(
        _summary(),
        _summary(cvar95_te_cm=20.0, recall_5cm_5deg_percent=89.9),
        seen_state_hashes=set(), candidate_state_hash="candidate",
        maximum_anchor_count=200, maximum_online_latency_ms=20.0,
    )
    assert decision["accepted"] is False
    assert decision["reason"] == "hard_guard"


def test_repeated_state_is_rejected() -> None:
    decision = accept_candidate(
        _summary(), _summary(cvar95_te_cm=40.0),
        seen_state_hashes={"candidate"}, candidate_state_hash="candidate",
        maximum_anchor_count=200, maximum_online_latency_ms=20.0,
    )
    assert decision["reason"] == "repeated_state_hash"
