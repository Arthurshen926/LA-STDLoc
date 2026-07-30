from localization_training.pair_reliability import (
    PairOutcome,
    aggregate_pair_reliability,
    reliable_pair,
    wilson_lower_bound,
)


def test_wilson_lower_bound_is_conservative():
    assert wilson_lower_bound(0, 4) == 0.0
    assert 0.0 < wilson_lower_bound(3, 4) < 0.75
    assert wilson_lower_bound(4, 4) < 1.0


def test_pair_reliability_excludes_heldout_trajectory():
    outcomes = [
        PairOutcome("seq1", 4, 8, True),
        PairOutcome("seq2", 4, 8, True),
        PairOutcome("seq3", 4, 8, False),
        PairOutcome("seq1", 5, 9, True),
    ]
    result = aggregate_pair_reliability(
        outcomes, excluded_trajectory="seq3"
    )
    pair = result[(4, 8)]
    assert pair.attempts == 2
    assert pair.successes == 2
    assert pair.positive_trajectories == 2
    assert reliable_pair(
        pair,
        minimum_attempts=2,
        minimum_successes=2,
        minimum_positive_trajectories=2,
        minimum_precision=1.0,
        minimum_wilson_lower_bound=0.4,
    )


def test_unobserved_pair_always_falls_back():
    assert not reliable_pair(
        None,
        minimum_attempts=1,
        minimum_successes=1,
        minimum_positive_trajectories=1,
        minimum_precision=0.5,
        minimum_wilson_lower_bound=0.1,
    )
