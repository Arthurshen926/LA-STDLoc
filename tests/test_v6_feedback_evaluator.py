from map_learning.v6_feedback_evaluator import _maximum_matching


def test_maximum_matching_prevents_duplicate_anchor_votes() -> None:
    count, pairs = _maximum_matching([[0, 1], [0], [1, 2]])
    assert count == 3
    assert len({anchor for _, anchor in pairs}) == 3
