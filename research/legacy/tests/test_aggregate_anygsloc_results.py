from scripts.aggregate_anygsloc_results import percentile


def test_percentile_uses_linear_interpolation():
    assert percentile([0.0, 10.0], 0.9) == 9.0
    assert percentile([3.0], 0.9) == 3.0
