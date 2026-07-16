import json

from scripts.select_candidate_checkpoint import (
    linear_quantile,
    select_checkpoint,
)


def write_results(root, tag, iteration, values):
    result_dir = root / f"{tag}_{iteration}_calibrated_validation"
    result_dir.mkdir(parents=True)
    (result_dir / "results.json").write_text(
        json.dumps([{"sparse_TE": value} for value in values])
    )


def test_linear_quantile_matches_interpolated_percentile():
    assert linear_quantile([0.0, 10.0, 20.0], 0.75) == 15.0


def test_selection_uses_validation_median_before_tail_metrics(tmp_path):
    tag = "candidate_f0"
    write_results(tmp_path, tag, 100, [1.0, 2.0, 100.0])
    write_results(tmp_path, tag, 200, [2.1, 2.1, 2.1])
    write_results(tmp_path, tag, 300, [3.0, 3.0, 3.0])

    report = select_checkpoint(tmp_path, tag, [100, 200, 300])

    assert report["selected_iteration"] == 100
    assert report["selection_protocol"]["test_metrics_used"] is False
    assert report["selected_metrics"]["median_te_cm"] == 2.0


def test_selection_uses_mean_as_first_tie_breaker(tmp_path):
    tag = "candidate_f0"
    write_results(tmp_path, tag, 100, [1.0, 2.0, 3.0])
    write_results(tmp_path, tag, 200, [0.0, 2.0, 2.5])

    report = select_checkpoint(tmp_path, tag, [100, 200])

    assert report["selected_iteration"] == 200
