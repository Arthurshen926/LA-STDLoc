import torch

from scripts.audit_render_real_descriptor_gap import _binned, _summary


def test_binned_attribution_is_deterministic_and_covers_rows():
    factor = torch.arange(100).float()
    outcome = factor / 100 - 0.5
    cosine = 1 - factor / 200
    first = _binned(factor, outcome, cosine)
    second = _binned(factor, outcome, cosine)
    assert first == second
    assert sum(row["count"] for row in first) == 100
    assert first[0]["rgb_correct_score_gain_mean"] < first[-1]["rgb_correct_score_gain_mean"]


def test_summary_filters_nonfinite_values():
    result = _summary(torch.tensor([1.0, 3.0, float("nan"), float("inf")]))
    assert result["mean"] == 2.0
    assert result["median"] == 1.0
