import pytest

from scripts.compare_rendered_track_artifact_stability import evaluate_mapping_gate


def _report(**summary):
    values = {
        "catastrophic_100cm_count": 2,
        "cvar95_te_cm": 100.0,
        "median_te_cm": 0.5,
        "p90_te_cm": 1.0,
        "query_count": 100,
        "raw_gt_precision_percent": 10.0,
        "recall_5cm_5deg_percent": 99.0,
    }
    values.update(summary)
    return {"summary": values}


def test_shopfacade_gate_accepts_exact_boundaries():
    baseline = _report()
    candidate = _report(
        median_te_cm=0.51,
        p90_te_cm=1.02,
        recall_5cm_5deg_percent=98.75,
    )
    result = evaluate_mapping_gate("shopfacade", baseline, candidate)
    assert result["passed"] is True
    assert all(result["gates"].values())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("median_te_cm", 0.510001),
        ("p90_te_cm", 1.020001),
        ("recall_5cm_5deg_percent", 98.7499),
        ("catastrophic_100cm_count", 3),
    ],
)
def test_shopfacade_gate_rejects_each_regression(field, value):
    result = evaluate_mapping_gate("shopfacade", _report(), _report(**{field: value}))
    assert result["passed"] is False


def test_stairs_gate_accepts_exact_boundaries():
    candidate = _report(raw_gt_precision_percent=9.95)
    result = evaluate_mapping_gate("stairs", _report(), candidate)
    assert result["passed"] is True
    assert all(result["gates"].values())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("p90_te_cm", 1.000001),
        ("cvar95_te_cm", 100.000001),
        ("raw_gt_precision_percent", 9.9499),
        ("catastrophic_100cm_count", 3),
    ],
)
def test_stairs_gate_rejects_each_regression(field, value):
    result = evaluate_mapping_gate("stairs", _report(), _report(**{field: value}))
    assert result["passed"] is False


def test_gate_rejects_query_count_or_unknown_scene():
    with pytest.raises(ValueError, match="query counts"):
        evaluate_mapping_gate("stairs", _report(), _report(query_count=99))
    with pytest.raises(ValueError, match="unsupported"):
        evaluate_mapping_gate("other", _report(), _report())
