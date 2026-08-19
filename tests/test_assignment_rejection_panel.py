import json
from pathlib import Path

import pytest
import torch

from scripts import summarize_v4_assignment_rejection_panel as panel


def _write_report(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    statistics = path.with_suffix(".pt")
    torch.save({"uses_test_queries": False, "queries": rows}, statistics)
    path.write_text(
        json.dumps(
            {
                "uses_test_queries": False,
                "statistics": str(statistics.resolve()),
            }
        )
    )


def test_panel_summary_uses_query_level_tail_metrics():
    rows = [
        {"te_cm": 0.1, "ae_deg": 0.1},
        {"te_cm": 10.0, "ae_deg": 0.1},
        {"te_cm": 200.0, "ae_deg": 20.0},
    ]
    result = panel.summary(rows)
    assert result["query_count"] == 3
    assert result["median_te_cm"] == 10.0
    assert result["cvar95_te_cm"] == 200.0
    assert result["catastrophic_100cm_count"] == 1
    assert result["recall_5cm_5deg_percent"] == pytest.approx(100.0 / 3.0)


def test_panel_report_loader_rejects_test_scope(tmp_path):
    report = tmp_path / "report.json"
    _write_report(report, [{"te_cm": 1.0, "ae_deg": 1.0}])
    assert panel.load_queries(report)[0]["te_cm"] == 1.0
    payload = json.loads(report.read_text())
    payload["uses_test_queries"] = True
    report.write_text(json.dumps(payload))
    try:
        panel.load_queries(report)
    except ValueError as error:
        assert "test queries" in str(error)
    else:
        raise AssertionError("test-scoped panel report was accepted")
