import math

import pytest
import torch

from topology.alias_risk import (
    aggregate_group_alias_evidence,
    alias_risk_from_counters,
    crossfit_alias_separability,
    wilson_lower,
    wilson_upper,
)


def _record(winners, *, clean, solver, harmful):
    winners = torch.tensor(winners, dtype=torch.int32)
    top = torch.stack((winners, 1 - winners), dim=1)
    flags = torch.zeros_like(top, dtype=torch.uint8)
    flags[:, 0] = torch.tensor(clean, dtype=torch.uint8) * 12
    return {
        "top_indices": top,
        "legal_flags": flags,
        "solver_inlier": torch.tensor(solver),
        "harmful_solver_inlier": torch.tensor(harmful),
    }


def test_group_alias_evidence_treats_ambiguous_winner_as_not_false():
    record = _record(
        [0, 1], clean=[False, True], solver=[True, True], harmful=[True, False]
    )
    record["legal_flags"][0, 0] = 8
    counters = aggregate_group_alias_evidence(
        {"anchor_count": 2, "records": [record]}, torch.tensor([3])
    )
    assert counters["winner"].tolist() == [[1, 1]]
    assert counters["false"].tolist() == [[0, 0]]
    assert counters["harmful"].tolist() == [[1, 0]]


def test_wilson_upper_keeps_missing_evidence_unknown():
    result = wilson_upper(torch.tensor([0, 1]), torch.tensor([0, 2]))
    assert math.isnan(float(result[0]))
    assert 0.5 < float(result[1]) <= 1.0
    lower = wilson_lower(torch.tensor([0, 1]), torch.tensor([0, 2]))
    assert math.isnan(float(lower[0]))
    assert 0.0 < float(lower[1]) < 0.5


def test_crossfit_alias_risk_separates_recurrent_harm_from_clean():
    counters = {
        name: torch.zeros((3, 2), dtype=torch.long)
        for name in ("winner", "clean", "false", "solver_inlier", "harmful")
    }
    counters["winner"][:] = torch.tensor([10, 10])
    counters["solver_inlier"][:] = torch.tensor([5, 5])
    counters["false"][:] = torch.tensor([8, 0])
    counters["harmful"][:] = torch.tensor([4, 0])
    counters["clean"][:] = torch.tensor([0, 8])
    risk = alias_risk_from_counters(counters)
    report = crossfit_alias_separability(counters)
    assert risk["recurrent_alias"].tolist() == [True, False]
    assert report["false_vs_clean_auc"] == pytest.approx(1.0)
    assert report["harmful_vs_clean_auc"] == pytest.approx(1.0)
