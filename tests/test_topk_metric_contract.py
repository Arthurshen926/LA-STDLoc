from __future__ import annotations

import hashlib

import pytest
import torch

from scripts.build_lafgs_topk_outcomes import _verify_metric_contract


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_explicit_metric_contract_rejects_different_state(tmp_path):
    expected = tmp_path / "expected.pt"
    other = tmp_path / "other.pt"
    torch.save({"value": 1}, expected)
    torch.save({"value": 2}, other)
    state = {
        "metric_state_contract": {
            "path": str(expected),
            "sha256": _sha256(expected),
        }
    }

    _verify_metric_contract(
        state=state,
        metric_payload={},
        metric_path=str(expected),
    )
    with pytest.raises(ValueError, match="materialized map contract"):
        _verify_metric_contract(
            state=state,
            metric_payload={},
            metric_path=str(other),
        )


def test_legacy_metric_provenance_checks_experiment_directory(tmp_path):
    expected_dir = tmp_path / "metric_expected"
    other_dir = tmp_path / "metric_other"
    expected_dir.mkdir()
    other_dir.mkdir()
    expected = expected_dir / "metric.pt"
    other = other_dir / "metric.pt"
    torch.save({}, expected)
    torch.save({}, other)
    state = {
        "v7_online_metric": {
            "config": {"output_dir": str(expected_dir)}
        }
    }

    _verify_metric_contract(
        state=state,
        metric_payload={"map_path": str(expected_dir / "map.pt")},
        metric_path=str(expected),
    )
    with pytest.raises(ValueError, match="provenance"):
        _verify_metric_contract(
            state=state,
            metric_payload={"map_path": str(other_dir / "map.pt")},
            metric_path=str(other),
        )
