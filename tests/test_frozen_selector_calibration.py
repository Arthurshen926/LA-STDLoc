from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import copy
import json

import pytest

from common.calibration import AdaptiveParameters, _canonical_sha256
from common.hashing import sha256_file
from map_learning import pipeline
from topology import adaptive_distillation


_INTEGER_PARAMETERS = {
    "matching_rows_target",
    "stage_a_steps",
    "metric_steps",
    "view_bin_count",
    "trajectory_bin_count",
}


def _parameters() -> dict:
    return {
        field.name: (1 if field.name in _INTEGER_PARAMETERS else 1.0)
        for field in fields(AdaptiveParameters)
    }


def _frozen_contract(tmp_path):
    query = tmp_path / "query_cache.pt"
    payload = tmp_path / "variant_tracks.pt"
    query.write_bytes(b"frozen query")
    payload.write_bytes(b"variant tracks")
    policy = {"matching_rows_fraction": 0.06}
    parent = {
        "schema": "lafgs_mapping_only_scene_calibration",
        "version": 2,
        "statistics": {"query_count": 2000, "track_count": 105782},
        "parameters": _parameters(),
        "policy": policy,
        "sources": {
            "query_cache": str(query.resolve()),
            "track_payload": str((tmp_path / "control_tracks.pt").resolve()),
            "uses_test_queries": False,
        },
    }
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(parent))
    frozen = {
        **parent,
        "uses_test_queries": False,
        "sources": {
            "query_cache": str(query.resolve()),
            "query_cache_sha256": sha256_file(query),
            "track_payload": str(payload.resolve()),
            "track_payload_sha256": sha256_file(payload),
            "uses_test_queries": False,
        },
        "lineage": {
            "mode": "frozen_numeric_pair_factor",
            "parent_calibration": str(parent_path.resolve()),
            "parent_calibration_sha256": sha256_file(parent_path),
            "expected_parent_calibration_sha256": sha256_file(parent_path),
            "statistics_reused_from_parent": True,
            "parameters_reused_from_parent": True,
            "parameters_sha256": _canonical_sha256(parent["parameters"]),
            "policy_sha256": _canonical_sha256(parent["policy"]),
            "uses_test_queries": False,
        },
    }
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text(json.dumps(frozen))
    return query, payload, policy, parent_path, frozen_path, frozen


def test_frozen_selector_calibration_is_not_reestimated(tmp_path, monkeypatch):
    query, payload, policy, _, frozen_path, frozen = _frozen_contract(tmp_path)

    def reject_derivation(*args, **kwargs):
        raise AssertionError("frozen selector calibration must not be re-estimated")

    monkeypatch.setattr(
        adaptive_distillation, "derive_mapping_statistics", reject_derivation
    )
    monkeypatch.setattr(
        adaptive_distillation, "derive_adaptive_parameters", reject_derivation
    )
    parameters, calibration = adaptive_distillation._resolve_selector_calibration(
        query_path=query,
        payload_path=payload,
        query_payload={},
        payload={},
        policy=policy,
        frozen_path=frozen_path,
        expected_frozen_sha256=sha256_file(frozen_path),
    )
    assert asdict(parameters) == frozen["parameters"]
    assert calibration == frozen


def test_evidence_and_metric_consumers_accept_only_exact_variant_binding(tmp_path):
    query, payload, policy, _, frozen_path, frozen = _frozen_contract(tmp_path)
    assert pipeline._read_exact_scene_calibration(
        query_cache=query,
        track_payload=payload,
        policy=policy,
        cached_path=frozen_path,
    ) == frozen
    payload.write_bytes(b"changed tracks")
    with pytest.raises(ValueError, match="Track-payload SHA"):
        pipeline._read_exact_scene_calibration(
            query_cache=query,
            track_payload=payload,
            policy=policy,
            cached_path=frozen_path,
        )


def test_selector_requires_explicit_frozen_calibration_sha(tmp_path):
    query, payload, policy, _, frozen_path, _ = _frozen_contract(tmp_path)
    with pytest.raises(ValueError, match="requires its expected SHA"):
        adaptive_distillation._resolve_selector_calibration(
            query_path=query,
            payload_path=payload,
            query_payload={},
            payload={},
            policy=policy,
            frozen_path=frozen_path,
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda value: value["sources"].__setitem__(
                "track_payload_sha256", "0" * 64
            ),
            "Track-payload SHA",
        ),
        (
            lambda value: value["lineage"].__setitem__(
                "parent_calibration_sha256", "0" * 64
            ),
            "parent lineage",
        ),
        (
            lambda value: value["parameters"].__setitem__("metric_steps", 1519),
            "parameters differ",
        ),
        (
            lambda value: value["statistics"].__setitem__("track_count", 1),
            "statistics differ",
        ),
    ],
)
def test_frozen_selector_calibration_fails_closed(tmp_path, mutation, match):
    query, payload, policy, _, frozen_path, frozen = _frozen_contract(tmp_path)
    frozen = copy.deepcopy(frozen)
    mutation(frozen)
    frozen_path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match=match):
        adaptive_distillation._resolve_selector_calibration(
            query_path=query,
            payload_path=payload,
            query_payload={},
            payload={},
            policy=policy,
            frozen_path=frozen_path,
            expected_frozen_sha256=sha256_file(frozen_path),
        )


def test_default_selector_calibration_remains_adaptive(tmp_path, monkeypatch):
    query = tmp_path / "query.pt"
    payload = tmp_path / "payload.pt"
    expected = AdaptiveParameters(**_parameters())
    @dataclass
    class Statistics:
        query_count: int = 1

    statistics = Statistics()
    monkeypatch.setattr(
        adaptive_distillation, "derive_mapping_statistics", lambda *a, **k: statistics
    )
    monkeypatch.setattr(
        adaptive_distillation, "derive_adaptive_parameters", lambda *a, **k: expected
    )
    parameters, calibration = adaptive_distillation._resolve_selector_calibration(
        query_path=query,
        payload_path=payload,
        query_payload={},
        payload={},
        policy={"matching_rows_fraction": 0.06},
        frozen_path=None,
    )
    assert parameters is expected
    assert calibration["sources"]["query_cache"] == str(query)
    assert calibration["sources"]["track_payload"] == str(payload)
    assert calibration["sources"]["uses_test_queries"] is False
