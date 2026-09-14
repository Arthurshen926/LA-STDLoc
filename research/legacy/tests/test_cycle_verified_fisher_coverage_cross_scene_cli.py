import json

import pytest

from common.hashing import sha256_file
from scripts import aggregate_cycle_verified_fisher_coverage_stage_a as aggregate_cli
from scripts.cycle_verified_fisher_cli_common import (
    SCENE_CONTRACTS,
    V2_FROZEN_SOURCE_CONTRACTS,
)


def _write_scene_gate(tmp_path, monkeypatch, *, scene, passed):
    scene_contract = {
        "mapping_keypoints": 3,
        "nms_radius": 1,
        "pair_budget": 5,
        "candidate_pair_count": 6,
        "candidate_component_count": 1,
    }
    source_contract = {
        "query_count": 5,
        "query_names_sha256": scene[0] * 64,
        "query_cache_sha256": scene[-1] * 64,
        "mapping_scope_mode": "query_cache_explicit_mapping_only",
        "mapping_scope_equivalence_sha256": None,
        "proposals_sha256": "1" * 64,
        "proposals_content_sha256": "2" * 64,
        "probe_sha256": "3" * 64,
        "probe_content_sha256": "4" * 64,
    }
    monkeypatch.setitem(SCENE_CONTRACTS, scene, scene_contract)
    monkeypatch.setitem(V2_FROZEN_SOURCE_CONTRACTS, scene, source_contract)
    names = {
        "query_cache",
        "pair_proposals",
        "pair_match_probe",
        "verified_cycle_table",
        "pair_selection",
    }
    if scene == "stairs":
        names.add("stairs_v1_pair_selection")
    inputs = {}
    for name in sorted(names):
        path = tmp_path / f"{scene}_{name}.artifact"
        path.write_text(f"{scene}:{name}\n")
        inputs[name] = {"path": str(path), "sha256": sha256_file(path)}
    gates = {"exact_contract": True, "scientific_metric": bool(passed)}
    path = tmp_path / f"{scene}_stage_a.json"
    path.write_text(
        json.dumps(
            {
                "schema": aggregate_cli.PER_SCENE_SCHEMA,
                "version": 1,
                "uses_test_queries": False,
                "mapping_only": True,
                "valid": True,
                "policy": "cycle_verified_fisher_coverage",
                "scene_contract": {"scene": scene, **scene_contract},
                "frozen_source_contract": {"scene": scene, **source_contract},
                "verified_geometry_independently_rematerialized_exact": True,
                "gates": gates,
                "stage_a_passed": bool(passed),
                "requires_other_scene": True,
                "requires_v2_aware_track_lineage_implementation": True,
                "advance_to_reuse_only_track_build": False,
                "decision": (
                    "SCENE_STAGE_A_PASS_REQUIRES_OTHER_SCENE"
                    if passed
                    else "STOP_BEFORE_TRACK_REUSE"
                ),
                "inputs": inputs,
            },
            sort_keys=True,
        )
    )
    return path, inputs


def _arguments(stairs, greatcourt, output):
    return [
        "--stairs-stage-a-gate",
        str(stairs),
        "--expected-stairs-stage-a-gate-sha256",
        sha256_file(stairs),
        "--greatcourt-stage-a-gate",
        str(greatcourt),
        "--expected-greatcourt-stage-a-gate-sha256",
        sha256_file(greatcourt),
        "--output",
        str(output),
    ]


def test_cross_scene_stage_a_is_only_track_authority(tmp_path, monkeypatch):
    monkeypatch.setattr(aggregate_cli, "_replay_scene_gate", lambda gate: [])
    stairs, _ = _write_scene_gate(
        tmp_path, monkeypatch, scene="stairs", passed=True
    )
    greatcourt, _ = _write_scene_gate(
        tmp_path, monkeypatch, scene="greatcourt", passed=True
    )
    output = tmp_path / "cross_scene_go.json"
    aggregate_cli.main(_arguments(stairs, greatcourt, output))
    gate = json.loads(output.read_text())
    assert gate["both_scene_stage_a_passed"] is True
    assert gate["advance_to_v2_aware_reuse_only_track_build"] is True
    assert gate["authorizes_existing_v1_track_runner"] is False
    assert gate["decision"] == "GO_TO_V2_AWARE_REUSE_ONLY_TRACK_BUILD"


def test_cross_scene_stage_a_scientific_stop_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(aggregate_cli, "_replay_scene_gate", lambda gate: [])
    stairs, _ = _write_scene_gate(
        tmp_path, monkeypatch, scene="stairs", passed=True
    )
    greatcourt, _ = _write_scene_gate(
        tmp_path, monkeypatch, scene="greatcourt", passed=False
    )
    output = tmp_path / "cross_scene_stop.json"
    with pytest.raises(SystemExit) as error:
        aggregate_cli.entrypoint(_arguments(stairs, greatcourt, output))
    assert error.value.code == 2
    gate = json.loads(output.read_text())
    assert gate["both_scene_stage_a_passed"] is False
    assert gate["advance_to_v2_aware_reuse_only_track_build"] is False
    assert gate["decision"] == "STOP_BEFORE_TRACK_REUSE"


def test_cross_scene_stage_a_rejects_changed_referenced_input(tmp_path, monkeypatch):
    stairs, stairs_inputs = _write_scene_gate(
        tmp_path, monkeypatch, scene="stairs", passed=True
    )
    greatcourt, _ = _write_scene_gate(
        tmp_path, monkeypatch, scene="greatcourt", passed=True
    )
    output = tmp_path / "invalid.json"
    referenced = stairs_inputs["pair_match_probe"]["path"]
    with open(referenced, "a") as handle:
        handle.write("changed\n")
    with pytest.raises(SystemExit) as error:
        aggregate_cli.entrypoint(_arguments(stairs, greatcourt, output))
    assert error.value.code == 1
    assert not output.exists()


def test_cross_scene_stage_a_rehashes_gates_after_both_recursive_replays(
    tmp_path, monkeypatch
):
    stairs, _ = _write_scene_gate(
        tmp_path, monkeypatch, scene="stairs", passed=True
    )
    greatcourt, _ = _write_scene_gate(
        tmp_path, monkeypatch, scene="greatcourt", passed=True
    )
    calls = 0

    def mutate_first_gate_after_second_load(gate):
        nonlocal calls
        del gate
        calls += 1
        if calls == 2:
            stairs.write_text(stairs.read_text() + "\n")
        return []

    monkeypatch.setattr(
        aggregate_cli, "_replay_scene_gate", mutate_first_gate_after_second_load
    )
    output = tmp_path / "gate_changed_during_replay.json"
    with pytest.raises(SystemExit) as error:
        aggregate_cli.entrypoint(_arguments(stairs, greatcourt, output))
    assert error.value.code == 1
    assert not output.exists()


def test_cross_scene_rejects_self_signed_all_true_gate_without_real_artifacts(
    tmp_path, monkeypatch
):
    stairs, _ = _write_scene_gate(
        tmp_path, monkeypatch, scene="stairs", passed=True
    )
    greatcourt, _ = _write_scene_gate(
        tmp_path, monkeypatch, scene="greatcourt", passed=True
    )
    output = tmp_path / "forged_all_true.json"
    with pytest.raises(SystemExit) as error:
        aggregate_cli.entrypoint(_arguments(stairs, greatcourt, output))
    assert error.value.code == 1
    assert not output.exists()
