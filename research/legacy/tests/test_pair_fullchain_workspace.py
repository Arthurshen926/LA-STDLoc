import json

import pytest

from scripts.pair_fullchain_workspace import (
    _validate_semantic_input,
    lock_inputs,
    preflight_workspace,
    verify_stage_manifest,
    write_stage_manifest,
)


def test_lock_inputs_requires_preregistered_hashes_and_verifies_them(tmp_path):
    root = tmp_path / "run"
    preflight = root / "contracts" / "preflight.json"
    preflight.parent.mkdir(parents=True)
    preflight.write_text(json.dumps({"valid": True}))
    external = tmp_path / "query.pt"
    external.write_bytes(b"query")
    import hashlib

    digest = hashlib.sha256(b"query").hexdigest()
    report_path = root / "contracts" / "inputs.json"
    report = lock_inputs(
        root=root,
        inputs={"query_cache": external},
        expected_sha256={"query_cache": digest},
        parent=preflight,
        report_path=report_path,
    )
    assert report["inputs"]["query_cache"]["expected_sha256"] == digest
    assert report["inputs"]["query_cache"]["inside_output_root"] is False
    assert verify_stage_manifest(report_path)["valid"]
    with pytest.raises(ValueError, match="differs"):
        lock_inputs(
            root=root,
            inputs={"query_cache": external},
            expected_sha256={"query_cache": "0" * 64},
            parent=preflight,
            report_path=report_path,
        )


def test_preflight_accepts_only_declared_contract_inputs(tmp_path):
    root = tmp_path / "run"
    calibration = root / "inputs" / "calibration.json"
    calibration.parent.mkdir(parents=True)
    calibration.write_text("{}")
    report = preflight_workspace(
        root=root,
        allowed_inputs=[calibration],
        report_path=root / "contracts" / "preflight.json",
    )
    assert report["valid"]
    assert report["scientific_artifact_count"] == 0


def test_preflight_rejects_old_graph_or_teacher(tmp_path):
    root = tmp_path / "run"
    stale = root / "evidence" / "function_graph.pt"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old graph")
    with pytest.raises(RuntimeError, match="not empty/contract-only"):
        preflight_workspace(
            root=root,
            allowed_inputs=[],
            report_path=root / "contracts" / "preflight.json",
        )


def test_stage_manifest_hashes_in_root_artifacts_and_valid_parents(tmp_path):
    root = tmp_path / "run"
    artifact = root / "evidence" / "canonical.pt"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"canonical")
    parent = root / "contracts" / "preflight.json"
    parent.parent.mkdir(parents=True)
    parent.write_text(json.dumps({"valid": True}))
    report = write_stage_manifest(
        root=root,
        stage="canonical_evidence",
        artifacts={"canonical_map": artifact},
        parents=[parent],
        report_path=root / "contracts" / "evidence.json",
    )
    assert report["valid"]
    assert report["silent_resume_authorized"] is False
    assert report["artifacts"]["canonical_map"]["size_bytes"] == 9
    assert verify_stage_manifest(root / "contracts" / "evidence.json")["valid"]
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs from manifest"):
        verify_stage_manifest(root / "contracts" / "evidence.json")


def test_stage_manifest_rejects_artifact_outside_run(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    outside = tmp_path / "old_teacher.pt"
    outside.write_bytes(b"old")
    with pytest.raises(ValueError, match="escapes output root"):
        write_stage_manifest(
            root=root,
            stage="canonical_evidence",
            artifacts={"teacher": outside},
            parents=[],
            report_path=root / "contracts" / "evidence.json",
        )


def test_lock_inputs_rejects_failed_mechanism_gate(tmp_path):
    root = tmp_path / "run"
    preflight = root / "contracts" / "preflight.json"
    preflight.parent.mkdir(parents=True)
    preflight.write_text(json.dumps({"valid": True}))
    gate = tmp_path / "mechanism.json"
    gate.write_text(
        json.dumps(
            {
                "schema": "lafgs_pair_policy_mechanism_gate",
                "version": 2,
                "uses_test_queries": False,
                "valid": True,
                "mechanism_gate_passed": False,
                "advance_to_full_pipeline_pose": False,
                "decision": "STOP_BEFORE_PIPELINE",
                "gates": {"quality": False},
            }
        )
    )
    import hashlib

    digest = hashlib.sha256(gate.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="does not authorize"):
        lock_inputs(
            root=root,
            inputs={"mechanism_gate": gate},
            expected_sha256={"mechanism_gate": digest},
            parent=preflight,
            report_path=root / "contracts" / "inputs.json",
        )


@pytest.mark.parametrize(
    "schema",
    [
        "lafgs_cycle_verified_fisher_coverage_mechanism_gate",
        "lafgs_cycle_verified_fisher_coverage_cross_scene_stage_b_gate",
    ],
)
def test_existing_fullchain_rejects_coverage_v2_stage_b_schemas(tmp_path, schema):
    gate = tmp_path / "coverage_v2_gate.json"
    gate.write_text(
        json.dumps(
            {
                "schema": schema,
                "version": 1,
                "uses_test_queries": False,
                "valid": True,
                "scene_specific_mechanism_pass": True,
                "both_scene_stage_b_passed": True,
                "decision": "GO_TO_V2_AWARE_FULLCHAIN_LINEAGE_IMPLEMENTATION",
            }
        )
    )
    with pytest.raises(ValueError, match="does not authorize fullchain"):
        _validate_semantic_input("mechanism_gate", gate)
