import json

import pytest

from scripts.pair_fullchain_workspace import (
    preflight_workspace,
    verify_stage_manifest,
    write_stage_manifest,
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
