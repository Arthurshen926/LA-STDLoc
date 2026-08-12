import json
from pathlib import Path

import pytest

from scripts.audit_equal_energy_pose_postmortem import (
    _json_sha256,
    _load_locked_protocol,
    _sha256,
    _validate_reproduction,
    _validate_source_summary,
)


def _record(path: Path) -> dict:
    digest = _sha256(path)
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "expected_sha256": digest,
        "expected_sha256_matches": True,
    }


def _locked_gate_fixture(tmp_path: Path):
    arm_paths = {}
    summary_roots = {}
    inputs = {}
    for local_arm, gate_arm in (("baseline", "baseline"), ("candidate", "variant")):
        arm_paths[local_arm] = {}
        for local_role, gate_role in (
            ("map", "map"),
            ("metric", "metric"),
            ("teacher", "teacher"),
            ("cache", "query_cache"),
        ):
            path = tmp_path / f"{local_arm}_{local_role}.pt"
            path.write_bytes(f"{local_arm}:{local_role}".encode())
            arm_paths[local_arm][local_role] = path
            inputs[f"{gate_arm}.{gate_role}"] = _record(path)
        calibration = tmp_path / f"{local_arm}_calibration.json"
        calibration.write_text("{}\n")
        inputs[f"{gate_arm}.calibration"] = _record(calibration)
        summary_root = tmp_path / f"{local_arm}_summaries"
        summary_roots[local_arm] = summary_root
        for seed in (11, 12, 13):
            summary_path = summary_root / f"seed{seed}" / "mapping_cache_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text("{}\n")
            inputs[f"{gate_arm}.seed{seed}_summary"] = _record(summary_path)

    selected = [0, 3]
    gate = {
        "schema": "lafgs_mapping_pose_pair_gate",
        "version": 1,
        "uses_test_queries": False,
        "valid": True,
        "decision": {"verdict": "STOP"},
        "preregistered_protocol": {
            "deployment_row_limit": 0,
            "query_count": 2,
            "seeds": [11, 12, 13],
        },
        "lineage": {
            "checks": {"paired": True},
            "arms": {
                arm: {
                    "uniform_q256_indices": selected,
                    "uniform_q256_indices_sha256": _json_sha256(selected),
                }
                for arm in ("baseline", "variant")
            },
            "inputs": inputs,
        },
    }
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate))
    return gate_path, arm_paths, summary_roots, inputs, selected


def test_locked_protocol_binds_gate_inputs_summaries_subset_and_seeds(tmp_path):
    gate_path, arm_paths, summary_roots, _, selected = _locked_gate_fixture(tmp_path)
    gate, calibration_paths, summaries, actual_selected = _load_locked_protocol(
        gate_path=gate_path,
        expected_gate_sha256=_sha256(gate_path),
        arm_paths=arm_paths,
        summary_roots=summary_roots,
        requested_query_count=2,
        requested_seeds=[11, 12, 13],
    )
    assert gate["uses_test_queries"] is False
    assert actual_selected == selected
    assert set(calibration_paths) == {"baseline", "candidate"}
    assert set(summaries["candidate"]) == {"11", "12", "13"}

    gate["uses_test_queries"] = True
    gate_path.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="mapping-only STOP"):
        _load_locked_protocol(
            gate_path=gate_path,
            expected_gate_sha256=_sha256(gate_path),
            arm_paths=arm_paths,
            summary_roots=summary_roots,
            requested_query_count=2,
            requested_seeds=[11, 12, 13],
        )


def test_source_summary_rejects_test_or_seed_drift(tmp_path):
    gate_path, arm_paths, _, inputs, selected = _locked_gate_fixture(tmp_path)
    del gate_path
    paths = arm_paths["baseline"]
    calibration_path = Path(inputs["baseline.calibration"]["path"])
    expected = {
        role: inputs[f"baseline.{gate_role}"]["sha256"]
        for role, gate_role in (
            ("map", "map"),
            ("metric", "metric"),
            ("teacher", "teacher"),
            ("cache", "query_cache"),
        )
    }
    expected["calibration"] = inputs["baseline.calibration"]["sha256"]
    report = {
        "schema": "lafgs_mapping_cache_evaluation",
        "version": 2,
        "uses_test_queries": False,
        "seed": 11,
        "query_count": 2,
        "query_selection": "uniform_mapping_gate",
        "map": str(paths["map"]),
        "metric_state": str(paths["metric"]),
        "complete_positive_teacher": str(paths["teacher"]),
        "query_cache": str(paths["cache"]),
        "scene_calibration": str(calibration_path),
        "evaluation_protocol": {
            "split": "mapping_only",
            "query_selection": "uniform_mapping_gate",
            "requested_query_count": 2,
            "evaluated_query_count": 2,
            "deployment_row_limit": 0,
            "selected_query_indices": selected,
            "selected_query_indices_sha256": _json_sha256(selected),
        },
        "artifacts": {
            "map": {"path": str(paths["map"]), "sha256": expected["map"]},
            "metric": {"path": str(paths["metric"]), "sha256": expected["metric"]},
            "teacher": {"path": str(paths["teacher"]), "sha256": expected["teacher"]},
            "query_cache": {"path": str(paths["cache"]), "sha256": expected["cache"]},
            "calibration": {
                "path": str(calibration_path),
                "sha256": expected["calibration"],
            },
        },
    }
    _validate_source_summary(
        report,
        arm="baseline",
        seed=11,
        selected=selected,
        paths=paths,
        calibration_path=calibration_path,
        expected_sha256=expected,
    )
    report["uses_test_queries"] = True
    with pytest.raises(ValueError, match="mapping-only replay"):
        _validate_source_summary(
            report,
            arm="baseline",
            seed=11,
            selected=selected,
            paths=paths,
            calibration_path=calibration_path,
            expected_sha256=expected,
        )


def test_reproduction_is_exact_for_candidate_and_bounded_for_baseline():
    original = {"mean_te_cm": 1.0, "mean_ae_deg": 2.0, "median_te_cm": 0.5}
    assert _validate_reproduction(
        arm="candidate", seed=2026, computed=dict(original), original=original
    ) == {"mean_ae_deg": 0.0, "mean_te_cm": 0.0, "median_te_cm": 0.0}
    _validate_reproduction(
        arm="baseline",
        seed=2026,
        computed={**original, "mean_te_cm": 1.00004},
        original=original,
    )
    with pytest.raises(ValueError, match="exactly reproduce"):
        _validate_reproduction(
            arm="candidate",
            seed=2026,
            computed={**original, "mean_te_cm": 1.000001},
            original=original,
        )
    with pytest.raises(ValueError, match="bounded CPU/GPU"):
        _validate_reproduction(
            arm="baseline",
            seed=2026,
            computed={**original, "median_te_cm": 0.500001},
            original=original,
        )
