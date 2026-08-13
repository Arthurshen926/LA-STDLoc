import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from common.hashing import sha256_file
from scripts import aggregate_cycle_verified_fisher_coverage_cross_scene as cross_b
from scripts import compare_cycle_verified_fisher_coverage_mechanism as stage_b
from scripts import materialize_cycle_verified_fisher_coverage_track_factor as runner
from scripts import materialize_cycle_verified_track_factor as v1_runner
from scripts.cycle_verified_fisher_coverage_track_common import (
    CONTROL_POLICY_NAME,
    CONTROL_SUBSET_ROLE,
    VARIANT_POLICY_NAME,
    VARIANT_SUBSET_ROLE,
    load_cross_scene_authority,
    pair_table_sha256,
    recursive_equal,
    validate_completion_manifest,
)


def _write_json(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return sha256_file(path)


def _producer() -> dict:
    return {
        "schema": "lafgs_cycle_verified_fisher_coverage_track_producer",
        "version": 1,
        "algorithm": "p8_cycle_verified_fisher_coverage_v2_reuse_track",
        "entrypoint": (
            "python -m "
            "scripts.materialize_cycle_verified_fisher_coverage_track_factor"
        ),
        "git_commit": "a" * 40,
        "required_source_paths_clean": True,
        "source_paths": ["synthetic.py"],
        "source_file_sha256": {"synthetic.py": "b" * 64},
        "runtime": {"python": "3.11.0", "torch": "2.0.0", "device": "cpu"},
    }


def _factor(*, policy: str, role: str, run_uuid: str, producer: dict) -> dict:
    return {
        "schema": "lafgs_pair_policy_track_factor",
        "version": 1,
        "uses_test_queries": False,
        "pair_policy": policy,
        "pair_policy_parameters": {
            "reuse_only": True,
            "pair_subset_role": role,
            "probe_matcher": {},
            "track_science_contract": {"shared": 1},
        },
        "paired_run_uuid": run_uuid,
        "track_producer_identity": copy.deepcopy(producer),
        "input_lineage": {
            "pair_subset_role": role,
            "paired_run_uuid": run_uuid,
            "track_producer_identity": copy.deepcopy(producer),
        },
    }


def test_pair_table_hash_is_order_and_membership_exact():
    pairs = [(0, 1), (1, 2)]
    assert pair_table_sha256(pairs) == hashlib.sha256(
        b"[[0,1],[1,2]]"
    ).hexdigest()
    assert pair_table_sha256(pairs) != pair_table_sha256(list(reversed(pairs)))
    assert pair_table_sha256(pairs) != pair_table_sha256([(0, 1), (0, 2)])


def test_recursive_tensor_parity_treats_aligned_nan_as_equal():
    left = {"x": torch.tensor([1.0, float("nan")]), "y": [torch.tensor([2])]}
    right = copy.deepcopy(left)
    assert recursive_equal(left, right)
    right["x"][0] = 2.0
    assert not recursive_equal(left, right)


def test_base_gates_read_every_exact_preregistered_threshold(monkeypatch):
    thresholds = {
        "minimum_triangulated_track_retention_ratio": 0.80,
        "minimum_broad_eligible_track_retention_ratio": 0.81,
        "minimum_high_confidence_track_retention_ratio": 0.82,
        "maximum_triangulated_covariance_p90_ratio": 1.25,
        "minimum_broad_query_coverage_delta": -0.1,
        "coverage_formula": "synthetic",
    }
    monkeypatch.setattr(
        stage_b,
        "preregistration",
        lambda: {"stage_b_thresholds": thresholds},
    )
    control = {
        "triangulated_tracks": 100,
        "broad_eligible_tracks": 100,
        "high_confidence_tracks": 100,
        "triangulated_covariance_p90_m2": 1.0,
        "mapping_query_with_broad_track_fraction": 0.8,
    }
    variant = {
        "triangulated_tracks": 80,
        "broad_eligible_tracks": 81,
        "high_confidence_tracks": 82,
        "triangulated_covariance_p90_m2": 1.25,
        "mapping_query_with_broad_track_fraction": 0.7000000000000001,
    }
    assert all(stage_b._base_gates(control=control, variant=variant).values())


@pytest.mark.parametrize(
    "gate_name,field,bad",
    [
        ("triangulated_tracks_retain_98pct", "triangulated_tracks", 97),
        ("broad_eligible_tracks_retain_98pct", "broad_eligible_tracks", 97),
        ("high_confidence_tracks_retain_98pct", "high_confidence_tracks", 97),
        (
            "triangulated_covariance_p90_not_worse_5pct",
            "triangulated_covariance_p90_m2",
            1.051,
        ),
        (
            "broad_mapping_query_coverage_not_lower",
            "mapping_query_with_broad_track_fraction",
            0.79,
        ),
    ],
)
def test_each_scientific_base_gate_can_stop(gate_name, field, bad):
    control = {
        "triangulated_tracks": 100,
        "broad_eligible_tracks": 100,
        "high_confidence_tracks": 100,
        "triangulated_covariance_p90_m2": 1.0,
        "mapping_query_with_broad_track_fraction": 0.8,
    }
    variant = copy.deepcopy(control)
    variant[field] = bad
    gates = stage_b._base_gates(control=control, variant=variant)
    assert gates[gate_name] is False


def test_runner_forbids_matcher_and_pair_selector_reentry(monkeypatch):
    seen = {}

    def fake_builder(**kwargs):
        seen["matcher"] = runner.triangulation.reciprocal_epipolar_matches
        seen["selector"] = runner.triangulation.candidate_camera_pairs
        with pytest.raises(RuntimeError, match="forbids"):
            seen["matcher"]()
        with pytest.raises(RuntimeError, match="forbids"):
            seen["selector"]()
        raise RuntimeError("sentinel")

    monkeypatch.setattr(
        runner.triangulation, "build_cycle_consistent_tracks", fake_builder
    )
    monkeypatch.setattr(
        runner,
        "probe_pair_subset_track_build_inputs",
        lambda _probe, pairs: {
            "precomputed_pairs": pairs,
            "precomputed_pair_matches": {},
            "precomputed_pair_match_diagnostics": {},
            "precomputed_confidence_includes_detector_scores": True,
        },
    )
    monkeypatch.setattr(
        runner,
        "_track_science_contract",
        lambda **kwargs: {
            "pair_neighbors": 1,
            "minimum_baseline_m": 0.0,
            "maximum_baseline_m": 1.0,
            "maximum_axis_angle_deg": 180.0,
            "minimum_track_views": 1,
            "require_cycle": True,
            "allow_chain_tracks": True,
        },
    )
    with pytest.raises(RuntimeError, match="sentinel"):
        runner._build_arm(
            role="control",
            pair_policy=CONTROL_POLICY_NAME,
            subset_role=CONTROL_SUBSET_ROLE,
            pairs=[(0, 1)],
            registry={
                "pair_match_probe": {
                    "payload": {
                        "matcher": {
                            "minimum_similarity": 0.65,
                            "minimum_margin": 0.01,
                            "maximum_epipolar_error_px": 2.0,
                            "epipolar_candidate_topk": 1,
                            "epipolar_recovered_minimum_similarity": -1.0,
                            "epipolar_recovered_minimum_margin": -1.0,
                        }
                    }
                },
                "query_cache": {"names": ["a", "b"]},
            },
            manifest={},
            frozen={},
            base_lineage={},
            run_uuid="a" * 32,
            producer=_producer(),
            descriptors=[torch.zeros(1), torch.zeros(1)],
            keypoints=[torch.zeros(1, 2), torch.zeros(1, 2)],
            scores=[torch.ones(1), torch.ones(1)],
            intrinsics=torch.eye(3).repeat(2, 1, 1),
            poses=torch.eye(4).repeat(2, 1, 1),
            image_hw=torch.ones(2, 2, dtype=torch.long),
            depth_at_keypoints=[torch.ones(1), torch.ones(1)],
            keypoint_counts=torch.ones(2, dtype=torch.long),
            mapping_keypoints=1,
            nms_radius=1,
            pair_budget=1,
            device="cpu",
        )
    assert seen["matcher"] is seen["selector"]


def test_completion_manifest_rejects_partial_and_wrong_relative_path(
    tmp_path, monkeypatch
):
    producer = _producer()
    monkeypatch.setattr(
        "scripts.cycle_verified_fisher_coverage_track_common.validate_track_producer_identity",
        lambda identity, label: identity,
    )
    monkeypatch.setattr(
        "scripts.cycle_verified_fisher_coverage_track_common.implementation_registry",
        lambda: {"implementation_commit": "c" * 40},
    )
    registry_path = tmp_path / "implementation.json"
    registry_path.write_text("registry")
    monkeypatch.setattr(
        "scripts.cycle_verified_fisher_coverage_track_common.IMPLEMENTATION_REGISTRY_PATH",
        registry_path,
    )
    stems = {
        "control_factor": "cycle_verified_fisher_coverage_nearest_control_track_factor.pt",
        "control_report": "cycle_verified_fisher_coverage_nearest_control_track_factor.json",
        "variant_factor": "cycle_verified_fisher_coverage_track_factor.pt",
        "variant_report": "cycle_verified_fisher_coverage_track_factor.json",
    }
    artifacts = {}
    for name, relative in stems.items():
        path = tmp_path / relative
        path.write_bytes(name.encode())
        artifacts[name] = {"path": str(path), "sha256": sha256_file(path)}
    payload = {
        "schema": "lafgs_cycle_verified_fisher_coverage_paired_track_completion",
        "version": 1,
        "uses_test_queries": False,
        "mapping_only": True,
        "complete": True,
        "partial": False,
        "resume_allowed": False,
        "scene": "greatcourt",
        "build_order": ["control", "variant"],
        "run_uuid": "a" * 32,
        "track_producer_identity": producer,
        "implementation_registry": {
            "path": str(registry_path),
            "sha256": sha256_file(registry_path),
            "implementation_commit": "c" * 40,
        },
        "inputs": {},
        "artifacts": artifacts,
        "summaries": {"control": {}, "variant": {}},
        "failure_recovery": (
            "isolate_entire_output_root_and_rebuild_both_arms_from_scratch"
        ),
    }
    manifest = tmp_path / "paired_track_completion.json"
    digest = _write_json(manifest, payload)
    assert validate_completion_manifest(
        path=manifest, expected_sha256=digest, expected_scene="greatcourt"
    )["payload"]["complete"] is True
    payload["partial"] = True
    digest = _write_json(manifest, payload)
    with pytest.raises(ValueError, match="invalid or partial"):
        validate_completion_manifest(
            path=manifest, expected_sha256=digest, expected_scene="greatcourt"
        )
    payload["partial"] = False
    payload["artifacts"]["control_factor"]["path"] = str(
        tmp_path / "alternate.pt"
    )
    digest = _write_json(manifest, payload)
    with pytest.raises(ValueError, match="missing or changed"):
        validate_completion_manifest(
            path=manifest, expected_sha256=digest, expected_scene="greatcourt"
        )


def test_completion_manifest_rejects_cross_run_splice(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.cycle_verified_fisher_coverage_track_common.validate_track_producer_identity",
        lambda identity, label: identity,
    )
    monkeypatch.setattr(
        "scripts.cycle_verified_fisher_coverage_track_common.implementation_registry",
        lambda: {"implementation_commit": "c" * 40},
    )
    registry_path = tmp_path / "implementation.json"
    registry_path.write_text("registry")
    monkeypatch.setattr(
        "scripts.cycle_verified_fisher_coverage_track_common.IMPLEMENTATION_REGISTRY_PATH",
        registry_path,
    )
    producer = _producer()
    run_uuid = "a" * 32
    artifacts = {}
    for role, policy, subset in (
        ("control", CONTROL_POLICY_NAME, CONTROL_SUBSET_ROLE),
        ("variant", VARIANT_POLICY_NAME, VARIANT_SUBSET_ROLE),
    ):
        stem = (
            "cycle_verified_fisher_coverage_nearest_control_track_factor"
            if role == "control"
            else "cycle_verified_fisher_coverage_track_factor"
        )
        factor_path = tmp_path / f"{stem}.pt"
        report_path = tmp_path / f"{stem}.json"
        factor = _factor(
            policy=policy, role=subset, run_uuid=run_uuid, producer=producer
        )
        torch.save(factor, factor_path)
        report_path.write_text("{}")
        artifacts[f"{role}_factor"] = {
            "path": str(factor_path),
            "sha256": sha256_file(factor_path),
        }
        artifacts[f"{role}_report"] = {
            "path": str(report_path),
            "sha256": sha256_file(report_path),
        }
    manifest = {
        "schema": "lafgs_cycle_verified_fisher_coverage_paired_track_completion",
        "version": 1,
        "uses_test_queries": False,
        "mapping_only": True,
        "complete": True,
        "partial": False,
        "resume_allowed": False,
        "scene": "greatcourt",
        "build_order": ["control", "variant"],
        "run_uuid": run_uuid,
        "track_producer_identity": producer,
        "implementation_registry": {
            "path": str(registry_path),
            "sha256": sha256_file(registry_path),
            "implementation_commit": "c" * 40,
        },
        "inputs": {},
        "artifacts": artifacts,
        "summaries": {"control": {}, "variant": {}},
        "failure_recovery": (
            "isolate_entire_output_root_and_rebuild_both_arms_from_scratch"
        ),
    }
    path = tmp_path / "paired_track_completion.json"
    digest = _write_json(path, manifest)
    completion = validate_completion_manifest(
        path=path, expected_sha256=digest, expected_scene="greatcourt"
    )
    variant = torch.load(artifacts["variant_factor"]["path"], weights_only=False)
    variant["paired_run_uuid"] = "b" * 32
    torch.save(variant, artifacts["variant_factor"]["path"])
    assert sha256_file(Path(artifacts["variant_factor"]["path"])) != completion[
        "artifacts"
    ]["variant_factor"]["sha256"]


def test_cross_authority_rejects_caller_selected_alternate(tmp_path, monkeypatch):
    gate = {
        "schema": "lafgs_cycle_verified_fisher_coverage_cross_scene_stage_a_gate",
        "version": 1,
        "uses_test_queries": False,
        "mapping_only": True,
        "valid": True,
        "policy": VARIANT_POLICY_NAME,
        "both_scene_stage_a_passed": True,
        "advance_to_v2_aware_reuse_only_track_build": True,
        "authorizes_existing_v1_track_runner": False,
        "decision": "GO_TO_V2_AWARE_REUSE_ONLY_TRACK_BUILD",
        "inputs": {
            "stairs": {"path": "stairs.json", "sha256": "1" * 64},
            "greatcourt": {"path": "greatcourt.json", "sha256": "2" * 64},
        },
    }
    path = tmp_path / "alternate.json"
    digest = _write_json(path, gate)
    monkeypatch.setattr(
        "scripts.cycle_verified_fisher_coverage_track_common.preregistration",
        lambda: {
            "authorization": {
                "cross_scene_stage_a_gate": {
                    "path": str(tmp_path / "compiled.json"),
                    "sha256": "0" * 64,
                    "schema": gate["schema"],
                    "version": 1,
                    "decision": gate["decision"],
                }
            }
        },
    )
    with pytest.raises(ValueError, match="not the compiled GO"):
        load_cross_scene_authority(
            path=path, expected_sha256=digest, scene="stairs"
        )


def test_stairs_runner_requires_greatcourt_pass_before_any_output(
    tmp_path, monkeypatch
):
    output = tmp_path / "stairs_output"
    monkeypatch.setattr(runner, "implementation_registry", lambda: {})
    monkeypatch.setattr(
        runner,
        "load_scene_inputs",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("Stairs inputs must remain untouched before GC Pass")
        ),
    )
    arguments = runner.build_parser().parse_args(
        [
            "--scene", "stairs",
            "--cross-scene-stage-a-gate", "cross.json",
            "--expected-cross-scene-stage-a-gate-sha256", "0" * 64,
            "--scene-stage-a-gate", "scene.json",
            "--expected-scene-stage-a-gate-sha256", "0" * 64,
            "--manifest", "manifest.json",
            "--expected-manifest-sha256", "0" * 64,
            "--frozen-track-payload", "frozen.pt",
            "--expected-frozen-track-payload-sha256", "0" * 64,
            "--query-cache", "cache.pt",
            "--expected-query-cache-sha256", "0" * 64,
            "--mapping-scope-equivalence", "scope.json",
            "--expected-mapping-scope-equivalence-sha256", "0" * 64,
            "--proposals", "proposals.pt",
            "--expected-proposals-sha256", "0" * 64,
            "--expected-proposals-content-sha256", "0" * 64,
            "--probe", "probe.pt",
            "--expected-probe-sha256", "0" * 64,
            "--expected-probe-content-sha256", "0" * 64,
            "--verified-cycle-table", "table.pt",
            "--expected-verified-cycle-table-sha256", "0" * 64,
            "--expected-verified-cycle-table-content-sha256", "0" * 64,
            "--selection", "selection.pt",
            "--expected-selection-sha256", "0" * 64,
            "--expected-selection-content-sha256", "0" * 64,
            "--expected-query-names-sha256", "0" * 64,
            "--expected-mapping-keypoints", "1",
            "--expected-nms-radius", "1",
            "--expected-pair-budget", "1",
            "--expected-candidate-pair-count", "1",
            "--expected-candidate-components", "1",
            "--device", "cpu",
            "--output-root", str(output),
        ]
    )
    with pytest.raises(ValueError, match="requires the prior GreatCourt"):
        runner.run(arguments)
    assert not output.exists()


def test_v1_runner_schema_loader_rejects_v2_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(v1_runner, "validate_scene_contract", lambda **kwargs: {})
    monkeypatch.setattr(v1_runner, "load_mapping_cache", lambda **kwargs: {})
    monkeypatch.setattr(v1_runner, "load_proposals", lambda **kwargs: {})
    monkeypatch.setattr(v1_runner, "load_probe", lambda **kwargs: {})
    monkeypatch.setattr(v1_runner, "validate_probe_proposal_lineage", lambda **kwargs: None)
    monkeypatch.setattr(
        v1_runner,
        "load_selection",
        lambda **kwargs: (_ for _ in ()).throw(
            ValueError("Unexpected cycle-verified Fisher selection contract")
        ),
    )
    args = type(
        "Args",
        (),
        {
            "scene": "stairs",
            "expected_mapping_keypoints": 1,
            "expected_nms_radius": 1,
            "expected_pair_budget": 1,
            "expected_candidate_pair_count": 1,
            "expected_candidate_components": 1,
            "query_cache": tmp_path / "cache",
            "expected_query_cache_sha256": "0" * 64,
            "expected_query_names_sha256": "0" * 64,
            "mapping_scope_equivalence": None,
            "expected_mapping_scope_equivalence_sha256": None,
            "proposals": tmp_path / "proposals",
            "expected_proposals_sha256": "0" * 64,
            "expected_proposals_content_sha256": "0" * 64,
            "probe": tmp_path / "probe",
            "expected_probe_sha256": "0" * 64,
            "expected_probe_content_sha256": "0" * 64,
            "selection": tmp_path / "v2_selection",
            "expected_selection_sha256": "0" * 64,
            "expected_selection_content_sha256": "0" * 64,
        },
    )()
    with pytest.raises(ValueError, match="Unexpected cycle-verified"):
        v1_runner.run(args)


def _scene_gate_payload(scene: str, *, passed: bool, compiled: dict, parent=None):
    base = {name: True for name in stage_b.BASE_GATE_NAMES}
    retention = (
        {name: True for name in stage_b.STAIRS_RETENTION_GATE_NAMES}
        if scene == "stairs"
        else {}
    )
    return {
        "scene": scene,
        "stage_a": {"cross_scene_gate": {"path": "cross", "sha256": "0" * 64}},
        "compiled_identity": compiled,
        "paired_track": {
            "track_producer_identity": {"commit": "same"},
            "greatcourt_stage_b_parent": parent,
        },
        "stage_b_producer_identity": {"commit": "same"},
        "stage_b": {
            "base_gates": base,
            "stairs_v1_retention_gates": retention,
            "stairs_control_parity_gate": (
                {"v1_nearest_control_scientific_projection_exact": True}
                if scene == "stairs"
                else {}
            ),
        },
        "stairs_v1_reference": {} if scene == "stairs" else None,
        "scene_specific_mechanism_pass": passed,
    }


def test_cross_b_keeps_base_and_stairs_gates_separate_and_does_not_overauthorize(
    tmp_path, monkeypatch
):
    stairs_path = tmp_path / "stairs.json"
    gc_path = tmp_path / "greatcourt.json"
    stairs_path.write_text("stairs")
    gc_path.write_text("greatcourt")
    gc_ref = {"path": str(gc_path.resolve()), "sha256": sha256_file(gc_path)}
    compiled = {"algorithm": "p8_cycle_verified_fisher_coverage_v2"}
    records = {
        "stairs": {
            "path": stairs_path.resolve(),
            "sha256": sha256_file(stairs_path),
            "payload": _scene_gate_payload(
                "stairs", passed=True, compiled=compiled, parent=gc_ref
            ),
        },
        "greatcourt": {
            "path": gc_path.resolve(),
            "sha256": sha256_file(gc_path),
            "payload": _scene_gate_payload(
                "greatcourt", passed=True, compiled=compiled
            ),
        },
    }
    monkeypatch.setattr(
        cross_b,
        "validate_stage_b_gate",
        lambda scene, path, expected_sha256: records[scene],
    )
    monkeypatch.setattr(cross_b, "cross_b_producer_identity", lambda: {"clean": True})
    monkeypatch.setattr(cross_b, "require_clean_identity", lambda *args, **kwargs: None)
    output = tmp_path / "cross_b.json"
    result = cross_b.run(
        type(
            "Args",
            (),
            {
                "stairs_stage_b_gate": stairs_path,
                "expected_stairs_stage_b_gate_sha256": records["stairs"]["sha256"],
                "greatcourt_stage_b_gate": gc_path,
                "expected_greatcourt_stage_b_gate_sha256": records["greatcourt"]["sha256"],
                "output": output,
            },
        )()
    )
    assert result["decision"] == "GO_TO_V2_AWARE_FULLCHAIN_LINEAGE_IMPLEMENTATION"
    assert result["authorizes_existing_fullchain"] is False
    assert result["advance_to_mapping_pose"] is False
    assert result["authorizes_test"] is False
    assert set(result["base_gates"]) == {"stairs", "greatcourt"}
    assert set(result["stairs_v1_retention_gates"]) == stage_b.STAIRS_RETENTION_GATE_NAMES


def test_cross_b_persists_scientific_stop(monkeypatch, tmp_path):
    stairs_path = tmp_path / "stairs.json"
    gc_path = tmp_path / "greatcourt.json"
    stairs_path.write_text("stairs")
    gc_path.write_text("greatcourt")
    gc_ref = {"path": str(gc_path.resolve()), "sha256": sha256_file(gc_path)}
    compiled = {"algorithm": "p8_cycle_verified_fisher_coverage_v2"}
    records = {
        "stairs": {
            "path": stairs_path.resolve(),
            "sha256": sha256_file(stairs_path),
            "payload": _scene_gate_payload(
                "stairs", passed=True, compiled=compiled, parent=gc_ref
            ),
        },
        "greatcourt": {
            "path": gc_path.resolve(),
            "sha256": sha256_file(gc_path),
            "payload": _scene_gate_payload(
                "greatcourt", passed=False, compiled=compiled
            ),
        },
    }
    monkeypatch.setattr(
        cross_b,
        "validate_stage_b_gate",
        lambda scene, path, expected_sha256: records[scene],
    )
    monkeypatch.setattr(cross_b, "cross_b_producer_identity", lambda: {"clean": True})
    monkeypatch.setattr(cross_b, "require_clean_identity", lambda *args, **kwargs: None)
    output = tmp_path / "cross_stop.json"
    result = cross_b.run(
        type(
            "Args",
            (),
            {
                "stairs_stage_b_gate": stairs_path,
                "expected_stairs_stage_b_gate_sha256": records["stairs"]["sha256"],
                "greatcourt_stage_b_gate": gc_path,
                "expected_greatcourt_stage_b_gate_sha256": records["greatcourt"]["sha256"],
                "output": output,
            },
        )()
    )
    assert output.exists()
    assert result["both_scene_stage_b_passed"] is False
    assert result["decision"] == "STOP_BEFORE_FULLCHAIN_LINEAGE_IMPLEMENTATION"
