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
from evidence.cycle_verified_fisher import COVERAGE_POLICY_NAME, COVERAGE_SELECTION_SCHEMA
from scripts.cycle_verified_fisher_cli_common import load_selection
from scripts import cycle_verified_fisher_coverage_track_common as track_common
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


def _valid_implementation_registry_payload() -> dict:
    root = Path(track_common.__file__).resolve().parents[1]
    implementation_commit = "c" * 40
    source_paths = sorted(
        set(track_common.TRACK_PRODUCER_SOURCE_PATHS)
        | set(track_common.STAGE_B_PRODUCER_SOURCE_PATHS)
        | set(track_common.CROSS_B_PRODUCER_SOURCE_PATHS)
    )
    return {
        "schema": (
            "lafgs_cycle_verified_fisher_coverage_stage_b_implementation_registry"
        ),
        "version": 1,
        "valid": True,
        "uses_test_queries": False,
        "mapping_only": True,
        "preregistration": {
            "path": "docs/evidence/"
            "p8_cycle_verified_fisher_coverage_v2_stage_b_preregistration.json",
            "commit": track_common.PREREGISTRATION_COMMIT,
            "blob_sha256": track_common.PREREGISTRATION_BLOB_SHA256,
        },
        "implementation_commit": implementation_commit,
        "required_source_paths": source_paths,
        "source_file_sha256": {
            name: sha256_file(root / name) for name in source_paths
        },
        "full_cpu_tests": {
            "passed": True,
            "result": "ALL_CPU_TESTS_PASSED",
            "implementation_commit": implementation_commit,
            "command": "python -m pytest -q",
            "test_count": 1,
        },
        "independent_review": {
            "passed": True,
            "result": "INDEPENDENT_REVIEW_COMPLETE_NO_FINDINGS",
            "implementation_commit": implementation_commit,
            "finding_counts": {"p0": 0, "p1": 0, "p2": 0},
        },
        "authorizes_real_track_execution": True,
        "authorizes_test": False,
        "authorizes_method_default_change": False,
    }


def _mock_registry_git(
    monkeypatch,
    *,
    payload: dict,
    registry_path: Path,
    committed_registry,
    implementation_is_ancestor: bool = True,
    prereg_is_ancestor: bool = True,
    committed_prereg=None,
) -> None:
    real_run = track_common.subprocess.run
    root = Path(track_common.__file__).resolve().parents[1]
    current_commit = "d" * 40
    registry_relative = str(registry_path.relative_to(root))
    prereg_relative = str(track_common.PREREGISTRATION_PATH.relative_to(root))

    def fake_run(command, **kwargs):
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return track_common.subprocess.CompletedProcess(
                command, 0, stdout=f"{current_commit}\n", stderr=""
            )
        if command[:3] == ["git", "merge-base", "--is-ancestor"]:
            is_prereg = command[3] == track_common.PREREGISTRATION_COMMIT
            ok = prereg_is_ancestor if is_prereg else implementation_is_ancestor
            return track_common.subprocess.CompletedProcess(command, 0 if ok else 1)
        if command[:2] == ["git", "show"]:
            reference = command[-1]
            if reference == (
                f"{track_common.PREREGISTRATION_COMMIT}:{prereg_relative}"
            ):
                content = (
                    track_common.PREREGISTRATION_PATH.read_bytes()
                    if committed_prereg is None
                    else committed_prereg
                )
            elif reference == f"{current_commit}:{registry_relative}":
                if committed_registry is None:
                    raise track_common.subprocess.CalledProcessError(128, command)
                content = committed_registry
            elif reference.startswith(f"{payload['implementation_commit']}:"):
                content = (root / reference.split(":", 1)[1]).read_bytes()
            else:
                return real_run(command, **kwargs)
            return track_common.subprocess.CompletedProcess(
                command, 0, stdout=content, stderr=b""
            )
        return real_run(command, **kwargs)

    monkeypatch.setattr(track_common.subprocess, "run", fake_run)


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
    left = {
        "x": torch.tensor([1.0, float("nan")]),
        "y": [torch.tensor([2])],
        "scalar": float("nan"),
    }
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


@pytest.mark.parametrize(
    "gate_name,field,bad",
    [
        ("v1_triangulated_tracks_retain_98pct", "triangulated_tracks", 17036),
        (
            "v1_broad_eligible_tracks_retain_98pct",
            "broad_eligible_tracks",
            16301,
        ),
        ("v1_high_confidence_tracks_retain_98pct", "high_confidence_tracks", 49),
        (
            "v1_triangulated_covariance_p90_not_worse_5pct",
            "triangulated_covariance_p90_m2",
            0.0383588852360845,
        ),
        (
            "v1_broad_mapping_query_coverage_not_lower",
            "mapping_query_with_broad_track_fraction",
            0.999,
        ),
    ],
)
def test_each_stairs_v1_retention_gate_can_stop(gate_name, field, bad):
    thresholds = {
        "triangulated_tracks_at_least": 17037,
        "broad_eligible_tracks_at_least": 16302,
        "high_confidence_tracks_at_least": 50,
        "triangulated_covariance_p90_m2_at_most": 0.03835888523608449,
        "mapping_query_with_broad_track_fraction_exact": 1.0,
    }
    metrics = {
        "triangulated_tracks": 17037,
        "broad_eligible_tracks": 16302,
        "high_confidence_tracks": 50,
        "triangulated_covariance_p90_m2": 0.03835888523608449,
        "mapping_query_with_broad_track_fraction": 1.0,
    }
    assert all(
        stage_b._stairs_retention_gates(
            metrics=metrics, thresholds=thresholds
        ).values()
    )
    metrics[field] = bad
    gates = stage_b._stairs_retention_gates(metrics=metrics, thresholds=thresholds)
    assert gates[gate_name] is False


@pytest.mark.parametrize(
    "mutation",
    ["query_bins", "tracks", "track_geometry", "pair_sidecar.pair", "metrics"],
)
def test_each_v1_control_scientific_projection_field_is_exact(mutation):
    v1_factor = {
        "query_bins": torch.tensor([0, 1]),
        "tracks": {"track_index": torch.tensor([0])},
        "track_geometry": {"triangulated": torch.tensor([True])},
        "pair_sidecar": {"pair": {"left_query_index": torch.tensor([0])}},
    }
    metrics = {
        "triangulated_tracks": 1,
        "broad_eligible_tracks": 1,
        "high_confidence_tracks": 1,
        "triangulated_covariance_p90_m2": 0.1,
        "mapping_query_with_broad_track_fraction": 1.0,
    }
    v2 = {
        "factor": {"payload": copy.deepcopy(v1_factor)},
        "metrics": copy.deepcopy(metrics),
    }
    assert all(
        stage_b._control_scientific_projection_status(
            v2_control=v2,
            v1_control_factor=v1_factor,
            v1_control_metrics=metrics,
        ).values()
    )
    if mutation == "pair_sidecar.pair":
        v2["factor"]["payload"]["pair_sidecar"]["pair"][
            "left_query_index"
        ][0] = 1
    elif mutation == "metrics":
        v2["metrics"]["triangulated_tracks"] = 2
    else:
        value = v2["factor"]["payload"][mutation]
        first_tensor = value if isinstance(value, torch.Tensor) else next(iter(value.values()))
        first_tensor.reshape(-1)[0] = 0 if bool(first_tensor.reshape(-1)[0]) else 1
    status = stage_b._control_scientific_projection_status(
        v2_control=v2,
        v1_control_factor=v1_factor,
        v1_control_metrics=metrics,
    )
    assert not all(status.values())


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
        artifacts[name] = {
            "relative_path": relative,
            "sha256": sha256_file(path),
        }
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
    payload["artifacts"]["control_factor"]["relative_path"] = "alternate.pt"
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
            "relative_path": factor_path.name,
            "sha256": sha256_file(factor_path),
        }
        artifacts[f"{role}_report"] = {
            "relative_path": report_path.name,
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
    variant_path = tmp_path / artifacts["variant_factor"]["relative_path"]
    variant = torch.load(variant_path, weights_only=False)
    variant["paired_run_uuid"] = "b" * 32
    torch.save(variant, variant_path)
    assert sha256_file(variant_path) != completion[
        "artifacts"
    ]["variant_factor"]["sha256"]
    with pytest.raises(ValueError, match="missing or changed"):
        validate_completion_manifest(
            path=path, expected_sha256=digest, expected_scene="greatcourt"
        )


def test_completion_is_not_allowed_when_written_factor_reload_differs(
    tmp_path, monkeypatch
):
    factor_path = tmp_path / "factor.pt"
    report_path = tmp_path / "report.json"
    expected_factor = {"x": torch.tensor([1])}
    torch.save({"x": torch.tensor([2])}, factor_path)
    report = {"valid": True}
    report_path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="Reloaded paired Track factor differs"):
        runner._validate_written_arm(
            factor=expected_factor,
            report=report,
            factor_path=factor_path,
            report_path=report_path,
        )


def test_completion_reload_accepts_aligned_scalar_nan(tmp_path):
    factor_path = tmp_path / "factor.pt"
    report_path = tmp_path / "report.json"
    factor = {"x": torch.tensor([1.0, float("nan")])}
    report = {"pair": {"mapping_point_parallax_below_1deg_fraction": float("nan")}}
    torch.save(factor, factor_path)
    report_path.write_text(json.dumps(report))
    runner._validate_written_arm(
        factor=factor,
        report=report,
        factor_path=factor_path,
        report_path=report_path,
    )


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


def test_paired_runner_refuses_preexisting_partial_root_without_resume(
    tmp_path, monkeypatch
):
    output = tmp_path / "partial_root"
    output.mkdir()
    partial = output / "orphaned_control.pt"
    partial.write_bytes(b"partial")
    monkeypatch.setattr(runner, "implementation_registry", lambda: {})
    with pytest.raises(FileExistsError, match="must not exist"):
        runner.run(
            type(
                "Args",
                (),
                {"scene": "greatcourt", "output_root": output},
            )()
        )
    assert partial.read_bytes() == b"partial"


def test_v1_runner_schema_loader_rejects_v2_selection(tmp_path):
    path = tmp_path / "v2_selection.pt"
    payload = {
        "schema": COVERAGE_SELECTION_SCHEMA,
        "version": 1,
        "policy": COVERAGE_POLICY_NAME,
        "uses_test_queries": False,
        "content_sha256": "0" * 64,
    }
    torch.save(payload, path)
    digest = sha256_file(path)
    with pytest.raises(ValueError, match="Unexpected cycle-verified"):
        load_selection(
            path=path,
            expected_file_sha256=digest,
            expected_content_sha256="0" * 64,
            probe={"payload": {}},
            expected_pair_budget=1,
        )


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


def _cross_b_fixture(tmp_path):
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
    return stairs_path, gc_path, records


def _cross_b_args(tmp_path, stairs_path, gc_path, records):
    return type(
        "Args",
        (),
        {
            "stairs_stage_b_gate": stairs_path,
            "expected_stairs_stage_b_gate_sha256": records["stairs"]["sha256"],
            "greatcourt_stage_b_gate": gc_path,
            "expected_greatcourt_stage_b_gate_sha256": records["greatcourt"][
                "sha256"
            ],
            "output": tmp_path / "cross_b.json",
        },
    )()


def test_cross_b_keeps_base_and_stairs_gates_separate_and_does_not_overauthorize(
    tmp_path, monkeypatch
):
    stairs_path, gc_path, records = _cross_b_fixture(tmp_path)
    monkeypatch.setattr(
        cross_b,
        "validate_stage_b_gate",
        lambda scene, path, expected_sha256: records[scene],
    )
    monkeypatch.setattr(cross_b, "cross_b_producer_identity", lambda: {"clean": True})
    monkeypatch.setattr(cross_b, "require_clean_identity", lambda *args, **kwargs: None)
    result = cross_b.run(_cross_b_args(tmp_path, stairs_path, gc_path, records))
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


@pytest.mark.parametrize(
    "contamination",
    [
        "cross_stage_a_root",
        "greatcourt_parent",
        "compiled_identity",
        "track_producer",
        "stage_b_producer",
        "missing_base_gate",
        "retention_leaks_to_greatcourt",
    ],
)
def test_cross_b_rejects_recursive_scene_contamination(
    contamination, monkeypatch, tmp_path
):
    stairs_path, gc_path, records = _cross_b_fixture(tmp_path)
    stairs = records["stairs"]["payload"]
    greatcourt = records["greatcourt"]["payload"]
    if contamination == "cross_stage_a_root":
        greatcourt["stage_a"]["cross_scene_gate"]["sha256"] = "1" * 64
    elif contamination == "greatcourt_parent":
        stairs["paired_track"]["greatcourt_stage_b_parent"]["sha256"] = "1" * 64
    elif contamination == "compiled_identity":
        greatcourt["compiled_identity"] = {"algorithm": "polluted"}
    elif contamination == "track_producer":
        greatcourt["paired_track"]["track_producer_identity"] = {"commit": "other"}
    elif contamination == "stage_b_producer":
        greatcourt["stage_b_producer_identity"] = {"commit": "other"}
    elif contamination == "missing_base_gate":
        stairs["stage_b"]["base_gates"].pop(next(iter(stage_b.BASE_GATE_NAMES)))
    elif contamination == "retention_leaks_to_greatcourt":
        greatcourt["stage_b"]["stairs_v1_retention_gates"] = {
            name: True for name in stage_b.STAIRS_RETENTION_GATE_NAMES
        }
    monkeypatch.setattr(
        cross_b,
        "validate_stage_b_gate",
        lambda scene, path, expected_sha256: records[scene],
    )
    with pytest.raises(ValueError):
        cross_b.run(_cross_b_args(tmp_path, stairs_path, gc_path, records))


def test_stage_b_entrypoint_invalid_is_exit1_without_gate(monkeypatch, tmp_path):
    output = tmp_path / "scene_gate.json"
    monkeypatch.setattr(
        stage_b,
        "evaluate_scene",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("invalid lineage")),
    )
    with pytest.raises(SystemExit) as raised:
        stage_b.entrypoint(
            [
                "--scene",
                "greatcourt",
                "--completion-manifest",
                str(tmp_path / "completion.json"),
                "--expected-completion-manifest-sha256",
                "0" * 64,
                "--output",
                str(output),
            ]
        )
    assert raised.value.code == 1
    assert not output.exists()


def test_stage_b_entrypoint_scientific_stop_is_exit2_with_gate(
    monkeypatch, tmp_path
):
    output = tmp_path / "scene_gate.json"
    evaluation = {
        "completion": {"path": tmp_path / "completion.json", "artifacts": {}},
        "passed": False,
    }
    monkeypatch.setattr(stage_b, "evaluate_scene", lambda **kwargs: evaluation)
    monkeypatch.setattr(stage_b, "stage_b_producer_identity", lambda: {})
    monkeypatch.setattr(stage_b, "require_clean_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        stage_b,
        "gate_payload",
        lambda **kwargs: {
            "scene_specific_mechanism_pass": False,
            "decision": "STOP_SCENE_MECHANISM",
        },
    )
    with pytest.raises(SystemExit) as raised:
        stage_b.entrypoint(
            [
                "--scene",
                "greatcourt",
                "--completion-manifest",
                str(tmp_path / "completion.json"),
                "--expected-completion-manifest-sha256",
                "0" * 64,
                "--output",
                str(output),
            ]
        )
    assert raised.value.code == 2
    assert json.loads(output.read_text())["decision"] == "STOP_SCENE_MECHANISM"


def test_cross_b_entrypoint_scientific_stop_is_exit2_with_gate(
    monkeypatch, tmp_path
):
    stairs_path, gc_path, records = _cross_b_fixture(tmp_path)
    records["greatcourt"]["payload"]["scene_specific_mechanism_pass"] = False
    output = tmp_path / "cross_stop.json"
    monkeypatch.setattr(
        cross_b,
        "validate_stage_b_gate",
        lambda scene, path, expected_sha256: records[scene],
    )
    monkeypatch.setattr(cross_b, "cross_b_producer_identity", lambda: {"clean": True})
    monkeypatch.setattr(cross_b, "require_clean_identity", lambda *args, **kwargs: None)
    with pytest.raises(SystemExit) as raised:
        cross_b.entrypoint(
            [
                "--stairs-stage-b-gate",
                str(stairs_path),
                "--expected-stairs-stage-b-gate-sha256",
                records["stairs"]["sha256"],
                "--greatcourt-stage-b-gate",
                str(gc_path),
                "--expected-greatcourt-stage-b-gate-sha256",
                records["greatcourt"]["sha256"],
                "--output",
                str(output),
            ]
        )
    assert raised.value.code == 2
    assert output.exists()


def test_cross_b_entrypoint_invalid_is_exit1_without_gate(monkeypatch, tmp_path):
    output = tmp_path / "invalid_cross.json"
    monkeypatch.setattr(
        cross_b,
        "validate_stage_b_gate",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("invalid lineage")),
    )
    with pytest.raises(SystemExit) as raised:
        cross_b.entrypoint(
            [
                "--stairs-stage-b-gate",
                str(tmp_path / "stairs.json"),
                "--expected-stairs-stage-b-gate-sha256",
                "0" * 64,
                "--greatcourt-stage-b-gate",
                str(tmp_path / "greatcourt.json"),
                "--expected-greatcourt-stage-b-gate-sha256",
                "0" * 64,
                "--output",
                str(output),
            ]
        )
    assert raised.value.code == 1
    assert not output.exists()


def test_implementation_registry_missing_and_pending_are_fail_closed(
    monkeypatch, tmp_path
):
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(track_common, "IMPLEMENTATION_REGISTRY_PATH", missing)
    with pytest.raises(RuntimeError, match="not committed"):
        track_common.implementation_registry()

    pending = tmp_path / "pending.json"
    payload = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/evidence/"
            "p8_cycle_verified_fisher_coverage_v2_stage_b_implementation.json"
        ).read_text()
    )
    payload["full_cpu_tests"]["passed"] = False
    payload["full_cpu_tests"]["result"] = "PENDING_SYNTHETIC_TEST"
    payload["independent_review"]["passed"] = False
    payload["independent_review"]["result"] = "PENDING_SYNTHETIC_REVIEW"
    payload["authorizes_real_track_execution"] = False
    pending.write_text(json.dumps(payload))
    monkeypatch.setattr(track_common, "IMPLEMENTATION_REGISTRY_PATH", pending)
    with pytest.raises(RuntimeError, match="invalid or stale"):
        track_common.implementation_registry()


@pytest.mark.parametrize(
    "tamper",
    [
        "cpu_false",
        "empty_cpu_result",
        "wrong_tested_commit",
        "review_finding",
        "source_hash",
        "authorization_false",
    ],
)
def test_implementation_registry_rejects_false_or_tampered_claims(
    tamper, monkeypatch, tmp_path
):
    payload = _valid_implementation_registry_payload()
    if tamper == "cpu_false":
        payload["full_cpu_tests"]["passed"] = False
    elif tamper == "empty_cpu_result":
        payload["full_cpu_tests"]["result"] = ""
    elif tamper == "wrong_tested_commit":
        payload["full_cpu_tests"]["implementation_commit"] = "e" * 40
    elif tamper == "review_finding":
        payload["independent_review"]["finding_counts"]["p1"] = 1
    elif tamper == "source_hash":
        first = payload["required_source_paths"][0]
        payload["source_file_sha256"][first] = "0" * 64
    elif tamper == "authorization_false":
        payload["authorizes_real_track_execution"] = False
    path = tmp_path / "tampered_registry.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(track_common, "IMPLEMENTATION_REGISTRY_PATH", path)
    with pytest.raises(RuntimeError, match="invalid or stale"):
        track_common.implementation_registry()


def test_implementation_registry_rejects_untracked_and_dirty_file(
    monkeypatch, tmp_path
):
    root = Path(track_common.__file__).resolve().parents[1]
    registry_dir = root / ".pytest_cache" / "p8_registry_contracts"
    registry_dir.mkdir(parents=True, exist_ok=True)
    registry_path = registry_dir / f"{tmp_path.name}.json"
    payload = _valid_implementation_registry_payload()
    registry_path.write_text(json.dumps(payload, sort_keys=True))
    monkeypatch.setattr(
        track_common, "IMPLEMENTATION_REGISTRY_PATH", registry_path
    )
    try:
        _mock_registry_git(
            monkeypatch,
            payload=payload,
            registry_path=registry_path,
            committed_registry=None,
        )
        with pytest.raises(RuntimeError, match="must be committed"):
            track_common.implementation_registry()

        dirty_bytes = registry_path.read_bytes()
        _mock_registry_git(
            monkeypatch,
            payload=payload,
            registry_path=registry_path,
            committed_registry=dirty_bytes + b"\ncommitted-version-differs",
        )
        with pytest.raises(RuntimeError, match="registry is dirty"):
            track_common.implementation_registry()
    finally:
        registry_path.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "boundary",
    ["implementation_ancestry", "prereg_ancestry", "prereg_blob"],
)
def test_implementation_registry_rejects_commit_or_prereg_forgery(
    boundary, monkeypatch, tmp_path
):
    root = Path(track_common.__file__).resolve().parents[1]
    registry_dir = root / ".pytest_cache" / "p8_registry_contracts"
    registry_dir.mkdir(parents=True, exist_ok=True)
    registry_path = registry_dir / f"{tmp_path.name}.json"
    payload = _valid_implementation_registry_payload()
    registry_path.write_text(json.dumps(payload, sort_keys=True))
    monkeypatch.setattr(
        track_common, "IMPLEMENTATION_REGISTRY_PATH", registry_path
    )
    try:
        _mock_registry_git(
            monkeypatch,
            payload=payload,
            registry_path=registry_path,
            committed_registry=registry_path.read_bytes(),
            implementation_is_ancestor=boundary != "implementation_ancestry",
            prereg_is_ancestor=boundary != "prereg_ancestry",
            committed_prereg=(b"forged-prereg" if boundary == "prereg_blob" else None),
        )
        expected = {
            "implementation_ancestry": "not in current history",
            "prereg_ancestry": "does not precede implementation",
            "prereg_blob": "commit/blob registry differs",
        }[boundary]
        with pytest.raises(RuntimeError, match=expected):
            track_common.implementation_registry()
    finally:
        registry_path.unlink(missing_ok=True)
