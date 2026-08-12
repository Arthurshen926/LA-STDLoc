import copy
import json

import pytest

from common.hashing import sha256_file
from evidence.cycle_verified_fisher import CONTROL_POLICY_NAME, POLICY_NAME
from scripts import aggregate_cycle_verified_fisher_cross_scene as cross_scene_cli


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _ref(path, **extra):
    return {"path": str(path.resolve()), "sha256": sha256_file(path), **extra}


def _science_contract(scene_contract, *, algorithm_override=None):
    payload = {
        "allow_chain_tracks": True,
        "depth_sampling": "native_depth_at_sparse_keypoints_or_nearest_pixel_v1",
        "exact_pair_budget": scene_contract["pair_budget"],
        "huber_delta_px": 2.0,
        "mapping_keypoints": scene_contract["mapping_keypoints"],
        "mapping_nms_radius": scene_contract["nms_radius"],
        "matcher": copy.deepcopy(cross_scene_cli.MATCHER_CONTRACT),
        "maximum_axis_angle_deg": 75.0,
        "maximum_baseline_m": 5.0,
        "maximum_condition_number": 1_000_000.0,
        "maximum_covariance_trace_m2": (
            0.001 if scene_contract["scene"] == "stairs" else 0.01
        ),
        "maximum_observations_per_landmark": 32,
        "maximum_rendered_depth_residual_m": (
            0.02 if scene_contract["scene"] == "stairs" else 0.7
        ),
        "maximum_reprojection_px": (
            0.6 if scene_contract["scene"] == "stairs" else 1.9
        ),
        "minimum_baseline_m": 0.03,
        "minimum_parallax_deg": 1.0,
        "minimum_rendered_depth_observations": 2,
        "minimum_track_views": 3,
        "minimum_view_bins": 2,
        "pair_neighbors": 6,
        "parallax_quantile": 0.75,
        "require_cycle": True,
        "surface_support_enabled": False,
        "triangulation_iterations": 3,
        "view_bins": 8,
        "view_direction_weight": 0.5,
    }
    if algorithm_override is not None:
        name, value = algorithm_override
        payload[name] = value
    return payload


def _make_scene_gate(
    root,
    scene,
    *,
    scientific_pass=True,
    uses_test_queries=False,
    algorithm_override=None,
    stage_a_pass=True,
):
    scene_root = root / scene
    scene_root.mkdir()
    contract = {
        "scene": scene,
        **cross_scene_cli.SCENE_CONTRACTS[scene],
    }

    artifacts = {}
    for name in (
        "query_cache",
        "pair_proposals",
        "pair_match_probe",
        "pair_selection",
        "control_factor",
        "variant_factor",
        "manifest",
        "frozen_track_payload",
        "source_cache",
        "parent_manifest",
        "equivalence_report",
    ):
        path = scene_root / f"{name}.bin"
        path.write_bytes(f"{scene}:{name}\n".encode())
        artifacts[name] = path

    query_ref = _ref(
        artifacts["query_cache"],
        mapping_scope={
            "mode": "query_cache_explicit_mapping_only",
            "uses_test_queries": False,
        },
    )
    proposal_ref = _ref(artifacts["pair_proposals"], content_sha256="1" * 64)
    probe_ref = _ref(artifacts["pair_match_probe"], content_sha256="2" * 64)
    selection_ref = _ref(artifacts["pair_selection"], content_sha256="3" * 64)
    stage_a_gates = {name: True for name in cross_scene_cli.STAGE_A_GATE_NAMES}
    if not stage_a_pass:
        stage_a_gates["verified_fisher_utility_improves_5pct"] = False
    stage_a_path = scene_root / "stage_a_gate.json"
    _write_json(
        stage_a_path,
        {
            "schema": "lafgs_cycle_verified_fisher_stage_a_gate",
            "version": 1,
            "valid": True,
            "mapping_only": True,
            "uses_test_queries": False,
            "policy": POLICY_NAME,
            "scene_contract": contract,
            "control_subset": "attested_nearest_pairs_from_same_probe",
            "gates": stage_a_gates,
            "stage_a_passed": stage_a_pass,
            "advance_to_reuse_only_track_build": stage_a_pass,
            "decision": (
                "GO_TO_TRACK_REUSE" if stage_a_pass else "STOP_BEFORE_TRACK_REUSE"
            ),
            "inputs": {
                "query_cache": query_ref,
                "pair_proposals": proposal_ref,
                "pair_match_probe": probe_ref,
                "pair_selection": selection_ref,
            },
        },
    )
    stage_a_ref = _ref(stage_a_path)
    frozen_ref = _ref(artifacts["frozen_track_payload"])
    rebind = {
        "schema": "lafgs_equivalent_query_cache_rebind",
        "version": 1,
        "uses_test_queries": False,
        "equivalence_report": _ref(artifacts["equivalence_report"]),
        "parent_manifest": _ref(artifacts["parent_manifest"]),
        "refreshed_cache": _ref(artifacts["query_cache"]),
        "source_cache": _ref(artifacts["source_cache"]),
        "source_track_payload": frozen_ref,
    }

    def track_report(role):
        is_control = role == "control"
        factor_name = "control_factor" if is_control else "variant_factor"
        factor_ref = _ref(artifacts[factor_name])
        pair_policy = CONTROL_POLICY_NAME if is_control else POLICY_NAME
        subset_role = (
            "attested_nearest_same_probe_control"
            if is_control
            else "cycle_verified_fisher_selection"
        )
        report_path = scene_root / f"{role}_report.json"
        science = _science_contract(contract, algorithm_override=algorithm_override)
        metrics = (
            {
                "triangulated_tracks": 100.0,
                "broad_eligible_tracks": 90.0,
                "high_confidence_tracks": 10.0,
                "triangulated_covariance_p90_m2": 0.1,
                "mapping_query_with_broad_track_fraction": 0.9,
            }
            if is_control
            else {
                "triangulated_tracks": 110.0 if scientific_pass else 90.0,
                "broad_eligible_tracks": 100.0,
                "high_confidence_tracks": 11.0,
                "triangulated_covariance_p90_m2": 0.09,
                "mapping_query_with_broad_track_fraction": 0.9,
            }
        )
        _write_json(
            report_path,
            {
                "schema": "lafgs_pair_policy_track_factor",
                "version": 1,
                "uses_test_queries": False,
                "reuse_only": True,
                "pair_policy": pair_policy,
                "scene_contract": contract,
                "mapping_keypoint_factor": contract["mapping_keypoints"],
                "mapping_nms_radius": contract["nms_radius"],
                "exact_pair_budget": contract["pair_budget"],
                "artifact": factor_ref["path"],
                "artifact_sha256": factor_ref["sha256"],
                "probe_matcher": copy.deepcopy(cross_scene_cli.MATCHER_CONTRACT),
                "pair_policy_parameters": {
                    "pair_subset_role": subset_role,
                    "probe_matcher": copy.deepcopy(cross_scene_cli.MATCHER_CONTRACT),
                    "reuse_only": True,
                    "track_science_contract": science,
                },
                "track": {
                    "triangulated_track_count": metrics["triangulated_tracks"],
                    "broad_eligible_track_count": metrics["broad_eligible_tracks"],
                    "high_confidence_track_count": metrics["high_confidence_tracks"],
                    "mapping_query_with_broad_track_fraction": metrics[
                        "mapping_query_with_broad_track_fraction"
                    ],
                    "triangulated_covariance_trace_m2": {
                        "p90": metrics["triangulated_covariance_p90_m2"]
                    },
                },
                "inputs": {
                    "equivalent_query_cache_rebind": copy.deepcopy(rebind),
                    "manifest": _ref(artifacts["manifest"]),
                    "frozen_track_payload": frozen_ref,
                    "query_cache": query_ref,
                    "pair_proposals": proposal_ref,
                    "pair_match_probe": probe_ref,
                    "pair_selection": selection_ref,
                    "stage_a_gate": stage_a_ref,
                    "pair_subset_role": subset_role,
                    "probe_matcher": copy.deepcopy(cross_scene_cli.MATCHER_CONTRACT),
                },
            },
        )
        return factor_ref, _ref(report_path), metrics

    control_factor_ref, control_report_ref, control_metrics = track_report("control")
    variant_factor_ref, variant_report_ref, variant_metrics = track_report("variant")
    stage_b_gates = {name: True for name in cross_scene_cli.STAGE_B_GATE_NAMES}
    if not scientific_pass:
        stage_b_gates["triangulated_tracks_retain_98pct"] = False
    gate_path = scene_root / "stage_b_gate.json"
    _write_json(
        gate_path,
        {
            "schema": "lafgs_cycle_verified_fisher_mechanism_gate",
            "version": 2,
            "valid": True,
            "mapping_only": True,
            "uses_test_queries": uses_test_queries,
            "policy": POLICY_NAME,
            "scene_contract": contract,
            "stage_a": {
                "gate_path": stage_a_ref["path"],
                "gate_sha256": stage_a_ref["sha256"],
                "passed": True,
            },
            "stage_b": {
                "comparisons": {
                    name: cross_scene_cli._comparison(
                        control_metrics[name], variant_metrics[name]
                    )
                    for name in cross_scene_cli.STAGE_B_COMPARISON_NAMES
                },
                "gates": stage_b_gates,
                "passed": scientific_pass,
            },
            "scene_specific_mechanism_pass": scientific_pass,
            "requires_other_scene": True,
            "decision": (
                "SCENE_PASS_REQUIRES_OTHER_SCENE"
                if scientific_pass
                else "STOP_SCENE_MECHANISM"
            ),
            "inputs": {
                "query_cache": query_ref,
                "pair_proposals": proposal_ref,
                "pair_match_probe": probe_ref,
                "pair_selection": selection_ref,
                "stage_a_gate": stage_a_ref,
                "control_factor": control_factor_ref,
                "control_report": control_report_ref,
                "variant_factor": variant_factor_ref,
                "variant_report": variant_report_ref,
            },
        },
    )
    return {
        "gate": gate_path,
        "sha256": sha256_file(gate_path),
        "artifacts": artifacts,
    }


def _arguments(stairs, greatcourt, output):
    return [
        "--stairs-stage-b-gate",
        str(stairs["gate"]),
        "--expected-stairs-stage-b-gate-sha256",
        stairs["sha256"],
        "--greatcourt-stage-b-gate",
        str(greatcourt["gate"]),
        "--expected-greatcourt-stage-b-gate-sha256",
        greatcourt["sha256"],
        "--output",
        str(output),
    ]


def test_cross_scene_gate_authorizes_mapping_only_stage_c(tmp_path):
    stairs = _make_scene_gate(tmp_path, "stairs")
    greatcourt = _make_scene_gate(tmp_path, "greatcourt")
    output = tmp_path / "cross_scene_gate.json"
    report = cross_scene_cli.run(
        cross_scene_cli.build_parser().parse_args(
            _arguments(stairs, greatcourt, output)
        )
    )
    persisted = json.loads(output.read_text())
    assert report["decision"] == "GO_TO_FULLCHAIN_MAPPING_POSE"
    assert persisted["cross_scene_mechanism_pass"] is True
    assert persisted["advance_to_fullchain_mapping_pose"] is True
    assert persisted["authorizes_test"] is False
    assert persisted["mapping_only"] is True
    assert set(persisted["scene_gates"]) == {"stairs", "greatcourt"}
    assert persisted["scene_gates"]["stairs"]["embedded_stage_a_passed"] is True
    assert (
        persisted["scene_gates"]["greatcourt"]["track_inputs_and_reuse_lineage_valid"]
        is True
    )


def test_cross_scene_scientific_stop_persists_and_exits_two(tmp_path):
    stairs = _make_scene_gate(tmp_path, "stairs")
    greatcourt = _make_scene_gate(tmp_path, "greatcourt", scientific_pass=False)
    output = tmp_path / "cross_scene_stop.json"
    with pytest.raises(SystemExit) as error:
        cross_scene_cli.entrypoint(_arguments(stairs, greatcourt, output))
    assert error.value.code == 2
    persisted = json.loads(output.read_text())
    assert persisted["decision"] == "STOP_BEFORE_FULLCHAIN"
    assert persisted["cross_scene_mechanism_pass"] is False
    assert persisted["advance_to_fullchain_mapping_pose"] is False
    assert persisted["authorizes_test"] is False


def test_cross_scene_rejects_test_query_contamination_without_output(tmp_path):
    stairs = _make_scene_gate(tmp_path, "stairs")
    greatcourt = _make_scene_gate(tmp_path, "greatcourt", uses_test_queries=True)
    output = tmp_path / "contaminated.json"
    with pytest.raises(SystemExit) as error:
        cross_scene_cli.entrypoint(_arguments(stairs, greatcourt, output))
    assert error.value.code == 1
    assert not output.exists()


def test_cross_scene_rejects_mutated_embedded_input_without_output(tmp_path):
    stairs = _make_scene_gate(tmp_path, "stairs")
    greatcourt = _make_scene_gate(tmp_path, "greatcourt")
    greatcourt["artifacts"]["query_cache"].write_bytes(b"mutated\n")
    output = tmp_path / "mutated.json"
    with pytest.raises(SystemExit) as error:
        cross_scene_cli.entrypoint(_arguments(stairs, greatcourt, output))
    assert error.value.code == 1
    assert not output.exists()


def test_cross_scene_rejects_duplicate_scene_gate_without_output(tmp_path):
    stairs = _make_scene_gate(tmp_path, "stairs")
    output = tmp_path / "duplicate.json"
    with pytest.raises(SystemExit) as error:
        cross_scene_cli.entrypoint(_arguments(stairs, stairs, output))
    assert error.value.code == 1
    assert not output.exists()


def test_cross_scene_rejects_different_compiled_policy_without_output(tmp_path):
    stairs = _make_scene_gate(tmp_path, "stairs")
    greatcourt = _make_scene_gate(
        tmp_path,
        "greatcourt",
        algorithm_override=("allow_chain_tracks", False),
    )
    output = tmp_path / "different_policy.json"
    with pytest.raises(SystemExit) as error:
        cross_scene_cli.entrypoint(_arguments(stairs, greatcourt, output))
    assert error.value.code == 1
    assert not output.exists()


def test_cross_scene_rejects_nonpassing_embedded_stage_a_without_output(tmp_path):
    stairs = _make_scene_gate(tmp_path, "stairs")
    greatcourt = _make_scene_gate(tmp_path, "greatcourt", stage_a_pass=False)
    output = tmp_path / "bad_stage_a.json"
    with pytest.raises(SystemExit) as error:
        cross_scene_cli.entrypoint(_arguments(stairs, greatcourt, output))
    assert error.value.code == 1
    assert not output.exists()
