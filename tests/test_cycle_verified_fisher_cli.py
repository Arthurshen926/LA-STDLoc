import copy
import hashlib
import json

import pytest
import torch

from common.hashing import sha256_file
from evidence.cycle_verified_fisher import CONTROL_POLICY_NAME, POLICY_NAME
from scripts import attest_cycle_verified_pair_proposals as proposal_cli
from scripts import compare_cycle_verified_fisher_mechanism as stage_b_cli
from scripts import compare_cycle_verified_fisher_stage_a as stage_a_cli
from scripts import compare_cycle_verified_fisher_coverage_stage_a as coverage_stage_a_cli
from scripts import materialize_cycle_verified_triangle_table as verified_table_cli
from scripts import materialize_cycle_verified_pair_probe as probe_cli
from scripts import materialize_cycle_verified_track_factor as track_cli
from scripts import select_cycle_verified_fisher_pairs as select_cli
from scripts import select_cycle_verified_fisher_coverage_pairs as coverage_select_cli
from scripts.cycle_verified_fisher_cli_common import (
    SCENE_CONTRACTS,
    load_mapping_cache,
    torch_load,
)


def _look_at_pose(center, target):
    center = torch.as_tensor(center, dtype=torch.float64)
    target = torch.as_tensor(target, dtype=torch.float64)
    forward = torch.nn.functional.normalize(target - center, dim=0)
    right = torch.nn.functional.normalize(
        torch.cross(
            forward,
            torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
            dim=0,
        ),
        dim=0,
    )
    down = torch.cross(forward, right, dim=0)
    pose = torch.eye(4, dtype=torch.float64)
    pose[:3, :3] = torch.stack((right, down, forward))
    pose[:3, 3] = -(pose[:3, :3] @ center)
    return pose


def _project(points, camera_K, pose):
    camera = torch.einsum("ij,pj->pi", pose[:3, :3], points) + pose[:3, 3]
    pixel = torch.einsum("ij,pj->pi", camera_K, camera)
    return pixel[:, :2] / pixel[:, 2:], camera[:, 2]


def _names_sha256(names):
    return hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest()


def _write_archived_pair_source(path, *, policy, pairs, names):
    payload = {
        "schema": "lafgs_pair_policy_track_factor",
        "version": 1,
        "uses_test_queries": False,
        "mapping_keypoint_factor": 3,
        "descriptor_factor_mutated": False,
        "density_factor_mutated": False,
        "selector_factor_mutated": False,
        "pair_policy": policy,
        "query_names": names,
        # Deliberately no mapping_nms_radius, query_names_sha256 or input_lineage.
        "pair_sidecar": {
            "policy": {"name": policy, "uses_test_queries": False},
            "pair": {
                "left_query_index": torch.tensor([left for left, _ in pairs]),
                "right_query_index": torch.tensor([right for _, right in pairs]),
            },
        },
    }
    torch.save(payload, path)


def _frozen_track_payload(names):
    track_count = 3
    track_index = torch.arange(track_count).repeat(5)
    query_index = torch.arange(5).repeat_interleave(track_count)
    keypoint_index = torch.arange(track_count).repeat(5)
    tracks = {
        "track_index": track_index,
        "query_index": query_index,
        "keypoint_index": keypoint_index,
        "confidence": torch.ones(track_index.numel()),
        "track_level": torch.full((track_count,), 2, dtype=torch.int8),
    }
    geometry = {
        "triangulated": torch.ones(track_count, dtype=torch.bool),
        "triangulated_xyz": torch.tensor(
            [[-0.2, 0.0, 4.0], [0.0, 0.15, 4.2], [0.2, -0.1, 3.8]]
        ),
        "triangulation_observation_count": torch.full((track_count,), 5),
        "triangulation_distinct_view_bin_count": torch.full((track_count,), 3),
        "triangulation_reprojection_median_px": torch.full((track_count,), 0.1),
        "triangulation_reprojection_p90_px": torch.full((track_count,), 0.2),
        "triangulation_parallax_deg": torch.full((track_count,), 2.0),
        "triangulation_covariance_trace": torch.full((track_count,), 0.01),
        "triangulation_high_confidence": torch.ones(track_count, dtype=torch.bool),
        "track_confidence_level": torch.full((track_count,), 2, dtype=torch.int8),
    }
    return {
        "schema": "lafgs_track_first_payload",
        "version": 1,
        "query_names": names,
        "tracks": tracks,
        "track_geometry": geometry,
        "diagnostics": {"track_camera_pair_candidate_count": 5},
    }


@pytest.fixture
def p8_cli_artifacts(tmp_path, monkeypatch):
    monkeypatch.setitem(
        SCENE_CONTRACTS,
        "stairs",
        {
            "mapping_keypoints": 3,
            "nms_radius": 1,
            "pair_budget": 5,
            "candidate_pair_count": 6,
            "candidate_component_count": 1,
        },
    )
    centers = torch.tensor(
        [
            [-2.0, 0.0, 0.0],
            [0.0, 0.15, 0.0],
            [2.0, 0.0, 0.0],
            [2.2, 0.05, 0.0],
            [2.4, -0.05, 0.0],
        ],
        dtype=torch.float64,
    )
    target = torch.tensor([0.0, 0.0, 4.0], dtype=torch.float64)
    poses = torch.stack([_look_at_pose(center, target) for center in centers])
    camera_K = torch.tensor(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    ).repeat(5, 1, 1)
    points = torch.tensor(
        [[-0.2, 0.0, 4.0], [0.0, 0.15, 4.2], [0.2, -0.1, 3.8]],
        dtype=torch.float64,
    )
    projected = [_project(points, camera_K[index], poses[index]) for index in range(5)]
    keypoints = [value[0] for value in projected]
    depths = [value[1] for value in projected]
    names = [f"mapping/{index}.png" for index in range(5)]
    names_sha256 = _names_sha256(names)
    cache_path = tmp_path / "query_cache.pt"
    cache = {
        "version": 3,
        "uses_test_queries": False,
        "signature_payload": {
            "native_sparse_keypoint_count": 3,
            "native_sparse_nms_radius": 1,
        },
        "queries": {
            name: {
                "native_descriptors": torch.eye(3),
                "native_keypoints": keypoints[index].float() - 0.5,
                "native_scores": torch.ones(3),
                "native_K": camera_K[index].float(),
                "pose_w2c": poses[index].float(),
                "native_depth_at_keypoints": depths[index].float(),
                "native_input_hw": torch.tensor([480, 640]),
                "native_sparse_metadata": {
                    "nms_radius": 1,
                    "requested_keypoint_count": 3,
                },
            }
            for index, name in enumerate(names)
        },
    }
    torch.save(cache, cache_path)
    cache_sha256 = sha256_file(cache_path)

    # Nearest completes the weak 2/3/4 triangle; geometry completes strong 0/1/2.
    nearest_pairs = [(0, 2), (1, 2), (2, 3), (2, 4), (3, 4)]
    geometry_pairs = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4)]
    nearest_path = tmp_path / "archived_nearest.pt"
    geometry_path = tmp_path / "archived_geometry.pt"
    _write_archived_pair_source(
        nearest_path, policy="nearest", pairs=nearest_pairs, names=names
    )
    _write_archived_pair_source(
        geometry_path,
        policy="parallax_diverse",
        pairs=geometry_pairs,
        names=names,
    )
    proposals_path = tmp_path / "proposals.pt"
    proposal_cli.main(
        [
            "--scene", "stairs",
            "--query-cache", str(cache_path),
            "--expected-query-cache-sha256", cache_sha256,
            "--nearest-source", str(nearest_path),
            "--expected-nearest-source-sha256", sha256_file(nearest_path),
            "--geometry-source", str(geometry_path),
            "--expected-geometry-source-sha256", sha256_file(geometry_path),
            "--expected-query-names-sha256", names_sha256,
            "--expected-mapping-keypoints", "3",
            "--expected-nms-radius", "1",
            "--expected-pair-budget", "5",
            "--expected-candidate-pair-count", "6",
            "--expected-candidate-components", "1",
            "--output", str(proposals_path),
        ]
    )
    proposals = torch_load(proposals_path)
    proposal_record = {
        "path": proposals_path,
        "sha256": sha256_file(proposals_path),
        "content_sha256": proposals["content_sha256"],
    }
    probe_path = tmp_path / "probe.pt"
    probe_cli.main(
        [
            "--scene", "stairs",
            "--query-cache", str(cache_path),
            "--expected-query-cache-sha256", cache_sha256,
            "--proposals", str(proposals_path),
            "--expected-proposals-sha256", proposal_record["sha256"],
            "--expected-proposals-content-sha256", proposal_record["content_sha256"],
            "--expected-query-names-sha256", names_sha256,
            "--expected-mapping-keypoints", "3",
            "--expected-nms-radius", "1",
            "--expected-pair-budget", "5",
            "--expected-candidate-pair-count", "6",
            "--expected-candidate-components", "1",
            "--minimum-similarity", "0.65",
            "--minimum-margin", "0.01",
            "--maximum-epipolar-error-px", "2.0",
            "--epipolar-candidate-topk", "1",
            "--epipolar-recovered-minimum-similarity", "-1.0",
            "--epipolar-recovered-minimum-margin", "-1.0",
            "--device", "cpu",
            "--output", str(probe_path),
        ]
    )
    probe = torch_load(probe_path)
    probe_record = {
        "path": probe_path,
        "sha256": sha256_file(probe_path),
        "content_sha256": probe["content_sha256"],
    }
    selection_path = tmp_path / "selection.pt"
    selection_args = [
        "--scene", "stairs",
        "--query-cache", str(cache_path),
        "--expected-query-cache-sha256", cache_sha256,
        "--probe", str(probe_path),
        "--expected-probe-sha256", probe_record["sha256"],
        "--expected-probe-content-sha256", probe_record["content_sha256"],
        "--expected-query-names-sha256", names_sha256,
        "--expected-mapping-keypoints", "3",
        "--expected-nms-radius", "1",
        "--expected-pair-budget", "5",
        "--expected-candidate-pair-count", "6",
        "--expected-candidate-components", "1",
        "--minimum-camera-degree", "1",
        "--maximum-cycle-reprojection-error-px", "2.0",
        "--output", str(selection_path),
    ]
    select_cli.main(selection_args)
    selection = torch_load(selection_path)
    selection_record = {
        "path": selection_path,
        "sha256": sha256_file(selection_path),
        "content_sha256": selection["content_sha256"],
    }
    stage_a_path = tmp_path / "stage_a_gate.json"
    stage_a_args = [
        "--scene", "stairs",
        "--query-cache", str(cache_path),
        "--expected-query-cache-sha256", cache_sha256,
        "--proposals", str(proposals_path),
        "--expected-proposals-sha256", proposal_record["sha256"],
        "--expected-proposals-content-sha256", proposal_record["content_sha256"],
        "--probe", str(probe_path),
        "--expected-probe-sha256", probe_record["sha256"],
        "--expected-probe-content-sha256", probe_record["content_sha256"],
        "--selection", str(selection_path),
        "--expected-selection-sha256", selection_record["sha256"],
        "--expected-selection-content-sha256", selection_record["content_sha256"],
        "--expected-query-names-sha256", names_sha256,
        "--expected-mapping-keypoints", "3",
        "--expected-nms-radius", "1",
        "--expected-pair-budget", "5",
        "--expected-candidate-pair-count", "6",
        "--expected-candidate-components", "1",
        "--maximum-cycle-reprojection-error-px", "2.0",
        "--output", str(stage_a_path),
    ]
    stage_a_cli.main(stage_a_args)
    stage_a = json.loads(stage_a_path.read_text())
    assert stage_a["stage_a_passed"] is True

    frozen_path = tmp_path / "frozen_track_payload.pt"
    torch.save(_frozen_track_payload(names), frozen_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "version": 1,
        "arguments": {
            "query_cache_path": str(cache_path),
            "native_keypoint_count": 3,
            "geometry_teacher_track_pair_neighbors": 6,
            "geometry_teacher_track_min_baseline_m": 0.0,
            "geometry_teacher_track_max_baseline_m": 10.0,
            "geometry_teacher_track_max_axis_angle_deg": 180.0,
            "geometry_teacher_min_views": 3,
            "geometry_teacher_track_require_cycle": True,
            "geometry_teacher_track_allow_chain_tracks": True,
            "geometry_teacher_view_bins": 8,
            "geometry_teacher_view_direction_weight": 0.5,
            "geometry_teacher_max_observations_per_landmark": 32,
            "geometry_teacher_min_view_bins": 1,
            "geometry_teacher_huber_delta_px": 2.0,
            "geometry_teacher_iterations": 3,
            "geometry_teacher_min_parallax_deg": 0.0,
            "geometry_teacher_parallax_quantile": 0.75,
            "geometry_teacher_max_reprojection_px": 5.0,
            "geometry_teacher_max_condition_number": 1e12,
            "geometry_teacher_max_covariance_trace_m2": 1e9,
            "geometry_teacher_max_rendered_depth_residual_m": 1e9,
            "geometry_teacher_min_rendered_depth_observations": 0,
        },
        "inputs": {
            "query_cache_path": {
                "path": str(cache_path),
                "sha256": cache_sha256,
            }
        },
    }
    manifest_path.write_text(json.dumps(manifest))
    return {
        "tmp_path": tmp_path,
        "cache_path": cache_path,
        "cache_sha256": cache_sha256,
        "names": names,
        "names_sha256": names_sha256,
        "nearest_path": nearest_path,
        "nearest_pairs": nearest_pairs,
        "proposals_path": proposals_path,
        "proposals": proposals,
        "proposal_record": proposal_record,
        "probe_path": probe_path,
        "probe": probe,
        "probe_record": probe_record,
        "selection_path": selection_path,
        "selection": selection,
        "selection_record": selection_record,
        "selection_args": selection_args,
        "stage_a_path": stage_a_path,
        "stage_a_args": stage_a_args,
        "manifest_path": manifest_path,
        "frozen_path": frozen_path,
    }


def _track_args(artifact, *, arm, output_dir):
    return [
        "--scene", "stairs",
        "--arm", arm,
        "--manifest", str(artifact["manifest_path"]),
        "--expected-manifest-sha256", sha256_file(artifact["manifest_path"]),
        "--frozen-track-payload", str(artifact["frozen_path"]),
        "--expected-frozen-track-payload-sha256", sha256_file(artifact["frozen_path"]),
        "--query-cache", str(artifact["cache_path"]),
        "--expected-query-cache-sha256", artifact["cache_sha256"],
        "--proposals", str(artifact["proposals_path"]),
        "--expected-proposals-sha256", artifact["proposal_record"]["sha256"],
        "--expected-proposals-content-sha256", artifact["proposal_record"]["content_sha256"],
        "--probe", str(artifact["probe_path"]),
        "--expected-probe-sha256", artifact["probe_record"]["sha256"],
        "--expected-probe-content-sha256", artifact["probe_record"]["content_sha256"],
        "--selection", str(artifact["selection_path"]),
        "--expected-selection-sha256", artifact["selection_record"]["sha256"],
        "--expected-selection-content-sha256", artifact["selection_record"]["content_sha256"],
        "--stage-a-gate", str(artifact["stage_a_path"]),
        "--expected-stage-a-gate-sha256", sha256_file(artifact["stage_a_path"]),
        "--expected-query-names-sha256", artifact["names_sha256"],
        "--expected-mapping-keypoints", "3",
        "--expected-nms-radius", "1",
        "--expected-pair-budget", "5",
        "--expected-candidate-pair-count", "6",
        "--expected-candidate-components", "1",
        "--device", "cpu",
        "--output-dir", str(output_dir),
    ]


def _stage_b_args(artifact, *, control_dir, variant_dir, output):
    control_factor = control_dir / f"{CONTROL_POLICY_NAME}_track_factor.pt"
    control_report = control_dir / f"{CONTROL_POLICY_NAME}_track_factor.json"
    variant_factor = variant_dir / f"{POLICY_NAME}_track_factor.pt"
    variant_report = variant_dir / f"{POLICY_NAME}_track_factor.json"
    return [
        "--scene", "stairs",
        "--query-cache", str(artifact["cache_path"]),
        "--expected-query-cache-sha256", artifact["cache_sha256"],
        "--proposals", str(artifact["proposals_path"]),
        "--expected-proposals-sha256", artifact["proposal_record"]["sha256"],
        "--expected-proposals-content-sha256", artifact["proposal_record"]["content_sha256"],
        "--probe", str(artifact["probe_path"]),
        "--expected-probe-sha256", artifact["probe_record"]["sha256"],
        "--expected-probe-content-sha256", artifact["probe_record"]["content_sha256"],
        "--selection", str(artifact["selection_path"]),
        "--expected-selection-sha256", artifact["selection_record"]["sha256"],
        "--expected-selection-content-sha256", artifact["selection_record"]["content_sha256"],
        "--stage-a-gate", str(artifact["stage_a_path"]),
        "--expected-stage-a-gate-sha256", sha256_file(artifact["stage_a_path"]),
        "--control-factor", str(control_factor),
        "--expected-control-factor-sha256", sha256_file(control_factor),
        "--control-report", str(control_report),
        "--expected-control-report-sha256", sha256_file(control_report),
        "--variant-factor", str(variant_factor),
        "--expected-variant-factor-sha256", sha256_file(variant_factor),
        "--variant-report", str(variant_report),
        "--expected-variant-report-sha256", sha256_file(variant_report),
        "--expected-query-names-sha256", artifact["names_sha256"],
        "--expected-mapping-keypoints", "3",
        "--expected-nms-radius", "1",
        "--expected-pair-budget", "5",
        "--expected-candidate-pair-count", "6",
        "--expected-candidate-components", "1",
        "--output", str(output),
    ]


def test_archived_proposal_attestation_is_pair_only(p8_cli_artifacts):
    proposals = p8_cli_artifacts["proposals"]
    assert proposals["source_contract"] == {
        "scope": "archived_pair_tables_only",
        "track_factor_lineage_reused": False,
        "track_or_geometry_measurements_reused": False,
        "fresh_cache_is_authoritative_for_query_order_k_nms": True,
    }
    assert proposals["arms"]["nearest"]["unavailable_source_lineage"] == [
        "input_lineage",
        "mapping_nms_radius",
        "query_names_sha256",
    ]
    assert proposals["candidate_union"]["pair_count"] == 6


def test_cache_without_scope_requires_hash_bound_v2_equivalence(p8_cli_artifacts):
    artifact = p8_cli_artifacts
    cache = torch_load(artifact["cache_path"])
    cache.pop("uses_test_queries")
    cache_path = artifact["tmp_path"] / "scope_missing_cache.pt"
    torch.save(cache, cache_path)
    cache_sha256 = sha256_file(cache_path)
    common = {
        "path": cache_path,
        "expected_file_sha256": cache_sha256,
        "expected_query_names_sha256": artifact["names_sha256"],
        "expected_mapping_keypoints": 3,
        "expected_nms_radius": 1,
    }
    with pytest.raises(ValueError, match="requires a mapping-scope equivalence"):
        load_mapping_cache(**common)
    equivalence_path = artifact["tmp_path"] / "mapping_scope_equivalence.json"
    query_count = len(artifact["names"])
    equivalence = {
        "schema": "lafgs_mapping_sparse_refresh_equivalence",
        "version": 2,
        "uses_test_queries": False,
        "valid": True,
        "checks": {"all_exact": True},
        "expected": {"mapping_keypoints": 3, "nms_radius": 1},
        "audit": {
            "content_equivalent_track_payload_reuse_authorized": True,
            "query_order_exact": True,
            "query_count": query_count,
            "effective_sparse_depth_exact_query_count": query_count,
            "native_alpha_exact_query_count": query_count,
            "refreshed_metadata_pass_count": query_count,
            "track_input_exact_query_count": query_count,
            "target_k_mapping": 3,
            "target_nms_radius": 1,
        },
        "sources": {
            "refreshed_cache": {
                "path": str(cache_path),
                "sha256": cache_sha256,
            }
        },
    }
    equivalence_path.write_text(json.dumps(equivalence))
    loaded = load_mapping_cache(
        **common,
        mapping_scope_equivalence=equivalence_path,
        expected_mapping_scope_equivalence_sha256=sha256_file(equivalence_path),
    )
    assert loaded["mapping_scope"]["mode"] == (
        "mapping_sparse_refresh_equivalence_v2"
    )
    assert loaded["mapping_scope"]["equivalence_report"]["sha256"] == (
        sha256_file(equivalence_path)
    )


def test_stage_a_input_failure_exits_one(p8_cli_artifacts):
    arguments = list(p8_cli_artifacts["selection_args"])
    index = arguments.index("--expected-probe-sha256") + 1
    arguments[index] = "0" * 64
    arguments.append("--overwrite")
    with pytest.raises(SystemExit) as error:
        select_cli.entrypoint(arguments)
    assert error.value.code == 1


def test_stage_a_scientific_stop_persists_and_exits_two(
    p8_cli_artifacts, monkeypatch
):
    original = stage_a_cli._stage_a_gates

    def regression(**kwargs):
        result = original(**kwargs)
        result["verified_fisher_utility_improves_5pct"] = False
        return result

    monkeypatch.setattr(stage_a_cli, "_stage_a_gates", regression)
    arguments = list(p8_cli_artifacts["stage_a_args"])
    output = p8_cli_artifacts["tmp_path"] / "stage_a_stop.json"
    arguments[arguments.index("--output") + 1] = str(output)
    with pytest.raises(SystemExit) as error:
        stage_a_cli.entrypoint(arguments)
    assert error.value.code == 2
    gate = json.loads(output.read_text())
    assert gate["valid"] is True
    assert gate["stage_a_passed"] is False
    assert gate["decision"] == "STOP_BEFORE_TRACK_REUSE"


def test_v2_coverage_cli_hard_constraint_and_scientific_stop_persist(
    p8_cli_artifacts,
    monkeypatch,
):
    artifact = p8_cli_artifacts
    monkeypatch.setattr(
        coverage_stage_a_cli,
        "STAIRS_V1_SELECTION_CONTRACT",
        {
            "sha256": artifact["selection_record"]["sha256"],
            "content_sha256": artifact["selection_record"]["content_sha256"],
        },
    )
    common = [
        "--scene", "stairs",
        "--query-cache", str(artifact["cache_path"]),
        "--expected-query-cache-sha256", artifact["cache_sha256"],
        "--probe", str(artifact["probe_path"]),
        "--expected-probe-sha256", artifact["probe_record"]["sha256"],
        "--expected-probe-content-sha256", artifact["probe_record"]["content_sha256"],
        "--expected-query-names-sha256", artifact["names_sha256"],
        "--expected-mapping-keypoints", "3",
        "--expected-nms-radius", "1",
        "--expected-pair-budget", "5",
        "--expected-candidate-pair-count", "6",
        "--expected-candidate-components", "1",
        "--maximum-cycle-reprojection-error-px", "2.0",
    ]
    verified_path = artifact["tmp_path"] / "verified_table.pt"
    verified_table_cli.main([*common, "--output", str(verified_path)])
    verified = torch_load(verified_path)

    selection_path = artifact["tmp_path"] / "coverage_selection.pt"
    proposal_args = [
        "--proposals", str(artifact["proposals_path"]),
        "--expected-proposals-sha256", artifact["proposal_record"]["sha256"],
        "--expected-proposals-content-sha256", artifact["proposal_record"]["content_sha256"],
    ]
    table_args = [
        "--verified-cycle-table", str(verified_path),
        "--expected-verified-cycle-table-sha256", sha256_file(verified_path),
        "--expected-verified-cycle-table-content-sha256", verified["content_sha256"],
    ]
    coverage_select_cli.main(
        [
            *common,
            *proposal_args,
            *table_args,
            "--minimum-camera-degree", "1",
            "--output", str(selection_path),
        ]
    )
    selection = torch_load(selection_path)
    assert selection["coverage_certificate"]["target_camera_index"] == [2, 3, 4]
    assert selection["coverage_certificate"]["all_target_cameras_covered"] is True

    gate_path = artifact["tmp_path"] / "coverage_stage_a_stop.json"
    arguments = [
        *common,
        *proposal_args,
        *table_args,
        "--selection", str(selection_path),
        "--expected-selection-sha256", sha256_file(selection_path),
        "--expected-selection-content-sha256", selection["content_sha256"],
        "--stairs-v1-selection", str(artifact["selection_path"]),
        "--expected-stairs-v1-selection-sha256", artifact["selection_record"]["sha256"],
        "--expected-stairs-v1-selection-content-sha256", artifact["selection_record"]["content_sha256"],
        "--output", str(gate_path),
    ]
    with pytest.raises(SystemExit) as error:
        coverage_stage_a_cli.entrypoint(arguments)
    assert error.value.code == 2
    gate = json.loads(gate_path.read_text())
    assert gate["valid"] is True
    assert gate["gates"]["control_target_membership_exact"] is True
    assert gate["gates"]["all_control_target_cameras_hard_covered"] is True
    assert gate["gates"]["verified_fisher_utility_improves_5pct"] is False
    assert gate["decision"] == "STOP_BEFORE_TRACK_REUSE"


def test_reuse_only_track_factors_and_stage_b_end_to_end(p8_cli_artifacts):
    artifact = p8_cli_artifacts
    control_dir = artifact["tmp_path"] / "control"
    variant_dir = artifact["tmp_path"] / "variant"
    control = track_cli.run(
        track_cli.build_parser().parse_args(
            _track_args(artifact, arm="nearest_control", output_dir=control_dir)
        )
    )
    variant = track_cli.run(
        track_cli.build_parser().parse_args(
            _track_args(artifact, arm="variant", output_dir=variant_dir)
        )
    )
    assert control["track_pair_matches_reused"] == 1
    assert variant["track_pair_matches_reused"] == 1
    assert control["uses_precomputed_pair_matches"] is True
    assert variant["uses_precomputed_pair_matches"] is True
    output = artifact["tmp_path"] / "stage_b_gate.json"
    arguments = _stage_b_args(
        artifact, control_dir=control_dir, variant_dir=variant_dir, output=output
    )
    try:
        stage_b_cli.main(arguments)
    except SystemExit as error:
        assert error.code == 2
    gate = json.loads(output.read_text())
    assert gate["valid"] is True
    assert gate["version"] == 2
    assert gate["stage_a"]["passed"] is True
    assert gate["stage_b"]["gates"]["control_probe_rows_reused"] is True
    assert gate["stage_b"]["gates"]["variant_probe_rows_reused"] is True
    assert gate["stage_b"]["gates"]["same_probe_matcher_contract"] is True
    assert gate["requires_other_scene"] is True
    assert "advance_to_fullchain_mapping_pose" not in gate
    if gate["scene_specific_mechanism_pass"]:
        assert gate["decision"] == "SCENE_PASS_REQUIRES_OTHER_SCENE"


def test_stage_b_rejects_non_reuse_control_lineage(p8_cli_artifacts):
    artifact = p8_cli_artifacts
    control_dir = artifact["tmp_path"] / "control_bad"
    variant_dir = artifact["tmp_path"] / "variant_bad"
    track_cli.run(
        track_cli.build_parser().parse_args(
            _track_args(artifact, arm="nearest_control", output_dir=control_dir)
        )
    )
    track_cli.run(
        track_cli.build_parser().parse_args(
            _track_args(artifact, arm="variant", output_dir=variant_dir)
        )
    )
    control_factor = control_dir / f"{CONTROL_POLICY_NAME}_track_factor.pt"
    payload = torch.load(control_factor, map_location="cpu", weights_only=False)
    payload = copy.deepcopy(payload)
    payload["diagnostics"]["track_pair_matches_reused"] = 0
    torch.save(payload, control_factor)
    output = artifact["tmp_path"] / "invalid_stage_b.json"
    arguments = _stage_b_args(
        artifact, control_dir=control_dir, variant_dir=variant_dir, output=output
    )
    with pytest.raises(SystemExit) as error:
        stage_b_cli.entrypoint(arguments)
    assert error.value.code == 1
    assert not output.exists()


def test_stage_b_rejects_different_manifest_and_science_contract(
    p8_cli_artifacts,
):
    artifact = p8_cli_artifacts
    control_dir = artifact["tmp_path"] / "control_manifest"
    variant_dir = artifact["tmp_path"] / "variant_manifest"
    track_cli.run(
        track_cli.build_parser().parse_args(
            _track_args(artifact, arm="nearest_control", output_dir=control_dir)
        )
    )
    manifest = json.loads(artifact["manifest_path"].read_text())
    manifest["arguments"]["geometry_teacher_max_condition_number"] = 5e11
    artifact["manifest_path"].write_text(json.dumps(manifest))
    track_cli.run(
        track_cli.build_parser().parse_args(
            _track_args(artifact, arm="variant", output_dir=variant_dir)
        )
    )
    output = artifact["tmp_path"] / "different_manifest_stage_b.json"
    arguments = _stage_b_args(
        artifact, control_dir=control_dir, variant_dir=variant_dir, output=output
    )
    with pytest.raises(SystemExit) as error:
        stage_b_cli.entrypoint(arguments)
    assert error.value.code == 1
    assert not output.exists()


def test_stage_b_scientific_stop_persists_and_exits_two(
    p8_cli_artifacts, monkeypatch
):
    artifact = p8_cli_artifacts
    control_dir = artifact["tmp_path"] / "control_stop"
    variant_dir = artifact["tmp_path"] / "variant_stop"
    track_cli.run(
        track_cli.build_parser().parse_args(
            _track_args(artifact, arm="nearest_control", output_dir=control_dir)
        )
    )
    track_cli.run(
        track_cli.build_parser().parse_args(
            _track_args(artifact, arm="variant", output_dir=variant_dir)
        )
    )
    original = stage_b_cli._stage_b_gates

    def regression(**kwargs):
        result = original(**kwargs)
        result["triangulated_tracks_retain_98pct"] = False
        return result

    monkeypatch.setattr(stage_b_cli, "_stage_b_gates", regression)
    output = artifact["tmp_path"] / "stage_b_stop.json"
    arguments = _stage_b_args(
        artifact, control_dir=control_dir, variant_dir=variant_dir, output=output
    )
    with pytest.raises(SystemExit) as error:
        stage_b_cli.entrypoint(arguments)
    assert error.value.code == 2
    gate = json.loads(output.read_text())
    assert gate["valid"] is True
    assert gate["stage_b"]["passed"] is False
    assert gate["decision"] == "STOP_SCENE_MECHANISM"
    assert gate["scene_specific_mechanism_pass"] is False
    assert gate["requires_other_scene"] is True
