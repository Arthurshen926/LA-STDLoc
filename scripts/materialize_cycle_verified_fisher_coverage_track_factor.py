#!/usr/bin/env python3
"""Build the paired P8 coverage-V2 Track factors from one frozen probe."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from typing import Sequence
import uuid

import torch

from common.hashing import sha256_file
from evidence.cycle_verified_fisher import (
    probe_pair_subset_track_build_inputs,
)
from evidence import triangulation
from scripts.cycle_verified_fisher_cli_common import (
    atomic_json_save,
    atomic_torch_save,
    selection_pairs,
)
from scripts.cycle_verified_fisher_coverage_track_common import (
    CONTROL_POLICY_NAME,
    CONTROL_SUBSET_ROLE,
    VARIANT_POLICY_NAME,
    VARIANT_SUBSET_ROLE,
    artifact_reference,
    completion_artifact_names,
    frozen_track_lineage,
    implementation_registry,
    load_scene_inputs,
    reference_registry_unchanged,
    require_clean_identity,
    track_producer_identity,
)
from scripts.run_track_pair_factor import (
    _build_report,
    _factor_payload,
    _image_hw,
    _load,
    _sample_depth_at_keypoints,
    _track_report,
    _validate_expected_factor_contract,
    _validate_factor_input_lineage,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("stairs", "greatcourt"), required=True)
    parser.add_argument("--cross-scene-stage-a-gate", type=Path, required=True)
    parser.add_argument(
        "--expected-cross-scene-stage-a-gate-sha256", required=True
    )
    parser.add_argument("--scene-stage-a-gate", type=Path, required=True)
    parser.add_argument("--expected-scene-stage-a-gate-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--frozen-track-payload", type=Path, required=True)
    parser.add_argument("--expected-frozen-track-payload-sha256", required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--expected-query-cache-sha256", required=True)
    parser.add_argument("--mapping-scope-equivalence", type=Path, required=True)
    parser.add_argument(
        "--expected-mapping-scope-equivalence-sha256", required=True
    )
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--expected-proposals-sha256", required=True)
    parser.add_argument("--expected-proposals-content-sha256", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--expected-probe-sha256", required=True)
    parser.add_argument("--expected-probe-content-sha256", required=True)
    parser.add_argument("--verified-cycle-table", type=Path, required=True)
    parser.add_argument("--expected-verified-cycle-table-sha256", required=True)
    parser.add_argument(
        "--expected-verified-cycle-table-content-sha256", required=True
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--expected-selection-content-sha256", required=True)
    parser.add_argument("--expected-query-names-sha256", required=True)
    parser.add_argument("--expected-mapping-keypoints", type=int, required=True)
    parser.add_argument("--expected-nms-radius", type=int, required=True)
    parser.add_argument("--expected-pair-budget", type=int, required=True)
    parser.add_argument("--expected-candidate-pair-count", type=int, required=True)
    parser.add_argument("--expected-candidate-components", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--greatcourt-stage-b-gate", type=Path)
    parser.add_argument("--expected-greatcourt-stage-b-gate-sha256")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def _required(manifest: dict, name: str, cast):
    if name not in manifest:
        raise ValueError(f"P8 Track manifest lacks required argument {name}")
    value = manifest[name]
    if cast is bool:
        if not isinstance(value, bool):
            raise ValueError(f"P8 Track manifest argument {name} must be boolean")
        return value
    return cast(value)


def _track_science_contract(
    *,
    manifest: dict,
    matcher: dict,
    mapping_keypoints: int,
    nms_radius: int,
    pair_budget: int,
) -> dict:
    return {
        "mapping_keypoints": int(mapping_keypoints),
        "mapping_nms_radius": int(nms_radius),
        "exact_pair_budget": int(pair_budget),
        "pair_neighbors": _required(
            manifest, "geometry_teacher_track_pair_neighbors", int
        ),
        "minimum_baseline_m": _required(
            manifest, "geometry_teacher_track_min_baseline_m", float
        ),
        "maximum_baseline_m": _required(
            manifest, "geometry_teacher_track_max_baseline_m", float
        ),
        "maximum_axis_angle_deg": _required(
            manifest, "geometry_teacher_track_max_axis_angle_deg", float
        ),
        "matcher": deepcopy(matcher),
        "minimum_track_views": _required(
            manifest, "geometry_teacher_min_views", int
        ),
        "require_cycle": _required(
            manifest, "geometry_teacher_track_require_cycle", bool
        ),
        "allow_chain_tracks": _required(
            manifest, "geometry_teacher_track_allow_chain_tracks", bool
        ),
        "view_bins": _required(manifest, "geometry_teacher_view_bins", int),
        "view_direction_weight": _required(
            manifest, "geometry_teacher_view_direction_weight", float
        ),
        "maximum_observations_per_landmark": _required(
            manifest, "geometry_teacher_max_observations_per_landmark", int
        ),
        "minimum_view_bins": _required(
            manifest, "geometry_teacher_min_view_bins", int
        ),
        "huber_delta_px": _required(
            manifest, "geometry_teacher_huber_delta_px", float
        ),
        "triangulation_iterations": _required(
            manifest, "geometry_teacher_iterations", int
        ),
        "minimum_parallax_deg": _required(
            manifest, "geometry_teacher_min_parallax_deg", float
        ),
        "parallax_quantile": _required(
            manifest, "geometry_teacher_parallax_quantile", float
        ),
        "maximum_reprojection_px": _required(
            manifest, "geometry_teacher_max_reprojection_px", float
        ),
        "maximum_condition_number": _required(
            manifest, "geometry_teacher_max_condition_number", float
        ),
        "maximum_covariance_trace_m2": _required(
            manifest, "geometry_teacher_max_covariance_trace_m2", float
        ),
        "maximum_rendered_depth_residual_m": _required(
            manifest, "geometry_teacher_max_rendered_depth_residual_m", float
        ),
        "minimum_rendered_depth_observations": _required(
            manifest, "geometry_teacher_min_rendered_depth_observations", int
        ),
        "surface_support_enabled": False,
        "depth_sampling": "native_depth_at_sparse_keypoints_or_nearest_pixel_v1",
    }


def _build_arm(
    *,
    role: str,
    pair_policy: str,
    subset_role: str,
    pairs: list[tuple[int, int]],
    registry: dict,
    manifest: dict,
    frozen: dict,
    base_lineage: dict,
    run_uuid: str,
    producer: dict,
    descriptors: list[torch.Tensor],
    keypoints: list[torch.Tensor],
    scores: list[torch.Tensor],
    intrinsics: torch.Tensor,
    poses: torch.Tensor,
    image_hw: torch.Tensor,
    depth_at_keypoints: list[torch.Tensor],
    keypoint_counts: torch.Tensor,
    mapping_keypoints: int,
    nms_radius: int,
    pair_budget: int,
    device: str,
) -> tuple[dict, dict]:
    started = time.perf_counter()
    reuse_inputs = probe_pair_subset_track_build_inputs(
        registry["pair_match_probe"]["payload"], pairs
    )
    if reuse_inputs["precomputed_pairs"] != pairs or len(pairs) != pair_budget:
        raise ValueError(f"P8 {role} Track subset violates the exact pair contract")
    matcher = registry["pair_match_probe"]["payload"]["matcher"]
    science = _track_science_contract(
        manifest=manifest,
        matcher=matcher,
        mapping_keypoints=mapping_keypoints,
        nms_radius=nms_radius,
        pair_budget=pair_budget,
    )
    shared_build_kwargs = {
        "descriptors": descriptors,
        "keypoints": keypoints,
        "detector_scores": scores,
        "camera_K": intrinsics,
        "pose_w2c": poses,
        "pair_neighbors": science["pair_neighbors"],
        "pair_budget": pair_budget,
        "pair_image_hw": image_hw,
        "pair_scene_points_xyz": None,
        "minimum_baseline_m": science["minimum_baseline_m"],
        "maximum_baseline_m": science["maximum_baseline_m"],
        "maximum_axis_angle_deg": science["maximum_axis_angle_deg"],
        "minimum_similarity": matcher["minimum_similarity"],
        "minimum_margin": matcher["minimum_margin"],
        "maximum_epipolar_error_px": matcher["maximum_epipolar_error_px"],
        "epipolar_candidate_topk": matcher["epipolar_candidate_topk"],
        "epipolar_recovered_minimum_similarity": matcher[
            "epipolar_recovered_minimum_similarity"
        ],
        "epipolar_recovered_minimum_margin": matcher[
            "epipolar_recovered_minimum_margin"
        ],
        "minimum_track_views": science["minimum_track_views"],
        "require_cycle": science["require_cycle"],
        "allow_chain_tracks": science["allow_chain_tracks"],
        "return_pair_sidecar": True,
        "device": device,
    }

    def forbidden(*_args, **_kwargs):
        raise RuntimeError("P8 reuse-only Track forbids matcher/pair-selector re-entry")

    original_matcher = triangulation.reciprocal_epipolar_matches
    original_selector = triangulation.candidate_camera_pairs
    triangulation.reciprocal_epipolar_matches = forbidden
    triangulation.candidate_camera_pairs = forbidden
    try:
        tracks, diagnostics, sidecar = triangulation.build_cycle_consistent_tracks(
            pair_policy=pair_policy,
            **shared_build_kwargs,
            **reuse_inputs,
        )
    finally:
        triangulation.reciprocal_epipolar_matches = original_matcher
        triangulation.candidate_camera_pairs = original_selector
    if (
        diagnostics.get("track_pair_matches_reused") != 1
        or diagnostics.get("track_camera_pair_policy") != pair_policy
        or diagnostics.get("track_camera_pair_budget") != pair_budget
        or sidecar.get("policy", {}).get("name") != pair_policy
        or sidecar.get("policy", {}).get("exact_pair_budget") != pair_budget
        or sidecar.get("policy", {}).get("uses_precomputed_pair_matches") is not True
    ):
        raise RuntimeError(f"P8 {role} Track did not attest exact probe reuse")

    observation_query = tracks["query_index"]
    observation_keypoint = tracks["keypoint_index"]
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), keypoint_counts.cumsum(0)))
    all_keypoints = torch.cat(keypoints)
    observation_uv = all_keypoints[offsets[observation_query] + observation_keypoint]
    rendered_depth = torch.empty(observation_query.numel(), dtype=torch.float32)
    order = torch.argsort(observation_query, stable=True)
    counts = torch.bincount(observation_query, minlength=len(registry["query_cache"]["names"]))
    observation_offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    for query in range(len(registry["query_cache"]["names"])):
        begin = int(observation_offsets[query])
        end = int(observation_offsets[query + 1])
        if begin == end:
            continue
        rows = order[begin:end]
        rendered_depth[rows] = depth_at_keypoints[query][observation_keypoint[rows]]
    query_bins = triangulation.camera_pose_bins(
        poses,
        science["view_bins"],
        direction_weight=science["view_direction_weight"],
    )
    geometry = triangulation.robust_triangulate_associations(
        landmark_count=int(diagnostics["track_count"]),
        landmark_index=tracks["track_index"],
        query_index=observation_query,
        uv=observation_uv,
        confidence=tracks["confidence"],
        camera_K=intrinsics,
        pose_w2c=poses,
        query_bin=query_bins,
        rendered_depth=rendered_depth,
        maximum_observations_per_landmark=science[
            "maximum_observations_per_landmark"
        ],
        minimum_views=science["minimum_track_views"],
        minimum_view_bins=science["minimum_view_bins"],
        huber_delta_px=science["huber_delta_px"],
        iterations=science["triangulation_iterations"],
        minimum_parallax_deg=science["minimum_parallax_deg"],
        parallax_quantile=science["parallax_quantile"],
        maximum_reprojection_px=science["maximum_reprojection_px"],
        maximum_condition_number=science["maximum_condition_number"],
        maximum_covariance_trace_m2=science["maximum_covariance_trace_m2"],
        maximum_rendered_depth_residual_m=science[
            "maximum_rendered_depth_residual_m"
        ],
        minimum_rendered_depth_observations=science[
            "minimum_rendered_depth_observations"
        ],
        surface_support_enabled=False,
    )
    geometry["track_confidence_level"] = tracks["track_level"].clone()
    triangulation.attach_pair_triangulation_statistics(
        sidecar, tracks, geometry, poses
    )
    lineage = frozen_track_lineage(registry, base_lineage)
    lineage.update(
        {
            "pair_subset_role": subset_role,
            "paired_run_uuid": run_uuid,
            "track_producer_identity": deepcopy(producer),
        }
    )
    factor = _factor_payload(
        mapping_keypoints=mapping_keypoints,
        nms_radius=nms_radius,
        pair_policy=pair_policy,
        pair_policy_parameters={
            "reuse_only": True,
            "pair_subset_role": subset_role,
            "probe_matcher": deepcopy(matcher),
            "track_science_contract": deepcopy(science),
        },
        query_names=registry["query_cache"]["names"],
        query_bins=query_bins,
        tracks=tracks,
        track_geometry=geometry,
        pair_sidecar=sidecar,
        diagnostics=diagnostics,
        input_lineage=lineage,
    )
    factor["paired_run_uuid"] = run_uuid
    factor["track_producer_identity"] = deepcopy(producer)
    report = _build_report(
        result=factor,
        frozen=frozen,
        sidecar=sidecar,
        keypoint_counts=keypoint_counts,
        scene_point_count=0,
        pair_budget=pair_budget,
        manifest_path=registry["manifest_path"],
        query_cache_path=registry["query_cache"]["path"],
        frozen_track_payload_path=registry["frozen_track_payload_path"],
    )
    report.update(
        {
            "reuse_only": True,
            "probe_matcher": deepcopy(matcher),
            "scene_contract": {
                "scene": registry["scene"],
                "mapping_keypoints": registry["compiled"]["mapping_keypoints"],
                "nms_radius": registry["compiled"]["mapping_nms_radius"],
                "pair_budget": registry["compiled"]["exact_pair_budget"],
                "candidate_pair_count": registry["compiled"]["candidate_pair_count"],
                "candidate_component_count": registry["compiled"][
                    "candidate_component_count"
                ],
            },
            "paired_run_uuid": run_uuid,
            "track_producer_identity": deepcopy(producer),
            "stage_seconds": {"total": float(time.perf_counter() - started)},
        }
    )
    return factor, report


def _completion_inputs(registry: dict) -> dict:
    result = {
        "cross_scene_stage_a_gate": artifact_reference(
            registry["cross_scene_stage_a_gate"]
        ),
        "scene_stage_a_gate": artifact_reference(registry["scene_stage_a_gate"]),
        "query_cache": artifact_reference(registry["query_cache"]),
        "pair_proposals": artifact_reference(registry["pair_proposals"]),
        "pair_match_probe": artifact_reference(registry["pair_match_probe"]),
        "verified_cycle_table": artifact_reference(registry["verified_cycle_table"]),
        "pair_selection": artifact_reference(registry["pair_selection"]),
        "manifest": {
            "path": str(registry["manifest_path"]),
            "sha256": registry["manifest_sha256"],
        },
        "frozen_track_payload": {
            "path": str(registry["frozen_track_payload_path"]),
            "sha256": registry["frozen_track_payload_sha256"],
        },
    }
    if "greatcourt_stage_b_parent" in registry:
        result["greatcourt_stage_b_parent"] = artifact_reference(
            registry["greatcourt_stage_b_parent"]
        )
    return result


def run(args: argparse.Namespace) -> dict:
    reviewed_implementation = implementation_registry()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(
            "Paired Track output root must not exist; isolate any failed root and rerun"
        )
    greatcourt_parent = None
    if args.scene == "stairs":
        if (
            args.greatcourt_stage_b_gate is None
            or args.expected_greatcourt_stage_b_gate_sha256 is None
        ):
            raise ValueError(
                "Stairs Track requires the prior GreatCourt V2 Stage-B Pass"
            )
        from scripts.compare_cycle_verified_fisher_coverage_mechanism import (
            validate_stage_b_gate,
        )

        greatcourt_parent = validate_stage_b_gate(
            scene="greatcourt",
            path=args.greatcourt_stage_b_gate,
            expected_sha256=args.expected_greatcourt_stage_b_gate_sha256,
        )
        if greatcourt_parent["payload"]["scene_specific_mechanism_pass"] is not True:
            raise ValueError("GreatCourt V2 Stage-B STOP forbids Stairs Track")
    elif (
        args.greatcourt_stage_b_gate is not None
        or args.expected_greatcourt_stage_b_gate_sha256 is not None
    ):
        raise ValueError("GreatCourt runner must not name itself as a prior parent")
    registry = load_scene_inputs(
        scene=args.scene,
        cross_scene_stage_a_gate=args.cross_scene_stage_a_gate,
        expected_cross_scene_stage_a_gate_sha256=(
            args.expected_cross_scene_stage_a_gate_sha256
        ),
        scene_stage_a_gate=args.scene_stage_a_gate,
        expected_scene_stage_a_gate_sha256=args.expected_scene_stage_a_gate_sha256,
        manifest=args.manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        frozen_track_payload=args.frozen_track_payload,
        expected_frozen_track_payload_sha256=(
            args.expected_frozen_track_payload_sha256
        ),
        query_cache=args.query_cache,
        expected_query_cache_sha256=args.expected_query_cache_sha256,
        mapping_scope_equivalence=args.mapping_scope_equivalence,
        expected_mapping_scope_equivalence_sha256=(
            args.expected_mapping_scope_equivalence_sha256
        ),
        proposals=args.proposals,
        expected_proposals_sha256=args.expected_proposals_sha256,
        expected_proposals_content_sha256=args.expected_proposals_content_sha256,
        probe=args.probe,
        expected_probe_sha256=args.expected_probe_sha256,
        expected_probe_content_sha256=args.expected_probe_content_sha256,
        verified_cycle_table=args.verified_cycle_table,
        expected_verified_cycle_table_sha256=(
            args.expected_verified_cycle_table_sha256
        ),
        expected_verified_cycle_table_content_sha256=(
            args.expected_verified_cycle_table_content_sha256
        ),
        selection=args.selection,
        expected_selection_sha256=args.expected_selection_sha256,
        expected_selection_content_sha256=args.expected_selection_content_sha256,
        expected_query_names_sha256=args.expected_query_names_sha256,
        expected_mapping_keypoints=args.expected_mapping_keypoints,
        expected_nms_radius=args.expected_nms_radius,
        expected_pair_budget=args.expected_pair_budget,
        expected_candidate_pair_count=args.expected_candidate_pair_count,
        expected_candidate_components=args.expected_candidate_components,
    )
    if args.scene == "stairs":
        registry["greatcourt_stage_b_parent"] = greatcourt_parent
    manifest_payload = json.loads(registry["manifest_path"].read_text())
    base_lineage = _validate_factor_input_lineage(
        manifest_payload=manifest_payload,
        manifest_path=registry["manifest_path"],
        query_cache_path=registry["query_cache"]["path"],
        frozen_track_payload_path=registry["frozen_track_payload_path"],
        expected_manifest_sha256=registry["manifest_sha256"],
        expected_query_cache_sha256=registry["query_cache"]["sha256"],
        expected_frozen_track_payload_sha256=(
            registry["frozen_track_payload_sha256"]
        ),
    )
    frozen = _load(registry["frozen_track_payload_path"])
    if frozen.get("schema") != "lafgs_track_first_payload" or [
        str(value) for value in frozen.get("query_names", [])
    ] != registry["query_cache"]["names"]:
        raise ValueError("Frozen Track payload differs from the mapping query registry")
    manifest = dict(manifest_payload.get("arguments", {}))
    mapping_keypoints, pair_budget, nms_radius = _validate_expected_factor_contract(
        expected_mapping_keypoints=args.expected_mapping_keypoints,
        expected_nms_radius=args.expected_nms_radius,
        expected_pair_budget=args.expected_pair_budget,
        manifest=manifest,
        query_cache_payload=registry["query_cache"]["payload"],
        frozen_track_payload=frozen,
    )
    records = registry["query_cache"]["payload"].get(
        "queries", registry["query_cache"]["payload"]
    )
    keypoints = registry["query_cache"]["keypoints"]
    image_hw = []
    depth_at_keypoints = []
    for name, uv in zip(registry["query_cache"]["names"], keypoints):
        record = records[name]
        depth_source = record.get(
            "native_depth_at_keypoints", record.get("native_depth")
        )
        if depth_source is None:
            raise ValueError(f"Mapping query {name} lacks native depth")
        image_hw.append(_image_hw(record))
        depth_at_keypoints.append(
            _sample_depth_at_keypoints(depth_source, uv).float()
        )
    image_hw = torch.stack(image_hw)
    keypoint_counts = torch.as_tensor(
        [int(value.shape[0]) for value in keypoints], dtype=torch.long
    )
    producer = track_producer_identity(args.device)
    require_clean_identity(producer, label="P8 coverage-V2 Track producer")
    run_uuid = uuid.uuid4().hex
    arms = {
        "control": (
            CONTROL_POLICY_NAME,
            CONTROL_SUBSET_ROLE,
            registry["pair_proposals"]["nearest_pairs"],
        ),
        "variant": (
            VARIANT_POLICY_NAME,
            VARIANT_SUBSET_ROLE,
            selection_pairs(registry["pair_selection"]["payload"]),
        ),
    }
    output_root.mkdir(parents=True, exist_ok=False)
    artifact_records = {}
    summaries = {}
    stems = completion_artifact_names()
    for role in ("control", "variant"):
        policy, subset_role, pairs = arms[role]
        factor, report = _build_arm(
            role=role,
            pair_policy=policy,
            subset_role=subset_role,
            pairs=pairs,
            registry=registry,
            manifest=manifest,
            frozen=frozen,
            base_lineage=base_lineage,
            run_uuid=run_uuid,
            producer=producer,
            descriptors=registry["query_cache"]["descriptors"],
            keypoints=keypoints,
            scores=registry["query_cache"]["scores"],
            intrinsics=registry["query_cache"]["camera_K"],
            poses=registry["query_cache"]["pose_w2c"],
            image_hw=image_hw,
            depth_at_keypoints=depth_at_keypoints,
            keypoint_counts=keypoint_counts,
            mapping_keypoints=mapping_keypoints,
            nms_radius=nms_radius,
            pair_budget=pair_budget,
            device=args.device,
        )
        stem = stems[role][2]
        factor_path = output_root / f"{stem}.pt"
        report_path = output_root / f"{stem}.json"
        atomic_torch_save(factor, factor_path, overwrite=False)
        report["artifact"] = str(factor_path)
        report["artifact_sha256"] = sha256_file(factor_path)
        if report["track"] != _track_report(
            factor["tracks"],
            factor["track_geometry"],
            query_count=len(registry["query_cache"]["names"]),
        ):
            raise RuntimeError(f"P8 {role} Track report failed independent replay")
        atomic_json_save(report, report_path, overwrite=False)
        artifact_records[f"{role}_factor"] = {
            "path": str(factor_path),
            "sha256": sha256_file(factor_path),
        }
        artifact_records[f"{role}_report"] = {
            "path": str(report_path),
            "sha256": sha256_file(report_path),
        }
        summaries[role] = {
            "pair_policy": policy,
            "pair_subset_role": subset_role,
            "track": deepcopy(report["track"]),
        }
        del factor, report
    reference_registry_unchanged(registry)
    if track_producer_identity(args.device) != producer:
        raise RuntimeError("P8 Track producer source identity changed during build")
    for artifact in artifact_records.values():
        if sha256_file(Path(artifact["path"])) != artifact["sha256"]:
            raise RuntimeError("A paired Track output changed before completion")
    completion = {
        "schema": "lafgs_cycle_verified_fisher_coverage_paired_track_completion",
        "version": 1,
        "uses_test_queries": False,
        "mapping_only": True,
        "complete": True,
        "partial": False,
        "resume_allowed": False,
        "scene": args.scene,
        "build_order": ["control", "variant"],
        "run_uuid": run_uuid,
        "track_producer_identity": deepcopy(producer),
        "implementation_registry": {
            "path": str(
                runner_registry_path := (
                    Path(__file__).resolve().parents[1]
                    / "docs/evidence/"
                    "p8_cycle_verified_fisher_coverage_v2_stage_b_implementation.json"
                )
            ),
            "sha256": sha256_file(runner_registry_path),
            "implementation_commit": reviewed_implementation[
                "implementation_commit"
            ],
        },
        "inputs": _completion_inputs(registry),
        "artifacts": artifact_records,
        "summaries": summaries,
        "failure_recovery": (
            "isolate_entire_output_root_and_rebuild_both_arms_from_scratch"
        ),
    }
    completion_path = output_root / "paired_track_completion.json"
    atomic_json_save(completion, completion_path, overwrite=False)
    return {
        "scene": args.scene,
        "completion_manifest": str(completion_path),
        "completion_manifest_sha256": sha256_file(completion_path),
        "run_uuid": run_uuid,
        "build_order": ["control", "variant"],
        "artifacts": artifact_records,
        "summaries": summaries,
    }


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2, sort_keys=True))


def entrypoint(argv: Sequence[str] | None = None) -> None:
    try:
        main(argv)
    except SystemExit:
        raise
    except Exception as error:
        print(
            "ERROR: "
            f"{error}. A created output root is non-resumable; isolate it before retry.",
            file=sys.stderr,
        )
        raise SystemExit(1) from error


if __name__ == "__main__":
    entrypoint()
