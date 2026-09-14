#!/usr/bin/env python3
"""Build one P8 Track factor by reusing a same-probe pair subset only."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from typing import Sequence

import torch

from common.hashing import sha256_file
from evidence.cycle_verified_fisher import (
    CONTROL_POLICY_NAME,
    POLICY_NAME,
    probe_pair_subset_track_build_inputs,
    probe_track_build_inputs,
)
from evidence import triangulation
from scripts.cycle_verified_fisher_cli_common import (
    add_mapping_scope_arguments,
    atomic_json_save,
    atomic_torch_save,
    load_mapping_cache,
    load_probe,
    load_proposals,
    load_selection,
    load_stage_a_gate,
    mapping_scope_kwargs,
    validate_output_target,
    validate_probe_proposal_lineage,
    validate_scene_contract,
)
from scripts.run_track_pair_factor import (
    _build_report,
    _factor_payload,
    _image_hw,
    _load,
    _sample_depth_at_keypoints,
    _validate_expected_factor_contract,
    _validate_factor_input_lineage,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("stairs", "greatcourt"), required=True)
    parser.add_argument("--arm", choices=("nearest_control", "variant"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--frozen-track-payload", type=Path, required=True)
    parser.add_argument("--expected-frozen-track-payload-sha256", required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--expected-query-cache-sha256", required=True)
    add_mapping_scope_arguments(parser)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--expected-proposals-sha256", required=True)
    parser.add_argument("--expected-proposals-content-sha256", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--expected-probe-sha256", required=True)
    parser.add_argument("--expected-probe-content-sha256", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--expected-selection-content-sha256", required=True)
    parser.add_argument("--stage-a-gate", type=Path, required=True)
    parser.add_argument("--expected-stage-a-gate-sha256", required=True)
    parser.add_argument("--expected-query-names-sha256", required=True)
    parser.add_argument("--expected-mapping-keypoints", type=int, required=True)
    parser.add_argument("--expected-nms-radius", type=int, required=True)
    parser.add_argument("--expected-pair-budget", type=int, required=True)
    parser.add_argument("--expected-candidate-pair-count", type=int, required=True)
    parser.add_argument("--expected-candidate-components", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    *, manifest: dict, matcher: dict, mapping_keypoints: int, nms_radius: int,
    pair_budget: int,
) -> dict:
    """Materialize every shared scientific axis used by both Track arms."""
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


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    contract = validate_scene_contract(
        scene=args.scene,
        mapping_keypoints=args.expected_mapping_keypoints,
        nms_radius=args.expected_nms_radius,
        pair_budget=args.expected_pair_budget,
        candidate_pair_count=args.expected_candidate_pair_count,
        candidate_component_count=args.expected_candidate_components,
    )
    cache = load_mapping_cache(
        path=args.query_cache,
        expected_file_sha256=args.expected_query_cache_sha256,
        expected_query_names_sha256=args.expected_query_names_sha256,
        expected_mapping_keypoints=args.expected_mapping_keypoints,
        expected_nms_radius=args.expected_nms_radius,
        **mapping_scope_kwargs(args),
    )
    proposals = load_proposals(
        path=args.proposals,
        expected_file_sha256=args.expected_proposals_sha256,
        expected_content_sha256=args.expected_proposals_content_sha256,
        cache=cache,
        expected_mapping_keypoints=args.expected_mapping_keypoints,
        expected_nms_radius=args.expected_nms_radius,
        expected_pair_budget=args.expected_pair_budget,
        expected_candidate_pair_count=args.expected_candidate_pair_count,
        expected_candidate_components=args.expected_candidate_components,
    )
    probe = load_probe(
        path=args.probe,
        expected_file_sha256=args.expected_probe_sha256,
        expected_content_sha256=args.expected_probe_content_sha256,
        cache=cache,
        expected_mapping_keypoints=args.expected_mapping_keypoints,
        expected_nms_radius=args.expected_nms_radius,
        expected_candidate_pair_count=args.expected_candidate_pair_count,
    )
    validate_probe_proposal_lineage(probe=probe, proposals=proposals)
    selection = load_selection(
        path=args.selection,
        expected_file_sha256=args.expected_selection_sha256,
        expected_content_sha256=args.expected_selection_content_sha256,
        probe=probe,
        expected_pair_budget=args.expected_pair_budget,
    )
    stage_a = load_stage_a_gate(
        path=args.stage_a_gate,
        expected_file_sha256=args.expected_stage_a_gate_sha256,
        cache=cache,
        proposals=proposals,
        probe=probe,
        selection=selection,
        require_go=True,
    )
    manifest_path = args.manifest.resolve()
    manifest_payload = json.loads(manifest_path.read_text())
    input_lineage = _validate_factor_input_lineage(
        manifest_payload=manifest_payload,
        manifest_path=manifest_path,
        query_cache_path=cache["path"],
        frozen_track_payload_path=args.frozen_track_payload,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_query_cache_sha256=cache["sha256"],
        expected_frozen_track_payload_sha256=(
            args.expected_frozen_track_payload_sha256
        ),
    )
    frozen_path = args.frozen_track_payload.resolve()
    frozen = _load(frozen_path)
    if frozen.get("schema") != "lafgs_track_first_payload":
        raise ValueError("Unexpected frozen Track payload schema")
    if [str(value) for value in frozen.get("query_names", [])] != cache["names"]:
        raise ValueError("Frozen Track payload differs from the mapping query order")
    manifest = dict(manifest_payload.get("arguments", {}))
    mapping_k, pair_budget, nms_radius = _validate_expected_factor_contract(
        expected_mapping_keypoints=args.expected_mapping_keypoints,
        expected_nms_radius=args.expected_nms_radius,
        expected_pair_budget=args.expected_pair_budget,
        manifest=manifest,
        query_cache_payload=cache["payload"],
        frozen_track_payload=frozen,
    )

    records = cache["payload"].get("queries", cache["payload"])
    descriptors = cache["descriptors"]
    keypoints = cache["keypoints"]
    scores = cache["scores"]
    intrinsics = cache["camera_K"]
    poses = cache["pose_w2c"]
    image_hw = []
    depth_at_keypoints = []
    for name, uv in zip(cache["names"], keypoints):
        record = records[name]
        depth_source = record.get(
            "native_depth_at_keypoints", record.get("native_depth")
        )
        if depth_source is None:
            raise ValueError(f"Mapping query {name} lacks native depth")
        image_hw.append(_image_hw(record))
        depth_at_keypoints.append(_sample_depth_at_keypoints(depth_source, uv).float())
    image_hw = torch.stack(image_hw)
    keypoint_counts = torch.as_tensor(
        [int(value.shape[0]) for value in keypoints], dtype=torch.long
    )

    if args.arm == "nearest_control":
        pair_policy = CONTROL_POLICY_NAME
        subset_role = "attested_nearest_same_probe_control"
        reuse_inputs = probe_pair_subset_track_build_inputs(
            probe["payload"], proposals["nearest_pairs"]
        )
    else:
        pair_policy = POLICY_NAME
        subset_role = "cycle_verified_fisher_selection"
        reuse_inputs = probe_track_build_inputs(
            probe["payload"], selection["payload"]
        )
    if len(reuse_inputs["precomputed_pairs"]) != pair_budget:
        raise ValueError("P8 Track subset violates the exact pair budget")

    matcher = probe["payload"]["matcher"]
    science_contract = _track_science_contract(
        manifest=manifest,
        matcher=matcher,
        mapping_keypoints=mapping_k,
        nms_radius=nms_radius,
        pair_budget=pair_budget,
    )

    def forbidden_matcher(*_args, **_kwargs):
        raise RuntimeError("P8 reuse-only Track runner forbids matcher re-entry")

    original_matcher = triangulation.reciprocal_epipolar_matches
    triangulation.reciprocal_epipolar_matches = forbidden_matcher
    try:
        tracks, diagnostics, sidecar = triangulation.build_cycle_consistent_tracks(
            descriptors=descriptors,
            keypoints=keypoints,
            detector_scores=scores,
            camera_K=intrinsics,
            pose_w2c=poses,
            pair_neighbors=_required(
                manifest, "geometry_teacher_track_pair_neighbors", int
            ),
            pair_policy=pair_policy,
            pair_budget=pair_budget,
            pair_image_hw=image_hw,
            pair_scene_points_xyz=None,
            minimum_baseline_m=_required(
                manifest, "geometry_teacher_track_min_baseline_m", float
            ),
            maximum_baseline_m=_required(
                manifest, "geometry_teacher_track_max_baseline_m", float
            ),
            maximum_axis_angle_deg=_required(
                manifest, "geometry_teacher_track_max_axis_angle_deg", float
            ),
            minimum_similarity=float(matcher["minimum_similarity"]),
            minimum_margin=float(matcher["minimum_margin"]),
            maximum_epipolar_error_px=float(matcher["maximum_epipolar_error_px"]),
            epipolar_candidate_topk=int(matcher["epipolar_candidate_topk"]),
            epipolar_recovered_minimum_similarity=float(
                matcher["epipolar_recovered_minimum_similarity"]
            ),
            epipolar_recovered_minimum_margin=float(
                matcher["epipolar_recovered_minimum_margin"]
            ),
            minimum_track_views=_required(
                manifest, "geometry_teacher_min_views", int
            ),
            require_cycle=_required(
                manifest, "geometry_teacher_track_require_cycle", bool
            ),
            allow_chain_tracks=_required(
                manifest, "geometry_teacher_track_allow_chain_tracks", bool
            ),
            return_pair_sidecar=True,
            device=args.device,
            **reuse_inputs,
        )
    finally:
        triangulation.reciprocal_epipolar_matches = original_matcher
    if (
        int(diagnostics.get("track_pair_matches_reused", 0)) != 1
        or sidecar.get("policy", {}).get("uses_precomputed_pair_matches") is not True
    ):
        raise RuntimeError("P8 Track builder did not attest exact probe reuse")

    observation_query = tracks["query_index"]
    observation_keypoint = tracks["keypoint_index"]
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), keypoint_counts.cumsum(0)))
    all_keypoints = torch.cat(keypoints)
    observation_uv = all_keypoints[offsets[observation_query] + observation_keypoint]
    rendered_depth = torch.empty(observation_query.numel(), dtype=torch.float32)
    order = torch.argsort(observation_query, stable=True)
    counts = torch.bincount(observation_query, minlength=len(cache["names"]))
    observation_offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    for query in range(len(cache["names"])):
        begin, end = int(observation_offsets[query]), int(observation_offsets[query + 1])
        if begin == end:
            continue
        rows = order[begin:end]
        rendered_depth[rows] = depth_at_keypoints[query][observation_keypoint[rows]]
    query_bins = triangulation.camera_pose_bins(
        poses,
        _required(manifest, "geometry_teacher_view_bins", int),
        direction_weight=_required(
            manifest, "geometry_teacher_view_direction_weight", float
        ),
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
        maximum_observations_per_landmark=_required(
            manifest, "geometry_teacher_max_observations_per_landmark", int
        ),
        minimum_views=_required(manifest, "geometry_teacher_min_views", int),
        minimum_view_bins=_required(
            manifest, "geometry_teacher_min_view_bins", int
        ),
        huber_delta_px=_required(manifest, "geometry_teacher_huber_delta_px", float),
        iterations=_required(manifest, "geometry_teacher_iterations", int),
        minimum_parallax_deg=_required(
            manifest, "geometry_teacher_min_parallax_deg", float
        ),
        parallax_quantile=_required(
            manifest, "geometry_teacher_parallax_quantile", float
        ),
        maximum_reprojection_px=_required(
            manifest, "geometry_teacher_max_reprojection_px", float
        ),
        maximum_condition_number=_required(
            manifest, "geometry_teacher_max_condition_number", float
        ),
        maximum_covariance_trace_m2=_required(
            manifest, "geometry_teacher_max_covariance_trace_m2", float
        ),
        maximum_rendered_depth_residual_m=_required(
            manifest, "geometry_teacher_max_rendered_depth_residual_m", float
        ),
        minimum_rendered_depth_observations=_required(
            manifest, "geometry_teacher_min_rendered_depth_observations", int
        ),
        surface_support_enabled=False,
    )
    geometry["track_confidence_level"] = tracks["track_level"].clone()
    triangulation.attach_pair_triangulation_statistics(sidecar, tracks, geometry, poses)

    input_lineage.update(
        {
            "pair_proposals": {
                "path": str(proposals["path"]),
                "sha256": proposals["sha256"],
                "content_sha256": proposals["content_sha256"],
            },
            "pair_match_probe": {
                "path": str(probe["path"]),
                "sha256": probe["sha256"],
                "content_sha256": probe["content_sha256"],
            },
            "pair_selection": {
                "path": str(selection["path"]),
                "sha256": selection["sha256"],
                "content_sha256": selection["content_sha256"],
            },
            "stage_a_gate": {
                "path": str(stage_a["path"]),
                "sha256": stage_a["sha256"],
            },
            "probe_matcher": deepcopy(matcher),
            "pair_subset_role": subset_role,
        }
    )
    input_lineage["query_cache"]["mapping_scope"] = deepcopy(
        cache["mapping_scope"]
    )
    factor = _factor_payload(
        mapping_keypoints=mapping_k,
        nms_radius=nms_radius,
        pair_policy=pair_policy,
        pair_policy_parameters={
            "reuse_only": True,
            "pair_subset_role": subset_role,
            "probe_matcher": deepcopy(matcher),
            "track_science_contract": deepcopy(science_contract),
        },
        query_names=cache["names"],
        query_bins=query_bins,
        tracks=tracks,
        track_geometry=geometry,
        pair_sidecar=sidecar,
        diagnostics=diagnostics,
        input_lineage=input_lineage,
    )
    report = _build_report(
        result=factor,
        frozen=frozen,
        sidecar=sidecar,
        keypoint_counts=keypoint_counts,
        scene_point_count=0,
        pair_budget=pair_budget,
        manifest_path=manifest_path,
        query_cache_path=cache["path"],
        frozen_track_payload_path=frozen_path,
    )
    report["reuse_only"] = True
    report["probe_matcher"] = deepcopy(matcher)
    report["scene_contract"] = contract
    report["stage_seconds"] = {"total": float(time.perf_counter() - started)}

    frozen_inputs = {
        "query_cache": cache,
        "pair_proposals": proposals,
        "pair_match_probe": probe,
        "pair_selection": selection,
        "stage_a_gate": stage_a,
        "manifest": input_lineage["manifest"],
        "frozen_track_payload": input_lineage["frozen_track_payload"],
    }
    for name, artifact in frozen_inputs.items():
        if sha256_file(artifact["path"]) != artifact["sha256"]:
            raise RuntimeError(f"P8 Track input changed during construction: {name}")

    output_dir = args.output_dir.resolve()
    factor_path = output_dir / f"{pair_policy}_track_factor.pt"
    report_path = output_dir / f"{pair_policy}_track_factor.json"
    validate_output_target(
        factor_path,
        protected_paths=[
            cache["path"],
            proposals["path"],
            probe["path"],
            selection["path"],
            stage_a["path"],
            manifest_path,
            frozen_path,
        ],
    )
    validate_output_target(report_path, protected_paths=[factor_path])
    if factor_path.exists() or report_path.exists():
        raise FileExistsError("P8 Track factor output already exists")
    atomic_torch_save(factor, factor_path, overwrite=False)
    report["artifact"] = str(factor_path)
    report["artifact_sha256"] = sha256_file(factor_path)
    atomic_json_save(report, report_path, overwrite=False)
    return {
        "arm": args.arm,
        "pair_policy": pair_policy,
        "factor": str(factor_path),
        "factor_sha256": report["artifact_sha256"],
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "track_pair_matches_reused": diagnostics["track_pair_matches_reused"],
        "uses_precomputed_pair_matches": sidecar["policy"][
            "uses_precomputed_pair_matches"
        ],
        "track": report["track"],
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True))


def entrypoint(argv: Sequence[str] | None = None) -> None:
    try:
        main(argv)
    except SystemExit:
        raise
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    entrypoint()
