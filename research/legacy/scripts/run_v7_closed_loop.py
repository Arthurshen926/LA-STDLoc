#!/usr/bin/env python3
"""Run the gated, fail-closed V7 safe closed-loop mainline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import torch
import torch.nn.functional as F

from common.v7_contracts import (
    V7_P0_REPORT_SCHEMA,
    audit_formal_import_graph,
    compare_deployment_contracts,
    compare_query_results,
    load_v7_config,
    sha256_file,
    tensor_tree_equal,
    validate_compact_map,
    require_view_role,
)
from data.datasets import ColmapDataset
from evidence.v7_query_planner import camera_centers
from evidence.v7_render_certificate import (
    CertificateThresholds,
    certify_v7_render,
    extreme_distortion_row_mask,
)
from features.extractor import FeatureExtractor
from map_learning.v7_feedback import (
    DiagnosticRegistry,
    diagnose_feedback_query,
    load_v7_fixed_plant,
    localize_rgb_query,
    pose_error,
)
from map_learning.v7_descriptor_controller import reconstruct_v7_descriptors
from priors.models import GaussianModel2D
from priors.rendering import render_from_pose_gsplat
from topology.v7_sufficiency_selector import (
    CompactEdgeRegistry,
    CompactPoseInformation,
    EligibilityThresholds,
    SufficiencyTargets,
    eligibility_mask,
    reconstruct_mapping_candidate_evidence,
    select_v7_sufficiency,
)


def _require_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected.lower():
        raise ValueError(f"{label} SHA256 differs")
    return actual


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(payload: dict, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def run_p0(args: argparse.Namespace) -> dict:
    root = Path(__file__).resolve().parents[1]
    load_v7_config(args.config)
    import_audit = audit_formal_import_graph(
        root=root,
        entrypoint=Path(__file__),
        allowlist_path=args.formal_source_allowlist,
    )
    source_map = args.baseline_map.resolve()
    source_metric = args.baseline_metric.resolve()
    map_sha = _require_sha(
        source_map, args.expected_baseline_map_sha256, "baseline map"
    )
    metric_sha = _require_sha(
        source_metric, args.expected_baseline_metric_sha256, "baseline metric"
    )
    source_state = torch.load(source_map, map_location="cpu", weights_only=False)
    map_summary = validate_compact_map(source_state)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    output_map = args.output_dir / "compact_map.pt"
    output_metric = args.output_dir / "identity_metric.pt"
    _atomic_copy(source_map, output_map)
    _atomic_copy(source_metric, output_metric)
    output_state = torch.load(output_map, map_location="cpu", weights_only=False)
    if not tensor_tree_equal(source_state, output_state):
        raise RuntimeError("V7 P0 no-op changed compact map tensors")
    if sha256_file(output_map) != map_sha or sha256_file(output_metric) != metric_sha:
        raise RuntimeError("V7 P0 no-op changed serialized baseline artifacts")

    query_parity = None
    deployment_parity = None
    if (args.reference_results is None) != (args.candidate_results is None):
        raise ValueError("reference and candidate results must be supplied together")
    if args.reference_results is not None:
        reference_results = args.reference_results.resolve()
        candidate_results = args.candidate_results.resolve()
        _require_sha(
            reference_results,
            args.expected_reference_results_sha256,
            "reference results",
        )
        _require_sha(
            candidate_results,
            args.expected_candidate_results_sha256,
            "candidate results",
        )
        query_parity = compare_query_results(
            json.loads(reference_results.read_text()),
            json.loads(candidate_results.read_text()),
        )
        if query_parity["query_count"] != int(args.expected_query_count):
            raise ValueError("P0 query count differs from the preregistered count")
        if (args.reference_deployment_contract is None) != (
            args.candidate_deployment_contract is None
        ):
            raise ValueError("deployment contracts must be supplied together")
        if args.reference_deployment_contract is not None:
            deployment_parity = compare_deployment_contracts(
                json.loads(args.reference_deployment_contract.read_text()),
                json.loads(args.candidate_deployment_contract.read_text()),
            )

    report = {
        "schema": V7_P0_REPORT_SCHEMA,
        "version": 1,
        "phase": "P0",
        "status": "PASS",
        "uses_source_mapping_rgb": False,
        "uses_test_queries_for_map_updates": False,
        "map_action": "identity_noop",
        "input": {
            "map": str(source_map),
            "map_sha256": map_sha,
            "metric": str(source_metric),
            "metric_sha256": metric_sha,
        },
        "output": {
            "map": str(output_map.resolve()),
            "map_sha256": sha256_file(output_map),
            "metric": str(output_metric.resolve()),
            "metric_sha256": sha256_file(output_metric),
        },
        "map_tensor_parity": {**map_summary, "exact": True},
        "query_parity": query_parity,
        "deployment_parity": deployment_parity,
        "formal_import_graph": import_audit,
    }
    (args.output_dir / "p0_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


@torch.inference_mode()
def run_p2(args: argparse.Namespace) -> dict:
    root = Path(__file__).resolve().parents[1]
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(False)
    config = load_v7_config(args.config)
    certificate_config = config.get("p2_render_certificate", {})
    expected_fusion_policy = (
        "basic_geometry_and_rank_extreme_distortion_and_rgb_structure_v2"
    )
    if certificate_config.get("quality_fusion_policy") != expected_fusion_policy:
        raise ValueError("P2 render-quality fusion policy differs from V7 v2")
    certificate_thresholds = CertificateThresholds(
        **{
            name: certificate_config[name]
            for name in CertificateThresholds.__dataclass_fields__
            if name in certificate_config
        }
    )
    import_audit = audit_formal_import_graph(
        root=root,
        entrypoint=Path(__file__),
        allowlist_path=args.formal_source_allowlist,
    )
    plan_path = args.query_plan.resolve()
    prior_path = args.gaussian_ply.resolve()
    plan_sha = _require_sha(plan_path, args.expected_query_plan_sha256, "query plan")
    prior_sha = _require_sha(
        prior_path, args.expected_gaussian_ply_sha256, "Gaussian prior"
    )
    plan = torch.load(plan_path, map_location="cpu", weights_only=False)
    role = str(plan.get("view_role"))
    if role not in {
        "feedback_query",
        "confirmation_query",
        "test_pose_render_diagnostic",
    }:
        raise ValueError("P2 received an unsupported query plan role")
    require_view_role(plan, role)
    if plan.get("render_protocol") != "clean_once_per_pose":
        raise ValueError("P2 formal path permits one clean render per pose only")
    if any(
        plan.get(field) is not False
        for field in (
            "enters_track_registry",
            "enters_anchor_observation_csr",
            "enters_descriptor_bank",
        )
    ):
        raise ValueError("P2 query plan violates non-mapping membership")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    records_dir = args.output_dir / "records"
    records_dir.mkdir()

    dataset = ColmapDataset(args.dataset, images=args.images)
    mapping = dataset.split("mapping")
    mapping_poses = torch.stack(
        [torch.as_tensor(camera.pose_w2c, dtype=torch.float64) for camera in mapping]
    )
    if int(plan.get("mapping_camera_count", -1)) != len(mapping):
        raise ValueError("P2 dataset mapping registry differs from the query plan")
    mapping_centers = camera_centers(mapping_poses)
    query_centers = camera_centers(plan["pose_w2c"])
    nearest_distances = torch.cdist(query_centers, mapping_centers).min(dim=1).values

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("P2 formal Gaussian rendering requires CUDA")
    model = GaussianModel2D(args.sh_degree, device=device)
    model.load_ply(prior_path, loc_feature_dim=0)
    model = model.to(device).eval()
    extractor = FeatureExtractor("sp", nms_radius=args.nms_radius).to(device).eval()
    extractor.requires_grad_(False)
    decisions = {"ACCEPT": 0, "UNCERTAIN": 0, "REJECT": 0}
    record_registry = []
    for index in range(int(plan["query_count"])):
        query_index = int(plan.get("query_indices", torch.arange(plan["query_count"]))[index])
        pose = torch.as_tensor(plan["pose_w2c"][index], device=device).float()
        intrinsic = torch.as_tensor(plan["intrinsics"][index]).float()
        height, width = map(int, torch.as_tensor(plan["image_hw"][index]).tolist())
        fov_x = (
            2.0
            * torch.atan(torch.tensor(width / (2.0 * float(intrinsic[0, 0])))).item()
        )
        fov_y = (
            2.0
            * torch.atan(torch.tensor(height / (2.0 * float(intrinsic[1, 1])))).item()
        )
        package = render_from_pose_gsplat(
            model,
            pose,
            fov_x,
            fov_y,
            width,
            height,
            bg_color=torch.zeros(3, device=device),
            render_mode="RGB+ED",
            rgb_only=True,
            rasterize_mode="antialiased",
        )
        rendered_rgb = package["render"].float().clamp(0.0, 1.0)
        # The detector and every later replay must consume the same persisted
        # tensor. Float16 is the canonical clean-render sensor representation;
        # converting it back to float32 is exact and replayable.
        rgb_float16 = rendered_rgb.detach().cpu().to(torch.float16)
        rgb = rgb_float16.float().to(device)
        alpha = package.get("alphas", package.get("rend_alpha"))
        depth = package.get("depth")
        if alpha is None or depth is None:
            raise ValueError("P2 renderer must return RGB, alpha, and depth")
        torch.use_deterministic_algorithms(True)
        sparse = extractor.detectAndCompute(
            rgb[None],
            top_k=args.keypoints,
            detection_threshold=args.detection_threshold,
        )[0]
        torch.use_deterministic_algorithms(False)
        keypoints_device = sparse["keypoints"].detach().float()
        keypoints = keypoints_device.cpu()
        median_depth_raster = package.get("rend_median")
        expected_median_depth = None
        if median_depth_raster is not None:
            median_values = median_depth_raster.detach().cpu().float().squeeze()
            median_valid = torch.isfinite(median_values) & (median_values > 0)
            if bool(median_valid.any()):
                expected_median_depth = float(median_values[median_valid].median())
        distortion = package.get("rend_dist")
        artifact_row_mask = (
            None
            if distortion is None
            else extreme_distortion_row_mask(
                distortion.detach(),
                keypoints_device,
                mad_multiplier=certificate_thresholds.distortion_mad_multiplier,
                tail_quantile=certificate_thresholds.distortion_tail_quantile,
            )
        )
        certificate = certify_v7_render(
            rgb=rgb,
            alpha=alpha.detach(),
            depth=depth.detach(),
            keypoints=keypoints_device,
            nearest_mapping_distance_m=float(nearest_distances[index]),
            median_adjacent_baseline_m=float(
                plan["trajectory_statistics"]["median_adjacent_baseline_m"]
            ),
            source_family_support=len(set(plan["source_mapping_indices"][index])),
            expected_median_depth_m=expected_median_depth,
            artifact_row_mask=artifact_row_mask,
            thresholds=certificate_thresholds,
        )
        certificate["row_valid"] = certificate["row_valid"].cpu()
        certificate["row_uncertain"] = certificate["row_uncertain"].cpu()
        certificate["row_reasons"] = {
            name: rows.cpu()
            for name, rows in certificate["row_reasons"].items()
        }
        decision = certificate["decision"]
        decisions[decision] += 1
        record = {
            "schema": "lafgs_v7_certified_clean_render",
            "version": 1,
            "view_role": role,
            "query_index": query_index,
            "pose_family_id": int(plan["pose_family_ids"][index]),
            "pose_w2c": torch.as_tensor(plan["pose_w2c"][index]).float(),
            "intrinsics": intrinsic,
            "image_hw": torch.tensor([height, width], dtype=torch.long),
            "rgb_float16": rgb_float16,
            "rgb_uint8_preview": (rgb_float16.float() * 255.0).round().to(torch.uint8),
            "alpha_float16": alpha.detach().cpu().to(torch.float16),
            "depth_float16": depth.detach().cpu().to(torch.float16),
            "surface_median_depth_float16": (
                None
                if median_depth_raster is None
                else median_depth_raster.detach().cpu().to(torch.float16)
            ),
            "keypoints": keypoints,
            "descriptors": F.normalize(
                sparse["descriptors"].detach().cpu().float(), dim=1
            ),
            "scores": sparse["keypoint_scores"].detach().cpu().float(),
            "certificate": certificate,
            "enters_track_registry": False,
            "enters_anchor_observation_csr": False,
            "enters_descriptor_bank": False,
        }
        if role == "test_pose_render_diagnostic":
            record.update(
                {
                    "formal_protocol_eligible": False,
                    "uses_test_pose_metadata": True,
                    "uses_test_rgb": False,
                    "transductive_pose_distribution_oracle": True,
                }
            )
        record_path = records_dir / f"query_{query_index:04d}.pt"
        temporary = record_path.with_name(f".{record_path.name}.{os.getpid()}.tmp")
        try:
            torch.save(record, temporary)
            os.replace(temporary, record_path)
        finally:
            temporary.unlink(missing_ok=True)
        record_registry.append(
            {
                "query_index": query_index,
                "path": str(record_path.resolve()),
                "sha256": sha256_file(record_path),
                "decision": decision,
            }
        )
    manifest = {
        "schema": "lafgs_v7_certified_clean_render_batch",
        "version": 1,
        "view_role": role,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": bool(plan.get("uses_test_queries", False)),
        "uses_test_pose_metadata": bool(plan.get("uses_test_pose_metadata", False)),
        "uses_test_rgb": bool(plan.get("uses_test_rgb", False)),
        "formal_protocol_eligible": bool(plan.get("formal_protocol_eligible", True)),
        "render_protocol": "clean_once_per_pose",
        "detector_input": "complete_unmasked_rgb",
        "detector_rgb_storage": "float16_replayed_as_float32",
        "quality_mask_stage": "post_detector_row_sampling",
        "map_input": None,
        "map_mutation_count": 0,
        "input": {
            "query_plan": str(plan_path),
            "query_plan_sha256": plan_sha,
            "gaussian_ply": str(prior_path),
            "gaussian_ply_sha256": prior_sha,
        },
        "query_count": int(plan["query_count"]),
        "decision_counts": decisions,
        "records": record_registry,
        "formal_import_graph": import_audit,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


@torch.inference_mode()
def run_p3_prepare(args: argparse.Namespace) -> dict:
    root = Path(__file__).resolve().parents[1]
    config = load_v7_config(args.config)
    import_audit = audit_formal_import_graph(
        root=root,
        entrypoint=Path(__file__),
        allowlist_path=args.formal_source_allowlist,
    )
    candidates_path = args.candidate_pool.resolve()
    cache_path = args.mapping_feature_cache.resolve()
    candidates_sha = _require_sha(
        candidates_path, args.expected_candidate_pool_sha256, "candidate pool"
    )
    cache_sha = _require_sha(
        cache_path, args.expected_mapping_feature_cache_sha256, "mapping feature cache"
    )
    candidates = torch.load(candidates_path, map_location="cpu", weights_only=False)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if (
        candidates.get("schema") != "projective_anchor_candidates_v2"
        or candidates.get("uses_source_mapping_rgb") is not False
        or candidates.get("uses_test_queries") is not False
    ):
        raise ValueError("P3 candidate pool is not mapping-only projective evidence")
    construction = candidates.get("contract", {})
    if (
        construction.get("final_xyz_source") != "fixed_camera_robust_ray_triangulation"
        or construction.get("gaussian_depth_used_for_final_xyz") is not False
        or construction.get("direct_gaussian_surface_anchor") is not False
    ):
        raise ValueError("P3 candidate xyz violates pure-ray construction")
    if (
        cache.get("schema") != "render_observation_cache_v2"
        or cache.get("uses_source_mapping_rgb") is not False
        or cache.get("uses_test_queries") is not False
    ):
        raise ValueError("P3 feature cache is not rendered mapping-only evidence")
    observations = candidates["projective_anchor_observations"]
    p3_config = config["p3_selector"]
    evidence = reconstruct_mapping_candidate_evidence(
        anchor_xyz=candidates["anchor_xyz"],
        anchor_features=candidates["anchor_features"],
        observation_offsets=observations["observation_offsets"],
        query_indices=observations["query_indices"],
        keypoint_indices=observations["keypoint_indices"],
        query_names=candidates["query_names"],
        query_bins=candidates["query_bins"],
        rendered_feature_records=cache["queries"],
        device=args.device,
        grid_shape=tuple(p3_config["image_grid"]),
    )
    evidence.update(
        {
            "candidate_pool": str(candidates_path),
            "candidate_pool_sha256": candidates_sha,
            "mapping_feature_cache": str(cache_path),
            "mapping_feature_cache_sha256": cache_sha,
            "candidate_count": int(candidates["anchor_ids"].numel()),
            "mapping_query_count": len(candidates["query_names"]),
            "mapping_intrinsics": torch.stack(
                [
                    torch.as_tensor(cache["queries"][name]["native_K"]).float()
                    for name in candidates["query_names"]
                ]
            ),
            "mapping_poses_w2c": torch.stack(
                [
                    torch.as_tensor(cache["queries"][name]["pose_w2c"]).float()
                    for name in candidates["query_names"]
                ]
            ),
        }
    )
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    evidence_path = args.output_dir / "candidate_evidence.pt"
    _atomic_torch_save(evidence, evidence_path)
    manifest = {
        "schema": "lafgs_v7_p3_candidate_evidence_manifest",
        "version": 1,
        "phase": "P3_PREPARE",
        "status": "PASS",
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "input": {
            "candidate_pool": str(candidates_path),
            "candidate_pool_sha256": candidates_sha,
            "mapping_feature_cache": str(cache_path),
            "mapping_feature_cache_sha256": cache_sha,
        },
        "output": {
            "candidate_evidence": str(evidence_path.resolve()),
            "candidate_evidence_sha256": sha256_file(evidence_path),
        },
        "candidate_count": evidence["candidate_count"],
        "mapping_query_count": evidence["mapping_query_count"],
        "formal_import_graph": import_audit,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _materialize_selected_map(source: dict, selected_rows: torch.Tensor) -> dict:
    selected_rows = torch.as_tensor(selected_rows).long().cpu()
    count = int(torch.as_tensor(source["anchor_ids"]).numel())
    output = {}
    for key, value in source.items():
        if key == "projective_anchor_observations":
            continue
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == count:
            output[key] = value[selected_rows].clone()
        elif isinstance(value, list) and len(value) == count:
            output[key] = [value[row] for row in selected_rows.tolist()]
        else:
            output[key] = value
    observations = source["projective_anchor_observations"]
    offsets = torch.as_tensor(observations["observation_offsets"]).long()
    queries = torch.as_tensor(observations["query_indices"]).long()
    keypoints = torch.as_tensor(observations["keypoint_indices"]).long()
    query_parts, keypoint_parts, selected_offsets = [], [], [0]
    for row in selected_rows.tolist():
        start, end = int(offsets[row]), int(offsets[row + 1])
        query_parts.append(queries[start:end])
        keypoint_parts.append(keypoints[start:end])
        selected_offsets.append(selected_offsets[-1] + end - start)
    output["projective_anchor_observations"] = {
        **{
            key: value
            for key, value in observations.items()
            if key not in {"observation_offsets", "query_indices", "keypoint_indices"}
        },
        "observation_offsets": torch.tensor(selected_offsets, dtype=torch.long),
        "query_indices": torch.cat(query_parts),
        "keypoint_indices": torch.cat(keypoint_parts),
    }
    selected_count = selected_rows.numel()
    output["base_anchor_count"] = 0
    output["canonical_anchor_count"] = selected_count
    output["micro_anchor_count"] = selected_count
    output["provenance"] = {
        **dict(source["provenance"]),
        "v7_selector_applied": True,
        "v7_selector_uses_test_queries": False,
        "v7_selector_uses_feedback_queries": False,
    }
    return output


def _v7_candidate_eligibility(
    candidates: dict, evidence: dict, p3: dict
) -> tuple[torch.Tensor, dict[str, int], EligibilityThresholds]:
    quality = p3["evidence"]
    covariance_trace = (
        torch.as_tensor(candidates["anchor_position_covariance"])
        .diagonal(dim1=1, dim2=2)
        .sum(1)
    )
    thresholds = EligibilityThresholds(
        minimum_geometry_reliability=float(quality["minimum_geometry_reliability"]),
        minimum_observation_count=int(quality["minimum_observation_count"]),
        minimum_view_family_count=int(quality["minimum_view_family_count"]),
        maximum_descriptor_dispersion=float(
            torch.quantile(
                evidence["descriptor_dispersion"],
                float(quality["maximum_descriptor_dispersion_quantile"]),
            )
        ),
        maximum_reprojection_error=float(
            torch.quantile(
                evidence["reprojection_error_px_mean"],
                float(quality["maximum_reprojection_error_quantile"]),
            )
        ),
        maximum_covariance_trace=float(
            torch.quantile(
                covariance_trace, float(quality["maximum_covariance_trace_quantile"])
            )
        ),
        minimum_parallax=float(
            torch.quantile(
                evidence["ray_angular_dispersion_deg"],
                float(quality["minimum_ray_angular_dispersion_quantile"]),
            )
        ),
    )
    eligible, exclusions = eligibility_mask(
        geometry_reliability=candidates["geometry_reliability"],
        observation_count=evidence["observation_count"],
        view_family_count=evidence["view_family_count"],
        descriptor_dispersion=evidence["descriptor_dispersion"],
        reprojection_error=evidence["reprojection_error_px_mean"],
        covariance_trace=covariance_trace,
        parallax=evidence["ray_angular_dispersion_deg"],
        render_artifact_supported=evidence["invalid_projection_count"] > 0,
        lineage_complete=torch.ones_like(candidates["anchor_ids"], dtype=torch.bool),
        thresholds=thresholds,
    )
    return eligible, exclusions, thresholds


@torch.inference_mode()
def run_p3_select(args: argparse.Namespace) -> dict:
    torch.set_num_threads(4)
    config = load_v7_config(args.config)
    candidate_path = args.candidate_pool.resolve()
    evidence_path = args.candidate_evidence.resolve()
    source_map_path = args.baseline_map.resolve()
    source_metric_path = args.baseline_metric.resolve()
    candidate_sha = _require_sha(
        candidate_path, args.expected_candidate_pool_sha256, "candidate pool"
    )
    evidence_sha = _require_sha(
        evidence_path, args.expected_candidate_evidence_sha256, "candidate evidence"
    )
    _require_sha(source_map_path, args.expected_baseline_map_sha256, "baseline map")
    _require_sha(
        source_metric_path, args.expected_baseline_metric_sha256, "baseline metric"
    )
    candidates = torch.load(candidate_path, map_location="cpu", weights_only=False)
    evidence = torch.load(evidence_path, map_location="cpu", weights_only=False)
    source_map = torch.load(source_map_path, map_location="cpu", weights_only=False)
    source_metric = torch.load(
        source_metric_path, map_location="cpu", weights_only=False
    )
    if (
        evidence.get("candidate_pool_sha256") != candidate_sha
        or evidence.get("contract", {}).get("uses_test_queries") is not False
    ):
        raise ValueError("P3 evidence lineage differs from candidate pool")
    for field in ("anchor_ids", "anchor_xyz", "anchor_features"):
        if not torch.equal(
            torch.as_tensor(candidates[field]), torch.as_tensor(source_map[field])
        ):
            raise ValueError(f"P3 candidate and baseline map differ: {field}")
    p3 = config["p3_selector"]
    quality = p3["evidence"]
    covariance_trace = (
        candidates["anchor_position_covariance"].diagonal(dim1=1, dim2=2).sum(1)
    )
    thresholds = EligibilityThresholds(
        minimum_geometry_reliability=float(quality["minimum_geometry_reliability"]),
        minimum_observation_count=int(quality["minimum_observation_count"]),
        minimum_view_family_count=int(quality["minimum_view_family_count"]),
        maximum_descriptor_dispersion=float(
            torch.quantile(
                evidence["descriptor_dispersion"],
                float(quality["maximum_descriptor_dispersion_quantile"]),
            )
        ),
        maximum_reprojection_error=float(
            torch.quantile(
                evidence["reprojection_error_px_mean"],
                float(quality["maximum_reprojection_error_quantile"]),
            )
        ),
        maximum_covariance_trace=float(
            torch.quantile(
                covariance_trace, float(quality["maximum_covariance_trace_quantile"])
            )
        ),
        minimum_parallax=float(
            torch.quantile(
                evidence["ray_angular_dispersion_deg"],
                float(quality["minimum_ray_angular_dispersion_quantile"]),
            )
        ),
    )
    eligible, exclusions = eligibility_mask(
        geometry_reliability=candidates["geometry_reliability"],
        observation_count=evidence["observation_count"],
        view_family_count=evidence["view_family_count"],
        descriptor_dispersion=evidence["descriptor_dispersion"],
        reprojection_error=evidence["reprojection_error_px_mean"],
        covariance_trace=covariance_trace,
        parallax=evidence["ray_angular_dispersion_deg"],
        render_artifact_supported=evidence["invalid_projection_count"] > 0,
        lineage_complete=torch.ones_like(candidates["anchor_ids"], dtype=torch.bool),
        thresholds=thresholds,
    )
    observations = candidates["projective_anchor_observations"]
    offsets, query_indices = (
        observations["observation_offsets"],
        observations["query_indices"],
    )
    names = candidates["query_names"]
    layers = {
        "matching": CompactEdgeRegistry(
            offsets, query_indices, observations["keypoint_indices"]
        ),
        "image_cell": CompactEdgeRegistry(
            offsets, query_indices, evidence["image_cell_identities"]
        ),
        "view_family": CompactEdgeRegistry(
            offsets, query_indices, evidence["view_family_identities"]
        ),
        "depth_range": CompactEdgeRegistry(
            offsets, query_indices, evidence["depth_range_identities"]
        ),
    }
    profile = p3["profiles"][args.selector_profile]
    observation_count = torch.as_tensor(evidence["observation_count"]).long()
    anchor_for_edge = torch.repeat_interleave(
        torch.arange(eligible.numel()), observation_count
    )
    edge_eligible = eligible[anchor_for_edge]
    query_indices_tensor = torch.as_tensor(query_indices).long()
    keypoint_indices_tensor = torch.as_tensor(observations["keypoint_indices"]).long()
    pair_base = int(keypoint_indices_tensor.max()) + 1
    pair_ids = query_indices_tensor * pair_base + keypoint_indices_tensor
    if torch.unique(pair_ids).numel() != pair_ids.numel():
        raise ValueError("P3 matching rows are not globally unique per mapping query")
    feasible_matching = torch.bincount(
        query_indices_tensor[edge_eligible], minlength=len(names)
    )

    def feasible_unique(identity: torch.Tensor) -> torch.Tensor:
        identity = torch.as_tensor(identity).long()
        valid = edge_eligible & (identity >= 0)
        base = int(identity[valid].max()) + 1
        unique = torch.unique(query_indices_tensor[valid] * base + identity[valid])
        return torch.bincount(
            torch.div(unique, base, rounding_mode="floor"), minlength=len(names)
        )

    feasible_cells = feasible_unique(evidence["image_cell_identities"])
    feasible_families = feasible_unique(evidence["view_family_identities"])
    feasible_depth = feasible_unique(evidence["depth_range_identities"])
    full_pose = torch.eye(6, dtype=torch.float64).repeat(len(names), 1, 1) * float(
        p3["pose_damping"]
    )
    full_pose_flat = full_pose.reshape(len(names), 36)
    full_pose_flat.index_add_(
        0,
        query_indices_tensor[edge_eligible],
        torch.as_tensor(evidence["pose_information_contributions"])[edge_eligible]
        .double()
        .reshape(-1, 36),
    )
    full_eigenvalues = torch.linalg.eigvalsh(full_pose)
    feasible_logdet = full_eigenvalues.clamp_min(1e-12).log().sum(1)
    feasible_minimum = full_eigenvalues[:, 0]

    def capped_integer_target(requested: int, feasible: torch.Tensor) -> list[int]:
        return torch.minimum(
            feasible, torch.full_like(feasible, int(requested))
        ).tolist()

    targets = SufficiencyTargets(
        precision_matching_rank=capped_integer_target(
            int(profile["precision_rank"]), feasible_matching
        ),
        completion_matching_rank=capped_integer_target(
            int(profile["matching_rank"]), feasible_matching
        ),
        image_cells=capped_integer_target(int(profile["image_cells"]), feasible_cells),
        view_families=capped_integer_target(
            int(profile["view_families"]), feasible_families
        ),
        depth_ranges=capped_integer_target(
            int(profile["depth_ranges"]), feasible_depth
        ),
        pose_logdet=torch.minimum(
            feasible_logdet,
            torch.full_like(feasible_logdet, float(profile["pose_logdet"])),
        ).tolist(),
        pose_minimum_eigenvalue=torch.minimum(
            feasible_minimum,
            torch.full_like(
                feasible_minimum, float(profile["pose_minimum_eigenvalue"])
            ),
        ).tolist(),
        maximum_anchors=int(profile["maximum_anchors"]),
        pose_damping=float(p3["pose_damping"]),
    )
    reliability = (
        candidates["geometry_reliability"].float()
        * candidates["identity_reliability"].float()
    )
    selection = select_v7_sufficiency(
        anchor_ids=candidates["anchor_ids"],
        reliability=reliability,
        eligible=eligible,
        layer_edges=layers,
        pose_information=CompactPoseInformation(
            offsets, query_indices, evidence["pose_information_contributions"]
        ),
        query_count=len(names),
        targets=targets,
        active_set_change_fraction=float(p3["active_set_change_fraction"]),
    )
    selection["profile"] = args.selector_profile
    selection["feasibility_limited_query_count"] = {
        "matching": int((feasible_matching < int(profile["matching_rank"])).sum()),
        "image_cell": int((feasible_cells < int(profile["image_cells"])).sum()),
        "view_family": int((feasible_families < int(profile["view_families"])).sum()),
        "depth_range": int((feasible_depth < int(profile["depth_ranges"])).sum()),
        "pose_logdet": int((feasible_logdet < float(profile["pose_logdet"])).sum()),
        "pose_minimum_eigenvalue": int(
            (feasible_minimum < float(profile["pose_minimum_eigenvalue"])).sum()
        ),
    }
    selection["eligibility"]["exclusions"] = exclusions
    selection["eligibility"]["thresholds"] = thresholds.__dict__
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    selection_path = args.output_dir / "selection.pt"
    _atomic_torch_save(selection, selection_path)
    selected_map = _materialize_selected_map(
        source_map, selection["selected_anchor_rows"]
    )
    map_path = args.output_dir / "compact_map.pt"
    _atomic_torch_save(selected_map, map_path)
    metric = {
        **source_metric,
        "landmark_indices": selected_map["anchor_ids"].clone(),
        "map_path": str(map_path.resolve()),
    }
    metric["map_sha256"] = sha256_file(map_path)
    metric_path = args.output_dir / "identity_metric.pt"
    _atomic_torch_save(metric, metric_path)
    manifest = {
        "schema": "lafgs_v7_p3_selection_manifest",
        "version": 1,
        "profile": args.selector_profile,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "uses_feedback_queries": False,
        "candidate_count": int(candidates["anchor_ids"].numel()),
        "eligible_count": int(eligible.sum()),
        "selected_count": int(selection["selected_anchor_ids"].numel()),
        "unmet": selection["unmet"],
        "input": {
            "candidate_pool_sha256": candidate_sha,
            "candidate_evidence_sha256": evidence_sha,
        },
        "output": {
            "selection": str(selection_path.resolve()),
            "selection_sha256": sha256_file(selection_path),
            "map": str(map_path.resolve()),
            "map_sha256": sha256_file(map_path),
            "metric": str(metric_path.resolve()),
            "metric_sha256": sha256_file(metric_path),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


@torch.inference_mode()
def run_p4(args: argparse.Namespace) -> dict:
    """Replay a certified novel-view batch through the immutable RGB plant."""

    root = Path(__file__).resolve().parents[1]
    config = load_v7_config(args.config)
    import_audit = audit_formal_import_graph(
        root=root,
        entrypoint=Path(__file__),
        allowlist_path=args.formal_source_allowlist,
    )
    batch_path = args.certified_batch.resolve()
    candidate_path = args.candidate_pool.resolve()
    evidence_path = args.candidate_evidence.resolve()
    map_path = args.baseline_map.resolve()
    metric_path = args.baseline_metric.resolve()
    input_shas = {
        "certified_batch": _require_sha(
            batch_path, args.expected_certified_batch_sha256, "certified batch"
        ),
        "candidate_pool": _require_sha(
            candidate_path, args.expected_candidate_pool_sha256, "candidate pool"
        ),
        "candidate_evidence": _require_sha(
            evidence_path,
            args.expected_candidate_evidence_sha256,
            "candidate evidence",
        ),
        "map": _require_sha(map_path, args.expected_baseline_map_sha256, "P4 map"),
        "metric": _require_sha(
            metric_path, args.expected_baseline_metric_sha256, "P4 metric"
        ),
    }
    batch = json.loads(batch_path.read_text())
    if (
        batch.get("schema") != "lafgs_v7_certified_clean_render_batch"
        or batch.get("view_role") not in {
            "feedback_query",
            "confirmation_query",
            "test_pose_render_diagnostic",
        }
        or (
            batch.get("view_role") != "test_pose_render_diagnostic"
            and batch.get("uses_test_queries") is not False
        )
        or (
            batch.get("view_role") == "test_pose_render_diagnostic"
            and not (
                batch.get("uses_test_queries") is True
                and batch.get("uses_test_pose_metadata") is True
                and batch.get("uses_test_rgb") is False
                and batch.get("formal_protocol_eligible") is False
            )
        )
        or batch.get("map_mutation_count") != 0
        or batch.get("detector_rgb_storage") != "float16_replayed_as_float32"
    ):
        raise ValueError("P4 requires a replayable non-test certified render batch")
    candidates = torch.load(candidate_path, map_location="cpu", weights_only=False)
    evidence = torch.load(evidence_path, map_location="cpu", weights_only=False)
    if evidence.get("candidate_pool_sha256") != input_shas["candidate_pool"]:
        raise ValueError("P4 candidate evidence lineage differs")
    eligible, exclusions, thresholds = _v7_candidate_eligibility(
        candidates, evidence, config["p3_selector"]
    )
    registry = DiagnosticRegistry(
        anchor_ids=torch.as_tensor(candidates["anchor_ids"]).long().cpu(),
        anchor_xyz=torch.as_tensor(candidates["anchor_xyz"]).float().cpu(),
        eligible=eligible.cpu(),
    )
    plant = load_v7_fixed_plant(
        map_path, metric_path, device=args.device, diagnostic_registry=registry
    )
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    records_dir = args.output_dir / "records"
    records_dir.mkdir()
    p4 = config["p4_feedback"]
    categories = {
        "representation_deficit": 0,
        "precision_deficit": 0,
        "coverage_deficit": 0,
        "unreliable_query": 0,
        "nominal_success": 0,
    }
    if batch["view_role"] == "test_pose_render_diagnostic":
        categories["pose_failure_diagnostic"] = 0
    replay_mismatch_count = 0
    output_records = []
    runtime_totals: dict[str, float] = {}
    for item in batch["records"]:
        record_path = Path(item["path"]).resolve()
        record_sha = _require_sha(record_path, item["sha256"], "P2 query record")
        record = torch.load(record_path, map_location="cpu", weights_only=False)
        if (
            record.get("view_role") != batch["view_role"]
            or record.get("enters_track_registry") is not False
            or record.get("enters_anchor_observation_csr") is not False
            or record.get("enters_descriptor_bank") is not False
        ):
            raise ValueError("P4 query record violates its frozen view role")
        localization = localize_rgb_query(
            record["rgb_float16"].float(), record["intrinsics"], plant
        )
        keypoint_score_replay_exact = torch.equal(
            localization.keypoints, record["keypoints"]
        ) and torch.equal(localization.scores, record["scores"])
        descriptor_replay_max_abs_error = float(
            (localization.descriptors - record["descriptors"]).abs().max()
        )
        replay_within_contract = keypoint_score_replay_exact and (
            descriptor_replay_max_abs_error
            <= float(p4["maximum_descriptor_replay_abs_error"])
        )
        replay_mismatch_count += int(not replay_within_contract)
        if (
            p4["require_bitwise_keypoint_score_replay"]
            and not keypoint_score_replay_exact
        ):
            raise RuntimeError("P4 keypoints/scores do not bitwise replay P2 RGB")
        if not replay_within_contract:
            raise RuntimeError("P4 descriptor replay exceeds its numeric contract")
        if batch["view_role"] == "test_pose_render_diagnostic":
            rotation, translation = pose_error(
                localization.pose.pose_w2c,
                torch.as_tensor(record["pose_w2c"]).numpy(),
            )
            decision = record["certificate"]["decision"]
            pose_success = bool(
                translation < float(p4["success_translation_cm"])
                and rotation < float(p4["success_rotation_deg"])
            )
            category = (
                "unreliable_query"
                if decision != "ACCEPT"
                else "nominal_success" if pose_success else "pose_failure_diagnostic"
            )
            diagnosis = {
                "category": category,
                "certificate_decision": decision,
                "can_drive_map_update": False,
                "included_in_feedback_statistics": decision == "ACCEPT",
                "translation_error_cm": translation,
                "rotation_error_deg": rotation,
                "task_error": float(
                    (
                        (translation / float(p4["success_translation_cm"])) ** 2
                        + (rotation / float(p4["success_rotation_deg"])) ** 2
                    )
                    ** 0.5
                ),
                "pose_success": pose_success,
                "transductive_pose_distribution_oracle": True,
                "uses_test_rgb": False,
                "oracle_correspondence_search_run": False,
                "oracle_used_online": False,
            }
        else:
            diagnosis = diagnose_feedback_query(
                localization,
                record["pose_w2c"],
                record["alpha_float16"],
                record["depth_float16"],
                record["certificate"],
                success_translation_cm=float(p4["success_translation_cm"]),
                success_rotation_deg=float(p4["success_rotation_deg"]),
                oracle_reprojection_px=float(p4["oracle_reprojection_px"]),
                minimum_oracle_correspondences=int(
                    p4["minimum_oracle_correspondences"]
                ),
                precision_spatial_grid=tuple(p4["precision_spatial_grid"]),
                minimum_precision_spatial_cells=int(
                    p4["minimum_precision_spatial_cells"]
                ),
                minimum_precision_supporting_rows=int(
                    p4["minimum_precision_supporting_rows"]
                ),
                minimum_precision_translation_improvement_cm=float(
                    p4["minimum_precision_translation_improvement_cm"]
                ),
                minimum_precision_translation_relative_improvement=float(
                    p4["minimum_precision_translation_relative_improvement"]
                ),
                minimum_precision_rotation_improvement_deg=float(
                    p4["minimum_precision_rotation_improvement_deg"]
                ),
            )
        if batch["view_role"] != "feedback_query":
            diagnosis["confirmation_only_would_drive_update"] = bool(
                diagnosis["can_drive_map_update"]
            )
            diagnosis["can_drive_map_update"] = False
        categories[diagnosis["category"]] += 1
        for key, value in localization.runtime_ms.items():
            runtime_totals[key] = runtime_totals.get(key, 0.0) + float(value)
        output = {
            "schema": "lafgs_v7_fixed_plant_feedback_record",
            "version": 1,
            "view_role": batch["view_role"],
            "query_index": int(record["query_index"]),
            "pose_family_id": int(record["pose_family_id"]),
            "p2_record": str(record_path),
            "p2_record_sha256": record_sha,
            "keypoint_score_replay_exact": keypoint_score_replay_exact,
            "descriptor_replay_max_abs_error": descriptor_replay_max_abs_error,
            "frontend_replay_within_contract": replay_within_contract,
            "estimated_pose_w2c": torch.from_numpy(localization.pose.pose_w2c),
            "pose_inliers": torch.from_numpy(localization.pose.inliers),
            "matches": {
                "keypoint_indices": localization.matches.keypoint_indices,
                "anchor_indices": localization.matches.anchor_indices,
                "scores": localization.matches.scores,
            },
            "runtime_ms": localization.runtime_ms,
            "diagnosis": diagnosis,
            "oracle_used_online": False,
            "map_mutation_count": 0,
        }
        output_path = records_dir / f"query_{int(record['query_index']):04d}.pt"
        _atomic_torch_save(output, output_path)
        output_records.append(
            {
                "query_index": int(record["query_index"]),
                "path": str(output_path.resolve()),
                "sha256": sha256_file(output_path),
                "category": diagnosis["category"],
                "can_drive_map_update": bool(diagnosis["can_drive_map_update"]),
            }
        )
    count = int(batch["query_count"])
    manifest = {
        "schema": "lafgs_v7_fixed_plant_feedback_batch",
        "version": 1,
        "phase": "P4",
        "status": "PASS",
        "view_role": batch["view_role"],
        "uses_source_mapping_rgb": False,
        "uses_test_queries": bool(batch.get("uses_test_queries", False)),
        "uses_test_pose_metadata": bool(
            batch.get("uses_test_pose_metadata", False)
        ),
        "uses_test_rgb": bool(batch.get("uses_test_rgb", False)),
        "formal_protocol_eligible": bool(
            batch.get("formal_protocol_eligible", True)
        ),
        "oracle_used_online": False,
        "map_mutation_count": 0,
        "query_count": count,
        "category_counts": categories,
        "update_authorized_count": sum(
            int(item["can_drive_map_update"]) for item in output_records
        ),
        "frontend_replay_mismatch_count": replay_mismatch_count,
        "runtime_ms_mean": {
            key: value / max(count, 1) for key, value in runtime_totals.items()
        },
        "eligibility": {
            "eligible_count": int(eligible.sum()),
            "exclusions": exclusions,
            "thresholds": thresholds.__dict__,
        },
        "input": {
            "certified_batch": str(batch_path),
            "candidate_pool": str(candidate_path),
            "candidate_evidence": str(evidence_path),
            "map": str(map_path),
            "metric": str(metric_path),
            **{f"{key}_sha256": value for key, value in input_shas.items()},
        },
        "records": output_records,
        "formal_import_graph": import_audit,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


@torch.inference_mode()
def run_p5(args: argparse.Namespace) -> dict:
    """Build bounded descriptor evidence; stop exactly when it is not actionable."""

    root = Path(__file__).resolve().parents[1]
    config = load_v7_config(args.config)
    import_audit = audit_formal_import_graph(
        root=root,
        entrypoint=Path(__file__),
        allowlist_path=args.formal_source_allowlist,
    )
    feedback_path = args.feedback_batch.resolve()
    candidate_path = args.candidate_pool.resolve()
    cache_path = args.mapping_feature_cache.resolve()
    map_path = args.baseline_map.resolve()
    metric_path = args.baseline_metric.resolve()
    input_shas = {
        "feedback_batch": _require_sha(
            feedback_path, args.expected_feedback_batch_sha256, "P4 feedback batch"
        ),
        "candidate_pool": _require_sha(
            candidate_path, args.expected_candidate_pool_sha256, "candidate pool"
        ),
        "mapping_feature_cache": _require_sha(
            cache_path,
            args.expected_mapping_feature_cache_sha256,
            "mapping feature cache",
        ),
        "map": _require_sha(map_path, args.expected_baseline_map_sha256, "P5 map"),
        "metric": _require_sha(
            metric_path, args.expected_baseline_metric_sha256, "P5 metric"
        ),
    }
    feedback = json.loads(feedback_path.read_text())
    if (
        feedback.get("schema") != "lafgs_v7_fixed_plant_feedback_batch"
        or feedback.get("view_role") != "feedback_query"
        or feedback.get("uses_test_queries") is not False
        or feedback.get("map_mutation_count") != 0
        or feedback.get("input", {}).get("map_sha256") != input_shas["map"]
    ):
        raise ValueError("P5 accepts only immutable non-test feedback on its M0")
    candidates = torch.load(candidate_path, map_location="cpu", weights_only=False)
    deployed_map = torch.load(map_path, map_location="cpu", weights_only=False)
    if deployed_map.get("schema") != "lafgs_materialized_anchor_map":
        raise ValueError("P5 baseline is not a materialized deployed map")
    evidence_rows = []
    signs_by_anchor: dict[int, dict[str, set[int]]] = {}
    for item in feedback["records"]:
        record_path = Path(item["path"]).resolve()
        _require_sha(record_path, item["sha256"], "P4 feedback record")
        record = torch.load(record_path, map_location="cpu", weights_only=False)
        diagnosis = record["diagnosis"]
        if not bool(diagnosis["can_drive_map_update"]):
            continue
        control = diagnosis["descriptor_control_evidence"]
        row = {
            "pose_family_id": int(record["pose_family_id"]),
            "query_descriptors": control["query_descriptors"],
            "positive_anchor_ids": control["positive_anchor_ids"],
            "false_attractor_anchor_ids": control["false_attractor_anchor_ids"],
        }
        evidence_rows.append(row)
        family = row["pose_family_id"]
        for anchor_id in torch.as_tensor(row["positive_anchor_ids"]).tolist():
            signs_by_anchor.setdefault(
                int(anchor_id), {"positive": set(), "harm": set()}
            )["positive"].add(family)
        for anchor_id in torch.as_tensor(row["false_attractor_anchor_ids"]).tolist():
            signs_by_anchor.setdefault(
                int(anchor_id), {"positive": set(), "harm": set()}
            )["harm"].add(family)
    p5 = config["p5_descriptor_controller"]
    minimum_families = int(p5["minimum_pose_families"])
    potential_ids = {
        anchor_id
        for anchor_id, signs in signs_by_anchor.items()
        if len(signs["positive"] | signs["harm"]) >= minimum_families
        and not bool(signs["positive"] & signs["harm"])
    }
    observation_banks = {}
    if potential_ids:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            cache.get("schema") != "render_observation_cache_v2"
            or cache.get("uses_source_mapping_rgb") is not False
            or cache.get("uses_test_queries") is not False
        ):
            raise ValueError("P5 observation bank is not frozen rendered mapping data")
        id_to_row = {
            int(anchor_id): row
            for row, anchor_id in enumerate(candidates["anchor_ids"].tolist())
        }
        observations = candidates["projective_anchor_observations"]
        offsets = torch.as_tensor(observations["observation_offsets"]).long()
        query_indices = torch.as_tensor(observations["query_indices"]).long()
        keypoint_indices = torch.as_tensor(observations["keypoint_indices"]).long()
        names = candidates["query_names"]
        bins = torch.as_tensor(candidates["query_bins"]).long()
        for anchor_id in potential_ids:
            if anchor_id not in id_to_row:
                continue
            anchor_row = id_to_row[anchor_id]
            start, end = int(offsets[anchor_row]), int(offsets[anchor_row + 1])
            descriptors = []
            families = []
            for query, keypoint in zip(
                query_indices[start:end].tolist(), keypoint_indices[start:end].tolist()
            ):
                descriptors.append(
                    torch.as_tensor(
                        cache["queries"][names[query]]["native_descriptors"][keypoint]
                    ).float()
                )
                families.append(int(bins[query]))
            observation_banks[anchor_id] = {
                "descriptors": torch.stack(descriptors),
                "view_families": torch.tensor(families, dtype=torch.long),
            }
    reconstruction = reconstruct_v7_descriptors(
        anchor_ids=deployed_map["anchor_ids"],
        current_descriptors=deployed_map["anchor_features"],
        feedback_evidence=evidence_rows,
        observation_banks=observation_banks,
        minimum_pose_families=minimum_families,
        learning_rate=float(p5["learning_rate"]),
        harmful_weight=float(p5["harmful_weight"]),
        minimum_relative_weight=float(p5["minimum_relative_weight"]),
        maximum_relative_weight=float(p5["maximum_relative_weight"]),
        trim_fraction=float(p5["trim_fraction"]),
        maximum_descriptor_angle_deg=float(p5["maximum_descriptor_angle_deg"]),
    )
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    reconstruction_path = args.output_dir / "descriptor_reconstruction.pt"
    _atomic_torch_save(reconstruction, reconstruction_path)
    changed = int(reconstruction["changed_anchor_count"])
    proposal_path = None
    proposal_sha = input_shas["map"]
    stop_reason = "no_executable_descriptor_deficit" if changed == 0 else None
    if changed:
        proposal = dict(deployed_map)
        proposal["anchor_features"] = reconstruction["anchor_features"]
        proposal["provenance"] = {
            **dict(deployed_map.get("provenance", {})),
            "v7_descriptor_reconstruction": True,
            "feedback_descriptors_copied_into_map": False,
        }
        proposal_path = args.output_dir / "deployed_map_proposal.pt"
        _atomic_torch_save(proposal, proposal_path)
        proposal_sha = sha256_file(proposal_path)
    manifest = {
        "schema": "lafgs_v7_p5_descriptor_control",
        "version": 1,
        "phase": "P5",
        "status": "PASS",
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "feedback_descriptors_copied_into_map": False,
        "feedback_query_count": int(feedback["query_count"]),
        "representation_deficit_query_count": int(
            feedback["category_counts"]["representation_deficit"]
        ),
        "precision_deficit_query_count": int(
            feedback["category_counts"].get("precision_deficit", 0)
        ),
        "update_authorized_query_count": int(feedback["update_authorized_count"]),
        "potential_anchor_count": len(potential_ids),
        "changed_anchor_count": changed,
        "stop_reason": stop_reason,
        "baseline_map_sha256": input_shas["map"],
        "proposal_map_sha256": proposal_sha,
        "input": {
            "feedback_batch": str(feedback_path),
            "candidate_pool": str(candidate_path),
            "mapping_feature_cache": str(cache_path),
            "map": str(map_path),
            "metric": str(metric_path),
            **{f"{key}_sha256": value for key, value in input_shas.items()},
        },
        "output": {
            "descriptor_reconstruction": str(reconstruction_path.resolve()),
            "descriptor_reconstruction_sha256": sha256_file(reconstruction_path),
            "deployed_map_proposal": (
                None if proposal_path is None else str(proposal_path.resolve())
            ),
            "deployed_map_proposal_sha256": (
                None if proposal_path is None else proposal_sha
            ),
        },
        "formal_import_graph": import_audit,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "p0",
            "p2",
            "p3-prepare",
            "p3-select",
            "p4-feedback",
            "p5-control",
        ),
        default="p0",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/v7_safe_closed_loop.yaml")
    )
    parser.add_argument(
        "--formal-source-allowlist",
        type=Path,
        default=Path("configs/v7_formal_source_allowlist.json"),
    )
    parser.add_argument("--baseline-map", type=Path)
    parser.add_argument("--expected-baseline-map-sha256")
    parser.add_argument("--baseline-metric", type=Path)
    parser.add_argument("--expected-baseline-metric-sha256")
    parser.add_argument("--reference-results", type=Path)
    parser.add_argument("--expected-reference-results-sha256")
    parser.add_argument("--candidate-results", type=Path)
    parser.add_argument("--expected-candidate-results-sha256")
    parser.add_argument("--reference-deployment-contract", type=Path)
    parser.add_argument("--candidate-deployment-contract", type=Path)
    parser.add_argument("--expected-query-count", type=int, default=530)
    parser.add_argument("--query-plan", type=Path)
    parser.add_argument("--expected-query-plan-sha256")
    parser.add_argument("--gaussian-ply", type=Path)
    parser.add_argument("--expected-gaussian-ply-sha256")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--nms-radius", type=int, default=4)
    parser.add_argument("--keypoints", type=int, default=2048)
    parser.add_argument("--detection-threshold", type=float, default=0.0)
    parser.add_argument("--candidate-pool", type=Path)
    parser.add_argument("--expected-candidate-pool-sha256")
    parser.add_argument("--mapping-feature-cache", type=Path)
    parser.add_argument("--expected-mapping-feature-cache-sha256")
    parser.add_argument("--candidate-evidence", type=Path)
    parser.add_argument("--expected-candidate-evidence-sha256")
    parser.add_argument("--certified-batch", type=Path)
    parser.add_argument("--expected-certified-batch-sha256")
    parser.add_argument("--feedback-batch", type=Path)
    parser.add_argument("--expected-feedback-batch-sha256")
    parser.add_argument(
        "--selector-profile",
        choices=(
            "large_sufficient",
            "medium_sufficient",
            "small_sufficient",
            "aggressive_minimum",
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    required_by_phase = {
        "p0": (
            "baseline_map",
            "expected_baseline_map_sha256",
            "baseline_metric",
            "expected_baseline_metric_sha256",
        ),
        "p2": (
            "query_plan",
            "expected_query_plan_sha256",
            "gaussian_ply",
            "expected_gaussian_ply_sha256",
            "dataset",
        ),
        "p3-prepare": (
            "candidate_pool",
            "expected_candidate_pool_sha256",
            "mapping_feature_cache",
            "expected_mapping_feature_cache_sha256",
        ),
        "p3-select": (
            "candidate_pool",
            "expected_candidate_pool_sha256",
            "candidate_evidence",
            "expected_candidate_evidence_sha256",
            "baseline_map",
            "expected_baseline_map_sha256",
            "baseline_metric",
            "expected_baseline_metric_sha256",
            "selector_profile",
        ),
        "p4-feedback": (
            "certified_batch",
            "expected_certified_batch_sha256",
            "candidate_pool",
            "expected_candidate_pool_sha256",
            "candidate_evidence",
            "expected_candidate_evidence_sha256",
            "baseline_map",
            "expected_baseline_map_sha256",
            "baseline_metric",
            "expected_baseline_metric_sha256",
        ),
        "p5-control": (
            "feedback_batch",
            "expected_feedback_batch_sha256",
            "candidate_pool",
            "expected_candidate_pool_sha256",
            "mapping_feature_cache",
            "expected_mapping_feature_cache_sha256",
            "baseline_map",
            "expected_baseline_map_sha256",
            "baseline_metric",
            "expected_baseline_metric_sha256",
        ),
    }
    required = required_by_phase[args.phase]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(
            f"--phase {args.phase} misses required arguments: {', '.join(missing)}"
        )
    runners = {
        "p0": run_p0,
        "p2": run_p2,
        "p3-prepare": run_p3_prepare,
        "p3-select": run_p3_select,
        "p4-feedback": run_p4,
        "p5-control": run_p5,
    }
    report = runners[args.phase](args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
