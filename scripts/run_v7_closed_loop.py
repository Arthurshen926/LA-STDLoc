#!/usr/bin/env python3
"""Run the fail-closed V7 mainline; P0 supports identity no-op only."""

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
    certify_v7_render,
    extreme_distortion_row_mask,
)
from features.extractor import FeatureExtractor
from priors.models import GaussianModel2D
from priors.rendering import render_from_pose_gsplat


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
    map_sha = _require_sha(source_map, args.expected_baseline_map_sha256, "baseline map")
    metric_sha = _require_sha(source_metric, args.expected_baseline_metric_sha256, "baseline metric")
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
        _require_sha(reference_results, args.expected_reference_results_sha256, "reference results")
        _require_sha(candidate_results, args.expected_candidate_results_sha256, "candidate results")
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
    load_v7_config(args.config)
    import_audit = audit_formal_import_graph(
        root=root,
        entrypoint=Path(__file__),
        allowlist_path=args.formal_source_allowlist,
    )
    plan_path = args.query_plan.resolve()
    prior_path = args.gaussian_ply.resolve()
    plan_sha = _require_sha(plan_path, args.expected_query_plan_sha256, "query plan")
    prior_sha = _require_sha(prior_path, args.expected_gaussian_ply_sha256, "Gaussian prior")
    plan = torch.load(plan_path, map_location="cpu", weights_only=False)
    role = str(plan.get("view_role"))
    if role not in {"feedback_query", "confirmation_query"}:
        raise ValueError("P2 accepts only feedback or confirmation query plans")
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
        pose = torch.as_tensor(plan["pose_w2c"][index], device=device).float()
        intrinsic = torch.as_tensor(plan["intrinsics"][index]).float()
        height, width = map(int, torch.as_tensor(plan["image_hw"][index]).tolist())
        fov_x = 2.0 * torch.atan(torch.tensor(width / (2.0 * float(intrinsic[0, 0])))).item()
        fov_y = 2.0 * torch.atan(torch.tensor(height / (2.0 * float(intrinsic[1, 1])))).item()
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
        rgb = package["render"].float().clamp(0.0, 1.0)
        alpha = package.get("alphas", package.get("rend_alpha"))
        depth = package.get("depth")
        if alpha is None or depth is None:
            raise ValueError("P2 renderer must return RGB, alpha, and depth")
        sparse = extractor.detectAndCompute(
            rgb[None],
            top_k=args.keypoints,
            detection_threshold=args.detection_threshold,
        )[0]
        keypoints = sparse["keypoints"].detach().cpu().float()
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
            else extreme_distortion_row_mask(distortion.detach().cpu(), keypoints)
        )
        certificate = certify_v7_render(
            rgb=rgb.detach().cpu(),
            alpha=alpha.detach().cpu(),
            depth=depth.detach().cpu(),
            keypoints=keypoints,
            nearest_mapping_distance_m=float(nearest_distances[index]),
            median_adjacent_baseline_m=float(
                plan["trajectory_statistics"]["median_adjacent_baseline_m"]
            ),
            source_family_support=len(set(plan["source_mapping_indices"][index])),
            expected_median_depth_m=expected_median_depth,
            artifact_row_mask=artifact_row_mask,
        )
        decision = certificate["decision"]
        decisions[decision] += 1
        record = {
            "schema": "lafgs_v7_certified_clean_render",
            "version": 1,
            "view_role": role,
            "query_index": index,
            "pose_family_id": int(plan["pose_family_ids"][index]),
            "pose_w2c": torch.as_tensor(plan["pose_w2c"][index]).float(),
            "intrinsics": intrinsic,
            "image_hw": torch.tensor([height, width], dtype=torch.long),
            "rgb_uint8": (rgb.detach().cpu() * 255.0).round().to(torch.uint8),
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
        record_path = records_dir / f"query_{index:04d}.pt"
        temporary = record_path.with_name(f".{record_path.name}.{os.getpid()}.tmp")
        try:
            torch.save(record, temporary)
            os.replace(temporary, record_path)
        finally:
            temporary.unlink(missing_ok=True)
        record_registry.append(
            {
                "query_index": index,
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
        "uses_test_queries": False,
        "render_protocol": "clean_once_per_pose",
        "detector_input": "complete_unmasked_rgb",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("p0", "p2"), default="p0")
    parser.add_argument("--config", type=Path, default=Path("configs/v7_safe_closed_loop.yaml"))
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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    required = (
        ("baseline_map", "expected_baseline_map_sha256", "baseline_metric", "expected_baseline_metric_sha256")
        if args.phase == "p0"
        else ("query_plan", "expected_query_plan_sha256", "gaussian_ply", "expected_gaussian_ply_sha256", "dataset")
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"--phase {args.phase} misses required arguments: {', '.join(missing)}")
    report = run_p0(args) if args.phase == "p0" else run_p2(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
