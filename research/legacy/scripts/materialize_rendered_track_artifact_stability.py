#!/usr/bin/env python3
"""Reweight a frozen rendered-Track map by raw/clean 2DGS stability.

This R1 materializer is descriptor-only.  It preserves the exact keypoint
rows, pair graph, Track components, ray-triangulated xyz, selected Track IDs,
row order, and map cardinality of a frozen R0 map.  A 2DGS distortion buffer
drives a second opacity-suppressed render; raw/clean stability only multiplies
the existing observation reliability used by Track descriptor fusion.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from data.datasets import ColmapDataset
from evidence.render_artifact_stability import (
    gaussian_opacity_multiplier,
    local_peak_stability,
    normalized_distortion_risk,
    observation_artifact_reliability,
    quantile_summary,
    sample_plane_nearest,
)
from evidence.tracks import fuse_track_descriptors
from features.extractor import FeatureExtractor
from features.superpoint import sample_descriptors
from map_learning.metric import SharedLowRankMetric
from priors.models import GaussianModel2D
from priors.rendering import render_from_pose_gsplat
from evidence.virtual_camera_registry import resolve_virtual_camera_registry
from scripts.probe_rendered_rgb_track_map import _intrinsic


RISK_REFERENCE_QUANTILE = 0.95
OPACITY_SUPPRESSION_STRENGTH = 0.50
ARTIFACT_ALPHA_MINIMUM = 0.05
PEAK_SEARCH_RADIUS_PX = 4
POSITION_SIGMA_PX = 2.0

_SOURCE_PATHS = (
    "docs/evidence/rendered_rgb_track_artifact_stability_preregistration.json",
    "evidence/render_artifact_stability.py",
    "evidence/tracks.py",
    "features/superpoint.py",
    "priors/rendering.py",
    "scripts/materialize_rendered_track_artifact_stability.py",
)


def _producer_identity() -> dict:
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("artifact-stability producer worktree must be clean")
    return {
        "git_commit": commit,
        "worktree_clean": True,
        "source_sha256": {
            relative: sha256_file(repository / relative) for relative in _SOURCE_PATHS
        },
        "torch_version": torch.__version__,
    }


def _require_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"{label} SHA differs: expected {expected}, got {actual}")
    return actual


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
        if reloaded.get("schema") != payload.get("schema"):
            raise RuntimeError("temporary artifact schema did not reload")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if json.loads(temporary.read_text()).get("schema") != payload.get("schema"):
            raise RuntimeError("temporary report schema did not reload")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _alpha(package: dict) -> torch.Tensor:
    value = package.get("rend_alpha")
    if value is None:
        value = package.get("alphas")
    value = torch.as_tensor(value).squeeze().float()
    if value.ndim != 2 or not bool(torch.isfinite(value).all()):
        raise ValueError("2DGS alpha must be a finite [H,W] plane")
    return value


def _distortion(package: dict) -> torch.Tensor:
    value = package.get("rend_dist")
    if value is None:
        raise ValueError("R1 requires the 2DGS distortion buffer")
    value = torch.as_tensor(value).squeeze().float()
    if value.ndim != 2:
        raise ValueError("2DGS distortion must be a [H,W] plane")
    return value


def _projected_centres(package: dict) -> torch.Tensor:
    meta = package.get("rgb_meta") or {}
    value = meta.get("means2d")
    if value is None:
        value = package.get("viewspace_points")
    value = torch.as_tensor(value).squeeze(0)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError("2DGS projected centres must be [N,2]")
    return value


def _same_rows(left: dict, right: dict, *, name: str) -> None:
    for key in ("native_keypoints", "native_K", "pose_w2c", "native_input_hw"):
        if not torch.equal(torch.as_tensor(left[key]), torch.as_tensor(right[key])):
            raise ValueError(f"{name} raw/appearance cache field differs: {key}")


def _topology_fields(state: dict) -> tuple[str, ...]:
    required = (
        "anchor_ids",
        "anchor_xyz",
        "source_primitive_ids",
        "track_cluster_ids",
        "anchor_type",
        "dependency_group_ids",
        "coarse_dependency_group_ids",
        "fine_identity_ids",
        "source_dependency_group_ids",
        "parent_source_track_ids",
        "repair_child_index",
        "repair_parent_child_count",
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"selected R0 map lacks frozen topology fields: {missing}")
    return required


@torch.inference_mode()
def materialize(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    if str(args.device) != "cuda:0":
        raise ValueError("formal R1 device must be cuda:0 under CUDA_VISIBLE_DEVICES")
    inputs = {
        "gaussian_ply": args.gaussian_ply.resolve(),
        "raw_source_cache": args.raw_source_cache.resolve(),
        "appearance_cache": args.appearance_cache.resolve(),
        "track_payload": args.track_payload.resolve(),
        "selected_map": args.selected_map.resolve(),
    }
    expected = {
        "gaussian_ply": args.expected_gaussian_ply_sha256,
        "raw_source_cache": args.expected_raw_source_cache_sha256,
        "appearance_cache": args.expected_appearance_cache_sha256,
        "track_payload": args.expected_track_payload_sha256,
        "selected_map": args.expected_selected_map_sha256,
    }
    input_sha256 = {
        label: _require_sha(path, expected[label], label)
        for label, path in inputs.items()
    }
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    raw_cache = torch.load(
        inputs["raw_source_cache"], map_location="cpu", weights_only=False
    )
    appearance_cache = torch.load(
        inputs["appearance_cache"], map_location="cpu", weights_only=False
    )
    payload = torch.load(
        inputs["track_payload"], map_location="cpu", weights_only=False
    )
    source_map = torch.load(
        inputs["selected_map"], map_location="cpu", weights_only=False
    )
    if (
        raw_cache.get("uses_source_mapping_rgb") is not False
        or raw_cache.get("uses_test_queries") is not False
        or appearance_cache.get("uses_source_mapping_rgb") is not False
        or appearance_cache.get("uses_test_queries") is not False
        or payload.get("rendered_rgb_only") is not True
    ):
        raise ValueError("R1 inputs are not source-image-free mapping-only evidence")
    if appearance_cache.get("schema") != "lafgs_rendered_rgb_appearance_ensemble_cache":
        raise ValueError("R1 requires the frozen V1.4 appearance cache")
    if args.gaussian_type != "2dgs":
        raise ValueError("R1 distortion suppression is defined only for 2DGS")
    names = list(payload["query_names"])
    raw_queries = raw_cache["queries"]
    appearance_queries = appearance_cache["queries"]
    if names != list(raw_queries) or names != list(appearance_queries):
        raise ValueError("R1 cache and Track query registries differ")
    selected_tracks = torch.as_tensor(source_map["track_cluster_ids"]).long()
    if selected_tracks.unique().numel() != selected_tracks.numel():
        raise ValueError("selected R0 Track identities are not unique")
    frozen_topology = {
        key: torch.as_tensor(source_map[key]).clone()
        for key in _topology_fields(source_map)
    }

    dataset = ColmapDataset(args.dataset.resolve(), images=args.images)
    mapping = dataset.split("mapping")
    cameras = resolve_virtual_camera_registry(
        mapping, raw_cache.get("virtual_camera_registry")
    )
    if names != [camera.image_name for camera in cameras]:
        raise ValueError("R1 dataset mapping order differs from the frozen cache")

    model = GaussianModel2D(args.sh_degree, device=args.device)
    model.load_ply(inputs["gaussian_ply"], loc_feature_dim=0)
    model = model.eval()
    extractor = (
        FeatureExtractor("sp", nms_radius=args.nms_radius).to(args.device).eval()
    )
    extractor.requires_grad_(False)
    records: dict[str, dict] = {}
    metrics: dict[str, list[torch.Tensor]] = {
        "descriptor_cosine": [],
        "detector_score_stability": [],
        "position_stability": [],
        "position_displacement_px": [],
        "artifact_exposure": [],
        "artifact_reliability": [],
        "combined_reliability": [],
        "gaussian_risk": [],
    }
    distortion_scales = []
    rerender_cosines = []
    for query_index, camera in enumerate(cameras):
        name = camera.image_name
        raw_record = raw_queries[name]
        appearance = appearance_queries[name]
        _same_rows(raw_record, appearance, name=name)
        pose = torch.from_numpy(camera.pose_w2c).float()
        if not torch.equal(pose, torch.as_tensor(raw_record["pose_w2c"]).float()):
            raise ValueError(f"mapping pose differs for {name}")
        if not torch.equal(_intrinsic(camera), torch.as_tensor(raw_record["native_K"])):
            raise ValueError(f"mapping intrinsic differs for {name}")
        raw_package = render_from_pose_gsplat(
            model,
            pose.to(args.device),
            camera.fov_x,
            camera.fov_y,
            camera.width,
            camera.height,
            bg_color=torch.zeros(3, device=args.device),
            # R1 consumes RGB, alpha, and the independent 2DGS distortion
            # buffer; it does not consume rendered depth.  Keeping the render
            # mode at RGB also avoids the known gsplat 1.4 RGB+ED background
            # channel mismatch without changing any signal used by R1.
            render_mode="RGB",
            rgb_only=True,
            rasterize_mode="antialiased",
            return_rgb_meta=True,
        )
        raw_rgb = raw_package["render"].float().clamp(0.0, 1.0)
        raw_alpha = _alpha(raw_package)
        risk, scale = normalized_distortion_risk(
            _distortion(raw_package),
            raw_alpha,
            reference_quantile=RISK_REFERENCE_QUANTILE,
            alpha_minimum=ARTIFACT_ALPHA_MINIMUM,
        )
        multiplier, gaussian_risk = gaussian_opacity_multiplier(
            risk,
            _projected_centres(raw_package),
            torch.as_tensor(raw_package["visibility_filter"]).bool(),
            suppression_strength=OPACITY_SUPPRESSION_STRENGTH,
        )
        clean_package = render_from_pose_gsplat(
            model,
            pose.to(args.device),
            camera.fov_x,
            camera.fov_y,
            camera.width,
            camera.height,
            bg_color=torch.zeros(3, device=args.device),
            render_mode="RGB",
            rgb_only=True,
            rasterize_mode="antialiased",
            opacity_multiplier=multiplier,
        )
        clean_rgb = clean_package["render"].float().clamp(0.0, 1.0)
        raw_dense, _ = extractor.detectAndComputeDense(raw_rgb[None])
        clean_dense, clean_score_map = extractor.detectAndComputeDense(clean_rgb[None])
        keypoints = (
            torch.as_tensor(raw_record["native_keypoints"]).float().to(args.device)
        )
        raw_descriptors = sample_descriptors(keypoints[None], raw_dense)[0].transpose(
            0, 1
        )
        clean_descriptors = sample_descriptors(keypoints[None], clean_dense)[
            0
        ].transpose(0, 1)
        frozen_raw = F.normalize(
            torch.as_tensor(raw_record["native_descriptors"]).float().to(args.device),
            dim=1,
        )
        rerender_cosine = (raw_descriptors * frozen_raw).sum(dim=1)
        if float(rerender_cosine.min()) < 0.999:
            raise ValueError(f"raw rerender does not reproduce frozen rows for {name}")
        score_stability, position_stability, displacement = local_peak_stability(
            clean_score_map[0, 0],
            keypoints,
            torch.as_tensor(raw_record["native_scores"]).float().to(args.device),
            nms_radius=args.nms_radius,
            search_radius=PEAK_SEARCH_RADIUS_PX,
            position_sigma_px=POSITION_SIGMA_PX,
        )
        stability = observation_artifact_reliability(
            raw_descriptors,
            clean_descriptors,
            score_stability,
            position_stability,
            sample_plane_nearest(risk, keypoints),
        )
        appearance_reliability = (
            torch.as_tensor(
                appearance.get(
                    "native_appearance_reliability",
                    torch.ones(keypoints.shape[0]),
                )
            )
            .float()
            .to(args.device)
        )
        combined = (appearance_reliability * stability["reliability"]).clamp(0.0, 1.0)
        records[name] = {
            **appearance,
            "native_appearance_reliability": combined.cpu(),
            "native_artifact_reliability": stability["reliability"].cpu(),
            "native_artifact_exposure": stability["artifact_exposure"].cpu(),
            "native_raw_clean_descriptor_cosine": stability["descriptor_cosine"].cpu(),
            "native_raw_clean_detector_score_stability": stability[
                "detector_score_stability"
            ].cpu(),
            "native_raw_clean_position_stability": stability[
                "position_stability"
            ].cpu(),
            "native_raw_clean_position_displacement_px": displacement.cpu(),
            "source": "gaussian_rendered_rgb_artifact_stability_r1",
        }
        for key in (
            "descriptor_cosine",
            "detector_score_stability",
            "position_stability",
            "artifact_exposure",
            "artifact_reliability",
        ):
            source = "reliability" if key == "artifact_reliability" else key
            metrics[key].append(stability[source].cpu())
        metrics["position_displacement_px"].append(displacement.cpu())
        metrics["combined_reliability"].append(combined.cpu())
        metrics["gaussian_risk"].append(gaussian_risk[gaussian_risk > 0].cpu())
        distortion_scales.append(scale.cpu().reshape(1))
        rerender_cosines.append(rerender_cosine.cpu())
        if (query_index + 1) % max(int(args.progress_interval), 1) == 0 or (
            query_index + 1 == len(cameras)
        ):
            print(
                json.dumps(
                    {
                        "completed_views": query_index + 1,
                        "mean_artifact_reliability": float(
                            torch.cat(metrics["artifact_reliability"]).mean()
                        ),
                    }
                ),
                flush=True,
            )

    stability_cache = {
        **appearance_cache,
        "schema": "lafgs_rendered_rgb_artifact_stability_cache",
        "version": 1,
        "queries": records,
        "artifact_stability": {
            "topology_frozen": True,
            "descriptors_remain_v14_appearance_descriptors": True,
            "risk_source": "2dgs_rendered_distortion",
            "risk_reference_quantile": RISK_REFERENCE_QUANTILE,
            "opacity_suppression_strength": OPACITY_SUPPRESSION_STRENGTH,
            "artifact_alpha_minimum": ARTIFACT_ALPHA_MINIMUM,
            "peak_search_radius_px": PEAK_SEARCH_RADIUS_PX,
            "position_sigma_px": POSITION_SIGMA_PX,
            "reliability_fusion": "geometric_mean_then_multiply_v14_reliability",
        },
    }
    fused = fuse_track_descriptors(
        payload=payload,
        query_cache=stability_cache,
        track_indices=selected_tracks,
        trim_fraction=args.descriptor_trim_fraction,
    )
    output_state = dict(source_map)
    output_state["anchor_features"] = fused.float()
    output_state["v7_metric_raw_features"] = fused.float()
    output_state["provenance"] = {
        **source_map.get("provenance", {}),
        "rendered_artifact_stability_r1": {
            "source_map": str(inputs["selected_map"]),
            "source_cache": str(inputs["appearance_cache"]),
            "fixed_keypoints_pairs_tracks_xyz_selection_and_size": True,
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
        },
    }
    for key, reference in frozen_topology.items():
        if not torch.equal(torch.as_tensor(output_state[key]), reference):
            raise RuntimeError(f"R1 modified frozen topology field {key}")

    identity = _producer_identity()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cache_path = args.output_dir / "artifact_stability_cache.pt"
    map_path = args.output_dir / "artifact_stability_anchor_map.pt"
    metric_path = args.output_dir / "artifact_stability_identity_metric.pt"
    _atomic_torch_save(stability_cache, cache_path)
    _atomic_torch_save(output_state, map_path)
    metric = SharedLowRankMetric(
        descriptor_dim=fused.shape[1], rank=1, max_residual_norm=0.0
    )
    with torch.no_grad():
        for parameter in metric.parameters():
            parameter.zero_()
    metric_state = {
        "schema": "lafgs_shared_metric_state",
        "version": 1,
        "landmark_indices": torch.arange(fused.shape[0], dtype=torch.long),
        "metric_config": metric.export_config(),
        "metric_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in metric.state_dict().items()
        },
        "map_path": str(map_path.resolve()),
        "map_sha256": sha256_file(map_path),
        "step": 0,
        "protocol": "rendered_track_artifact_stability_r1_identity",
    }
    _atomic_torch_save(metric_state, metric_path)
    if _producer_identity() != identity:
        raise RuntimeError("R1 producer identity changed during materialization")
    for label, path in inputs.items():
        _require_sha(path, input_sha256[label], label)
    summaries = {
        key: quantile_summary(torch.cat(values))
        for key, values in metrics.items()
        if values and any(value.numel() for value in values)
    }
    report = {
        "schema": "lafgs_rendered_track_artifact_stability_r1_materialization",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "uses_gaussian_primitive_geometry_for_anchor_xyz": False,
        "fixed_keypoint_rows": True,
        "fixed_pair_graph": True,
        "fixed_track_components": True,
        "fixed_anchor_xyz": True,
        "fixed_selector_membership": True,
        "fixed_map_cardinality": True,
        "only_observation_fusion_weight_changes": True,
        "mapping_query_count": len(cameras),
        "anchor_count": int(fused.shape[0]),
        "configuration": stability_cache["artifact_stability"],
        "producer_identity": identity,
        "inputs": {key: str(path) for key, path in inputs.items()},
        "input_sha256": input_sha256,
        "outputs": {
            "query_cache": str(cache_path.resolve()),
            "anchor_map": str(map_path.resolve()),
            "identity_metric": str(metric_path.resolve()),
        },
        "output_sha256": {
            "query_cache": sha256_file(cache_path),
            "anchor_map": sha256_file(map_path),
            "identity_metric": sha256_file(metric_path),
        },
        "raw_rerender_descriptor_cosine_minimum": float(
            torch.cat(rerender_cosines).min()
        ),
        "distortion_reference_scale": quantile_summary(torch.cat(distortion_scales)),
        "stability": summaries,
        "timing_seconds": {"total": time.perf_counter() - started},
    }
    _atomic_json(report, args.output_dir / "artifact_stability_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--gaussian-ply", type=Path, required=True)
    parser.add_argument("--expected-gaussian-ply-sha256", required=True)
    parser.add_argument("--raw-source-cache", type=Path, required=True)
    parser.add_argument("--expected-raw-source-cache-sha256", required=True)
    parser.add_argument("--appearance-cache", type=Path, required=True)
    parser.add_argument("--expected-appearance-cache-sha256", required=True)
    parser.add_argument("--track-payload", type=Path, required=True)
    parser.add_argument("--expected-track-payload-sha256", required=True)
    parser.add_argument("--selected-map", type=Path, required=True)
    parser.add_argument("--expected-selected-map-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gaussian-type", choices=("2dgs",), default="2dgs")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--nms-radius", type=int, default=4)
    parser.add_argument("--descriptor-trim-fraction", type=float, default=0.2)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(materialize(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
