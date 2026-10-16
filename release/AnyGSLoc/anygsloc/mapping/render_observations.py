#!/usr/bin/env python3


"""Frozen mainline kernel extracted from scripts.probe_rendered_rgb_track_map."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import anygsloc.priors.mapping_supersampling as mapping_supersampling_module
from anygsloc.data.datasets import ColmapDataset
from anygsloc.features.extractor import FeatureExtractor
from anygsloc.features.superpoint import render_validity_mask_from_alpha
from anygsloc.mapping.contracts import RENDER_OBSERVATION_SCHEMA
from anygsloc.mapping.virtual_camera_registry import (
    build_virtual_camera_registry,
    camera_intrinsic,
    resolve_virtual_camera_registry,
)
from anygsloc.priors.mapping_supersampling import (
    mapping_render_sampling_contract,
    render_mapping_rgbd,
)
from anygsloc.priors.models import GaussianModel2D, GaussianModel3D
from anygsloc.priors.rendering import render_from_pose_gsplat
from anygsloc.utils.hashing import sha256_file


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _intrinsic(camera) -> torch.Tensor:
    return camera_intrinsic(camera)


def _resolve_render_cameras(mapping, args):
    camera_registry_path = getattr(args, "camera_registry", None)
    if camera_registry_path is not None:
        if int(args.max_views) != 0:
            raise ValueError(
                "an explicit camera registry cannot be combined with max-views"
            )
        expected_registry_sha = getattr(args, "expected_camera_registry_sha256", None)
        if not expected_registry_sha:
            raise ValueError("an explicit camera registry requires its expected SHA")
        actual_registry_sha = sha256_file(camera_registry_path)
        if actual_registry_sha != expected_registry_sha:
            raise ValueError("explicit camera registry SHA differs")
        camera_registry = json.loads(camera_registry_path.read_text())
        cameras = resolve_virtual_camera_registry(mapping, camera_registry)
        return cameras, camera_registry, actual_registry_sha
    cameras, camera_registry = build_virtual_camera_registry(
        mapping,
        args.max_views,
        deduplicate_geometry=args.deduplicate_camera_geometry,
    )
    return cameras, camera_registry, None


@torch.inference_mode()
def _render_feature_cache(args, output: Path) -> dict:
    started = time.perf_counter()
    prior_sha256 = sha256_file(args.gaussian_ply)
    dataset = ColmapDataset(args.dataset, images=args.images)
    mapping = dataset.split("mapping")
    cameras, camera_registry, actual_registry_sha = _resolve_render_cameras(
        mapping, args
    )
    camera_registry_path = getattr(args, "camera_registry", None)

    model = (
        GaussianModel2D(args.sh_degree)
        if args.gaussian_type == "2dgs"
        else GaussianModel3D(args.sh_degree)
    )
    # This experiment consumes only Gaussian RGB appearance.  Do not
    # materialize the fallback random localization-feature bank when the PLY
    # has no loc_* fields; it is unused by rgb_only rendering and can dominate
    # prior-loading time and memory for million-Gaussian scenes.
    model.load_ply(args.gaussian_ply, loc_feature_dim=0)
    model = model.cuda().eval()
    extractor = FeatureExtractor("sp", nms_radius=args.nms_radius).cuda().eval()
    extractor.requires_grad_(False)
    background = (
        torch.ones(3, device="cuda")
        if args.white_background
        else torch.zeros(3, device="cuda")
    )

    records: dict[str, dict] = {}
    render_seconds = 0.0
    feature_seconds = 0.0
    raw_detector_rows = 0
    valid_detector_rows = 0
    valid_pixels = 0
    total_pixels = 0
    render_sampling_contract = None
    for row, camera in enumerate(cameras):
        pose = torch.from_numpy(camera.pose_w2c).cuda().float()
        render_started = time.perf_counter()
        render_valid = args.render_valid_alpha_minimum is not None
        if int(args.render_supersampling_factor) == 1:
            package = render_from_pose_gsplat(
                model,
                pose,
                camera.fov_x,
                camera.fov_y,
                camera.width,
                camera.height,
                bg_color=background,
                render_mode="RGB+ED" if render_valid else "RGB",
                # V6 needs RGB, alpha, and proposal-only depth, never the legacy
                # learned/random Gaussian localization-feature render.
                rgb_only=True,
                rasterize_mode="antialiased",
            )
            render_sampling_contract = mapping_render_sampling_contract(
                gaussian_type=args.gaussian_type, supersampling_factor=1
            )
        else:
            package, render_sampling_contract = render_mapping_rgbd(
                model,
                pose,
                fov_x=camera.fov_x,
                fov_y=camera.fov_y,
                width=camera.width,
                height=camera.height,
                background=background,
                gaussian_type=args.gaussian_type,
                supersampling_factor=args.render_supersampling_factor,
            )
        rendered_payload = package["render"].float()
        rendered = rendered_payload[:3].clamp(0.0, 1.0)
        validity_mask = None
        rendered_alpha = None
        if render_valid:
            rendered_alpha = package.get("alphas")
            if rendered_alpha is None:
                rendered_alpha = package.get("rend_alpha")
            if rendered_alpha is None:
                raise ValueError("render-valid extraction requires renderer alpha")
            rendered_depth = package.get("depth")
            if rendered_depth is None:
                raise ValueError("V6 observation extraction requires proposal depth")
            validity_mask = render_validity_mask_from_alpha(
                rendered_alpha,
                minimum_alpha=float(args.render_valid_alpha_minimum),
                neighborhood_radius=int(args.render_valid_neighborhood_radius),
            )
        torch.cuda.synchronize()
        render_seconds += time.perf_counter() - render_started

        feature_started = time.perf_counter()
        if validity_mask is None:
            sparse = extractor.detectAndCompute(
                rendered[None],
                top_k=args.keypoints,
                detection_threshold=args.detection_threshold,
            )[0]
            raw = sparse
        else:
            valid_batch, raw_batch = extractor.detectAndComputeWithValidityAudit(
                rendered[None],
                top_k=args.keypoints,
                detection_threshold=args.detection_threshold,
                validity_mask=validity_mask,
            )
            sparse, raw = valid_batch[0], raw_batch[0]
        torch.cuda.synchronize()
        feature_seconds += time.perf_counter() - feature_started
        keypoints = sparse["keypoints"].detach().cpu().float()
        descriptors = F.normalize(sparse["descriptors"].detach().cpu().float(), dim=1)
        scores = sparse["keypoint_scores"].detach().cpu().float()
        raw_count = int(raw["keypoints"].shape[0])
        valid_count = int(keypoints.shape[0])
        raw_detector_rows += raw_count
        valid_detector_rows += valid_count
        if validity_mask is not None:
            valid_pixels += int(validity_mask.sum())
            total_pixels += int(validity_mask.numel())
        records[camera.image_name] = {
            "native_keypoints": keypoints,
            "native_descriptors": descriptors,
            "native_scores": scores,
            "native_K": _intrinsic(camera),
            "pose_w2c": torch.from_numpy(camera.pose_w2c).float(),
            "native_input_hw": torch.tensor(
                [camera.height, camera.width], dtype=torch.long
            ),
            "source": "gaussian_rendered_rgb_only",
            "render_valid_diagnostics": {
                "raw_detector_rows": raw_count,
                "valid_detector_rows": valid_count,
                "invalid_or_boundary_rows_removed": raw_count - valid_count,
            },
            **(
                {
                    "native_valid_mask": validity_mask[0].detach().cpu(),
                    "native_alpha": rendered_alpha.squeeze().detach().cpu().half(),
                    "native_depth": rendered_depth.squeeze().detach().cpu().float(),
                }
                if validity_mask is not None
                else {}
            ),
        }
        if (row + 1) % max(int(args.progress_interval), 1) == 0 or row + 1 == len(
            cameras
        ):
            print(
                f"rendered {row + 1}/{len(cameras)} views; rows={keypoints.shape[0]}",
                flush=True,
            )

    if sha256_file(args.gaussian_ply) != prior_sha256:
        raise ValueError("Gaussian prior changed during render materialization")
    payload = {
        "schema": (
            RENDER_OBSERVATION_SCHEMA
            if args.render_valid_alpha_minimum is not None
            else "lafgs_rendered_rgb_only_sparse_mapping_cache"
        ),
        "version": 2 if args.render_valid_alpha_minimum is not None else 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "uses_rendered_depth": args.render_valid_alpha_minimum is not None,
        "uses_gaussian_geometry_for_triangulation": False,
        "input": {
            "gaussian_ply": str(Path(args.gaussian_ply).resolve()),
            "gaussian_ply_sha256": prior_sha256,
        },
        "implementation_sha256": {
            "observation_materializer": sha256_file(Path(__file__)),
            "mapping_supersampling": sha256_file(
                Path(mapping_supersampling_module.__file__)
            ),
        },
        "mapping_query_count": len(cameras),
        "full_mapping_query_count": len(mapping),
        # Kept only for archived artifact readers. New stages resolve by the
        # attested names below, never by this dataset-order-dependent vector.
        "source_mapping_indices": torch.tensor(
            camera_registry["selected_legacy_dataset_indices"], dtype=torch.long
        ),
        "virtual_camera_registry": camera_registry,
        "explicit_camera_registry": (
            None
            if camera_registry_path is None
            else {
                "path": str(camera_registry_path.resolve()),
                "sha256": actual_registry_sha,
            }
        ),
        "queries": records,
        "configuration": {
            "gaussian_type": args.gaussian_type,
            "sh_degree": args.sh_degree,
            "keypoints": args.keypoints,
            "nms_radius": args.nms_radius,
            "detection_threshold": args.detection_threshold,
            "background": "white" if args.white_background else "black",
            "render_mode": (
                "RGB+ED" if args.render_valid_alpha_minimum is not None else "RGB"
            ),
            "rasterize_mode": render_sampling_contract["effective_rasterize_mode"],
            "mapping_render_sampling": render_sampling_contract,
            "render_valid_observation_extraction": (
                None
                if args.render_valid_alpha_minimum is None
                else {
                    "schema": "lafgs_v6_render_valid_observation_policy",
                    "version": 1,
                    "score_gate_stage": "before_native_nms_and_topk",
                    "minimum_alpha": float(args.render_valid_alpha_minimum),
                    "neighborhood_radius": int(args.render_valid_neighborhood_radius),
                    "appearance_reliability_used": False,
                }
            ),
        },
        "timing_seconds": {
            "render": render_seconds,
            "superpoint": feature_seconds,
            "total": time.perf_counter() - started,
        },
        "render_valid_diagnostics": {
            "raw_detector_rows": raw_detector_rows,
            "valid_detector_rows": valid_detector_rows,
            "invalid_or_boundary_rows_removed": (
                raw_detector_rows - valid_detector_rows
            ),
            "valid_region_pixel_fraction": (
                None if total_pixels == 0 else valid_pixels / total_pixels
            ),
            "valid_rows_per_megapixel": (
                None
                if valid_pixels == 0
                else valid_detector_rows * 1000000.0 / valid_pixels
            ),
        },
    }
    _atomic_torch_save(payload, output)
    print(f"wrote rendered feature cache {output}", flush=True)
    return payload


def main(argv=None):
    """Render only the frozen observations, never the historical Track probe."""
    import argparse

    from anygsloc.mapping.build import MAPPING

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--gaussian-ply", type=Path, required=True)
    parser.add_argument("--gaussian-type", choices=["2dgs", "3dgs"], required=True)
    parser.add_argument("--sh-degree", type=int, required=True)
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.max_views = 0
    args.deduplicate_camera_geometry = True
    args.keypoints = MAPPING.keypoints
    args.nms_radius = MAPPING.nms_radius
    args.detection_threshold = 0.0
    args.render_valid_alpha_minimum = MAPPING.render_alpha_minimum
    args.render_valid_neighborhood_radius = MAPPING.render_neighborhood_radius
    args.render_supersampling_factor = MAPPING.render_supersampling
    args.progress_interval = 25
    args.device = "cuda:0"
    torch.set_num_threads(MAPPING.cpu_threads)
    if not torch.cuda.is_available():
        raise RuntimeError("Gaussian observation rendering requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _render_feature_cache(args, args.output_dir / "rendered_rgb_feature_cache.pt")


if __name__ == "__main__":
    main()
