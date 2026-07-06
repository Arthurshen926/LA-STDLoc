#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from arguments import ModelParams
from la_artifacts.detector import ArtifactDetector
from la_artifacts.no_reference_valid_mask import (
    NoReferenceValidMaskBuilder,
    NoReferenceValidMaskConfig,
    save_no_reference_valid_mask_pngs,
)
from la_artifacts.pseudo_query import PseudoQueryManifest, PseudoTeacherCache
from la_artifacts.valid_mask import ArtifactValidMaskBuilder, ArtifactValidMaskConfig, save_valid_mask_png
from scene import Scene
from scene.gaussian_model import GaussianModel
from stdloc import STDLoc
from utils.pose_utils import cal_pose_error

try:
    from la_diagnostics.teacher_stage import classify_teacher_stage
except Exception:
    classify_teacher_stage = None


def _load_train_camera_map(scene):
    return {str(camera.image_name).replace("\\", "/").lstrip("./"): camera for camera in scene.getTrainCameras()}


def _record_camera(record, train_camera_by_name):
    if record.source == "train_rgb" and record.image_name in train_camera_by_name:
        return train_camera_by_name[record.image_name]
    return record.to_camera(device="cuda")


def _dense_final(loc_res):
    dense = loc_res.get("dense") or []
    return dense[-1] if dense else loc_res["sparse"]


def _comma_set(value):
    return {item.strip() for item in str(value or "").split(",") if item.strip()}


def _record_mask_path(output_dir, record):
    safe = str(record.query_id).replace(":", "__").replace("/", "_")
    return Path(output_dir) / f"{safe}.sparse_valid_mask.png"


def _record_no_reference_prefix(output_dir, record):
    safe = str(record.query_id).replace(":", "__").replace("/", "_")
    return Path(output_dir) / f"{safe}.no_reference"


def _resize_query_image(query_image, scale):
    scale = float(scale or 1.0)
    if abs(scale - 1.0) < 1e-6:
        return query_image
    height, width = query_image.shape[-2:]
    target_hw = (max(8, int(round(height * scale))), max(8, int(round(width * scale))))
    return torch.nn.functional.interpolate(
        query_image.detach().float()[None],
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )[0]


def _build_sparse_valid_mask_for_record(
    record,
    query_image,
    enabled=False,
    allowed_sources=None,
    mode="no_reference",
    output_dir="",
    max_artifact_score=0.45,
    erosion_radius=3,
    min_component_area=64,
    min_component_area_frac=0.0,
    min_valid_frac=0.0,
    no_reference_image_scale=1.0,
    no_reference_support_threshold=0.22,
    no_reference_support_dilate_radius=5,
    no_reference_support_min_area=24,
    no_reference_invalid_min_area=96,
    detector=None,
    builder=None,
):
    valid_mask, _support_score, summary = _build_sparse_guidance_for_record(
        record,
        query_image,
        enabled=enabled,
        allowed_sources=allowed_sources,
        mode=mode,
        output_dir=output_dir,
        max_artifact_score=max_artifact_score,
        erosion_radius=erosion_radius,
        min_component_area=min_component_area,
        min_component_area_frac=min_component_area_frac,
        min_valid_frac=min_valid_frac,
        no_reference_image_scale=no_reference_image_scale,
        no_reference_support_threshold=no_reference_support_threshold,
        no_reference_support_dilate_radius=no_reference_support_dilate_radius,
        no_reference_support_min_area=no_reference_support_min_area,
        no_reference_invalid_min_area=no_reference_invalid_min_area,
        detector=detector,
        builder=builder,
    )
    return valid_mask, summary


def _build_sparse_guidance_for_record(
    record,
    query_image,
    enabled=False,
    allowed_sources=None,
    mode="no_reference",
    output_dir="",
    max_artifact_score=0.45,
    erosion_radius=3,
    min_component_area=64,
    min_component_area_frac=0.0,
    min_valid_frac=0.0,
    no_reference_image_scale=1.0,
    no_reference_support_threshold=0.22,
    no_reference_support_dilate_radius=5,
    no_reference_support_min_area=24,
    no_reference_invalid_min_area=96,
    detector=None,
    builder=None,
):
    allowed_sources = set(allowed_sources or [])
    if not enabled:
        return None, None, {"enabled": False, "reason": "disabled"}
    if allowed_sources and record.source not in allowed_sources:
        return None, None, {"enabled": False, "reason": "source_not_enabled"}
    mode = str(mode or "no_reference")
    if mode in {"no_reference", "support_mask", "support_mask_score", "valid_mask"}:
        builder = builder or NoReferenceValidMaskBuilder(
            NoReferenceValidMaskConfig(
                support_threshold=float(no_reference_support_threshold),
                support_dilate_radius=int(no_reference_support_dilate_radius),
                support_min_area=int(no_reference_support_min_area),
                invalid_min_area=int(no_reference_invalid_min_area),
            )
        )
        mask_image = _resize_query_image(query_image.detach().cpu(), no_reference_image_scale)
        result = builder.build(mask_image)
        if mode == "support_mask":
            selected_mask = result.support_mask
            support_score = None
        elif mode == "support_mask_score":
            selected_mask = result.support_mask
            support_score = result.support_score
        elif mode == "valid_mask":
            selected_mask = result.valid_mask
            support_score = None
        else:
            selected_mask = result.valid_mask
            support_score = result.support_score
        summary = {
            "enabled": True,
            "reason": "ok",
            "mode": mode,
            **result.summary,
            "no_reference_image_scale": float(no_reference_image_scale),
            "no_reference_support_threshold": float(no_reference_support_threshold),
            "no_reference_support_dilate_radius": int(no_reference_support_dilate_radius),
            "no_reference_support_min_area": int(no_reference_support_min_area),
            "no_reference_invalid_min_area": int(no_reference_invalid_min_area),
        }
        if output_dir:
            paths = save_no_reference_valid_mask_pngs(result, _record_no_reference_prefix(output_dir, record))
            summary["mask_path"] = os.path.abspath(paths["support_mask" if mode == "support_mask" else "valid_mask"])
            summary["no_reference_paths"] = {key: os.path.abspath(value) for key, value in paths.items()}
        return selected_mask, support_score, summary
    if mode != "artifact_selector":
        raise ValueError(f"Unsupported sparse valid mask mode: {mode}")
    detector = detector or ArtifactDetector()
    builder = builder or ArtifactValidMaskBuilder(
        ArtifactValidMaskConfig(
            max_artifact_score=float(max_artifact_score),
            erosion_radius=int(erosion_radius),
            min_component_area=int(min_component_area),
            min_component_area_frac=float(min_component_area_frac),
            min_valid_frac=float(min_valid_frac),
        )
    )
    evidence = detector.detect(rendered_rgb=query_image.detach().cpu())
    result = builder.build(evidence)
    summary = {
        "enabled": True,
        "reason": "ok",
        "mode": mode,
        **result.summary,
        "artifact_score_mean": float(evidence.summary.get("artifact_score_mean", 0.0)),
        "artifact_score_p95": float(evidence.summary.get("artifact_score_p95", 0.0)),
    }
    if output_dir:
        path = _record_mask_path(output_dir, record)
        save_valid_mask_png(result, path)
        summary["mask_path"] = os.path.abspath(os.fspath(path))
    return result.mask, None, summary


def main():
    parser = argparse.ArgumentParser(description="Run STDLoc teacher on pseudo-query manifest and cache results.")
    model = ModelParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--sources", default="", help="Comma-separated pseudo-query sources to cache.")
    parser.add_argument("--summary_json", default="")
    parser.add_argument("--sparse_valid_mask", action="store_true", default=False)
    parser.add_argument(
        "--sparse_valid_mask_mode",
        choices=("no_reference", "support_mask", "support_mask_score", "valid_mask", "artifact_selector"),
        default="no_reference",
    )
    parser.add_argument("--sparse_valid_mask_sources", default="synthetic_rgb")
    parser.add_argument("--sparse_valid_mask_output_dir", default="")
    parser.add_argument("--sparse_valid_mask_max_artifact_score", type=float, default=0.45)
    parser.add_argument("--sparse_valid_mask_erosion_radius", type=int, default=3)
    parser.add_argument("--sparse_valid_mask_min_component_area", type=int, default=64)
    parser.add_argument("--sparse_valid_mask_min_component_area_frac", type=float, default=0.0)
    parser.add_argument("--sparse_valid_mask_min_valid_frac", type=float, default=0.0)
    parser.add_argument("--sparse_valid_mask_min_fraction", type=float, default=0.5)
    parser.add_argument("--sparse_valid_mask_candidate_multiplier", type=float, default=2.0)
    parser.add_argument("--sparse_support_score_weight", type=float, default=0.5)
    parser.add_argument("--sparse_support_score_min_multiplier", type=float, default=0.75)
    parser.add_argument("--no_reference_image_scale", type=float, default=0.25)
    parser.add_argument("--no_reference_support_threshold", type=float, default=0.22)
    parser.add_argument("--no_reference_support_dilate_radius", type=int, default=5)
    parser.add_argument("--no_reference_support_min_area", type=int, default=24)
    parser.add_argument("--no_reference_invalid_min_area", type=int, default=96)
    parser.add_argument("--no_sparse_valid_mask_refill", action="store_false", dest="sparse_valid_mask_refill")
    parser.set_defaults(sparse_valid_mask_refill=True)
    args = parser.parse_args()
    args.eval = False

    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    config = yaml.load(open(args.cfg), Loader=yaml.FullLoader)
    config.setdefault("dense", {})["norm_before_render"] = dataset.norm_before_render
    config["feature_type"] = dataset.feature_type
    config["longest_edge"] = dataset.longest_edge
    config["model_path"] = dataset.model_path
    config.setdefault("sparse", {})["valid_mask_min_fraction"] = float(args.sparse_valid_mask_min_fraction)
    config.setdefault("sparse", {})["valid_mask_candidate_multiplier"] = float(args.sparse_valid_mask_candidate_multiplier)
    config.setdefault("sparse", {})["valid_mask_refill"] = bool(args.sparse_valid_mask_refill)
    config.setdefault("sparse", {})["support_score_weight"] = float(args.sparse_support_score_weight)
    config.setdefault("sparse", {})["support_score_min_multiplier"] = float(args.sparse_support_score_min_multiplier)
    stdloc = STDLoc(gaussians, config)

    sources = [item.strip() for item in str(args.sources).split(",") if item.strip()]
    manifest = PseudoQueryManifest.load(args.manifest).accepted(sources=sources or None)
    records = manifest.records
    if int(args.max_queries) > 0:
        records = records[: int(args.max_queries)]
    train_camera_by_name = _load_train_camera_map(scene)
    cache = PseudoTeacherCache()
    stage_counts = {}
    use_artifact_selector = bool(args.sparse_valid_mask) and args.sparse_valid_mask_mode == "artifact_selector"
    use_no_reference_mask = bool(args.sparse_valid_mask) and args.sparse_valid_mask_mode in {
        "no_reference",
        "support_mask",
        "support_mask_score",
        "valid_mask",
    }
    mask_detector = ArtifactDetector() if use_artifact_selector else None
    mask_builder = (
        ArtifactValidMaskBuilder(
            ArtifactValidMaskConfig(
                max_artifact_score=float(args.sparse_valid_mask_max_artifact_score),
                erosion_radius=int(args.sparse_valid_mask_erosion_radius),
                min_component_area=int(args.sparse_valid_mask_min_component_area),
                min_component_area_frac=float(args.sparse_valid_mask_min_component_area_frac),
                min_valid_frac=float(args.sparse_valid_mask_min_valid_frac),
            )
        )
        if use_artifact_selector
        else None
    )
    if use_no_reference_mask:
        mask_builder = NoReferenceValidMaskBuilder(
            NoReferenceValidMaskConfig(
                support_threshold=float(args.no_reference_support_threshold),
                support_dilate_radius=int(args.no_reference_support_dilate_radius),
                support_min_area=int(args.no_reference_support_min_area),
                invalid_min_area=int(args.no_reference_invalid_min_area),
            )
        )
    mask_sources = _comma_set(args.sparse_valid_mask_sources)

    for record in tqdm(records, desc="Pseudo teacher cache"):
        camera = _record_camera(record, train_camera_by_name)
        query_image = camera.original_image.to("cuda")
        sparse_valid_mask, sparse_support_score, sparse_valid_mask_summary = _build_sparse_guidance_for_record(
            record,
            query_image,
            enabled=bool(args.sparse_valid_mask),
            allowed_sources=mask_sources,
            mode=args.sparse_valid_mask_mode,
            output_dir=args.sparse_valid_mask_output_dir,
            max_artifact_score=args.sparse_valid_mask_max_artifact_score,
            erosion_radius=args.sparse_valid_mask_erosion_radius,
            min_component_area=args.sparse_valid_mask_min_component_area,
            min_component_area_frac=args.sparse_valid_mask_min_component_area_frac,
            min_valid_frac=args.sparse_valid_mask_min_valid_frac,
            no_reference_image_scale=args.no_reference_image_scale,
            no_reference_support_threshold=args.no_reference_support_threshold,
            no_reference_support_dilate_radius=args.no_reference_support_dilate_radius,
            no_reference_support_min_area=args.no_reference_support_min_area,
            no_reference_invalid_min_area=args.no_reference_invalid_min_area,
            detector=mask_detector,
            builder=mask_builder,
        )
        loc_res = stdloc.localize(
            query_image,
            record.fovx,
            record.fovy,
            sparse_valid_mask=sparse_valid_mask,
            sparse_support_score=sparse_support_score,
        )
        gt_w2c = torch.as_tensor(record.pose_w2c, dtype=torch.float32).cpu().numpy()
        sparse = loc_res["sparse"]
        dense = _dense_final(loc_res)
        sparse_ae, sparse_te = cal_pose_error(sparse["pose_w2c"], gt_w2c)
        dense_ae, dense_te = cal_pose_error(dense["pose_w2c"], gt_w2c)
        stage = "unknown"
        if classify_teacher_stage is not None:
            stage = classify_teacher_stage(
                sparse_te=sparse_te,
                dense_te=dense_te,
                sparse_inliers=sparse.get("inliers", 0),
                sparse_correct_rate=None,
            )
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        cache.items[record.teacher_cache_key or record.query_id] = {
            "pose_w2c": torch.as_tensor(sparse["pose_w2c"]).detach().cpu().float(),
            "inliers": int(sparse.get("inliers", 0)),
            "matches": int(sparse.get("matches", 0)),
            "matches_before_selector": int(sparse.get("matches_before_selector", sparse.get("matches", 0))),
            "detected_keypoints_raw": int(sparse.get("detected_keypoints_raw", sparse.get("detected_keypoints", 0))),
            "detected_keypoints": int(sparse.get("detected_keypoints", 0)),
            "sparse_valid_mask_filtered_keypoints": int(sparse.get("sparse_valid_mask_filtered_keypoints", 0)),
            "sparse_valid_mask_valid_frac": float(sparse.get("sparse_valid_mask_valid_frac", 1.0)),
            "sparse_valid_mask_invalid_candidates": int(sparse.get("sparse_valid_mask_invalid_candidates", 0)),
            "sparse_valid_mask_selected_valid_keypoints": int(sparse.get("sparse_valid_mask_selected_valid_keypoints", 0)),
            "sparse_valid_mask_refill_keypoints": int(sparse.get("sparse_valid_mask_refill_keypoints", 0)),
            "sparse_valid_mask": sparse_valid_mask_summary,
            "sparse_support_score_prior_weight": float(sparse.get("sparse_support_score_prior_weight", 0.0)),
            "sparse_support_score_prior_min_multiplier": float(sparse.get("sparse_support_score_prior_min_multiplier", 1.0)),
            "sparse_support_score_prior_multiplier_mean": float(sparse.get("sparse_support_score_prior_multiplier_mean", 1.0)),
            "sparse_support_score_prior_score_mean": float(sparse.get("sparse_support_score_prior_score_mean", 0.0)),
            "ae": float(sparse_ae),
            "te": float(sparse_te),
            "failed": bool(sparse.get("inliers", 0) <= 0),
            "dense_pose_w2c": torch.as_tensor(dense["pose_w2c"]).detach().cpu().float(),
            "dense_inliers": int(dense.get("inliers", 0)),
            "dense_ae": float(dense_ae),
            "dense_te": float(dense_te),
            "dense_valid_mask_enabled": bool(dense.get("dense_valid_mask_enabled", False)),
            "dense_valid_mask_valid_cells": int(dense.get("dense_valid_mask_valid_cells", 0)),
            "dense_valid_mask_valid_frac": float(dense.get("dense_valid_mask_valid_frac", 1.0)),
            "failure_stage": stage,
            "source": record.source,
            "image_name": record.image_name,
            "artifact_score": float(record.artifact_score),
            "repair_action": record.repair_action,
        }
    cache.save(args.output)
    summary = {
        "manifest": os.path.abspath(args.manifest),
        "output": os.path.abspath(args.output),
        "count": len(cache.items),
        "stage_counts": stage_counts,
        "sparse_valid_mask": {
            "enabled": bool(args.sparse_valid_mask),
            "mode": args.sparse_valid_mask_mode,
            "sources": sorted(mask_sources),
            "output_dir": os.path.abspath(args.sparse_valid_mask_output_dir) if args.sparse_valid_mask_output_dir else "",
            "max_artifact_score": float(args.sparse_valid_mask_max_artifact_score),
            "erosion_radius": int(args.sparse_valid_mask_erosion_radius),
            "min_component_area": int(args.sparse_valid_mask_min_component_area),
            "min_component_area_frac": float(args.sparse_valid_mask_min_component_area_frac),
            "min_valid_frac": float(args.sparse_valid_mask_min_valid_frac),
            "min_fraction": float(args.sparse_valid_mask_min_fraction),
            "candidate_multiplier": float(args.sparse_valid_mask_candidate_multiplier),
            "refill": bool(args.sparse_valid_mask_refill),
            "support_score_weight": float(args.sparse_support_score_weight),
            "support_score_min_multiplier": float(args.sparse_support_score_min_multiplier),
            "no_reference_image_scale": float(args.no_reference_image_scale),
            "no_reference_support_threshold": float(args.no_reference_support_threshold),
            "no_reference_support_dilate_radius": int(args.no_reference_support_dilate_radius),
            "no_reference_support_min_area": int(args.no_reference_support_min_area),
            "no_reference_invalid_min_area": int(args.no_reference_invalid_min_area),
        },
    }
    if args.summary_json:
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
