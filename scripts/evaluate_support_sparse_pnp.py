#!/usr/bin/env python3
import argparse
import json
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from la_artifacts.no_reference_valid_mask import NoReferenceValidMaskBuilder, NoReferenceValidMaskConfig
from la_artifacts.pseudo_query import PseudoQueryManifest
from scene import Scene
from scene.gaussian_model import GaussianModel, GaussianModel_2dgs
from stdloc import STDLoc, resize_sparse_valid_mask_to_feature_grid
from utils.pose_utils import cal_pose_error


def _comma_list(value):
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _load_rgb_tensor(path, device="cuda"):
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).to(device=device, dtype=torch.float32)


def _resize_image(image, scale):
    scale = float(scale or 1.0)
    if abs(scale - 1.0) < 1e-6:
        return image
    width = max(8, int(round(image.width * scale)))
    height = max(8, int(round(image.height * scale)))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _image_to_tensor_cpu(image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def select_support_keypoints(kp_ids, support_mask, height, width, target_count=0, refill=True, min_fraction=0.5):
    kp_ids = torch.as_tensor(kp_ids, dtype=torch.long)
    raw_count = int(kp_ids.numel())
    target_count = int(target_count or 0)
    if support_mask is None:
        selected = kp_ids[:target_count] if target_count > 0 else kp_ids
        return selected, {
            "support_detected_keypoints_raw": raw_count,
            "support_detected_keypoints": int(selected.numel()),
            "support_mask_frac": 1.0,
            "support_invalid_candidates": 0,
            "support_selected_keypoints": int(selected.numel()),
            "support_refill_keypoints": 0,
        }

    mask = resize_sparse_valid_mask_to_feature_grid(
        support_mask,
        int(height),
        int(width),
        min_fraction=min_fraction,
    ).reshape(-1)
    if mask.device != kp_ids.device:
        mask = mask.to(kp_ids.device)
    keep = mask[kp_ids] if raw_count else torch.zeros(0, dtype=torch.bool, device=kp_ids.device)
    supported = kp_ids[keep]
    unsupported = kp_ids[~keep]
    if target_count <= 0:
        selected = supported
        refill_count = 0
    else:
        selected_supported = supported[:target_count]
        if bool(refill) and selected_supported.numel() < target_count:
            refill_ids = unsupported[: target_count - selected_supported.numel()]
        else:
            refill_ids = unsupported[:0]
        selected = torch.cat([selected_supported, refill_ids], dim=0)
        refill_count = int(refill_ids.numel())
    selected_supported_count = int(mask[selected].sum().item()) if selected.numel() else 0
    return selected, {
        "support_detected_keypoints_raw": raw_count,
        "support_detected_keypoints": int(selected.numel()),
        "support_mask_frac": float(mask.float().mean().item()) if mask.numel() else 0.0,
        "support_invalid_candidates": int((~keep).sum().item()) if keep.numel() else 0,
        "support_selected_keypoints": selected_supported_count,
        "support_refill_keypoints": refill_count,
    }


def _mean(values):
    values = [float(value) for value in values]
    return float(sum(values) / len(values)) if values else 0.0


def _median(values):
    values = [float(value) for value in values]
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else 0.0


def _recall(rows, branch, max_te, max_ae):
    if not rows:
        return 0.0
    ok = 0
    for row in rows:
        item = row.get(branch, {})
        if float(item.get("sparse_te", np.inf)) <= float(max_te) and float(item.get("sparse_ae", np.inf)) <= float(max_ae):
            ok += 1
    return float(ok / len(rows))


def summarize_support_ablation(rows):
    rows = list(rows)
    baseline = [row.get("baseline", {}) for row in rows]
    support = [row.get("support", {}) for row in rows]
    support_wins_te = 0
    support_wins_inliers = 0
    support_rescues = 0
    support_regressions = 0
    for row in rows:
        base = row.get("baseline", {})
        supp = row.get("support", {})
        base_te = float(base.get("sparse_te", np.inf))
        supp_te = float(supp.get("sparse_te", np.inf))
        base_inliers = int(base.get("inliers", 0))
        supp_inliers = int(supp.get("inliers", 0))
        base_failed = bool(base.get("failed", False))
        supp_failed = bool(supp.get("failed", False))
        if supp_te < base_te:
            support_wins_te += 1
        if supp_inliers > base_inliers:
            support_wins_inliers += 1
        if base_failed and not supp_failed:
            support_rescues += 1
        if (not base_failed and supp_failed) or (supp_te > base_te and supp_inliers < base_inliers):
            support_regressions += 1

    baseline_inliers = [item.get("inliers", 0) for item in baseline]
    support_inliers = [item.get("inliers", 0) for item in support]
    baseline_matches = [item.get("matches", 0) for item in baseline]
    support_matches = [item.get("matches", 0) for item in support]
    baseline_te = [item.get("sparse_te", np.inf) for item in baseline]
    support_te = [item.get("sparse_te", np.inf) for item in support]
    baseline_ae = [item.get("sparse_ae", np.inf) for item in baseline]
    support_ae = [item.get("sparse_ae", np.inf) for item in support]
    return {
        "count": int(len(rows)),
        "baseline_avg_inliers": _mean(baseline_inliers),
        "support_avg_inliers": _mean(support_inliers),
        "delta_avg_inliers": _mean(support_inliers) - _mean(baseline_inliers),
        "baseline_avg_matches": _mean(baseline_matches),
        "support_avg_matches": _mean(support_matches),
        "delta_avg_matches": _mean(support_matches) - _mean(baseline_matches),
        "baseline_median_te": _median(baseline_te),
        "support_median_te": _median(support_te),
        "delta_median_te": _median(support_te) - _median(baseline_te),
        "baseline_median_ae": _median(baseline_ae),
        "support_median_ae": _median(support_ae),
        "delta_median_ae": _median(support_ae) - _median(baseline_ae),
        "baseline_recall_5m_10d": _recall(rows, "baseline", 500.0, 10.0),
        "support_recall_5m_10d": _recall(rows, "support", 500.0, 10.0),
        "baseline_recall_2m_5d": _recall(rows, "baseline", 200.0, 5.0),
        "support_recall_2m_5d": _recall(rows, "support", 200.0, 5.0),
        "support_wins_te": int(support_wins_te),
        "support_wins_inliers": int(support_wins_inliers),
        "support_rescues": int(support_rescues),
        "support_regressions": int(support_regressions),
    }


def _load_model_args(model_path, source_path, images, eval_mode=True):
    parser = ArgumentParser(add_help=False)
    ModelParams(parser, sentinel=True)
    PipelineParams(parser)
    argv = ["--model_path", os.fspath(model_path)]
    if source_path:
        argv += ["--source_path", os.fspath(source_path)]
    if images:
        argv += ["--images", str(images)]
    cmd = parser.parse_args(argv)
    merged = {}
    cfg_path = Path(model_path) / "cfg_args"
    if cfg_path.exists():
        merged.update(vars(eval(cfg_path.read_text())))
    for key, value in vars(cmd).items():
        if value is not None:
            merged[key] = value
    merged["model_path"] = os.path.abspath(os.fspath(model_path))
    if source_path:
        merged["source_path"] = os.path.abspath(os.fspath(source_path))
    merged["eval"] = bool(eval_mode)
    return Namespace(**merged)


def load_stdloc(model_path, source_path, cfg_path, iteration=30000, images="processed"):
    dataset = _load_model_args(model_path, source_path, images=images, eval_mode=True)
    if dataset.gaussian_type == "3dgs":
        gaussians = GaussianModel(dataset.sh_degree)
    elif dataset.gaussian_type == "2dgs":
        gaussians = GaussianModel_2dgs(dataset.sh_degree)
    else:
        raise ValueError(f"Unsupported gaussian_type: {dataset.gaussian_type}")
    Scene(dataset, gaussians, load_iteration=int(iteration), shuffle=False, preload_cameras=False)
    config = yaml.load(open(cfg_path), Loader=yaml.FullLoader)
    config.setdefault("sparse", {})["sparse_only"] = True
    config.setdefault("dense", {})["norm_before_render"] = dataset.norm_before_render
    config["feature_type"] = dataset.feature_type
    config["longest_edge"] = dataset.longest_edge
    config["model_path"] = dataset.model_path
    landmark_meta_path = Path(dataset.model_path) / config["sparse"].get("landmark_meta_path", "detector/landmark_meta.pt")
    if not landmark_meta_path.exists():
        config["sparse"]["use_landmark_prior"] = False
    return STDLoc(gaussians, config)


def _sparse_metrics(result, gt_w2c, min_inliers=4):
    ae, te = cal_pose_error(result["pose_w2c"], gt_w2c)
    inliers = int(result.get("inliers", 0))
    return {
        "pose_w2c": np.asarray(result["pose_w2c"], dtype=np.float32).tolist(),
        "inliers": inliers,
        "matches": int(result.get("matches", 0)),
        "matches_before_selector": int(result.get("matches_before_selector", result.get("matches", 0))),
        "sparse_ae": float(ae),
        "sparse_te": float(te),
        "failed": bool(inliers < int(min_inliers)),
        "diagnostics": {key: value for key, value in result.items() if key not in {"pose_w2c"}},
    }


def _build_support_mask(record, builder, image_scale=0.5):
    image = Image.open(record.image_path).convert("RGB")
    image = _resize_image(image, image_scale)
    result = builder.build(_image_to_tensor_cpu(image))
    return result.support_mask, result.support_score, result.summary


def evaluate_support_sparse_pnp(
    model_path,
    source_path,
    cfg_path,
    manifest_path,
    output_dir,
    iteration=30000,
    images="processed",
    sources=("synthetic_rgb",),
    include_rejected=False,
    max_records=0,
    support_threshold=0.22,
    support_dilate_radius=5,
    support_image_scale=0.5,
    support_mode="mask",
    support_score_weight=0.5,
    support_score_min_multiplier=0.75,
    min_inliers=4,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = PseudoQueryManifest.load(manifest_path)
    allowed_sources = {str(item) for item in sources}
    records = [row for row in manifest.records if row.source in allowed_sources and (include_rejected or row.accepted)]
    if int(max_records or 0) > 0:
        records = records[: int(max_records)]
    stdloc = load_stdloc(model_path, source_path, cfg_path, iteration=iteration, images=images)
    stdloc.config.setdefault("sparse", {})["support_score_weight"] = float(support_score_weight)
    stdloc.config.setdefault("sparse", {})["support_score_min_multiplier"] = float(support_score_min_multiplier)
    builder = NoReferenceValidMaskBuilder(
        NoReferenceValidMaskConfig(
            support_threshold=float(support_threshold),
            support_dilate_radius=int(support_dilate_radius),
        )
    )
    rows = []
    for record in tqdm(records, desc="support sparse pnp"):
        query_image = _load_rgb_tensor(record.image_path, device="cuda")
        fine_feature_map, _ = stdloc.get_feature_map(query_image)
        gt_w2c = np.asarray(record.pose_w2c, dtype=np.float32)
        baseline_result = stdloc.loc_sparse(fine_feature_map, record.fovx, record.fovy, valid_mask=None)
        support_mask, support_score, support_summary = _build_support_mask(record, builder, image_scale=support_image_scale)
        mode = str(support_mode)
        support_valid_mask = support_mask if mode in {"mask", "score_mask"} else None
        support_score_map = support_score if mode in {"score", "score_mask"} else None
        support_result = stdloc.loc_sparse(
            fine_feature_map,
            record.fovx,
            record.fovy,
            valid_mask=support_valid_mask,
            support_score=support_score_map,
        )
        row = {
            "query_id": record.query_id,
            "scene": record.scene,
            "source": record.source,
            "image_path": record.image_path,
            "support_summary": support_summary,
            "baseline": _sparse_metrics(baseline_result, gt_w2c, min_inliers=min_inliers),
            "support": _sparse_metrics(support_result, gt_w2c, min_inliers=min_inliers),
        }
        rows.append(row)
    records_path = output_dir / "support_sparse_pnp_records.jsonl"
    with records_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "model_path": os.path.abspath(os.fspath(model_path)),
        "source_path": os.path.abspath(os.fspath(source_path)),
        "cfg_path": os.path.abspath(os.fspath(cfg_path)),
        "manifest": os.path.abspath(os.fspath(manifest_path)),
        "output_dir": os.path.abspath(os.fspath(output_dir)),
        "records_jsonl": str(records_path),
        "sources": sorted(allowed_sources),
        "include_rejected": bool(include_rejected),
        "max_records": int(max_records or 0),
        "support_config": {
            "support_threshold": float(support_threshold),
            "support_dilate_radius": int(support_dilate_radius),
            "support_image_scale": float(support_image_scale),
            "support_mode": str(support_mode),
            "support_score_weight": float(support_score_weight),
            "support_score_min_multiplier": float(support_score_min_multiplier),
        },
        **summarize_support_ablation(rows),
    }
    summary_path = output_dir / "support_sparse_pnp_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate no-reference support masks in STDLoc sparse PnP.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--cfg", default="/root/STDLoc/configs/stdloc_cambridge.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--sources", default="synthetic_rgb")
    parser.add_argument("--include_rejected", action="store_true", default=False)
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--support_threshold", type=float, default=0.22)
    parser.add_argument("--support_dilate_radius", type=int, default=5)
    parser.add_argument("--support_image_scale", type=float, default=0.5)
    parser.add_argument("--support_mode", choices=["mask", "score", "score_mask"], default="mask")
    parser.add_argument("--support_score_weight", type=float, default=0.5)
    parser.add_argument("--support_score_min_multiplier", type=float, default=0.75)
    parser.add_argument("--min_inliers", type=int, default=4)
    args = parser.parse_args()
    evaluate_support_sparse_pnp(
        model_path=args.model_path,
        source_path=args.source_path,
        cfg_path=args.cfg,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        iteration=args.iteration,
        images=args.images,
        sources=_comma_list(args.sources),
        include_rejected=args.include_rejected,
        max_records=args.max_records,
        support_threshold=args.support_threshold,
        support_dilate_radius=args.support_dilate_radius,
        support_image_scale=args.support_image_scale,
        support_mode=args.support_mode,
        support_score_weight=args.support_score_weight,
        support_score_min_multiplier=args.support_score_min_multiplier,
        min_inliers=args.min_inliers,
    )


if __name__ == "__main__":
    main()
