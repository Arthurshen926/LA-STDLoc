"""Evaluate only frozen Compact + Fast; no training, model selection or map writes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from anygsloc.artifacts import sha256
from anygsloc.config import FROZEN
from anygsloc.data.calibration import resolve_scene_calibration
from anygsloc.data.datasets import ColmapDataset
from anygsloc.evaluation.metrics import pose_error, summarize_pose_errors
from anygsloc.online.runtime import Localizer


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def summarize(rows, wall_seconds):
    rotation = [r["rotation_error_deg"] if r["success"] else float("inf") for r in rows]
    translation = [
        r["translation_error_cm"] if r["success"] else float("inf") for r in rows
    ]
    summary = summarize_pose_errors(rotation, translation)
    t = np.asarray(translation)
    r = np.asarray(rotation)
    for cm in [5, 15, 50]:
        summary[f"recall_{cm}cm_5deg_percent"] = 100 * float(
            np.mean((t < cm) & (r < 5))
        )
    timing = np.asarray([row["runtime_ms"] for row in rows])
    summary.update(
        failed_query_count=sum(not row["success"] for row in rows),
        catastrophic_100cm_count=int((t >= 100).sum()),
        runtime_ms_mean=float(timing.mean()),
        runtime_ms_p50=float(np.percentile(timing, 50)),
        runtime_ms_p90=float(np.percentile(timing, 90)),
        runtime_ms_p95=float(np.percentile(timing, 95)),
        over_100ms_percent=100 * float(np.mean(timing > 100)),
        loop_wall_seconds=wall_seconds,
        loop_queries_per_second=len(rows) / max(wall_seconds, 1e-12),
        timing_protocol="unwarmed_localize_telemetry_not_formal_benchmark",
        nonfinite_errors_serialized_as_null=True,
    )
    return json_safe(summary)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--images", default="processed")
    p.add_argument("--map", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["test", "mapping"], default="test")
    p.add_argument("--image-domain", choices=["rectified"], required=True)
    p.add_argument(
        "--mask-policy", choices=["none", "explicit_valid_mask", "geometric_validity"], required=True
    )
    p.add_argument('--rectification-contract', choices=['opencv_same_k'], default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    dataset = ColmapDataset(
        args.dataset, images=args.images,
        load_masks=args.mask_policy == "explicit_valid_mask",
    )
    cameras = dataset.split(args.split)
    geometric_mask = None
    if args.mask_policy == 'geometric_validity':
        if args.rectification_contract != 'opencv_same_k':
            raise ValueError('geometric validity requires an explicit audited rectification contract')
        from anygsloc.data.geometric_validity import SameKRectificationSupport
        geometric_mask = SameKRectificationSupport(dataset)
    if not cameras:
        raise ValueError("evaluation split is empty")
    calibration = resolve_scene_calibration(dataset.split("mapping"))
    config = FROZEN.resolved(
        keypoint_count=calibration["keypoint_count"],
        reprojection_error_px=calibration["reprojection_error_px"],
        mask_policy=args.mask_policy,
    )
    manifest = dict(
        schema="anygsloc_evaluation_v1",
        map_sha256=sha256(args.map),
        dataset=str(dataset.root),
        images=args.images,
        image_domain=args.image_domain,
        split=args.split,
        seed=args.seed,
        queries=[c.image_name for c in cameras],
        calibration=calibration,
        config=config,
        rectification_contract=args.rectification_contract,
    )
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    torch.set_num_threads(FROZEN.cpu_threads)
    engine = Localizer(
        args.map,
        device=args.device,
        seed=args.seed,
        keypoint_count=calibration["keypoint_count"],
        reprojection_error_px=calibration["reprojection_error_px"],
        mask_policy=args.mask_policy,
    )
    rows = []
    started = time.perf_counter()
    for camera in cameras:
        mask = (
            dataset.valid_mask(camera)
            if args.mask_policy == "explicit_valid_mask"
            else None
        )
        if geometric_mask is not None:
            mask = geometric_mask(camera)
        result = engine.localize(
            dataset.load_image(camera),
            fov_x=camera.fov_x,
            fov_y=camera.fov_y,
            valid_mask=mask,
        )
        re, te = (
            pose_error(result["pose"].pose_w2c, camera.pose_w2c)
            if result["success"]
            else (None, None)
        )
        row = dict(
            image_name=camera.image_name,
            success=result["success"],
            rotation_error_deg=re,
            translation_error_cm=te,
            runtime_ms=result["runtime_ms"],
            inlier_count=len(result["pose"].inliers),
            pose_w2c=result["pose"].pose_w2c.tolist(),
        )
        rows.append(row)
        with (output / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
    (output / "results.json").write_text(json.dumps(rows, indent=2))
    (output / "summary.json").write_text(
        json.dumps(
            summarize(rows, time.perf_counter() - started), indent=2, allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
