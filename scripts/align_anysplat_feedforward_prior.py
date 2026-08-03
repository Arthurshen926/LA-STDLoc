#!/usr/bin/env python3
"""Align mapping-only AnySplat windows and export a Graphdeco SH-0 prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from lafgs.priors.anysplat import (
    colmap_qvec_to_rotation,
    fit_similarity_from_camera_poses,
    spatial_confidence_coreset,
    transform_gaussian_moments,
    write_graphdeco_dc_ply,
)
from scripts.prepare_cambridge_mapping_only_colmap import read_images_binary


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(values.max()),
    }


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--windows-manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--maximum-windows", type=int, default=0)
    parser.add_argument("--maximum-primitives", type=int, default=0)
    parser.add_argument("--preselection-multiplier", type=float, default=1.0)
    args = parser.parse_args()
    manifest = json.loads(args.windows_manifest.read_text())
    images = read_images_binary(args.dataset / "sparse" / "0" / "images.bin")
    by_name = {image.name: image for image in images.values()}
    all_means = []
    all_covariances = []
    all_f_dc = []
    all_opacities = []
    reports = []
    windows = manifest["windows"]
    if args.maximum_windows > 0:
        windows = windows[: args.maximum_windows]
    total_available_views = sum(int(w["available_view_count"]) for w in windows)
    preselection_budget = args.maximum_primitives
    if args.maximum_primitives > 0:
        preselection_budget = int(
            np.ceil(args.maximum_primitives * args.preselection_multiplier)
        )
        preselection_budget = max(args.maximum_primitives, preselection_budget)
    retained_total = 0
    for window_index, window in enumerate(windows):
        window_id = str(window["window_id"])
        payload_path = args.raw_root / f"{window_id}.pt"
        if not payload_path.is_file():
            raise FileNotFoundError(payload_path)
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        names = payload["run_record"]["image_names"]
        predicted_c2w = payload["predicted_c2w"].numpy().astype(np.float64)
        predicted_centers = predicted_c2w[:, :3, 3]
        target_c2w = []
        for name in names:
            image = by_name[name]
            rotation_w2c = colmap_qvec_to_rotation(image.qvec)
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = rotation_w2c.T
            c2w[:3, 3] = -(rotation_w2c.T @ image.tvec)
            target_c2w.append(c2w)
        target_c2w = np.stack(target_c2w)
        (
            transform,
            inliers,
            center_residual,
            orientation_inliers,
            orientation_residual,
        ) = fit_similarity_from_camera_poses(
            predicted_centers,
            target_c2w[:, :3, 3],
            predicted_c2w[:, :3, :3],
            target_c2w[:, :3, :3],
        )
        aligned_rotation = transform.rotation[None] @ predicted_c2w[:, :3, :3]
        relative_rotation = np.swapaxes(aligned_rotation, 1, 2) @ target_c2w[:, :3, :3]
        rotation_error = np.rad2deg(Rotation.from_matrix(relative_rotation).magnitude())
        means, covariances = transform_gaussian_moments(
            payload["means"].numpy(),
            payload["covariances"].numpy(),
            transform,
        )
        finite = (
            np.isfinite(means).all(axis=1)
            & np.isfinite(covariances).all(axis=(1, 2))
            & np.isfinite(payload["f_dc"].numpy()).all(axis=1)
            & np.isfinite(payload["opacity_probability"].numpy())
        )
        means = means[finite]
        covariances = covariances[finite]
        f_dc = payload["f_dc"].numpy()[finite]
        opacities = payload["opacity_probability"].numpy()[finite]
        quota = len(means)
        if preselection_budget > 0:
            if window_index == len(windows) - 1:
                quota = preselection_budget - retained_total
            else:
                quota = int(
                    round(
                        preselection_budget
                        * int(window["available_view_count"])
                        / total_available_views
                    )
                )
            quota = max(1, min(quota, len(means)))
        retained = spatial_confidence_coreset(
            means,
            covariances,
            opacities,
            quota,
        )
        retained_total += len(retained)
        all_means.append(means[retained].astype(np.float32))
        all_covariances.append(covariances[retained].astype(np.float32))
        all_f_dc.append(f_dc[retained].astype(np.float32))
        all_opacities.append(opacities[retained].astype(np.float32))
        reports.append(
            {
                "window_id": window_id,
                "view_count": len(names),
                "primitive_count": int(finite.sum()),
                "retained_primitive_count": int(len(retained)),
                "dropped_nonfinite_count": int((~finite).sum()),
                "sim3_scale": transform.scale,
                "sim3_rotation": transform.rotation.tolist(),
                "sim3_translation": transform.translation.tolist(),
                "alignment_inlier_count": int(inliers.sum()),
                "orientation_alignment_inlier_count": int(
                    orientation_inliers.sum()
                ),
                "camera_center_error_m": _quantiles(center_residual),
                "camera_rotation_error_deg": _quantiles(rotation_error),
                "orientation_fit_residual_deg": _quantiles(
                    np.rad2deg(orientation_residual)
                ),
            }
        )
    means = np.concatenate(all_means)
    covariances = np.concatenate(all_covariances)
    f_dc = np.concatenate(all_f_dc)
    opacities = np.concatenate(all_opacities)
    preselection_primitive_count = len(means)
    if args.maximum_primitives > 0 and len(means) > args.maximum_primitives:
        retained = spatial_confidence_coreset(
            means,
            covariances,
            opacities,
            args.maximum_primitives,
        )
        means = means[retained]
        covariances = covariances[retained]
        f_dc = f_dc[retained]
        opacities = opacities[retained]
    write_graphdeco_dc_ply(
        args.output_ply,
        means,
        covariances,
        f_dc,
        opacities,
    )
    report = {
        "schema": "lafgs_anysplat_mapping_only_sim3_alignment",
        "version": 1,
        "test_poses_used": False,
        "post_optimization_used": False,
        "primitive_fusion": (
            "mapping-trajectory-balanced spatial-confidence coreset; "
            "no localization labels or test data"
            if args.maximum_primitives > 0
            else "concatenation only; no localization-aware pruning"
        ),
        "maximum_primitive_budget": args.maximum_primitives,
        "preselection_multiplier": args.preselection_multiplier,
        "preselection_primitive_count": preselection_primitive_count,
        "primitive_count": len(means),
        "alignment_and_fusion_seconds": time.perf_counter() - started,
        "windows": reports,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
