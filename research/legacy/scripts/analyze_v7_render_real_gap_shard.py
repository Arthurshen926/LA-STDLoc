#!/usr/bin/env python3
"""Run one read-only shard of the V7 P0.5 render--real causal diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from common.v7_contracts import sha256_file
from data.datasets import ColmapDataset
from evaluation.evaluator import pose_error
from evidence.v7_render_certificate import CertificateThresholds
from evidence.v7_render_real_gap import (
    feather_support,
    mutual_spatial_pairs,
    oracle_visible_correspondences,
    projected_match_correctness,
    sample_pixel_mask,
    shared_support_mask,
)
from localization.localizer import SparseLocalizer
from localization.matcher import global_cosine_top2
from localization.pose_solver import poselib_camera, solve_absolute_pose


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if value.get("schema") != "lafgs_v7_render_real_gap_causal_diagnostic":
        raise ValueError("unsupported P0.5 diagnostic config")
    protocol = value["protocol"]
    if not (
        protocol.get("formal_protocol_eligible") is False
        and protocol.get("posthoc_test_rgb_diagnostic") is True
        and protocol.get("may_update_or_select_map") is False
        and protocol.get("threshold_tuning_from_results") is False
        and protocol.get("map_mutation_count") == 0
    ):
        raise ValueError("P0.5 config violates read-only diagnostic isolation")
    return value


def _condition_row(name: str, result, camera) -> dict[str, Any]:
    rotation, translation = pose_error(result.pose.pose_w2c, camera.pose_w2c)
    return {
        "condition": name,
        "translation_error_cm": float(translation),
        "rotation_error_deg": float(rotation),
        "estimated_pose_w2c": result.pose.pose_w2c.tolist(),
        "gt_pose_w2c": np.asarray(camera.pose_w2c).tolist(),
        "keypoint_count": int(result.sparse_features.keypoints.shape[0]),
        "match_count": int(result.matches.scores.numel()),
        "inlier_count": int(result.pose.inliers.size),
    }


def _solve_rows(localizer, sparse, matches, keep: torch.Tensor, intrinsic: np.ndarray):
    keep = torch.as_tensor(keep, device=matches.anchor_indices.device).bool()
    query_rows = matches.keypoint_indices[keep]
    anchor_rows = matches.anchor_indices[keep]
    return solve_absolute_pose(
        sparse.keypoints[query_rows].cpu().numpy() + 0.5,
        localizer.anchor_xyz[anchor_rows].cpu().numpy(),
        intrinsic,
        reprojection_error_px=localizer.reprojection_error_px,
        confidence=localizer.confidence,
        max_iterations=localizer.max_iterations,
        min_iterations=localizer.min_iterations,
        seed=localizer.seed,
        camera=poselib_camera(intrinsic),
    )


def _pose_row(name: str, estimate, camera, *, rows: int) -> dict[str, Any]:
    rotation, translation = pose_error(estimate.pose_w2c, camera.pose_w2c)
    return {
        "condition": name,
        "translation_error_cm": float(translation),
        "rotation_error_deg": float(rotation),
        "estimated_pose_w2c": estimate.pose_w2c.tolist(),
        "gt_pose_w2c": np.asarray(camera.pose_w2c).tolist(),
        "keypoint_count": int(rows),
        "match_count": int(rows),
        "inlier_count": int(estimate.inliers.size),
    }


def _ratio(correct: torch.Tensor, mask: torch.Tensor) -> tuple[int, int]:
    selected = torch.as_tensor(mask, device=correct.device).bool()
    return int((correct & selected).sum()), int(selected.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--metric", type=Path, required=True)
    parser.add_argument("--render-manifest", type=Path, action="append", required=True)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("invalid shard index/count")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    config = _load_config(args.config.resolve())
    protocol = config["protocol"]
    if sha256_file(args.map) != protocol["full_map_sha256"]:
        raise ValueError("P0.5 Full map SHA differs from preregistration")
    if sha256_file(args.metric) != protocol["identity_metric_sha256"]:
        raise ValueError("P0.5 identity metric SHA differs from preregistration")

    render_records: dict[int, dict[str, Any]] = {}
    render_inputs = []
    for manifest_path in args.render_manifest:
        manifest_path = manifest_path.resolve()
        manifest = json.loads(manifest_path.read_text())
        if not (
            manifest.get("view_role") == "test_pose_render_diagnostic"
            and manifest.get("uses_test_pose_metadata") is True
            and manifest.get("uses_test_rgb") is False
            and manifest.get("formal_protocol_eligible") is False
            and manifest.get("detector_input") == "complete_unmasked_rgb"
            and manifest.get("map_mutation_count") == 0
        ):
            raise ValueError("P0.5 requires isolated test-pose render manifests")
        render_inputs.append({"path": str(manifest_path), "sha256": sha256_file(manifest_path)})
        for item in manifest["records"]:
            query_index = int(item["query_index"])
            if query_index in render_records:
                raise ValueError("duplicate render query index")
            render_records[query_index] = item

    dataset = ColmapDataset(args.dataset, images=args.images)
    cameras = dataset.split("test")
    reference = json.loads(args.reference_results.read_text())
    if len(reference) != len(cameras) or set(render_records) != set(range(len(cameras))):
        raise ValueError("P0.5 camera/render/reference registries differ")
    reference_by_name = {item["image_name"]: item for item in reference}
    frontend = config["frontend"]
    localizer = SparseLocalizer(
        args.map,
        args.metric,
        device=args.device,
        keypoint_count=int(frontend["keypoints"]),
        nms_radius=int(frontend["nms_radius"]),
        reprojection_error_px=float(frontend["reprojection_error_px"]),
        confidence=float(frontend["confidence"]),
        max_iterations=int(frontend["maximum_iterations"]),
        min_iterations=int(frontend["minimum_iterations"]),
        seed=int(frontend["seed"]),
        profile_mode=True,
    )
    anchor_xyz_cpu = localizer.anchor_xyz.detach().cpu()
    support_config = config["shared_support"]
    thresholds = CertificateThresholds(
        alpha_minimum=float(support_config["alpha_minimum"]),
        depth_discontinuity_relative=float(support_config["depth_discontinuity_relative"]),
        border_fraction=float(support_config["border_fraction"]),
        rgb_structure_support_threshold=float(support_config["rgb_structure_support_threshold"]),
        rgb_structure_window_reference_px=int(support_config["rgb_structure_window_reference_px"]),
        rgb_structure_dilate_reference_px=int(support_config["rgb_structure_dilate_reference_px"]),
        rgb_structure_reference_short_side=int(support_config["rgb_structure_reference_short_side"]),
    )
    conditions = config["conditions"]
    hybrid_stride = int(conditions["hybrid_query_stride"])
    rows = []
    replay_mismatches = []
    for query_index in range(args.shard_index, len(cameras), args.shard_count):
        camera = cameras[query_index]
        reference_row = reference_by_name[camera.image_name]
        real = dataset.load_image(camera)
        dataset_mask = dataset.valid_mask(camera)
        if dataset_mask is None:
            raise ValueError("P0.5 existing-mask control requires a dataset mask")
        item = render_records[query_index]
        record_path = Path(item["path"])
        if sha256_file(record_path) != item["sha256"]:
            raise ValueError("P0.5 render record SHA mismatch")
        render_record = torch.load(record_path, map_location="cpu", weights_only=False)
        render = torch.as_tensor(render_record["rgb_float16"]).float()
        if tuple(real.shape) != tuple(render.shape):
            raise ValueError("real/render image shapes differ")
        support = shared_support_mask(
            render.to(localizer.device),
            torch.as_tensor(render_record["alpha_float16"], device=localizer.device),
            torch.as_tensor(render_record["depth_float16"], device=localizer.device),
            thresholds=thresholds,
        )

        real_masked = localizer.localize(
            real, fov_x=camera.fov_x, fov_y=camera.fov_y, valid_mask=dataset_mask
        )
        exact_pose = np.array_equal(
            real_masked.pose.pose_w2c,
            np.asarray(reference_row["pose_w2c"], dtype=np.float32),
        )
        exact_counts = (
            int(real_masked.sparse_features.keypoints.shape[0]) == int(reference_row["keypoints"])
            and int(real_masked.pose.inliers.size) == int(reference_row["inliers"])
        )
        if not (exact_pose and exact_counts):
            replay_mismatches.append(query_index)
        condition_rows = [
            _condition_row("real_dataset_masked", real_masked, camera)
        ]
        real_unmasked = localizer.localize(
            real, fov_x=camera.fov_x, fov_y=camera.fov_y, valid_mask=None
        )
        render_unmasked = localizer.localize(
            render, fov_x=camera.fov_x, fov_y=camera.fov_y, valid_mask=None
        )
        render_masked = localizer.localize(
            render, fov_x=camera.fov_x, fov_y=camera.fov_y, valid_mask=dataset_mask
        )
        condition_rows.extend(
            [
                _condition_row("real_unmasked", real_unmasked, camera),
                _condition_row("render_unmasked", render_unmasked, camera),
                _condition_row("render_dataset_masked", render_masked, camera),
            ]
        )

        real_support_rows = sample_pixel_mask(
            support, real_masked.sparse_features.keypoints.to(localizer.device)
        )
        supported_pose = _solve_rows(
            localizer,
            real_masked.sparse_features,
            real_masked.matches,
            real_support_rows,
            real_masked.intrinsic,
        )
        condition_rows.append(
            _pose_row(
                "real_dataset_masked_support_rows",
                supported_pose,
                camera,
                rows=int(real_support_rows.sum()),
            )
        )

        oracle_xy, oracle_xyz = oracle_visible_correspondences(
            real_masked.sparse_features.keypoints,
            anchor_xyz_cpu,
            camera.pose_w2c,
            real_masked.intrinsic,
            render_record["depth_float16"],
            real_support_rows.cpu(),
            maximum_reprojection_px=float(support_config["oracle_maximum_reprojection_px"]),
            search_neighbors=int(support_config["oracle_search_neighbors"]),
            absolute_depth_tolerance_m=float(support_config["oracle_depth_absolute_tolerance_m"]),
            relative_depth_tolerance=float(support_config["oracle_depth_relative_tolerance"]),
        )
        oracle_pose = solve_absolute_pose(
            oracle_xy,
            oracle_xyz,
            real_masked.intrinsic,
            reprojection_error_px=localizer.reprojection_error_px,
            confidence=localizer.confidence,
            max_iterations=localizer.max_iterations,
            min_iterations=localizer.min_iterations,
            seed=localizer.seed,
            camera=poselib_camera(real_masked.intrinsic),
        )
        condition_rows.append(
            _pose_row("oracle_geometry", oracle_pose, camera, rows=len(oracle_xy))
        )

        if query_index % hybrid_stride == 0:
            soft = feather_support(
                support, reference_px=int(support_config["feather_reference_px"])
            )[None]
            real_device = real.to(localizer.device)
            render_device = render.to(localizer.device)
            real_shared = soft * real_device + (1.0 - soft) * render_device
            render_shared = soft * render_device + (1.0 - soft) * real_device
            condition_rows.extend(
                [
                    _condition_row(
                        "real_shared_render_else_dataset_masked",
                        localizer.localize(
                            real_shared,
                            fov_x=camera.fov_x,
                            fov_y=camera.fov_y,
                            valid_mask=dataset_mask,
                        ),
                        camera,
                    ),
                    _condition_row(
                        "render_shared_real_else_dataset_masked",
                        localizer.localize(
                            render_shared,
                            fov_x=camera.fov_x,
                            fov_y=camera.fov_y,
                            valid_mask=dataset_mask,
                        ),
                        camera,
                    ),
                ]
            )

        real_correct = projected_match_correctness(
            real_masked.sparse_features.keypoints[real_masked.matches.keypoint_indices],
            localizer.anchor_xyz[real_masked.matches.anchor_indices],
            camera.pose_w2c,
            real_masked.intrinsic,
            maximum_reprojection_px=float(support_config["gt_correct_reprojection_px"]),
        )
        render_correct = projected_match_correctness(
            render_unmasked.sparse_features.keypoints[render_unmasked.matches.keypoint_indices],
            localizer.anchor_xyz[render_unmasked.matches.anchor_indices],
            camera.pose_w2c,
            render_unmasked.intrinsic,
            maximum_reprojection_px=float(support_config["gt_correct_reprojection_px"]),
        )
        render_support_rows = sample_pixel_mask(
            support, render_unmasked.sparse_features.keypoints.to(localizer.device)
        )
        real_inside_correct, real_inside_count = _ratio(real_correct, real_support_rows)
        real_outside_correct, real_outside_count = _ratio(real_correct, ~real_support_rows)
        render_inside_correct, render_inside_count = _ratio(render_correct, render_support_rows)
        render_outside_correct, render_outside_count = _ratio(render_correct, ~render_support_rows)

        real_top2 = global_cosine_top2(
            real_masked.sparse_features.descriptors,
            localizer.anchor_features,
            anchor_descriptors_normalized=True,
        )
        render_top2 = global_cosine_top2(
            render_unmasked.sparse_features.descriptors,
            localizer.anchor_features,
            anchor_descriptors_normalized=True,
        )
        left, right, pair_distance = mutual_spatial_pairs(
            real_masked.sparse_features.keypoints.to(localizer.device),
            render_unmasked.sparse_features.keypoints.to(localizer.device),
            maximum_distance_px=float(support_config["descriptor_pair_maximum_distance_px"]),
        )
        pair_support = real_support_rows[left] & render_support_rows[right]
        left = left[pair_support]
        right = right[pair_support]
        descriptor_cosine = (
            real_masked.sparse_features.descriptors[left]
            * render_unmasked.sparse_features.descriptors[right]
        ).sum(1)
        same_anchor = (
            real_masked.matches.anchor_indices[left]
            == render_unmasked.matches.anchor_indices[right]
        )
        descriptor = {
            "mutual_pair_count": int(left.numel()),
            "real_keypoint_count": int(real_masked.sparse_features.keypoints.shape[0]),
            "render_keypoint_count": int(render_unmasked.sparse_features.keypoints.shape[0]),
            "descriptor_cosine_sum": float(descriptor_cosine.sum()),
            "same_top1_anchor_count": int(same_anchor.sum()),
            "real_top1_margin_sum": float(
                (real_top2.scores[left, 0] - real_top2.scores[left, 1]).sum()
            ),
            "render_top1_margin_sum": float(
                (render_top2.scores[right, 0] - render_top2.scores[right, 1]).sum()
            ),
            "real_pair_correct_count": int(real_correct[left].sum()),
            "render_pair_correct_count": int(render_correct[right].sum()),
            "pair_distance_sum_px": float(pair_distance[pair_support].sum()),
        }
        rows.append(
            {
                "query_index": query_index,
                "image_name": camera.image_name,
                "reference_replay_exact": bool(exact_pose and exact_counts),
                "dataset_mask_fraction": float(dataset_mask.float().mean()),
                "shared_support_pixel_fraction": float(support.float().mean()),
                "conditions": condition_rows,
                "correspondence_partition": {
                    "real_inside": {"correct": real_inside_correct, "count": real_inside_count},
                    "real_outside": {"correct": real_outside_correct, "count": real_outside_count},
                    "render_inside": {"correct": render_inside_correct, "count": render_inside_count},
                    "render_outside": {"correct": render_outside_correct, "count": render_outside_count},
                },
                "descriptor_pairing": descriptor,
                "oracle_correspondence_count": int(len(oracle_xy)),
            }
        )

    args.output_dir.mkdir(parents=True)
    result_path = args.output_dir / "results.json"
    payload = {
        "schema": "lafgs_v7_render_real_gap_causal_diagnostic_shard",
        "version": 1,
        "formal_protocol_eligible": False,
        "posthoc_test_rgb_diagnostic": True,
        "may_update_or_select_map": False,
        "map_mutation_count": 0,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "query_count": len(rows),
        "reference_replay_mismatch_count": len(replay_mismatches),
        "reference_replay_mismatch_query_indices": replay_mismatches,
        "inputs": {
            "config": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config),
            "dataset": str(args.dataset.resolve()),
            "map": str(args.map.resolve()),
            "map_sha256": sha256_file(args.map),
            "metric": str(args.metric.resolve()),
            "metric_sha256": sha256_file(args.metric),
            "reference_results": str(args.reference_results.resolve()),
            "reference_results_sha256": sha256_file(args.reference_results),
            "render_manifests": render_inputs,
        },
        "rows": rows,
    }
    temporary = result_path.with_name(f".{result_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, result_path)
    manifest = {
        "schema": "lafgs_v7_render_real_gap_causal_diagnostic_shard_manifest",
        "version": 1,
        "result": str(result_path.resolve()),
        "result_sha256": sha256_file(result_path),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "query_count": len(rows),
        "reference_replay_mismatch_count": len(replay_mismatches),
        "map_mutation_count": 0,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
