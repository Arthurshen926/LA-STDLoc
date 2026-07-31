#!/usr/bin/env python
"""Audit whether group-saturated scoring repairs A2 false pose consensus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import poselib
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localization_training.gaussian_prior import GaussianPriorGeometry
from localization_training.group_saturated_consensus import (
    build_surface_component_groups,
    image_cell_ids,
    image_cell_normalization,
    reprojection_errors,
    score_pose,
    support_concentration,
)
from utils.pose_utils import cal_pose_error


def _load_source_normals(gaussian_ply, source_ids):
    source_ids = np.asarray(source_ids, dtype=np.int64).reshape(-1)
    vertex = PlyData.read(str(gaussian_ply))["vertex"].data
    if source_ids.size and (
        source_ids.min() < 0 or source_ids.max() >= len(vertex)
    ):
        raise ValueError("source Gaussian IDs are outside the prior PLY")
    rotation = np.column_stack(
        [vertex[f"rot_{axis}"][source_ids] for axis in range(4)]
    )
    scaling = np.column_stack(
        [np.exp(vertex[f"scale_{axis}"][source_ids]) for axis in range(2)]
    )
    xyz = np.column_stack(
        [vertex[axis][source_ids] for axis in ("x", "y", "z")]
    )
    prior = GaussianPriorGeometry(
        gaussian_type="2dgs",
        xyz=np_to_torch(xyz),
        rotation=np_to_torch(rotation),
        scaling=np_to_torch(scaling),
    )
    return prior.proxy_normals.cpu().numpy()


def np_to_torch(value):
    import torch

    return torch.from_numpy(np.asarray(value, dtype=np.float32))


def _camera_dict(K, width, height):
    return {
        "model": "PINHOLE",
        "width": int(width),
        "height": int(height),
        "params": [
            float(K[0, 0]),
            float(K[1, 1]),
            float(K[0, 2]),
            float(K[1, 2]),
        ],
    }


def refine_correct_basin(points2d, points3d, K, gt_pose, threshold, width, height):
    """Refine only GT-consistent matches while staying in the GT basin."""
    errors = reprojection_errors(points2d, points3d, K, gt_pose)
    clean = np.isfinite(errors) & (errors <= float(threshold))
    diagnostics = {
        "gt_inlier_count": int(clean.sum()),
        "used_refinement": False,
        "fallback_reason": "",
    }
    if int(clean.sum()) < 6:
        diagnostics["fallback_reason"] = "fewer_than_six_gt_inliers"
        return np.asarray(gt_pose, dtype=np.float64), diagnostics
    initial = poselib.CameraPose()
    initial.R = np.asarray(gt_pose, dtype=np.float64)[:3, :3]
    initial.t = np.asarray(gt_pose, dtype=np.float64)[:3, 3]
    try:
        refined, _ = poselib.refine_absolute_pose(
            np.asarray(points2d, dtype=np.float64)[clean],
            np.asarray(points3d, dtype=np.float64)[clean],
            initial,
            _camera_dict(K, width, height),
            {"verbose": False},
        )
    except RuntimeError as error:
        diagnostics["fallback_reason"] = f"poselib:{error}"
        return np.asarray(gt_pose, dtype=np.float64), diagnostics
    pose = np.eye(4, dtype=np.float64)
    pose[:3] = np.asarray(refined.Rt, dtype=np.float64)
    ae_deg, te_cm = cal_pose_error(pose, gt_pose)
    diagnostics.update(
        {
            "refined_te_cm": float(te_cm),
            "refined_ae_deg": float(ae_deg),
        }
    )
    if not np.isfinite(pose).all() or te_cm > 100.0 or ae_deg > 5.0:
        diagnostics["fallback_reason"] = "refinement_left_gt_basin"
        return np.asarray(gt_pose, dtype=np.float64), diagnostics
    diagnostics["used_refinement"] = True
    return pose, diagnostics


def _variant_score(
    points2d,
    points3d,
    K,
    pose,
    threshold,
    *,
    groups=None,
    cap=None,
    weights=None,
):
    score, errors, support = score_pose(
        points2d,
        points3d,
        K,
        pose,
        threshold,
        group_ids=groups,
        cap=cap,
        weights=weights,
    )
    return {
        "score": float(score),
        "inlier_count": int(np.sum(errors <= threshold)),
        "soft_support": float(np.sum(support)),
        "errors": errors,
        "support": support,
    }


def _compare_hypotheses(
    points2d,
    points3d,
    K,
    correct_pose,
    predicted_pose,
    threshold,
    group_sets,
    cell_weights,
    caps,
):
    output = {}
    correct = _variant_score(
        points2d, points3d, K, correct_pose, threshold
    )
    predicted = _variant_score(
        points2d, points3d, K, predicted_pose, threshold
    )
    standard_margin = correct["score"] - predicted["score"]
    output["standard_msac"] = {
        "correct_score": correct["score"],
        "predicted_score": predicted["score"],
        "margin": float(standard_margin),
        "correct_wins": bool(standard_margin > 1e-9),
        "correct_inlier_count": correct["inlier_count"],
        "predicted_inlier_count": predicted["inlier_count"],
    }
    for group_name, groups in group_sets.items():
        for cap in caps:
            for suffix, weights in (("", None), ("_image", cell_weights)):
                correct_grouped = _variant_score(
                    points2d,
                    points3d,
                    K,
                    correct_pose,
                    threshold,
                    groups=groups,
                    cap=cap,
                    weights=weights,
                )
                predicted_grouped = _variant_score(
                    points2d,
                    points3d,
                    K,
                    predicted_pose,
                    threshold,
                    groups=groups,
                    cap=cap,
                    weights=weights,
                )
                margin = (
                    correct_grouped["score"] - predicted_grouped["score"]
                )
                output[f"{group_name}{suffix}_cap{cap:g}"] = {
                    "correct_score": correct_grouped["score"],
                    "predicted_score": predicted_grouped["score"],
                    "margin": float(margin),
                    "normalized_margin": float(
                        margin
                        / max(
                            abs(correct_grouped["score"])
                            + abs(predicted_grouped["score"]),
                            1e-12,
                        )
                    ),
                    "correct_wins": bool(margin > 1e-9),
                }
    for group_name, groups in group_sets.items():
        output["standard_msac"][
            f"predicted_{group_name}_concentration"
        ] = support_concentration(predicted["support"], groups)
        output["standard_msac"][
            f"correct_{group_name}_concentration"
        ] = support_concentration(correct["support"], groups)
    return output


def _topk_coverage(query, bank_xyz, bank_source, K, gt_pose):
    keypoints = np.asarray(query["keypoint_xy"], dtype=np.float64) + 0.5
    topk = np.asarray(query["topk_landmark_idx"], dtype=np.int64)
    candidate_xyz = bank_xyz[topk]
    flat_error = reprojection_errors(
        np.repeat(keypoints, topk.shape[1], axis=0),
        candidate_xyz.reshape(-1, 3),
        K,
        gt_pose,
    ).reshape(topk.shape)
    result = {
        f"reprojection_recall_{threshold}px": float(
            np.mean(np.min(flat_error, axis=1) <= threshold)
        )
        for threshold in (2.0, 4.0, 12.0)
    }
    if "splat_provenance_source_gaussian_idx" in query:
        provenance = np.asarray(
            query["splat_provenance_source_gaussian_idx"], dtype=np.int64
        )
        provenance_valid = np.asarray(
            query["splat_provenance_valid"], dtype=bool
        ).reshape(-1)
        candidate_source = bank_source[topk]
        overlap = np.any(
            candidate_source[:, :, None] == provenance[:, None, :],
            axis=(1, 2),
        )
        result["source_provenance_recall"] = float(
            np.mean(overlap[provenance_valid])
            if bool(provenance_valid.any())
            else 0.0
        )
        result["source_provenance_valid_fraction"] = float(
            np.mean(provenance_valid)
        )
    return result


def _summarize_variants(query_rows):
    variants = sorted(query_rows[0]["variants"]) if query_rows else []
    summary = {}
    for variant in variants:
        by_role = {}
        for role in ("all", "failure", "control"):
            rows = (
                query_rows
                if role == "all"
                else [row for row in query_rows if row["role"] == role]
            )
            if not rows:
                continue
            winner = np.asarray(
                [row["variants"][variant]["correct_wins"] for row in rows],
                dtype=bool,
            )
            standard = np.asarray(
                [
                    row["variants"]["standard_msac"]["correct_wins"]
                    for row in rows
                ],
                dtype=bool,
            )
            margins = np.asarray(
                [row["variants"][variant]["margin"] for row in rows],
                dtype=np.float64,
            )
            by_role[role] = {
                "query_count": len(rows),
                "correct_win_rate": float(np.mean(winner)),
                "standard_correct_win_rate": float(np.mean(standard)),
                "win_rate_delta": float(np.mean(winner) - np.mean(standard)),
                "recovered_count": int(np.sum(winner & ~standard)),
                "regressed_count": int(np.sum(~winner & standard)),
                "median_margin": float(np.median(margins)),
            }
        summary[variant] = by_role
    return summary


def evaluate(args):
    dump_dir = Path(args.dump_dir)
    manifest = json.loads((dump_dir / "manifest.json").read_text())
    bank = np.load(dump_dir / manifest["landmark_bank"])
    bank_xyz = np.asarray(bank["landmark_xyz"], dtype=np.float64)
    bank_source = np.asarray(bank["source_gaussian_idx"], dtype=np.int64)
    bank_dependency = np.asarray(bank["dependency_group_id"], dtype=np.int64)
    bank_track = np.asarray(bank["track_cluster_id"], dtype=np.int64)
    normals = _load_source_normals(args.gaussian_ply, bank_source)
    surface_groups, surface_diagnostics = build_surface_component_groups(
        bank_xyz,
        normals,
        voxel_scale_ratio=args.surface_voxel_scale_ratio,
        minimum_voxel_size=args.surface_minimum_voxel_size,
        maximum_normal_angle_degrees=args.surface_normal_angle,
    )
    selection = json.loads(Path(args.selection_report).read_text())
    selection_by_name = {
        str(row["image_name"]): row for row in selection["queries"]
    }
    query_rows = []
    for query_file in manifest["query_files"]:
        query = np.load(dump_dir / query_file)
        image_name = str(np.asarray(query["image_name"]).item())
        selection_row = selection_by_name[image_name]
        landmark_idx = np.asarray(
            query["hard_post_landmark_idx"], dtype=np.int64
        )
        keypoint_idx = np.asarray(
            query["hard_post_keypoint_idx"], dtype=np.int64
        )
        keypoint_xy = np.asarray(query["keypoint_xy"], dtype=np.float64)
        points2d = keypoint_xy[keypoint_idx] + 0.5
        points3d = bank_xyz[landmark_idx]
        K = np.asarray(query["K"], dtype=np.float64)
        gt_pose = np.asarray(query["gt_pose_w2c"], dtype=np.float64)
        predicted_pose = np.asarray(query["pred_pose_w2c"], dtype=np.float64)
        threshold = float(np.asarray(query["reprojection_error"]).item())
        width = int(np.asarray(query["width"]).item())
        height = int(np.asarray(query["height"]).item())
        correct_pose, basin_diagnostics = refine_correct_basin(
            points2d,
            points3d,
            K,
            gt_pose,
            threshold,
            width,
            height,
        )
        cells = image_cell_ids(
            points2d,
            width,
            height,
            rows=args.image_grid_rows,
            cols=args.image_grid_cols,
        )
        group_sets = {
            "source": bank_source[landmark_idx],
            "dependency": bank_dependency[landmark_idx],
            "track": bank_track[landmark_idx],
            "surface": surface_groups[landmark_idx],
        }
        variants = _compare_hypotheses(
            points2d,
            points3d,
            K,
            correct_pose,
            predicted_pose,
            threshold,
            group_sets,
            image_cell_normalization(cells),
            args.caps,
        )
        pred_ae, pred_te = cal_pose_error(predicted_pose, gt_pose)
        query_rows.append(
            {
                "image_name": image_name,
                "role": selection_row["role"],
                "baseline_mean_te_cm": selection_row["mean_te_cm"],
                "dump_predicted_te_cm": float(pred_te),
                "dump_predicted_ae_deg": float(pred_ae),
                "match_count": int(landmark_idx.size),
                "correct_basin": basin_diagnostics,
                "topk_coverage": _topk_coverage(
                    query, bank_xyz, bank_source, K, gt_pose
                ),
                "variants": variants,
            }
        )
    output = {
        "schema": "lafgs_group_saturated_consensus_oracle_v1",
        "scene": args.scene,
        "query_count": len(query_rows),
        "failure_count": sum(row["role"] == "failure" for row in query_rows),
        "control_count": sum(row["role"] == "control" for row in query_rows),
        "caps": args.caps,
        "surface_group_diagnostics": surface_diagnostics,
        "variant_summary": _summarize_variants(query_rows),
        "queries": query_rows,
    }
    return output


def _write_markdown(payload, path):
    lines = [
        f"# {payload['scene']} Group-Saturated Consensus Oracle",
        "",
        (
            f"Queries: {payload['query_count']} "
            f"({payload['failure_count']} failures, "
            f"{payload['control_count']} controls)"
        ),
        "",
        "| Variant | All delta | Failure delta | Control delta | Recovered | Regressed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in payload["variant_summary"].items():
        if name == "standard_msac":
            continue
        all_row = summary["all"]
        failure = summary.get("failure", {})
        control = summary.get("control", {})
        lines.append(
            f"| {name} | {all_row['win_rate_delta']:+.3f} | "
            f"{failure.get('win_rate_delta', 0.0):+.3f} | "
            f"{control.get('win_rate_delta', 0.0):+.3f} | "
            f"{all_row['recovered_count']} | {all_row['regressed_count']} |"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--gaussian-ply", required=True)
    parser.add_argument("--selection-report", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--caps", nargs="+", type=float, default=[1, 2, 4, 8])
    parser.add_argument("--image-grid-rows", type=int, default=4)
    parser.add_argument("--image-grid-cols", type=int, default=4)
    parser.add_argument("--surface-voxel-scale-ratio", type=float, default=0.02)
    parser.add_argument("--surface-minimum-voxel-size", type=float, default=0.5)
    parser.add_argument("--surface-normal-angle", type=float, default=25.0)
    args = parser.parse_args()
    payload = evaluate(args)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_markdown(payload, args.output_markdown)
    print(json.dumps(payload["variant_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
