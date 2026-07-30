#!/usr/bin/env python3
"""Audit clean, independent, non-degenerate, and correct-basin P3P triplets."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import poselib
import torch
from scipy.stats import spearmanr

from localization_training.basis_utility import (
    deterministic_triplets,
    group_independent_triplets,
    image_triangle_area_fraction,
    triangle_shape_quality,
)
from utils.pose_utils import cal_pose_error


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _records(payload: dict) -> dict[str, dict]:
    return {
        str(record["query_name"]): record for record in payload["records"]
    }


def _camera_pose_matrix(pose) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(pose.R, dtype=np.float64)
    matrix[:3, 3] = np.asarray(pose.t, dtype=np.float64)
    return matrix


def _best_p3p_solution_error(
    image_points: torch.Tensor,
    world_points: torch.Tensor,
    intrinsics: torch.Tensor,
    ground_truth_w2c: torch.Tensor,
) -> tuple[float, float]:
    homogeneous = torch.cat(
        (
            image_points.double(),
            torch.ones(3, 1, dtype=torch.float64),
        ),
        dim=1,
    )
    bearings = homogeneous @ torch.linalg.inv(intrinsics.double()).T
    bearings = bearings / torch.linalg.norm(
        bearings, dim=1, keepdim=True
    ).clamp_min(1e-12)
    try:
        solutions = poselib.p3p(
            bearings.numpy(),
            world_points.double().numpy(),
        )
    except Exception:
        return float("inf"), float("inf")
    ground_truth = ground_truth_w2c.double().numpy()
    best = (float("inf"), float("inf"))
    best_task_error = float("inf")
    for solution in solutions:
        rotation_error, translation_error = cal_pose_error(
            _camera_pose_matrix(solution),
            ground_truth,
        )
        task_error = translation_error / 5.0 + rotation_error / 5.0
        if task_error < best_task_error:
            best_task_error = task_error
            best = float(translation_error), float(rotation_error)
    return best


def _safe_correlation(x: list[float], y: list[float]) -> float:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--dynamic-outcomes", required=True)
    parser.add_argument(
        "--selection",
        action="append",
        default=[],
        help="NAME=selected_topk.pt; omit to audit only the full baseline rows",
    )
    parser.add_argument(
        "--pose-results",
        action="append",
        default=[],
        help="NAME=replay.json; supplies each variant's own TE/hypotheses",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--triplets-per-query", type=int, default=256)
    parser.add_argument("--image-area-fraction-min", type=float, default=1e-4)
    parser.add_argument("--image-shape-quality-min", type=float, default=0.01)
    parser.add_argument("--world-shape-quality-min", type=float, default=0.01)
    parser.add_argument("--correct-basin-translation-cm", type=float, default=5.0)
    parser.add_argument("--correct-basin-rotation-deg", type=float, default=5.0)
    parser.add_argument(
        "--loose-basin-translation-cm", type=float, default=50.0
    )
    parser.add_argument("--loose-basin-rotation-deg", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    # These operations are tiny and repeated per query; a large BLAS thread
    # pool is substantially slower than one deterministic CPU worker.
    torch.set_num_threads(1)

    state = torch.load(args.map, map_location="cpu", weights_only=False)
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    dynamic = torch.load(
        args.dynamic_outcomes, map_location="cpu", weights_only=False
    )
    dynamic_by_name = _records(dynamic)
    variants: dict[str, dict[str, dict] | None] = {"baseline": None}
    selection_paths = {}
    for value in args.selection:
        if "=" not in value:
            raise ValueError("selection must use NAME=PATH")
        name, path = value.split("=", 1)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if list(payload["query_names"]) != list(dynamic["query_names"]):
            raise ValueError(f"selection {name} query ordering differs")
        variants[name] = _records(payload)
        selection_paths[name] = str(Path(path).resolve())
    pose_results = {}
    pose_result_paths = {}
    for value in args.pose_results:
        if "=" not in value:
            raise ValueError("pose-results must use NAME=PATH")
        name, path = value.split("=", 1)
        if name not in variants:
            raise ValueError(f"pose results refer to unknown variant {name}")
        payload = json.loads(Path(path).read_text())
        pose_results[name] = {
            str(record["query"]): record for record in payload["results"]
        }
        pose_result_paths[name] = str(Path(path).resolve())

    anchor_xyz = torch.as_tensor(state["anchor_xyz"]).float()
    dependency_by_anchor = torch.as_tensor(
        state.get(
            "coarse_dependency_group_ids",
            state["dependency_group_ids"],
        )
    ).long()
    source_by_anchor = torch.as_tensor(
        state["source_primitive_ids"]
    ).long()
    results: dict[str, list[dict]] = {name: [] for name in variants}
    for query_number, query_name in enumerate(dynamic["query_names"], start=1):
        dynamic_record = dynamic_by_name[query_name]
        cached = cache[query_name]
        rows = torch.as_tensor(dynamic_record["query_rows"]).long()
        anchors = torch.as_tensor(
            dynamic_record["top1_anchor_indices"]
        ).long()
        keypoints = (
            torch.as_tensor(cached["native_keypoints"]).float()[rows]
            + float(cached.get("pixel_center_offset", 0.5))
        )
        clean2 = (
            torch.as_tensor(
                dynamic_record["gt_reprojection_errors_px"]
            ).float()
            <= 2.0
        )
        ransac_inlier = torch.as_tensor(
            dynamic_record["ransac_inlier_mask"]
        ).bool()
        harmful = torch.as_tensor(
            dynamic_record["harmful_inlier_mask"]
        ).bool()
        dependency = dependency_by_anchor[anchors]
        source = source_by_anchor[anchors]
        world = anchor_xyz[anchors]
        for variant_name, selection_by_name in variants.items():
            if selection_by_name is None:
                selected = torch.ones(len(rows), dtype=torch.bool)
            else:
                selection_record = selection_by_name[query_name]
                if not torch.equal(
                    rows,
                    torch.as_tensor(
                        selection_record["query_rows"]
                    ).long(),
                ):
                    raise ValueError(
                        f"selection rows differ for {query_name}"
                    )
                selected = torch.as_tensor(
                    selection_record["selected_row_mask"]
                ).bool()
            triplets = deterministic_triplets(
                torch.where(selected)[0],
                count=int(args.triplets_per_query),
                seed=int(args.seed),
                query_name=f"{variant_name}:{query_name}",
            )
            clean_basis = clean2[triplets].all(dim=1)
            independent = group_independent_triplets(
                triplets, dependency, source
            )
            image_area = image_triangle_area_fraction(
                keypoints[triplets],
                cached["native_input_hw"],
            )
            image_quality = triangle_shape_quality(
                keypoints[triplets]
            )
            world_quality = triangle_shape_quality(world[triplets])
            non_degenerate = (
                (image_area >= float(args.image_area_fraction_min))
                & (
                    image_quality
                    >= float(args.image_shape_quality_min)
                )
                & (
                    world_quality
                    >= float(args.world_shape_quality_min)
                )
            )
            structurally_good = clean_basis & independent & non_degenerate
            harmful_consensus = (
                ransac_inlier[triplets].all(dim=1)
                & harmful[triplets].any(dim=1)
            )
            strict_correct_count = 0
            loose_correct_count = 0
            for triplet in triplets[structurally_good]:
                translation_error, rotation_error = _best_p3p_solution_error(
                    keypoints[triplet],
                    world[triplet],
                    torch.as_tensor(cached["native_K"]).float(),
                    torch.as_tensor(cached["pose_w2c"]).float(),
                )
                if (
                    translation_error
                    <= float(args.correct_basin_translation_cm)
                    and rotation_error
                    <= float(args.correct_basin_rotation_deg)
                ):
                    strict_correct_count += 1
                if (
                    translation_error
                    <= float(args.loose_basin_translation_cm)
                    and rotation_error
                    <= float(args.loose_basin_rotation_deg)
                ):
                    loose_correct_count += 1
            sample_count = max(len(triplets), 1)
            variant_pose = pose_results.get(variant_name, {}).get(
                query_name
            )
            results[variant_name].append(
                {
                    "query": query_name,
                    "selected_count": int(selected.sum()),
                    "sampled_basis_count": len(triplets),
                    "clean2_basis_rate": float(
                        clean_basis.sum() / sample_count
                    ),
                    "independent_basis_rate": float(
                        independent.sum() / sample_count
                    ),
                    "nondegenerate_basis_rate": float(
                        non_degenerate.sum() / sample_count
                    ),
                    "structurally_good_basis_rate": float(
                        structurally_good.sum() / sample_count
                    ),
                    "correct_basin_basis_rate": float(
                        strict_correct_count / sample_count
                    ),
                    "loose_basin_basis_rate": float(
                        loose_correct_count / sample_count
                    ),
                    "correct_given_structurally_good": float(
                        strict_correct_count
                        / max(int(structurally_good.sum()), 1)
                    ),
                    "loose_given_structurally_good": float(
                        loose_correct_count
                        / max(int(structurally_good.sum()), 1)
                    ),
                    "harmful_consensus_basis_rate": float(
                        harmful_consensus.sum() / sample_count
                    ),
                    "estimated_correct_basis_count": float(
                        math.comb(int(selected.sum()), 3)
                        * strict_correct_count
                        / sample_count
                    ),
                    "estimated_loose_basis_count": float(
                        math.comb(int(selected.sum()), 3)
                        * loose_correct_count
                        / sample_count
                    ),
                    "te_cm": float(
                        variant_pose["te_cm"]
                        if variant_pose is not None
                        else dynamic_record["te_cm"]
                    ),
                    "re_deg": float(
                        variant_pose["re_deg"]
                        if variant_pose is not None
                        else dynamic_record["re_deg"]
                    ),
                    "hypotheses": int(
                        variant_pose["hypotheses"]
                        if variant_pose is not None
                        else dynamic_record["hypotheses"]
                    ),
                }
            )
        if query_number % 50 == 0:
            print(f"{query_number}/{len(dynamic['query_names'])}", flush=True)

    summary = {}
    for name, records in results.items():
        fields = (
            "selected_count",
            "clean2_basis_rate",
            "independent_basis_rate",
            "nondegenerate_basis_rate",
            "structurally_good_basis_rate",
            "correct_basin_basis_rate",
            "loose_basin_basis_rate",
            "correct_given_structurally_good",
            "loose_given_structurally_good",
            "harmful_consensus_basis_rate",
            "estimated_correct_basis_count",
            "estimated_loose_basis_count",
        )
        aggregate = {
            f"mean_{field}": float(
                np.mean([record[field] for record in records])
            )
            for field in fields
        }
        correct_rates = [
            record["correct_basin_basis_rate"] for record in records
        ]
        correct_counts = [
            math.log1p(record["estimated_correct_basis_count"])
            for record in records
        ]
        loose_rates = [
            record["loose_basin_basis_rate"] for record in records
        ]
        loose_counts = [
            math.log1p(record["estimated_loose_basis_count"])
            for record in records
        ]
        te = [record["te_cm"] for record in records]
        hypotheses = [record["hypotheses"] for record in records]
        aggregate.update(
            {
                "query_count": len(records),
                "zero_correct_basis_query_fraction": float(
                    np.mean(np.asarray(correct_rates) <= 0)
                ),
                "correct_basis_rate_spearman_te": _safe_correlation(
                    correct_rates, te
                ),
                "log_correct_basis_count_spearman_te": _safe_correlation(
                    correct_counts, te
                ),
                "correct_basis_rate_spearman_hypotheses": _safe_correlation(
                    correct_rates, hypotheses
                ),
                "loose_basis_rate_spearman_te": _safe_correlation(
                    loose_rates, te
                ),
                "log_loose_basis_count_spearman_te": _safe_correlation(
                    loose_counts, te
                ),
                "loose_basis_rate_spearman_hypotheses": _safe_correlation(
                    loose_rates, hypotheses
                ),
            }
        )
        summary[name] = aggregate

    payload = {
        "schema": "lafgs_v25_basis_audit",
        "config": vars(args),
        "selection_paths": selection_paths,
        "pose_result_paths": pose_result_paths,
        "summary": summary,
        "records": results,
    }
    _atomic_json(Path(args.output).resolve(), payload)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
