#!/usr/bin/env python3
"""Measure exact fixed-seed leave-one-out R5 leverage on the final set."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import torch

from localization_training.artifact_contract import sha256_file
from localization_training.exact_counterfactual_pose_teacher import (
    ExactCounterfactualConfig,
    serialize_config,
    solve_counterfactual_pose,
)


def _atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--dynamic-outcomes", required=True)
    parser.add_argument("--exact-teacher", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--maximum-rows-per-query", type=int, default=32)
    parser.add_argument("--reprojection-error", type=float, default=12.0)
    parser.add_argument("--confidence", type=float, default=0.99999)
    parser.add_argument("--maximum-iterations", type=int, default=100000)
    parser.add_argument("--minimum-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--clean-reprojection", type=float, default=4.0)
    parser.add_argument("--strict-translation-cm", type=float, default=5.0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    paths = {
        "map": Path(args.map).resolve(),
        "selection": Path(args.selection).resolve(),
        "dynamic_outcomes": Path(args.dynamic_outcomes).resolve(),
        "exact_teacher": Path(args.exact_teacher).resolve(),
        "query_cache": Path(args.query_cache).resolve(),
    }
    payload = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in paths.items()
    }
    state = payload["map"]
    selection = payload["selection"]
    dynamic = payload["dynamic_outcomes"]
    exact = payload["exact_teacher"]
    cache = payload["query_cache"].get(
        "queries", payload["query_cache"]
    )
    names = list(selection["query_names"])
    if (
        names != list(dynamic["query_names"])
        or names != list(exact["query_names"])
    ):
        raise ValueError("exact LOO query registries differ")
    if not (
        int(selection["anchor_count"])
        == int(dynamic["anchor_count"])
        == int(exact["anchor_count"])
        == len(state["anchor_xyz"])
    ):
        raise ValueError("exact LOO map registries differ")
    config = ExactCounterfactualConfig(
        reprojection_error_px=args.reprojection_error,
        confidence=args.confidence,
        maximum_iterations=args.maximum_iterations,
        minimum_iterations=args.minimum_iterations,
        seed=args.seed,
        clean_reprojection_px=args.clean_reprojection,
        strict_translation_cm=args.strict_translation_cm,
    )
    xyz = torch.as_tensor(state["anchor_xyz"]).double().numpy()
    dependency = torch.as_tensor(
        state.get(
            "coarse_dependency_group_ids",
            state["dependency_group_ids"],
        )
    ).long().numpy()
    sources = torch.as_tensor(
        state["source_primitive_ids"]
    ).long().numpy()
    start = max(int(args.query_start), 0)
    stop = len(names)
    if int(args.query_limit) > 0:
        stop = min(stop, start + int(args.query_limit))
    records = []
    totals = Counter()
    for query_index in range(start, stop):
        name = names[query_index]
        selected_record = selection["records"][query_index]
        dynamic_record = dynamic["records"][query_index]
        exact_record = exact["records"][query_index]
        if not bool(exact_record["base_strict_translation_success"]):
            records.append(
                {
                    "query_index": query_index,
                    "query_name": name,
                    "query_rows": torch.empty(0, dtype=torch.long),
                    "selected_local_positions": torch.empty(
                        0, dtype=torch.long
                    ),
                    "anchor_indices": torch.empty(0, dtype=torch.long),
                    "loo_translation_error_cm": torch.empty(0),
                    "loo_rotation_error_degrees": torch.empty(0),
                    "loo_valid": torch.empty(0, dtype=torch.bool),
                    "threshold_crossing": torch.empty(
                        0, dtype=torch.bool
                    ),
                }
            )
            totals["queries"] += 1
            continue
        rows = torch.as_tensor(selected_record["query_rows"]).long()
        selected = torch.as_tensor(
            selected_record["selected_row_mask"]
        ).bool()
        selected_positions = torch.where(selected)[0]
        selected_rows = rows[selected_positions]
        anchors = torch.as_tensor(
            selected_record["topk_anchor_indices"]
        ).long()[:, 0][selected_positions]
        cached = cache[name]
        points2d = (
            torch.as_tensor(cached["native_keypoints"]).double()[selected_rows]
            + float(cached.get("pixel_center_offset", 0.5))
        ).numpy()
        intrinsics = torch.as_tensor(
            cached["native_K"]
        ).double().numpy()
        ground_truth = torch.as_tensor(
            cached["pose_w2c"]
        ).double().numpy()
        base = solve_counterfactual_pose(
            points2d=points2d,
            points3d=xyz[anchors.numpy()],
            intrinsics=intrinsics,
            ground_truth_w2c=ground_truth,
            dependency_groups=dependency[anchors.numpy()],
            source_groups=sources[anchors.numpy()],
            config=config,
        )
        if not bool(base["strict_translation_success"]):
            raise ValueError(
                "LOO base replay differs from exact-teacher base"
            )
        dynamic_errors = torch.as_tensor(
            dynamic_record["gt_reprojection_errors_px"]
        ).float()[selected_positions]
        inliers = torch.as_tensor(
            base["inlier_indices"]
        ).long().reshape(-1)
        clean_inliers = inliers[
            dynamic_errors[inliers]
            <= float(config.clean_reprojection_px)
        ]
        strict_probability = torch.as_tensor(
            selected_record["strict_probability"]
        ).float()[selected_positions]
        order = clean_inliers[
            torch.argsort(
                strict_probability[clean_inliers],
                descending=True,
                stable=True,
            )
        ][: max(int(args.maximum_rows_per_query), 0)]
        loo_translation = []
        loo_rotation = []
        loo_valid = []
        crossing = []
        for selected_local in order.tolist():
            keep = torch.ones(len(anchors), dtype=torch.bool)
            keep[selected_local] = False
            kept_anchors = anchors[keep].numpy()
            outcome = solve_counterfactual_pose(
                points2d=points2d[keep.numpy()],
                points3d=xyz[kept_anchors],
                intrinsics=intrinsics,
                ground_truth_w2c=ground_truth,
                dependency_groups=dependency[kept_anchors],
                source_groups=sources[kept_anchors],
                config=config,
            )
            crossed = bool(
                not outcome["valid"]
                or outcome["translation_error_cm"]
                > float(config.strict_translation_cm)
            )
            loo_translation.append(outcome["translation_error_cm"])
            loo_rotation.append(outcome["rotation_error_degrees"])
            loo_valid.append(outcome["valid"])
            crossing.append(crossed)
        records.append(
            {
                "query_index": query_index,
                "query_name": name,
                "query_rows": selected_rows[order],
                "selected_local_positions": order,
                "anchor_indices": anchors[order],
                "base_translation_error_cm": float(
                    base["translation_error_cm"]
                ),
                "base_rotation_error_degrees": float(
                    base["rotation_error_degrees"]
                ),
                "loo_translation_error_cm": torch.as_tensor(
                    loo_translation, dtype=torch.float32
                ),
                "loo_rotation_error_degrees": torch.as_tensor(
                    loo_rotation, dtype=torch.float32
                ),
                "loo_valid": torch.as_tensor(
                    loo_valid, dtype=torch.bool
                ),
                "threshold_crossing": torch.as_tensor(
                    crossing, dtype=torch.bool
                ),
            }
        )
        totals["queries"] += 1
        totals["strict_base_queries"] += 1
        totals["loo_replays"] += len(order)
        totals["threshold_crossings"] += sum(crossing)
        if (query_index - start + 1) % 10 == 0 or query_index + 1 == stop:
            print(
                json.dumps(
                    {
                        "completed": query_index - start + 1,
                        "total": stop - start,
                        **dict(totals),
                    }
                ),
                flush=True,
            )
    output = {
        "schema": "lafgs_exact_loo_threshold_teacher",
        "version": 1,
        "query_names": names,
        "query_start": start,
        "query_stop": stop,
        "anchor_count": len(xyz),
        "records": records,
        "summary": dict(totals),
        "config": {
            **serialize_config(config),
            "maximum_rows_per_query": int(args.maximum_rows_per_query),
            "candidate_policy": (
                "fixed-seed base inlier AND GT-clean, ranked by OOF "
                "strict probability"
            ),
        },
        "provenance": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
    }
    output_path = Path(args.output).resolve()
    _atomic_torch(output_path, output)
    output_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": output["schema"],
                "version": output["version"],
                "query_start": start,
                "query_stop": stop,
                "summary": output["summary"],
                "config": output["config"],
                "provenance": output["provenance"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
