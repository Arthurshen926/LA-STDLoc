#!/usr/bin/env python3
"""Build exact fixed-seed PoseLib replacement targets for harmful rows."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from localization_training.artifact_contract import sha256_file
from localization_training.exact_counterfactual_pose_teacher import (
    ExactCounterfactualConfig,
    improves_lexicographically,
    serialize_config,
    solve_counterfactual_pose,
)
from localization_training.harmful_outcome_triage import (
    RANK_FAILURE,
    TEACHER_MISS,
)


def _atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _records(payload: dict) -> dict[str, dict]:
    return {
        str(record["query_name"]): record
        for record in payload["records"]
    }


def _ragged_values(
    offsets: torch.Tensor, values: torch.Tensor, index: int
) -> torch.Tensor:
    start = int(offsets[index])
    stop = int(offsets[index + 1])
    return values[start:stop]


def _pack_candidates(records: list[dict]) -> dict:
    offsets = [0]
    for record in records:
        offsets.append(offsets[-1] + len(record["candidate_anchor_indices"]))
    fields = {
        "candidate_anchor_indices": torch.long,
        "candidate_accepted": torch.bool,
        "candidate_valid": torch.bool,
        "candidate_correct_basin": torch.bool,
        "candidate_strict_translation_success": torch.bool,
        "candidate_translation_error_cm": torch.float32,
        "candidate_rotation_error_degrees": torch.float32,
        "candidate_inlier_count": torch.int32,
        "candidate_harmful_consensus_count": torch.int32,
        "candidate_hypotheses": torch.int32,
        "candidate_geometry_diversity": torch.float32,
    }
    packed = {"candidate_offsets": torch.as_tensor(offsets, dtype=torch.long)}
    for name, dtype in fields.items():
        values = [
            value
            for record in records
            for value in record[name]
        ]
        packed[name] = torch.as_tensor(values, dtype=dtype)
    return packed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--triage", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--maximum-candidates-per-row", type=int, default=8)
    parser.add_argument("--reprojection-error", type=float, default=12.0)
    parser.add_argument("--confidence", type=float, default=0.99999)
    parser.add_argument("--maximum-iterations", type=int, default=100000)
    parser.add_argument("--minimum-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--clean-reprojection", type=float, default=4.0)
    parser.add_argument("--strict-translation-cm", type=float, default=5.0)
    parser.add_argument("--basin-translation-cm", type=float, default=50.0)
    parser.add_argument("--basin-rotation-degrees", type=float, default=5.0)
    args = parser.parse_args()
    torch.set_num_threads(1)

    paths = {
        "map": Path(args.map).resolve(),
        "selection": Path(args.selection).resolve(),
        "triage": Path(args.triage).resolve(),
        "query_cache": Path(args.query_cache).resolve(),
    }
    state = torch.load(paths["map"], map_location="cpu", weights_only=False)
    selection = torch.load(
        paths["selection"], map_location="cpu", weights_only=False
    )
    triage = torch.load(
        paths["triage"], map_location="cpu", weights_only=False
    )
    cache_payload = torch.load(
        paths["query_cache"], map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    names = list(selection["query_names"])
    if names != list(triage["query_names"]):
        raise ValueError("selection and triage query registries differ")
    if int(selection["anchor_count"]) != len(state["anchor_xyz"]):
        raise ValueError("selection does not align with map")
    if int(triage["active_anchor_count"]) != len(state["anchor_xyz"]):
        raise ValueError("triage does not align with map")
    selection_by_name = _records(selection)
    triage_by_name = _records(triage)
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
    config = ExactCounterfactualConfig(
        reprojection_error_px=args.reprojection_error,
        confidence=args.confidence,
        maximum_iterations=args.maximum_iterations,
        minimum_iterations=args.minimum_iterations,
        seed=args.seed,
        clean_reprojection_px=args.clean_reprojection,
        strict_translation_cm=args.strict_translation_cm,
        basin_translation_cm=args.basin_translation_cm,
        basin_rotation_degrees=args.basin_rotation_degrees,
        maximum_candidates_per_row=args.maximum_candidates_per_row,
    )
    start = max(int(args.query_start), 0)
    stop = len(names)
    if int(args.query_limit) > 0:
        stop = min(stop, start + int(args.query_limit))
    output_records = []
    totals = Counter()
    for query_index in range(start, stop):
        name = names[query_index]
        selected_record = selection_by_name[name]
        triage_record = triage_by_name[name]
        cached = cache[name]
        rows = torch.as_tensor(selected_record["query_rows"]).long()
        selected = torch.as_tensor(
            selected_record["selected_row_mask"]
        ).bool()
        selected_positions = torch.where(selected)[0]
        selected_rows = rows[selected_positions]
        anchors = torch.as_tensor(
            selected_record["topk_anchor_indices"]
        ).long()[:, 0][selected_positions]
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
        base_points3d = xyz[anchors.numpy()]
        base_outcome = solve_counterfactual_pose(
            points2d=points2d,
            points3d=base_points3d,
            intrinsics=intrinsics,
            ground_truth_w2c=ground_truth,
            dependency_groups=dependency[anchors.numpy()],
            source_groups=sources[anchors.numpy()],
            config=config,
        )
        local_by_full_position = {
            int(position): local
            for local, position in enumerate(selected_positions.tolist())
        }
        triage_positions = torch.as_tensor(
            triage_record["selected_row_positions"]
        ).long()
        categories = torch.as_tensor(
            triage_record["category"]
        ).long()
        candidate_offsets = torch.as_tensor(
            triage_record["active_positive_offsets"]
        ).long()
        candidate_values = torch.as_tensor(
            triage_record["active_positive_indices"]
        ).long()
        row_records = []
        for triage_local, full_position in enumerate(
            triage_positions.tolist()
        ):
            if int(categories[triage_local]) not in (
                RANK_FAILURE,
                TEACHER_MISS,
            ):
                continue
            selected_local = local_by_full_position.get(int(full_position))
            if selected_local is None:
                raise ValueError("triage row is absent from selected set")
            candidates = _ragged_values(
                candidate_offsets, candidate_values, triage_local
            )[: int(config.maximum_candidates_per_row)]
            wrong_anchor = int(anchors[selected_local])
            candidates = candidates[
                (candidates >= 0)
                & (candidates < len(xyz))
                & (candidates != wrong_anchor)
            ].unique(sorted=False)
            outcomes = []
            for candidate_anchor in candidates.tolist():
                replacement_anchors = anchors.numpy().copy()
                replacement_anchors[selected_local] = int(candidate_anchor)
                outcome = solve_counterfactual_pose(
                    points2d=points2d,
                    points3d=xyz[replacement_anchors],
                    intrinsics=intrinsics,
                    ground_truth_w2c=ground_truth,
                    dependency_groups=dependency[replacement_anchors],
                    source_groups=sources[replacement_anchors],
                    config=config,
                )
                outcome["candidate_anchor"] = int(candidate_anchor)
                outcomes.append(outcome)
            improving = [
                outcome
                for outcome in outcomes
                if improves_lexicographically(outcome, base_outcome)
            ]
            best = (
                max(
                    improving,
                    key=lambda value: (
                        bool(value["valid"]),
                        bool(value["correct_basin"]),
                        bool(value["strict_translation_success"]),
                        -float(value["translation_error_cm"]),
                        -float(value["rotation_error_degrees"]),
                        -int(value["harmful_consensus_count"]),
                        float(value["geometry_diversity"]),
                    ),
                )
                if improving
                else None
            )
            row_records.append(
                {
                    "query_row": int(rows[full_position]),
                    "selected_row_position": int(full_position),
                    "selected_local_position": int(selected_local),
                    "wrong_anchor_index": wrong_anchor,
                    "target_anchor_index": (
                        int(best["candidate_anchor"]) if best else -1
                    ),
                    "accepted": best is not None,
                    "candidate_anchor_indices": [
                        int(value["candidate_anchor"]) for value in outcomes
                    ],
                    "candidate_accepted": [
                        best is not None
                        and int(value["candidate_anchor"])
                        == int(best["candidate_anchor"])
                        for value in outcomes
                    ],
                    "candidate_valid": [
                        bool(value["valid"]) for value in outcomes
                    ],
                    "candidate_correct_basin": [
                        bool(value["correct_basin"]) for value in outcomes
                    ],
                    "candidate_strict_translation_success": [
                        bool(value["strict_translation_success"])
                        for value in outcomes
                    ],
                    "candidate_translation_error_cm": [
                        float(value["translation_error_cm"])
                        for value in outcomes
                    ],
                    "candidate_rotation_error_degrees": [
                        float(value["rotation_error_degrees"])
                        for value in outcomes
                    ],
                    "candidate_inlier_count": [
                        int(value["inlier_count"]) for value in outcomes
                    ],
                    "candidate_harmful_consensus_count": [
                        int(value["harmful_consensus_count"])
                        for value in outcomes
                    ],
                    "candidate_hypotheses": [
                        int(value["hypotheses"])
                        if value["hypotheses"] is not None
                        else -1
                        for value in outcomes
                    ],
                    "candidate_geometry_diversity": [
                        float(value["geometry_diversity"])
                        for value in outcomes
                    ],
                }
            )
        packed = _pack_candidates(row_records)
        packed.update(
            {
                "query_index": int(query_index),
                "query_name": name,
                "query_rows": torch.as_tensor(
                    [value["query_row"] for value in row_records],
                    dtype=torch.long,
                ),
                "selected_row_positions": torch.as_tensor(
                    [
                        value["selected_row_position"]
                        for value in row_records
                    ],
                    dtype=torch.long,
                ),
                "selected_local_positions": torch.as_tensor(
                    [
                        value["selected_local_position"]
                        for value in row_records
                    ],
                    dtype=torch.long,
                ),
                "wrong_anchor_indices": torch.as_tensor(
                    [
                        value["wrong_anchor_index"]
                        for value in row_records
                    ],
                    dtype=torch.long,
                ),
                "target_anchor_indices": torch.as_tensor(
                    [
                        value["target_anchor_index"]
                        for value in row_records
                    ],
                    dtype=torch.long,
                ),
                "accepted": torch.as_tensor(
                    [value["accepted"] for value in row_records],
                    dtype=torch.bool,
                ),
                "base_valid": bool(base_outcome["valid"]),
                "base_translation_error_cm": float(
                    base_outcome["translation_error_cm"]
                ),
                "base_rotation_error_degrees": float(
                    base_outcome["rotation_error_degrees"]
                ),
                "base_correct_basin": bool(
                    base_outcome["correct_basin"]
                ),
                "base_strict_translation_success": bool(
                    base_outcome["strict_translation_success"]
                ),
                "base_inlier_count": int(base_outcome["inlier_count"]),
                "base_harmful_consensus_count": int(
                    base_outcome["harmful_consensus_count"]
                ),
                "base_hypotheses": (
                    int(base_outcome["hypotheses"])
                    if base_outcome["hypotheses"] is not None
                    else -1
                ),
            }
        )
        output_records.append(packed)
        totals["queries"] += 1
        totals["eligible_rows"] += len(row_records)
        totals["candidate_replays"] += len(packed["candidate_anchor_indices"])
        totals["accepted_rows"] += int(packed["accepted"].sum())
        totals["base_strict_queries"] += int(
            base_outcome["strict_translation_success"]
        )
        accepted_mask = packed["candidate_accepted"]
        if not bool(base_outcome["strict_translation_success"]):
            totals["strict_crossings"] += int(
                (
                    accepted_mask
                    & packed["candidate_strict_translation_success"]
                ).sum()
            )
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
    payload = {
        "schema": "lafgs_exact_counterfactual_pose_teacher",
        "version": 1,
        "query_names": names,
        "query_start": start,
        "query_stop": stop,
        "anchor_count": len(xyz),
        "records": output_records,
        "summary": dict(totals),
        "config": serialize_config(config),
        "provenance": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
    }
    output = Path(args.output)
    _atomic_torch(output, payload)
    summary_output = Path(
        args.summary_output
        or output.with_suffix(".json")
    )
    _atomic_json(
        summary_output,
        {
            "schema": payload["schema"],
            "version": payload["version"],
            "query_start": start,
            "query_stop": stop,
            "summary": dict(totals),
            "config": payload["config"],
            "provenance": payload["provenance"],
        },
    )


if __name__ == "__main__":
    main()
