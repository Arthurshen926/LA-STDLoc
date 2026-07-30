#!/usr/bin/env python3
"""Build balanced correct/harmful/near-miss P3P basin hyperedges."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from localization_training.artifact_contract import sha256_file
from localization_training.basin_distillation import (
    GOOD_SET,
    HARMFUL_SET,
    NEAR_MISS_SET,
    aggregate_edge_credit,
    evaluate_p3p_triplet,
    expanded_positive_lookup,
    image_cell_ids,
    is_diverse_triplet,
    proposal_propensity,
)


def _risk(outcome: dict) -> float:
    if not outcome["valid"]:
        return 20.0
    return min(float(outcome["te_cm"]) / 50.0, 10.0) + min(
        float(outcome["re_deg"]) / 5.0, 10.0
    )


def _draw_diverse(
    rng: np.random.Generator,
    pool: np.ndarray,
    points3d: np.ndarray,
    dependency: np.ndarray,
    cells: np.ndarray,
    surfaces: np.ndarray,
    attempts: int,
):
    for attempt in range(1, int(attempts) + 1):
        if pool.size < 3:
            break
        sample = rng.choice(pool, size=3, replace=False)
        if is_diverse_triplet(
            sample, points3d, dependency, cells, surfaces
        ):
            return sample, attempt
    return None, int(attempts)


def _as_record(
    *,
    query_rows,
    anchors,
    set_type,
    outcome,
    propensity,
    proposal_attempts,
    parent=-1,
    replaced_position=-1,
    blame=0.0,
):
    return {
        "query_rows": [int(value) for value in query_rows],
        "anchor_indices": [int(value) for value in anchors],
        "set_type": int(set_type),
        "correct_basin": bool(outcome["correct_basin"]),
        "valid": bool(outcome["valid"]),
        "inlier_count": int(outcome["inlier_count"]),
        "msac_cost": float(outcome["msac_cost"]),
        "te_cm": float(outcome["te_cm"]),
        "re_deg": float(outcome["re_deg"]),
        "sampling_propensity": float(max(propensity, 1e-12)),
        "proposal_attempts": int(proposal_attempts),
        "parent_set_index": int(parent),
        "replaced_position": int(replaced_position),
        "counterfactual_blame": float(max(blame, 0.0)),
    }


def _pack_records(records: list[dict]) -> dict:
    if not records:
        return {
            "set_query_rows": torch.empty((0, 3), dtype=torch.long),
            "set_anchor_indices": torch.empty((0, 3), dtype=torch.long),
            "set_types": torch.empty(0, dtype=torch.int8),
            "correct_basin": torch.empty(0, dtype=torch.bool),
            "inlier_count": torch.empty(0, dtype=torch.int32),
            "msac_cost": torch.empty(0),
            "te_cm": torch.empty(0),
            "re_deg": torch.empty(0),
            "sampling_propensity": torch.empty(0),
            "proposal_attempts": torch.empty(0, dtype=torch.int16),
            "parent_set_index": torch.empty(0, dtype=torch.long),
            "replaced_position": torch.empty(0, dtype=torch.int8),
            "counterfactual_blame": torch.empty(0),
        }
    packed = {
        "set_query_rows": torch.as_tensor(
            [record["query_rows"] for record in records], dtype=torch.long
        ),
        "set_anchor_indices": torch.as_tensor(
            [record["anchor_indices"] for record in records], dtype=torch.long
        ),
        "set_types": torch.as_tensor(
            [record["set_type"] for record in records], dtype=torch.int8
        ),
        "correct_basin": torch.as_tensor(
            [record["correct_basin"] for record in records], dtype=torch.bool
        ),
        "inlier_count": torch.as_tensor(
            [record["inlier_count"] for record in records], dtype=torch.int32
        ),
        "msac_cost": torch.as_tensor(
            [record["msac_cost"] for record in records], dtype=torch.float32
        ),
        "te_cm": torch.as_tensor(
            [record["te_cm"] for record in records], dtype=torch.float32
        ),
        "re_deg": torch.as_tensor(
            [record["re_deg"] for record in records], dtype=torch.float32
        ),
        "sampling_propensity": torch.as_tensor(
            [record["sampling_propensity"] for record in records],
            dtype=torch.float64,
        ),
        "proposal_attempts": torch.as_tensor(
            [record["proposal_attempts"] for record in records], dtype=torch.int16
        ),
        "parent_set_index": torch.as_tensor(
            [record["parent_set_index"] for record in records], dtype=torch.long
        ),
        "replaced_position": torch.as_tensor(
            [record["replaced_position"] for record in records], dtype=torch.int8
        ),
        "counterfactual_blame": torch.as_tensor(
            [record["counterfactual_blame"] for record in records],
            dtype=torch.float32,
        ),
    }
    severity = (
        torch.log1p(packed["inlier_count"].float())
        * (
            1.0
            + (packed["te_cm"] / 100.0).nan_to_num(posinf=10.0).clamp(max=10.0)
        )
    )
    packed["edge_credit"] = aggregate_edge_credit(
        packed["set_query_rows"],
        packed["set_anchor_indices"],
        packed["set_types"],
        packed["correct_basin"],
        packed["sampling_propensity"],
        severity,
    )
    blame_mask = (
        (packed["set_types"] == NEAR_MISS_SET)
        & (packed["parent_set_index"] >= 0)
        & (packed["replaced_position"] >= 0)
        & (packed["counterfactual_blame"] > 0)
    )
    blame_records = []
    for child_index in torch.nonzero(blame_mask, as_tuple=False).reshape(-1).tolist():
        parent_index = int(packed["parent_set_index"][child_index])
        position = int(packed["replaced_position"][child_index])
        blame_records.append(
            (
                int(packed["set_query_rows"][child_index, position]),
                int(packed["set_anchor_indices"][parent_index, position]),
                int(packed["set_anchor_indices"][child_index, position]),
                float(packed["counterfactual_blame"][child_index]),
            )
        )
    packed["blame_rows"] = torch.as_tensor(
        [value[0] for value in blame_records], dtype=torch.long
    )
    packed["blame_harmful_anchors"] = torch.as_tensor(
        [value[1] for value in blame_records], dtype=torch.long
    )
    packed["blame_positive_anchors"] = torch.as_tensor(
        [value[2] for value in blame_records], dtype=torch.long
    )
    packed["blame_weights"] = torch.as_tensor(
        [value[3] for value in blame_records], dtype=torch.float32
    )
    return packed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--dynamic-outcomes", required=True)
    parser.add_argument(
        "--selection",
        default="",
        help="Optional final correspondence-set selection (for example P1-512).",
    )
    parser.add_argument("--complete-positive-teacher", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--good-sets-per-query", type=int, default=8)
    parser.add_argument("--harmful-sets-per-query", type=int, default=8)
    parser.add_argument("--near-miss-per-query", type=int, default=8)
    parser.add_argument("--proposal-multiplier", type=int, default=8)
    parser.add_argument("--diversity-attempts", type=int, default=32)
    parser.add_argument("--clean-threshold-px", type=float, default=4.0)
    parser.add_argument("--harmful-threshold-px", type=float, default=12.0)
    parser.add_argument("--correct-translation-cm", type=float, default=50.0)
    parser.add_argument("--correct-rotation-deg", type=float, default=5.0)
    parser.add_argument("--minimum-harmful-inliers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-limit", type=int, default=0)
    args = parser.parse_args()

    paths = {
        "map": Path(args.map).resolve(),
        "dynamic_outcomes": Path(args.dynamic_outcomes).resolve(),
        "complete_positive_teacher": Path(
            args.complete_positive_teacher
        ).resolve(),
        "query_cache": Path(args.query_cache).resolve(),
    }
    if args.selection:
        paths["selection"] = Path(args.selection).resolve()
    state = torch.load(paths["map"], map_location="cpu", weights_only=False)
    outcomes = torch.load(
        paths["dynamic_outcomes"], map_location="cpu", weights_only=False
    )
    selection = (
        torch.load(
            paths["selection"], map_location="cpu", weights_only=False
        )
        if "selection" in paths
        else None
    )
    positives = torch.load(
        paths["complete_positive_teacher"], map_location="cpu", weights_only=False
    )
    cache_payload = torch.load(
        paths["query_cache"], map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    names = list(outcomes["query_names"])
    if names != list(positives["query_names"]):
        raise ValueError("outcome and positive-teacher query registries differ")
    anchor_count = int(torch.as_tensor(state["anchor_xyz"]).shape[0])
    if int(outcomes["anchor_count"]) != anchor_count:
        raise ValueError("dynamic outcomes do not align with map")
    if int(positives["anchor_count"]) != anchor_count:
        raise ValueError("positive teacher does not align with map")
    if selection is not None:
        if names != list(selection["query_names"]):
            raise ValueError("basis selection query registry differs")
        if int(selection["anchor_count"]) != anchor_count:
            raise ValueError("basis selection does not align with map")
    torch.set_num_threads(1)

    xyz = torch.as_tensor(state["anchor_xyz"]).double().numpy()
    dependency = torch.as_tensor(state["dependency_group_ids"]).long().numpy()
    surfaces = torch.as_tensor(state["source_primitive_ids"]).long().numpy()
    start = max(int(args.query_start), 0)
    stop = len(names)
    if int(args.query_limit) > 0:
        stop = min(stop, start + int(args.query_limit))
    records = []
    totals = {
        "good": 0,
        "harmful": 0,
        "near_miss": 0,
        "queries_with_good": 0,
        "queries_with_harmful": 0,
        "queries_with_near_miss": 0,
    }
    for query_index in range(start, stop):
        name = names[query_index]
        dynamic = outcomes["records"][query_index]
        selected_record = (
            selection["records"][query_index]
            if selection is not None
            else None
        )
        positive_lookup = expanded_positive_lookup(positives["records"][query_index])
        cached = cache[name]
        rows_tensor = torch.as_tensor(dynamic["query_rows"]).long()
        anchors_tensor = torch.as_tensor(
            dynamic["top1_anchor_indices"]
        ).long()
        if selected_record is not None:
            selected_rows = torch.as_tensor(
                selected_record["query_rows"]
            ).long()
            if not torch.equal(rows_tensor, selected_rows):
                raise ValueError("basis selection rows differ")
            selected_top1 = torch.as_tensor(
                selected_record["topk_anchor_indices"]
            ).long()[:, 0]
            if not torch.equal(anchors_tensor, selected_top1):
                raise ValueError(
                    "basis selection and dynamic top1 identities differ"
                )
            selected_mask = torch.as_tensor(
                selected_record["selected_row_mask"]
            ).bool()
        else:
            selected_mask = torch.ones(len(rows_tensor), dtype=torch.bool)
        rows = rows_tensor[selected_mask].numpy()
        anchors = anchors_tensor[selected_mask].numpy()
        points2d_all = (
            torch.as_tensor(cached["native_keypoints"]).double().numpy()
            + float(cached.get("pixel_center_offset", 0.5))
        )
        points2d = points2d_all[rows]
        points3d = xyz[anchors]
        K = torch.as_tensor(cached["native_K"]).double().numpy()
        gt_pose = torch.as_tensor(cached["pose_w2c"]).double().numpy()
        height, width = cached["native_input_hw"]
        cells = image_cell_ids(points2d, int(width), int(height))
        edge_dependency = dependency[anchors]
        edge_surfaces = surfaces[anchors]
        errors = (
            torch.as_tensor(dynamic["gt_reprojection_errors_px"])
            .float()[selected_mask]
            .numpy()
        )
        dynamic_harmful = torch.as_tensor(
            dynamic["harmful_inlier_mask"]
        ).bool()[selected_mask].numpy()
        clean_pool = np.flatnonzero(errors <= float(args.clean_threshold_px))
        harmful_pool = np.flatnonzero(
            (errors > float(args.harmful_threshold_px))
            | dynamic_harmful
        )
        repairable_harmful_pool = np.asarray(
            [
                edge
                for edge in harmful_pool.tolist()
                if any(
                    anchor != int(anchors[edge])
                    for anchor in positive_lookup.get(int(rows[edge]), [])
                )
            ],
            dtype=np.int64,
        )
        broad_pool = np.arange(rows.size)
        rng = np.random.default_rng(int(args.seed) + query_index * 104729)
        query_sets: list[dict] = []
        seen = set()

        good_trials = max(
            int(args.good_sets_per_query) * int(args.proposal_multiplier), 1
        )
        good_accepted_attempts = 0
        for _ in range(good_trials):
            if sum(record["set_type"] == GOOD_SET for record in query_sets) >= int(
                args.good_sets_per_query
            ):
                break
            sample, attempts = _draw_diverse(
                rng,
                clean_pool,
                points3d,
                edge_dependency,
                cells,
                edge_surfaces,
                args.diversity_attempts,
            )
            good_accepted_attempts += attempts
            if sample is None:
                continue
            identity = tuple(sorted((int(rows[i]), int(anchors[i])) for i in sample))
            if identity in seen:
                continue
            outcome = evaluate_p3p_triplet(
                sample,
                points2d,
                points3d,
                K,
                gt_pose,
                correct_translation_cm=args.correct_translation_cm,
                correct_rotation_deg=args.correct_rotation_deg,
            )
            if not outcome["correct_basin"]:
                continue
            seen.add(identity)
            base = proposal_propensity(clean_pool.size, 0.4)
            acceptance = max(
                (len(query_sets) + 1) / max(good_accepted_attempts, 1), 1e-6
            )
            query_sets.append(
                _as_record(
                    query_rows=rows[sample],
                    anchors=anchors[sample],
                    set_type=GOOD_SET,
                    outcome=outcome,
                    propensity=min(base / acceptance, 1.0),
                    proposal_attempts=attempts,
                )
            )

        harmful_records = []
        harmful_trials = max(
            int(args.harmful_sets_per_query) * int(args.proposal_multiplier), 1
        )
        harmful_attempt_count = 0
        harmful_proposal_pool = broad_pool
        for _ in range(harmful_trials):
            if len(harmful_records) >= int(args.harmful_sets_per_query):
                break
            sample = None
            attempts = 0
            # Most useful hard sets contain two already-correct edges and one
            # deceptive edge; fully random triplets overwhelmingly contain
            # three bad correspondences and rarely admit a one-edge repair.
            mixed_harmful_pool = (
                repairable_harmful_pool
                if repairable_harmful_pool.size
                else harmful_pool
            )
            if clean_pool.size >= 2 and mixed_harmful_pool.size:
                for attempts in range(1, int(args.diversity_attempts) + 1):
                    proposed = np.concatenate(
                        (
                            rng.choice(clean_pool, size=2, replace=False),
                            rng.choice(mixed_harmful_pool, size=1, replace=False),
                        )
                    )
                    rng.shuffle(proposed)
                    if is_diverse_triplet(
                        proposed,
                        points3d,
                        edge_dependency,
                        cells,
                        edge_surfaces,
                    ):
                        sample = proposed
                        break
            if sample is None:
                sample, extra_attempts = _draw_diverse(
                    rng,
                    harmful_proposal_pool,
                    points3d,
                    edge_dependency,
                    cells,
                    edge_surfaces,
                    args.diversity_attempts,
                )
                attempts += extra_attempts
            harmful_attempt_count += attempts
            if sample is None or not bool(
                (errors[sample] > float(args.harmful_threshold_px)).any()
            ):
                continue
            identity = tuple(sorted((int(rows[i]), int(anchors[i])) for i in sample))
            if identity in seen:
                continue
            outcome = evaluate_p3p_triplet(
                sample,
                points2d,
                points3d,
                K,
                gt_pose,
                correct_translation_cm=args.correct_translation_cm,
                correct_rotation_deg=args.correct_rotation_deg,
            )
            if outcome["correct_basin"] or int(outcome["inlier_count"]) < int(
                args.minimum_harmful_inliers
            ):
                continue
            seen.add(identity)
            if clean_pool.size >= 2 and mixed_harmful_pool.size:
                base = 0.4 / max(
                    math.comb(int(clean_pool.size), 2)
                    * int(mixed_harmful_pool.size),
                    1,
                )
            else:
                base = proposal_propensity(harmful_proposal_pool.size, 0.4)
            acceptance = max(
                (len(harmful_records) + 1) / max(harmful_attempt_count, 1), 1e-6
            )
            record = _as_record(
                query_rows=rows[sample],
                anchors=anchors[sample],
                set_type=HARMFUL_SET,
                outcome=outcome,
                propensity=min(base / acceptance, 1.0),
                proposal_attempts=attempts,
            )
            record["_sample"] = sample
            harmful_records.append(record)
            query_sets.append(record)

        near_count = 0
        for parent_record in harmful_records:
            if near_count >= int(args.near_miss_per_query):
                break
            sample = parent_record["_sample"]
            parent_risk = _risk(parent_record)
            best = None
            replaceable = []
            for position, edge_index in enumerate(sample.tolist()):
                legal = positive_lookup.get(int(rows[edge_index]), [])
                legal = [anchor for anchor in legal if anchor != int(anchors[edge_index])]
                if legal:
                    replaceable.append((position, edge_index, legal))
            for position, edge_index, legal in replaceable:
                for replacement in legal[:4]:
                    basis = points3d[sample].copy()
                    basis[position] = xyz[int(replacement)]
                    child = evaluate_p3p_triplet(
                        sample,
                        points2d,
                        points3d,
                        K,
                        gt_pose,
                        basis_points3d=basis,
                        correct_translation_cm=args.correct_translation_cm,
                        correct_rotation_deg=args.correct_rotation_deg,
                    )
                    blame = parent_risk - _risk(child)
                    candidate = (
                        bool(child["correct_basin"]),
                        float(blame),
                        -float(child["te_cm"]),
                        position,
                        edge_index,
                        int(replacement),
                        child,
                        len(legal),
                    )
                    if best is None or candidate[:3] > best[:3]:
                        best = candidate
            if best is None or not best[0] or best[1] <= 0:
                continue
            _, blame, _, position, edge_index, replacement, child, legal_count = best
            child_anchors = anchors[sample].copy()
            child_anchors[position] = replacement
            parent_index = query_sets.index(parent_record)
            conditional = 1.0 / max(len(replaceable) * legal_count, 1)
            query_sets.append(
                _as_record(
                    query_rows=rows[sample],
                    anchors=child_anchors,
                    set_type=NEAR_MISS_SET,
                    outcome=child,
                    propensity=min(
                        float(parent_record["sampling_propensity"])
                        * 0.2
                        * conditional,
                        1.0,
                    ),
                    proposal_attempts=1,
                    parent=parent_index,
                    replaced_position=position,
                    blame=blame,
                )
            )
            near_count += 1
        for record in query_sets:
            record.pop("_sample", None)
        packed = _pack_records(query_sets)
        packed.update({"query_index": query_index, "query_name": name})
        records.append(packed)
        counts = {
            "good": int((packed["set_types"] == GOOD_SET).sum()),
            "harmful": int((packed["set_types"] == HARMFUL_SET).sum()),
            "near_miss": int((packed["set_types"] == NEAR_MISS_SET).sum()),
        }
        for key, value in counts.items():
            totals[key] += value
            totals[f"queries_with_{key}"] += int(value > 0)
        if (query_index - start + 1) % 25 == 0:
            print(
                json.dumps(
                    {
                        "completed": query_index - start + 1,
                        "total": stop - start,
                        **totals,
                    }
                ),
                flush=True,
            )

    output = {
        "schema": "lafgs_basin_teacher",
        "version": 3 if selection is not None else 2,
        "query_names": names[start:stop],
        "query_start": start,
        "query_stop": stop,
        "anchor_count": anchor_count,
        "records": records,
        "summary": {
            **totals,
            "query_count": len(records),
            "good_sets_per_query": float(totals["good"] / max(len(records), 1)),
            "harmful_sets_per_query": float(
                totals["harmful"] / max(len(records), 1)
            ),
            "near_miss_per_query": float(
                totals["near_miss"] / max(len(records), 1)
            ),
        },
        "config": vars(args),
        "correspondence_set": (
            "selected_final_set" if selection is not None else "all_rows"
        ),
        "artifacts": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in paths.items()
        },
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(output, temporary)
    os.replace(temporary, output_path)
    output_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": output["schema"],
                "version": output["version"],
                "summary": output["summary"],
                "config": output["config"],
                "artifacts": output["artifacts"],
            },
            indent=2,
        )
        + "\n"
    )
    print(output_path)


if __name__ == "__main__":
    main()
