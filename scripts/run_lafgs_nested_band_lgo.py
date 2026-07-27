#!/usr/bin/env python3
"""Measure exact affected-query pose influence of redundancy groups."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from scripts.build_lafgs_lgo_baselines import (
    _active_mask,
    _assignment_positions,
)
from scripts.run_lafgs_alternating_structure import _pose_error_m
from utils.pose_utils import solve_pose


def _screen_groups(
    groups: list[dict], per_band: int, mode: str = "legacy"
) -> list[dict]:
    selected = []
    for band in (1, 2, 3):
        candidates = [
            group for group in groups if int(group["band"]) == band
        ]
        if per_band <= 0 or len(candidates) <= per_band:
            selected.extend(candidates)
            continue
        harmful = sorted(
            candidates,
            key=lambda group: (
                -group["harmful_consensus_count"]
                / max(group["opportunity_count"], 1),
                -group["harmful_consensus_count"],
                group["group_id"],
            ),
        )
        rescue = sorted(
            candidates,
            key=lambda group: (
                -group["gtclean_inlier_count"],
                -group["support_edge_count"],
                group["group_id"],
            ),
        )
        if mode == "active_diverse":
            uncertain = sorted(
                candidates,
                key=lambda group: (
                    abs(
                        group["gtclean_inlier_count"]
                        - group["harmful_consensus_count"]
                    )
                    / max(group["opportunity_count"], 1),
                    -group["opportunity_count"],
                    group["group_id"],
                ),
            )
            high_frequency = sorted(
                candidates,
                key=lambda group: (
                    -group["opportunity_count"],
                    -group["support_edge_count"],
                    group["group_id"],
                ),
            )
            large_family = sorted(
                candidates,
                key=lambda group: (
                    -group["size"],
                    -group["support_edge_count"],
                    group["group_id"],
                ),
            )
            strata = (
                harmful,
                rescue,
                uncertain,
                high_frequency,
                large_family,
            )
            take = max(per_band // len(strata), 1)
            by_id = {
                int(group["group_id"]): group
                for ranking in strata
                for group in ranking[:take]
            }
            remainder = [
                group for ranking in strata for group in ranking[take:]
            ]
        else:
            take = max(per_band // 2, 1)
            by_id = {
                int(group["group_id"]): group
                for group in harmful[:take] + rescue[:take]
            }
            remainder = harmful[take:] + rescue[take:]
        if len(by_id) < per_band:
            for group in remainder:
                by_id.setdefault(int(group["group_id"]), group)
                if len(by_id) >= per_band:
                    break
        selected.extend(by_id.values())
    return sorted(selected, key=lambda group: int(group["group_id"]))


def _run_pose(
    cached: dict,
    xyz: torch.Tensor,
    record: dict,
    positions: torch.Tensor,
    valid: torch.Tensor,
    *,
    seed: int,
    max_iterations: int,
    min_iterations: int,
):
    rows = torch.as_tensor(record["query_rows"]).long()[valid]
    indices = torch.as_tensor(record["top_indices"]).long()
    scores = torch.as_tensor(record["top_scores"]).float()
    chosen = indices[valid].gather(
        1, positions[valid, None].long()
    ).squeeze(1)
    chosen_scores = scores[valid].gather(
        1, positions[valid, None].long()
    ).squeeze(1)
    keypoints = (
        torch.as_tensor(cached["native_keypoints"])[rows].float()
        + float(cached.get("pixel_center_offset", 0.5))
    )
    start = time.perf_counter()
    pose, _, diagnostics = solve_pose(
        keypoints.numpy(),
        xyz[chosen].numpy(),
        torch.as_tensor(cached["native_K"]).numpy(),
        solver="poselib",
        reprojection_error=12.0,
        confidence=0.99999,
        max_iterations=max_iterations,
        min_iterations=min_iterations,
        scores=chosen_scores.numpy(),
        ransac_seed=seed,
        return_diagnostics=True,
    )
    return {
        "translation_error_m": _pose_error_m(
            pose, cached["pose_w2c"]
        ),
        "hypotheses": float(
            diagnostics.get("ransac_actual_hypotheses", float("nan"))
        ),
        "runtime_seconds": time.perf_counter() - start,
        "match_count": int(valid.sum()),
    }


def _state_metrics(errors: np.ndarray) -> dict:
    return {
        "median_te_m": float(np.median(errors)),
        "mean_te_m": float(errors.mean()),
        "p90_te_m": float(np.quantile(errors, 0.9)),
        "r5": float((errors <= 0.05).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-map", required=True)
    parser.add_argument("--function-graph", required=True)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--baselines", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups-per-band", type=int, default=256)
    parser.add_argument(
        "--screen-mode",
        choices=("legacy", "active_diverse"),
        default="legacy",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument("--min-iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-operation-rows",
        type=int,
        default=0,
        help=(
            "Limit a large community operation to its highest-value "
            "representative anchors; zero keeps the whole group."
        ),
    )
    parser.add_argument(
        "--counterfactual-seeds",
        nargs="*",
        type=int,
        default=[],
        help=(
            "Optional fixed seeds for paired baseline/proposal replay on "
            "affected queries."
        ),
    )
    args = parser.parse_args()
    state = torch.load(
        args.anchor_map, map_location="cpu", weights_only=False
    )
    graph = torch.load(
        args.function_graph, map_location="cpu", weights_only=False
    )
    group_payload = torch.load(
        args.groups, map_location="cpu", weights_only=False
    )
    baselines = torch.load(
        args.baselines, map_location="cpu", weights_only=False
    )
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    query_cache = cache_payload.get("queries", cache_payload)
    names = graph["query_names"]
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    count = int(xyz.shape[0])
    groups = _screen_groups(
        group_payload["groups"], args.groups_per_band, args.screen_mode
    )
    groups = [
        group
        for index, group in enumerate(groups)
        if index % args.num_shards == args.shard_index
    ]
    graph_records = {
        int(record["query_index"]): record
        for record in graph["records"]
    }
    baseline_states = {}
    for budget, state_payload in baselines["states"].items():
        baseline_states[int(budget)] = {
            "active": _active_mask(
                count,
                torch.as_tensor(state_payload["active_rows"]).long(),
            ),
            "records": {
                int(record["query_index"]): record
                for record in state_payload["records"]
            },
        }
    ordered_queries = sorted(graph_records)
    query_position = {
        query_index: position
        for position, query_index in enumerate(ordered_queries)
    }
    winner_queries = {
        budget: defaultdict(set) for budget in baseline_states
    }
    add_to_core_queries = defaultdict(set)
    for query_index in ordered_queries:
        record = graph_records[query_index]
        indices = torch.as_tensor(record["top_indices"]).long()
        for budget, baseline in baseline_states.items():
            base_record = baseline["records"][query_index]
            positions = torch.as_tensor(
                base_record["positions"]
            ).long()
            valid = torch.as_tensor(base_record["valid"]).bool()
            chosen = indices[valid].gather(
                1, positions[valid, None]
            ).squeeze(1)
            for anchor in torch.unique(chosen).tolist():
                winner_queries[budget][int(anchor)].add(query_index)
        core_record = baseline_states[30000]["records"][query_index]
        core_positions = torch.as_tensor(
            core_record["positions"]
        ).long()
        core_valid = torch.as_tensor(core_record["valid"]).bool()
        position_grid = torch.arange(indices.shape[1])[None]
        precedes_core = (
            (~core_valid[:, None])
            | (position_grid < core_positions[:, None])
        )
        candidate_rows = torch.unique(indices[precedes_core]).tolist()
        for anchor in candidate_rows:
            add_to_core_queries[int(anchor)].add(query_index)

    output_path = Path(args.output)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    influences = []
    completed_group_ids = set()
    if args.resume and partial_path.exists():
        partial = torch.load(
            partial_path, map_location="cpu", weights_only=False
        )
        influences = list(partial.get("influences", []))
        completed_group_ids = {
            int(item["group_id"]) for item in influences
        }
    upper_budget = {1: 35000, 2: 40000, 3: 45000}
    for completed, group in enumerate(groups, start=1):
        if int(group["group_id"]) in completed_group_ids:
            continue
        group_rows = torch.as_tensor(group["rows"]).long()
        operations = {}
        for operation, budget in (
            ("remove", upper_budget[int(group["band"])]),
            ("add_to_30k", 30000),
        ):
            operation_rows = group_rows
            if (
                int(args.max_operation_rows) > 0
                and group_rows.numel() > int(args.max_operation_rows)
            ):
                opportunities = torch.as_tensor(
                    graph[
                        "provenance_opportunity_count"
                        if "provenance_opportunity_count" in graph
                        else "candidate_opportunity_count"
                    ]
                )[group_rows].float().clamp_min(1)
                if operation == "remove":
                    values = torch.as_tensor(
                        graph[
                            "provenance_harmful_solver_inlier_count"
                            if "provenance_harmful_solver_inlier_count"
                            in graph
                            else "harmful_solver_inlier_count"
                        ]
                    )[group_rows].float() / opportunities
                else:
                    values = torch.as_tensor(
                        graph[
                            "provenance_solver_inlier_gtclean_4px_count"
                            if (
                                "provenance_solver_inlier_gtclean_4px_count"
                                in graph
                            )
                            else "solver_inlier_gtclean_4px_count"
                        ]
                    )[group_rows].float() / opportunities
                keep = torch.topk(
                    values,
                    k=int(args.max_operation_rows),
                ).indices
                operation_rows = group_rows[keep]
            baseline = baseline_states[budget]
            active = baseline["active"].clone()
            if operation == "remove":
                active[operation_rows] = False
            else:
                active[operation_rows] = True
            base_errors = np.asarray(
                [
                    baseline["records"][query_index][
                        "translation_error_m"
                    ]
                    for query_index in sorted(graph_records)
                ],
                dtype=np.float64,
            )
            new_errors = base_errors.copy()
            query_effects = []
            if operation == "remove":
                affected_queries = set()
                for row in operation_rows.tolist():
                    affected_queries.update(
                        winner_queries[budget].get(int(row), ())
                    )
            else:
                affected_queries = set()
                for row in operation_rows.tolist():
                    affected_queries.update(
                        add_to_core_queries.get(int(row), ())
                    )
            for query_index in sorted(affected_queries):
                position = query_position[query_index]
                record = graph_records[query_index]
                indices = torch.as_tensor(
                    record["top_indices"]
                ).long()
                new_positions, new_valid = _assignment_positions(
                    indices, active
                )
                base_record = baseline["records"][query_index]
                base_positions = torch.as_tensor(
                    base_record["positions"]
                ).to(torch.uint8)
                base_valid = torch.as_tensor(
                    base_record["valid"]
                ).bool()
                changed = (
                    (new_valid != base_valid)
                    | (
                        new_valid
                        & base_valid
                        & (new_positions != base_positions)
                    )
                )
                changed_count = int(changed.sum())
                if changed_count == 0:
                    continue
                seed_effects = []
                if args.counterfactual_seeds:
                    for fixed_seed in args.counterfactual_seeds:
                        paired_seed = int(fixed_seed) + query_index
                        baseline_result = _run_pose(
                            query_cache[names[query_index]],
                            xyz,
                            record,
                            base_positions,
                            base_valid,
                            seed=paired_seed,
                            max_iterations=args.max_iterations,
                            min_iterations=args.min_iterations,
                        )
                        proposal_result = _run_pose(
                            query_cache[names[query_index]],
                            xyz,
                            record,
                            new_positions,
                            new_valid,
                            seed=paired_seed,
                            max_iterations=args.max_iterations,
                            min_iterations=args.min_iterations,
                        )
                        seed_effects.append(
                            {
                                "seed": int(fixed_seed),
                                "baseline_te_m": baseline_result[
                                    "translation_error_m"
                                ],
                                "proposal_te_m": proposal_result[
                                    "translation_error_m"
                                ],
                                "delta_te_m": (
                                    proposal_result["translation_error_m"]
                                    - baseline_result[
                                        "translation_error_m"
                                    ]
                                ),
                                "baseline_hypotheses": baseline_result[
                                    "hypotheses"
                                ],
                                "proposal_hypotheses": proposal_result[
                                    "hypotheses"
                                ],
                                "baseline_runtime_seconds": (
                                    baseline_result["runtime_seconds"]
                                ),
                                "proposal_runtime_seconds": (
                                    proposal_result["runtime_seconds"]
                                ),
                            }
                        )
                    result = proposal_result
                else:
                    result = _run_pose(
                        query_cache[names[query_index]],
                        xyz,
                        record,
                        new_positions,
                        new_valid,
                        seed=int(args.seed) + query_index,
                        max_iterations=args.max_iterations,
                        min_iterations=args.min_iterations,
                    )
                new_errors[position] = result["translation_error_m"]
                query_effects.append(
                    {
                        "query_index": query_index,
                        "changed_assignment_count": changed_count,
                        "baseline_te_m": float(base_errors[position]),
                        "proposal_te_m": result["translation_error_m"],
                        "delta_te_m": float(
                            result["translation_error_m"]
                            - base_errors[position]
                        ),
                        "baseline_hypotheses": float(
                            base_record["hypotheses"]
                        ),
                        "proposal_hypotheses": result["hypotheses"],
                        "runtime_seconds": result["runtime_seconds"],
                        "paired_seed_effects": seed_effects,
                    }
                )
            baseline_metrics = _state_metrics(base_errors)
            proposal_metrics = _state_metrics(new_errors)
            paired_seed_summary = {}
            for fixed_seed in args.counterfactual_seeds:
                seed_rows = [
                    seed_effect
                    for effect in query_effects
                    for seed_effect in effect["paired_seed_effects"]
                    if int(seed_effect["seed"]) == int(fixed_seed)
                ]
                if not seed_rows:
                    continue
                paired_seed_summary[str(fixed_seed)] = {
                    "affected_query_count": len(seed_rows),
                    "delta_te_m_mean": float(
                        np.mean(
                            [row["delta_te_m"] for row in seed_rows]
                        )
                    ),
                    "delta_te_m_median": float(
                        np.median(
                            [row["delta_te_m"] for row in seed_rows]
                        )
                    ),
                    "improved_query_fraction": float(
                        np.mean(
                            [row["delta_te_m"] < 0 for row in seed_rows]
                        )
                    ),
                    "delta_hypotheses_mean": float(
                        np.mean(
                            [
                                row["proposal_hypotheses"]
                                - row["baseline_hypotheses"]
                                for row in seed_rows
                            ]
                        )
                    ),
                    "delta_runtime_seconds_mean": float(
                        np.mean(
                            [
                                row["proposal_runtime_seconds"]
                                - row["baseline_runtime_seconds"]
                                for row in seed_rows
                            ]
                        )
                    ),
                }
            operations[operation] = {
                "baseline_budget": budget,
                "operation_row_count": int(operation_rows.numel()),
                "affected_query_count": len(query_effects),
                "baseline_metrics": baseline_metrics,
                "proposal_metrics": proposal_metrics,
                "delta": {
                    key: proposal_metrics[key] - baseline_metrics[key]
                    for key in baseline_metrics
                },
                "improved_query_count": sum(
                    effect["delta_te_m"] < -1e-6
                    for effect in query_effects
                ),
                "regressed_query_count": sum(
                    effect["delta_te_m"] > 1e-6
                    for effect in query_effects
                ),
                "query_effects": query_effects,
                "paired_seed_summary": paired_seed_summary,
            }
        influences.append({**group, "operations": operations})
        if (
            int(args.checkpoint_every) > 0
            and len(influences) % int(args.checkpoint_every) == 0
        ):
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema": "lafgs_nested_band_lgo_partial",
                    "version": 1,
                    "influences": influences,
                    "config": vars(args),
                },
                partial_path,
            )
        if completed % 25 == 0 or completed == len(groups):
            print(
                f"LGO shard {args.shard_index}: {completed}/{len(groups)}",
                flush=True,
            )
    output = {
        "schema": "lafgs_nested_band_lgo_shard",
        "version": 1,
        "groups_screened_total": len(
            _screen_groups(
                group_payload["groups"],
                args.groups_per_band,
                args.screen_mode,
            )
        ),
        "influences": influences,
        "config": vars(args),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(
        json.dumps(
            {
                "evaluated_group_count": len(influences),
                "affected_remove_queries": sum(
                    item["operations"]["remove"][
                        "affected_query_count"
                    ]
                    for item in influences
                ),
                "affected_add_queries": sum(
                    item["operations"]["add_to_30k"][
                        "affected_query_count"
                    ]
                    for item in influences
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
