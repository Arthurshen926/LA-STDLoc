#!/usr/bin/env python3
"""Evaluate nested-map assignments used by leave-group-out screening."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.run_lafgs_alternating_structure import _pose_error_m
from utils.pose_utils import solve_pose


def _selected_rows(path: str) -> torch.Tensor:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return torch.as_tensor(
        state["functional_pruning"]["selected_source_rows"]
    ).long()


def _active_mask(count: int, rows: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(count, dtype=torch.bool)
    mask[rows] = True
    return mask


def _assignment_positions(
    indices: torch.Tensor, active: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    eligible = active[indices.long()]
    valid = eligible.any(dim=1)
    positions = eligible.to(torch.uint8).argmax(dim=1)
    positions[~valid] = 255
    return positions, valid


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
    elapsed = time.perf_counter() - start
    return {
        "translation_error_m": _pose_error_m(
            pose, cached["pose_w2c"]
        ),
        "hypotheses": float(
            diagnostics.get("ransac_actual_hypotheses", float("nan"))
        ),
        "runtime_seconds": elapsed,
        "match_count": int(valid.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-map", required=True)
    parser.add_argument("--function-graph", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--budget-maps", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument("--min-iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    state = torch.load(
        args.anchor_map, map_location="cpu", weights_only=False
    )
    graph = torch.load(
        args.function_graph, map_location="cpu", weights_only=False
    )
    payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    query_cache = payload.get("queries", payload)
    names = graph["query_names"]
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    count = int(xyz.shape[0])
    states = {}
    for path in args.budget_maps:
        budget = int(Path(path).stem.rsplit("_", 1)[-1])
        states[budget] = _active_mask(count, _selected_rows(path))
    if not set(states).issubset({30000, 35000, 40000, 45000}):
        raise ValueError("only 30K, 35K, 40K and 45K maps are supported")

    output_states = {}
    for budget in sorted(states):
        active = states[budget]
        records = []
        errors = []
        for completed, record in enumerate(graph["records"], start=1):
            query_index = int(record["query_index"])
            indices = torch.as_tensor(record["top_indices"]).long()
            positions, valid = _assignment_positions(indices, active)
            result = _run_pose(
                query_cache[names[query_index]],
                xyz,
                record,
                positions,
                valid,
                seed=int(args.seed) + query_index,
                max_iterations=args.max_iterations,
                min_iterations=args.min_iterations,
            )
            records.append(
                {
                    "query_index": query_index,
                    "positions": positions,
                    "valid": valid,
                    **result,
                }
            )
            errors.append(result["translation_error_m"])
            if completed % 50 == 0 or completed == len(graph["records"]):
                print(
                    f"LGO baseline {budget}: {completed}/"
                    f"{len(graph['records'])}",
                    flush=True,
                )
        errors_np = np.asarray(errors)
        output_states[str(budget)] = {
            "active_rows": torch.nonzero(
                active, as_tuple=False
            ).reshape(-1).to(torch.int32),
            "records": records,
            "metrics": {
                "median_te_m": float(np.median(errors_np)),
                "mean_te_m": float(errors_np.mean()),
                "p90_te_m": float(np.quantile(errors_np, 0.9)),
                "r5": float((errors_np <= 0.05).mean()),
            },
        }
    output = {
        "schema": "lafgs_lgo_baselines",
        "version": 1,
        "anchor_map": str(Path(args.anchor_map).resolve()),
        "function_graph": str(Path(args.function_graph).resolve()),
        "query_cache": str(Path(args.query_cache).resolve()),
        "states": output_states,
        "config": vars(args),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(
        json.dumps(
            {
                budget: value["metrics"]
                for budget, value in output_states.items()
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
