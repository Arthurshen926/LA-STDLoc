#!/usr/bin/env python3
"""Materialize bottom-up rescue and top-down LGO operating points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


PROFILES = {
    "balanced": {
        "median": 300.0,
        "mean": 1000.0,
        "p90": 200.0,
        "r5": 80.0,
        "regression": 0.02,
    },
    "tail": {
        "median": 100.0,
        "mean": 500.0,
        "p90": 800.0,
        "r5": 80.0,
        "regression": 0.01,
    },
    "precision": {
        "median": 300.0,
        "mean": 500.0,
        "p90": 100.0,
        "r5": 300.0,
        "regression": 0.05,
    },
}


def _selected_rows(state: dict) -> torch.Tensor:
    return torch.as_tensor(
        state["functional_pruning"]["selected_source_rows"]
    ).long()


def _utility(operation: dict, weights: dict) -> float:
    delta = operation["delta"]
    return (
        -weights["median"] * float(delta["median_te_m"])
        - weights["mean"] * float(delta["mean_te_m"])
        - weights["p90"] * float(delta["p90_te_m"])
        + weights["r5"] * float(delta["r5"])
        - weights["regression"]
        * max(
            int(operation["regressed_query_count"])
            - int(operation["improved_query_count"]),
            0,
        )
    )


def _materialize(
    state: dict, rows: torch.Tensor, metadata: dict
) -> dict:
    count = int(state["anchor_xyz"].shape[0])
    rows = torch.sort(torch.unique(rows.long())).values
    output = {}
    for key, value in state.items():
        if torch.is_tensor(value) and value.ndim and value.shape[0] == count:
            output[key] = value[rows]
        else:
            output[key] = value
    output["anchor_ids"] = torch.arange(rows.numel(), dtype=torch.long)
    output["canonical_anchor_count"] = int(rows.numel())
    anchor_type = torch.as_tensor(output["anchor_type"])
    output["base_anchor_count"] = int((anchor_type == 0).sum())
    output["micro_anchor_count"] = int((anchor_type != 0).sum())
    output["requested_micro_anchor_budget"] = output[
        "micro_anchor_count"
    ]
    output["functional_pruning"] = {
        **metadata,
        "selected_source_rows": rows,
    }
    return output


def _select_groups(
    ranked_groups: list[dict],
    target_delta: int,
) -> tuple[list[dict], int]:
    ranked = []
    for influence in ranked_groups:
        utility = float(influence["predicted_utility"])
        size = int(influence["size"])
        ranked.append(
            (
                utility / max(np.sqrt(size), 1.0),
                utility,
                -size,
                -int(influence["group_id"]),
                influence,
            )
        )
    ranked.sort(reverse=True, key=lambda item: item[:4])
    selected = []
    used = 0
    for _, utility, _, _, influence in ranked:
        size = int(influence["size"])
        if used + size > target_delta:
            continue
        selected.append(
            {
                "group_id": int(influence["group_id"]),
                "band": int(influence["band"]),
                "size": size,
                "utility": float(utility),
                "rows": torch.as_tensor(influence["rows"]).long(),
            }
        )
        used += size
        if used >= target_delta:
            break
    return selected, used


def _group_features(groups: list[dict]) -> np.ndarray:
    rows = []
    for group in groups:
        opportunity = max(int(group["opportunity_count"]), 1)
        harmful = int(group["harmful_consensus_count"])
        clean = int(group["gtclean_inlier_count"])
        band = int(group["band"])
        rows.append(
            [
                1.0,
                np.log1p(int(group["size"])),
                np.log1p(int(group["support_edge_count"])),
                np.log1p(harmful),
                np.log1p(clean),
                harmful / opportunity,
                clean / opportunity,
                *(1.0 if band == value else 0.0 for value in (1, 2, 3, 4)),
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def _fit_lgo_surrogate(
    all_groups: list[dict],
    influences: list[dict],
    operation_name: str,
    profile: str,
    ridge: float,
) -> tuple[list[dict], dict]:
    by_id = {
        int(group["group_id"]): group for group in all_groups
    }
    training_groups = [
        by_id[int(influence["group_id"])] for influence in influences
    ]
    x = _group_features(training_groups)
    y = np.asarray(
        [
            _utility(
                influence["operations"][operation_name],
                PROFILES[profile],
            )
            for influence in influences
        ],
        dtype=np.float64,
    )
    mean = x[:, 1:].mean(axis=0)
    scale = x[:, 1:].std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = x.copy()
    normalized[:, 1:] = (normalized[:, 1:] - mean) / scale
    regularizer = np.eye(normalized.shape[1]) * float(ridge)
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(
        normalized.T @ normalized + regularizer,
        normalized.T @ y,
    )
    all_x = _group_features(all_groups)
    all_x[:, 1:] = (all_x[:, 1:] - mean) / scale
    prediction = all_x @ coefficients
    fitted = normalized @ coefficients
    variance = float(np.var(y))
    r2 = (
        1.0 - float(np.mean((fitted - y) ** 2)) / variance
        if variance > 1e-12
        else 0.0
    )
    ranked = [
        {
            **group,
            "predicted_utility": float(prediction[index]),
        }
        for index, group in enumerate(all_groups)
    ]
    diagnostics = {
        "operation": operation_name,
        "profile": profile,
        "training_group_count": len(training_groups),
        "target_mean": float(y.mean()),
        "target_std": float(y.std()),
        "training_r2": r2,
        "coefficients": coefficients.tolist(),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
    }
    return ranked, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-map", required=True)
    parser.add_argument("--core-30k-map", required=True)
    parser.add_argument("--upper-40k-map", required=True)
    parser.add_argument("--lgo-influence", required=True)
    parser.add_argument("--all-groups", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--surrogate-ridge", type=float, default=1.0)
    parser.add_argument(
        "--bottom-up-budgets",
        nargs="+",
        type=int,
        default=[32000, 35000, 37500, 40000],
    )
    parser.add_argument(
        "--top-down-budgets",
        nargs="+",
        type=int,
        default=[37500, 35000],
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=sorted(PROFILES),
        default=sorted(PROFILES),
    )
    args = parser.parse_args()
    state = torch.load(
        args.anchor_map, map_location="cpu", weights_only=False
    )
    core = torch.load(
        args.core_30k_map, map_location="cpu", weights_only=False
    )
    upper = torch.load(
        args.upper_40k_map, map_location="cpu", weights_only=False
    )
    lgo = torch.load(
        args.lgo_influence, map_location="cpu", weights_only=False
    )
    influences = lgo["influences"]
    group_payload = torch.load(
        args.all_groups, map_location="cpu", weights_only=False
    )
    all_groups = group_payload["groups"]
    core_rows = _selected_rows(core)
    upper_rows = _selected_rows(upper)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "lafgs_lgo_candidate_maps",
        "version": 1,
        "profiles": {},
        "surrogates": {},
    }
    for profile in args.profiles:
        add_ranked, add_diagnostics = _fit_lgo_surrogate(
            all_groups,
            influences,
            "add_to_30k",
            profile,
            args.surrogate_ridge,
        )
        remove_ranked, remove_diagnostics = _fit_lgo_surrogate(
            all_groups,
            influences,
            "remove",
            profile,
            args.surrogate_ridge,
        )
        report["surrogates"][profile] = {
            "add_to_30k": add_diagnostics,
            "remove": remove_diagnostics,
        }
        profile_dir = output_dir / profile
        profile_dir.mkdir(parents=True, exist_ok=True)
        entries = {}
        for budget in args.bottom_up_budgets:
            target_delta = max(int(budget) - int(core_rows.numel()), 0)
            selected, used = _select_groups(
                [
                    group
                    for group in add_ranked
                    if int(group["band"]) in (1, 2, 3, 4)
                ],
                target_delta,
            )
            added = torch.cat(
                [item["rows"] for item in selected],
                dim=0,
            ) if selected else torch.empty(0, dtype=torch.long)
            rows = torch.unique(torch.cat((core_rows, added)))
            label = (
                f"bottomup_{profile}_req{budget:05d}_"
                f"actual{int(rows.numel()):05d}"
            )
            metadata = {
                "schema": "lafgs_lgo_pareto_map",
                "version": 1,
                "path": "bottom_up_rescue",
                "profile": profile,
                "requested_budget": budget,
                "selected_group_count": len(selected),
                "selected_group_ids": [
                    item["group_id"] for item in selected
                ],
                "predicted_group_utility_sum": float(
                    sum(item["utility"] for item in selected)
                ),
            }
            path = profile_dir / f"{label}.pt"
            torch.save(_materialize(state, rows, metadata), path)
            entries[label] = {
                **metadata,
                "actual_anchor_count": int(rows.numel()),
                "path": str(path),
            }
        for budget in args.top_down_budgets:
            target_delta = max(int(upper_rows.numel()) - int(budget), 0)
            eligible = [
                influence
                for influence in remove_ranked
                if int(influence["band"]) in (1, 2)
            ]
            selected, used = _select_groups(
                eligible, target_delta
            )
            removed = torch.cat(
                [item["rows"] for item in selected],
                dim=0,
            ) if selected else torch.empty(0, dtype=torch.long)
            keep = torch.ones(
                state["anchor_xyz"].shape[0], dtype=torch.bool
            )
            keep[removed] = False
            rows = upper_rows[keep[upper_rows]]
            label = (
                f"topdown_{profile}_req{budget:05d}_"
                f"actual{int(rows.numel()):05d}"
            )
            metadata = {
                "schema": "lafgs_lgo_pareto_map",
                "version": 1,
                "path": "top_down_retirement",
                "profile": profile,
                "requested_budget": budget,
                "selected_group_count": len(selected),
                "selected_group_ids": [
                    item["group_id"] for item in selected
                ],
                "predicted_group_utility_sum": float(
                    sum(item["utility"] for item in selected)
                ),
            }
            path = profile_dir / f"{label}.pt"
            torch.save(_materialize(state, rows, metadata), path)
            entries[label] = {
                **metadata,
                "actual_anchor_count": int(rows.numel()),
                "path": str(path),
            }
        report["profiles"][profile] = entries
    (output_dir / "lgo_candidate_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                profile: {
                    label: entry["actual_anchor_count"]
                    for label, entry in entries.items()
                }
                for profile, entries in report["profiles"].items()
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
