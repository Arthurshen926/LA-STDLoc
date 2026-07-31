#!/usr/bin/env python3
"""Fit a scene-specific SLPS risk atlas from exact current-policy outcomes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _records_by_name(payload: dict) -> dict[str, dict]:
    records = {str(record["query_name"]): record for record in payload["records"]}
    if len(records) != len(payload["records"]):
        raise ValueError("SLPS atlas top-K records contain duplicate names")
    return records


def _outcome_by_seed(subset: dict) -> dict[int, dict]:
    output = {}
    for outcome in subset["outcomes"]:
        seed = int(outcome["seed"])
        if seed in output:
            raise ValueError("SLPS atlas subset repeats one PoseLib seed")
        output[seed] = outcome
    return output


def _policy_metrics(
    chosen_budgets: np.ndarray,
    *,
    budgets: tuple[int, ...],
    baseline: list[dict],
    outcomes: list[dict[int, dict]],
) -> dict[str, float]:
    rows = []
    for query_index, budget in enumerate(chosen_budgets.tolist()):
        rows.append(
            baseline[query_index]
            if int(budget) == 0
            else outcomes[query_index][int(budget)]
        )
    te = np.asarray([float(row["te_cm"]) for row in rows])
    re = np.asarray([float(row["re_deg"]) for row in rows])
    hypotheses = np.asarray(
        [float(row.get("hypotheses") or 100000) for row in rows]
    )
    return {
        "median_te_cm": float(np.median(te)),
        "mean_te_cm": float(np.mean(te)),
        "p90_te_cm": float(np.quantile(te, 0.9)),
        "recall_5cm_percent": float(100.0 * np.mean((te <= 5.0) & (re <= 5.0))),
        "catastrophic_count": int(
            sum(bool(row["catastrophic"]) for row in rows)
        ),
        "mean_hypotheses": float(np.mean(hypotheses)),
        "compact_query_fraction": float(np.mean(chosen_budgets > 0)),
        "selected_budget_mean": float(
            np.mean(
                [
                    int(budget) if int(budget) > 0 else 0
                    for budget in chosen_budgets
                ]
            )
        ),
    }


def _joint_score(candidate: dict, baseline: dict) -> float:
    score = 0.0
    for key, weight in (
        ("median_te_cm", 1.0),
        ("mean_te_cm", 0.35),
        ("p90_te_cm", 0.50),
    ):
        scale = max(float(baseline[key]), 1.0)
        score += weight * (
            float(baseline[key]) - float(candidate[key])
        ) / scale
    score += 2.0 * (
        float(candidate["recall_5cm_percent"])
        - float(baseline["recall_5cm_percent"])
    ) / 100.0
    score += 0.05 * np.log(
        max(float(baseline["mean_hypotheses"]), 1.0)
        / max(float(candidate["mean_hypotheses"]), 1.0)
    )
    return float(score)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--topk-outcomes", required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budgets", default="256,384,512,768")
    parser.add_argument("--neighbor-count", type=int, default=8)
    parser.add_argument("--similarity-power", type=float, default=8.0)
    parser.add_argument(
        "--context-grid-size",
        type=int,
        default=2,
        help=(
            "Image-grid width/height used for joint (anchor, cell) "
            "self-localization evidence; zero disables view context."
        ),
    )
    parser.add_argument(
        "--context-weight",
        type=float,
        default=0.5,
        help="Weight of joint (anchor, cell) similarity versus anchor overlap.",
    )
    parser.add_argument(
        "--set-query-context-weight",
        type=float,
        default=0.25,
        help=(
            "Weight of full-query context in each budget-conditioned "
            "selected-set similarity."
        ),
    )
    parser.add_argument("--calibration-seed", type=int, default=2026)
    args = parser.parse_args()

    corpus_path = Path(args.corpus).resolve()
    topk_path = Path(args.topk_outcomes).resolve()
    selector_path = Path(args.selector).resolve()
    output = Path(args.output).resolve()
    corpus = torch.load(corpus_path, map_location="cpu", weights_only=False)
    topk = torch.load(topk_path, map_location="cpu", weights_only=False)
    selector = torch.load(selector_path, map_location="cpu", weights_only=False)
    if (
        corpus.get("schema") != "lafgs_slps_set_outcomes"
        or topk.get("schema") != "lafgs_exact_topk_outcomes"
        or selector.get("schema") != "lafgs_slps_selector"
    ):
        raise ValueError("unsupported SLPS atlas input")
    if (
        dict(corpus["candidate_graph_contract"])
        != dict(topk["provenance"])
        or dict(selector["candidate_graph_contract"])
        != dict(topk["provenance"])
    ):
        raise ValueError("SLPS atlas candidate graph contract differs")
    if (
        corpus["anchor_ids_sha256"] != topk["anchor_ids_sha256"]
        or selector["anchor_ids_sha256"] != topk["anchor_ids_sha256"]
    ):
        raise ValueError("SLPS atlas anchor identity differs")

    budgets = tuple(
        sorted(
            {
                int(value)
                for value in str(args.budgets).split(",")
                if value.strip()
            }
        )
    )
    context_grid_size = max(int(args.context_grid_size), 0)
    context_weight = float(args.context_weight)
    set_query_context_weight = float(args.set_query_context_weight)
    if not 0.0 <= context_weight <= 1.0:
        raise ValueError("--context-weight must be in [0, 1]")
    if not 0.0 <= set_query_context_weight <= 1.0:
        raise ValueError("--set-query-context-weight must be in [0, 1]")
    if context_grid_size == 0 and context_weight > 0.0:
        raise ValueError(
            "--context-weight must be zero when --context-grid-size is zero"
        )
    selector_sha256 = _sha256_file(selector_path)
    corpus_policy_matches = (
        corpus.get("self_mining", {}).get("selector_sha256")
        == selector_sha256
    )
    records = _records_by_name(topk)
    query_names = [str(query["query_name"]) for query in corpus["queries"]]
    support = torch.zeros(
        (len(query_names), int(topk["anchor_count"])), dtype=torch.bool
    )
    context_cell_count = context_grid_size * context_grid_size
    support_context = (
        torch.zeros(
            (
                len(query_names),
                int(topk["anchor_count"]) * context_cell_count,
            ),
            dtype=torch.bool,
        )
        if context_cell_count
        else None
    )
    support_set = torch.zeros(
        (
            len(query_names),
            len(budgets),
            int(topk["anchor_count"]),
        ),
        dtype=torch.bool,
    )
    support_set_context = (
        torch.zeros(
            (
                len(query_names),
                len(budgets),
                int(topk["anchor_count"]) * context_cell_count,
            ),
            dtype=torch.bool,
        )
        if context_cell_count
        else None
    )
    try:
        keypoint_x_index = list(corpus["feature_names"]).index("keypoint_x")
        keypoint_y_index = list(corpus["feature_names"]).index("keypoint_y")
    except ValueError as error:
        raise ValueError(
            "SLPS atlas corpus lacks normalized keypoint coordinates"
        ) from error
    safe_targets = torch.zeros((len(query_names), len(budgets)))
    catastrophic_targets = torch.zeros_like(safe_targets)
    utility_targets = torch.zeros_like(safe_targets)
    primary_baseline = []
    primary_outcomes: list[dict[int, dict]] = []

    for query_index, query in enumerate(corpus["queries"]):
        name = str(query["query_name"])
        record = records[name]
        top1 = torch.as_tensor(
            record["topk_anchor_indices"]
        ).long()[:, 0]
        anchors = torch.unique(top1)
        support[query_index, anchors] = True
        features = torch.as_tensor(query["features"]).float()
        if len(features) != len(top1):
            raise ValueError(
                f"{name} feature rows differ from its top-1 graph"
            )
        cells = None
        if support_context is not None:
            x = (
                features[:, keypoint_x_index] * context_grid_size
            ).floor().long().clamp(0, context_grid_size - 1)
            y = (
                features[:, keypoint_y_index] * context_grid_size
            ).floor().long().clamp(0, context_grid_size - 1)
            cells = x + context_grid_size * y
            tokens = torch.unique(top1 * context_cell_count + cells)
            support_context[query_index, tokens] = True
        all_subset = next(
            subset for subset in query["subsets"] if subset["name"] == "all"
        )
        all_by_seed = _outcome_by_seed(all_subset)
        if int(args.calibration_seed) not in all_by_seed:
            raise ValueError(f"{name} misses the calibration all-set seed")
        primary_baseline.append(all_by_seed[int(args.calibration_seed)])
        query_primary = {}
        for column, budget in enumerate(budgets):
            expected_name = f"learned_nested_{budget}"
            candidates = [
                subset
                for subset in query["subsets"]
                if subset["name"] == expected_name
                and (
                    (
                        bool(subset.get("deployment_calibration", False))
                        and subset.get("self_mining_selector_sha256")
                        == selector_sha256
                    )
                    or (
                        corpus_policy_matches
                        and "deployment_calibration" not in subset
                    )
                )
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"{name} has {len(candidates)} exact current-policy "
                    f"sets for budget {budget}"
                )
            subset = candidates[0]
            subset_indices = torch.as_tensor(subset["indices"]).long()
            if (
                not len(subset_indices)
                or int(subset_indices.min()) < 0
                or int(subset_indices.max()) >= len(top1)
            ):
                raise ValueError(
                    f"{name} budget {budget} has invalid selected rows"
                )
            set_anchors = torch.unique(top1[subset_indices])
            support_set[query_index, column, set_anchors] = True
            if support_set_context is not None:
                set_tokens = torch.unique(
                    top1[subset_indices] * context_cell_count
                    + cells[subset_indices]
                )
                support_set_context[
                    query_index, column, set_tokens
                ] = True
            subset_by_seed = _outcome_by_seed(subset)
            common_seeds = sorted(set(all_by_seed) & set(subset_by_seed))
            if not common_seeds:
                raise ValueError(f"{name} budget {budget} has no paired seed")
            safe_targets[query_index, column] = float(
                all(
                    bool(subset_by_seed[seed]["safe_relative_all"])
                    and not bool(subset_by_seed[seed]["catastrophic"])
                    for seed in common_seeds
                )
            )
            catastrophic_targets[query_index, column] = float(
                any(
                    bool(subset_by_seed[seed]["catastrophic"])
                    for seed in common_seeds
                )
            )
            utility_targets[query_index, column] = float(
                np.mean(
                    [
                        float(subset_by_seed[seed]["target_utility"])
                        - float(all_by_seed[seed]["target_utility"])
                        for seed in common_seeds
                    ]
                )
            )
            if int(args.calibration_seed) not in subset_by_seed:
                raise ValueError(
                    f"{name} budget {budget} misses calibration seed"
                )
            query_primary[int(budget)] = subset_by_seed[
                int(args.calibration_seed)
            ]
        primary_outcomes.append(query_primary)

    def cosine_incidence(matrix: torch.Tensor) -> torch.Tensor:
        counts = matrix.sum(dim=1).float().clamp_min(1.0)
        overlap = matrix.float() @ matrix.float().T
        return overlap / torch.sqrt(counts[:, None] * counts[None, :])

    query_similarity = cosine_incidence(support)
    if support_context is not None and context_weight > 0.0:
        query_similarity = (
            (1.0 - context_weight) * query_similarity
            + context_weight * cosine_incidence(support_context)
        )
    budget_similarity = []
    for column in range(len(budgets)):
        selected_similarity = cosine_incidence(support_set[:, column])
        if support_set_context is not None and context_weight > 0.0:
            selected_similarity = (
                (1.0 - context_weight) * selected_similarity
                + context_weight
                * cosine_incidence(support_set_context[:, column])
            )
        selected_similarity = (
            (1.0 - set_query_context_weight) * selected_similarity
            + set_query_context_weight * query_similarity
        )
        selected_similarity.fill_diagonal_(float("-inf"))
        budget_similarity.append(selected_similarity)
    similarity = torch.stack(budget_similarity, dim=1)
    neighbor_count = min(
        max(int(args.neighbor_count), 1), max(len(query_names) - 1, 1)
    )
    values, indices = torch.topk(
        similarity, k=neighbor_count, dim=2, largest=True, sorted=True
    )
    values = values.clamp_min(0.0)
    maximum = values.max(dim=2, keepdim=True).values.clamp_min(1e-8)
    weights = (values / maximum).pow(max(float(args.similarity_power), 0.0))
    weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-8)
    predicted_safe = torch.zeros_like(safe_targets)
    predicted_catastrophic = torch.zeros_like(catastrophic_targets)
    predicted_utility = torch.zeros_like(utility_targets)
    for column in range(len(budgets)):
        neighbors = indices[:, column]
        predicted_safe[:, column] = (
            weights[:, column] * safe_targets[:, column][neighbors]
        ).sum(dim=1)
        predicted_catastrophic[:, column] = (
            weights[:, column]
            * catastrophic_targets[:, column][neighbors]
        ).sum(dim=1)
        predicted_utility[:, column] = (
            weights[:, column] * utility_targets[:, column][neighbors]
        ).sum(dim=1)
    maximum_similarity = values[:, :, 0]

    all_choice = np.zeros(len(query_names), dtype=np.int64)
    baseline_metrics = _policy_metrics(
        all_choice,
        budgets=budgets,
        baseline=primary_baseline,
        outcomes=primary_outcomes,
    )
    best = None
    safe_thresholds = np.linspace(0.60, 1.00, 17)
    catastrophic_thresholds = (0.05, 0.10, 0.15, 0.20, 0.30)
    utility_thresholds = np.linspace(-0.20, 0.30, 11)
    similarity_thresholds = sorted(
        {
            0.0,
            *[
                float(value)
                for value in torch.quantile(
                    maximum_similarity,
                    torch.tensor([0.10, 0.25, 0.50]),
                )
            ],
        }
    )
    for safe_threshold in safe_thresholds:
        for catastrophic_threshold in catastrophic_thresholds:
            for utility_threshold in utility_thresholds:
                for similarity_threshold in similarity_thresholds:
                    accepted = (
                        (predicted_safe >= float(safe_threshold))
                        & (
                            predicted_catastrophic
                            <= float(catastrophic_threshold)
                        )
                        & (predicted_utility >= float(utility_threshold))
                        & (
                            maximum_similarity
                            >= float(similarity_threshold)
                        )
                    )
                    choice = np.zeros(len(query_names), dtype=np.int64)
                    for query_index in range(len(query_names)):
                        columns = torch.where(accepted[query_index])[0]
                        if not len(columns):
                            continue
                        utilities = predicted_utility[
                            query_index, columns
                        ]
                        best_column = columns[
                            int(torch.argmax(utilities))
                        ].item()
                        choice[query_index] = budgets[int(best_column)]
                    metrics = _policy_metrics(
                        choice,
                        budgets=budgets,
                        baseline=primary_baseline,
                        outcomes=primary_outcomes,
                    )
                    feasible = (
                        metrics["catastrophic_count"]
                        <= baseline_metrics["catastrophic_count"]
                        and metrics["recall_5cm_percent"] + 1e-8
                        >= baseline_metrics["recall_5cm_percent"]
                    )
                    score = _joint_score(metrics, baseline_metrics)
                    candidate = {
                        "score": score,
                        "feasible": feasible,
                        "atlas_safe_probability_threshold": float(
                            safe_threshold
                        ),
                        "atlas_catastrophic_probability_threshold": float(
                            catastrophic_threshold
                        ),
                        "atlas_relative_utility_threshold": float(
                            utility_threshold
                        ),
                        "atlas_minimum_similarity": float(
                            similarity_threshold
                        ),
                        "metrics": metrics,
                    }
                    key = (
                        int(feasible),
                        score if feasible else -float(
                            metrics["catastrophic_count"]
                        ),
                        metrics["compact_query_fraction"],
                    )
                    if best is None or key > best[0]:
                        best = (key, candidate)
    assert best is not None
    calibration = {
        **best[1],
        "baseline_metrics": baseline_metrics,
        "neighbor_count": neighbor_count,
        "similarity_power": float(args.similarity_power),
        "context_grid_size": context_grid_size,
        "context_weight": context_weight,
        "set_query_context_weight": set_query_context_weight,
        "maximum_similarity_quantiles": {
            str(quantile): float(
                torch.quantile(maximum_similarity.reshape(-1), quantile)
            )
            for quantile in (0.0, 0.1, 0.5, 0.9, 1.0)
        },
    }
    if not calibration["feasible"] or calibration["score"] <= 0.0:
        raise RuntimeError(
            "SLPS outcome atlas has no positive jointly feasible LOO policy"
        )

    atlas = {
        "schema": "lafgs_slps_outcome_atlas",
        "version": 1,
        "support_query_names": query_names,
        "support_anchor_mask": support,
        "support_context_mask": support_context,
        "support_set_anchor_mask": support_set,
        "support_set_context_mask": support_set_context,
        "context_grid_size": context_grid_size,
        "context_weight": context_weight,
        "set_query_context_weight": set_query_context_weight,
        "budgets": list(budgets),
        "safe_probability_targets": safe_targets,
        "catastrophic_probability_targets": catastrophic_targets,
        "relative_utility_targets": utility_targets,
        "neighbor_count": neighbor_count,
        "similarity_power": float(args.similarity_power),
        "calibration": calibration,
        "source": {
            "corpus": str(corpus_path),
            "corpus_sha256": _sha256_file(corpus_path),
            "topk_outcomes": str(topk_path),
            "topk_outcomes_sha256": _sha256_file(topk_path),
            "selector": str(selector_path),
            "selector_sha256": selector_sha256,
        },
    }
    payload = copy.deepcopy(selector)
    payload["version"] = max(int(payload.get("version", 1)), 2)
    payload["outcome_atlas"] = atlas
    payload["selector_config"] = {
        **dict(payload["selector_config"]),
        "risk_gate_mode": "atlas",
        "atlas_safe_probability_threshold": calibration[
            "atlas_safe_probability_threshold"
        ],
        "atlas_catastrophic_probability_threshold": calibration[
            "atlas_catastrophic_probability_threshold"
        ],
        "atlas_relative_utility_threshold": calibration[
            "atlas_relative_utility_threshold"
        ],
        "atlas_minimum_similarity": calibration[
            "atlas_minimum_similarity"
        ],
        "execution_device": "cpu",
    }
    payload["summary"] = {
        **dict(payload["summary"]),
        "outcome_atlas_calibration": calibration,
    }
    _atomic_torch(output, payload)
    summary = {
        "output": str(output),
        "support_query_count": len(query_names),
        "anchor_count": int(topk["anchor_count"]),
        "context_grid_size": context_grid_size,
        "context_weight": context_weight,
        "set_query_context_weight": set_query_context_weight,
        "calibration": calibration,
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
