#!/usr/bin/env python3
"""Train OOF quality heads and basis-aware core-reserve correspondence sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from localization_training.pose_sufficient_selector import (
    basis_aware_core_reserve_mask,
    image_grid_cells,
)
from scripts.train_lafgs_pose_sufficient_selector import (
    _anchor_statistics,
    _atomic_torch,
    _features_for_query,
    _model_payload,
    _records_by_name,
    _sha256_tensor,
    _trajectory,
)


HEAD_NAMES = ("strict_clean", "solver_clean", "harmful")


def _labels(dynamic_record: dict) -> dict[str, torch.Tensor]:
    return {
        "strict_clean": (
            torch.as_tensor(
                dynamic_record["gt_reprojection_errors_px"]
            ).float()
            <= 2.0
        ),
        "solver_clean": torch.as_tensor(
            dynamic_record["clean_inlier_mask"]
        ).bool(),
        "harmful": torch.as_tensor(
            dynamic_record["harmful_inlier_mask"]
        ).bool(),
    }


def _fit_probability_model(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    maximum_fit_rows: int,
    seed: int,
) -> tuple[LogisticRegression, np.ndarray, np.ndarray]:
    """Fit an unweighted logistic head while preserving the true prevalence."""

    rng = np.random.default_rng(int(seed))
    maximum = min(max(int(maximum_fit_rows), 1000), len(labels))
    chosen = (
        np.arange(len(labels))
        if maximum == len(labels)
        else rng.choice(len(labels), maximum, replace=False)
    )
    mean = features[chosen].mean(axis=0)
    scale = features[chosen].std(axis=0).clip(1e-6)
    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        max_iter=200,
        random_state=int(seed),
        solver="lbfgs",
    )
    model.fit((features[chosen] - mean) / scale, labels[chosen])
    return model, mean, scale


def _predict(model, mean, scale, features) -> np.ndarray:
    return model.predict_proba((features - mean) / scale)[:, 1]


def _expected_calibration_error(
    labels: np.ndarray,
    probability: np.ndarray,
    *,
    bins: int = 15,
) -> float:
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    total = max(len(labels), 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probability >= lower) & (
            probability < upper if upper < 1.0 else probability <= upper
        )
        if mask.any():
            value += (
                mask.sum()
                / total
                * abs(float(probability[mask].mean() - labels[mask].mean()))
            )
    return float(value)


def _quality_diagnostics(
    labels: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    return {
        "positive_rate": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, probability)),
        "average_precision": float(
            average_precision_score(labels, probability)
        ),
        "brier": float(brier_score_loss(labels, probability)),
        "ece15": _expected_calibration_error(labels, probability),
        "probability_mean": float(probability.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--topk-outcomes", required=True)
    parser.add_argument("--dynamic-outcomes", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--maximum-fit-rows", type=int, default=500000)
    parser.add_argument("--core-budget", type=int, default=384)
    parser.add_argument("--minimum-budget", type=int, default=512)
    parser.add_argument("--maximum-budget", type=int, default=768)
    parser.add_argument("--minimum-strict-lcb", type=float, default=80.0)
    parser.add_argument("--minimum-dependency-groups", type=int, default=96)
    parser.add_argument("--minimum-image-cells", type=int, default=16)
    parser.add_argument("--minimum-log-expected-basis", type=float, default=11.0)
    parser.add_argument("--representative-count", type=int, default=64)
    parser.add_argument("--pair-count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    state = torch.load(args.map, map_location="cpu", weights_only=False)
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    topk = torch.load(
        args.topk_outcomes, map_location="cpu", weights_only=False
    )
    dynamic = torch.load(
        args.dynamic_outcomes, map_location="cpu", weights_only=False
    )
    if topk.get("schema") != "lafgs_exact_topk_outcomes":
        raise ValueError("unsupported top-K outcomes")
    if dynamic.get("schema") != "lafgs_dynamic_self_localization_outcomes":
        raise ValueError("unsupported dynamic outcomes")
    if list(topk["query_names"]) != list(dynamic["query_names"]):
        raise ValueError("top-K and dynamic query ordering differ")
    anchor_ids = torch.as_tensor(state["anchor_ids"]).long()
    if (
        int(topk["anchor_count"]) != len(anchor_ids)
        or topk["anchor_ids_sha256"] != _sha256_tensor(anchor_ids)
    ):
        raise ValueError("top-K outcomes do not align with map")

    topk_by_name = _records_by_name(topk)
    dynamic_by_name = _records_by_name(dynamic)
    labels_by_name = {
        name: _labels(dynamic_by_name[name])
        for name in topk["query_names"]
    }
    trajectories = sorted({_trajectory(name) for name in topk["query_names"]})
    trajectory_lookup = {
        trajectory: index for index, trajectory in enumerate(trajectories)
    }
    trajectory_indices = torch.as_tensor(
        [trajectory_lookup[_trajectory(name)] for name in topk["query_names"]]
    ).long()
    source = torch.as_tensor(state["source_primitive_ids"]).long()
    dependency = torch.as_tensor(
        state.get("coarse_dependency_group_ids", state["dependency_group_ids"])
    ).long()
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    statistics = _anchor_statistics(
        [dynamic_by_name[name] for name in topk["query_names"]],
        trajectory_indices,
        trajectory_count=len(trajectories),
        anchor_count=len(anchor_ids),
    )

    probabilities_by_name = {
        name: {} for name in topk["query_names"]
    }
    fold_models = {}
    fold_diagnostics = {}
    for fold_index, trajectory in enumerate(trajectories):
        folded_statistics = {
            name: values.sum(dim=0) - values[fold_index]
            for name, values in statistics.items()
        }
        features_by_name = {}
        for name in topk["query_names"]:
            features, _ = _features_for_query(
                topk_by_name[name],
                dynamic_by_name[name],
                cache[name],
                source_groups=source,
                dependency_groups=dependency,
                anchor_statistics=statistics,
                excluded_trajectory_index=fold_index,
                folded_statistics=folded_statistics,
                positive_target="gt_clean",
            )
            features_by_name[name] = features
        train_names = [
            name
            for name in topk["query_names"]
            if _trajectory(name) != trajectory
        ]
        heldout_names = [
            name
            for name in topk["query_names"]
            if _trajectory(name) == trajectory
        ]
        train_features = torch.cat(
            [features_by_name[name] for name in train_names]
        ).numpy()
        heldout_features = torch.cat(
            [features_by_name[name] for name in heldout_names]
        ).numpy()
        fold_models[trajectory] = {}
        fold_diagnostics[trajectory] = {}
        for head_index, head_name in enumerate(HEAD_NAMES):
            train_labels = torch.cat(
                [labels_by_name[name][head_name] for name in train_names]
            ).numpy()
            heldout_labels = torch.cat(
                [labels_by_name[name][head_name] for name in heldout_names]
            ).numpy()
            model, mean, scale = _fit_probability_model(
                train_features,
                train_labels,
                maximum_fit_rows=int(args.maximum_fit_rows),
                seed=int(args.seed) + fold_index * 10 + head_index,
            )
            probability = _predict(
                model, mean, scale, heldout_features
            )
            cursor = 0
            for name in heldout_names:
                count = len(features_by_name[name])
                probabilities_by_name[name][head_name] = torch.from_numpy(
                    probability[cursor : cursor + count]
                ).float()
                cursor += count
            fold_models[trajectory][head_name] = _model_payload(
                model, mean, scale
            )
            fold_diagnostics[trajectory][head_name] = (
                _quality_diagnostics(heldout_labels, probability)
            )
        print(
            json.dumps({trajectory: fold_diagnostics[trajectory]}),
            flush=True,
        )

    summary = {"folds": fold_diagnostics, "heads": {}}
    for head_name in HEAD_NAMES:
        labels = torch.cat(
            [labels_by_name[name][head_name] for name in topk["query_names"]]
        ).numpy()
        probability = torch.cat(
            [
                probabilities_by_name[name][head_name]
                for name in topk["query_names"]
            ]
        ).numpy()
        summary["heads"][head_name] = _quality_diagnostics(
            labels, probability
        )

    policies = {
        "fixed0512": {
            "minimum_budget": 512,
            "maximum_budget": 512,
        },
        "adaptive": {
            "minimum_budget": int(args.minimum_budget),
            "maximum_budget": int(args.maximum_budget),
        },
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selector_common = {
        "core_budget": int(args.core_budget),
        "minimum_strict_lcb": float(args.minimum_strict_lcb),
        "minimum_dependency_groups": int(args.minimum_dependency_groups),
        "minimum_image_cells": int(args.minimum_image_cells),
        "minimum_log_expected_basis": float(
            args.minimum_log_expected_basis
        ),
        "representative_count": int(args.representative_count),
        "pair_count": int(args.pair_count),
    }
    torch.set_num_threads(1)
    for policy_name, policy_budget in policies.items():
        records = []
        policy_diagnostics = []
        selected_labels = {name: 0 for name in HEAD_NAMES}
        selected_total = 0
        selector_config = {**selector_common, **policy_budget}
        for query_number, name in enumerate(topk["query_names"], start=1):
            record = topk_by_name[name]
            rows = torch.as_tensor(record["query_rows"]).long()
            anchors = torch.as_tensor(
                record["topk_anchor_indices"]
            ).long()[:, 0]
            keypoints = torch.as_tensor(
                cache[name]["native_keypoints"]
            ).float()[rows]
            cells = image_grid_cells(
                keypoints, cache[name]["native_input_hw"]
            )
            probability = probabilities_by_name[name]
            selected, diagnostics = basis_aware_core_reserve_mask(
                probability["strict_clean"],
                probability["solver_clean"],
                probability["harmful"],
                image_points=keypoints,
                image_hw=cache[name]["native_input_hw"],
                image_cells=cells,
                dependency_groups=dependency[anchors],
                source_groups=source[anchors],
                xyz=xyz[anchors],
                **selector_config,
            )
            selected_total += int(selected.sum())
            for head_name in HEAD_NAMES:
                selected_labels[head_name] += int(
                    (
                        selected
                        & labels_by_name[name][head_name]
                    ).sum()
                )
            policy_diagnostics.append(diagnostics)
            records.append(
                {
                    "query_name": name,
                    "query_rows": rows,
                    "topk_anchor_indices": torch.as_tensor(
                        record["topk_anchor_indices"]
                    ).long(),
                    "topk_scores": torch.as_tensor(
                        record["topk_scores"]
                    ).float(),
                    "selected_row_mask": selected,
                    "strict_probability": probability["strict_clean"],
                    "solver_probability": probability["solver_clean"],
                    "harmful_probability": probability["harmful"],
                    "basis_diagnostics": diagnostics,
                }
            )
            if query_number % 100 == 0:
                print(
                    f"{policy_name}: {query_number}/{len(topk['query_names'])}",
                    flush=True,
                )
        policy_summary = {
            "selected_count_mean": selected_total / len(records),
            "selected_strict_precision": (
                selected_labels["strict_clean"] / max(selected_total, 1)
            ),
            "selected_solver_precision": (
                selected_labels["solver_clean"] / max(selected_total, 1)
            ),
            "selected_harmful_rate": (
                selected_labels["harmful"] / max(selected_total, 1)
            ),
            "strict_lcb_mean": float(
                np.mean(
                    [value["strict_lcb"] for value in policy_diagnostics]
                )
            ),
            "log_expected_basis_mean": float(
                np.mean(
                    [
                        value["log_expected_basis"]
                        for value in policy_diagnostics
                    ]
                )
            ),
        }
        summary[policy_name] = policy_summary
        _atomic_torch(
            output_dir / f"selected_{policy_name}.pt",
            {
                "schema": "lafgs_exact_topk_outcomes",
                "version": 4,
                "query_names": list(topk["query_names"]),
                "query_start": int(topk.get("query_start", 0)),
                "topk": int(topk["topk"]),
                "anchor_count": int(topk["anchor_count"]),
                "anchor_ids_sha256": topk["anchor_ids_sha256"],
                "records": records,
                "method": "oof_basis_aware_core_reserve_selection",
                "summary": {**summary["heads"], **policy_summary},
                "provenance": {
                    "topk_outcomes": str(
                        Path(args.topk_outcomes).resolve()
                    ),
                    "dynamic_outcomes": str(
                        Path(args.dynamic_outcomes).resolve()
                    ),
                },
            },
        )

    full_statistics = {
        name: values.sum(dim=0) for name, values in statistics.items()
    }
    full_features = []
    for name in topk["query_names"]:
        features, _ = _features_for_query(
            topk_by_name[name],
            dynamic_by_name[name],
            cache[name],
            source_groups=source,
            dependency_groups=dependency,
            anchor_statistics=statistics,
            excluded_trajectory_index=None,
            folded_statistics=full_statistics,
            positive_target="gt_clean",
        )
        full_features.append(features)
    full_features_numpy = torch.cat(full_features).numpy()
    full_models = {}
    for head_index, head_name in enumerate(HEAD_NAMES):
        labels = torch.cat(
            [labels_by_name[name][head_name] for name in topk["query_names"]]
        ).numpy()
        model, mean, scale = _fit_probability_model(
            full_features_numpy,
            labels,
            maximum_fit_rows=int(args.maximum_fit_rows),
            seed=int(args.seed) + 1000 + head_index,
        )
        full_models[head_name] = _model_payload(model, mean, scale)
    for policy_name, policy_budget in policies.items():
        _atomic_torch(
            output_dir / f"selector_model_{policy_name}.pt",
            {
                "schema": "lafgs_basis_core_reserve_selector",
                "version": 1,
                "fold_contract": "leave_one_trajectory_out",
                "fold_models": fold_models,
                "full_models": full_models,
                "anchor_statistics": full_statistics,
                "selector_config": {
                    **selector_common,
                    **policy_budget,
                },
                "summary": summary,
                "training_config": vars(args),
                "anchor_count": len(anchor_ids),
                "anchor_ids_sha256": _sha256_tensor(anchor_ids),
                "candidate_graph_contract": dict(
                    topk.get("provenance", {})
                ),
                "retrieval_topk": int(topk["topk"]),
                "entropy_temperature": 0.05,
                "prior_strength": 12.0,
            },
        )
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
