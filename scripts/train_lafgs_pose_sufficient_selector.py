#!/usr/bin/env python3
"""Train an OOF matchability model and constrained sparse correspondence sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from localization_training.pose_sufficient_selector import (
    FEATURE_NAMES,
    build_pose_sufficient_features,
    constrained_pose_sufficient_mask,
    image_grid_cells,
)


def _trajectory(name: str) -> str:
    return str(name).split("/", 1)[0]


def _records_by_name(payload: dict) -> dict[str, dict]:
    return {
        str(record["query_name"]): record for record in payload["records"]
    }


def _sha256_tensor(value: torch.Tensor) -> str:
    value = torch.as_tensor(value).detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _anchor_statistics(
    dynamic_records: list[dict],
    trajectory_indices: torch.Tensor,
    *,
    trajectory_count: int,
    anchor_count: int,
) -> dict[str, torch.Tensor]:
    statistics = {
        name: torch.zeros(trajectory_count, anchor_count)
        for name in ("attempts", "clean", "clean_inlier", "harmful_inlier")
    }
    for query_index, record in enumerate(dynamic_records):
        trajectory = int(trajectory_indices[query_index])
        anchors = torch.as_tensor(record["top1_anchor_indices"]).long()
        ones = torch.ones(len(anchors))
        statistics["attempts"][trajectory].index_add_(0, anchors, ones)
        clean = (
            torch.as_tensor(record["gt_reprojection_errors_px"]).float() <= 2
        ).float()
        statistics["clean"][trajectory].index_add_(0, anchors, clean)
        statistics["clean_inlier"][trajectory].index_add_(
            0,
            anchors,
            torch.as_tensor(record["clean_inlier_mask"]).float(),
        )
        statistics["harmful_inlier"][trajectory].index_add_(
            0,
            anchors,
            torch.as_tensor(record["harmful_inlier_mask"]).float(),
        )
    return statistics


def _features_for_query(
    topk_record: dict,
    dynamic_record: dict,
    cached: dict,
    *,
    source_groups: torch.Tensor,
    dependency_groups: torch.Tensor,
    anchor_statistics: dict[str, torch.Tensor],
    excluded_trajectory_index: int | None,
    folded_statistics: dict[str, torch.Tensor] | None = None,
    positive_target: str = "gt_clean",
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.as_tensor(topk_record["query_rows"]).long()
    if not torch.equal(
        rows, torch.as_tensor(dynamic_record["query_rows"]).long()
    ):
        raise ValueError("top-K and dynamic query rows differ")
    keypoints = torch.as_tensor(cached["native_keypoints"]).float()[rows]
    if folded_statistics is not None:
        folded = folded_statistics
    elif excluded_trajectory_index is None:
        folded = {
            name: values.sum(dim=0)
            for name, values in anchor_statistics.items()
        }
    else:
        folded = {
            name: values.sum(dim=0) - values[int(excluded_trajectory_index)]
            for name, values in anchor_statistics.items()
        }
    features = build_pose_sufficient_features(
        torch.as_tensor(topk_record["topk_scores"]).float(),
        torch.as_tensor(topk_record["topk_anchor_indices"]).long(),
        keypoints=keypoints,
        keypoint_scores=torch.as_tensor(
            dynamic_record["keypoint_scores"]
        ).float(),
        image_hw=cached["native_input_hw"],
        source_groups=source_groups,
        dependency_groups=dependency_groups,
        anchor_statistics=folded,
    )
    if positive_target == "gt_clean":
        labels = (
            torch.as_tensor(
                dynamic_record["gt_reprojection_errors_px"]
            ).float()
            <= 2
        )
    elif positive_target == "clean_inlier":
        labels = torch.as_tensor(
            dynamic_record["clean_inlier_mask"]
        ).bool()
    else:
        raise ValueError(f"unsupported positive target {positive_target!r}")
    return features, labels


def _fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    maximum_fit_rows: int,
) -> tuple[LogisticRegression, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    positive = np.flatnonzero(labels)
    negative = np.flatnonzero(~labels)
    maximum = max(int(maximum_fit_rows), 1000)
    positive_budget = min(len(positive), maximum // 2)
    negative_budget = min(len(negative), maximum - positive_budget)
    chosen = np.concatenate(
        (
            rng.choice(positive, positive_budget, replace=False),
            rng.choice(negative, negative_budget, replace=False),
        )
    )
    rng.shuffle(chosen)
    mean = features[chosen].mean(axis=0)
    scale = features[chosen].std(axis=0).clip(1e-6)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=150,
        random_state=int(seed),
        solver="lbfgs",
    )
    model.fit((features[chosen] - mean) / scale, labels[chosen])
    return model, mean, scale


def _model_payload(
    model: LogisticRegression, mean: np.ndarray, scale: np.ndarray
) -> dict:
    return {
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": torch.from_numpy(mean).float(),
        "feature_scale": torch.from_numpy(scale).float(),
        "coefficients": torch.from_numpy(model.coef_[0]).float(),
        "intercept": float(model.intercept_[0]),
    }


def _predict(
    model: LogisticRegression,
    mean: np.ndarray,
    scale: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    return model.predict_proba((features - mean) / scale)[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--topk-outcomes", required=True)
    parser.add_argument("--dynamic-outcomes", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budgets", default="512,768,1024")
    parser.add_argument("--maximum-fit-rows", type=int, default=400000)
    parser.add_argument("--minimum-per-image-cell", type=int, default=8)
    parser.add_argument("--minimum-per-spatial-bin", type=int, default=4)
    parser.add_argument("--maximum-per-dependency", type=int, default=4)
    parser.add_argument("--maximum-per-source", type=int, default=2)
    parser.add_argument(
        "--positive-target",
        choices=("gt_clean", "clean_inlier"),
        default="gt_clean",
        help=(
            "gt_clean selects GT reprojection-clean@2 rows; clean_inlier "
            "selects baseline RANSAC inliers that are GT-clean@4"
        ),
    )
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

    if args.positive_target == "gt_clean":
        labels_by_name = {
            name: (
                torch.as_tensor(
                    dynamic_by_name[name]["gt_reprojection_errors_px"]
                ).float()
                <= 2
            )
            for name in topk["query_names"]
        }
    else:
        labels_by_name = {
            name: torch.as_tensor(
                dynamic_by_name[name]["clean_inlier_mask"]
            ).bool()
            for name in topk["query_names"]
        }

    probabilities_by_name = {}
    fold_models = {}
    fold_diagnostics = {}
    for fold_index, trajectory in enumerate(trajectories):
        folded_statistics = {
            name: values.sum(dim=0) - values[fold_index]
            for name, values in statistics.items()
        }
        fold_features_by_name = {}
        for name in topk["query_names"]:
            features, labels = _features_for_query(
                topk_by_name[name],
                dynamic_by_name[name],
                cache[name],
                source_groups=source,
                dependency_groups=dependency,
                anchor_statistics=statistics,
                excluded_trajectory_index=fold_index,
                folded_statistics=folded_statistics,
                positive_target=str(args.positive_target),
            )
            if not torch.equal(labels, labels_by_name[name]):
                raise AssertionError("fold feature labels changed")
            fold_features_by_name[name] = features
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
            [fold_features_by_name[name] for name in train_names]
        ).numpy()
        train_labels = torch.cat(
            [labels_by_name[name] for name in train_names]
        ).numpy()
        model, mean, scale = _fit_logistic(
            train_features,
            train_labels,
            seed=int(args.seed) + fold_index,
            maximum_fit_rows=int(args.maximum_fit_rows),
        )
        heldout_features = torch.cat(
            [fold_features_by_name[name] for name in heldout_names]
        ).numpy()
        heldout_labels = torch.cat(
            [labels_by_name[name] for name in heldout_names]
        ).numpy()
        probability = _predict(model, mean, scale, heldout_features)
        cursor = 0
        for name in heldout_names:
            count = len(fold_features_by_name[name])
            probabilities_by_name[name] = torch.from_numpy(
                probability[cursor : cursor + count]
            ).float()
            cursor += count
        del fold_features_by_name
        fold_models[trajectory] = _model_payload(model, mean, scale)
        fold_diagnostics[trajectory] = {
            "row_count": int(len(heldout_labels)),
            "positive_rate": float(heldout_labels.mean()),
            "auroc": float(roc_auc_score(heldout_labels, probability)),
            "average_precision": float(
                average_precision_score(heldout_labels, probability)
            ),
        }
        print(json.dumps({trajectory: fold_diagnostics[trajectory]}), flush=True)

    oof_labels = torch.cat(
        [labels_by_name[name] for name in topk["query_names"]]
    ).numpy()
    oof_probability = torch.cat(
        [probabilities_by_name[name] for name in topk["query_names"]]
    ).numpy()
    summary = {
        "row_count": int(len(oof_labels)),
        "positive_rate": float(oof_labels.mean()),
        "oof_auroc": float(roc_auc_score(oof_labels, oof_probability)),
        "oof_average_precision": float(
            average_precision_score(oof_labels, oof_probability)
        ),
        "folds": fold_diagnostics,
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    budgets = sorted(
        {int(value) for value in args.budgets.split(",") if value.strip()}
    )
    for budget in budgets:
        records = []
        selected_clean = 0
        selected_total = 0
        for name in topk["query_names"]:
            record = topk_by_name[name]
            rows = torch.as_tensor(record["query_rows"]).long()
            anchors = torch.as_tensor(
                record["topk_anchor_indices"]
            ).long()[:, 0]
            keypoints = torch.as_tensor(
                cache[name]["native_keypoints"]
            ).float()[rows]
            cells = image_grid_cells(
                keypoints,
                cache[name]["native_input_hw"],
                rows=4,
                cols=4,
            )
            selected = constrained_pose_sufficient_mask(
                probabilities_by_name[name],
                image_cells=cells,
                dependency_groups=dependency[anchors],
                source_groups=source[anchors],
                xyz=xyz[anchors],
                budget=budget,
                minimum_per_image_cell=int(args.minimum_per_image_cell),
                minimum_per_spatial_bin=int(args.minimum_per_spatial_bin),
                maximum_per_dependency=int(args.maximum_per_dependency),
                maximum_per_source=int(args.maximum_per_source),
            )
            selected_total += int(selected.sum())
            selected_clean += int(
                (selected & labels_by_name[name]).sum()
            )
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
                    "matchability_probability": probabilities_by_name[name],
                }
            )
        path = output_dir / f"selected_{budget:04d}.pt"
        _atomic_torch(
            path,
            {
                "schema": "lafgs_exact_topk_outcomes",
                "version": 3,
                "query_names": list(topk["query_names"]),
                "query_start": int(topk.get("query_start", 0)),
                "topk": int(topk["topk"]),
                "anchor_count": int(topk["anchor_count"]),
                "anchor_ids_sha256": topk["anchor_ids_sha256"],
                "records": records,
                "method": "oof_pose_sufficient_set_selection",
                "summary": {
                    **summary,
                    "budget": budget,
                    "selected_row_count": selected_total,
                    "selected_gt_precision_2px": selected_clean
                    / max(selected_total, 1),
                },
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

    full_features = []
    full_labels = []
    full_statistics = {
        name: values.sum(dim=0) for name, values in statistics.items()
    }
    for name in topk["query_names"]:
        features, labels = _features_for_query(
            topk_by_name[name],
            dynamic_by_name[name],
            cache[name],
            source_groups=source,
            dependency_groups=dependency,
            anchor_statistics=statistics,
            excluded_trajectory_index=None,
            folded_statistics=full_statistics,
            positive_target=str(args.positive_target),
        )
        full_features.append(features)
        full_labels.append(labels)
    full_features_numpy = torch.cat(full_features).numpy()
    full_labels_numpy = torch.cat(full_labels).numpy()
    full_model, full_mean, full_scale = _fit_logistic(
        full_features_numpy,
        full_labels_numpy,
        seed=int(args.seed) + 1000,
        maximum_fit_rows=int(args.maximum_fit_rows),
    )
    _atomic_torch(
        output_dir / "selector_model.pt",
        {
            "schema": "lafgs_pose_sufficient_selector",
            "version": 1,
            "fold_contract": "leave_one_trajectory_out",
            "fold_models": fold_models,
            "full_model": _model_payload(
                full_model, full_mean, full_scale
            ),
            "anchor_statistics": {
                name: values.sum(dim=0)
                for name, values in statistics.items()
            },
            "selector_config": {
                "minimum_per_image_cell": int(
                    args.minimum_per_image_cell
                ),
                "minimum_per_spatial_bin": int(
                    args.minimum_per_spatial_bin
                ),
                "maximum_per_dependency": int(
                    args.maximum_per_dependency
                ),
                "maximum_per_source": int(args.maximum_per_source),
            },
            "summary": summary,
            "training_config": vars(args),
            "anchor_count": len(anchor_ids),
            "anchor_ids_sha256": _sha256_tensor(anchor_ids),
            "candidate_graph_contract": dict(topk.get("provenance", {})),
            "retrieval_topk": int(topk["topk"]),
            "entropy_temperature": 0.05,
            "prior_strength": 12.0,
            "positive_target": str(args.positive_target),
        },
    )
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
