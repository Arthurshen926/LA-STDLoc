#!/usr/bin/env python3
"""Partition mapping queries and fit OOF minimal-set sampling/risk models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


FEATURE_NAMES = ("top1_score", "top1_margin", "keypoint_score")


def _sequence(name: str) -> str:
    return name.split("/", 1)[0]


def partition(function_graph: str, output_dir: str, fold_count: int) -> None:
    graph = torch.load(function_graph, map_location="cpu", weights_only=False)
    names = list(graph["query_names"])
    folds = np.zeros(len(names), dtype=np.int64)
    by_sequence: dict[str, list[tuple[str, int]]] = {}
    for index, name in enumerate(names):
        by_sequence.setdefault(_sequence(name), []).append((name, index))
    for rows in by_sequence.values():
        rows.sort()
        for rank, (_, index) in enumerate(rows):
            folds[index] = min(
                int(rank * fold_count / max(len(rows), 1)), fold_count - 1
            )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    query_to_fold = {}
    for fold in range(fold_count):
        fold_names = [name for name, value in zip(names, folds) if value == fold]
        (output / f"fold{fold}.txt").write_text("\n".join(fold_names) + "\n")
        query_to_fold.update({name: fold for name in fold_names})
    manifest = {
        "schema": "lafgs_sampler_oof_partition",
        "version": 1,
        "function_graph": str(Path(function_graph).resolve()),
        "fold_count": fold_count,
        "query_count": len(names),
        "query_to_fold": query_to_fold,
        "fold_query_counts": {
            str(fold): int((folds == fold).sum()) for fold in range(fold_count)
        },
        "partition": "sequence_contiguous_blocks",
    }
    (output / "partition.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    return float(
        (ranks[labels].sum() - positive * (positive + 1) / 2)
        / (positive * negative)
    )


def _fit_logistic(features: np.ndarray, labels: np.ndarray) -> dict:
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-6)
    normalized = (features - mean) / scale
    classifier = LogisticRegression(
        class_weight="balanced",
        C=10.0,
        max_iter=500,
        random_state=0,
        solver="lbfgs",
    )
    classifier.fit(normalized, labels.astype(np.int64))
    return {
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coefficients": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "positive_count": int(labels.sum()),
        "negative_count": int((~labels).sum()),
    }


def _triplet_features(record: dict) -> np.ndarray | None:
    values = (
        record.get("sampling_scores"),
        record.get("sampling_margins"),
        record.get("keypoint_scores"),
    )
    if any(value is None for value in values):
        return None
    array = np.asarray(values, dtype=np.float64).T
    if array.shape != (3, len(FEATURE_NAMES)) or not np.isfinite(array).all():
        return None
    return array.mean(axis=0)


def fit(partition_path: str, teacher_paths: list[str], output: str) -> None:
    partition_payload = json.loads(Path(partition_path).read_text())
    query_to_fold = {
        str(name): int(fold)
        for name, fold in partition_payload["query_to_fold"].items()
    }
    queries = {}
    for teacher_path in teacher_paths:
        payload = torch.load(teacher_path, map_location="cpu", weights_only=False)
        for record in payload["records"]:
            name = str(record["query_name"])
            if name in queries:
                raise ValueError(f"duplicate teacher query: {name}")
            queries[name] = record
    missing = sorted(set(query_to_fold) - set(queries))
    if missing:
        raise ValueError(f"teacher records miss {len(missing)} OOF queries")

    set_features = []
    set_labels = []
    set_folds = []
    query_features = []
    query_labels = []
    query_folds = []
    for name, fold in query_to_fold.items():
        query = queries[name]
        top1 = torch.as_tensor(query["top1_scores"]).float().numpy()
        margins = torch.as_tensor(query["top1_margins"]).float().numpy()
        keypoint = torch.as_tensor(query["keypoint_scores"]).float().numpy()
        query_features.append(
            np.asarray(
                [np.median(top1), np.median(margins), np.median(keypoint)]
            )
        )
        query_labels.append(
            float(query["te_cm"]) > 20.0
            or float(torch.as_tensor(query["ransac_inlier_mask"]).float().mean()) < 0.04
        )
        query_folds.append(fold)
        for minimal_set in query["minimal_set_records"]:
            features = _triplet_features(minimal_set)
            if features is None:
                continue
            set_features.append(features)
            set_labels.append(bool(minimal_set["correct_basin"]))
            set_folds.append(fold)
    set_features = np.asarray(set_features, dtype=np.float64)
    set_labels = np.asarray(set_labels, dtype=bool)
    set_folds = np.asarray(set_folds, dtype=np.int64)
    query_features = np.asarray(query_features, dtype=np.float64)
    query_labels = np.asarray(query_labels, dtype=bool)
    query_folds = np.asarray(query_folds, dtype=np.int64)

    models = {}
    risk_models = {}
    diagnostics = {}
    fold_count = int(partition_payload["fold_count"])
    for fold in range(fold_count):
        train = set_folds != fold
        heldout = set_folds == fold
        model = _fit_logistic(set_features[train], set_labels[train])
        logits = (
            (set_features[heldout] - np.asarray(model["feature_mean"]))
            / np.asarray(model["feature_scale"])
        ) @ np.asarray(model["coefficients"]) + float(model["intercept"])
        baseline = set_features[heldout, 0]
        models[str(fold)] = model

        risk_train = query_folds != fold
        risk_heldout = query_folds == fold
        risk_model = _fit_logistic(
            query_features[risk_train], query_labels[risk_train]
        )
        risk_logits = (
            (query_features[risk_heldout] - np.asarray(risk_model["feature_mean"]))
            / np.asarray(risk_model["feature_scale"])
        ) @ np.asarray(risk_model["coefficients"]) + float(risk_model["intercept"])
        risk_models[str(fold)] = risk_model
        diagnostics[str(fold)] = {
            "set_count": int(heldout.sum()),
            "positive_set_count": int(set_labels[heldout].sum()),
            "sampling_logit_auc": _auc(set_labels[heldout], logits),
            "top1_score_auc": _auc(set_labels[heldout], baseline),
            "query_count": int(risk_heldout.sum()),
            "risk_query_count": int(query_labels[risk_heldout].sum()),
            "query_risk_auc": _auc(query_labels[risk_heldout], risk_logits),
        }
    models["all"] = _fit_logistic(set_features, set_labels)
    risk_models["all"] = _fit_logistic(query_features, query_labels)
    output_payload = {
        "schema": "lafgs_minimal_set_oof_sampling_model",
        "version": 1,
        "partition": partition_payload,
        "teacher_paths": [str(Path(path).resolve()) for path in teacher_paths],
        "query_to_fold": {
            name: str(fold) for name, fold in query_to_fold.items()
        },
        "models": models,
        "risk_models": risk_models,
        "risk_threshold": 0.5,
        "diagnostics": diagnostics,
    }
    torch.save(output_payload, output)
    print(json.dumps(diagnostics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    partition_parser = subparsers.add_parser("partition")
    partition_parser.add_argument("--function-graph", required=True)
    partition_parser.add_argument("--output-dir", required=True)
    partition_parser.add_argument("--fold-count", type=int, default=5)
    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--partition", required=True)
    fit_parser.add_argument("--teacher", nargs="+", required=True)
    fit_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "partition":
        partition(args.function_graph, args.output_dir, args.fold_count)
    else:
        fit(args.partition, args.teacher, args.output)


if __name__ == "__main__":
    main()
