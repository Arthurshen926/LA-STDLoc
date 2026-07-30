#!/usr/bin/env python3
"""Fit a cross-trajectory OOF gate for frozen candidate-context proposals."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


FEATURE_NAMES = (
    "local_top1_score",
    "local_top1_margin",
    "challenger_local_score",
    "challenger_local_gap",
    "context_score_delta",
    "rescued_score_margin",
    "challenger_rank",
    "keypoint_score",
    "keypoint_x_normalized",
    "keypoint_y_normalized",
    "edge_log_occurrences",
    "edge_log_trajectories",
    "edge_harmful_fraction",
    "edge_survivor_fraction",
    "edge_log_pose_blame",
    "anchor_log_distance",
    "same_source",
    "same_dependency",
)


def _records_by_name(payload: dict) -> dict[str, dict]:
    return {
        str(record["query_name"]): record
        for record in payload["records"]
    }


def _trajectory(name: str) -> str:
    return str(name).split("/", 1)[0]


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


def _fit(features: np.ndarray, labels: np.ndarray, seed: int) -> dict:
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-6)
    normalized = (features - mean) / scale
    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        max_iter=500,
        random_state=int(seed),
        solver="lbfgs",
    )
    model.fit(normalized, labels.astype(np.int64))
    return {
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": mean,
        "feature_scale": scale,
        "coefficients": model.coef_[0],
        "intercept": float(model.intercept_[0]),
    }


def _probability(model: dict, features: np.ndarray) -> np.ndarray:
    logits = (
        (features - model["feature_mean"]) / model["feature_scale"]
    ) @ model["coefficients"] + float(model["intercept"])
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def _protected_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    false_positive_cost: float,
) -> tuple[float, dict]:
    order = np.argsort(-probabilities, kind="stable")
    sorted_labels = labels[order].astype(bool)
    true_positive = np.cumsum(sorted_labels)
    false_positive = np.cumsum(~sorted_labels)
    utility = true_positive - float(false_positive_cost) * false_positive
    best = int(np.argmax(utility))
    if float(utility[best]) <= 0.0:
        return 1.0, {
            "selected": 0,
            "precision": 1.0,
            "recall": 0.0,
            "utility": 0.0,
        }
    threshold = float(probabilities[order[best]])
    selected = best + 1
    return threshold, {
        "selected": int(selected),
        "precision": float(true_positive[best] / selected),
        "recall": float(
            true_positive[best] / max(int(labels.sum()), 1)
        ),
        "utility": float(utility[best]),
    }


def _serializable_model(model: dict, threshold: float) -> dict:
    return {
        "feature_names": list(model["feature_names"]),
        "feature_mean": model["feature_mean"].tolist(),
        "feature_scale": model["feature_scale"].tolist(),
        "coefficients": model["coefficients"].tolist(),
        "intercept": float(model["intercept"]),
        "threshold": float(threshold),
    }


def _atomic_torch(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--baseline-topk", required=True)
    parser.add_argument("--edge-rescue-topk", required=True)
    parser.add_argument("--strict-oracle-topk", required=True)
    parser.add_argument("--confusion-graph", required=True)
    parser.add_argument("--output-topk", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--false-positive-cost", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    state = torch.load(args.map, map_location="cpu", weights_only=False)
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    baseline = torch.load(
        args.baseline_topk, map_location="cpu", weights_only=False
    )
    rescue = torch.load(
        args.edge_rescue_topk, map_location="cpu", weights_only=False
    )
    oracle = torch.load(
        args.strict_oracle_topk, map_location="cpu", weights_only=False
    )
    graph = torch.load(
        args.confusion_graph, map_location="cpu", weights_only=False
    )
    for payload in (baseline, rescue, oracle):
        if payload.get("schema") != "lafgs_exact_topk_outcomes":
            raise ValueError("candidate gate requires exact top-K states")
    if not (
        baseline["query_names"]
        == rescue["query_names"]
        == oracle["query_names"]
    ):
        raise ValueError("candidate gate query registries differ")
    if not (
        baseline["anchor_ids_sha256"]
        == rescue["anchor_ids_sha256"]
        == oracle["anchor_ids_sha256"]
    ):
        raise ValueError("candidate gate maps differ")

    edge_metadata = {
        (int(edge["correct_anchor"]), int(edge["confusing_anchor"])): edge
        for edge in graph["edges"]
    }
    baseline_by_name = _records_by_name(baseline)
    rescue_by_name = _records_by_name(rescue)
    oracle_by_name = _records_by_name(oracle)
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    source = torch.as_tensor(state["source_primitive_ids"]).long()
    dependency = torch.as_tensor(
        state.get("coarse_dependency_group_ids", state["dependency_group_ids"])
    ).long()
    samples = []
    for name in baseline["query_names"]:
        before = baseline_by_name[name]
        after = rescue_by_name[name]
        strict = oracle_by_name[name]
        rows = torch.as_tensor(before["query_rows"]).long()
        before_indices = torch.as_tensor(
            before["topk_anchor_indices"]
        ).long()
        before_scores = torch.as_tensor(before["topk_scores"]).float()
        after_indices = torch.as_tensor(
            after["topk_anchor_indices"]
        ).long()
        after_scores = torch.as_tensor(after["topk_scores"]).float()
        strict_indices = torch.as_tensor(
            strict["topk_anchor_indices"]
        ).long()
        if not torch.equal(rows, torch.as_tensor(after["query_rows"]).long()):
            raise ValueError(f"rescue rows differ for {name}")
        changed = torch.nonzero(
            after_indices[:, 0] != before_indices[:, 0],
            as_tuple=False,
        ).reshape(-1)
        cached = cache[name]
        keypoint_scores = torch.as_tensor(
            cached["native_scores"]
        ).float()[rows]
        keypoints = torch.as_tensor(
            cached["native_keypoints"]
        ).float()[rows]
        height, width = cached["native_input_hw"]
        for slot in changed.tolist():
            confusing = int(before_indices[slot, 0])
            correct = int(after_indices[slot, 0])
            metadata = edge_metadata.get((correct, confusing))
            if metadata is None:
                raise ValueError("rescued pair is absent from confusion graph")
            candidate_slots = torch.nonzero(
                before_indices[slot] == correct, as_tuple=False
            ).reshape(-1)
            if candidate_slots.numel() != 1:
                raise ValueError("rescued challenger is not unique in baseline top-K")
            rank = int(candidate_slots[0])
            local_top1 = float(before_scores[slot, 0])
            challenger = float(before_scores[slot, rank])
            rescued_score = float(after_scores[slot, 0])
            occurrences = max(int(metadata.get("occurrences", 0)), 1)
            harmful = int(metadata.get("harmful_occurrences", 0))
            survivors = int(
                metadata.get("harmful_ransac_survivors", 0)
            )
            distance = float(
                torch.linalg.vector_norm(xyz[correct] - xyz[confusing])
            )
            features = np.asarray(
                [
                    local_top1,
                    local_top1 - float(before_scores[slot, 1]),
                    challenger,
                    local_top1 - challenger,
                    rescued_score - challenger,
                    rescued_score - local_top1,
                    rank / max(before_indices.shape[1] - 1, 1),
                    float(keypoint_scores[slot]),
                    float(keypoints[slot, 0]) / max(float(width), 1.0),
                    float(keypoints[slot, 1]) / max(float(height), 1.0),
                    np.log1p(occurrences),
                    np.log1p(int(metadata.get("trajectory_count", 0))),
                    harmful / occurrences,
                    survivors / max(harmful, 1),
                    np.log1p(max(float(metadata.get("pose_blame", 0.0)), 0.0)),
                    np.log1p(distance),
                    float(source[correct] == source[confusing]),
                    float(dependency[correct] == dependency[confusing]),
                ],
                dtype=np.float64,
            )
            samples.append(
                {
                    "query_name": name,
                    "row_slot": slot,
                    "features": features,
                    "label": bool(
                        int(strict_indices[slot, 0]) == correct
                    ),
                }
            )
    features = np.stack([sample["features"] for sample in samples])
    labels = np.asarray([sample["label"] for sample in samples], dtype=bool)
    trajectories = np.asarray(
        [_trajectory(sample["query_name"]) for sample in samples]
    )
    unique_trajectories = sorted(set(trajectories.tolist()))
    probabilities = np.zeros(len(samples), dtype=np.float64)
    accepted = np.zeros(len(samples), dtype=bool)
    fold_models = {}
    diagnostics = {}
    for fold_index, trajectory in enumerate(unique_trajectories):
        train = trajectories != trajectory
        heldout = trajectories == trajectory
        model = _fit(features[train], labels[train], args.seed + fold_index)
        train_probability = _probability(model, features[train])
        threshold, threshold_diagnostics = _protected_threshold(
            labels[train],
            train_probability,
            false_positive_cost=float(args.false_positive_cost),
        )
        heldout_probability = _probability(model, features[heldout])
        probabilities[heldout] = heldout_probability
        accepted[heldout] = heldout_probability >= threshold
        heldout_labels = labels[heldout]
        heldout_accept = accepted[heldout]
        true_positive = int((heldout_accept & heldout_labels).sum())
        false_positive = int((heldout_accept & ~heldout_labels).sum())
        fold_models[trajectory] = _serializable_model(model, threshold)
        diagnostics[trajectory] = {
            "sample_count": int(heldout.sum()),
            "positive_count": int(heldout_labels.sum()),
            "auc": _auc(heldout_labels, heldout_probability),
            "accepted_count": int(heldout_accept.sum()),
            "accepted_precision": float(
                true_positive / max(true_positive + false_positive, 1)
            ),
            "accepted_recall": float(
                true_positive / max(int(heldout_labels.sum()), 1)
            ),
            "training_threshold": threshold_diagnostics,
        }

    output_records = []
    accepted_by_query: dict[str, set[int]] = {}
    for sample, use_rescue in zip(samples, accepted):
        if use_rescue:
            accepted_by_query.setdefault(
                sample["query_name"], set()
            ).add(int(sample["row_slot"]))
    for name in baseline["query_names"]:
        before = baseline_by_name[name]
        after = rescue_by_name[name]
        indices = torch.as_tensor(
            before["topk_anchor_indices"]
        ).long().clone()
        scores = torch.as_tensor(before["topk_scores"]).float().clone()
        selected = sorted(accepted_by_query.get(name, ()))
        if selected:
            selected_tensor = torch.as_tensor(selected).long()
            indices[selected_tensor] = torch.as_tensor(
                after["topk_anchor_indices"]
            ).long()[selected_tensor]
            scores[selected_tensor] = torch.as_tensor(
                after["topk_scores"]
            ).float()[selected_tensor]
        output_records.append(
            {
                "query_name": name,
                "query_rows": torch.as_tensor(before["query_rows"]).long(),
                "topk_anchor_indices": indices,
                "topk_scores": scores,
            }
        )

    full_model = _fit(features, labels, args.seed + 1000)
    full_probability = _probability(full_model, features)
    full_threshold, full_threshold_diagnostics = _protected_threshold(
        labels,
        full_probability,
        false_positive_cost=float(args.false_positive_cost),
    )
    true_positive = int((accepted & labels).sum())
    false_positive = int((accepted & ~labels).sum())
    summary = {
        "proposal_count": int(len(samples)),
        "strict_positive_count": int(labels.sum()),
        "oof_auc": _auc(labels, probabilities),
        "oof_accepted_count": int(accepted.sum()),
        "oof_accepted_precision": float(
            true_positive / max(true_positive + false_positive, 1)
        ),
        "oof_accepted_recall": float(
            true_positive / max(int(labels.sum()), 1)
        ),
        "false_positive_cost": float(args.false_positive_cost),
    }
    output_topk = Path(args.output_topk).resolve()
    _atomic_torch(
        output_topk,
        {
            "schema": "lafgs_exact_topk_outcomes",
            "version": 2,
            "query_names": list(baseline["query_names"]),
            "query_start": int(baseline.get("query_start", 0)),
            "topk": int(baseline["topk"]),
            "anchor_count": int(baseline["anchor_count"]),
            "anchor_ids_sha256": baseline["anchor_ids_sha256"],
            "records": output_records,
            "method": "candidate_conditioned_context_oof_pair_gate",
            "summary": summary,
            "provenance": {
                "baseline_topk": str(Path(args.baseline_topk).resolve()),
                "edge_rescue_topk": str(
                    Path(args.edge_rescue_topk).resolve()
                ),
                "strict_oracle_topk": str(
                    Path(args.strict_oracle_topk).resolve()
                ),
                "confusion_graph": str(
                    Path(args.confusion_graph).resolve()
                ),
            },
        },
    )
    output_model = Path(args.output_model).resolve()
    _atomic_torch(
        output_model,
        {
            "schema": "lafgs_candidate_context_pair_gate",
            "version": 1,
            "feature_names": list(FEATURE_NAMES),
            "fold_contract": "leave_one_trajectory_out",
            "fold_models": fold_models,
            "full_model": _serializable_model(
                full_model, full_threshold
            ),
            "full_threshold_diagnostics": full_threshold_diagnostics,
            "diagnostics": diagnostics,
            "summary": summary,
            "provenance": {
                "map": str(Path(args.map).resolve()),
                "query_cache": str(Path(args.query_cache).resolve()),
                "output_topk": str(output_topk),
            },
        },
    )
    print(json.dumps({"summary": summary, "folds": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
