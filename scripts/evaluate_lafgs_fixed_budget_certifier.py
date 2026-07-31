#!/usr/bin/env python3
"""Evaluate cross-scene risk certification of the frozen P1-512 set."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from localization_training.fixed_budget_certifier import (
    CERTIFIER_FEATURE_NAMES,
    calibrate_selective_risk,
    certify_fixed_budget,
    fit_linear_risk_certifier,
    fixed_budget_certifier_features,
    predict_unsafe_probability,
)


DEFAULT_SCENES = (
    "GreatCourt",
    "KingsCollege",
    "ShopFacade",
    "StMarysChurch",
)
DEFAULT_SEEDS = (2026, 2027, 2028)


def _result_path(root: Path, scene: str, variant: str, seed: int) -> Path:
    paths = list(
        (root / scene / "evaluation" / variant / f"seed{seed}").glob(
            "results/*/results.json"
        )
    )
    if len(paths) != 1:
        raise FileNotFoundError(
            f"expected one {scene}/{variant}/seed{seed} result, got {paths}"
        )
    return paths[0]


def _load_rows(
    root: Path, scene: str, variant: str, seed: int
) -> dict[str, dict]:
    with _result_path(root, scene, variant, seed).open() as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        rows = rows["results"]
    output = {str(row["image_name"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate query names in {scene}/{variant}/seed{seed}")
    return output


def _relative_safe(baseline: dict, compact: dict) -> bool:
    baseline_te = float(baseline["sparse_TE"])
    baseline_re = float(baseline["sparse_AE"])
    compact_te = float(compact["sparse_TE"])
    compact_re = float(compact["sparse_AE"])
    catastrophic = compact_te > 100.0 or compact_re > 10.0
    baseline_r5 = baseline_te <= 5.0 and baseline_re <= 5.0
    compact_r5 = compact_te <= 5.0 and compact_re <= 5.0
    return bool(
        not catastrophic
        and compact_te <= max(baseline_te * 1.15, baseline_te + 0.5)
        and compact_re <= max(baseline_re * 1.15, baseline_re + 0.25)
        and (not baseline_r5 or compact_r5)
    )


def _trajectory_frame(query: str) -> tuple[str, int]:
    match = re.search(r"(^|/)([^/]+)/frame(\d+)", query)
    if match is None:
        return query.rsplit("/", 1)[0], 0
    return match.group(2), int(match.group(3))


def _split_trajectory_tail(records: list[dict]) -> tuple[list, list]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["trajectory"]].append(record)
    train: list[dict] = []
    calibration: list[dict] = []
    for trajectory in sorted(grouped):
        rows = sorted(grouped[trajectory], key=lambda row: row["frame"])
        boundary = max(1, int(0.75 * len(rows)))
        train.extend(rows[:boundary])
        calibration.extend(rows[boundary:])
    return train, calibration


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels.astype(bool)
    if not positive.any() or positive.all():
        return float("nan")
    ranks = np.argsort(np.argsort(scores, kind="stable"), kind="stable") + 1
    positive_ranks = ranks[positive].sum()
    positives = int(positive.sum())
    negatives = len(labels) - positives
    return float(
        (positive_ranks - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def _scene_records(
    root: Path, scene: str, seeds: tuple[int, ...]
) -> list[dict]:
    baseline = {
        seed: _load_rows(root, scene, "A1_reconstructed", seed)
        for seed in seeds
    }
    compact = {
        seed: _load_rows(root, scene, "A3_p1_fixed512", seed)
        for seed in seeds
    }
    names = tuple(sorted(baseline[seeds[0]]))
    for seed in seeds:
        if set(baseline[seed]) != set(names) or set(compact[seed]) != set(
            names
        ):
            raise ValueError(f"paired query contract differs for {scene}/{seed}")
    records = []
    for name in names:
        trajectory, frame = _trajectory_frame(name)
        features = fixed_budget_certifier_features(
            compact[seeds[0]][name]["sparse"]
        )
        unsafe = not all(
            _relative_safe(baseline[seed][name], compact[seed][name])
            for seed in seeds
        )
        catastrophic = any(
            float(compact[seed][name]["sparse_TE"]) > 100.0
            or float(compact[seed][name]["sparse_AE"]) > 10.0
            for seed in seeds
        )
        records.append(
            {
                "scene": scene,
                "query": name,
                "trajectory": trajectory,
                "frame": frame,
                "features": features,
                "unsafe": unsafe,
                "catastrophic": catastrophic,
                "baseline": {seed: baseline[seed][name] for seed in seeds},
                "compact": {seed: compact[seed][name] for seed in seeds},
            }
        )
    return records


def _pose_metrics(records: list[dict], accepted: np.ndarray, seed: int) -> dict:
    translation = []
    rotation = []
    hypotheses = []
    baseline_hypotheses = []
    for record, use_compact in zip(records, accepted):
        baseline = record["baseline"][seed]
        selected = record["compact"][seed] if use_compact else baseline
        translation.append(float(selected["sparse_TE"]))
        rotation.append(float(selected["sparse_AE"]))
        hypotheses.append(
            float(
                selected["sparse"].get(
                    "sparse_diag_ransac_actual_hypotheses", 0.0
                )
            )
        )
        baseline_hypotheses.append(
            float(
                baseline["sparse"].get(
                    "sparse_diag_ransac_actual_hypotheses", 0.0
                )
            )
        )
    te = np.asarray(translation)
    re = np.asarray(rotation)
    successful = (te <= 5.0) & (re <= 5.0)
    catastrophic = (te > 100.0) | (re > 10.0)
    denominator = max(float(np.sum(baseline_hypotheses)), 1.0)
    return {
        "median_te_cm": float(np.median(te)),
        "mean_te_cm": float(np.mean(te)),
        "p90_te_cm": float(np.quantile(te, 0.90)),
        "r2_percent": float(np.mean((te <= 2.0) & (re <= 2.0)) * 100.0),
        "r5_percent": float(np.mean(successful) * 100.0),
        "catastrophic_rate": float(np.mean(catastrophic)),
        "hypotheses_reduction": float(
            1.0 - np.sum(hypotheses) / denominator
        ),
    }


def _mean_metrics(rows: list[dict]) -> dict:
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument(
        "--seeds", default=",".join(map(str, DEFAULT_SEEDS))
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--risk-limit", type=float, default=0.02)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    scenes = tuple(filter(None, args.scenes.split(",")))
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if len(scenes) < 3:
        raise ValueError("LOSO certification requires at least three scenes")

    by_scene = {
        scene: _scene_records(root, scene, seeds) for scene in scenes
    }
    folds = []
    states = {}
    for held_out in scenes:
        training = []
        calibration = []
        for scene in scenes:
            if scene == held_out:
                continue
            scene_train, scene_calibration = _split_trajectory_tail(
                by_scene[scene]
            )
            training.extend(scene_train)
            calibration.extend(scene_calibration)
        train_features = torch.stack(
            [record["features"] for record in training]
        )
        train_labels = torch.tensor(
            [record["unsafe"] for record in training], dtype=torch.float32
        )
        state = fit_linear_risk_certifier(
            train_features,
            train_labels,
            device=args.device,
        )
        calibration_features = torch.stack(
            [record["features"] for record in calibration]
        )
        calibration_labels = torch.tensor(
            [record["unsafe"] for record in calibration], dtype=torch.bool
        )
        calibration_probability = predict_unsafe_probability(
            state, calibration_features
        )
        risk_calibration = calibrate_selective_risk(
            calibration_probability,
            calibration_labels,
            risk_limit=args.risk_limit,
            confidence=args.confidence,
        )
        state = replace(
            state,
            unsafe_probability_threshold=risk_calibration.threshold,
        )
        test = by_scene[held_out]
        test_features = torch.stack([record["features"] for record in test])
        start = time.perf_counter()
        for _ in range(100):
            accepted_tensor = certify_fixed_budget(state, test_features)
        elapsed_ms = 1000.0 * (time.perf_counter() - start) / 100.0
        accepted = accepted_tensor.cpu().numpy().astype(bool)
        test_labels = np.asarray([record["unsafe"] for record in test])
        test_catastrophic = np.asarray(
            [record["catastrophic"] for record in test]
        )
        test_probability = predict_unsafe_probability(
            state, test_features
        ).cpu().numpy()
        seed_metrics = [
            _pose_metrics(test, accepted, seed) for seed in seeds
        ]
        fold = {
            "held_out_scene": held_out,
            "training_count": len(training),
            "calibration_count": len(calibration),
            "test_count": len(test),
            "training_unsafe_rate": float(train_labels.mean()),
            "calibration_unsafe_rate": float(
                calibration_labels.float().mean()
            ),
            "test_unsafe_rate": float(np.mean(test_labels)),
            "calibration_auc": _roc_auc(
                calibration_labels.cpu().numpy(),
                calibration_probability.cpu().numpy(),
            ),
            "held_out_auc": _roc_auc(test_labels, test_probability),
            "risk_calibration": risk_calibration.__dict__,
            "accepted_count": int(accepted.sum()),
            "accepted_rate": float(accepted.mean()),
            "accepted_false_safe_rate": float(
                test_labels[accepted].mean() if accepted.any() else 0.0
            ),
            "accepted_catastrophic_rate": float(
                test_catastrophic[accepted].mean()
                if accepted.any()
                else 0.0
            ),
            "certifier_batch_runtime_ms": elapsed_ms,
            "certifier_runtime_us_per_query": float(
                1000.0 * elapsed_ms / max(len(test), 1)
            ),
            "seed_metrics": {
                str(seed): metrics
                for seed, metrics in zip(seeds, seed_metrics)
            },
            "seed_mean_metrics": _mean_metrics(seed_metrics),
        }
        folds.append(fold)
        states[held_out] = state.to_dict()

    macro = {
        "accepted_rate": float(np.mean([fold["accepted_rate"] for fold in folds])),
        "accepted_false_safe_rate": float(
            np.mean([fold["accepted_false_safe_rate"] for fold in folds])
        ),
        "held_out_auc": float(np.mean([fold["held_out_auc"] for fold in folds])),
        "hypotheses_reduction": float(
            np.mean(
                [
                    fold["seed_mean_metrics"]["hypotheses_reduction"]
                    for fold in folds
                ]
            )
        ),
        "certifier_runtime_us_per_query": float(
            np.mean(
                [fold["certifier_runtime_us_per_query"] for fold in folds]
            )
        ),
    }
    success = {
        "risk_control": all(
            fold["accepted_false_safe_rate"] <= args.risk_limit
            for fold in folds
        ),
        "nontrivial_acceptance": all(
            fold["accepted_count"] > 0 for fold in folds
        ),
        "hypotheses_reduction_at_least_50_percent": (
            macro["hypotheses_reduction"] >= 0.50
        ),
        "certifier_under_3ms": all(
            fold["certifier_batch_runtime_ms"] < 3.0 for fold in folds
        ),
    }
    success["overall"] = all(success.values())
    payload = {
        "schema": "lafgs_cross_scene_fixed_budget_certifier_v1",
        "decision_scope": "fixed_p1_512_or_all_fallback",
        "ordering_learned": False,
        "descriptor_writeback": False,
        "post_pnp_or_gt_input_features": False,
        "feature_names": CERTIFIER_FEATURE_NAMES,
        "scenes": scenes,
        "seeds": seeds,
        "protocol": {
            "split": "leave_one_scene_out",
            "calibration": "last_25_percent_per_training_trajectory",
            "query_label": "unsafe_if_any_seed_violates_relative_A1_gate",
            "risk_limit": args.risk_limit,
            "confidence": args.confidence,
            "fallback": "A1_all_correspondences",
        },
        "folds": folds,
        "macro": macro,
        "success": success,
        "states": states,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(output), **macro, "success": success}))


if __name__ == "__main__":
    main()
