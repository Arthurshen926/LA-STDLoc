#!/usr/bin/env python3
"""Audit whether fixed equal-energy descriptor fusion can be enabled safely.

This is a mapping-only, CPU-only stopping audit.  It does not search a fusion
weight or a decision threshold.  It reuses the frozen A1 and equal-energy
assignments from the hardened Stairs postmortem, measures their query-policy
oracle, and then evaluates fixed image-level policies under leave-one-sequence-
out (LOSO) transfer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import torch

from localization.pose_solver import pose_error, solve_absolute_pose
from map_learning.context_policy_oracle import (
    pose_policy_loss,
    summarize_policy_oracle,
)
from map_learning.trainer import _pose_error_cm


SCHEMA = "lafgs_equal_energy_descriptor_consensus_stop_audit"
VERSION = 1
LOCKED_SEEDS = (2026, 2027, 2028)
LOCKED_QUERY_COUNT = 256
LOCKED_CLASSIFIER_ALPHA = 1.0
LOCKED_REGRESSOR_ALPHA = 1.0
LOCKED_DECISION_THRESHOLD = 0.0
LOCKED_CONFORMAL_QUANTILE = 0.95
LOCKED_ORACLE_RECOVERY_FRACTION = 1.0 / 3.0
LOCKED_CPU_THREADS = 32
LOCKED_SOLVER = {
    "reprojection_error_px": 12.0,
    "confidence": 0.99999,
    "max_iterations": 100000,
    "min_iterations": 1000,
    "seed": 2026,
    "progressive_sampling": False,
}
FEATURE_NAMES = (
    "superpoint_mean_concentration",
    "superpoint_dimension_variance_max",
    "superpoint_dimension_variance_std",
    "superpoint_mean_projection_std",
    "superpoint_mean_projection_p90_minus_p10",
    "xfeat_mean_concentration",
    "xfeat_dimension_variance_max",
    "xfeat_dimension_variance_std",
    "xfeat_mean_projection_std",
    "xfeat_mean_projection_p90_minus_p10",
    "dense_feature_mean_concentration",
    "dense_feature_dimension_variance_max",
    "dense_feature_dimension_variance_std",
    "dense_feature_mean_projection_std",
    "dense_feature_mean_projection_p90_minus_p10",
    "keypoint_score_mean",
    "keypoint_score_std",
    "keypoint_score_p10",
    "keypoint_score_p90",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: str, *, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    return digest


def _locked_path(path: Path, expected: str, *, label: str) -> tuple[Path, dict]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} is not a file: {resolved}")
    expected = _require_sha256(expected, label=f"{label} SHA-256")
    actual = _sha256(resolved)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return resolved, {
        "path": str(resolved),
        "sha256": actual,
        "expected_sha256": expected,
        "expected_sha256_matches": True,
    }


def _load_torch(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu", weights_only=False)


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()


def _require_gate_record(record: dict, actual: dict, *, label: str) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"formal gate misses {label}")
    if not _same_path(record.get("path", ""), actual["path"]):
        raise ValueError(f"formal gate {label} path differs")
    if record.get("sha256") != actual["sha256"]:
        raise ValueError(f"formal gate {label} SHA-256 differs")
    if record.get("expected_sha256") != actual["sha256"]:
        raise ValueError(f"formal gate {label} expected SHA-256 differs")
    if record.get("expected_sha256_matches") is not True:
        raise ValueError(f"formal gate did not fail-close {label}")


def _validate_lineage(
    *,
    gate: dict,
    postmortem: dict,
    sidecar: np.lib.npyio.NpzFile,
    records: dict[str, dict],
) -> list[int]:
    if (
        gate.get("schema") != "lafgs_mapping_pose_pair_gate"
        or gate.get("version") != 1
        or gate.get("uses_test_queries") is not False
        or gate.get("valid") is not True
        or gate.get("decision", {}).get("verdict") != "STOP"
    ):
        raise ValueError("formal gate is not the valid mapping-only equal-energy STOP")
    protocol = gate.get("preregistered_protocol", {})
    factor = protocol.get("descriptor_factor", {})
    if (
        int(protocol.get("query_count", -1)) != LOCKED_QUERY_COUNT
        or tuple(protocol.get("seeds", ())) != LOCKED_SEEDS
        or int(factor.get("effective_descriptor_dim", -1)) != 320
        or factor.get("formula")
        != "concat(l2(v3_metric(superpoint256)),l2(xfeat64))/sqrt(2)"
    ):
        raise ValueError("formal gate protocol differs from the locked audit")
    checks = gate.get("lineage", {}).get("checks", {})
    if not checks or any(value is not True for value in checks.values()):
        raise ValueError("formal gate lineage checks are incomplete")
    inputs = gate["lineage"]["inputs"]
    for role, gate_key in (
        ("candidate_cache", "variant.query_cache"),
        ("candidate_map", "variant.map"),
        ("candidate_teacher", "variant.teacher"),
    ):
        _require_gate_record(inputs.get(gate_key), records[role], label=gate_key)

    if (
        postmortem.get("schema") != "lafgs_equal_energy_pose_postmortem"
        or postmortem.get("version") != 1
        or postmortem.get("uses_test_queries") is not False
        or postmortem.get("scope") != "mapping_only_q256_frozen_pair"
        or int(postmortem.get("protocol", {}).get("query_count", -1))
        != LOCKED_QUERY_COUNT
        or tuple(postmortem.get("protocol", {}).get("seeds", ())) != LOCKED_SEEDS
    ):
        raise ValueError("postmortem is not the locked mapping-only q256x3 report")
    for role, post_key in (
        ("candidate_cache", "cache"),
        ("candidate_map", "map"),
        ("candidate_teacher", "teacher"),
    ):
        observed = postmortem.get("lineage", {}).get("candidate", {}).get("paths", {}).get(
            post_key, {}
        )
        if not _same_path(observed.get("path", ""), records[role]["path"]):
            raise ValueError(f"postmortem candidate {post_key} path differs")
        if observed.get("sha256") != records[role]["sha256"]:
            raise ValueError(f"postmortem candidate {post_key} SHA-256 differs")
    missing = postmortem.get("missing_primary_sidecar", {})
    if not _same_path(missing.get("reconstructed_sidecar", ""), records["sidecar"]["path"]):
        raise ValueError("postmortem reconstructed sidecar path differs")
    if missing.get("reconstructed_sidecar_sha256") != records["sidecar"]["sha256"]:
        raise ValueError("postmortem reconstructed sidecar SHA-256 differs")

    selected = [int(value) for value in postmortem["protocol"]["selected_query_indices"]]
    if len(selected) != LOCKED_QUERY_COUNT or len(set(selected)) != len(selected):
        raise ValueError("postmortem selected-query registry differs")
    for arm in ("baseline", "variant"):
        gate_selected = gate["lineage"]["arms"][arm]["uniform_q256_indices"]
        if [int(value) for value in gate_selected] != selected:
            raise ValueError(f"formal gate {arm} query registry differs")
    if not np.array_equal(
        np.asarray(sidecar["selected_query_indices"], dtype=np.int64),
        np.asarray(selected, dtype=np.int64),
    ):
        raise ValueError("sidecar selected-query registry differs")
    offsets = np.asarray(sidecar["query_row_offsets"], dtype=np.int64)
    if offsets.shape != (LOCKED_QUERY_COUNT + 1,) or offsets[0] != 0:
        raise ValueError("sidecar query-row offsets differ")
    row_count = int(offsets[-1])
    for key in (
        "baseline_winners",
        "baseline_correct",
        "candidate_winners",
        "candidate_correct",
    ):
        if np.asarray(sidecar[key]).shape != (row_count,):
            raise ValueError(f"sidecar {key} does not align with query rows")
    return selected


def _descriptor_factor_is_mapping_only(payload: dict, *, label: str) -> None:
    factor = payload.get("descriptor_factor", {})
    if (
        factor.get("schema") != "lafgs_mapping_equal_energy_descriptor_factor"
        or factor.get("mapping_only") is not True
        or factor.get("uses_test_queries") is not False
    ):
        raise ValueError(f"{label} descriptor factor is not mapping-only")


def _branch_statistics(values: torch.Tensor) -> list[float]:
    rows = torch.as_tensor(values).float()
    if rows.ndim != 2 or rows.shape[0] < 2:
        raise ValueError("feature statistics require at least two descriptor rows")
    mean = rows.mean(dim=0)
    projections = rows @ mean
    variances = rows.var(dim=0, unbiased=False)
    return [
        float(torch.linalg.norm(mean)),
        float(variances.max()),
        float(variances.std(unbiased=True)),
        float(projections.std(unbiased=True)),
        float(torch.quantile(projections, 0.9) - torch.quantile(projections, 0.1)),
    ]


def extract_query_features(record: dict, rows: torch.Tensor) -> np.ndarray:
    """Return the locked 19 retrieval-before-policy image statistics."""
    selected = torch.as_tensor(rows).long().reshape(-1)
    descriptor = torch.as_tensor(record["native_descriptors"]).float()[selected]
    if descriptor.ndim != 2 or descriptor.shape[1] != 320:
        raise ValueError("candidate cache must contain the locked 320D descriptor")
    # The frozen candidate stores both unit branches divided by sqrt(2).
    superpoint = descriptor[:, :256] * math.sqrt(2.0)
    xfeat = descriptor[:, 256:] * math.sqrt(2.0)
    for name, branch in (("SuperPoint", superpoint), ("XFeat", xfeat)):
        if not torch.allclose(
            torch.linalg.norm(branch, dim=1),
            torch.ones(branch.shape[0]),
            atol=3e-5,
            rtol=0.0,
        ):
            raise ValueError(f"{name} branch is not unit normalized")
    dense = torch.as_tensor(record["feature_map"]).float()
    if dense.ndim != 3 or dense.shape[0] != 256:
        raise ValueError("candidate cache dense feature map must have shape [256,H,W]")
    dense_rows = dense.flatten(start_dim=1).T
    scores = torch.as_tensor(record["native_scores"]).float()[selected]
    if scores.numel() < 2:
        raise ValueError("query requires at least two keypoint scores")
    output = (
        _branch_statistics(superpoint)
        + _branch_statistics(xfeat)
        + _branch_statistics(dense_rows)
        + [
            float(scores.mean()),
            float(scores.std(unbiased=True)),
            float(torch.quantile(scores, 0.1)),
            float(torch.quantile(scores, 0.9)),
        ]
    )
    if len(output) != len(FEATURE_NAMES) or not np.isfinite(output).all():
        raise ValueError("locked query feature vector is invalid")
    return np.asarray(output, dtype=np.float64)


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def _fit_ridge(train: np.ndarray, target: np.ndarray, alpha: float) -> Callable:
    x = np.asarray(train, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    centered = x - x_mean
    coefficient = np.linalg.solve(
        centered.T @ centered + float(alpha) * np.eye(x.shape[1]),
        centered.T @ (y - y_mean),
    )
    intercept = y_mean - float(x_mean @ coefficient)
    return lambda values: np.asarray(values, dtype=np.float64) @ coefficient + intercept


def _fit_logistic(
    train: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray,
    alpha: float,
) -> Callable:
    x = np.asarray(train, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    design = np.column_stack((x, np.ones(x.shape[0], dtype=np.float64)))

    def objective(parameter: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ parameter
        loss = float(np.sum(weights * (np.logaddexp(0.0, logits) - y * logits)))
        loss += 0.5 * float(alpha) * float(parameter[:-1] @ parameter[:-1])
        gradient = design.T @ (weights * (expit(logits) - y))
        gradient[:-1] += float(alpha) * parameter[:-1]
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(design.shape[1], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"fixed logistic fit failed: {result.message}")
    parameter = np.asarray(result.x, dtype=np.float64)
    return lambda values: np.asarray(values, dtype=np.float64) @ parameter[:-1] + parameter[-1]


def _loso_predictions(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    kind: str,
) -> tuple[np.ndarray, list[dict]]:
    unique = sorted(set(str(value) for value in groups))
    if len(unique) < 4:
        raise ValueError("strict LOSO audit requires at least four sequences")
    predictions = np.zeros(target.shape[0], dtype=np.float64)
    folds = []
    for held_out in unique:
        test = groups == held_out
        train = ~test
        x_train, x_test = _standardize(features[train], features[test])
        if kind == "regression":
            model = _fit_ridge(x_train, target[train], LOCKED_REGRESSOR_ALPHA)
        elif kind == "classification":
            advantage = np.abs(np.asarray(target, dtype=np.float64)[train])
            weights = advantage / max(float(advantage.mean()), 1e-12)
            labels = (np.asarray(target, dtype=np.float64)[train] > 0.0).astype(float)
            model = _fit_logistic(
                x_train,
                labels,
                weights,
                LOCKED_CLASSIFIER_ALPHA,
            )
        else:
            raise ValueError(f"unsupported LOSO model: {kind}")
        predictions[test] = model(x_test)
        folds.append(
            {
                "held_out_sequence": held_out,
                "support_query_count": int(train.sum()),
                "held_out_query_count": int(test.sum()),
            }
        )
    return predictions, folds


def _conformal_lower_bounds(
    features: np.ndarray,
    advantage: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    predictions = np.zeros(advantage.shape[0], dtype=np.float64)
    lower = np.zeros_like(predictions)
    folds = []
    unique = sorted(set(str(value) for value in groups))
    for outer in unique:
        support = groups != outer
        test = ~support
        residuals = []
        for inner in sorted(set(str(value) for value in groups[support])):
            inner_train = support & (groups != inner)
            inner_test = support & (groups == inner)
            x_train, x_test = _standardize(
                features[inner_train], features[inner_test]
            )
            model = _fit_ridge(
                x_train,
                advantage[inner_train],
                LOCKED_REGRESSOR_ALPHA,
            )
            residuals.extend((model(x_test) - advantage[inner_test]).tolist())
        quantile = float(
            np.quantile(
                np.asarray(residuals, dtype=np.float64),
                LOCKED_CONFORMAL_QUANTILE,
                method="higher",
            )
        )
        x_train, x_test = _standardize(features[support], features[test])
        model = _fit_ridge(
            x_train,
            advantage[support],
            LOCKED_REGRESSOR_ALPHA,
        )
        predictions[test] = model(x_test)
        lower[test] = predictions[test] - quantile
        folds.append(
            {
                "held_out_sequence": outer,
                "support_only_error_p95": quantile,
                "enabled_query_count": int((lower[test] > 0.0).sum()),
            }
        )
    return predictions, lower, folds


def _rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=bool)
    values = np.asarray(scores, dtype=np.float64)
    positive = int(y.sum())
    negative = int((~y).sum())
    if not positive or not negative:
        raise ValueError("AUC requires both classes")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    cursor = 0
    while cursor < values.size:
        end = cursor + 1
        while end < values.size and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * (cursor + 1 + end)
        cursor = end
    return float((ranks[y].sum() - positive * (positive + 1) / 2) / (positive * negative))


def _balanced_accuracy(labels: np.ndarray, choices: np.ndarray) -> float:
    y = np.asarray(labels, dtype=bool)
    selected = np.asarray(choices, dtype=bool)
    return float(0.5 * (selected[y].mean() + (~selected[~y]).mean()))


def _pose_summary(postmortem: dict, choices: np.ndarray, seeds: tuple[int, ...]) -> dict:
    te, ae, hypotheses = [], [], []
    for seed in seeds:
        pose = postmortem["pose"][str(seed)]
        for index, selected in enumerate(choices):
            row = pose["candidate" if selected else "baseline"][index]
            te.append(float(row["te_cm"]))
            ae.append(float(row["ae_deg"]))
            hypotheses.append(int(row["hypotheses"]))
    translation = np.asarray(te, dtype=np.float64)
    rotation = np.asarray(ae, dtype=np.float64)
    tail = max(int(math.ceil(0.05 * translation.size)), 1)
    return {
        "query_seed_count": int(translation.size),
        "median_te_cm": float(np.median(translation)),
        "mean_te_cm": float(translation.mean()),
        "p90_te_cm": float(np.percentile(translation, 90)),
        "cvar95_te_cm": float(np.sort(translation)[-tail:].mean()),
        "median_ae_deg": float(np.median(rotation)),
        "mean_ae_deg": float(rotation.mean()),
        "p90_ae_deg": float(np.percentile(rotation, 90)),
        "recall_5cm_5deg_percent": float(
            100.0 * np.mean((translation < 5.0) & (rotation < 5.0))
        ),
        "catastrophic_100cm_count": int(np.count_nonzero(translation >= 100.0)),
        "mean_hypotheses": float(np.mean(hypotheses)),
    }


def _raw_precision(
    postmortem: dict,
    choices: np.ndarray,
) -> dict:
    rows = postmortem["per_query_matching"]
    total = int(sum(int(row["row_count"]) for row in rows))
    baseline = int(sum(int(row["baseline_correct_count"]) for row in rows))
    selected = int(
        sum(
            int(row["candidate_correct_count"] if choices[index] else row["baseline_correct_count"])
            for index, row in enumerate(rows)
        )
    )
    return {
        "row_count": total,
        "correct_count": selected,
        "precision_percent": float(100.0 * selected / total),
        "delta_from_a1_pp": float(100.0 * (selected - baseline) / total),
    }


def _policy_report(
    *,
    name: str,
    predictions: np.ndarray,
    choices: np.ndarray,
    advantages: np.ndarray,
    a1_risk: np.ndarray,
    equal_risk: np.ndarray,
    groups: np.ndarray,
    postmortem: dict,
    oracle_risk: float,
) -> dict:
    hybrid_risk = np.where(choices, equal_risk, a1_risk)
    headroom = float(a1_risk.mean() - oracle_risk)
    a1_choices = np.zeros(choices.shape, dtype=bool)
    pooled_pose = _pose_summary(postmortem, choices, LOCKED_SEEDS)
    baseline_pose = _pose_summary(postmortem, a1_choices, LOCKED_SEEDS)
    per_seed = {
        str(seed): {
            "a1": _pose_summary(postmortem, a1_choices, (seed,)),
            "policy": _pose_summary(postmortem, choices, (seed,)),
        }
        for seed in LOCKED_SEEDS
    }
    outer = []
    for group in sorted(set(str(value) for value in groups)):
        mask = groups == group
        local_choices = choices.copy()
        # Summaries operate on all 256 query positions.  Select the relevant
        # rows directly here to avoid relabelling their frozen registry.
        def subset_summary(selected: np.ndarray) -> dict:
            te, ae, hypotheses = [], [], []
            for seed in LOCKED_SEEDS:
                pose = postmortem["pose"][str(seed)]
                for index in np.flatnonzero(mask):
                    row = pose["candidate" if selected[index] else "baseline"][index]
                    te.append(float(row["te_cm"]))
                    ae.append(float(row["ae_deg"]))
                    hypotheses.append(int(row["hypotheses"]))
            translation = np.asarray(te)
            rotation = np.asarray(ae)
            tail = max(int(math.ceil(0.05 * translation.size)), 1)
            return {
                "mean_te_cm": float(translation.mean()),
                "p90_te_cm": float(np.percentile(translation, 90)),
                "cvar95_te_cm": float(np.sort(translation)[-tail:].mean()),
                "mean_ae_deg": float(rotation.mean()),
                "recall_5cm_5deg_percent": float(
                    100.0 * np.mean((translation < 5.0) & (rotation < 5.0))
                ),
                "catastrophic_100cm_count": int(
                    np.count_nonzero(translation >= 100.0)
                ),
                "mean_hypotheses": float(np.mean(hypotheses)),
            }

        a1_pose = subset_summary(a1_choices)
        policy_pose = subset_summary(local_choices)
        non_regression = {
            "risk": float(hybrid_risk[mask].mean()) <= float(a1_risk[mask].mean()),
            "mean_te": policy_pose["mean_te_cm"] <= a1_pose["mean_te_cm"],
            "p90_te": policy_pose["p90_te_cm"] <= a1_pose["p90_te_cm"],
            "cvar95_te": policy_pose["cvar95_te_cm"] <= a1_pose["cvar95_te_cm"],
            "recall_5cm_5deg": policy_pose["recall_5cm_5deg_percent"]
            >= a1_pose["recall_5cm_5deg_percent"],
            "catastrophic_100cm": policy_pose["catastrophic_100cm_count"]
            <= a1_pose["catastrophic_100cm_count"],
        }
        outer.append(
            {
                "held_out_sequence": group,
                "query_count": int(mask.sum()),
                "enabled_query_count": int(choices[mask].sum()),
                "a1_risk": float(a1_risk[mask].mean()),
                "equal_energy_risk": float(equal_risk[mask].mean()),
                "policy_risk": float(hybrid_risk[mask].mean()),
                "a1_pose": a1_pose,
                "policy_pose": policy_pose,
                "non_regression": non_regression,
                "all_non_regression": bool(all(non_regression.values())),
            }
        )
    recovery = float((a1_risk.mean() - hybrid_risk.mean()) / max(headroom, 1e-12))
    return {
        "name": name,
        "enabled_query_count": int(choices.sum()),
        "selection_fraction": float(choices.mean()),
        "a1_risk": float(a1_risk.mean()),
        "policy_risk": float(hybrid_risk.mean()),
        "oracle_headroom_recovered_fraction": recovery,
        "advantage_sign_auc": _rank_auc(advantages > 0.0, predictions),
        "advantage_sign_balanced_accuracy": _balanced_accuracy(
            advantages > 0.0, choices
        ),
        "pose": {"a1": baseline_pose, "policy": pooled_pose},
        "per_seed": per_seed,
        "raw_precision": _raw_precision(postmortem, choices),
        "outer_folds": outer,
        "outer_fold_pass_count": int(sum(row["all_non_regression"] for row in outer)),
        "outer_fold_count": len(outer),
        "passes_all_outer_folds": bool(all(row["all_non_regression"] for row in outer)),
        "passes_minimum_oracle_recovery": recovery >= LOCKED_ORACLE_RECOVERY_FRACTION,
    }


def restricted_agreement_choices(
    *,
    query_superpoint: torch.Tensor,
    query_xfeat: torch.Tensor,
    map_superpoint: torch.Tensor,
    map_xfeat: torch.Tensor,
    baseline_winners: np.ndarray,
    candidate_winners: np.ndarray,
) -> np.ndarray:
    """Choose between the two frozen winners with a parameter-free min score."""
    baseline = torch.as_tensor(baseline_winners).long()
    candidate = torch.as_tensor(candidate_winners).long()
    sp_baseline = (query_superpoint * map_superpoint[baseline]).sum(dim=1)
    sp_candidate = (query_superpoint * map_superpoint[candidate]).sum(dim=1)
    xf_baseline = (query_xfeat * map_xfeat[baseline]).sum(dim=1)
    xf_candidate = (query_xfeat * map_xfeat[candidate]).sum(dim=1)
    return (
        torch.minimum(sp_candidate, xf_candidate)
        > torch.minimum(sp_baseline, xf_baseline)
    ).cpu().numpy()


def _agreement_pose(
    *,
    choices: np.ndarray,
    sidecar: np.lib.npyio.NpzFile,
    cache: dict,
    teacher: dict,
    state: dict,
    selected: list[int],
) -> dict:
    offsets = np.asarray(sidecar["query_row_offsets"], dtype=np.int64)
    baseline = np.asarray(sidecar["baseline_winners"], dtype=np.int64)
    candidate = np.asarray(sidecar["candidate_winners"], dtype=np.int64)
    winners = np.where(choices, candidate, baseline)
    names = list(teacher["query_names"])
    queries = cache.get("queries", cache)
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    te, ae, hypotheses = [], [], []
    for local_index, query_index in enumerate(selected):
        name = names[query_index]
        record = teacher["records"][query_index]
        rows = torch.as_tensor(record["query_rows"]).long()
        cached = queries[name]
        segment = slice(offsets[local_index], offsets[local_index + 1])
        if rows.numel() != offsets[local_index + 1] - offsets[local_index]:
            raise ValueError("agreement rows differ from sidecar")
        keypoints = (
            torch.as_tensor(cached["native_keypoints"]).float()[rows]
            + float(cached.get("pixel_center_offset", 0.5))
        )
        intrinsic = torch.as_tensor(cached["native_K"]).float()
        gt_pose = torch.as_tensor(cached["pose_w2c"]).float()
        estimate = solve_absolute_pose(
            keypoints.numpy(),
            xyz[torch.as_tensor(winners[segment]).long()].numpy(),
            intrinsic.numpy(),
            reprojection_error_px=float(LOCKED_SOLVER["reprojection_error_px"]),
            confidence=float(LOCKED_SOLVER["confidence"]),
            max_iterations=int(LOCKED_SOLVER["max_iterations"]),
            min_iterations=int(LOCKED_SOLVER["min_iterations"]),
            seed=int(LOCKED_SOLVER["seed"]),
            progressive_sampling=bool(LOCKED_SOLVER["progressive_sampling"]),
        )
        rotation, _ = pose_error(estimate.pose_w2c, gt_pose.numpy())
        te.append(float(_pose_error_cm(estimate.pose_w2c, gt_pose)))
        ae.append(float(rotation))
        hypotheses.append(int(estimate.diagnostics.get("iterations", 0)))
        if (local_index + 1) % 32 == 0:
            print(
                json.dumps(
                    {
                        "event": "descriptor_consensus_agreement_pose",
                        "queries_complete": local_index + 1,
                        "query_count": len(selected),
                    }
                ),
                flush=True,
            )
    translation = np.asarray(te)
    rotation = np.asarray(ae)
    tail = max(int(math.ceil(0.05 * translation.size)), 1)
    return {
        "seed": int(LOCKED_SOLVER["seed"]),
        "median_te_cm": float(np.median(translation)),
        "mean_te_cm": float(translation.mean()),
        "p90_te_cm": float(np.percentile(translation, 90)),
        "cvar95_te_cm": float(np.sort(translation)[-tail:].mean()),
        "median_ae_deg": float(np.median(rotation)),
        "mean_ae_deg": float(rotation.mean()),
        "p90_ae_deg": float(np.percentile(rotation, 90)),
        "recall_5cm_5deg_percent": float(
            100.0 * np.mean((translation < 5.0) & (rotation < 5.0))
        ),
        "catastrophic_100cm_count": int(np.count_nonzero(translation >= 100.0)),
        "mean_hypotheses": float(np.mean(hypotheses)),
        "maximum_te_cm": float(translation.max()),
        "failure_query_indices": [
            int(selected[index]) for index in np.flatnonzero(translation >= 5.0)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "formal_pose_gate",
        "postmortem_report",
        "sidecar",
        "candidate_query_cache",
        "candidate_map",
        "candidate_teacher",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
        parser.add_argument(
            f"--expected-{name.replace('_', '-')}-sha256", required=True
        )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.is_initialized():
        raise RuntimeError("descriptor consensus audit must run before CUDA initialization")
    torch.set_num_threads(LOCKED_CPU_THREADS)

    records = {}
    for name in (
        "formal_pose_gate",
        "postmortem_report",
        "sidecar",
        "candidate_query_cache",
        "candidate_map",
        "candidate_teacher",
    ):
        path, record = _locked_path(
            getattr(args, name),
            getattr(args, f"expected_{name}_sha256"),
            label=name.replace("_", " "),
        )
        records[name] = record
        records[name]["resolved"] = path

    gate = json.loads(records["formal_pose_gate"]["resolved"].read_text())
    postmortem = json.loads(records["postmortem_report"]["resolved"].read_text())
    sidecar = np.load(records["sidecar"]["resolved"], allow_pickle=False)
    lineage_records = {
        "candidate_cache": records["candidate_query_cache"],
        "candidate_map": records["candidate_map"],
        "candidate_teacher": records["candidate_teacher"],
        "sidecar": records["sidecar"],
    }
    selected = _validate_lineage(
        gate=gate,
        postmortem=postmortem,
        sidecar=sidecar,
        records=lineage_records,
    )
    cache = _load_torch(records["candidate_query_cache"]["resolved"])
    state = _load_torch(records["candidate_map"]["resolved"])
    teacher = _load_torch(records["candidate_teacher"]["resolved"])
    _descriptor_factor_is_mapping_only(cache, label="candidate query cache")
    _descriptor_factor_is_mapping_only(state, label="candidate map")
    if list(cache.get("queries", cache)) != list(teacher["query_names"]):
        raise ValueError("candidate cache and teacher query registries differ")

    pose_rows = []
    a1_risk, equal_risk = [], []
    for seed in LOCKED_SEEDS:
        paired = postmortem["pose"][str(seed)]
        if len(paired["baseline"]) != LOCKED_QUERY_COUNT or len(
            paired["candidate"]
        ) != LOCKED_QUERY_COUNT:
            raise ValueError("postmortem pose rows differ from q256")
        for index, (baseline_row, candidate_row) in enumerate(
            zip(paired["baseline"], paired["candidate"])
        ):
            pose_rows.append(
                {
                    "query_index": index,
                    "image_name": baseline_row["image_name"],
                    "seed": seed,
                    "a1_te_cm": baseline_row["te_cm"],
                    "a1_ae_deg": baseline_row["ae_deg"],
                    "a1_hypotheses": baseline_row["hypotheses"],
                    "a1_failed": False,
                    "equal_energy_te_cm": candidate_row["te_cm"],
                    "equal_energy_ae_deg": candidate_row["ae_deg"],
                    "equal_energy_hypotheses": candidate_row["hypotheses"],
                    "equal_energy_failed": False,
                }
            )
    for index in range(LOCKED_QUERY_COUNT):
        baseline_losses, candidate_losses = [], []
        for seed in LOCKED_SEEDS:
            baseline_row = postmortem["pose"][str(seed)]["baseline"][index]
            candidate_row = postmortem["pose"][str(seed)]["candidate"][index]
            baseline_losses.append(
                pose_policy_loss(
                    te_cm=baseline_row["te_cm"],
                    ae_deg=baseline_row["ae_deg"],
                    hypotheses=baseline_row["hypotheses"],
                )
            )
            candidate_losses.append(
                pose_policy_loss(
                    te_cm=candidate_row["te_cm"],
                    ae_deg=candidate_row["ae_deg"],
                    hypotheses=candidate_row["hypotheses"],
                )
            )
        a1_risk.append(float(np.mean(baseline_losses)))
        equal_risk.append(float(np.mean(candidate_losses)))
    a1_risk = np.asarray(a1_risk)
    equal_risk = np.asarray(equal_risk)
    advantage = a1_risk - equal_risk
    oracle = summarize_policy_oracle(
        pose_rows,
        context_protocol="equal_energy",
        seeds=LOCKED_SEEDS,
        bootstrap_samples=10000,
        bootstrap_seed=2026,
    )

    features, groups, query_rows = [], [], []
    queries = cache.get("queries", cache)
    names = list(teacher["query_names"])
    offsets = np.asarray(sidecar["query_row_offsets"], dtype=np.int64)
    map_features = torch.as_tensor(state["anchor_features"]).float() * math.sqrt(2.0)
    if map_features.ndim != 2 or map_features.shape[1] != 320:
        raise ValueError("candidate map does not contain a 320D bank")
    agreement_parts = []
    for local_index, query_index in enumerate(selected):
        name = names[query_index]
        record = teacher["records"][query_index]
        rows = torch.as_tensor(record["query_rows"]).long()
        cached = queries[name]
        if postmortem["per_query_matching"][local_index]["image_name"] != name:
            raise ValueError("postmortem and teacher query names differ")
        features.append(extract_query_features(cached, rows))
        groups.append(name.split("/", 1)[0])
        segment = slice(offsets[local_index], offsets[local_index + 1])
        descriptor = (
            torch.as_tensor(cached["native_descriptors"]).float()[rows]
            * math.sqrt(2.0)
        )
        agreement_parts.append(
            restricted_agreement_choices(
                query_superpoint=descriptor[:, :256],
                query_xfeat=descriptor[:, 256:],
                map_superpoint=map_features[:, :256],
                map_xfeat=map_features[:, 256:],
                baseline_winners=np.asarray(sidecar["baseline_winners"])[segment],
                candidate_winners=np.asarray(sidecar["candidate_winners"])[segment],
            )
        )
        query_rows.append(
            {
                "local_query_index": local_index,
                "mapping_query_index": query_index,
                "image_name": name,
                "sequence": groups[-1],
                "a1_risk": float(a1_risk[local_index]),
                "equal_energy_risk": float(equal_risk[local_index]),
                "equal_energy_advantage": float(advantage[local_index]),
            }
        )
        if (local_index + 1) % 64 == 0:
            print(
                json.dumps(
                    {
                        "event": "descriptor_consensus_features",
                        "queries_complete": local_index + 1,
                        "query_count": len(selected),
                    }
                ),
                flush=True,
            )
    features = np.asarray(features)
    groups = np.asarray(groups)

    classification_prediction, classification_folds = _loso_predictions(
        features,
        advantage,
        groups,
        kind="classification",
    )
    regression_prediction, regression_folds = _loso_predictions(
        features,
        advantage,
        groups,
        kind="regression",
    )
    conformal_prediction, conformal_lower, conformal_folds = _conformal_lower_bounds(
        features,
        advantage,
        groups,
    )
    policies = {
        "loso_logistic_advantage_sign": _policy_report(
            name="loso_logistic_advantage_sign",
            predictions=classification_prediction,
            choices=classification_prediction > LOCKED_DECISION_THRESHOLD,
            advantages=advantage,
            a1_risk=a1_risk,
            equal_risk=equal_risk,
            groups=groups,
            postmortem=postmortem,
            oracle_risk=float(oracle["oracle_risk"]),
        ),
        "loso_ridge_continuous_advantage": _policy_report(
            name="loso_ridge_continuous_advantage",
            predictions=regression_prediction,
            choices=regression_prediction > LOCKED_DECISION_THRESHOLD,
            advantages=advantage,
            a1_risk=a1_risk,
            equal_risk=equal_risk,
            groups=groups,
            postmortem=postmortem,
            oracle_risk=float(oracle["oracle_risk"]),
        ),
        "nested_loso_conformal_lower_bound": _policy_report(
            name="nested_loso_conformal_lower_bound",
            predictions=conformal_lower,
            choices=conformal_lower > LOCKED_DECISION_THRESHOLD,
            advantages=advantage,
            a1_risk=a1_risk,
            equal_risk=equal_risk,
            groups=groups,
            postmortem=postmortem,
            oracle_risk=float(oracle["oracle_risk"]),
        ),
    }
    policies["loso_logistic_advantage_sign"]["fit_folds"] = classification_folds
    policies["loso_ridge_continuous_advantage"]["fit_folds"] = regression_folds
    policies["nested_loso_conformal_lower_bound"]["fit_folds"] = conformal_folds

    classification_choice = classification_prediction > 0.0
    regression_choice = regression_prediction > 0.0
    conformal_choice = conformal_lower > 0.0
    for index, row in enumerate(query_rows):
        row.update(
            {
                "classification_logit": float(classification_prediction[index]),
                "classification_enable": bool(classification_choice[index]),
                "regression_predicted_advantage": float(regression_prediction[index]),
                "regression_enable": bool(regression_choice[index]),
                "conformal_predicted_advantage": float(conformal_prediction[index]),
                "conformal_lower_bound": float(conformal_lower[index]),
                "conformal_enable": bool(conformal_choice[index]),
            }
        )

    agreement_row_choice = np.concatenate(agreement_parts)
    baseline_winners = np.asarray(sidecar["baseline_winners"])
    candidate_winners = np.asarray(sidecar["candidate_winners"])
    baseline_correct = np.asarray(sidecar["baseline_correct"], dtype=bool)
    candidate_correct = np.asarray(sidecar["candidate_correct"], dtype=bool)
    agreement_winners = np.where(
        agreement_row_choice, candidate_winners, baseline_winners
    )
    agreement_correct = np.where(
        agreement_row_choice, candidate_correct, baseline_correct
    )
    agreement = {
        "scope": "restricted_choice_between_existing_a1_and_equal_energy_winners",
        "score": "min(superpoint_cosine,xfeat_cosine)",
        "global_min_retrieval_claimed": False,
        "candidate_choice_row_count": int(agreement_row_choice.sum()),
        "changed_from_a1_row_count": int(
            np.count_nonzero(agreement_winners != baseline_winners)
        ),
        "correct_count": int(agreement_correct.sum()),
        "raw_precision_percent": float(100.0 * agreement_correct.mean()),
        "raw_precision_delta_from_a1_pp": float(
            100.0 * (agreement_correct.mean() - baseline_correct.mean())
        ),
        "solver_protocol": LOCKED_SOLVER,
        "pose": _agreement_pose(
            choices=agreement_row_choice,
            sidecar=sidecar,
            cache=cache,
            teacher=teacher,
            state=state,
            selected=selected,
        ),
    }
    a1_seed2026 = _pose_summary(
        postmortem,
        np.zeros(LOCKED_QUERY_COUNT, dtype=bool),
        (2026,),
    )
    agreement_checks = {
        "mean_te_non_regression": agreement["pose"]["mean_te_cm"]
        <= a1_seed2026["mean_te_cm"],
        "p90_te_non_regression": agreement["pose"]["p90_te_cm"]
        <= a1_seed2026["p90_te_cm"],
        "cvar95_te_non_regression": agreement["pose"]["cvar95_te_cm"]
        <= a1_seed2026["cvar95_te_cm"],
        "recall_non_regression": agreement["pose"]["recall_5cm_5deg_percent"]
        >= a1_seed2026["recall_5cm_5deg_percent"],
        "catastrophe_non_regression": agreement["pose"]["catastrophic_100cm_count"]
        <= a1_seed2026["catastrophic_100cm_count"],
    }
    agreement["a1_pose_seed2026"] = a1_seed2026
    agreement["non_regression"] = agreement_checks
    agreement["passes"] = bool(all(agreement_checks.values()))

    regression = policies["loso_ridge_continuous_advantage"]
    decision_checks = {
        "mapping_only_and_test_free": True,
        "oracle_headroom_is_nonzero": float(oracle["oracle_headroom_absolute"]) > 0.0,
        "regression_recovers_at_least_one_third_oracle": regression[
            "passes_minimum_oracle_recovery"
        ],
        "regression_passes_all_outer_sequences": regression[
            "passes_all_outer_folds"
        ],
        "conformal_recovers_at_least_one_third_oracle": policies[
            "nested_loso_conformal_lower_bound"
        ]["passes_minimum_oracle_recovery"],
        "agreement_non_regression": agreement["passes"],
    }
    output = {
        "schema": SCHEMA,
        "version": VERSION,
        "scope": "mapping_only_stairs_q256x3_cpu_postmortem",
        "uses_test_queries": False,
        "device": "cpu",
        "inputs": {
            key: {field: value for field, value in record.items() if field != "resolved"}
            for key, record in records.items()
        },
        "code": {
            "entrypoint": "scripts/audit_equal_energy_descriptor_consensus.py",
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "protocol": {
            "query_count": LOCKED_QUERY_COUNT,
            "seeds": list(LOCKED_SEEDS),
            "sequence_group": "first_path_component",
            "feature_count": len(FEATURE_NAMES),
            "feature_names": list(FEATURE_NAMES),
            "features_available_before_retrieval": True,
            "classification": {
                "model": "weighted_l2_logistic",
                "alpha": LOCKED_CLASSIFIER_ALPHA,
                "sample_weight": "absolute_pose_policy_advantage_normalized_on_support",
                "decision_threshold": LOCKED_DECISION_THRESHOLD,
            },
            "regression": {
                "model": "l2_ridge_continuous_pose_policy_advantage",
                "alpha": LOCKED_REGRESSOR_ALPHA,
                "decision_threshold": LOCKED_DECISION_THRESHOLD,
            },
            "conformal": {
                "calibration": "inner_sequence_loso_on_outer_support_only",
                "one_sided_error_quantile": LOCKED_CONFORMAL_QUANTILE,
                "quantile_method": "higher",
                "decision": "enable_only_if_advantage_lower_bound_gt_zero",
            },
            "threshold_or_weight_search": False,
            "fusion_policies": ["a1_alpha_0", "equal_energy_alpha_0p5"],
            "minimum_oracle_recovery_fraction": LOCKED_ORACLE_RECOVERY_FRACTION,
        },
        "oracle": oracle,
        "policies": policies,
        "agreement_audit": agreement,
        "decision": {
            "verdict": "STOP_DESCRIPTOR_FUSION",
            "checks": decision_checks,
            "failed_checks": [
                name for name, passed in decision_checks.items() if not passed
            ],
            "reason": (
                "continuous advantage has pooled signal but fails the preregistered "
                "4/4 held-out-sequence non-regression gate; conformal safety removes "
                "the required oracle recovery and parameter-free agreement still "
                "creates a coherent pose tail"
            ),
            "do_not_advance": [
                "production_query_gate",
                "fusion_weight_or_threshold_sweep",
                "office2_5b",
                "outdoor_guard",
                "formal_test",
            ],
            "p8_is_independent_and_not_a_descriptor_remedy": True,
        },
        "query_predictions": query_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "event": "descriptor_consensus_stop_audit_complete",
                "output": str(args.output.resolve()),
                "verdict": output["decision"]["verdict"],
                "failed_checks": output["decision"]["failed_checks"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
