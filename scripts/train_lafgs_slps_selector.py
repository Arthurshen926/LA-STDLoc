#!/usr/bin/env python3
"""Train a scene-specific SLPS selector from exact PoseLib set outcomes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from localization_training.slps_selector import (
    RELATION_NAMES,
    SLPS_BIAS_AWARE_FEATURE_NAMES,
    SLPS_FEATURE_NAMES,
    SLPSModelConfig,
    SLPSSelector,
    normalize_relation_groups,
)


def _atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _balanced_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.float()
    positive = labels.sum()
    negative = len(labels) - positive
    if positive <= 0 or negative <= 0:
        return F.binary_cross_entropy_with_logits(logits, labels)
    positive_weight = (negative / positive).clamp(0.25, 20.0)
    return F.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=positive_weight
    )


def _pinball_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    error = target - prediction
    value = float(quantile)
    return torch.maximum(value * error, (value - 1.0) * error).mean()


def _examples_for_seed(query: dict, seed: int) -> list[tuple[dict, dict]]:
    examples = []
    for subset in query["subsets"]:
        for outcome in subset["outcomes"]:
            if int(outcome["seed"]) == int(seed):
                examples.append((subset, outcome))
    return examples


def _sample_training_examples(
    query: dict,
    *,
    seed: int,
    generator: random.Random,
    maximum_sets: int,
) -> list[tuple[dict, dict]]:
    """Keep deployment-style self-mined sets visible in every update."""

    all_examples = _examples_for_seed(query, seed)
    maximum = max(int(maximum_sets), 1)
    if len(all_examples) <= maximum:
        return all_examples
    baseline = next(
        (value for value in all_examples if value[0]["name"] == "all"),
        None,
    )
    learned = [
        value
        for value in all_examples
        if str(value[0]["name"]).startswith("learned_")
    ]
    static = [
        value
        for value in all_examples
        if value is not baseline and value not in learned
    ]
    selected = []
    if baseline is not None:
        selected.append(baseline)
    learned_budget = min(
        len(learned),
        max((maximum - len(selected)) * 2 // 3, 1),
    )
    if learned_budget:
        selected.extend(generator.sample(learned, learned_budget))
    remaining = maximum - len(selected)
    if remaining > 0:
        pool = [value for value in static if value not in selected]
        selected.extend(generator.sample(pool, min(remaining, len(pool))))
    remaining = maximum - len(selected)
    if remaining > 0:
        pool = [value for value in all_examples if value not in selected]
        selected.extend(generator.sample(pool, min(remaining, len(pool))))
    return selected


def _set_mask(
    subset: dict, count: int, device: torch.device
) -> torch.Tensor:
    mask = torch.zeros(count, dtype=torch.bool, device=device)
    mask[torch.as_tensor(subset["indices"], device=device).long()] = True
    return mask


def _query_loss(
    model: SLPSSelector,
    query: dict,
    *,
    generator: random.Random,
    maximum_sets: int,
    bias_loss_weight: float = 0.0,
    residual_utility_regularization_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = model.feature_mean.device
    features = query["features"].to(device=device, dtype=torch.float32)
    relation_groups = query["relation_groups"].to(device=device)
    encoded = model.encode(features, relation_groups)
    available_seeds = sorted(
        {
            int(outcome["seed"])
            for subset in query["subsets"]
            for outcome in subset["outcomes"]
        }
    )
    seed = generator.choice(available_seeds)
    examples = _sample_training_examples(
        query,
        seed=seed,
        generator=generator,
        maximum_sets=maximum_sets,
    )
    masks = [
        _set_mask(subset, len(features), device)
        for subset, _ in examples
    ]
    set_scores = model.score_sets(
        encoded,
        relation_groups,
        torch.stack(masks, dim=0),
    )
    all_index = next(
        index
        for index, (subset, _) in enumerate(examples)
        if subset["name"] == "all"
    )
    all_set_utility = set_scores[all_index]
    safe_logits = []
    catastrophic_logits = []
    hypotheses_predictions = []
    relative_lcb_predictions = []
    relative_median_predictions = []
    target_utility = []
    safe_labels = []
    catastrophic_labels = []
    hypotheses_targets = []
    for (subset, outcome), mask, set_utility in zip(
        examples, masks, set_scores
    ):
        predicted = model.predict_set_outcome(
            encoded, relation_groups, mask
        )
        safe_logits.append(predicted["safe_logit"])
        catastrophic_logits.append(predicted["catastrophic_logit"])
        hypotheses_predictions.append(predicted["log_hypotheses"])
        if model.config.relative_outcome_heads:
            relative = model.predict_relative_outcome(
                encoded,
                relation_groups,
                mask,
                selected_utility=set_utility,
                all_utility=all_set_utility,
            )
            relative_lcb_predictions.append(
                relative["relative_utility_lcb"]
            )
            relative_median_predictions.append(
                relative["relative_utility_median"]
            )
        target_utility.append(float(outcome["target_utility"]))
        safe_labels.append(float(outcome["safe_relative_all"]))
        catastrophic_labels.append(float(outcome["catastrophic"]))
        hypotheses_targets.append(
            math.log1p(max(int(outcome.get("hypotheses") or 100000), 1))
        )
    scores = set_scores
    target = scores.new_tensor(target_utility)
    centered_scores = scores - scores[all_index]
    centered_target = target - target[all_index]
    difference = target[:, None] - target[None, :]
    valid_pairs = difference.abs() >= 0.1
    upper = torch.triu(
        torch.ones_like(valid_pairs, dtype=torch.bool), diagonal=1
    )
    valid_pairs = valid_pairs & upper
    if bool(valid_pairs.any()):
        direction = torch.sign(difference[valid_pairs])
        predicted_difference = (
            scores[:, None] - scores[None, :]
        )[valid_pairs]
        rank_loss = F.softplus(
            0.2 - direction * predicted_difference
        ).mean()
        rank_accuracy = float(
            (direction * predicted_difference > 0).float().mean().detach()
        )
    else:
        rank_loss = scores.sum() * 0.0
        rank_accuracy = 1.0
    utility_regression_loss = F.smooth_l1_loss(
        centered_scores,
        centered_target,
        beta=0.5,
    )
    safe_logits = torch.stack(safe_logits)
    catastrophic_logits = torch.stack(catastrophic_logits)
    hypotheses_predictions = torch.stack(hypotheses_predictions)
    safe_labels_tensor = scores.new_tensor(safe_labels)
    catastrophic_labels_tensor = scores.new_tensor(catastrophic_labels)
    hypotheses_targets_tensor = scores.new_tensor(hypotheses_targets)
    safe_loss = _balanced_bce(safe_logits, safe_labels_tensor)
    catastrophic_loss = _balanced_bce(
        catastrophic_logits, catastrophic_labels_tensor
    )
    hypotheses_loss = F.smooth_l1_loss(
        hypotheses_predictions, hypotheses_targets_tensor
    )
    relative_outcome_loss = scores.sum() * 0.0
    if model.config.relative_outcome_heads:
        relative_lcb = torch.stack(relative_lcb_predictions)
        relative_median = torch.stack(relative_median_predictions)
        relative_outcome_loss = (
            _pinball_loss(relative_lcb, centered_target, 0.1)
            + _pinball_loss(relative_median, centered_target, 0.5)
        )
    strict_labels = query["strict_clean"].to(device=device).float()
    solver_labels = query["solver_clean"].to(device=device).float()
    harmful_labels = query["harmful"].to(device=device).float()
    auxiliary = (
        _balanced_bce(
            torch.logit(
                encoded["strict_probability"].clamp(1e-6, 1.0 - 1e-6)
            ),
            strict_labels,
        )
        + _balanced_bce(
            torch.logit(
                encoded["solver_probability"].clamp(1e-6, 1.0 - 1e-6)
            ),
            solver_labels,
        )
        + _balanced_bce(
            torch.logit(
                encoded["harmful_probability"].clamp(1e-6, 1.0 - 1e-6)
            ),
            harmful_labels,
        )
    ) / 3.0
    bias_loss = scores.sum() * 0.0
    if model.bias_head is not None:
        if (
            "signed_residual_target" not in query
            or "signed_residual_weight" not in query
        ):
            raise ValueError("bias-aware SLPS misses signed residual targets")
        target = query["signed_residual_target"].to(
            device=device, dtype=torch.float32
        )
        weight = query["signed_residual_weight"].to(
            device=device, dtype=torch.float32
        ).reshape(-1)
        row_loss = F.smooth_l1_loss(
            encoded["bias_vector"], target, reduction="none", beta=0.1
        ).mean(dim=1)
        bias_loss = (row_loss * weight).sum() / weight.sum().clamp_min(1.0)
    residual_utility_regularization = encoded[
        "utility_residual_unit"
    ].square().mean()
    loss = (
        rank_loss
        + 0.20 * utility_regression_loss
        + 0.35 * safe_loss
        + 0.35 * catastrophic_loss
        + 0.05 * hypotheses_loss
        + 0.20 * relative_outcome_loss
        + 0.15 * auxiliary
        + float(bias_loss_weight) * bias_loss
        + float(residual_utility_regularization_weight)
        * residual_utility_regularization
    )
    return loss, {
        "loss": float(loss.detach()),
        "rank_loss": float(rank_loss.detach()),
        "rank_accuracy": rank_accuracy,
        "utility_regression_loss": float(
            utility_regression_loss.detach()
        ),
        "safe_loss": float(safe_loss.detach()),
        "catastrophic_loss": float(catastrophic_loss.detach()),
        "hypotheses_loss": float(hypotheses_loss.detach()),
        "relative_outcome_loss": float(relative_outcome_loss.detach()),
        "auxiliary_loss": float(auxiliary.detach()),
        "bias_loss": float(bias_loss.detach()),
        "residual_utility_regularization": float(
            residual_utility_regularization.detach()
        ),
    }


@torch.no_grad()
def _validation_metrics(
    model: SLPSSelector,
    queries: list[dict],
    *,
    maximum_sets: int,
    seed: int,
    bias_loss_weight: float = 0.0,
    residual_utility_regularization_weight: float = 0.0,
) -> dict[str, float]:
    if not queries:
        return {"loss": float("nan"), "rank_accuracy": float("nan")}
    generator = random.Random(int(seed))
    metrics = []
    for query in queries:
        _, diagnostics = _query_loss(
            model,
            query,
            generator=generator,
            maximum_sets=maximum_sets,
            bias_loss_weight=bias_loss_weight,
            residual_utility_regularization_weight=(
                residual_utility_regularization_weight
            ),
        )
        metrics.append(diagnostics)
    return {
        name: float(np.mean([row[name] for row in metrics]))
        for name in metrics[0]
    }


@torch.no_grad()
def _calibrate_risk_thresholds(
    model: SLPSSelector,
    queries: list[dict],
) -> tuple[dict[str, float], dict[str, float]]:
    rows = []
    device = model.feature_mean.device
    has_deployment_profiles = any(
        bool(subset.get("deployment_calibration", False))
        for query in queries
        for subset in query["subsets"]
    )
    if not has_deployment_profiles:
        has_deployment_profiles = any(
            str(subset["name"]).startswith("learned_nested_")
            for query in queries
            for subset in query["subsets"]
        )
    for query in queries:
        features = query["features"].to(device=device, dtype=torch.float32)
        groups = query["relation_groups"].to(device=device)
        encoded = model.encode(features, groups)
        all_utility = None
        if model.config.relative_outcome_heads:
            all_utility = model.score_set(
                encoded,
                groups,
                torch.ones(
                    len(features), dtype=torch.bool, device=device
                ),
            )
        for subset in query["subsets"]:
            if has_deployment_profiles:
                explicit = any(
                    "deployment_calibration" in candidate
                    for candidate in query["subsets"]
                )
                if explicit and not bool(
                    subset.get("deployment_calibration", False)
                ):
                    continue
                if (
                    not explicit
                    and not str(subset["name"]).startswith(
                        "learned_nested_"
                    )
                ):
                    continue
            mask = _set_mask(subset, len(features), device)
            predicted = model.predict_set_outcome(encoded, groups, mask)
            relative_lcb = float("-inf")
            if model.config.relative_outcome_heads:
                selected_utility = model.score_set(
                    encoded, groups, mask
                )
                relative_lcb = float(
                    model.predict_relative_outcome(
                        encoded,
                        groups,
                        mask,
                        selected_utility=selected_utility,
                        all_utility=all_utility,
                    )["relative_utility_lcb"]
                )
            outcomes = list(subset["outcomes"])
            rows.append(
                {
                    "safe": float(torch.sigmoid(predicted["safe_logit"])),
                    "cat": float(
                        torch.sigmoid(predicted["catastrophic_logit"])
                    ),
                    "relative_lcb": relative_lcb,
                    "safe_label": all(
                        bool(outcome["safe_relative_all"])
                        for outcome in outcomes
                    ),
                    "cat_label": any(
                        bool(outcome["catastrophic"])
                        for outcome in outcomes
                    ),
                }
            )
    if not rows:
        raise ValueError("SLPS risk calibration has no eligible sets")
    best = None
    best_feasible = None
    relative_thresholds = (
        (-0.50, -0.25, -0.10, 0.0, 0.10, 0.25)
        if model.config.relative_outcome_heads
        else (float("-inf"),)
    )
    for safe_threshold in (0.55, 0.65, 0.75, 0.85, 0.9):
        for cat_threshold in (0.05, 0.1, 0.15, 0.2, 0.3):
            for relative_threshold in relative_thresholds:
                accepted = [
                    row
                    for row in rows
                    if row["safe"] >= safe_threshold
                    and row["cat"] <= cat_threshold
                    and row["relative_lcb"] >= relative_threshold
                ]
                false_safe = (
                    np.mean(
                        [
                            not row["safe_label"] or row["cat_label"]
                            for row in accepted
                        ]
                    )
                    if accepted
                    else 1.0
                )
                safe_recall = sum(
                    row["safe_label"] and not row["cat_label"]
                    for row in accepted
                ) / max(
                    sum(
                        row["safe_label"] and not row["cat_label"]
                        for row in rows
                    ),
                    1,
                )
                candidate = {
                    "safe_probability_threshold": float(safe_threshold),
                    "catastrophic_probability_threshold": float(
                        cat_threshold
                    ),
                    "relative_utility_lcb_threshold": float(
                        relative_threshold
                    ),
                    "accepted_rate": float(
                        len(accepted) / max(len(rows), 1)
                    ),
                    "false_safe_rate": float(false_safe),
                    "safe_recall": float(safe_recall),
                    "calibration_profile": (
                        "learned_nested"
                        if has_deployment_profiles
                        else "all_available"
                    ),
                    "calibration_set_count": len(rows),
                }
                feasible = false_safe <= 0.05 and len(accepted) > 0
                objective = (
                    safe_recall if feasible else -false_safe,
                    len(accepted),
                )
                if best is None or objective > best[0]:
                    best = (objective, candidate)
                if feasible and (
                    best_feasible is None or objective > best_feasible[0]
                ):
                    best_feasible = (objective, candidate)
    assert best is not None
    if best_feasible is None:
        calibration = {
            **best[1],
            "feasible": False,
            "fallback_reason": "no_threshold_satisfies_false_safe_limit",
        }
        return {
            "safe_probability_threshold": 1.01,
            "catastrophic_probability_threshold": -0.01,
            "minimum_probability_margin": 0.1,
            "relative_utility_lcb_threshold": float("inf"),
        }, calibration
    calibration = {**best_feasible[1], "feasible": True}
    thresholds = {
        "safe_probability_threshold": best_feasible[1][
            "safe_probability_threshold"
        ],
        "catastrophic_probability_threshold": best_feasible[1][
            "catastrophic_probability_threshold"
        ],
        "minimum_probability_margin": 0.1,
        "relative_utility_lcb_threshold": best_feasible[1][
            "relative_utility_lcb_threshold"
        ],
    }
    return thresholds, calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--maximum-sets-per-query", type=int, default=16)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--budgets", default="256,384,512,768")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--greedy-block-size", type=int, default=32)
    parser.add_argument(
        "--bias-aware-utility",
        action="store_true",
        help="Learn and cancel signed self-localization residual bias.",
    )
    parser.add_argument("--bias-loss-weight", type=float, default=0.20)
    parser.add_argument(
        "--decoupled-risk-encoder",
        action="store_true",
        help="Keep set-risk evidence separate from the frozen ordering encoder.",
    )
    parser.add_argument(
        "--bounded-residual-utility-fraction",
        type=float,
        default=0.0,
        help=(
            "Freeze the proven ordering and learn only a query-normalized "
            "bounded residual utility with this fraction of its robust spread."
        ),
    )
    parser.add_argument(
        "--residual-utility-regularization-weight",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--initial-selector",
        default="",
        help="Warm-start one exact self-mining macro-round.",
    )
    parser.add_argument(
        "--quality-utility-heads",
        action="store_true",
        help="Use learned strict/solver probabilities in set utility.",
    )
    parser.add_argument(
        "--relative-outcome-heads",
        action="store_true",
        help=(
            "Learn conservative set utility gain relative to the exact "
            "all-correspondence solve."
        ),
    )
    parser.add_argument(
        "--train-all-queries",
        action="store_true",
        help=(
            "Fit the scene-specific selector on every mapping query; use "
            "the same exact outcomes for conservative deployment calibration."
        ),
    )
    parser.add_argument(
        "--risk-heads-only",
        action="store_true",
        help=(
            "Freeze row encoding and set ordering after exact deployment "
            "self-mining; fit only set outcome and relative risk heads."
        ),
    )
    args = parser.parse_args()
    output = Path(args.output).resolve()
    partial_output = output.with_suffix(output.suffix + ".partial.pt")

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    corpus = torch.load(
        args.corpus, map_location="cpu", weights_only=False
    )
    if corpus.get("schema") != "lafgs_slps_set_outcomes":
        raise ValueError("unsupported SLPS corpus")
    corpus_feature_names = tuple(corpus.get("feature_names", ()))
    supported_feature_contracts = {
        tuple(SLPS_FEATURE_NAMES),
        tuple(SLPS_BIAS_AWARE_FEATURE_NAMES),
    }
    if corpus_feature_names not in supported_feature_contracts:
        raise ValueError("SLPS corpus feature contract differs")
    if args.bias_aware_utility and corpus_feature_names != tuple(
        SLPS_BIAS_AWARE_FEATURE_NAMES
    ):
        raise ValueError("bias-aware utility requires residual signature features")
    if corpus["candidate_graph_contract"].get(
        "family_prototype_state_sha256"
    ) is not None:
        raise ValueError("SLPS training requires a single-descriptor graph")
    queries = list(corpus["queries"])
    for query in queries:
        query["relation_groups"] = normalize_relation_groups(
            query["relation_groups"]
        )
    generator = random.Random(int(args.seed))
    shuffled = list(range(len(queries)))
    generator.shuffle(shuffled)
    if args.train_all_queries:
        training_queries = queries
        validation_queries = queries
    else:
        validation_count = min(
            max(round(len(queries) * float(args.validation_fraction)), 1),
            max(len(queries) - 1, 1),
        )
        validation_indices = set(shuffled[:validation_count])
        training_queries = [
            query
            for index, query in enumerate(queries)
            if index not in validation_indices
        ]
        validation_queries = [
            query
            for index, query in enumerate(queries)
            if index in validation_indices
        ]
    if not training_queries:
        training_queries = queries
        validation_queries = queries

    device = torch.device(args.device)
    config = SLPSModelConfig(
        input_dim=len(corpus_feature_names),
        greedy_block_size=max(int(args.greedy_block_size), 1),
        quality_utility_heads=bool(args.quality_utility_heads),
        relative_outcome_heads=bool(args.relative_outcome_heads),
        bias_aware_utility=bool(args.bias_aware_utility),
        decoupled_risk_encoder=bool(args.decoupled_risk_encoder),
        bounded_residual_utility_fraction=max(
            float(args.bounded_residual_utility_fraction), 0.0
        ),
    )
    model = SLPSSelector(
        config,
        feature_mean=corpus["feature_mean"],
        feature_scale=corpus["feature_scale"],
    ).to(device)
    initial_selector_sha256 = None
    if args.initial_selector:
        initial_path = Path(args.initial_selector).resolve()
        initial = torch.load(
            initial_path, map_location="cpu", weights_only=False
        )
        if initial.get("schema") != "lafgs_slps_selector":
            raise ValueError("initial SLPS selector schema differs")
        initial_feature_names = tuple(initial.get("feature_names", ()))
        expandable_legacy = (
            initial_feature_names == tuple(SLPS_FEATURE_NAMES)
            and corpus_feature_names == tuple(SLPS_BIAS_AWARE_FEATURE_NAMES)
        )
        if (
            (initial_feature_names != corpus_feature_names and not expandable_legacy)
            or dict(initial["candidate_graph_contract"])
            != dict(corpus["candidate_graph_contract"])
            or initial["anchor_ids_sha256"] != corpus["anchor_ids_sha256"]
        ):
            raise ValueError("initial SLPS selector contract differs")
        initial_parameters = dict(initial["model_state_dict"])
        if expandable_legacy:
            expanded = model.row_encoder[0].weight.detach().clone()
            old_weight = torch.as_tensor(
                initial_parameters["row_encoder.0.weight"]
            )
            expanded[:, : old_weight.shape[1]] = old_weight
            expanded[:, old_weight.shape[1] :] = 0.0
            initial_parameters["row_encoder.0.weight"] = expanded
            initial_parameters["feature_mean"] = corpus["feature_mean"].float()
            initial_parameters["feature_scale"] = corpus["feature_scale"].float()
            for key in list(initial_parameters):
                if key.startswith("set_outcome_head."):
                    expected = model.state_dict().get(key)
                    if expected is None or expected.shape != initial_parameters[key].shape:
                        initial_parameters.pop(key)
        incompatible = model.load_state_dict(initial_parameters, strict=False)
        allowed_missing = {
            "strict_weight_raw",
            "solver_weight_raw",
        }
        allowed_missing.update(
            key
            for key in incompatible.missing_keys
            if key.startswith("relative_outcome_head.")
        )
        allowed_missing.update(
            key
            for key in incompatible.missing_keys
            if key.startswith("bias_head.")
            or key.startswith("set_outcome_head.")
            or key == "bias_weight_raw"
            or key.startswith("risk_row_encoder.")
            or key.startswith("risk_relation_layers.")
            or key.startswith("residual_utility_head.")
        )
        if (
            set(incompatible.missing_keys) - allowed_missing
            or incompatible.unexpected_keys
        ):
            raise ValueError("initial SLPS selector parameters differ")
        initial_selector_sha256 = hashlib.sha256(
            initial_path.read_bytes()
        ).hexdigest()
    if args.risk_heads_only and float(
        args.bounded_residual_utility_fraction
    ) > 0.0:
        raise ValueError(
            "bounded residual utility and --risk-heads-only are exclusive"
        )
    if float(args.bounded_residual_utility_fraction) > 0.0:
        if not args.initial_selector:
            raise ValueError(
                "bounded residual utility requires --initial-selector"
            )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        assert model.residual_utility_head is not None
        for parameter in model.residual_utility_head.parameters():
            parameter.requires_grad_(True)
        for module in (
            model.set_outcome_head,
            model.relative_outcome_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        if model.risk_row_encoder is not None:
            for module in (
                model.risk_row_encoder,
                model.risk_relation_layers,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
    elif args.risk_heads_only:
        if not args.initial_selector:
            raise ValueError("--risk-heads-only requires --initial-selector")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for module in (
            model.set_outcome_head,
            model.relative_outcome_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        if model.risk_row_encoder is not None:
            for module in (
                model.risk_row_encoder,
                model.risk_relation_layers,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(args.steps), 1), eta_min=2e-6
    )
    best_state = None
    best_validation = float("inf")
    stale = 0
    history = []
    start_step = 0
    if partial_output.is_file():
        partial = torch.load(
            partial_output, map_location="cpu", weights_only=False
        )
        identity = {
            "corpus": str(Path(args.corpus).resolve()),
            "corpus_sha256": hashlib.sha256(
                Path(args.corpus).read_bytes()
            ).hexdigest(),
            "steps": int(args.steps),
            "seed": int(args.seed),
            "initial_selector_sha256": initial_selector_sha256,
            "model_config": model.export_config(),
            "maximum_sets_per_query": int(args.maximum_sets_per_query),
            "train_all_queries": bool(args.train_all_queries),
            "risk_heads_only": bool(args.risk_heads_only),
        }
        if partial.get("identity") != identity:
            raise ValueError("SLPS training partial identity differs")
        model.load_state_dict(partial["model_state_dict"])
        optimizer.load_state_dict(partial["optimizer_state_dict"])
        scheduler.load_state_dict(partial["scheduler_state_dict"])
        best_state = partial.get("best_state")
        best_validation = float(partial["best_validation"])
        stale = int(partial["stale"])
        history = list(partial["history"])
        start_step = int(partial["step"])
        generator.setstate(partial["generator_state"])
    for step in range(start_step + 1, int(args.steps) + 1):
        model.train()
        query = generator.choice(training_queries)
        optimizer.zero_grad(set_to_none=True)
        loss, diagnostics = _query_loss(
            model,
            query,
            generator=generator,
            maximum_sets=int(args.maximum_sets_per_query),
            bias_loss_weight=float(args.bias_loss_weight),
            residual_utility_regularization_weight=float(
                args.residual_utility_regularization_weight
            ),
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("SLPS training loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        if step % int(args.validation_interval) == 0 or step == int(
            args.steps
        ):
            model.eval()
            validation = _validation_metrics(
                model,
                validation_queries,
                maximum_sets=int(args.maximum_sets_per_query),
                seed=int(args.seed) + 99173,
                bias_loss_weight=float(args.bias_loss_weight),
                residual_utility_regularization_weight=float(
                    args.residual_utility_regularization_weight
                ),
            )
            row = {
                "step": step,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "train": diagnostics,
                "validation": validation,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            value = float(validation["loss"])
            if value < best_validation - 1e-4:
                best_validation = value
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            _atomic_torch(
                partial_output,
                {
                    "schema": "lafgs_slps_selector_training_partial",
                    "identity": {
                        "corpus": str(Path(args.corpus).resolve()),
                        "corpus_sha256": hashlib.sha256(
                            Path(args.corpus).read_bytes()
                        ).hexdigest(),
                        "steps": int(args.steps),
                        "seed": int(args.seed),
                        "initial_selector_sha256": (
                            initial_selector_sha256
                        ),
                        "model_config": model.export_config(),
                        "maximum_sets_per_query": int(
                            args.maximum_sets_per_query
                        ),
                        "train_all_queries": bool(
                            args.train_all_queries
                        ),
                        "risk_heads_only": bool(args.risk_heads_only),
                    },
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_state": best_state,
                    "best_validation": best_validation,
                    "stale": stale,
                    "history": history,
                    "generator_state": generator.getstate(),
                },
            )
            if stale >= int(args.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    thresholds, calibration = _calibrate_risk_thresholds(
        model, validation_queries
    )
    budgets = sorted(
        {
            int(value)
            for value in str(args.budgets).split(",")
            if value.strip()
        }
    )
    state_dict = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }
    payload = {
        "schema": "lafgs_slps_selector",
        "version": 1,
        "method": "self_localization_guided_pose_sufficient_set_learning",
        "model_config": model.export_config(),
        "model_state_dict": state_dict,
        "feature_names": list(corpus_feature_names),
        "relation_names": list(RELATION_NAMES),
        "feature_mean": corpus["feature_mean"].float(),
        "feature_scale": corpus["feature_scale"].float(),
        "anchor_statistics": corpus["anchor_statistics"],
        "anchor_track_stability": corpus["anchor_track_stability"],
        "residual_signature_state": corpus.get("residual_signature_state"),
        "selector_config": {
            "budgets": budgets,
            **thresholds,
        },
        "anchor_count": int(corpus["anchor_count"]),
        "anchor_ids_sha256": corpus["anchor_ids_sha256"],
        "candidate_graph_contract": dict(
            corpus["candidate_graph_contract"]
        ),
        "retrieval_topk": int(corpus["retrieval_topk"]),
        "entropy_temperature": 0.05,
        "prior_strength": 12.0,
        "training_config": vars(args),
        "initial_selector_sha256": initial_selector_sha256,
        "training_query_names": [
            query["query_name"] for query in training_queries
        ],
        "validation_query_names": [
            query["query_name"] for query in validation_queries
        ],
        "summary": {
            "best_validation_loss": best_validation,
            "completed_steps": history[-1]["step"] if history else 0,
            "risk_calibration": calibration,
            "corpus_summary": corpus["summary"],
            "history": history,
        },
    }
    _atomic_torch(output, payload)
    summary = {
        "output": str(output),
        "best_validation_loss": best_validation,
        "completed_steps": payload["summary"]["completed_steps"],
        "training_query_count": len(training_queries),
        "validation_query_count": len(validation_queries),
        "selector_config": payload["selector_config"],
        "risk_calibration": calibration,
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    partial_output.unlink(missing_ok=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
