"""Query-level PoseLib policy oracle for A1 versus shared context."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Sequence

import numpy as np


def pose_policy_loss(
    *,
    te_cm: float,
    ae_deg: float,
    hypotheses: int,
    translation_scale_cm: float = 5.0,
    rotation_scale_deg: float = 5.0,
    catastrophe_cm: float = 100.0,
    catastrophe_weight: float = 2.0,
    hypothesis_scale: float = 1000.0,
    hypothesis_weight: float = 0.05,
) -> float:
    """Task-normalized localization risk used to choose one image policy."""
    if min(
        float(translation_scale_cm),
        float(rotation_scale_deg),
        float(catastrophe_cm),
        float(hypothesis_scale),
    ) <= 0.0:
        raise ValueError("policy-loss scales must be positive")
    return float(
        math.log1p(max(float(te_cm), 0.0) / float(translation_scale_cm))
        + math.log1p(max(float(ae_deg), 0.0) / float(rotation_scale_deg))
        + float(catastrophe_weight) * (float(te_cm) >= float(catastrophe_cm))
        + float(hypothesis_weight)
        * math.log1p(max(int(hypotheses), 0) / float(hypothesis_scale))
    )


def _pose_summary(rows: list[dict], policy_by_query: dict[int, str], label: str) -> dict:
    if not rows:
        return {"query_seed_count": 0}
    te = []
    ae = []
    hypotheses = []
    failures = []
    for row in rows:
        prefix = label if label != "oracle" else policy_by_query[int(row["query_index"])]
        te.append(float(row[f"{prefix}_te_cm"]))
        ae.append(float(row[f"{prefix}_ae_deg"]))
        hypotheses.append(int(row[f"{prefix}_hypotheses"]))
        failures.append(bool(row[f"{prefix}_failed"]))
    translations = np.asarray(te, dtype=np.float64)
    rotations = np.asarray(ae, dtype=np.float64)
    tail_count = max(int(math.ceil(0.05 * translations.size)), 1)
    return {
        "query_seed_count": int(translations.size),
        "median_te_cm": float(np.median(translations)),
        "mean_te_cm": float(translations.mean()),
        "p90_te_cm": float(np.percentile(translations, 90)),
        "cvar95_te_cm": float(np.sort(translations)[-tail_count:].mean()),
        "median_ae_deg": float(np.median(rotations)),
        "mean_ae_deg": float(rotations.mean()),
        "p90_ae_deg": float(np.percentile(rotations, 90)),
        "recall_2cm_2deg_percent": float(
            100.0 * np.mean((translations <= 2.0) & (rotations <= 2.0))
        ),
        "recall_5cm_5deg_percent": float(
            100.0 * np.mean((translations <= 5.0) & (rotations <= 5.0))
        ),
        "catastrophic_100cm_count": int(np.count_nonzero(translations >= 100.0)),
        "failure_count": int(sum(failures)),
        "mean_hypotheses": float(np.mean(hypotheses)),
    }


def summarize_policy_oracle(
    rows: list[dict],
    *,
    context_protocol: str,
    seeds: Sequence[int],
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 2026,
    loss_config: dict | None = None,
) -> dict:
    """Select the lower-risk policy per query and quantify oracle headroom."""
    if not rows:
        raise ValueError("policy oracle requires pose rows")
    expected_seeds = tuple(int(value) for value in seeds)
    if not expected_seeds:
        raise ValueError("policy oracle requires at least one seed")
    config = dict(loss_config or {})
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["query_index"])].append(row)
    query_rows = []
    policy_by_query = {}
    for query_index, values in sorted(grouped.items()):
        actual_seeds = tuple(sorted(int(value["seed"]) for value in values))
        if actual_seeds != tuple(sorted(expected_seeds)):
            raise ValueError(f"query {query_index} does not contain every seed")
        a1_losses = [
            pose_policy_loss(
                te_cm=value["a1_te_cm"],
                ae_deg=value["a1_ae_deg"],
                hypotheses=value["a1_hypotheses"],
                **config,
            )
            for value in values
        ]
        context_losses = [
            pose_policy_loss(
                te_cm=value[f"{context_protocol}_te_cm"],
                ae_deg=value[f"{context_protocol}_ae_deg"],
                hypotheses=value[f"{context_protocol}_hypotheses"],
                **config,
            )
            for value in values
        ]
        a1_risk = float(np.mean(a1_losses))
        context_risk = float(np.mean(context_losses))
        policy = context_protocol if context_risk < a1_risk else "a1"
        policy_by_query[query_index] = policy
        query_rows.append(
            {
                "query_index": query_index,
                "image_name": values[0]["image_name"],
                "direction": values[0].get("direction"),
                "a1_risk": a1_risk,
                "context_risk": context_risk,
                "context_advantage": a1_risk - context_risk,
                "oracle_policy": policy,
                "oracle_risk": min(a1_risk, context_risk),
            }
        )

    a1 = np.asarray([value["a1_risk"] for value in query_rows])
    context = np.asarray([value["context_risk"] for value in query_rows])
    oracle = np.minimum(a1, context)
    best_fixed_name = context_protocol if context.mean() < a1.mean() else "a1"
    best_fixed = min(float(a1.mean()), float(context.mean()))
    oracle_mean = float(oracle.mean())
    headroom = best_fixed - oracle_mean
    generator = np.random.default_rng(int(bootstrap_seed))
    query_count = len(query_rows)
    bootstrap_headroom = np.empty(int(bootstrap_samples), dtype=np.float64)
    bootstrap_advantage = np.empty(int(bootstrap_samples), dtype=np.float64)
    for index in range(int(bootstrap_samples)):
        sample = generator.integers(0, query_count, size=query_count)
        sample_a1 = float(a1[sample].mean())
        sample_context = float(context[sample].mean())
        sample_oracle = float(np.minimum(a1[sample], context[sample]).mean())
        bootstrap_headroom[index] = min(sample_a1, sample_context) - sample_oracle
        bootstrap_advantage[index] = float((a1[sample] - context[sample]).mean())

    return {
        "query_count": query_count,
        "seed_count": len(expected_seeds),
        "query_seed_count": len(rows),
        "context_protocol": context_protocol,
        "loss_config": config,
        "fixed_policy_risk": {
            "a1": float(a1.mean()),
            context_protocol: float(context.mean()),
        },
        "best_fixed_policy": best_fixed_name,
        "oracle_risk": oracle_mean,
        "oracle_headroom_absolute": float(headroom),
        "oracle_headroom_relative_to_best_fixed_percent": float(
            100.0 * headroom / max(best_fixed, 1e-12)
        ),
        "oracle_headroom_bootstrap_95ci": [
            float(np.percentile(bootstrap_headroom, 2.5)),
            float(np.percentile(bootstrap_headroom, 97.5)),
        ],
        "mean_context_advantage": float((a1 - context).mean()),
        "mean_context_advantage_bootstrap_95ci": [
            float(np.percentile(bootstrap_advantage, 2.5)),
            float(np.percentile(bootstrap_advantage, 97.5)),
        ],
        "oracle_policy_counts": {
            "a1": int(sum(value == "a1" for value in policy_by_query.values())),
            context_protocol: int(
                sum(value == context_protocol for value in policy_by_query.values())
            ),
        },
        "pose": {
            "a1": _pose_summary(rows, policy_by_query, "a1"),
            context_protocol: _pose_summary(rows, policy_by_query, context_protocol),
            "oracle": _pose_summary(rows, policy_by_query, "oracle"),
        },
        "queries": query_rows,
    }
