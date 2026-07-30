#!/usr/bin/env python3
"""Build leave-one-trajectory-out candidate switches from pair reliability."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from localization_training.pair_reliability import (
    PairOutcome,
    aggregate_pair_reliability,
    reliable_pair,
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


def _load_aligned(path: str, reference: dict) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "lafgs_exact_topk_outcomes":
        raise ValueError(f"unsupported top-K payload: {path}")
    for key in ("anchor_count", "anchor_ids_sha256", "query_names"):
        if payload[key] != reference[key]:
            raise ValueError(f"{path} differs from baseline field {key}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-topk", required=True)
    parser.add_argument("--proposal-topk", required=True)
    parser.add_argument("--strict-oracle-topk", required=True)
    parser.add_argument("--output-topk", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--minimum-attempts", type=int, default=3)
    parser.add_argument("--minimum-successes", type=int, default=2)
    parser.add_argument("--minimum-positive-trajectories", type=int, default=2)
    parser.add_argument("--minimum-precision", type=float, default=0.75)
    parser.add_argument("--minimum-wilson-lower-bound", type=float, default=0.45)
    parser.add_argument("--wilson-z", type=float, default=1.0)
    args = parser.parse_args()

    baseline = torch.load(
        args.baseline_topk, map_location="cpu", weights_only=False
    )
    if baseline.get("schema") != "lafgs_exact_topk_outcomes":
        raise ValueError("unsupported baseline top-K payload")
    proposal = _load_aligned(args.proposal_topk, baseline)
    oracle = _load_aligned(args.strict_oracle_topk, baseline)
    baseline_by_name = _records_by_name(baseline)
    proposal_by_name = _records_by_name(proposal)
    oracle_by_name = _records_by_name(oracle)

    outcomes: list[PairOutcome] = []
    event_rows: dict[str, list[tuple[int, tuple[int, int], bool]]] = {}
    for name in baseline["query_names"]:
        before = baseline_by_name[name]
        after = proposal_by_name[name]
        strict = oracle_by_name[name]
        rows = torch.as_tensor(before["query_rows"]).long()
        before_indices = torch.as_tensor(before["topk_anchor_indices"]).long()
        after_indices = torch.as_tensor(after["topk_anchor_indices"]).long()
        strict_indices = torch.as_tensor(strict["topk_anchor_indices"]).long()
        if not (
            torch.equal(rows, torch.as_tensor(after["query_rows"]).long())
            and torch.equal(rows, torch.as_tensor(strict["query_rows"]).long())
        ):
            raise ValueError(f"query rows do not align for {name}")
        changed = torch.nonzero(
            before_indices[:, 0] != after_indices[:, 0], as_tuple=False
        ).reshape(-1)
        local_events = []
        trajectory = _trajectory(name)
        for slot in changed.tolist():
            pair = (
                int(before_indices[slot, 0]),
                int(after_indices[slot, 0]),
            )
            success = int(strict_indices[slot, 0]) == pair[1]
            outcomes.append(
                PairOutcome(
                    trajectory=trajectory,
                    confusing_anchor=pair[0],
                    correct_anchor=pair[1],
                    success=success,
                )
            )
            local_events.append((slot, pair, success))
        event_rows[name] = local_events

    trajectories = sorted({_trajectory(name) for name in baseline["query_names"]})
    fold_statistics = {
        trajectory: aggregate_pair_reliability(
            outcomes,
            excluded_trajectory=trajectory,
            z=float(args.wilson_z),
        )
        for trajectory in trajectories
    }
    full_statistics = aggregate_pair_reliability(
        outcomes, z=float(args.wilson_z)
    )
    gate = {
        "minimum_attempts": int(args.minimum_attempts),
        "minimum_successes": int(args.minimum_successes),
        "minimum_positive_trajectories": int(
            args.minimum_positive_trajectories
        ),
        "minimum_precision": float(args.minimum_precision),
        "minimum_wilson_lower_bound": float(
            args.minimum_wilson_lower_bound
        ),
    }

    output_records = []
    accepted = 0
    accepted_positive = 0
    accepted_pairs: set[tuple[int, int]] = set()
    for name in baseline["query_names"]:
        before = baseline_by_name[name]
        after = proposal_by_name[name]
        indices = torch.as_tensor(before["topk_anchor_indices"]).long().clone()
        scores = torch.as_tensor(before["topk_scores"]).float().clone()
        statistics = fold_statistics[_trajectory(name)]
        selected = []
        for slot, pair, success in event_rows[name]:
            if reliable_pair(statistics.get(pair), **gate):
                selected.append(slot)
                accepted += 1
                accepted_positive += int(success)
                accepted_pairs.add(pair)
        if selected:
            selected = torch.as_tensor(selected).long()
            indices[selected] = torch.as_tensor(
                after["topk_anchor_indices"]
            ).long()[selected]
            scores[selected] = torch.as_tensor(
                after["topk_scores"]
            ).float()[selected]
        output_records.append(
            {
                "query_name": name,
                "query_rows": torch.as_tensor(before["query_rows"]).long(),
                "topk_anchor_indices": indices,
                "topk_scores": scores,
            }
        )

    full_reliable = {
        pair: reliability
        for pair, reliability in full_statistics.items()
        if reliable_pair(reliability, **gate)
    }
    summary = {
        "proposal_count": len(outcomes),
        "strict_positive_count": sum(int(outcome.success) for outcome in outcomes),
        "oof_accepted_count": accepted,
        "oof_accepted_pair_count": len(accepted_pairs),
        "oof_accepted_precision": accepted_positive / max(accepted, 1),
        "oof_accepted_recall": accepted_positive
        / max(sum(int(outcome.success) for outcome in outcomes), 1),
        "full_reliable_pair_count": len(full_reliable),
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
            "method": "candidate_pair_reliability_oof",
            "summary": summary,
            "provenance": {
                "baseline_topk": str(Path(args.baseline_topk).resolve()),
                "proposal_topk": str(Path(args.proposal_topk).resolve()),
                "strict_oracle_topk": str(
                    Path(args.strict_oracle_topk).resolve()
                ),
            },
        },
    )
    output_model = Path(args.output_model).resolve()
    serializable_statistics = {
        f"{confusing}:{correct}": {
            "attempts": value.attempts,
            "successes": value.successes,
            "positive_trajectories": value.positive_trajectories,
            "empirical_precision": value.empirical_precision,
            "wilson_lower_bound": value.wilson_lower_bound,
        }
        for (confusing, correct), value in full_reliable.items()
    }
    _atomic_torch(
        output_model,
        {
            "schema": "lafgs_candidate_pair_reliability",
            "version": 1,
            "fold_contract": "leave_one_trajectory_out",
            "gate": gate,
            "wilson_z": float(args.wilson_z),
            "full_reliable_pairs": serializable_statistics,
            "summary": summary,
            "training_config": vars(args),
            "anchor_count": int(baseline["anchor_count"]),
            "anchor_ids_sha256": baseline["anchor_ids_sha256"],
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
