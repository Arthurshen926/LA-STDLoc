#!/usr/bin/env python3
"""Adapt the shared descriptor head on strict rescue and protected correspondences."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from localization_training.confusion_directed_metric import (
    candidate_margin_loss,
    protected_top1_mask,
    select_stratified_rows,
    topk_distribution_distillation,
)
from localization_training.positive_retrieval import (
    csr_candidate_positive_mask,
)
from localization_training.shared_metric import SharedLowRankMetric


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _file_stamp(path: str | Path) -> dict:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _csr_positive_mask(
    candidates: torch.Tensor,
    offsets: torch.Tensor,
    positive_indices: torch.Tensor,
) -> torch.Tensor:
    return csr_candidate_positive_mask(
        candidates, offsets, positive_indices
    )


def _build_examples(
    *,
    cache: dict,
    teacher: dict,
    outcomes: dict,
    topk: dict,
    clean_threshold_px: float,
    harmful_threshold_px: float,
    maximum_rows_per_type_per_query: int,
    seed: int,
) -> tuple[dict, dict]:
    teacher_by_name = {
        str(name): record
        for name, record in zip(teacher["query_names"], teacher["records"])
    }
    outcome_by_name = {
        str(record["query_name"]): record for record in outcomes["records"]
    }
    topk_by_name = {
        str(record["query_name"]): record for record in topk["records"]
    }
    names = sorted(set(teacher_by_name) & set(outcome_by_name) & set(topk_by_name))
    if len(names) != len(teacher["query_names"]):
        raise ValueError("training artifacts do not cover the same query set")
    generator = torch.Generator().manual_seed(int(seed))
    values = {
        "query": [],
        "candidates": [],
        "positive_mask": [],
        "kind": [],
        "query_index": [],
        "trajectory": [],
    }
    diagnostics = {
        "query_count": len(names),
        "rescue_available": 0,
        "protected_available": 0,
        "neutral_available": 0,
        "hard_available": 0,
    }
    trajectories = sorted({name.split("/", 1)[0] for name in names})
    trajectory_index = {
        trajectory: index for index, trajectory in enumerate(trajectories)
    }
    for query_index, name in enumerate(names):
        teacher_record = teacher_by_name[name]
        outcome_record = outcome_by_name[name]
        topk_record = topk_by_name[name]
        rows = torch.as_tensor(topk_record["query_rows"]).long()
        if not torch.equal(rows, torch.as_tensor(teacher_record["query_rows"]).long()):
            raise ValueError(f"teacher rows do not align for {name}")
        if not torch.equal(rows, torch.as_tensor(outcome_record["query_rows"]).long()):
            raise ValueError(f"outcome rows do not align for {name}")
        candidates = torch.as_tensor(topk_record["topk_anchor_indices"]).long()
        if candidates.ndim != 2 or candidates.shape[1] < 2:
            raise ValueError("confusion-directed adaptation requires top-K >= 2")
        positive_mask = _csr_positive_mask(
            candidates,
            torch.as_tensor(teacher_record["positive_offsets"]).long(),
            torch.as_tensor(teacher_record["positive_indices"]).long(),
        )
        error = torch.as_tensor(
            outcome_record["gt_reprojection_errors_px"]
        ).float()
        if error.numel() != len(rows):
            raise ValueError(f"outcome errors do not align for {name}")
        has_legal = positive_mask.any(dim=1)
        rescue = (
            (error > float(harmful_threshold_px))
            & has_legal
            & ~positive_mask[:, 0]
        )
        protected = (error <= float(clean_threshold_px)) & positive_mask[:, 0]
        neutral = ~has_legal
        harmful_inlier = torch.as_tensor(
            outcome_record.get(
                "harmful_inlier_mask", torch.zeros_like(rescue)
            )
        ).bool()
        if harmful_inlier.numel() != len(rows):
            raise ValueError(f"harmful-inlier rows do not align for {name}")
        hard = rescue & harmful_inlier
        diagnostics["rescue_available"] += int(rescue.sum())
        diagnostics["protected_available"] += int(protected.sum())
        diagnostics["neutral_available"] += int(neutral.sum())
        diagnostics["hard_available"] += int(hard.sum())
        rescue_rows = select_stratified_rows(
            rescue,
            maximum=maximum_rows_per_type_per_query,
            generator=generator,
        )
        protected_rows = select_stratified_rows(
            protected,
            maximum=maximum_rows_per_type_per_query,
            generator=generator,
        )
        neutral_rows = select_stratified_rows(
            neutral,
            maximum=maximum_rows_per_type_per_query,
            generator=generator,
        )
        hard_rows = select_stratified_rows(
            hard,
            maximum=maximum_rows_per_type_per_query,
            generator=generator,
        )
        selected = torch.cat(
            (rescue_rows, protected_rows, neutral_rows, hard_rows)
        )
        if not selected.numel():
            continue
        native = F.normalize(
            torch.as_tensor(cache[name]["native_descriptors"]).float()[rows[selected]],
            dim=1,
        )
        selected_mask = positive_mask[selected]
        if protected_rows.numel():
            protected_start = rescue_rows.numel()
            protected_end = protected_start + protected_rows.numel()
            selected_mask[protected_start:protected_end] = protected_top1_mask(
                len(protected_rows), candidates.shape[1]
            )
        values["query"].append(native)
        values["candidates"].append(candidates[selected])
        values["positive_mask"].append(selected_mask)
        values["kind"].append(
            torch.cat(
                (
                    torch.zeros(len(rescue_rows), dtype=torch.long),
                    torch.ones(len(protected_rows), dtype=torch.long),
                    torch.full(
                        (len(neutral_rows),), 2, dtype=torch.long
                    ),
                    torch.full((len(hard_rows),), 3, dtype=torch.long),
                )
            )
        )
        values["query_index"].append(
            torch.full((len(selected),), query_index, dtype=torch.long)
        )
        values["trajectory"].append(
            torch.full(
                (len(selected),),
                trajectory_index[name.split("/", 1)[0]],
                dtype=torch.long,
            )
        )
    output = {
        key: torch.cat(parts, dim=0)
        for key, parts in values.items()
        if parts
    }
    if not output or not bool((output["kind"] == 0).any()):
        raise RuntimeError("no strict rescue examples were found")
    diagnostics["rescue_selected"] = int((output["kind"] == 0).sum())
    diagnostics["protected_selected"] = int((output["kind"] == 1).sum())
    diagnostics["neutral_selected"] = int((output["kind"] == 2).sum())
    diagnostics["hard_selected"] = int((output["kind"] == 3).sum())
    diagnostics["trajectory_count"] = len(trajectories)
    diagnostics["training_rows"] = int(len(output["kind"]))
    return output, diagnostics


def _sample(rows: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    if not rows.numel():
        return rows
    return rows[
        torch.randint(rows.numel(), (int(count),), generator=generator)
    ]


def _hierarchical_sampling_index(
    examples: dict, kind: int
) -> dict[int, dict[int, torch.Tensor]]:
    selected = torch.nonzero(
        torch.as_tensor(examples["kind"]).long() == int(kind),
        as_tuple=False,
    ).reshape(-1)
    output: dict[int, dict[int, torch.Tensor]] = {}
    trajectories = torch.as_tensor(examples["trajectory"]).long()
    queries = torch.as_tensor(examples["query_index"]).long()
    for trajectory in trajectories[selected].unique(sorted=True).tolist():
        trajectory_rows = selected[trajectories[selected] == int(trajectory)]
        output[int(trajectory)] = {}
        for query in queries[trajectory_rows].unique(sorted=True).tolist():
            output[int(trajectory)][int(query)] = trajectory_rows[
                queries[trajectory_rows] == int(query)
            ]
    return output


def _sample_hierarchical(
    index: dict[int, dict[int, torch.Tensor]],
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if count <= 0:
        return torch.empty(0, dtype=torch.long)
    if not index:
        raise ValueError("requested sample kind has no eligible rows")
    trajectories = sorted(index)
    output = []
    for _ in range(int(count)):
        trajectory = trajectories[
            int(torch.randint(len(trajectories), (), generator=generator))
        ]
        queries = sorted(index[trajectory])
        query = queries[
            int(torch.randint(len(queries), (), generator=generator))
        ]
        rows = index[trajectory][query]
        output.append(
            rows[int(torch.randint(len(rows), (), generator=generator))]
        )
    return torch.stack(output)


@torch.inference_mode()
def _assignment_contract_diagnostics(
    *,
    metric: SharedLowRankMetric,
    teacher_metric: SharedLowRankMetric,
    raw_bank: torch.Tensor,
    examples: dict,
    device: torch.device,
    temperature: float,
    batch_size: int = 4096,
) -> dict:
    totals = {
        "rescue_success": 0,
        "rescue_total": 0,
        "protected_retained": 0,
        "protected_total": 0,
        "neutral_top1_changed": 0,
        "neutral_total": 0,
        "neutral_kl_sum": 0.0,
    }
    for start in range(0, len(examples["kind"]), int(batch_size)):
        stop = min(start + int(batch_size), len(examples["kind"]))
        query_raw = examples["query"][start:stop].to(device)
        candidates = examples["candidates"][start:stop].to(device)
        positive = examples["positive_mask"][start:stop].to(device)
        kind = examples["kind"][start:stop].to(device)
        query, _ = metric(query_raw)
        candidate, _ = metric(raw_bank[candidates])
        score = torch.einsum("bd,bkd->bk", query, candidate)
        old_query, _ = teacher_metric(query_raw)
        old_candidate, _ = teacher_metric(raw_bank[candidates])
        old_score = torch.einsum("bd,bkd->bk", old_query, old_candidate)
        winner = score.argmax(dim=1)
        old_winner = old_score.argmax(dim=1)
        rescue = kind == 0
        protected = kind == 1
        neutral = kind == 2
        if bool(rescue.any()):
            totals["rescue_success"] += int(
                positive[rescue]
                .gather(1, winner[rescue, None])
                .sum()
            )
            totals["rescue_total"] += int(rescue.sum())
        if bool(protected.any()):
            totals["protected_retained"] += int(
                (winner[protected] == 0).sum()
            )
            totals["protected_total"] += int(protected.sum())
        if bool(neutral.any()):
            totals["neutral_top1_changed"] += int(
                (winner[neutral] != old_winner[neutral]).sum()
            )
            totals["neutral_total"] += int(neutral.sum())
            totals["neutral_kl_sum"] += float(
                topk_distribution_distillation(
                    score[neutral],
                    old_score[neutral],
                    temperature=temperature,
                )
            ) * int(neutral.sum())
    return {
        "rescue_candidate_success_percent": (
            100.0 * totals["rescue_success"] / max(totals["rescue_total"], 1)
        ),
        "protected_top1_retained_percent": (
            100.0
            * totals["protected_retained"]
            / max(totals["protected_total"], 1)
        ),
        "neutral_top1_change_percent": (
            100.0
            * totals["neutral_top1_changed"]
            / max(totals["neutral_total"], 1)
        ),
        "neutral_top16_kl_mean": (
            totals["neutral_kl_sum"] / max(totals["neutral_total"], 1)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--initial-metric-state", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--complete-positive-teacher", required=True)
    parser.add_argument("--dynamic-outcomes", required=True)
    parser.add_argument("--topk-outcomes", required=True)
    parser.add_argument(
        "--examples-cache",
        default="",
        help="Reusable strict rescue/protected rows bound to source file stamps.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--clean-threshold-px", type=float, default=2.0)
    parser.add_argument("--harmful-threshold-px", type=float, default=12.0)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--protected-weight", type=float, default=2.0)
    parser.add_argument("--distill-weight", type=float, default=2.0)
    parser.add_argument("--neutral-replay-weight", type=float, default=4.0)
    parser.add_argument("--neutral-temperature", type=float, default=0.05)
    parser.add_argument("--hard-weight", type=float, default=1.0)
    parser.add_argument("--parameter-trust-weight", type=float, default=0.1)
    parser.add_argument("--rescue-fraction", type=float, default=0.25)
    parser.add_argument("--protected-fraction", type=float, default=0.25)
    parser.add_argument("--neutral-fraction", type=float, default=0.40)
    parser.add_argument("--hard-fraction", type=float, default=0.10)
    parser.add_argument("--maximum-rows-per-type-per-query", type=int, default=96)
    parser.add_argument("--example-seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    fractions = (
        args.rescue_fraction,
        args.protected_fraction,
        args.neutral_fraction,
        args.hard_fraction,
    )
    if any(value < 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("training fractions must be non-negative and sum to one")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    metric_payload = torch.load(
        args.initial_metric_state, map_location="cpu", weights_only=False
    )
    teacher = torch.load(
        args.complete_positive_teacher, map_location="cpu", weights_only=False
    )
    outcomes = torch.load(
        args.dynamic_outcomes, map_location="cpu", weights_only=False
    )
    topk = torch.load(args.topk_outcomes, map_location="cpu", weights_only=False)
    anchor_count = int(torch.as_tensor(state["anchor_xyz"]).shape[0])
    for label, payload in (
        ("teacher", teacher),
        ("outcomes", outcomes),
        ("top-K", topk),
    ):
        if int(payload["anchor_count"]) != anchor_count:
            raise ValueError(f"{label} anchor count does not align with map")
    landmark_indices = torch.as_tensor(
        metric_payload["landmark_indices"]
    ).long()
    if not torch.equal(landmark_indices, torch.arange(anchor_count)):
        raise ValueError("metric landmark registry does not align with map")

    example_sources = {
        key: _file_stamp(value)
        for key, value in {
            "query_cache": args.query_cache,
            "complete_positive_teacher": args.complete_positive_teacher,
            "dynamic_outcomes": args.dynamic_outcomes,
            "topk_outcomes": args.topk_outcomes,
        }.items()
    }
    example_config = {
        "clean_threshold_px": float(args.clean_threshold_px),
        "harmful_threshold_px": float(args.harmful_threshold_px),
        "maximum_rows_per_type_per_query": int(
            args.maximum_rows_per_type_per_query
        ),
        "example_seed": int(args.example_seed),
        "schema_version": 2,
    }
    examples_cache = Path(args.examples_cache).resolve() if args.examples_cache else None
    if examples_cache is not None and examples_cache.is_file():
        cached_examples = torch.load(
            examples_cache, map_location="cpu", weights_only=False
        )
        if cached_examples.get("schema") != "lafgs_confusion_metric_examples":
            raise ValueError("unsupported confusion metric examples cache")
        if cached_examples.get("sources") != example_sources:
            raise ValueError("examples cache sources do not match this run")
        if cached_examples.get("config") != example_config:
            raise ValueError("examples cache configuration does not match this run")
        examples = cached_examples["examples"]
        data_report = cached_examples["data_report"]
    else:
        cache_payload = torch.load(
            args.query_cache, map_location="cpu", weights_only=False
        )
        cache = cache_payload.get("queries", cache_payload)
        examples, data_report = _build_examples(
            cache=cache,
            teacher=teacher,
            outcomes=outcomes,
            topk=topk,
            clean_threshold_px=args.clean_threshold_px,
            harmful_threshold_px=args.harmful_threshold_px,
            maximum_rows_per_type_per_query=args.maximum_rows_per_type_per_query,
            seed=args.example_seed,
        )
        if examples_cache is not None:
            examples_cache.parent.mkdir(parents=True, exist_ok=True)
            _atomic_torch(
                examples_cache,
                {
                    "schema": "lafgs_confusion_metric_examples",
                    "version": 1,
                    "sources": example_sources,
                    "config": example_config,
                    "examples": examples,
                    "data_report": data_report,
                },
            )
    raw_key = (
        "v7_metric_raw_features"
        if "v7_metric_raw_features" in state
        else "anchor_features"
    )
    raw_bank = F.normalize(torch.as_tensor(state[raw_key]).float(), dim=1).to(
        device
    )
    metric = SharedLowRankMetric(**metric_payload["metric_config"]).to(device)
    metric.load_state_dict(metric_payload["metric_state_dict"])
    teacher_metric = copy.deepcopy(metric).eval()
    for parameter in teacher_metric.parameters():
        parameter.requires_grad_(False)
    initial_parameters = {
        name: value.detach().clone()
        for name, value in metric.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        metric.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    sampling_indices = {
        kind: _hierarchical_sampling_index(examples, kind)
        for kind in range(4)
    }
    generator = torch.Generator().manual_seed(args.seed + 1)
    history = []
    counts = [
        int(round(int(args.batch_size) * float(value)))
        for value in fractions
    ]
    counts[2] += int(args.batch_size) - sum(counts)
    for step in range(1, int(args.steps) + 1):
        selected = torch.cat(
            tuple(
                _sample_hierarchical(
                    sampling_indices[kind], counts[kind], generator
                )
                for kind in range(4)
            )
        )
        query_raw = examples["query"][selected].to(device)
        candidates = examples["candidates"][selected].to(device)
        positive_mask = examples["positive_mask"][selected].to(device)
        kind = examples["kind"][selected].to(device)
        query, _ = metric(query_raw)
        candidate, _ = metric(raw_bank[candidates])
        scores = torch.einsum("bd,bkd->bk", query, candidate)
        with torch.no_grad():
            old_query, _ = teacher_metric(query_raw)
            old_candidate, _ = teacher_metric(raw_bank[candidates])
            old_scores = torch.einsum(
                "bd,bkd->bk", old_query, old_candidate
            )
        rescue = kind == 0
        protected = kind == 1
        neutral = kind == 2
        hard = kind == 3
        rescue_loss = candidate_margin_loss(
            scores[rescue],
            positive_mask[rescue],
            margin=args.margin,
            temperature=args.temperature,
        )
        protected_loss = (
            candidate_margin_loss(
                scores[protected],
                positive_mask[protected],
                margin=args.margin,
                temperature=args.temperature,
            )
            if bool(protected.any())
            else scores.new_zeros(())
        )
        neutral_loss = topk_distribution_distillation(
            scores[neutral],
            old_scores[neutral],
            temperature=args.neutral_temperature,
        )
        hard_loss = candidate_margin_loss(
            scores[hard],
            positive_mask[hard],
            margin=args.margin,
            temperature=args.temperature,
        )
        distill = F.mse_loss(scores, old_scores)
        parameter_trust = sum(
            (value - initial_parameters[name]).square().mean()
            for name, value in metric.named_parameters()
        )
        loss = (
            rescue_loss
            + float(args.protected_weight) * protected_loss
            + float(args.neutral_replay_weight) * neutral_loss
            + float(args.hard_weight) * hard_loss
            + float(args.distill_weight) * distill
            + float(args.parameter_trust_weight) * parameter_trust
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(metric.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "rescue_loss": float(rescue_loss.detach()),
                "protected_loss": float(protected_loss.detach()),
                "neutral_loss": float(neutral_loss.detach()),
                "hard_loss": float(hard_loss.detach()),
                "distill_loss": float(distill.detach()),
                "parameter_trust": float(parameter_trust.detach()),
            }
            history.append(row)
            print(json.dumps(row), flush=True)

    with torch.inference_mode():
        transformed_bank, residual = metric(raw_bank)
        old_bank, _ = teacher_metric(raw_bank)
        bank_cosine = F.cosine_similarity(transformed_bank, old_bank, dim=1)
    contract_diagnostics = _assignment_contract_diagnostics(
        metric=metric,
        teacher_metric=teacher_metric,
        raw_bank=raw_bank,
        examples=examples,
        device=device,
        temperature=args.neutral_temperature,
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    map_path = output_dir / "anchor_map_confusion_metric.pt"
    metric_path = output_dir / "metric_state_confusion_metric.pt"
    output_state = dict(state)
    output_state["anchor_features"] = transformed_bank.cpu()
    output_state["v7_metric_raw_features"] = raw_bank.cpu()
    output_state["confusion_directed_metric"] = {
        "schema": "lafgs_confusion_directed_metric",
        "version": 1,
        "config": vars(args),
        "data_report": data_report,
        "history": history,
        "mean_bank_cosine_to_initial": float(bank_cosine.mean()),
        "p01_bank_cosine_to_initial": float(torch.quantile(bank_cosine, 0.01)),
        "assignment_contract": contract_diagnostics,
    }
    output_metric = {
        **metric_payload,
        "metric_state_dict": {
            key: value.detach().cpu() for key, value in metric.state_dict().items()
        },
        "map_path": str(map_path),
        "confusion_directed_metric": output_state["confusion_directed_metric"],
    }
    _atomic_torch(map_path, output_state)
    _atomic_torch(metric_path, output_metric)
    provenance = {
        key: {"path": str(Path(value).resolve()), "sha256": _sha256(value)}
        for key, value in {
            "map": args.map,
            "initial_metric_state": args.initial_metric_state,
            "query_cache": args.query_cache,
            "complete_positive_teacher": args.complete_positive_teacher,
            "dynamic_outcomes": args.dynamic_outcomes,
            "topk_outcomes": args.topk_outcomes,
        }.items()
    }
    report = {
        "schema": "lafgs_confusion_directed_metric_report",
        "version": 1,
        "map": str(map_path),
        "metric_state": str(metric_path),
        "data_report": data_report,
        "history": history,
        "mean_bank_cosine_to_initial": float(bank_cosine.mean()),
        "p01_bank_cosine_to_initial": float(torch.quantile(bank_cosine, 0.01)),
        "assignment_contract": contract_diagnostics,
        "provenance": provenance,
    }
    _atomic_json(output_dir / "training_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
