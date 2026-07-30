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
    if offsets.numel() != len(candidates) + 1:
        raise ValueError("positive CSR offsets do not align with candidates")
    if not positive_indices.numel():
        return torch.zeros_like(candidates, dtype=torch.bool)
    stride = int(
        max(
            int(candidates.max()),
            int(positive_indices.max()),
        )
        + 1
    )
    positive_rows = torch.repeat_interleave(
        torch.arange(len(candidates), dtype=torch.long),
        offsets[1:] - offsets[:-1],
    )
    positive_keys = torch.sort(
        positive_rows * stride + positive_indices
    ).values
    candidate_keys = (
        torch.arange(len(candidates), dtype=torch.long)[:, None] * stride
        + candidates
    )
    flat = candidate_keys.reshape(-1)
    slot = torch.searchsorted(positive_keys, flat)
    valid = slot < positive_keys.numel()
    matched = torch.zeros_like(valid)
    matched[valid] = positive_keys[slot[valid]] == flat[valid]
    return matched.reshape_as(candidates)


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
    }
    diagnostics = {
        "query_count": len(names),
        "rescue_available": 0,
        "protected_available": 0,
        "miss_or_reject_rows": 0,
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
        diagnostics["rescue_available"] += int(rescue.sum())
        diagnostics["protected_available"] += int(protected.sum())
        diagnostics["miss_or_reject_rows"] += int((~has_legal).sum())
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
        selected = torch.cat((rescue_rows, protected_rows))
        if not selected.numel():
            continue
        native = F.normalize(
            torch.as_tensor(cache[name]["native_descriptors"]).float()[rows[selected]],
            dim=1,
        )
        selected_mask = positive_mask[selected]
        if protected_rows.numel():
            selected_mask[rescue_rows.numel() :] = protected_top1_mask(
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
                )
            )
        )
        values["query_index"].append(
            torch.full((len(selected),), query_index, dtype=torch.long)
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
    diagnostics["training_rows"] = int(len(output["kind"]))
    return output, diagnostics


def _sample(rows: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    if not rows.numel():
        return rows
    return rows[
        torch.randint(rows.numel(), (int(count),), generator=generator)
    ]


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
    parser.add_argument("--parameter-trust-weight", type=float, default=0.1)
    parser.add_argument("--maximum-rows-per-type-per-query", type=int, default=96)
    parser.add_argument("--example-seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

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
    rescue_rows = torch.nonzero(
        examples["kind"] == 0, as_tuple=False
    ).reshape(-1)
    protected_rows = torch.nonzero(
        examples["kind"] == 1, as_tuple=False
    ).reshape(-1)
    generator = torch.Generator().manual_seed(args.seed + 1)
    history = []
    half = max(int(args.batch_size) // 2, 1)
    for step in range(1, int(args.steps) + 1):
        selected = torch.cat(
            (
                _sample(rescue_rows, half, generator),
                _sample(protected_rows, int(args.batch_size) - half, generator),
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
        distill = F.mse_loss(scores, old_scores)
        parameter_trust = sum(
            (value - initial_parameters[name]).square().mean()
            for name, value in metric.named_parameters()
        )
        loss = (
            rescue_loss
            + float(args.protected_weight) * protected_loss
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
                "distill_loss": float(distill.detach()),
                "parameter_trust": float(parameter_trust.detach()),
            }
            history.append(row)
            print(json.dumps(row), flush=True)

    with torch.inference_mode():
        transformed_bank, residual = metric(raw_bank)
        old_bank, _ = teacher_metric(raw_bank)
        bank_cosine = F.cosine_similarity(transformed_bank, old_bank, dim=1)
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
        "provenance": provenance,
    }
    _atomic_json(output_dir / "training_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
