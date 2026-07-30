#!/usr/bin/env python3
"""Train bounded 2D-query/3D-map context alignment on real self-localization."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from localization_training.contextual_descriptor import (
    BoundedDualContextEncoder,
    flatten_context,
    joint_local_context_similarity,
    multiscale_sparse_query_context,
)
from localization_training.shared_metric import SharedLowRankMetric
from localization_training.relational_context import (
    AsymmetricBoundedDualContextEncoder,
    relational_sparse_query_context,
)


def _load_metric(path: str, device: torch.device) -> SharedLowRankMetric:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metric = SharedLowRankMetric(**payload["metric_config"]).to(device)
    metric.load_state_dict(payload["metric_state_dict"])
    metric.eval()
    for parameter in metric.parameters():
        parameter.requires_grad_(False)
    return metric


def _dynamic_by_name(payload: dict) -> dict[str, dict]:
    return {
        str(record["query_name"]): record
        for record in payload["records"]
    }


def _confusion_events_by_name(payload: dict | None) -> dict[str, dict[int, list[dict]]]:
    if payload is None:
        return {}
    output: dict[str, dict[int, list[dict]]] = {}
    for event in payload["events"]:
        name = str(event["query_name"])
        row = int(event["query_row"])
        output.setdefault(name, {}).setdefault(row, []).append(event)
    return output


def _aligned_topk(
    record: dict,
    rows: torch.Tensor,
    valid_slots: torch.Tensor,
) -> torch.Tensor:
    topk_rows = torch.as_tensor(record["query_rows"]).long()
    topk = torch.as_tensor(record["topk_anchor_indices"]).long()
    if topk.ndim != 2 or topk.shape[0] != topk_rows.numel():
        raise ValueError("top-K outcome rows and candidates do not align")
    if torch.equal(topk_rows, rows):
        return topk[valid_slots]
    row_to_slot = {
        int(row): index for index, row in enumerate(topk_rows.tolist())
    }
    missing = [
        int(rows[slot])
        for slot in valid_slots.tolist()
        if int(rows[slot]) not in row_to_slot
    ]
    if missing:
        raise ValueError(f"top-K outcomes miss teacher rows: {missing[:3]}")
    return torch.stack(
        [topk[row_to_slot[int(rows[slot])]] for slot in valid_slots]
    )


def _sample_slots_by_category(
    categories: tuple[torch.Tensor, ...],
    valid_slots: torch.Tensor,
    *,
    maximum_rows: int,
    generator: torch.Generator,
    fractions: tuple[float, ...] = (0.40, 0.25, 0.20, 0.15),
) -> torch.Tensor:
    """Sample disjoint row categories, then deterministically fill shortages."""

    if len(categories) != len(fractions):
        raise ValueError("sampling categories and fractions must align")
    if maximum_rows <= 0 or valid_slots.numel() <= int(maximum_rows):
        return valid_slots
    selected_parts = []
    selected_mask = torch.zeros(valid_slots.numel(), dtype=torch.bool)
    remaining_budget = int(maximum_rows)
    for category, fraction in zip(categories, fractions):
        requested = min(
            int(round(int(maximum_rows) * float(fraction))),
            remaining_budget,
        )
        available = torch.nonzero(
            torch.as_tensor(category).bool() & ~selected_mask,
            as_tuple=False,
        ).reshape(-1)
        if requested > 0 and available.numel():
            order = torch.randperm(available.numel(), generator=generator)
            chosen = available[order[:requested]]
            selected_parts.append(chosen)
            selected_mask[chosen] = True
            remaining_budget -= int(chosen.numel())
    if remaining_budget > 0:
        available = torch.nonzero(
            ~selected_mask, as_tuple=False
        ).reshape(-1)
        order = torch.randperm(available.numel(), generator=generator)
        selected_parts.append(available[order[:remaining_budget]])
    selected = torch.cat(selected_parts)
    return valid_slots[selected].sort().values


def _selected_training_batch(
    teacher: dict,
    dynamic: dict,
    *,
    maximum_rows: int,
    generator: torch.Generator,
    anchor_count: int,
    topk: dict | None = None,
    confusion_events: dict[int, list[dict]] | None = None,
    row_sampling: str = "uniform",
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    rows = torch.as_tensor(teacher["query_rows"]).long()
    offsets = torch.as_tensor(teacher["positive_offsets"]).long()
    positives = torch.as_tensor(teacher["positive_indices"]).long()
    valid_slots = torch.nonzero(
        offsets[1:] > offsets[:-1], as_tuple=False
    ).reshape(-1)
    if row_sampling not in {"uniform", "stratified"}:
        raise ValueError("row_sampling must be uniform or stratified")
    if (
        row_sampling == "stratified"
        and maximum_rows > 0
        and valid_slots.numel() > int(maximum_rows)
    ):
        dynamic_rows = torch.as_tensor(dynamic["query_rows"]).long()
        dynamic_top1 = torch.as_tensor(
            dynamic["top1_anchor_indices"]
        ).long()
        top1_by_row = {
            int(row): int(anchor)
            for row, anchor in zip(
                dynamic_rows.tolist(), dynamic_top1.tolist()
            )
        }
        top1_all = torch.as_tensor(
            [top1_by_row[int(rows[slot])] for slot in valid_slots]
        ).long()
        topk_all = (
            _aligned_topk(topk, rows, valid_slots)
            if topk is not None
            else top1_all[:, None]
        )
        clean = torch.zeros(valid_slots.numel(), dtype=torch.bool)
        recoverable = torch.zeros_like(clean)
        confusion_rows = torch.zeros_like(clean)
        for local_slot, source_slot in enumerate(valid_slots.tolist()):
            positive_set = set(
                int(value)
                for value in positives[
                    offsets[source_slot] : offsets[source_slot + 1]
                ].tolist()
            )
            clean[local_slot] = int(top1_all[local_slot]) in positive_set
            recoverable[local_slot] = any(
                int(value) in positive_set
                for value in topk_all[local_slot].tolist()
            )
            confusion_rows[local_slot] = bool(
                confusion_events
                and int(rows[source_slot]) in confusion_events
            )
        categories = (
            confusion_rows,
            clean & ~confusion_rows,
            recoverable & ~clean & ~confusion_rows,
            ~confusion_rows & ~clean & ~recoverable,
        )
        valid_slots = _sample_slots_by_category(
            categories,
            valid_slots,
            maximum_rows=int(maximum_rows),
            generator=generator,
        )
    elif maximum_rows > 0 and valid_slots.numel() > int(maximum_rows):
        order = torch.randperm(valid_slots.numel(), generator=generator)
        valid_slots = valid_slots[order[: int(maximum_rows)]]
    valid_slots = valid_slots.sort().values
    if not valid_slots.numel():
        return (
            rows.new_empty(0),
            rows.new_empty((0, 0), dtype=torch.bool),
            rows.new_empty(0),
            rows.new_empty(0, dtype=torch.bool),
            rows.new_empty(0),
            rows.new_empty(0),
            rows.new_empty(0),
            rows.new_empty(0),
            torch.empty(0, dtype=torch.float32),
        )
    slot_lookup = torch.full((rows.numel(),), -1, dtype=torch.long)
    slot_lookup[valid_slots] = torch.arange(valid_slots.numel())
    edge_slots = torch.repeat_interleave(
        torch.arange(rows.numel()), offsets[1:] - offsets[:-1]
    )
    retained_edges = slot_lookup[edge_slots] >= 0
    local_slots = slot_lookup[edge_slots[retained_edges]]
    positive_anchors = positives[retained_edges]

    dynamic_rows = torch.as_tensor(dynamic["query_rows"]).long()
    dynamic_top1 = torch.as_tensor(
        dynamic["top1_anchor_indices"]
    ).long()
    if torch.equal(dynamic_rows, rows):
        top1 = dynamic_top1[valid_slots]
    else:
        row_to_top1 = {
            int(row): int(anchor)
            for row, anchor in zip(
                dynamic_rows.tolist(), dynamic_top1.tolist()
            )
        }
        top1 = torch.as_tensor(
            [row_to_top1[int(rows[slot])] for slot in valid_slots]
        ).long()
    candidate_parts = [positive_anchors, top1]
    if topk is not None:
        candidate_parts.append(
            _aligned_topk(topk, rows, valid_slots).reshape(-1)
        )
    selected_rows = rows[valid_slots]
    selected_row_lookup = {
        int(row): local_slot
        for local_slot, row in enumerate(selected_rows.tolist())
    }
    directed_events: list[tuple[int, int, int, float]] = []
    if confusion_events:
        graph_anchors = []
        for row, local_slot in selected_row_lookup.items():
            for event in confusion_events.get(row, ()):
                correct = int(event["correct_anchor"])
                confusing = int(event["confusing_anchor"])
                if not (
                    0 <= correct < int(anchor_count)
                    and 0 <= confusing < int(anchor_count)
                ):
                    raise ValueError("confusion event anchor is outside map")
                graph_anchors.extend((correct, confusing))
                directed_events.append(
                    (
                        local_slot,
                        correct,
                        confusing,
                        max(float(event.get("pose_blame", 0.0)), 0.0),
                    )
                )
        if graph_anchors:
            candidate_parts.append(torch.as_tensor(graph_anchors).long())
    candidates = torch.unique(torch.cat(candidate_parts), sorted=True)
    candidate_lookup = torch.full(
        (int(anchor_count),), -1, dtype=torch.long
    )
    candidate_lookup[candidates] = torch.arange(candidates.numel())
    positive_mask = torch.zeros(
        (valid_slots.numel(), candidates.numel()), dtype=torch.bool
    )
    positive_mask[
        local_slots, candidate_lookup[positive_anchors]
    ] = True
    top1_positions = candidate_lookup[top1]
    clean_top1 = positive_mask[
        torch.arange(valid_slots.numel()), top1_positions
    ]
    if directed_events:
        directed_rows = torch.as_tensor(
            [event[0] for event in directed_events]
        ).long()
        directed_correct = candidate_lookup[
            torch.as_tensor([event[1] for event in directed_events]).long()
        ]
        directed_confusing = candidate_lookup[
            torch.as_tensor([event[2] for event in directed_events]).long()
        ]
        directed_weights = torch.as_tensor(
            [event[3] for event in directed_events], dtype=torch.float32
        )
        directed_weights = 1.0 + torch.log1p(directed_weights)
        directed_weights = directed_weights / directed_weights.mean().clamp_min(
            1e-8
        )
    else:
        directed_rows = rows.new_empty(0)
        directed_correct = rows.new_empty(0)
        directed_confusing = rows.new_empty(0)
        directed_weights = torch.empty(0, dtype=torch.float32)
    return (
        selected_rows,
        positive_mask,
        candidates,
        clean_top1,
        top1_positions,
        directed_rows,
        directed_correct,
        directed_confusing,
        directed_weights,
    )


def _multi_positive_loss(
    scores: torch.Tensor,
    positive_mask: torch.Tensor,
    clean_top1: torch.Tensor,
    *,
    clean_weight: float,
    error_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    numerator = torch.logsumexp(
        scores.masked_fill(~positive_mask, -torch.inf), dim=1
    )
    denominator = torch.logsumexp(scores, dim=1)
    weights = torch.where(
        clean_top1,
        scores.new_full((len(scores),), float(clean_weight)),
        scores.new_full((len(scores),), float(error_weight)),
    )
    loss = ((denominator - numerator) * weights).sum() / weights.sum().clamp_min(
        1e-8
    )
    return loss, numerator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--complete-positive-teacher", required=True)
    parser.add_argument("--dynamic-outcomes", required=True)
    parser.add_argument(
        "--topk-outcomes",
        default="",
        help="Optional exact top-K candidates aligned to the fixed deployment matcher.",
    )
    parser.add_argument(
        "--confusion-graph",
        default="",
        help="Optional query-specific correct/confusing event graph.",
    )
    parser.add_argument("--raw-context-state", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--maximum-rows-per-query", type=int, default=512)
    parser.add_argument(
        "--row-sampling",
        choices=("uniform", "stratified"),
        default="uniform",
    )
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--maximum-residual", type=float, default=0.35)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--hard-negative-weight", type=float, default=0.5)
    parser.add_argument("--keep-top1-weight", type=float, default=1.0)
    parser.add_argument("--keep-top1-margin", type=float, default=0.02)
    parser.add_argument("--trust-weight", type=float, default=0.2)
    parser.add_argument("--directed-confusion-weight", type=float, default=0.0)
    parser.add_argument("--directed-confusion-margin", type=float, default=0.05)
    parser.add_argument("--clean-weight", type=float, default=2.0)
    parser.add_argument("--error-weight", type=float, default=3.0)
    parser.add_argument(
        "--context-weight",
        type=float,
        default=0.05,
        help="Context weight used by both the training score and deployment.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    teacher = torch.load(
        args.complete_positive_teacher,
        map_location="cpu",
        weights_only=False,
    )
    dynamic_payload = torch.load(
        args.dynamic_outcomes, map_location="cpu", weights_only=False
    )
    dynamic = _dynamic_by_name(dynamic_payload)
    topk_payload = (
        torch.load(
            args.topk_outcomes, map_location="cpu", weights_only=False
        )
        if args.topk_outcomes
        else None
    )
    topk = (
        _dynamic_by_name(topk_payload) if topk_payload is not None else {}
    )
    if topk_payload is not None:
        if topk_payload.get("schema") != "lafgs_exact_topk_outcomes":
            raise ValueError("unsupported top-K outcome state")
        if int(topk_payload["anchor_count"]) != len(state["anchor_xyz"]):
            raise ValueError("top-K outcomes do not align with active map")
    confusion_payload = (
        torch.load(
            args.confusion_graph, map_location="cpu", weights_only=False
        )
        if args.confusion_graph
        else None
    )
    confusion = _confusion_events_by_name(confusion_payload)
    if confusion_payload is not None:
        if (
            confusion_payload.get("schema")
            != "lafgs_anchor_family_confusion_graph"
        ):
            raise ValueError("unsupported confusion graph")
        if int(confusion_payload["anchor_count"]) != len(state["anchor_xyz"]):
            raise ValueError("confusion graph does not align with active map")
    raw_context_payload = torch.load(
        args.raw_context_state, map_location="cpu", weights_only=False
    )
    if raw_context_payload.get("schema") != "lafgs_fixed_3d_context_state":
        raise ValueError("dual context training requires fixed 3D context")
    if not torch.equal(
        torch.as_tensor(raw_context_payload["anchor_ids"]).long(),
        torch.as_tensor(state["anchor_ids"]).long(),
    ):
        raise ValueError("raw context state does not align with active map")
    names = list(teacher["query_names"])
    if args.query_limit > 0:
        names = names[: int(args.query_limit)]
    if any(
        name not in cache
        or name not in dynamic
        or (topk_payload is not None and name not in topk)
        for name in names
    ):
        raise ValueError("dual context training query registries differ")
    teacher_by_name = {
        str(record["query_name"]): record
        for record in teacher["records"]
    }
    raw_map_context = F.normalize(
        torch.as_tensor(
            raw_context_payload["anchor_context"]
        ).float().to(device),
        dim=1,
    )
    map_local = F.normalize(
        torch.as_tensor(state["anchor_features"]).float().to(device),
        dim=1,
    )
    input_dim = int(raw_map_context.shape[1])
    context_config = raw_context_payload["config"]
    if (
        context_config.get("query_representation")
        == "relational_sparse_2d_v1"
    ):
        model = AsymmetricBoundedDualContextEncoder(
            query_input_dim=int(context_config["query_input_dim"]),
            map_input_dim=input_dim,
            output_dim=args.output_dim,
            rank=args.rank,
            maximum_residual=args.maximum_residual,
            seed=args.seed,
        ).to(device)
    else:
        model = BoundedDualContextEncoder(
            input_dim=input_dim,
            output_dim=args.output_dim,
            rank=args.rank,
            maximum_residual=args.maximum_residual,
            seed=args.seed,
        ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    metric = _load_metric(args.metric_state, device)
    history = []
    for epoch in range(args.epochs):
        order = list(names)
        random.Random(args.seed + epoch).shuffle(order)
        records = []
        model.train()
        for query_index, name in enumerate(
            tqdm(order, desc=f"Dual context {epoch + 1}/{args.epochs}")
        ):
            generator = torch.Generator().manual_seed(
                args.seed * 1000003 + epoch * len(order) + query_index
            )
            (
                selected_rows,
                positive_mask,
                candidates,
                clean_top1,
                top1_positions,
                directed_rows,
                directed_correct,
                directed_confusing,
                directed_weights,
            ) = _selected_training_batch(
                teacher_by_name[name],
                dynamic[name],
                maximum_rows=args.maximum_rows_per_query,
                generator=generator,
                anchor_count=len(raw_map_context),
                topk=topk.get(name),
                confusion_events=confusion.get(name),
                row_sampling=args.row_sampling,
            )
            if not selected_rows.numel():
                continue
            cached = cache[name]
            all_descriptors = F.normalize(
                torch.as_tensor(
                    cached["native_descriptors"]
                ).float().to(device),
                dim=1,
            )
            if (
                context_config.get("query_representation")
                == "relational_sparse_2d_v1"
            ):
                with torch.no_grad():
                    transformed_all, _ = metric(all_descriptors)
                    query_local = transformed_all[selected_rows.to(device)]
                query_context = relational_sparse_query_context(
                    transformed_all,
                    torch.as_tensor(
                        cached["native_keypoints"]
                    ).float().to(device),
                    torch.as_tensor(
                        cached["native_scores"]
                    ).float().to(device),
                    neighbor_count=int(
                        context_config["query_neighbor_count"]
                    ),
                    chunk_size=int(
                        context_config.get("context_chunk_size", 256)
                    ),
                )
                query_context = query_context[selected_rows.to(device)]
            else:
                query_context = multiscale_sparse_query_context(
                    all_descriptors,
                    torch.as_tensor(
                        cached["native_keypoints"]
                    ).float().to(device),
                    torch.as_tensor(
                        cached["native_scores"]
                    ).float().to(device),
                    radii_px=tuple(
                        float(value)
                        for value in context_config["sparse_radii_px"]
                    ),
                    maximum_neighbors=int(
                        context_config["maximum_sparse_neighbors"]
                    ),
                    chunk_size=int(
                        context_config.get("context_chunk_size", 256)
                    ),
                )
                shape = query_context.shape
                with torch.no_grad():
                    transformed, _ = metric(
                        query_context.reshape(-1, shape[-1])
                    )
                    query_local, _ = metric(
                        all_descriptors[selected_rows.to(device)]
                    )
                query_context = flatten_context(
                    transformed.reshape(shape)
                )[selected_rows.to(device)]
            candidates = candidates.to(device)
            query_embedding, query_residual = model.query(query_context)
            map_embedding, map_residual = model.map(
                raw_map_context[candidates]
            )
            scores = joint_local_context_similarity(
                query_local,
                map_local[candidates],
                query_embedding,
                map_embedding,
                context_weight=float(args.context_weight),
            )
            scaled_scores = scores / max(float(args.temperature), 1e-6)
            positive_mask = positive_mask.to(device)
            clean_top1 = clean_top1.to(device)
            top1_positions = top1_positions.to(device)
            metric_loss, positive_score = _multi_positive_loss(
                scaled_scores,
                positive_mask,
                clean_top1,
                clean_weight=args.clean_weight,
                error_weight=args.error_weight,
            )
            hard = ~clean_top1
            if bool(hard.any()):
                negative_score = scaled_scores[hard].masked_fill(
                    positive_mask[hard], -torch.inf
                ).max(dim=1).values
                hard_loss = F.softplus(
                    negative_score
                    - positive_score[hard]
                    + float(args.margin)
                ).mean()
            else:
                hard_loss = scores.new_zeros(())
            if bool(clean_top1.any()):
                clean_scores = scaled_scores[clean_top1]
                clean_positive_mask = positive_mask[clean_top1]
                protected_score = clean_scores[
                    torch.arange(
                        int(clean_top1.sum()), device=device
                    ),
                    top1_positions[clean_top1],
                ]
                strongest_negative = clean_scores.masked_fill(
                    clean_positive_mask, -torch.inf
                ).max(dim=1).values
                keep_loss = F.softplus(
                    strongest_negative
                    - protected_score
                    + float(args.keep_top1_margin)
                ).mean()
            else:
                keep_loss = scores.new_zeros(())
            if directed_rows.numel():
                directed_rows = directed_rows.to(device)
                directed_correct = directed_correct.to(device)
                directed_confusing = directed_confusing.to(device)
                directed_weights = directed_weights.to(device)
                directed_delta = (
                    scores[directed_rows, directed_confusing]
                    - scores[directed_rows, directed_correct]
                    + float(args.directed_confusion_margin)
                ) / max(float(args.temperature), 1e-6)
                directed_loss = (
                    F.softplus(directed_delta) * directed_weights
                ).mean()
                directed_accuracy = (
                    directed_delta
                    < float(args.directed_confusion_margin)
                    / max(float(args.temperature), 1e-6)
                ).float().mean()
            else:
                directed_loss = scores.new_zeros(())
                directed_accuracy = scores.new_zeros(())
            trust_loss = (
                query_residual.square().sum(dim=1).mean()
                + map_residual.square().sum(dim=1).mean()
            )
            loss = (
                metric_loss
                + float(args.hard_negative_weight) * hard_loss
                + float(args.keep_top1_weight) * keep_loss
                + float(args.directed_confusion_weight) * directed_loss
                + float(args.trust_weight) * trust_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            records.append(
                {
                    "loss": float(loss.detach()),
                    "metric": float(metric_loss.detach()),
                    "hard": float(hard_loss.detach()),
                    "keep": float(keep_loss.detach()),
                    "directed": float(directed_loss.detach()),
                    "directed_accuracy": float(
                        directed_accuracy.detach()
                    ),
                    "trust": float(trust_loss.detach()),
                    "clean_fraction": float(clean_top1.float().mean()),
                    "query_residual": float(
                        query_residual.norm(dim=1).mean().detach()
                    ),
                    "map_residual": float(
                        map_residual.norm(dim=1).mean().detach()
                    ),
                }
            )
        epoch_summary = {
            "epoch": epoch + 1,
            **{
                key: sum(record[key] for record in records) / len(records)
                for key in records[0]
                if key != "epoch"
            },
        }
        history.append(epoch_summary)
        print(json.dumps(epoch_summary), flush=True)

    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(raw_map_context), 2048):
            embedding, _ = model.map(
                raw_map_context[start : start + 2048]
            )
            chunks.append(embedding.cpu())
    anchor_context = torch.cat(chunks)
    output = {
        "schema": "lafgs_fixed_3d_context_state",
        "version": 2,
        "anchor_ids": torch.as_tensor(state["anchor_ids"]).long().cpu(),
        "anchor_ids_sha256": raw_context_payload["anchor_ids_sha256"],
        "anchor_context": anchor_context.half(),
        "context_dim": int(anchor_context.shape[1]),
        "query_projector_config": model.query.export_config(),
        "query_projector_state_dict": {
            key: value.detach().cpu()
            for key, value in model.query.state_dict().items()
        },
        "training": {
            "history": history,
            "config": vars(args),
        },
        "config": {
            **context_config,
            "representation": "bounded_dual_2d3d_context_v1",
            "output_dim": int(args.output_dim),
        },
        "provenance": {
            "map": str(Path(args.map).resolve()),
            "metric_state": str(Path(args.metric_state).resolve()),
            "query_cache": str(Path(args.query_cache).resolve()),
            "complete_positive_teacher": str(
                Path(args.complete_positive_teacher).resolve()
            ),
            "dynamic_outcomes": str(
                Path(args.dynamic_outcomes).resolve()
            ),
            "topk_outcomes": str(Path(args.topk_outcomes).resolve())
            if args.topk_outcomes
            else None,
            "confusion_graph": str(Path(args.confusion_graph).resolve())
            if args.confusion_graph
            else None,
            "raw_context_state": str(
                Path(args.raw_context_state).resolve()
            ),
        },
    }
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, path)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": output["schema"],
                "version": output["version"],
                "context_dim": output["context_dim"],
                "training": output["training"],
                "config": output["config"],
                "provenance": output["provenance"],
            },
            indent=2,
        )
        + "\n"
    )
    print({"output": str(path), "history": history})


if __name__ == "__main__":
    main()
