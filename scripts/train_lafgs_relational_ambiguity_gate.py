#!/usr/bin/env python3
"""Train a query-only gate from recoverable top-16 assignment failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from localization_training.contextual_descriptor import (
    BoundedContextProjector,
)
from localization_training.relational_context import (
    QueryAmbiguityGate,
    relational_sparse_query_context,
)
from localization_training.shared_metric import SharedLowRankMetric


def _records(payload: dict) -> dict[str, dict]:
    return {
        str(record["query_name"]): record for record in payload["records"]
    }


def _load_metric(path: str, device: torch.device) -> SharedLowRankMetric:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metric = SharedLowRankMetric(**payload["metric_config"]).to(device)
    metric.load_state_dict(payload["metric_state_dict"])
    metric.eval()
    return metric


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--context-state", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--complete-positive-teacher", required=True)
    parser.add_argument("--topk-outcomes", required=True)
    parser.add_argument("--confusion-graph", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--targeted-weight", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    context_state = torch.load(
        args.context_state, map_location="cpu", weights_only=False
    )
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    teacher = torch.load(
        args.complete_positive_teacher,
        map_location="cpu",
        weights_only=False,
    )
    topk = torch.load(
        args.topk_outcomes, map_location="cpu", weights_only=False
    )
    graph = torch.load(
        args.confusion_graph, map_location="cpu", weights_only=False
    )
    if (
        context_state["config"].get("query_representation")
        != "relational_sparse_2d_v1"
    ):
        raise ValueError("ambiguity gate requires relational context")
    teacher_by_name = _records(teacher)
    topk_by_name = _records(topk)
    targeted = {
        (str(event["query_name"]), int(event["query_row"]))
        for event in graph["events"]
    }
    metric = _load_metric(args.metric_state, device)
    projector = BoundedContextProjector(
        **context_state["query_projector_config"]
    ).to(device)
    projector.load_state_dict(
        context_state["query_projector_state_dict"]
    )
    projector.eval()
    config = context_state["config"]
    all_context = []
    all_scores = []
    all_labels = []
    all_weights = []
    with torch.no_grad():
        for name in tqdm(teacher["query_names"], desc="Gate evidence"):
            teacher_record = teacher_by_name[name]
            topk_record = topk_by_name[name]
            rows = torch.as_tensor(teacher_record["query_rows"]).long()
            if not torch.equal(
                rows, torch.as_tensor(topk_record["query_rows"]).long()
            ):
                raise ValueError(f"gate top-K rows differ for {name}")
            offsets = torch.as_tensor(
                teacher_record["positive_offsets"]
            ).long()
            positives = torch.as_tensor(
                teacher_record["positive_indices"]
            ).long()
            valid_slots = torch.nonzero(
                offsets[1:] > offsets[:-1], as_tuple=False
            ).reshape(-1)
            if not valid_slots.numel():
                continue
            top16 = torch.as_tensor(
                topk_record["topk_anchor_indices"]
            ).long()[valid_slots]
            labels = []
            retained = []
            weights = []
            for local_index, slot in enumerate(valid_slots.tolist()):
                positive = set(
                    int(value)
                    for value in positives[
                        offsets[slot] : offsets[slot + 1]
                    ].tolist()
                )
                top = [int(value) for value in top16[local_index].tolist()]
                if top[0] in positive:
                    label = 0.0
                elif positive.intersection(top):
                    label = 1.0
                else:
                    continue
                retained.append(slot)
                labels.append(label)
                weights.append(
                    float(args.targeted_weight)
                    if (name, int(rows[slot])) in targeted
                    else 1.0
                )
            if not retained:
                continue
            cached = cache[name]
            descriptors = F.normalize(
                torch.as_tensor(
                    cached["native_descriptors"]
                ).float().to(device),
                dim=1,
            )
            descriptors, _ = metric(descriptors)
            relation = relational_sparse_query_context(
                descriptors,
                torch.as_tensor(
                    cached["native_keypoints"]
                ).float().to(device),
                torch.as_tensor(
                    cached["native_scores"]
                ).float().to(device),
                neighbor_count=int(config["query_neighbor_count"]),
                chunk_size=int(config.get("context_chunk_size", 256)),
            )
            embedding, _ = projector(
                relation[rows[torch.as_tensor(retained)].to(device)]
            )
            all_context.append(embedding.cpu())
            all_scores.append(
                torch.as_tensor(cached["native_scores"]).float()[
                    rows[torch.as_tensor(retained)]
                ]
            )
            all_labels.append(torch.as_tensor(labels).float())
            all_weights.append(torch.as_tensor(weights).float())

    contexts = torch.cat(all_context)
    keypoint_scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    weights = torch.cat(all_weights)
    gate = QueryAmbiguityGate(
        context_dim=contexts.shape[1],
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        gate.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    positives = float(labels.sum())
    positive_weight = (len(labels) - positives) / max(positives, 1.0)
    history = []
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(args.epochs):
        order = torch.randperm(len(labels), generator=generator)
        losses = []
        accuracies = []
        for start in range(0, len(order), 4096):
            batch = order[start : start + 4096]
            prediction = gate(
                contexts[batch].to(device),
                keypoint_scores[batch].to(device),
            )
            batch_labels = labels[batch].to(device)
            batch_weights = weights[batch].to(device) * torch.where(
                batch_labels > 0.5,
                batch_labels.new_full((), positive_weight),
                batch_labels.new_ones(()),
            )
            loss = (
                F.binary_cross_entropy(
                    prediction, batch_labels, reduction="none"
                )
                * batch_weights
            ).sum() / batch_weights.sum().clamp_min(1e-8)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            accuracies.append(
                float(
                    ((prediction > 0.5) == (batch_labels > 0.5))
                    .float()
                    .mean()
                )
            )
        history.append(
            {
                "epoch": epoch + 1,
                "loss": sum(losses) / len(losses),
                "accuracy": sum(accuracies) / len(accuracies),
            }
        )
    output = dict(context_state)
    output["version"] = 5
    output["query_gate_config"] = gate.export_config()
    output["query_gate_state_dict"] = {
        key: value.detach().cpu()
        for key, value in gate.state_dict().items()
    }
    output["query_gate_training"] = {
        "history": history,
        "row_count": len(labels),
        "positive_fraction": float(labels.mean()),
        "positive_weight": positive_weight,
        "config": vars(args),
    }
    output["config"] = {
        **context_state["config"],
        "query_gate": "top16_recoverable_query_only_v1",
    }
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, path)
    path.with_suffix(".json").write_text(
        json.dumps(output["query_gate_training"], indent=2) + "\n"
    )
    print(
        {
            "output": str(path),
            **{
                key: output["query_gate_training"][key]
                for key in (
                    "row_count",
                    "positive_fraction",
                    "positive_weight",
                )
            },
            "last": history[-1],
        }
    )


if __name__ == "__main__":
    main()
