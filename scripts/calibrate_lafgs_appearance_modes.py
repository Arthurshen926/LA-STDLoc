#!/usr/bin/env python3
"""Calibrate high-recall appearance modes against the deployed hard assignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from localization_training.shared_metric import SharedLowRankMetric


def _parse_biases(value: str) -> torch.Tensor:
    biases = sorted(
        {min(float(item), 0.0) for item in value.split(",") if item.strip()},
        reverse=True,
    )
    if not biases:
        raise ValueError("at least one non-positive bias is required")
    return torch.as_tensor(biases, dtype=torch.float32)


def _legal_matrix(
    record: dict,
    deployment_rows: torch.Tensor,
    prototype_parents: torch.Tensor,
    parent_columns: dict[int, torch.Tensor],
    *,
    device: torch.device,
) -> torch.Tensor:
    output = torch.zeros(
        (deployment_rows.numel(), prototype_parents.numel()),
        dtype=torch.bool,
        device=device,
    )
    teacher_rows = torch.as_tensor(record["query_rows"]).long()
    offsets = torch.as_tensor(record["positive_offsets"]).long()
    positives = torch.as_tensor(record["positive_indices"]).long()
    deployment_lookup = {
        int(row): index for index, row in enumerate(deployment_rows.tolist())
    }
    for row_index, row in enumerate(teacher_rows.tolist()):
        target_row = deployment_lookup.get(int(row))
        if target_row is None:
            continue
        for anchor in positives[
            offsets[row_index] : offsets[row_index + 1]
        ].tolist():
            columns = parent_columns.get(int(anchor))
            if columns is not None:
                output[target_row, columns.to(device)] = True
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode-pool", required=True)
    parser.add_argument("--base-family", default="")
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--complete-positive-teacher", required=True)
    parser.add_argument("--dynamic-outcomes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--biases", default="0,-0.01,-0.02,-0.03,-0.05,-0.08")
    parser.add_argument("--minimum-legal-activations", type=int, default=3)
    parser.add_argument("--minimum-activation-precision", type=float, default=0.8)
    parser.add_argument("--false-activation-cost", type=float, default=2.0)
    parser.add_argument("--maximum-selected", type=int, default=2048)
    args = parser.parse_args()

    device = torch.device(args.device)
    pool = torch.load(args.mode_pool, map_location="cpu", weights_only=False)
    positives = torch.load(
        args.complete_positive_teacher, map_location="cpu", weights_only=False
    )
    dynamic = torch.load(
        args.dynamic_outcomes, map_location="cpu", weights_only=False
    )
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    metric_payload = torch.load(
        args.metric_state, map_location="cpu", weights_only=False
    )
    metric = SharedLowRankMetric(**metric_payload["metric_config"]).to(device)
    metric.load_state_dict(metric_payload["metric_state_dict"])
    metric.eval()
    names = list(dynamic["query_names"])
    if names != list(positives["query_names"]):
        raise ValueError("dynamic and positive-teacher query registries differ")

    prototypes = F.normalize(
        torch.as_tensor(pool["prototype_features"]).float(), dim=1
    ).to(device)
    parents = torch.as_tensor(pool["prototype_anchor_indices"]).long()
    biases = _parse_biases(args.biases).to(device)
    true_counts = torch.zeros(
        (len(biases), len(prototypes)), dtype=torch.long, device=device
    )
    false_counts = torch.zeros_like(true_counts)
    parent_columns: dict[int, torch.Tensor] = {}
    for parent in parents.unique().tolist():
        parent_columns[int(parent)] = torch.nonzero(
            parents == int(parent), as_tuple=False
        ).reshape(-1)

    with torch.no_grad():
        for query_index, name in enumerate(names):
            outcome = dynamic["records"][query_index]
            rows = torch.as_tensor(outcome["query_rows"]).long()
            raw = F.normalize(
                torch.as_tensor(cache[name]["native_descriptors"]).float()[rows],
                dim=1,
            ).to(device)
            query, _ = metric(raw)
            delta = query @ prototypes.T
            delta -= torch.as_tensor(outcome["top1_scores"]).float().to(device)[
                :, None
            ]
            legal = _legal_matrix(
                positives["records"][query_index],
                rows,
                parents,
                parent_columns,
                device=device,
            )
            for bias_index, bias in enumerate(biases):
                active = delta + bias > 0
                true_counts[bias_index] += (active & legal).sum(dim=0)
                false_counts[bias_index] += (active & ~legal).sum(dim=0)
            if (query_index + 1) % 25 == 0:
                print(
                    json.dumps(
                        {
                            "completed": query_index + 1,
                            "query_count": len(names),
                        }
                    ),
                    flush=True,
                )

    precision = true_counts.float() / (
        true_counts + false_counts
    ).float().clamp_min(1)
    utility = true_counts.float() - float(args.false_activation_cost) * false_counts
    valid = (
        (true_counts >= int(args.minimum_legal_activations))
        & (precision >= float(args.minimum_activation_precision))
    )
    utility = utility.masked_fill(~valid, -torch.inf)
    best_utility, best_bias_index = utility.max(dim=0)
    keep = torch.isfinite(best_utility)
    selected = torch.nonzero(keep, as_tuple=False).reshape(-1)
    if selected.numel():
        order = torch.argsort(best_utility[selected], descending=True)
        selected = selected[order]
    if int(args.maximum_selected) > 0:
        selected = selected[: int(args.maximum_selected)]

    selected_features = prototypes[selected.to(device)].detach().cpu()
    selected_parents = parents[selected.cpu()]
    selected_bias = biases[best_bias_index[selected.to(device)]].cpu()
    selected_temperature = torch.ones(len(selected))
    selected_families = []
    for index in selected.tolist():
        metadata = dict(pool["families"][index])
        bias_index = int(best_bias_index[index])
        metadata.update(
            {
                "candidate_index": int(index),
                "calibrated_bias": float(biases[bias_index]),
                "legal_activation_count": int(true_counts[bias_index, index]),
                "false_activation_count": int(false_counts[bias_index, index]),
                "activation_precision": float(precision[bias_index, index]),
                "activation_utility": float(best_utility[index]),
            }
        )
        selected_families.append(metadata)

    if args.base_family:
        base = torch.load(
            args.base_family, map_location="cpu", weights_only=False
        )
        base_count = int(torch.as_tensor(base["prototype_features"]).shape[0])
        selected_features = torch.cat(
            (
                torch.as_tensor(base["prototype_features"]).float(),
                selected_features,
            )
        )
        selected_parents = torch.cat(
            (
                torch.as_tensor(base["prototype_anchor_indices"]).long(),
                selected_parents,
            )
        )
        selected_bias = torch.cat(
            (
                torch.as_tensor(
                    base.get("prototype_bias", torch.zeros(base_count))
                ).float(),
                selected_bias,
            )
        )
        selected_temperature = torch.cat(
            (
                torch.as_tensor(
                    base.get(
                        "prototype_temperature", torch.ones(base_count)
                    )
                ).float(),
                selected_temperature,
            )
        )
        selected_families = [
            {**dict(value), "candidate_source": "base_family"}
            for value in base["families"]
        ] + [
            {**value, "candidate_source": "appearance_pool"}
            for value in selected_families
        ]
    selected_features = selected_features.detach()
    selected_bias = selected_bias.detach()
    selected_temperature = selected_temperature.detach()

    output = {
        "schema": "lafgs_basin_family_prototypes",
        "version": 2,
        "landmark_indices": torch.as_tensor(pool["landmark_indices"]).long(),
        "prototype_features": selected_features,
        "prototype_anchor_indices": selected_parents,
        "prototype_bias": selected_bias,
        "prototype_temperature": selected_temperature,
        "families": selected_families,
        "calibration": {
            "candidate_count": len(prototypes),
            "selected_candidate_count": int(selected.numel()),
            "biases": biases.cpu(),
            "true_activation_counts": true_counts.cpu(),
            "false_activation_counts": false_counts.cpu(),
            "best_bias_indices": best_bias_index.cpu(),
            "best_utility": best_utility.cpu(),
        },
        "config": vars(args),
    }
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, path)
    summary = {
        "schema": output["schema"],
        "candidate_count": len(prototypes),
        "selected_candidate_count": int(selected.numel()),
        "base_prototype_count": (
            int(torch.as_tensor(base["prototype_features"]).shape[0])
            if args.base_family
            else 0
        ),
        "deployed_prototype_count": len(selected_features),
        "selected_true_activation_count": int(
            sum(value["legal_activation_count"] for value in selected_families if "legal_activation_count" in value)
        ),
        "selected_false_activation_count": int(
            sum(value["false_activation_count"] for value in selected_families if "false_activation_count" in value)
        ),
        "config": vars(args),
    }
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
