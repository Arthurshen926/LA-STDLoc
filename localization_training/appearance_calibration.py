"""Hard-assignment calibration for same-geometry appearance families."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from localization_training.shared_metric import SharedLowRankMetric


def validate_dynamic_baseline_binding(
    dynamic: dict,
    *,
    base_family_path: str,
    allow_unbound: bool = False,
) -> None:
    """Require calibration scores to come from the deployed base family."""
    from pathlib import Path

    expected = dynamic.get("family_prototype_state")
    if not expected:
        if allow_unbound:
            return
        raise ValueError(
            "dynamic outcomes do not bind a family baseline; pass an "
            "explicit legacy override only after verifying provenance"
        )
    if Path(expected).resolve() != Path(base_family_path).resolve():
        raise ValueError(
            "dynamic outcome family baseline does not match --base-family"
        )


def parse_nonpositive_biases(value: str | list[float]) -> torch.Tensor:
    items = value.split(",") if isinstance(value, str) else value
    biases = sorted(
        {min(float(item), 0.0) for item in items if str(item).strip()},
        reverse=True,
    )
    if not biases:
        raise ValueError("at least one non-positive bias is required")
    return torch.as_tensor(biases, dtype=torch.float32)


def legal_activation_matrix(
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


def calibrate_appearance_modes(
    *,
    pool: dict,
    positives: dict,
    dynamic: dict,
    cache: dict,
    metric: SharedLowRankMetric,
    biases: torch.Tensor,
    minimum_legal_activations: int,
    minimum_activation_precision: float,
    false_activation_cost: float,
    maximum_selected: int,
    device: torch.device,
    base_family: dict | None = None,
    config: dict | None = None,
    progress=None,
) -> dict:
    """Calibrate modes against the exact deployed hard top-1 assignment."""
    names = list(dynamic["query_names"])
    if names != list(positives["query_names"]):
        raise ValueError("dynamic and positive-teacher query registries differ")
    prototypes = F.normalize(
        torch.as_tensor(pool["prototype_features"]).float(), dim=1
    ).to(device)
    parents = torch.as_tensor(pool["prototype_anchor_indices"]).long()
    biases = torch.as_tensor(biases).float().to(device)
    if bool((biases > 0).any()):
        raise ValueError("appearance calibration biases must be non-positive")
    true_counts = torch.zeros(
        (len(biases), len(prototypes)), dtype=torch.long, device=device
    )
    false_counts = torch.zeros_like(true_counts)
    parent_columns = {
        int(parent): torch.nonzero(
            parents == int(parent), as_tuple=False
        ).reshape(-1)
        for parent in parents.unique().tolist()
    }
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
            legal = legal_activation_matrix(
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
            if progress is not None:
                progress(query_index + 1, len(names))
    precision = true_counts.float() / (
        true_counts + false_counts
    ).float().clamp_min(1)
    utility = (
        true_counts.float()
        - float(false_activation_cost) * false_counts.float()
    )
    valid = (
        (true_counts >= int(minimum_legal_activations))
        & (precision >= float(minimum_activation_precision))
    )
    utility = utility.masked_fill(~valid, -torch.inf)
    best_utility, best_bias_index = utility.max(dim=0)
    selected = torch.nonzero(
        torch.isfinite(best_utility), as_tuple=False
    ).reshape(-1)
    if selected.numel():
        selected = selected[
            torch.argsort(best_utility[selected], descending=True)
        ]
    if int(maximum_selected) > 0:
        selected = selected[: int(maximum_selected)]
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
                "candidate_source": "appearance_pool",
            }
        )
        selected_families.append(metadata)
    base_count = 0
    if base_family is not None:
        base_count = int(
            torch.as_tensor(base_family["prototype_features"]).shape[0]
        )
        selected_features = torch.cat(
            (
                torch.as_tensor(base_family["prototype_features"]).float(),
                selected_features,
            )
        )
        selected_parents = torch.cat(
            (
                torch.as_tensor(
                    base_family["prototype_anchor_indices"]
                ).long(),
                selected_parents,
            )
        )
        selected_bias = torch.cat(
            (
                torch.as_tensor(
                    base_family.get(
                        "prototype_bias", torch.zeros(base_count)
                    )
                ).float(),
                selected_bias,
            )
        )
        selected_temperature = torch.cat(
            (
                torch.as_tensor(
                    base_family.get(
                        "prototype_temperature", torch.ones(base_count)
                    )
                ).float(),
                selected_temperature,
            )
        )
        selected_families = [
            {**dict(value), "candidate_source": "base_family"}
            for value in base_family["families"]
        ] + selected_families
    output = {
        "schema": "lafgs_basin_family_prototypes",
        "version": 2,
        "landmark_indices": torch.as_tensor(pool["landmark_indices"]).long(),
        "prototype_features": selected_features.detach(),
        "prototype_anchor_indices": selected_parents,
        "prototype_bias": selected_bias.detach(),
        "prototype_temperature": selected_temperature.detach(),
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
        "config": dict(config or {}),
    }
    output["summary"] = {
        "candidate_count": len(prototypes),
        "selected_candidate_count": int(selected.numel()),
        "base_prototype_count": base_count,
        "deployed_prototype_count": len(selected_features),
        "selected_true_activation_count": int(
            sum(
                value.get("legal_activation_count", 0)
                for value in selected_families
            )
        ),
        "selected_false_activation_count": int(
            sum(
                value.get("false_activation_count", 0)
                for value in selected_families
            )
        ),
    }
    return output
