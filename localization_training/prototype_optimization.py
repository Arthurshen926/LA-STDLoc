"""Bounded prototype-only optimization for candidate-aware Basin hyperedges."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from localization_training.shared_metric import SharedLowRankMetric


@dataclass(frozen=True)
class PrototypeOptimizationConfig:
    steps: int = 300
    learning_rate: float = 3e-4
    maximum_residual: float = 0.05
    maximum_negative_bias: float = 0.12
    minimum_temperature: float = 0.85
    maximum_temperature: float = 1.15
    train_temperature: bool = False
    hyperedge_weight: float = 1.0
    sibling_weight: float = 2.0
    trust_weight: float = 0.2
    bias_trust_weight: float = 0.2
    margin: float = 0.05
    translation_reward_scale_cm: float = 15.0
    rotation_reward_scale_deg: float = 2.0
    seed: int = 2026


def inverse_sigmoid(value: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(value).float().clamp(1e-5, 1.0 - 1e-5)
    return torch.log(value) - torch.log1p(-value)


def materialize_prototypes(
    initial: torch.Tensor,
    residual: torch.Tensor,
    bias_raw: torch.Tensor,
    temperature_raw: torch.Tensor,
    *,
    maximum_residual: float,
    maximum_negative_bias: float,
    minimum_temperature: float,
    maximum_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    residual_norm = torch.linalg.norm(residual, dim=1, keepdim=True)
    ratio = residual_norm / max(float(maximum_residual), 1e-8)
    residual_scale = torch.where(
        ratio < 1e-4,
        torch.ones_like(ratio),
        torch.tanh(ratio) / ratio.clamp_min(1e-8),
    )
    bounded = residual * residual_scale
    features = F.normalize(initial + bounded, dim=1)
    bias = -float(maximum_negative_bias) * torch.sigmoid(bias_raw)
    temperature = float(minimum_temperature) + (
        float(maximum_temperature) - float(minimum_temperature)
    ) * torch.sigmoid(temperature_raw)
    temperature = temperature.clamp(
        min=float(minimum_temperature), max=float(maximum_temperature)
    )
    return features, bias, temperature, bounded


def teacher_set_scores(
    query: torch.Tensor,
    bank: torch.Tensor,
    family_features: torch.Tensor,
    family_bias: torch.Tensor,
    family_temperature: torch.Tensor,
    anchors: torch.Tensor,
    modes: torch.Tensor,
) -> torch.Tensor:
    flat_query = query.reshape(-1, query.shape[-1])
    flat_anchors = anchors.reshape(-1)
    flat_modes = modes.reshape(-1)
    scores = (flat_query * bank[flat_anchors]).sum(dim=1)
    secondary = flat_modes >= 0
    if bool(secondary.any().item()):
        mode_index = flat_modes[secondary]
        scores[secondary] = (
            (flat_query[secondary] * family_features[mode_index]).sum(dim=1)
            / family_temperature[mode_index]
            + family_bias[mode_index]
        )
    return scores.reshape(anchors.shape).sum(dim=1)


def hyperedge_loss(
    set_scores: torch.Tensor,
    set_types: torch.Tensor,
    levels: torch.Tensor,
    te_cm: torch.Tensor,
    re_deg: torch.Tensor,
    parents: torch.Tensor,
    *,
    margin: float,
    translation_scale_cm: float,
    rotation_scale_deg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    good = (set_types != 1) & (levels > 0)
    harmful = set_types == 1
    reward = torch.exp(
        -te_cm.clamp_min(0) / float(translation_scale_cm)
        - re_deg.clamp_min(0) / float(rotation_scale_deg)
    )
    quality = 0.25 + reward + (levels >= 2).float() + (levels >= 3).float()
    if bool(good.any().item()) and bool(harmful.any().item()):
        positive = torch.logsumexp(
            set_scores[good] + torch.log(quality[good].clamp_min(1e-8)),
            dim=0,
        )
        negative = torch.logsumexp(set_scores[harmful], dim=0)
        contrastive = F.softplus(negative - positive + float(margin))
    else:
        contrastive = set_scores.new_zeros(())
    repaired = torch.nonzero(
        (set_types == 2) & (parents >= 0), as_tuple=False
    ).reshape(-1)
    repaired = repaired[parents[repaired] < len(set_scores)]
    if repaired.numel():
        parent_indices = parents[repaired]
        valid = set_types[parent_indices] == 1
        repaired = repaired[valid]
        parent_indices = parent_indices[valid]
    sibling = (
        F.softplus(
            set_scores[parent_indices]
            - set_scores[repaired]
            + float(margin)
        ).mean()
        if repaired.numel()
        else set_scores.new_zeros(())
    )
    return contrastive, sibling


def initial_parameter_state(
    family: dict,
    train_indices: torch.Tensor,
    *,
    maximum_negative_bias: float,
    minimum_temperature: float,
    maximum_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    bias = torch.as_tensor(
        family.get(
            "prototype_bias",
            torch.zeros(len(family["prototype_features"])),
        )
    ).float()[train_indices]
    bias_fraction = (-bias / float(maximum_negative_bias)).clamp(
        1e-5, 1.0 - 1e-5
    )
    temperature = torch.as_tensor(
        family.get(
            "prototype_temperature",
            torch.ones(len(family["prototype_features"])),
        )
    ).float()[train_indices]
    temperature_fraction = (
        (temperature - float(minimum_temperature))
        / (float(maximum_temperature) - float(minimum_temperature))
    ).clamp(1e-5, 1.0 - 1e-5)
    return inverse_sigmoid(bias_fraction), inverse_sigmoid(
        temperature_fraction
    )


def optimize_basin_prototypes(
    *,
    state: dict,
    metric: SharedLowRankMetric,
    family: dict,
    teacher: dict,
    cache: dict,
    config: PrototypeOptimizationConfig,
    device: torch.device,
    checkpoint_steps: set[int] | None = None,
    checkpoint_callback=None,
    progress=None,
) -> tuple[dict, list[dict]]:
    """Optimize only appearance-pool prototypes; map and metric stay frozen."""
    if teacher.get("schema") != "lafgs_candidate_aware_basin_teacher":
        raise ValueError("prototype-only training requires Basin Teacher V3")
    if not torch.equal(
        torch.as_tensor(family["landmark_indices"]).long(),
        torch.arange(torch.as_tensor(state["anchor_xyz"]).shape[0]),
    ):
        raise ValueError("family state does not align with map")
    torch.manual_seed(int(config.seed))
    all_features = F.normalize(
        torch.as_tensor(family["prototype_features"]).float(), dim=1
    ).detach().to(device)
    all_bias = torch.as_tensor(
        family.get("prototype_bias", torch.zeros(len(all_features)))
    ).float().detach().to(device)
    all_temperature = torch.as_tensor(
        family.get("prototype_temperature", torch.ones(len(all_features)))
    ).float().detach().to(device)
    train_indices = torch.as_tensor(
        [
            index
            for index, metadata in enumerate(family["families"])
            if metadata.get("candidate_source") == "appearance_pool"
        ],
        dtype=torch.long,
    )
    if not train_indices.numel():
        raise ValueError("family state has no trainable appearance-pool modes")
    train_lookup = torch.full((len(all_features),), -1, dtype=torch.long)
    train_lookup[train_indices] = torch.arange(len(train_indices))
    initial = all_features[train_indices.to(device)].detach()
    residual = torch.nn.Parameter(torch.zeros_like(initial))
    bias_initial_raw, temperature_initial_raw = initial_parameter_state(
        family,
        train_indices,
        maximum_negative_bias=config.maximum_negative_bias,
        minimum_temperature=config.minimum_temperature,
        maximum_temperature=config.maximum_temperature,
    )
    bias_raw = torch.nn.Parameter(bias_initial_raw.to(device))
    temperature_raw = torch.nn.Parameter(temperature_initial_raw.to(device))
    parameters = [residual, bias_raw]
    if config.train_temperature:
        parameters.append(temperature_raw)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config.learning_rate),
        weight_decay=1e-4,
    )
    bank = F.normalize(
        torch.as_tensor(state["anchor_features"]).float(), dim=1
    ).to(device)
    metric = metric.to(device).eval()
    for parameter in metric.parameters():
        parameter.requires_grad_(False)
    records = [
        record
        for record in teacher["records"]
        if torch.as_tensor(record["set_types"]).numel()
        and bool(
            (
                train_lookup[
                    torch.as_tensor(record["set_mode_indices"])
                    .long()
                    .clamp_min(0)
                ]
                >= 0
            ).any()
        )
    ]
    if not records:
        raise ValueError("teacher does not use any trainable appearance mode")
    generator = torch.Generator().manual_seed(int(config.seed) + 1)
    checkpoint_steps = set(checkpoint_steps or {int(config.steps)})
    checkpoint_steps.add(int(config.steps))
    history = []
    output = None
    for step in range(1, int(config.steps) + 1):
        record = records[
            int(torch.randint(len(records), (1,), generator=generator))
        ]
        name = str(record["query_name"])
        set_rows = torch.as_tensor(record["set_query_rows"]).long()
        flat_rows = set_rows.reshape(-1)
        raw = F.normalize(
            torch.as_tensor(cache[name]["native_descriptors"]).float()[
                flat_rows
            ],
            dim=1,
        ).to(device)
        with torch.no_grad():
            query, _ = metric(raw)
        query = query.reshape(len(set_rows), 3, -1)
        trained, trained_bias, trained_temperature, bounded = (
            materialize_prototypes(
                initial,
                residual,
                bias_raw,
                temperature_raw,
                maximum_residual=config.maximum_residual,
                maximum_negative_bias=config.maximum_negative_bias,
                minimum_temperature=config.minimum_temperature,
                maximum_temperature=config.maximum_temperature,
            )
        )
        family_features = all_features.clone()
        family_bias = all_bias.clone()
        family_temperature = all_temperature.clone()
        family_features[train_indices.to(device)] = trained
        family_bias[train_indices.to(device)] = trained_bias
        family_temperature[train_indices.to(device)] = trained_temperature
        set_scores = teacher_set_scores(
            query,
            bank,
            family_features,
            family_bias,
            family_temperature,
            torch.as_tensor(record["set_anchor_indices"]).long().to(device),
            torch.as_tensor(record["set_mode_indices"]).long().to(device),
        )
        hyperedge, sibling = hyperedge_loss(
            set_scores,
            torch.as_tensor(record["set_types"]).long().to(device),
            torch.as_tensor(record["basin_level"]).long().to(device),
            torch.as_tensor(record["te_cm"]).float().to(device),
            torch.as_tensor(record["re_deg"]).float().to(device),
            torch.as_tensor(record["parent_set_index"]).long().to(device),
            margin=config.margin,
            translation_scale_cm=config.translation_reward_scale_cm,
            rotation_scale_deg=config.rotation_reward_scale_deg,
        )
        trust = bounded.square().sum(dim=1).mean()
        bias_trust = (
            trained_bias - all_bias[train_indices.to(device)]
        ).square().mean()
        loss = (
            float(config.hyperedge_weight) * hyperedge
            + float(config.sibling_weight) * sibling
            + float(config.trust_weight) * trust
            + float(config.bias_trust_weight) * bias_trust
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step == 1 or step % 25 == 0:
            event = {
                "step": step,
                "loss": float(loss.detach()),
                "hyperedge": float(hyperedge.detach()),
                "sibling": float(sibling.detach()),
                "residual_mean": float(
                    bounded.norm(dim=1).mean().detach()
                ),
                "residual_max": float(
                    bounded.norm(dim=1).max().detach()
                ),
                "bias_mean": float(trained_bias.mean().detach()),
                "temperature_mean": float(
                    trained_temperature.mean().detach()
                ),
            }
            history.append(event)
            if progress is not None:
                progress(dict(event))
        if step in checkpoint_steps:
            with torch.no_grad():
                (
                    trained,
                    trained_bias,
                    trained_temperature,
                    bounded,
                ) = materialize_prototypes(
                    initial,
                    residual,
                    bias_raw,
                    temperature_raw,
                    maximum_residual=config.maximum_residual,
                    maximum_negative_bias=config.maximum_negative_bias,
                    minimum_temperature=config.minimum_temperature,
                    maximum_temperature=config.maximum_temperature,
                )
                output_features = all_features.clone()
                output_bias = all_bias.clone()
                output_temperature = all_temperature.clone()
                output_features[train_indices.to(device)] = trained
                output_bias[train_indices.to(device)] = trained_bias
                output_temperature[train_indices.to(device)] = (
                    trained_temperature
                )
            output = {
                **family,
                "version": 3,
                "prototype_features": output_features.cpu(),
                "prototype_bias": output_bias.cpu(),
                "prototype_temperature": output_temperature.cpu(),
                "prototype_only_training": {
                    "schema": "lafgs_basin_prototype_only_training",
                    "version": 1,
                    "step": step,
                    "trainable_prototype_indices": train_indices,
                    "teacher_query_count": len(records),
                    "history": list(history),
                    "config": asdict(config),
                },
            }
            if checkpoint_callback is not None:
                checkpoint_callback(step, output)
    assert output is not None
    return output, history
