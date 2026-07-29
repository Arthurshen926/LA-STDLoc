#!/usr/bin/env python3
"""Train only validated secondary prototypes with Basin hyperedges and replay."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from localization_training.shared_metric import SharedLowRankMetric


def _inverse_sigmoid(value: torch.Tensor) -> torch.Tensor:
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


def _teacher_set_scores(
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
            (
                flat_query[secondary] * family_features[mode_index]
            ).sum(dim=1)
            / family_temperature[mode_index]
            + family_bias[mode_index]
        )
    return scores.reshape(anchors.shape).sum(dim=1)


def _hyperedge_loss(
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
            set_scores[good] + torch.log(quality[good].clamp_min(1e-8)), dim=0
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


def _initial_parameter_state(
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
    return _inverse_sigmoid(bias_fraction), _inverse_sigmoid(
        temperature_fraction
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--family-state", required=True)
    parser.add_argument("--basin-teacher", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--checkpoint-steps", default="50,100,200,300")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--maximum-residual", type=float, default=0.05)
    parser.add_argument("--maximum-negative-bias", type=float, default=0.12)
    parser.add_argument("--minimum-temperature", type=float, default=0.85)
    parser.add_argument("--maximum-temperature", type=float, default=1.15)
    parser.add_argument("--train-temperature", action="store_true")
    parser.add_argument("--hyperedge-weight", type=float, default=1.0)
    parser.add_argument("--sibling-weight", type=float, default=2.0)
    parser.add_argument("--trust-weight", type=float, default=0.2)
    parser.add_argument("--bias-trust-weight", type=float, default=0.2)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--translation-reward-scale-cm", type=float, default=15.0)
    parser.add_argument("--rotation-reward-scale-deg", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    metric_payload = torch.load(
        args.metric_state, map_location="cpu", weights_only=False
    )
    family = torch.load(
        args.family_state, map_location="cpu", weights_only=False
    )
    teacher = torch.load(
        args.basin_teacher, map_location="cpu", weights_only=False
    )
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    if teacher.get("schema") != "lafgs_candidate_aware_basin_teacher":
        raise ValueError("prototype-only training requires Basin Teacher V3")
    if not torch.equal(
        torch.as_tensor(family["landmark_indices"]).long(),
        torch.arange(torch.as_tensor(state["anchor_xyz"]).shape[0]),
    ):
        raise ValueError("family state does not align with map")

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
    bias_initial_raw, temperature_initial_raw = _initial_parameter_state(
        family,
        train_indices,
        maximum_negative_bias=args.maximum_negative_bias,
        minimum_temperature=args.minimum_temperature,
        maximum_temperature=args.maximum_temperature,
    )
    bias_raw = torch.nn.Parameter(bias_initial_raw.to(device))
    temperature_raw = torch.nn.Parameter(temperature_initial_raw.to(device))
    parameters = [residual, bias_raw]
    if args.train_temperature:
        parameters.append(temperature_raw)
    optimizer = torch.optim.AdamW(
        parameters, lr=float(args.learning_rate), weight_decay=1e-4
    )
    bank = F.normalize(
        torch.as_tensor(state["anchor_features"]).float(), dim=1
    ).to(device)
    metric = SharedLowRankMetric(**metric_payload["metric_config"]).to(device)
    metric.load_state_dict(metric_payload["metric_state_dict"])
    metric.eval()
    for parameter in metric.parameters():
        parameter.requires_grad_(False)

    records = [
        record
        for record in teacher["records"]
        if torch.as_tensor(record["set_types"]).numel()
        and bool(
            (
                train_lookup[
                    torch.as_tensor(record["set_mode_indices"]).long().clamp_min(0)
                ]
                >= 0
            ).any()
        )
    ]
    if not records:
        raise ValueError("teacher does not use any trainable appearance mode")
    generator = torch.Generator().manual_seed(int(args.seed) + 1)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        int(value)
        for value in args.checkpoint_steps.split(",")
        if value.strip()
    }
    history = []

    for step in range(1, int(args.steps) + 1):
        record = records[
            int(torch.randint(len(records), (1,), generator=generator))
        ]
        name = str(record["query_name"])
        set_rows = torch.as_tensor(record["set_query_rows"]).long()
        flat_rows = set_rows.reshape(-1)
        raw = F.normalize(
            torch.as_tensor(cache[name]["native_descriptors"]).float()[flat_rows],
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
                maximum_residual=args.maximum_residual,
                maximum_negative_bias=args.maximum_negative_bias,
                minimum_temperature=args.minimum_temperature,
                maximum_temperature=args.maximum_temperature,
            )
        )
        family_features = all_features.clone()
        family_bias = all_bias.clone()
        family_temperature = all_temperature.clone()
        family_features[train_indices.to(device)] = trained
        family_bias[train_indices.to(device)] = trained_bias
        family_temperature[train_indices.to(device)] = trained_temperature
        set_scores = _teacher_set_scores(
            query,
            bank,
            family_features,
            family_bias,
            family_temperature,
            torch.as_tensor(record["set_anchor_indices"]).long().to(device),
            torch.as_tensor(record["set_mode_indices"]).long().to(device),
        )
        hyperedge, sibling = _hyperedge_loss(
            set_scores,
            torch.as_tensor(record["set_types"]).long().to(device),
            torch.as_tensor(record["basin_level"]).long().to(device),
            torch.as_tensor(record["te_cm"]).float().to(device),
            torch.as_tensor(record["re_deg"]).float().to(device),
            torch.as_tensor(record["parent_set_index"]).long().to(device),
            margin=args.margin,
            translation_scale_cm=args.translation_reward_scale_cm,
            rotation_scale_deg=args.rotation_reward_scale_deg,
        )
        trust = bounded.square().sum(dim=1).mean()
        bias_trust = (trained_bias - all_bias[train_indices.to(device)]).square().mean()
        loss = (
            float(args.hyperedge_weight) * hyperedge
            + float(args.sibling_weight) * sibling
            + float(args.trust_weight) * trust
            + float(args.bias_trust_weight) * bias_trust
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
                "residual_mean": float(bounded.norm(dim=1).mean().detach()),
                "residual_max": float(bounded.norm(dim=1).max().detach()),
                "bias_mean": float(trained_bias.mean().detach()),
                "temperature_mean": float(trained_temperature.mean().detach()),
            }
            history.append(event)
            print(json.dumps(event), flush=True)
        if step in checkpoints or step == int(args.steps):
            with torch.no_grad():
                trained, trained_bias, trained_temperature, bounded = (
                    materialize_prototypes(
                        initial,
                        residual,
                        bias_raw,
                        temperature_raw,
                        maximum_residual=args.maximum_residual,
                        maximum_negative_bias=args.maximum_negative_bias,
                        minimum_temperature=args.minimum_temperature,
                        maximum_temperature=args.maximum_temperature,
                    )
                )
                output_features = all_features.clone()
                output_bias = all_bias.clone()
                output_temperature = all_temperature.clone()
                output_features[train_indices.to(device)] = trained
                output_bias[train_indices.to(device)] = trained_bias
                output_temperature[train_indices.to(device)] = trained_temperature
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
                    "history": history,
                    "config": vars(args),
                },
            }
            torch.save(
                output,
                output_dir / f"family_prototypes_step_{step:04d}.pt",
            )
    (output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "schema": "lafgs_basin_prototype_only_training",
                "trainable_prototype_count": int(train_indices.numel()),
                "teacher_query_count": len(records),
                "history": history,
                "config": vars(args),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
