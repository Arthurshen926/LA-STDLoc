#!/usr/bin/env python
"""Build a guarded second view prototype for each active landmark."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localization_training.full_primitive_retrieval import (
    chunked_exact_topk,
    chunked_exact_topk_dual_prototype,
)
from train_lafgs_map import _cached_native_observations


def collect_clean_observations(cache, visibility, bank_xyz, maximum):
    args = SimpleNamespace(
        grid_rows=8,
        grid_cols=8,
        native_association_radius_px=2.0,
        native_unmatched_fraction=0.5,
        native_sampling_mode="detector_grid",
    )
    feature_parts = []
    positive_count_parts = []
    positive_index_parts = []
    for name in tqdm(
        sorted(set(cache) & set(visibility)), desc="Prototype observations"
    ):
        observations = _cached_native_observations(
            cache[name],
            bank_xyz,
            args,
            max_observations=maximum,
            bank_visibility_mask=visibility[name].cuda(),
            prediction_bank_xyz=bank_xyz,
        )
        offsets = observations.positive_offsets
        counts = offsets[1:] - offsets[:-1]
        clean = counts > 0
        if bool(clean.any().item()):
            feature_parts.append(
                F.normalize(observations.query_features[clean], dim=1)
                .half()
                .cpu()
            )
            positive_count_parts.append(counts[clean].cpu())
            positive_index_parts.append(
                observations.positive_indices.cpu()
            )
    if not feature_parts:
        raise RuntimeError("no GT-clean native observations were found")
    counts = torch.cat(positive_count_parts)
    offsets = torch.cat(
        [counts.new_zeros(1), counts.cumsum(dim=0)]
    )
    return (
        torch.cat(feature_parts).float(),
        offsets,
        torch.cat(positive_index_parts),
    )


def csr_candidate_membership(
    candidate_indices,
    row_ids,
    positive_offsets,
    positive_indices,
    landmark_count,
):
    candidate_indices = torch.as_tensor(candidate_indices)
    positive_offsets = torch.as_tensor(
        positive_offsets,
        device=candidate_indices.device,
        dtype=torch.long,
    ).reshape(-1)
    positive_indices = torch.as_tensor(
        positive_indices,
        device=candidate_indices.device,
        dtype=torch.long,
    ).reshape(-1)
    row_ids = torch.as_tensor(
        row_ids, device=candidate_indices.device, dtype=torch.long
    ).reshape(-1)
    if candidate_indices.ndim == 1:
        candidate_indices = candidate_indices[:, None]
        squeeze = True
    else:
        squeeze = False
    counts = positive_offsets[1:] - positive_offsets[:-1]
    positive_rows = torch.repeat_interleave(
        torch.arange(
            counts.numel(), device=candidate_indices.device, dtype=torch.long
        ),
        counts,
    )
    positive_keys = (
        positive_rows * int(landmark_count) + positive_indices
    )
    candidate_keys = (
        row_ids[:, None] * int(landmark_count) + candidate_indices
    )
    result = torch.isin(candidate_keys, positive_keys)
    return result[:, 0] if squeeze else result


def sampled_row_ids(row_count, maximum_rows, device):
    rows = torch.arange(row_count, device=device, dtype=torch.long)
    if int(maximum_rows) > 0 and row_count > int(maximum_rows):
        generator = torch.Generator(device=device).manual_seed(2026)
        rows = torch.randperm(
            row_count, generator=generator, device=device
        )[: int(maximum_rows)]
    return rows


def build_secondary_prototypes(
    primary,
    sources,
    observations,
    *,
    minimum_observations,
    minimum_secondary_observations,
    maximum_primary_secondary_cosine,
):
    primary = F.normalize(primary.float(), dim=1)
    sources = sources.long()
    observations = F.normalize(observations.float(), dim=1)
    landmark_count = int(primary.shape[0])
    count = torch.bincount(sources, minlength=landmark_count)
    primary_similarity = (observations * primary[sources]).sum(dim=1)

    seed_score = primary.new_full((landmark_count,), torch.inf)
    seed_score.scatter_reduce_(
        0, sources, primary_similarity, reduce="amin", include_self=True
    )
    row_ids = torch.arange(sources.numel(), device=sources.device)
    seed_candidates = torch.where(
        primary_similarity <= seed_score[sources] + 1e-7,
        row_ids,
        row_ids.new_full(row_ids.shape, sources.numel()),
    )
    seed_row = torch.full(
        (landmark_count,),
        sources.numel(),
        dtype=torch.long,
        device=sources.device,
    )
    seed_row.scatter_reduce_(
        0, sources, seed_candidates, reduce="amin", include_self=True
    )
    seeded = seed_row < sources.numel()
    secondary_seed = primary.clone()
    secondary_seed[seeded] = observations[seed_row[seeded]]

    secondary_similarity = (
        observations * secondary_seed[sources]
    ).sum(dim=1)
    secondary_assignment = secondary_similarity > primary_similarity
    secondary_count = torch.bincount(
        sources[secondary_assignment], minlength=landmark_count
    )
    secondary_sum = torch.zeros_like(primary)
    secondary_sum.index_add_(
        0, sources[secondary_assignment], observations[secondary_assignment]
    )
    secondary = primary.clone()
    has_secondary = secondary_count > 0
    secondary[has_secondary] = F.normalize(
        secondary_sum[has_secondary], dim=1
    )
    prototype_cosine = (primary * secondary).sum(dim=1)
    mask = (
        (count >= int(minimum_observations))
        & (secondary_count >= int(minimum_secondary_observations))
        & (prototype_cosine <= float(maximum_primary_secondary_cosine))
    )
    return secondary, mask, {
        "observation_count": count,
        "secondary_observation_count": secondary_count,
        "primary_secondary_cosine": prototype_cosine,
    }


def retrieval_diagnostics(
    primary,
    secondary,
    mask,
    observations,
    positive_offsets,
    positive_indices,
    maximum_rows=20000,
):
    rows = sampled_row_ids(
        observations.shape[0], maximum_rows, observations.device
    )
    sampled_observations = observations[rows]
    baseline = chunked_exact_topk(
        sampled_observations, primary, topk=4, chunk_size=8192
    )
    dual = chunked_exact_topk_dual_prototype(
        sampled_observations,
        primary,
        secondary,
        mask,
        topk=4,
        chunk_size=8192,
    )
    baseline_positive = csr_candidate_membership(
        baseline.indices,
        rows,
        positive_offsets,
        positive_indices,
        primary.shape[0],
    )
    dual_positive = csr_candidate_membership(
        dual.indices,
        rows,
        positive_offsets,
        positive_indices,
        primary.shape[0],
    )
    return {
        "rows": int(rows.numel()),
        "baseline_recall_at_1": float(
            baseline_positive[:, 0].float().mean().item()
        ),
        "baseline_recall_at_4": float(
            baseline_positive.any(dim=1).float().mean().item()
        ),
        "dual_recall_at_1": float(
            dual_positive[:, 0].float().mean().item()
        ),
        "dual_recall_at_4": float(
            dual_positive.any(dim=1).float().mean().item()
        ),
        "top1_changed_rate": float(
            (dual.indices[:, 0] != baseline.indices[:, 0])
            .float()
            .mean()
            .item()
        ),
    }


def calibrate_secondary_activation(
    primary,
    secondary,
    initial_mask,
    observations,
    positive_offsets,
    positive_indices,
    *,
    maximum_rows,
    minimum_benefits,
    minimum_precision,
):
    rows = sampled_row_ids(
        observations.shape[0], maximum_rows, observations.device
    )
    sampled_observations = observations[rows]
    baseline = chunked_exact_topk(
        sampled_observations, primary, topk=1, chunk_size=8192
    )
    dual = chunked_exact_topk_dual_prototype(
        sampled_observations,
        primary,
        secondary,
        initial_mask,
        topk=1,
        chunk_size=8192,
    )
    baseline_top1 = baseline.indices[:, 0]
    dual_top1 = dual.indices[:, 0]
    baseline_positive = csr_candidate_membership(
        baseline_top1,
        rows,
        positive_offsets,
        positive_indices,
        primary.shape[0],
    )
    dual_positive = csr_candidate_membership(
        dual_top1,
        rows,
        positive_offsets,
        positive_indices,
        primary.shape[0],
    )
    changed = dual_top1 != baseline_top1
    beneficial = changed & dual_positive & ~baseline_positive
    harmful = changed & ~dual_positive
    landmark_count = int(primary.shape[0])
    benefit_count = torch.bincount(
        dual_top1[beneficial], minlength=landmark_count
    )
    harmful_count = torch.bincount(
        dual_top1[harmful], minlength=landmark_count
    )
    precision = benefit_count.float() / (
        benefit_count + harmful_count
    ).clamp_min(1)
    protected_mask = (
        initial_mask
        & (benefit_count >= int(minimum_benefits))
        & (precision >= float(minimum_precision))
    )
    return protected_mask, {
        "prototype_benefit_count": benefit_count,
        "prototype_harmful_count": harmful_count,
        "prototype_activation_precision": precision,
        "calibration_changed_rows": changed.sum(),
        "calibration_beneficial_rows": beneficial.sum(),
        "calibration_harmful_rows": harmful.sum(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_cache", required=True)
    parser.add_argument("--visibility_cache", required=True)
    parser.add_argument("--map_state", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_observations", type=int, default=2048)
    parser.add_argument("--minimum_observations", type=int, default=8)
    parser.add_argument("--minimum_secondary_observations", type=int, default=4)
    parser.add_argument("--diagnostic_rows", type=int, default=20000)
    parser.add_argument("--calibrate_activation", action="store_true")
    parser.add_argument("--activation_minimum_benefits", type=int, default=2)
    parser.add_argument("--activation_minimum_precision", type=float, default=0.5)
    parser.add_argument(
        "--maximum_primary_secondary_cosine", type=float, default=0.98
    )
    args = parser.parse_args()

    state = torch.load(args.map_state, map_location="cpu")
    bank_xyz = torch.as_tensor(state["landmark_xyz"]).float().cuda()
    primary = F.normalize(
        torch.as_tensor(state["landmark_features"]).float().cuda(), dim=1
    )
    cache = torch.load(args.query_cache, map_location="cpu")["queries"]
    visibility = torch.load(args.visibility_cache, map_location="cpu")[
        "visibility"
    ]
    observations, positive_offsets, positive_indices = collect_clean_observations(
        cache, visibility, bank_xyz, args.max_observations
    )
    observations = observations.cuda()
    positive_offsets = positive_offsets.cuda()
    positive_indices = positive_indices.cuda()
    positive_counts = positive_offsets[1:] - positive_offsets[:-1]
    positive_rows = torch.repeat_interleave(
        torch.arange(
            positive_counts.numel(),
            device=observations.device,
            dtype=torch.long,
        ),
        positive_counts,
    )
    sources = positive_indices
    prototype_observations = observations[positive_rows]
    secondary, mask, statistics = build_secondary_prototypes(
        primary,
        sources,
        prototype_observations,
        minimum_observations=args.minimum_observations,
        minimum_secondary_observations=args.minimum_secondary_observations,
        maximum_primary_secondary_cosine=args.maximum_primary_secondary_cosine,
    )
    activation_statistics = {}
    initial_mask = mask.clone()
    if args.calibrate_activation:
        mask, activation_statistics = calibrate_secondary_activation(
            primary,
            secondary,
            initial_mask,
            observations,
            positive_offsets,
            positive_indices,
            maximum_rows=args.diagnostic_rows,
            minimum_benefits=args.activation_minimum_benefits,
            minimum_precision=args.activation_minimum_precision,
        )
    diagnostics = retrieval_diagnostics(
        primary,
        secondary,
        mask,
        observations,
        positive_offsets,
        positive_indices,
        maximum_rows=args.diagnostic_rows,
    )
    diagnostics.update(
        {
            "active_secondary_count": int(mask.sum().item()),
            "active_secondary_fraction": float(mask.float().mean().item()),
            "initial_secondary_count": int(initial_mask.sum().item()),
            "active_primary_secondary_cosine_mean": float(
                statistics["primary_secondary_cosine"][mask].mean().item()
                if bool(mask.any().item())
                else 1.0
            ),
        }
    )
    artifact = {
        "version": 1,
        "landmark_indices": torch.as_tensor(state["landmark_indices"]).cpu(),
        "secondary_features": secondary.detach().cpu(),
        "secondary_mask": mask.detach().cpu(),
        "statistics": {
            key: value.detach().cpu()
            for key, value in {**statistics, **activation_statistics}.items()
        },
        "config": vars(args),
        "diagnostics": diagnostics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    output.with_suffix(".json").write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
