#!/usr/bin/env python3
"""Locally refit only anchors that replace correspondences removed by pruning."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F

from localization_training.micro_anchors import (
    protected_micro_anchor_descriptor_loss,
)
from scripts.run_lafgs_alternating_structure import _deployment_valid_mask


def _project_error(xyz, indices, K, pose_w2c, keypoints):
    points = xyz[indices].float()
    pose = pose_w2c.float()
    camera = points @ pose[:3, :3].T + pose[:3, 3]
    projected = camera @ K.float().T
    uv = projected[..., :2] / projected[..., 2:].clamp_min(1e-8)
    error = torch.linalg.norm(uv - keypoints[:, None], dim=-1)
    return torch.where(
        camera[..., 2] > 1e-6,
        error,
        torch.full_like(error, 1e6),
    )


def _first_frozen_score(scores, indices, train_mask):
    frozen = ~train_mask[indices]
    has_frozen = frozen.any(dim=1)
    first = frozen.float().argmax(dim=1)
    value = scores.gather(1, first[:, None]).squeeze(1)
    return value, has_frozen


@torch.no_grad()
def _build_data(
    original,
    pruned,
    query_cache,
    deployment_masks,
    *,
    topk,
    clean_radius_px,
    max_positive_per_query,
    guard_rows_per_query,
):
    device = torch.device("cuda")
    original_features = F.normalize(
        torch.as_tensor(original["anchor_features"]).float(), dim=1
    ).to(device)
    pruned_features = F.normalize(
        torch.as_tensor(pruned["anchor_features"]).float(), dim=1
    ).to(device)
    original_xyz = torch.as_tensor(original["anchor_xyz"]).float()
    pruned_xyz = torch.as_tensor(pruned["anchor_xyz"]).float()
    selected_source_rows = torch.as_tensor(
        pruned["functional_pruning"]["selected_source_rows"]
    ).long()
    retained = torch.zeros(
        original_features.shape[0], dtype=torch.bool
    )
    retained[selected_source_rows] = True
    positive_records, guard_records = [], []
    names = list(query_cache)
    for query_index, name in enumerate(names):
        cached = query_cache[name]
        valid = _deployment_valid_mask(cached, name, deployment_masks)
        query_rows = torch.nonzero(valid, as_tuple=False).reshape(-1)
        descriptors = F.normalize(
            torch.as_tensor(cached["native_descriptors"])[query_rows].float(),
            dim=1,
        )
        descriptors_device = descriptors.to(device)
        original_score, original_index = (
            descriptors_device @ original_features.T
        ).max(dim=1)
        pruned_score, pruned_index = torch.topk(
            descriptors_device @ pruned_features.T,
            k=min(int(topk), pruned_features.shape[0]),
            dim=1,
        )
        original_score = original_score.cpu()
        original_index = original_index.cpu()
        pruned_score = pruned_score.cpu()
        pruned_index = pruned_index.cpu()
        keypoints = (
            torch.as_tensor(cached["native_keypoints"])[query_rows].float()
            + float(cached.get("pixel_center_offset", 0.5))
        )
        original_error = _project_error(
            original_xyz,
            original_index[:, None],
            torch.as_tensor(cached["native_K"]),
            torch.as_tensor(cached["pose_w2c"]),
            keypoints,
        )[:, 0]
        pruned_error = _project_error(
            pruned_xyz,
            pruned_index,
            torch.as_tensor(cached["native_K"]),
            torch.as_tensor(cached["pose_w2c"]),
            keypoints,
        )
        clean = pruned_error <= float(clean_radius_px)
        deleted_winner = ~retained[original_index]
        recoverable = deleted_winner & clean.any(dim=1)
        recoverable_rows = torch.nonzero(
            recoverable, as_tuple=False
        ).reshape(-1)
        if recoverable_rows.numel() > max_positive_per_query:
            native_scores = torch.as_tensor(cached["native_scores"])[
                query_rows[recoverable_rows]
            ].float()
            keep = torch.topk(
                native_scores, k=max_positive_per_query
            ).indices
            recoverable_rows = recoverable_rows[keep]
        if recoverable_rows.numel():
            clean_position = clean[recoverable_rows].float().argmax(dim=1)
            target = pruned_index[recoverable_rows].gather(
                1, clean_position[:, None]
            ).squeeze(1)
            positive_records.append(
                {
                    "descriptors": descriptors[recoverable_rows],
                    "targets": target,
                    "topk_indices": pruned_index[recoverable_rows],
                    "topk_scores": pruned_score[recoverable_rows],
                }
            )
        guard = retained[original_index] & (
            original_error <= float(clean_radius_px)
        )
        guard_rows = torch.nonzero(guard, as_tuple=False).reshape(-1)
        if guard_rows.numel() > guard_rows_per_query:
            native_scores = torch.as_tensor(cached["native_scores"])[
                query_rows[guard_rows]
            ].float()
            keep = torch.topk(
                native_scores, k=guard_rows_per_query
            ).indices
            guard_rows = guard_rows[keep]
        if guard_rows.numel():
            guard_records.append(
                {
                    "descriptors": descriptors[guard_rows],
                    "topk_indices": pruned_index[guard_rows],
                    "topk_scores": pruned_score[guard_rows],
                }
            )
        if (query_index + 1) % 25 == 0 or query_index + 1 == len(names):
            print(f"Refit mining {query_index + 1}/{len(names)}", flush=True)

    if not positive_records:
        raise RuntimeError("No recoverable deleted-winner correspondences")
    train_rows = torch.unique(
        torch.cat([record["targets"] for record in positive_records])
    )
    row_to_local = torch.full(
        (pruned_features.shape[0],), -1, dtype=torch.long
    )
    row_to_local[train_rows] = torch.arange(train_rows.numel())
    train_mask = row_to_local >= 0
    positives, targets, positive_old_best = [], [], []
    for record in positive_records:
        old_best, valid_old = _first_frozen_score(
            record["topk_scores"],
            record["topk_indices"],
            train_mask,
        )
        positives.append(record["descriptors"][valid_old])
        targets.append(row_to_local[record["targets"][valid_old]])
        positive_old_best.append(old_best[valid_old])
    guards, guard_old_best = [], []
    for record in guard_records:
        old_best, valid_old = _first_frozen_score(
            record["topk_scores"],
            record["topk_indices"],
            train_mask,
        )
        guards.append(record["descriptors"][valid_old])
        guard_old_best.append(old_best[valid_old])
    return {
        "train_rows": train_rows,
        "positive_descriptors": torch.cat(positives),
        "positive_targets": torch.cat(targets),
        "positive_old_best": torch.cat(positive_old_best),
        "guard_descriptors": torch.cat(guards),
        "guard_old_best": torch.cat(guard_old_best),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-map", required=True)
    parser.add_argument("--pruned-map", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--deployment-mask-cache", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--checkpoint-steps", default="100,300")
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--clean-radius-px", type=float, default=4.0)
    parser.add_argument("--max-positive-per-query", type=int, default=256)
    parser.add_argument("--guard-rows-per-query", type=int, default=64)
    parser.add_argument("--positive-batch-size", type=int, default=256)
    parser.add_argument("--guard-batch-size", type=int, default=256)
    parser.add_argument("--positive-margin", type=float, default=0.01)
    parser.add_argument("--guard-margin", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--guard-weight", type=float, default=4.0)
    parser.add_argument("--trust-weight", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-residual-norm", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Local replacement refit requires CUDA")
    torch.manual_seed(args.seed)
    original = torch.load(
        args.original_map, map_location="cpu", weights_only=False
    )
    pruned = torch.load(
        args.pruned_map, map_location="cpu", weights_only=False
    )
    payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    query_cache = payload.get("queries", payload)
    deployment_masks = None
    if args.deployment_mask_cache:
        with Path(args.deployment_mask_cache).open("rb") as handle:
            deployment_masks = pickle.load(handle)
    data = _build_data(
        original,
        pruned,
        query_cache,
        deployment_masks,
        topk=args.topk,
        clean_radius_px=args.clean_radius_px,
        max_positive_per_query=args.max_positive_per_query,
        guard_rows_per_query=args.guard_rows_per_query,
    )
    del payload, query_cache, original
    device = torch.device("cuda")
    all_features = F.normalize(
        torch.as_tensor(pruned["anchor_features"]).float(), dim=1
    )
    train_rows = data["train_rows"]
    initial = all_features[train_rows].to(device)
    residual = torch.nn.Parameter(torch.zeros_like(initial))
    optimizer = torch.optim.AdamW(
        [residual], lr=args.learning_rate, weight_decay=0.0
    )
    target_frequency = torch.bincount(
        data["positive_targets"], minlength=train_rows.numel()
    ).float()
    sampling_weight = torch.reciprocal(
        target_frequency[data["positive_targets"]].clamp_min(1)
    )
    sampling_weight /= sampling_weight.sum()
    checkpoints = {
        int(value)
        for value in args.checkpoint_steps.split(",")
        if value.strip()
    }
    checkpoints.add(args.steps)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    history = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        positive_index = torch.multinomial(
            sampling_weight,
            num_samples=args.positive_batch_size,
            replacement=True,
            generator=generator,
        )
        guard_index = torch.randint(
            data["guard_descriptors"].shape[0],
            (args.guard_batch_size,),
            generator=generator,
        )
        candidate = F.normalize(initial + residual, dim=1)
        loss, diagnostics = protected_micro_anchor_descriptor_loss(
            candidate_features=candidate,
            positive_descriptors=data["positive_descriptors"][
                positive_index
            ].to(device),
            positive_targets=data["positive_targets"][positive_index].to(
                device
            ),
            positive_old_best=data["positive_old_best"][
                positive_index
            ].to(device),
            guard_descriptors=data["guard_descriptors"][guard_index].to(
                device
            ),
            guard_old_best=data["guard_old_best"][guard_index].to(device),
            initial_features=initial,
            positive_margin=args.positive_margin,
            guard_margin=args.guard_margin,
            temperature=args.temperature,
            guard_weight=args.guard_weight,
            trust_weight=args.trust_weight,
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            norm = torch.linalg.norm(residual, dim=1, keepdim=True)
            residual.mul_(
                (
                    args.max_residual_norm / norm.clamp_min(1e-8)
                ).clamp(max=1.0)
            )
        if step == 1 or step % 50 == 0 or step in checkpoints:
            record = {
                "step": step,
                **{
                    key: float(value.item())
                    for key, value in diagnostics.items()
                },
                "residual_norm_mean": float(
                    torch.linalg.norm(residual, dim=1).mean()
                ),
                "residual_norm_max": float(
                    torch.linalg.norm(residual, dim=1).max()
                ),
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
        if step in checkpoints:
            output = dict(pruned)
            output_features = all_features.clone()
            output_features[train_rows] = F.normalize(
                initial + residual, dim=1
            ).detach().cpu()
            output["anchor_features"] = output_features
            output["local_replacement_refit"] = {
                "step": step,
                "train_row_count": int(train_rows.numel()),
                "positive_count": int(
                    data["positive_descriptors"].shape[0]
                ),
                "guard_count": int(data["guard_descriptors"].shape[0]),
                "train_rows": train_rows,
                "config": vars(args),
                "history": history,
            }
            torch.save(
                output, output_dir / f"anchor_map_step_{step:04d}.pt"
            )
    (output_dir / "training_manifest.json").write_text(
        json.dumps(
            {
                "schema": "lafgs_local_replacement_refit",
                "version": 1,
                "train_row_count": int(train_rows.numel()),
                "positive_count": int(
                    data["positive_descriptors"].shape[0]
                ),
                "guard_count": int(data["guard_descriptors"].shape[0]),
                "history": history,
                "config": vars(args),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
