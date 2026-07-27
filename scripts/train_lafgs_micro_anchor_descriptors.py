#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F

from localization_training.micro_anchors import (
    protected_micro_anchor_descriptor_loss,
)
from localization_training.ulf_initializer import sample_mask_at_grid_uv


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_selected(xyz, K, pose_w2c):
    camera = xyz @ pose_w2c[:3, :3].T + pose_w2c[:3, 3]
    homogeneous = camera @ K.T
    uv = homogeneous[:, :2] / camera[:, 2:].clamp_min(1e-8)
    return uv, camera[:, 2]


@torch.no_grad()
def _old_bank_best(descriptors, old_features, *, chunk_size):
    values = []
    indices = []
    for start in range(0, descriptors.shape[0], int(chunk_size)):
        score = descriptors[start : start + int(chunk_size)] @ old_features.T
        value, index = score.max(dim=1)
        values.append(value.cpu())
        indices.append(index.cpu())
    return torch.cat(values), torch.cat(indices)


def _load_payload(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "lafgs_track_first_payload":
        raise ValueError(f"Unsupported track payload: {path}")
    return payload


def _load_anchor_map(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema") != "lafgs_materialized_anchor_map":
        raise ValueError(f"Unsupported anchor map: {path}")
    return state


def _compact_training_data(
    *,
    state,
    payload,
    query_cache,
    device,
    guard_rows_per_query,
    clean_radius_px,
    score_chunk_size,
    train_start_row,
    deployment_masks,
):
    base_count = int(train_start_row)
    features = F.normalize(
        torch.as_tensor(state["anchor_features"]).float(), dim=1
    )
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    old_features = features[:base_count].to(device)
    old_xyz = xyz[:base_count].to(device)
    track_ids = torch.as_tensor(state["track_cluster_ids"]).long()[
        base_count:
    ]
    track_to_new = {
        int(track): row for row, track in enumerate(track_ids.tolist())
    }
    tracks = payload["tracks"]
    query_names = payload["query_names"]
    observations_by_query = {}
    for observation, track in enumerate(tracks["track_index"].tolist()):
        if int(track) not in track_to_new:
            continue
        query = int(tracks["query_index"][observation])
        observations_by_query.setdefault(query, []).append(observation)

    positive_descriptors = []
    positive_targets = []
    guard_descriptors = []
    guard_old_scores = []
    positive_counts = torch.zeros(len(track_to_new), dtype=torch.long)
    for query_index, name in enumerate(query_names):
        cached = query_cache[name]
        descriptors = F.normalize(
            torch.as_tensor(cached["native_descriptors"]).float(), dim=1
        )
        native_keypoints = torch.as_tensor(
            cached["native_keypoints"]
        ).float()
        valid_rows = torch.ones(descriptors.shape[0], dtype=torch.bool)
        if cached.get("native_valid_mask") is not None:
            valid_rows &= sample_mask_at_grid_uv(
                torch.as_tensor(cached["native_valid_mask"]),
                native_keypoints,
            ).cpu()
        if deployment_masks is not None and name in deployment_masks:
            channels = deployment_masks[name]
            if len(channels) < 3:
                raise ValueError(
                    f"deployment mask for {name!r} needs three channels"
                )
            target_hw = tuple(
                int(value)
                for value in cached.get("native_input_hw", ())
            )
            if len(target_hw) != 2:
                raise ValueError(
                    "native_input_hw is required for deployment masks"
                )
            resized = []
            for channel in channels[:3]:
                mask = torch.as_tensor(channel).detach().cpu().float()
                while mask.ndim > 2:
                    mask = mask.squeeze(0)
                resized.append(
                    F.interpolate(
                        mask[None, None],
                        size=target_hw,
                        mode="nearest",
                    )[0, 0].bool()
                )
            deployment_valid = resized[0] & resized[1] & resized[2]
            valid_rows &= sample_mask_at_grid_uv(
                deployment_valid, native_keypoints
            ).cpu()
        if query_index in observations_by_query:
            observations = observations_by_query[query_index]
            keypoint_index = tracks["keypoint_index"][observations].long()
            observation_valid = valid_rows[keypoint_index]
            observations = torch.as_tensor(
                observations, dtype=torch.long
            )[observation_valid].tolist()
            keypoint_index = keypoint_index[observation_valid]
            if not observations:
                continue
            targets = torch.as_tensor(
                [
                    track_to_new[int(tracks["track_index"][observation])]
                    for observation in observations
                ],
                dtype=torch.long,
            )
            positive_descriptors.append(descriptors[keypoint_index])
            positive_targets.append(targets)
            positive_counts.index_add_(
                0, targets, torch.ones_like(targets, dtype=torch.long)
            )

        valid_indices = torch.nonzero(
            valid_rows, as_tuple=False
        ).reshape(-1)
        keep = min(int(guard_rows_per_query), valid_indices.numel())
        if keep <= 0:
            continue
        scores = torch.as_tensor(cached["native_scores"]).float()
        local_guard = torch.topk(
            scores[valid_indices], k=keep, sorted=False
        ).indices
        guard_index = valid_indices[local_guard]
        guard = descriptors[guard_index].to(device)
        old_score, old_index = _old_bank_best(
            guard, old_features, chunk_size=score_chunk_size
        )
        selected_xyz = old_xyz[old_index.to(device)]
        projected, depth = _project_selected(
            selected_xyz,
            torch.as_tensor(cached["native_K"]).float().to(device),
            torch.as_tensor(cached["pose_w2c"]).float().to(device),
        )
        physical_keypoints = (
            torch.as_tensor(cached["native_keypoints"]).float()[guard_index]
            + float(cached.get("pixel_center_offset", 0.5))
        ).to(device)
        clean = (depth > 0) & (
            torch.linalg.norm(projected - physical_keypoints, dim=1)
            <= float(clean_radius_px)
        )
        if bool(clean.any()):
            guard_descriptors.append(guard[clean].cpu())
            guard_old_scores.append(old_score[clean.cpu()])

    if not positive_descriptors:
        raise RuntimeError("No selected micro-anchor track observations were found")
    if bool((positive_counts == 0).any()):
        raise RuntimeError("At least one micro-anchor has no track observations")
    positives = torch.cat(positive_descriptors)
    targets = torch.cat(positive_targets)
    positive_old_best, _ = _old_bank_best(
        positives.to(device), old_features, chunk_size=score_chunk_size
    )
    guards = (
        torch.cat(guard_descriptors)
        if guard_descriptors
        else positives.new_zeros((0, positives.shape[1]))
    )
    guard_scores = (
        torch.cat(guard_old_scores)
        if guard_old_scores
        else positives.new_zeros((0,))
    )
    return {
        "positive_descriptors": positives,
        "positive_targets": targets,
        "positive_old_best": positive_old_best,
        "positive_counts": positive_counts,
        "guard_descriptors": guards,
        "guard_old_best": guard_scores,
    }


def _full_diagnostics(candidate_features, initial_features, data, args):
    with torch.no_grad():
        _, diagnostics = protected_micro_anchor_descriptor_loss(
            candidate_features=candidate_features,
            positive_descriptors=data["positive_descriptors"].to(
                candidate_features.device
            ),
            positive_targets=data["positive_targets"].to(
                candidate_features.device
            ),
            positive_old_best=data["positive_old_best"].to(
                candidate_features.device
            ),
            guard_descriptors=data["guard_descriptors"].to(
                candidate_features.device
            ),
            guard_old_best=data["guard_old_best"].to(candidate_features.device),
            initial_features=initial_features,
            positive_margin=args.positive_margin,
            guard_margin=args.guard_margin,
            temperature=args.temperature,
            guard_weight=args.guard_weight,
            trust_weight=args.trust_weight,
        )
    return {name: float(value.item()) for name, value in diagnostics.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-map", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument(
        "--deployment-mask-cache",
        default="",
        help="masks.pkl used by deployment keypoint filtering.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--checkpoint-steps", default="500,1000")
    parser.add_argument("--positive-batch-size", type=int, default=256)
    parser.add_argument("--guard-batch-size", type=int, default=256)
    parser.add_argument("--guard-rows-per-query", type=int, default=96)
    parser.add_argument("--clean-radius-px", type=float, default=2.0)
    parser.add_argument("--positive-margin", type=float, default=0.03)
    parser.add_argument("--guard-margin", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--guard-weight", type=float, default=2.0)
    parser.add_argument("--trust-weight", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--max-residual-norm", type=float, default=0.2)
    parser.add_argument("--score-chunk-size", type=int, default=1024)
    parser.add_argument(
        "--train-start-row",
        type=int,
        default=-1,
        help=(
            "First trainable anchor row. Defaults to base_anchor_count for "
            "backward compatibility; alternating refresh should pass the "
            "pre-update active-map size."
        ),
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Micro-anchor descriptor training requires CUDA")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    anchor_path = Path(args.anchor_map).resolve()
    payload_path = Path(args.track_payload).resolve()
    cache_path = Path(args.query_cache).resolve()
    mask_path = (
        Path(args.deployment_mask_cache).resolve()
        if args.deployment_mask_cache
        else None
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_hashes = {
        "anchor_map_sha256": _sha256(anchor_path),
        "track_payload_sha256": _sha256(payload_path),
        "query_cache_sha256": _sha256(cache_path),
    }
    state = _load_anchor_map(anchor_path)
    train_start_row = (
        int(state["base_anchor_count"])
        if int(args.train_start_row) < 0
        else int(args.train_start_row)
    )
    if not 0 < train_start_row < int(state["anchor_ids"].numel()):
        raise ValueError("train-start-row must leave old and new anchors")
    payload = _load_payload(payload_path)
    print(f"Loading native query cache: {cache_path}", flush=True)
    query_payload = torch.load(
        cache_path, map_location="cpu", weights_only=False
    )
    query_cache = query_payload.get("queries", query_payload)
    deployment_masks = None
    if mask_path is not None:
        with mask_path.open("rb") as handle:
            deployment_masks = pickle.load(handle)
    print(
        f"Building protected data from {len(payload['query_names'])} train queries",
        flush=True,
    )
    data = _compact_training_data(
        state=state,
        payload=payload,
        query_cache=query_cache,
        device=device,
        guard_rows_per_query=args.guard_rows_per_query,
        clean_radius_px=args.clean_radius_px,
        score_chunk_size=args.score_chunk_size,
        train_start_row=train_start_row,
        deployment_masks=deployment_masks,
    )
    del query_payload, query_cache, payload

    base_count = train_start_row
    all_features = F.normalize(
        torch.as_tensor(state["anchor_features"]).float(), dim=1
    )
    initial = all_features[base_count:].to(device)
    residual = torch.nn.Parameter(torch.zeros_like(initial))
    optimizer = torch.optim.AdamW(
        [residual], lr=args.learning_rate, weight_decay=0.0
    )
    positive_sampling_weight = torch.reciprocal(
        data["positive_counts"][data["positive_targets"]].float()
    )
    positive_sampling_weight /= positive_sampling_weight.sum()
    checkpoints = {
        int(value)
        for value in args.checkpoint_steps.split(",")
        if value.strip()
    }
    checkpoints.add(int(args.steps))
    history = []
    initial_diagnostics = _full_diagnostics(initial, initial, data, args)
    print(
        "Initial diagnostics: " + json.dumps(initial_diagnostics, sort_keys=True),
        flush=True,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    for step in range(1, int(args.steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        positive_index = torch.multinomial(
            positive_sampling_weight,
            num_samples=int(args.positive_batch_size),
            replacement=True,
            generator=generator,
        )
        guard_count = data["guard_descriptors"].shape[0]
        guard_index = torch.randint(
            max(guard_count, 1),
            (int(args.guard_batch_size),),
            generator=generator,
        )
        if guard_count == 0:
            guard_index = guard_index[:0]
        candidate = F.normalize(initial + residual, dim=1)
        loss, diagnostics = protected_micro_anchor_descriptor_loss(
            candidate_features=candidate,
            positive_descriptors=data["positive_descriptors"][
                positive_index
            ].to(device),
            positive_targets=data["positive_targets"][positive_index].to(
                device
            ),
            positive_old_best=data["positive_old_best"][positive_index].to(
                device
            ),
            guard_descriptors=data["guard_descriptors"][guard_index].to(device),
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
            scale = (
                float(args.max_residual_norm) / norm.clamp_min(1e-8)
            ).clamp(max=1.0)
            residual.mul_(scale)
        if step == 1 or step % 50 == 0 or step in checkpoints:
            record = {
                "step": step,
                **{
                    name: float(value.item())
                    for name, value in diagnostics.items()
                },
                "residual_norm_mean": float(
                    torch.linalg.norm(residual, dim=1).mean().item()
                ),
                "residual_norm_max": float(
                    torch.linalg.norm(residual, dim=1).max().item()
                ),
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
        if step in checkpoints:
            final_features = F.normalize(initial + residual, dim=1)
            output_state = dict(state)
            output_state["anchor_features"] = torch.cat(
                (all_features[:base_count], final_features.detach().cpu())
            )
            output_state["descriptor_training"] = {
                "mode": "protected_gap_only_alternating_v2",
                "step": step,
                "train_start_row": int(train_start_row),
                "deployment_mask_cache_path": (
                    str(mask_path) if mask_path is not None else None
                ),
                "deployment_mask_cache_sha256": (
                    _sha256(mask_path) if mask_path is not None else None
                ),
                "old_anchor_descriptors_frozen": True,
                "old_anchor_geometry_frozen": True,
                "new_anchor_geometry_frozen": True,
                "anchor_map_path": str(anchor_path),
                "anchor_map_sha256": input_hashes["anchor_map_sha256"],
                "track_payload_path": str(payload_path),
                "track_payload_sha256": input_hashes[
                    "track_payload_sha256"
                ],
                "query_cache_path": str(cache_path),
                "query_cache_sha256": input_hashes["query_cache_sha256"],
                "config": vars(args),
                "initial_diagnostics": initial_diagnostics,
                "final_diagnostics": _full_diagnostics(
                    final_features, initial, data, args
                ),
            }
            output_path = output_dir / f"anchor_map_step_{step:04d}.pt"
            torch.save(output_state, output_path)
            print(f"Saved checkpoint: {output_path}", flush=True)

    manifest = {
        "schema": "lafgs_protected_micro_anchor_training",
        "version": 1,
        "anchor_map_path": str(anchor_path),
        "track_payload_path": str(payload_path),
        "query_cache_path": str(cache_path),
        "positive_observation_count": int(
            data["positive_descriptors"].shape[0]
        ),
        "guard_observation_count": int(data["guard_descriptors"].shape[0]),
        "new_anchor_count": int(initial.shape[0]),
        "initial_diagnostics": initial_diagnostics,
        "history": history,
        "config": vars(args),
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
