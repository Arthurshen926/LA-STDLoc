#!/usr/bin/env python3
"""Online-refreshed shared-metric reconstruction for a Track-centric map."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from localization_training.shared_metric import (
    NativeNullHead,
    SharedLowRankMetric,
    build_native_null_features,
    select_native_matchable_rows,
)
from utils.pose_utils import cal_pose_error, solve_pose


def _first_k(values: torch.Tensor, mask: torch.Tensor, width: int) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError("candidate values and mask must align")
    width = min(int(width), values.shape[1])
    output = torch.full(
        (values.shape[0], width), -1, dtype=values.dtype, device=values.device
    )
    rank = mask.to(torch.int64).cumsum(dim=1) - 1
    selected = mask & (rank < width)
    rows, columns = torch.nonzero(selected, as_tuple=True)
    output[rows, rank[rows, columns]] = values[rows, columns]
    return output


def _track_observations(payload: dict, track_to_local: torch.Tensor):
    by_query: dict[int, dict[int, int]] = defaultdict(dict)
    tracks = payload["tracks"]
    for track, query, keypoint in zip(
        tracks["track_index"].tolist(),
        tracks["query_index"].tolist(),
        tracks["keypoint_index"].tolist(),
    ):
        local = int(track_to_local[int(track)])
        if local >= 0:
            by_query[int(query)][int(keypoint)] = local
    return by_query


def _build_rotating_shards(
    groups: torch.Tensor, shard_count: int
) -> list[list[int]]:
    """Round-robin every stable query group across deterministic shards."""
    groups = torch.as_tensor(groups).long().reshape(-1)
    shard_count = max(min(int(shard_count), groups.numel()), 1)
    shards: list[list[int]] = [[] for _ in range(shard_count)]
    for group in torch.unique(groups, sorted=True).tolist():
        indices = torch.nonzero(
            groups == int(group), as_tuple=False
        ).reshape(-1)
        for offset, query_index in enumerate(indices.tolist()):
            shards[offset % shard_count].append(int(query_index))
    for shard in shards:
        shard.sort()
    if sorted(index for shard in shards for index in shard) != list(
        range(groups.numel())
    ):
        raise RuntimeError("rotating query shards must cover every query once")
    return shards


def _group_pose_risk(errors_cm: list[float]) -> float:
    errors = torch.as_tensor(errors_cm, dtype=torch.float32)
    if errors.numel() == 0:
        return 0.0
    smooth_mean = torch.log1p(errors / 10.0).mean()
    tail_count = max(int(math.ceil(0.2 * errors.numel())), 1)
    tail = torch.topk(errors, k=tail_count).values.mean() / 20.0
    near_five = F.softplus((errors - 5.0) / 2.0).mean() / 5.0
    return float(smooth_mean + 0.5 * tail + 0.5 * near_five)


def _build_training_records(
    graph: dict,
    payload: dict,
    state: dict,
    max_positives: int,
    *,
    device: torch.device | str = "cpu",
    query_chunk_size: int = 32,
):
    metadata = state["track_centric_reconstruction"]
    track_indices = torch.as_tensor(metadata["track_indices"]).long()
    base_rows = torch.as_tensor(metadata["base_canonical_rows"]).long()
    track_count = int(track_indices.numel())
    canonical_count = int(graph["anchor_count"])
    canonical_to_local = torch.full(
        (canonical_count,), -1, dtype=torch.long
    )
    canonical_to_local[base_rows] = (
        torch.arange(base_rows.numel()) + track_count
    )
    payload_track_count = int(
        payload["track_geometry"]["triangulated_xyz"].shape[0]
    )
    track_to_local = torch.full(
        (payload_track_count,), -1, dtype=torch.long
    )
    track_to_local[track_indices] = torch.arange(track_count)
    exact = _track_observations(payload, track_to_local)
    graph_records = graph["records"]
    row_counts = [
        int(torch.as_tensor(record["query_rows"]).numel())
        for record in graph_records
    ]
    build_device = torch.device(device)
    canonical_to_local_build = canonical_to_local.to(build_device)
    positive_blocks = []
    legal4_blocks = []
    query_chunk_size = max(int(query_chunk_size), 1)
    for start in range(0, len(graph_records), query_chunk_size):
        chunk = graph_records[start : start + query_chunk_size]
        chunk_counts = row_counts[start : start + query_chunk_size]
        candidates = torch.cat(
            [
                torch.as_tensor(record["top_indices"]).long()
                for record in chunk
            ],
            dim=0,
        ).to(build_device)
        flags = torch.cat(
            [
                torch.as_tensor(record["legal_flags"]).to(torch.uint8)
                for record in chunk
            ],
            dim=0,
        ).to(build_device)
        candidate_valid = (candidates >= 0) & (
            candidates < canonical_count
        )
        local = canonical_to_local_build[
            candidates.clamp(min=0, max=canonical_count - 1)
        ]
        local = torch.where(
            candidate_valid, local, torch.full_like(local, -1)
        )
        chunk_positives = _first_k(
            local,
            (local >= 0) & candidate_valid & ((flags & 2) != 0),
            max_positives,
        ).cpu()
        chunk_legal4 = (
            candidate_valid & ((flags & 4) != 0)
        ).any(dim=1).cpu()
        positive_blocks.extend(chunk_positives.split(chunk_counts))
        legal4_blocks.extend(chunk_legal4.split(chunk_counts))
    query_bins = torch.as_tensor(payload["query_bins"]).long()
    del canonical_to_local_build
    records = []
    positive_rows = 0
    for query_index, record in enumerate(graph_records):
        cache_rows = torch.as_tensor(record["query_rows"]).long()
        positives = positive_blocks[query_index].clone()
        query_exact = exact.get(query_index, {})
        if query_exact and cache_rows.numel():
            exact_keypoints = torch.as_tensor(
                list(query_exact.keys()), dtype=torch.long
            )
            exact_tracks = torch.as_tensor(
                list(query_exact.values()), dtype=torch.long
            )
            lookup_size = int(
                max(cache_rows.max(), exact_keypoints.max()).item()
            ) + 1
            row_lookup = torch.full((lookup_size,), -1, dtype=torch.long)
            row_lookup[cache_rows] = torch.arange(cache_rows.numel())
            exact_rows = row_lookup[exact_keypoints]
            present = exact_rows >= 0
            exact_rows = exact_rows[present]
            exact_tracks = exact_tracks[present]
            current = positives[exact_rows]
            missing = ~(current == exact_tracks[:, None]).any(dim=1)
            if bool(missing.any()):
                exact_rows = exact_rows[missing]
                exact_tracks = exact_tracks[missing]
                empty = positives[exact_rows] < 0
                has_empty = empty.any(dim=1)
                columns = torch.where(
                    has_empty,
                    empty.to(torch.int64).argmax(dim=1),
                    torch.full(
                        (exact_rows.numel(),),
                        positives.shape[1] - 1,
                        dtype=torch.long,
                    ),
                )
                positives[exact_rows, columns] = exact_tracks
        valid = (positives >= 0).any(dim=1)
        canonical_legal4 = legal4_blocks[query_index]
        null_weight = torch.where(
            valid,
            torch.ones_like(valid, dtype=torch.float32),
            torch.where(
                canonical_legal4,
                torch.full_like(valid, 0.25, dtype=torch.float32),
                torch.ones_like(valid, dtype=torch.float32),
            ),
        )
        positive_rows += int(valid.sum())
        records.append(
            {
                "deployment_rows": cache_rows,
                "cache_rows": cache_rows,
                "positives": positives,
                "matchable": valid,
                "null_weight": null_weight,
                "group": int(query_bins[query_index]),
            }
        )
    return records, {
        "positive_rows": positive_rows,
        "track_anchor_count": track_count,
        "base_anchor_count": int(base_rows.numel()),
    }


def _bounded_anchor_features(
    raw: torch.Tensor, residual: torch.Tensor, maximum: float
) -> tuple[torch.Tensor, torch.Tensor]:
    norm = torch.linalg.norm(residual, dim=1, keepdim=True)
    bounded = residual * torch.clamp(
        float(maximum) / norm.clamp_min(1e-8), max=1.0
    )
    return F.normalize(raw + bounded, dim=1), bounded


def _multi_positive_list_loss(
    query: torch.Tensor,
    bank: torch.Tensor,
    positives: torch.Tensor,
    topk: int,
    temperature: float,
    harmful_prior: torch.Tensor | None,
    harmful_weight: float,
    harmful_indices: torch.Tensor | None = None,
):
    scores = query @ bank.T
    top_scores, top_indices = torch.topk(
        scores, k=min(int(topk), bank.shape[0]), dim=1
    )
    safe = positives.clamp_min(0)
    positive_scores = torch.einsum(
        "bd,bpd->bp", query, bank[safe]
    )
    positive_mask = positives >= 0
    numerator = torch.logsumexp(
        (positive_scores / temperature).masked_fill(
            ~positive_mask, -torch.inf
        ),
        dim=1,
    )
    top_is_positive = (
        (top_indices[:, :, None] == positives[:, None, :])
        & positive_mask[:, None, :]
    ).any(dim=2)
    denominator_scores = torch.cat([top_scores, positive_scores], dim=1)
    denominator_mask = torch.cat(
        [
            ~top_is_positive,
            positive_mask,
        ],
        dim=1,
    )
    denominator = torch.logsumexp(
        (denominator_scores / temperature).masked_fill(
            ~denominator_mask, -torch.inf
        ),
        dim=1,
    )
    list_loss = (denominator - numerator) * temperature
    harmful_loss = torch.zeros_like(list_loss)
    if harmful_prior is not None:
        retrieved_harm = harmful_prior[top_indices]
        harmful_loss = harmful_loss + (
            torch.softmax(top_scores / temperature, dim=1) * retrieved_harm
        ).sum(dim=1)
    if harmful_indices is not None:
        harmful_valid = harmful_indices >= 0
        safe_harmful = harmful_indices.clamp_min(0)
        harmful_scores = torch.einsum(
            "bd,bhd->bh", query, bank[safe_harmful]
        )
        harmful_scores = harmful_scores.masked_fill(
            ~harmful_valid, -torch.inf
        )
        hardest_harmful = harmful_scores.max(dim=1).values
        has_harmful = harmful_valid.any(dim=1)
        positive_aggregate = numerator * float(temperature)
        harmful_loss = harmful_loss + torch.where(
            has_harmful,
            F.softplus(
                (hardest_harmful - positive_aggregate)
                / float(temperature)
            )
            * float(temperature),
            torch.zeros_like(harmful_loss),
        )
    return (
        list_loss + float(harmful_weight) * harmful_loss,
        top_indices,
        top_scores,
    )


def _project_errors(xyz, keypoints, K, pose):
    camera = xyz @ pose[:3, :3].T + pose[:3, 3]
    depth = camera[:, 2]
    projected = torch.empty_like(keypoints)
    projected[:, 0] = K[0, 0] * camera[:, 0] / depth.clamp_min(1e-8) + K[0, 2]
    projected[:, 1] = K[1, 1] * camera[:, 1] / depth.clamp_min(1e-8) + K[1, 2]
    return torch.linalg.norm(projected - keypoints, dim=1)


@torch.no_grad()
def _refresh_ransac_outcomes(
    *,
    metric,
    null_head,
    null_temperature,
    null_threshold,
    null_minimum_total,
    null_grid_rows,
    null_grid_cols,
    null_minimum_per_cell,
    raw_features,
    anchor_residual,
    maximum_anchor_residual,
    state,
    cache,
    names,
    groups,
    training_records,
    device,
    query_limit,
    query_indices,
    seed,
):
    anchor, _ = _bounded_anchor_features(
        raw_features, anchor_residual, maximum_anchor_residual
    )
    bank, _ = metric(anchor)
    xyz_cpu = torch.as_tensor(state["anchor_xyz"]).float()
    harmful = torch.zeros(bank.shape[0])
    clean = torch.zeros(bank.shape[0])
    harmful_pairs: dict[int, dict[int, int]] = defaultdict(dict)
    clean_pairs: dict[int, dict[int, int]] = defaultdict(dict)
    group_error: dict[int, list[float]] = defaultdict(list)
    order = (
        np.asarray(query_indices, dtype=int)
        if query_indices is not None
        else np.linspace(
            0,
            len(names) - 1,
            min(int(query_limit), len(names)),
            dtype=int,
        )
    )
    records = []
    for query_index in order.tolist():
        cached = cache[names[query_index]]
        deployment_rows = training_records[query_index][
            "deployment_rows"
        ].long()
        descriptors = F.normalize(
            torch.as_tensor(cached["native_descriptors"]).float()[
                deployment_rows
            ],
            dim=1,
        ).to(device)
        adapted, _ = metric(descriptors)
        score_matrix = adapted @ bank.T
        null_top_scores, null_top_indices = torch.topk(
            score_matrix, k=min(8, score_matrix.shape[1]), dim=1
        )
        keypoint_grid = torch.as_tensor(
            cached["native_keypoints"]
        ).float()[deployment_rows]
        null_features = build_native_null_features(
            null_top_scores,
            torch.as_tensor(cached["native_scores"]).float()[
                deployment_rows
            ].to(device),
            temperature=float(null_temperature),
        )
        matchable_probability = torch.sigmoid(null_head(null_features))
        native_height, native_width = cached["native_input_hw"]
        keep = select_native_matchable_rows(
            matchable_probability,
            keypoint_grid.to(device),
            width=int(native_width),
            height=int(native_height),
            threshold=float(null_threshold),
            minimum_total=int(null_minimum_total),
            grid_rows=int(null_grid_rows),
            grid_cols=int(null_grid_cols),
            minimum_per_cell=int(null_minimum_per_cell),
        )
        score = null_top_scores[keep, 0]
        index = null_top_indices[keep, 0]
        deployment_rows = deployment_rows[keep.cpu()]
        keypoint = keypoint_grid[keep.cpu()] + float(
            cached.get("pixel_center_offset", 0.5)
        )
        K = torch.as_tensor(cached["native_K"]).float()
        pose, inliers, diagnostics = solve_pose(
            keypoint.numpy(),
            xyz_cpu[index.cpu()].numpy(),
            K.numpy(),
            solver="poselib",
            reprojection_error=12.0,
            confidence=0.99999,
            max_iterations=100000,
            min_iterations=1000,
            scores=score.cpu().numpy(),
            ransac_seed=int(seed),
            return_diagnostics=True,
        )
        _, te_cm = cal_pose_error(
            pose, torch.as_tensor(cached["pose_w2c"]).numpy()
        )
        group = int(groups[query_index])
        group_error[group].append(float(te_cm))
        inliers = torch.as_tensor(inliers).long().reshape(-1)
        if inliers.numel():
            gt_pose = torch.as_tensor(cached["pose_w2c"]).float()
            errors = _project_errors(
                xyz_cpu[index.cpu()[inliers]],
                keypoint[inliers],
                K,
                gt_pose,
            )
            clean_mask = errors <= 4.0
            clean.index_add_(
                0,
                index.cpu()[inliers[clean_mask]],
                torch.ones(int(clean_mask.sum())),
            )
            harmful.index_add_(
                0,
                index.cpu()[inliers[~clean_mask]],
                torch.ones(int((~clean_mask).sum())),
            )
            inlier_cache_rows = deployment_rows[inliers]
            inlier_anchors = index.cpu()[inliers]
            for cache_row, anchor, is_clean in zip(
                inlier_cache_rows.tolist(),
                inlier_anchors.tolist(),
                clean_mask.tolist(),
            ):
                target = clean_pairs if is_clean else harmful_pairs
                target[query_index][int(cache_row)] = int(anchor)
        records.append(
            {
                "query_index": query_index,
                "group": group,
                "te_cm": float(te_cm),
                "inliers": int(inliers.numel()),
                "candidate_count": int(keep.numel()),
                "hypotheses": diagnostics.get("ransac_actual_hypotheses"),
            }
        )
    prior = harmful / (harmful + clean + 1.0)
    risks = {
        group: _group_pose_risk(values)
        for group, values in group_error.items()
    }
    return (
        prior.to(device),
        risks,
        records,
        dict(clean_pairs),
        dict(harmful_pairs),
    )


def _save_checkpoint(
    *,
    output_dir,
    step,
    state,
    metric,
    null_head,
    raw_features,
    anchor_residual,
    maximum_anchor_residual,
    history,
    config,
):
    with torch.no_grad():
        anchor, bounded = _bounded_anchor_features(
            raw_features, anchor_residual, maximum_anchor_residual
        )
        transformed, _ = metric(anchor)
    output = dict(state)
    output["v7_metric_raw_features"] = anchor.detach().cpu()
    output["anchor_features"] = transformed.detach().cpu()
    output["v7_online_metric"] = {
        "schema": "lafgs_v7_online_shared_metric",
        "version": 1,
        "step": int(step),
        "anchor_residual_mean": float(bounded.norm(dim=1).mean()),
        "anchor_residual_max": float(bounded.norm(dim=1).max()),
        "config": config,
        "history": history,
    }
    map_path = output_dir / f"anchor_map_step_{step:04d}.pt"
    metric_path = output_dir / f"metric_state_step_{step:04d}.pt"
    torch.save(output, map_path)
    torch.save(
        {
            "schema": "lafgs_v7_shared_metric_state",
            "version": 1,
            "landmark_indices": torch.arange(
                transformed.shape[0], dtype=torch.long
            ),
            "metric_config": metric.export_config(),
            "metric_state_dict": {
                key: value.detach().cpu()
                for key, value in metric.state_dict().items()
            },
            "null_head_config": {
                "feature_dim": int(null_head.feature_dim),
                "temperature": float(config["null_temperature"]),
                "threshold": float(config["null_threshold"]),
                "minimum_total": int(config["null_minimum_total"]),
                "grid_rows": int(config["null_grid_rows"]),
                "grid_cols": int(config["null_grid_cols"]),
                "minimum_per_cell": int(config["null_minimum_per_cell"]),
            },
            "null_head_state_dict": {
                key: value.detach().cpu()
                for key, value in null_head.state_dict().items()
            },
            "map_path": str(map_path),
            "step": int(step),
        },
        metric_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--function-graph", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--checkpoint-steps", default="100,250,500")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--max-positives", type=int, default=4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--metric-residual", type=float, default=0.10)
    parser.add_argument("--anchor-residual", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.04)
    parser.add_argument("--harmful-weight", type=float, default=0.05)
    parser.add_argument("--trust-weight", type=float, default=0.1)
    parser.add_argument("--group-dro-eta", type=float, default=0.03)
    parser.add_argument("--refresh-interval", type=int, default=100)
    parser.add_argument("--refresh-query-limit", type=int, default=128)
    parser.add_argument("--refresh-shards", type=int, default=7)
    parser.add_argument("--null-weight", type=float, default=0.2)
    parser.add_argument("--null-temperature", type=float, default=0.05)
    parser.add_argument("--null-threshold", type=float, default=0.5)
    parser.add_argument("--null-minimum-total", type=int, default=384)
    parser.add_argument("--null-grid-rows", type=int, default=4)
    parser.add_argument("--null-grid-cols", type=int, default=4)
    parser.add_argument("--null-minimum-per-cell", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    graph = torch.load(
        args.function_graph, map_location="cpu", weights_only=False
    )
    payload = torch.load(
        args.track_payload, map_location="cpu", weights_only=False
    )
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    names = graph["query_names"]
    records, data_report = _build_training_records(
        graph,
        payload,
        state,
        args.max_positives,
        device=device,
    )
    del graph
    raw_features = F.normalize(
        torch.as_tensor(
            state.get("v7_metric_raw_features", state["anchor_features"])
        ).float(),
        dim=1,
    ).to(device)
    anchor_residual = torch.nn.Parameter(torch.zeros_like(raw_features))
    metric = SharedLowRankMetric(
        descriptor_dim=raw_features.shape[1],
        rank=args.rank,
        max_residual_norm=args.metric_residual,
    ).to(device)
    null_head = NativeNullHead().to(device)
    optimizer = torch.optim.AdamW(
        [*metric.parameters(), *null_head.parameters(), anchor_residual],
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    group_count = int(torch.as_tensor(payload["query_bins"]).max()) + 1
    group_weights = torch.ones(group_count, device=device) / group_count
    harmful_prior = torch.zeros(raw_features.shape[0], device=device)
    clean_pairs = {}
    harmful_pairs = {}
    generator = torch.Generator().manual_seed(args.seed + 1)
    checkpoints = {
        int(value)
        for value in args.checkpoint_steps.split(",")
        if value.strip()
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    groups = torch.as_tensor(payload["query_bins"]).long()
    refresh_shards = _build_rotating_shards(groups, args.refresh_shards)
    refresh_index = 0
    for step in range(1, args.steps + 1):
        if step == 1 or (
            args.refresh_interval > 0
            and (step - 1) % args.refresh_interval == 0
        ):
            active_shard = refresh_index % len(refresh_shards)
            (
                harmful_prior,
                group_risks,
                outcome,
                refreshed_clean_pairs,
                refreshed_harmful_pairs,
            ) = _refresh_ransac_outcomes(
                metric=metric,
                null_head=null_head,
                null_temperature=args.null_temperature,
                null_threshold=args.null_threshold,
                null_minimum_total=args.null_minimum_total,
                null_grid_rows=args.null_grid_rows,
                null_grid_cols=args.null_grid_cols,
                null_minimum_per_cell=args.null_minimum_per_cell,
                raw_features=raw_features,
                anchor_residual=anchor_residual,
                maximum_anchor_residual=args.anchor_residual,
                state=state,
                cache=cache,
                names=names,
                groups=groups,
                training_records=records,
                device=device,
                query_limit=args.refresh_query_limit,
                query_indices=refresh_shards[active_shard],
                seed=args.seed,
            )
            clean_pairs.update(refreshed_clean_pairs)
            harmful_pairs.update(refreshed_harmful_pairs)
            risk = torch.zeros_like(group_weights)
            for group, value in group_risks.items():
                risk[group] = float(value)
            group_weights *= torch.exp(float(args.group_dro_eta) * risk)
            group_weights /= group_weights.sum().clamp_min(1e-8)
            history.append(
                {
                    "step": step - 1,
                    "event": "deployment_refresh",
                    "shard": int(active_shard),
                    "shard_query_count": len(refresh_shards[active_shard]),
                    "covered_query_count": int(
                        sum(
                            len(refresh_shards[index])
                            for index in range(
                                min(refresh_index + 1, len(refresh_shards))
                            )
                        )
                    ),
                    "mean_te_cm": float(
                        np.mean([row["te_cm"] for row in outcome])
                    ),
                    "mean_candidate_count": float(
                        np.mean(
                            [row["candidate_count"] for row in outcome]
                        )
                    ),
                    "mean_hypotheses": float(
                        np.mean(
                            [
                                row["hypotheses"]
                                for row in outcome
                                if row["hypotheses"] is not None
                            ]
                        )
                    ),
                    "harmful_anchor_fraction": float(
                        (harmful_prior > 0).float().mean()
                    ),
                    "group_weight_max": float(group_weights.max()),
                }
            )
            print(json.dumps(history[-1]), flush=True)
            refresh_index += 1

        query_index = int(
            torch.randint(len(records), (1,), generator=generator)
        )
        record = records[query_index]
        count = int(record["cache_rows"].numel())
        if count == 0:
            continue
        rows = torch.randint(
            count,
            (min(args.batch_size, count),),
            generator=generator,
        )
        cache_rows = record["cache_rows"][rows]
        query = F.normalize(
            torch.as_tensor(
                cache[names[query_index]]["native_descriptors"]
            ).float()[cache_rows],
            dim=1,
        ).to(device)
        positives = record["positives"][rows].to(device)
        matchable = record["matchable"][rows].to(device)
        null_weight = record["null_weight"][rows].to(device)
        current_clean = clean_pairs.get(query_index, {})
        clean_survivors = torch.as_tensor(
            [current_clean.get(int(row), -1) for row in cache_rows],
            dtype=torch.long,
            device=device,
        )
        add_clean = (clean_survivors >= 0) & ~(
            positives == clean_survivors[:, None]
        ).any(dim=1)
        if bool(add_clean.any()):
            positives = positives.clone()
            replace = torch.where(
                (positives < 0).any(dim=1),
                (positives < 0).to(torch.int64).argmax(dim=1),
                torch.full(
                    (positives.shape[0],),
                    positives.shape[1] - 1,
                    device=device,
                ),
            )
            positives[
                torch.arange(positives.shape[0], device=device)[add_clean],
                replace[add_clean],
            ] = clean_survivors[add_clean]
            matchable = matchable | add_clean
        current_harmful = harmful_pairs.get(query_index, {})
        harmful_survivors = torch.as_tensor(
            [current_harmful.get(int(row), -1) for row in cache_rows],
            dtype=torch.long,
            device=device,
        )[:, None]
        anchor, bounded_anchor = _bounded_anchor_features(
            raw_features, anchor_residual, args.anchor_residual
        )
        adapted_query, query_metric_residual = metric(query)
        adapted_anchor, anchor_metric_residual = metric(anchor)
        score_matrix = adapted_query @ adapted_anchor.T
        top_scores = torch.topk(
            score_matrix, k=min(args.topk, score_matrix.shape[1]), dim=1
        ).values
        list_loss = torch.zeros(
            adapted_query.shape[0], device=device, dtype=adapted_query.dtype
        )
        if bool(matchable.any()):
            list_loss[matchable] = _multi_positive_list_loss(
                adapted_query[matchable],
                adapted_anchor,
                positives[matchable],
                args.topk,
                args.temperature,
                None,
                args.harmful_weight,
                harmful_indices=harmful_survivors[matchable],
            )[0]
        keypoint_score = torch.as_tensor(
            cache[names[query_index]]["native_scores"]
        ).float()[cache_rows].to(device)
        null_features = build_native_null_features(
            top_scores,
            keypoint_score,
            temperature=args.null_temperature,
        )
        null_logits = null_head(null_features)
        null_loss = F.binary_cross_entropy_with_logits(
            null_logits,
            matchable.float(),
            weight=null_weight,
            reduction="mean",
        )
        group_weight = (
            group_weights[int(record["group"])] * float(group_count)
        )
        task_loss = (
            list_loss[matchable].mean()
            if bool(matchable.any())
            else torch.zeros((), device=device)
        ) * group_weight
        trust = (
            query_metric_residual.square().sum(dim=1).mean()
            + anchor_metric_residual.square().sum(dim=1).mean()
            + bounded_anchor.square().sum(dim=1).mean()
        )
        loss = (
            task_loss
            + float(args.null_weight) * null_loss
            + float(args.trust_weight) * trust
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*metric.parameters(), anchor_residual], 1.0
        )
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == args.steps:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "task_loss": float(task_loss.detach()),
                "trust_loss": float(trust.detach()),
                "null_loss": float(null_loss.detach()),
                "matchable_fraction": float(matchable.float().mean()),
                "group": int(record["group"]),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
        if step in checkpoints or step == args.steps:
            _save_checkpoint(
                output_dir=output_dir,
                step=step,
                state=state,
                metric=metric,
                null_head=null_head,
                raw_features=raw_features,
                anchor_residual=anchor_residual,
                maximum_anchor_residual=args.anchor_residual,
                history=history,
                config={**vars(args), **data_report},
            )
    (output_dir / "training_report.json").write_text(
        json.dumps(
            {
                "schema": "lafgs_v7_online_metric_training",
                "config": {**vars(args), **data_report},
                "history": history,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
