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

from localization_training.shared_metric import SharedLowRankMetric
from utils.pose_utils import cal_pose_error, solve_pose


def _first_k(values: torch.Tensor, mask: torch.Tensor, width: int) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError("candidate values and mask must align")
    width = min(int(width), values.shape[1])
    position = torch.arange(values.shape[1])[None].expand_as(values)
    sentinel = torch.full_like(position, values.shape[1])
    selected_position = torch.topk(
        torch.where(mask, position, sentinel),
        k=width,
        dim=1,
        largest=False,
        sorted=True,
    ).values
    valid = selected_position < values.shape[1]
    output = values.gather(
        1, selected_position.clamp_max(values.shape[1] - 1)
    )
    return torch.where(valid, output, torch.full_like(output, -1))


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


def _build_training_records(
    graph: dict, payload: dict, state: dict, max_positives: int
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
    records = []
    positive_rows = 0
    for query_index, record in enumerate(graph["records"]):
        cache_rows = torch.as_tensor(record["query_rows"]).long()
        candidates = torch.as_tensor(record["top_indices"]).long()
        flags = torch.as_tensor(record["legal_flags"]).to(torch.uint8)
        local = canonical_to_local[candidates]
        legal = (local >= 0) & ((flags & 2) != 0)
        positives = _first_k(local, legal, max_positives)
        for row, keypoint in enumerate(cache_rows.tolist()):
            track_local = exact.get(query_index, {}).get(int(keypoint), -1)
            if track_local < 0:
                continue
            current = positives[row]
            if bool((current == track_local).any()):
                continue
            empty = torch.nonzero(current < 0, as_tuple=False).reshape(-1)
            if empty.numel():
                current[empty[0]] = track_local
            else:
                current[-1] = track_local
        valid = (positives >= 0).any(dim=1)
        positive_rows += int(valid.sum())
        records.append(
            {
                "deployment_rows": cache_rows,
                "cache_rows": cache_rows[valid],
                "positives": positives[valid],
                "group": int(torch.as_tensor(payload["query_bins"])[query_index]),
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
    order = np.linspace(
        0, len(names) - 1, min(int(query_limit), len(names)), dtype=int
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
        score, index = (adapted @ bank.T).max(dim=1)
        keypoint = (
            torch.as_tensor(cached["native_keypoints"]).float()[
                deployment_rows
            ]
            + float(cached.get("pixel_center_offset", 0.5))
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
                "hypotheses": diagnostics.get("ransac_actual_hypotheses"),
            }
        )
    prior = harmful / (harmful + clean + 1.0)
    risks = {
        group: float(np.mean(values))
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
        graph, payload, state, args.max_positives
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
    optimizer = torch.optim.AdamW(
        [*metric.parameters(), anchor_residual],
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
    for step in range(1, args.steps + 1):
        if step == 1 or (
            args.refresh_interval > 0
            and (step - 1) % args.refresh_interval == 0
        ):
            (
                harmful_prior,
                group_risks,
                outcome,
                clean_pairs,
                harmful_pairs,
            ) = _refresh_ransac_outcomes(
                metric=metric,
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
                seed=args.seed,
            )
            risk = torch.zeros_like(group_weights)
            for group, value in group_risks.items():
                risk[group] = float(value) / 100.0
            group_weights *= torch.exp(float(args.group_dro_eta) * risk)
            group_weights /= group_weights.sum().clamp_min(1e-8)
            history.append(
                {
                    "step": step - 1,
                    "event": "deployment_refresh",
                    "mean_te_cm": float(
                        np.mean([row["te_cm"] for row in outcome])
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
        per_row, _, _ = _multi_positive_list_loss(
            adapted_query,
            adapted_anchor,
            positives,
            args.topk,
            args.temperature,
            None,
            args.harmful_weight,
            harmful_indices=harmful_survivors,
        )
        group_weight = (
            group_weights[int(record["group"])] * float(group_count)
        )
        task_loss = per_row.mean() * group_weight
        trust = (
            query_metric_residual.square().sum(dim=1).mean()
            + anchor_metric_residual.square().sum(dim=1).mean()
            + bounded_anchor.square().sum(dim=1).mean()
        )
        loss = task_loss + float(args.trust_weight) * trust
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
