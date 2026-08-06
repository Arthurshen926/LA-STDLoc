"""Self-localization-guided descriptor reconstruction for a compact map.

The trainer contains only the frozen paper path: complete-positive retrieval,
current-map hard outcomes, trajectory-group DRO, and a bounded shared metric.
All geometry and anchor identities remain fixed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from map_learning.metric import SharedLowRankMetric
from localization.pose_solver import solve_absolute_pose


def _query_index_remap(source: list[str], target: list[str]) -> torch.Tensor:
    if len(set(source)) != len(source) or len(set(target)) != len(target):
        raise ValueError("query registries must be unique")
    target_by_name = {name: index for index, name in enumerate(target)}
    if set(source) != set(target):
        raise ValueError("query registries differ")
    return torch.as_tensor([target_by_name[name] for name in source]).long()


def _csr_first_k(offsets, indices, width: int) -> torch.Tensor:
    offsets = torch.as_tensor(offsets).long().reshape(-1)
    indices = torch.as_tensor(indices).long().reshape(-1)
    output = torch.full((offsets.numel() - 1, int(width)), -1, dtype=torch.long)
    counts = offsets[1:] - offsets[:-1]
    rows = torch.repeat_interleave(torch.arange(counts.numel()), counts)
    rank = torch.arange(indices.numel()) - offsets[rows]
    keep = rank < int(width)
    output[rows[keep], rank[keep]] = indices[keep]
    return output


def _build_training_records(
    graph: dict,
    track_payload: dict,
    state: dict,
    teacher: dict,
    max_positives: int,
) -> tuple[list[dict], dict]:
    names = list(graph["query_names"])
    if list(teacher["query_names"]) != names:
        raise ValueError("complete-positive teacher query order mismatch")
    if int(teacher["anchor_count"]) != int(state["anchor_xyz"].shape[0]):
        raise ValueError("complete-positive teacher anchor count mismatch")
    graph_records = graph["records"]
    if len(teacher["records"]) != len(graph_records):
        raise ValueError("complete-positive teacher query count mismatch")

    payload_to_graph = _query_index_remap(track_payload["query_names"], names)
    payload_bins = torch.as_tensor(track_payload["query_bins"]).long()
    query_bins = torch.empty_like(payload_bins)
    query_bins[payload_to_graph] = payload_bins

    records = []
    positive_rows = 0
    ambiguous_pairs = 0
    ignored_ambiguous_pairs = 0
    for graph_record, teacher_record, group in zip(
        graph_records, teacher["records"], query_bins.tolist()
    ):
        query_rows = torch.as_tensor(graph_record["query_rows"]).long()
        teacher_rows = torch.as_tensor(teacher_record["query_rows"]).long()
        if not torch.equal(query_rows, teacher_rows):
            raise ValueError("complete-positive teacher row mismatch")
        positives = _csr_first_k(
            teacher_record["positive_offsets"],
            teacher_record["positive_indices"],
            max_positives,
        )
        ambiguous = _csr_first_k(
            teacher_record["ambiguous_offsets"],
            teacher_record["ambiguous_indices"],
            max_positives,
        )
        ignore_ambiguous = graph_record.get("ambiguous_training_policy") == "ignore"
        ignored = ambiguous if ignore_ambiguous else torch.full_like(ambiguous, -1)
        matchable = (positives >= 0).any(dim=1)
        positive_rows += int(matchable.sum())
        ambiguous_pairs += int((ambiguous >= 0).sum())
        if ignore_ambiguous:
            ignored_ambiguous_pairs += int((ambiguous >= 0).sum())
        records.append(
            {
                "deployment_rows": query_rows,
                "cache_rows": query_rows,
                "positives": positives,
                "ignored_anchors": ignored,
                "matchable": matchable,
                "group": int(group),
            }
        )
    metadata = state["track_centric_reconstruction"]
    return records, {
        "positive_rows": positive_rows,
        "complete_positive_pair_count": int(
            teacher["diagnostics"]["strong_pair_count"]
        ),
        "ambiguous_pair_count": ambiguous_pairs,
        "ignored_ambiguous_pair_count": ignored_ambiguous_pairs,
        "track_anchor_count": int(
            torch.as_tensor(metadata["track_indices"]).numel()
        ),
        "base_anchor_count": int(
            torch.as_tensor(metadata["base_canonical_rows"]).numel()
        ),
        "query_groups": query_bins.tolist(),
    }


def _build_rotating_shards(groups: torch.Tensor, count: int) -> list[list[int]]:
    groups = torch.as_tensor(groups).long().reshape(-1)
    count = max(min(int(count), groups.numel()), 1)
    shards: list[list[int]] = [[] for _ in range(count)]
    for group in torch.unique(groups, sorted=True).tolist():
        indices = torch.nonzero(groups == int(group), as_tuple=False).reshape(-1)
        for offset, query_index in enumerate(indices.tolist()):
            shards[offset % count].append(int(query_index))
    for shard in shards:
        shard.sort()
    if sorted(index for shard in shards for index in shard) != list(
        range(groups.numel())
    ):
        raise RuntimeError("refresh shards do not cover each mapping query once")
    return shards


def _replace_refreshed_pairs(
    clean_pairs: dict,
    harmful_pairs: dict,
    query_indices: list[int],
    refreshed_clean: dict,
    refreshed_harmful: dict,
) -> dict[str, int]:
    old_clean = sum(len(clean_pairs.get(int(query), {})) for query in query_indices)
    old_harmful = sum(
        len(harmful_pairs.get(int(query), {})) for query in query_indices
    )
    for query in query_indices:
        clean_pairs.pop(int(query), None)
        harmful_pairs.pop(int(query), None)
    clean_pairs.update({int(key): dict(value) for key, value in refreshed_clean.items()})
    harmful_pairs.update(
        {int(key): dict(value) for key, value in refreshed_harmful.items()}
    )
    return {
        "old_clean_pair_count": old_clean,
        "old_harmful_pair_count": old_harmful,
        "new_clean_pair_count": sum(
            len(clean_pairs.get(int(query), {})) for query in query_indices
        ),
        "new_harmful_pair_count": sum(
            len(harmful_pairs.get(int(query), {})) for query in query_indices
        ),
    }


def _group_pose_risk(errors_cm: list[float]) -> float:
    errors = torch.as_tensor(errors_cm, dtype=torch.float32)
    if errors.numel() == 0:
        return 0.0
    smooth_mean = torch.log1p(errors / 10.0).mean()
    tail_count = max(int(math.ceil(0.2 * errors.numel())), 1)
    tail = torch.topk(errors, k=tail_count).values.mean() / 20.0
    near_five = F.softplus((errors - 5.0) / 2.0).mean() / 5.0
    return float(smooth_mean + 0.5 * tail + 0.5 * near_five)


def _project_errors(xyz, keypoints, intrinsic, pose):
    camera = xyz @ pose[:3, :3].T + pose[:3, 3]
    depth = camera[:, 2]
    projected = torch.empty_like(keypoints)
    projected[:, 0] = (
        intrinsic[0, 0] * camera[:, 0] / depth.clamp_min(1e-8) + intrinsic[0, 2]
    )
    projected[:, 1] = (
        intrinsic[1, 1] * camera[:, 1] / depth.clamp_min(1e-8) + intrinsic[1, 2]
    )
    return torch.linalg.norm(projected - keypoints, dim=1)


def _pose_error_cm(predicted: np.ndarray, target: torch.Tensor) -> float:
    target = torch.as_tensor(target).cpu().numpy()
    predicted_center = np.linalg.inv(predicted)[:3, -1]
    target_center = np.linalg.inv(target)[:3, -1]
    return float(np.linalg.norm(predicted_center - target_center) * 100.0)


@torch.no_grad()
def _refresh_ransac_outcomes(
    *,
    metric: SharedLowRankMetric,
    raw_features: torch.Tensor,
    state: dict,
    cache: dict,
    names: list[str],
    groups: torch.Tensor,
    training_records: list[dict],
    device: torch.device,
    query_indices: list[int],
    seed: int,
    ransac_reprojection_px: float,
    clean_reprojection_px: float,
):
    bank, _ = metric(raw_features)
    xyz_cpu = torch.as_tensor(state["anchor_xyz"]).float()
    harmful = torch.zeros(bank.shape[0])
    clean = torch.zeros(bank.shape[0])
    harmful_pairs: dict[int, dict[int, int]] = defaultdict(dict)
    clean_pairs: dict[int, dict[int, int]] = defaultdict(dict)
    group_error: dict[int, list[float]] = defaultdict(list)
    rows = []
    for query_index in np.asarray(query_indices, dtype=int).tolist():
        cached = cache[names[query_index]]
        deployment_rows = training_records[query_index]["deployment_rows"].long()
        descriptors = F.normalize(
            torch.as_tensor(cached["native_descriptors"]).float()[deployment_rows],
            dim=1,
        ).to(device)
        adapted, _ = metric(descriptors)
        top_scores, top_indices = torch.topk(
            adapted @ bank.T, k=min(8, bank.shape[0]), dim=1
        )
        index = top_indices[:, 0]
        keypoint = torch.as_tensor(cached["native_keypoints"]).float()[
            deployment_rows
        ] + float(cached.get("pixel_center_offset", 0.5))
        intrinsic = torch.as_tensor(cached["native_K"]).float()
        estimate = solve_absolute_pose(
            keypoint.numpy(),
            xyz_cpu[index.cpu()].numpy(),
            intrinsic.numpy(),
            reprojection_error_px=float(ransac_reprojection_px),
            confidence=0.99999,
            max_iterations=100000,
            min_iterations=1000,
            seed=int(seed),
        )
        te_cm = _pose_error_cm(
            estimate.pose_w2c, torch.as_tensor(cached["pose_w2c"])
        )
        group = int(groups[query_index])
        group_error[group].append(te_cm)
        inliers = torch.as_tensor(estimate.inliers).long().reshape(-1)
        if inliers.numel():
            errors = _project_errors(
                xyz_cpu[index.cpu()[inliers]],
                keypoint[inliers],
                intrinsic,
                torch.as_tensor(cached["pose_w2c"]).float(),
            )
            clean_mask = errors <= float(clean_reprojection_px)
            clean.index_add_(
                0, index.cpu()[inliers[clean_mask]], torch.ones(int(clean_mask.sum()))
            )
            harmful.index_add_(
                0,
                index.cpu()[inliers[~clean_mask]],
                torch.ones(int((~clean_mask).sum())),
            )
            for cache_row, anchor, is_clean in zip(
                deployment_rows[inliers].tolist(),
                index.cpu()[inliers].tolist(),
                clean_mask.tolist(),
            ):
                target = clean_pairs if is_clean else harmful_pairs
                target[query_index][int(cache_row)] = int(anchor)
        rows.append(
            {
                "query_index": query_index,
                "group": group,
                "te_cm": te_cm,
                "inliers": int(inliers.numel()),
                "candidate_count": int(deployment_rows.numel()),
                "hypotheses": estimate.diagnostics.get("iterations"),
            }
        )
    return (
        (harmful / (harmful + clean + 1.0)).to(device),
        {group: _group_pose_risk(value) for group, value in group_error.items()},
        rows,
        dict(clean_pairs),
        dict(harmful_pairs),
    )


def _multi_positive_list_loss(
    query: torch.Tensor,
    bank: torch.Tensor,
    positives: torch.Tensor,
    ignored: torch.Tensor,
    harmful: torch.Tensor,
    *,
    topk: int,
    temperature: float,
    harmful_weight: float,
) -> torch.Tensor:
    scores = query @ bank.T
    top_scores, top_indices = torch.topk(
        scores, k=min(int(topk), bank.shape[0]), dim=1
    )
    positive_mask = positives >= 0
    positive_scores = torch.einsum(
        "bd,bpd->bp", query, bank[positives.clamp_min(0)]
    )
    top_is_positive = (
        (top_indices[:, :, None] == positives[:, None, :])
        & positive_mask[:, None, :]
    ).any(dim=2)
    ignored_valid = ignored >= 0
    top_is_ignored = (
        (top_indices[:, :, None] == ignored[:, None, :])
        & ignored_valid[:, None, :]
    ).any(dim=2)
    denominator = torch.logsumexp(
        (
            torch.cat((top_scores, positive_scores), dim=1)
            / float(temperature)
        ).masked_fill(
            ~torch.cat((~top_is_positive & ~top_is_ignored, positive_mask), dim=1),
            -torch.inf,
        ),
        dim=1,
    )
    target = positive_mask.float()
    target /= target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    positive_aggregate = (target * positive_scores).sum(dim=1)
    list_loss = denominator * float(temperature) - positive_aggregate

    harmful_valid = harmful >= 0
    harmful_scores = torch.einsum(
        "bd,bhd->bh", query, bank[harmful.clamp_min(0)]
    ).masked_fill(~harmful_valid, -torch.inf)
    hardest = harmful_scores.max(dim=1).values
    harmful_loss = torch.where(
        harmful_valid.any(dim=1),
        F.softplus((hardest - positive_aggregate) / float(temperature))
        * float(temperature),
        torch.zeros_like(list_loss),
    )
    return list_loss + float(harmful_weight) * harmful_loss


def _save_checkpoint(
    output_dir: Path,
    step: int,
    state: dict,
    metric: SharedLowRankMetric,
    raw_features: torch.Tensor,
    history: list[dict],
    config: dict,
) -> None:
    with torch.no_grad():
        transformed, _ = metric(raw_features)
    output = dict(state)
    output["v7_metric_raw_features"] = raw_features.detach().cpu()
    output["anchor_features"] = transformed.detach().cpu()
    output["v7_online_metric"] = {
        "schema": "lafgs_self_localization_descriptor_reconstruction",
        "version": 1,
        "step": int(step),
        "config": config,
        "history": history,
    }
    map_path = output_dir / f"anchor_map_step_{step:04d}.pt"
    torch.save(output, map_path)
    torch.save(
        {
            "schema": "lafgs_shared_metric_state",
            "version": 1,
            "landmark_indices": torch.arange(transformed.shape[0]).long(),
            "metric_config": metric.export_config(),
            "metric_state_dict": {
                key: value.detach().cpu() for key, value in metric.state_dict().items()
            },
            "map_path": str(map_path),
            "step": int(step),
        },
        output_dir / f"metric_state_step_{step:04d}.pt",
    )


def train(
    *,
    map_path: str | Path,
    function_graph_path: str | Path,
    track_payload_path: str | Path,
    query_cache_path: str | Path,
    positive_teacher_path: str | Path,
    output_dir: str | Path,
    steps: int = 175,
    checkpoint_steps: tuple[int, ...] = (175,),
    batch_size: int = 512,
    topk: int = 64,
    max_positives: int = 8,
    rank: int = 16,
    metric_residual: float = 0.05,
    learning_rate: float = 2e-4,
    temperature: float = 0.04,
    harmful_weight: float = 0.1,
    trust_weight: float = 1.0,
    group_dro_eta: float = 0.03,
    refresh_interval: int = 0,
    refresh_shards: int = 7,
    ransac_reprojection_px: float = 12.0,
    clean_reprojection_px: float = 4.0,
    seed: int = 2026,
) -> dict:
    torch.manual_seed(int(seed))
    device = torch.device("cuda")
    state = torch.load(map_path, map_location="cpu", weights_only=False)
    graph = torch.load(function_graph_path, map_location="cpu", weights_only=False)
    payload = torch.load(track_payload_path, map_location="cpu", weights_only=False)
    cache_payload = torch.load(query_cache_path, map_location="cpu", weights_only=False)
    teacher = torch.load(positive_teacher_path, map_location="cpu", weights_only=False)
    cache = cache_payload.get("queries", cache_payload)
    names = list(graph["query_names"])
    records, data_report = _build_training_records(
        graph, payload, state, teacher, max_positives
    )
    raw_features = F.normalize(
        torch.as_tensor(state["anchor_features"]).float(), dim=1
    ).to(device)
    metric = SharedLowRankMetric(
        descriptor_dim=raw_features.shape[1],
        rank=rank,
        max_residual_norm=metric_residual,
    ).to(device)
    optimizer = torch.optim.AdamW(
        metric.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    groups = torch.as_tensor(data_report["query_groups"]).long()
    group_count = int(groups.max()) + 1
    group_weights = torch.ones(group_count, device=device) / group_count
    clean_pairs: dict = {}
    harmful_pairs: dict = {}
    generator = torch.Generator().manual_seed(int(seed) + 1)
    shards = _build_rotating_shards(groups, refresh_shards)
    refresh_index = 0
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    config = {
        "steps": int(steps),
        "batch_size": int(batch_size),
        "topk": int(topk),
        "max_positives": int(max_positives),
        "rank": int(rank),
        "metric_residual": float(metric_residual),
        "learning_rate": float(learning_rate),
        "temperature": float(temperature),
        "harmful_weight": float(harmful_weight),
        "trust_weight": float(trust_weight),
        "group_dro_eta": float(group_dro_eta),
        "refresh_interval": int(refresh_interval),
        "refresh_shards": int(refresh_shards),
        "ransac_reprojection_px": float(ransac_reprojection_px),
        "clean_reprojection_px": float(clean_reprojection_px),
        "seed": int(seed),
        **data_report,
    }
    checkpoints = set(int(value) for value in checkpoint_steps)
    for step in range(1, int(steps) + 1):
        if step == 1 or (
            refresh_interval > 0 and (step - 1) % int(refresh_interval) == 0
        ):
            shard_index = refresh_index % len(shards)
            harmful_prior, group_risks, outcomes, clean, harmful = (
                _refresh_ransac_outcomes(
                    metric=metric,
                    raw_features=raw_features,
                    state=state,
                    cache=cache,
                    names=names,
                    groups=groups,
                    training_records=records,
                    device=device,
                    query_indices=shards[shard_index],
                    seed=seed,
                    ransac_reprojection_px=ransac_reprojection_px,
                    clean_reprojection_px=clean_reprojection_px,
                )
            )
            churn = _replace_refreshed_pairs(
                clean_pairs, harmful_pairs, shards[shard_index], clean, harmful
            )
            risk = torch.zeros_like(group_weights)
            for group, value in group_risks.items():
                risk[group] = float(value)
            group_weights *= torch.exp(float(group_dro_eta) * risk)
            group_weights /= group_weights.sum().clamp_min(1e-8)
            row = {
                "step": step - 1,
                "event": "self_localization_refresh",
                "shard": shard_index,
                "shard_query_count": len(shards[shard_index]),
                "mean_te_cm": float(np.mean([value["te_cm"] for value in outcomes])),
                "mean_candidate_count": float(
                    np.mean([value["candidate_count"] for value in outcomes])
                ),
                "mean_hypotheses": float(
                    np.mean(
                        [
                            value["hypotheses"]
                            for value in outcomes
                            if value["hypotheses"] is not None
                        ]
                    )
                ),
                "harmful_anchor_fraction": float(
                    (harmful_prior > 0).float().mean()
                ),
                "group_weight_max": float(group_weights.max()),
                **churn,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            refresh_index += 1

        query_index = int(torch.randint(len(records), (1,), generator=generator))
        record = records[query_index]
        count = int(record["cache_rows"].numel())
        if count == 0:
            continue
        rows = torch.randint(
            count, (min(int(batch_size), count),), generator=generator
        )
        cache_rows = record["cache_rows"][rows]
        query = F.normalize(
            torch.as_tensor(cache[names[query_index]]["native_descriptors"])
            .float()[cache_rows],
            dim=1,
        ).to(device)
        positives = record["positives"][rows].to(device)
        ignored = record["ignored_anchors"][rows].to(device)
        matchable = record["matchable"][rows].to(device)

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
                    (positives.shape[0],), positives.shape[1] - 1, device=device
                ),
            )
            positives[
                torch.arange(positives.shape[0], device=device)[add_clean],
                replace[add_clean],
            ] = clean_survivors[add_clean]
            matchable |= add_clean
        current_harmful = harmful_pairs.get(query_index, {})
        harmful_survivors = torch.as_tensor(
            [current_harmful.get(int(row), -1) for row in cache_rows],
            dtype=torch.long,
            device=device,
        )[:, None]
        harmful_survivors = torch.where(
            (
                (harmful_survivors == ignored) & (ignored >= 0)
            ).any(dim=1, keepdim=True),
            torch.full_like(harmful_survivors, -1),
            harmful_survivors,
        )

        adapted_query, query_residual = metric(query)
        adapted_anchor, anchor_residual = metric(raw_features)
        list_loss = torch.zeros(adapted_query.shape[0], device=device)
        if bool(matchable.any()):
            list_loss[matchable] = _multi_positive_list_loss(
                adapted_query[matchable],
                adapted_anchor,
                positives[matchable],
                ignored[matchable],
                harmful_survivors[matchable],
                topk=topk,
                temperature=temperature,
                harmful_weight=harmful_weight,
            )
        if bool(matchable.any()):
            row_weights = torch.ones_like(list_loss[matchable])
            task_loss = (
                (list_loss[matchable] * row_weights).sum()
                / row_weights.sum().clamp_min(1e-8)
            )
        else:
            task_loss = torch.zeros((), device=device)
        task_loss *= group_weights[int(record["group"])] * float(group_count)
        trust_loss = (
            query_residual.square().sum(dim=1).mean()
            + anchor_residual.square().sum(dim=1).mean()
            + torch.zeros((), device=device)
        )
        loss = task_loss + float(trust_weight) * trust_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(metric.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == steps:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "task_loss": float(task_loss.detach()),
                "trust_loss": float(trust_loss.detach()),
                "matchable_fraction": float(matchable.float().mean()),
                "group": int(record["group"]),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
        if step in checkpoints or step == steps:
            _save_checkpoint(
                output_dir,
                step,
                state,
                metric,
                raw_features,
                history,
                config,
            )
    report = {
        "schema": "lafgs_self_localization_training",
        "version": 1,
        "config": config,
        "history": history,
    }
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--function-graph", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--complete-positive-teacher", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=175)
    parser.add_argument("--checkpoint-steps", default="175")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--max-positives", type=int, default=8)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--metric-residual", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--temperature", type=float, default=0.04)
    parser.add_argument("--harmful-weight", type=float, default=0.1)
    parser.add_argument("--trust-weight", type=float, default=1.0)
    parser.add_argument("--group-dro-eta", type=float, default=0.03)
    parser.add_argument("--refresh-interval", type=int, default=0)
    parser.add_argument("--refresh-shards", type=int, default=7)
    parser.add_argument("--ransac-reprojection-px", type=float, default=12.0)
    parser.add_argument("--clean-reprojection-px", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    train(
        map_path=args.map,
        function_graph_path=args.function_graph,
        track_payload_path=args.track_payload,
        query_cache_path=args.query_cache,
        positive_teacher_path=args.complete_positive_teacher,
        output_dir=args.output_dir,
        steps=args.steps,
        checkpoint_steps=tuple(
            int(value) for value in args.checkpoint_steps.split(",") if value
        ),
        batch_size=args.batch_size,
        topk=args.topk,
        max_positives=args.max_positives,
        rank=args.rank,
        metric_residual=args.metric_residual,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        harmful_weight=args.harmful_weight,
        trust_weight=args.trust_weight,
        group_dro_eta=args.group_dro_eta,
        refresh_interval=args.refresh_interval,
        refresh_shards=args.refresh_shards,
        ransac_reprojection_px=args.ransac_reprojection_px,
        clean_reprojection_px=args.clean_reprojection_px,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
