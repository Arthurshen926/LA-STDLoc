#!/usr/bin/env python3
"""Distill OOF view-conditioned 2D relation prototypes onto 3D anchors."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from localization_training.contextual_descriptor import BoundedContextProjector
from localization_training.relational_context import (
    relational_sparse_query_context,
)
from localization_training.shared_metric import SharedLowRankMetric


def _sha256_tensor(value: torch.Tensor) -> str:
    value = torch.as_tensor(value).detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _trajectory(name: str) -> str:
    return str(name).split("/", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--complete-positive-teacher", required=True)
    parser.add_argument("--context-state", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--view-bin-count", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    metric_payload = torch.load(
        args.metric_state, map_location="cpu", weights_only=False
    )
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    teacher = torch.load(
        args.complete_positive_teacher, map_location="cpu", weights_only=False
    )
    context_state = torch.load(
        args.context_state, map_location="cpu", weights_only=False
    )
    if context_state.get("schema") != "lafgs_fixed_3d_context_state":
        raise ValueError("unsupported context state")
    if context_state["config"].get(
        "query_representation"
    ) != "relational_sparse_2d_v1":
        raise ValueError("observed families require relational query context")
    anchor_ids = torch.as_tensor(state["anchor_ids"]).long()
    if not torch.equal(
        torch.as_tensor(context_state["anchor_ids"]).long(), anchor_ids
    ):
        raise ValueError("context state does not align with map")

    metric = SharedLowRankMetric(**metric_payload["metric_config"]).to(device)
    metric.load_state_dict(metric_payload["metric_state_dict"])
    metric.eval()
    projector = BoundedContextProjector(
        **context_state["query_projector_config"]
    ).to(device)
    projector.load_state_dict(context_state["query_projector_state_dict"])
    projector.eval()
    config = context_state["config"]
    trajectories = sorted(
        {_trajectory(name) for name in teacher["query_names"]}
    )
    trajectory_index = {
        trajectory: index
        for index, trajectory in enumerate(trajectories)
    }
    anchor_count = len(anchor_ids)
    view_bin_count = max(int(args.view_bin_count), 1)
    context_dim = int(context_state["context_dim"])
    anchor_xyz = torch.as_tensor(state["anchor_xyz"]).float().to(device)
    sums = torch.zeros(
        len(trajectories),
        view_bin_count,
        anchor_count,
        context_dim,
        device=device,
    )
    counts = torch.zeros(
        len(trajectories), view_bin_count, anchor_count, device=device
    )

    with torch.inference_mode():
        for record in tqdm(
            teacher["records"], desc="Observed context families"
        ):
            name = str(record["query_name"])
            cached = cache[name]
            rows = torch.as_tensor(record["query_rows"]).long()
            offsets = torch.as_tensor(record["positive_offsets"]).long()
            positives = torch.as_tensor(record["positive_indices"]).long()
            edge_counts = offsets[1:] - offsets[:-1]
            valid = edge_counts > 0
            if not bool(valid.any()):
                continue
            all_descriptors = F.normalize(
                torch.as_tensor(cached["native_descriptors"]).float().to(device),
                dim=1,
            )
            transformed, _ = metric(all_descriptors)
            query_context = relational_sparse_query_context(
                transformed,
                torch.as_tensor(cached["native_keypoints"]).float().to(device),
                torch.as_tensor(cached["native_scores"]).float().to(device),
                neighbor_count=int(config["query_neighbor_count"]),
                chunk_size=int(config.get("context_chunk_size", 256)),
            )[rows.to(device)]
            query_context, _ = projector(query_context)
            repeated_context = torch.repeat_interleave(
                query_context[valid.to(device)],
                edge_counts[valid].to(device),
                dim=0,
            )
            retained_positive = positives[
                torch.repeat_interleave(
                    valid, edge_counts
                )
            ].to(device)
            trajectory = trajectory_index[_trajectory(name)]
            pose = torch.as_tensor(cached["pose_w2c"]).float().to(device)
            camera_center = -(pose[:3, :3].T @ pose[:3, 3])
            viewing_direction = (
                anchor_xyz[retained_positive] - camera_center
            )
            azimuth = torch.atan2(
                viewing_direction[:, 1], viewing_direction[:, 0]
            )
            view_bins = torch.floor(
                torch.remainder(azimuth, 2.0 * torch.pi)
                * view_bin_count
                / (2.0 * torch.pi)
            ).long().clamp(0, view_bin_count - 1)
            flattened = view_bins * anchor_count + retained_positive
            sums[trajectory].reshape(
                view_bin_count * anchor_count, context_dim
            ).index_add_(0, flattened, repeated_context)
            counts[trajectory].reshape(-1).index_add_(
                0,
                flattened,
                torch.ones(
                    len(retained_positive), device=device
                ),
            )

    prototypes = F.normalize(
        sums / counts.clamp_min(1.0)[..., None], dim=3
    )
    prototypes[counts == 0] = 0.0
    output = {
        "schema": "lafgs_observed_context_families",
        "version": 1,
        "anchor_ids": anchor_ids,
        "anchor_ids_sha256": _sha256_tensor(anchor_ids),
        "trajectories": trajectories,
        "prototype_context": prototypes.half().cpu(),
        "observation_counts": counts.int().cpu(),
        "context_dim": context_dim,
        "config": {
            "representation": "real_observation_projected_2d_relation_v1",
            "fold_contract": "exclude_query_trajectory_at_replay",
            "view_bin_count": view_bin_count,
            "view_bin_frame": "world_azimuth_family_max_no_pose_at_replay",
            "query_representation": "relational_sparse_2d_v1",
            "query_neighbor_count": int(config["query_neighbor_count"]),
            "context_chunk_size": int(
                config.get("context_chunk_size", 256)
            ),
            "minimum_observations": 1,
        },
        "summary": {
            "trajectory_count": len(trajectories),
            "view_bin_count": view_bin_count,
            "observed_anchor_view_pairs": int((counts > 0).sum()),
            "observed_anchor_trajectory_pairs": int(
                (counts.sum(dim=1) > 0).sum()
            ),
            "observed_anchor_count": int(
                (counts.sum(dim=(0, 1)) > 0).sum()
            ),
            "observation_count": int(counts.sum()),
        },
        "provenance": {
            "map": str(Path(args.map).resolve()),
            "metric_state": str(Path(args.metric_state).resolve()),
            "query_cache": str(Path(args.query_cache).resolve()),
            "complete_positive_teacher": str(
                Path(args.complete_positive_teacher).resolve()
            ),
            "query_projector_context_state": str(
                Path(args.context_state).resolve()
            ),
        },
    }
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(output, temporary)
    os.replace(temporary, path)
    print({"output": str(path), **output["summary"]})


if __name__ == "__main__":
    main()
