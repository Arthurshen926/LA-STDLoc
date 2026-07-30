#!/usr/bin/env python3
"""Build exact deployment top-K assignments without running pose estimation."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from localization_training.contextual_descriptor import (
    BoundedContextProjector,
    flatten_context,
    fuse_local_and_context,
    multiscale_sparse_query_context,
)
from localization_training.shared_metric import SharedLowRankMetric
from localization_training.relational_context import (
    relational_sparse_query_context,
)


def _sha256_tensor(value: torch.Tensor) -> str:
    value = torch.as_tensor(value).detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _load_metric(path: str, device: torch.device) -> SharedLowRankMetric:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metric = SharedLowRankMetric(**payload["metric_config"]).to(device)
    metric.load_state_dict(payload["metric_state_dict"])
    metric.eval()
    return metric


def _atomic_torch(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--function-graph", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--context-state", default="")
    parser.add_argument("--context-weight", type=float, default=0.0)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    graph = torch.load(
        args.function_graph, map_location="cpu", weights_only=False
    )
    names = [str(value) for value in graph["query_names"]]
    start = int(args.query_start)
    if start < 0 or start >= len(names):
        raise ValueError("query_start is outside the function graph")
    names = names[start:]
    if args.query_limit > 0:
        names = names[: int(args.query_limit)]
    bank = F.normalize(
        torch.as_tensor(state["anchor_features"]).float().to(device), dim=1
    )
    metric = _load_metric(args.metric_state, device)
    topk = min(max(int(args.topk), 1), len(bank))

    context_state = None
    context_map = None
    context_config = None
    query_projector = None
    if args.context_state:
        if float(args.context_weight) <= 0:
            raise ValueError("context state requires a positive weight")
        context_state = torch.load(
            args.context_state, map_location="cpu", weights_only=False
        )
        if context_state.get("schema") != "lafgs_fixed_3d_context_state":
            raise ValueError("unsupported context state")
        if not torch.equal(
            torch.as_tensor(context_state["anchor_ids"]).long(),
            torch.as_tensor(state["anchor_ids"]).long(),
        ):
            raise ValueError("context state does not align with map")
        context_map = F.normalize(
            torch.as_tensor(context_state["anchor_context"]).float(), dim=1
        ).to(device)
        context_config = dict(context_state["config"])
        if "query_projector_state_dict" in context_state:
            query_projector = BoundedContextProjector(
                **context_state["query_projector_config"]
            ).to(device)
            query_projector.load_state_dict(
                context_state["query_projector_state_dict"]
            )
            query_projector.eval()
        bank = fuse_local_and_context(
            bank,
            context_map,
            context_weight=float(args.context_weight),
        )

    graph_by_name = {
        str(name): record
        for name, record in zip(graph["query_names"], graph["records"])
    }
    records = []
    with torch.inference_mode():
        for name in tqdm(names, desc="Exact top-K"):
            cached = cache[name]
            rows = torch.as_tensor(
                graph_by_name[name]["query_rows"]
            ).long()
            all_descriptors = F.normalize(
                torch.as_tensor(
                    cached["native_descriptors"]
                ).float().to(device),
                dim=1,
            )
            query, _ = metric(all_descriptors[rows.to(device)])
            if context_state is not None:
                if (
                    context_config.get("query_representation")
                    == "relational_sparse_2d_v1"
                ):
                    transformed_all, _ = metric(all_descriptors)
                    query_context = relational_sparse_query_context(
                        transformed_all,
                        torch.as_tensor(
                            cached["native_keypoints"]
                        ).float().to(device),
                        torch.as_tensor(
                            cached["native_scores"]
                        ).float().to(device),
                        neighbor_count=int(
                            context_config["query_neighbor_count"]
                        ),
                        chunk_size=int(
                            context_config.get("context_chunk_size", 256)
                        ),
                    )[rows.to(device)]
                else:
                    query_context = multiscale_sparse_query_context(
                        all_descriptors,
                        torch.as_tensor(
                            cached["native_keypoints"]
                        ).float().to(device),
                        torch.as_tensor(
                            cached["native_scores"]
                        ).float().to(device),
                        radii_px=tuple(
                            float(value)
                            for value in context_config["sparse_radii_px"]
                        ),
                        maximum_neighbors=int(
                            context_config["maximum_sparse_neighbors"]
                        ),
                        chunk_size=int(
                            context_config.get("context_chunk_size", 256)
                        ),
                    )
                    shape = query_context.shape
                    transformed, _ = metric(
                        query_context.reshape(-1, shape[-1])
                    )
                    query_context = flatten_context(
                        transformed.reshape(shape)
                    )[rows.to(device)]
                if query_projector is not None:
                    query_context, _ = query_projector(query_context)
                query = fuse_local_and_context(
                    query,
                    query_context,
                    context_weight=float(args.context_weight),
                )
            scores, indices = torch.topk(
                query @ bank.T, k=topk, dim=1
            )
            records.append(
                {
                    "query_name": name,
                    "query_rows": rows.cpu(),
                    "topk_anchor_indices": indices.cpu(),
                    "topk_scores": scores.cpu(),
                }
            )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch(
        output,
        {
            "schema": "lafgs_exact_topk_outcomes",
            "version": 1,
            "query_names": names,
            "query_start": start,
            "topk": topk,
            "anchor_count": len(bank),
            "anchor_ids_sha256": _sha256_tensor(state["anchor_ids"]),
            "records": records,
            "provenance": {
                "map": str(Path(args.map).resolve()),
                "metric_state": str(Path(args.metric_state).resolve()),
                "query_cache": str(Path(args.query_cache).resolve()),
                "function_graph": str(Path(args.function_graph).resolve()),
                "context_state": str(Path(args.context_state).resolve())
                if args.context_state
                else None,
                "context_weight": float(args.context_weight),
            },
        },
    )
    print(
        {
            "output": str(output),
            "query_count": len(records),
            "topk": topk,
            "context": context_state is not None,
        }
    )


if __name__ == "__main__":
    main()
