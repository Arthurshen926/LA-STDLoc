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
from localization_training.full_primitive_retrieval import (
    chunked_exact_topk_family_prototype,
)
from localization_training.shared_metric import SharedLowRankMetric
from localization_training.relational_context import (
    relational_sparse_query_context,
)


def _sha256_tensor(value: torch.Tensor) -> str:
    value = torch.as_tensor(value).detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_metric_contract(
    *,
    state: dict,
    metric_payload: dict,
    metric_path: str,
) -> None:
    contract = state.get("metric_state_contract")
    if contract:
        expected = contract.get("sha256")
        actual = _sha256_file(metric_path)
        if not expected or expected != actual:
            raise ValueError(
                "metric state does not match the materialized map contract"
            )
        return

    # Legacy V7+ maps predate explicit hashes but retain the metric output
    # directory. Reject a different experiment state while allowing older
    # maps without either form of provenance.
    metric_metadata = state.get("v7_online_metric", {})
    expected_dir = metric_metadata.get("config", {}).get("output_dir")
    source_map = metric_payload.get("map_path")
    if expected_dir and source_map:
        expected_dir = Path(expected_dir).resolve()
        source_dir = Path(source_map).resolve().parent
        metric_dir = Path(metric_path).resolve().parent
        if source_dir != expected_dir or metric_dir != expected_dir:
            raise ValueError(
                "metric state provenance does not match the materialized map"
            )


def _load_family_prototypes(
    path: str,
    *,
    state: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "lafgs_basin_family_prototypes":
        raise ValueError("unsupported family prototype state")
    if not torch.equal(
        torch.as_tensor(payload["landmark_indices"]).long().reshape(-1),
        torch.as_tensor(state["anchor_ids"]).long().reshape(-1),
    ):
        raise ValueError("family prototype state does not align with the map")
    features = torch.as_tensor(payload["prototype_features"]).float()
    parents = torch.as_tensor(
        payload["prototype_anchor_indices"]
    ).long().reshape(-1)
    descriptor_dim = int(
        torch.as_tensor(state["anchor_features"]).shape[1]
    )
    if (
        features.ndim != 2
        or features.shape[0] != parents.numel()
        or features.shape[1] != descriptor_dim
    ):
        raise ValueError("family prototype descriptors are malformed")
    if parents.numel() and (
        int(parents.min()) < 0
        or int(parents.max()) >= len(state["anchor_ids"])
    ):
        raise ValueError("family prototype parent is outside the map")
    bias = torch.as_tensor(
        payload.get("prototype_bias", torch.zeros(len(features)))
    ).float().reshape(-1)
    temperature = torch.as_tensor(
        payload.get("prototype_temperature", torch.ones(len(features)))
    ).float().reshape(-1)
    if bias.numel() != len(features) or bool((bias > 1e-8).any()):
        raise ValueError("family prototype bias is malformed")
    if temperature.numel() != len(features) or bool((temperature <= 0).any()):
        raise ValueError("family prototype temperature is malformed")
    return (
        features.to(device),
        parents.to(device),
        bias.to(device),
        temperature.to(device),
    )


def _load_metric(
    path: str,
    device: torch.device,
    *,
    state: dict,
) -> SharedLowRankMetric:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _verify_metric_contract(
        state=state,
        metric_payload=payload,
        metric_path=path,
    )
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
    parser.add_argument("--family-prototype-state", default="")
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
    metric = _load_metric(args.metric_state, device, state=state)
    topk = min(max(int(args.topk), 1), len(bank))

    context_state = None
    context_map = None
    context_config = None
    query_projector = None
    family_prototypes = None
    if args.family_prototype_state:
        if args.context_state:
            raise ValueError(
                "family prototypes and fused global context are mutually exclusive"
            )
        family_prototypes = _load_family_prototypes(
            args.family_prototype_state,
            state=state,
            device=device,
        )
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
            if family_prototypes is None:
                scores, indices = torch.topk(
                    query @ bank.T, k=topk, dim=1
                )
            else:
                retrieval = chunked_exact_topk_family_prototype(
                    query,
                    bank,
                    family_prototypes[0],
                    family_prototypes[1],
                    prototype_bias=family_prototypes[2],
                    prototype_temperature=family_prototypes[3],
                    topk=topk,
                )
                scores, indices = retrieval.scores, retrieval.indices
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
                "map_sha256": _sha256_file(args.map),
                "metric_state": str(Path(args.metric_state).resolve()),
                "metric_state_sha256": _sha256_file(args.metric_state),
                "query_cache": str(Path(args.query_cache).resolve()),
                "function_graph": str(Path(args.function_graph).resolve()),
                "context_state": str(Path(args.context_state).resolve())
                if args.context_state
                else None,
                "context_state_sha256": (
                    _sha256_file(args.context_state)
                    if args.context_state
                    else None
                ),
                "context_weight": float(args.context_weight),
                "family_prototype_state": (
                    str(Path(args.family_prototype_state).resolve())
                    if args.family_prototype_state
                    else None
                ),
                "family_prototype_state_sha256": (
                    _sha256_file(args.family_prototype_state)
                    if args.family_prototype_state
                    else None
                ),
            },
        },
    )
    print(
        {
            "output": str(output),
            "query_count": len(records),
            "topk": topk,
            "context": context_state is not None,
            "family_prototypes": family_prototypes is not None,
        }
    )


if __name__ == "__main__":
    main()
