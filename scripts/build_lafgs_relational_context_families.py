#!/usr/bin/env python3
"""Build view-conditioned relation families from GT-clean mapping evidence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from localization_training.contextual_descriptor import (
    BoundedContextProjector,
)
from localization_training.relational_context import (
    relational_sparse_query_context,
)
from localization_training.shared_metric import SharedLowRankMetric


def _load_metric(path: str, device: torch.device) -> SharedLowRankMetric:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metric = SharedLowRankMetric(**payload["metric_config"]).to(device)
    metric.load_state_dict(payload["metric_state_dict"])
    metric.eval()
    return metric


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--context-state", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--complete-positive-teacher", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--confusion-graph", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-observations", type=int, default=2)
    parser.add_argument("--map-blend", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    context_state = torch.load(
        args.context_state, map_location="cpu", weights_only=False
    )
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    teacher = torch.load(
        args.complete_positive_teacher,
        map_location="cpu",
        weights_only=False,
    )
    track = torch.load(
        args.track_payload, map_location="cpu", weights_only=False
    )
    graph = torch.load(
        args.confusion_graph, map_location="cpu", weights_only=False
    )
    if (
        context_state["config"].get("query_representation")
        != "relational_sparse_2d_v1"
    ):
        raise ValueError("context families require relational context")
    if not torch.equal(
        torch.as_tensor(context_state["anchor_ids"]).long(),
        torch.as_tensor(state["anchor_ids"]).long(),
    ):
        raise ValueError("context state does not align with active map")
    teacher_by_name = {
        str(record["query_name"]): record
        for record in teacher["records"]
    }
    query_bins = {
        str(name): int(view_bin)
        for name, view_bin in zip(
            track["query_names"],
            torch.as_tensor(track["query_bins"]).tolist(),
        )
    }
    confusion_anchors = {
        int(edge[key])
        for edge in graph["edges"]
        for key in ("correct_anchor", "confusing_anchor")
    }
    metric = _load_metric(args.metric_state, device)
    projector = BoundedContextProjector(
        **context_state["query_projector_config"]
    ).to(device)
    projector.load_state_dict(
        context_state["query_projector_state_dict"]
    )
    projector.eval()
    map_context = F.normalize(
        torch.as_tensor(context_state["anchor_context"]).float(),
        dim=1,
    )
    sums: dict[tuple[int, int], torch.Tensor] = {}
    counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    config = context_state["config"]
    with torch.no_grad():
        for name in tqdm(
            teacher["query_names"], desc="View-conditioned context"
        ):
            if name not in query_bins:
                raise ValueError(f"track payload misses query {name}")
            record = teacher_by_name[name]
            rows = torch.as_tensor(record["query_rows"]).long()
            offsets = torch.as_tensor(record["positive_offsets"]).long()
            positives = torch.as_tensor(record["positive_indices"]).long()
            valid_slots = torch.nonzero(
                offsets[1:] > offsets[:-1], as_tuple=False
            ).reshape(-1)
            if not valid_slots.numel():
                continue
            cached = cache[name]
            descriptors = F.normalize(
                torch.as_tensor(
                    cached["native_descriptors"]
                ).float().to(device),
                dim=1,
            )
            descriptors, _ = metric(descriptors)
            relation = relational_sparse_query_context(
                descriptors,
                torch.as_tensor(
                    cached["native_keypoints"]
                ).float().to(device),
                torch.as_tensor(
                    cached["native_scores"]
                ).float().to(device),
                neighbor_count=int(config["query_neighbor_count"]),
                chunk_size=int(config.get("context_chunk_size", 256)),
            )
            embedding, _ = projector(relation[rows[valid_slots].to(device)])
            embedding = embedding.cpu()
            for local_slot, teacher_slot in enumerate(valid_slots.tolist()):
                anchors = positives[
                    offsets[teacher_slot] : offsets[teacher_slot + 1]
                ]
                for anchor in anchors.tolist():
                    anchor = int(anchor)
                    if anchor not in confusion_anchors:
                        continue
                    key = (anchor, query_bins[name])
                    if key in sums:
                        sums[key] += embedding[local_slot]
                    else:
                        sums[key] = embedding[local_slot].clone()
                    counts[key] += 1

    parents = []
    bins = []
    prototype_context = []
    blend = min(max(float(args.map_blend), 0.0), 1.0)
    for key in sorted(sums):
        if counts[key] < int(args.minimum_observations):
            continue
        anchor, view_bin = key
        observed = F.normalize(sums[key], dim=0)
        prototype_context.append(
            F.normalize(
                blend * map_context[anchor] + (1.0 - blend) * observed,
                dim=0,
            )
        )
        parents.append(anchor)
        bins.append(view_bin)
    output = dict(context_state)
    output["version"] = 4
    output["prototype_context"] = torch.stack(prototype_context).half()
    output["prototype_anchor_indices"] = torch.as_tensor(parents).long()
    output["prototype_view_bins"] = torch.as_tensor(bins).long()
    output["prototype_observation_counts"] = torch.as_tensor(
        [counts[key] for key in sorted(sums) if counts[key] >= args.minimum_observations]
    ).long()
    output["config"] = {
        **context_state["config"],
        "view_conditioning": "gt_clean_query_relation_family_v1",
        "prototype_map_blend": blend,
        "prototype_minimum_observations": int(args.minimum_observations),
        "prototype_scope": "confusion_graph_anchors_only",
    }
    output["provenance"] = {
        **context_state.get("provenance", {}),
        "family_builder": {
            "query_cache": str(Path(args.query_cache).resolve()),
            "complete_positive_teacher": str(
                Path(args.complete_positive_teacher).resolve()
            ),
            "track_payload": str(Path(args.track_payload).resolve()),
            "confusion_graph": str(Path(args.confusion_graph).resolve()),
        },
    }
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, path)
    summary = {
        "output": str(path),
        "prototype_count": len(parents),
        "prototype_anchor_count": len(set(parents)),
        "map_blend": blend,
        "minimum_observations": int(args.minimum_observations),
    }
    path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(summary)


if __name__ == "__main__":
    main()
