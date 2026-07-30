#!/usr/bin/env python3
"""Audit real-only 2D/3D context separability on repeated assignments."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from encoders.sp_encoder.export_image_embeddings import SuperPoint
from localization_training.confusion_evidence import family_pair_scores
from localization_training.contextual_descriptor import (
    flatten_context,
    multiscale_dense_query_context,
    multiscale_map_3d_context,
    multiscale_sparse_query_context,
)
from localization_training.shared_metric import SharedLowRankMetric


def _trajectory(name: str) -> str:
    return str(name).split("/", 1)[0]


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _file_identity(path: str) -> dict:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _load_metric(path: str, device: torch.device) -> SharedLowRankMetric:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metric = SharedLowRankMetric(**payload["metric_config"]).to(device)
    metric.load_state_dict(payload["metric_state_dict"])
    metric.eval()
    for parameter in metric.parameters():
        parameter.requires_grad_(False)
    return metric


def _apply_metric(
    context: torch.Tensor, metric: SharedLowRankMetric
) -> torch.Tensor:
    shape = context.shape
    flat = context.reshape(-1, shape[-1])
    transformed, _ = metric(flat)
    return F.normalize(transformed, dim=1).reshape(shape)


def _selected_graph(
    graph: dict,
    *,
    maximum_edges: int,
    maximum_events_per_edge: int,
) -> tuple[list[dict], list[dict]]:
    edges = sorted(
        graph["edges"],
        key=lambda value: (
            -float(value["weight"]),
            -int(value["occurrences"]),
            int(value["edge_index"]),
        ),
    )[: max(int(maximum_edges), 1)]
    selected_edge_ids = {int(edge["edge_index"]) for edge in edges}
    by_edge: dict[int, list[dict]] = defaultdict(list)
    for event in graph["events"]:
        edge_index = int(event["edge_index"])
        if edge_index in selected_edge_ids:
            by_edge[edge_index].append(event)
    events = []
    for edge in edges:
        edge_events = sorted(
            by_edge[int(edge["edge_index"])],
            key=lambda value: (
                -float(value["pose_blame"]),
                -float(value["score_margin"]),
                str(value["query_name"]),
                int(value["query_row"]),
            ),
        )
        events.extend(edge_events[: max(int(maximum_events_per_edge), 1)])
    return edges, events


def _target_support_rows(
    teacher: dict,
    target_anchors: set[int],
    event_rows: dict[str, set[int]],
) -> tuple[dict[str, dict[int, list[int]]], dict[str, set[int]]]:
    target_lookup = torch.zeros(
        int(teacher["anchor_count"]), dtype=torch.bool
    )
    target_lookup[torch.as_tensor(sorted(target_anchors)).long()] = True
    support: dict[str, dict[int, list[int]]] = {}
    required = {name: set(rows) for name, rows in event_rows.items()}
    for record in teacher["records"]:
        name = str(record["query_name"])
        rows = torch.as_tensor(record["query_rows"]).long()
        offsets = torch.as_tensor(record["positive_offsets"]).long()
        positives = torch.as_tensor(record["positive_indices"]).long()
        if not positives.numel():
            continue
        selected = target_lookup[positives]
        if not bool(selected.any()):
            continue
        slots = torch.repeat_interleave(
            torch.arange(rows.numel()), offsets[1:] - offsets[:-1]
        )[selected]
        anchors = positives[selected]
        by_row: dict[int, list[int]] = defaultdict(list)
        for slot, anchor in zip(slots.tolist(), anchors.tolist()):
            by_row[int(rows[slot])].append(int(anchor))
        support[name] = dict(by_row)
        required.setdefault(name, set()).update(by_row)
    return support, required


def _context_cache_identity(args, target_anchors: set[int]) -> dict:
    digest = hashlib.sha256(
        ",".join(str(value) for value in sorted(target_anchors)).encode()
    ).hexdigest()
    return {
        "schema": "lafgs_real_context_rows",
        "version": 1,
        "mode": str(args.mode),
        "query_cache": _file_identity(args.query_cache),
        "metric_state": _file_identity(args.metric_state),
        "images_root": str(Path(args.images_root).resolve())
        if args.images_root
        else None,
        "sparse_radii_px": list(args.sparse_radii_px),
        "dense_radii_cells": list(args.dense_radii_cells),
        "maximum_sparse_neighbors": int(args.maximum_sparse_neighbors),
        "target_anchor_sha256": digest,
    }


def _build_context_rows(
    *,
    args,
    cache: dict,
    required_rows: dict[str, set[int]],
    metric: SharedLowRankMetric,
    target_anchors: set[int],
    device: torch.device,
) -> dict:
    identity = _context_cache_identity(args, target_anchors)
    cache_path = Path(args.context_cache).resolve()
    if cache_path.is_file():
        payload = torch.load(
            cache_path, map_location="cpu", weights_only=False
        )
        if payload.get("identity") != identity:
            raise ValueError("context cache identity does not match this run")
        return payload

    use_sparse = args.mode in {"sparse", "sparse_dense"}
    use_dense = args.mode in {"dense", "sparse_dense"}
    extractor = SuperPoint().to(device).eval() if use_dense else None
    output_rows = {}
    parity_cosines = []
    names = sorted(required_rows)
    for name in tqdm(names, desc=f"{args.mode} context"):
        cached = cache[name]
        selected_rows = torch.as_tensor(
            sorted(required_rows[name]), device=device, dtype=torch.long
        )
        if selected_rows.numel() == 0:
            continue
        descriptors = F.normalize(
            torch.as_tensor(cached["native_descriptors"]).float().to(device),
            dim=1,
        )
        keypoints = torch.as_tensor(
            cached["native_keypoints"]
        ).float().to(device)
        scores = torch.as_tensor(cached["native_scores"]).float().to(device)
        parts = []
        sparse_part = None
        if use_sparse:
            sparse_part = multiscale_sparse_query_context(
                descriptors,
                keypoints,
                scores,
                radii_px=args.sparse_radii_px,
                maximum_neighbors=args.maximum_sparse_neighbors,
                chunk_size=args.context_chunk_size,
            )
            sparse_part = _apply_metric(sparse_part, metric)
            parts.append(sparse_part[selected_rows])
        if use_dense:
            image_path = Path(args.images_root) / name
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            image = (
                pil_to_tensor(Image.open(image_path).convert("RGB"))
                .float()
                .div_(255.0)
                .to(device)
            )
            expected_hw = tuple(
                int(value) for value in cached["native_input_hw"]
            )
            if tuple(image.shape[-2:]) != expected_hw:
                raise ValueError(
                    f"processed image size mismatch for {name}: "
                    f"{tuple(image.shape[-2:])} != {expected_hw}"
                )
            dense_map, _ = extractor.detectAndComputeDense(image[None])
            dense_part = multiscale_dense_query_context(
                dense_map,
                keypoints,
                radii_cells=args.dense_radii_cells,
            )
            dense_part = _apply_metric(dense_part, metric)
            parts.append(dense_part[selected_rows])
            dense_local = multiscale_dense_query_context(
                dense_map, keypoints[selected_rows], radii_cells=(0,)
            )[:, 0]
            parity_cosines.extend(
                (
                    F.normalize(dense_local, dim=1)
                    * descriptors[selected_rows]
                )
                .sum(dim=1)
                .detach()
                .cpu()
                .tolist()
            )
        context = F.normalize(
            torch.cat(parts, dim=1).reshape(selected_rows.numel(), -1),
            dim=1,
        )
        sparse_flat = (
            flatten_context(sparse_part[selected_rows])
            if sparse_part is not None
            else None
        )
        output_rows[name] = {
            "rows": selected_rows.cpu(),
            "context": context.detach().cpu().half(),
            "sparse_context": (
                None
                if sparse_flat is None
                else sparse_flat.detach().cpu().half()
            ),
        }
    parity = torch.as_tensor(parity_cosines)
    payload = {
        "identity": identity,
        "rows": output_rows,
        "dense_sparse_parity": {
            "count": int(parity.numel()),
            "p50": float(parity.quantile(0.50)) if parity.numel() else None,
            "p90": float(parity.quantile(0.90)) if parity.numel() else None,
            "p99": float(parity.quantile(0.99)) if parity.numel() else None,
            "minimum": float(parity.min()) if parity.numel() else None,
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch(cache_path, payload)
    return payload


def _row_context_lookup(payload: dict) -> dict[str, dict[int, int]]:
    return {
        name: {
            int(row): index
            for index, row in enumerate(record["rows"].tolist())
        }
        for name, record in payload["rows"].items()
    }


def _anchor_observations(
    *,
    payload: dict,
    support_rows: dict[str, dict[int, list[int]]],
    query_bins: dict[str, int],
) -> dict[int, list[tuple[str, int, torch.Tensor]]]:
    lookup = _row_context_lookup(payload)
    observations: dict[
        int, list[tuple[str, int, torch.Tensor]]
    ] = defaultdict(list)
    for name, by_row in support_rows.items():
        if name not in payload["rows"]:
            continue
        record = payload["rows"][name]
        context = record["context"].float()
        for row, anchors in by_row.items():
            slot = lookup[name].get(int(row))
            if slot is None:
                continue
            value = context[slot]
            for anchor in anchors:
                observations[int(anchor)].append(
                    (_trajectory(name), int(query_bins[name]), value)
                )
    return observations


def _cross_trajectory_prototypes(
    observations: dict[int, list[tuple[str, int, torch.Tensor]]],
    trajectories: set[str],
) -> dict[int, dict[str, tuple[torch.Tensor, torch.Tensor, int]]]:
    """Pre-aggregate leave-one-trajectory-out prototypes once per anchor."""

    output = {}
    trajectory_names = sorted(
        set(trajectories)
        | {
            record[0]
            for anchor_records in observations.values()
            for record in anchor_records
        }
    )
    trajectory_lookup = {
        name: index for index, name in enumerate(trajectory_names)
    }
    for anchor, records in tqdm(
        observations.items(), desc="Cross-trajectory prototypes"
    ):
        record_trajectories = torch.as_tensor(
            [trajectory_lookup[value[0]] for value in records]
        ).long()
        original_bins = torch.as_tensor(
            [value[1] for value in records]
        ).long()
        unique_bins, record_bins = torch.unique(
            original_bins, sorted=True, return_inverse=True
        )
        values = torch.stack([value[2] for value in records]).float()
        dimension = values.shape[1]
        trajectory_sums = values.new_zeros(
            (len(trajectory_names), dimension)
        )
        trajectory_sums.index_add_(0, record_trajectories, values)
        trajectory_counts = torch.bincount(
            record_trajectories, minlength=len(trajectory_names)
        )
        bin_sums = values.new_zeros((len(unique_bins), dimension))
        bin_sums.index_add_(0, record_bins, values)
        bin_counts = torch.bincount(
            record_bins, minlength=len(unique_bins)
        )
        pair_indices = (
            record_trajectories * len(unique_bins) + record_bins
        )
        pair_sums = values.new_zeros(
            (len(trajectory_names) * len(unique_bins), dimension)
        )
        pair_sums.index_add_(0, pair_indices, values)
        pair_sums = pair_sums.reshape(
            len(trajectory_names), len(unique_bins), dimension
        )
        pair_counts = torch.bincount(
            pair_indices,
            minlength=len(trajectory_names) * len(unique_bins),
        ).reshape(len(trajectory_names), len(unique_bins))
        global_sum = values.sum(dim=0)
        global_count = int(values.shape[0])
        per_trajectory = {}
        for excluded, excluded_index in trajectory_lookup.items():
            retained_count = (
                global_count - int(trajectory_counts[excluded_index])
            )
            if retained_count <= 0:
                continue
            static = F.normalize(
                global_sum - trajectory_sums[excluded_index], dim=0
            )
            retained_bin_counts = (
                bin_counts - pair_counts[excluded_index]
            )
            valid_bins = retained_bin_counts > 0
            view_sums = (
                bin_sums[valid_bins]
                - pair_sums[excluded_index, valid_bins]
            )
            view_prototypes = F.normalize(view_sums, dim=1)
            per_trajectory[str(excluded)] = (
                static,
                view_prototypes,
                retained_count,
            )
        output[int(anchor)] = per_trajectory
    return output


def _cross_trajectory_scores(
    query: torch.Tensor,
    prototypes: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor, int]]],
    anchor: int,
    excluded_trajectory: str,
) -> tuple[float | None, float | None, int]:
    payload = prototypes.get(int(anchor), {}).get(str(excluded_trajectory))
    if payload is None:
        return None, None, 0
    static, views, count = payload
    return float(query @ static), float((views @ query).max()), int(count)


def _summarize_method(
    records: list[dict], method: str, baseline: str = "O0_local"
) -> dict:
    valid = [
        record
        for record in records
        if record["margins"].get(method) is not None
    ]
    margins = torch.as_tensor(
        [record["margins"][method] for record in valid]
    )
    baseline_margins = torch.as_tensor(
        [record["margins"][baseline] for record in valid]
    )
    edges: dict[int, list[float]] = defaultdict(list)
    for record in valid:
        edges[int(record["edge_index"])].append(
            float(record["margins"][method])
        )
    edge_positive = [
        sum(value > 0.0 for value in values) > len(values) / 2
        for values in edges.values()
    ]
    return {
        "event_count": len(valid),
        "edge_count": len(edges),
        "median_margin": float(margins.median()) if len(valid) else None,
        "mean_margin": float(margins.mean()) if len(valid) else None,
        "positive_event_percent": (
            float(100 * (margins > 0).float().mean()) if len(valid) else None
        ),
        "positive_edge_percent": (
            float(100 * sum(edge_positive) / len(edge_positive))
            if edge_positive
            else None
        ),
        "beneficial_switch_count": int(
            ((baseline_margins <= 0) & (margins > 0)).sum()
        )
        if len(valid)
        else 0,
        "harmful_switch_count": int(
            ((baseline_margins > 0) & (margins <= 0)).sum()
        )
        if len(valid)
        else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--family-state", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--complete-positive-teacher", required=True)
    parser.add_argument("--confusion-graph", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--context-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--images-root", default="")
    parser.add_argument(
        "--mode",
        choices=("sparse", "dense", "sparse_dense"),
        default="sparse",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum-edges", type=int, default=256)
    parser.add_argument("--maximum-events-per-edge", type=int, default=24)
    parser.add_argument("--maximum-sparse-neighbors", type=int, default=48)
    parser.add_argument("--context-chunk-size", type=int, default=256)
    parser.add_argument("--sparse-radii-px", default="48,96,192")
    parser.add_argument("--dense-radii-cells", default="1,3,7")
    parser.add_argument("--map-neighbor-counts", default="8,24,64")
    parser.add_argument(
        "--context-weights", default="0.025,0.05,0.1,0.2,0.4,0.8"
    )
    args = parser.parse_args()
    # The prototype stage consists of many small reductions. Large OpenMP
    # pools make it substantially slower and do not change the result.
    torch.set_num_threads(1)
    args.sparse_radii_px = tuple(
        float(value) for value in args.sparse_radii_px.split(",")
    )
    args.dense_radii_cells = tuple(
        int(value) for value in args.dense_radii_cells.split(",")
    )
    args.map_neighbor_counts = tuple(
        int(value) for value in args.map_neighbor_counts.split(",")
    )
    context_weights = tuple(
        float(value) for value in args.context_weights.split(",")
    )
    if args.mode in {"dense", "sparse_dense"} and not args.images_root:
        raise ValueError("dense context requires --images-root")

    device = torch.device(args.device)
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    teacher = torch.load(
        args.complete_positive_teacher,
        map_location="cpu",
        weights_only=False,
    )
    graph = torch.load(
        args.confusion_graph, map_location="cpu", weights_only=False
    )
    track = torch.load(
        args.track_payload, map_location="cpu", weights_only=False
    )
    family = torch.load(
        args.family_state, map_location="cpu", weights_only=False
    )
    metric = _load_metric(args.metric_state, device)
    query_bins = {
        str(name): int(value)
        for name, value in zip(
            track["query_names"],
            torch.as_tensor(track["query_bins"]).tolist(),
        )
    }
    edges, events = _selected_graph(
        graph,
        maximum_edges=args.maximum_edges,
        maximum_events_per_edge=args.maximum_events_per_edge,
    )
    target_anchors = {
        int(anchor)
        for edge in edges
        for anchor in (edge["correct_anchor"], edge["confusing_anchor"])
    }
    event_rows: dict[str, set[int]] = defaultdict(set)
    for event in events:
        event_rows[str(event["query_name"])].add(int(event["query_row"]))
    support_rows, required_rows = _target_support_rows(
        teacher, target_anchors, event_rows
    )
    context_payload = _build_context_rows(
        args=args,
        cache=cache,
        required_rows=required_rows,
        metric=metric,
        target_anchors=target_anchors,
        device=device,
    )
    context_lookup = _row_context_lookup(context_payload)
    anchor_observations = _anchor_observations(
        payload=context_payload,
        support_rows=support_rows,
        query_bins=query_bins,
    )
    context_prototypes = _cross_trajectory_prototypes(
        anchor_observations,
        {_trajectory(str(event["query_name"])) for event in events},
    )

    target_list = torch.as_tensor(sorted(target_anchors)).long()
    target_position = {
        int(anchor): index for index, anchor in enumerate(target_list.tolist())
    }
    bank = F.normalize(
        torch.as_tensor(state["anchor_features"]).float(), dim=1
    ).to(device)
    map_context = multiscale_map_3d_context(
        bank,
        torch.as_tensor(state["anchor_xyz"]).float().to(device),
        query_indices=target_list.to(device),
        neighbor_counts=args.map_neighbor_counts,
        chunk_size=args.context_chunk_size,
    )
    map_context = flatten_context(map_context).cpu()

    query_descriptors = []
    correct_anchors = []
    confusing_anchors = []
    retained_events = []
    for event in events:
        name = str(event["query_name"])
        row = int(event["query_row"])
        slot = context_lookup.get(name, {}).get(row)
        if slot is None:
            continue
        local = F.normalize(
            torch.as_tensor(cache[name]["native_descriptors"])[row].float(),
            dim=0,
        ).to(device)
        transformed, _ = metric(local[None])
        query_descriptors.append(transformed[0])
        correct_anchors.append(int(event["correct_anchor"]))
        confusing_anchors.append(int(event["confusing_anchor"]))
        retained_events.append(event)
    query_descriptors = torch.stack(query_descriptors)
    correct_tensor = torch.as_tensor(correct_anchors, device=device)
    confusing_tensor = torch.as_tensor(confusing_anchors, device=device)
    correct_scores, _ = family_pair_scores(
        query_descriptors, correct_tensor, bank, family
    )
    confusing_scores, _ = family_pair_scores(
        query_descriptors, confusing_tensor, bank, family
    )
    base_margins = (correct_scores - confusing_scores).cpu()

    records = []
    graph_margin_delta = []
    for index, event in enumerate(retained_events):
        name = str(event["query_name"])
        row = int(event["query_row"])
        context_slot = context_lookup[name][row]
        query_context = F.normalize(
            context_payload["rows"][name]["context"][context_slot].float(),
            dim=0,
        )
        query_sparse = context_payload["rows"][name].get("sparse_context")
        map_margin = None
        if query_sparse is not None:
            query_map_context = F.normalize(
                query_sparse[context_slot].float(), dim=0
            )
            correct_map = map_context[
                target_position[int(event["correct_anchor"])]
            ]
            confusing_map = map_context[
                target_position[int(event["confusing_anchor"])]
            ]
            map_margin = float(
                query_map_context @ correct_map
                - query_map_context @ confusing_map
            )
        trajectory = _trajectory(name)
        correct_static, correct_view, correct_count = (
            _cross_trajectory_scores(
                query_context,
                context_prototypes,
                int(event["correct_anchor"]),
                trajectory,
            )
        )
        confusing_static, confusing_view, confusing_count = (
            _cross_trajectory_scores(
                query_context,
                context_prototypes,
                int(event["confusing_anchor"]),
                trajectory,
            )
        )
        static_margin = (
            None
            if correct_static is None or confusing_static is None
            else float(correct_static - confusing_static)
        )
        view_margin = (
            None
            if correct_view is None or confusing_view is None
            else float(correct_view - confusing_view)
        )
        base_margin = float(base_margins[index])
        margins = {
            "O0_local": base_margin,
            "O1_cross_trajectory_2d": static_margin,
            "O2_3d_context": map_margin,
        }
        for weight in context_weights:
            tag = str(weight).replace(".", "p")
            margins[f"O3_joint_3d_w{tag}"] = (
                None
                if map_margin is None
                else base_margin + weight * map_margin
            )
            margins[f"O4_view_family_w{tag}"] = (
                None
                if view_margin is None
                else base_margin + weight * view_margin
            )
        graph_margin_delta.append(
            abs(base_margin + float(event["score_margin"]))
        )
        records.append(
            {
                "edge_index": int(event["edge_index"]),
                "query_name": name,
                "query_row": row,
                "trajectory": trajectory,
                "correct_anchor": int(event["correct_anchor"]),
                "confusing_anchor": int(event["confusing_anchor"]),
                "correct_support_count": int(correct_count),
                "confusing_support_count": int(confusing_count),
                "margins": margins,
            }
        )
    methods = list(records[0]["margins"]) if records else []
    summaries = {
        method: _summarize_method(records, method) for method in methods
    }
    output = {
        "schema": "lafgs_real_2d3d_context_oracle",
        "version": 1,
        "mode": args.mode,
        "selected_edge_count": len(edges),
        "selected_event_count": len(events),
        "evaluated_event_count": len(records),
        "target_anchor_count": len(target_anchors),
        "support_query_count": len(support_rows),
        "context_query_count": len(context_payload["rows"]),
        "dense_sparse_parity": context_payload["dense_sparse_parity"],
        "baseline_graph_margin_abs_delta_median": (
            median(graph_margin_delta) if graph_margin_delta else None
        ),
        "methods": summaries,
        "records": records,
        "provenance": {
            key: _file_identity(value)
            for key, value in {
                "map": args.map,
                "metric_state": args.metric_state,
                "family_state": args.family_state,
                "query_cache": args.query_cache,
                "complete_positive_teacher": args.complete_positive_teacher,
                "confusion_graph": args.confusion_graph,
                "track_payload": args.track_payload,
            }.items()
        },
        "config": {
            "maximum_edges": args.maximum_edges,
            "maximum_events_per_edge": args.maximum_events_per_edge,
            "maximum_sparse_neighbors": args.maximum_sparse_neighbors,
            "sparse_radii_px": list(args.sparse_radii_px),
            "dense_radii_cells": list(args.dense_radii_cells),
            "map_neighbor_counts": list(args.map_neighbor_counts),
            "context_weights": list(context_weights),
        },
    }
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, output)
    print(
        json.dumps(
            {key: value for key, value in output.items() if key != "records"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
