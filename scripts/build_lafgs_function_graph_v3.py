#!/usr/bin/env python3
"""Align the Active Map functional graph with sparse raster provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _padded_anchor_sources(provenance: dict):
    offsets = torch.as_tensor(
        provenance["anchor_source_offsets"]
    ).long()
    ids = torch.as_tensor(
        provenance["anchor_source_primitive_ids"]
    ).long()
    weights = torch.as_tensor(
        provenance["anchor_source_weights"]
    ).float()
    counts = offsets[1:] - offsets[:-1]
    width = max(int(counts.max()), 1)
    padded_ids = torch.full((counts.numel(), width), -1, dtype=torch.long)
    padded_weights = torch.zeros((counts.numel(), width))
    for row in range(counts.numel()):
        count = int(counts[row])
        if count:
            start = int(offsets[row])
            padded_ids[row, :count] = ids[start : start + count]
            padded_weights[row, :count] = weights[start : start + count]
    return padded_ids, padded_weights, counts


def _candidate_provenance_mass(
    top_indices: torch.Tensor,
    primitive_ids: torch.Tensor,
    primitive_mass: torch.Tensor,
    source_ids: torch.Tensor,
    source_weights: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    output = torch.zeros(top_indices.shape, dtype=torch.float32)
    top_indices_device = top_indices.to(device)
    primitive_ids_device = primitive_ids.to(device)
    primitive_mass_device = primitive_mass.float().to(device)
    source_ids_device = source_ids.to(device)
    source_weights_device = source_weights.to(device)
    for start in range(0, top_indices.shape[0], chunk_size):
        end = min(start + chunk_size, top_indices.shape[0])
        selected = top_indices_device[start:end]
        candidate_ids = source_ids_device[selected]
        candidate_weights = source_weights_device[selected]
        raster_ids = primitive_ids_device[start:end]
        raster_mass = primitive_mass_device[start:end]
        matches = (
            candidate_ids[..., None]
            == raster_ids[:, None, None, :]
        )
        output[start:end] = (
            matches.float()
            * candidate_weights[..., None]
            * raster_mass[:, None, None, :]
        ).sum(dim=(-1, -2)).cpu()
    return output


def _increment(target: torch.Tensor, indices: torch.Tensor) -> None:
    if indices.numel():
        target.index_add_(
            0,
            indices.long(),
            torch.ones_like(indices, dtype=target.dtype),
        )


def main() -> None:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--function-graph-v2", required=True)
    parser.add_argument("--raster-provenance", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-contribution-mass", type=float, default=0.02)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    graph = torch.load(
        args.function_graph_v2, map_location="cpu", weights_only=False
    )
    provenance = torch.load(
        args.raster_provenance, map_location="cpu", weights_only=False
    )
    if graph["anchor_map"] != provenance["anchor_map"]:
        raise ValueError("functional graph and provenance anchor maps differ")
    if graph["query_names"] != provenance["query_names"]:
        raise ValueError("functional graph and provenance query names differ")
    source_ids, source_weights, source_counts = _padded_anchor_sources(
        provenance
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    provenance_records = {
        int(record["query_index"]): record
        for record in provenance["records"]
    }
    anchor_count = int(graph["anchor_count"])
    counters = {
        key: torch.zeros(anchor_count, dtype=torch.int64)
        for key in (
            "provenance_opportunity_count",
            "provenance_legal_hit_2px_count",
            "provenance_legal_hit_4px_count",
            "provenance_legal_hit_8px_count",
            "provenance_legal_winner_2px_count",
            "provenance_solver_inlier_gtclean_2px_count",
            "provenance_solver_inlier_gtclean_4px_count",
            "provenance_harmful_solver_inlier_count",
        )
    }
    records = []
    mass_total = 0.0
    mass_positive = 0
    candidate_total = 0
    for record in graph["records"]:
        query_index = int(record["query_index"])
        raster = provenance_records[query_index]
        rows = torch.as_tensor(record["query_rows"]).long()
        if not torch.equal(
            rows, torch.as_tensor(raster["query_rows"]).long()
        ):
            raise ValueError(f"query row mismatch at {query_index}")
        indices = torch.as_tensor(record["top_indices"]).long()
        mass = _candidate_provenance_mass(
            indices,
            torch.as_tensor(raster["primitive_ids"]).long(),
            torch.as_tensor(raster["contribution_mass"]).float(),
            source_ids,
            source_weights,
            args.chunk_size,
            device,
        )
        provenance_legal = mass >= float(args.minimum_contribution_mass)
        old_flags = torch.as_tensor(record["legal_flags"])
        depth_legal = (old_flags & 1) != 0
        legal2 = provenance_legal & ((old_flags & 2) != 0)
        legal4 = provenance_legal & ((old_flags & 4) != 0)
        legal8 = provenance_legal & ((old_flags & 8) != 0)
        legal_flags = (
            (provenance_legal & depth_legal).to(torch.uint8)
            | (legal2.to(torch.uint8) << 1)
            | (legal4.to(torch.uint8) << 2)
            | (legal8.to(torch.uint8) << 3)
        )
        flat = indices[provenance_legal]
        _increment(counters["provenance_opportunity_count"], flat)
        _increment(
            counters["provenance_legal_hit_2px_count"], indices[legal2]
        )
        _increment(
            counters["provenance_legal_hit_4px_count"], indices[legal4]
        )
        _increment(
            counters["provenance_legal_hit_8px_count"], indices[legal8]
        )
        _increment(
            counters["provenance_legal_winner_2px_count"],
            indices[:, 0][legal2[:, 0]],
        )
        solver_inlier = torch.as_tensor(record["solver_inlier"]).bool()
        _increment(
            counters["provenance_solver_inlier_gtclean_2px_count"],
            indices[:, 0][solver_inlier & legal2[:, 0]],
        )
        _increment(
            counters["provenance_solver_inlier_gtclean_4px_count"],
            indices[:, 0][solver_inlier & legal4[:, 0]],
        )
        harmful = solver_inlier & ~legal4[:, 0]
        _increment(
            counters["provenance_harmful_solver_inlier_count"],
            indices[:, 0][harmful],
        )
        records.append(
            {
                **record,
                "legal_flags_v2_depth": old_flags,
                "legal_flags": legal_flags,
                "provenance_mass": mass.to(torch.float16),
            }
        )
        positive = mass > 0
        mass_total += float(mass[positive].sum())
        mass_positive += int(positive.sum())
        candidate_total += mass.numel()
        if len(records) % 25 == 0:
            print(f"function graph V3: {len(records)} queries", flush=True)

    output = {
        **graph,
        "schema": "lafgs_keypoint_function_graph",
        "version": 3,
        "records": records,
        "raster_provenance": str(
            Path(args.raster_provenance).resolve()
        ),
        "raster_visibility_enabled": True,
        "raster_provenance_mode": "candidate_family_2dgs_composition",
        "anchor_source_family_size": source_counts,
        "config_v3": vars(args),
        **counters,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(
        json.dumps(
            {
                "anchor_count": anchor_count,
                "anchors_provenance_legal_p2": int(
                    (
                        counters["provenance_legal_hit_2px_count"] > 0
                    ).sum()
                ),
                "anchors_gtclean_solver_inlier": int(
                    (
                        counters[
                            "provenance_solver_inlier_gtclean_2px_count"
                        ]
                        > 0
                    ).sum()
                ),
                "anchors_harmful_solver_inlier": int(
                    (
                        counters[
                            "provenance_harmful_solver_inlier_count"
                        ]
                        > 0
                    ).sum()
                ),
                "positive_candidate_mass_rate": (
                    mass_positive / max(candidate_total, 1)
                ),
                "positive_mass_mean": (
                    mass_total / max(mass_positive, 1)
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
