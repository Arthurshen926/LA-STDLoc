#!/usr/bin/env python3
"""Aggregate mapping V2 row audits and materialize reversible map ablations."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from evidence.v7_anchor_contamination import (
    aggregate_anchor_reliability,
    bounded_descriptor_reconstruction,
    gather_observation_rows,
)


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _quantiles(value: torch.Tensor) -> dict[str, float]:
    tensor = torch.as_tensor(value).float().reshape(-1)
    return {
        "minimum": float(tensor.min()),
        "p10": float(torch.quantile(tensor, 0.10)),
        "median": float(tensor.median()),
        "p90": float(torch.quantile(tensor, 0.90)),
        "maximum": float(tensor.max()),
        "mean": float(tensor.mean()),
    }


def _materialize_map(source: dict, selected: torch.Tensor) -> dict:
    rows = torch.as_tensor(selected).long().cpu()
    count = int(torch.as_tensor(source["anchor_ids"]).numel())
    output = {}
    for key, value in source.items():
        if key == "projective_anchor_observations":
            continue
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == count:
            output[key] = value[rows].clone()
        elif isinstance(value, list) and len(value) == count:
            output[key] = [value[row] for row in rows.tolist()]
        else:
            output[key] = value
    observations = source["projective_anchor_observations"]
    offsets = torch.as_tensor(observations["observation_offsets"]).long()
    query = torch.as_tensor(observations["query_indices"]).long()
    keypoint = torch.as_tensor(observations["keypoint_indices"]).long()
    lengths = offsets[rows + 1] - offsets[rows]
    output_offsets = torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0)))
    parts = [torch.arange(int(offsets[row]), int(offsets[row + 1])) for row in rows]
    observation_rows = torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)
    output["projective_anchor_observations"] = {
        **{
            key: value
            for key, value in observations.items()
            if key not in {"observation_offsets", "query_indices", "keypoint_indices"}
        },
        "observation_offsets": output_offsets,
        "query_indices": query[observation_rows],
        "keypoint_indices": keypoint[observation_rows],
    }
    selected_count = int(rows.numel())
    output["base_anchor_count"] = 0
    output["canonical_anchor_count"] = selected_count
    output["micro_anchor_count"] = selected_count
    return output


def _write_metric(source: dict, map_payload: dict, map_path: Path, metric_path: Path) -> None:
    metric = dict(source)
    metric["landmark_indices"] = torch.as_tensor(map_payload["anchor_ids"]).long().clone()
    metric["map_path"] = str(map_path.resolve())
    metric["map_sha256"] = sha256_file(map_path)
    _atomic_save(metric, metric_path)


def _valid_descriptor_means(
    *,
    cache: dict,
    names: list[str],
    observation_valid: torch.Tensor,
    observation_offsets: torch.Tensor,
    observation_query_indices: torch.Tensor,
    observation_keypoint_indices: torch.Tensor,
    anchor_count: int,
    descriptor_dim: int,
) -> torch.Tensor:
    """Accumulate valid mapping descriptors without packing the 3M-row cache."""

    valid = torch.as_tensor(observation_valid).bool()
    offsets = torch.as_tensor(observation_offsets).long()
    query = torch.as_tensor(observation_query_indices).long()
    keypoint = torch.as_tensor(observation_keypoint_indices).long()
    anchor_for_observation = torch.repeat_interleave(
        torch.arange(anchor_count, dtype=torch.long), offsets[1:] - offsets[:-1]
    )
    order = torch.argsort(query, stable=True)
    sorted_query = query[order]
    query_counts = torch.bincount(sorted_query, minlength=len(names))
    query_offsets = torch.cat((torch.zeros(1, dtype=torch.long), query_counts.cumsum(0)))
    sums = torch.zeros(anchor_count, descriptor_dim, dtype=torch.float32)
    for query_index, name in enumerate(names):
        start, end = int(query_offsets[query_index]), int(query_offsets[query_index + 1])
        positions = order[start:end]
        positions = positions[valid[positions]]
        if positions.numel() == 0:
            continue
        descriptors = torch.as_tensor(
            cache["queries"][name]["native_descriptors"]
        ).float()[keypoint[positions]]
        sums.index_add_(0, anchor_for_observation[positions], descriptors)
        if (query_index + 1) % 100 == 0 or query_index + 1 == len(names):
            print(f"descriptor accumulation {query_index + 1}/{len(names)}", flush=True)
    return F.normalize(sums, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--observation-cache", type=Path, required=True)
    parser.add_argument("--anchor-map", type=Path, required=True)
    parser.add_argument("--identity-metric", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-descriptor-angle-deg", type=float, default=5.0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    cache_path = args.observation_cache.resolve()
    map_path = args.anchor_map.resolve()
    metric_path = args.identity_metric.resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    source_map = torch.load(map_path, map_location="cpu", weights_only=False)
    source_metric = torch.load(metric_path, map_location="cpu", weights_only=False)
    names = list(source_map["v6_mapping_query_names"])
    records: list[dict | None] = [None] * len(names)
    shard_payloads = []
    for path in args.shards:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != "lafgs_v7_mapping_render_quality_audit_shard":
            raise ValueError(f"unexpected audit shard: {path}")
        if payload["input"]["anchor_map_sha256"] != sha256_file(map_path):
            raise ValueError("audit shard map lineage differs")
        if payload["input"]["observation_cache_sha256"] != sha256_file(cache_path):
            raise ValueError("audit shard cache lineage differs")
        shard_payloads.append(payload)
        for record in payload["records"]:
            index = int(record["query_index"])
            if records[index] is not None:
                raise ValueError("duplicate query across audit shards")
            records[index] = record
    if any(record is None for record in records):
        raise ValueError("audit shards do not cover the mapping registry")
    complete = [record for record in records if record is not None]
    if [record["query_name"] for record in complete] != names:
        raise ValueError("audit record names differ from map lineage")

    observations = source_map["projective_anchor_observations"]
    query_indices = torch.as_tensor(observations["query_indices"]).long()
    keypoint_indices = torch.as_tensor(observations["keypoint_indices"]).long()
    offsets = torch.as_tensor(observations["observation_offsets"]).long()
    valid_by_query = [record["row_valid"] for record in complete]
    structure_by_query = [record["row_structure_supported"] for record in complete]
    observation_valid = gather_observation_rows(
        valid_by_query, query_indices, keypoint_indices
    )
    observation_structure = gather_observation_rows(
        structure_by_query, query_indices, keypoint_indices
    )
    reliability = aggregate_anchor_reliability(
        observation_valid=observation_valid,
        observation_structure_supported=observation_structure,
        observation_offsets=offsets,
        observation_query_indices=query_indices,
        query_family_ids=source_map["v6_mapping_query_bins"],
    )
    anchor_count = int(source_map["anchor_ids"].numel())
    proposed = _valid_descriptor_means(
        cache=cache,
        names=names,
        observation_valid=observation_valid,
        observation_offsets=offsets,
        observation_query_indices=query_indices,
        observation_keypoint_indices=keypoint_indices,
        anchor_count=anchor_count,
        descriptor_dim=int(source_map["anchor_features"].shape[1]),
    )
    bounded, angles = bounded_descriptor_reconstruction(
        source_map["anchor_features"],
        proposed,
        reliability["descriptor_reconstructable"],
        maximum_angle_deg=args.maximum_descriptor_angle_deg,
    )
    strict_keep = ~reliability["pure_contamination"]

    evidence_path = args.output_dir / "anchor_contamination_evidence.pt"
    evidence = {
        "schema": "lafgs_v7_anchor_contamination_evidence",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "threshold_tuning_from_results": False,
        "map_mutation_count": 0,
        "anchor_ids": source_map["anchor_ids"].clone(),
        **reliability,
        "bounded_descriptor_angle_deg": angles,
        "observation_valid": observation_valid,
        "observation_structure_supported": observation_structure,
        "input": {
            "anchor_map": str(map_path),
            "anchor_map_sha256": sha256_file(map_path),
            "observation_cache": str(cache_path),
            "observation_cache_sha256": sha256_file(cache_path),
            "audit_shards": [str(path.resolve()) for path in args.shards],
            "audit_shard_sha256": [sha256_file(path) for path in args.shards],
        },
    }
    _atomic_save(evidence, evidence_path)

    variants = {}
    for name, retire, reconstruct in (
        ("strict_retire", True, False),
        ("bounded_reaggregate", False, True),
        ("strict_retire_bounded_reaggregate", True, True),
    ):
        selected = torch.nonzero(strict_keep if retire else torch.ones(anchor_count, dtype=torch.bool)).flatten()
        payload = _materialize_map(source_map, selected)
        if reconstruct:
            payload["anchor_features"] = bounded[selected].to(
                dtype=source_map["anchor_features"].dtype
            )
        payload["provenance"] = {
            **dict(source_map["provenance"]),
            "v7_anchor_contamination_audit": True,
            "v7_anchor_contamination_policy": name,
            "v7_anchor_contamination_evidence_sha256": sha256_file(evidence_path),
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
            "formal_method_selected": False,
        }
        variant_dir = args.output_dir / name
        variant_dir.mkdir()
        output_map = variant_dir / "map.pt"
        output_metric = variant_dir / "identity_metric.pt"
        _atomic_save(payload, output_map)
        _write_metric(source_metric, payload, output_map, output_metric)
        variants[name] = {
            "anchor_count": int(payload["anchor_ids"].numel()),
            "map": str(output_map.resolve()),
            "map_sha256": sha256_file(output_map),
            "metric": str(output_metric.resolve()),
            "metric_sha256": sha256_file(output_metric),
        }

    reason_counts = {}
    total_cached_rows = sum(int(record["keypoint_count"]) for record in complete)
    for reason in complete[0]["row_reasons"]:
        reason_counts[reason] = sum(
            int(torch.as_tensor(record["row_reasons"][reason]).sum())
            for record in complete
        )
    report = {
        "schema": "lafgs_v7_anchor_contamination_audit",
        "version": 1,
        "status": "PASS",
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "threshold_tuning_from_results": False,
        "formal_method_selected": False,
        "mapping_query_count": len(names),
        "anchor_count": anchor_count,
        "anchor_observation_count": int(observation_valid.numel()),
        "cached_detector_row_count": total_cached_rows,
        "cached_detector_valid_count_v2": sum(
            int(torch.as_tensor(record["row_valid"]).sum()) for record in complete
        ),
        "cached_detector_valid_fraction_v2": sum(
            int(torch.as_tensor(record["row_valid"]).sum()) for record in complete
        ) / max(total_cached_rows, 1),
        "cached_row_reason_counts": reason_counts,
        "anchor_valid_fraction_quantiles": _quantiles(
            reliability["valid_observation_fraction"]
        ),
        "pure_contamination_anchor_count": int(
            reliability["pure_contamination"].sum()
        ),
        "mixed_contamination_anchor_count": int(
            reliability["mixed_contamination"].sum()
        ),
        "descriptor_reconstructable_anchor_count": int(
            reliability["descriptor_reconstructable"].sum()
        ),
        "bounded_descriptor_changed_anchor_count": int((angles > 1e-3).sum()),
        "bounded_descriptor_angle_quantiles_changed": (
            _quantiles(angles[angles > 1e-3]) if bool((angles > 1e-3).any()) else None
        ),
        "strict_rule": {
            "retire_only_when_zero_v2_valid_observations": True,
            "minimum_original_view_families": 2,
            "descriptor_reaggregate_minimum_valid_observations": 3,
            "descriptor_reaggregate_minimum_valid_view_families": 2,
            "maximum_descriptor_angle_deg": args.maximum_descriptor_angle_deg,
        },
        "artifacts": {
            "evidence": str(evidence_path.resolve()),
            "evidence_sha256": sha256_file(evidence_path),
            "variants": variants,
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
