#!/usr/bin/env python3
"""Materialize zero-Gaussian selector inputs and crossfit Track exclusions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
        if reloaded.get("schema") != payload.get("schema"):
            raise RuntimeError("temporary fullchain input did not reload")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize(
    *,
    anchor_map_path: Path,
    track_payload_path: Path,
    query_cache_path: Path,
    capacity_report_path: Path,
    output_dir: Path,
) -> dict:
    for path in (
        anchor_map_path,
        track_payload_path,
        query_cache_path,
        capacity_report_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    state = torch.load(anchor_map_path, map_location="cpu", weights_only=False)
    payload = torch.load(track_payload_path, map_location="cpu", weights_only=False)
    cache = torch.load(query_cache_path, map_location="cpu", weights_only=False)
    report = json.loads(capacity_report_path.read_text())
    if (
        state.get("schema") != "lafgs_materialized_anchor_map"
        or payload.get("schema") != "lafgs_track_first_payload"
        or payload.get("rendered_rgb_only") is not True
        or cache.get("uses_source_mapping_rgb") is not False
        or cache.get("uses_test_queries") is not False
        or report.get("schema") != "lafgs_rendered_track_train_only_capacity_selection"
        or report.get("uses_test_queries") is not False
    ):
        raise ValueError("inputs do not describe a rendered-RGB mapping-only run")
    source_path = Path(str(report["inputs"]["anchor_map"])).resolve()
    if source_path != anchor_map_path:
        raise ValueError("capacity report names a different source Anchor map")
    if report["input_sha256"]["anchor_map"] != sha256_file(anchor_map_path):
        raise ValueError("capacity report source Anchor-map SHA differs")
    track_rows = torch.as_tensor(state["track_cluster_ids"]).long()
    pruned_rows = torch.as_tensor(report["pruned_anchor_rows"]).long()
    if (
        pruned_rows.ndim != 1
        or torch.unique(pruned_rows).numel() != pruned_rows.numel()
    ):
        raise ValueError("capacity report pruned rows must be unique")
    if pruned_rows.numel() and (
        int(pruned_rows.min()) < 0 or int(pruned_rows.max()) >= track_rows.numel()
    ):
        raise ValueError("capacity report pruned row is outside its source map")
    excluded_ids = track_rows[pruned_rows]
    track_count = int(
        torch.as_tensor(payload["track_geometry"]["triangulated"]).numel()
    )
    if excluded_ids.numel() and (
        int(excluded_ids.min()) < 0 or int(excluded_ids.max()) >= track_count
    ):
        raise ValueError("capacity report resolves outside the Track universe")
    descriptor_dim = int(torch.as_tensor(state["anchor_features"]).shape[1])
    empty = {
        "schema": "lafgs_materialized_anchor_map",
        "version": 1,
        "anchor_ids": torch.empty(0, dtype=torch.long),
        "anchor_xyz": torch.empty((0, 3), dtype=torch.float32),
        "anchor_features": torch.empty((0, descriptor_dim), dtype=torch.float32),
        "source_primitive_ids": torch.empty(0, dtype=torch.long),
        "track_cluster_ids": torch.empty(0, dtype=torch.long),
        "anchor_type": torch.empty(0, dtype=torch.long),
        "dependency_group_ids": torch.empty(0, dtype=torch.long),
        "coarse_dependency_group_ids": torch.empty(0, dtype=torch.long),
        "fine_identity_ids": torch.empty(0, dtype=torch.long),
        "base_anchor_count": 0,
        "canonical_anchor_count": 0,
        "micro_anchor_count": 0,
        "provenance": {
            "mapping_rgb_source": "gaussian_render_only",
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
            "uses_gaussian_anchor_candidates": False,
            "gaussian_support_scope": "optional_annotation_only",
            "track_payload": str(track_payload_path),
            "track_payload_sha256": sha256_file(track_payload_path),
        },
    }
    graph = {
        "schema": "lafgs_rendered_track_only_empty_base_graph",
        "version": 1,
        "query_names": list(payload["query_names"]),
        "records": [],
        "uses_test_queries": False,
        "base_anchor_count": 0,
    }
    exclusions = {
        "schema": "lafgs_rendered_track_crossfit_exclusions",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "track_payload": str(track_payload_path),
        "track_payload_sha256": sha256_file(track_payload_path),
        "source_anchor_map": str(anchor_map_path),
        "source_anchor_map_sha256": sha256_file(anchor_map_path),
        "capacity_report": str(capacity_report_path),
        "capacity_report_sha256": sha256_file(capacity_report_path),
        "excluded_track_ids": excluded_ids,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "empty_canonical_map": output_dir / "empty_canonical_map.pt",
        "empty_base_graph": output_dir / "empty_base_graph.pt",
        "track_exclusions": output_dir / "track_exclusions.pt",
    }
    _atomic_torch_save(empty, outputs["empty_canonical_map"])
    _atomic_torch_save(graph, outputs["empty_base_graph"])
    _atomic_torch_save(exclusions, outputs["track_exclusions"])
    result = {
        "schema": "lafgs_rendered_track_fullchain_inputs",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "candidate_track_count": track_count,
        "excluded_track_count": int(excluded_ids.numel()),
        "outputs": {key: str(path.resolve()) for key, path in outputs.items()},
        "output_sha256": {key: sha256_file(path) for key, path in outputs.items()},
    }
    report_path = output_dir / "fullchain_inputs.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-map", type=Path, required=True)
    parser.add_argument("--track-payload", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--capacity-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(
        anchor_map_path=args.anchor_map.resolve(),
        track_payload_path=args.track_payload.resolve(),
        query_cache_path=args.query_cache.resolve(),
        capacity_report_path=args.capacity_report.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
