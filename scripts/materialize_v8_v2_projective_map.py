#!/usr/bin/env python3
"""Rebuild the complete Projective Anchor chain with V2 before association."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

from common.hashing import sha256_file
from common.v6_contracts import RENDER_OBSERVATION_SCHEMA, require_schema
from evidence.projective_association import build_projective_association_graph
from evidence.projective_completion import build_projective_completion
from evidence.projective_reconstruction import reconstruct_projective_anchors
from evidence.v2_filtered_observations import (
    build_v2_filtered_provider,
    remap_candidate_rows_to_source,
)
from map_learning.v24_anchor_view_support import build_anchor_view_support
from topology.v6_anchor_map import (
    identity_metric_state,
    materialize_projective_anchor_map,
    merge_projective_candidates,
)


def _save(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _load_v2_rows(
    paths: list[Path],
    names: list[str],
    *,
    observation_cache: Path,
    observation_cache_sha256: str,
) -> list[torch.Tensor]:
    records = {}
    shard_indices = set()
    expected_shard_count = len(paths)
    for path in paths:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        shard_index = int(shard.get("shard_index", -1))
        shard_input = dict(shard.get("input", {}))
        if (
            shard.get("schema") != "lafgs_v7_mapping_render_quality_audit_shard"
            or shard.get("status") != "PASS"
            or shard.get("uses_source_mapping_rgb") is not False
            or shard.get("uses_test_queries") is not False
            or int(shard.get("shard_count", -1)) != expected_shard_count
            or shard_index < 0
            or shard_index >= expected_shard_count
            or shard_index in shard_indices
            or Path(str(shard_input.get("observation_cache", ""))).resolve()
            != observation_cache.resolve()
            or shard_input.get("observation_cache_sha256")
            != observation_cache_sha256
            or int(shard.get("mapping_query_count", -1)) != len(names)
        ):
            raise ValueError("V2 audit shard is outside mapping-only scope")
        shard_indices.add(shard_index)
        for record in shard["records"]:
            index = int(record["query_index"])
            if index in records:
                raise ValueError("duplicate V2 query row")
            if record["query_name"] != names[index]:
                raise ValueError("V2 query registry differs from observation cache")
            records[index] = torch.as_tensor(record["row_valid"]).bool()
    if shard_indices != set(range(expected_shard_count)) or sorted(records) != list(
        range(len(names))
    ):
        raise ValueError("V2 audit shards do not exactly cover mapping queries")
    return [records[index] for index in range(len(names))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-cache", type=Path, required=True)
    parser.add_argument("--v2-audit-shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--triangulation-workers", type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(args.cpu_threads)
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
    started = time.perf_counter()
    cache = torch.load(args.observation_cache, map_location="cpu", weights_only=False)
    require_schema(cache, RENDER_OBSERVATION_SCHEMA, label="V8 observations")
    names = list(cache["queries"])
    observation_cache_sha256 = sha256_file(args.observation_cache)
    audit_shard_sha256 = [sha256_file(path) for path in args.v2_audit_shards]
    rows = _load_v2_rows(
        args.v2_audit_shards,
        names,
        observation_cache=args.observation_cache,
        observation_cache_sha256=observation_cache_sha256,
    )
    provider, source_rows, filter_report = build_v2_filtered_provider(
        cache, rows_by_query=rows, allow_empty_queries=True
    )
    # Release the unfiltered sparse tensors. Dense alpha/depth tensors remain
    # referenced by the filtered records for completion proposals.
    del cache

    association_started = time.perf_counter()
    association = build_projective_association_graph(
        provider, pair_neighbors=6, minimum_similarity=0.65,
        minimum_margin=0.01, maximum_epipolar_error_px=2.0,
        minimum_track_views=3, device=args.device,
    )
    association["input_sha256"] = {
        "observation_cache": observation_cache_sha256,
        "v2_audit_shards": audit_shard_sha256,
    }
    association["v2_preassociation_filter"] = filter_report
    association["source_keypoint_indices_by_query"] = source_rows
    association_seconds = time.perf_counter() - association_started

    reconstruction_started = time.perf_counter()
    base = reconstruct_projective_anchors(
        provider, association, minimum_views=3, minimum_view_bins=1,
        minimum_parallax_deg=1.0, maximum_reprojection_px=2.0,
        parallel_workers=args.triangulation_workers,
        parallel_minimum_tracks=5000,
    )
    base["candidate_kind"] = "v2_projective_track"
    completion = None
    completion_unavailable_reason = None
    try:
        completion = build_projective_completion(
            provider, association, voxel_size_m=0.05, alpha_minimum=0.05,
            minimum_similarity=0.7, minimum_margin=0.01,
            maximum_epipolar_error_px=2.0, minimum_observations=3,
            minimum_camera_families=2, maximum_rows_per_view=256,
            safety_maximum_components=100000, device=args.device,
        )
    except ValueError as error:
        unavailable = {
            "no unused render-valid observations for completion": (
                "no_unused_render_valid_completion_observations"
            ),
            "depth proposals produced no descriptor-consistent component": (
                "no_descriptor_consistent_projective_completion"
            ),
            "association graph contains no ray-triangulated Anchor": (
                "no_ray_triangulated_completion_anchor"
            ),
        }
        if str(error) not in unavailable:
            raise
        completion_unavailable_reason = unavailable[str(error)]
    base = remap_candidate_rows_to_source(base, source_rows)
    candidate_parts = [base]
    if completion is not None:
        completion = remap_candidate_rows_to_source(completion, source_rows)
        candidate_parts.append(completion)
    candidates = merge_projective_candidates(candidate_parts)
    reconstruction_seconds = time.perf_counter() - reconstruction_started

    association_path = args.output_dir / "association_graph.pt"
    candidates_path = args.output_dir / "projective_anchor_candidates.pt"
    map_path = args.output_dir / "projective_anchor_map.pt"
    metric_path = args.output_dir / "identity_metric.pt"
    _save(association, association_path)
    _save(candidates, candidates_path)
    lineage = {
        "v8_observation_cache": str(args.observation_cache.resolve()),
        "v8_observation_cache_sha256": observation_cache_sha256,
        "v8_v2_audit_shards": [str(path.resolve()) for path in args.v2_audit_shards],
        "v8_v2_audit_shard_sha256": audit_shard_sha256,
        "v8_association_graph": str(association_path.resolve()),
        "v8_association_graph_sha256": sha256_file(association_path),
        "v8_filter_stage": "after_detection_before_pair_association",
        "projective_completion_attempted": True,
        "projective_completion_enabled": completion is not None,
        "projective_completion_unavailable_reason": completion_unavailable_reason,
    }
    state = materialize_projective_anchor_map(candidates, lineage=lineage)
    # The candidate CSR has already been remapped from filtered-local row ids
    # to the immutable source-cache row ids.  Preserve that contract in the
    # final map instead of relying on implicit knowledge in downstream audits.
    state["projective_anchor_observations"]["keypoint_index_semantics"] = (
        "original_unfiltered_observation_cache_row"
    )
    state["projective_anchor_construction"] = {
        **dict(state["projective_anchor_construction"]),
        "schema": "lafgs_v8_v2_projective_anchor_construction",
        "v2_preassociation_filter": True,
    }
    state["provenance"] = {
        **dict(state["provenance"]),
        "mapping_source": "gaussian_render_v2_filtered_before_projective_association",
    }
    mapping_poses = torch.stack(
        [provider.build_view(index).pose_w2c.float().cpu() for index in range(len(provider))]
    )
    observation_csr = state["projective_anchor_observations"]
    state["anchor_view_support"] = build_anchor_view_support(
        anchor_xyz=state["anchor_xyz"],
        observation_offsets=observation_csr["observation_offsets"],
        observation_query_indices=observation_csr["query_indices"],
        mapping_pose_w2c=mapping_poses,
    )
    state["provenance"]["v24_mapping_only_anchor_view_support"] = True
    _save(state, map_path)
    metric = identity_metric_state(
        state, map_path=str(map_path.resolve()), map_sha256=sha256_file(map_path)
    )
    _save(metric, metric_path)
    report = {
        "schema": "lafgs_v8_v2_projective_map_materialization_report", "version": 1,
        "uses_source_mapping_rgb": False, "uses_test_queries": False,
        "filter": filter_report,
        "counts": {
            "mapping_views": len(provider),
            "association_components": int(association["diagnostics"]["track_count"]),
            "base_projective_anchors": int(base["anchor_xyz"].shape[0]),
            "completion_anchors": int(
                0 if completion is None else completion["anchor_xyz"].shape[0]
            ),
            "total_anchors": int(state["anchor_xyz"].shape[0]),
        },
        "association_diagnostics": dict(association["diagnostics"]),
        "output": {
            "association_graph": str(association_path.resolve()),
            "association_graph_sha256": sha256_file(association_path),
            "candidates": str(candidates_path.resolve()),
            "candidates_sha256": sha256_file(candidates_path),
            "map": str(map_path.resolve()), "map_sha256": sha256_file(map_path),
            "metric": str(metric_path.resolve()), "metric_sha256": sha256_file(metric_path),
        },
        "contracts": {
            "superpoint_complete_unmasked_rgb": True,
            "v2_before_pair_association": True,
            "all_tracks_rebuilt": True, "all_geometry_retriangulated": True,
            "all_descriptors_fused_from_v2_valid_observations": True,
            "query_detector_used": False, "feedback_used": False,
            "mapping_only_anchor_view_support": True,
            "empty_v2_query_policy": "retain_zero_row_mapping_view",
            "projective_completion_attempted": True,
            "projective_completion_available": completion is not None,
            "projective_completion_unavailable_reason": (
                completion_unavailable_reason
            ),
        },
        "timing_seconds": {
            "association": association_seconds,
            "reconstruction_and_completion": reconstruction_seconds,
            "total": time.perf_counter() - started,
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
