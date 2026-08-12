#!/usr/bin/env python3
"""Materialize the complete eligible candidate universe for alias-risk audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common.config import load_mainline_config
from evidence.tracks import fuse_track_descriptors
from topology.adaptive_distillation import (
    _adaptive_track_eligibility,
    _deployment_track_geometry,
    _graph_counter,
    _image_only_core_eligibility,
)
from topology.track_core import _materialize


def _load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-map", type=Path, required=True)
    parser.add_argument("--function-graph", type=Path, required=True)
    parser.add_argument("--track-payload", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--scene-calibration", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    canonical = _load(args.canonical_map)
    graph = _load(args.function_graph)
    payload = _load(args.track_payload)
    query_cache = _load(args.query_cache)
    calibration = json.loads(args.scene_calibration.read_text())
    if calibration.get("schema") != "lafgs_mapping_only_scene_calibration":
        raise ValueError("candidate universe requires mapping-only calibration")
    uses_test_queries = calibration.get(
        "uses_test_queries", calibration.get("sources", {}).get("uses_test_queries")
    )
    if uses_test_queries is not False:
        raise ValueError("candidate universe must not use test queries")
    parameters = calibration["parameters"]
    policy = load_mainline_config(args.config).values["adaptive"]

    geometry = payload["track_geometry"]
    track_count = int(torch.as_tensor(geometry["triangulated"]).numel())
    base_count = int(canonical["base_anchor_count"])
    broad = _adaptive_track_eligibility(
        geometry,
        median_px=float(parameters["track_reprojection_median_px"]),
        p90_px=float(parameters["track_reprojection_p90_px"]),
        covariance_m2=float(parameters["track_covariance_trace_m2"]),
        broad=True,
    )
    image_only_core = _image_only_core_eligibility(
        geometry,
        median_px=float(parameters["track_reprojection_median_px"]),
        p90_px=float(parameters["track_reprojection_p90_px"]),
        covariance_m2=float(parameters["track_covariance_trace_m2"]),
    )
    deployment_payload = dict(payload)
    deployment_payload["track_geometry"] = _deployment_track_geometry(
        geometry, image_only_core
    )
    legal = _graph_counter(
        graph,
        "provenance_legal_hit_strong_count",
        "provenance_legal_hit_2px_count",
    )[:base_count] > 0
    selected_tracks = torch.nonzero(broad, as_tuple=False).reshape(-1)
    selected_base = torch.nonzero(legal, as_tuple=False).reshape(-1)

    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        track_features = fuse_track_descriptors(
            payload=payload,
            query_cache=query_cache,
            track_indices=selected_tracks,
            trim_fraction=float(policy["descriptor_trim_fraction"]),
        )
    finally:
        torch.set_num_threads(original_threads)
    budget = int(selected_tracks.numel() + selected_base.numel())
    state = _materialize(
        canonical,
        deployment_payload,
        selected_tracks,
        track_features,
        selected_base,
        budget=budget,
        quality_tier="all_eligible_alias_audit_only",
        source_map=args.canonical_map.resolve(),
        payload_path=args.track_payload.resolve(),
        dependency_voxel_size=float(parameters["dependency_voxel_m"]),
        separate_spatial_dependency=True,
    )
    candidate_universe_ids = torch.cat(
        (selected_tracks, track_count + selected_base)
    )
    if candidate_universe_ids.numel() != torch.as_tensor(
        state["anchor_ids"]
    ).numel():
        raise ValueError("candidate universe IDs do not align with audit map")
    state["candidate_universe_ids"] = candidate_universe_ids
    state["audit_contract"] = {
        "schema": "lafgs_selector_candidate_universe",
        "version": 1,
        "uses_test_queries": False,
        "deployment_authorized": False,
        "candidate_definition": (
            "broad_observation_track_or_mapping_legal_surface_candidate"
        ),
        "risk_filter_applied": False,
        "track_candidate_universe_count": track_count,
        "base_candidate_universe_count": base_count,
        "eligible_track_count": int(selected_tracks.numel()),
        "eligible_base_count": int(selected_base.numel()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    print(json.dumps(state["audit_contract"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
