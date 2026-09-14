#!/usr/bin/env python3
"""Audit duplicate Anchor identities using mapping-only evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common.hashing import sha256_file
from topology.anchor_equivalence import (
    SCHEMA,
    VERSION,
    anchor_functional_evidence,
    audit_component_ids,
    build_equivalence_candidates,
    equivalence_edge_masks,
    summarize_equivalence_audit,
)
from topology.anchor_registry import validate_registry_compatibility


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--function-graph", type=Path)
    parser.add_argument("--distance-scale-m", type=float)
    parser.add_argument("--maximum-anchors-per-observation", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    registry = torch.load(args.registry, map_location="cpu", weights_only=False)
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    validate_registry_compatibility(registry, state)
    distance_scale = args.distance_scale_m
    if distance_scale is None:
        distance_scale = (
            state.get("track_centric_reconstruction", {})
            .get("calibration", {})
            .get("parameters", {})
            .get("assignment_distance_m")
        )
    if distance_scale is None:
        raise ValueError(
            "distance scale is absent; pass --distance-scale-m from mapping calibration"
        )
    candidates = build_equivalence_candidates(
        registry,
        maximum_anchors_per_observation=args.maximum_anchors_per_observation,
    )
    functional = None
    graph_hash = None
    if args.function_graph is not None:
        function_graph = torch.load(
            args.function_graph, map_location="cpu", weights_only=False
        )
        functional = anchor_functional_evidence(registry, state, function_graph)
        graph_hash = sha256_file(args.function_graph)
    report = summarize_equivalence_audit(
        registry,
        candidates,
        distance_scale_m=float(distance_scale),
        functional_evidence=functional,
    )
    edge_masks = equivalence_edge_masks(
        candidates, distance_scale_m=float(distance_scale)
    )
    independent_component_ids = audit_component_ids(
        candidates,
        edge_masks["independent_support"],
        anchor_count=int(torch.as_tensor(registry["anchor_ids"]).numel()),
    )
    artifact = {
        "schema": SCHEMA,
        "version": VERSION,
        "registry_path": str(args.registry.resolve()),
        "registry_sha256": sha256_file(args.registry),
        "map_path": str(args.map.resolve()),
        "map_sha256": sha256_file(args.map),
        "function_graph_path": (
            str(args.function_graph.resolve())
            if args.function_graph is not None
            else None
        ),
        "function_graph_sha256": graph_hash,
        "candidates": candidates,
        "edge_masks": edge_masks,
        "independent_support_component_ids": independent_component_ids,
        "functional_evidence": functional,
        "report": report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
