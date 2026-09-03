#!/usr/bin/env python3
"""Materialize a test-unaware reliability subset of a Projective Anchor map."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.mapping_reliability_subset import select_mapping_reliable_anchors
from topology.v6_anchor_map import identity_metric_state, subset_projective_anchor_map


def _save_fresh(value: dict, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--expected-map-sha256", required=True)
    parser.add_argument("--expected-candidates-sha256", required=True)
    parser.add_argument("--minimum-identity-reliability", type=float, required=True)
    parser.add_argument("--minimum-geometry-reliability", type=float, required=True)
    parser.add_argument("--minimum-observations", type=int, required=True)
    parser.add_argument("--maximum-covariance-trace-m2", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    map_path = args.map.expanduser().resolve()
    candidates_path = args.candidates.expanduser().resolve()
    if sha256_file(map_path) != args.expected_map_sha256:
        raise ValueError("map SHA256 mismatch")
    if sha256_file(candidates_path) != args.expected_candidates_sha256:
        raise ValueError("candidate SHA256 mismatch")
    state = torch.load(map_path, map_location="cpu", weights_only=False)
    candidates = torch.load(candidates_path, map_location="cpu", weights_only=False)
    if state.get("schema") != "lafgs_materialized_anchor_map":
        raise ValueError("input is not a materialized Anchor map")
    count = int(torch.as_tensor(state["anchor_xyz"]).shape[0])
    if (
        torch.as_tensor(candidates["anchor_xyz"]).shape[0] != count
        or not torch.equal(
            torch.as_tensor(state["anchor_xyz"]),
            torch.as_tensor(candidates["anchor_xyz"]),
        )
        or not torch.equal(
            torch.as_tensor(state["anchor_features"]),
            torch.as_tensor(candidates["anchor_features"]),
        )
    ):
        raise ValueError("candidate rows do not exactly bind to the input map")

    selected, selection = select_mapping_reliable_anchors(
        candidates,
        minimum_identity_reliability=args.minimum_identity_reliability,
        minimum_geometry_reliability=args.minimum_geometry_reliability,
        minimum_observations=args.minimum_observations,
        maximum_covariance_trace_m2=args.maximum_covariance_trace_m2,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    output_map_path = output_dir / "projective_anchor_map.pt"
    output_metric_path = output_dir / "identity_metric.pt"
    subset = subset_projective_anchor_map(state, selected)
    subset["provenance"] = {
        **dict(subset.get("provenance", {})),
        "mapping_reliability_subset": selection,
        "source_map": str(map_path),
        "source_map_sha256": args.expected_map_sha256,
        "source_candidates": str(candidates_path),
        "source_candidates_sha256": args.expected_candidates_sha256,
        "uses_test_queries": False,
        "localization_outcomes_consumed": False,
    }
    _save_fresh(subset, output_map_path)
    metric = identity_metric_state(
        subset,
        map_path=str(output_map_path),
        map_sha256=sha256_file(output_map_path),
    )
    _save_fresh(metric, output_metric_path)
    report = {
        **selection,
        "inputs": {
            "map": str(map_path),
            "map_sha256": args.expected_map_sha256,
            "candidates": str(candidates_path),
            "candidates_sha256": args.expected_candidates_sha256,
        },
        "outputs": {
            "map": str(output_map_path),
            "map_sha256": sha256_file(output_map_path),
            "metric": str(output_metric_path),
            "metric_sha256": sha256_file(output_metric_path),
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
