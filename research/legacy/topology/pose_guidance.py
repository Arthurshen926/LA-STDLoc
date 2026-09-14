#!/usr/bin/env python3
"""Attach mapping-derived matchability to a covariance-bearing compact map."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from topology.adaptive_distillation import _candidate_matchability


def attach_matchability(state: dict, payload: dict, graph: dict) -> tuple[dict, dict]:
    metadata = state["track_centric_reconstruction"]
    track_ids = torch.as_tensor(metadata["track_indices"]).long()
    base_rows = torch.as_tensor(metadata["base_canonical_rows"]).long()
    base_count = int(graph["anchor_count"])
    threshold = float(
        metadata["calibration"]["parameters"]["track_reprojection_median_px"]
    )
    universe = _candidate_matchability(payload, graph, base_count, threshold)
    selected = torch.cat(
        (universe[track_ids], universe[len(universe) - base_count + base_rows])
    )
    output = dict(state)
    output["anchor_matchability"] = selected.float()
    output["provenance"] = {
        **state.get("provenance", {}),
        "pose_guidance": {
            "matchability": "mapping_detector_repeatability_x_global_reliability",
            "uncertainty": "anchor_position_covariance_trace_normalized_by_map_median",
            "uses_test_queries": False,
        },
    }
    return output, {
        "anchor_count": int(selected.numel()),
        "matchability_p10": float(torch.quantile(selected, 0.1)),
        "matchability_median": float(torch.median(selected)),
        "matchability_p90": float(torch.quantile(selected, 0.9)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--function-graph", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    payload = torch.load(args.track_payload, map_location="cpu", weights_only=False)
    graph = torch.load(args.function_graph, map_location="cpu", weights_only=False)
    output, report = attach_matchability(state, payload, graph)
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, destination)
    print({"output": str(destination), **report})


if __name__ == "__main__":
    main()
