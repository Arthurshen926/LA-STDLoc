#!/usr/bin/env python3
"""Build a reversible clean-Anchor candidate from V2-valid observations only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from topology.v6_anchor_map import subset_projective_anchor_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-map", type=Path, required=True)
    parser.add_argument("--observation-cache", type=Path, required=True)
    parser.add_argument("--anchor-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-valid-observations", type=int, default=3)
    parser.add_argument("--minimum-valid-view-families", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    state = torch.load(args.anchor_map, map_location="cpu", weights_only=False)
    cache = torch.load(args.observation_cache, map_location="cpu", weights_only=False)
    evidence = torch.load(args.anchor_evidence, map_location="cpu", weights_only=False)
    if not torch.equal(torch.as_tensor(state["anchor_ids"]), torch.as_tensor(evidence["anchor_ids"])):
        raise ValueError("map/evidence Anchor registries differ")
    selected = torch.nonzero(
        (torch.as_tensor(evidence["valid_observation_count"]) >= args.minimum_valid_observations)
        & (torch.as_tensor(evidence["valid_view_family_count"]) >= args.minimum_valid_view_families),
        as_tuple=False,
    ).reshape(-1)
    output = subset_projective_anchor_map(state, selected)
    source_csr = state["projective_anchor_observations"]
    offsets = torch.as_tensor(source_csr["observation_offsets"]).long()
    queries = torch.as_tensor(source_csr["query_indices"]).long()
    keypoints = torch.as_tensor(source_csr["keypoint_indices"]).long()
    observation_valid = torch.as_tensor(evidence["observation_valid"]).bool()
    names = list(state["v6_mapping_query_names"])
    output_offsets = [0]
    output_queries = []
    output_keypoints = []
    descriptors = []
    for completed, source_row in enumerate(selected.tolist(), 1):
        start, stop = int(offsets[source_row]), int(offsets[source_row + 1])
        keep = observation_valid[start:stop]
        query_rows = queries[start:stop][keep]
        keypoint_rows = keypoints[start:stop][keep]
        observation_descriptors = torch.stack([
            torch.as_tensor(cache["queries"][names[int(q)]]["native_descriptors"])[int(k)].float()
            for q, k in zip(query_rows.tolist(), keypoint_rows.tolist())
        ])
        descriptors.append(F.normalize(observation_descriptors.mean(0), dim=0))
        output_queries.append(query_rows)
        output_keypoints.append(keypoint_rows)
        output_offsets.append(output_offsets[-1] + query_rows.numel())
        if completed % 5000 == 0 or completed == selected.numel():
            print(f"clean Anchor descriptors: {completed}/{selected.numel()}", flush=True)
    output["anchor_features"] = torch.stack(descriptors)
    output["anchor_matchability"] = (
        torch.as_tensor(output["anchor_matchability"]).float()
        * torch.as_tensor(evidence["valid_observation_fraction"])[selected].float()
    )
    output["projective_anchor_observations"] = {
        **dict(source_csr),
        "observation_offsets": torch.tensor(output_offsets, dtype=torch.long),
        "query_indices": torch.cat(output_queries),
        "keypoint_indices": torch.cat(output_keypoints),
    }
    output["v8_clean_anchor"] = {
        "schema": "lafgs_v8_clean_anchor_map", "version": 1,
        "source_anchor_rows": selected,
        "minimum_valid_observations": args.minimum_valid_observations,
        "minimum_valid_view_families": args.minimum_valid_view_families,
        "descriptors_from_v2_valid_original_render_observations_only": True,
        "uses_source_mapping_rgb": False, "uses_test_queries": False,
        "source_map_sha256": sha256_file(args.anchor_map),
        "anchor_evidence_sha256": sha256_file(args.anchor_evidence),
    }
    output.setdefault("provenance", {})["v8_clean_anchor_candidate"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    torch.save(output, temporary)
    os.replace(temporary, args.output)
    report = {
        "schema": "lafgs_v8_clean_anchor_map_report", "version": 1,
        "source_anchor_count": int(torch.as_tensor(state["anchor_ids"]).numel()),
        "clean_anchor_count": int(selected.numel()),
        "retained_fraction": float(selected.numel() / torch.as_tensor(state["anchor_ids"]).numel()),
        "output": str(args.output.resolve()), "output_sha256": sha256_file(args.output),
        "uses_test_queries": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
