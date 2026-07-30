#!/usr/bin/env python3
"""Gate candidate-context proposals with a strict GT cleanliness oracle."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from localization_training.candidate_context_rescue import (
    oracle_acceptance_mask,
)


def _records_by_name(payload: dict) -> dict[str, dict]:
    return {
        str(record["query_name"]): record
        for record in payload["records"]
    }


def _project_errors(
    xyz: torch.Tensor,
    keypoints: torch.Tensor,
    K: torch.Tensor,
    pose_w2c: torch.Tensor,
) -> torch.Tensor:
    camera = xyz @ pose_w2c[:3, :3].T + pose_w2c[:3, 3]
    projected = camera[:, :2] / camera[:, 2:3].clamp_min(1e-8)
    projected = projected @ K[:2, :2].T + K[:2, 2]
    errors = torch.linalg.vector_norm(projected - keypoints, dim=1)
    errors[camera[:, 2] <= 1e-6] = torch.inf
    return errors


def main() -> None:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--baseline-topk", required=True)
    parser.add_argument("--proposal-topk", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold-px", type=float, default=2.0)
    args = parser.parse_args()

    state = torch.load(args.map, map_location="cpu", weights_only=False)
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    baseline = torch.load(
        args.baseline_topk, map_location="cpu", weights_only=False
    )
    proposal = torch.load(
        args.proposal_topk, map_location="cpu", weights_only=False
    )
    if baseline["query_names"] != proposal["query_names"]:
        raise ValueError("oracle query registries differ")
    if baseline["anchor_ids_sha256"] != proposal["anchor_ids_sha256"]:
        raise ValueError("oracle maps differ")
    proposal_by_name = _records_by_name(proposal)
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    records = []
    proposed_count = 0
    accepted_count = 0
    protected_clean_count = 0
    for before in baseline["records"]:
        name = str(before["query_name"])
        after = proposal_by_name[name]
        rows = torch.as_tensor(before["query_rows"]).long()
        baseline_indices = torch.as_tensor(
            before["topk_anchor_indices"]
        ).long()
        baseline_scores = torch.as_tensor(before["topk_scores"]).float()
        proposal_indices = torch.as_tensor(
            after["topk_anchor_indices"]
        ).long()
        proposal_scores = torch.as_tensor(after["topk_scores"]).float()
        changed = proposal_indices[:, 0] != baseline_indices[:, 0]
        cached = cache[name]
        keypoints = (
            torch.as_tensor(cached["native_keypoints"]).float()[rows]
            + float(cached.get("pixel_center_offset", 0.5))
        )
        K = torch.as_tensor(cached["native_K"]).float()
        pose = torch.as_tensor(cached["pose_w2c"]).float()
        baseline_errors = _project_errors(
            xyz[baseline_indices[:, 0]], keypoints, K, pose
        )
        proposal_errors = _project_errors(
            xyz[proposal_indices[:, 0]], keypoints, K, pose
        )
        accepted = oracle_acceptance_mask(
            baseline_errors,
            proposal_errors,
            changed,
            strict_threshold_px=float(args.threshold_px),
        )
        indices = baseline_indices.clone()
        scores = baseline_scores.clone()
        indices[accepted] = proposal_indices[accepted]
        scores[accepted] = proposal_scores[accepted]
        proposed_count += int(changed.sum())
        accepted_count += int(accepted.sum())
        protected_clean_count += int(
            (changed & (baseline_errors <= float(args.threshold_px))).sum()
        )
        records.append(
            {
                "query_name": name,
                "query_rows": rows,
                "topk_anchor_indices": indices,
                "topk_scores": scores,
            }
        )
    output = {
        "schema": "lafgs_exact_topk_outcomes",
        "version": 2,
        "query_names": list(baseline["query_names"]),
        "query_start": int(baseline.get("query_start", 0)),
        "topk": int(baseline["topk"]),
        "anchor_count": int(baseline["anchor_count"]),
        "anchor_ids_sha256": baseline["anchor_ids_sha256"],
        "records": records,
        "method": "candidate_conditioned_context_strict_clean_oracle",
        "summary": {
            "proposed_count": proposed_count,
            "accepted_count": accepted_count,
            "protected_clean_proposal_count": protected_clean_count,
            "threshold_px": float(args.threshold_px),
        },
        "provenance": {
            "baseline_topk": str(Path(args.baseline_topk).resolve()),
            "proposal_topk": str(Path(args.proposal_topk).resolve()),
            "map": str(Path(args.map).resolve()),
            "query_cache": str(Path(args.query_cache).resolve()),
        },
    }
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(output, temporary)
    os.replace(temporary, path)
    print({"output": str(path), **output["summary"]})


if __name__ == "__main__":
    main()
