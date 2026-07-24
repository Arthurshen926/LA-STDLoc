#!/usr/bin/env python3
"""Re-select a LaFGS landmark bank from immutable train-only statistics."""

import argparse
import json
from argparse import Namespace
from pathlib import Path

import torch

from train_lafgs_map import _distill_final_landmark_bank


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics_run", type=Path, required=True)
    parser.add_argument("--source_state", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument(
        "--profile",
        choices=("distinct_core", "core_reserve", "core_reserve_switch"),
        required=True,
    )
    return parser.parse_args()


def main():
    cli = parse_args()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (cli.statistics_run / "reproducibility_manifest.json").read_text()
    )
    args = Namespace(**manifest["arguments"])
    args.output_dir = str(cli.output_dir)
    args.distill_budget = int(cli.budget)
    args.distill_require_exact_budget = True
    args.distill_allow_coverage_fill = True
    args.distill_global_attractor_weight = 1.0
    args.distill_protected_core_ratio = 0.0
    if cli.profile == "distinct_core":
        args.distill_hard_matchability_core_ratio = 1.0
        args.distill_rescue_weight = 0.0
        args.distill_harmful_switch_weight = 0.0
    elif cli.profile == "core_reserve":
        args.distill_hard_matchability_core_ratio = 0.75
        args.distill_rescue_weight = 1.0
        args.distill_harmful_switch_weight = 0.0
    else:
        args.distill_hard_matchability_core_ratio = 0.75
        args.distill_rescue_weight = 1.0
        args.distill_harmful_switch_weight = 1.0

    source = torch.load(cli.source_state, map_location="cpu")
    statistics_payload = torch.load(
        cli.statistics_run / "landmark_statistics_full.pt",
        map_location="cpu",
    )
    statistics = statistics_payload.get("statistics", statistics_payload)
    prior_payload = torch.load(
        cli.statistics_run / "distill_global_attractor_prior.pt",
        map_location="cpu",
    )
    global_statistics = prior_payload.get("statistics", prior_payload)
    landmark_indices = torch.as_tensor(source["landmark_indices"]).reshape(-1)
    features = torch.as_tensor(source["landmark_features"])
    bank_xyz = torch.as_tensor(source["landmark_xyz"])
    raw_anchor_offset = torch.as_tensor(
        source.get("raw_anchor_offset", torch.zeros_like(bank_xyz))
    )
    observation_count = torch.as_tensor(
        source.get(
            "mvinit_observation_count",
            torch.zeros(landmark_indices.numel()),
        )
    )
    if any(
        torch.as_tensor(value).shape[:1] != landmark_indices.shape
        for value in statistics.values()
        if torch.is_tensor(value) and value.ndim > 0
    ):
        raise ValueError("Saved statistics do not align with the source bank")

    result = _distill_final_landmark_bank(
        cli.output_dir,
        landmark_indices,
        features,
        bank_xyz,
        raw_anchor_offset,
        statistics,
        args,
        dict(source.get("config", {})),
        observation_count,
        source.get("dustbin_score"),
        global_attractor_statistics=global_statistics,
    )
    distilled = torch.load(result["state_path"], map_location="cpu")
    selected_ids = torch.as_tensor(
        distilled["landmark_indices"]
    ).reshape(-1)
    source_id_to_local = {
        int(value): index
        for index, value in enumerate(landmark_indices.tolist())
    }
    selected_local = torch.tensor(
        [source_id_to_local[int(value)] for value in selected_ids.tolist()],
        dtype=torch.long,
    )
    torch.save(
        {
            "version": statistics_payload.get("version", 1),
            "split": statistics_payload.get("split", "train_only"),
            "train_camera_names_sha256": statistics_payload.get(
                "train_camera_names_sha256"
            ),
            "landmark_indices": selected_ids,
            "statistics": distilled["landmark_statistics"],
            "diagnostics": statistics_payload.get("diagnostics", {}),
        },
        cli.output_dir / "landmark_statistics_full.pt",
    )
    selected_global_statistics = {
        key: (
            torch.as_tensor(value)[selected_local]
            if torch.is_tensor(value)
            and value.ndim > 0
            and value.shape[0] == landmark_indices.numel()
            else value
        )
        for key, value in global_statistics.items()
    }
    torch.save(
        {
            "version": prior_payload.get("version", 1),
            "split": prior_payload.get("split", "train_only"),
            "train_camera_names_sha256": prior_payload.get(
                "train_camera_names_sha256"
            ),
            "landmark_indices": selected_ids,
            "statistics": selected_global_statistics,
            "diagnostics": prior_payload.get("diagnostics", {}),
        },
        cli.output_dir / "distill_global_attractor_prior.pt",
    )
    summary = {
        "profile": cli.profile,
        "budget": int(cli.budget),
        "statistics_run": str(cli.statistics_run.resolve()),
        "source_state": str(cli.source_state.resolve()),
        **result,
    }
    (cli.output_dir / "redistill_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
