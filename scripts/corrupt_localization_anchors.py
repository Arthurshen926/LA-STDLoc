#!/usr/bin/env python3

import argparse
import json
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(
        description="Apply controlled localization-only anchor corruption"
    )
    parser.add_argument("--source_state", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--magnitude_m", type=float, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not 0.0 < float(args.fraction) < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    if float(args.magnitude_m) <= 0.0:
        raise ValueError("magnitude_m must be positive")

    source_path = Path(args.source_state).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state = torch.load(source_path, map_location="cpu", weights_only=False)
    xyz = torch.as_tensor(state["landmark_xyz"], dtype=torch.float32).clone()
    count = xyz.shape[0]
    corrupted_count = max(int(round(float(args.fraction) * count)), 1)
    generator = torch.Generator().manual_seed(int(args.seed))
    selected = torch.randperm(count, generator=generator)[:corrupted_count]
    direction = F.normalize(
        torch.randn(corrupted_count, 3, generator=generator), dim=1
    )
    displacement = direction * float(args.magnitude_m)
    xyz[selected] += displacement
    state["landmark_xyz"] = xyz
    state["raw_anchor_offset"] = torch.zeros_like(xyz)
    config = dict(state.get("config", {}))
    config["controlled_anchor_corruption"] = {
        "enabled": True,
        "fraction": float(args.fraction),
        "magnitude_m": float(args.magnitude_m),
        "direction": "isotropic_random",
        "seed": int(args.seed),
        "rgb_map_modified": False,
    }
    state["config"] = config
    diagnostics = dict(state.get("diagnostics", {}))
    diagnostics["controlled_anchor_corruption_count"] = int(corrupted_count)
    state["diagnostics"] = diagnostics
    state_path = output_dir / "corrupted_lafgs_map_state.pt"
    torch.save(state, state_path)
    with (output_dir / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(state["landmark_indices"], handle)
    torch.save(
        {
            "version": 1,
            "landmark_indices": state["landmark_indices"],
            "fixed_bank": True,
            "one_time_landmark_distillation": False,
            "feature_dim": int(state["landmark_features"].shape[1]),
            "state_path": str(state_path),
        },
        output_dir / "landmark_meta.pt",
    )
    torch.save(
        {
            "version": 1,
            "corrupted_mask": torch.zeros(count, dtype=torch.bool).scatter_(
                0, selected, True
            ),
            "selected_relative_indices": selected,
            "displacement": displacement,
            "fraction": float(args.fraction),
            "magnitude_m": float(args.magnitude_m),
            "seed": int(args.seed),
        },
        output_dir / "corruption_labels.pt",
    )
    report = {
        "source_state": str(source_path),
        "output_state": str(state_path),
        "landmark_count": int(count),
        "corrupted_count": int(corrupted_count),
        "fraction": float(args.fraction),
        "magnitude_m": float(args.magnitude_m),
        "rgb_map_modified": False,
    }
    (output_dir / "corruption_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
