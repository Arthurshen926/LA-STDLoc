#!/usr/bin/env python3
"""Write a content-addressed manifest for one immutable V3 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from common.artifact_contract import anchor_registry, git_worktree_state
from common.config import load_mainline_config
from common.hashing import sha256_file


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("artifact must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("artifact name must not be empty")
    return name, Path(raw_path).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--config", type=Path, default="configs/paper_mainline.yaml")
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        type=_named_path,
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--effective-nms-radius",
        type=int,
        default=4,
        help="Observed historical frontend behavior; V3 used SuperPoint default 4.",
    )
    parser.add_argument(
        "--declared-nms-radius-at-run",
        type=int,
        help="Optional historical config value when the run predates the NMS fix.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_mainline_config(args.config)
    map_path = args.map.expanduser().resolve()
    state = torch.load(map_path, map_location="cpu", weights_only=False)
    if state.get("schema") != "lafgs_materialized_anchor_map":
        raise ValueError("unsupported anchor map schema")
    paths = {"map": map_path, **dict(args.artifact)}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    anchor_count = int(torch.as_tensor(state["anchor_ids"]).numel())
    manifest = {
        "schema": "lafgs_v3_frozen_baseline",
        "version": 1,
        "immutable": True,
        "config": config.manifest(),
        "producer_git": git_worktree_state(Path(__file__).resolve().parents[1]),
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sorted(paths.items())
        },
        "anchor_registry": anchor_registry(state),
        "localization_tensor_sha256": {
            key: _tensor_sha256(state[key])
            for key in ("anchor_ids", "anchor_xyz", "anchor_features")
        },
        "counts": {
            "anchor": anchor_count,
            "track": int((torch.as_tensor(state["anchor_type"]) == 1).sum()),
            "gaussian_fallback": int(
                (torch.as_tensor(state["anchor_type"]) == 0).sum()
            ),
        },
        "frontend": {
            "resolved_declared_nms_radius": int(
                config.values["deployment"]["nms"]
            ),
            "declared_nms_radius_at_run": (
                int(args.declared_nms_radius_at_run)
                if args.declared_nms_radius_at_run is not None
                else int(config.values["deployment"]["nms"])
            ),
            "effective_nms_radius": int(args.effective_nms_radius),
            "historical_mismatch": bool(
                (
                    int(args.declared_nms_radius_at_run)
                    if args.declared_nms_radius_at_run is not None
                    else int(config.values["deployment"]["nms"])
                )
                != int(args.effective_nms_radius)
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
