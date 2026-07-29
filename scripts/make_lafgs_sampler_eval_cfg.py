#!/usr/bin/env python3
"""Create a locked STDLoc config for the OOF dependency sampler."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--sampling-model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = Path(args.base_config).resolve()
    sampling_model = Path(args.sampling_model).resolve()
    output = Path(args.output).resolve()
    config = yaml.safe_load(base.read_text())
    sparse = config["sparse"]
    sparse.update(
        {
            "dependency_sampling_model_path": str(sampling_model),
            "dependency_max_iterations": 3000,
            "dependency_min_iterations": 500,
            "dependency_rescue_max_iterations": 30000,
            "dependency_rescue_inlier_ratio": 0.04,
            "dependency_guided_mixture": 0.8,
            "dependency_guided_rank_power": 1.0,
            "ransac_seed": 2026,
            "detect_num": 2048,
            "max_matches_per_keypoint": 0,
            "max_matches_per_landmark": 0,
            "unique_landmark_matches": False,
            "sparse_frontend": "ulfloc_native",
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = {
        "schema": "lafgs_locked_sampler_eval_config",
        "version": 1,
        "base_config": str(base),
        "base_config_sha256": _sha256(base),
        "sampling_model": str(sampling_model),
        "sampling_model_sha256": _sha256(sampling_model),
        "output_config": str(output),
        "output_config_sha256": _sha256(output),
        "protocol": {
            "full_resolution_native_superpoint": True,
            "detect_num": 2048,
            "landmark_match_cap": 0,
            "single_sparse_pose": True,
            "dense_refinement": False,
            "oof_mapping_model_for_test": "all",
        },
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
