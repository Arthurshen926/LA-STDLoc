#!/usr/bin/env python3
"""Write the immutable input lineage for a pair-policy full-pipeline gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.hashing import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-state", type=Path, required=True)
    parser.add_argument("--track-payload", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--visibility-cache", type=Path, required=True)
    parser.add_argument("--gaussian-ply", type=Path, required=True)
    parser.add_argument("--scene-calibration", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        name: Path(value).resolve()
        for name, value in (
            ("base_state", args.base_state),
            ("track_payload", args.track_payload),
            ("query_cache", args.query_cache),
            ("visibility_cache", args.visibility_cache),
            ("gaussian_ply", args.gaussian_ply),
            ("scene_calibration", args.scene_calibration),
            ("config", args.config),
        )
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    calibration = json.loads(paths["scene_calibration"].read_text())
    if calibration.get("sources", {}).get("uses_test_queries") is not False:
        raise ValueError("Frozen calibration must be mapping-only")
    report = {
        "schema": "lafgs_pair_policy_full_pipeline_lineage",
        "version": 1,
        "uses_test_queries": False,
        "single_factor": "camera_pair_policy",
        "mapping_keypoints": 1024,
        "canonical_policy": "rebuild_from_frozen_stage_a_with_track_budget_zero",
        "canonical_reused": False,
        "canonical_function_graph_reused": False,
        "canonical_provenance_reused": False,
        "canonical_teacher_reused": False,
        "compact_function_graph_reused": False,
        "compact_provenance_reused": False,
        "compact_teacher_reused": False,
        "scene_calibration_frozen": True,
        "resolved_thresholds": calibration["parameters"],
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
        "forbidden_inputs": ["test split", "old Track identity graph", "K=2048 cache"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
