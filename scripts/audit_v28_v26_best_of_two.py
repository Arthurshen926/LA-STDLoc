"""Materialize a strict V26 T0/T1 pose-selection upper-bound audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.hashing import sha256_file
from map_learning.v28_best_of_two_oracle import audit_best_of_two
from scripts.fixed_pair_matcher_ceiling_common import atomic_json_save_fresh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-pass-results", type=Path, required=True)
    parser.add_argument("--refinement-results", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first_path = args.first_pass_results.resolve()
    refinement_path = args.refinement_results.resolve()
    first_hash = sha256_file(first_path)
    refinement_hash = sha256_file(refinement_path)
    audit = audit_best_of_two(
        first_pass_records=json.loads(first_path.read_text()),
        refinement_records=json.loads(refinement_path.read_text()),
    )
    payload = {
        "schema": "lafgs_v28_v26_best_of_two_oracle",
        "version": 1,
        "scene": args.scene,
        "uses_test_ground_truth": True,
        "diagnostic_only": True,
        "online_deployment_authorized": False,
        "selection_objective": "minimize_max_te_over_5cm_re_over_5deg_keep_t0_on_tie",
        "inputs": {
            "first_pass_results": {
                "path": str(first_path),
                "sha256": first_hash,
            },
            "refinement_results": {
                "path": str(refinement_path),
                "sha256": refinement_hash,
            },
        },
        **audit,
    }
    if sha256_file(first_path) != first_hash or sha256_file(refinement_path) != refinement_hash:
        raise RuntimeError("V28 oracle input changed during audit")
    output = atomic_json_save_fresh(payload, args.output.resolve())
    print(json.dumps({"output": str(output), **audit["best_of_two_oracle"]}, indent=2))


if __name__ == "__main__":
    main()

