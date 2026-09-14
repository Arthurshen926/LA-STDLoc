#!/usr/bin/env python3
"""Verify a unified-selector compatibility map against a frozen V3 map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common.hashing import sha256_file


LOCALIZATION_TENSORS = (
    "anchor_ids",
    "anchor_xyz",
    "anchor_features",
    "anchor_type",
    "source_primitive_ids",
    "track_cluster_ids",
    "fine_identity_ids",
    "dependency_group_ids",
    "coarse_dependency_group_ids",
    "source_dependency_group_ids",
)


def _load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _reason_ids(artifact: dict, reason: str) -> torch.Tensor:
    return torch.as_tensor(
        [
            row["candidate_universe_id"]
            for row in artifact["trace"]
            if row["primary_reason"] == reason
        ],
        dtype=torch.long,
    )


def _equal_set(first: torch.Tensor, second: torch.Tensor) -> bool:
    first = torch.as_tensor(first).long().sort().values
    second = torch.as_tensor(second).long().sort().values
    return bool(torch.equal(first, second))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-map", type=Path, required=True)
    parser.add_argument("--compatibility-map", type=Path, required=True)
    parser.add_argument("--unified-selection", type=Path, required=True)
    parser.add_argument("--reference-build-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = _load(args.reference_map)
    candidate = _load(args.compatibility_map)
    selection = _load(args.unified_selection)
    if selection.get("schema") != "lafgs_unified_sufficiency_selection":
        raise ValueError("unsupported unified selection artifact")
    if selection.get("policy") != "v3_compatibility":
        raise ValueError("compatibility audit refuses a behavior-changing policy")

    tensor_equal = {}
    for key in LOCALIZATION_TENSORS:
        if key not in reference and key not in candidate:
            continue
        tensor_equal[key] = bool(
            key in reference
            and key in candidate
            and torch.equal(torch.as_tensor(reference[key]), torch.as_tensor(candidate[key]))
        )

    provenance = reference.get("track_centric_reconstruction", {}).get(
        "selection_provenance"
    )
    provenance_exact = isinstance(provenance, dict)
    reason_set_equal = {}
    if provenance_exact:
        expected = {
            "precision": torch.as_tensor(
                provenance["track_core_universe_ids"]
            ).long(),
            "matching_completion": torch.cat(
                (
                    torch.as_tensor(
                        provenance["coverage_track_universe_ids"]
                    ).long(),
                    torch.as_tensor(
                        provenance["coverage_gaussian_universe_ids"]
                    ).long(),
                )
            ),
            "observability_completion": torch.cat(
                (
                    torch.as_tensor(provenance["pose_track_universe_ids"]).long(),
                    torch.as_tensor(
                        provenance["pose_gaussian_universe_ids"]
                    ).long(),
                )
            ),
        }
        reason_set_equal = {
            reason: _equal_set(_reason_ids(selection, reason), values)
            for reason, values in expected.items()
        }

    reason_count_equal = {}
    if args.reference_build_report is not None:
        build = json.loads(args.reference_build_report.read_text())
        expected_counts = {
            "precision": int(build["track_core"]["realized_count"]),
            "matching_completion": int(build["coverage"]["reserve_count"]),
            "observability_completion": int(
                build["pose_reserve"]["selected_count"]
            ),
        }
        reason_count_equal = {
            reason: int(_reason_ids(selection, reason).numel()) == count
            for reason, count in expected_counts.items()
        }
    selection_checks = [*reason_set_equal.values(), *reason_count_equal.values()]
    passed = bool(
        all(tensor_equal.values())
        and selection_checks
        and all(selection_checks)
    )
    report = {
        "schema": "lafgs_unified_sufficiency_compatibility_audit",
        "version": 1,
        "reference_map": str(args.reference_map.resolve()),
        "reference_map_sha256": sha256_file(args.reference_map),
        "compatibility_map": str(args.compatibility_map.resolve()),
        "compatibility_map_sha256": sha256_file(args.compatibility_map),
        "selection_artifact": str(args.unified_selection.resolve()),
        "selection_artifact_sha256": sha256_file(args.unified_selection),
        "localization_tensor_bitwise_equal": tensor_equal,
        "reference_selection_provenance_exact": provenance_exact,
        "selection_reason_sets_equal": reason_set_equal,
        "selection_reason_counts_equal": reason_count_equal,
        "selected_count": int(selection["selected_count"]),
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
