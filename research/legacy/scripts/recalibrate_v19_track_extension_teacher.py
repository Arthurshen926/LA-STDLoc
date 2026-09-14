#!/usr/bin/env python3
"""Reapply the fail-closed V19 tier selector to a frozen evidence graph."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from scripts.evaluate_v19_track_extension_teacher import _select_tiers


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--anchor-map", type=Path, required=True)
    parser.add_argument("--mapping-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    evidence = torch.load(args.evidence, map_location="cpu", weights_only=False)
    state = torch.load(args.anchor_map, map_location="cpu", weights_only=False)
    provenance = torch.load(
        args.mapping_provenance, map_location="cpu", weights_only=False
    )
    if not (
        evidence.get("schema") == "lafgs_v19_track_extension_teacher_validation"
        and evidence.get("uses_test_queries") is False
        and evidence.get("feedback_enters_track_registry") is False
        and evidence.get("reference_source")
        == "mapping_observation_track_membership"
        and evidence.get("inputs", {}).get("anchor_map_sha256")
        == sha256_file(args.anchor_map)
        and evidence.get("inputs", {}).get("mapping_provenance_sha256")
        == sha256_file(args.mapping_provenance)
    ):
        raise ValueError("V19 frozen evidence lineage differs")
    observations = state["projective_anchor_observations"]
    query_indices = torch.as_tensor(observations["query_indices"]).long()
    evaluation_rows = torch.as_tensor(evidence["evaluation_rows"]).long()
    mapping_families = torch.as_tensor(
        provenance["mapping_view_family_ids"]
    ).long()
    evaluation_families = mapping_families[query_indices[evaluation_rows]]
    equivalence = torch.as_tensor(
        state.get("fine_identity_ids", torch.arange(state["anchor_xyz"].shape[0]))
    ).long()
    ground_truth = torch.as_tensor(evidence["ground_truth_anchor_rows"]).long()
    calibration_count = int(evidence["row_counts"]["threshold_calibration"])
    calibration_rows = torch.arange(calibration_count)
    validation_rows = torch.arange(calibration_count, evaluation_rows.numel())
    selected, trial_count = _select_tiers(
        graph=evidence["candidate_graph"],
        consensus=evidence["consensus"],
        equivalence=equivalence,
        ground_truth=ground_truth,
        calibration_rows=calibration_rows,
        validation_rows=validation_rows,
        evaluation_families=evaluation_families,
    )
    artifact = {
        **evidence,
        "version": 2,
        "selection_uses_validation": False,
        "authorization_uses_wilson_lower_bound": True,
        "authorization_requires_independent_mapping_families": True,
        "candidate_trial_count": trial_count,
        "selected_tiers": selected,
        "evaluation_view_family_ids": evaluation_families,
        "recalibration": {
            "source_evidence": str(args.evidence.resolve()),
            "source_evidence_sha256": sha256_file(args.evidence),
            "reason": "conservative_missing_track_support_and_family_aware_authorization",
        },
    }
    _atomic_save(artifact, args.output.resolve())
    report = {
        key: value
        for key, value in artifact.items()
        if key
        not in {
            "selected_tiers",
            "evaluation_rows",
            "ground_truth_anchor_rows",
            "evaluation_view_family_ids",
            "candidate_graph",
            "consensus",
        }
    }
    report["selected_tiers"] = {
        name: {key: value for key, value in item.items() if key != "truth"}
        for name, item in selected.items()
    }
    report["output"] = str(args.output.resolve())
    report["output_sha256"] = sha256_file(args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
