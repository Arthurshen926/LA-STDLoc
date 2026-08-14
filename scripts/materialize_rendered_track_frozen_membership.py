#!/usr/bin/env python3
"""Transfer a frozen source-Track selection onto repaired Track children.

This is the E+R ablation for the source-image-free pipeline.  It preserves the
selected source-Track identity set from V1.1 and chooses, in the already frozen
repaired-candidate quality order, at most one broad child for each source
Track.  It does not run the sufficiency selector, fill missing identities, or
introduce Gaussian anchors.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import torch

from common.hashing import sha256_file


_SOURCE_PATHS = (
    "scripts/materialize_rendered_track_frozen_membership.py",
    "common/hashing.py",
)


def _producer_identity() -> dict:
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("frozen-membership producer worktree must be clean")
    return {
        "git_commit": commit,
        "worktree_clean": True,
        "source_sha256": {
            relative: sha256_file(repository / relative) for relative in _SOURCE_PATHS
        },
        "torch_version": torch.__version__,
    }


def _require_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"{label} SHA differs: expected {expected}, got {actual}")
    return actual


def _atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
        if reloaded.get("schema") != payload.get("schema"):
            raise RuntimeError("frozen-membership map did not reload")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def transfer_frozen_membership(
    source_selection: dict, repaired_payload: dict, candidate_map: dict
) -> tuple[dict, dict]:
    source_tracks = torch.as_tensor(source_selection["track_cluster_ids"]).long()
    candidate_tracks = torch.as_tensor(candidate_map["track_cluster_ids"]).long()
    source_by_child = torch.as_tensor(
        repaired_payload["tracks"]["source_track_index"]
    ).long()
    if (
        source_tracks.ndim != 1
        or source_tracks.unique().numel() != source_tracks.numel()
    ):
        raise ValueError("source selection Track IDs must be unique 1-D rows")
    if (
        candidate_tracks.ndim != 1
        or candidate_tracks.unique().numel() != candidate_tracks.numel()
    ):
        raise ValueError("repaired candidate Track IDs must be unique 1-D rows")
    if candidate_tracks.numel() and (
        int(candidate_tracks.min()) < 0
        or int(candidate_tracks.max()) >= int(source_by_child.numel())
    ):
        raise ValueError("candidate Track ID is outside repaired child registry")

    # Candidate rows were materialized in descending frozen Track quality.
    # First occurrence is therefore the deterministic best broad child.
    best_row_by_source: dict[int, int] = {}
    for row, source in enumerate(source_by_child[candidate_tracks].tolist()):
        best_row_by_source.setdefault(int(source), row)
    selected_rows = []
    retained_sources = []
    missing_sources = []
    for source in source_tracks.tolist():
        row = best_row_by_source.get(int(source))
        if row is None:
            missing_sources.append(int(source))
        else:
            selected_rows.append(int(row))
            retained_sources.append(int(source))
    rows = torch.as_tensor(selected_rows, dtype=torch.long)
    candidate_count = int(candidate_tracks.numel())
    output = dict(candidate_map)
    for key, value in candidate_map.items():
        if torch.is_tensor(value) and value.ndim and value.shape[0] == candidate_count:
            output[key] = value[rows].clone()
    output["anchor_ids"] = torch.arange(rows.numel(), dtype=torch.long)
    selected_children = candidate_tracks[rows]
    output["track_cluster_ids"] = selected_children.clone()
    output["base_anchor_count"] = 0
    output["micro_anchor_count"] = int(rows.numel())
    output["canonical_anchor_count"] = int(rows.numel())
    output["track_centric_reconstruction"] = {
        "track_indices": selected_children.clone(),
        "base_canonical_rows": torch.empty(0, dtype=torch.long),
    }
    output["rendered_track_frozen_membership"] = {
        "schema": "lafgs_rendered_track_frozen_membership",
        "version": 1,
        "source_selection_count": int(source_tracks.numel()),
        "retained_source_count": len(retained_sources),
        "missing_source_count": len(missing_sources),
        "retained_source_track_ids": torch.as_tensor(retained_sources).long(),
        "missing_source_track_ids": torch.as_tensor(missing_sources).long(),
        "selected_child_track_ids": selected_children.clone(),
        "child_policy": "highest_frozen_repaired_broad_quality_one_per_source",
        "runs_sufficiency_selector": False,
        "uses_gaussian_anchors": False,
    }
    diagnostics = {
        "source_selection_count": int(source_tracks.numel()),
        "retained_source_count": len(retained_sources),
        "missing_source_count": len(missing_sources),
        "retention_fraction": len(retained_sources)
        / max(int(source_tracks.numel()), 1),
        "candidate_count": candidate_count,
        "selected_child_count": int(rows.numel()),
        "missing_source_track_ids": missing_sources,
    }
    return output, diagnostics


def run(args: argparse.Namespace) -> dict:
    identity = _producer_identity()
    inputs = {
        "source_selection": args.source_selection.resolve(),
        "repaired_track_payload": args.repaired_track_payload.resolve(),
        "repaired_candidate_map": args.repaired_candidate_map.resolve(),
    }
    expected = {
        "source_selection": args.expected_source_selection_sha256,
        "repaired_track_payload": args.expected_repaired_track_payload_sha256,
        "repaired_candidate_map": args.expected_repaired_candidate_map_sha256,
    }
    input_sha256 = {
        label: _require_sha(path, expected[label], label)
        for label, path in inputs.items()
    }
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    source = torch.load(
        inputs["source_selection"], map_location="cpu", weights_only=False
    )
    repaired = torch.load(
        inputs["repaired_track_payload"], map_location="cpu", weights_only=False
    )
    candidate = torch.load(
        inputs["repaired_candidate_map"], map_location="cpu", weights_only=False
    )
    if repaired.get("rendered_rgb_only") is not True:
        raise ValueError("repaired payload is not rendered-RGB-only")
    if (
        repaired.get("support_repair", {}).get("forbids_cross_source_track_merge")
        is not True
    ):
        raise ValueError("repaired payload lacks the source-component boundary")
    if int(source.get("base_anchor_count", 0)) != 0:
        raise ValueError("frozen membership accepts a Track-only source selection")
    output, diagnostics = transfer_frozen_membership(source, repaired, candidate)
    map_path = args.output_dir / "frozen_membership_anchor_map.pt"
    _atomic_save(output, map_path)
    if _producer_identity() != identity:
        raise RuntimeError("frozen-membership producer identity changed")
    for label, path in inputs.items():
        _require_sha(path, input_sha256[label], label)
    report = {
        "schema": "lafgs_rendered_track_frozen_membership_report",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "producer_identity": identity,
        "diagnostics": diagnostics,
        "inputs": {label: str(path) for label, path in inputs.items()},
        "input_sha256": input_sha256,
        "output": str(map_path.resolve()),
        "output_sha256": sha256_file(map_path),
    }
    _atomic_json(report, args.output_dir / "frozen_membership_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-selection", type=Path, required=True)
    parser.add_argument("--expected-source-selection-sha256", required=True)
    parser.add_argument("--repaired-track-payload", type=Path, required=True)
    parser.add_argument("--expected-repaired-track-payload-sha256", required=True)
    parser.add_argument("--repaired-candidate-map", type=Path, required=True)
    parser.add_argument("--expected-repaired-candidate-map-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
