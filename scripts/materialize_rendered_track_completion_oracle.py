#!/usr/bin/env python3
"""Materialize the full broad-Track pool as a mapping-only completion oracle.

This artifact is deliberately not a deployable selector result.  It asks only
whether the already reconstructed, support-repaired broad Track universe has
enough localization headroom to justify a later deficit-targeted lazy
completion method.  No test query is loaded and no scientific gate is emitted.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import torch

from common.hashing import sha256_file
from common.tensor_identity import recursive_bitwise_equal


_SOURCE_PATHS = ("scripts/materialize_rendered_track_completion_oracle.py",)
_TRACK_TOPOLOGY_FIELDS = (
    "anchor_xyz",
    "source_primitive_ids",
    "track_cluster_ids",
    "anchor_type",
    "dependency_group_ids",
    "coarse_dependency_group_ids",
    "fine_identity_ids",
    "parent_source_track_ids",
    "repair_child_index",
    "repair_parent_child_count",
)


def _producer_identity() -> dict[str, Any]:
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
        raise RuntimeError("completion-oracle producer worktree must be clean")
    return {
        "git_commit": commit,
        "worktree_clean": True,
        "torch_version": torch.__version__,
        "source_sha256": {
            relative: sha256_file(repository / relative) for relative in _SOURCE_PATHS
        },
    }


def _require_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != str(expected).strip().lower():
        raise ValueError(f"{label} SHA differs: expected {expected}, got {actual}")
    return actual


def _load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu", weights_only=False)


def _atomic_save(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        reloaded = _load(temporary)
        if not recursive_bitwise_equal(payload, reloaded):
            raise RuntimeError("temporary completion oracle did not reload exactly")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if json.loads(temporary.read_text()) != payload:
            raise RuntimeError("temporary completion report did not reload exactly")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    candidate_path = args.candidate_map.resolve()
    selected_path = args.selected_map.resolve()
    statistics_path = args.mapping_statistics.resolve()
    inputs = {
        "candidate_map": candidate_path,
        "selected_map": selected_path,
        "mapping_statistics": statistics_path,
    }
    expected = {
        "candidate_map": args.expected_candidate_map_sha256,
        "selected_map": args.expected_selected_map_sha256,
        "mapping_statistics": args.expected_mapping_statistics_sha256,
    }
    input_sha256 = {
        label: _require_sha(path, expected[label], label)
        for label, path in inputs.items()
    }
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise FileExistsError(output if output.exists() else report_path)
    if output in inputs.values():
        raise ValueError("completion oracle cannot overwrite an input")

    candidate = _load(candidate_path)
    selected = _load(selected_path)
    statistics = _load(statistics_path)
    if (
        candidate.get("schema") != "lafgs_materialized_anchor_map"
        or selected.get("schema") != "lafgs_materialized_anchor_map"
    ):
        raise ValueError("completion inputs are not anchor maps")
    if (
        statistics.get("uses_source_mapping_rgb") is not False
        or statistics.get("uses_test_queries") is not False
        or statistics.get("schema")
        != "lafgs_rendered_track_full_mapping_loo_statistics"
    ):
        raise ValueError("completion oracle requires mapping-only LOO statistics")
    missing = [key for key in _TRACK_TOPOLOGY_FIELDS if key not in candidate]
    if missing:
        raise ValueError(f"candidate map lacks Track topology fields: {missing}")
    candidate_count = int(torch.as_tensor(candidate["anchor_ids"]).numel())
    selected_count = int(torch.as_tensor(selected["anchor_ids"]).numel())
    if candidate_count <= selected_count:
        raise ValueError(
            "completion candidate universe does not expand the selected map"
        )
    if not torch.equal(
        torch.as_tensor(candidate["anchor_ids"]).long(),
        torch.arange(candidate_count),
    ):
        raise ValueError("completion candidate rows are not canonical and contiguous")
    if not bool(
        (torch.as_tensor(candidate["anchor_type"]).long() == 1).all()
    ) or not bool(
        (torch.as_tensor(candidate["source_primitive_ids"]).long() == -1).all()
    ):
        raise ValueError("completion universe is not pure ray-triangulated Tracks")
    candidate_tracks = torch.as_tensor(candidate["track_cluster_ids"]).long()
    selected_tracks = torch.as_tensor(selected["track_cluster_ids"]).long()
    if (
        candidate_tracks.unique().numel() != candidate_count
        or selected_tracks.unique().numel() != selected_count
    ):
        raise ValueError("completion Track identities are not unique")
    row_by_track = {
        int(track): row for row, track in enumerate(candidate_tracks.tolist())
    }
    if not set(selected_tracks.tolist()) <= set(candidate_tracks.tolist()):
        raise ValueError("selected Track set is not a subset of the candidate universe")
    selected_candidate_rows = torch.as_tensor(
        [row_by_track[int(track)] for track in selected_tracks.tolist()]
    ).long()
    for field in _TRACK_TOPOLOGY_FIELDS:
        if field == "track_cluster_ids" or field not in selected:
            continue
        if not recursive_bitwise_equal(
            torch.as_tensor(candidate[field])[selected_candidate_rows],
            torch.as_tensor(selected[field]),
        ):
            raise ValueError(
                f"selected and candidate Track topology differs for {field}"
            )
    selected_source_dependency = torch.as_tensor(
        selected.get(
            "source_dependency_group_ids",
            torch.full((selected_count,), -1, dtype=torch.long),
        )
    )
    if not bool((selected_source_dependency == -1).all()):
        raise ValueError("pure Track selection has nonempty Gaussian dependencies")

    query_rows = list(statistics.get("queries", ()))
    task_failure_count = sum(
        float(row["te_cm"]) >= float(args.task_translation_cm)
        or float(row["ae_deg"]) >= float(args.task_rotation_deg)
        for row in query_rows
    )
    catastrophic_count = sum(float(row["te_cm"]) >= 100.0 for row in query_rows)
    identity = _producer_identity()
    oracle = dict(candidate)
    oracle["anchor_ids"] = torch.arange(candidate_count, dtype=torch.long)
    oracle["source_dependency_group_ids"] = torch.full(
        (candidate_count,), -1, dtype=torch.long
    )
    oracle["provenance"] = {
        **candidate.get("provenance", {}),
        "rendered_track_completion_oracle": {
            "method_role": "mapping_only_full_broad_track_upper_bound_not_deployable",
            "selected_map": str(selected_path),
            "mapping_statistics": str(statistics_path),
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
            "selection_policy": "all_support_repaired_broad_tracks",
            "input_sha256": input_sha256,
            "producer_identity": identity,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save(oracle, output)
    for label, path in inputs.items():
        _require_sha(path, input_sha256[label], label)
    report = {
        "schema": "lafgs_rendered_track_completion_upper_bound",
        "version": 1,
        "method_role": "mapping_only_oracle_not_a_gate_not_deployable",
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "inputs": {label: str(path) for label, path in inputs.items()},
        "input_sha256": input_sha256,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "selected_anchor_count": selected_count,
        "oracle_anchor_count": candidate_count,
        "added_anchor_count": candidate_count - selected_count,
        "control_task_failure_count": int(task_failure_count),
        "control_catastrophic_count": int(catastrophic_count),
        "producer_identity": identity,
    }
    _atomic_json(report, report_path)
    if _producer_identity() != identity:
        raise RuntimeError("completion-oracle producer identity changed")
    for label, path in inputs.items():
        _require_sha(path, input_sha256[label], label)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-map", type=Path, required=True)
    parser.add_argument("--expected-candidate-map-sha256", required=True)
    parser.add_argument("--selected-map", type=Path, required=True)
    parser.add_argument("--expected-selected-map-sha256", required=True)
    parser.add_argument("--mapping-statistics", type=Path, required=True)
    parser.add_argument("--expected-mapping-statistics-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-translation-cm", type=float, default=5.0)
    parser.add_argument("--task-rotation-deg", type=float, default=5.0)
    args = parser.parse_args()
    print(json.dumps(materialize(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
