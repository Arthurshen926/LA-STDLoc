#!/usr/bin/env python3
"""Materialize the preregistered C2 budget control from a frozen R0 map.

The V1.4 selector emits its Precision Core in deterministic quality order
before Matching and Observability Completion.  C2 takes an exact prefix that
is no longer than that Precision Core, preserving every source tensor row and
changing only map cardinality.  It is an attribution control, not a proposed
deployment map.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import torch

from common.hashing import sha256_file
from map_learning.metric import SharedLowRankMetric


_SOURCE_PATHS = ("scripts/materialize_rendered_track_budget_control.py",)


def _identity() -> dict:
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
        raise RuntimeError("budget-control producer worktree must be clean")
    return {
        "git_commit": commit,
        "worktree_clean": True,
        "source_sha256": {
            path: sha256_file(repository / path) for path in _SOURCE_PATHS
        },
        "torch_version": torch.__version__,
    }


def _require_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"{label} SHA differs: expected {expected}, got {actual}")
    return actual


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
        if reloaded.get("schema") != payload.get("schema"):
            raise RuntimeError("temporary budget artifact did not reload")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def budget_prefix_map(state: dict, selection: dict, target_count: int) -> dict:
    """Return the exact C2 prefix after validating selector ordering."""

    if selection.get("schema") != "lafgs_unified_sufficiency_selection":
        raise ValueError("unexpected selection artifact schema")
    source_count = int(torch.as_tensor(state["anchor_ids"]).numel())
    target_count = int(target_count)
    if not 0 < target_count <= source_count:
        raise ValueError("target count must be inside the source map")
    selected = torch.as_tensor(selection.get("selected_universe_ids")).long()
    tracks = torch.as_tensor(state.get("track_cluster_ids")).long()
    if selected.shape != tracks.shape or not torch.equal(selected, tracks):
        raise ValueError("R0 map rows do not follow the frozen selector trace")
    reasons = selection.get("primary_selection_reasons")
    if not isinstance(reasons, list) or len(reasons) != source_count:
        raise ValueError("selector reasons do not align with R0 rows")
    if any(reason != "precision" for reason in reasons[:target_count]):
        raise ValueError("C2 target extends beyond the ordered Precision Core")
    precision_count = int(
        selection.get("reports", {}).get("precision", {}).get("realized_count", -1)
    )
    if target_count > precision_count:
        raise ValueError("C2 target exceeds the reported Precision Core")

    output = {}
    for key, value in state.items():
        if (
            torch.is_tensor(value)
            and value.ndim >= 1
            and value.shape[0] == source_count
        ):
            output[key] = value[:target_count].clone()
        else:
            output[key] = value
    if not torch.equal(
        torch.as_tensor(output["anchor_ids"]), torch.arange(target_count)
    ):
        raise ValueError("R0 prefix anchor IDs are not contiguous")
    output["canonical_anchor_count"] = target_count
    output["micro_anchor_count"] = target_count
    output["requested_micro_anchor_budget"] = target_count
    output["base_anchor_count"] = 0
    reconstruction = dict(state.get("track_centric_reconstruction", {}))
    if not torch.equal(torch.as_tensor(reconstruction.get("track_indices")), tracks):
        raise ValueError("Track reconstruction rows differ from the source map")
    reconstruction["track_indices"] = tracks[:target_count].clone()
    reconstruction["base_canonical_rows"] = torch.empty(0, dtype=torch.long)
    output["track_centric_reconstruction"] = reconstruction
    output["provenance"] = {
        **state.get("provenance", {}),
        "rendered_track_attribution_C2": {
            "policy": "v14_selector_precision_order_prefix",
            "source_anchor_count": source_count,
            "target_anchor_count": target_count,
            "attribution_only": True,
        },
    }
    return output


def run(args: argparse.Namespace) -> dict:
    source_map = args.source_map.resolve()
    selection_path = args.selection.resolve()
    source_sha = _require_sha(source_map, args.expected_source_map_sha256, "source map")
    selection_sha = _require_sha(
        selection_path, args.expected_selection_sha256, "selection"
    )
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    state = torch.load(source_map, map_location="cpu", weights_only=False)
    selection = torch.load(selection_path, map_location="cpu", weights_only=False)
    output = budget_prefix_map(state, selection, args.target_count)
    identity = _identity()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    map_path = args.output_dir / "budget_control_anchor_map.pt"
    metric_path = args.output_dir / "budget_control_identity_metric.pt"
    _atomic_torch_save(output, map_path)
    descriptor_dim = int(torch.as_tensor(output["anchor_features"]).shape[1])
    metric = SharedLowRankMetric(
        descriptor_dim=descriptor_dim, rank=1, max_residual_norm=0.0
    )
    with torch.no_grad():
        for parameter in metric.parameters():
            parameter.zero_()
    metric_state = {
        "schema": "lafgs_shared_metric_state",
        "version": 1,
        "landmark_indices": torch.arange(args.target_count, dtype=torch.long),
        "metric_config": metric.export_config(),
        "metric_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in metric.state_dict().items()
        },
        "map_path": str(map_path.resolve()),
        "map_sha256": sha256_file(map_path),
        "step": 0,
        "protocol": "rendered_track_attribution_C2_budget_control_identity",
    }
    _atomic_torch_save(metric_state, metric_path)
    if _identity() != identity:
        raise RuntimeError("budget-control producer identity changed")
    _require_sha(source_map, source_sha, "source map")
    _require_sha(selection_path, selection_sha, "selection")
    report = {
        "schema": "lafgs_rendered_track_attribution_budget_control",
        "version": 1,
        "arm": "C2",
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "policy": "v14_selector_precision_order_prefix",
        "target_count": int(args.target_count),
        "source_count": int(torch.as_tensor(state["anchor_ids"]).numel()),
        "producer_identity": identity,
        "inputs": {
            "source_map": str(source_map),
            "selection": str(selection_path),
        },
        "input_sha256": {
            "source_map": source_sha,
            "selection": selection_sha,
        },
        "outputs": {
            "map": str(map_path.resolve()),
            "metric": str(metric_path.resolve()),
        },
        "output_sha256": {
            "map": sha256_file(map_path),
            "metric": sha256_file(metric_path),
        },
    }
    report_path = args.output_dir / "budget_control_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-map", type=Path, required=True)
    parser.add_argument("--expected-source-map-sha256", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--target-count", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
