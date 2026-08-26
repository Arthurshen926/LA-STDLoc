#!/usr/bin/env python3
"""Run the fail-closed V7 mainline; P0 supports identity no-op only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import torch

from common.v7_contracts import (
    V7_P0_REPORT_SCHEMA,
    audit_formal_import_graph,
    compare_deployment_contracts,
    compare_query_results,
    load_v7_config,
    sha256_file,
    tensor_tree_equal,
    validate_compact_map,
)


def _require_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected.lower():
        raise ValueError(f"{label} SHA256 differs")
    return actual


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def run_p0(args: argparse.Namespace) -> dict:
    root = Path(__file__).resolve().parents[1]
    load_v7_config(args.config)
    import_audit = audit_formal_import_graph(
        root=root,
        entrypoint=Path(__file__),
        allowlist_path=args.formal_source_allowlist,
    )
    source_map = args.baseline_map.resolve()
    source_metric = args.baseline_metric.resolve()
    map_sha = _require_sha(source_map, args.expected_baseline_map_sha256, "baseline map")
    metric_sha = _require_sha(source_metric, args.expected_baseline_metric_sha256, "baseline metric")
    source_state = torch.load(source_map, map_location="cpu", weights_only=False)
    map_summary = validate_compact_map(source_state)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    output_map = args.output_dir / "compact_map.pt"
    output_metric = args.output_dir / "identity_metric.pt"
    _atomic_copy(source_map, output_map)
    _atomic_copy(source_metric, output_metric)
    output_state = torch.load(output_map, map_location="cpu", weights_only=False)
    if not tensor_tree_equal(source_state, output_state):
        raise RuntimeError("V7 P0 no-op changed compact map tensors")
    if sha256_file(output_map) != map_sha or sha256_file(output_metric) != metric_sha:
        raise RuntimeError("V7 P0 no-op changed serialized baseline artifacts")

    query_parity = None
    deployment_parity = None
    if (args.reference_results is None) != (args.candidate_results is None):
        raise ValueError("reference and candidate results must be supplied together")
    if args.reference_results is not None:
        reference_results = args.reference_results.resolve()
        candidate_results = args.candidate_results.resolve()
        _require_sha(reference_results, args.expected_reference_results_sha256, "reference results")
        _require_sha(candidate_results, args.expected_candidate_results_sha256, "candidate results")
        query_parity = compare_query_results(
            json.loads(reference_results.read_text()),
            json.loads(candidate_results.read_text()),
        )
        if query_parity["query_count"] != int(args.expected_query_count):
            raise ValueError("P0 query count differs from the preregistered count")
        if (args.reference_deployment_contract is None) != (
            args.candidate_deployment_contract is None
        ):
            raise ValueError("deployment contracts must be supplied together")
        if args.reference_deployment_contract is not None:
            deployment_parity = compare_deployment_contracts(
                json.loads(args.reference_deployment_contract.read_text()),
                json.loads(args.candidate_deployment_contract.read_text()),
            )

    report = {
        "schema": V7_P0_REPORT_SCHEMA,
        "version": 1,
        "phase": "P0",
        "status": "PASS",
        "uses_source_mapping_rgb": False,
        "uses_test_queries_for_map_updates": False,
        "map_action": "identity_noop",
        "input": {
            "map": str(source_map),
            "map_sha256": map_sha,
            "metric": str(source_metric),
            "metric_sha256": metric_sha,
        },
        "output": {
            "map": str(output_map.resolve()),
            "map_sha256": sha256_file(output_map),
            "metric": str(output_metric.resolve()),
            "metric_sha256": sha256_file(output_metric),
        },
        "map_tensor_parity": {**map_summary, "exact": True},
        "query_parity": query_parity,
        "deployment_parity": deployment_parity,
        "formal_import_graph": import_audit,
    }
    (args.output_dir / "p0_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("p0",), default="p0")
    parser.add_argument("--config", type=Path, default=Path("configs/v7_safe_closed_loop.yaml"))
    parser.add_argument(
        "--formal-source-allowlist",
        type=Path,
        default=Path("configs/v7_formal_source_allowlist.json"),
    )
    parser.add_argument("--baseline-map", type=Path, required=True)
    parser.add_argument("--expected-baseline-map-sha256", required=True)
    parser.add_argument("--baseline-metric", type=Path, required=True)
    parser.add_argument("--expected-baseline-metric-sha256", required=True)
    parser.add_argument("--reference-results", type=Path)
    parser.add_argument("--expected-reference-results-sha256")
    parser.add_argument("--candidate-results", type=Path)
    parser.add_argument("--expected-candidate-results-sha256")
    parser.add_argument("--reference-deployment-contract", type=Path)
    parser.add_argument("--candidate-deployment-contract", type=Path)
    parser.add_argument("--expected-query-count", type=int, default=530)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_p0(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
