#!/usr/bin/env python3
"""Run one bounded selection-aware descriptor reconstruction macro-round."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from localization_training.selection_aware_reconstruction import (
    SelectionAwareOptimizationConfig,
    SelectionAwareTrainingData,
    build_selection_aware_training_data,
    optimize_selection_aware_representations,
)
from localization_training.shared_metric import SharedLowRankMetric


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _resolved(path: str | Path) -> str:
    return str(Path(path).resolve())


def _verify_contract(
    *,
    selector_state: dict,
    selected: dict,
    map_path: str,
    metric_path: str,
    family_path: str,
    topk_path: str,
    dynamic_path: str,
) -> None:
    if selector_state.get("schema") != "lafgs_basis_core_reserve_selector":
        raise ValueError("selection-aware reconstruction requires a V25 selector")
    if selected.get("method") != "oof_basis_aware_core_reserve_selection":
        raise ValueError("selection-aware reconstruction requires OOF selections")
    provenance = selected.get("provenance", {})
    if _resolved(provenance.get("topk_outcomes", "")) != _resolved(topk_path):
        raise ValueError("selected outcomes use a different exact candidate graph")
    if _resolved(provenance.get("dynamic_outcomes", "")) != _resolved(
        dynamic_path
    ):
        raise ValueError("selected outcomes use different localization outcomes")
    contract = selector_state.get("candidate_graph_contract", {})
    expected = {
        "map_sha256": _sha256_file(map_path),
        "metric_state_sha256": _sha256_file(metric_path),
        "family_prototype_state_sha256": _sha256_file(family_path),
    }
    for name, actual in expected.items():
        if contract.get(name) != actual:
            raise ValueError(f"selection-aware {name} contract differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--family-prototype-state", required=True)
    parser.add_argument(
        "--family-training-state",
        default="",
        help=(
            "Optional M-step family initialization. Candidate-graph "
            "identity remains tied to --family-prototype-state."
        ),
    )
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--complete-positive-teacher", required=True)
    parser.add_argument("--topk-outcomes", required=True)
    parser.add_argument("--dynamic-outcomes", required=True)
    parser.add_argument("--selected-outcomes", required=True)
    parser.add_argument("--selector-state", required=True)
    parser.add_argument("--counterfactual-audit", default="")
    parser.add_argument("--basis-teacher", default="")
    parser.add_argument("--loo-teacher", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum-protected-per-query", type=int, default=96)
    parser.add_argument("--maximum-neutral-per-query", type=int, default=64)
    parser.add_argument("--maximum-harmful-per-query", type=int, default=64)
    parser.add_argument("--maximum-critical-per-query", type=int, default=32)
    parser.add_argument(
        "--maximum-basis-hyperedges-per-query", type=int, default=8
    )
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--maximum-descriptor-delta", type=float, default=0.03)
    parser.add_argument("--maximum-bias-delta", type=float, default=0.02)
    parser.add_argument("--ranking-margin", type=float, default=0.02)
    parser.add_argument("--preserve-tolerance", type=float, default=0.01)
    parser.add_argument("--ranking-temperature", type=float, default=0.05)
    parser.add_argument("--descriptor-replay-weight", type=float, default=0.001)
    parser.add_argument("--bias-replay-weight", type=float, default=0.02)
    parser.add_argument("--topk-replay-weight", type=float, default=0.1)
    parser.add_argument(
        "--topk-replay-temperature", type=float, default=0.05
    )
    parser.add_argument(
        "--basis-hyperedge-weight", type=float, default=0.2
    )
    parser.add_argument(
        "--basis-hyperedge-margin", type=float, default=0.01
    )
    parser.add_argument(
        "--basis-hyperedge-temperature", type=float, default=0.05
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--reuse-training-data",
        action="store_true",
        help="Reuse output-dir/selection_teacher.pt after contract validation.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    family = torch.load(
        args.family_training_state or args.family_prototype_state,
        map_location="cpu",
        weights_only=False,
    )
    metric_payload = torch.load(
        args.metric_state, map_location="cpu", weights_only=False
    )
    metric = SharedLowRankMetric(**metric_payload["metric_config"]).to(device)
    metric.load_state_dict(metric_payload["metric_state_dict"])
    metric.eval()
    for parameter in metric.parameters():
        parameter.requires_grad_(False)
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    positives = torch.load(
        args.complete_positive_teacher,
        map_location="cpu",
        weights_only=False,
    )
    topk = torch.load(
        args.topk_outcomes, map_location="cpu", weights_only=False
    )
    dynamic = torch.load(
        args.dynamic_outcomes, map_location="cpu", weights_only=False
    )
    selected = torch.load(
        args.selected_outcomes, map_location="cpu", weights_only=False
    )
    counterfactual = (
        torch.load(
            args.counterfactual_audit,
            map_location="cpu",
            weights_only=False,
        )
        if args.counterfactual_audit
        else None
    )
    basis_teacher = (
        torch.load(
            args.basis_teacher,
            map_location="cpu",
            weights_only=False,
        )
        if args.basis_teacher
        else None
    )
    loo_teacher = (
        torch.load(
            args.loo_teacher,
            map_location="cpu",
            weights_only=False,
        )
        if args.loo_teacher
        else None
    )
    selector_state = torch.load(
        args.selector_state, map_location="cpu", weights_only=False
    )
    _verify_contract(
        selector_state=selector_state,
        selected=selected,
        map_path=args.map,
        metric_path=args.metric_state,
        family_path=args.family_prototype_state,
        topk_path=args.topk_outcomes,
        dynamic_path=args.dynamic_outcomes,
    )
    if list(topk["query_names"]) != list(selected["query_names"]):
        raise ValueError("selected outcomes do not cover the exact graph")
    # Basis scoring is dominated by small CPU tensors; a large OpenMP pool
    # makes this deterministic preprocessing substantially slower.
    torch.set_num_threads(1)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_path = output_dir / "selection_teacher.pt"

    def data_progress(completed: int, total: int) -> None:
        if completed % 50 == 0 or completed == total:
            print(
                json.dumps(
                    {
                        "stage": "build_teacher",
                        "completed": completed,
                        "total": total,
                    }
                ),
                flush=True,
            )

    if args.reuse_training_data:
        teacher = torch.load(
            teacher_path, map_location="cpu", weights_only=False
        )
        expected = {
            "map_sha256": _sha256_file(args.map),
            "family_sha256": _sha256_file(
                args.family_training_state or args.family_prototype_state
            ),
            "selected_sha256": _sha256_file(args.selected_outcomes),
            "counterfactual_audit_sha256": (
                _sha256_file(args.counterfactual_audit)
                if args.counterfactual_audit
                else None
            ),
            "basis_teacher_sha256": (
                _sha256_file(args.basis_teacher)
                if args.basis_teacher
                else None
            ),
            "loo_teacher_sha256": (
                _sha256_file(args.loo_teacher)
                if args.loo_teacher
                else None
            ),
        }
        if any(teacher.get("provenance", {}).get(key) != value for key, value in expected.items()):
            raise ValueError("selection teacher cache provenance differs")
        data = SelectionAwareTrainingData(**teacher["training_data"])
    else:
        data = build_selection_aware_training_data(
            state=state,
            family=family,
            selected_outcomes=selected,
            dynamic_outcomes=dynamic,
            complete_positive_teacher=positives,
            cache=cache,
            metric=metric,
            selector_config=dict(selector_state["selector_config"]),
            device=device,
            maximum_protected_per_query=args.maximum_protected_per_query,
            maximum_neutral_per_query=args.maximum_neutral_per_query,
            maximum_harmful_per_query=args.maximum_harmful_per_query,
            maximum_critical_per_query=args.maximum_critical_per_query,
            counterfactual_audit=counterfactual,
            basis_teacher=basis_teacher,
            loo_teacher=loo_teacher,
            maximum_basis_hyperedges_per_query=(
                args.maximum_basis_hyperedges_per_query
            ),
            progress=data_progress,
        )
        _atomic_torch(
            teacher_path,
            {
                "schema": "lafgs_selection_aware_teacher",
                "version": 1,
                "training_data": data.__dict__,
                "provenance": {
                    "map_sha256": _sha256_file(args.map),
                    "family_sha256": _sha256_file(
                        args.family_training_state
                        or args.family_prototype_state
                    ),
                    "selected_sha256": _sha256_file(args.selected_outcomes),
                    "counterfactual_audit_sha256": (
                        _sha256_file(args.counterfactual_audit)
                        if args.counterfactual_audit
                        else None
                    ),
                    "basis_teacher_sha256": (
                        _sha256_file(args.basis_teacher)
                        if args.basis_teacher
                        else None
                    ),
                    "loo_teacher_sha256": (
                        _sha256_file(args.loo_teacher)
                        if args.loo_teacher
                        else None
                    ),
                },
            },
        )
    config = SelectionAwareOptimizationConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        maximum_descriptor_delta=args.maximum_descriptor_delta,
        maximum_bias_delta=args.maximum_bias_delta,
        ranking_margin=args.ranking_margin,
        preserve_tolerance=args.preserve_tolerance,
        ranking_temperature=args.ranking_temperature,
        descriptor_replay_weight=args.descriptor_replay_weight,
        bias_replay_weight=args.bias_replay_weight,
        topk_replay_weight=args.topk_replay_weight,
        topk_replay_temperature=args.topk_replay_temperature,
        basis_hyperedge_weight=args.basis_hyperedge_weight,
        basis_hyperedge_margin=args.basis_hyperedge_margin,
        basis_hyperedge_temperature=args.basis_hyperedge_temperature,
        seed=args.seed,
    )

    def training_progress(record: dict[str, float]) -> None:
        print(
            json.dumps({"stage": "optimize", **record}),
            flush=True,
        )

    updated_map, updated_family, history = (
        optimize_selection_aware_representations(
            state=state,
            family=family,
            data=data,
            config=config,
            device=device,
            progress=training_progress,
        )
    )
    metric_contract = {
        "path": _resolved(args.metric_state),
        "sha256": _sha256_file(args.metric_state),
    }
    updated_map["metric_state_contract"] = metric_contract
    updated_family["metric_state_contract"] = metric_contract
    updated_map["selection_aware_reconstruction"][
        "metric_state_contract"
    ] = metric_contract
    updated_family["selection_aware_reconstruction"][
        "metric_state_contract"
    ] = metric_contract
    map_path = output_dir / "anchor_map_macro_round_01.pt"
    family_path = output_dir / "family_macro_round_01.pt"
    _atomic_torch(map_path, updated_map)
    _atomic_torch(family_path, updated_family)
    provenance = {
        name: {
            "path": _resolved(path),
            "sha256": _sha256_file(path),
        }
        for name, path in {
            "map": args.map,
            "metric_state": args.metric_state,
            "family_prototype_state": args.family_prototype_state,
            **(
                {"family_training_state": args.family_training_state}
                if args.family_training_state
                else {}
            ),
            "query_cache": args.query_cache,
            "complete_positive_teacher": args.complete_positive_teacher,
            "topk_outcomes": args.topk_outcomes,
            "dynamic_outcomes": args.dynamic_outcomes,
            "selected_outcomes": args.selected_outcomes,
            "selector_state": args.selector_state,
            **(
                {"counterfactual_audit": args.counterfactual_audit}
                if args.counterfactual_audit
                else {}
            ),
            **(
                {"basis_teacher": args.basis_teacher}
                if args.basis_teacher
                else {}
            ),
            **(
                {"loo_teacher": args.loo_teacher}
                if args.loo_teacher
                else {}
            ),
        }.items()
    }
    report = {
        "schema": "lafgs_selection_aware_reconstruction_report",
        "version": 1,
        "config": vars(args),
        "optimization_config": config.__dict__,
        "training_data": data.diagnostics,
        "reconstruction": updated_map["selection_aware_reconstruction"],
        "history": history,
        "provenance": provenance,
        "outputs": {
            "map": {
                "path": str(map_path),
                "sha256": _sha256_file(map_path),
            },
            "family": {
                "path": str(family_path),
                "sha256": _sha256_file(family_path),
            },
        },
    }
    _atomic_json(output_dir / "training_report.json", report)
    print(json.dumps(report["outputs"], indent=2))


if __name__ == "__main__":
    main()
