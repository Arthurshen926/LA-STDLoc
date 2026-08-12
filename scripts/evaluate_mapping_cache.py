#!/usr/bin/env python3
"""Replay one map on mapping descriptors and report translation/rotation pose error."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from common.evaluation_code import mapping_pose_evaluation_code_identity
from common.hashing import sha256_file
from map_learning.equal_energy_descriptor_factor import (
    validate_descriptor_factor_contract,
)
from topology.deployment_revision import collect_deployment_statistics


ARTIFACT_PATH_ARGUMENTS = {
    "map": "map",
    "metric": "metric_state",
    "teacher": "complete_positive_teacher",
    "query_cache": "query_cache",
    "calibration": "scene_calibration",
}


def _json_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_records(args: argparse.Namespace) -> tuple[dict[str, Path], dict]:
    paths = {
        role: Path(getattr(args, argument)).expanduser().resolve()
        for role, argument in ARTIFACT_PATH_ARGUMENTS.items()
    }
    for role, path in paths.items():
        if not path.is_file():
            raise ValueError(f"{role} input is not a file: {path}")
    records = {
        role: {"path": str(path), "sha256": sha256_file(path)}
        for role, path in paths.items()
    }
    return paths, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--metric-state", type=Path, required=True)
    parser.add_argument("--complete-positive-teacher", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument(
        "--descriptor-factor-contract",
        type=Path,
        help=(
            "Optional mapping-only descriptor-source contract. Map, metric, "
            "query cache, teacher, and calibration must be its candidate outputs."
        ),
    )
    parser.add_argument("--scene-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deployment-row-limit", type=int, default=0)
    parser.add_argument(
        "--query-count",
        type=int,
        default=0,
        help="Deterministic uniformly spaced mapping gate; zero evaluates all queries.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    evaluation_code = mapping_pose_evaluation_code_identity(require_clean=True)
    artifact_paths, artifact_records = _artifact_records(args)
    state = torch.load(artifact_paths["map"], map_location="cpu", weights_only=False)
    teacher = torch.load(
        artifact_paths["teacher"], map_location="cpu", weights_only=False
    )
    cache = torch.load(
        artifact_paths["query_cache"], map_location="cpu", weights_only=False
    )
    calibration = json.loads(artifact_paths["calibration"].read_text())
    calibration_sources = dict(calibration.get("sources", {}))
    if (
        calibration.get("schema") != "lafgs_mapping_only_scene_calibration"
        or int(calibration.get("version", 0)) < 2
        or calibration.get("uses_test_queries", False) is not False
        or calibration_sources.get("uses_test_queries") is not False
    ):
        raise ValueError("scene calibration is not a mapping-only contract")
    if (
        Path(str(calibration_sources.get("query_cache", ""))).resolve()
        != artifact_paths["query_cache"]
    ):
        raise ValueError("scene calibration names a different query cache")
    if (
        Path(str(teacher.get("query_cache", ""))).resolve()
        != artifact_paths["query_cache"]
    ):
        raise ValueError("complete-positive teacher names a different query cache")
    descriptor_factor = None
    deployment_cache = cache
    if args.descriptor_factor_contract is not None:
        descriptor_factor = validate_descriptor_factor_contract(
            args.descriptor_factor_contract,
            variant_map_path=artifact_paths["map"],
            variant_metric_path=artifact_paths["metric"],
            variant_query_cache_path=artifact_paths["query_cache"],
            variant_teacher_path=artifact_paths["teacher"],
            variant_calibration_path=artifact_paths["calibration"],
        )
        descriptor_cache_path = descriptor_factor["descriptor_cache_path"]
        if descriptor_cache_path != artifact_paths["query_cache"]:
            raise ValueError("descriptor-factor output is not the evaluated query cache")
        artifact_paths["descriptor_factor"] = descriptor_factor["path"]
        artifact_records["descriptor_factor"] = {
            "path": str(descriptor_factor["path"]),
            "sha256": descriptor_factor["sha256"],
        }
    parameters = calibration["parameters"]
    query_names = list(teacher["query_names"])
    total_queries = len(teacher["records"])
    if total_queries != len(query_names):
        raise ValueError("teacher query names and records are not row-aligned")
    if int(args.query_count) < 0:
        raise ValueError("query count must be non-negative")
    query_indices = None
    if 0 < int(args.query_count) < total_queries:
        query_indices = (
            torch.linspace(0, total_queries - 1, steps=int(args.query_count))
            .round()
            .long()
            .unique(sorted=True)
        )
    selected_query_indices = (
        list(range(total_queries))
        if query_indices is None
        else [int(value) for value in query_indices.tolist()]
    )
    selected_query_names = [query_names[index] for index in selected_query_indices]
    query_selection = "all" if query_indices is None else "uniform_mapping_gate"
    statistics = collect_deployment_statistics(
        state=state,
        metric_state_path=artifact_paths["metric"],
        teacher=teacher,
        query_cache=deployment_cache,
        device=torch.device(args.device),
        ransac_reprojection_px=float(parameters["ransac_reprojection_px"]),
        clean_reprojection_px=float(parameters["clean_radius_px"]),
        task_translation_m=float(parameters["task_translation_m"]),
        task_rotation_deg=float(parameters["task_rotation_deg"]),
        seed=args.seed,
        query_indices=query_indices,
        deployment_row_limit=args.deployment_row_limit,
        collect_anchor_statistics=False,
        progress_label="mapping_cache_evaluation",
    )
    report = {
        "schema": "lafgs_mapping_cache_evaluation",
        "version": 2,
        "uses_test_queries": False,
        "seed": int(args.seed),
        "evaluation_code": evaluation_code,
        "map": str(artifact_paths["map"]),
        "metric_state": str(artifact_paths["metric"]),
        "complete_positive_teacher": str(artifact_paths["teacher"]),
        "query_cache": str(artifact_paths["query_cache"]),
        "descriptor_cache": str(
            artifact_paths.get("descriptor_cache", artifact_paths["query_cache"])
        ),
        "descriptor_factor_contract": (
            str(artifact_paths["descriptor_factor"])
            if descriptor_factor is not None
            else None
        ),
        "scene_calibration": str(artifact_paths["calibration"]),
        "artifacts": artifact_records,
        "deployment_row_limit": int(args.deployment_row_limit),
        "pose_error_units": {"translation": "cm", "rotation": "deg"},
        "query_count": len(selected_query_indices),
        "query_selection": query_selection,
        "evaluation_protocol": {
            "split": "mapping_only",
            "query_selection": query_selection,
            "requested_query_count": int(args.query_count),
            "evaluated_query_count": len(selected_query_indices),
            "teacher_query_count": total_queries,
            "ordered_teacher_query_names_sha256": _json_sha256(query_names),
            "selected_query_indices": selected_query_indices,
            "selected_query_indices_sha256": _json_sha256(selected_query_indices),
            "selected_query_names_sha256": _json_sha256(selected_query_names),
            "deployment_row_limit": int(args.deployment_row_limit),
            "descriptor_protocol": (
                {
                    "kind": "equal_energy_descriptor_factor",
                    "factor_id": descriptor_factor["factor_id"],
                    "source_descriptor_dim": 256,
                    "xfeat_descriptor_dim": 64,
                    "effective_descriptor_dim": 320,
                    "strict_identity_metric": True,
                    "one_materialized_bank": True,
                    "one_global_top1": True,
                    "one_poselib_call_per_query": True,
                }
                if descriptor_factor is not None
                else {
                    "kind": "canonical_query_cache_shared_metric",
                    "descriptor_cache_equals_query_cache": True,
                }
            ),
        },
        "summary": statistics["summary"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "mapping_cache_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
