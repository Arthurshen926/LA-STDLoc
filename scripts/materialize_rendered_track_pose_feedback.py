#!/usr/bin/env python3
"""Materialize one bounded mapping-pose feedback revision of a Track-only map.

The input statistics are produced by the exact full-mapping LOO deployment
replay.  The revision removes a small deterministic set of harmful attractors,
restores any rows required by the mapping-feasible matching rank, and writes a
map-bound identity metric plus a remapped positive teacher.  It never reads
source mapping RGB or test queries.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import torch

from common.hashing import sha256_file
from topology.deployment_revision import (
    select_revision,
    subset_map_and_metric,
    subset_teacher,
)


_SOURCE_PATHS = (
    "scripts/materialize_rendered_track_pose_feedback.py",
    "topology/deployment_revision.py",
    "topology/matching_coverage.py",
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
        raise RuntimeError("pose-feedback producer worktree must be clean")
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
            raise RuntimeError(f"temporary {path.name} did not reload")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if json.loads(temporary.read_text()).get("schema") != payload.get("schema"):
            raise RuntimeError(f"temporary {path.name} did not reload")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_inputs(
    *,
    state: dict,
    metric: dict,
    teacher: dict,
    statistics: dict,
    baseline_report: dict,
    calibration: dict,
    paths: dict[str, Path],
    input_sha256: dict[str, str],
) -> int:
    if state.get("schema") != "lafgs_materialized_anchor_map":
        raise ValueError("pose feedback requires a materialized anchor map")
    count = int(torch.as_tensor(state.get("anchor_ids", ())).numel())
    if count <= 0 or not torch.equal(
        torch.as_tensor(state["anchor_ids"]).long(), torch.arange(count)
    ):
        raise ValueError("anchor registry must be non-empty, ordered, and contiguous")
    if not bool((torch.as_tensor(state.get("anchor_type", ())).long() == 1).all()):
        raise ValueError("render-only pose feedback accepts Track anchors only")
    if int(teacher.get("anchor_count", -1)) != count:
        raise ValueError("teacher and map anchor registries differ")
    if metric.get("schema") != "lafgs_shared_metric_state":
        raise ValueError("pose feedback requires a shared metric state")
    if (
        Path(str(metric.get("map_path", ""))).resolve() != paths["map"]
        or str(metric.get("map_sha256", "")) != input_sha256["map"]
    ):
        raise ValueError("metric is not bound to the exact input map")
    if statistics.get("schema") != "lafgs_rendered_track_full_mapping_loo_statistics":
        raise ValueError("statistics are not full-mapping rendered Track LOO feedback")
    if (
        statistics.get("uses_source_mapping_rgb") is not False
        or statistics.get("uses_test_queries") is not False
    ):
        raise ValueError(
            "pose feedback statistics are not source-image-free mapping data"
        )
    counters = statistics.get("counters", {})
    required_counters = {
        "winner_count",
        "correct_winner_count",
        "false_attractor_count",
        "ambiguous_winner_count",
        "clean_inlier_count",
        "harmful_inlier_count",
        "counterfactual_clean_gain",
        "information_deletion_loss",
    }
    if not required_counters.issubset(counters):
        raise ValueError("pose feedback statistics miss required anchor counters")
    for name in required_counters:
        values = torch.as_tensor(counters[name])
        if values.shape != (count,) or not bool(torch.isfinite(values).all()):
            raise ValueError(f"counter {name} is not a finite anchor vector")
    statistic_queries = list(statistics.get("queries", ()))
    query_names = list(teacher.get("query_names", ()))
    if len(statistic_queries) != len(query_names):
        raise ValueError("statistics do not cover the complete mapping query registry")
    if [int(row.get("query_index", -1)) for row in statistic_queries] != list(
        range(len(query_names))
    ) or [str(row.get("image_name", "")) for row in statistic_queries] != query_names:
        raise ValueError("statistics and teacher mapping query registries differ")
    if baseline_report.get("schema") != "lafgs_rendered_track_full_mapping_loo_report":
        raise ValueError("baseline report schema differs")
    report_inputs = baseline_report.get("inputs", {})
    report_hashes = baseline_report.get("input_sha256", {})
    for role in ("map", "metric", "teacher", "query_cache", "scene_calibration"):
        if (
            Path(str(report_inputs.get(role, ""))).resolve() != paths[role]
            or str(report_hashes.get(role, "")) != input_sha256[role]
        ):
            raise ValueError(f"baseline report does not bind exact {role}")
    if (
        Path(str(baseline_report.get("statistics", ""))).resolve()
        != paths["statistics"]
        or str(baseline_report.get("statistics_sha256", ""))
        != input_sha256["statistics"]
    ):
        raise ValueError("baseline report does not bind exact statistics")
    sources = calibration.get("sources", {})
    if (
        calibration.get("schema") != "lafgs_mapping_only_scene_calibration"
        or calibration.get("uses_test_queries", False) is not False
        or sources.get("uses_source_mapping_rgb") is not False
        or sources.get("uses_test_queries") is not False
        or sources.get("mapping_source") != "gaussian_render"
        or Path(str(sources.get("query_cache", ""))).resolve() != paths["query_cache"]
    ):
        raise ValueError("calibration is not exact rendered mapping-only evidence")
    if (
        Path(str(teacher.get("query_cache", ""))).resolve() != paths["query_cache"]
        or str(teacher.get("query_cache_sha256", "")) != input_sha256["query_cache"]
    ):
        raise ValueError("teacher does not bind the exact query cache")
    return count


def run(args: argparse.Namespace) -> dict:
    if not 0.0 < float(args.maximum_prune_fraction) <= 0.02:
        raise ValueError("maximum prune fraction must lie in (0, 0.02]")
    if float(args.minimum_counterfactual_gain) < 0.0:
        raise ValueError("minimum counterfactual gain must be non-negative")
    if int(args.matching_rows_target) <= 0:
        raise ValueError("matching rows target must be positive")
    identity = _producer_identity()
    paths = {
        "map": args.anchor_map.resolve(),
        "metric": args.metric_state.resolve(),
        "teacher": args.teacher.resolve(),
        "statistics": args.statistics.resolve(),
        "baseline_report": args.baseline_report.resolve(),
        "query_cache": args.query_cache.resolve(),
        "scene_calibration": args.scene_calibration.resolve(),
    }
    expected = {
        "map": args.expected_map_sha256,
        "metric": args.expected_metric_sha256,
        "teacher": args.expected_teacher_sha256,
        "statistics": args.expected_statistics_sha256,
        "baseline_report": args.expected_baseline_report_sha256,
        "query_cache": args.expected_query_cache_sha256,
        "scene_calibration": args.expected_scene_calibration_sha256,
    }
    input_sha256 = {
        role: _require_sha(path, expected[role], role) for role, path in paths.items()
    }
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    temporary_dir = output_dir.with_name(f".{output_dir.name}.{os.getpid()}.tmp")
    if temporary_dir.exists():
        raise FileExistsError(temporary_dir)

    state = torch.load(paths["map"], map_location="cpu", weights_only=False)
    metric = torch.load(paths["metric"], map_location="cpu", weights_only=False)
    teacher = torch.load(paths["teacher"], map_location="cpu", weights_only=False)
    statistics = torch.load(paths["statistics"], map_location="cpu", weights_only=False)
    baseline_report = json.loads(paths["baseline_report"].read_text())
    calibration = json.loads(paths["scene_calibration"].read_text())
    count = _validate_inputs(
        state=state,
        metric=metric,
        teacher=teacher,
        statistics=statistics,
        baseline_report=baseline_report,
        calibration=calibration,
        paths=paths,
        input_sha256=input_sha256,
    )
    pruned, selection = select_revision(
        teacher,
        statistics,
        matching_rows_target=int(args.matching_rows_target),
        revisable_mask=torch.ones(count, dtype=torch.bool),
        maximum_prune_fraction=float(args.maximum_prune_fraction),
        minimum_counterfactual_gain=float(args.minimum_counterfactual_gain),
        maximum_tail_nonimproving_wins=int(args.maximum_tail_nonimproving_wins),
    )
    if pruned.numel() == 0:
        raise RuntimeError("pose feedback found no safe revision candidate")
    keep = torch.ones(count, dtype=torch.bool)
    keep[pruned] = False

    final_paths = {
        "map": output_dir / "pose_feedback_anchor_map.pt",
        "metric": output_dir / "pose_feedback_identity_metric.pt",
        "teacher": output_dir / "pose_feedback_positive_teacher.pt",
    }
    temporary_dir.mkdir(parents=True, exist_ok=False)
    try:
        temporary_paths = {
            role: temporary_dir / path.name for role, path in final_paths.items()
        }
        revised_map, revised_metric = subset_map_and_metric(
            state, metric, keep, output_map=final_paths["map"]
        )
        source_rows = torch.nonzero(keep, as_tuple=False).reshape(-1)
        revised_map["pose_feedback_source_anchor_rows"] = source_rows
        revised_map["provenance"] = {
            **revised_map.get("provenance", {}),
            "rendered_track_pose_feedback": {
                "source_map": str(paths["map"]),
                "source_map_sha256": input_sha256["map"],
                "statistics": str(paths["statistics"]),
                "statistics_sha256": input_sha256["statistics"],
                "source_anchor_count": count,
                "retained_anchor_count": int(keep.sum()),
                "pruned_anchor_count": int(pruned.numel()),
                "selection_split": "all_mapping_train",
                "uses_source_mapping_rgb": False,
                "uses_test_queries": False,
            },
        }
        revised_teacher = subset_teacher(teacher, keep, final_paths["map"])
        _atomic_save(revised_map, temporary_paths["map"])
        map_sha256 = sha256_file(temporary_paths["map"])
        revised_metric["map_path"] = str(final_paths["map"])
        revised_metric["map_sha256"] = map_sha256
        revised_metric["protocol"] = "rendered_track_pose_feedback_identity"
        revised_metric["pose_feedback_source_metric"] = {
            "path": str(paths["metric"]),
            "sha256": input_sha256["metric"],
        }
        revised_teacher["anchor_map"] = str(final_paths["map"])
        revised_teacher["anchor_map_sha256"] = map_sha256
        _atomic_save(revised_metric, temporary_paths["metric"])
        _atomic_save(revised_teacher, temporary_paths["teacher"])
        output_sha256 = {
            role: sha256_file(temporary_paths[role]) for role in temporary_paths
        }
        counters = statistics["counters"]
        report = {
            "schema": "lafgs_rendered_track_pose_feedback_revision",
            "version": 1,
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
            "producer_identity": identity,
            "protocol": {
                "feedback_source": "full_mapping_query_local_loo_top1_poselib",
                "revision_rounds": 1,
                "geometry_changed": False,
                "descriptor_metric_changed": False,
                "matching_rank_restored": True,
                "test_used_for_selection": False,
            },
            "configuration": {
                "matching_rows_target": int(args.matching_rows_target),
                "maximum_prune_fraction": float(args.maximum_prune_fraction),
                "minimum_counterfactual_gain": float(args.minimum_counterfactual_gain),
                "maximum_tail_nonimproving_wins": int(
                    args.maximum_tail_nonimproving_wins
                ),
            },
            "source_anchor_count": count,
            "retained_anchor_count": int(keep.sum()),
            "pruned_anchor_count": int(pruned.numel()),
            "pruned_anchor_rows": pruned.tolist(),
            "pruned_track_cluster_ids": torch.as_tensor(state["track_cluster_ids"])
            .long()[pruned]
            .tolist(),
            "pruned_statistics": {
                name: float(torch.as_tensor(values)[pruned].sum())
                for name, values in counters.items()
            },
            "selection": selection,
            "inputs": {role: str(path) for role, path in paths.items()},
            "input_sha256": input_sha256,
            "outputs": {role: str(path) for role, path in final_paths.items()},
            "output_sha256": output_sha256,
        }
        report_path = temporary_dir / "pose_feedback_revision.json"
        _atomic_json(report, report_path)
        completion = {
            "schema": "lafgs_rendered_track_pose_feedback_revision_completion",
            "version": 1,
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
            "producer_identity": identity,
            "report": str(output_dir / report_path.name),
            "report_sha256": sha256_file(report_path),
            "outputs": report["outputs"],
            "output_sha256": output_sha256,
        }
        _atomic_json(
            completion, temporary_dir / "pose_feedback_revision_completion.json"
        )
        if _producer_identity() != identity:
            raise RuntimeError("pose-feedback producer identity changed")
        for role, path in paths.items():
            _require_sha(path, input_sha256[role], role)
        os.replace(temporary_dir, output_dir)
        return report
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-map", type=Path, required=True)
    parser.add_argument("--expected-map-sha256", required=True)
    parser.add_argument("--metric-state", type=Path, required=True)
    parser.add_argument("--expected-metric-sha256", required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--expected-teacher-sha256", required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--expected-statistics-sha256", required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--expected-baseline-report-sha256", required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--expected-query-cache-sha256", required=True)
    parser.add_argument("--scene-calibration", type=Path, required=True)
    parser.add_argument("--expected-scene-calibration-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matching-rows-target", type=int, required=True)
    parser.add_argument("--maximum-prune-fraction", type=float, default=0.01)
    parser.add_argument("--minimum-counterfactual-gain", type=float, default=1.0)
    parser.add_argument("--maximum-tail-nonimproving-wins", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
