#!/usr/bin/env python3
"""Run the canonical V6 feedback-core pipeline with independent proposal arms.

This is the sole formal runner for new V6 experiments.  The older
``run_v6_mainline_convergence.py`` and
``run_closed_loop_projective_distillation.py`` entrypoints are retained only
for reproducing legacy diagnostics.  This runner never chains candidates,
never accesses test queries, and never accepts or selects a winner.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch

from common.hashing import sha256_file
from common.v6_contracts import (
    DESCRIPTOR_SPLIT_SCHEMA,
    FEEDBACK_VERSION,
    require_exact_identity_positive_contract,
    require_mapping_only,
    require_schema,
)
from common.v6_pipeline_contract import validate_v6_pipeline_inputs
from topology.v6_anchor_map import validate_v6_identity_metric


RUN_SCHEMA = "lafgs_v6_feedback_core_independent_arms_run"
RUN_VERSION = 1
SCENE_CALIBRATION_SCHEMA = "lafgs_mapping_only_scene_calibration"
ARM_CHOICES = ("descriptor_loss", "selection", "reconstruction")
_SOURCE_PATHS = (
    "scripts/run_v6_feedback_core_pipeline.py",
    "scripts/evaluate_v6_self_localization.py",
    "scripts/propose_v6_round.py",
    "scripts/compare_v6_feedback.py",
    "common/v6_pipeline_contract.py",
    "common/v6_contracts.py",
)


def _is_sha256(value: object) -> bool:
    text = str(value).lower()
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _require_file_sha(path: Path, expected: str, *, label: str) -> str:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not _is_sha256(expected):
        raise ValueError(f"{label} expected SHA256 is invalid")
    actual = sha256_file(path)
    if actual != str(expected).lower():
        raise ValueError(f"{label} SHA differs: expected {expected}, got {actual}")
    return actual


def _artifact(path: Path, expected: str | None = None, *, label: str) -> dict:
    path = path.resolve()
    actual = (
        _require_file_sha(path, expected, label=label)
        if expected is not None
        else sha256_file(path)
    )
    return {
        "path": str(path),
        "sha256": actual,
        "size_bytes": path.stat().st_size,
    }


def _load_torch(path: Path, expected: str, *, label: str) -> tuple[dict, dict]:
    artifact = _artifact(path, expected, label=label)
    value = torch.load(path.resolve(), map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a dictionary")
    return value, artifact


def _load_json(path: Path, expected: str, *, label: str) -> tuple[dict, dict]:
    artifact = _artifact(path, expected, label=label)
    value = json.loads(path.resolve().read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value, artifact


def _load_scene_calibration(
    path: Path,
    expected: str,
    *,
    requested_ransac_reprojection_px: float | None,
) -> tuple[dict, dict, float]:
    calibration, artifact = _load_json(
        path,
        expected,
        label="mapping-only scene calibration",
    )
    sources = calibration.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("scene calibration source registry is missing")
    uses_test_queries = calibration.get(
        "uses_test_queries", sources.get("uses_test_queries")
    )
    if (
        calibration.get("schema") != SCENE_CALIBRATION_SCHEMA
        or uses_test_queries is not False
        or sources.get("uses_source_mapping_rgb") is not False
        or sources.get("mapping_source") != "gaussian_render"
    ):
        raise ValueError(
            "scene calibration is not a Gaussian-render mapping-only contract"
        )
    parameters = calibration.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("scene calibration parameter registry is missing")
    calibrated_value = parameters.get("ransac_reprojection_px")
    if (
        isinstance(calibrated_value, bool)
        or not isinstance(calibrated_value, (int, float))
        or not math.isfinite(float(calibrated_value))
        or float(calibrated_value) <= 0.0
    ):
        raise ValueError("scene calibration has no valid RANSAC threshold")
    resolved = float(calibrated_value)
    if requested_ransac_reprojection_px is not None:
        requested = float(requested_ransac_reprojection_px)
        if not math.isfinite(requested) or requested != resolved:
            raise ValueError(
                "requested RANSAC threshold differs from scene calibration: "
                f"{requested!r} != {resolved!r}"
            )
    return calibration, artifact, resolved


def _producer(root: Path) -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("formal V6 feedback-core runner requires a clean worktree")
    return {
        "git_commit": commit,
        "worktree_clean": True,
        "source_sha256": {path: sha256_file(root / path) for path in _SOURCE_PATHS},
        "python_executable": sys.executable,
        "python_version": sys.version,
        "torch_version": torch.__version__,
    }


def _run_command(command: list[str], *, root: Path) -> None:
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(root)
        if not existing_pythonpath
        else os.pathsep.join((str(root), existing_pythonpath))
    )
    subprocess.run(command, cwd=root, check=True, env=environment)


def _value(command: Sequence[str], flag: str) -> str:
    try:
        index = command.index(flag)
    except ValueError as error:
        raise ValueError(f"command is missing {flag}") from error
    if index + 1 >= len(command):
        raise ValueError(f"command value is missing for {flag}")
    return str(command[index + 1])


def _evaluation_command(
    args: argparse.Namespace,
    *,
    root: Path,
    map_path: Path,
    map_sha256: str,
    metric_path: Path,
    metric_sha256: str,
    output_dir: Path,
) -> list[str]:
    if args.ransac_reprojection_px is None:
        raise ValueError("RANSAC threshold was not resolved from scene calibration")
    return [
        sys.executable,
        str(root / "scripts/evaluate_v6_self_localization.py"),
        "--map",
        str(map_path.resolve()),
        "--expected-map-sha256",
        map_sha256,
        "--metric",
        str(metric_path.resolve()),
        "--expected-metric-sha256",
        metric_sha256,
        "--observation-cache",
        str(args.observation_cache.resolve()),
        "--expected-observation-cache-sha256",
        args.expected_observation_cache_sha256,
        "--output-dir",
        str(output_dir.resolve()),
        "--device",
        args.device,
        "--cpu-threads",
        str(args.cpu_threads),
        "--positive-radius-px",
        str(args.positive_radius_px),
        "--alpha-minimum",
        str(args.alpha_minimum),
        "--required-rank",
        str(args.required_rank),
        "--required-visibility-rank",
        str(args.required_visibility_rank),
        "--required-detectable-rank",
        str(args.required_detectable_rank),
        "--pose-logdet-target",
        str(args.pose_logdet_target),
        "--pose-min-eigenvalue-target",
        str(args.pose_min_eigenvalue_target),
        "--loo-pose-neighbors",
        str(args.loo_pose_neighbors),
        "--loo-affected-anchor-policy",
        "rebuild",
        "--ransac-reprojection-px",
        str(args.ransac_reprojection_px),
        "--seed",
        str(args.seed),
    ]


def _proposal_command(
    args: argparse.Namespace,
    *,
    root: Path,
    arm: str,
    feedback: dict,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(root / "scripts/propose_v6_round.py"),
        "--arm",
        arm,
        "--map",
        str(args.map.resolve()),
        "--expected-map-sha256",
        args.expected_map_sha256,
        "--observation-cache",
        str(args.observation_cache.resolve()),
        "--expected-observation-cache-sha256",
        args.expected_observation_cache_sha256,
        "--feedback",
        feedback["feedback"]["path"],
        "--expected-feedback-sha256",
        feedback["feedback"]["sha256"],
        "--output-dir",
        str(output_dir.resolve()),
        "--device",
        args.device,
        "--descriptor-trust-region",
        str(args.descriptor_trust_region),
        "--descriptor-margin",
        str(args.descriptor_margin),
        "--descriptor-temperature",
        str(args.descriptor_temperature),
        "--descriptor-learning-rate",
        str(args.descriptor_learning_rate),
        "--descriptor-epochs",
        str(args.descriptor_epochs),
        "--descriptor-batch-size",
        str(args.descriptor_batch_size),
        "--descriptor-maximum-triplets-per-query",
        str(args.descriptor_maximum_triplets_per_query),
        "--descriptor-clean-fraction",
        str(args.descriptor_clean_fraction),
        "--descriptor-clean-weight",
        str(args.descriptor_clean_weight),
        "--descriptor-trust-weight",
        str(args.descriptor_trust_weight),
        "--descriptor-pose-critical-weight",
        str(args.descriptor_pose_critical_weight),
        "--descriptor-tail-query-weight",
        str(args.descriptor_tail_query_weight),
        "--maximum-anchors",
        str(args.selection_maximum_anchors),
        "--visibility-target",
        str(args.required_visibility_rank),
        "--detectability-target",
        str(args.required_detectable_rank),
        "--matching-target",
        str(args.required_rank),
        "--pose-logdet-target",
        str(args.pose_logdet_target),
        "--pose-min-eigenvalue-target",
        str(args.pose_min_eigenvalue_target),
        "--completion-voxel-size-m",
        str(args.completion_voxel_size_m),
        "--alpha-minimum",
        str(args.alpha_minimum),
        "--completion-minimum-similarity",
        str(args.completion_minimum_similarity),
        "--minimum-margin",
        str(args.completion_minimum_margin),
        "--maximum-epipolar-error-px",
        str(args.maximum_epipolar_error_px),
        "--minimum-views",
        str(args.minimum_views),
        "--minimum-camera-families",
        str(args.minimum_camera_families),
        "--completion-maximum-rows-per-view",
        str(args.completion_maximum_rows_per_view),
        "--completion-safety-maximum-components",
        str(args.completion_safety_maximum_components),
    ]
    if args.mapping_training_query_indices is not None and arm in ARM_CHOICES:
        command.extend(
            [
                "--mapping-training-query-indices",
                str(args.mapping_training_query_indices.resolve()),
                "--expected-mapping-training-query-indices-sha256",
                args.expected_mapping_training_query_indices_sha256,
            ]
        )
    if arm == "reconstruction":
        command.extend(
            [
                "--association-graph",
                str(args.association_graph.resolve()),
                "--expected-association-graph-sha256",
                args.expected_association_graph_sha256,
            ]
        )
    return command


def _paired_command(
    *,
    root: Path,
    baseline: dict,
    baseline_map: dict,
    candidate: dict,
    candidate_map: dict,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(root / "scripts/compare_v6_feedback.py"),
        "--baseline-feedback",
        baseline["feedback"]["path"],
        "--expected-baseline-feedback-sha256",
        baseline["feedback"]["sha256"],
        "--candidate-feedback",
        candidate["feedback"]["path"],
        "--expected-candidate-feedback-sha256",
        candidate["feedback"]["sha256"],
        "--baseline-map",
        baseline_map["path"],
        "--expected-baseline-map-sha256",
        baseline_map["sha256"],
        "--candidate-map",
        candidate_map["path"],
        "--expected-candidate-map-sha256",
        candidate_map["sha256"],
        "--output",
        str(output.resolve()),
    ]


def _validate_feedback_summary(
    summary_path: Path,
    *,
    args: argparse.Namespace,
    expected_summary_sha256: str | None,
    map_sha256: str,
    metric_sha256: str,
    cache_sha256: str,
) -> dict:
    summary_path = summary_path.resolve()
    summary_artifact = _artifact(
        summary_path,
        expected_summary_sha256,
        label="feedback summary",
    )
    summary = json.loads(summary_path.read_text())
    if (
        not isinstance(summary, dict)
        or summary.get("schema") != "lafgs_v6_query_local_feedback_summary"
        or int(summary.get("version", -1)) != FEEDBACK_VERSION
    ):
        raise ValueError("feedback summary is not the current V6 identity-safe summary")
    require_mapping_only(summary, label="feedback summary")
    expected_inputs = {
        "map": map_sha256,
        "metric": metric_sha256,
        "observation_cache": cache_sha256,
    }
    if summary.get("input_sha256") != expected_inputs:
        raise ValueError("feedback summary input SHA registry differs")
    contract = summary.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("feedback summary contract is missing")
    if args.ransac_reprojection_px is None:
        raise ValueError("RANSAC threshold was not resolved from scene calibration")
    expected_contract = {
        "positive_radius_px": float(args.positive_radius_px),
        "alpha_minimum": float(args.alpha_minimum),
        "required_matching_rank": int(args.required_rank),
        "required_visibility_rank": int(args.required_visibility_rank),
        "required_detectable_rank": int(args.required_detectable_rank),
        "pose_logdet_target": float(args.pose_logdet_target),
        "pose_min_eigenvalue_target": float(args.pose_min_eigenvalue_target),
        "loo_pose_neighbors": int(args.loo_pose_neighbors),
        "pose_neighborhood_loo": int(args.loo_pose_neighbors) > 1,
        "affected_anchor_policy": "rebuild",
        "affected_anchor_holdout_is_exact_rebuild": True,
        "descriptor_identity_supervision_available": True,
        "diagnostic_purge_suppresses_descriptor_triplets": False,
        "ransac_reprojection_px": float(args.ransac_reprojection_px),
        "ransac_seed": int(args.seed),
        "evaluation_device": str(args.device),
        "global_top1": True,
        "pose_solves_per_query": 1,
        "retrieval": False,
        "refinement": False,
    }

    def matches(actual: object, expected: object) -> bool:
        if isinstance(expected, bool):
            return actual is expected
        if isinstance(expected, int):
            return (
                isinstance(actual, int)
                and not isinstance(actual, bool)
                and actual == expected
            )
        if isinstance(expected, float):
            return (
                isinstance(actual, (int, float))
                and not isinstance(actual, bool)
                and math.isfinite(float(actual))
                and float(actual) == expected
            )
        return actual == expected

    for field, expected in expected_contract.items():
        actual = contract.get(field)
        if not matches(actual, expected):
            raise ValueError(
                "feedback summary contract differs at "
                f"{field}: {actual!r} != {expected!r}"
            )
    require_exact_identity_positive_contract(
        contract.get("positive_identity"),
        label="feedback summary positive identity contract",
    )
    cpu_threads = summary.get("cpu_threads")
    if (
        not isinstance(cpu_threads, int)
        or isinstance(cpu_threads, bool)
        or cpu_threads != int(args.cpu_threads)
    ):
        raise ValueError(
            "feedback summary CPU thread count differs: "
            f"{cpu_threads!r} != {int(args.cpu_threads)!r}"
        )
    feedback_path = Path(summary.get("feedback_path", "")).resolve()
    feedback_artifact = _artifact(
        feedback_path,
        summary.get("feedback_sha256"),
        label="feedback artifact",
    )
    return {
        "summary": summary_artifact,
        "feedback": feedback_artifact,
        "metrics": summary.get("summary"),
        "failure_layer_counts": summary.get("failure_layer_counts"),
        "contract": contract,
    }


def _evaluate(
    args: argparse.Namespace,
    *,
    root: Path,
    map_artifact: dict,
    metric_artifact: dict,
    output_dir: Path,
) -> tuple[dict, list[str]]:
    command = _evaluation_command(
        args,
        root=root,
        map_path=Path(map_artifact["path"]),
        map_sha256=map_artifact["sha256"],
        metric_path=Path(metric_artifact["path"]),
        metric_sha256=metric_artifact["sha256"],
        output_dir=output_dir,
    )
    _run_command(command, root=root)
    result = _validate_feedback_summary(
        output_dir / "summary.json",
        args=args,
        expected_summary_sha256=None,
        map_sha256=map_artifact["sha256"],
        metric_sha256=metric_artifact["sha256"],
        cache_sha256=args.expected_observation_cache_sha256,
    )
    return result, command


def _validate_split(args: argparse.Namespace, *, feedback_sha256: str) -> dict | None:
    path = args.mapping_training_query_indices
    expected = args.expected_mapping_training_query_indices_sha256
    if (path is None) != (expected is None):
        raise ValueError("mapping training split path and SHA must be paired")
    if path is None:
        return None
    split, artifact = _load_json(path, expected, label="mapping training split")
    require_schema(split, DESCRIPTOR_SPLIT_SCHEMA, label="mapping training split")
    if split.get("source_feedback_sha256") != feedback_sha256:
        raise ValueError("mapping training split is not bound to baseline feedback")
    training = split.get("training_query_indices")
    validation = split.get("validation_query_indices")
    if (
        not isinstance(training, list)
        or not isinstance(validation, list)
        or not training
        or not validation
        or sorted(training + validation) != list(range(len(training) + len(validation)))
        or len(training + validation) != len(set(training + validation))
    ):
        raise ValueError("mapping training split is not an exact non-empty partition")
    return artifact


def _load_proposal(
    report_path: Path,
    *,
    arm: str,
    args: argparse.Namespace,
    baseline_feedback: dict,
) -> dict:
    report_path = report_path.resolve()
    report_artifact = _artifact(report_path, label=f"{arm} proposal report")
    report = json.loads(report_path.read_text())
    if (
        not isinstance(report, dict)
        or report.get("schema") != "lafgs_v6_round_proposal"
        or int(report.get("version", -1)) != 2
        or report.get("uses_source_mapping_rgb") is not False
        or report.get("uses_test_queries") is not False
        or report.get("arm") != arm
    ):
        raise ValueError(f"{arm} proposal report contract differs")
    expected_input = {
        "map": args.expected_map_sha256,
        "observation_cache": args.expected_observation_cache_sha256,
        "feedback": baseline_feedback["feedback"]["sha256"],
    }
    if args.mapping_training_query_indices is not None and arm in ARM_CHOICES:
        split_role = (
            "reconstruction_training_query_indices"
            if arm == "reconstruction"
            else "descriptor_training_query_indices"
        )
        expected_input.update(
            {
                "mapping_training_query_indices": (
                    args.expected_mapping_training_query_indices_sha256
                ),
                split_role: (
                    args.expected_mapping_training_query_indices_sha256
                ),
            }
        )
    if arm == "reconstruction":
        expected_input["association_graph"] = args.expected_association_graph_sha256
    if report.get("input_sha256") != expected_input:
        raise ValueError(f"{arm} proposal input SHA registry differs")
    if report.get("proposal_available") is False:
        return {
            "available": False,
            "report": report_artifact,
            "unavailable_reason": report.get("unavailable_reason"),
        }
    if report.get("proposal_available") is not True:
        raise ValueError(f"{arm} proposal availability is missing")
    output = report.get("output")
    if not isinstance(output, Mapping):
        raise ValueError(f"{arm} proposal output registry is missing")
    fields = {
        "map": ("map", "map_sha256"),
        "metric": ("metric", "metric_sha256"),
        "deployment_map": ("deployment_map", "deployment_map_sha256"),
        "deployment_metric": ("deployment_metric", "deployment_metric_sha256"),
    }
    artifacts = {
        name: _artifact(
            Path(output[path_field]),
            output[sha_field],
            label=f"{arm} {name}",
        )
        for name, (path_field, sha_field) in fields.items()
    }
    state = torch.load(artifacts["map"]["path"], map_location="cpu", weights_only=False)
    metric = torch.load(
        artifacts["metric"]["path"], map_location="cpu", weights_only=False
    )
    deployment = torch.load(
        artifacts["deployment_map"]["path"],
        map_location="cpu",
        weights_only=False,
    )
    deployment_metric = torch.load(
        artifacts["deployment_metric"]["path"],
        map_location="cpu",
        weights_only=False,
    )
    validate_v6_identity_metric(
        metric,
        state=state,
        map_path=artifacts["map"]["path"],
        map_sha256=artifacts["map"]["sha256"],
    )
    if deployment.get("provenance", {}).get("v6_compact_deployment_export") is not True:
        raise ValueError(f"{arm} deployment map is not compact")
    for field in ("anchor_ids", "anchor_xyz", "anchor_features"):
        if (
            field not in state
            or field not in deployment
            or not torch.equal(
                torch.as_tensor(state[field]), torch.as_tensor(deployment[field])
            )
        ):
            raise ValueError(f"{arm} compact deployment differs at {field}")
    validate_v6_identity_metric(
        deployment_metric,
        state=deployment,
        map_path=artifacts["deployment_map"]["path"],
        map_sha256=artifacts["deployment_map"]["sha256"],
    )
    return {
        "available": True,
        "report": report_artifact,
        "artifacts": artifacts,
        "configuration": report.get("configuration"),
        "deployment_equivalence": {
            "anchor_ids_equal": True,
            "anchor_xyz_equal": True,
            "deployed_anchor_features_equal": True,
            "compact_map_exact_loo_rebuild_capable": False,
        },
    }


def _selected_arms(args: argparse.Namespace) -> list[str]:
    if args.arms is None:
        arms = ["descriptor_loss", "selection"]
    else:
        arms = [str(arm) for arm in args.arms]
        if not arms:
            raise ValueError("at least one V6 proposal arm is required")
        if len(arms) != len(set(arms)):
            raise ValueError("V6 proposal arms must be unique")
    if bool(args.run_reconstruction) and "reconstruction" not in arms:
        arms.append("reconstruction")
    return arms


def _configuration(args: argparse.Namespace) -> dict:
    if args.ransac_reprojection_px is None:
        raise ValueError("RANSAC threshold was not resolved from scene calibration")
    return {
        "device": args.device,
        "cpu_threads": int(args.cpu_threads),
        "seed": int(args.seed),
        "positive_radius_px": float(args.positive_radius_px),
        "alpha_minimum": float(args.alpha_minimum),
        "required_matching_rank": int(args.required_rank),
        "required_visibility_rank": int(args.required_visibility_rank),
        "required_detectable_rank": int(args.required_detectable_rank),
        "loo_pose_neighbors": int(args.loo_pose_neighbors),
        "loo_affected_anchor_policy": "rebuild",
        "ransac_reprojection_px": float(args.ransac_reprojection_px),
        "ransac_reprojection_source": (
            "scene_calibration.parameters.ransac_reprojection_px"
        ),
        "descriptor_trust_region": float(args.descriptor_trust_region),
        "descriptor_margin": float(args.descriptor_margin),
        "descriptor_temperature": float(args.descriptor_temperature),
        "descriptor_learning_rate": float(args.descriptor_learning_rate),
        "descriptor_epochs": int(args.descriptor_epochs),
        "descriptor_batch_size": int(args.descriptor_batch_size),
        "descriptor_maximum_triplets_per_query": int(
            args.descriptor_maximum_triplets_per_query
        ),
        "descriptor_clean_fraction": float(args.descriptor_clean_fraction),
        "descriptor_clean_weight": float(args.descriptor_clean_weight),
        "descriptor_trust_weight": float(args.descriptor_trust_weight),
        "descriptor_pose_critical_weight": float(args.descriptor_pose_critical_weight),
        "descriptor_tail_query_weight": float(args.descriptor_tail_query_weight),
        "selection_maximum_anchors": int(args.selection_maximum_anchors),
        "pose_logdet_target": float(args.pose_logdet_target),
        "pose_min_eigenvalue_target": float(args.pose_min_eigenvalue_target),
        "requested_arms": _selected_arms(args),
        "run_reconstruction": "reconstruction" in _selected_arms(args),
        "completion_voxel_size_m": float(args.completion_voxel_size_m),
        "completion_minimum_similarity": float(args.completion_minimum_similarity),
        "completion_minimum_margin": float(args.completion_minimum_margin),
        "maximum_epipolar_error_px": float(args.maximum_epipolar_error_px),
        "minimum_views": int(args.minimum_views),
        "minimum_camera_families": int(args.minimum_camera_families),
        "completion_maximum_rows_per_view": int(args.completion_maximum_rows_per_view),
        "completion_safety_maximum_components": int(
            args.completion_safety_maximum_components
        ),
    }


def _atomic_json(payload: dict, output: Path) -> None:
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if json.loads(temporary.read_text()) != payload:
            raise RuntimeError("temporary V6 pipeline report did not reload exactly")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def run(
    args: argparse.Namespace,
    *,
    invocation_argv: Sequence[str] | None = None,
) -> dict:
    root = Path(__file__).resolve().parents[1]
    path_fields = (
        "map",
        "metric",
        "observation_cache",
        "association_graph",
        "materialization_report",
        "scene_calibration",
        "mapping_training_query_indices",
        "baseline_feedback_summary",
        "output_dir",
    )
    for field in path_fields:
        value = getattr(args, field, None)
        if value is not None:
            setattr(args, field, Path(value).resolve())
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if (args.baseline_feedback_summary is None) != (
        args.expected_baseline_feedback_summary_sha256 is None
    ):
        raise ValueError("precomputed baseline summary path and SHA must be paired")
    if (args.mapping_training_query_indices is None) != (
        args.expected_mapping_training_query_indices_sha256 is None
    ):
        raise ValueError("mapping training split path and SHA must be paired")

    calibration, calibration_artifact, calibrated_ransac_reprojection_px = (
        _load_scene_calibration(
            args.scene_calibration,
            args.expected_scene_calibration_sha256,
            requested_ransac_reprojection_px=args.ransac_reprojection_px,
        )
    )
    args.ransac_reprojection_px = calibrated_ransac_reprojection_px

    state, map_artifact = _load_torch(
        args.map, args.expected_map_sha256, label="baseline map"
    )
    metric, metric_artifact = _load_torch(
        args.metric, args.expected_metric_sha256, label="baseline metric"
    )
    cache, cache_artifact = _load_torch(
        args.observation_cache,
        args.expected_observation_cache_sha256,
        label="Gaussian-render observation cache",
    )
    association, association_artifact = _load_torch(
        args.association_graph,
        args.expected_association_graph_sha256,
        label="association graph",
    )
    materialization, materialization_artifact = _load_json(
        args.materialization_report,
        args.expected_materialization_report_sha256,
        label="materialization report",
    )
    pipeline_contract = validate_v6_pipeline_inputs(
        state=state,
        observation_cache=cache,
        observation_cache_sha256=cache_artifact["sha256"],
        map_sha256=map_artifact["sha256"],
        association_graph=association,
        association_graph_sha256=association_artifact["sha256"],
        materialization_report=materialization,
    )
    statistics = calibration.get("statistics")
    if not isinstance(statistics, Mapping):
        raise ValueError("scene calibration statistics registry is missing")
    calibrated_query_count = statistics.get("query_count")
    if (
        not isinstance(calibrated_query_count, int)
        or isinstance(calibrated_query_count, bool)
        or calibrated_query_count <= 0
    ):
        raise ValueError("scene calibration mapping query count is invalid")
    observation_query_count = pipeline_contract.get("mapping_query_count")
    if observation_query_count != calibrated_query_count:
        raise ValueError(
            "scene calibration and observation query counts differ: "
            f"{calibrated_query_count!r} != {observation_query_count!r}"
        )
    validate_v6_identity_metric(
        metric,
        state=state,
        map_path=map_artifact["path"],
        map_sha256=map_artifact["sha256"],
    )
    # The formal runner only needs immutable artifact metadata after validating
    # the inputs.  Releasing the 25+ GB observation payload here keeps one
    # parent process per independent arm cheap enough to run D2/D3/S1/R1 in
    # parallel; each child still reloads and SHA-validates its own exact input.
    del state, metric, cache, association, materialization, calibration
    gc.collect()
    producer = _producer(root)
    args.output_dir.mkdir(parents=True)

    if args.baseline_feedback_summary is None:
        baseline, baseline_command = _evaluate(
            args,
            root=root,
            map_artifact=map_artifact,
            metric_artifact=metric_artifact,
            output_dir=args.output_dir / "baseline_evaluation",
        )
        baseline_reused = False
    else:
        baseline = _validate_feedback_summary(
            args.baseline_feedback_summary,
            args=args,
            expected_summary_sha256=(args.expected_baseline_feedback_summary_sha256),
            map_sha256=map_artifact["sha256"],
            metric_sha256=metric_artifact["sha256"],
            cache_sha256=cache_artifact["sha256"],
        )
        baseline_command = None
        baseline_reused = True
    split_artifact = _validate_split(
        args, feedback_sha256=baseline["feedback"]["sha256"]
    )

    arms = _selected_arms(args)
    arm_reports = []
    for arm in arms:
        arm_root = args.output_dir / f"arm_{arm}"
        proposal_dir = arm_root / "proposal"
        proposal_command = _proposal_command(
            args,
            root=root,
            arm=arm,
            feedback=baseline,
            output_dir=proposal_dir,
        )
        _run_command(proposal_command, root=root)
        proposal = _load_proposal(
            proposal_dir / "proposal.json",
            arm=arm,
            args=args,
            baseline_feedback=baseline,
        )
        stage = {
            "arm": arm,
            "parent": {
                "map_sha256": map_artifact["sha256"],
                "feedback_sha256": baseline["feedback"]["sha256"],
            },
            "proposal": proposal,
            "commands": {"proposal": proposal_command},
        }
        if not proposal["available"]:
            stage["evaluation"] = None
            stage["paired_diagnostics"] = None
            arm_reports.append(stage)
            continue
        candidate = proposal["artifacts"]
        evaluation, evaluation_command = _evaluate(
            args,
            root=root,
            map_artifact=candidate["map"],
            metric_artifact=candidate["metric"],
            output_dir=arm_root / "evaluation",
        )
        paired_output = arm_root / "paired_diagnostics.json"
        paired_command = _paired_command(
            root=root,
            baseline=baseline,
            baseline_map=map_artifact,
            candidate=evaluation,
            candidate_map=candidate["map"],
            output=paired_output,
        )
        _run_command(paired_command, root=root)
        paired = json.loads(paired_output.read_text())
        if (
            not isinstance(paired, dict)
            or paired.get("schema") != "lafgs_v6_paired_feedback_diagnostics"
            or int(paired.get("version", -1)) != 1
            or paired.get("uses_source_mapping_rgb") is not False
            or paired.get("uses_test_queries") is not False
            or paired.get("valid") is not True
        ):
            raise ValueError(f"{arm} paired diagnostics contract differs")
        stage["evaluation"] = evaluation
        stage["paired_diagnostics"] = {
            "artifact": _artifact(paired_output, label=f"{arm} paired diagnostics"),
            "scopes": paired.get("scopes"),
            "comparison_contract": paired.get("comparison_contract"),
        }
        stage["commands"].update(
            {"evaluation": evaluation_command, "paired_diagnostics": paired_command}
        )
        stage["evaluation_target_contract"] = {
            "fresh_mapping_feedback_target": "full_proposal_training_checkpoint",
            "fresh_mapping_feedback_map_sha256": candidate["map"]["sha256"],
            "exact_affected_anchor_rebuild_required": True,
            "compact_deployment_target": "final_online_localization_and_size_loading",
            "compact_deployment_map_sha256": candidate["deployment_map"]["sha256"],
            "deployed_anchor_features_equal": True,
            "equivalence_boundary": (
                "online anchor ids/xyz/features are identical; exact LOO rebuild "
                "is replayable only from the full training checkpoint"
            ),
        }
        arm_reports.append(stage)

    result = {
        "schema": RUN_SCHEMA,
        "version": RUN_VERSION,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "automatic_hard_gate_acceptance": False,
        "candidate_acceptance_status": "external_review_required",
        "winner_selection_performed": False,
        "selected_winner": None,
        "runner_role": "canonical_v6_feedback_core_independent_arms",
        "legacy_runner_note": (
            "older V6 convergence/closed-loop runners are reproduction-only diagnostics"
        ),
        "argv": list(sys.argv if invocation_argv is None else invocation_argv),
        "producer": producer,
        "pipeline_input_contract": pipeline_contract,
        "inputs": {
            "map": map_artifact,
            "metric": metric_artifact,
            "observation_cache": cache_artifact,
            "association_graph": association_artifact,
            "materialization_report": materialization_artifact,
            "scene_calibration": calibration_artifact,
            "mapping_training_split": split_artifact,
        },
        "configuration": _configuration(args),
        "baseline": {
            **baseline,
            "reused_precomputed": baseline_reused,
            "command": baseline_command,
        },
        "independent_arm_contract": {
            "all_arms_parent_map_sha256": map_artifact["sha256"],
            "all_arms_feedback_sha256": baseline["feedback"]["sha256"],
            "candidate_chaining": False,
            "parent_input_payloads_released_before_subprocesses": True,
            "arms": arms,
        },
        "arms": arm_reports,
        "online_protocol": "native_superpoint_global_top1_one_standard_poselib",
    }
    _atomic_json(result, args.output_dir / "run.json")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--expected-map-sha256", required=True)
    parser.add_argument("--metric", type=Path, required=True)
    parser.add_argument("--expected-metric-sha256", required=True)
    parser.add_argument("--observation-cache", type=Path, required=True)
    parser.add_argument("--expected-observation-cache-sha256", required=True)
    parser.add_argument("--association-graph", type=Path, required=True)
    parser.add_argument("--expected-association-graph-sha256", required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--expected-materialization-report-sha256", required=True)
    parser.add_argument("--scene-calibration", type=Path, required=True)
    parser.add_argument("--expected-scene-calibration-sha256", required=True)
    parser.add_argument("--mapping-training-query-indices", type=Path)
    parser.add_argument("--expected-mapping-training-query-indices-sha256")
    parser.add_argument("--baseline-feedback-summary", type=Path)
    parser.add_argument("--expected-baseline-feedback-summary-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--positive-radius-px", type=float, default=2.0)
    parser.add_argument("--alpha-minimum", type=float, default=0.05)
    parser.add_argument("--required-rank", type=int, default=16)
    parser.add_argument("--required-visibility-rank", type=int, default=4)
    parser.add_argument("--required-detectable-rank", type=int, default=16)
    parser.add_argument("--loo-pose-neighbors", type=int, default=3)
    parser.add_argument(
        "--ransac-reprojection-px",
        type=float,
        help=(
            "Optional assertion that must exactly match the mapping-only scene "
            "calibration; the calibration always supplies the formal threshold."
        ),
    )
    parser.add_argument("--descriptor-trust-region", type=float, default=0.05)
    parser.add_argument("--descriptor-margin", type=float, default=0.05)
    parser.add_argument("--descriptor-temperature", type=float, default=0.04)
    parser.add_argument("--descriptor-learning-rate", type=float, default=0.02)
    parser.add_argument("--descriptor-epochs", type=int, default=5)
    parser.add_argument("--descriptor-batch-size", type=int, default=8192)
    parser.add_argument(
        "--descriptor-maximum-triplets-per-query", type=int, default=128
    )
    parser.add_argument("--descriptor-clean-fraction", type=float, default=0.25)
    parser.add_argument("--descriptor-clean-weight", type=float, default=0.25)
    parser.add_argument("--descriptor-trust-weight", type=float, default=0.1)
    parser.add_argument("--descriptor-pose-critical-weight", type=float, default=0.0)
    parser.add_argument("--descriptor-tail-query-weight", type=float, default=0.0)
    parser.add_argument("--selection-maximum-anchors", type=int, default=20000)
    parser.add_argument("--pose-logdet-target", type=float, default=0.0)
    parser.add_argument("--pose-min-eigenvalue-target", type=float, default=0.0)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=ARM_CHOICES,
        help=(
            "Independent proposal arms to run. Defaults to descriptor_loss and "
            "selection; --run-reconstruction remains a compatibility shortcut."
        ),
    )
    parser.add_argument("--run-reconstruction", action="store_true")
    parser.add_argument("--completion-voxel-size-m", type=float, default=0.05)
    parser.add_argument("--completion-minimum-similarity", type=float, default=0.7)
    parser.add_argument("--completion-minimum-margin", type=float, default=0.01)
    parser.add_argument("--maximum-epipolar-error-px", type=float, default=2.0)
    parser.add_argument("--minimum-views", type=int, default=3)
    parser.add_argument("--minimum-camera-families", type=int, default=2)
    parser.add_argument("--completion-maximum-rows-per-view", type=int, default=256)
    parser.add_argument(
        "--completion-safety-maximum-components", type=int, default=100000
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    result = run(
        build_parser().parse_args(arguments),
        invocation_argv=[str(Path(__file__).resolve()), *arguments],
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
