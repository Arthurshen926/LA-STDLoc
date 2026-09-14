#!/usr/bin/env python3
"""Fail-closed comparison of paired three-seed mapping-only pose replays."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import torch

from common.evaluation_code import mapping_pose_evaluation_code_identity
from common.hashing import sha256_file
from map_learning.equal_energy_descriptor_factor import audit_descriptor_factor_pair


REQUIRED_SEEDS = (2026, 2027, 2028)
ARTIFACT_ROLES = ("map", "metric", "teacher", "query_cache", "calibration")
SUMMARY_METRICS = (
    "raw_gt_precision_percent",
    "median_te_cm",
    "mean_te_cm",
    "p90_te_cm",
    "cvar95_te_cm",
    "median_ae_deg",
    "mean_ae_deg",
    "p90_ae_deg",
    "p95_ae_deg",
    "recall_5cm_5deg_percent",
    "catastrophic_100cm_count",
)
TRANSLATION_METRICS = (
    "median_te_cm",
    "mean_te_cm",
    "p90_te_cm",
    "cvar95_te_cm",
)
ROTATION_METRICS = (
    "median_ae_deg",
    "mean_ae_deg",
    "p90_ae_deg",
    "p95_ae_deg",
)


def default_thresholds() -> dict:
    """Return the preregistered P7 pair-policy pose gate."""
    return {
        "per_seed_non_regression": {
            "raw_precision_tolerance_pp": 0.005,
            "translation_absolute_tolerance_cm": 0.02,
            "translation_relative_tolerance": 0.01,
            "rotation_absolute_tolerance_deg": 0.02,
            "rotation_relative_tolerance": 0.01,
            "recall_5cm_5deg_tolerance_pp": 0.1,
            "catastrophic_count_tolerance": 0,
        },
        "three_seed_mean_substantive_improvement": {
            "central_translation_cm": 0.03,
            "tail_translation_cm": 0.05,
            "rotation_deg": 0.02,
            "recall_5cm_5deg_pp": 0.2,
            "raw_precision_pp": 0.01,
        },
    }


def _json_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _require_file(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} is not a file: {resolved}")
    return resolved


def _resolve_artifacts(
    values: Mapping[str, str | Path], *, arm: str
) -> dict[str, Path]:
    if set(values) != set(ARTIFACT_ROLES):
        raise ValueError(
            f"{arm} artifacts must be exactly {list(ARTIFACT_ROLES)}, got {sorted(values)}"
        )
    return {
        role: _require_file(values[role], label=f"{arm}.{role}")
        for role in ARTIFACT_ROLES
    }


def _resolve_summaries(
    values: Mapping[int, str | Path], *, arm: str
) -> dict[int, Path]:
    seeds = tuple(sorted(int(seed) for seed in values))
    if seeds != REQUIRED_SEEDS:
        raise ValueError(
            f"{arm} summaries must bind exactly seeds {list(REQUIRED_SEEDS)}, got {list(seeds)}"
        )
    return {
        seed: _require_file(values[seed], label=f"{arm}.seed{seed}_summary")
        for seed in REQUIRED_SEEDS
    }


def _normalize_thresholds(value: dict | None) -> dict:
    thresholds = (
        default_thresholds() if value is None else json.loads(json.dumps(value))
    )
    if set(thresholds) != {
        "per_seed_non_regression",
        "three_seed_mean_substantive_improvement",
    }:
        raise ValueError("threshold contract has unexpected top-level fields")
    per_seed = thresholds["per_seed_non_regression"]
    substantive = thresholds["three_seed_mean_substantive_improvement"]
    expected_per_seed = {
        "raw_precision_tolerance_pp",
        "translation_absolute_tolerance_cm",
        "translation_relative_tolerance",
        "rotation_absolute_tolerance_deg",
        "rotation_relative_tolerance",
        "recall_5cm_5deg_tolerance_pp",
        "catastrophic_count_tolerance",
    }
    expected_substantive = {
        "central_translation_cm",
        "tail_translation_cm",
        "rotation_deg",
        "recall_5cm_5deg_pp",
        "raw_precision_pp",
    }
    if set(per_seed) != expected_per_seed or set(substantive) != expected_substantive:
        raise ValueError(
            "threshold contract fields differ from the preregistered schema"
        )
    for section in (per_seed, substantive):
        for name, raw in section.items():
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"threshold {name} must be numeric")
            if not math.isfinite(float(raw)) or float(raw) < 0:
                raise ValueError(f"threshold {name} must be finite and non-negative")
    if int(per_seed["catastrophic_count_tolerance"]) != 0:
        raise ValueError("catastrophic count tolerance is preregistered at zero")
    per_seed["catastrophic_count_tolerance"] = 0
    return thresholds


def _normalize_expected_sha256(
    values: Mapping[str, str] | None, *, allowed_keys: set[str]
) -> dict[str, str]:
    expected = dict(values or {})
    unknown = set(expected) - allowed_keys
    if unknown:
        raise ValueError(f"unexpected expected-SHA keys: {sorted(unknown)}")
    result = {}
    for name, value in expected.items():
        digest = str(value).strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"expected SHA-256 for {name} must be 64 hex digits")
        result[name] = digest
    return result


def _hash_inputs(
    *,
    artifacts: dict[str, dict[str, Path]],
    summaries: dict[str, dict[int, Path]],
    expected_sha256: Mapping[str, str] | None,
    extra_paths: Mapping[str, Path] | None = None,
) -> tuple[dict[str, dict], dict[str, str]]:
    paths: dict[str, Path] = {}
    for arm in ("baseline", "variant"):
        paths.update({f"{arm}.{role}": artifacts[arm][role] for role in ARTIFACT_ROLES})
        paths.update(
            {
                f"{arm}.seed{seed}_summary": summaries[arm][seed]
                for seed in REQUIRED_SEEDS
            }
        )
    paths.update(dict(extra_paths or {}))
    expected = _normalize_expected_sha256(expected_sha256, allowed_keys=set(paths))
    digest_cache: dict[Path, str] = {}
    records = {}
    actual = {}
    for name, path in paths.items():
        if path not in digest_cache:
            # This is a streaming byte hash. In particular, the multi-GB query
            # cache is never deserialized, and a shared path is hashed once.
            digest_cache[path] = sha256_file(path)
        digest = digest_cache[path]
        actual[name] = digest
        expected_digest = expected.get(name)
        if expected_digest is not None and digest != expected_digest:
            raise ValueError(
                f"SHA-256 mismatch for {name}: expected {expected_digest}, got {digest}"
            )
        records[name] = {
            "path": str(path),
            "sha256": digest,
            "expected_sha256": expected_digest,
            "expected_sha256_matches": (None if expected_digest is None else True),
        }
    return records, actual


def _audit_arm_artifacts(*, arm: str, paths: dict[str, Path]) -> dict:
    state = torch.load(paths["map"], map_location="cpu", weights_only=False)
    metric = torch.load(paths["metric"], map_location="cpu", weights_only=False)
    teacher = torch.load(paths["teacher"], map_location="cpu", weights_only=False)
    calibration = json.loads(paths["calibration"].read_text())

    anchor_ids = torch.as_tensor(state["anchor_ids"]).long().reshape(-1)
    metric_ids = torch.as_tensor(metric["landmark_indices"]).long().reshape(-1)
    if anchor_ids.numel() == 0 or anchor_ids.unique().numel() != anchor_ids.numel():
        raise ValueError(f"{arm} map anchor IDs are empty or non-unique")
    if not torch.equal(anchor_ids, metric_ids):
        raise ValueError(f"{arm} map and metric anchor IDs do not align exactly")
    if "map_path" not in metric:
        raise ValueError(f"{arm} metric does not bind map_path")
    if Path(str(metric["map_path"])).resolve() != paths["map"]:
        raise ValueError(f"{arm} metric map_path names a different map")

    names = list(teacher["query_names"])
    records = list(teacher["records"])
    if not names or len(names) != len(set(names)) or len(records) != len(names):
        raise ValueError(
            f"{arm} teacher query registry is empty, duplicated, or incomplete"
        )
    if int(teacher["anchor_count"]) != int(anchor_ids.numel()):
        raise ValueError(f"{arm} teacher and map anchor counts differ")
    for index, (name, record) in enumerate(zip(names, records)):
        if int(record.get("query_index", -1)) != index:
            raise ValueError(f"{arm} teacher query_index differs at row {index}")
        if str(record.get("query_name", "")) != str(name):
            raise ValueError(f"{arm} teacher query_name differs at row {index}")
    if "query_cache" not in teacher:
        raise ValueError(f"{arm} teacher does not bind query_cache")
    if Path(str(teacher["query_cache"])).resolve() != paths["query_cache"]:
        raise ValueError(f"{arm} teacher names a different query cache")
    if "anchor_map" not in teacher:
        raise ValueError(f"{arm} teacher does not bind anchor_map")
    teacher_map_path = Path(str(teacher["anchor_map"])).expanduser().resolve()
    if not teacher_map_path.is_file():
        raise ValueError(f"{arm} teacher anchor_map is not a file")
    teacher_map = (
        state
        if teacher_map_path == paths["map"]
        else torch.load(teacher_map_path, map_location="cpu", weights_only=False)
    )
    registry_fields = (
        "anchor_ids",
        "source_primitive_ids",
        "track_cluster_ids",
        "anchor_xyz",
        "anchor_type",
        "dependency_group_ids",
        "coarse_dependency_group_ids",
        "fine_identity_ids",
        "source_dependency_group_ids",
    )
    if teacher_map_path != paths["map"]:
        for field in registry_fields:
            if (
                field not in state
                or field not in teacher_map
                or not torch.equal(
                    torch.as_tensor(state[field]), torch.as_tensor(teacher_map[field])
                )
            ):
                raise ValueError(
                    f"{arm} teacher anchor-map registry differs for {field}"
                )

    sources = dict(calibration.get("sources", {}))
    if (
        calibration.get("schema") != "lafgs_mapping_only_scene_calibration"
        or int(calibration.get("version", 0)) < 2
        or calibration.get("uses_test_queries", False) is not False
        or sources.get("uses_test_queries") is not False
    ):
        raise ValueError(f"{arm} calibration is not a mapping-only contract")
    if Path(str(sources.get("query_cache", ""))).resolve() != paths["query_cache"]:
        raise ValueError(f"{arm} calibration names a different query cache")

    selected = (
        torch.linspace(0, len(names) - 1, steps=256).round().long().unique(sorted=True)
    )
    if selected.numel() != 256:
        raise ValueError(f"{arm} teacher cannot define the preregistered q256 gate")
    indices = selected.tolist()
    selected_names = [names[index] for index in indices]
    return {
        "anchor_count": int(anchor_ids.numel()),
        "anchor_ids_sha256": _tensor_sha256(anchor_ids),
        "metric_anchor_ids_bitwise_equal_map": True,
        "teacher_anchor_count_equal_map": True,
        "teacher_anchor_map": str(teacher_map_path),
        "teacher_anchor_map_sha256": sha256_file(teacher_map_path),
        "teacher_anchor_registry_equal_map": True,
        "teacher_query_count": len(names),
        "ordered_teacher_query_names_sha256": _json_sha256(names),
        "uniform_q256_indices_sha256": _json_sha256(indices),
        "uniform_q256_query_names_sha256": _json_sha256(selected_names),
        "uniform_q256_indices": indices,
        "uniform_q256_first_index": int(indices[0]),
        "uniform_q256_last_index": int(indices[-1]),
        "query_cache_path_bound_by_teacher": True,
        "query_cache_path_bound_by_calibration": True,
        "calibration_numeric_contract": {
            name: calibration[name] for name in ("statistics", "parameters", "policy")
        },
        "calibration_uses_test_queries": False,
        "query_names": names,
    }


def _numeric_metric(summary: dict, name: str, *, seed: int, arm: str) -> float | int:
    value = summary.get(name)
    if name == "catastrophic_100cm_count":
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{arm} seed {seed} {name} must be a non-negative integer")
        return int(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{arm} seed {seed} {name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{arm} seed {seed} {name} must be finite")
    if value < 0:
        raise ValueError(f"{arm} seed {seed} {name} must be non-negative")
    if name.endswith("_percent") and value > 100.0:
        raise ValueError(f"{arm} seed {seed} {name} exceeds 100 percent")
    return value


def _is_exact_int(value, expected: int) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value == expected


def _load_summary(
    *,
    arm: str,
    seed: int,
    path: Path,
    artifacts: dict[str, Path],
    artifact_sha256: dict[str, str],
    arm_audit: dict,
    expected_evaluation_code: dict,
    extra_artifacts: Mapping[str, Path] | None = None,
) -> dict:
    report = json.loads(path.read_text())
    if (
        report.get("schema") != "lafgs_mapping_cache_evaluation"
        or report.get("version") != 2
        or report.get("uses_test_queries") is not False
    ):
        raise ValueError(f"{arm} seed {seed} is not a mapping-only evaluation report")
    if not _is_exact_int(report.get("deployment_row_limit"), 0):
        raise ValueError(f"{arm} seed {seed} changes deployment row count")
    if report.get("query_selection") != "uniform_mapping_gate":
        raise ValueError(f"{arm} seed {seed} does not use uniform_mapping_gate")
    if not _is_exact_int(report.get("query_count"), 256):
        raise ValueError(f"{arm} seed {seed} does not contain exactly q256")
    if report.get("pose_error_units") != {"translation": "cm", "rotation": "deg"}:
        raise ValueError(f"{arm} seed {seed} has absent or unexpected pose-error units")
    if not _is_exact_int(report.get("seed"), seed):
        raise ValueError(f"{arm} seed {seed} summary embeds a different seed")
    if report.get("evaluation_code") != expected_evaluation_code:
        raise ValueError(f"{arm} seed {seed} evaluation-code identity differs")
    required_paths = {
        "map": "map",
        "metric_state": "metric",
        "complete_positive_teacher": "teacher",
        "query_cache": "query_cache",
        "scene_calibration": "calibration",
    }
    extra_artifacts = dict(extra_artifacts or {})
    if extra_artifacts:
        required_paths["descriptor_factor_contract"] = "descriptor_factor"
    for field, role in required_paths.items():
        expected_path = ({**artifacts, **extra_artifacts})[role]
        if Path(str(report.get(field, ""))).resolve() != expected_path:
            raise ValueError(f"{arm} seed {seed} summary {field} path differs")
    descriptor_cache = report.get("descriptor_cache")
    if extra_artifacts and descriptor_cache is None:
        raise ValueError(f"{arm} seed {seed} descriptor cache is absent")
    if descriptor_cache is not None and Path(str(descriptor_cache)).resolve() != (
        artifacts["query_cache"]
    ):
        raise ValueError(f"{arm} seed {seed} descriptor cache differs")
    if not extra_artifacts:
        if report.get("descriptor_factor_contract") is not None:
            raise ValueError(f"{arm} seed {seed} unexpectedly binds a descriptor factor")
    embedded_artifacts = report.get("artifacts")
    expected_artifact_roles = set(ARTIFACT_ROLES) | set(extra_artifacts)
    if not isinstance(embedded_artifacts, dict) or set(embedded_artifacts) != (
        expected_artifact_roles
    ):
        raise ValueError(f"{arm} seed {seed} summary artifact bindings are incomplete")
    summary_artifacts = {**artifacts, **extra_artifacts}
    for role in expected_artifact_roles:
        record = embedded_artifacts[role]
        if not isinstance(record, dict):
            raise ValueError(f"{arm} seed {seed} summary {role} binding is invalid")
        if Path(str(record.get("path", ""))).resolve() != summary_artifacts[role]:
            raise ValueError(f"{arm} seed {seed} summary {role} artifact path differs")
        if str(record.get("sha256", "")).lower() != artifact_sha256[role]:
            raise ValueError(
                f"{arm} seed {seed} summary {role} artifact SHA-256 differs"
            )

    protocol = report.get("evaluation_protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"{arm} seed {seed} has no evaluation protocol")
    if (
        protocol.get("split") != "mapping_only"
        or protocol.get("query_selection") != "uniform_mapping_gate"
        or not _is_exact_int(protocol.get("requested_query_count"), 256)
        or not _is_exact_int(protocol.get("evaluated_query_count"), 256)
        or not _is_exact_int(
            protocol.get("teacher_query_count"),
            int(arm_audit["teacher_query_count"]),
        )
        or not _is_exact_int(protocol.get("deployment_row_limit"), 0)
    ):
        raise ValueError(f"{arm} seed {seed} evaluation protocol differs")
    expected_indices = arm_audit["uniform_q256_indices"]
    actual_indices = protocol.get("selected_query_indices")
    if (
        not isinstance(actual_indices, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in actual_indices
        )
        or actual_indices != expected_indices
    ):
        raise ValueError(f"{arm} seed {seed} selected query indices differ")
    protocol_hashes = {
        "ordered_teacher_query_names_sha256": arm_audit[
            "ordered_teacher_query_names_sha256"
        ],
        "selected_query_indices_sha256": arm_audit["uniform_q256_indices_sha256"],
        "selected_query_names_sha256": arm_audit["uniform_q256_query_names_sha256"],
    }
    for name, expected in protocol_hashes.items():
        if protocol.get(name) != expected:
            raise ValueError(f"{arm} seed {seed} {name} differs")
    descriptor_protocol = protocol.get("descriptor_protocol")
    if extra_artifacts:
        if descriptor_protocol != {
            "kind": "equal_energy_descriptor_factor",
            "factor_id": arm_audit["descriptor_factor_id"],
            "source_descriptor_dim": 256,
            "xfeat_descriptor_dim": 64,
            "effective_descriptor_dim": 320,
            "strict_identity_metric": True,
            "one_materialized_bank": True,
            "one_global_top1": True,
            "one_poselib_call_per_query": True,
        }:
            raise ValueError(f"{arm} seed {seed} descriptor protocol differs")
    elif descriptor_protocol not in (
        None,
        {
            "kind": "canonical_query_cache_shared_metric",
            "descriptor_cache_equals_query_cache": True,
        },
    ):
        raise ValueError(f"{arm} seed {seed} baseline descriptor protocol differs")
    summary = dict(report.get("summary", {}))
    if not _is_exact_int(summary.get("query_count"), 256):
        raise ValueError(f"{arm} seed {seed} nested summary query count differs")
    return {
        name: _numeric_metric(summary, name, seed=seed, arm=arm)
        for name in SUMMARY_METRICS
    }


def _per_seed_comparison(*, baseline: dict, variant: dict, thresholds: dict) -> dict:
    delta = {
        name: float(variant[name]) - float(baseline[name]) for name in SUMMARY_METRICS
    }
    config = thresholds["per_seed_non_regression"]
    translation_tolerance = {
        name: max(
            float(config["translation_absolute_tolerance_cm"]),
            float(config["translation_relative_tolerance"]) * float(baseline[name]),
        )
        for name in TRANSLATION_METRICS
    }
    rotation_tolerance = {
        name: max(
            float(config["rotation_absolute_tolerance_deg"]),
            float(config["rotation_relative_tolerance"]) * float(baseline[name]),
        )
        for name in ROTATION_METRICS
    }
    checks = {
        "raw_precision_non_regression": delta["raw_gt_precision_percent"]
        >= -float(config["raw_precision_tolerance_pp"]),
        "median_te_non_regression": delta["median_te_cm"]
        <= translation_tolerance["median_te_cm"],
        "mean_te_non_regression": delta["mean_te_cm"]
        <= translation_tolerance["mean_te_cm"],
        "p90_te_non_regression": delta["p90_te_cm"]
        <= translation_tolerance["p90_te_cm"],
        "cvar95_te_non_regression": delta["cvar95_te_cm"]
        <= translation_tolerance["cvar95_te_cm"],
        "median_ae_non_regression": delta["median_ae_deg"]
        <= rotation_tolerance["median_ae_deg"],
        "mean_ae_non_regression": delta["mean_ae_deg"]
        <= rotation_tolerance["mean_ae_deg"],
        "p90_ae_non_regression": delta["p90_ae_deg"]
        <= rotation_tolerance["p90_ae_deg"],
        "p95_ae_non_regression": delta["p95_ae_deg"]
        <= rotation_tolerance["p95_ae_deg"],
        "recall_5cm_5deg_non_regression": delta["recall_5cm_5deg_percent"]
        >= -float(config["recall_5cm_5deg_tolerance_pp"]),
        "catastrophic_count_non_regression": delta["catastrophic_100cm_count"] <= 0,
    }
    return {
        "baseline": baseline,
        "variant": variant,
        "delta_variant_minus_baseline": delta,
        "translation_tolerance_cm": translation_tolerance,
        "rotation_tolerance_deg": rotation_tolerance,
        "checks": checks,
        "passes_non_regression": all(checks.values()),
    }


def _three_seed_mean(per_seed: dict[str, dict], thresholds: dict) -> dict:
    baseline = {
        name: sum(float(row["baseline"][name]) for row in per_seed.values())
        / len(per_seed)
        for name in SUMMARY_METRICS
    }
    variant = {
        name: sum(float(row["variant"][name]) for row in per_seed.values())
        / len(per_seed)
        for name in SUMMARY_METRICS
    }
    delta = {name: variant[name] - baseline[name] for name in SUMMARY_METRICS}
    gain = thresholds["three_seed_mean_substantive_improvement"]
    checks = {
        "raw_precision": delta["raw_gt_precision_percent"]
        >= float(gain["raw_precision_pp"]),
        "median_te": -delta["median_te_cm"] >= float(gain["central_translation_cm"]),
        "mean_te": -delta["mean_te_cm"] >= float(gain["central_translation_cm"]),
        "p90_te": -delta["p90_te_cm"] >= float(gain["tail_translation_cm"]),
        "cvar95_te": -delta["cvar95_te_cm"] >= float(gain["tail_translation_cm"]),
        "median_ae": -delta["median_ae_deg"] >= float(gain["rotation_deg"]),
        "mean_ae": -delta["mean_ae_deg"] >= float(gain["rotation_deg"]),
        "p90_ae": -delta["p90_ae_deg"] >= float(gain["rotation_deg"]),
        "p95_ae": -delta["p95_ae_deg"] >= float(gain["rotation_deg"]),
        "recall_5cm_5deg": delta["recall_5cm_5deg_percent"]
        >= float(gain["recall_5cm_5deg_pp"]),
    }
    return {
        "baseline": baseline,
        "variant": variant,
        "delta_variant_minus_baseline": delta,
        "substantive_improvement_checks": checks,
        "has_substantive_improvement": any(checks.values()),
    }


def compare_mapping_pose_gate(
    *,
    baseline_summaries: Mapping[int, str | Path],
    variant_summaries: Mapping[int, str | Path],
    baseline_artifacts: Mapping[str, str | Path],
    variant_artifacts: Mapping[str, str | Path],
    thresholds: dict | None = None,
    expected_sha256: Mapping[str, str] | None = None,
    variant_descriptor_factor: str | Path | None = None,
) -> dict:
    """Audit paired inputs and apply the preregistered mapping-pose gate."""
    thresholds = _normalize_thresholds(thresholds)
    evaluation_code = mapping_pose_evaluation_code_identity(require_clean=True)
    artifacts = {
        "baseline": _resolve_artifacts(baseline_artifacts, arm="baseline"),
        "variant": _resolve_artifacts(variant_artifacts, arm="variant"),
    }
    summaries = {
        "baseline": _resolve_summaries(baseline_summaries, arm="baseline"),
        "variant": _resolve_summaries(variant_summaries, arm="variant"),
    }
    all_summary_paths = [
        summaries[arm][seed]
        for arm in ("baseline", "variant")
        for seed in REQUIRED_SEEDS
    ]
    if len(set(all_summary_paths)) != len(all_summary_paths):
        raise ValueError("all six seed summaries must be distinct files")

    descriptor_factor = None
    extra_artifacts = {"baseline": {}, "variant": {}}
    extra_input_paths = {}
    if variant_descriptor_factor is not None:
        expected_factor_sha256 = dict(expected_sha256 or {}).get(
            "variant.descriptor_factor"
        )
        if expected_factor_sha256 is None:
            raise ValueError(
                "descriptor factor requires expected SHA-256 for "
                "variant.descriptor_factor"
            )
        descriptor_factor = audit_descriptor_factor_pair(
            variant_descriptor_factor,
            source_map_path=artifacts["baseline"]["map"],
            source_metric_path=artifacts["baseline"]["metric"],
            source_query_cache_path=artifacts["baseline"]["query_cache"],
            teacher_path=artifacts["baseline"]["teacher"],
            calibration_path=artifacts["baseline"]["calibration"],
            variant_map_path=artifacts["variant"]["map"],
            variant_metric_path=artifacts["variant"]["metric"],
            variant_query_cache_path=artifacts["variant"]["query_cache"],
            variant_teacher_path=artifacts["variant"]["teacher"],
            variant_calibration_path=artifacts["variant"]["calibration"],
        )
        if descriptor_factor["producer_git_commit"] != evaluation_code["git_commit"]:
            raise ValueError(
                "descriptor-factor producer and pose evaluator Git commits differ"
            )
        extra_artifacts["variant"] = {
            "descriptor_factor": descriptor_factor["path"],
        }
        extra_input_paths = {
            "variant.descriptor_factor": descriptor_factor["path"],
        }

    input_records, actual_sha256 = _hash_inputs(
        artifacts=artifacts,
        summaries=summaries,
        expected_sha256=expected_sha256,
        extra_paths=extra_input_paths,
    )
    arm_audits = {
        arm: _audit_arm_artifacts(arm=arm, paths=artifacts[arm])
        for arm in ("baseline", "variant")
    }
    baseline_audit = arm_audits["baseline"]
    variant_audit = arm_audits["variant"]
    if descriptor_factor is not None:
        variant_audit["descriptor_factor_id"] = descriptor_factor["factor_id"]
        variant_audit["descriptor_factor"] = {
            "path": str(descriptor_factor["path"]),
            "sha256": descriptor_factor["sha256"],
            "descriptor_cache": str(descriptor_factor["descriptor_cache_path"]),
            "descriptor_cache_sha256": descriptor_factor[
                "descriptor_cache_sha256"
            ],
            "source_metric_descriptor_dim": descriptor_factor[
                "source_metric_descriptor_dim"
            ],
            "variant_metric_descriptor_dim": descriptor_factor[
                "variant_metric_descriptor_dim"
            ],
            "strict_identity_metric": descriptor_factor["strict_identity_metric"],
            "producer_git_commit": descriptor_factor["producer_git_commit"],
            "deployment_extension": descriptor_factor["contract"][
                "deployment_extension_audit"
            ],
            "anchor_registry_bitwise_equal": descriptor_factor[
                "anchor_registry_bitwise_equal"
            ],
            "teacher_anchor_map_registry_bitwise_equal": descriptor_factor[
                "teacher_anchor_map_registry_bitwise_equal"
            ],
        }
    lineage_checks = {
        "ordered_teacher_query_names_equal": (
            baseline_audit["ordered_teacher_query_names_sha256"]
            == variant_audit["ordered_teacher_query_names_sha256"]
        ),
        "uniform_q256_indices_equal": (
            baseline_audit["uniform_q256_indices_sha256"]
            == variant_audit["uniform_q256_indices_sha256"]
        ),
        "uniform_q256_query_names_equal": (
            baseline_audit["uniform_q256_query_names_sha256"]
            == variant_audit["uniform_q256_query_names_sha256"]
        ),
        "calibration_statistics_equal": (
            baseline_audit["calibration_numeric_contract"]["statistics"]
            == variant_audit["calibration_numeric_contract"]["statistics"]
        ),
        "calibration_parameters_equal": (
            baseline_audit["calibration_numeric_contract"]["parameters"]
            == variant_audit["calibration_numeric_contract"]["parameters"]
        ),
        "calibration_policy_equal": (
            baseline_audit["calibration_numeric_contract"]["policy"]
            == variant_audit["calibration_numeric_contract"]["policy"]
        ),
    }
    if descriptor_factor is not None:
        lineage_checks.update(
            {
                "descriptor_factor_contract_valid": True,
                "anchor_registry_bitwise_equal": (
                    descriptor_factor["anchor_registry_bitwise_equal"] is True
                ),
                "teacher_anchor_map_registry_bitwise_equal": (
                    descriptor_factor[
                        "teacher_anchor_map_registry_bitwise_equal"
                    ]
                    is True
                ),
                "strict_320d_identity_metric": (
                    descriptor_factor["strict_identity_metric"] is True
                ),
                "query_caches_descriptor_factor_equivalent": True,
                "teacher_rebind_only": descriptor_factor["teacher_rebind_only"]
                is True,
                "calibration_rebind_only": descriptor_factor[
                    "calibration_rebind_only"
                ]
                is True,
                "one_global_top1_and_one_poselib": (
                    descriptor_factor["contract"]["checks"]["one_global_top1"]
                    is True
                    and descriptor_factor["contract"]["checks"][
                        "one_poselib_call_per_query"
                    ]
                    is True
                ),
                "deployment_extension_compiled_before_pose": (
                    descriptor_factor["contract"]["deployment_extension_audit"][
                        "compiled_expected_support_exact"
                    ]
                    is True
                    and descriptor_factor["contract"][
                        "deployment_extension_audit"
                    ]["proxy_to_deployment_transfer_preregistered"]
                    is True
                    and descriptor_factor["contract"][
                        "deployment_extension_audit"
                    ]["stairs_q256_three_seed_tail_gate_required"]
                    is True
                ),
            }
        )
    else:
        lineage_checks.update(
            {
                "query_cache_paths_equal": (
                    artifacts["baseline"]["query_cache"]
                    == artifacts["variant"]["query_cache"]
                ),
                "query_cache_sha256_equal": (
                    actual_sha256["baseline.query_cache"]
                    == actual_sha256["variant.query_cache"]
                ),
            }
        )
    if not all(lineage_checks.values()):
        failed = [name for name, passed in lineage_checks.items() if not passed]
        raise ValueError(f"paired mapping-pose lineage differs: {failed}")

    # Keep compact indices/hashes in the JSON; never serialize all query names or
    # the large frozen numeric calibration payload twice.
    for audit in arm_audits.values():
        audit.pop("query_names")
        numeric = audit.pop("calibration_numeric_contract")
        audit["calibration_numeric_sha256"] = {
            name: _json_sha256(value) for name, value in numeric.items()
        }

    loaded = {
        arm: {
            seed: _load_summary(
                arm=arm,
                seed=seed,
                path=summaries[arm][seed],
                artifacts=artifacts[arm],
                artifact_sha256={
                    role: actual_sha256[f"{arm}.{role}"]
                    for role in (*ARTIFACT_ROLES, *extra_artifacts[arm])
                },
                arm_audit=arm_audits[arm],
                expected_evaluation_code=evaluation_code,
                extra_artifacts=extra_artifacts[arm],
            )
            for seed in REQUIRED_SEEDS
        }
        for arm in ("baseline", "variant")
    }
    per_seed = {
        str(seed): {
            "seed_binding": "embedded_report_and_explicit_cli",
            "baseline_summary": input_records[f"baseline.seed{seed}_summary"],
            "variant_summary": input_records[f"variant.seed{seed}_summary"],
            **_per_seed_comparison(
                baseline=loaded["baseline"][seed],
                variant=loaded["variant"][seed],
                thresholds=thresholds,
            ),
        }
        for seed in REQUIRED_SEEDS
    }
    means = _three_seed_mean(per_seed, thresholds)
    per_seed_safe = all(row["passes_non_regression"] for row in per_seed.values())
    substantive = bool(means["has_substantive_improvement"])
    passed = per_seed_safe and substantive
    if passed:
        reason = "ALL_SEEDS_SAFE_AND_SUBSTANTIVE_MEAN_GAIN"
    elif not per_seed_safe:
        reason = "PER_SEED_NON_REGRESSION_FAILED"
    else:
        reason = "NO_SUBSTANTIVE_THREE_SEED_MEAN_GAIN"

    return {
        "schema": "lafgs_mapping_pose_pair_gate",
        "version": 1,
        "valid": True,
        "uses_test_queries": False,
        "comparison": "baseline_vs_single_factor_variant",
        "preregistered_protocol": {
            "seeds": list(REQUIRED_SEEDS),
            "query_count": 256,
            "query_selection": "uniform_mapping_gate",
            "deployment_row_limit": 0,
            "thresholds": thresholds,
            "descriptor_factor": (
                {
                    "required": True,
                    "factor_id": descriptor_factor["factor_id"],
                    "formula": descriptor_factor["contract"]["formula"],
                    "effective_descriptor_dim": 320,
                }
                if descriptor_factor is not None
                else {"required": False}
            ),
        },
        "lineage": {
            "checks": lineage_checks,
            "evaluation_code": evaluation_code,
            "arms": arm_audits,
            "inputs": input_records,
        },
        "per_seed": per_seed,
        "three_seed_mean": means,
        "decision": {
            "verdict": "PASS" if passed else "STOP",
            "reason": reason,
            "all_per_seed_non_regression_checks_pass": per_seed_safe,
            "has_substantive_three_seed_mean_improvement": substantive,
            "authorizes_next_stage": passed,
            "authorized_next_stage": (
                "12Scenes/office2_5b_mapping_pose_tail_guard" if passed else None
            ),
            "authorizes_deployment": False,
            "establishes_test_accuracy": False,
        },
    }


def _parse_named_paths(values: list[str], *, label: str) -> dict[int, Path]:
    result = {}
    for raw in values:
        seed_text, separator, path_text = str(raw).partition("=")
        if not separator or not seed_text or not path_text:
            raise ValueError(f"{label} must use SEED=PATH syntax")
        seed = int(seed_text)
        if seed in result:
            raise ValueError(f"duplicate {label} seed: {seed}")
        result[seed] = Path(path_text)
    return result


def _parse_expected_sha256(values: list[str]) -> dict[str, str]:
    result = {}
    for raw in values:
        name, separator, digest = str(raw).partition("=")
        if not separator or not name or not digest:
            raise ValueError("--expected-sha256 must use ARM.ROLE=SHA256 syntax")
        if name in result:
            raise ValueError(f"duplicate expected SHA-256 key: {name}")
        result[name] = digest
    return result


def _artifact_args(args, arm: str) -> dict[str, Path]:
    return {role: getattr(args, f"{arm}_{role}") for role in ARTIFACT_ROLES}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-seed", action="append", required=True)
    parser.add_argument("--variant-seed", action="append", required=True)
    parser.add_argument(
        "--variant-descriptor-factor",
        type=Path,
        help=(
            "Required for the 320D equal-energy arm; binds its descriptor cache "
            "and proves descriptor-only mutation."
        ),
    )
    for arm in ("baseline", "variant"):
        for role in ARTIFACT_ROLES:
            parser.add_argument(
                f"--{arm}-{role.replace('_', '-')}", type=Path, required=True
            )
    parser.add_argument(
        "--expected-sha256",
        action="append",
        default=[],
        help=(
            "Optional ARM.ROLE=SHA256 lock, including seed2026_summary roles; "
            "repeat for each known digest. The variant descriptor-factor SHA "
            "is mandatory when that factor is enabled."
        ),
    )
    parser.add_argument(
        "--thresholds-json",
        type=Path,
        help="Optional complete threshold contract; defaults are preregistered P7 values.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    thresholds = None
    if args.thresholds_json is not None:
        thresholds = json.loads(args.thresholds_json.read_text())
    report = compare_mapping_pose_gate(
        baseline_summaries=_parse_named_paths(
            args.baseline_seed, label="--baseline-seed"
        ),
        variant_summaries=_parse_named_paths(args.variant_seed, label="--variant-seed"),
        baseline_artifacts=_artifact_args(args, "baseline"),
        variant_artifacts=_artifact_args(args, "variant"),
        thresholds=thresholds,
        expected_sha256=_parse_expected_sha256(args.expected_sha256),
        variant_descriptor_factor=args.variant_descriptor_factor,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
