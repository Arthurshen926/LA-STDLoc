"""Materialize and audit the fixed SP-metric + XFeat descriptor factor.

The factor deliberately changes only descriptor-bearing tensors.  Geometry,
anchor identity, topology, the complete-positive teacher, and the canonical
mapping query cache remain frozen.  A separate descriptor cache is consumed by
the mapping replay only after its JSON contract has been verified.
"""

from __future__ import annotations

from collections.abc import Mapping
import gc
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from evidence.mapping_density_factor import audit_sparse_refresh_equivalence
from localization.localizer import load_shared_metric
from map_learning.context_booster_crossfit import accumulate_view_descriptors
from map_learning.frontend_upper_bound import (
    query_cache_queries,
    tensor_sha256,
    validate_probe,
)
from map_learning.metric import SharedLowRankMetric
from map_learning.repeated_assignment_audit import _selected_csr_edges


FACTOR_SCHEMA = "lafgs_mapping_equal_energy_descriptor_factor"
FACTOR_VERSION = 1
SOURCE_DESCRIPTOR_DIM = 256
XFEAT_DESCRIPTOR_DIM = 64
EFFECTIVE_DESCRIPTOR_DIM = 320
REGISTRY_FIELDS = (
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
MAP_DESCRIPTOR_MUTATIONS = frozenset(
    {
        "anchor_features",
        "v7_metric_raw_features",
        "v7_anchor_residual_parameter",
        "v7_anchor_residual",
        "v7_online_metric",
        "descriptor_factor",
    }
)
CACHE_TOP_LEVEL_MUTATIONS = frozenset(
    {"signature", "signature_payload", "descriptor_factor"}
)
CACHE_QUERY_MUTATIONS = frozenset({"native_descriptors"})
TEACHER_REBIND_MUTATIONS = frozenset({"anchor_map", "query_cache"})
CALIBRATION_REBIND_MUTATIONS = frozenset({"sources"})
MECHANISM_GATE_FIELDS = frozenset(
    {
        "selection_to_gate_candidate_r1_strictly_positive",
        "gate_to_selection_candidate_r1_strictly_positive",
        "pooled_r8_non_regression",
        "pooled_track_core_r1_non_regression",
        "pooled_gaussian_reserve_r1_non_regression",
    }
)
DEPLOYMENT_EXTENSION_SCHEMA = (
    "lafgs_equal_energy_descriptor_deployment_extension_preregistration"
)
DEPLOYMENT_ESTIMATOR = (
    "uniform_mean_of_all_available_mapping_views_after_per_view_positive_mean"
)
MECHANISM_SP_BRANCH = (
    "raw_l2_superpoint_query_vs_crossfit_positive_view_superpoint_bank"
)
DEPLOYMENT_SCORE = (
    "0.5*cos(frozen_v3_metric_superpoint_query,frozen_v3_anchor_features)"
    "+0.5*cos(xfeat_query,all_mapping_view_xfeat_bank)"
)
DEPLOYMENT_ADJUDICATION = {
    "stairs_mapping_pose_gate": {
        "query_selection": "uniform_mapping_gate",
        "query_count": 256,
        "seeds": [2026, 2027, 2028],
        "requires_translation_rotation_recall_and_catastrophe_non_regression": True,
        "requires_tail_metrics": ["p90_te_cm", "cvar95_te_cm"],
    },
    "post_stairs_pass_required_guard_order": [
        "12Scenes/office2_5b_mapping_pose_tail_non_regression",
        "outdoor_mapping_guard",
    ],
    "office2_5b": {
        "mapping_only": True,
        "no_threshold_tuning": True,
        "catastrophic_100cm_count_tolerance": 0,
        "requires_tail_non_regression": True,
    },
    "formal_test_frozen": True,
}
QUERY_AUDIT_FIELDS = (
    "query_count",
    "descriptor_row_count",
    "ordered_query_names_sha256",
    "native_keypoint_registry_sha256",
    "candidate_descriptor_registry_sha256",
)
SUPPORT_AUDIT_FIELDS = (
    "query_count",
    "positive_edge_count",
    "anchor_view_count",
    "supported_anchor_count",
    "unsupported_anchor_count",
    "minimum_support_views",
    "mechanism_crossfit_minimum_support_views",
    "full_mapping_minimum_support_views",
    "mechanism_support_domain_min",
    "deployment_estimator_domain_min",
    "single_view_extension_preregistered",
    "deployment_estimator",
    "full_mapping_support_policy",
    "single_view_anchor_count",
    "single_view_anchor_indices_sha256",
    "single_view_anchor_ids_sha256",
    "single_view_anchor_type_histogram",
    "support_view_counts_sha256",
    "xfeat_query_descriptor_registry_sha256",
    "candidate_anchor_features_sha256",
)


def _load_mmap(path: str | Path) -> dict:
    try:
        return torch.load(
            path, map_location="cpu", weights_only=False, mmap=True
        )
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def descriptor_factor_producer_identity(*, require_clean: bool = True) -> dict:
    repository = Path(__file__).resolve().parents[1]
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    clean = not status.strip()
    if require_clean and not clean:
        raise RuntimeError("descriptor-factor materialization requires a clean worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    entrypoints = (
        "map_learning/equal_energy_descriptor_factor.py",
        "scripts/materialize_equal_energy_descriptor_factor.py",
        "scripts/evaluate_mapping_cache.py",
        "scripts/compare_mapping_pose_gate.py",
    )
    return {
        "schema": "lafgs_equal_energy_descriptor_factor_producer_code",
        "version": 1,
        "repository": str(repository),
        "git_commit": commit,
        "git_worktree_clean": clean,
        "entrypoints": {
            relative: sha256_file(repository / relative) for relative in entrypoints
        },
    }


def _require_sha256(value: str, *, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    return digest


def _locked_file(
    path: str | Path, expected_sha256: str, *, label: str
) -> tuple[Path, dict[str, str]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} is not a file: {resolved}")
    expected = _require_sha256(expected_sha256, label=f"{label} SHA-256")
    actual = sha256_file(resolved)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return resolved, {"path": str(resolved), "sha256": actual}


def _equal_value(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        try:
            return torch.equal(torch.as_tensor(left), torch.as_tensor(right))
        except (TypeError, ValueError):
            return False
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _equal_value(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _equal_value(a, b) for a, b in zip(left, right)
        )
    return type(left) is type(right) and left == right


def _assert_only_fields_changed(
    source: Mapping,
    candidate: Mapping,
    *,
    allowed: frozenset[str],
    label: str,
) -> None:
    for field in sorted((set(source) | set(candidate)) - set(allowed)):
        if field not in source or field not in candidate:
            raise ValueError(f"{label} immutable field set differs at {field}")
        if not _equal_value(source[field], candidate[field]):
            raise ValueError(f"{label} immutable field differs at {field}")


def _assert_audit_fields_equal(
    independent: Mapping, recorded: Mapping, fields: tuple[str, ...], *, label: str
) -> None:
    for field in fields:
        if independent.get(field) != recorded.get(field):
            raise ValueError(f"{label} differs for {field}")


def _registry_hashes(state: Mapping) -> dict[str, str]:
    hashes = {}
    count = int(torch.as_tensor(state["anchor_ids"]).numel())
    for field in REGISTRY_FIELDS:
        if field not in state:
            raise ValueError(f"anchor map misses registry field {field}")
        value = torch.as_tensor(state[field])
        if value.shape[0] != count:
            raise ValueError(f"anchor registry field {field} is not row-aligned")
        hashes[field] = tensor_sha256(value)
    return hashes


def _metric_is_strict_identity(metric_state: Mapping, *, anchor_ids: torch.Tensor) -> bool:
    if metric_state.get("schema") != "lafgs_shared_metric_state":
        return False
    config = dict(metric_state.get("metric_config", {}))
    if (
        int(config.get("descriptor_dim", -1)) != EFFECTIVE_DESCRIPTOR_DIM
        or float(config.get("max_residual_norm", -1.0)) != 0.0
    ):
        return False
    if not torch.equal(
        torch.as_tensor(metric_state.get("landmark_indices")).long().reshape(-1),
        torch.as_tensor(anchor_ids).long().reshape(-1),
    ):
        return False
    values = metric_state.get("metric_state_dict", {})
    expected = {"down.weight", "down.bias", "up.weight"}
    return set(values) == expected and all(
        bool((torch.as_tensor(value) == 0).all()) for value in values.values()
    )


def _validate_equivalence_report(
    report: Mapping,
    *,
    source_record: Mapping[str, str],
    refreshed_record: Mapping[str, str],
    observed: Mapping,
) -> None:
    if (
        report.get("schema") != "lafgs_mapping_sparse_refresh_equivalence"
        or report.get("version") != 2
        or report.get("uses_test_queries") is not False
        or report.get("valid") is not True
    ):
        raise ValueError("query-cache equivalence is not a valid mapping-only V2 report")
    checks = report.get("checks")
    if not isinstance(checks, Mapping) or not checks or not all(checks.values()):
        raise ValueError("query-cache equivalence checks did not all pass")
    sources = report.get("sources", {})
    for name, expected in (
        ("source_cache", source_record),
        ("refreshed_cache", refreshed_record),
    ):
        record = sources.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"query-cache equivalence misses {name}")
        if Path(str(record.get("path", ""))).resolve() != Path(expected["path"]):
            raise ValueError(f"query-cache equivalence names a different {name}")
        if str(record.get("sha256", "")).lower() != expected["sha256"]:
            raise ValueError(f"query-cache equivalence SHA differs for {name}")
    report_audit = dict(report.get("audit", {}))
    if report_audit != dict(observed):
        raise ValueError("serialized and freshly recomputed cache equivalence differ")
    if observed.get("content_equivalent_track_payload_reuse_authorized") is not True:
        raise ValueError("source and refreshed query caches are not exactly equivalent")


def _artifact_binding_exact(
    value: object, *, expected: Mapping[str, str], label: str
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"mechanism evidence misses {label} artifact binding")
    if Path(str(value.get("path", ""))).expanduser().resolve() != Path(
        expected["path"]
    ):
        raise ValueError(f"mechanism evidence names a different {label}")
    if str(value.get("sha256", "")).lower() != expected["sha256"]:
        raise ValueError(f"mechanism evidence SHA-256 differs for {label}")


def _validate_mechanism_go(
    *,
    report: Mapping,
    gate: Mapping,
    report_record: Mapping[str, str],
    source_map_record: Mapping[str, str],
    refreshed_query_cache_record: Mapping[str, str],
    teacher_record: Mapping[str, str],
    probe_record: Mapping[str, str],
    xfeat_weights_record: Mapping[str, str],
) -> dict:
    if (
        report.get("schema") != "lafgs_frontend_ceiling_probe_audit_bundle"
        or report.get("version") != 1
        or report.get("mapping_only") is not True
        or report.get("uses_test_queries") is not False
        or report.get("deployment_modified") is not False
    ):
        raise ValueError("equal-energy descriptor report is not a frozen mapping audit")
    descriptor = report.get("descriptor_identity", {})
    protocol = descriptor.get("protocol", {})
    if (
        descriptor.get("schema")
        != "lafgs_mapping_descriptor_equal_energy_ceiling_probe"
        or descriptor.get("mapping_only") is not True
        or descriptor.get("uses_test_queries") is not False
        or protocol.get("candidate_representation")
        != "l2_concat(l2(superpoint),l2(candidate))/sqrt(2)"
        or protocol.get("score_identity")
        != "0.5*cosine_superpoint+0.5*cosine_candidate"
        or protocol.get("ranking") != "single_global_cosine"
        or protocol.get("map_bank")
        != "same_positive_edges_view_balanced_support_only"
        or protocol.get("query_coordinates")
        != "exact_frozen_superpoint_keypoint_rows"
        or protocol.get("learned_fusion_parameters") is not False
        or protocol.get("source_specific_descriptor_routing") is not False
        or protocol.get("candidate_detector_used") is not False
        or int(protocol.get("source_candidate_descriptor_dim", -1))
        != XFEAT_DESCRIPTOR_DIM
        or int(protocol.get("effective_candidate_descriptor_dim", -1))
        != EFFECTIVE_DESCRIPTOR_DIM
        or int(protocol.get("minimum_support_views", -1)) != 2
    ):
        raise ValueError("equal-energy descriptor report protocol differs")
    if Path(str(report.get("probe_cache", ""))).expanduser().resolve() != Path(
        probe_record["path"]
    ):
        raise ValueError("equal-energy descriptor report names a different probe")
    report_sources = report.get("source_artifacts", {})
    for name, expected in (
        ("state", source_map_record),
        ("query_cache", refreshed_query_cache_record),
        ("teacher", teacher_record),
        ("probe_cache", probe_record),
    ):
        _artifact_binding_exact(
            report_sources.get(name), expected=expected, label=f"report.{name}"
        )
    attestation = descriptor.get("attestation", {})
    _artifact_binding_exact(
        attestation.get("artifact"),
        expected=xfeat_weights_record,
        label="report.XFeat weights",
    )
    reference_artifacts = attestation.get("reference_artifacts", {})
    _artifact_binding_exact(
        reference_artifacts.get("query_cache"),
        expected=refreshed_query_cache_record,
        label="report reference query cache",
    )
    _artifact_binding_exact(
        reference_artifacts.get("teacher"),
        expected=teacher_record,
        label="report reference teacher",
    )
    if (
        int(attestation.get("reference_descriptor_dim", -1))
        != SOURCE_DESCRIPTOR_DIM
        or int(attestation.get("candidate_descriptor_dim", -1))
        != XFEAT_DESCRIPTOR_DIM
        or int(attestation.get("query_count", -1)) <= 0
        or int(attestation.get("validated_descriptor_rows", -1)) <= 0
    ):
        raise ValueError("equal-energy descriptor report attestation differs")

    if (
        gate.get("schema")
        != "lafgs_frontend_descriptor_equal_energy_mechanism_gate"
        or gate.get("version") != 1
        or gate.get("valid") is not True
        or gate.get("mapping_only") is not True
        or gate.get("uses_test_queries") is not False
        or gate.get("mechanism_gate_passed") is not True
        or gate.get("advance_to_mapping_only_descriptor_rebuild") is not True
        or gate.get("decision") != "GO"
        or gate.get("single_factor")
        != "equal_energy_single_descriptor_at_exact_superpoint_rows"
    ):
        raise ValueError("equal-energy mechanism evidence is not a mapping-only GO")
    gates = gate.get("gates")
    if (
        not isinstance(gates, Mapping)
        or set(gates) != set(MECHANISM_GATE_FIELDS)
        or not all(value is True for value in gates.values())
    ):
        raise ValueError("equal-energy mechanism GO gates did not all pass")
    gate_protocol = gate.get("protocol", {})
    if (
        gate_protocol.get("candidate_representation")
        != "equal_energy_superpoint_candidate"
        or int(gate_protocol.get("source_candidate_descriptor_dim", -1))
        != XFEAT_DESCRIPTOR_DIM
        or int(gate_protocol.get("effective_candidate_descriptor_dim", -1))
        != EFFECTIVE_DESCRIPTOR_DIM
        or gate_protocol.get("crossfit") != "bidirectional_temporal_block"
        or int(gate_protocol.get("minimum_support_views", -1)) != 2
    ):
        raise ValueError("equal-energy mechanism gate protocol differs")
    gate_inputs = gate.get("inputs", {})
    _artifact_binding_exact(
        gate_inputs.get("descriptor_report"),
        expected=report_record,
        label="gate descriptor report",
    )
    _artifact_binding_exact(
        gate_inputs.get("candidate_weights"),
        expected=xfeat_weights_record,
        label="gate XFeat weights",
    )
    gate_sources = gate_inputs.get("source_artifacts", {})
    for name, expected in (
        ("state", source_map_record),
        ("query_cache", refreshed_query_cache_record),
        ("teacher", teacher_record),
        ("probe_cache", probe_record),
    ):
        _artifact_binding_exact(
            gate_sources.get(name), expected=expected, label=f"gate.{name}"
        )
    if gate_inputs.get("evaluation_code") != report.get("evaluation_code"):
        raise ValueError("mechanism report and gate evaluation-code identity differ")
    evaluation_code = gate_inputs.get("evaluation_code", {})
    if (
        evaluation_code.get("schema")
        != "lafgs_frontend_descriptor_evaluation_code"
        or evaluation_code.get("version") != 1
        or evaluation_code.get("git_worktree_clean") is not True
    ):
        raise ValueError("mechanism evidence does not bind a clean evaluation revision")
    return {
        "report_schema": report["schema"],
        "gate_schema": gate["schema"],
        "decision": "GO",
        "all_gate_checks_passed": True,
        "report_gate_evaluation_code_equal": True,
        "source_artifacts_exact": True,
        "probe_and_weights_exact": True,
        "mapping_only": True,
        "uses_test_queries": False,
    }


def _validate_deployment_extension_preregistration(
    *,
    preregistration: Mapping,
    preregistration_record: Mapping[str, str],
    source_map_record: Mapping[str, str],
    source_metric_record: Mapping[str, str],
    teacher_record: Mapping[str, str],
    mechanism_report_record: Mapping[str, str],
    mechanism_gate_record: Mapping[str, str],
    support_audit: Mapping,
) -> dict:
    """Compile the pose-before-deployment extension against live support facts."""
    if (
        preregistration.get("schema") != DEPLOYMENT_EXTENSION_SCHEMA
        or preregistration.get("version") != 1
        or preregistration.get("valid") is not True
        or preregistration.get("mapping_only") is not True
        or preregistration.get("uses_test_queries") is not False
        or preregistration.get("single_factor")
        != "fixed_equal_energy_descriptor_representation"
    ):
        raise ValueError("descriptor deployment extension preregistration is invalid")
    sources = preregistration.get("source_artifacts", {})
    for name, expected in (
        ("source_map", source_map_record),
        ("source_metric", source_metric_record),
        ("teacher", teacher_record),
        ("mechanism_report", mechanism_report_record),
        ("mechanism_gate", mechanism_gate_record),
    ):
        _artifact_binding_exact(
            sources.get(name), expected=expected, label=f"deployment extension.{name}"
        )
    estimator = preregistration.get("estimator_extension", {})
    expected_estimator = {
        "mechanism_support_domain_min": 2,
        "deployment_estimator_domain_min": 1,
        "single_view_extension_preregistered": True,
        "estimator": DEPLOYMENT_ESTIMATOR,
        "all_frozen_anchors_retained": True,
        "no_unsupported_anchor_fallback": True,
        "no_source_or_anchor_type_routing": True,
        "no_anchor_removal": True,
        "anchor_ids_geometry_topology_frozen": True,
    }
    if estimator != expected_estimator:
        raise ValueError("descriptor deployment estimator extension differs")
    transfer = preregistration.get("proxy_to_deployment_transfer", {})
    expected_transfer = {
        "preregistered_before_pose": True,
        "mechanism_superpoint_branch": MECHANISM_SP_BRANCH,
        "deployment_score": DEPLOYMENT_SCORE,
        "mechanism_and_deployment_superpoint_banks_bitwise_identical": False,
        "reason": "preserve_the_frozen_v3_baseline_branch",
        "requires_pose_tail_and_cross_domain_adjudication": True,
    }
    if transfer != expected_transfer:
        raise ValueError("descriptor proxy-to-deployment transfer differs")
    if preregistration.get("adjudication") != DEPLOYMENT_ADJUDICATION:
        raise ValueError("descriptor deployment-extension adjudication differs")

    expected_support = preregistration.get("expected_support", {})
    observed_support = {
        "anchor_count": int(support_audit["supported_anchor_count"]),
        "single_view_anchor_count": int(support_audit["single_view_anchor_count"]),
        "single_view_anchor_indices_sha256": support_audit[
            "single_view_anchor_indices_sha256"
        ],
        "single_view_anchor_ids_sha256": support_audit[
            "single_view_anchor_ids_sha256"
        ],
        "single_view_anchor_type_histogram": support_audit[
            "single_view_anchor_type_histogram"
        ],
    }
    if expected_support != observed_support:
        raise ValueError(
            "live descriptor support differs from the preregistered deployment extension"
        )
    if (
        int(support_audit.get("mechanism_support_domain_min", -1)) != 2
        or int(support_audit.get("deployment_estimator_domain_min", -1)) != 1
        or support_audit.get("single_view_extension_preregistered") is not True
        or support_audit.get("deployment_estimator") != DEPLOYMENT_ESTIMATOR
    ):
        raise ValueError("compiled descriptor support-domain semantics differ")
    if int(support_audit["unsupported_anchor_count"]) != 0:
        raise ValueError("deployment extension cannot authorize unsupported anchors")
    if int(support_audit["minimum_support_views"]) != 1:
        raise ValueError("deployment extension is only defined for minimum support one")
    return {
        "preregistration": dict(preregistration_record),
        "mechanism_support_domain_min": 2,
        "deployment_estimator_domain_min": 1,
        "single_view_extension_preregistered": True,
        "compiled_expected_support_exact": True,
        "uniform_all_available_view_estimator": True,
        "no_fallback_routing_or_removal": True,
        "proxy_to_deployment_transfer_preregistered": True,
        "stairs_q256_three_seed_tail_gate_required": True,
        "office2_5b_tail_guard_required_after_stairs_pass": True,
        "outdoor_guard_required_after_office2_5b": True,
        "formal_test_frozen": True,
    }


def _build_xfeat_anchor_bank(
    *,
    teacher: Mapping,
    probe: Mapping,
    anchor_ids: torch.Tensor,
    anchor_type: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    anchor_ids = torch.as_tensor(anchor_ids).long().reshape(-1)
    anchor_type = torch.as_tensor(anchor_type).long().reshape(-1)
    anchor_count = int(anchor_ids.numel())
    if anchor_type.shape != (anchor_count,):
        raise ValueError("anchor type is not row-aligned for descriptor fusion")
    names = list(teacher["query_names"])
    records = list(teacher["records"])
    if not names or len(names) != len(records) or len(names) != len(set(names)):
        raise ValueError("teacher query registry is empty, duplicated, or incomplete")
    accumulator = torch.zeros((anchor_count, XFEAT_DESCRIPTOR_DIM), dtype=torch.float32)
    view_counts = torch.zeros(anchor_count, dtype=torch.long)
    positive_edge_count = 0
    observed_view_count = 0
    descriptor_digest = hashlib.sha256()
    for query_index, (name, record) in enumerate(zip(names, records)):
        if int(record.get("query_index", -1)) != query_index:
            raise ValueError(f"teacher query index differs at row {query_index}")
        if str(record.get("query_name", name)) != str(name):
            raise ValueError(f"teacher query name differs at row {query_index}")
        rows = torch.as_tensor(record["query_rows"]).long()
        selected = torch.arange(rows.numel())
        _, edge_rows, edge_anchors = _selected_csr_edges(record, "positive", selected)
        if edge_anchors.numel() and (
            int(edge_anchors.min()) < 0 or int(edge_anchors.max()) >= anchor_count
        ):
            raise ValueError(f"teacher positive anchor is out of range for {name}")
        candidate = F.normalize(
            torch.as_tensor(
                probe["queries"][name]["descriptor_at_reference_keypoints"]
            ).float(),
            dim=1,
        )
        if candidate.ndim != 2 or candidate.shape[1] != XFEAT_DESCRIPTOR_DIM:
            raise ValueError(f"XFeat descriptor dimension differs for {name}")
        if rows.numel() and int(rows.max()) >= candidate.shape[0]:
            raise ValueError(f"teacher rows exceed XFeat descriptor rows for {name}")
        descriptor_digest.update(name.encode("utf-8"))
        descriptor_digest.update(tensor_sha256(candidate).encode("ascii"))
        if edge_rows.numel():
            native_rows = rows[edge_rows]
            observed_view_count += accumulate_view_descriptors(
                accumulator,
                view_counts,
                edge_anchors,
                candidate[native_rows],
            )
            positive_edge_count += int(edge_rows.numel())
    unsupported = torch.nonzero(view_counts == 0, as_tuple=False).reshape(-1)
    if unsupported.numel():
        raise ValueError(
            "fixed equal-energy factor forbids unsupported-anchor fallback; "
            f"found {unsupported.numel()} unsupported anchors"
        )
    bank = F.normalize(accumulator, dim=1)
    if not bool(torch.isfinite(bank).all()):
        raise ValueError("XFeat anchor bank contains non-finite values")
    single_view = torch.nonzero(view_counts == 1, as_tuple=False).reshape(-1)
    single_types = anchor_type[single_view]
    type_histogram = {
        str(int(value)): int((single_types == value).sum())
        for value in torch.unique(single_types, sorted=True)
    }
    return bank, view_counts, {
        "query_count": len(names),
        "positive_edge_count": int(positive_edge_count),
        "anchor_view_count": int(observed_view_count),
        "supported_anchor_count": int((view_counts > 0).sum()),
        "unsupported_anchor_count": int(unsupported.numel()),
        "minimum_support_views": int(view_counts.min()),
        "mechanism_crossfit_minimum_support_views": 2,
        "full_mapping_minimum_support_views": 1,
        "mechanism_support_domain_min": 2,
        "deployment_estimator_domain_min": 1,
        "single_view_extension_preregistered": True,
        "deployment_estimator": DEPLOYMENT_ESTIMATOR,
        "full_mapping_support_policy": (
            "retain_all_frozen_anchors_with_at_least_one_mapping_view_no_fallback"
        ),
        "single_view_anchor_count": int(single_view.numel()),
        "single_view_anchor_indices_sha256": tensor_sha256(single_view),
        "single_view_anchor_ids_sha256": tensor_sha256(anchor_ids[single_view]),
        "single_view_anchor_type_histogram": type_histogram,
        "support_view_counts_sha256": tensor_sha256(view_counts),
        "xfeat_query_descriptor_registry_sha256": descriptor_digest.hexdigest(),
    }


def _candidate_query_cache(
    *,
    source_cache: Mapping,
    teacher: Mapping,
    probe: Mapping,
    source_metric: SharedLowRankMetric,
    factor_id: str,
) -> tuple[dict, dict]:
    if "queries" not in source_cache or not isinstance(source_cache["queries"], Mapping):
        raise ValueError("descriptor factor requires a versioned query cache payload")
    source_queries = query_cache_queries(source_cache)
    names = list(teacher["query_names"])
    if list(source_queries) != names:
        raise ValueError("source query-cache order differs from the teacher")
    output_queries = {}
    descriptor_digest = hashlib.sha256()
    keypoint_digest = hashlib.sha256()
    total_rows = 0
    for name in names:
        source = source_queries[name]
        superpoint = torch.as_tensor(source["native_descriptors"]).float()
        if superpoint.ndim != 2 or superpoint.shape[1] != SOURCE_DESCRIPTOR_DIM:
            raise ValueError(f"source descriptor dimension differs for {name}")
        xfeat = torch.as_tensor(
            probe["queries"][name]["descriptor_at_reference_keypoints"]
        ).float()
        if xfeat.shape != (superpoint.shape[0], XFEAT_DESCRIPTOR_DIM):
            raise ValueError(f"XFeat/source descriptor rows differ for {name}")
        adapted_superpoint, _ = source_metric(F.normalize(superpoint, dim=1))
        descriptor = torch.cat(
            (adapted_superpoint, F.normalize(xfeat, dim=1)), dim=1
        ) / (2.0**0.5)
        descriptor = descriptor.detach().cpu().float()
        if descriptor.shape[1] != EFFECTIVE_DESCRIPTOR_DIM:
            raise AssertionError("equal-energy descriptor dimension is not 320")
        if not torch.allclose(
            torch.linalg.norm(descriptor, dim=1),
            torch.ones(descriptor.shape[0]),
            atol=2e-6,
            rtol=0.0,
        ):
            raise ValueError(f"candidate descriptor is not unit length for {name}")
        candidate = dict(source)
        candidate["native_descriptors"] = descriptor
        _assert_only_fields_changed(
            source,
            candidate,
            allowed=CACHE_QUERY_MUTATIONS,
            label=f"query cache row {name}",
        )
        output_queries[name] = candidate
        descriptor_digest.update(name.encode("utf-8"))
        descriptor_digest.update(tensor_sha256(descriptor).encode("ascii"))
        keypoint_digest.update(name.encode("utf-8"))
        keypoint_digest.update(
            tensor_sha256(torch.as_tensor(source["native_keypoints"])).encode("ascii")
        )
        total_rows += int(descriptor.shape[0])

    output = dict(source_cache)
    output["queries"] = output_queries
    source_signature_payload = dict(source_cache.get("signature_payload", {}))
    signature_payload = {
        **source_signature_payload,
        "descriptor_source": "equal_energy_v3_metric_superpoint256_xfeat64",
        "descriptor_factor_id": factor_id,
        "source_superpoint_descriptor_dim": SOURCE_DESCRIPTOR_DIM,
        "source_xfeat_descriptor_dim": XFEAT_DESCRIPTOR_DIM,
        "effective_descriptor_dim": EFFECTIVE_DESCRIPTOR_DIM,
        "mapping_only": True,
        "uses_test_queries": False,
    }
    output["signature_payload"] = signature_payload
    output["signature"] = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    output["descriptor_factor"] = {
        "schema": FACTOR_SCHEMA,
        "version": FACTOR_VERSION,
        "factor_id": factor_id,
        "mapping_only": True,
        "uses_test_queries": False,
        "effective_descriptor_dim": EFFECTIVE_DESCRIPTOR_DIM,
    }
    _assert_only_fields_changed(
        source_cache,
        output,
        allowed=CACHE_TOP_LEVEL_MUTATIONS | {"queries"},
        label="query cache",
    )
    return output, {
        "query_count": len(names),
        "descriptor_row_count": int(total_rows),
        "ordered_query_names_sha256": _canonical_json_sha256(names),
        "native_keypoint_registry_sha256": keypoint_digest.hexdigest(),
        "candidate_descriptor_registry_sha256": descriptor_digest.hexdigest(),
    }


@torch.inference_mode()
def _audit_candidate_query_cache_formula(
    *,
    source_cache: Mapping,
    candidate_cache: Mapping,
    teacher: Mapping,
    probe: Mapping,
    source_metric: SharedLowRankMetric,
    factor_id: str,
) -> dict:
    """Independently replay every candidate descriptor and immutable query field."""
    if "queries" not in source_cache or "queries" not in candidate_cache:
        raise ValueError("live descriptor audit requires versioned query caches")
    _assert_only_fields_changed(
        source_cache,
        candidate_cache,
        allowed=CACHE_TOP_LEVEL_MUTATIONS | {"queries"},
        label="live candidate query cache",
    )
    source_queries = query_cache_queries(source_cache)
    candidate_queries = query_cache_queries(candidate_cache)
    names = list(teacher["query_names"])
    if list(source_queries) != names or list(candidate_queries) != names:
        raise ValueError("live candidate/source cache order differs from the teacher")
    if set(probe.get("queries", {})) != set(names):
        raise ValueError("live XFeat probe query registry differs from the teacher")
    expected_signature_payload = {
        **dict(source_cache.get("signature_payload", {})),
        "descriptor_source": "equal_energy_v3_metric_superpoint256_xfeat64",
        "descriptor_factor_id": factor_id,
        "source_superpoint_descriptor_dim": SOURCE_DESCRIPTOR_DIM,
        "source_xfeat_descriptor_dim": XFEAT_DESCRIPTOR_DIM,
        "effective_descriptor_dim": EFFECTIVE_DESCRIPTOR_DIM,
        "mapping_only": True,
        "uses_test_queries": False,
    }
    if candidate_cache.get("signature_payload") != expected_signature_payload:
        raise ValueError("live candidate query-cache signature payload differs")
    expected_signature = hashlib.sha256(
        json.dumps(expected_signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if candidate_cache.get("signature") != expected_signature:
        raise ValueError("live candidate query-cache signature differs")
    if candidate_cache.get("descriptor_factor") != {
        "schema": FACTOR_SCHEMA,
        "version": FACTOR_VERSION,
        "factor_id": factor_id,
        "mapping_only": True,
        "uses_test_queries": False,
        "effective_descriptor_dim": EFFECTIVE_DESCRIPTOR_DIM,
    }:
        raise ValueError("live candidate query-cache factor metadata differs")

    descriptor_digest = hashlib.sha256()
    keypoint_digest = hashlib.sha256()
    total_rows = 0
    for name in names:
        source = source_queries[name]
        candidate = candidate_queries[name]
        _assert_only_fields_changed(
            source,
            candidate,
            allowed=CACHE_QUERY_MUTATIONS,
            label=f"live candidate query {name}",
        )
        source_descriptor = torch.as_tensor(source["native_descriptors"]).float()
        candidate_descriptor = torch.as_tensor(
            candidate["native_descriptors"]
        ).float()
        xfeat = torch.as_tensor(
            probe["queries"][name]["descriptor_at_reference_keypoints"]
        ).float()
        if source_descriptor.ndim != 2 or source_descriptor.shape[1] != (
            SOURCE_DESCRIPTOR_DIM
        ):
            raise ValueError(f"live source descriptor dimension differs for {name}")
        if candidate_descriptor.shape != (
            source_descriptor.shape[0],
            EFFECTIVE_DESCRIPTOR_DIM,
        ):
            raise ValueError(f"live candidate descriptor shape differs for {name}")
        if xfeat.shape != (source_descriptor.shape[0], XFEAT_DESCRIPTOR_DIM):
            raise ValueError(f"live XFeat descriptor shape differs for {name}")
        if probe["queries"][name].get("reference_keypoints_sha256") != (
            tensor_sha256(torch.as_tensor(source["native_keypoints"]).float())
        ):
            raise ValueError(f"live XFeat keypoint registry differs for {name}")
        adapted_superpoint, _ = source_metric(
            F.normalize(source_descriptor, dim=1)
        )
        expected = torch.cat(
            (adapted_superpoint, F.normalize(xfeat, dim=1)), dim=1
        ) / (2.0**0.5)
        expected = expected.detach().cpu().float()
        if not torch.equal(candidate_descriptor, expected):
            maximum_error = float((candidate_descriptor - expected).abs().max())
            raise ValueError(
                f"live candidate descriptor formula differs for {name}: "
                f"max_abs_error={maximum_error}"
            )
        descriptor_digest.update(name.encode("utf-8"))
        descriptor_digest.update(tensor_sha256(candidate_descriptor).encode("ascii"))
        keypoint_digest.update(name.encode("utf-8"))
        keypoint_digest.update(
            tensor_sha256(torch.as_tensor(source["native_keypoints"])).encode("ascii")
        )
        total_rows += int(candidate_descriptor.shape[0])
    return {
        "query_count": len(names),
        "descriptor_row_count": int(total_rows),
        "ordered_query_names_sha256": _canonical_json_sha256(names),
        "native_keypoint_registry_sha256": keypoint_digest.hexdigest(),
        "candidate_descriptor_registry_sha256": descriptor_digest.hexdigest(),
        "all_nondescriptor_query_fields_bitwise_equal": True,
        "all_candidate_descriptors_formula_bitwise_equal": True,
        "formula": "concat(l2(v3_metric(superpoint256)),l2(xfeat64))/sqrt(2)",
    }


@torch.inference_mode()
def _audit_candidate_map_formula(
    *, source_map: Mapping, candidate_map: Mapping, teacher: Mapping, probe: Mapping
) -> dict:
    """Independently rebuild the view-balanced XFeat bank and final 320D map."""
    anchor_count = int(torch.as_tensor(source_map["anchor_ids"]).numel())
    accumulator = torch.zeros((anchor_count, XFEAT_DESCRIPTOR_DIM), dtype=torch.float32)
    view_counts = torch.zeros(anchor_count, dtype=torch.long)
    positive_edge_count = 0
    anchor_view_count = 0
    descriptor_digest = hashlib.sha256()
    names = list(teacher["query_names"])
    records = list(teacher["records"])
    if len(names) != len(records):
        raise ValueError("live map replay teacher registry is incomplete")
    for query_index, (name, record) in enumerate(zip(names, records)):
        if int(record.get("query_index", -1)) != query_index:
            raise ValueError(f"live map replay teacher index differs for {name}")
        rows = torch.as_tensor(record["query_rows"]).long().reshape(-1)
        offsets = torch.as_tensor(record["positive_offsets"]).long().reshape(-1)
        anchors = torch.as_tensor(record["positive_indices"]).long().reshape(-1)
        if offsets.shape != (rows.numel() + 1,):
            raise ValueError(f"live map replay positive CSR shape differs for {name}")
        counts = offsets[1:] - offsets[:-1]
        if (
            bool((counts < 0).any())
            or int(offsets[0]) != 0
            or int(offsets[-1]) != anchors.numel()
        ):
            raise ValueError(f"live map replay positive CSR is invalid for {name}")
        if not anchors.numel():
            continue
        if int(anchors.min()) < 0 or int(anchors.max()) >= anchor_count:
            raise ValueError(f"live map replay anchor is out of range for {name}")
        local_rows = torch.repeat_interleave(torch.arange(rows.numel()), counts)
        xfeat = F.normalize(
            torch.as_tensor(
                probe["queries"][name]["descriptor_at_reference_keypoints"]
            ).float(),
            dim=1,
        )
        descriptor_digest.update(name.encode("utf-8"))
        descriptor_digest.update(tensor_sha256(xfeat).encode("ascii"))
        if rows.numel() and int(rows.max()) >= xfeat.shape[0]:
            raise ValueError(f"live map replay teacher rows exceed XFeat rows for {name}")
        edge_descriptors = xfeat[rows[local_rows]]
        unique, inverse = torch.unique(anchors, sorted=True, return_inverse=True)
        per_view = torch.zeros((unique.numel(), XFEAT_DESCRIPTOR_DIM))
        per_view.index_add_(0, inverse, edge_descriptors)
        per_view = F.normalize(per_view, dim=1)
        accumulator.index_add_(0, unique, per_view)
        view_counts[unique] += 1
        positive_edge_count += int(anchors.numel())
        anchor_view_count += int(unique.numel())
    if bool((view_counts == 0).any()):
        raise ValueError("live map replay found an unsupported anchor")
    xfeat_bank = F.normalize(accumulator, dim=1)
    source_bank = F.normalize(
        torch.as_tensor(source_map["anchor_features"]).float(), dim=1
    )
    expected = torch.cat((source_bank, xfeat_bank), dim=1) / (2.0**0.5)
    actual = torch.as_tensor(candidate_map["anchor_features"]).float()
    if not torch.equal(actual, expected):
        maximum_error = float((actual - expected).abs().max())
        raise ValueError(
            "live candidate map formula differs: "
            f"max_abs_error={maximum_error}"
        )
    metadata = candidate_map.get("descriptor_factor", {})
    if not torch.equal(
        torch.as_tensor(metadata.get("support_view_counts")).long(), view_counts
    ):
        raise ValueError("live candidate map support-view counts differ")
    single_view = torch.nonzero(view_counts == 1, as_tuple=False).reshape(-1)
    anchor_ids = torch.as_tensor(source_map["anchor_ids"]).long().reshape(-1)
    anchor_type = torch.as_tensor(source_map["anchor_type"]).long().reshape(-1)
    single_types = anchor_type[single_view]
    type_histogram = {
        str(int(value)): int((single_types == value).sum())
        for value in torch.unique(single_types, sorted=True)
    }
    return {
        "query_count": len(names),
        "positive_edge_count": int(positive_edge_count),
        "anchor_view_count": int(anchor_view_count),
        "supported_anchor_count": int((view_counts > 0).sum()),
        "unsupported_anchor_count": int((view_counts == 0).sum()),
        "minimum_support_views": int(view_counts.min()),
        "mechanism_crossfit_minimum_support_views": 2,
        "full_mapping_minimum_support_views": 1,
        "mechanism_support_domain_min": 2,
        "deployment_estimator_domain_min": 1,
        "single_view_extension_preregistered": True,
        "deployment_estimator": DEPLOYMENT_ESTIMATOR,
        "full_mapping_support_policy": (
            "retain_all_frozen_anchors_with_at_least_one_mapping_view_no_fallback"
        ),
        "single_view_anchor_count": int((view_counts == 1).sum()),
        "single_view_anchor_indices_sha256": tensor_sha256(single_view),
        "single_view_anchor_ids_sha256": tensor_sha256(anchor_ids[single_view]),
        "single_view_anchor_type_histogram": type_histogram,
        "support_view_counts_sha256": tensor_sha256(view_counts),
        "xfeat_query_descriptor_registry_sha256": descriptor_digest.hexdigest(),
        "candidate_anchor_features_sha256": tensor_sha256(actual),
        "candidate_map_formula_bitwise_equal": True,
        "view_balance_independently_replayed": True,
    }


def _strict_map_audit(source: Mapping, candidate: Mapping) -> dict:
    _assert_only_fields_changed(
        source,
        candidate,
        allowed=MAP_DESCRIPTOR_MUTATIONS,
        label="anchor map",
    )
    source_hashes = _registry_hashes(source)
    candidate_hashes = _registry_hashes(candidate)
    if source_hashes != candidate_hashes:
        raise ValueError("candidate changes anchor identity, geometry, or topology")
    source_features = F.normalize(
        torch.as_tensor(source["anchor_features"]).float(), dim=1
    )
    candidate_features = torch.as_tensor(candidate["anchor_features"]).float()
    if source_features.shape[1] != SOURCE_DESCRIPTOR_DIM:
        raise ValueError("source map descriptor dimension is not 256")
    if candidate_features.shape != (
        source_features.shape[0],
        EFFECTIVE_DESCRIPTOR_DIM,
    ):
        raise ValueError("candidate map descriptor shape differs from [N,320]")
    branch = candidate_features[:, :SOURCE_DESCRIPTOR_DIM] * (2.0**0.5)
    if not torch.allclose(branch, source_features, atol=1e-7, rtol=0.0):
        raise ValueError("candidate SuperPoint branch differs from the frozen V3 bank")
    expected_branch_norm = torch.full(
        (candidate_features.shape[0],), (2.0**-0.5)
    )
    for value in (
        candidate_features[:, :SOURCE_DESCRIPTOR_DIM],
        candidate_features[:, SOURCE_DESCRIPTOR_DIM:],
    ):
        if not torch.allclose(
            torch.linalg.norm(value, dim=1),
            expected_branch_norm,
            atol=2e-6,
            rtol=0.0,
        ):
            raise ValueError("candidate map branches do not have equal energy")
    if not torch.allclose(
        torch.linalg.norm(candidate_features, dim=1),
        torch.ones(candidate_features.shape[0]),
        atol=2e-6,
        rtol=0.0,
    ):
        raise ValueError("candidate map descriptor is not unit length")
    return {
        "registry_field_sha256": source_hashes,
        "anchor_count": int(source_features.shape[0]),
        "source_descriptor_dim": SOURCE_DESCRIPTOR_DIM,
        "effective_descriptor_dim": EFFECTIVE_DESCRIPTOR_DIM,
        "equal_branch_norm": True,
        "unit_descriptor_norm": True,
    }


def _strict_teacher_rebind_audit(
    source: Mapping,
    candidate: Mapping,
    *,
    variant_map_path: Path,
    variant_query_cache_path: Path,
) -> dict:
    _assert_only_fields_changed(
        source,
        candidate,
        allowed=TEACHER_REBIND_MUTATIONS,
        label="complete-positive teacher",
    )
    if Path(str(candidate.get("anchor_map", ""))).resolve() != variant_map_path:
        raise ValueError("candidate teacher does not bind the candidate map")
    if Path(str(candidate.get("query_cache", ""))).resolve() != (
        variant_query_cache_path
    ):
        raise ValueError("candidate teacher does not bind the candidate query cache")
    return {
        "allowed_mutations": sorted(TEACHER_REBIND_MUTATIONS),
        "query_registry_bitwise_preserved": True,
        "positive_and_ambiguous_csr_bitwise_preserved": True,
        "config_and_diagnostics_bitwise_preserved": True,
    }


def _strict_calibration_rebind_audit(
    source: Mapping,
    candidate: Mapping,
    *,
    source_query_cache_path: Path,
    variant_query_cache_path: Path,
    variant_query_cache_sha256: str,
) -> dict:
    _assert_only_fields_changed(
        source,
        candidate,
        allowed=CALIBRATION_REBIND_MUTATIONS,
        label="scene calibration",
    )
    source_sources = dict(source.get("sources", {}))
    candidate_sources = dict(candidate.get("sources", {}))
    if Path(str(source_sources.get("query_cache", ""))).resolve() != (
        source_query_cache_path
    ):
        raise ValueError("source calibration does not bind the source query cache")
    for field in sorted(
        (set(source_sources) | set(candidate_sources))
        - {"query_cache", "query_cache_sha256"}
    ):
        if field not in source_sources or field not in candidate_sources:
            raise ValueError(f"calibration immutable source field set differs at {field}")
        if not _equal_value(source_sources[field], candidate_sources[field]):
            raise ValueError(f"calibration immutable source field differs at {field}")
    if Path(str(candidate_sources.get("query_cache", ""))).resolve() != (
        variant_query_cache_path
    ):
        raise ValueError("candidate calibration does not bind the candidate query cache")
    if str(candidate_sources.get("query_cache_sha256", "")).lower() != (
        variant_query_cache_sha256
    ):
        raise ValueError("candidate calibration query-cache SHA-256 differs")
    if (
        candidate.get("schema") != "lafgs_mapping_only_scene_calibration"
        or int(candidate.get("version", 0)) < 2
        or candidate.get("uses_test_queries", False) is not False
        or candidate_sources.get("uses_test_queries") is not False
    ):
        raise ValueError("candidate calibration is not mapping-only")
    return {
        "allowed_top_level_mutations": sorted(CALIBRATION_REBIND_MUTATIONS),
        "allowed_source_mutations": ["query_cache", "query_cache_sha256"],
        "statistics_parameters_policy_bitwise_preserved": True,
        "uses_test_queries_false": True,
    }


def materialize_equal_energy_descriptor_factor(
    *,
    source_map_path: str | Path,
    source_map_sha256: str,
    source_metric_path: str | Path,
    source_metric_sha256: str,
    source_query_cache_path: str | Path,
    source_query_cache_sha256: str,
    refreshed_query_cache_path: str | Path,
    refreshed_query_cache_sha256: str,
    mechanism_report_path: str | Path,
    mechanism_report_sha256: str,
    mechanism_gate_path: str | Path,
    mechanism_gate_sha256: str,
    deployment_extension_path: str | Path,
    deployment_extension_sha256: str,
    teacher_path: str | Path,
    teacher_sha256: str,
    calibration_path: str | Path,
    calibration_sha256: str,
    probe_path: str | Path,
    probe_sha256: str,
    xfeat_weights_path: str | Path,
    xfeat_weights_sha256: str,
    output_dir: str | Path,
    require_clean_producer: bool = True,
) -> dict:
    """Write the 320D map/cache/identity metric and their strict factor contract."""
    producer_identity = descriptor_factor_producer_identity(
        require_clean=require_clean_producer
    )
    locked = {}
    paths = {}
    for name, path, digest in (
        ("source_map", source_map_path, source_map_sha256),
        ("source_metric", source_metric_path, source_metric_sha256),
        ("source_query_cache", source_query_cache_path, source_query_cache_sha256),
        (
            "refreshed_query_cache",
            refreshed_query_cache_path,
            refreshed_query_cache_sha256,
        ),
        ("mechanism_report", mechanism_report_path, mechanism_report_sha256),
        ("mechanism_gate", mechanism_gate_path, mechanism_gate_sha256),
        (
            "deployment_extension",
            deployment_extension_path,
            deployment_extension_sha256,
        ),
        ("teacher", teacher_path, teacher_sha256),
        ("calibration", calibration_path, calibration_sha256),
        ("probe", probe_path, probe_sha256),
        ("xfeat_weights", xfeat_weights_path, xfeat_weights_sha256),
    ):
        paths[name], locked[name] = _locked_file(path, digest, label=name)

    output = Path(output_dir).expanduser().resolve()
    map_output = output / "anchor_map_equal_energy_320d.pt"
    metric_output = output / "metric_state_equal_energy_320d.pt"
    cache_output = output / "query_cache_equal_energy_320d.pt"
    teacher_output = output / "complete_positive_teacher_equal_energy_320d.pt"
    calibration_output = output / "scene_calibration_equal_energy_320d.json"
    equivalence_output = output / "query_cache_equivalence_v2.json"
    contract_output = output / "descriptor_factor_contract.json"
    targets = (
        map_output,
        metric_output,
        cache_output,
        teacher_output,
        calibration_output,
        equivalence_output,
        contract_output,
    )
    if any(path.exists() for path in targets):
        raise ValueError("refusing to overwrite an equal-energy factor artifact")
    output.mkdir(parents=True, exist_ok=True)

    source_map = torch.load(paths["source_map"], map_location="cpu", weights_only=False)
    source_metric_state = torch.load(
        paths["source_metric"], map_location="cpu", weights_only=False
    )
    source_cache = _load_mmap(paths["source_query_cache"])
    refreshed_cache = _load_mmap(paths["refreshed_query_cache"])
    teacher = torch.load(paths["teacher"], map_location="cpu", weights_only=False)
    calibration = json.loads(paths["calibration"].read_text(encoding="utf-8"))
    probe = _load_mmap(paths["probe"])
    mechanism_report = json.loads(paths["mechanism_report"].read_text())
    mechanism_gate = json.loads(paths["mechanism_gate"].read_text())
    deployment_extension = json.loads(paths["deployment_extension"].read_text())

    if teacher.get("uses_test_queries", False) is not False:
        raise ValueError("teacher must be mapping-only")
    if Path(str(teacher.get("query_cache", ""))).resolve() != paths[
        "source_query_cache"
    ]:
        raise ValueError("teacher does not bind the frozen source query cache")
    if Path(str(teacher.get("anchor_map", ""))).resolve() != paths["source_map"]:
        raise ValueError("teacher does not bind the frozen source map")
    calibration_sources = dict(calibration.get("sources", {}))
    if (
        calibration.get("schema") != "lafgs_mapping_only_scene_calibration"
        or int(calibration.get("version", 0)) < 2
        or calibration.get("uses_test_queries", False) is not False
        or calibration_sources.get("uses_test_queries") is not False
        or Path(str(calibration_sources.get("query_cache", ""))).resolve()
        != paths["source_query_cache"]
    ):
        raise ValueError("source calibration is not the frozen mapping-only contract")
    if int(teacher.get("anchor_count", -1)) != int(
        torch.as_tensor(source_map["anchor_ids"]).numel()
    ):
        raise ValueError("teacher and source map anchor counts differ")
    if not torch.equal(
        torch.as_tensor(source_metric_state["landmark_indices"]).long().reshape(-1),
        torch.as_tensor(source_map["anchor_ids"]).long().reshape(-1),
    ):
        raise ValueError("source metric and map anchor IDs differ")
    if Path(str(source_metric_state.get("map_path", ""))).resolve() != paths[
        "source_map"
    ]:
        raise ValueError("source metric does not bind the source map")
    if int(source_metric_state["metric_config"]["descriptor_dim"]) != (
        SOURCE_DESCRIPTOR_DIM
    ):
        raise ValueError("source shared metric is not 256D")

    observed_equivalence = audit_sparse_refresh_equivalence(
        source_cache, refreshed_cache
    )
    if observed_equivalence.get("content_equivalent_track_payload_reuse_authorized") is not True:
        raise ValueError("freshly recomputed source/refreshed cache equivalence failed")
    equivalence_v2 = {
        "schema": "lafgs_mapping_sparse_refresh_equivalence",
        "version": 2,
        "valid": True,
        "mapping_only": True,
        "uses_test_queries": False,
        "sources": {
            "source_cache": locked["source_query_cache"],
            "refreshed_cache": locked["refreshed_query_cache"],
        },
        "checks": {
            "source_cache_sha256_locked": True,
            "refreshed_cache_sha256_locked": True,
            "query_order_exact": observed_equivalence["query_order_exact"] is True,
            "content_equivalent_track_payload_reuse_authorized": (
                observed_equivalence[
                    "content_equivalent_track_payload_reuse_authorized"
                ]
                is True
            ),
        },
        "audit": observed_equivalence,
    }
    _validate_equivalence_report(
        equivalence_v2,
        source_record=locked["source_query_cache"],
        refreshed_record=locked["refreshed_query_cache"],
        observed=observed_equivalence,
    )
    equivalence_output.write_text(
        json.dumps(equivalence_v2, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    probe_audit = validate_probe(
        probe,
        refreshed_cache,
        teacher,
        require_descriptor=True,
        verify_weight_artifact=True,
        query_cache_path=paths["refreshed_query_cache"],
        teacher_path=paths["teacher"],
    )
    if int(probe_audit["reference_descriptor_dim"]) != SOURCE_DESCRIPTOR_DIM:
        raise ValueError("probe reference descriptor dimension is not 256")
    if int(probe_audit["candidate_descriptor_dim"]) != XFEAT_DESCRIPTOR_DIM:
        raise ValueError("probe XFeat descriptor dimension is not 64")
    probe_weight = probe_audit["artifact"]
    if (
        Path(str(probe_weight.get("path", ""))).resolve() != paths["xfeat_weights"]
        or str(probe_weight.get("sha256", "")).lower()
        != locked["xfeat_weights"]["sha256"]
    ):
        raise ValueError("probe binds different XFeat weights")
    mechanism_audit = _validate_mechanism_go(
        report=mechanism_report,
        gate=mechanism_gate,
        report_record=locked["mechanism_report"],
        source_map_record=locked["source_map"],
        refreshed_query_cache_record=locked["refreshed_query_cache"],
        teacher_record=locked["teacher"],
        probe_record=locked["probe"],
        xfeat_weights_record=locked["xfeat_weights"],
    )
    del refreshed_cache
    gc.collect()

    xfeat_bank, view_counts, support_audit = _build_xfeat_anchor_bank(
        teacher=teacher,
        probe=probe,
        anchor_ids=source_map["anchor_ids"],
        anchor_type=source_map["anchor_type"],
    )
    deployment_extension_audit = _validate_deployment_extension_preregistration(
        preregistration=deployment_extension,
        preregistration_record=locked["deployment_extension"],
        source_map_record=locked["source_map"],
        source_metric_record=locked["source_metric"],
        teacher_record=locked["teacher"],
        mechanism_report_record=locked["mechanism_report"],
        mechanism_gate_record=locked["mechanism_gate"],
        support_audit=support_audit,
    )
    factor_basis = {
        "formula": "concat(l2(v3_metric(superpoint256)),l2(xfeat64))/sqrt(2)",
        "source_descriptor_dim": SOURCE_DESCRIPTOR_DIM,
        "xfeat_descriptor_dim": XFEAT_DESCRIPTOR_DIM,
        "effective_descriptor_dim": EFFECTIVE_DESCRIPTOR_DIM,
        "sources": {name: record["sha256"] for name, record in locked.items()},
    }
    factor_id = _canonical_json_sha256(factor_basis)

    source_bank = F.normalize(
        torch.as_tensor(source_map["anchor_features"]).float(), dim=1
    )
    candidate_bank = torch.cat((source_bank, xfeat_bank), dim=1) / (2.0**0.5)
    support_audit["candidate_anchor_features_sha256"] = tensor_sha256(candidate_bank)
    candidate_map = dict(source_map)
    candidate_map["anchor_features"] = candidate_bank.cpu().float()
    candidate_map["v7_metric_raw_features"] = candidate_bank.cpu().float()
    candidate_map.pop("v7_anchor_residual_parameter", None)
    candidate_map.pop("v7_anchor_residual", None)
    candidate_map["v7_online_metric"] = {
        "schema": FACTOR_SCHEMA,
        "version": FACTOR_VERSION,
        "factor_id": factor_id,
        "protocol": "strict_320d_identity_after_fixed_equal_energy_composition",
    }
    candidate_map["descriptor_factor"] = {
        "schema": FACTOR_SCHEMA,
        "version": FACTOR_VERSION,
        "factor_id": factor_id,
        "mapping_only": True,
        "uses_test_queries": False,
        "support_view_counts": view_counts,
    }
    map_audit = _strict_map_audit(source_map, candidate_map)

    identity_metric = SharedLowRankMetric(
        descriptor_dim=EFFECTIVE_DESCRIPTOR_DIM,
        rank=1,
        max_residual_norm=0.0,
    )
    with torch.no_grad():
        for parameter in identity_metric.parameters():
            parameter.zero_()
    candidate_metric = {
        "schema": "lafgs_shared_metric_state",
        "version": 1,
        "landmark_indices": torch.as_tensor(source_map["anchor_ids"]).long().clone(),
        "metric_config": identity_metric.export_config(),
        "metric_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in identity_metric.state_dict().items()
        },
        "map_path": str(map_output),
        "step": int(source_metric_state.get("step", -1)),
        "descriptor_factor": {
            "schema": FACTOR_SCHEMA,
            "version": FACTOR_VERSION,
            "factor_id": factor_id,
            "strict_identity": True,
        },
    }
    if not _metric_is_strict_identity(
        candidate_metric, anchor_ids=source_map["anchor_ids"]
    ):
        raise AssertionError("materialized 320D metric is not strict identity")

    source_metric = load_shared_metric(
        paths["source_metric"],
        anchor_ids=torch.as_tensor(source_map["anchor_ids"]).long(),
        device=torch.device("cpu"),
    )
    candidate_cache, query_audit = _candidate_query_cache(
        source_cache=source_cache,
        teacher=teacher,
        probe=probe,
        source_metric=source_metric,
        factor_id=factor_id,
    )
    producer_query_replay = _audit_candidate_query_cache_formula(
        source_cache=source_cache,
        candidate_cache=candidate_cache,
        teacher=teacher,
        probe=probe,
        source_metric=source_metric,
        factor_id=factor_id,
    )
    _assert_audit_fields_equal(
        producer_query_replay,
        query_audit,
        QUERY_AUDIT_FIELDS,
        label="producer independent query replay",
    )
    producer_map_replay = _audit_candidate_map_formula(
        source_map=source_map,
        candidate_map=candidate_map,
        teacher=teacher,
        probe=probe,
    )
    _assert_audit_fields_equal(
        producer_map_replay,
        support_audit,
        SUPPORT_AUDIT_FIELDS,
        label="producer independent map replay",
    )

    torch.save(candidate_map, map_output)
    torch.save(candidate_metric, metric_output)
    torch.save(candidate_cache, cache_output)
    candidate_cache_sha256 = sha256_file(cache_output)

    candidate_teacher = dict(teacher)
    candidate_teacher["anchor_map"] = str(map_output)
    candidate_teacher["query_cache"] = str(cache_output)
    teacher_audit = _strict_teacher_rebind_audit(
        teacher,
        candidate_teacher,
        variant_map_path=map_output,
        variant_query_cache_path=cache_output,
    )
    torch.save(candidate_teacher, teacher_output)

    candidate_calibration = dict(calibration)
    candidate_calibration["sources"] = {
        **calibration_sources,
        "query_cache": str(cache_output),
        "query_cache_sha256": candidate_cache_sha256,
        "uses_test_queries": False,
    }
    calibration_audit = _strict_calibration_rebind_audit(
        calibration,
        candidate_calibration,
        source_query_cache_path=paths["source_query_cache"],
        variant_query_cache_path=cache_output,
        variant_query_cache_sha256=candidate_cache_sha256,
    )
    calibration_output.write_text(
        json.dumps(candidate_calibration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs = {
        "map": {"path": str(map_output), "sha256": sha256_file(map_output)},
        "metric": {
            "path": str(metric_output),
            "sha256": sha256_file(metric_output),
        },
        "descriptor_cache": {
            "path": str(cache_output),
            "sha256": candidate_cache_sha256,
        },
        "teacher": {
            "path": str(teacher_output),
            "sha256": sha256_file(teacher_output),
        },
        "calibration": {
            "path": str(calibration_output),
            "sha256": sha256_file(calibration_output),
        },
        "equivalence": {
            "path": str(equivalence_output),
            "sha256": sha256_file(equivalence_output),
        },
    }
    checks = {
        "mapping_only": True,
        "uses_test_queries_false": True,
        "source_refreshed_cache_equivalent": True,
        "probe_row_registry_valid": True,
        "equal_energy_mechanism_go_bound": True,
        "deployment_extension_preregistered_before_pose": True,
        "producer_independent_formula_replay_passed": True,
        "all_anchors_supported_without_fallback": (
            support_audit["unsupported_anchor_count"] == 0
        ),
        "anchor_registry_bitwise_preserved": True,
        "geometry_and_topology_bitwise_preserved": True,
        "query_nondescriptor_fields_bitwise_preserved": True,
        "teacher_only_rebound_to_candidate_artifacts": True,
        "calibration_numbers_and_policy_bitwise_preserved": True,
        "equal_energy_branch_norms": True,
        "strict_320d_identity_metric": True,
        "one_materialized_bank": True,
        "one_global_top1": True,
        "one_poselib_call_per_query": True,
    }
    if not all(checks.values()):
        raise AssertionError("equal-energy factor audit did not pass")
    contract = {
        "schema": FACTOR_SCHEMA,
        "version": FACTOR_VERSION,
        "valid": True,
        "mapping_only": True,
        "uses_test_queries": False,
        "factor_id": factor_id,
        "single_factor": "descriptor_representation_only",
        "producer_identity": producer_identity,
        "formula": factor_basis["formula"],
        "score_identity": (
            "dot(zq,zm)=0.5*cos(v3_metric_superpoint)+0.5*cos(xfeat)"
        ),
        "dimensions": {
            "source_superpoint": SOURCE_DESCRIPTOR_DIM,
            "source_xfeat": XFEAT_DESCRIPTOR_DIM,
            "effective": EFFECTIVE_DESCRIPTOR_DIM,
        },
        "sources": locked,
        "outputs": outputs,
        "allowed_mutations": {
            "map_top_level": sorted(MAP_DESCRIPTOR_MUTATIONS),
            "query_cache_top_level": sorted(CACHE_TOP_LEVEL_MUTATIONS),
            "query_record": sorted(CACHE_QUERY_MUTATIONS),
            "teacher_top_level": sorted(TEACHER_REBIND_MUTATIONS),
            "calibration_top_level": sorted(CALIBRATION_REBIND_MUTATIONS),
            "calibration_sources": ["query_cache", "query_cache_sha256"],
        },
        "map_audit": map_audit,
        "teacher_audit": teacher_audit,
        "calibration_audit": calibration_audit,
        "query_audit": query_audit,
        "support_audit": support_audit,
        "equivalence_audit": observed_equivalence,
        "probe_audit": {
            key: probe_audit[key]
            for key in (
                "query_count",
                "requested_keypoint_count",
                "reference_descriptor_dim",
                "candidate_descriptor_dim",
                "validated_descriptor_rows",
            )
        },
        "mechanism_audit": mechanism_audit,
        "deployment_extension_audit": deployment_extension_audit,
        "producer_independent_replay": {
            "query": producer_query_replay,
            "map": producer_map_replay,
        },
        "limitations": [
            (
                "The mechanism GO estimates identity with cross-fit minimum "
                "support_views=2; full deployment retains every frozen anchor "
                "with minimum support_views=1 to preserve topology."
            ),
            (
                f"{support_audit['single_view_anchor_count']} single-view anchors "
                "are outside the two-fold identity mechanism estimate and are "
                "adjudicated only by the mapping pose/tail gate."
            ),
            (
                "The mechanism GO uses a raw normalized SuperPoint cross-fit "
                "proxy, while deployment intentionally preserves the frozen V3 "
                "metric and anchor-feature branch; this transfer requires the "
                "q256x3 pose-tail and office2_5b cross-domain guards."
            ),
        ],
        "metric_audit": {
            "schema": "lafgs_shared_metric_state",
            "descriptor_dim": EFFECTIVE_DESCRIPTOR_DIM,
            "rank": 1,
            "max_residual_norm": 0.0,
            "all_parameters_zero": True,
            "strict_identity": True,
        },
        "checks": checks,
    }
    contract_output.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "contract": contract,
        "contract_path": str(contract_output),
        "contract_sha256": sha256_file(contract_output),
    }


def validate_descriptor_factor_contract(
    contract_path: str | Path,
    *,
    source_map_path: str | Path | None = None,
    source_metric_path: str | Path | None = None,
    source_query_cache_path: str | Path | None = None,
    teacher_path: str | Path | None = None,
    calibration_path: str | Path | None = None,
    variant_map_path: str | Path | None = None,
    variant_metric_path: str | Path | None = None,
    variant_query_cache_path: str | Path | None = None,
    variant_teacher_path: str | Path | None = None,
    variant_calibration_path: str | Path | None = None,
) -> dict:
    """Fail closed on a materialized descriptor-factor contract and its files."""
    path = Path(contract_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"descriptor-factor contract is not a file: {path}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        contract.get("schema") != FACTOR_SCHEMA
        or contract.get("version") != FACTOR_VERSION
        or contract.get("valid") is not True
        or contract.get("mapping_only") is not True
        or contract.get("uses_test_queries") is not False
        or contract.get("single_factor") != "descriptor_representation_only"
    ):
        raise ValueError("invalid equal-energy descriptor-factor contract")
    if contract.get("formula") != (
        "concat(l2(v3_metric(superpoint256)),l2(xfeat64))/sqrt(2)"
    ):
        raise ValueError("descriptor-factor formula differs")
    if contract.get("score_identity") != (
        "dot(zq,zm)=0.5*cos(v3_metric_superpoint)+0.5*cos(xfeat)"
    ):
        raise ValueError("descriptor-factor score identity differs")
    producer_identity = contract.get("producer_identity", {})
    if (
        producer_identity.get("schema")
        != "lafgs_equal_energy_descriptor_factor_producer_code"
        or producer_identity.get("version") != 1
        or producer_identity.get("git_worktree_clean") is not True
        or len(str(producer_identity.get("git_commit", ""))) != 40
    ):
        raise ValueError("descriptor-factor producer identity is invalid")
    current_identity = descriptor_factor_producer_identity(require_clean=True)
    if producer_identity.get("git_commit") != current_identity.get("git_commit"):
        raise ValueError("descriptor-factor producer Git commit differs")
    if producer_identity.get("entrypoints") != current_identity.get("entrypoints"):
        raise ValueError("descriptor-factor producer entrypoint identity differs")
    if contract.get("dimensions") != {
        "source_superpoint": SOURCE_DESCRIPTOR_DIM,
        "source_xfeat": XFEAT_DESCRIPTOR_DIM,
        "effective": EFFECTIVE_DESCRIPTOR_DIM,
    }:
        raise ValueError("descriptor-factor dimensions differ")
    checks = contract.get("checks")
    if not isinstance(checks, Mapping) or not checks or not all(checks.values()):
        raise ValueError("descriptor-factor audit checks did not all pass")
    metric_audit = contract.get("metric_audit", {})
    if metric_audit != {
        "schema": "lafgs_shared_metric_state",
        "descriptor_dim": EFFECTIVE_DESCRIPTOR_DIM,
        "rank": 1,
        "max_residual_norm": 0.0,
        "all_parameters_zero": True,
        "strict_identity": True,
    }:
        raise ValueError("descriptor-factor identity-metric audit differs")

    sources = contract.get("sources", {})
    outputs = contract.get("outputs", {})
    required_sources = {
        "source_map",
        "source_metric",
        "source_query_cache",
        "refreshed_query_cache",
        "mechanism_report",
        "mechanism_gate",
        "deployment_extension",
        "teacher",
        "calibration",
        "probe",
        "xfeat_weights",
    }
    if set(sources) != required_sources or set(outputs) != {
        "map",
        "metric",
        "descriptor_cache",
        "teacher",
        "calibration",
        "equivalence",
    }:
        raise ValueError("descriptor-factor artifact bindings are incomplete")
    factor_basis = {
        "formula": contract["formula"],
        "source_descriptor_dim": SOURCE_DESCRIPTOR_DIM,
        "xfeat_descriptor_dim": XFEAT_DESCRIPTOR_DIM,
        "effective_descriptor_dim": EFFECTIVE_DESCRIPTOR_DIM,
        "sources": {
            name: str(record.get("sha256", "")).lower()
            for name, record in sources.items()
        },
    }
    if contract.get("factor_id") != _canonical_json_sha256(factor_basis):
        raise ValueError("descriptor-factor ID does not match its frozen source basis")
    resolved = {}
    for group_name, group in (("sources", sources), ("outputs", outputs)):
        for name, record in group.items():
            if not isinstance(record, Mapping):
                raise ValueError(f"descriptor-factor {group_name}.{name} is invalid")
            artifact = Path(str(record.get("path", ""))).expanduser().resolve()
            if not artifact.is_file():
                raise ValueError(
                    f"descriptor-factor {group_name}.{name} is not a file: {artifact}"
                )
            expected = _require_sha256(
                str(record.get("sha256", "")),
                label=f"descriptor-factor {group_name}.{name} SHA-256",
            )
            actual = sha256_file(artifact)
            if actual != expected:
                raise ValueError(
                    f"descriptor-factor {group_name}.{name} SHA-256 differs"
                )
            resolved[f"{group_name}.{name}"] = artifact

    mechanism_audit = _validate_mechanism_go(
        report=json.loads(
            resolved["sources.mechanism_report"].read_text(encoding="utf-8")
        ),
        gate=json.loads(
            resolved["sources.mechanism_gate"].read_text(encoding="utf-8")
        ),
        report_record=sources["mechanism_report"],
        source_map_record=sources["source_map"],
        refreshed_query_cache_record=sources["refreshed_query_cache"],
        teacher_record=sources["teacher"],
        probe_record=sources["probe"],
        xfeat_weights_record=sources["xfeat_weights"],
    )
    if mechanism_audit != contract.get("mechanism_audit"):
        raise ValueError("live mechanism GO audit differs from descriptor contract")
    deployment_extension_audit = _validate_deployment_extension_preregistration(
        preregistration=json.loads(
            resolved["sources.deployment_extension"].read_text(encoding="utf-8")
        ),
        preregistration_record=sources["deployment_extension"],
        source_map_record=sources["source_map"],
        source_metric_record=sources["source_metric"],
        teacher_record=sources["teacher"],
        mechanism_report_record=sources["mechanism_report"],
        mechanism_gate_record=sources["mechanism_gate"],
        support_audit=contract.get("support_audit", {}),
    )
    if deployment_extension_audit != contract.get("deployment_extension_audit"):
        raise ValueError(
            "live deployment-extension audit differs from descriptor contract"
        )
    producer_replay = contract.get("producer_independent_replay", {})
    if (
        not isinstance(producer_replay, Mapping)
        or producer_replay.get("query", {}).get(
            "all_candidate_descriptors_formula_bitwise_equal"
        )
        is not True
        or producer_replay.get("query", {}).get(
            "all_nondescriptor_query_fields_bitwise_equal"
        )
        is not True
        or producer_replay.get("map", {}).get(
            "candidate_map_formula_bitwise_equal"
        )
        is not True
        or producer_replay.get("map", {}).get(
            "view_balance_independently_replayed"
        )
        is not True
    ):
        raise ValueError("descriptor-factor producer replay contract is invalid")
    equivalence = json.loads(
        resolved["outputs.equivalence"].read_text(encoding="utf-8")
    )
    _validate_equivalence_report(
        equivalence,
        source_record=sources["source_query_cache"],
        refreshed_record=sources["refreshed_query_cache"],
        observed=contract.get("equivalence_audit", {}),
    )

    explicit = {
        "sources.source_map": source_map_path,
        "sources.source_metric": source_metric_path,
        "sources.source_query_cache": source_query_cache_path,
        "sources.teacher": teacher_path,
        "sources.calibration": calibration_path,
        "outputs.map": variant_map_path,
        "outputs.metric": variant_metric_path,
        "outputs.descriptor_cache": variant_query_cache_path,
        "outputs.teacher": variant_teacher_path,
        "outputs.calibration": variant_calibration_path,
    }
    for name, value in explicit.items():
        if value is not None and Path(value).expanduser().resolve() != resolved[name]:
            raise ValueError(f"descriptor-factor contract binds a different {name}")
    return {
        "path": path,
        "sha256": sha256_file(path),
        "contract": contract,
        "descriptor_cache_path": resolved["outputs.descriptor_cache"],
        "descriptor_cache_sha256": outputs["descriptor_cache"]["sha256"],
        "teacher_path": resolved["outputs.teacher"],
        "calibration_path": resolved["outputs.calibration"],
        "factor_id": contract["factor_id"],
        "producer_git_commit": producer_identity["git_commit"],
    }


def audit_descriptor_factor_pair(
    contract_path: str | Path,
    *,
    source_map_path: str | Path,
    source_metric_path: str | Path,
    source_query_cache_path: str | Path,
    teacher_path: str | Path,
    calibration_path: str | Path,
    variant_map_path: str | Path,
    variant_metric_path: str | Path,
    variant_query_cache_path: str | Path,
    variant_teacher_path: str | Path,
    variant_calibration_path: str | Path,
) -> dict:
    """Independently replay every query/map descriptor and immutable factor field."""
    validated = validate_descriptor_factor_contract(
        contract_path,
        source_map_path=source_map_path,
        source_metric_path=source_metric_path,
        source_query_cache_path=source_query_cache_path,
        teacher_path=teacher_path,
        calibration_path=calibration_path,
        variant_map_path=variant_map_path,
        variant_metric_path=variant_metric_path,
        variant_query_cache_path=variant_query_cache_path,
        variant_teacher_path=variant_teacher_path,
        variant_calibration_path=variant_calibration_path,
    )
    source_map = torch.load(source_map_path, map_location="cpu", weights_only=False)
    variant_map = torch.load(variant_map_path, map_location="cpu", weights_only=False)
    source_metric = torch.load(
        source_metric_path, map_location="cpu", weights_only=False
    )
    variant_metric = torch.load(
        variant_metric_path, map_location="cpu", weights_only=False
    )
    source_teacher = torch.load(teacher_path, map_location="cpu", weights_only=False)
    variant_teacher = torch.load(
        variant_teacher_path, map_location="cpu", weights_only=False
    )
    source_cache = _load_mmap(source_query_cache_path)
    variant_cache = _load_mmap(variant_query_cache_path)
    refreshed_cache_path = Path(
        validated["contract"]["sources"]["refreshed_query_cache"]["path"]
    ).expanduser().resolve()
    refreshed_cache = _load_mmap(refreshed_cache_path)
    live_equivalence = audit_sparse_refresh_equivalence(
        source_cache, refreshed_cache
    )
    if live_equivalence != validated["contract"].get("equivalence_audit"):
        raise ValueError("live source/refreshed cache equivalence differs from contract")
    equivalence_path = Path(
        validated["contract"]["outputs"]["equivalence"]["path"]
    ).expanduser().resolve()
    _validate_equivalence_report(
        json.loads(equivalence_path.read_text(encoding="utf-8")),
        source_record=validated["contract"]["sources"]["source_query_cache"],
        refreshed_record=validated["contract"]["sources"][
            "refreshed_query_cache"
        ],
        observed=live_equivalence,
    )
    probe_path = Path(
        validated["contract"]["sources"]["probe"]["path"]
    ).expanduser().resolve()
    probe = _load_mmap(probe_path)
    source_calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    variant_calibration = json.loads(
        Path(variant_calibration_path).read_text(encoding="utf-8")
    )
    if int(source_metric.get("metric_config", {}).get("descriptor_dim", -1)) != (
        SOURCE_DESCRIPTOR_DIM
    ):
        raise ValueError("descriptor-factor source metric is not 256D")
    if Path(str(source_metric.get("map_path", ""))).resolve() != Path(
        source_map_path
    ).resolve():
        raise ValueError("descriptor-factor source metric binds another map")
    if Path(str(variant_metric.get("map_path", ""))).resolve() != Path(
        variant_map_path
    ).resolve():
        raise ValueError("descriptor-factor identity metric binds another map")
    map_audit = _strict_map_audit(source_map, variant_map)
    if map_audit != validated["contract"].get("map_audit"):
        raise ValueError("live descriptor-factor map audit differs from its contract")
    if not _metric_is_strict_identity(
        variant_metric, anchor_ids=variant_map["anchor_ids"]
    ):
        raise ValueError("descriptor-factor variant metric is not strict 320D identity")
    source_metric_module = load_shared_metric(
        source_metric_path,
        anchor_ids=torch.as_tensor(source_map["anchor_ids"]).long(),
        device=torch.device("cpu"),
    )
    query_audit = _audit_candidate_query_cache_formula(
        source_cache=source_cache,
        candidate_cache=variant_cache,
        teacher=source_teacher,
        probe=probe,
        source_metric=source_metric_module,
        factor_id=validated["factor_id"],
    )
    contract_query_audit = dict(validated["contract"].get("query_audit", {}))
    _assert_audit_fields_equal(
        query_audit,
        contract_query_audit,
        QUERY_AUDIT_FIELDS,
        label="live descriptor-factor query audit",
    )
    if query_audit != validated["contract"]["producer_independent_replay"]["query"]:
        raise ValueError("live query replay differs from producer independent replay")
    map_formula_audit = _audit_candidate_map_formula(
        source_map=source_map,
        candidate_map=variant_map,
        teacher=source_teacher,
        probe=probe,
    )
    contract_support_audit = dict(validated["contract"].get("support_audit", {}))
    _assert_audit_fields_equal(
        map_formula_audit,
        contract_support_audit,
        SUPPORT_AUDIT_FIELDS,
        label="live descriptor-factor support audit",
    )
    if map_formula_audit != validated["contract"]["producer_independent_replay"]["map"]:
        raise ValueError("live map replay differs from producer independent replay")
    teacher_audit = _strict_teacher_rebind_audit(
        source_teacher,
        variant_teacher,
        variant_map_path=Path(variant_map_path).resolve(),
        variant_query_cache_path=Path(variant_query_cache_path).resolve(),
    )
    variant_query_cache_sha256 = validated["descriptor_cache_sha256"]
    calibration_audit = _strict_calibration_rebind_audit(
        source_calibration,
        variant_calibration,
        source_query_cache_path=Path(source_query_cache_path).resolve(),
        variant_query_cache_path=Path(variant_query_cache_path).resolve(),
        variant_query_cache_sha256=variant_query_cache_sha256,
    )
    if teacher_audit != validated["contract"].get("teacher_audit"):
        raise ValueError("live descriptor-factor teacher audit differs from contract")
    if calibration_audit != validated["contract"].get("calibration_audit"):
        raise ValueError("live descriptor-factor calibration audit differs from contract")
    if validated["contract"].get("allowed_mutations") != {
        "map_top_level": sorted(MAP_DESCRIPTOR_MUTATIONS),
        "query_cache_top_level": sorted(CACHE_TOP_LEVEL_MUTATIONS),
        "query_record": sorted(CACHE_QUERY_MUTATIONS),
        "teacher_top_level": sorted(TEACHER_REBIND_MUTATIONS),
        "calibration_top_level": sorted(CALIBRATION_REBIND_MUTATIONS),
        "calibration_sources": ["query_cache", "query_cache_sha256"],
    }:
        raise ValueError("descriptor-factor allowed-mutation contract differs")
    return {
        **validated,
        "map_audit": map_audit,
        "query_formula_audit": query_audit,
        "map_formula_audit": map_formula_audit,
        "source_metric_descriptor_dim": SOURCE_DESCRIPTOR_DIM,
        "variant_metric_descriptor_dim": EFFECTIVE_DESCRIPTOR_DIM,
        "strict_identity_metric": True,
        "anchor_registry_bitwise_equal": True,
        "teacher_rebind_only": True,
        "calibration_rebind_only": True,
    }
