"""Fail-closed correspondence certification for V21 test adaptation.

Gaussian depth is used only to enumerate geometrically plausible full-map
Anchors.  Identity is certified by the frozen mapping Track observations via
the independently calibrated V19 teacher.  A diagnostic assignment becomes
an actionable positive only when the selected teacher tier explicitly
authorizes the requested map/metric mutation.  Planner-only authorization
never exposes an action CSR.  Ambiguous and unlabelled rows are never
converted into negatives.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import os
from pathlib import Path
import re
import uuid

import torch

from common.hashing import canonical_json
from map_learning.v18_provenance_truth import (
    TRUTH_AMBIGUOUS,
    TRUTH_EQUIVALENT,
    TRUTH_INVALID,
    TRUTH_NONE,
    TRUTH_UNIQUE,
)
from map_learning.v21_gaussian_support import validate_support_payload
from map_learning.v21_test_cache import (
    tensor_sha256,
    validate_cache_payload,
    validate_shard_registry,
)


SCHEMA = "lafgs_v21_adaptation_correspondence_truth"
VERSION = 1
ROLE = "adaptation"
STATUS_NO_TRUTH = 0
STATUS_UNIQUE = 1
STATUS_EQUIVALENT = 2
STATUS_AMBIGUOUS = 3
STATUS_NAMES = ("NO_TRUTH", "UNIQUE", "EQUIVALENT", "AMBIGUOUS")
ACTIONABLE_STATUSES = frozenset({STATUS_UNIQUE, STATUS_EQUIVALENT})
MUTATING_ACTIONS = frozenset(
    {"destructive_map_control", "strong_metric_control"}
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

SEMANTICS = {
    "candidate_generation": "full_map_gt_pose_gaussian_depth_projection",
    "identity_evidence": "frozen_mapping_track_multiview_geometry_and_native_descriptor_consensus",
    "candidate_source_uses_descriptor_retrieval": False,
    "deployed_metric_used_for_identity": False,
    "ground_truth_pose_authority": "delayed_adaptation_feedback_only",
    "gaussian_support_identity_claim": False,
    "feedback_enters_mapping_track_registry": False,
    "negative_labels_created": False,
    "ambiguous_or_unlabelled_are_negative": False,
    "action_rule": "teacher_authorized_and_status_in_unique_or_equivalent",
}


def _require_sha256(value: object, *, label: str) -> str:
    digest = str(value)
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    return digest


def _source_record(value: object, *, label: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"V21 correspondence {label} source is missing")
    path = str(value.get("path", ""))
    digest = _require_sha256(value.get("sha256"), label=f"{label} SHA256")
    size = int(value.get("size_bytes", 0))
    if not path or size <= 0:
        raise ValueError(f"V21 correspondence {label} source is empty")
    return {"path": path, "sha256": digest, "size_bytes": size}


def sha256_json(value: Mapping) -> str:
    return __import__("hashlib").sha256(
        canonical_json(value).encode("ascii")
    ).hexdigest()


def resolve_teacher_action(
    teacher_validation: Mapping,
    *,
    tier_name: str,
    requested_action: str,
) -> dict:
    """Resolve an action without treating a calibrated tier as authorized.

    The V19 recalibration artifact separates permitted from authorized actions.
    Both lists must contain the exact requested action.  Missing safety fields
    make the decision fail closed.
    """

    if not (
        teacher_validation.get("schema")
        == "lafgs_v19_track_extension_teacher_validation"
        and int(teacher_validation.get("version", 0)) >= 2
        and teacher_validation.get("uses_test_queries") is False
        and teacher_validation.get("loo_used") is False
        and teacher_validation.get("feedback_enters_track_registry") is False
        and teacher_validation.get("reference_source")
        == "mapping_observation_track_membership"
        and teacher_validation.get("reference_available_for_novel_query") is False
        and teacher_validation.get("selection_uses_validation") is False
        and teacher_validation.get("authorization_uses_wilson_lower_bound") is True
        and teacher_validation.get(
            "authorization_requires_independent_mapping_families"
        )
        is True
    ):
        raise ValueError("V21 correspondence requires the fail-closed V19 teacher")
    tiers = teacher_validation.get("selected_tiers")
    if not isinstance(tiers, Mapping) or tier_name not in tiers:
        raise ValueError("requested V19 teacher tier is absent")
    tier = tiers[tier_name]
    if not isinstance(tier, Mapping) or not isinstance(
        tier.get("thresholds"), Mapping
    ):
        raise ValueError("requested V19 teacher tier is malformed")
    permitted = tuple(str(value) for value in tier.get("permitted_actions_if_authorized", ()))
    authorized = tuple(str(value) for value in tier.get("authorized_actions", ()))
    if requested_action not in permitted:
        raise ValueError("requested action is outside the selected tier's scope")
    teacher_authorized = requested_action in authorized
    planner_diagnostic_authorized = teacher_authorized and requested_action in {
        "soft_diagnostic",
        "planner_priority",
    }
    action_authorized = teacher_authorized and requested_action in MUTATING_ACTIONS
    if action_authorized:
        block_reason = None
    elif planner_diagnostic_authorized:
        block_reason = "planner_diagnostic_is_not_map_or_metric_action"
    else:
        block_reason = "requested_action_not_authorized_by_teacher_validation"
    return {
        "tier_name": tier_name,
        "requested_action": requested_action,
        "permitted_actions_if_authorized": permitted,
        "authorized_actions": authorized,
        "teacher_authorized": teacher_authorized,
        "planner_diagnostic_authorized": planner_diagnostic_authorized,
        "action_authorized": action_authorized,
        "action_block_reason": block_reason,
        "thresholds": dict(tier["thresholds"]),
        "calibration": dict(tier.get("calibration", {})),
        "validation": dict(tier.get("validation", {})),
        "authorization_basis": (
            "exact_requested_action_present_in_teacher_authorized_actions"
            if action_authorized
            else block_reason
        ),
    }


def gaussian_row_validity(
    support_record: Mapping,
    *,
    minimum_alpha: float,
    maximum_relative_depth_spread: float,
    minimum_local_valid_fraction: float,
) -> torch.Tensor:
    """Return a conservative geometry gate; rejected rows remain unlabelled."""

    if not (
        0.0 <= float(minimum_alpha) <= 1.0
        and float(maximum_relative_depth_spread) >= 0.0
        and 0.0 <= float(minimum_local_valid_fraction) <= 1.0
    ):
        raise ValueError("V21 Gaussian correspondence gates are invalid")
    base = torch.as_tensor(support_record["gaussian_support_valid"]).bool().cpu()
    alpha = torch.as_tensor(
        support_record["gaussian_alpha_at_keypoints"]
    ).float().cpu()
    spread = torch.as_tensor(
        support_record["gaussian_relative_depth_spread_3x3"]
    ).float().cpu()
    fraction = torch.as_tensor(
        support_record["gaussian_local_valid_fraction_3x3"]
    ).float().cpu()
    if not (base.shape == alpha.shape == spread.shape == fraction.shape):
        raise ValueError("V21 Gaussian support columns do not align")
    return (
        base
        & torch.isfinite(alpha)
        & (alpha >= float(minimum_alpha))
        & torch.isfinite(spread)
        & (spread <= float(maximum_relative_depth_spread))
        & torch.isfinite(fraction)
        & (fraction >= float(minimum_local_valid_fraction))
    )


def _canonical_status(v19_status: torch.Tensor) -> torch.Tensor:
    source = torch.as_tensor(v19_status).to(torch.int8).cpu().reshape(-1)
    if source.numel() and (
        int(source.min()) < TRUTH_NONE or int(source.max()) > TRUTH_INVALID
    ):
        raise ValueError("V19 truth status is outside its registry")
    output = torch.full_like(source, STATUS_NO_TRUTH)
    output[source == TRUTH_UNIQUE] = STATUS_UNIQUE
    output[source == TRUTH_EQUIVALENT] = STATUS_EQUIVALENT
    output[source == TRUTH_AMBIGUOUS] = STATUS_AMBIGUOUS
    return output


def _filter_truth_csr(
    *,
    offsets: torch.Tensor,
    anchors: torch.Tensor,
    keep_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.as_tensor(offsets).long().cpu()
    anchors = torch.as_tensor(anchors).long().cpu()
    keep = torch.as_tensor(keep_rows).bool().cpu().reshape(-1)
    if (
        offsets.shape != (keep.numel() + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != anchors.numel()
        or bool((offsets[1:] < offsets[:-1]).any())
    ):
        raise ValueError("V21 source truth CSR is invalid")
    counts = offsets[1:] - offsets[:-1]
    edge_keep = torch.repeat_interleave(keep, counts)
    selected = anchors[edge_keep]
    selected_counts = counts * keep.long()
    selected_offsets = torch.cat(
        (torch.zeros(1, dtype=torch.long), selected_counts.cumsum(0))
    )
    return selected_offsets, selected


def build_query_truth_record(
    *,
    frontend_record: Mapping,
    support_record: Mapping,
    v19_truth: Mapping,
    projection_candidate_offsets: torch.Tensor,
    geometry_valid: torch.Tensor,
    action_authorized: bool,
    tier_name: str,
    requested_action: str,
) -> dict:
    """Create compact diagnostic and action-gated per-query positive CSRs."""

    if (
        str(frontend_record.get("role")) != ROLE
        or str(support_record.get("role")) != ROLE
    ):
        raise ValueError("V21 correspondence accepts adaptation records only")
    identity_fields = (
        "query_index",
        "image_name",
        "image_sha256",
        "source_record_sha256",
        "pose_w2c_sha256",
    )
    if any(frontend_record.get(name) != support_record.get(name) for name in identity_fields):
        raise ValueError("V21 frontend/Gaussian query identities differ")
    keypoints = torch.as_tensor(frontend_record["keypoints"]).float().cpu()
    descriptors = torch.as_tensor(frontend_record["descriptors"]).float().cpu()
    count = int(keypoints.shape[0])
    if keypoints.shape != (count, 2) or descriptors.ndim != 2 or descriptors.shape[0] != count:
        raise ValueError("V21 correspondence frontend rows do not align")
    if int(support_record.get("keypoint_count", -1)) != count:
        raise ValueError("V21 correspondence Gaussian keypoint count differs")
    if support_record.get("keypoints_sha256") != tensor_sha256(keypoints):
        raise ValueError("V21 correspondence Gaussian keypoint registry differs")

    source_status = torch.as_tensor(v19_truth["truth_status"]).to(torch.int8).cpu()
    diagnostic_status = _canonical_status(source_status)
    geometry = torch.as_tensor(geometry_valid).bool().cpu().reshape(-1)
    if diagnostic_status.shape != (count,) or geometry.shape != (count,):
        raise ValueError("V21 correspondence truth rows do not align")
    # Geometry gates are abstentions and can never manufacture negatives.
    diagnostic_status[~geometry] = STATUS_NO_TRUTH
    diagnostic_keep = torch.tensor(
        [int(value) in ACTIONABLE_STATUSES for value in diagnostic_status],
        dtype=torch.bool,
    )
    diagnostic_offsets, diagnostic_anchors = _filter_truth_csr(
        offsets=v19_truth["truth_offsets"],
        anchors=v19_truth["truth_anchor_rows"],
        keep_rows=diagnostic_keep,
    )

    certified_status = diagnostic_status.clone()
    if not action_authorized:
        certified_status[diagnostic_keep] = STATUS_NO_TRUTH
    action_keep = diagnostic_keep & bool(action_authorized)
    action_offsets, action_anchors = _filter_truth_csr(
        offsets=v19_truth["truth_offsets"],
        anchors=v19_truth["truth_anchor_rows"],
        keep_rows=action_keep,
    )
    projection_offsets = torch.as_tensor(projection_candidate_offsets).long().cpu()
    if projection_offsets.shape != (count + 1,):
        raise ValueError("V21 projection candidate offsets do not align")
    record = {
        "query_index": int(frontend_record["query_index"]),
        "image_name": str(frontend_record["image_name"]),
        "image_sha256": str(frontend_record["image_sha256"]),
        "sequence_id": str(frontend_record["sequence_id"]),
        "frame_index": int(frontend_record["frame_index"]),
        "block_id": str(frontend_record["block_id"]),
        "role": ROLE,
        "source_record_sha256": str(frontend_record["source_record_sha256"]),
        "pose_w2c_sha256": str(frontend_record["pose_w2c_sha256"]),
        "keypoint_count": count,
        "keypoints_sha256": tensor_sha256(keypoints),
        "descriptors_sha256": tensor_sha256(descriptors),
        "tier_name": str(tier_name),
        "requested_action": str(requested_action),
        "action_authorized": bool(action_authorized),
        "geometry_valid": geometry.contiguous(),
        "source_v19_invalid": (source_status == TRUTH_INVALID).contiguous(),
        "projection_candidate_count": (
            projection_offsets[1:] - projection_offsets[:-1]
        ).contiguous(),
        "diagnostic_truth_status": diagnostic_status.contiguous(),
        "diagnostic_positive_offsets": diagnostic_offsets.contiguous(),
        "diagnostic_positive_anchor_rows": diagnostic_anchors.contiguous(),
        "truth_status": certified_status.contiguous(),
        "truth_status_names": STATUS_NAMES,
        "positive_offsets": action_offsets.contiguous(),
        "positive_anchor_rows": action_anchors.contiguous(),
        "negative_anchor_rows": None,
        "ambiguous_or_unlabelled_are_negative": False,
    }
    validate_query_truth_record(record)
    return record


def validate_query_truth_record(record: Mapping, *, anchor_count: int | None = None) -> None:
    count = int(record.get("keypoint_count", -1))
    if (
        count < 0
        or int(record.get("query_index", -1)) < 0
        or record.get("role") != ROLE
        or record.get("truth_status_names") != STATUS_NAMES
        or record.get("negative_anchor_rows") is not None
        or record.get("ambiguous_or_unlabelled_are_negative") is not False
    ):
        raise ValueError("V21 correspondence query contract is invalid")
    for name in (
        "image_sha256",
        "source_record_sha256",
        "pose_w2c_sha256",
        "keypoints_sha256",
        "descriptors_sha256",
    ):
        _require_sha256(record.get(name), label=name)
    geometry = torch.as_tensor(record.get("geometry_valid"))
    invalid = torch.as_tensor(record.get("source_v19_invalid"))
    candidates = torch.as_tensor(record.get("projection_candidate_count"))
    diagnostic = torch.as_tensor(record.get("diagnostic_truth_status"))
    status = torch.as_tensor(record.get("truth_status"))
    if any(value.shape != (count,) for value in (geometry, invalid, candidates, diagnostic, status)):
        raise ValueError("V21 correspondence row columns do not align")
    if geometry.dtype != torch.bool or invalid.dtype != torch.bool:
        raise ValueError("V21 correspondence row masks must be boolean")
    if count and (
        int(diagnostic.min()) < STATUS_NO_TRUTH
        or int(diagnostic.max()) > STATUS_AMBIGUOUS
        or int(status.min()) < STATUS_NO_TRUTH
        or int(status.max()) > STATUS_AMBIGUOUS
        or bool((candidates < 0).any())
    ):
        raise ValueError("V21 correspondence status/count columns are invalid")
    if bool((~geometry & (diagnostic != STATUS_NO_TRUTH)).any()):
        raise ValueError("invalid Gaussian rows cannot carry diagnostic truth")
    unauthorized_actionable = (status == STATUS_UNIQUE) | (
        status == STATUS_EQUIVALENT
    )
    if record.get("action_authorized") is not True and bool(
        unauthorized_actionable.any()
    ):
        raise ValueError("unauthorized V21 correspondence status did not fail closed")

    for prefix, source_status in (
        ("diagnostic_positive", diagnostic),
        ("positive", status),
    ):
        offsets = torch.as_tensor(record.get(f"{prefix}_offsets")).long()
        anchors = torch.as_tensor(record.get(f"{prefix}_anchor_rows")).long()
        if (
            offsets.shape != (count + 1,)
            or int(offsets[0]) != 0
            or int(offsets[-1]) != anchors.numel()
            or bool((offsets[1:] < offsets[:-1]).any())
        ):
            raise ValueError(f"V21 {prefix} CSR is invalid")
        row_has_positive = offsets[1:] > offsets[:-1]
        expected = (source_status == STATUS_UNIQUE) | (
            source_status == STATUS_EQUIVALENT
        )
        if not torch.equal(row_has_positive, expected):
            raise ValueError(f"V21 {prefix} CSR/status differs")
        if anchors.numel() and (
            int(anchors.min()) < 0
            or (anchor_count is not None and int(anchors.max()) >= anchor_count)
        ):
            raise ValueError(f"V21 {prefix} Anchor row is outside the map")
        counts = offsets[1:] - offsets[:-1]
        if bool((source_status == STATUS_UNIQUE).any()) and bool(
            (counts[source_status == STATUS_UNIQUE] != 1).any()
        ):
            raise ValueError("V21 UNIQUE status must have exactly one positive")
        if bool((source_status == STATUS_EQUIVALENT).any()) and bool(
            (counts[source_status == STATUS_EQUIVALENT] < 2).any()
        ):
            raise ValueError("V21 EQUIVALENT status must have multiple positives")


def validate_correspondence_payload(payload: Mapping) -> None:
    if not (
        payload.get("schema") == SCHEMA
        and payload.get("version") == VERSION
        and payload.get("protocol") == "test_adapted"
        and payload.get("uses_test_queries") is True
        and payload.get("test_adapted") is True
        and payload.get("role") == ROLE
        and payload.get("training_consumers_allowed")
        is bool(payload.get("action_authorized"))
        and payload.get("control_or_confirmation_forbidden") is True
        and payload.get("negative_labels_created") is False
        and payload.get("ambiguous_or_unlabelled_are_negative") is False
        and payload.get("feedback_enters_mapping_track_registry") is False
        and payload.get("artifact_writes_map") is False
        and payload.get("exact_poselib_recovery_is_identity_truth") is False
        and payload.get("semantics") == SEMANTICS
    ):
        raise ValueError("unsupported V21 correspondence payload")
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("V21 correspondence input lineage is missing")
    required_sources = (
        "stable_map",
        "gaussian_support",
        "mapping_provenance",
        "mapping_feature_cache",
        "teacher_validation",
    )
    sources = {
        name: _source_record(inputs.get(name), label=name)
        for name in required_sources
    }
    frontend_sources = inputs.get("frontend_caches")
    producer_sources = inputs.get("producer_sources")
    if not isinstance(frontend_sources, list) or not frontend_sources:
        raise ValueError("V21 correspondence frontend lineage is empty")
    if not isinstance(producer_sources, list) or not producer_sources:
        raise ValueError("V21 correspondence producer lineage is empty")
    [_source_record(value, label="frontend cache") for value in frontend_sources]
    [_source_record(value, label="producer") for value in producer_sources]
    if (
        payload.get("stable_map_sha256") != sources["stable_map"]["sha256"]
        or payload.get("gaussian_support_sha256")
        != sources["gaussian_support"]["sha256"]
        or payload.get("mapping_provenance_sha256")
        != sources["mapping_provenance"]["sha256"]
        or payload.get("mapping_feature_cache_sha256")
        != sources["mapping_feature_cache"]["sha256"]
        or payload.get("teacher_validation_sha256")
        != sources["teacher_validation"]["sha256"]
    ):
        raise ValueError("V21 correspondence primary SHA lineage differs")
    decision = payload.get("teacher_action_decision")
    if not isinstance(decision, Mapping):
        raise ValueError("V21 teacher action decision is missing")
    if payload.get("action_authorized") is not bool(
        decision.get("action_authorized")
    ):
        raise ValueError("V21 action authorization fields differ")
    if payload.get("planner_diagnostic_consumers_allowed") is not bool(
        decision.get("planner_diagnostic_authorized")
    ):
        raise ValueError("V21 planner diagnostic authorization differs")
    gates = payload.get("gaussian_geometry_gates")
    if not isinstance(gates, Mapping) or payload.get(
        "gaussian_geometry_gates_sha256"
    ) != sha256_json(gates):
        raise ValueError("V21 Gaussian geometry gate binding is invalid")
    for name in (
        "minimum_alpha",
        "maximum_relative_depth_spread",
        "minimum_local_valid_fraction",
    ):
        if not math.isfinite(float(gates.get(name, math.nan))):
            raise ValueError("V21 Gaussian geometry gate is non-finite")

    registry = payload.get("frontend_shard_registry")
    if not isinstance(registry, Mapping):
        raise ValueError("V21 correspondence frontend registry is missing")
    validate_shard_registry(registry)
    if payload.get("frontend_shard_registry_sha256") != registry.get(
        "registry_sha256"
    ):
        raise ValueError("V21 correspondence frontend registry SHA differs")
    records = payload.get("records")
    registry_rows = sorted(registry["rows"], key=lambda value: int(value["ordinal"]))
    anchor_count = int(payload.get("anchor_count", 0))
    if (
        anchor_count <= 0
        or not isinstance(records, list)
        or len(records) != len(registry_rows)
        or int(payload.get("query_count", -1)) != len(records)
    ):
        raise ValueError("V21 correspondence coverage is incomplete")
    totals = {name: 0 for name in STATUS_NAMES}
    diagnostic_totals = {name: 0 for name in STATUS_NAMES}
    positive_edges = 0
    for record, row in zip(records, registry_rows):
        validate_query_truth_record(record, anchor_count=anchor_count)
        if (
            int(record["query_index"]) != int(row["query_index"])
            or record["image_name"] != row["image_name"]
            or record["image_sha256"] != row["image_sha256"]
            or record["source_record_sha256"] != row["source_record_sha256"]
            or record["action_authorized"] is not payload["action_authorized"]
            or record["tier_name"] != decision.get("tier_name")
            or record["requested_action"] != decision.get("requested_action")
        ):
            raise ValueError("V21 correspondence record registry differs")
        for code, name in enumerate(STATUS_NAMES):
            totals[name] += int((torch.as_tensor(record["truth_status"]) == code).sum())
            diagnostic_totals[name] += int(
                (torch.as_tensor(record["diagnostic_truth_status"]) == code).sum()
            )
        positive_edges += int(torch.as_tensor(record["positive_anchor_rows"]).numel())
    if (
        payload.get("status_counts") != totals
        or payload.get("diagnostic_status_counts") != diagnostic_totals
        or int(payload.get("positive_edge_count", -1)) != positive_edges
    ):
        raise ValueError("V21 correspondence aggregate diagnostics differ")
    if not payload["action_authorized"] and positive_edges != 0:
        raise ValueError("unauthorized V21 artifact contains action positives")
    diagnostic_rows = diagnostic_totals["UNIQUE"] + diagnostic_totals[
        "EQUIVALENT"
    ]
    diagnostic_edges = sum(
        int(torch.as_tensor(record["diagnostic_positive_anchor_rows"]).numel())
        for record in records
    )
    expected_blocked_rows = 0 if payload["action_authorized"] else diagnostic_rows
    expected_blocked_edges = 0 if payload["action_authorized"] else diagnostic_edges
    if (
        int(payload.get("diagnostic_positive_edge_count", -1))
        != diagnostic_edges
        or int(payload.get("blocked_diagnostic_positive_row_count", -1))
        != expected_blocked_rows
        or int(payload.get("blocked_diagnostic_positive_edge_count", -1))
        != expected_blocked_edges
        or (
            expected_blocked_rows > 0
            and payload.get("blocked_diagnostic_positive_reason")
            != decision.get("action_block_reason")
        )
        or (
            expected_blocked_rows == 0
            and payload.get("blocked_diagnostic_positive_reason") is not None
        )
    ):
        raise ValueError("V21 blocked diagnostic-positive report differs")


def validate_frontend_support_alignment(
    frontend_caches: Sequence[Mapping], support_payload: Mapping
) -> tuple[dict, list[tuple[dict, dict]]]:
    """Validate complete adaptation shards and align them with support rows."""

    if not frontend_caches:
        raise ValueError("V21 correspondence requires frontend cache shards")
    for cache in frontend_caches:
        validate_cache_payload(cache)
        if cache.get("role") != ROLE:
            raise ValueError("V21 correspondence frontend role is not adaptation")
    first = frontend_caches[0]
    registry = first["shard_registry"]
    validate_shard_registry(registry)
    shard_count = int(registry["shard_count"])
    by_shard = {}
    records_by_query = {}
    for cache in frontend_caches:
        if cache.get("shard_registry") != registry:
            raise ValueError("V21 correspondence frontend registries differ")
        shard = int(cache["shard_index"])
        if shard in by_shard or int(cache["shard_count"]) != shard_count:
            raise ValueError("V21 correspondence frontend shard is duplicated")
        by_shard[shard] = cache
        for record in cache["records"]:
            query = int(record["query_index"])
            if query in records_by_query:
                raise ValueError("V21 correspondence frontend query is duplicated")
            records_by_query[query] = record
    if sorted(by_shard) != list(range(shard_count)):
        raise ValueError("V21 correspondence requires complete frontend shards")
    validate_support_payload(support_payload)
    if (
        support_payload.get("role") != ROLE
        or support_payload.get("frontend_shard_registry") != registry
    ):
        raise ValueError("V21 Gaussian support uses another frontend registry")
    support_by_query = {
        int(record["query_index"]): record for record in support_payload["records"]
    }
    ordered = []
    for row in sorted(registry["rows"], key=lambda value: int(value["ordinal"])):
        query = int(row["query_index"])
        frontend = records_by_query.get(query)
        support = support_by_query.get(query)
        if frontend is None or support is None:
            raise ValueError("V21 frontend/Gaussian support coverage is incomplete")
        ordered.append((frontend, support))
    if len(support_by_query) != len(ordered) or len(records_by_query) != len(ordered):
        raise ValueError("V21 frontend/Gaussian query registries differ")
    return dict(registry), ordered


def status_counts(records: Sequence[Mapping], *, diagnostic: bool = False) -> dict:
    field = "diagnostic_truth_status" if diagnostic else "truth_status"
    output = {name: 0 for name in STATUS_NAMES}
    for record in records:
        status = torch.as_tensor(record[field])
        for code, name in enumerate(STATUS_NAMES):
            output[name] += int((status == code).sum())
    return output


def atomic_torch_save_fresh(payload: Mapping, output: str | Path) -> Path:
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"V21 correspondence output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f".{output.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        torch.save(dict(payload), temporary)
        validate_correspondence_payload(
            torch.load(temporary, map_location="cpu", weights_only=False)
        )
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(
                f"V21 correspondence output appeared while running: {output}"
            ) from error
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return output


__all__ = [
    "ACTIONABLE_STATUSES",
    "MUTATING_ACTIONS",
    "ROLE",
    "SCHEMA",
    "SEMANTICS",
    "STATUS_AMBIGUOUS",
    "STATUS_EQUIVALENT",
    "STATUS_NAMES",
    "STATUS_NO_TRUTH",
    "STATUS_UNIQUE",
    "VERSION",
    "atomic_torch_save_fresh",
    "build_query_truth_record",
    "gaussian_row_validity",
    "resolve_teacher_action",
    "sha256_json",
    "status_counts",
    "validate_correspondence_payload",
    "validate_frontend_support_alignment",
    "validate_query_truth_record",
]
