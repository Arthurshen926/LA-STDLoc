"""Fail-closed serialized contracts for the V6 closed-loop mainline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path


RENDER_OBSERVATION_SCHEMA = "render_observation_cache_v2"
ASSOCIATION_GRAPH_SCHEMA = "projective_association_graph_v2"
ANCHOR_CANDIDATE_SCHEMA = "projective_anchor_candidates_v2"
FEEDBACK_SCHEMA = "self_localization_feedback_v2"
FEEDBACK_VERSION = 3
POSITIVE_IDENTITY_CONTRACT_SCHEMA = "lafgs_v6_exact_identity_positive_contract"
ROUND_SCHEMA = "closed_loop_distillation_round_v1"
DESCRIPTOR_SPLIT_SCHEMA = "lafgs_v6_sequence_block_descriptor_split"


def require_mapping_only(payload: Mapping, *, label: str) -> None:
    if payload.get("uses_source_mapping_rgb") is not False:
        raise ValueError(f"{label} must be source-mapping-RGB-free")
    if payload.get("uses_test_queries") is not False:
        raise ValueError(f"{label} must be test-query-free")


def require_schema(payload: Mapping, schema: str, *, label: str) -> None:
    if payload.get("schema") != schema:
        raise ValueError(f"{label} schema differs: {payload.get('schema')} != {schema}")
    require_mapping_only(payload, label=label)
    if schema == FEEDBACK_SCHEMA:
        if int(payload.get("version", -1)) != FEEDBACK_VERSION:
            raise ValueError(f"{label} feedback version is not identity-safe")
        require_exact_identity_positive_contract(
            payload.get("positive_identity_contract"),
            label=f"{label} positive identity contract",
        )


def exact_identity_positive_contract() -> dict:
    """Return the immutable semantics of V6 descriptor-positive labels."""

    return {
        "schema": POSITIVE_IDENTITY_CONTRACT_SCHEMA,
        "version": 1,
        "identity_source": "projective_anchor_observations_csr",
        "strong_positive": ("exact_observation_identity_and_loo_projective_compatible"),
        "geometry_compatible_nonidentity": "ignore",
        "identity_projective_incompatible": "ignore",
        "negative": "neither_identity_nor_projective_compatible",
        "missing_identity": "fail_closed",
        "duplicate_query_keypoint_identity": "fail_closed",
    }


def require_exact_identity_positive_contract(
    payload: Mapping, *, label: str = "positive identity contract"
) -> None:
    """Reject radius-only or otherwise ambiguous descriptor supervision."""

    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} is missing")
    expected = exact_identity_positive_contract()
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"{label} differs at {key}: {payload.get(key)!r} != {value!r}"
            )


def round_directory(root: Path, index: int) -> Path:
    if int(index) < 0:
        raise ValueError("round index must be non-negative")
    return Path(root) / f"round_{int(index)}"


def validate_ordered_query_registry(names: Sequence[str]) -> tuple[str, ...]:
    ordered = tuple(str(name) for name in names)
    if not ordered or any(not name for name in ordered):
        raise ValueError("ordered query registry must be non-empty")
    if len(set(ordered)) != len(ordered):
        raise ValueError("ordered query registry contains duplicates")
    return ordered


def ordered_query_registry_sha256(names: Sequence[str]) -> str:
    ordered = validate_ordered_query_registry(names)
    serialized = json.dumps(
        list(ordered), ensure_ascii=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(serialized).hexdigest()
