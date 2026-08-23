"""Fail-closed serialized contracts for the V6 closed-loop mainline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path


RENDER_OBSERVATION_SCHEMA = "render_observation_cache_v2"
ASSOCIATION_GRAPH_SCHEMA = "projective_association_graph_v2"
ANCHOR_CANDIDATE_SCHEMA = "projective_anchor_candidates_v2"
FEEDBACK_SCHEMA = "self_localization_feedback_v1"
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
