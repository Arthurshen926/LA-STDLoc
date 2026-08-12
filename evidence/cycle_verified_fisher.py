"""Cycle-verified, Fisher-aware fixed-budget camera-pair selection.

The selector in this module deliberately consumes *candidate-pool match
correspondences*, not the aggregate pair sidecar produced after Track building.
An exact three-camera keypoint cycle cannot be inferred from pair-level counts.
Keeping the probe as a separate, hash-bound artifact makes that causal boundary
explicit and lets the selected pairs reuse the probed matches without paying for
the same descriptor matrices twice.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import heapq
import json
import math
import re
import struct
from typing import Iterable

import torch
import torch.nn.functional as F

from evidence.camera_pair_policy import _camera_centers_and_axes


PROPOSAL_SCHEMA = "lafgs_cycle_verified_pair_proposal_table"
PROBE_SCHEMA = "lafgs_cycle_verified_pair_match_probe"
SELECTION_SCHEMA = "lafgs_cycle_verified_fisher_selection"
POLICY_NAME = "cycle_verified_fisher"
CONTROL_POLICY_NAME = "cycle_verified_fisher_nearest_control"

_MATCHER_PARAMETER_NAMES = (
    "minimum_similarity",
    "minimum_margin",
    "maximum_epipolar_error_px",
    "epipolar_candidate_topk",
    "epipolar_recovered_minimum_similarity",
    "epipolar_recovered_minimum_margin",
)
_MATCH_DIAGNOSTIC_NAMES = (
    "source_keypoint_count",
    "target_keypoint_count",
    "raw_top1_reciprocal_count",
    "descriptor_accepted_before_epipolar_count",
    "epipolar_accepted_top1_count",
    "epipolar_rejected_after_descriptor_count",
    "ambiguity_rejected_count",
    "final_reciprocal_epipolar_count",
    "epipolar_recovered_final_count",
)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _canonical_pairs(
    pairs: Iterable[tuple[int, int]], *, query_count: int
) -> list[tuple[int, int]]:
    canonical = []
    for value in pairs:
        if len(value) != 2:
            raise ValueError("Every candidate pair must contain two query indices")
        left, right = int(value[0]), int(value[1])
        if left >= right:
            raise ValueError("Candidate pairs must be canonical with left < right")
        if left < 0 or right >= int(query_count):
            raise ValueError("Candidate pair query index is out of range")
        canonical.append((left, right))
    if canonical != sorted(set(canonical)):
        raise ValueError("Candidate pairs must be unique and lexicographically sorted")
    return canonical


def _candidate_pool_sha256(
    pairs: list[tuple[int, int]], keypoint_counts: list[int]
) -> str:
    payload = {
        "keypoint_counts": [int(value) for value in keypoint_counts],
        "pairs": [[int(left), int(right)] for left, right in pairs],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _hash_tensor(hasher, value: torch.Tensor, *, floating: bool = False) -> None:
    tensor = torch.as_tensor(value).detach().cpu().reshape(-1)
    hasher.update(struct.pack("<Q", int(tensor.numel())))
    if floating:
        for item in tensor.double().tolist():
            hasher.update(struct.pack("<d", float(item)))
    else:
        for item in tensor.long().tolist():
            hasher.update(struct.pack("<q", int(item)))


def _probe_content_sha256(payload: dict) -> str:
    """Hash all scientific probe content, excluding paths and the hash itself."""
    hasher = hashlib.sha256()
    header = {
        "schema": payload.get("schema"),
        "version": payload.get("version"),
        "uses_test_queries": payload.get("uses_test_queries"),
        "query_count": payload.get("query_count"),
        "query_names_sha256": payload.get("query_names_sha256"),
        "query_cache_sha256": payload.get("query_cache_sha256"),
        "mapping_keypoint_count": payload.get("mapping_keypoint_count"),
        "mapping_nms_radius": payload.get("mapping_nms_radius"),
        "mapping_scope": payload.get("mapping_scope"),
        "candidate_pool_construction": payload.get("candidate_pool", {}).get(
            "construction"
        ),
        "candidate_pool_parameters": payload.get("candidate_pool", {}).get(
            "parameters"
        ),
        "candidate_pool_sha256": payload.get("candidate_pool", {}).get("sha256"),
        "matcher": payload.get("matcher"),
        "detector_scores_applied": payload.get("detector_scores_applied"),
    }
    hasher.update(
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    )
    candidate = payload["candidate_pool"]
    matches = payload["matches"]
    _hash_tensor(hasher, candidate["left_query_index"])
    _hash_tensor(hasher, candidate["right_query_index"])
    _hash_tensor(hasher, payload["keypoint_counts"])
    _hash_tensor(hasher, matches["offsets"])
    _hash_tensor(hasher, matches["source_keypoint_index"])
    _hash_tensor(hasher, matches["target_keypoint_index"])
    _hash_tensor(hasher, matches["confidence"], floating=True)
    diagnostics = payload.get("pair_diagnostics", {})
    for name in sorted(diagnostics):
        hasher.update(name.encode())
        _hash_tensor(hasher, diagnostics[name])
    return hasher.hexdigest()


def _selection_content_sha256(payload: dict) -> str:
    hasher = hashlib.sha256()
    header = {
        name: value
        for name, value in payload.items()
        if name not in {"content_sha256", "selected_pair"}
    }
    hasher.update(
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    )
    selected = payload.get("selected_pair", {})
    for name in sorted(selected):
        hasher.update(name.encode())
        _hash_tensor(
            hasher,
            selected[name],
            floating=torch.as_tensor(selected[name]).is_floating_point(),
        )
    return hasher.hexdigest()


def _proposal_content_sha256(payload: dict) -> str:
    """Hash the proposal science content without treating old factors as lineage."""
    hasher = hashlib.sha256()
    arms = payload.get("arms", {})
    header = {
        "schema": payload.get("schema"),
        "version": payload.get("version"),
        "uses_test_queries": payload.get("uses_test_queries"),
        "query_count": payload.get("query_count"),
        "query_names_sha256": payload.get("query_names_sha256"),
        "query_cache_sha256": payload.get("query_cache_sha256"),
        "mapping_keypoint_count": payload.get("mapping_keypoint_count"),
        "mapping_nms_radius": payload.get("mapping_nms_radius"),
        "exact_pair_budget": payload.get("exact_pair_budget"),
        "source_contract": payload.get("source_contract"),
        "arm_sources": {
            name: {
                "source_policy": value.get("source_policy"),
                "source_artifact_sha256": value.get("source_artifact", {}).get(
                    "sha256"
                ),
                "unavailable_source_lineage": value.get(
                    "unavailable_source_lineage"
                ),
            }
            for name, value in sorted(arms.items())
        },
        "candidate_union": payload.get("candidate_union"),
    }
    hasher.update(
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    )
    for name, value in sorted(arms.items()):
        hasher.update(name.encode())
        pair = value.get("pair", {})
        _hash_tensor(hasher, pair.get("left_query_index"))
        _hash_tensor(hasher, pair.get("right_query_index"))
    return hasher.hexdigest()


def materialize_pair_proposal_table(
    *,
    nearest_pairs: list[tuple[int, int]],
    geometry_pairs: list[tuple[int, int]],
    query_count: int,
    query_names_sha256: str,
    query_cache_path: str,
    query_cache_sha256: str,
    mapping_keypoint_count: int,
    mapping_nms_radius: int,
    mapping_scope: dict,
    exact_pair_budget: int,
    nearest_source_path: str,
    nearest_source_sha256: str,
    nearest_unavailable_source_lineage: list[str],
    geometry_source_path: str,
    geometry_source_sha256: str,
    geometry_unavailable_source_lineage: list[str],
) -> dict:
    """Attest only two archived proposal pair tables against a fresh cache.

    Old Track factors are used solely as immutable sources of ordered camera
    pairs.  Their Track/geometry values and incomplete factor lineage are not
    copied or upgraded into a new factor claim.
    """
    query_count = int(query_count)
    pair_budget = int(exact_pair_budget)
    mapping_k = int(mapping_keypoint_count)
    mapping_nms = int(mapping_nms_radius)
    if query_count <= 1 or pair_budget <= 0 or mapping_k <= 0 or mapping_nms <= 0:
        raise ValueError("Proposal query/K/NMS/budget contracts must be positive")
    if not _is_sha256(query_names_sha256) or not _is_sha256(query_cache_sha256):
        raise ValueError("Proposal cache lineage requires exact SHA-256 values")
    if (
        not isinstance(mapping_scope, dict)
        or mapping_scope.get("uses_test_queries") is not False
    ):
        raise ValueError("Proposal requires an explicit mapping-scope proof")
    source_values = (nearest_source_sha256, geometry_source_sha256)
    if not all(_is_sha256(value) for value in source_values):
        raise ValueError("Proposal source artifacts require exact SHA-256 values")
    nearest_pairs = _canonical_pairs(nearest_pairs, query_count=query_count)
    geometry_pairs = _canonical_pairs(geometry_pairs, query_count=query_count)
    if len(nearest_pairs) != pair_budget or len(geometry_pairs) != pair_budget:
        raise ValueError("Each proposal arm must preserve the exact pair budget")
    union, graph = bounded_union_candidate_pool(
        pair_sets=(nearest_pairs, geometry_pairs),
        query_count=query_count,
        maximum_pair_count=2 * pair_budget,
    )

    def arm(
        *,
        policy: str,
        pairs: list[tuple[int, int]],
        source_path: str,
        source_sha256: str,
        unavailable: list[str],
    ) -> dict:
        return {
            "source_policy": policy,
            "source_artifact": {
                "path": str(source_path),
                "sha256": str(source_sha256),
            },
            "unavailable_source_lineage": sorted(set(map(str, unavailable))),
            "pair": {
                "left_query_index": torch.as_tensor(
                    [left for left, _ in pairs], dtype=torch.long
                ),
                "right_query_index": torch.as_tensor(
                    [right for _, right in pairs], dtype=torch.long
                ),
            },
        }

    payload = {
        "schema": PROPOSAL_SCHEMA,
        "version": 1,
        "uses_test_queries": False,
        "query_count": query_count,
        "query_names_sha256": str(query_names_sha256),
        "query_cache_path": str(query_cache_path),
        "query_cache_sha256": str(query_cache_sha256),
        "mapping_keypoint_count": mapping_k,
        "mapping_nms_radius": mapping_nms,
        "mapping_scope": deepcopy(mapping_scope),
        "exact_pair_budget": pair_budget,
        "source_contract": {
            "scope": "archived_pair_tables_only",
            "track_factor_lineage_reused": False,
            "track_or_geometry_measurements_reused": False,
            "fresh_cache_is_authoritative_for_query_order_k_nms": True,
        },
        "arms": {
            "nearest": arm(
                policy="nearest",
                pairs=nearest_pairs,
                source_path=nearest_source_path,
                source_sha256=nearest_source_sha256,
                unavailable=nearest_unavailable_source_lineage,
            ),
            "mapping_geometry": arm(
                policy="parallax_diverse",
                pairs=geometry_pairs,
                source_path=geometry_source_path,
                source_sha256=geometry_source_sha256,
                unavailable=geometry_unavailable_source_lineage,
            ),
        },
        "candidate_union": {
            **graph,
            "sha256": _candidate_pool_sha256(union, []),
        },
    }
    payload["content_sha256"] = _proposal_content_sha256(payload)
    validate_pair_proposal_table(payload)
    return payload


def proposal_arm_pairs(payload: dict, arm: str) -> list[tuple[int, int]]:
    """Return one validated proposal arm's canonical pair table."""
    validate_pair_proposal_table(payload)
    if arm not in {"nearest", "mapping_geometry"}:
        raise ValueError("Proposal arm must be nearest or mapping_geometry")
    pair = payload["arms"][arm]["pair"]
    return list(
        zip(
            torch.as_tensor(pair["left_query_index"]).long().tolist(),
            torch.as_tensor(pair["right_query_index"]).long().tolist(),
        )
    )


def validate_pair_proposal_table(
    payload: dict,
    *,
    expected_query_names_sha256: str | None = None,
    expected_query_cache_path: str | None = None,
    expected_query_cache_sha256: str | None = None,
    expected_mapping_keypoint_count: int | None = None,
    expected_mapping_nms_radius: int | None = None,
    expected_mapping_scope: dict | None = None,
    expected_pair_budget: int | None = None,
    expected_candidate_pair_count: int | None = None,
    expected_candidate_component_count: int | None = None,
    expected_content_sha256: str | None = None,
) -> None:
    """Validate a pair-only attestation without inferring missing old lineage."""
    if (
        payload.get("schema") != PROPOSAL_SCHEMA
        or int(payload.get("version", -1)) != 1
        or payload.get("uses_test_queries") is not False
    ):
        raise ValueError("Unexpected pair-proposal table contract")
    source_contract = payload.get("source_contract")
    if (
        not isinstance(source_contract, dict)
        or source_contract.get("scope") != "archived_pair_tables_only"
        or source_contract.get("track_factor_lineage_reused") is not False
        or source_contract.get("track_or_geometry_measurements_reused") is not False
        or source_contract.get("fresh_cache_is_authoritative_for_query_order_k_nms")
        is not True
    ):
        raise ValueError("Proposal table overclaims its archived source lineage")
    query_count = int(payload.get("query_count", -1))
    pair_budget = int(payload.get("exact_pair_budget", -1))
    if query_count <= 1 or pair_budget <= 0:
        raise ValueError("Proposal table has invalid query/budget axes")
    if not _is_sha256(payload.get("query_names_sha256")) or not _is_sha256(
        payload.get("query_cache_sha256")
    ):
        raise ValueError("Proposal table lacks exact cache lineage")
    mapping_scope = payload.get("mapping_scope")
    if (
        not isinstance(mapping_scope, dict)
        or mapping_scope.get("uses_test_queries") is not False
        or mapping_scope.get("mode")
        not in {
            "query_cache_explicit_mapping_only",
            "mapping_sparse_refresh_equivalence_v2",
        }
    ):
        raise ValueError("Proposal table lacks a valid mapping-scope proof")
    if expected_mapping_scope is not None and mapping_scope != expected_mapping_scope:
        raise ValueError("Proposal table mapping scope differs from expected")
    expected_values = (
        ("query_names_sha256", expected_query_names_sha256),
        ("query_cache_sha256", expected_query_cache_sha256),
        ("mapping_keypoint_count", expected_mapping_keypoint_count),
        ("mapping_nms_radius", expected_mapping_nms_radius),
        ("exact_pair_budget", expected_pair_budget),
        ("content_sha256", expected_content_sha256),
    )
    for name, expected in expected_values:
        if expected is not None and payload.get(name) != expected:
            raise ValueError(f"Proposal table {name} differs from expected")
    if expected_query_cache_path is not None and str(
        payload.get("query_cache_path", "")
    ) != str(expected_query_cache_path):
        raise ValueError("Proposal table names a different query-cache path")
    arms = payload.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"nearest", "mapping_geometry"}:
        raise ValueError("Proposal table must contain exactly two named arms")
    policies = {"nearest": "nearest", "mapping_geometry": "parallax_diverse"}
    pair_sets = []
    for name, expected_policy in policies.items():
        value = arms[name]
        source = value.get("source_artifact") if isinstance(value, dict) else None
        if (
            not isinstance(source, dict)
            or value.get("source_policy") != expected_policy
            or not source.get("path")
            or not _is_sha256(source.get("sha256"))
            or not isinstance(value.get("unavailable_source_lineage"), list)
        ):
            raise ValueError(f"Proposal arm {name} lacks source-table attestation")
        pair = value.get("pair", {})
        left = torch.as_tensor(pair.get("left_query_index"), dtype=torch.long).reshape(-1)
        right = torch.as_tensor(pair.get("right_query_index"), dtype=torch.long).reshape(-1)
        if left.numel() != pair_budget or right.numel() != pair_budget:
            raise ValueError(f"Proposal arm {name} violates the exact pair budget")
        pair_sets.append(
            _canonical_pairs(list(zip(left.tolist(), right.tolist())), query_count=query_count)
        )
    union, graph = bounded_union_candidate_pool(
        pair_sets=pair_sets,
        query_count=query_count,
        maximum_pair_count=2 * pair_budget,
    )
    expected_union = {
        **graph,
        "sha256": _candidate_pool_sha256(union, []),
    }
    if payload.get("candidate_union") != expected_union:
        raise ValueError("Proposal candidate-union diagnostics are stale")
    if expected_candidate_pair_count is not None and len(union) != int(
        expected_candidate_pair_count
    ):
        raise ValueError("Proposal candidate-union count differs from expected")
    if expected_candidate_component_count is not None and int(
        graph["component_count"]
    ) != int(expected_candidate_component_count):
        raise ValueError("Proposal candidate components differ from expected")
    actual_content = _proposal_content_sha256(payload)
    if payload.get("content_sha256") != actual_content:
        raise ValueError("Proposal table content SHA-256 is stale")
    if expected_content_sha256 is not None and actual_content != expected_content_sha256:
        raise ValueError("Proposal table differs from expected content SHA-256")


def materialize_pair_match_probe(
    *,
    candidate_pairs: list[tuple[int, int]],
    pair_matches: dict[
        tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ],
    pair_diagnostics: dict[tuple[int, int], dict[str, int]] | None,
    keypoint_counts: list[int],
    query_names_sha256: str,
    query_cache_sha256: str,
    mapping_keypoint_count: int,
    mapping_nms_radius: int,
    candidate_pool_construction: str,
    candidate_pool_parameters: dict,
    matcher_parameters: dict,
    detector_scores_applied: bool,
) -> dict:
    """Pack exact candidate correspondences into a fail-closed probe contract."""
    query_count = len(keypoint_counts)
    pairs = _canonical_pairs(candidate_pairs, query_count=query_count)
    if set(pair_matches) != set(pairs):
        raise ValueError("Probe matches must contain every candidate pair exactly once")
    if not _is_sha256(query_names_sha256) or not _is_sha256(query_cache_sha256):
        raise ValueError("Probe input lineage requires exact SHA-256 values")
    if set(matcher_parameters) != set(_MATCHER_PARAMETER_NAMES):
        raise ValueError("Probe matcher parameters are incomplete or contain extras")
    if not candidate_pool_construction:
        raise ValueError("Candidate-pool construction must be named")
    if int(mapping_keypoint_count) <= 0 or int(mapping_nms_radius) <= 0:
        raise ValueError("Mapping K and NMS contracts must be positive")
    keypoint_counts = [int(value) for value in keypoint_counts]
    if any(value < 0 or value > int(mapping_keypoint_count) for value in keypoint_counts):
        raise ValueError("Probe keypoint counts violate the mapping-K contract")

    offsets = [0]
    sources = []
    targets = []
    confidences = []
    diagnostics = pair_diagnostics or {}
    if set(diagnostics) != set(pairs) or any(
        set(value) != set(_MATCH_DIAGNOSTIC_NAMES)
        for value in diagnostics.values()
    ):
        raise ValueError(
            "Probe diagnostics must contain the exact matcher funnel for every pair"
        )
    diagnostic_names = sorted(_MATCH_DIAGNOSTIC_NAMES)
    diagnostic_columns = {
        name: torch.zeros(len(pairs), dtype=torch.long) for name in diagnostic_names
    }
    for pair_index, pair in enumerate(pairs):
        source, target, confidence = pair_matches[pair]
        source = torch.as_tensor(source, dtype=torch.long).detach().cpu().reshape(-1)
        target = torch.as_tensor(target, dtype=torch.long).detach().cpu().reshape(-1)
        confidence = (
            torch.as_tensor(confidence, dtype=torch.float64)
            .detach()
            .cpu()
            .reshape(-1)
        )
        if source.numel() != target.numel() or source.numel() != confidence.numel():
            raise ValueError("Per-pair probe match columns must align")
        left, right = pair
        if source.numel() and (
            int(source.min()) < 0
            or int(source.max()) >= keypoint_counts[left]
            or int(target.min()) < 0
            or int(target.max()) >= keypoint_counts[right]
        ):
            raise ValueError("Probe keypoint index is out of range")
        if source.unique().numel() != source.numel() or target.unique().numel() != target.numel():
            raise ValueError("Probe pairs must preserve reciprocal one-to-one matches")
        if not bool(torch.isfinite(confidence).all()) or bool((confidence < 0).any()):
            raise ValueError("Probe confidences must be finite and non-negative")
        sources.append(source)
        targets.append(target)
        confidences.append(confidence)
        offsets.append(offsets[-1] + int(source.numel()))
        for name, value in diagnostics.get(pair, {}).items():
            diagnostic_columns[name][pair_index] = int(value)

    payload = {
        "schema": PROBE_SCHEMA,
        "version": 1,
        "uses_test_queries": False,
        "query_count": int(query_count),
        "query_names_sha256": str(query_names_sha256),
        "query_cache_sha256": str(query_cache_sha256),
        "mapping_keypoint_count": int(mapping_keypoint_count),
        "mapping_nms_radius": int(mapping_nms_radius),
        "keypoint_counts": torch.as_tensor(keypoint_counts, dtype=torch.long),
        "candidate_pool": {
            "construction": str(candidate_pool_construction),
            "parameters": deepcopy(candidate_pool_parameters),
            "sha256": _candidate_pool_sha256(pairs, keypoint_counts),
            "left_query_index": torch.as_tensor(
                [pair[0] for pair in pairs], dtype=torch.long
            ),
            "right_query_index": torch.as_tensor(
                [pair[1] for pair in pairs], dtype=torch.long
            ),
        },
        "matcher": deepcopy(matcher_parameters),
        "detector_scores_applied": bool(detector_scores_applied),
        "matches": {
            "offsets": torch.as_tensor(offsets, dtype=torch.long),
            "source_keypoint_index": (
                torch.cat(sources) if sources else torch.zeros(0, dtype=torch.long)
            ),
            "target_keypoint_index": (
                torch.cat(targets) if targets else torch.zeros(0, dtype=torch.long)
            ),
            "confidence": (
                torch.cat(confidences)
                if confidences
                else torch.zeros(0, dtype=torch.float64)
            ),
        },
        "pair_diagnostics": diagnostic_columns,
    }
    payload["content_sha256"] = _probe_content_sha256(payload)
    validate_pair_match_probe(payload)
    return payload


def validate_pair_match_probe(
    payload: dict,
    *,
    expected_query_names_sha256: str | None = None,
    expected_query_cache_sha256: str | None = None,
    expected_mapping_keypoint_count: int | None = None,
    expected_mapping_nms_radius: int | None = None,
    expected_content_sha256: str | None = None,
) -> None:
    """Validate structure, lineage and correspondence bounds of a probe."""
    if payload.get("schema") != PROBE_SCHEMA or int(payload.get("version", -1)) != 1:
        raise ValueError("Unexpected cycle-verified pair probe schema")
    if payload.get("uses_test_queries") is not False:
        raise ValueError("Pair probe must attest mapping-only queries")
    query_count = int(payload.get("query_count", -1))
    if query_count <= 1:
        raise ValueError("Pair probe requires at least two mapping queries")
    if not _is_sha256(payload.get("query_names_sha256")) or not _is_sha256(
        payload.get("query_cache_sha256")
    ):
        raise ValueError("Pair probe lacks exact input SHA-256 lineage")
    expected_values = (
        ("query_names_sha256", expected_query_names_sha256),
        ("query_cache_sha256", expected_query_cache_sha256),
        ("mapping_keypoint_count", expected_mapping_keypoint_count),
        ("mapping_nms_radius", expected_mapping_nms_radius),
        ("content_sha256", expected_content_sha256),
    )
    for name, expected in expected_values:
        if expected is not None and payload.get(name) != expected:
            raise ValueError(f"Pair probe {name} differs from the expected contract")
    if set(payload.get("matcher", {})) != set(_MATCHER_PARAMETER_NAMES):
        raise ValueError("Pair probe matcher contract is incomplete")
    if payload.get("detector_scores_applied") is not True:
        raise ValueError("Pair probe must bind detector-score confidence weighting")

    keypoint_counts = torch.as_tensor(
        payload.get("keypoint_counts"), dtype=torch.long
    ).reshape(-1)
    if keypoint_counts.numel() != query_count or bool((keypoint_counts < 0).any()):
        raise ValueError("Pair probe keypoint-count table is invalid")
    mapping_k = int(payload.get("mapping_keypoint_count", -1))
    mapping_nms = int(payload.get("mapping_nms_radius", -1))
    if mapping_k <= 0 or mapping_nms <= 0 or bool((keypoint_counts > mapping_k).any()):
        raise ValueError("Pair probe violates its mapping K/NMS contract")

    candidate = payload.get("candidate_pool", {})
    left = torch.as_tensor(candidate.get("left_query_index"), dtype=torch.long).reshape(-1)
    right = torch.as_tensor(candidate.get("right_query_index"), dtype=torch.long).reshape(-1)
    if left.numel() == 0 or left.numel() != right.numel():
        raise ValueError("Pair probe candidate pool is empty or misaligned")
    pairs = list(zip(left.tolist(), right.tolist()))
    _canonical_pairs(pairs, query_count=query_count)
    if not candidate.get("construction") or not isinstance(
        candidate.get("parameters"), dict
    ):
        raise ValueError("Pair probe candidate-pool contract is incomplete")
    expected_pool_sha = _candidate_pool_sha256(pairs, keypoint_counts.tolist())
    if candidate.get("sha256") != expected_pool_sha:
        raise ValueError("Pair probe candidate-pool SHA-256 is stale")

    matches = payload.get("matches", {})
    offsets = torch.as_tensor(matches.get("offsets"), dtype=torch.long).reshape(-1)
    source = torch.as_tensor(
        matches.get("source_keypoint_index"), dtype=torch.long
    ).reshape(-1)
    target = torch.as_tensor(
        matches.get("target_keypoint_index"), dtype=torch.long
    ).reshape(-1)
    confidence = torch.as_tensor(
        matches.get("confidence"), dtype=torch.float64
    ).reshape(-1)
    if offsets.numel() != len(pairs) + 1 or int(offsets[0]) != 0:
        raise ValueError("Pair probe match offsets are invalid")
    if bool((offsets[1:] < offsets[:-1]).any()) or int(offsets[-1]) != source.numel():
        raise ValueError("Pair probe match offsets are not monotonic/exact")
    if source.numel() != target.numel() or source.numel() != confidence.numel():
        raise ValueError("Pair probe flattened match columns do not align")
    if not bool(torch.isfinite(confidence).all()) or bool((confidence < 0).any()):
        raise ValueError("Pair probe confidence contains invalid values")
    for pair_index, (pair_left, pair_right) in enumerate(pairs):
        begin, end = int(offsets[pair_index]), int(offsets[pair_index + 1])
        pair_source = source[begin:end]
        pair_target = target[begin:end]
        if pair_source.numel() and (
            int(pair_source.min()) < 0
            or int(pair_source.max()) >= int(keypoint_counts[pair_left])
            or int(pair_target.min()) < 0
            or int(pair_target.max()) >= int(keypoint_counts[pair_right])
        ):
            raise ValueError("Pair probe flattened keypoint index is out of range")
        if (
            pair_source.unique().numel() != pair_source.numel()
            or pair_target.unique().numel() != pair_target.numel()
        ):
            raise ValueError("Pair probe no longer contains reciprocal one-to-one matches")
    diagnostics = payload.get("pair_diagnostics", {})
    if (
        not isinstance(diagnostics, dict)
        or set(diagnostics) != set(_MATCH_DIAGNOSTIC_NAMES)
        or any(
            torch.as_tensor(value).numel() != len(pairs)
            for value in diagnostics.values()
        )
    ):
        raise ValueError("Pair probe diagnostic columns do not align with candidates")
    diagnostic_tensors = {
        name: torch.as_tensor(value, dtype=torch.long).reshape(-1)
        for name, value in diagnostics.items()
    }
    if any(bool((value < 0).any()) for value in diagnostic_tensors.values()):
        raise ValueError("Pair probe diagnostic counts must be non-negative")
    if not torch.equal(
        diagnostic_tensors["source_keypoint_count"], keypoint_counts[left]
    ) or not torch.equal(
        diagnostic_tensors["target_keypoint_count"], keypoint_counts[right]
    ):
        raise ValueError("Pair probe diagnostic keypoint counts are stale")
    emitted = offsets[1:] - offsets[:-1]
    if not torch.equal(
        diagnostic_tensors["final_reciprocal_epipolar_count"], emitted
    ):
        raise ValueError("Pair probe emitted-match diagnostics are stale")
    if bool(
        (
            diagnostic_tensors["epipolar_recovered_final_count"]
            > diagnostic_tensors["final_reciprocal_epipolar_count"]
        ).any()
    ):
        raise ValueError("Pair probe recovered-match diagnostics are invalid")
    actual_content_sha = _probe_content_sha256(payload)
    if payload.get("content_sha256") != actual_content_sha:
        raise ValueError("Pair probe content SHA-256 is stale")


def pair_matches_from_probe(
    payload: dict,
    *,
    selected_pairs: Iterable[tuple[int, int]] | None = None,
) -> tuple[
    dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    dict[tuple[int, int], dict[str, int]],
]:
    """Recover exact correspondences, optionally for an attested selected subset."""
    validate_pair_match_probe(payload)
    candidate = payload["candidate_pool"]
    pairs = list(
        zip(
            torch.as_tensor(candidate["left_query_index"]).long().tolist(),
            torch.as_tensor(candidate["right_query_index"]).long().tolist(),
        )
    )
    index = {pair: position for position, pair in enumerate(pairs)}
    requested = pairs if selected_pairs is None else list(selected_pairs)
    if requested != sorted(set(requested)) or any(pair not in index for pair in requested):
        raise ValueError("Selected pair set is not an exact candidate-pool subset")
    matches = payload["matches"]
    offsets = torch.as_tensor(matches["offsets"]).long()
    source = torch.as_tensor(matches["source_keypoint_index"]).long()
    target = torch.as_tensor(matches["target_keypoint_index"]).long()
    confidence = torch.as_tensor(matches["confidence"]).double()
    diagnostic_columns = payload.get("pair_diagnostics", {})
    result = {}
    diagnostics = {}
    for pair in requested:
        position = index[pair]
        begin, end = int(offsets[position]), int(offsets[position + 1])
        result[pair] = (
            source[begin:end].clone(),
            target[begin:end].clone(),
            confidence[begin:end].clone(),
        )
        diagnostics[pair] = {
            name: int(torch.as_tensor(column)[position])
            for name, column in diagnostic_columns.items()
        }
    return result, diagnostics


def probe_track_build_inputs(
    pair_match_probe: dict,
    selection_sidecar: dict,
) -> dict:
    """Return the exact precomputed-match inputs for Track construction."""
    validate_cycle_verified_fisher_selection(
        selection_sidecar, pair_match_probe=pair_match_probe
    )
    selected = selection_sidecar.get("selected_pair", {})
    left = torch.as_tensor(selected.get("left_query_index"), dtype=torch.long)
    right = torch.as_tensor(selected.get("right_query_index"), dtype=torch.long)
    if left.numel() != right.numel() or left.numel() != int(
        selection_sidecar.get("exact_pair_budget", -1)
    ):
        raise ValueError("Selection pair table violates its exact budget")
    pairs = list(zip(left.tolist(), right.tolist()))
    return probe_pair_subset_track_build_inputs(pair_match_probe, pairs)


def probe_pair_subset_track_build_inputs(
    pair_match_probe: dict,
    selected_pairs: list[tuple[int, int]],
) -> dict:
    """Return reuse-only Track inputs for any exact, attested probe subset."""
    validate_pair_match_probe(pair_match_probe)
    pairs = _canonical_pairs(
        selected_pairs, query_count=int(pair_match_probe["query_count"])
    )
    matches, diagnostics = pair_matches_from_probe(
        pair_match_probe, selected_pairs=pairs
    )
    return {
        "precomputed_pairs": pairs,
        "precomputed_pair_matches": matches,
        "precomputed_pair_match_diagnostics": diagnostics,
        "precomputed_confidence_includes_detector_scores": True,
    }


def validate_cycle_verified_fisher_selection(
    payload: dict,
    *,
    pair_match_probe: dict | None = None,
    expected_content_sha256: str | None = None,
) -> None:
    """Validate the selected graph, exact budget and probe lineage."""
    if (
        payload.get("schema") != SELECTION_SCHEMA
        or int(payload.get("version", -1)) != 1
        or payload.get("policy") != POLICY_NAME
    ):
        raise ValueError("Unexpected cycle-verified Fisher selection contract")
    if payload.get("uses_test_queries") is not False:
        raise ValueError("Cycle-verified Fisher selection must be mapping-only")
    actual_sha = _selection_content_sha256(payload)
    if payload.get("content_sha256") != actual_sha:
        raise ValueError("Cycle-verified Fisher selection content SHA-256 is stale")
    if expected_content_sha256 is not None and actual_sha != expected_content_sha256:
        raise ValueError("Cycle-verified Fisher selection differs from expected SHA-256")
    budget = int(payload.get("exact_pair_budget", -1))
    selected = payload.get("selected_pair", {})
    required_columns = {
        "left_query_index",
        "right_query_index",
        "connectivity_backbone",
        "candidate_verified_triangle_count",
        "candidate_fisher_utility_sum",
        "selected_completed_triangle_count",
    }
    if set(selected) != required_columns or any(
        torch.as_tensor(value).numel() != budget for value in selected.values()
    ):
        raise ValueError("Cycle-verified Fisher selected-pair columns are invalid")
    left = torch.as_tensor(selected["left_query_index"], dtype=torch.long)
    right = torch.as_tensor(selected["right_query_index"], dtype=torch.long)
    pairs = list(zip(left.tolist(), right.tolist()))
    graph = payload.get("graph", {})
    candidate_graph = payload.get("candidate_graph", {})
    query_count = max(
        max(right.tolist(), default=-1), max(left.tolist(), default=-1)
    ) + 1
    # The probe supplies the authoritative query count below.  Without it, the
    # selected indices still receive canonical/order checks over their range.
    _canonical_pairs(pairs, query_count=max(query_count, 2))
    if (
        int(graph.get("isolated_camera_count", -1)) != 0
        or int(graph.get("minimum_degree", -1))
        < int(payload.get("parameters", {}).get("minimum_camera_degree", -1))
        or int(graph.get("component_count", -1))
        != int(candidate_graph.get("component_count", -2))
    ):
        raise ValueError("Cycle-verified Fisher hard graph constraints are invalid")

    if pair_match_probe is None:
        return
    validate_pair_match_probe(pair_match_probe)
    if payload.get("probe_content_sha256") != pair_match_probe.get("content_sha256"):
        raise ValueError("Selection is not bound to this exact match probe")
    candidate = pair_match_probe["candidate_pool"]
    if payload.get("candidate_pool_sha256") != candidate.get("sha256"):
        raise ValueError("Selection candidate-pool lineage differs from the probe")
    authoritative_query_count = int(pair_match_probe["query_count"])
    _canonical_pairs(pairs, query_count=authoritative_query_count)
    candidate_pairs = list(
        zip(
            candidate["left_query_index"].long().tolist(),
            candidate["right_query_index"].long().tolist(),
        )
    )
    candidate_pair_set = set(candidate_pairs)
    if any(pair not in candidate_pair_set for pair in pairs):
        raise ValueError("Selection contains a pair outside the probed candidate pool")
    actual_candidate_graph = _graph_diagnostics(
        candidate_pairs,
        set(range(len(candidate_pairs))),
        authoritative_query_count,
    )
    actual_selected_graph = _graph_diagnostics(
        pairs, set(range(len(pairs))), authoritative_query_count
    )
    if candidate_graph != actual_candidate_graph or graph != actual_selected_graph:
        raise ValueError("Selection graph diagnostics are stale")


@torch.no_grad()
def build_pair_match_probe(
    *,
    candidate_pairs: list[tuple[int, int]],
    descriptors: list[torch.Tensor],
    keypoints: list[torch.Tensor],
    detector_scores: list[torch.Tensor],
    camera_K: torch.Tensor,
    pose_w2c: torch.Tensor,
    query_names_sha256: str,
    query_cache_sha256: str,
    mapping_keypoint_count: int,
    mapping_nms_radius: int,
    candidate_pool_construction: str,
    candidate_pool_parameters: dict,
    matcher_parameters: dict,
    device: str | torch.device,
) -> dict:
    """Run the one unavoidable candidate matcher probe and retain its matches.

    This function is intentionally not called by the selector.  A real runner
    must materialize and hash this artifact first, then reuse its selected
    subset during Track construction.
    """
    # Local import avoids a module cycle: triangulation imports camera policies.
    from evidence.triangulation import reciprocal_epipolar_matches

    query_count = len(descriptors)
    if (
        len(keypoints) != query_count
        or len(detector_scores) != query_count
        or torch.as_tensor(camera_K).shape[0] != query_count
        or torch.as_tensor(pose_w2c).shape[0] != query_count
    ):
        raise ValueError("Probe camera tables do not align")
    pairs = _canonical_pairs(candidate_pairs, query_count=query_count)
    device = torch.device(device)
    pair_matches = {}
    pair_diagnostics = {}
    for left, right in pairs:
        source, target, confidence, diagnostics = reciprocal_epipolar_matches(
            descriptors[left].to(device),
            descriptors[right].to(device),
            keypoints[left],
            keypoints[right],
            camera_K[left],
            pose_w2c[left],
            camera_K[right],
            pose_w2c[right],
            minimum_similarity=float(matcher_parameters["minimum_similarity"]),
            minimum_margin=float(matcher_parameters["minimum_margin"]),
            maximum_epipolar_error_px=float(
                matcher_parameters["maximum_epipolar_error_px"]
            ),
            epipolar_candidate_topk=int(
                matcher_parameters["epipolar_candidate_topk"]
            ),
            recovered_minimum_similarity=float(
                matcher_parameters["epipolar_recovered_minimum_similarity"]
            ),
            recovered_minimum_margin=float(
                matcher_parameters["epipolar_recovered_minimum_margin"]
            ),
            return_diagnostics=True,
        )
        confidence = confidence.cpu() * torch.sqrt(
            detector_scores[left][source.cpu()].float().clamp_min(0.0)
            * detector_scores[right][target.cpu()].float().clamp_min(0.0)
        )
        pair_matches[(left, right)] = (
            source.cpu(),
            target.cpu(),
            confidence.cpu(),
        )
        pair_diagnostics[(left, right)] = diagnostics
    return materialize_pair_match_probe(
        candidate_pairs=pairs,
        pair_matches=pair_matches,
        pair_diagnostics=pair_diagnostics,
        keypoint_counts=[int(value.shape[0]) for value in keypoints],
        query_names_sha256=query_names_sha256,
        query_cache_sha256=query_cache_sha256,
        mapping_keypoint_count=mapping_keypoint_count,
        mapping_nms_radius=mapping_nms_radius,
        candidate_pool_construction=candidate_pool_construction,
        candidate_pool_parameters=candidate_pool_parameters,
        matcher_parameters=matcher_parameters,
        detector_scores_applied=True,
    )


def _world_bearing_table(
    keypoints: list[torch.Tensor], camera_K: torch.Tensor, pose_w2c: torch.Tensor
) -> list[torch.Tensor]:
    result = []
    for query, uv_value in enumerate(keypoints):
        uv = torch.as_tensor(uv_value, dtype=torch.float64).reshape(-1, 2)
        homogeneous = torch.cat(
            (uv, torch.ones((uv.shape[0], 1), dtype=torch.float64)), dim=1
        )
        camera_ray = torch.linalg.solve(
            torch.as_tensor(camera_K[query], dtype=torch.float64), homogeneous.T
        ).T
        world_ray = camera_ray @ torch.as_tensor(
            pose_w2c[query, :3, :3], dtype=torch.float64
        )
        result.append(F.normalize(world_ray, dim=1))
    return result


def _cycle_geometry_batch(
    *,
    cameras: tuple[int, int, int],
    keypoint_index: torch.Tensor,
    edge_confidence: torch.Tensor,
    world_bearings: list[torch.Tensor],
    camera_centers: torch.Tensor,
    camera_K: torch.Tensor,
    pose_w2c: torch.Tensor,
    scene_scale_m: float,
    maximum_reprojection_error_px: float,
) -> dict[str, torch.Tensor]:
    count = int(keypoint_index.shape[0])
    if count == 0:
        empty = torch.zeros(0, dtype=torch.float64)
        return {
            "valid": torch.zeros(0, dtype=torch.bool),
            "fisher_logdet_gain": empty,
            "confidence": empty,
            "utility": empty,
            "reprojection_max_px": empty,
        }
    camera_index = torch.as_tensor(cameras, dtype=torch.long)
    centers = camera_centers[camera_index]
    rays = torch.stack(
        [
            world_bearings[query][keypoint_index[:, position]]
            for position, query in enumerate(cameras)
        ],
        dim=1,
    )
    identity = torch.eye(3, dtype=torch.float64)
    projectors = identity - rays[..., :, None] * rays[..., None, :]
    normal = projectors.sum(dim=1)
    rhs = torch.einsum("ntij,tj->ni", projectors, centers)
    eigenvalues = torch.linalg.eigvalsh(normal)
    numerically_valid = eigenvalues[:, 0] > 1e-10
    xyz = torch.einsum("nij,nj->ni", torch.linalg.pinv(normal), rhs)

    reprojection_errors = []
    positive_depth = torch.ones(count, dtype=torch.bool)
    for position, query in enumerate(cameras):
        camera_xyz = torch.einsum(
            "ij,nj->ni", pose_w2c[query, :3, :3].double(), xyz
        ) + pose_w2c[query, :3, 3].double()
        positive_depth &= camera_xyz[:, 2] > 1e-6
        projected = torch.einsum("ij,nj->ni", camera_K[query].double(), camera_xyz)
        uv = projected[:, :2] / projected[:, 2:].clamp_min(1e-12)
        observed_ray = world_bearings[query][keypoint_index[:, position]]
        # Recover the observed pixel from its camera ray without retaining a
        # second, potentially inconsistent keypoint table in this helper.
        observed_camera_ray = torch.einsum(
            "ij,nj->ni", pose_w2c[query, :3, :3].double(), observed_ray
        )
        observed_pixel = torch.einsum(
            "ij,nj->ni", camera_K[query].double(), observed_camera_ray
        )
        observed_uv = observed_pixel[:, :2] / observed_pixel[:, 2:].clamp_min(1e-12)
        reprojection_errors.append(torch.linalg.norm(uv - observed_uv, dim=1))
    reprojection_max = torch.stack(reprojection_errors, dim=1).max(dim=1).values

    delta = xyz[:, None, :] - centers[None, :, :]
    distance = torch.linalg.norm(delta, dim=2).clamp_min(1e-9)
    direction = delta / distance[..., None]
    tangent = identity - direction[..., :, None] * direction[..., None, :]
    fisher = (
        tangent / distance[..., None, None].square()
    ).sum(dim=1) * float(scene_scale_m) ** 2
    sign, logdet = torch.linalg.slogdet(identity + fisher)
    triangle_confidence = edge_confidence.double().clamp_min(0.0).prod(dim=1).pow(
        1.0 / 3.0
    )
    utility = logdet * triangle_confidence
    valid = (
        numerically_valid
        & positive_depth
        & torch.isfinite(reprojection_max)
        & (reprojection_max <= float(maximum_reprojection_error_px))
        & (sign > 0)
        & torch.isfinite(logdet)
        & (logdet > 0)
        & torch.isfinite(utility)
        & (utility > 0)
    )
    return {
        "valid": valid,
        "fisher_logdet_gain": logdet,
        "confidence": triangle_confidence,
        "utility": utility,
        "reprojection_max_px": reprojection_max,
    }


def _verified_cycle_table(
    *,
    pairs: list[tuple[int, int]],
    pair_matches: dict[
        tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ],
    keypoints: list[torch.Tensor],
    camera_K: torch.Tensor,
    pose_w2c: torch.Tensor,
    maximum_reprojection_error_px: float,
) -> dict[str, torch.Tensor | float]:
    keypoint_counts = [int(value.shape[0]) for value in keypoints]
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    neighbors = defaultdict(set)
    for left, right in pairs:
        neighbors[left].add(right)
        neighbors[right].add(left)
    centers, _ = _camera_centers_and_axes(pose_w2c)
    positive_baselines = torch.linalg.norm(
        centers[torch.as_tensor([left for left, _ in pairs])]
        - centers[torch.as_tensor([right for _, right in pairs])],
        dim=1,
    )
    positive_baselines = positive_baselines[positive_baselines > 1e-9]
    if positive_baselines.numel() == 0:
        raise ValueError("Candidate pool has no positive camera baseline")
    scene_scale = float(positive_baselines.median())
    world_bearings = _world_bearing_table(keypoints, camera_K, pose_w2c)

    all_cameras = []
    all_keypoints = []
    all_edges = []
    all_fisher = []
    all_confidence = []
    all_utility = []
    all_reprojection = []
    for left in sorted(neighbors):
        right_candidates = sorted(index for index in neighbors[left] if index > left)
        for position, middle in enumerate(right_candidates):
            for right in right_candidates[position + 1 :]:
                pair_lm = (left, middle)
                pair_lr = (left, right)
                pair_mr = (middle, right)
                if pair_mr not in pair_index:
                    continue
                lm_left, lm_middle, lm_confidence = pair_matches[pair_lm]
                lr_left, lr_right, lr_confidence = pair_matches[pair_lr]
                mr_middle, mr_right, mr_confidence = pair_matches[pair_mr]
                if lm_left.numel() == 0 or lr_left.numel() == 0 or mr_middle.numel() == 0:
                    continue
                lookup_lr = torch.full((keypoint_counts[left],), -1, dtype=torch.long)
                edge_lr = torch.full_like(lookup_lr, -1)
                confidence_lr = torch.zeros(keypoint_counts[left], dtype=torch.float64)
                lookup_lr[lr_left] = lr_right
                edge_lr[lr_left] = torch.arange(lr_left.numel())
                confidence_lr[lr_left] = lr_confidence.double()
                lookup_mr = torch.full((keypoint_counts[middle],), -1, dtype=torch.long)
                edge_mr = torch.full_like(lookup_mr, -1)
                confidence_mr = torch.zeros(keypoint_counts[middle], dtype=torch.float64)
                lookup_mr[mr_middle] = mr_right
                edge_mr[mr_middle] = torch.arange(mr_middle.numel())
                confidence_mr[mr_middle] = mr_confidence.double()
                cycle_right = lookup_lr[lm_left]
                cycle_mr_edge = edge_mr[lm_middle]
                cycle = (
                    (cycle_right >= 0)
                    & (cycle_mr_edge >= 0)
                    & (lookup_mr[lm_middle] == cycle_right)
                )
                if not bool(cycle.any()):
                    continue
                lm_edge = torch.nonzero(cycle, as_tuple=False).reshape(-1)
                keypoint_index = torch.stack(
                    (lm_left[lm_edge], lm_middle[lm_edge], cycle_right[lm_edge]),
                    dim=1,
                )
                edge_confidence = torch.stack(
                    (
                        lm_confidence[lm_edge].double(),
                        confidence_lr[lm_left[lm_edge]],
                        confidence_mr[lm_middle[lm_edge]],
                    ),
                    dim=1,
                )
                geometry = _cycle_geometry_batch(
                    cameras=(left, middle, right),
                    keypoint_index=keypoint_index,
                    edge_confidence=edge_confidence,
                    world_bearings=world_bearings,
                    camera_centers=centers,
                    camera_K=torch.as_tensor(camera_K),
                    pose_w2c=torch.as_tensor(pose_w2c),
                    scene_scale_m=scene_scale,
                    maximum_reprojection_error_px=maximum_reprojection_error_px,
                )
                valid = geometry["valid"]
                if not bool(valid.any()):
                    continue
                count = int(valid.sum())
                all_cameras.append(
                    torch.tensor([[left, middle, right]], dtype=torch.long).repeat(
                        count, 1
                    )
                )
                all_keypoints.append(keypoint_index[valid])
                all_edges.append(
                    torch.tensor(
                        [[pair_index[pair_lm], pair_index[pair_lr], pair_index[pair_mr]]],
                        dtype=torch.long,
                    ).repeat(count, 1)
                )
                all_fisher.append(geometry["fisher_logdet_gain"][valid])
                all_confidence.append(geometry["confidence"][valid])
                all_utility.append(geometry["utility"][valid])
                all_reprojection.append(geometry["reprojection_max_px"][valid])
    empty_long = torch.zeros((0, 3), dtype=torch.long)
    empty_float = torch.zeros(0, dtype=torch.float64)
    return {
        "scene_scale_m": scene_scale,
        "camera_index": torch.cat(all_cameras) if all_cameras else empty_long,
        "keypoint_index": torch.cat(all_keypoints) if all_keypoints else empty_long,
        "pair_index": torch.cat(all_edges) if all_edges else empty_long,
        "fisher_logdet_gain": torch.cat(all_fisher) if all_fisher else empty_float,
        "confidence": torch.cat(all_confidence) if all_confidence else empty_float,
        "utility": torch.cat(all_utility) if all_utility else empty_float,
        "reprojection_max_px": (
            torch.cat(all_reprojection) if all_reprojection else empty_float
        ),
    }


class _DisjointSet:
    def __init__(self, count: int):
        self.parent = list(range(count))
        self.rank = [0] * count

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> bool:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


def _graph_diagnostics(
    pairs: list[tuple[int, int]], selected: set[int], query_count: int
) -> dict[str, int]:
    degree = torch.zeros(query_count, dtype=torch.long)
    disjoint = _DisjointSet(query_count)
    for edge in selected:
        left, right = pairs[edge]
        degree[left] += 1
        degree[right] += 1
        disjoint.union(left, right)
    return {
        "component_count": len({disjoint.find(index) for index in range(query_count)}),
        "isolated_camera_count": int((degree == 0).sum()),
        "minimum_degree": int(degree.min()) if degree.numel() else 0,
        "maximum_degree": int(degree.max()) if degree.numel() else 0,
    }


def _complete_verified_triangles_bruteforce(
    *,
    triangle_edges: torch.Tensor,
    triangle_utility: torch.Tensor,
    selected: set[int],
    pair_budget: int,
) -> set[int]:
    """Reference implementation of the registered V1 closure objective.

    This intentionally mirrors the original full-scan loop.  Production uses
    the incremental implementation below; retaining this small oracle makes the
    exact selection semantics and tie-break executable in randomized tests.
    """
    selected = set(selected)
    while len(selected) < int(pair_budget) and triangle_edges.numel():
        remaining = int(pair_budget) - len(selected)
        best = None
        best_key = None
        for triangle_index, edge_tensor in enumerate(triangle_edges):
            edge_tuple = tuple(int(value) for value in edge_tensor.tolist())
            missing = tuple(edge for edge in edge_tuple if edge not in selected)
            if not missing or len(missing) > remaining:
                continue
            utility = float(triangle_utility[triangle_index])
            key = (
                utility / len(missing),
                utility,
                -len(missing),
                tuple(-value for value in missing),
            )
            if best_key is None or key > best_key:
                best_key = key
                best = missing
        if best is None:
            break
        selected.update(best)
    return selected


def _edge_triangle_csr(
    triangle_edges: torch.Tensor, *, edge_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a deterministic edge-to-triangle CSR incidence table."""
    edges = torch.as_tensor(triangle_edges).long().cpu().reshape(-1, 3)
    edge_count = int(edge_count)
    if edge_count < 0:
        raise ValueError("edge count must be non-negative")
    flat_edges = edges.reshape(-1)
    if flat_edges.numel() and (
        int(flat_edges.min()) < 0 or int(flat_edges.max()) >= edge_count
    ):
        raise ValueError("verified triangle references an out-of-range pair")
    counts = torch.bincount(flat_edges, minlength=edge_count)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), torch.cumsum(counts, dim=0)))
    if not flat_edges.numel():
        return offsets, torch.zeros(0, dtype=torch.long)
    triangle_indices = torch.arange(edges.shape[0], dtype=torch.long).repeat_interleave(
        3
    )
    order = torch.argsort(flat_edges, stable=True)
    return offsets, triangle_indices[order]


def _closure_heap_entry(
    *, triangle_index: int, missing: tuple[int, ...], utility: float, version: int
) -> tuple:
    """Invert the registered max-key into an exactly ordered min-heap key."""
    return (
        -(utility / len(missing)),
        -utility,
        len(missing),
        missing,
        int(triangle_index),
        int(version),
    )


def _complete_verified_triangles_incremental(
    *,
    triangle_edges: torch.Tensor,
    triangle_utility: torch.Tensor,
    selected: set[int],
    pair_budget: int,
    edge_count: int,
) -> set[int]:
    """Apply the registered closure objective with incremental incidence updates.

    The original implementation rescanned every verified triangle after every
    one-to-three-edge bundle.  Here each triangle is inserted into a heap for
    its current missing-edge count, and is reconsidered only when one of its
    three incident edges is selected.  Version tags discard stale heap entries.

    Heap ordering is the exact inverse of the original max-key, including the
    missing-edge tuple and first-triangle-on-exact-tie behavior.  Separate
    buckets for one, two, and three missing edges preserve the final-slot
    eligibility rule as ``remaining`` falls below three.
    """
    edges = torch.as_tensor(triangle_edges).long().cpu().reshape(-1, 3)
    utilities = torch.as_tensor(triangle_utility).double().cpu().reshape(-1)
    if utilities.numel() != edges.shape[0]:
        raise ValueError("verified triangle edges and utilities do not align")
    selected = set(int(edge) for edge in selected)
    pair_budget = int(pair_budget)
    edge_count = int(edge_count)
    if selected and (min(selected) < 0 or max(selected) >= edge_count):
        raise ValueError("preselected edge is outside the candidate pair registry")
    if pair_budget < len(selected):
        raise ValueError("pair budget is smaller than the preselected edge set")
    if not edges.numel() or len(selected) >= pair_budget:
        return selected

    offsets, incident_triangles = _edge_triangle_csr(edges, edge_count=edge_count)
    edge_rows = [tuple(int(value) for value in row) for row in edges.tolist()]
    utility_values = [float(value) for value in utilities.tolist()]
    versions = [0] * len(edge_rows)
    missing_by_triangle = [
        tuple(edge for edge in row if edge not in selected) for row in edge_rows
    ]
    heaps: list[list[tuple]] = [[], [], [], []]
    for triangle_index, missing in enumerate(missing_by_triangle):
        if missing:
            heapq.heappush(
                heaps[len(missing)],
                _closure_heap_entry(
                    triangle_index=triangle_index,
                    missing=missing,
                    utility=utility_values[triangle_index],
                    version=versions[triangle_index],
                ),
            )

    def valid_head(missing_count: int) -> tuple | None:
        heap = heaps[missing_count]
        while heap:
            entry = heap[0]
            triangle_index = int(entry[-2])
            version = int(entry[-1])
            if (
                version == versions[triangle_index]
                and len(missing_by_triangle[triangle_index]) == missing_count
            ):
                return entry
            heapq.heappop(heap)
        return None

    while len(selected) < pair_budget:
        remaining = pair_budget - len(selected)
        best_entry = None
        for missing_count in range(1, min(remaining, 3) + 1):
            entry = valid_head(missing_count)
            if entry is not None and (best_entry is None or entry < best_entry):
                best_entry = entry
        if best_entry is None:
            break

        triangle_index = int(best_entry[-2])
        new_edges = missing_by_triangle[triangle_index]
        selected.update(new_edges)
        impacted: set[int] = set()
        for edge in new_edges:
            begin, end = int(offsets[edge]), int(offsets[edge + 1])
            impacted.update(
                int(value) for value in incident_triangles[begin:end].tolist()
            )
        for impacted_triangle in impacted:
            previous = missing_by_triangle[impacted_triangle]
            missing = tuple(edge for edge in previous if edge not in selected)
            if missing == previous:
                continue
            versions[impacted_triangle] += 1
            missing_by_triangle[impacted_triangle] = missing
            if missing:
                heapq.heappush(
                    heaps[len(missing)],
                    _closure_heap_entry(
                        triangle_index=impacted_triangle,
                        missing=missing,
                        utility=utility_values[impacted_triangle],
                        version=versions[impacted_triangle],
                    ),
                )
    return selected


def bounded_union_candidate_pool(
    *,
    pair_sets: Iterable[Iterable[tuple[int, int]]],
    query_count: int,
    maximum_pair_count: int,
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    """Union preregistered proposal arms and fail closed on coverage/bounds.

    The intended V1 universe is the union of the frozen nearest graph and one
    equally budgeted mapping-geometry proposal graph.  The union is a proposal
    mechanism only; its overlap/parallax scores never enter the final utility.
    This caps the unavoidable match probe at no more than two frozen budgets.
    """
    query_count = int(query_count)
    maximum_pair_count = int(maximum_pair_count)
    if query_count <= 1 or maximum_pair_count <= 0:
        raise ValueError("Candidate-union query/pair bounds must be positive")
    union: set[tuple[int, int]] = set()
    arm_count = 0
    for arm in pair_sets:
        arm_pairs = [(int(left), int(right)) for left, right in arm]
        _canonical_pairs(arm_pairs, query_count=query_count)
        union.update(arm_pairs)
        arm_count += 1
    if arm_count < 2:
        raise ValueError("Candidate union requires at least two preregistered arms")
    pairs = sorted(union)
    if not pairs or len(pairs) > maximum_pair_count:
        raise ValueError("Candidate union is empty or exceeds its hard probe bound")
    graph = _graph_diagnostics(pairs, set(range(len(pairs))), query_count)
    if graph["isolated_camera_count"]:
        raise RuntimeError("Candidate union contains isolated mapping cameras")
    return pairs, {"arm_count": arm_count, "pair_count": len(pairs), **graph}


def select_cycle_verified_fisher_pairs(
    *,
    pair_match_probe: dict,
    keypoints: list[torch.Tensor],
    camera_K: torch.Tensor,
    pose_w2c: torch.Tensor,
    pair_budget: int,
    minimum_camera_degree: int = 1,
    maximum_cycle_reprojection_error_px: float = 2.0,
) -> tuple[list[tuple[int, int]], dict]:
    """Select an exact-budget connected graph from verified Fisher triangles.

    The objective is the sum of dimensionless landmark information gains
    ``confidence * logdet(I + s^2 F_bearing)`` for exact keypoint triangles
    whose three camera edges are all selected.  Connectivity is lexicographic,
    not a soft penalty: a maximum-utility spanning tree is installed first.
    """
    validate_pair_match_probe(pair_match_probe)
    query_count = int(pair_match_probe["query_count"])
    if (
        len(keypoints) != query_count
        or torch.as_tensor(camera_K).shape[0] != query_count
        or torch.as_tensor(pose_w2c).shape[0] != query_count
    ):
        raise ValueError("Selector camera tables do not align with the probe")
    expected_counts = torch.as_tensor(
        [int(value.shape[0]) for value in keypoints], dtype=torch.long
    )
    if not torch.equal(expected_counts, pair_match_probe["keypoint_counts"].long()):
        raise ValueError("Selector keypoints differ from the probed keypoint tables")
    pair_budget = int(pair_budget)
    minimum_camera_degree = int(minimum_camera_degree)
    if minimum_camera_degree < 1:
        raise ValueError("cycle_verified_fisher requires a positive degree floor")
    candidate = pair_match_probe["candidate_pool"]
    pairs = list(
        zip(
            candidate["left_query_index"].long().tolist(),
            candidate["right_query_index"].long().tolist(),
        )
    )
    if pair_budget > len(pairs):
        raise ValueError("Exact pair budget exceeds the probed candidate pool")
    candidate_graph = _graph_diagnostics(
        pairs, set(range(len(pairs))), query_count
    )
    if candidate_graph["isolated_camera_count"]:
        raise RuntimeError("Probed candidate graph contains isolated cameras")
    minimum_backbone_size = query_count - candidate_graph["component_count"]
    if pair_budget < minimum_backbone_size:
        raise ValueError(
            "Pair budget cannot preserve every candidate-graph component"
        )
    pair_matches, _ = pair_matches_from_probe(pair_match_probe)
    triangle = _verified_cycle_table(
        pairs=pairs,
        pair_matches=pair_matches,
        keypoints=keypoints,
        camera_K=torch.as_tensor(camera_K),
        pose_w2c=torch.as_tensor(pose_w2c),
        maximum_reprojection_error_px=float(maximum_cycle_reprojection_error_px),
    )
    triangle_edges = triangle["pair_index"]
    triangle_utility = triangle["utility"]
    edge_utility = torch.zeros(len(pairs), dtype=torch.float64)
    edge_cycle_count = torch.zeros(len(pairs), dtype=torch.long)
    if triangle_edges.numel():
        repeated_utility = triangle_utility[:, None].expand_as(triangle_edges)
        edge_utility.scatter_add_(0, triangle_edges.reshape(-1), repeated_utility.reshape(-1))
        edge_cycle_count.scatter_add_(
            0,
            triangle_edges.reshape(-1),
            torch.ones(triangle_edges.numel(), dtype=torch.long),
        )

    # Connectivity is a hard lexicographic constraint.  Kruskal on descending
    # verified information gives the best spanning forest available in each
    # candidate-universe component.  This does not invent cross-trajectory
    # edges when the preregistered universe is physically disconnected.
    edge_order = sorted(
        range(len(pairs)),
        key=lambda edge: (
            -float(edge_utility[edge]),
            -int(edge_cycle_count[edge]),
            pairs[edge][0],
            pairs[edge][1],
        ),
    )
    disjoint = _DisjointSet(query_count)
    selected: set[int] = set()
    backbone: set[int] = set()
    degree = torch.zeros(query_count, dtype=torch.long)
    for edge in edge_order:
        left, right = pairs[edge]
        if not disjoint.union(left, right):
            continue
        selected.add(edge)
        backbone.add(edge)
        degree[left] += 1
        degree[right] += 1
        if len(selected) == minimum_backbone_size:
            break
    if len(selected) != minimum_backbone_size:
        raise RuntimeError("Failed to span every candidate-graph component")

    # A degree floor above one is also hard and is satisfied before spending
    # budget on the information objective.
    while bool((degree < minimum_camera_degree).any()):
        if len(selected) >= pair_budget:
            raise RuntimeError("Pair budget cannot satisfy the camera-degree floor")
        deficient = degree < minimum_camera_degree
        candidates = []
        for edge in edge_order:
            if edge in selected:
                continue
            left, right = pairs[edge]
            coverage = int(deficient[left]) + int(deficient[right])
            if coverage:
                candidates.append((coverage, edge))
        if not candidates:
            raise RuntimeError("Candidate graph cannot satisfy the camera-degree floor")
        _, edge = max(
            candidates,
            key=lambda item: (
                item[0],
                float(edge_utility[item[1]]),
                int(edge_cycle_count[item[1]]),
                -pairs[item[1]][0],
                -pairs[item[1]][1],
            ),
        )
        selected.add(edge)
        backbone.add(edge)
        left, right = pairs[edge]
        degree[left] += 1
        degree[right] += 1

    # Complete whole verified triangles whenever their missing-edge bundle fits.
    # This is exactly the registered closure objective and tie-break, evaluated
    # through edge-to-triangle incidence updates instead of O(T * budget) scans.
    selected = _complete_verified_triangles_incremental(
        triangle_edges=triangle_edges,
        triangle_utility=triangle_utility,
        selected=selected,
        pair_budget=pair_budget,
        edge_count=len(pairs),
    )

    # Exact budget is a scientific contract.  Remaining slots use only verified
    # cycle/Fisher evidence for ranking; no overlap/parallax surrogate re-enters.
    for edge in edge_order:
        if len(selected) >= pair_budget:
            break
        selected.add(edge)
    if len(selected) != pair_budget:
        raise AssertionError("cycle_verified_fisher exact-budget contract failed")

    graph = _graph_diagnostics(pairs, selected, query_count)
    if (
        graph["component_count"] != candidate_graph["component_count"]
        or graph["isolated_camera_count"] != 0
        or graph["minimum_degree"] < minimum_camera_degree
    ):
        raise AssertionError("cycle_verified_fisher graph hard constraints failed")
    completed = torch.zeros(int(triangle_edges.shape[0]), dtype=torch.bool)
    for triangle_index, edge_tensor in enumerate(triangle_edges):
        completed[triangle_index] = all(
            int(edge) in selected for edge in edge_tensor.tolist()
        )
    completed_camera_mask = torch.zeros(query_count, dtype=torch.bool)
    if bool(completed.any()):
        completed_camera_mask[triangle["camera_index"][completed].reshape(-1)] = True
    selected_indices = sorted(selected)
    selected_pairs = [pairs[index] for index in selected_indices]
    selected_index_tensor = torch.as_tensor(selected_indices, dtype=torch.long)
    selected_completed_edge = torch.zeros(len(pairs), dtype=torch.long)
    if bool(completed.any()):
        selected_completed_edge.scatter_add_(
            0,
            triangle_edges[completed].reshape(-1),
            torch.ones(triangle_edges[completed].numel(), dtype=torch.long),
        )
    sidecar = {
        "schema": SELECTION_SCHEMA,
        "version": 1,
        "uses_test_queries": False,
        "policy": POLICY_NAME,
        "probe_content_sha256": pair_match_probe["content_sha256"],
        "candidate_pool_sha256": candidate["sha256"],
        "candidate_pair_count": len(pairs),
        "exact_pair_budget": pair_budget,
        "parameters": {
            "minimum_camera_degree": minimum_camera_degree,
            "maximum_cycle_reprojection_error_px": float(
                maximum_cycle_reprojection_error_px
            ),
            "information_model": "bearing_fisher_logdet_v1",
            "information_formula": "confidence_geomean*logdet(I+s^2*F_bearing)",
            "scene_scale_semantics": "median_positive_candidate_baseline_m",
            "connectivity": (
                "hard_maximum_utility_spanning_forest_preserving_"
                "candidate_components"
            ),
        },
        "candidate_graph": candidate_graph,
        "graph": graph,
        "verified_triangle": {
            "candidate_count": int(triangle_edges.shape[0]),
            "selected_completed_count": int(completed.sum()),
            "selected_camera_count": int(completed_camera_mask.sum()),
            "selected_camera_fraction": float(
                completed_camera_mask.double().mean()
            ),
            "selected_cycle_verified_edge_count": int(
                (selected_completed_edge[selected_index_tensor] > 0).sum()
            ),
            "selected_fisher_logdet_gain_sum": float(
                triangle["fisher_logdet_gain"][completed].sum()
            ),
            "selected_confidence_weighted_utility_sum": float(
                triangle_utility[completed].sum()
            ),
            "scene_scale_m": float(triangle["scene_scale_m"]),
            "reprojection_p90_px": (
                float(torch.quantile(triangle["reprojection_max_px"][completed], 0.9))
                if bool(completed.any())
                else math.nan
            ),
        },
        "selected_pair": {
            "left_query_index": torch.as_tensor(
                [pair[0] for pair in selected_pairs], dtype=torch.long
            ),
            "right_query_index": torch.as_tensor(
                [pair[1] for pair in selected_pairs], dtype=torch.long
            ),
            "connectivity_backbone": torch.as_tensor(
                [index in backbone for index in selected_indices], dtype=torch.bool
            ),
            "candidate_verified_triangle_count": edge_cycle_count[
                selected_index_tensor
            ],
            "candidate_fisher_utility_sum": edge_utility[selected_index_tensor],
            "selected_completed_triangle_count": selected_completed_edge[
                selected_index_tensor
            ],
        },
    }
    sidecar["content_sha256"] = _selection_content_sha256(sidecar)
    validate_cycle_verified_fisher_selection(
        sidecar, pair_match_probe=pair_match_probe
    )
    return selected_pairs, sidecar


__all__ = [
    "CONTROL_POLICY_NAME",
    "POLICY_NAME",
    "PROBE_SCHEMA",
    "PROPOSAL_SCHEMA",
    "SELECTION_SCHEMA",
    "bounded_union_candidate_pool",
    "build_pair_match_probe",
    "materialize_pair_match_probe",
    "materialize_pair_proposal_table",
    "pair_matches_from_probe",
    "probe_pair_subset_track_build_inputs",
    "probe_track_build_inputs",
    "proposal_arm_pairs",
    "select_cycle_verified_fisher_pairs",
    "validate_cycle_verified_fisher_selection",
    "validate_pair_match_probe",
    "validate_pair_proposal_table",
]
