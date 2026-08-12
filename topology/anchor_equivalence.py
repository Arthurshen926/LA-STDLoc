"""Audit candidate identity overlap without mutating the deployed anchor map."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from topology.anchor_registry import SCHEMA as REGISTRY_SCHEMA


SCHEMA = "lafgs_anchor_equivalence_audit"
VERSION = 1

PAIR_TRACK_TRACK = 0
PAIR_TRACK_GAUSSIAN = 1
PAIR_GAUSSIAN_GAUSSIAN = 2


def _quantiles(value: torch.Tensor) -> dict[str, float | None]:
    value = torch.as_tensor(value).detach().cpu().float().reshape(-1)
    value = value[torch.isfinite(value)]
    if value.numel() == 0:
        return {key: None for key in ("min", "p10", "p25", "p50", "p75", "p90", "p99", "max")}
    probabilities = torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    values = torch.quantile(value, probabilities).tolist()
    return dict(zip(("min", "p10", "p25", "p50", "p75", "p90", "p99", "max"), values))


def _validate_observations(registry: Mapping) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise ValueError("unsupported Anchor Registry schema")
    count = int(torch.as_tensor(registry["anchor_ids"]).numel())
    offsets = torch.as_tensor(registry["observation_offsets"]).long().reshape(-1)
    queries = torch.as_tensor(registry["observation_query_indices"]).long().reshape(-1)
    keypoints = torch.as_tensor(registry["observation_keypoint_indices"]).long().reshape(-1)
    if offsets.numel() != count + 1:
        raise ValueError("observation offsets do not align with anchors")
    if queries.numel() != keypoints.numel() or int(offsets[-1]) != queries.numel():
        raise ValueError("observation CSR arrays do not align")
    if int(offsets[0]) != 0 or bool((offsets[1:] < offsets[:-1]).any()):
        raise ValueError("observation CSR offsets are invalid")
    if queries.numel() and (int(queries.min()) < 0 or int(keypoints.min()) < 0):
        raise ValueError("observation references must be non-negative")
    return offsets, queries, keypoints


def _shared_observation_pairs(
    registry: Mapping,
    *,
    maximum_anchors_per_observation: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    offsets, queries, keypoints = _validate_observations(registry)
    count = offsets.numel() - 1
    if int(maximum_anchors_per_observation) < 2:
        raise ValueError("maximum anchors per observation must be at least two")
    if queries.numel() == 0:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty, empty, empty, empty, {
            "unique_observation_count": 0,
            "colliding_observation_count": 0,
            "maximum_observation_multiplicity": 0,
            "skipped_high_multiplicity_observation_count": 0,
        }
    anchor_rows = torch.repeat_interleave(
        torch.arange(count), offsets[1:] - offsets[:-1]
    )
    key_base = int(keypoints.max()) + 1
    observation_keys = queries * key_base + keypoints
    order = torch.argsort(observation_keys, stable=True)
    observation_keys = observation_keys[order]
    anchor_rows = anchor_rows[order]
    sorted_queries = queries[order]
    query_groups = torch.as_tensor(
        registry.get("query_group_ids", torch.empty(0, dtype=torch.long))
    ).long()
    if query_groups.numel() and int(queries.max()) >= query_groups.numel():
        raise ValueError("query groups do not cover all observations")
    _, multiplicity = torch.unique_consecutive(
        observation_keys, return_counts=True
    )
    group_offsets = torch.zeros(multiplicity.numel() + 1, dtype=torch.long)
    group_offsets[1:] = torch.cumsum(multiplicity, dim=0)
    pair_counts: Counter[tuple[int, int]] = Counter()
    pair_queries: dict[tuple[int, int], set[int]] = {}
    pair_groups: dict[tuple[int, int], set[int]] = {}
    skipped = 0
    for group, raw_count in enumerate(multiplicity.tolist()):
        if raw_count < 2:
            continue
        begin, end = int(group_offsets[group]), int(group_offsets[group + 1])
        anchors = torch.unique(anchor_rows[begin:end], sorted=True).tolist()
        if len(anchors) < 2:
            continue
        if len(anchors) > int(maximum_anchors_per_observation):
            skipped += 1
            continue
        query_index = int(sorted_queries[begin])
        group_index = (
            int(query_groups[query_index]) if query_groups.numel() else -1
        )
        for left_rank, left in enumerate(anchors[:-1]):
            for right in anchors[left_rank + 1 :]:
                pair = (int(left), int(right))
                pair_counts[pair] += 1
                pair_queries.setdefault(pair, set()).add(query_index)
                if group_index >= 0:
                    pair_groups.setdefault(pair, set()).add(group_index)
    if pair_counts:
        ordered = sorted(pair_counts)
        left = torch.tensor([pair[0] for pair in ordered], dtype=torch.long)
        right = torch.tensor([pair[1] for pair in ordered], dtype=torch.long)
        shared = torch.tensor([pair_counts[pair] for pair in ordered], dtype=torch.long)
        shared_queries = torch.tensor(
            [len(pair_queries[pair]) for pair in ordered], dtype=torch.long
        )
        shared_groups = torch.tensor(
            [len(pair_groups.get(pair, ())) for pair in ordered], dtype=torch.long
        )
    else:
        left = right = shared = shared_queries = shared_groups = torch.empty(
            0, dtype=torch.long
        )
    return left, right, shared, shared_queries, shared_groups, {
        "unique_observation_count": int(multiplicity.numel()),
        "colliding_observation_count": int((multiplicity > 1).sum()),
        "maximum_observation_multiplicity": int(multiplicity.max()),
        "skipped_high_multiplicity_observation_count": int(skipped),
    }


def _mahalanobis_squared(
    delta: torch.Tensor,
    covariance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = int(delta.shape[0])
    result = torch.full((count,), float("nan"), dtype=torch.float32)
    complete = torch.isfinite(covariance).reshape(count, -1).all(dim=1)
    if not bool(complete.any()):
        return result, complete
    matrices = covariance[complete].double()
    eigenvalues = torch.linalg.eigvalsh(matrices)
    positive = eigenvalues[:, 0] > 0.0
    complete_rows = torch.nonzero(complete, as_tuple=False).reshape(-1)
    complete[complete_rows[~positive]] = False
    if bool(positive.any()):
        rows = complete_rows[positive]
        inverse = torch.linalg.pinv(matrices[positive], hermitian=True)
        values = torch.einsum(
            "ni,nij,nj->n", delta[rows].double(), inverse, delta[rows].double()
        )
        result[rows] = values.float()
    return result, complete


def build_equivalence_candidates(
    registry: Mapping,
    *,
    maximum_anchors_per_observation: int = 64,
) -> dict:
    """Build weighted candidate pairs from exact shared real observations.

    No descriptor-only or geometry-only pair is introduced.  The result is an
    audit graph and deliberately contains no final merge decision.
    """
    left, right, shared, shared_queries, shared_groups, collision_report = _shared_observation_pairs(
        registry,
        maximum_anchors_per_observation=maximum_anchors_per_observation,
    )
    anchor_type = torch.as_tensor(registry["anchor_type"]).long().reshape(-1)
    observation_count = torch.as_tensor(registry["observation_count"]).long()
    xyz = torch.as_tensor(registry["anchor_xyz"]).float()
    features = torch.as_tensor(registry["anchor_features"]).float()
    source = torch.as_tensor(registry["source_primitive_ids"]).long()
    dependency = torch.as_tensor(
        registry.get(
            "coarse_dependency_group_ids",
            registry.get("dependency_group_ids", torch.arange(anchor_type.numel())),
        )
    ).long()
    covariance = torch.as_tensor(registry["anchor_position_covariance"]).float()
    if covariance.shape != (anchor_type.numel(), 3, 3):
        raise ValueError("anchor covariance does not align with anchors")
    if left.numel() == 0:
        empty_float = torch.empty(0, dtype=torch.float32)
        empty_bool = torch.empty(0, dtype=torch.bool)
        return {
            "anchor_left": left,
            "anchor_right": right,
            "pair_type": torch.empty(0, dtype=torch.int8),
            "shared_observation_count": shared,
            "shared_query_count": shared_queries,
            "shared_query_group_count": shared_groups,
            "observation_containment": empty_float,
            "observation_jaccard": empty_float,
            "distance_m": empty_float,
            "descriptor_cosine_audit_only": empty_float,
            "same_source_lineage": empty_bool,
            "same_coarse_dependency_group": empty_bool,
            "covariance_complete": empty_bool,
            "mahalanobis_squared": empty_float,
            "collision_report": collision_report,
        }
    left_type, right_type = anchor_type[left], anchor_type[right]
    pair_type = torch.full((left.numel(),), PAIR_TRACK_GAUSSIAN, dtype=torch.int8)
    pair_type[(left_type == 1) & (right_type == 1)] = PAIR_TRACK_TRACK
    pair_type[(left_type == 0) & (right_type == 0)] = PAIR_GAUSSIAN_GAUSSIAN
    left_count = observation_count[left].clamp_min(1)
    right_count = observation_count[right].clamp_min(1)
    containment = shared.float() / torch.minimum(left_count, right_count).float()
    jaccard = shared.float() / (left_count + right_count - shared).clamp_min(1).float()
    delta = xyz[left] - xyz[right]
    pair_covariance = covariance[left] + covariance[right]
    mahalanobis, covariance_complete = _mahalanobis_squared(
        delta, pair_covariance
    )
    return {
        "anchor_left": left,
        "anchor_right": right,
        "pair_type": pair_type,
        "shared_observation_count": shared,
        "shared_query_count": shared_queries,
        "shared_query_group_count": shared_groups,
        "observation_containment": containment,
        "observation_jaccard": jaccard,
        "distance_m": torch.linalg.norm(delta, dim=1),
        # This is intentionally diagnostic and is never used to create a pair.
        "descriptor_cosine_audit_only": F.cosine_similarity(
            features[left], features[right], dim=1
        ),
        "same_source_lineage": (source[left] == source[right]) & (source[left] >= 0),
        "same_coarse_dependency_group": dependency[left] == dependency[right],
        "covariance_complete": covariance_complete,
        "mahalanobis_squared": mahalanobis,
        "collision_report": collision_report,
    }


def anchor_functional_evidence(
    registry: Mapping,
    state: Mapping,
    function_graph: Mapping,
) -> dict[str, torch.Tensor]:
    """Map known canonical-base functional counters onto final Anchor rows."""
    count = int(torch.as_tensor(registry["anchor_ids"]).numel())
    harmful = torch.full((count,), -1, dtype=torch.long)
    opportunity = torch.full((count,), -1, dtype=torch.long)
    anchor_type = torch.as_tensor(registry["anchor_type"]).long()
    base_final_rows = torch.nonzero(anchor_type == 0, as_tuple=False).reshape(-1)
    base_canonical_rows = torch.as_tensor(
        state.get("track_centric_reconstruction", {}).get(
            "base_canonical_rows", torch.empty(0, dtype=torch.long)
        )
    ).long()
    if base_final_rows.numel() != base_canonical_rows.numel():
        raise ValueError("base canonical rows do not align with final map")
    harmful_source = function_graph.get(
        "provenance_harmful_solver_inlier_count",
        function_graph.get("harmful_solver_inlier_count"),
    )
    opportunity_source = function_graph.get(
        "provenance_opportunity_count",
        function_graph.get("candidate_opportunity_count"),
    )
    if harmful_source is None or opportunity_source is None:
        raise ValueError("function graph lacks functional evidence counters")
    harmful_source = torch.as_tensor(harmful_source).long()
    opportunity_source = torch.as_tensor(opportunity_source).long()
    if base_canonical_rows.numel() and (
        int(base_canonical_rows.min()) < 0
        or int(base_canonical_rows.max()) >= harmful_source.numel()
        or int(base_canonical_rows.max()) >= opportunity_source.numel()
    ):
        raise ValueError("base map references rows outside the function graph")
    harmful[base_final_rows] = harmful_source[base_canonical_rows]
    opportunity[base_final_rows] = opportunity_source[base_canonical_rows]
    return {
        "known_harmful_count": harmful,
        "known_opportunity_count": opportunity,
    }


def _pair_type_counts(pair_type: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, int]:
    pair_type = torch.as_tensor(pair_type)
    if mask is not None:
        pair_type = pair_type[torch.as_tensor(mask).bool()]
    return {
        "track_track": int((pair_type == PAIR_TRACK_TRACK).sum()),
        "track_gaussian": int((pair_type == PAIR_TRACK_GAUSSIAN).sum()),
        "gaussian_gaussian": int((pair_type == PAIR_GAUSSIAN_GAUSSIAN).sum()),
    }


def _component_report(
    left: torch.Tensor,
    right: torch.Tensor,
    mask: torch.Tensor,
    anchor_count: int,
) -> dict[str, int]:
    selected = torch.nonzero(mask, as_tuple=False).reshape(-1).tolist()
    parent = list(range(int(anchor_count)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left_value: int, right_value: int) -> None:
        left_root, right_root = find(left_value), find(right_value)
        if left_root != right_root:
            parent[right_root] = left_root

    anchors: set[int] = set()
    for edge in selected:
        left_value, right_value = int(left[edge]), int(right[edge])
        anchors.update((left_value, right_value))
        union(left_value, right_value)
    sizes = Counter(find(anchor) for anchor in anchors)
    return {
        "edge_count": len(selected),
        "anchor_count": len(anchors),
        "component_count": len(sizes),
        "largest_component_size": max(sizes.values(), default=0),
        "one_identity_upper_bound_reduction": len(anchors) - len(sizes),
        "one_identity_upper_bound_fraction_of_map": (
            float(len(anchors) - len(sizes)) / max(int(anchor_count), 1)
        ),
    }


def equivalence_edge_masks(
    candidates: Mapping,
    *,
    distance_scale_m: float,
) -> dict[str, torch.Tensor]:
    """Return reproducible audit masks; neither mask authorizes a merge."""
    if float(distance_scale_m) <= 0.0:
        raise ValueError("distance scale must be positive")
    lineage_proxy = torch.as_tensor(candidates["same_source_lineage"]).bool() | torch.as_tensor(
        candidates["same_coarse_dependency_group"]
    ).bool()
    triage = lineage_proxy & (
        torch.as_tensor(candidates["distance_m"]).float()
        <= float(distance_scale_m)
    )
    independent = triage & (
        torch.as_tensor(candidates["shared_query_group_count"]).long() >= 2
    )
    return {
        "calibrated_triage": triage,
        "independent_support": independent,
    }


def audit_component_ids(
    candidates: Mapping,
    mask: torch.Tensor,
    *,
    anchor_count: int,
) -> torch.Tensor:
    """Label non-trivial audit components, leaving isolated anchors at -1."""
    left = torch.as_tensor(candidates["anchor_left"]).long()
    right = torch.as_tensor(candidates["anchor_right"]).long()
    mask = torch.as_tensor(mask).bool()
    if mask.numel() != left.numel():
        raise ValueError("edge mask does not align with candidate pairs")
    parent = list(range(int(anchor_count)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left_value: int, right_value: int) -> None:
        left_root, right_root = find(left_value), find(right_value)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    anchors: set[int] = set()
    for edge in torch.nonzero(mask, as_tuple=False).reshape(-1).tolist():
        left_value, right_value = int(left[edge]), int(right[edge])
        anchors.update((left_value, right_value))
        union(left_value, right_value)
    roots = sorted({find(anchor) for anchor in anchors})
    compact = {root: index for index, root in enumerate(roots)}
    result = torch.full((int(anchor_count),), -1, dtype=torch.long)
    for anchor in anchors:
        result[anchor] = compact[find(anchor)]
    return result


def summarize_equivalence_audit(
    registry: Mapping,
    candidates: Mapping,
    *,
    distance_scale_m: float,
    functional_evidence: Mapping | None = None,
    containment_thresholds: Sequence[float] = (0.0, 0.1, 0.25, 0.5),
    distance_multipliers: Sequence[float] = (0.5, 1.0, 2.0),
) -> dict:
    if float(distance_scale_m) <= 0.0:
        raise ValueError("distance scale must be positive")
    left = torch.as_tensor(candidates["anchor_left"]).long()
    right = torch.as_tensor(candidates["anchor_right"]).long()
    pair_type = torch.as_tensor(candidates["pair_type"])
    containment = torch.as_tensor(candidates["observation_containment"]).float()
    distance = torch.as_tensor(candidates["distance_m"]).float()
    same_source = torch.as_tensor(candidates["same_source_lineage"]).bool()
    same_dependency = torch.as_tensor(
        candidates["same_coarse_dependency_group"]
    ).bool()
    covariance_complete = torch.as_tensor(candidates["covariance_complete"]).bool()
    shared_queries = torch.as_tensor(candidates["shared_query_count"]).long()
    shared_groups = torch.as_tensor(candidates["shared_query_group_count"]).long()
    lineage_proxy = same_source | same_dependency
    masks = equivalence_edge_masks(
        candidates, distance_scale_m=float(distance_scale_m)
    )
    triage = masks["calibrated_triage"]
    independent = masks["independent_support"]
    report = {
        "schema": SCHEMA,
        "version": VERSION,
        "decision_scope": "audit_only_no_anchor_mutation",
        "distance_scale_m": float(distance_scale_m),
        "observation_collisions": dict(candidates["collision_report"]),
        "candidate_pairs": {
            "count": int(left.numel()),
            "by_type": _pair_type_counts(pair_type),
            "same_source_lineage_count": int(same_source.sum()),
            "same_coarse_dependency_group_count": int(same_dependency.sum()),
            "lineage_proxy_count": int(lineage_proxy.sum()),
            "covariance_complete_count": int(covariance_complete.sum()),
            "shared_observation_count": _quantiles(
                candidates["shared_observation_count"]
            ),
            "shared_query_count": _quantiles(shared_queries),
            "shared_query_group_count": _quantiles(shared_groups),
            "observation_containment": _quantiles(containment),
            "observation_jaccard": _quantiles(candidates["observation_jaccard"]),
            "distance_m": _quantiles(distance),
            "descriptor_cosine_audit_only": _quantiles(
                candidates["descriptor_cosine_audit_only"]
            ),
            "mahalanobis_squared": _quantiles(
                candidates["mahalanobis_squared"]
            ),
        },
        "calibrated_triage_graph": {
            **_component_report(
                left,
                right,
                triage,
                int(torch.as_tensor(registry["anchor_ids"]).numel()),
            ),
            "by_type": _pair_type_counts(pair_type, triage),
            "rule": (
                "exact_shared_observation AND "
                "(same_source OR same_coarse_dependency_group) AND "
                "distance<=mapping_assignment_distance"
            ),
            "is_merge_decision": False,
        },
        "independent_support_graph": {
            **_component_report(
                left,
                right,
                independent,
                int(torch.as_tensor(registry["anchor_ids"]).numel()),
            ),
            "by_type": _pair_type_counts(pair_type, independent),
            "rule": "calibrated_triage_edge AND shared_query_group_count>=2",
            "query_group_semantics": str(
                registry.get("query_group_semantics", "unavailable")
            ),
            "is_merge_decision": False,
        },
        "strict_equivalence_readiness": {
            "triage_edge_count": int(triage.sum()),
            "covariance_complete_triage_edge_count": int(
                (triage & covariance_complete).sum()
            ),
            "ready_for_mahalanobis_calibration": bool(
                int(triage.sum()) > 0 and bool(covariance_complete[triage].all())
            ),
            "missing_requirement": (
                None
                if int(triage.sum()) > 0 and bool(covariance_complete[triage].all())
                else "complete per-anchor covariance for every triage edge"
            ),
        },
        "threshold_sweep": [],
    }
    for containment_threshold in containment_thresholds:
        for distance_multiplier in distance_multipliers:
            mask = (
                lineage_proxy
                & (containment >= float(containment_threshold))
                & (distance <= float(distance_scale_m) * float(distance_multiplier))
            )
            report["threshold_sweep"].append(
                {
                    "minimum_observation_containment": float(
                        containment_threshold
                    ),
                    "maximum_distance_scale": float(distance_multiplier),
                    "edge_count": int(mask.sum()),
                    "by_type": _pair_type_counts(pair_type, mask),
                }
            )
    if functional_evidence is not None:
        harmful = torch.as_tensor(functional_evidence["known_harmful_count"]).long()
        opportunity = torch.as_tensor(
            functional_evidence["known_opportunity_count"]
        ).long()
        if harmful.numel() != torch.as_tensor(registry["anchor_ids"]).numel():
            raise ValueError("functional evidence does not align with anchors")
        known_endpoint = (harmful[left] >= 0) | (harmful[right] >= 0)
        harmful_endpoint = (harmful[left] > 0) | (harmful[right] > 0)
        known_harmful_events = harmful[left].clamp_min(0) + harmful[right].clamp_min(0)
        known_opportunities = opportunity[left].clamp_min(0) + opportunity[right].clamp_min(0)
        report["functional_relevance"] = {
            "known_endpoint_pair_count": int(known_endpoint.sum()),
            "harmful_endpoint_pair_count": int(harmful_endpoint.sum()),
            "known_harmful_event_count": int(known_harmful_events.sum()),
            "known_opportunity_count": int(known_opportunities.sum()),
            "triage_harmful_endpoint_pair_count": int(
                (triage & harmful_endpoint).sum()
            ),
            "triage_known_harmful_event_count": int(
                known_harmful_events[triage].sum()
            ),
            "independent_support_harmful_endpoint_pair_count": int(
                (independent & harmful_endpoint).sum()
            ),
            "independent_support_known_harmful_event_count": int(
                known_harmful_events[independent].sum()
            ),
            "track_endpoint_counters_available": False,
        }
    return report
