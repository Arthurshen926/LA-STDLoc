"""Contracts for the bounded virtual-render Track closed-loop experiment."""

from __future__ import annotations

from dataclasses import replace
import torch

from topology.anchor_construction import AnchorCandidateBatch, UnifiedAnchorConstructor
from map_learning.metric import SharedLowRankMetric, validate_map_bound_identity_metric


DRY_RUN_THRESHOLDS = {
    "selected_view_count": 8,
    "median_detector_rows": 512,
    "median_supported_detector_rows": 256,
    "raw_track_count": 50,
    "stable_broad_track_count": 20,
    "new_anchor_count": 20,
    "distinct_view_bins": 2,
}


def enforce_one_observation_per_family(
    tracks: dict, pose_family: torch.Tensor
) -> tuple[dict, dict]:
    """Keep at most one deterministic highest-confidence row per Track/family.

    This is a hard evidence-independence contract, not a selector heuristic.
    Track IDs are compacted after filtering so triangulation receives a closed
    registry and cannot accidentally count siblings as independent views.
    """

    family = torch.as_tensor(pose_family, dtype=torch.long).reshape(-1)
    track = torch.as_tensor(tracks["track_index"], dtype=torch.long).reshape(-1)
    query = torch.as_tensor(tracks["query_index"], dtype=torch.long).reshape(-1)
    keypoint = torch.as_tensor(tracks["keypoint_index"], dtype=torch.long).reshape(-1)
    confidence = torch.as_tensor(tracks["confidence"], dtype=torch.float32).reshape(-1)
    if not (track.numel() == query.numel() == keypoint.numel() == confidence.numel()):
        raise ValueError("Track observation columns must align")
    if query.numel() and (int(query.min()) < 0 or int(query.max()) >= family.numel()):
        raise ValueError("Track query row exceeds pose-family registry")
    keep = []
    for track_id in torch.unique(track, sorted=True).tolist():
        rows = torch.nonzero(track == int(track_id), as_tuple=False).reshape(-1)
        for family_id in torch.unique(family[query[rows]], sorted=True).tolist():
            candidates = rows[family[query[rows]] == int(family_id)]
            rank = torch.argsort(confidence[candidates], descending=True, stable=True)
            keep.append(int(candidates[rank[0]]))
    keep = torch.tensor(sorted(keep), dtype=torch.long)
    retained_track = track[keep]
    unique_track = torch.unique(retained_track, sorted=True)
    remap = torch.full(
        (int(track.max()) + 1 if track.numel() else 0,), -1, dtype=torch.long
    )
    if unique_track.numel():
        remap[unique_track] = torch.arange(unique_track.numel())
    level = torch.as_tensor(tracks.get("track_level", torch.ones(
        int(track.max()) + 1 if track.numel() else 0, dtype=torch.int8
    )), dtype=torch.int8)
    result = {
        "track_index": remap[retained_track],
        "query_index": query[keep],
        "keypoint_index": keypoint[keep],
        "confidence": confidence[keep],
        "track_level": level[unique_track],
    }
    _, compact_family = torch.unique(family, sorted=True, return_inverse=True)
    pair = (
        result["track_index"] * max(int(torch.unique(family).numel()), 1)
        + compact_family[result["query_index"]]
    )
    if pair.unique().numel() != pair.numel():
        raise AssertionError("family observation contract was not enforced")
    return result, {
        "input_track_count": int(torch.unique(track).numel()),
        "retained_track_count": int(unique_track.numel()),
        "input_observation_count": int(track.numel()),
        "retained_observation_count": int(keep.numel()),
        "duplicate_family_observation_count": int(track.numel() - keep.numel()),
        "maximum_observations_per_track_family": 1,
    }


def dry_run_passes(metrics: dict) -> tuple[bool, list[str]]:
    failures = []
    exact = {
        "selected_view_count": DRY_RUN_THRESHOLDS["selected_view_count"],
    }
    minima = {
        key: value for key, value in DRY_RUN_THRESHOLDS.items() if key not in exact
    }
    for key, expected in exact.items():
        if int(metrics.get(key, -1)) != int(expected):
            failures.append(f"{key}!={expected}")
    for key, minimum in minima.items():
        if float(metrics.get(key, float("-inf"))) < float(minimum):
            failures.append(f"{key}<{minimum}")
    if metrics.get("family_contract_passed") is not True:
        failures.append("family_contract_passed!=true")
    if metrics.get("gt_visible_diagnostic", "missing") is not None:
        failures.append("gt_visible_diagnostic!=null")
    return not failures, failures


def _formal_anchor_batch(formal_map: dict) -> AnchorCandidateBatch:
    count = int(torch.as_tensor(formal_map["anchor_ids"]).numel())
    observations = formal_map["projective_anchor_observations"]
    batch = AnchorCandidateBatch(
        xyz=torch.as_tensor(formal_map["anchor_xyz"]).float(),
        features=torch.as_tensor(formal_map["anchor_features"]).float(),
        source_primitive_ids=torch.as_tensor(formal_map["source_primitive_ids"]).long(),
        track_cluster_ids=torch.as_tensor(formal_map["track_cluster_ids"]).long(),
        anchor_type=torch.as_tensor(formal_map["anchor_type"]).long(),
        parent_identity_ids=torch.as_tensor(formal_map["anchor_parent_identity_ids"]).long(),
        correlation_group_ids=torch.as_tensor(formal_map["anchor_correlation_group_ids"]).long(),
        covariance=torch.zeros(count, 3, 3, dtype=torch.float32),
        matchability=torch.ones(count, dtype=torch.float32),
        surface_support_weight=torch.as_tensor(formal_map["anchor_surface_support_weight"]).float(),
        candidate_kind=torch.as_tensor(formal_map["anchor_candidate_kind"]).long(),
        observation_offsets=torch.as_tensor(observations["observation_offsets"]).long(),
        observation_query_indices=torch.as_tensor(observations["query_indices"]).long(),
        observation_keypoint_indices=torch.as_tensor(observations["keypoint_indices"]).long(),
    )
    batch.validate()
    return batch


def augment_formal_anchor_map(
    formal_map: dict,
    virtual_batch: AnchorCandidateBatch,
    *,
    formal_query_count: int,
    virtual_registry: dict,
    lineage: dict,
) -> dict:
    """Append virtual Track Anchors while preserving the formal map prefix."""
    formal = _formal_anchor_batch(formal_map)
    virtual_batch.validate()
    if int(formal.features.shape[1]) != int(virtual_batch.features.shape[1]):
        raise ValueError("formal and virtual Anchor descriptor dimensions differ")
    existing_identity = torch.cat((
        formal.track_cluster_ids[formal.track_cluster_ids >= 0],
        formal.parent_identity_ids[formal.parent_identity_ids >= 0],
        formal.correlation_group_ids[formal.correlation_group_ids >= 0],
        torch.as_tensor(formal_map["fine_identity_ids"]).long().clamp_min(0),
    ))
    identity_offset = int(existing_identity.max()) + 1 if existing_identity.numel() else 0
    new_count = int(virtual_batch.xyz.shape[0])
    new_track = torch.arange(identity_offset, identity_offset + new_count, dtype=torch.long)
    new_identity = torch.arange(
        identity_offset, identity_offset + new_count, dtype=torch.long
    )
    virtual = replace(
        virtual_batch,
        track_cluster_ids=new_track,
        parent_identity_ids=new_identity,
        correlation_group_ids=new_identity.clone(),
        observation_query_indices=(
            virtual_batch.observation_query_indices + int(formal_query_count)
        ),
    )
    combined = UnifiedAnchorConstructor.materialize([
        type("_Provider", (), {"materialize": lambda self: formal})(),
        type("_Provider", (), {"materialize": lambda self: virtual})(),
    ])
    state = dict(formal_map)
    total = int(combined.xyz.shape[0])
    formal_count = int(formal.xyz.shape[0])
    state.update({
        "schema": "lafgs_materialized_anchor_map",
        "anchor_ids": torch.arange(total, dtype=torch.long),
        "anchor_xyz": combined.xyz,
        "anchor_features": combined.features,
        "source_primitive_ids": combined.source_primitive_ids,
        "track_cluster_ids": combined.track_cluster_ids,
        "anchor_type": combined.anchor_type,
        "dependency_group_ids": torch.cat((
            torch.as_tensor(formal_map["dependency_group_ids"]).long(), new_identity
        )),
        "coarse_dependency_group_ids": torch.cat((
            torch.as_tensor(formal_map["coarse_dependency_group_ids"]).long(), new_identity
        )),
        "fine_identity_ids": torch.cat((
            torch.as_tensor(formal_map["fine_identity_ids"]).long(), new_track
        )),
        "source_dependency_group_ids": torch.cat((
            torch.as_tensor(formal_map["source_dependency_group_ids"]).long(),
            torch.full((new_count,), -1, dtype=torch.long),
        )),
        "parent_source_track_ids": torch.cat((
            torch.as_tensor(formal_map["parent_source_track_ids"]).long(), new_identity
        )),
        "repair_child_index": torch.cat((
            torch.as_tensor(formal_map["repair_child_index"]).long(),
            torch.zeros(new_count, dtype=torch.long),
        )),
        "repair_parent_child_count": torch.cat((
            torch.as_tensor(formal_map["repair_parent_child_count"]).long(),
            torch.ones(new_count, dtype=torch.long),
        )),
        "canonical_anchor_count": total,
        "micro_anchor_count": int(formal_map.get("micro_anchor_count", formal_count)) + new_count,
        "requested_micro_anchor_budget": int(formal_map.get("requested_micro_anchor_budget", formal_count)) + new_count,
        "projective_anchor_construction": {
            **dict(formal_map["projective_anchor_construction"]),
            "track_anchor_count": int((combined.candidate_kind == 1).sum()),
            "surface_completion_anchor_count": int((combined.candidate_kind == 0).sum()),
            "virtual_track_augmentation_count": new_count,
            "augmentation_policy": "append_to_frozen_formal_unified_map",
        },
        "virtual_observation_registry": virtual_registry,
        "virtual_anchor_augmentation_lineage": lineage,
    })
    UnifiedAnchorConstructor.attach_to_map(state, combined)
    suffix_ids = torch.arange(formal_count, total, dtype=torch.long)
    state["virtual_anchor_selection_registry"] = {
        "schema": "lafgs_virtual_anchor_selection_registry",
        "version": 1,
        "candidate_anchor_ids": suffix_ids,
        "selected_anchor_ids": suffix_ids.clone(),
        "selected_state": torch.ones(new_count, dtype=torch.bool),
        "primary_selection_reasons": [
            "stable_broad_virtual_track_augmentation"
        ] * new_count,
        "candidate_count": new_count,
        "selected_count": new_count,
        "append_start": formal_count,
        "append_stop": total,
        "formal_parent_anchor_count": formal_count,
        "selection_uses_test_queries": False,
    }
    if not torch.equal(state["anchor_xyz"][:formal_count], formal.xyz):
        raise AssertionError("formal Anchor prefix changed during augmentation")
    return state


def validate_augmented_mapping_guard(
    augmented: dict, formal_map: dict, *, virtual_query_count: int
) -> dict:
    formal_count = int(torch.as_tensor(formal_map["anchor_ids"]).numel())
    total = int(torch.as_tensor(augmented["anchor_ids"]).numel())
    if augmented.get("schema") != "lafgs_materialized_anchor_map" or total <= formal_count:
        raise ValueError("augmented map is not a nonempty formal map extension")
    prefix_fields = (
        "anchor_xyz", "anchor_features", "source_primitive_ids",
        "track_cluster_ids", "anchor_type", "anchor_candidate_kind",
        "anchor_parent_identity_ids", "anchor_correlation_group_ids",
    )
    for field in prefix_fields:
        if not torch.equal(
            torch.as_tensor(augmented[field])[:formal_count],
            torch.as_tensor(formal_map[field]),
        ):
            raise ValueError(f"formal prefix changed: {field}")
    new_kind = torch.as_tensor(augmented["anchor_candidate_kind"])[formal_count:]
    new_source = torch.as_tensor(augmented["source_primitive_ids"])[formal_count:]
    if not bool((new_kind == 1).all()) or not bool((new_source == -1).all()):
        raise ValueError("virtual additions are not pure observation-defined Tracks")
    formal_track = torch.as_tensor(formal_map["track_cluster_ids"]).long()
    new_track = torch.as_tensor(augmented["track_cluster_ids"])[formal_count:].long()
    formal_identity = torch.as_tensor(formal_map["anchor_parent_identity_ids"]).long()
    new_identity = torch.as_tensor(augmented["anchor_parent_identity_ids"])[formal_count:].long()
    if (
        new_track.unique().numel() != new_track.numel()
        or new_identity.unique().numel() != new_identity.numel()
        or bool(torch.isin(new_track, formal_track[formal_track >= 0]).any())
        or bool(torch.isin(new_identity, formal_identity[formal_identity >= 0]).any())
    ):
        raise ValueError("virtual Track/identity namespace collides with formal map")
    registry = augmented.get("virtual_observation_registry", {})
    if int(registry.get("query_count", -1)) != int(virtual_query_count):
        raise ValueError("virtual observation registry count differs")
    selection = augmented.get("virtual_anchor_selection_registry", {})
    expected_suffix = torch.arange(formal_count, total, dtype=torch.long)
    if (
        selection.get("schema") != "lafgs_virtual_anchor_selection_registry"
        or int(selection.get("candidate_count", -1)) != total - formal_count
        or int(selection.get("selected_count", -1)) != total - formal_count
        or int(selection.get("append_start", -1)) != formal_count
        or int(selection.get("append_stop", -1)) != total
        or selection.get("selection_uses_test_queries") is not False
        or not torch.equal(
            torch.as_tensor(selection.get("candidate_anchor_ids")).long(),
            expected_suffix,
        )
        or not torch.equal(
            torch.as_tensor(selection.get("selected_anchor_ids")).long(),
            expected_suffix,
        )
        or not bool(torch.as_tensor(selection.get("selected_state")).bool().all())
        or selection.get("primary_selection_reasons") != [
            "stable_broad_virtual_track_augmentation"
        ] * (total - formal_count)
    ):
        raise ValueError("virtual suffix selection registry is incomplete")
    observations = augmented["projective_anchor_observations"]
    offsets = torch.as_tensor(observations["observation_offsets"]).long()
    query = torch.as_tensor(observations["query_indices"]).long()
    new_start = int(offsets[formal_count])
    new_query = query[new_start:]
    formal_query_count = int(registry["formal_query_count"])
    if new_query.numel() and (
        int(new_query.min()) < formal_query_count
        or int(new_query.max()) >= formal_query_count + int(virtual_query_count)
    ):
        raise ValueError("virtual observations escape the combined query registry")
    return {
        "formal_prefix_preserved": True,
        "formal_anchor_count": formal_count,
        "virtual_anchor_count": total - formal_count,
        "augmented_anchor_count": total,
        "virtual_observation_count": int(new_query.numel()),
        "identity_namespace_disjoint": True,
        "selection_registry_complete": True,
        "selection_reason": "stable_broad_virtual_track_augmentation",
        "mapping_only": True,
        "gt_visible_diagnostic": None,
    }


def build_map_bound_identity_metric(
    *, map_path: str, map_sha256: str, anchor_ids: torch.Tensor,
    descriptor_dim: int, producer: dict,
) -> dict:
    """Create the strict no-learn metric shim for an exact augmented map."""
    anchor_ids = torch.as_tensor(anchor_ids).long()
    expected = torch.arange(anchor_ids.numel(), dtype=torch.long)
    if not torch.equal(anchor_ids, expected):
        raise ValueError("identity metric requires canonical contiguous anchor IDs")
    metric = SharedLowRankMetric(
        descriptor_dim=int(descriptor_dim), rank=1, max_residual_norm=0.0
    )
    with torch.no_grad():
        for parameter in metric.parameters():
            parameter.zero_()
    payload = {
        "schema": "lafgs_shared_metric_state",
        "version": 1,
        "landmark_indices": anchor_ids.clone(),
        "metric_config": metric.export_config(),
        "metric_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in metric.state_dict().items()
        },
        "map_path": str(map_path),
        "map_sha256": str(map_sha256),
        "step": 0,
        "protocol": "rendered_track_map_bound_identity",
        "producer": {
            **dict(producer),
            "learned_descriptor_transform": False,
            "uses_test_queries": False,
            "virtual_suffix_training_steps": 0,
        },
    }
    validate_map_bound_identity_metric(
        payload, descriptor_dim=int(descriptor_dim),
        anchor_count=int(anchor_ids.numel()), map_path=str(map_path),
        map_sha256=str(map_sha256),
    )
    return payload
