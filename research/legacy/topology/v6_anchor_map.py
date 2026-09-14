"""Materialize V6 projective candidates into the unchanged online map API."""

from __future__ import annotations

import torch

from common.v6_contracts import ANCHOR_CANDIDATE_SCHEMA, require_schema
from map_learning.metric import SharedLowRankMetric


def merge_projective_candidates(parts: list[dict]) -> dict:
    """Concatenate independently constructed candidate sets without reselecting."""

    if not parts:
        raise ValueError("at least one candidate set is required")
    for index, part in enumerate(parts):
        require_schema(part, ANCHOR_CANDIDATE_SCHEMA, label=f"candidate[{index}]")
    names = list(parts[0]["query_names"])
    bins = torch.as_tensor(parts[0]["query_bins"]).long()
    for part in parts[1:]:
        if list(part["query_names"]) != names or not torch.equal(
            torch.as_tensor(part["query_bins"]).long(), bins
        ):
            raise ValueError("candidate mapping registries differ")
    feature_dims = {int(torch.as_tensor(part["anchor_features"]).shape[1]) for part in parts}
    if len(feature_dims) != 1:
        raise ValueError("candidate descriptor dimensions differ")
    offsets = [0]
    query = []
    keypoint = []
    for part in parts:
        csr = part["projective_anchor_observations"]
        local_offsets = torch.as_tensor(csr["observation_offsets"]).long()
        if local_offsets.ndim != 1 or int(local_offsets[0]) != 0:
            raise ValueError("candidate CSR offsets are invalid")
        if int(local_offsets[-1]) != len(torch.as_tensor(csr["query_indices"])):
            raise ValueError("candidate CSR columns are not aligned")
        base = offsets[-1]
        offsets.extend((local_offsets[1:] + base).tolist())
        query.append(torch.as_tensor(csr["query_indices"]).long())
        keypoint.append(torch.as_tensor(csr["keypoint_indices"]).long())
    count = sum(int(torch.as_tensor(part["anchor_xyz"]).shape[0]) for part in parts)
    kind = []
    for part in parts:
        value = str(part.get("candidate_kind", "projective_track"))
        kind.extend([value] * int(torch.as_tensor(part["anchor_xyz"]).shape[0]))
    return {
        "schema": ANCHOR_CANDIDATE_SCHEMA,
        "version": 2,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "query_names": names,
        "query_bins": bins,
        "anchor_ids": torch.arange(count, dtype=torch.long),
        "anchor_xyz": torch.cat([torch.as_tensor(part["anchor_xyz"]).float() for part in parts]),
        "anchor_features": torch.cat(
            [torch.as_tensor(part["anchor_features"]).float() for part in parts]
        ),
        "anchor_observation_features": torch.cat(
            [
                torch.as_tensor(
                    part.get("anchor_observation_features", part["anchor_features"])
                ).float()
                for part in parts
            ]
        ),
        "anchor_descriptor_residual": torch.cat(
            [
                torch.as_tensor(
                    part.get(
                        "anchor_descriptor_residual",
                        torch.zeros_like(torch.as_tensor(part["anchor_features"])),
                    )
                ).float()
                for part in parts
            ]
        ),
        "anchor_position_covariance": torch.cat(
            [torch.as_tensor(part["anchor_position_covariance"]).float() for part in parts]
        ),
        "identity_reliability": torch.cat(
            [torch.as_tensor(part["identity_reliability"]).float() for part in parts]
        ),
        "geometry_reliability": torch.cat(
            [torch.as_tensor(part["geometry_reliability"]).float() for part in parts]
        ),
        "candidate_kind": kind,
        "projective_anchor_observations": {
            "observation_offsets": torch.tensor(offsets, dtype=torch.long),
            "query_indices": torch.cat(query),
            "keypoint_indices": torch.cat(keypoint),
        },
        "contract": {
            "final_xyz_source": "fixed_camera_robust_ray_triangulation",
            "gaussian_depth_used_for_final_xyz": False,
            "direct_gaussian_surface_anchor": False,
        },
    }


def materialize_projective_anchor_map(candidates: dict, *, lineage: dict) -> dict:
    require_schema(candidates, ANCHOR_CANDIDATE_SCHEMA, label="candidates")
    xyz = torch.as_tensor(candidates["anchor_xyz"]).float()
    features = torch.as_tensor(candidates["anchor_features"]).float()
    count = int(xyz.shape[0])
    if count == 0 or features.ndim != 2 or features.shape[0] != count:
        raise ValueError("candidate rows are empty or misaligned")
    if not torch.isfinite(xyz).all() or not torch.isfinite(features).all():
        raise ValueError("candidate geometry/descriptors must be finite")
    anchor_ids = torch.arange(count, dtype=torch.long)
    kinds = list(candidates.get("candidate_kind", ["projective_track"] * count))
    if len(kinds) != count:
        raise ValueError("candidate kinds do not align")
    csr = candidates["projective_anchor_observations"]
    return {
        "schema": "lafgs_materialized_anchor_map",
        "version": 1,
        "anchor_ids": anchor_ids,
        "anchor_xyz": xyz,
        "anchor_features": features,
        "anchor_observation_features": torch.as_tensor(
            candidates.get("anchor_observation_features", features)
        ).float(),
        "anchor_descriptor_residual": torch.as_tensor(
            candidates.get("anchor_descriptor_residual", torch.zeros_like(features))
        ).float(),
        "source_primitive_ids": torch.full((count,), -1, dtype=torch.long),
        "track_cluster_ids": torch.arange(count, dtype=torch.long),
        "anchor_type": torch.ones(count, dtype=torch.long),
        "dependency_group_ids": torch.arange(count, dtype=torch.long),
        "coarse_dependency_group_ids": torch.arange(count, dtype=torch.long),
        "fine_identity_ids": torch.arange(count, dtype=torch.long),
        "anchor_parent_identity_ids": torch.arange(count, dtype=torch.long),
        "anchor_correlation_group_ids": torch.arange(count, dtype=torch.long),
        "anchor_position_covariance": torch.as_tensor(
            candidates["anchor_position_covariance"]
        ).float(),
        "anchor_matchability": (
            torch.as_tensor(candidates["identity_reliability"]).float()
            * torch.as_tensor(candidates["geometry_reliability"]).float()
        ).clamp(0, 1),
        "anchor_candidate_kind": kinds,
        "base_anchor_count": 0,
        "canonical_anchor_count": count,
        "micro_anchor_count": count,
        "projective_anchor_observations": {
            "schema": "lafgs_projective_anchor_observations",
            "version": 1,
            "observation_offsets": torch.as_tensor(csr["observation_offsets"]).long(),
            "query_indices": torch.as_tensor(csr["query_indices"]).long(),
            "keypoint_indices": torch.as_tensor(csr["keypoint_indices"]).long(),
        },
        "projective_anchor_construction": {
            "schema": "lafgs_v6_closed_loop_projective_anchor_construction",
            "version": 1,
            "final_xyz_source": "fixed_camera_robust_ray_triangulation",
            "gaussian_depth_role": "proposal_and_visibility_only",
            "direct_gaussian_surface_anchor": False,
            "posthoc_support_repair": False,
            "parent_child_semantics": False,
        },
        "v6_mapping_query_names": list(candidates["query_names"]),
        "v6_mapping_query_bins": torch.as_tensor(candidates["query_bins"]).long(),
        "provenance": {
            **dict(lineage),
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
            "uses_gaussian_geometry_for_triangulation": False,
            "mapping_source": "gaussian_render_valid_projective_v6",
        },
    }


def compact_projective_deployment_map(state: dict) -> dict:
    """Bake descriptors and remove training-only dense tensors.

    The returned map preserves the online ``anchor_features`` API and query
    registry, but cannot be used for another descriptor-training round or for
    exact affected-Anchor rebuild.  The full proposal checkpoint remains the
    auditable training artifact.
    """

    if state.get("schema") != "lafgs_materialized_anchor_map":
        raise ValueError("compact deployment export requires a V6 anchor map")
    output = dict(state)
    removed = []
    for field in ("anchor_observation_features", "anchor_descriptor_residual"):
        if field in output:
            output.pop(field)
            removed.append(field)
    report = output.get("v6_descriptor_distillation")
    if isinstance(report, dict):
        report = dict(report)
        for rows_field, count_field in (
            ("updated_anchor_rows", "updated_anchor_count"),
            ("round_updated_anchor_rows", "round_updated_anchor_count"),
        ):
            updated = report.pop(rows_field, None)
            if updated is not None:
                report.setdefault(
                    count_field, int(torch.as_tensor(updated).numel())
                )
        report["training_state_available"] = False
        report["deployed_features_baked"] = True
        output["v6_descriptor_distillation"] = report
    selection = output.get("v6_selection_distillation")
    if isinstance(selection, dict):
        selection = dict(selection)
        selected = selection.pop("selected_source_rows", None)
        if selected is not None:
            selection["selected_anchor_count"] = int(
                torch.as_tensor(selected).numel()
            )
        selection.pop("report", None)
        selection["training_diagnostics_available"] = False
        output["v6_selection_distillation"] = selection
    output["provenance"] = {
        **dict(state.get("provenance", {})),
        "v6_compact_deployment_export": True,
        "v6_training_only_fields_removed": removed,
    }
    return output


def identity_metric_state(state: dict, *, map_path: str, map_sha256: str) -> dict:
    features = torch.as_tensor(state["anchor_features"]).float()
    metric = SharedLowRankMetric(
        descriptor_dim=int(features.shape[1]), rank=1, max_residual_norm=0.0
    )
    with torch.no_grad():
        for parameter in metric.parameters():
            parameter.zero_()
    return {
        "schema": "lafgs_shared_metric_state",
        "version": 1,
        "landmark_indices": torch.as_tensor(state["anchor_ids"]).long(),
        "metric_config": metric.export_config(),
        "metric_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in metric.state_dict().items()
        },
        "map_path": str(map_path),
        "map_sha256": str(map_sha256),
        "step": 0,
        "protocol": "v6_identity_shared_metric",
    }


def validate_v6_identity_metric(
    payload: dict, *, state: dict, map_path: str, map_sha256: str
) -> None:
    features = torch.as_tensor(state["anchor_features"])
    expected_config = {
        "descriptor_dim": int(features.shape[1]),
        "rank": 1,
        "max_residual_norm": 0.0,
    }
    if (
        payload.get("schema") != "lafgs_shared_metric_state"
        or payload.get("protocol") != "v6_identity_shared_metric"
        or payload.get("map_path") != str(map_path)
        or payload.get("map_sha256") != str(map_sha256)
        or payload.get("metric_config") != expected_config
        or payload.get("step") != 0
    ):
        raise ValueError("V6 metric is not the exact map-bound identity shim")
    expected_rows = torch.as_tensor(state["anchor_ids"]).long()
    if not torch.equal(torch.as_tensor(payload.get("landmark_indices")).long(), expected_rows):
        raise ValueError("V6 identity metric rows differ from the map")
    metric = SharedLowRankMetric(**expected_config)
    try:
        metric.load_state_dict(payload["metric_state_dict"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise ValueError("V6 identity metric state differs") from error
    if any(bool(torch.count_nonzero(parameter)) for parameter in metric.parameters()):
        raise ValueError("V6 formal map forbids a learned online metric")


def subset_projective_anchor_map(state: dict, selected: torch.Tensor) -> dict:
    selected = torch.as_tensor(selected).long().reshape(-1)
    count = int(torch.as_tensor(state["anchor_ids"]).numel())
    if selected.numel() == 0 or bool((selected < 0).any()) or bool((selected >= count).any()):
        raise ValueError("selected V6 Anchor rows are empty or out of range")
    if not torch.equal(selected, torch.unique(selected, sorted=True)):
        raise ValueError("selected V6 Anchor rows must be unique and sorted")
    output = dict(state)
    row_fields = (
        "anchor_xyz", "anchor_features", "anchor_observation_features",
        "anchor_descriptor_residual", "source_primitive_ids", "track_cluster_ids",
        "anchor_type", "dependency_group_ids", "coarse_dependency_group_ids",
        "fine_identity_ids", "anchor_parent_identity_ids",
        "anchor_correlation_group_ids", "anchor_position_covariance",
        "anchor_matchability", "anchor_candidate_kind",
    )
    for field in row_fields:
        value = state.get(field)
        if value is None:
            continue
        if isinstance(value, list):
            output[field] = [value[index] for index in selected.tolist()]
        else:
            output[field] = torch.as_tensor(value)[selected].clone()
    output["anchor_ids"] = torch.arange(selected.numel(), dtype=torch.long)
    for field in (
        "track_cluster_ids", "dependency_group_ids", "coarse_dependency_group_ids",
        "fine_identity_ids", "anchor_parent_identity_ids", "anchor_correlation_group_ids",
    ):
        if field in output:
            output[field] = torch.arange(selected.numel(), dtype=torch.long)
    source_csr = state["projective_anchor_observations"]
    offsets = torch.as_tensor(source_csr["observation_offsets"]).long()
    query = torch.as_tensor(source_csr["query_indices"]).long()
    keypoint = torch.as_tensor(source_csr["keypoint_indices"]).long()
    output_offsets = [0]
    output_query = []
    output_keypoint = []
    for row in selected.tolist():
        start, stop = int(offsets[row]), int(offsets[row + 1])
        output_query.append(query[start:stop])
        output_keypoint.append(keypoint[start:stop])
        output_offsets.append(output_offsets[-1] + stop - start)
    output["projective_anchor_observations"] = {
        **dict(source_csr),
        "observation_offsets": torch.tensor(output_offsets, dtype=torch.long),
        "query_indices": torch.cat(output_query),
        "keypoint_indices": torch.cat(output_keypoint),
    }
    output["canonical_anchor_count"] = int(selected.numel())
    output["micro_anchor_count"] = int(selected.numel())
    view_support = state.get("anchor_view_support")
    if isinstance(view_support, dict):
        subset_support = dict(view_support)
        for field in (
            "direction_modes",
            "direction_radius_deg",
            "mode_count",
            "minimum_distance_m",
            "maximum_distance_m",
            "observation_count",
        ):
            value = view_support.get(field)
            if value is None:
                continue
            tensor = torch.as_tensor(value)
            if tensor.ndim < 1 or tensor.shape[0] != count:
                raise ValueError("Anchor view support does not align with the map")
            subset_support[field] = tensor[selected].clone()
        output["anchor_view_support"] = subset_support
    report = state.get("v6_descriptor_distillation")
    if isinstance(report, dict):
        report = dict(report)
        old_to_new = torch.full((count,), -1, dtype=torch.long)
        old_to_new[selected] = torch.arange(selected.numel(), dtype=torch.long)
        for rows_field, count_field in (
            ("updated_anchor_rows", "updated_anchor_count"),
            ("round_updated_anchor_rows", "round_updated_anchor_count"),
        ):
            updated = torch.as_tensor(
                report.get(rows_field, ()), dtype=torch.long
            ).reshape(-1)
            if updated.numel() and (
                int(updated.min()) < 0 or int(updated.max()) >= count
            ):
                raise ValueError("descriptor updated Anchor registry is invalid")
            remapped = old_to_new[updated] if updated.numel() else updated
            remapped = remapped[remapped >= 0]
            report[rows_field] = remapped
            report[count_field] = int(remapped.numel())
        output["v6_descriptor_distillation"] = report
    return output


def projective_candidates_from_map(state: dict) -> dict:
    construction = state.get("projective_anchor_construction", {})
    if construction.get("final_xyz_source") != "fixed_camera_robust_ray_triangulation":
        raise ValueError("only a V6 pure-ray map can become projective candidates")
    return {
        "schema": ANCHOR_CANDIDATE_SCHEMA,
        "version": 2,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "query_names": list(state["v6_mapping_query_names"]),
        "query_bins": torch.as_tensor(state["v6_mapping_query_bins"]).long(),
        "anchor_xyz": torch.as_tensor(state["anchor_xyz"]).float(),
        "anchor_features": torch.as_tensor(state["anchor_features"]).float(),
        "anchor_observation_features": torch.as_tensor(
            state.get("anchor_observation_features", state["anchor_features"])
        ).float(),
        "anchor_descriptor_residual": torch.as_tensor(
            state.get(
                "anchor_descriptor_residual",
                torch.zeros_like(torch.as_tensor(state["anchor_features"])),
            )
        ).float(),
        "anchor_position_covariance": torch.as_tensor(
            state["anchor_position_covariance"]
        ).float(),
        "identity_reliability": torch.as_tensor(state["anchor_matchability"]).float(),
        "geometry_reliability": torch.ones_like(
            torch.as_tensor(state["anchor_matchability"]).float()
        ),
        "candidate_kind": list(state["anchor_candidate_kind"]),
        "projective_anchor_observations": {
            key: value
            for key, value in state["projective_anchor_observations"].items()
            if key in {"observation_offsets", "query_indices", "keypoint_indices"}
        },
    }
