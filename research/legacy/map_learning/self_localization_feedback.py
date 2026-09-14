"""Leakage-free L1--L4 feedback records for V6 closed-loop distillation."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch

from common.v6_contracts import (
    DESCRIPTOR_CLEAN_LABEL_SEMANTICS,
    DESCRIPTOR_POSE_WEIGHT_SEMANTICS,
    FEEDBACK_SCHEMA,
    FEEDBACK_VERSION,
    require_exact_identity_positive_contract,
    validate_ordered_query_registry,
)
from topology.layered_sufficiency import DEFAULT_VISIBILITY_GRID


FAILURE_LAYERS = ("L1", "L2", "L3", "L4")


def active_failure_layers(
    *,
    visible_rank: int,
    detectable_rank: int,
    matching_rank: int,
    required_rank: int,
    pose_information_sufficient: bool,
    pose_success: bool,
    required_visibility_rank: int | None = None,
    required_detectable_rank: int | None = None,
) -> tuple[str, ...]:
    """Return all active deficits while retaining hierarchical diagnostics."""

    layers = []
    visibility_target = (
        int(required_rank)
        if required_visibility_rank is None
        else int(required_visibility_rank)
    )
    detectable_target = (
        int(required_rank)
        if required_detectable_rank is None
        else int(required_detectable_rank)
    )
    if int(visible_rank) < visibility_target:
        layers.append("L1")
    if int(detectable_rank) < detectable_target:
        layers.append("L2")
    if int(matching_rank) < int(required_rank):
        layers.append("L3")
    if not bool(pose_information_sufficient) or not bool(pose_success):
        layers.append("L4")
    return tuple(layers)


def classify_failure_layer(
    *,
    visible_rank: int,
    detectable_rank: int,
    matching_rank: int,
    required_rank: int,
    pose_information_sufficient: bool,
    pose_success: bool,
) -> str | None:
    if int(visible_rank) < int(required_rank):
        return "L1"
    if int(detectable_rank) < int(required_rank):
        return "L2"
    if int(matching_rank) < int(required_rank):
        return "L3"
    if not bool(pose_information_sufficient) or not bool(pose_success):
        return "L4"
    return None


def build_self_localization_feedback(
    *,
    query_names: Sequence[str],
    records: Sequence[dict],
    required_rank: int,
    source_map_sha256: str,
    query_cache_sha256: str,
    scene_calibration_sha256: str,
    feedback_calibration_binding_sha256: str,
    positive_identity_contract: dict,
    required_visibility_rank: int | None = None,
    required_detectable_rank: int | None = None,
    pose_logdet_target: float = 0.0,
    pose_min_eigenvalue_target: float = 0.0,
    pose_information_regularization: float = 1e-9,
) -> dict:
    names = validate_ordered_query_registry(query_names)
    require_exact_identity_positive_contract(positive_identity_contract)
    if not math.isfinite(float(pose_logdet_target)):
        raise ValueError("pose logdet target must be finite")
    if (
        not math.isfinite(float(pose_min_eigenvalue_target))
        or float(pose_min_eigenvalue_target) < 0.0
    ):
        raise ValueError(
            "pose minimum-eigenvalue target must be finite and non-negative"
        )
    if (
        not math.isfinite(float(pose_information_regularization))
        or float(pose_information_regularization) <= 0.0
    ):
        raise ValueError("pose information regularization must be finite and positive")
    if len(records) != len(names):
        raise ValueError("feedback records do not align with mapping queries")
    normalized = []
    counts = {layer: 0 for layer in FAILURE_LAYERS}
    success_count = 0
    for query_index, (name, source) in enumerate(zip(names, records)):
        if str(source.get("image_name")) != name:
            raise ValueError("feedback record registry differs")
        pose_rank = int(source.get("pose_information_rank", -1))
        pose_logdet = float(source.get("pose_information_logdet", float("nan")))
        pose_min_eigenvalue = float(
            source.get("pose_information_min_eigenvalue", float("nan"))
        )
        record_logdet_target = float(
            source.get("pose_logdet_target", float("nan"))
        )
        record_min_eigenvalue_target = float(
            source.get("pose_min_eigenvalue_target", float("nan"))
        )
        record_regularization = float(
            source.get("pose_information_regularization", float("nan"))
        )
        if (
            not 0 <= pose_rank <= 6
            or not math.isfinite(pose_logdet)
            or not math.isfinite(pose_min_eigenvalue)
            or pose_min_eigenvalue < 0.0
        ):
            raise ValueError("feedback pose-information diagnostics are invalid")
        if (
            record_logdet_target != float(pose_logdet_target)
            or record_min_eigenvalue_target != float(pose_min_eigenvalue_target)
            or record_regularization != float(pose_information_regularization)
        ):
            raise ValueError("feedback pose-information threshold contract differs")
        pose_sufficient = source.get("pose_information_sufficient")
        expected_pose_sufficient = bool(
            pose_rank >= 6
            and pose_logdet >= float(pose_logdet_target)
            and pose_min_eigenvalue >= float(pose_min_eigenvalue_target)
        )
        if not isinstance(pose_sufficient, bool) or (
            pose_sufficient != expected_pose_sufficient
        ):
            raise ValueError("feedback pose-information sufficiency label differs")
        layers = active_failure_layers(
            visible_rank=int(source["visible_rank"]),
            detectable_rank=int(source["detectable_rank"]),
            matching_rank=int(source["matching_rank"]),
            required_rank=int(required_rank),
            pose_information_sufficient=pose_sufficient,
            pose_success=bool(source["pose_success"]),
            required_visibility_rank=required_visibility_rank,
            required_detectable_rank=required_detectable_rank,
        )
        layer = layers[0] if layers else None
        for active in layers:
            counts[active] += 1
        if not layers:
            success_count += 1
        required = max(int(required_rank), 1)
        visibility_required = max(
            required
            if required_visibility_rank is None
            else int(required_visibility_rank),
            1,
        )
        detectable_required = max(
            required
            if required_detectable_rank is None
            else int(required_detectable_rank),
            1,
        )
        query_row_values = torch.as_tensor(
            source.get("query_rows", ()), dtype=torch.long
        ).reshape(-1)
        winner_anchor_ids = torch.as_tensor(
            source.get("winner_anchor_ids", ()), dtype=torch.long
        ).reshape(-1)
        winner_scores = torch.as_tensor(
            source.get("winner_scores", ()), dtype=torch.float32
        ).reshape(-1)
        winner_masks = {
            key: torch.as_tensor(source.get(key, ()), dtype=torch.bool).reshape(-1)
            for key in (
                "top1_exact_identity_correct_mask",
                "top1_geometry_compatible_ambiguous_mask",
                "top1_identity_projective_incompatible_mask",
                "top1_negative_mask",
            )
        }
        aligned = [
            query_row_values,
            winner_anchor_ids,
            winner_scores,
            *winner_masks.values(),
        ]
        if any(value.numel() != query_row_values.numel() for value in aligned):
            raise ValueError("feedback winner identity rows are not aligned")
        if query_row_values.numel():
            partition_count = sum(mask.long() for mask in winner_masks.values())
            if not bool((partition_count == 1).all()):
                raise ValueError("feedback winner identity labels are not a partition")
        descriptor_triplets = torch.as_tensor(
            source.get("descriptor_triplets", ()), dtype=torch.long
        ).reshape(-1, 4)
        descriptor_triplet_harmful = torch.as_tensor(
            source.get("descriptor_triplet_harmful_inlier_mask", ()),
            dtype=torch.bool,
        ).reshape(-1)
        if descriptor_triplet_harmful.numel() != descriptor_triplets.shape[0]:
            raise ValueError("descriptor triplet harmful-inlier rows are not aligned")
        descriptor_triplet_pose_weights = torch.as_tensor(
            source.get("descriptor_triplet_pose_weights", ()),
            dtype=torch.float32,
        ).reshape(-1)
        if descriptor_triplet_pose_weights.numel() != descriptor_triplets.shape[0]:
            raise ValueError("descriptor triplet pose-weight rows are not aligned")
        if not bool(torch.isfinite(descriptor_triplet_pose_weights).all()) or bool(
            ((descriptor_triplet_pose_weights < 0.0)
             | (descriptor_triplet_pose_weights > 1.0)).any()
        ):
            raise ValueError("descriptor triplet pose weights must be finite in [0,1]")
        descriptor_triplet_legal_pair_clean = torch.as_tensor(
            source.get("descriptor_triplet_legal_pair_clean_mask", ()),
            dtype=torch.bool,
        ).reshape(-1)
        if descriptor_triplet_legal_pair_clean.numel() != descriptor_triplets.shape[0]:
            raise ValueError("descriptor triplet legal-pair rows are not aligned")
        if not torch.equal(
            descriptor_triplet_legal_pair_clean,
            descriptor_triplets[:, 3].bool(),
        ):
            raise ValueError("descriptor triplet clean labels differ from legal pairs")
        descriptor_identity_supervision_available = source.get(
            "descriptor_identity_supervision_available"
        )
        if not isinstance(descriptor_identity_supervision_available, bool):
            raise ValueError("descriptor identity supervision status is required")
        affected_anchor_policy = source.get("affected_anchor_policy")
        if affected_anchor_policy is None:
            raise ValueError("affected-Anchor policy is required")
        if not isinstance(affected_anchor_policy, str) or (
            affected_anchor_policy
            not in {"fixed_map", "descriptor_only", "rebuild", "purge"}
        ):
            raise ValueError("affected-Anchor policy is invalid")
        if descriptor_identity_supervision_available != (
            affected_anchor_policy != "purge"
        ):
            raise ValueError(
                "descriptor identity supervision is unavailable only for purge"
            )
        feedback_observer_mode = str(source.get("feedback_observer_mode", "full_loo"))
        expected_policy = {
            "fixed_map": "fixed_map",
            "descriptor_leave_self_out": "descriptor_only",
            "full_loo": affected_anchor_policy,
        }.get(feedback_observer_mode)
        if expected_policy is None or expected_policy != affected_anchor_policy:
            raise ValueError("feedback observer mode and Anchor policy differ")
        if (
            not descriptor_identity_supervision_available
            and descriptor_triplets.numel()
        ):
            raise ValueError(
                "diagnostic purge feedback cannot contain descriptor triplets"
            )
        identity_positive_count = int(source.get("identity_positive_count", -1))
        identity_active_count = int(source.get("identity_active_count", -1))
        identity_lineage_count = int(source.get("identity_lineage_count", -1))
        geometry_ambiguous_count = int(source.get("geometry_ambiguous_count", -1))
        certified_pose_valid_count = int(
            source.get("certified_pose_valid_alternative_count", -1)
        )
        if (
            min(
                identity_positive_count,
                identity_active_count,
                identity_lineage_count,
                geometry_ambiguous_count,
                certified_pose_valid_count,
            )
            < 0
        ):
            raise ValueError("feedback exact identity counts are required")
        exact_identity_pairs = torch.as_tensor(
            source.get("exact_identity_pairs", ()), dtype=torch.long
        ).reshape(-1, 2)
        active_identity_pairs = torch.as_tensor(
            source.get("active_identity_pairs", ()), dtype=torch.long
        ).reshape(-1, 2)
        exact_identity_positive_pairs = torch.as_tensor(
            source.get("exact_identity_positive_pairs", ()), dtype=torch.long
        ).reshape(-1, 2)
        certified_pose_valid_pairs = torch.as_tensor(
            source.get("certified_pose_valid_alternative_pairs", ()),
            dtype=torch.long,
        ).reshape(-1, 2)
        if (
            exact_identity_pairs.shape[0] != identity_lineage_count
            or active_identity_pairs.shape[0] != identity_active_count
            or exact_identity_positive_pairs.shape[0] != identity_positive_count
            or certified_pose_valid_pairs.shape[0] != certified_pose_valid_count
        ):
            raise ValueError("feedback exact identity pairs and counts differ")
        pose_anchor_ids = torch.as_tensor(
            source.get("clean_inlier_pose_anchor_ids", ()), dtype=torch.long
        ).reshape(-1)
        pose_query_rows = torch.as_tensor(
            source.get("clean_inlier_pose_query_rows", ()), dtype=torch.long
        ).reshape(-1)
        pose_reprojection_errors = torch.as_tensor(
            source.get("clean_inlier_pose_reprojection_errors_px", ()),
            dtype=torch.float64,
        ).reshape(-1)
        pose_information = torch.as_tensor(
            source.get("clean_inlier_pose_information", ()), dtype=torch.float64
        ).reshape(-1, 6, 6)
        if (
            not (
                pose_anchor_ids.shape
                == pose_query_rows.shape
                == pose_reprojection_errors.shape
            )
            or pose_information.shape[0] != pose_anchor_ids.numel()
        ):
            raise ValueError("Anchor-unique pose information rows are not aligned")
        if pose_anchor_ids.numel() != torch.unique(pose_anchor_ids).numel():
            raise ValueError("pose information contains duplicate Anchors")
        if source.get("pose_information_anchor_unique") is not True:
            raise ValueError("pose information must certify Anchor uniqueness")
        estimated_pose_value = source.get("estimated_pose_w2c")
        if estimated_pose_value is None:
            raise ValueError("estimated pose is required")
        estimated_pose_w2c = torch.as_tensor(estimated_pose_value, dtype=torch.float64)
        if estimated_pose_w2c.shape != (4, 4) or not bool(
            torch.isfinite(estimated_pose_w2c).all()
        ):
            raise ValueError("estimated pose must be a finite 4x4 matrix")
        visible_anchor_ids = torch.as_tensor(
            source.get("visible_anchor_ids", ()), dtype=torch.long
        ).reshape(-1)
        visible_anchor_image_cells = torch.as_tensor(
            source.get("visible_anchor_image_cells", ()), dtype=torch.long
        ).reshape(-1)
        if visible_anchor_ids.shape != visible_anchor_image_cells.shape:
            raise ValueError("visible Anchor IDs and image cells do not align")
        if bool((visible_anchor_image_cells < 0).any()):
            raise ValueError("visibility image-cell IDs must be non-negative")
        normalized.append(
            {
                "query_index": query_index,
                "image_name": name,
                "failure_layer": layer,
                "failure_layers": list(layers),
                "deficits": {
                    "visibility": max(
                        visibility_required - int(source["visible_rank"]), 0
                    )
                    / visibility_required,
                    "detectability": max(
                        detectable_required - int(source["detectable_rank"]), 0
                    )
                    / detectable_required,
                    "matching": max(required - int(source["matching_rank"]), 0)
                    / required,
                    "pose": float(
                        not bool(source["pose_information_sufficient"])
                        or not bool(source["pose_success"])
                    ),
                },
                "legal_positive_exists": identity_positive_count > 0,
                "detector_accessible": bool(source["detectable_rank"] > 0),
                "visible_rank": int(source["visible_rank"]),
                "visible_anchor_count": int(
                    source.get("visible_anchor_count", source["visible_rank"])
                ),
                "detectable_rank": int(source["detectable_rank"]),
                "correct_anchor_rank": int(source["correct_anchor_rank"]),
                "matching_rank": int(source["matching_rank"]),
                "winner_anchor": int(source["winner_anchor"]),
                "best_positive_score": float(source["best_positive_score"]),
                "best_wrong_score": float(source["best_wrong_score"]),
                "positive_wrong_margin": float(
                    source["best_positive_score"] - source["best_wrong_score"]
                ),
                "clean_inlier_anchor_ids": torch.as_tensor(
                    source.get("clean_inlier_anchor_ids", ())
                ).long(),
                "harmful_inlier_anchor_ids": torch.as_tensor(
                    source.get("harmful_inlier_anchor_ids", ())
                ).long(),
                "ambiguous_inlier_anchor_ids": torch.as_tensor(
                    source.get("ambiguous_inlier_anchor_ids", ())
                ).long(),
                "query_rows": query_row_values,
                "winner_anchor_ids": winner_anchor_ids,
                "winner_scores": winner_scores,
                "winner_identity_correct_mask": winner_masks[
                    "top1_exact_identity_correct_mask"
                ],
                **winner_masks,
                "inlier_query_rows": torch.as_tensor(
                    source.get("inlier_query_rows", ())
                ).long(),
                "inlier_clean_mask": torch.as_tensor(
                    source.get("inlier_clean_mask", ())
                ).bool(),
                "visible_anchor_ids": visible_anchor_ids,
                "visible_anchor_image_cells": visible_anchor_image_cells,
                "exact_identity_pairs": exact_identity_pairs,
                "exact_identity_lineage_pairs": exact_identity_pairs,
                "active_identity_pairs": active_identity_pairs,
                "exact_identity_positive_pairs": exact_identity_positive_pairs,
                "identity_inactive_pairs": torch.as_tensor(
                    source.get("identity_inactive_pairs", ()), dtype=torch.long
                ).reshape(-1, 2),
                "identity_projective_incompatible_pairs": torch.as_tensor(
                    source.get("identity_projective_incompatible_pairs", ()),
                    dtype=torch.long,
                ).reshape(-1, 2),
                "projective_compatible_ambiguous_pairs": torch.as_tensor(
                    source.get("projective_compatible_ambiguous_pairs", ()),
                    dtype=torch.long,
                ).reshape(-1, 2),
                "certified_pose_valid_alternative_pairs": (
                    certified_pose_valid_pairs
                ),
                "pose_valid_depth_supervision_available": bool(
                    source.get("pose_valid_depth_supervision_available", False)
                ),
                "pose_valid_depth_absolute_tolerance_m": float(
                    source.get("pose_valid_depth_absolute_tolerance_m", 0.25)
                ),
                "pose_valid_depth_relative_tolerance": float(
                    source.get("pose_valid_depth_relative_tolerance", 0.05)
                ),
                "identity_positive_count": identity_positive_count,
                "identity_active_count": identity_active_count,
                "identity_lineage_count": identity_lineage_count,
                "identity_supervision_available": identity_lineage_count > 0,
                "identity_inactive_count": int(
                    source.get("identity_inactive_count", 0)
                ),
                "identity_projective_incompatible_count": int(
                    source.get("identity_projective_incompatible_count", 0)
                ),
                "geometry_ambiguous_count": geometry_ambiguous_count,
                "certified_pose_valid_alternative_count": (
                    certified_pose_valid_count
                ),
                "detectable_pairs": torch.as_tensor(
                    source.get("detectable_pairs", ()), dtype=torch.long
                ).reshape(-1, 2),
                "matching_pairs": torch.as_tensor(
                    source.get("matching_pairs", ()), dtype=torch.long
                ).reshape(-1, 2),
                "confusion_pairs": torch.as_tensor(
                    source.get("confusion_pairs", ()), dtype=torch.long
                ).reshape(-1, 3),
                "descriptor_triplets": descriptor_triplets,
                "descriptor_triplet_harmful_inlier_mask": (descriptor_triplet_harmful),
                "descriptor_triplet_pose_weights": (descriptor_triplet_pose_weights),
                "descriptor_triplet_legal_pair_clean_mask": (
                    descriptor_triplet_legal_pair_clean
                ),
                "descriptor_identity_supervision_available": (
                    descriptor_identity_supervision_available
                ),
                "excluded_query_indices": torch.as_tensor(
                    source.get("excluded_query_indices", (query_index,))
                ).long(),
                "dependency_group_ids": torch.as_tensor(
                    source.get("dependency_group_ids", ())
                ).long(),
                "pose_information_contribution": float(
                    source["pose_information_contribution"]
                ),
                "pose_information_rank": pose_rank,
                "pose_information_logdet": pose_logdet,
                "pose_information_min_eigenvalue": pose_min_eigenvalue,
                "pose_information_sufficient": pose_sufficient,
                "pose_logdet_target": float(pose_logdet_target),
                "pose_min_eigenvalue_target": float(
                    pose_min_eigenvalue_target
                ),
                "pose_information_regularization": float(
                    pose_information_regularization
                ),
                "clean_inlier_pose_anchor_ids": pose_anchor_ids,
                "clean_inlier_pose_query_rows": pose_query_rows,
                "clean_inlier_pose_reprojection_errors_px": (pose_reprojection_errors),
                "pose_information_duplicate_rows_removed": int(
                    source.get("pose_information_duplicate_rows_removed", 0)
                ),
                "pose_information_anchor_unique": bool(
                    source.get("pose_information_anchor_unique", False)
                ),
                "clean_inlier_pose_information": pose_information,
                "pose_success": bool(source["pose_success"]),
                "estimated_pose_w2c": estimated_pose_w2c,
                "te_cm": float(source["te_cm"]),
                "ae_deg": float(source["ae_deg"]),
                "query_descriptor_loo": bool(source.get("query_descriptor_loo", True)),
                "observer_descriptor_leave_self_out": bool(
                    source.get("observer_descriptor_leave_self_out", False)
                ),
                "feedback_observer_mode": feedback_observer_mode,
                "descriptor_training_query_reused": bool(
                    source.get("descriptor_training_query_reused", False)
                ),
                "descriptor_training_split_member": bool(
                    source.get("descriptor_training_split_member", False)
                ),
                "reconstruction_target_query_reused": bool(
                    source.get("reconstruction_target_query_reused", False)
                ),
                "reconstruction_training_query_reused": bool(
                    source.get("reconstruction_training_query_reused", False)
                ),
                "query_geometry_loo": bool(source["query_geometry_loo"]),
                "query_raw_geometry_observation_loo": bool(
                    source.get(
                        "query_raw_geometry_observation_loo",
                        source["query_geometry_loo"],
                    )
                ),
                "query_candidate_topology_loo": bool(
                    source.get("query_candidate_topology_loo", True)
                ),
                "pose_neighborhood_loo": bool(
                    source.get("pose_neighborhood_loo", False)
                ),
                "affected_anchor_policy": affected_anchor_policy,
                "selection_training_query_reused": bool(
                    source.get("selection_training_query_reused", False)
                ),
                "independent_mapping_validation_query": bool(
                    source.get("independent_mapping_validation_query", False)
                ),
            }
        )
    return {
        "schema": FEEDBACK_SCHEMA,
        "version": FEEDBACK_VERSION,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "query_names": list(names),
        "positive_identity_contract": dict(positive_identity_contract),
        "visibility_evidence_contract": {
            "edge_identity": "query_image_grid_cell",
            "grid_shape": list(DEFAULT_VISIBILITY_GRID),
            "raw_visible_anchor_count_is_not_visibility_rank": True,
        },
        "descriptor_triplet_pose_weight_semantics": (
            DESCRIPTOR_POSE_WEIGHT_SEMANTICS
        ),
        "descriptor_triplet_pose_weight_formula": (
            "max(0,((I[pos@gt]-I[pos@current])-"
            "(I[neg@gt]-I[neg@current]))/2) when neg_is_deployed_winner"
        ),
        "descriptor_triplet_clean_semantics": (
            DESCRIPTOR_CLEAN_LABEL_SEMANTICS
        ),
        "estimated_pose_w2c_role": ("paired_winning_pose_diagnostic_only_not_training"),
        "required_matching_rank": int(required_rank),
        "required_visibility_rank": int(
            required_rank
            if required_visibility_rank is None
            else required_visibility_rank
        ),
        "required_detectable_rank": int(
            required_rank
            if required_detectable_rank is None
            else required_detectable_rank
        ),
        "pose_logdet_target": float(pose_logdet_target),
        "pose_min_eigenvalue_target": float(pose_min_eigenvalue_target),
        "pose_information_regularization": float(pose_information_regularization),
        "pose_information_sufficiency": (
            "unregularized_rank_ge_6_and_task_scaled_regularized_"
            "logdet_and_min_eigenvalue_meet_targets"
        ),
        "records": normalized,
        "failure_layer_counts": counts,
        "failure_layer_counts_are_overlapping": True,
        "failure_query_count": len(names) - success_count,
        "multi_layer_failure_query_count": sum(
            int(len(record["failure_layers"]) > 1) for record in normalized
        ),
        "success_count": success_count,
        "identity_lineage_count": sum(
            record["identity_lineage_count"] for record in normalized
        ),
        "identity_supervision_unavailable_query_count": sum(
            int(not record["identity_supervision_available"]) for record in normalized
        ),
        "identity_positive_count": sum(
            record["identity_positive_count"] for record in normalized
        ),
        "identity_active_count": sum(
            record["identity_active_count"] for record in normalized
        ),
        "identity_inactive_count": sum(
            record["identity_inactive_count"] for record in normalized
        ),
        "identity_projective_incompatible_count": sum(
            record["identity_projective_incompatible_count"] for record in normalized
        ),
        "geometry_compatible_ambiguous_count": sum(
            record["geometry_ambiguous_count"] for record in normalized
        ),
        "certified_pose_valid_alternative_count": sum(
            record["certified_pose_valid_alternative_count"]
            for record in normalized
        ),
        "top1_exact_identity_correct_count": sum(
            int(record["top1_exact_identity_correct_mask"].sum())
            for record in normalized
        ),
        "top1_geometry_compatible_ambiguous_count": sum(
            int(record["top1_geometry_compatible_ambiguous_mask"].sum())
            for record in normalized
        ),
        "top1_identity_projective_incompatible_count": sum(
            int(record["top1_identity_projective_incompatible_mask"].sum())
            for record in normalized
        ),
        "top1_negative_count": sum(
            int(record["top1_negative_mask"].sum()) for record in normalized
        ),
        "descriptor_triplet_harmful_inlier_count": sum(
            int(record["descriptor_triplet_harmful_inlier_mask"].sum())
            for record in normalized
        ),
        "descriptor_triplet_positive_pose_weight_count": sum(
            int((record["descriptor_triplet_pose_weights"] > 0).sum())
            for record in normalized
        ),
        "descriptor_triplet_pose_weight_sum": sum(
            float(record["descriptor_triplet_pose_weights"].sum())
            for record in normalized
        ),
        "descriptor_triplet_legal_pair_clean_count": sum(
            int(record["descriptor_triplet_legal_pair_clean_mask"].sum())
            for record in normalized
        ),
        "descriptor_identity_supervision_available": all(
            record["descriptor_identity_supervision_available"] for record in normalized
        ),
        "pose_information_duplicate_rows_removed": sum(
            record["pose_information_duplicate_rows_removed"] for record in normalized
        ),
        "pose_information_anchor_unique": all(
            record["pose_information_anchor_unique"] for record in normalized
        ),
        "query_descriptor_loo_count": sum(
            int(record["query_descriptor_loo"]) for record in normalized
        ),
        "descriptor_gradient_reuse_query_count": sum(
            int(record["descriptor_training_query_reused"]) for record in normalized
        ),
        "descriptor_training_split_query_count": sum(
            int(record["descriptor_training_split_member"]) for record in normalized
        ),
        "reconstruction_target_reuse_query_count": sum(
            int(record["reconstruction_target_query_reused"]) for record in normalized
        ),
        "reconstruction_training_reuse_query_count": sum(
            int(record["reconstruction_training_query_reused"])
            for record in normalized
        ),
        "selection_training_reuse_query_count": sum(
            int(record["selection_training_query_reused"]) for record in normalized
        ),
        "query_raw_geometry_observation_loo_count": sum(
            int(record["query_raw_geometry_observation_loo"]) for record in normalized
        ),
        "query_candidate_topology_loo_count": sum(
            int(record["query_candidate_topology_loo"]) for record in normalized
        ),
        "independent_mapping_validation_query_count": sum(
            int(record["independent_mapping_validation_query"]) for record in normalized
        ),
        "affected_anchor_policies": sorted(
            {record["affected_anchor_policy"] for record in normalized}
        ),
        "feedback_observer_modes": sorted(
            {record["feedback_observer_mode"] for record in normalized}
        ),
        "input_sha256": {
            "map": str(source_map_sha256),
            "query_cache": str(query_cache_sha256),
            "scene_calibration": str(scene_calibration_sha256),
            "feedback_calibration_binding": str(
                feedback_calibration_binding_sha256
            ),
        },
        "deployment_protocol": (
            "query_local_affected_anchor_"
            + "_or_".join(
                sorted({record["affected_anchor_policy"] for record in normalized})
            )
            + "_global_top1_one_standard_poselib"
        ),
    }
