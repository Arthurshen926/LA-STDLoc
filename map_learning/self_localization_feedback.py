"""Leakage-free L1--L4 feedback records for V6 closed-loop distillation."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from common.v6_contracts import FEEDBACK_SCHEMA, validate_ordered_query_registry


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
    required_visibility_rank: int | None = None,
    required_detectable_rank: int | None = None,
) -> dict:
    names = validate_ordered_query_registry(query_names)
    if len(records) != len(names):
        raise ValueError("feedback records do not align with mapping queries")
    normalized = []
    counts = {layer: 0 for layer in FAILURE_LAYERS}
    success_count = 0
    for query_index, (name, source) in enumerate(zip(names, records)):
        if str(source.get("image_name")) != name:
            raise ValueError("feedback record registry differs")
        layers = active_failure_layers(
            visible_rank=int(source["visible_rank"]),
            detectable_rank=int(source["detectable_rank"]),
            matching_rank=int(source["matching_rank"]),
            required_rank=int(required_rank),
            pose_information_sufficient=bool(source["pose_information_sufficient"]),
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
                    "matching": max(required - int(source["matching_rank"]), 0) / required,
                    "pose": float(
                        not bool(source["pose_information_sufficient"])
                        or not bool(source["pose_success"])
                    ),
                },
                "legal_positive_exists": bool(source["visible_rank"] > 0),
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
                "query_rows": torch.as_tensor(source.get("query_rows", ())).long(),
                "winner_anchor_ids": torch.as_tensor(
                    source.get("winner_anchor_ids", ())
                ).long(),
                "winner_scores": torch.as_tensor(
                    source.get("winner_scores", ())
                ).float(),
                "inlier_query_rows": torch.as_tensor(
                    source.get("inlier_query_rows", ())
                ).long(),
                "inlier_clean_mask": torch.as_tensor(
                    source.get("inlier_clean_mask", ())
                ).bool(),
                "visible_anchor_ids": torch.as_tensor(
                    source.get("visible_anchor_ids", ())
                ).long(),
                "detectable_pairs": torch.as_tensor(
                    source.get("detectable_pairs", ()), dtype=torch.long
                ).reshape(-1, 2),
                "matching_pairs": torch.as_tensor(
                    source.get("matching_pairs", ()), dtype=torch.long
                ).reshape(-1, 2),
                "confusion_pairs": torch.as_tensor(
                    source.get("confusion_pairs", ()), dtype=torch.long
                ).reshape(-1, 3),
                "descriptor_triplets": torch.as_tensor(
                    source.get("descriptor_triplets", ()), dtype=torch.long
                ).reshape(-1, 4),
                "excluded_query_indices": torch.as_tensor(
                    source.get("excluded_query_indices", (query_index,))
                ).long(),
                "dependency_group_ids": torch.as_tensor(
                    source.get("dependency_group_ids", ())
                ).long(),
                "pose_information_contribution": float(
                    source["pose_information_contribution"]
                ),
                "pose_information_rank": int(source.get("pose_information_rank", 0)),
                "pose_information_logdet": float(
                    source.get("pose_information_logdet", float("-inf"))
                ),
                "clean_inlier_pose_anchor_ids": torch.as_tensor(
                    source.get("clean_inlier_pose_anchor_ids", ())
                ).long(),
                "clean_inlier_pose_information": torch.as_tensor(
                    source.get("clean_inlier_pose_information", ()),
                    dtype=torch.float64,
                ).reshape(-1, 6, 6),
                "pose_success": bool(source["pose_success"]),
                "query_descriptor_loo": True,
                "query_geometry_loo": bool(source["query_geometry_loo"]),
                "pose_neighborhood_loo": bool(
                    source.get("pose_neighborhood_loo", False)
                ),
            }
        )
    return {
        "schema": FEEDBACK_SCHEMA,
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "query_names": list(names),
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
        "records": normalized,
        "failure_layer_counts": counts,
        "success_count": success_count,
        "input_sha256": {
            "map": str(source_map_sha256),
            "query_cache": str(query_cache_sha256),
        },
        "deployment_protocol": "query_local_loo_global_top1_one_standard_poselib",
    }
