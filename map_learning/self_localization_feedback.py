"""Leakage-free L1--L4 feedback records for V6 closed-loop distillation."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from common.v6_contracts import FEEDBACK_SCHEMA, validate_ordered_query_registry


FAILURE_LAYERS = ("L1", "L2", "L3", "L4")


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
) -> dict:
    names = validate_ordered_query_registry(query_names)
    if len(records) != len(names):
        raise ValueError("feedback records do not align with mapping queries")
    normalized = []
    counts = {layer: 0 for layer in FAILURE_LAYERS}
    for query_index, (name, source) in enumerate(zip(names, records)):
        if str(source.get("image_name")) != name:
            raise ValueError("feedback record registry differs")
        layer = classify_failure_layer(
            visible_rank=int(source["visible_rank"]),
            detectable_rank=int(source["detectable_rank"]),
            matching_rank=int(source["matching_rank"]),
            required_rank=int(required_rank),
            pose_information_sufficient=bool(source["pose_information_sufficient"]),
            pose_success=bool(source["pose_success"]),
        )
        if layer is not None:
            counts[layer] += 1
        normalized.append(
            {
                "query_index": query_index,
                "image_name": name,
                "failure_layer": layer,
                "legal_positive_exists": bool(source["visible_rank"] > 0),
                "detector_accessible": bool(source["detectable_rank"] > 0),
                "visible_rank": int(source["visible_rank"]),
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
                ).reshape(-1, 2),
                "dependency_group_ids": torch.as_tensor(
                    source.get("dependency_group_ids", ())
                ).long(),
                "pose_information_contribution": float(
                    source["pose_information_contribution"]
                ),
                "pose_success": bool(source["pose_success"]),
                "query_descriptor_loo": True,
                "query_geometry_loo": bool(source["query_geometry_loo"]),
            }
        )
    return {
        "schema": FEEDBACK_SCHEMA,
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "query_names": list(names),
        "required_matching_rank": int(required_rank),
        "records": normalized,
        "failure_layer_counts": counts,
        "success_count": len(names) - sum(counts.values()),
        "input_sha256": {
            "map": str(source_map_sha256),
            "query_cache": str(query_cache_sha256),
        },
        "deployment_protocol": "query_local_loo_global_top1_one_standard_poselib",
    }
