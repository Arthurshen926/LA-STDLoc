#!/usr/bin/env python3
"""Build one frozen scene graph for cross-scene joint assignment training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localization_training.exact_counterfactual_pose_teacher import (
    ExactCounterfactualConfig,
    geometry_diversity_score,
    improves_lexicographically,
    solve_counterfactual_pose,
)
from localization_training.joint_assignment_training import (
    ambiguous_mask_from_teacher_record,
    beta_smoothed_anchor_features,
    positive_mask_from_teacher_record,
    raw_tensor_sha256,
    mask_stale_dynamic_harmful_evidence,
    select_balanced_training_rows,
    select_counterfactual_replacement_rows,
    temporal_view_bin_support_from_positive_teacher,
    trajectory_support_from_positive_teacher,
)
from localization_training.full_primitive_retrieval import (
    chunked_exact_topk_preserve_top1,
)
from localization_training.local_assignment import (
    JOINT_ASSIGNMENT_V1_FEATURE_NAMES,
    build_one_of_k_features,
)
from localization_training.shared_metric import SharedLowRankMetric
from scripts.train_one_of_k_reranker import load_assignment_map_state


def _baseline_outcome(record, xyz, dependency, source, config):
    te = float(record["te_cm"])
    re = float(record["re_deg"])
    inliers = torch.as_tensor(record["ransac_inlier_mask"]).bool()
    harmful = torch.as_tensor(record["harmful_inlier_mask"]).bool()
    valid = bool(inliers.sum() >= 4 and np.isfinite(te) and np.isfinite(re))
    return {
        "valid": valid,
        "translation_error_cm": te,
        "rotation_error_degrees": re,
        "correct_basin": bool(
            valid
            and te <= float(config.basin_translation_cm)
            and re <= float(config.basin_rotation_degrees)
        ),
        "strict_translation_success": bool(
            valid and te <= float(config.strict_translation_cm)
        ),
        "inlier_count": int(inliers.sum()),
        "harmful_consensus_count": int(harmful.sum()),
        "geometry_diversity": geometry_diversity_score(xyz, dependency, source),
    }


def _pose_target_weight(candidate, baseline):
    if improves_lexicographically(candidate, baseline):
        return 2.0, "improves"
    if not candidate["valid"] or (
        baseline["correct_basin"] and not candidate["correct_basin"]
    ):
        return 0.0, "basin_harmful"
    if (
        candidate["translation_error_cm"]
        <= baseline["translation_error_cm"] + 0.1
        and candidate["rotation_error_degrees"]
        <= baseline["rotation_error_degrees"] + 0.02
    ):
        return 1.0, "non_worse"
    if candidate["translation_error_cm"] > baseline["translation_error_cm"] + 2.0:
        return 0.0, "translation_harmful"
    return 0.25, "weak"


def _index_records(payload):
    names = list(payload["query_names"])
    records = list(payload["records"])
    if len(names) != len(records):
        raise ValueError("query registry and records do not align")
    return {name: record for name, record in zip(names, records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--positive-teacher", required=True)
    parser.add_argument("--dynamic-outcomes", required=True)
    parser.add_argument("--selector-state", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--topk", type=int, default=8, choices=(4, 8))
    parser.add_argument("--patch-radius", type=int, default=2)
    parser.add_argument("--patch-step-px", type=float, default=8.0)
    parser.add_argument("--maximum-rows", type=int, default=1024)
    parser.add_argument("--null-to-positive-ratio", type=float, default=2.0)
    parser.add_argument("--exact-query-budget", type=int, default=20)
    parser.add_argument("--exact-rows-per-query", type=int, default=2)
    parser.add_argument("--exact-candidates-per-row", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    paths = {
        name: Path(value).resolve()
        for name, value in {
            "map": args.map,
            "metric_state": args.metric_state,
            "query_cache": args.query_cache,
            "positive_teacher": args.positive_teacher,
            "dynamic_outcomes": args.dynamic_outcomes,
            "selector_state": args.selector_state,
        }.items()
    }
    device = torch.device("cuda")
    state = torch.load(paths["map"], map_location="cpu", weights_only=False)
    map_rows = load_assignment_map_state(state)
    anchor_ids = map_rows["identities"].cpu()
    anchor_count = len(anchor_ids)
    bank = F.normalize(map_rows["features"].float(), dim=1).to(device)
    xyz = map_rows["xyz"].float()
    source = map_rows["source_groups"].long()
    dependency = map_rows["dependency_groups"].long()

    metric_payload = torch.load(
        paths["metric_state"], map_location="cpu", weights_only=False
    )
    if not torch.equal(
        torch.as_tensor(metric_payload["landmark_indices"]).long().cpu(), anchor_ids
    ):
        raise ValueError("metric state does not align with the scene map")
    metric = SharedLowRankMetric(**metric_payload["metric_config"]).to(device)
    metric.load_state_dict(metric_payload["metric_state_dict"])
    metric.eval()
    for parameter in metric.parameters():
        parameter.requires_grad_(False)

    positive_teacher = torch.load(
        paths["positive_teacher"], map_location="cpu", weights_only=False
    )
    dynamic = torch.load(
        paths["dynamic_outcomes"], map_location="cpu", weights_only=False
    )
    if int(positive_teacher["anchor_count"]) != anchor_count or int(
        dynamic["anchor_count"]
    ) != anchor_count:
        raise ValueError("teacher artifacts do not align with the scene map")
    positive_by_name = _index_records(positive_teacher)
    dynamic_by_name = _index_records(dynamic)
    if set(positive_by_name) != set(dynamic_by_name):
        raise ValueError("positive and dynamic query registries differ")
    cache_payload = torch.load(
        paths["query_cache"], map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    names = sorted(set(positive_by_name) & set(dynamic_by_name) & set(cache))
    if len(names) != len(positive_by_name):
        raise ValueError("query cache does not cover the complete teacher registry")

    selector = torch.load(
        paths["selector_state"], map_location="cpu", weights_only=False
    )
    if int(selector["anchor_count"]) != anchor_count or selector[
        "anchor_ids_sha256"
    ] != raw_tensor_sha256(anchor_ids):
        raise ValueError("selector statistics do not align with the scene map")
    trajectory_support = trajectory_support_from_positive_teacher(
        positive_teacher, anchor_count
    )
    temporal_view_bin_support = temporal_view_bin_support_from_positive_teacher(
        positive_teacher, anchor_count, bins_per_trajectory=8
    )
    anchor_statistics = beta_smoothed_anchor_features(
        selector["anchor_statistics"],
        trajectory_support,
        temporal_view_bin_support,
    ).to(device)

    processing_names = sorted(
        names,
        key=lambda name: (
            float(dynamic_by_name[name]["te_cm"]),
            float(dynamic_by_name[name]["re_deg"]),
            name,
        ),
        reverse=True,
    )
    exact_config = ExactCounterfactualConfig(seed=args.seed)
    records = []
    totals = {
        "rows_full": 0,
        "rows_training": 0,
        "positive_rows": 0,
        "ambiguous_only_rows": 0,
        "ambiguous_only_rows_excluded": 0,
        "exact_queries": 0,
        "exact_replacements": 0,
        "exact_improves": 0,
        "exact_non_worse": 0,
        "exact_harmful": 0,
        "top1_dynamic_identity_mismatches": 0,
        "top1_dynamic_mismatch_queries": 0,
        "top1_dynamic_mismatch_max_score_delta": 0.0,
        "stale_dynamic_harmful_rows_masked": 0,
    }

    for name in tqdm(
        processing_names, desc=f"{args.scene} joint-assignment graph"
    ):
        cached = cache[name]
        positive_record = positive_by_name[name]
        dynamic_record = dynamic_by_name[name]
        rows = torch.as_tensor(positive_record["query_rows"]).long()
        dynamic_rows = torch.as_tensor(dynamic_record["query_rows"]).long()
        if not torch.equal(rows, dynamic_rows):
            raise ValueError(f"dynamic and positive rows differ for {name}")
        raw_query = F.normalize(
            torch.as_tensor(cached["native_descriptors"]).float()[rows].to(device),
            dim=1,
        )
        with torch.no_grad():
            query, _ = metric(raw_query)
            retrieval = chunked_exact_topk_preserve_top1(
                query,
                bank,
                topk=min(int(args.topk), anchor_count),
                chunk_size=8192,
            )
            top_scores, top_indices = retrieval.scores, retrieval.indices
        expected_top1 = torch.as_tensor(
            dynamic_record["top1_anchor_indices"]
        ).long().to(device)
        if expected_top1.shape != top_indices[:, 0].shape:
            raise ValueError(f"dynamic top-1 rows differ for {name}")
        dynamic_harmful = torch.as_tensor(
            dynamic_record["harmful_inlier_mask"]
        ).bool().to(device)
        harmful, identity_changed = mask_stale_dynamic_harmful_evidence(
            dynamic_harmful,
            expected_top1,
            top_indices[:, 0],
        )
        changed_count = int(identity_changed.sum())
        if changed_count:
            expected_scores = (query * bank[expected_top1]).sum(dim=1)
            score_delta = (
                top_scores[:, 0] - expected_scores
            )[identity_changed].abs().max()
            totals["top1_dynamic_identity_mismatches"] += changed_count
            totals["top1_dynamic_mismatch_queries"] += 1
            totals["top1_dynamic_mismatch_max_score_delta"] = max(
                float(totals["top1_dynamic_mismatch_max_score_delta"]),
                float(score_delta),
            )
            totals["stale_dynamic_harmful_rows_masked"] += int(
                (dynamic_harmful & identity_changed).sum()
            )
        positive = positive_mask_from_teacher_record(
            positive_record, top_indices.cpu()
        ).to(device)
        ambiguous = ambiguous_mask_from_teacher_record(
            positive_record, top_indices.cpu()
        ).to(device)
        ambiguous_only = ambiguous.any(dim=1) & ~positive.any(dim=1)
        keypoints = torch.as_tensor(cached["native_keypoints"]).float()[rows].to(device)
        keypoint_scores = torch.as_tensor(cached["native_scores"]).float()[rows].to(device)
        with torch.no_grad():
            features = build_one_of_k_features(
                torch.as_tensor(cached["feature_map"]).float().to(device),
                keypoints,
                top_indices,
                top_scores,
                bank,
                cached["native_input_hw"],
                radius=args.patch_radius,
                step_px=args.patch_step_px,
                landmark_statistics=anchor_statistics,
                context_version=1,
                keypoint_scores=keypoint_scores,
                landmark_xyz=xyz.to(device),
                source_groups=source.to(device),
                dependency_groups=dependency.to(device),
                query_metric=metric,
            )

        target_weights = positive.float()
        row_weights = torch.ones(len(rows), device=device)
        row_weights[harmful] = 2.0
        exact_evidence = []
        if totals["exact_queries"] < max(int(args.exact_query_budget), 0):
            exact_rows = select_counterfactual_replacement_rows(
                positive, harmful, top_scores, maximum_rows=args.exact_rows_per_query
            )
            if exact_rows.numel():
                totals["exact_queries"] += 1
            points2d = (
                keypoints.detach().cpu().double().numpy()
                + float(cached.get("pixel_center_offset", 0.5))
            )
            top1 = top_indices[:, 0].detach().cpu().long()
            base_xyz = xyz[top1].double().numpy()
            base_dependency = dependency[top1].numpy()
            base_source = source[top1].numpy()
            baseline = (
                solve_counterfactual_pose(
                    points2d=points2d,
                    points3d=base_xyz,
                    intrinsics=torch.as_tensor(cached["native_K"]).double().numpy(),
                    ground_truth_w2c=torch.as_tensor(
                        cached["pose_w2c"]
                    ).double().numpy(),
                    dependency_groups=base_dependency,
                    source_groups=base_source,
                    config=exact_config,
                )
                if exact_rows.numel()
                else None
            )
            for row in exact_rows.detach().cpu().tolist():
                legal = torch.nonzero(
                    positive[row], as_tuple=False
                ).reshape(-1)
                legal = legal[legal != 0][: int(args.exact_candidates_per_row)]
                for candidate_position in legal.detach().cpu().tolist():
                    candidate_anchor = int(top_indices[row, candidate_position])
                    candidate_xyz = base_xyz.copy()
                    candidate_dependency = base_dependency.copy()
                    candidate_source = base_source.copy()
                    candidate_xyz[row] = xyz[candidate_anchor].double().numpy()
                    candidate_dependency[row] = int(dependency[candidate_anchor])
                    candidate_source[row] = int(source[candidate_anchor])
                    outcome = solve_counterfactual_pose(
                        points2d=points2d,
                        points3d=candidate_xyz,
                        intrinsics=torch.as_tensor(cached["native_K"]).double().numpy(),
                        ground_truth_w2c=torch.as_tensor(
                            cached["pose_w2c"]
                        ).double().numpy(),
                        dependency_groups=candidate_dependency,
                        source_groups=candidate_source,
                        config=exact_config,
                    )
                    weight, category = _pose_target_weight(outcome, baseline)
                    target_weights[row, candidate_position] = float(weight)
                    row_weights[row] = max(float(row_weights[row]), 3.0)
                    totals["exact_replacements"] += 1
                    totals["exact_improves"] += int(category == "improves")
                    totals["exact_non_worse"] += int(category == "non_worse")
                    totals["exact_harmful"] += int("harmful" in category)
                    exact_evidence.append(
                        {
                            "row": int(row),
                            "candidate_position": int(candidate_position),
                            "candidate_anchor": candidate_anchor,
                            "target_weight": float(weight),
                            "category": category,
                            "baseline_te_cm": float(
                                baseline["translation_error_cm"]
                            ),
                            "candidate_te_cm": float(
                                outcome["translation_error_cm"]
                            ),
                            "baseline_re_deg": float(
                                baseline["rotation_error_degrees"]
                            ),
                            "candidate_re_deg": float(
                                outcome["rotation_error_degrees"]
                            ),
                        }
                    )

        selected = select_balanced_training_rows(
            positive,
            top_scores,
            harmful_inlier_mask=harmful,
            ignored_row_mask=ambiguous_only,
            maximum_rows=args.maximum_rows,
            null_to_positive_ratio=args.null_to_positive_ratio,
        )
        records.append(
            {
                "query_name": name,
                "trajectory": name.split("/", 1)[0],
                "features": features[selected].detach().cpu().half(),
                "positive_mask": positive[selected].detach().cpu(),
                "ambiguous_mask": ambiguous[selected].detach().cpu(),
                "candidate_target_weights": target_weights[selected].detach().cpu().half(),
                "row_weights": row_weights[selected].detach().cpu().half(),
                "topk_anchor_indices": top_indices[selected].detach().cpu().int(),
                "dynamic_top1_identity_changed": identity_changed[selected]
                .detach()
                .cpu(),
                "exact_evidence": exact_evidence,
            }
        )
        totals["rows_full"] += len(rows)
        totals["rows_training"] += len(selected)
        totals["positive_rows"] += int(positive.any(dim=1).sum())
        totals["ambiguous_only_rows"] += int(ambiguous_only.sum())
        totals["ambiguous_only_rows_excluded"] += int(
            ambiguous_only.sum()
        )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    feature_names = list(JOINT_ASSIGNMENT_V1_FEATURE_NAMES)
    artifact = {
        "schema": "lafgs_joint_assignment_scene_graph",
        "version": 5,
        "scene": args.scene,
        "anchor_count": anchor_count,
        "anchor_ids_sha256": raw_tensor_sha256(anchor_ids),
        "anchor_statistics": anchor_statistics.detach().cpu(),
        "feature_names": feature_names,
        "topk": int(args.topk),
        "query_names": processing_names,
        "records": records,
        "summary": totals,
        "config": {
            **vars(args),
            **{key: str(value) for key, value in paths.items()},
            "exact_counterfactual": vars(exact_config),
            "candidate_top1_reference": "deployment_chunked_exact_top1",
            "dynamic_identity_switch_policy": (
                "retain_gt_multi_positive_and_mask_stale_dynamic_harmful"
            ),
            "ambiguous_candidate_policy": (
                "exclude_loose_radius_only_rows_from_assignment_and_null_supervision"
            ),
            "exact_query_sampling": (
                "difficulty_ordered_skip_unrepairable_until_budget"
            ),
        },
    }
    torch.save(artifact, output)
    report = {
        "scene": args.scene,
        "output": str(output),
        "anchor_count": anchor_count,
        "query_count": len(records),
        "feature_dim": len(feature_names),
        "summary": totals,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
