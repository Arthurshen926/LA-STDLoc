"""Mapping-only no-delete replay for audited Anchor identity components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np
import torch
import torch.nn.functional as F

from localization.localizer import load_shared_metric
from localization.matcher import (
    Top1Matches,
    global_cosine_top1,
    suppress_duplicate_anchor_matches,
    suppress_duplicate_entity_matches,
)
from localization.pose_solver import pose_error, solve_absolute_pose


def _selected_csr_contains(
    record: Mapping,
    prefix: str,
    selected_local_rows: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.as_tensor(record[f"{prefix}_offsets"]).long().reshape(-1)
    indices = torch.as_tensor(record[f"{prefix}_indices"]).long().reshape(-1)
    rows = torch.as_tensor(selected_local_rows).long().reshape(-1)
    values = torch.as_tensor(values).long().reshape(-1)
    if rows.numel() != values.numel():
        raise ValueError(f"{prefix} selected rows and values do not align")
    if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= offsets.numel() - 1):
        raise ValueError(f"{prefix} selected row is outside the teacher CSR")
    matched = torch.zeros(rows.numel(), dtype=torch.bool)
    nonempty = torch.zeros(rows.numel(), dtype=torch.bool)
    for output_row, (local, value) in enumerate(zip(rows.tolist(), values.tolist())):
        begin, end = int(offsets[local]), int(offsets[local + 1])
        nonempty[output_row] = end > begin
        if end > begin:
            matched[output_row] = bool((indices[begin:end] == int(value)).any())
    return matched, nonempty


def _entity_ids(anchor_indices: torch.Tensor, component_ids: torch.Tensor) -> torch.Tensor:
    components = torch.as_tensor(component_ids).long()
    anchors = torch.as_tensor(anchor_indices).long()
    component_count = int(components.max()) + 1 if bool((components >= 0).any()) else 0
    selected = components[anchors]
    return torch.where(selected >= 0, selected, component_count + anchors)


def _ground_truth_errors(
    xyz: torch.Tensor,
    keypoints: torch.Tensor,
    intrinsic: torch.Tensor,
    pose_w2c: torch.Tensor,
) -> torch.Tensor:
    camera = xyz @ pose_w2c[:3, :3].T + pose_w2c[:3, 3]
    projected = camera @ intrinsic.T
    valid = camera[:, 2] > 1e-8
    uv = projected[:, :2] / projected[:, 2:].clamp_min(1e-8)
    error = torch.linalg.norm(uv - keypoints, dim=1)
    return torch.where(valid, error, torch.full_like(error, float("inf")))


def _evaluate_matches(
    matches: Top1Matches,
    *,
    teacher_local_rows: torch.Tensor,
    keypoints: torch.Tensor,
    xyz: torch.Tensor,
    intrinsic: torch.Tensor,
    ground_truth_pose: torch.Tensor,
    component_ids: torch.Tensor,
    record: Mapping,
    reprojection_error_px: float,
    clean_reprojection_px: float,
    seed: int,
) -> dict:
    selected = matches.keypoint_indices.cpu().long()
    winners = matches.anchor_indices.cpu().long()
    selected_teacher_rows = teacher_local_rows[selected]
    selected_keypoints = keypoints[selected]
    correct, has_positive = _selected_csr_contains(
        record, "positive", selected_teacher_rows, winners
    )
    ambiguous, _ = _selected_csr_contains(
        record, "ambiguous", selected_teacher_rows, winners
    )
    false = ~correct & ~ambiguous & has_positive
    estimate = solve_absolute_pose(
        selected_keypoints.numpy(),
        xyz[winners].numpy(),
        intrinsic.numpy(),
        reprojection_error_px=float(reprojection_error_px),
        confidence=0.99999,
        max_iterations=100000,
        min_iterations=1000,
        seed=int(seed),
    )
    ae_deg, te_cm = pose_error(estimate.pose_w2c, ground_truth_pose.numpy())
    inliers = torch.as_tensor(estimate.inliers).long().reshape(-1)
    gt_error = _ground_truth_errors(
        xyz[winners], selected_keypoints, intrinsic, ground_truth_pose
    )
    inlier_clean = (
        gt_error[inliers] <= float(clean_reprojection_px)
        if inliers.numel()
        else torch.empty(0, dtype=torch.bool)
    )
    entities = _entity_ids(winners, component_ids)
    return {
        "correspondence_count": int(winners.numel()),
        "unique_anchor_count": int(torch.unique(winners).numel()),
        "unique_entity_count": int(torch.unique(entities).numel()),
        "labeled_row_count": int(has_positive.sum()),
        "correct_winner_count": int(correct.sum()),
        "ambiguous_winner_count": int((~correct & ambiguous).sum()),
        "false_winner_count": int(false.sum()),
        "inlier_count": int(inliers.numel()),
        "clean_inlier_count": int(inlier_clean.sum()),
        "harmful_inlier_count": int(inliers.numel() - int(inlier_clean.sum())),
        "te_cm": float(te_cm),
        "ae_deg": float(ae_deg),
        "failed": bool(inliers.numel() < 4),
        "hypotheses": int(estimate.diagnostics.get("iterations", 0)),
    }


def _percent(numerator: int | float, denominator: int | float) -> float:
    return 100.0 * float(numerator) / max(float(denominator), 1.0)


def _variant_summary(rows: list[dict], key: str) -> dict:
    values = [row[key] for row in rows]
    te = np.asarray([value["te_cm"] for value in values], dtype=np.float64)
    ae = np.asarray([value["ae_deg"] for value in values], dtype=np.float64)
    correspondences = sum(value["correspondence_count"] for value in values)
    labeled = sum(value["labeled_row_count"] for value in values)
    correct = sum(value["correct_winner_count"] for value in values)
    inliers = sum(value["inlier_count"] for value in values)
    clean = sum(value["clean_inlier_count"] for value in values)
    harmful = sum(value["harmful_inlier_count"] for value in values)
    tail_count = max(int(math.ceil(0.05 * te.size)), 1)
    return {
        "query_count": len(values),
        "correspondence_count": int(correspondences),
        "correspondences_mean": float(np.mean([value["correspondence_count"] for value in values])),
        "unique_anchor_mean": float(np.mean([value["unique_anchor_count"] for value in values])),
        "unique_entity_mean": float(np.mean([value["unique_entity_count"] for value in values])),
        "labeled_row_count": int(labeled),
        "correct_winner_count": int(correct),
        "raw_gt_precision_percent": _percent(correct, labeled),
        "inlier_count": int(inliers),
        "clean_inlier_count": int(clean),
        "harmful_inlier_count": int(harmful),
        "inlier_gt_precision_percent": _percent(clean, inliers),
        "solver_inlier_ratio_percent": _percent(inliers, correspondences),
        "median_te_cm": float(np.median(te)),
        "mean_te_cm": float(np.mean(te)),
        "p90_te_cm": float(np.percentile(te, 90)),
        "cvar95_te_cm": float(np.sort(te)[-tail_count:].mean()),
        "median_ae_deg": float(np.median(ae)),
        "mean_ae_deg": float(np.mean(ae)),
        "p90_ae_deg": float(np.percentile(ae, 90)),
        "recall_2cm_2deg_percent": float(100.0 * np.mean((te < 2.0) & (ae < 2.0))),
        "recall_5cm_5deg_percent": float(100.0 * np.mean((te < 5.0) & (ae < 5.0))),
        "catastrophic_100cm_count": int(np.count_nonzero(te >= 100.0)),
        "failure_count": int(sum(value["failed"] for value in values)),
        "mean_hypotheses": float(np.mean([value["hypotheses"] for value in values])),
    }


def summarize_identity_folding(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("identity-folding audit needs at least one query")
    baseline = _variant_summary(rows, "baseline")
    anchor_unique = _variant_summary(rows, "anchor_unique_control")
    folded = _variant_summary(rows, "entity_folded")
    anchor_unique_te = np.asarray(
        [row["anchor_unique_control"]["te_cm"] for row in rows]
    )
    folded_te = np.asarray([row["entity_folded"]["te_cm"] for row in rows])
    tolerance = 1e-9
    paired = {
        "query_count": len(rows),
        "query_with_folded_correspondence_count": int(
            sum(row["component_correspondence_removed_count"] > 0 for row in rows)
        ),
        "correspondence_removed_count": int(
            sum(row["component_correspondence_removed_count"] for row in rows)
        ),
        "correspondence_removed_percent": _percent(
            anchor_unique["correspondence_count"] - folded["correspondence_count"],
            anchor_unique["correspondence_count"],
        ),
        "raw_gt_precision_delta_pp": float(
            folded["raw_gt_precision_percent"] - anchor_unique["raw_gt_precision_percent"]
        ),
        "inlier_gt_precision_delta_pp": float(
            folded["inlier_gt_precision_percent"] - anchor_unique["inlier_gt_precision_percent"]
        ),
        "harmful_inlier_delta": int(
            folded["harmful_inlier_count"] - anchor_unique["harmful_inlier_count"]
        ),
        "clean_inlier_delta": int(
            folded["clean_inlier_count"] - anchor_unique["clean_inlier_count"]
        ),
        "median_te_delta_cm": float(folded["median_te_cm"] - anchor_unique["median_te_cm"]),
        "mean_te_delta_cm": float(folded["mean_te_cm"] - anchor_unique["mean_te_cm"]),
        "p90_te_delta_cm": float(folded["p90_te_cm"] - anchor_unique["p90_te_cm"]),
        "cvar95_te_delta_cm": float(folded["cvar95_te_cm"] - anchor_unique["cvar95_te_cm"]),
        "te_improved_query_count": int(np.count_nonzero(folded_te < anchor_unique_te - tolerance)),
        "te_worsened_query_count": int(np.count_nonzero(folded_te > anchor_unique_te + tolerance)),
        "te_unchanged_query_count": int(np.count_nonzero(np.abs(folded_te - anchor_unique_te) <= tolerance)),
        "recall_5cm_5deg_delta_pp": float(
            folded["recall_5cm_5deg_percent"] - anchor_unique["recall_5cm_5deg_percent"]
        ),
        "catastrophic_count_delta": int(
            folded["catastrophic_100cm_count"] - anchor_unique["catastrophic_100cm_count"]
        ),
        "failure_count_delta": int(
            folded["failure_count"] - anchor_unique["failure_count"]
        ),
        "comparison": "entity_folded_minus_anchor_unique_control",
    }
    raw_control = {
        "duplicate_anchor_correspondence_removed_count": int(
            baseline["correspondence_count"] - anchor_unique["correspondence_count"]
        ),
        "duplicate_anchor_correspondence_removed_percent": _percent(
            baseline["correspondence_count"] - anchor_unique["correspondence_count"],
            baseline["correspondence_count"],
        ),
        "raw_gt_precision_delta_pp": float(
            anchor_unique["raw_gt_precision_percent"] - baseline["raw_gt_precision_percent"]
        ),
        "median_te_delta_cm": float(
            anchor_unique["median_te_cm"] - baseline["median_te_cm"]
        ),
        "comparison": "anchor_unique_control_minus_current_deployment",
    }
    harmful_improved = paired["harmful_inlier_delta"] < 0
    precision_safe = paired["raw_gt_precision_delta_pp"] >= -1e-9
    pose_safe = (
        paired["median_te_delta_cm"] <= 1e-9
        and paired["mean_te_delta_cm"] <= 1e-9
        and paired["p90_te_delta_cm"] <= 1e-9
        and paired["cvar95_te_delta_cm"] <= 1e-9
        and paired["catastrophic_count_delta"] <= 0
        and paired["recall_5cm_5deg_delta_pp"] >= -1e-9
    )
    if harmful_improved and precision_safe and pose_safe:
        routing = "go_to_evidence_transfer_prototype"
    elif (
        not precision_safe
        or paired["recall_5cm_5deg_delta_pp"] < -1e-9
        or paired["catastrophic_count_delta"] > 0
        or paired["failure_count_delta"] > 0
    ):
        routing = "stop_physical_dedup_keep_semantic_components"
    else:
        routing = "inconclusive_expand_mapping_sentinels"
    return {
        "baseline": baseline,
        "anchor_unique_control": anchor_unique,
        "entity_folded": folded,
        "paired": paired,
        "raw_duplicate_control": raw_control,
        "routing": routing,
        "routing_is_mapping_only": True,
        "physical_map_mutated": False,
    }


@torch.inference_mode()
def evaluate_identity_folding(
    *,
    state: Mapping,
    metric_state_path: str,
    teacher: Mapping,
    query_cache: Mapping,
    component_ids: torch.Tensor,
    device: torch.device,
    reprojection_error_px: float,
    clean_reprojection_px: float,
    seed: int,
    query_indices: Sequence[int] | torch.Tensor | None = None,
    deployment_row_limit: int = 0,
    progress_interval: int = 25,
) -> dict:
    anchor_ids = torch.as_tensor(state["anchor_ids"]).long()
    xyz = torch.as_tensor(state["anchor_xyz"]).float().cpu()
    bank = F.normalize(
        torch.as_tensor(state["anchor_features"]).float().to(device), dim=1
    )
    components = torch.as_tensor(component_ids).long().reshape(-1)
    if components.numel() != anchor_ids.numel():
        raise ValueError("component IDs do not align with the compact map")
    if int(teacher["anchor_count"]) != anchor_ids.numel():
        raise ValueError("teacher and compact map anchor counts differ")
    metric = load_shared_metric(
        metric_state_path, anchor_ids=anchor_ids, device=device
    )
    cache = query_cache.get("queries", query_cache)
    names = list(teacher["query_names"])
    selected_queries = (
        list(range(len(names)))
        if query_indices is None
        else [int(value) for value in torch.as_tensor(query_indices).tolist()]
    )
    if not selected_queries:
        raise ValueError("identity-folding query subset is empty")
    if min(selected_queries) < 0 or max(selected_queries) >= len(names):
        raise ValueError("identity-folding query index is out of range")
    rows = []
    for completed, query_index in enumerate(selected_queries, start=1):
        record = teacher["records"][query_index]
        cached = cache[names[query_index]]
        all_rows = torch.as_tensor(record["query_rows"]).long()
        teacher_local_rows = torch.arange(all_rows.numel())
        if int(deployment_row_limit) > 0:
            keep = all_rows < int(deployment_row_limit)
            all_rows = all_rows[keep]
            teacher_local_rows = teacher_local_rows[keep]
        if all_rows.numel() == 0:
            raise ValueError(f"query {names[query_index]} has no deployment rows")
        descriptors = F.normalize(
            torch.as_tensor(cached["native_descriptors"]).float()[all_rows], dim=1
        ).to(device)
        adapted, _ = metric(descriptors)
        baseline_matches = global_cosine_top1(adapted, bank)
        anchor_unique_matches = suppress_duplicate_anchor_matches(baseline_matches)
        folded_matches = suppress_duplicate_entity_matches(
            baseline_matches, components.to(device)
        )
        keypoints = torch.as_tensor(cached["native_keypoints"]).float()[all_rows]
        keypoints += float(cached.get("pixel_center_offset", 0.5))
        intrinsic = torch.as_tensor(cached["native_K"]).float()
        ground_truth_pose = torch.as_tensor(cached["pose_w2c"]).float()
        common = {
            "teacher_local_rows": teacher_local_rows,
            "keypoints": keypoints,
            "xyz": xyz,
            "intrinsic": intrinsic,
            "ground_truth_pose": ground_truth_pose,
            "component_ids": components,
            "record": record,
            "reprojection_error_px": reprojection_error_px,
            "clean_reprojection_px": clean_reprojection_px,
            "seed": seed,
        }
        baseline = _evaluate_matches(baseline_matches, **common)
        anchor_unique = _evaluate_matches(anchor_unique_matches, **common)
        folded = _evaluate_matches(folded_matches, **common)
        rows.append(
            {
                "query_index": query_index,
                "image_name": names[query_index],
                "baseline": baseline,
                "anchor_unique_control": anchor_unique,
                "entity_folded": folded,
                "duplicate_anchor_correspondence_removed_count": int(
                    baseline["correspondence_count"]
                    - anchor_unique["correspondence_count"]
                ),
                "component_correspondence_removed_count": int(
                    anchor_unique["correspondence_count"]
                    - folded["correspondence_count"]
                ),
            }
        )
        if int(progress_interval) > 0 and (
            completed % int(progress_interval) == 0
            or completed == len(selected_queries)
        ):
            print(
                {
                    "event": "identity_folding_mapping_replay",
                    "queries_complete": completed,
                    "query_count": len(selected_queries),
                },
                flush=True,
            )
    return {"queries": rows, "summary": summarize_identity_folding(rows)}


def aggregate_identity_folding_summaries(reports: Sequence[Mapping]) -> dict:
    """Aggregate pre-registered RANSAC seeds under a conservative gate.

    All reports must replay the same map and mapping-query subset.  Physical
    dedup is authorized only when every seed satisfies the complete pose and
    correspondence safety gate.  A coverage failure in any seed, or a
    consistent CVaR95 regression without one fully safe pose seed, stops the
    physical branch while retaining semantic component annotations.
    """
    items = list(reports)
    if not items:
        raise ValueError("identity-folding aggregation needs at least one report")
    reference = items[0]
    identity_keys = ("map_sha256", "query_count", "query_selection")
    for report in items:
        for key in identity_keys:
            if report.get(key) != reference.get(key):
                raise ValueError(f"identity-folding reports differ on {key}")
        if "summary" not in report or "paired" not in report["summary"]:
            raise ValueError("identity-folding report is missing paired summary")
    seeds = [int(report["seed"]) for report in items]
    if len(set(seeds)) != len(seeds):
        raise ValueError("identity-folding aggregation received duplicate seeds")

    rows = []
    for report in items:
        paired = report["summary"]["paired"]
        pose_safe = (
            float(paired["median_te_delta_cm"]) <= 1e-9
            and float(paired["mean_te_delta_cm"]) <= 1e-9
            and float(paired["p90_te_delta_cm"]) <= 1e-9
            and float(paired["cvar95_te_delta_cm"]) <= 1e-9
            and float(paired["recall_5cm_5deg_delta_pp"]) >= -1e-9
            and int(paired["catastrophic_count_delta"]) <= 0
            and int(paired.get("failure_count_delta", 0)) <= 0
        )
        rows.append(
            {
                "seed": int(report["seed"]),
                "routing": report["summary"]["routing"],
                "pose_safe": bool(pose_safe),
                **{key: paired[key] for key in (
                    "correspondence_removed_count",
                    "correspondence_removed_percent",
                    "raw_gt_precision_delta_pp",
                    "inlier_gt_precision_delta_pp",
                    "harmful_inlier_delta",
                    "clean_inlier_delta",
                    "median_te_delta_cm",
                    "mean_te_delta_cm",
                    "p90_te_delta_cm",
                    "cvar95_te_delta_cm",
                    "recall_5cm_5deg_delta_pp",
                    "catastrophic_count_delta",
                )},
                "failure_count_delta": int(paired.get("failure_count_delta", 0)),
            }
        )

    metric_keys = (
        "raw_gt_precision_delta_pp",
        "inlier_gt_precision_delta_pp",
        "harmful_inlier_delta",
        "clean_inlier_delta",
        "median_te_delta_cm",
        "mean_te_delta_cm",
        "p90_te_delta_cm",
        "cvar95_te_delta_cm",
        "recall_5cm_5deg_delta_pp",
    )
    statistics = {}
    for key in metric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        statistics[key] = {
            "mean": float(values.mean()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }

    all_safe = all(
        row["pose_safe"]
        and float(row["raw_gt_precision_delta_pp"]) >= -1e-9
        and int(row["harmful_inlier_delta"]) < 0
        for row in rows
    )
    critical_regression = any(
        float(row["raw_gt_precision_delta_pp"]) < -1e-9
        or float(row["recall_5cm_5deg_delta_pp"]) < -1e-9
        or int(row["catastrophic_count_delta"]) > 0
        or int(row["failure_count_delta"]) > 0
        for row in rows
    )
    consistent_tail_regression = all(
        float(row["cvar95_te_delta_cm"]) > 1e-9 for row in rows
    ) and not any(row["pose_safe"] for row in rows)
    if all_safe:
        routing = "go_to_evidence_transfer_prototype"
    elif critical_regression or consistent_tail_regression:
        routing = "stop_physical_dedup_keep_semantic_components"
    else:
        routing = "inconclusive_expand_mapping_sentinels"
    return {
        "schema": "lafgs_identity_folding_seed_aggregate",
        "version": 1,
        "uses_test_queries": False,
        "physical_map_mutated": False,
        "map_sha256": reference["map_sha256"],
        "query_count": int(reference["query_count"]),
        "query_selection": reference["query_selection"],
        "seed_count": len(rows),
        "seeds": seeds,
        "per_seed": rows,
        "statistics": statistics,
        "pose_safe_seed_count": int(sum(row["pose_safe"] for row in rows)),
        "critical_regression": bool(critical_regression),
        "consistent_tail_regression": bool(consistent_tail_regression),
        "routing": routing,
    }
