"""One-pass deployment-aware revision of a trained compact localization map.

The revision is learned exclusively from mapping-view global top-1 and PnP
outcomes.  It removes anchors whose absence improves the observed assignment
matrix, while preserving the mapping-feasible rank target.  Geometry and the
shared descriptor metric remain fixed during selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from localization.localizer import load_shared_metric
from localization.pose_solver import pose_error, solve_absolute_pose
from map_learning.trainer import _pose_error_cm, _project_errors
from topology.matching_coverage import (
    IncrementalBipartiteCoverage,
    base_candidate_edges,
    greedy_matching_reserve,
)
from topology.pose_information import (
    conditional_delete_loss,
    fisher_contributions,
    pose_jacobian_analytic,
    task_scaled_pose_jacobian,
)


def _csr_values(record: dict, prefix: str, row: int) -> torch.Tensor:
    offsets = torch.as_tensor(record[f"{prefix}_offsets"]).long()
    indices = torch.as_tensor(record[f"{prefix}_indices"]).long()
    return indices[int(offsets[row]) : int(offsets[row + 1])]


def _csr_contains_per_row(
    record: dict, prefix: str, values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized CSR membership and non-empty masks for one value per row."""
    offsets = torch.as_tensor(record[f"{prefix}_offsets"]).long()
    indices = torch.as_tensor(record[f"{prefix}_indices"]).long()
    values = torch.as_tensor(values).long().reshape(-1)
    if offsets.numel() < values.numel() + 1:
        raise ValueError(f"{prefix} CSR has fewer rows than deployment values")
    offsets = offsets[: values.numel() + 1]
    indices = indices[: int(offsets[-1])]
    counts = offsets[1:] - offsets[:-1]
    row_ids = torch.repeat_interleave(torch.arange(values.numel()), counts)
    matched = torch.zeros(values.numel(), dtype=torch.bool)
    if indices.numel():
        hits = indices == values[row_ids]
        if bool(hits.any()):
            matched[row_ids[hits]] = True
    return matched, counts > 0


def _safe_percent(numerator: int, denominator: int) -> float:
    return 100.0 * float(numerator) / max(int(denominator), 1)


def _summary(query_rows: list[dict], counters: dict[str, torch.Tensor]) -> dict:
    translation_errors = np.asarray(
        [row["te_cm"] for row in query_rows], dtype=np.float64
    )
    rotation_errors = np.asarray(
        [row["ae_deg"] for row in query_rows], dtype=np.float64
    )
    raw = int(counters["winner_count"].sum())
    correct = int(counters["correct_winner_count"].sum())
    inliers = int(counters["clean_inlier_count"].sum()) + int(
        counters["harmful_inlier_count"].sum()
    )
    clean = int(counters["clean_inlier_count"].sum())
    tail_count = max(int(math.ceil(0.05 * translation_errors.size)), 1)
    return {
        "query_count": int(translation_errors.size),
        "median_te_cm": float(np.median(translation_errors)),
        "mean_te_cm": float(np.mean(translation_errors)),
        "p90_te_cm": float(np.percentile(translation_errors, 90)),
        "p95_te_cm": float(np.percentile(translation_errors, 95)),
        "p99_te_cm": float(np.percentile(translation_errors, 99)),
        "cvar95_te_cm": float(np.sort(translation_errors)[-tail_count:].mean()),
        "median_ae_deg": float(np.median(rotation_errors)),
        "mean_ae_deg": float(np.mean(rotation_errors)),
        "p90_ae_deg": float(np.percentile(rotation_errors, 90)),
        "p95_ae_deg": float(np.percentile(rotation_errors, 95)),
        "recall_5cm_5deg_percent": float(
            100.0 * np.mean((translation_errors < 5.0) & (rotation_errors < 5.0))
        ),
        "catastrophic_100cm_count": int(np.count_nonzero(translation_errors >= 100.0)),
        "raw_gt_precision_percent": _safe_percent(correct, raw),
        "inlier_gt_precision_percent": _safe_percent(clean, inliers),
        "solver_inlier_ratio_percent": _safe_percent(inliers, raw),
        "retained_matches_mean": float(
            np.mean([row["correspondences"] for row in query_rows])
        ),
        "mean_hypotheses": float(np.mean([row["hypotheses"] for row in query_rows])),
    }


@torch.inference_mode()
def collect_deployment_statistics(
    *,
    state: dict,
    metric_state_path: str | Path,
    teacher: dict,
    query_cache: dict,
    device: torch.device,
    ransac_reprojection_px: float,
    clean_reprojection_px: float,
    task_translation_m: float,
    task_rotation_deg: float,
    seed: int,
    retrieval_topk: int = 8,
    progress_label: str = "mapping replay",
    query_indices: list[int] | torch.Tensor | None = None,
    deployment_row_limit: int = 0,
    collect_anchor_statistics: bool = True,
) -> dict:
    """Replay exact deployment matching and collect anchor-level outcomes."""
    count = int(torch.as_tensor(state["anchor_xyz"]).shape[0])
    if int(teacher["anchor_count"]) != count:
        raise ValueError("teacher and deployment map anchor counts differ")
    metric = load_shared_metric(
        metric_state_path,
        anchor_ids=torch.as_tensor(state["anchor_ids"]).long(),
        device=device,
    )
    bank = F.normalize(
        torch.as_tensor(state["anchor_features"]).float().to(device), dim=1
    )
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    cache = query_cache.get("queries", query_cache)
    names = list(teacher["query_names"])
    if set(names) != set(cache):
        missing = sorted(set(names) - set(cache))
        if missing:
            raise ValueError(f"query cache misses mapping queries: {missing[:3]}")

    names_of_counters = (
        "winner_count",
        "correct_winner_count",
        "false_attractor_count",
        "ambiguous_winner_count",
        "clean_inlier_count",
        "harmful_inlier_count",
        "counterfactual_clean_gain",
        "information_deletion_loss",
    )
    counters = {
        name: torch.zeros(count, dtype=torch.float64) for name in names_of_counters
    }
    selected_queries = (
        list(range(len(names)))
        if query_indices is None
        else [int(value) for value in torch.as_tensor(query_indices).tolist()]
    )
    if not selected_queries:
        raise ValueError("deployment replay query subset is empty")
    if min(selected_queries) < 0 or max(selected_queries) >= len(names):
        raise ValueError("deployment replay query index is out of range")
    query_rows = []
    for completed, query_index in enumerate(selected_queries, start=1):
        record = teacher["records"][query_index]
        cached = cache[names[query_index]]
        rows = torch.as_tensor(record["query_rows"]).long()
        if int(deployment_row_limit) > 0:
            # Native detector rows are score-ranked cache indices.  A K-prefix
            # is therefore row < K, not the first K entries of a potentially
            # sparse teacher record.
            rows = rows[rows < int(deployment_row_limit)]
            if rows.numel() == 0:
                raise ValueError(
                    f"query {names[query_index]} has no teacher rows in requested "
                    f"deployment prefix {int(deployment_row_limit)}"
                )
        descriptors = F.normalize(
            torch.as_tensor(cached["native_descriptors"]).float()[rows], dim=1
        ).to(device)
        adapted, _ = metric(descriptors)
        effective_topk = int(retrieval_topk) if collect_anchor_statistics else 1
        scores, indices = torch.topk(
            adapted @ bank.T, k=min(effective_topk, count), dim=1
        )
        del scores
        indices_cpu = indices.cpu()
        winners = indices_cpu[:, 0]
        counters["winner_count"].index_add_(
            0, winners, torch.ones(winners.numel(), dtype=torch.float64)
        )
        current_correct, has_positive = _csr_contains_per_row(
            record, "positive", winners
        )
        current_ambiguous, _ = _csr_contains_per_row(record, "ambiguous", winners)
        correct_winners = winners[current_correct]
        ambiguous_winners = winners[~current_correct & current_ambiguous]
        false_winners = winners[~current_correct & ~current_ambiguous & has_positive]
        for counter_name, selected in (
            ("correct_winner_count", correct_winners),
            ("ambiguous_winner_count", ambiguous_winners),
            ("false_attractor_count", false_winners),
        ):
            counters[counter_name].index_add_(
                0, selected, torch.ones(selected.numel(), dtype=torch.float64)
            )
        if collect_anchor_statistics:
            for local, winner in enumerate(winners.tolist()):
                positives = _csr_values(record, "positive", local)
                replacement_correct = False
                for alternative in indices_cpu[local, 1:].tolist():
                    if alternative == winner:
                        continue
                    replacement_correct = bool((positives == alternative).any())
                    break
                counters["counterfactual_clean_gain"][winner] += float(
                    replacement_correct
                ) - float(current_correct[local])

        keypoints = torch.as_tensor(cached["native_keypoints"]).float()[rows]
        keypoints = keypoints + float(cached.get("pixel_center_offset", 0.5))
        intrinsic = torch.as_tensor(cached["native_K"]).float()
        estimate = solve_absolute_pose(
            keypoints.numpy(),
            xyz[winners].numpy(),
            intrinsic.numpy(),
            reprojection_error_px=float(ransac_reprojection_px),
            confidence=0.99999,
            max_iterations=100000,
            min_iterations=1000,
            seed=int(seed),
        )
        inliers = torch.as_tensor(estimate.inliers).long().reshape(-1)
        clean_mask = torch.zeros(inliers.numel(), dtype=torch.bool)
        if inliers.numel():
            errors = _project_errors(
                xyz[winners[inliers]],
                keypoints[inliers],
                intrinsic,
                torch.as_tensor(cached["pose_w2c"]).float(),
            )
            clean_mask = errors <= float(clean_reprojection_px)
            clean_anchor = winners[inliers[clean_mask]]
            harmful_anchor = winners[inliers[~clean_mask]]
            counters["clean_inlier_count"].index_add_(
                0, clean_anchor, torch.ones(clean_anchor.numel(), dtype=torch.float64)
            )
            counters["harmful_inlier_count"].index_add_(
                0,
                harmful_anchor,
                torch.ones(harmful_anchor.numel(), dtype=torch.float64),
            )
            if clean_anchor.numel() and collect_anchor_statistics:
                clean_points = xyz[clean_anchor].double()
                jacobian = task_scaled_pose_jacobian(
                    pose_jacobian_analytic(
                        clean_points,
                        intrinsic.double(),
                        torch.as_tensor(cached["pose_w2c"]).double(),
                    ),
                    translation_scale=float(task_translation_m),
                    rotation_scale=math.radians(float(task_rotation_deg)),
                )
                contribution = fisher_contributions(
                    jacobian,
                    measurement_covariance=torch.full(
                        (clean_points.shape[0],),
                        max(float(clean_reprojection_px), 0.5) ** 2,
                        dtype=torch.float64,
                    ),
                )
                full = (
                    contribution.sum(dim=0) + torch.eye(6, dtype=torch.float64) * 1e-4
                )
                for anchor in torch.unique(clean_anchor).tolist():
                    selected = clean_anchor == int(anchor)
                    loss = conditional_delete_loss(
                        full, contribution[selected].sum(dim=0), objective="full"
                    )
                    counters["information_deletion_loss"][anchor] += float(
                        loss.clamp_min(0)
                    )
        ae_deg, _ = pose_error(
            estimate.pose_w2c,
            torch.as_tensor(cached["pose_w2c"]).cpu().numpy(),
        )
        te_cm = _pose_error_cm(estimate.pose_w2c, torch.as_tensor(cached["pose_w2c"]))
        query_rows.append(
            {
                "query_index": query_index,
                "image_name": names[query_index],
                "te_cm": float(te_cm),
                "ae_deg": float(ae_deg),
                "inliers": int(inliers.numel()),
                "clean_inliers": int(clean_mask.sum()),
                "hypotheses": int(estimate.diagnostics.get("iterations", 0)),
                "correspondences": int(rows.numel()),
            }
        )
        if completed % 25 == 0 or completed == len(selected_queries):
            print(
                json.dumps(
                    {
                        "event": progress_label,
                        "queries_complete": completed,
                        "query_count": len(selected_queries),
                    }
                ),
                flush=True,
            )
    return {
        "counters": counters,
        "queries": query_rows,
        "summary": _summary(query_rows, counters),
    }


def _matching_assignments(teacher: dict) -> IncrementalBipartiteCoverage:
    count = int(teacher["anchor_count"])
    state = IncrementalBipartiteCoverage(
        len(teacher["query_names"]), base_candidate_edges(teacher, count)
    )
    for anchor in range(count):
        state.add(anchor)
    return state


@torch.inference_mode()
def add_tail_competition_protection(
    statistics: dict,
    *,
    state: dict,
    metric_state_path: str | Path,
    teacher: dict,
    query_cache: dict,
    device: torch.device,
    tail_fraction: float = 0.10,
    retrieval_topk: int = 8,
) -> dict:
    """Protect anchors whose removal has no clean alternative in risky views."""
    if not 0.0 < float(tail_fraction) <= 1.0:
        raise ValueError("tail fraction must lie in (0, 1]")
    count = int(teacher["anchor_count"])
    errors = torch.as_tensor(
        [row["te_cm"] for row in statistics["queries"]], dtype=torch.float64
    )
    tail_count = max(int(math.ceil(errors.numel() * float(tail_fraction))), 1)
    risky_queries = set(torch.topk(errors, k=tail_count).indices.tolist())
    protected = torch.zeros(count, dtype=torch.float64)
    cache = query_cache.get("queries", query_cache)
    metric = load_shared_metric(
        metric_state_path,
        anchor_ids=torch.as_tensor(state["anchor_ids"]).long(),
        device=device,
    )
    bank = F.normalize(
        torch.as_tensor(state["anchor_features"]).float().to(device), dim=1
    )
    for query_index in sorted(risky_queries):
        record = teacher["records"][query_index]
        cached = cache[teacher["query_names"][query_index]]
        rows = torch.as_tensor(record["query_rows"]).long()
        descriptor = F.normalize(
            torch.as_tensor(cached["native_descriptors"]).float()[rows], dim=1
        ).to(device)
        adapted, _ = metric(descriptor)
        indices = torch.topk(
            adapted @ bank.T, k=min(int(retrieval_topk), count), dim=1
        ).indices.cpu()
        for local, winner in enumerate(indices[:, 0].tolist()):
            positives = _csr_values(record, "positive", local)
            current_correct = bool((positives == winner).any())
            replacement_correct = False
            for alternative in indices[local, 1:].tolist():
                if alternative != winner:
                    replacement_correct = bool((positives == alternative).any())
                    break
            if int(replacement_correct) - int(current_correct) <= 0:
                protected[winner] += 1
    statistics["counters"]["tail_nonimproving_winner_count"] = protected
    statistics["tail_competition_protection"] = {
        "tail_fraction": float(tail_fraction),
        "tail_query_count": len(risky_queries),
        "protected_anchor_count": int((protected > 0).sum()),
        "uses_test_queries": False,
    }
    return statistics


def select_revision(
    teacher: dict,
    statistics: dict,
    *,
    matching_rows_target: int,
    revisable_mask: torch.Tensor | None = None,
    maximum_prune_fraction: float = 0.02,
    minimum_counterfactual_gain: float = 1.0,
    maximum_tail_nonimproving_wins: int = 2,
) -> tuple[torch.Tensor, dict]:
    """Select a safe harmful-anchor prune set and restore rank if needed."""
    counters = statistics["counters"]
    count = int(teacher["anchor_count"])
    original_matching = _matching_assignments(teacher)
    clean = counters["clean_inlier_count"]
    harmful = counters["harmful_inlier_count"]
    correct = counters["correct_winner_count"]
    false = counters["false_attractor_count"]
    info = counters["information_deletion_loss"]
    gain = counters["counterfactual_clean_gain"]
    tail_nonimproving = counters.get(
        "tail_nonimproving_winner_count", torch.zeros_like(gain)
    )
    revisable = (
        torch.ones(count, dtype=torch.bool)
        if revisable_mask is None
        else torch.as_tensor(revisable_mask).bool().reshape(-1)
    )
    if revisable.numel() != count:
        raise ValueError("revisable mask and teacher anchor registry differ")
    opportunity = clean + harmful
    harmful_rate = harmful / opportunity.clamp_min(1)
    false_rate = false / (false + correct).clamp_min(1)
    finite_info = info[torch.isfinite(info)]
    info_limit = float(torch.quantile(finite_info, 0.5)) if finite_info.numel() else 0.0
    eligible_without_tail = (
        (gain >= float(minimum_counterfactual_gain))
        & revisable
        & (correct == 0)
        & (info <= info_limit)
        & ((harmful_rate >= 0.25) | (false_rate >= 0.75))
    )
    eligible = eligible_without_tail & (
        tail_nonimproving <= int(maximum_tail_nonimproving_wins)
    )
    risk = (
        4.0 * gain
        + torch.log1p(false)
        + 2.0 * torch.log1p(harmful)
        - 2.0 * torch.log1p(clean)
        - info
    )
    limit = max(int(math.floor(count * float(maximum_prune_fraction))), 0)
    ordered = torch.argsort(risk, descending=True, stable=True)
    proposed = ordered[eligible[ordered]][:limit]
    keep = torch.ones(count, dtype=torch.bool)
    keep[proposed] = False
    edges = base_candidate_edges(teacher, count)
    retained = torch.nonzero(keep, as_tuple=False).reshape(-1)
    removed = torch.nonzero(~keep, as_tuple=False).reshape(-1)
    utility = correct + clean + info - false - harmful
    restored, final_matching, matching_report = greedy_matching_reserve(
        edges,
        retained.tolist(),
        removed.tolist(),
        utility,
        torch.zeros(len(teacher["query_names"]), dtype=torch.long),
        requested_rows_per_query=int(matching_rows_target),
        maximum_reserve=int(removed.numel()),
    )
    keep[restored] = True
    pruned = torch.nonzero(~keep, as_tuple=False).reshape(-1)
    return pruned, {
        "proposed_prune_count": int(proposed.numel()),
        "restored_for_matching_count": int(restored.numel()),
        "final_prune_count": int(pruned.numel()),
        "maximum_prune_fraction": float(maximum_prune_fraction),
        "minimum_counterfactual_gain": float(minimum_counterfactual_gain),
        "maximum_tail_nonimproving_wins": int(maximum_tail_nonimproving_wins),
        "information_deletion_median": info_limit,
        "tail_protected_candidate_count": int(
            (
                eligible_without_tail
                & (tail_nonimproving > int(maximum_tail_nonimproving_wins))
            ).sum()
        ),
        "matching_rank_before_p10": float(np.percentile(original_matching.counts, 10)),
        "matching_rank_after_p10": float(np.percentile(final_matching.counts, 10)),
        "matching_rank_before_median": float(np.median(original_matching.counts)),
        "matching_rank_after_median": float(np.median(final_matching.counts)),
        "matching_constraint": matching_report,
        "selection_uses_test_queries": False,
        "revisable_anchor_count": int(revisable.sum()),
    }


def subset_teacher(
    teacher: dict,
    keep: torch.Tensor,
    output_map: Path,
    *,
    source_anchor_type: torch.Tensor | None = None,
) -> dict:
    keep = torch.as_tensor(keep).bool().reshape(-1)
    remap = torch.full((keep.numel(),), -1, dtype=torch.long)
    remap[keep] = torch.arange(int(keep.sum()))
    output = dict(teacher)
    output["anchor_count"] = int(keep.sum())
    output["anchor_map"] = str(output_map)
    records = []
    if source_anchor_type is not None:
        source_anchor_type = torch.as_tensor(source_anchor_type).long().reshape(-1)
        if source_anchor_type.numel() != keep.numel():
            raise ValueError("anchor types and teacher anchor registry differ")
        if bool(((~keep) & (source_anchor_type == 1)).any()):
            raise ValueError("deployment revision must not remove Track Core anchors")
    positive_rows = strong_pairs = ambiguous_pairs = 0
    for record in teacher["records"]:
        revised = dict(record)
        for prefix in ("positive", "ambiguous"):
            offsets = torch.as_tensor(record[f"{prefix}_offsets"]).long()
            indices = torch.as_tensor(record[f"{prefix}_indices"]).long()
            new_offsets = [0]
            new_indices = []
            for row in range(offsets.numel() - 1):
                values = indices[offsets[row] : offsets[row + 1]]
                values = remap[values[keep[values]]]
                new_indices.extend(values.tolist())
                new_offsets.append(len(new_indices))
                if prefix == "positive" and values.numel():
                    positive_rows += 1
            revised[f"{prefix}_offsets"] = torch.as_tensor(new_offsets).long()
            revised[f"{prefix}_indices"] = torch.as_tensor(new_indices).long()
            if prefix == "positive":
                strong_pairs += len(new_indices)
            else:
                ambiguous_pairs += len(new_indices)
        records.append(revised)
    output["records"] = records
    output["diagnostics"] = {
        **teacher["diagnostics"],
        "positive_rows": positive_rows,
        "strong_pair_count": strong_pairs,
        "ambiguous_pair_count": ambiguous_pairs,
    }
    output["deployment_revision"] = {
        "source_anchor_count": int(keep.numel()),
        "retained_anchor_count": int(keep.sum()),
        "uses_test_queries": False,
    }
    return output


def subset_map_and_metric(
    state: dict,
    metric_state: dict,
    keep: torch.Tensor,
    *,
    output_map: Path | None = None,
) -> tuple[dict, dict]:
    keep = torch.as_tensor(keep).bool().reshape(-1)
    count = int(keep.numel())
    if torch.as_tensor(state["anchor_ids"]).numel() != count:
        raise ValueError("map and revision mask do not align")
    output = dict(state)
    for key, value in state.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == count:
            output[key] = value[keep]
    output["anchor_ids"] = torch.arange(int(keep.sum()), dtype=torch.long)
    track_mask = keep & (torch.as_tensor(state["anchor_type"]).long() == 1)
    base_mask = keep & ~track_mask
    if "track_centric_reconstruction" in state:
        metadata = dict(state["track_centric_reconstruction"])
        source_track_indices = torch.as_tensor(metadata["track_indices"]).long()
        source_base_rows = torch.as_tensor(metadata["base_canonical_rows"]).long()
        source_track_count = source_track_indices.numel()
        if source_track_count + source_base_rows.numel() != count:
            raise ValueError("track reconstruction registry and map do not align")
        metadata.update(
            {
                "budget": int(keep.sum()),
                "track_anchor_count": int(track_mask.sum()),
                "base_reserve_count": int(base_mask.sum()),
                "final_track_count": int(track_mask.sum()),
                "final_base_count": int(base_mask.sum()),
                "track_indices": source_track_indices[keep[:source_track_count]],
                "base_canonical_rows": source_base_rows[keep[source_track_count:]],
                "deployment_revision_applied": True,
            }
        )
        output["track_centric_reconstruction"] = metadata
    output["provenance"] = {
        **state.get("provenance", {}),
        "deployment_revision": {
            "source_anchor_count": count,
            "retained_anchor_count": int(keep.sum()),
            "selection_split": "all_mapping_train",
            "uses_test_queries": False,
        },
    }
    output["base_anchor_count"] = int(base_mask.sum())
    output["micro_anchor_count"] = int(track_mask.sum())
    output["requested_micro_anchor_budget"] = int(track_mask.sum())
    output["canonical_anchor_count"] = int(keep.sum())
    revised_metric = dict(metric_state)
    revised_metric["landmark_indices"] = output["anchor_ids"].clone()
    revised_metric["map_path"] = (
        str(Path(output_map).resolve()) if output_map is not None else None
    )
    return output, revised_metric


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--complete-positive-teacher", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--matching-rows-target", type=int, required=True)
    parser.add_argument("--ransac-reprojection-px", type=float, required=True)
    parser.add_argument("--clean-reprojection-px", type=float, required=True)
    parser.add_argument("--task-translation-m", type=float, default=0.05)
    parser.add_argument("--task-rotation-deg", type=float, default=5.0)
    parser.add_argument("--maximum-prune-fraction", type=float, default=0.02)
    parser.add_argument("--minimum-counterfactual-gain", type=float, default=1.0)
    parser.add_argument("--maximum-tail-nonimproving-wins", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    metric_state = torch.load(args.metric_state, map_location="cpu", weights_only=False)
    teacher = torch.load(
        args.complete_positive_teacher, map_location="cpu", weights_only=False
    )
    cache = torch.load(args.query_cache, map_location="cpu", weights_only=False)
    before_cache = output / "before_statistics.pt"
    if before_cache.is_file():
        before = torch.load(before_cache, map_location="cpu", weights_only=False)
    else:
        before = collect_deployment_statistics(
            state=state,
            metric_state_path=args.metric_state,
            teacher=teacher,
            query_cache=cache,
            device=torch.device(args.device),
            ransac_reprojection_px=args.ransac_reprojection_px,
            clean_reprojection_px=args.clean_reprojection_px,
            task_translation_m=args.task_translation_m,
            task_rotation_deg=args.task_rotation_deg,
            seed=args.seed,
            progress_label="before_revision_replay",
        )
        torch.save(before, before_cache)
    before = add_tail_competition_protection(
        before,
        state=state,
        metric_state_path=args.metric_state,
        teacher=teacher,
        query_cache=cache,
        device=torch.device(args.device),
    )
    torch.save(before, before_cache)
    pruned, selection = select_revision(
        teacher,
        before,
        matching_rows_target=args.matching_rows_target,
        revisable_mask=torch.as_tensor(state["anchor_type"]).long() == 0,
        maximum_prune_fraction=args.maximum_prune_fraction,
        minimum_counterfactual_gain=args.minimum_counterfactual_gain,
        maximum_tail_nonimproving_wins=args.maximum_tail_nonimproving_wins,
    )
    keep = torch.ones(int(teacher["anchor_count"]), dtype=torch.bool)
    keep[pruned] = False
    revised_map_path = output / "revised_anchor_map.pt"
    revised_metric_path = output / "revised_metric_state.pt"
    revised_teacher_path = output / "revised_complete_positive_teacher.pt"
    revised_map, revised_metric = subset_map_and_metric(
        state, metric_state, keep, output_map=revised_map_path
    )
    revised_teacher = subset_teacher(
        teacher,
        keep,
        revised_map_path,
        source_anchor_type=state["anchor_type"],
    )
    torch.save(revised_map, revised_map_path)
    torch.save(revised_metric, revised_metric_path)
    torch.save(revised_teacher, revised_teacher_path)
    keep_signature = hashlib.sha256(keep.numpy().tobytes()).hexdigest()[:12]
    after_cache = output / f"after_statistics_{keep_signature}.pt"
    if after_cache.is_file():
        after = torch.load(after_cache, map_location="cpu", weights_only=False)
    else:
        after = collect_deployment_statistics(
            state=revised_map,
            metric_state_path=revised_metric_path,
            teacher=revised_teacher,
            query_cache=cache,
            device=torch.device(args.device),
            ransac_reprojection_px=args.ransac_reprojection_px,
            clean_reprojection_px=args.clean_reprojection_px,
            task_translation_m=args.task_translation_m,
            task_rotation_deg=args.task_rotation_deg,
            seed=args.seed,
            progress_label="after_revision_replay",
        )
        torch.save(after, after_cache)
    gate = {
        "matching_target_preserved": selection["matching_constraint"][
            "unmet_query_count"
        ]
        == 0,
        "median_non_degraded": after["summary"]["median_te_cm"]
        <= 1.02 * before["summary"]["median_te_cm"],
        "mean_non_degraded": after["summary"]["mean_te_cm"]
        <= 1.02 * before["summary"]["mean_te_cm"],
        "p90_non_degraded": after["summary"]["p90_te_cm"]
        <= 1.02 * before["summary"]["p90_te_cm"],
        "p95_non_degraded": after["summary"]["p95_te_cm"]
        <= 1.02 * before["summary"]["p95_te_cm"],
        "cvar95_non_degraded": after["summary"]["cvar95_te_cm"]
        <= 1.02 * before["summary"]["cvar95_te_cm"],
        "catastrophic_failures_non_degraded": after["summary"][
            "catastrophic_100cm_count"
        ]
        <= before["summary"]["catastrophic_100cm_count"],
        "raw_precision_non_degraded": after["summary"]["raw_gt_precision_percent"]
        + 0.05
        >= before["summary"]["raw_gt_precision_percent"],
    }
    relative_map_reduction = float(pruned.numel()) / max(
        int(teacher["anchor_count"]), 1
    )
    meaningful_improvement = (
        after["summary"]["raw_gt_precision_percent"]
        >= before["summary"]["raw_gt_precision_percent"] + 0.01
        or after["summary"]["p90_te_cm"] <= 0.995 * before["summary"]["p90_te_cm"]
        or after["summary"]["mean_te_cm"] <= 0.995 * before["summary"]["mean_te_cm"]
        or after["summary"]["cvar95_te_cm"] <= 0.995 * before["summary"]["cvar95_te_cm"]
        or relative_map_reduction >= 0.005
    )
    gate["meaningful_improvement"] = bool(meaningful_improvement)
    accepted = bool(pruned.numel()) and all(gate.values())
    report = {
        "schema": "lafgs_deployment_aware_topology_revision",
        "version": 1,
        "uses_test_queries": False,
        "source_map": str(Path(args.map).resolve()),
        "source_metric_state": str(Path(args.metric_state).resolve()),
        "source_teacher": str(Path(args.complete_positive_teacher).resolve()),
        "revised_map": str(revised_map_path),
        "revised_metric_state": str(revised_metric_path),
        "revised_teacher": str(revised_teacher_path),
        "before": before["summary"],
        "after": after["summary"],
        "selection": selection,
        "gate": gate,
        "accepted": accepted,
        "pruned_anchor_rows": pruned.tolist(),
        "relative_map_reduction": relative_map_reduction,
    }
    torch.save(
        {"before": before["counters"], "after": after["counters"]},
        output / "revision_statistics.pt",
    )
    (output / "deployment_revision_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
