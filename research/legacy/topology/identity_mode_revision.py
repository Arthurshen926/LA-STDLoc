#!/usr/bin/env python3
"""Cross-fitted appearance-mode revision of a compact localization topology.

The exact pose-set oracle is used only to seed pose-critical identities.  It
never supplies a query-specific descriptor target.  Secondary descriptor modes
must instead recur across mapping trajectory folds, rescue held-out legal
positives, and introduce no sampled clean false attraction.  Accepted modes
replace functionally unused anchors one-for-one, so deployment remains global
cosine top-1 followed by one standard PoseLib solve at fixed map capacity.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from localization.localizer import load_shared_metric
from map_learning.pose_set_refinement import _csr_values, _selected_joint_actions
from topology.crossfit_swap_revision import temporal_crossfit_split
from topology.deployment_revision import collect_deployment_statistics


def interleaved_inner_fold_assignments(
    query_names: list[str],
    selection_queries: list[int],
    outer_assignments: dict[str, int],
    inner_fold_count: int = 4,
) -> tuple[dict[str, int], dict]:
    """Interleave discovery views inside each outer trajectory block.

    The outer split remains a contiguous trajectory holdout.  Inner folds are
    only a recurrence test for modes mined from the discovery partition, so
    using the outer block identity itself would collapse local tracks into one
    fold and make the test vacuous.
    """
    if int(inner_fold_count) < 2:
        raise ValueError("inner fold count must be at least two")
    selected = {int(value) for value in selection_queries}
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for query_index in sorted(selected):
        name = str(query_names[query_index])
        if name not in outer_assignments:
            raise KeyError(f"outer split has no assignment for {name}")
        sequence = name.rsplit("/", 1)[0] if "/" in name else ""
        grouped[(sequence, int(outer_assignments[name]))].append(query_index)

    assignments: dict[str, int] = {}
    group_sizes = {}
    fold_counts = torch.zeros(int(inner_fold_count), dtype=torch.long)
    for (sequence, outer_block), indices in sorted(grouped.items()):
        ordered = sorted(indices, key=lambda index: str(query_names[index]))
        group_sizes[f"{sequence}|outer_{outer_block}"] = len(ordered)
        for rank, query_index in enumerate(ordered):
            fold = int(rank % int(inner_fold_count))
            assignments[str(query_names[query_index])] = fold
            fold_counts[fold] += 1
    if set(assignments) != {str(query_names[index]) for index in selected}:
        raise RuntimeError("inner folds do not cover exactly the discovery queries")
    return assignments, {
        "policy": "interleaved_within_sequence_and_outer_discovery_block",
        "fold_count": int(inner_fold_count),
        "query_count": len(assignments),
        "fold_query_counts": fold_counts.tolist(),
        "group_sizes": group_sizes,
        "assignments": assignments,
        "uses_outer_gate_queries": False,
        "uses_test_queries": False,
    }


def _oracle_selection_targets(
    oracle: dict, selection_queries: set[int]
) -> dict[int, float]:
    gains: dict[int, list[float]] = defaultdict(list)
    for row in oracle["queries"]:
        if int(row["query_index"]) not in selection_queries:
            continue
        gain = max(float(row["current_risk"]) - float(row["joint_risk"]), 0.0)
        for action in _selected_joint_actions(row):
            if action["kind"] == "swap":
                gains[int(action["anchor"])].append(gain)
    return {
        anchor: float(np.median(np.asarray(values, dtype=np.float64)))
        for anchor, values in gains.items()
    }


def _fit_secondary_medoid(
    observations: list[dict],
    source_feature: torch.Tensor,
    *,
    assignment_margin: float,
    minimum_observations: int,
    minimum_views: int,
    iterations: int = 6,
) -> dict | None:
    """Fit one secondary spherical mode while leaving the primary fixed."""
    if len(observations) < int(minimum_observations):
        return None
    if len({int(value["query_index"]) for value in observations}) < int(
        minimum_views
    ):
        return None
    adapted = F.normalize(
        torch.stack([value["adapted"] for value in observations]).float(), dim=1
    )
    source = F.normalize(torch.as_tensor(source_feature).float(), dim=0)
    source_score = adapted @ source
    medoid = int(torch.argmin(source_score))
    candidate = adapted[medoid]
    assigned = torch.zeros(len(observations), dtype=torch.bool)
    for _ in range(int(iterations)):
        assigned = adapted @ candidate > source_score + float(assignment_margin)
        if int(assigned.sum()) < int(minimum_observations):
            return None
        center = F.normalize(adapted[assigned].mean(dim=0), dim=0)
        assigned_rows = torch.nonzero(assigned, as_tuple=False).reshape(-1)
        medoid = int(assigned_rows[torch.argmax(adapted[assigned] @ center)])
        candidate = adapted[medoid]
    assigned_views = {
        int(observations[index]["query_index"])
        for index in torch.nonzero(assigned, as_tuple=False).reshape(-1).tolist()
    }
    if len(assigned_views) < int(minimum_views):
        return None
    return {
        "raw": F.normalize(observations[medoid]["raw"].float(), dim=0),
        "adapted": candidate,
        "assigned_count": int(assigned.sum()),
        "assigned_view_count": len(assigned_views),
        "source_cosine": float(candidate @ source),
    }


@torch.inference_mode()
def _collect_selection_evidence(
    *,
    state: dict,
    metric_state_path: str | Path,
    teacher: dict,
    query_cache: dict,
    target_gains: dict[int, float],
    selection_queries: list[int],
    block_assignments: dict[str, int],
    deployment_row_limit: int,
    maximum_clean_samples: int,
    device: torch.device,
) -> dict:
    """Collect complete target support and current deployment behavior."""
    count = int(torch.as_tensor(state["anchor_ids"]).numel())
    target_lookup = torch.zeros(count, dtype=torch.bool)
    target_lookup[torch.as_tensor(sorted(target_gains)).long()] = True
    metric = load_shared_metric(
        metric_state_path,
        anchor_ids=torch.as_tensor(state["anchor_ids"]).long(),
        device=device,
    )
    bank = F.normalize(
        torch.as_tensor(state["anchor_features"]).float().to(device), dim=1
    )
    cache = query_cache.get("queries", query_cache)
    names = list(teacher["query_names"])
    winner_count = torch.zeros(count, dtype=torch.long)
    positive_count = torch.zeros(count, dtype=torch.long)
    sole_positive_count = torch.zeros(count, dtype=torch.long)
    observations: dict[int, list[dict]] = defaultdict(list)
    clean = []
    clean_per_query = max(
        int(np.ceil(maximum_clean_samples / max(len(selection_queries), 1))), 1
    )
    for completed, query_index in enumerate(selection_queries, start=1):
        record = teacher["records"][query_index]
        cached = cache[names[query_index]]
        all_rows = torch.as_tensor(record["query_rows"]).long()
        selected_local = torch.arange(all_rows.numel())
        if int(deployment_row_limit) > 0:
            selected_local = selected_local[
                all_rows < int(deployment_row_limit)
            ]
        if selected_local.numel() == 0:
            continue
        native_rows = all_rows[selected_local]
        raw = F.normalize(
            torch.as_tensor(cached["native_descriptors"]).float()[native_rows],
            dim=1,
        ).to(device)
        adapted, _ = metric(raw)
        top_score, winner = torch.max(adapted @ bank.T, dim=1)
        winner_cpu = winner.cpu()
        winner_count.index_add_(
            0, winner_cpu, torch.ones(winner_cpu.numel(), dtype=torch.long)
        )

        offsets = torch.as_tensor(record["positive_offsets"]).long()
        positive_indices = torch.as_tensor(record["positive_indices"]).long()
        lengths = offsets[1:] - offsets[:-1]
        occurrence_rows = torch.repeat_interleave(
            torch.arange(all_rows.numel()), lengths
        )
        selected_mask = torch.zeros(all_rows.numel(), dtype=torch.bool)
        selected_mask[selected_local] = True
        selected_occurrence = selected_mask[occurrence_rows]
        if bool(selected_occurrence.any()):
            positive_count.index_add_(
                0,
                positive_indices[selected_occurrence],
                torch.ones(int(selected_occurrence.sum()), dtype=torch.long),
            )
        sole_local = selected_local[lengths[selected_local] == 1]
        if sole_local.numel():
            sole_indices = positive_indices[offsets[sole_local]]
            sole_positive_count.index_add_(
                0,
                sole_indices,
                torch.ones(sole_indices.numel(), dtype=torch.long),
            )

        local_to_batch = torch.full((all_rows.numel(),), -1, dtype=torch.long)
        local_to_batch[selected_local] = torch.arange(selected_local.numel())
        target_occurrence = selected_occurrence & target_lookup[positive_indices]
        target_pairs = (
            torch.stack(
                [
                    occurrence_rows[target_occurrence],
                    positive_indices[target_occurrence],
                ],
                dim=1,
            ).unique(dim=0)
            if bool(target_occurrence.any())
            else torch.empty((0, 2), dtype=torch.long)
        )
        block = int(block_assignments[names[query_index]])
        for local, target in target_pairs.tolist():
            batch = int(local_to_batch[local])
            positives = _csr_values(record, "positive", local)
            current_winner = int(winner_cpu[batch])
            observations[int(target)].append(
                {
                    "query_index": int(query_index),
                    "fold": block,
                    "raw": raw[batch].cpu(),
                    "adapted": adapted[batch].cpu(),
                    "current_score": float(top_score[batch]),
                    "current_winner": current_winner,
                    "current_correct": bool(
                        (positives == current_winner).any()
                    ),
                }
            )

        sample_count = min(clean_per_query, int(selected_local.numel()))
        sampled_batch = (
            torch.linspace(0, selected_local.numel() - 1, steps=sample_count)
            .round()
            .long()
            .unique(sorted=True)
        )
        for batch in sampled_batch.tolist():
            local = int(selected_local[batch])
            positives = _csr_values(record, "positive", local)
            current_winner = int(winner_cpu[batch])
            if not bool((positives == current_winner).any()):
                continue
            legal_targets = tuple(
                int(value)
                for value in positives[target_lookup[positives]].tolist()
            )
            clean.append(
                {
                    "query_index": int(query_index),
                    "fold": block,
                    "adapted": adapted[batch].cpu(),
                    "current_score": float(top_score[batch]),
                    "legal_targets": legal_targets,
                }
            )
        if completed % 50 == 0 or completed == len(selection_queries):
            print(
                json.dumps(
                    {
                        "event": "identity_mode_evidence",
                        "queries_complete": completed,
                        "query_count": len(selection_queries),
                    }
                ),
                flush=True,
            )
    return {
        "observations": observations,
        "clean": clean,
        "winner_count": winner_count,
        "positive_count": positive_count,
        "sole_positive_count": sole_positive_count,
    }


def _candidate_effect(
    candidate: torch.Tensor,
    source: torch.Tensor,
    observations: list[dict],
    clean: list[dict],
    target: int,
    *,
    rescue_margin: float,
    assignment_margin: float,
    attraction_margin: float,
) -> tuple[int, int, int]:
    candidate = F.normalize(torch.as_tensor(candidate).float(), dim=0)
    source = F.normalize(torch.as_tensor(source).float(), dim=0)
    rescue = 0
    assigned = 0
    for value in observations:
        descriptor = value["adapted"]
        candidate_score = float(descriptor @ candidate)
        if candidate_score > float(descriptor @ source) + float(
            assignment_margin
        ):
            assigned += 1
        if (
            not value["current_correct"]
            and candidate_score
            > float(value["current_score"]) + float(rescue_margin)
        ):
            rescue += 1
    harm = sum(
        target not in value["legal_targets"]
        and float(value["adapted"] @ candidate)
        > float(value["current_score"]) + float(attraction_margin)
        for value in clean
    )
    return int(rescue), int(harm), int(assigned)


def discover_identity_modes(
    *,
    state: dict,
    target_gains: dict[int, float],
    evidence: dict,
    minimum_mode_observations: int,
    minimum_mode_views: int,
    minimum_stable_folds: int,
    minimum_total_rescues: int,
    assignment_margin: float,
    rescue_margin: float,
    attraction_margin: float,
    maximum_modes: int,
) -> tuple[list[dict], dict]:
    """Select modes using only inner cross-fit folds of the discovery split."""
    bank = F.normalize(torch.as_tensor(state["anchor_features"]).float(), dim=1)
    observations_by_target = evidence["observations"]
    clean = evidence["clean"]
    candidates = []
    rejected = defaultdict(int)
    diagnostics = []
    for target in sorted(target_gains):
        observations = observations_by_target.get(target, [])
        folds = sorted({int(value["fold"]) for value in observations})
        if len(folds) < int(minimum_stable_folds) + 1:
            rejected["insufficient_folds"] += 1
            continue
        fold_rows = []
        stable_folds = 0
        total_rescues = 0
        total_harm = 0
        for fold in folds:
            train = [value for value in observations if int(value["fold"]) != fold]
            heldout = [value for value in observations if int(value["fold"]) == fold]
            fitted = _fit_secondary_medoid(
                train,
                bank[target],
                assignment_margin=assignment_margin,
                minimum_observations=minimum_mode_observations,
                minimum_views=minimum_mode_views,
            )
            if fitted is None:
                fold_rows.append(
                    {"fold": fold, "fit": False, "rescue": 0, "harm": 0}
                )
                continue
            heldout_clean = [
                value for value in clean if int(value["fold"]) == fold
            ]
            rescue, harm, assigned = _candidate_effect(
                fitted["adapted"],
                bank[target],
                heldout,
                heldout_clean,
                target,
                rescue_margin=rescue_margin,
                assignment_margin=assignment_margin,
                attraction_margin=attraction_margin,
            )
            stable = rescue > 0 and harm == 0
            stable_folds += int(stable)
            total_rescues += rescue
            total_harm += harm
            fold_rows.append(
                {
                    "fold": fold,
                    "fit": True,
                    "rescue": rescue,
                    "harm": harm,
                    "assigned": assigned,
                    "stable": stable,
                    "source_cosine": fitted["source_cosine"],
                }
            )
        if stable_folds < int(minimum_stable_folds):
            rejected["unstable_crossfit"] += 1
            continue
        if total_rescues < int(minimum_total_rescues) or total_harm > 0:
            rejected["insufficient_net_rescue"] += 1
            continue
        final = _fit_secondary_medoid(
            observations,
            bank[target],
            assignment_margin=assignment_margin,
            minimum_observations=minimum_mode_observations,
            minimum_views=minimum_mode_views,
        )
        if final is None:
            rejected["final_fit_failed"] += 1
            continue
        final_rescue, final_harm, final_assigned = _candidate_effect(
            final["adapted"],
            bank[target],
            observations,
            clean,
            target,
            rescue_margin=rescue_margin,
            assignment_margin=assignment_margin,
            attraction_margin=attraction_margin,
        )
        if final_harm > 0 or final_rescue < int(minimum_total_rescues):
            rejected["final_attraction_gate"] += 1
            continue
        score = (
            100.0 * float(total_rescues)
            + 20.0 * float(stable_folds)
            + float(final_assigned)
            + 10.0 * float(target_gains[target])
        )
        candidate = {
            "target": int(target),
            "score": score,
            "pose_gain": float(target_gains[target]),
            "stable_fold_count": stable_folds,
            "crossfit_rescue_count": total_rescues,
            "crossfit_harm_count": total_harm,
            "final_rescue_count": final_rescue,
            "final_harm_count": final_harm,
            "final_assigned_count": final_assigned,
            "source_cosine": final["source_cosine"],
            "assigned_view_count": final["assigned_view_count"],
            "raw": final["raw"],
            "folds": fold_rows,
        }
        candidates.append(candidate)
        diagnostics.append(candidate)
    candidates.sort(key=lambda value: (-value["score"], value["target"]))
    selected = candidates[: max(int(maximum_modes), 0)]
    return selected, {
        "oracle_target_count": len(target_gains),
        "accepted_before_capacity": len(candidates),
        "selected_mode_count": len(selected),
        "rejected": dict(rejected),
        "candidates": [
            {key: value for key, value in candidate.items() if key != "raw"}
            for candidate in diagnostics
        ],
    }


def select_one_for_one_retirements(
    *,
    state: dict,
    evidence: dict,
    modes: list[dict],
) -> tuple[list[dict], torch.Tensor, dict]:
    """Retire anchors unused by discovery top-1 and unique-positive coverage."""
    count = int(torch.as_tensor(state["anchor_ids"]).numel())
    protected = {int(value["target"]) for value in modes}
    anchor_type = torch.as_tensor(state["anchor_type"]).long()
    winner = evidence["winner_count"]
    positive = evidence["positive_count"]
    sole = evidence["sole_positive_count"]
    eligible = [
        row
        for row in range(count)
        if row not in protected and int(winner[row]) == 0 and int(sole[row]) == 0
    ]
    eligible.sort(
        key=lambda row: (
            int(anchor_type[row] != 0),
            int(positive[row]),
            int(row),
        )
    )
    realized = min(len(modes), len(eligible))
    modes = modes[:realized]
    retire = torch.as_tensor(eligible[:realized], dtype=torch.long)
    return modes, retire, {
        "eligible_retirement_count": len(eligible),
        "realized_one_for_one_count": realized,
        "retired_base_count": int((anchor_type[retire] == 0).sum()),
        "retired_track_count": int((anchor_type[retire] != 0).sum()),
        "retired_positive_support_sum": int(positive[retire].sum()),
        "retired_winner_count_sum": int(winner[retire].sum()),
        "retired_sole_positive_count_sum": int(sole[retire].sum()),
    }


@torch.inference_mode()
def materialize_identity_modes(
    *,
    state: dict,
    metric_state: dict,
    modes: list[dict],
    retire: torch.Tensor,
    device: torch.device,
) -> tuple[dict, dict, torch.Tensor, torch.Tensor]:
    count = int(torch.as_tensor(state["anchor_ids"]).numel())
    retire = torch.as_tensor(retire).long()
    keep_mask = torch.ones(count, dtype=torch.bool)
    keep_mask[retire] = False
    keep = torch.nonzero(keep_mask, as_tuple=False).reshape(-1)
    sources = torch.as_tensor([value["target"] for value in modes]).long()
    if keep.numel() + sources.numel() != count:
        raise RuntimeError("identity-mode revision changed map capacity")
    raw_modes = F.normalize(
        torch.stack([value["raw"] for value in modes]).float(), dim=1
    )
    metric = load_shared_metric(
        metric_state["path"],
        anchor_ids=torch.as_tensor(state["anchor_ids"]).long(),
        device=device,
    )
    mode_features, _ = metric(raw_modes.to(device))
    mode_features = mode_features.cpu()

    output = dict(state)
    for key, value in state.items():
        if not torch.is_tensor(value) or value.ndim == 0 or value.shape[0] != count:
            continue
        revised = torch.cat([value[keep], value[sources]], dim=0)
        if key == "v7_metric_raw_features":
            revised[-sources.numel() :] = raw_modes.to(revised)
        elif key == "anchor_features":
            revised[-sources.numel() :] = mode_features.to(revised)
        output[key] = revised
    output["anchor_ids"] = torch.arange(count, dtype=torch.long)
    row_origin = torch.cat([keep, sources])
    anchor_type = torch.as_tensor(output["anchor_type"]).long()
    metadata = dict(state["track_centric_reconstruction"])
    original_type = torch.as_tensor(state["anchor_type"]).long()
    original_base_rows = torch.nonzero(
        original_type == 0, as_tuple=False
    ).reshape(-1)
    base_ids = torch.as_tensor(metadata["base_canonical_rows"]).long()
    base_lookup = torch.full((count,), -1, dtype=torch.long)
    base_lookup[original_base_rows] = base_ids
    metadata.update(
        {
            "track_indices": torch.as_tensor(output["track_cluster_ids"])[
                anchor_type != 0
            ].long(),
            "base_canonical_rows": base_lookup[row_origin[anchor_type == 0]],
            "track_anchor_count": int((anchor_type != 0).sum()),
            "base_reserve_count": int((anchor_type == 0).sum()),
            "identity_mode_count": len(modes),
            "identity_mode_alias_rows": torch.arange(
                count - len(modes), count, dtype=torch.long
            ),
            "identity_mode_source_original_rows": sources,
            "identity_mode_retired_original_rows": retire,
            "topology_frozen_after_identity_modes": True,
        }
    )
    output["track_centric_reconstruction"] = metadata
    output["base_anchor_count"] = int((anchor_type == 0).sum())
    output["micro_anchor_count"] = int((anchor_type != 0).sum())
    output["canonical_anchor_count"] = count
    output["requested_micro_anchor_budget"] = int((anchor_type != 0).sum())
    output["provenance"] = {
        **state.get("provenance", {}),
        "identity_mode_revision": {
            "mode_count": len(modes),
            "capacity_preserved": True,
            "uses_test_queries": False,
        },
    }
    revised_metric = dict(metric_state["state"])
    revised_metric["landmark_indices"] = output["anchor_ids"].clone()
    return output, revised_metric, keep, sources


def _remap_csr(
    offsets: torch.Tensor,
    indices: torch.Tensor,
    old_to_new: torch.Tensor,
    alias_lookup: torch.Tensor,
    new_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.as_tensor(offsets).long()
    indices = torch.as_tensor(indices).long()
    row_count = offsets.numel() - 1
    rows = torch.repeat_interleave(torch.arange(row_count), offsets[1:] - offsets[:-1])
    mapped = old_to_new[indices]
    valid = mapped >= 0
    pair_rows = rows[valid]
    pair_values = mapped[valid]
    aliases = alias_lookup[indices]
    alias_valid = aliases >= 0
    if bool(alias_valid.any()):
        pair_rows = torch.cat([pair_rows, rows[alias_valid]])
        pair_values = torch.cat([pair_values, aliases[alias_valid]])
    if pair_rows.numel() == 0:
        return torch.zeros(row_count + 1, dtype=torch.long), torch.empty(
            (0,), dtype=torch.long
        )
    key = pair_rows * int(new_count) + pair_values
    key = torch.unique(key, sorted=True)
    pair_rows = torch.div(key, int(new_count), rounding_mode="floor")
    pair_values = key.remainder(int(new_count))
    counts = torch.bincount(pair_rows, minlength=row_count)
    revised_offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    return revised_offsets, pair_values


def remap_complete_positive_teacher(
    *,
    teacher: dict,
    keep: torch.Tensor,
    sources: torch.Tensor,
    revised_map_path: Path,
) -> dict:
    old_count = int(teacher["anchor_count"])
    new_count = old_count
    old_to_new = torch.full((old_count,), -1, dtype=torch.long)
    old_to_new[keep] = torch.arange(keep.numel())
    alias_lookup = torch.full((old_count,), -1, dtype=torch.long)
    alias_lookup[sources] = torch.arange(
        keep.numel(), keep.numel() + sources.numel(), dtype=torch.long
    )
    records = []
    positive_rows = 0
    strong_pairs = 0
    ambiguous_pairs = 0
    for record in teacher["records"]:
        positive_offsets, positive_indices = _remap_csr(
            record["positive_offsets"],
            record["positive_indices"],
            old_to_new,
            alias_lookup,
            new_count,
        )
        ambiguous_offsets, ambiguous_indices = _remap_csr(
            record["ambiguous_offsets"],
            record["ambiguous_indices"],
            old_to_new,
            alias_lookup,
            new_count,
        )
        positive_rows += int(((positive_offsets[1:] - positive_offsets[:-1]) > 0).sum())
        strong_pairs += int(positive_indices.numel())
        ambiguous_pairs += int(ambiguous_indices.numel())
        records.append(
            {
                **record,
                "positive_offsets": positive_offsets,
                "positive_indices": positive_indices,
                "ambiguous_offsets": ambiguous_offsets,
                "ambiguous_indices": ambiguous_indices,
            }
        )
    return {
        **teacher,
        "anchor_count": new_count,
        "anchor_map": str(revised_map_path),
        "records": records,
        "diagnostics": {
            **teacher.get("diagnostics", {}),
            "positive_rows": positive_rows,
            "strong_pair_count": strong_pairs,
            "ambiguous_pair_count": ambiguous_pairs,
            "identity_mode_alias_count": int(sources.numel()),
        },
    }


def _heldout_gate(before: dict, after: dict) -> dict:
    baseline = before["summary"]
    revised = after["summary"]
    gate = {
        "median_non_degraded": revised["median_te_cm"]
        <= 1.005 * baseline["median_te_cm"],
        "mean_non_degraded": revised["mean_te_cm"]
        <= 1.005 * baseline["mean_te_cm"],
        "p90_non_degraded": revised["p90_te_cm"]
        <= 1.005 * baseline["p90_te_cm"],
        "cvar95_non_degraded": revised["cvar95_te_cm"]
        <= 1.005 * baseline["cvar95_te_cm"],
        "catastrophic_non_degraded": revised["catastrophic_100cm_count"]
        <= baseline["catastrophic_100cm_count"],
        "raw_precision_non_degraded": revised["raw_gt_precision_percent"] + 0.02
        >= baseline["raw_gt_precision_percent"],
    }
    gate["meaningful_improvement"] = bool(
        revised["mean_te_cm"] <= 0.995 * baseline["mean_te_cm"]
        or revised["p90_te_cm"] <= 0.995 * baseline["p90_te_cm"]
        or revised["cvar95_te_cm"] <= 0.995 * baseline["cvar95_te_cm"]
        or revised["raw_gt_precision_percent"]
        >= baseline["raw_gt_precision_percent"] + 0.02
    )
    gate["accepted"] = bool(all(gate.values()))
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--metric-state", type=Path, required=True)
    parser.add_argument("--complete-positive-teacher", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--pose-set-oracle", type=Path, required=True)
    parser.add_argument("--scene-calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deployment-row-limit", type=int, default=0)
    parser.add_argument("--crossfit-blocks", type=int, default=8)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--maximum-clean-samples", type=int, default=8192)
    parser.add_argument("--minimum-mode-observations", type=int, default=5)
    parser.add_argument("--minimum-mode-views", type=int, default=5)
    parser.add_argument("--minimum-stable-folds", type=int, default=2)
    parser.add_argument("--minimum-total-rescues", type=int, default=3)
    parser.add_argument("--assignment-margin", type=float, default=0.01)
    parser.add_argument("--rescue-margin", type=float, default=0.0)
    parser.add_argument("--attraction-margin", type=float, default=0.0)
    parser.add_argument("--maximum-modes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    metric_payload = torch.load(
        args.metric_state, map_location="cpu", weights_only=False
    )
    teacher = torch.load(
        args.complete_positive_teacher, map_location="cpu", weights_only=False
    )
    query_cache = torch.load(args.query_cache, map_location="cpu", weights_only=False)
    oracle = json.loads(args.pose_set_oracle.read_text())
    calibration = json.loads(args.scene_calibration.read_text())["parameters"]
    selection, gate_queries, split = temporal_crossfit_split(
        list(teacher["query_names"]), args.crossfit_blocks
    )
    inner_assignments, inner_split = interleaved_inner_fold_assignments(
        list(teacher["query_names"]),
        selection,
        split["assignments"],
        args.inner_folds,
    )
    target_gains = _oracle_selection_targets(oracle, set(selection))
    if not target_gains:
        raise ValueError("selection split contains no exact pose-critical identity")
    evidence = _collect_selection_evidence(
        state=state,
        metric_state_path=args.metric_state,
        teacher=teacher,
        query_cache=query_cache,
        target_gains=target_gains,
        selection_queries=selection,
        block_assignments=inner_assignments,
        deployment_row_limit=args.deployment_row_limit,
        maximum_clean_samples=args.maximum_clean_samples,
        device=device,
    )
    modes, discovery = discover_identity_modes(
        state=state,
        target_gains=target_gains,
        evidence=evidence,
        minimum_mode_observations=args.minimum_mode_observations,
        minimum_mode_views=args.minimum_mode_views,
        minimum_stable_folds=args.minimum_stable_folds,
        minimum_total_rescues=args.minimum_total_rescues,
        assignment_margin=args.assignment_margin,
        rescue_margin=args.rescue_margin,
        attraction_margin=args.attraction_margin,
        maximum_modes=args.maximum_modes,
    )
    modes, retire, retirement = select_one_for_one_retirements(
        state=state, evidence=evidence, modes=modes
    )
    report = {
        "schema": "lafgs_crossfit_identity_mode_revision",
        "version": 1,
        "uses_test_queries": False,
        "changes_default_mainline": False,
        "source_map": str(args.map.resolve()),
        "metric_state": str(args.metric_state.resolve()),
        "split": split,
        "inner_split": inner_split,
        "discovery": discovery,
        "retirement": retirement,
        "configuration": {
            key: getattr(args, key)
            for key in (
                "deployment_row_limit",
                "crossfit_blocks",
                "inner_folds",
                "maximum_clean_samples",
                "minimum_mode_observations",
                "minimum_mode_views",
                "minimum_stable_folds",
                "minimum_total_rescues",
                "assignment_margin",
                "rescue_margin",
                "attraction_margin",
                "maximum_modes",
                "seed",
            )
        },
    }
    if not modes:
        report["accepted"] = False
        report["stop_reason"] = "no_cross_fitted_zero_attraction_mode"
        (output / "identity_mode_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    metric_wrapper = {"path": args.metric_state, "state": metric_payload}
    revised, revised_metric, keep, sources = materialize_identity_modes(
        state=state,
        metric_state=metric_wrapper,
        modes=modes,
        retire=retire,
        device=device,
    )
    map_path = output / "identity_mode_anchor_map.pt"
    metric_path = output / "identity_mode_metric_state.pt"
    teacher_path = output / "identity_mode_complete_positive_teacher.pt"
    revised_metric["map_path"] = str(map_path)
    revised_teacher = remap_complete_positive_teacher(
        teacher=teacher,
        keep=keep,
        sources=sources,
        revised_map_path=map_path,
    )
    torch.save(revised, map_path)
    torch.save(revised_metric, metric_path)
    torch.save(revised_teacher, teacher_path)

    common = {
        "query_cache": query_cache,
        "device": device,
        "ransac_reprojection_px": float(calibration["ransac_reprojection_px"]),
        "clean_reprojection_px": float(calibration["clean_radius_px"]),
        "task_translation_m": float(calibration["task_translation_m"]),
        "task_rotation_deg": float(calibration["task_rotation_deg"]),
        "seed": int(args.seed),
        "query_indices": gate_queries,
        "deployment_row_limit": int(args.deployment_row_limit),
        "collect_anchor_statistics": False,
    }
    before = collect_deployment_statistics(
        state=state,
        metric_state_path=args.metric_state,
        teacher=teacher,
        progress_label="identity_mode_gate_before",
        **common,
    )
    after = collect_deployment_statistics(
        state=revised,
        metric_state_path=metric_path,
        teacher=revised_teacher,
        progress_label="identity_mode_gate_after",
        **common,
    )
    heldout_gate = _heldout_gate(before, after)
    report.update(
        {
            "revised_map": str(map_path),
            "revised_metric_state": str(metric_path),
            "revised_teacher": str(teacher_path),
            "mode_sources": [int(value["target"]) for value in modes],
            "retired_rows": retire.tolist(),
            "gate_before": before["summary"],
            "gate_after": after["summary"],
            "gate": heldout_gate,
            "accepted": bool(heldout_gate["accepted"]),
            "stop_reason": (
                "heldout_gate_passed"
                if heldout_gate["accepted"]
                else "heldout_pose_gate_failed"
            ),
        }
    )
    (output / "identity_mode_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
