#!/usr/bin/env python3
"""Cross-fitted one-for-one revision of the Gaussian reserve.

Candidate swaps are proposed from alternating mapping trajectory blocks.  The
complete global top-1 plus PoseLib outcome is then gated on disjoint mapping
blocks.  Track Core rows are never removed and the deployed map size is fixed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evidence.tracks import fuse_track_descriptors
from localization.localizer import load_shared_metric
from map_learning.metric import SharedLowRankMetric
from map_learning.observations import _query_index_remap, build_teacher
from topology.adaptive_distillation import _adaptive_track_eligibility
from topology.deployment_revision import (
    _csr_values,
    collect_deployment_statistics,
)
from topology.matching_coverage import (
    IncrementalBipartiteCoverage,
    base_candidate_edges,
    track_candidate_edges,
)
from topology.track_core import _graph_counter, _materialize


def temporal_crossfit_split(
    query_names: list[str], block_count: int = 8
) -> tuple[list[int], list[int], dict]:
    """Split each mapping trajectory into alternating contiguous blocks."""
    if int(block_count) < 2:
        raise ValueError("cross-fit block count must be at least two")
    by_sequence: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(query_names):
        sequence = str(name).rsplit("/", 1)[0] if "/" in str(name) else ""
        by_sequence[sequence].append(index)
    selection: list[int] = []
    gate: list[int] = []
    assignments = {}
    for sequence, indices in sorted(by_sequence.items()):
        ordered = sorted(indices, key=lambda index: str(query_names[index]))
        size = len(ordered)
        for rank, query in enumerate(ordered):
            block = min(int(rank * int(block_count) / max(size, 1)), block_count - 1)
            assignments[str(query_names[query])] = int(block)
            (selection if block % 2 == 0 else gate).append(query)
    if not selection or not gate:
        raise ValueError("cross-fit split produced an empty partition")
    return sorted(selection), sorted(gate), {
        "policy": "per_sequence_alternating_contiguous_temporal_blocks",
        "block_count": int(block_count),
        "selection_query_count": len(selection),
        "gate_query_count": len(gate),
        "assignments": assignments,
        "uses_test_queries": False,
    }


def _matching_counts(edges, selected: set[int], query_count: int) -> np.ndarray:
    matching = IncrementalBipartiteCoverage(query_count, edges)
    for candidate in sorted(selected):
        matching.add(candidate)
    return matching.counts


def _select_rank_feasible_pairs(
    proposals: list[tuple[int, int, float, int, float]],
    *,
    selected: set[int],
    edges,
    query_count: int,
    matching_rows_target: int,
    maximum_swaps: int,
) -> tuple[list[tuple[int, int, float, int, float]], dict]:
    before = _matching_counts(edges, selected, query_count)
    required = np.minimum(before, int(matching_rows_target))
    accepted = []
    current = set(selected)
    used_remove: set[int] = set()
    used_add: set[int] = set()
    rejected_rank = 0
    for proposal in proposals:
        remove, add = int(proposal[0]), int(proposal[1])
        if remove in used_remove or add in used_add or add in current:
            continue
        trial = set(current)
        trial.remove(remove)
        trial.add(add)
        after = _matching_counts(edges, trial, query_count)
        if bool((after >= required).all()):
            accepted.append(proposal)
            current = trial
            used_remove.add(remove)
            used_add.add(add)
            if len(accepted) >= int(maximum_swaps):
                break
        else:
            rejected_rank += 1
    after = _matching_counts(edges, current, query_count)
    return accepted, {
        "matching_rows_target": int(matching_rows_target),
        "matching_rank_before_p10": float(np.percentile(before, 10)),
        "matching_rank_after_p10": float(np.percentile(after, 10)),
        "matching_rank_before_median": float(np.median(before)),
        "matching_rank_after_median": float(np.median(after)),
        "required_rank_unmet_query_count": int((after < required).sum()),
        "rank_rejected_proposal_count": int(rejected_rank),
    }


@torch.inference_mode()
def propose_swaps(
    *,
    state: dict,
    metric_state_path: str | Path,
    current_teacher: dict,
    canonical: dict,
    base_teacher: dict,
    graph: dict,
    payload: dict,
    query_cache: dict,
    selection_queries: list[int],
    device: torch.device,
    maximum_swaps: int,
    minimum_clean_replacements: int,
    matching_rows_target: int,
) -> tuple[list[tuple[int, int, float, int, float]], dict, dict]:
    """Mine swaps that replace a wrong Gaussian winner with a positive."""
    names = list(current_teacher["query_names"])
    if list(base_teacher["query_names"]) != names:
        raise ValueError("base and compact teacher query registries differ")
    metadata = state["track_centric_reconstruction"]
    selected_tracks = torch.as_tensor(metadata["track_indices"]).long()
    selected_bases = torch.as_tensor(metadata["base_canonical_rows"]).long()
    track_count = int(
        torch.as_tensor(payload["track_geometry"]["triangulated"]).numel()
    )
    base_count = int(canonical["base_anchor_count"])
    selected_global = set(selected_tracks.tolist()) | {
        track_count + int(value) for value in selected_bases.tolist()
    }
    map_to_global = torch.cat((selected_tracks, selected_bases + track_count))

    calibration = metadata["calibration"]
    parameters = calibration["parameters"]
    policy = calibration["policy"]
    broad_tracks = _adaptive_track_eligibility(
        payload["track_geometry"],
        median_px=float(parameters["track_reprojection_median_px"]),
        p90_px=float(parameters["track_reprojection_p90_px"]),
        covariance_m2=float(parameters["track_covariance_trace_m2"]),
        broad=True,
    )
    opportunity = torch.as_tensor(
        graph["provenance_opportunity_count"][:base_count]
    ).float()
    harmful = torch.as_tensor(
        graph["provenance_harmful_solver_inlier_count"][:base_count]
    ).float()
    base_eligible = (
        _graph_counter(
            graph,
            "provenance_legal_hit_strong_count",
            "provenance_legal_hit_2px_count",
        )[:base_count]
        > 0
    ) & (
        harmful / opportunity.clamp_min(1)
        <= float(policy["maximum_harmful_rate"])
    )
    unused_tracks = torch.tensor(
        [
            index
            for index in torch.nonzero(broad_tracks, as_tuple=False).reshape(-1).tolist()
            if index not in selected_global
        ],
        dtype=torch.long,
    )
    unused_bases = torch.tensor(
        [
            index
            for index in torch.nonzero(base_eligible, as_tuple=False).reshape(-1).tolist()
            if track_count + index not in selected_global
        ],
        dtype=torch.long,
    )
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        track_raw = (
            fuse_track_descriptors(
                payload=payload,
                query_cache=query_cache,
                track_indices=unused_tracks,
                trim_fraction=float(policy["descriptor_trim_fraction"]),
            )
            if unused_tracks.numel()
            else torch.empty((0, state["anchor_features"].shape[1]))
        )
    finally:
        torch.set_num_threads(original_threads)
    base_raw = torch.as_tensor(canonical["anchor_features"])[unused_bases].float()
    candidate_ids = torch.cat((unused_tracks, unused_bases + track_count))
    candidate_raw = F.normalize(torch.cat((track_raw, base_raw)).float(), dim=1)
    metric = load_shared_metric(
        metric_state_path,
        anchor_ids=torch.as_tensor(state["anchor_ids"]).long(),
        device=device,
    )
    candidate_features, _ = metric(candidate_raw.to(device))
    candidate_features = F.normalize(candidate_features, dim=1)
    candidate_local = {
        int(candidate): local for local, candidate in enumerate(candidate_ids.tolist())
    }

    payload_to_teacher = _query_index_remap(payload["query_names"], names)
    edges = [
        *track_candidate_edges(payload, query_index_remap=payload_to_teacher),
        *base_candidate_edges(base_teacher, base_count),
    ]
    selection_set = set(int(value) for value in selection_queries)
    reverse: list[dict[int, list[int]]] = [defaultdict(list) for _ in names]
    for candidate in candidate_ids.tolist():
        local = candidate_local[int(candidate)]
        for query, rows in edges[int(candidate)].items():
            if int(query) not in selection_set:
                continue
            for row in rows:
                reverse[int(query)][int(row)].append(local)

    cache = query_cache.get("queries", query_cache)
    bank = F.normalize(
        torch.as_tensor(state["anchor_features"]).float().to(device), dim=1
    )
    anchor_type = torch.as_tensor(state["anchor_type"]).long()
    aggregate: dict[tuple[int, int], list[float]] = defaultdict(list)
    eligible_wrong_rows = 0
    for query in selection_queries:
        record = current_teacher["records"][query]
        cached = cache[names[query]]
        rows = torch.as_tensor(record["query_rows"]).long()
        descriptor = F.normalize(
            torch.as_tensor(cached["native_descriptors"]).float()[rows], dim=1
        ).to(device)
        adapted, _ = metric(descriptor)
        scores, winners = torch.topk(adapted @ bank.T, k=2, dim=1)
        scores = scores.cpu()
        winners = winners.cpu()
        for local_row, native_row in enumerate(rows.tolist()):
            winner = int(winners[local_row, 0])
            if int(anchor_type[winner]) != 0:
                continue
            positives = _csr_values(record, "positive", local_row)
            if bool((positives == winner).any()) or not positives.numel():
                continue
            candidate_rows = reverse[query].get(int(native_row), [])
            if not candidate_rows:
                continue
            eligible_wrong_rows += 1
            local_candidates = torch.as_tensor(candidate_rows, device=device)
            replacement_scores = (
                adapted[local_row] @ candidate_features[local_candidates].T
            ).cpu()
            best_position = int(torch.argmax(replacement_scores))
            best_score = float(replacement_scores[best_position])
            alternative_score = float(scores[local_row, 1])
            if best_score <= alternative_score:
                continue
            remove = int(map_to_global[winner])
            add = int(candidate_ids[candidate_rows[best_position]])
            aggregate[(remove, add)].append(best_score - alternative_score)
    proposals = []
    for (remove, add), margins in aggregate.items():
        if len(margins) < int(minimum_clean_replacements):
            continue
        score = float(len(margins) + np.mean(margins))
        proposals.append((remove, add, score, len(margins), float(np.mean(margins))))
    proposals.sort(key=lambda value: (-value[2], value[0], value[1]))
    accepted, matching = _select_rank_feasible_pairs(
        proposals,
        selected=selected_global,
        edges=edges,
        query_count=len(names),
        matching_rows_target=matching_rows_target,
        maximum_swaps=maximum_swaps,
    )
    report = {
        "candidate_track_count": int(unused_tracks.numel()),
        "candidate_gaussian_count": int(unused_bases.numel()),
        "eligible_wrong_row_count": int(eligible_wrong_rows),
        "proposed_pair_count": len(proposals),
        "accepted_pair_count": len(accepted),
        "minimum_clean_replacements": int(minimum_clean_replacements),
        "maximum_swaps": int(maximum_swaps),
        "matching": matching,
    }
    candidate_payload = {
        "candidate_ids": candidate_ids,
        "candidate_raw_features": candidate_raw.cpu(),
        "edges": edges,
    }
    return accepted, report, candidate_payload


def materialize_swaps(
    *,
    state: dict,
    metric_state: dict,
    canonical: dict,
    payload: dict,
    candidate_payload: dict,
    swaps: list[tuple[int, int, float, int, float]],
    canonical_path: Path,
    payload_path: Path,
    device: torch.device,
) -> tuple[dict, dict]:
    metadata = state["track_centric_reconstruction"]
    current_tracks = torch.as_tensor(metadata["track_indices"]).long().tolist()
    current_bases = torch.as_tensor(metadata["base_canonical_rows"]).long().tolist()
    track_count = int(
        torch.as_tensor(payload["track_geometry"]["triangulated"]).numel()
    )
    remove = {int(value[0]) for value in swaps}
    additions = [int(value[1]) for value in swaps]
    tracks = [*current_tracks, *[value for value in additions if value < track_count]]
    bases = [
        value for value in current_bases if track_count + int(value) not in remove
    ]
    bases.extend(value - track_count for value in additions if value >= track_count)
    if len(tracks) + len(bases) != int(torch.as_tensor(state["anchor_ids"]).numel()):
        raise RuntimeError("one-for-one swap changed map capacity")

    current_raw = torch.as_tensor(state["v7_metric_raw_features"]).float()
    existing_track_raw = {
        int(track): current_raw[index]
        for index, track in enumerate(current_tracks)
    }
    candidate_ids = torch.as_tensor(candidate_payload["candidate_ids"]).long()
    candidate_raw = torch.as_tensor(candidate_payload["candidate_raw_features"]).float()
    candidate_lookup = {
        int(candidate): candidate_raw[index]
        for index, candidate in enumerate(candidate_ids.tolist())
    }
    track_raw = torch.stack(
        [
            existing_track_raw[track]
            if track in existing_track_raw
            else candidate_lookup[track]
            for track in tracks
        ]
    )
    revised = _materialize(
        canonical,
        payload,
        torch.as_tensor(tracks),
        track_raw,
        torch.as_tensor(bases),
        budget=len(tracks) + len(bases),
        quality_tier="adaptive_crossfit_swap",
        source_map=canonical_path,
        payload_path=payload_path,
        dependency_voxel_size=float(metadata["dependency_voxel_size"]),
        separate_spatial_dependency=True,
    )
    revised_metadata = dict(metadata)
    revised_metadata.update(revised["track_centric_reconstruction"])
    revised_metadata.update(
        {
            "track_anchor_count": len(tracks),
            "base_reserve_count": len(bases),
            "final_track_count": len(tracks),
            "final_base_count": len(bases),
            "crossfit_swap_count": len(swaps),
        }
    )
    revised["track_centric_reconstruction"] = revised_metadata
    if "v7_online_metric" in state:
        revised["v7_online_metric"] = state["v7_online_metric"]
    revised["provenance"] = {
        **state.get("provenance", {}),
        "crossfit_swap_revision": {
            "swap_count": len(swaps),
            "uses_test_queries": False,
        },
    }
    raw = F.normalize(
        torch.cat(
            (
                track_raw,
                torch.as_tensor(canonical["anchor_features"])[torch.as_tensor(bases)],
            )
        ).float(),
        dim=1,
    )
    metric = SharedLowRankMetric(**metric_state["metric_config"]).to(device)
    metric.load_state_dict(metric_state["metric_state_dict"])
    metric.eval()
    with torch.inference_mode():
        transformed, _ = metric(raw.to(device))
    revised["v7_metric_raw_features"] = raw.cpu()
    revised["anchor_features"] = transformed.cpu()
    revised_metric = dict(metric_state)
    revised_metric["landmark_indices"] = revised["anchor_ids"].clone()
    revised_metric["map_path"] = None
    return revised, revised_metric


def _heldout_gate(before: dict, after: dict) -> dict:
    before_summary, after_summary = before["summary"], after["summary"]
    gate = {
        "median_non_degraded": after_summary["median_te_cm"]
        <= 1.02 * before_summary["median_te_cm"],
        "mean_non_degraded": after_summary["mean_te_cm"]
        <= 1.02 * before_summary["mean_te_cm"],
        "p90_non_degraded": after_summary["p90_te_cm"]
        <= 1.02 * before_summary["p90_te_cm"],
        "cvar95_non_degraded": after_summary["cvar95_te_cm"]
        <= 1.02 * before_summary["cvar95_te_cm"],
        "catastrophic_non_degraded": after_summary["catastrophic_100cm_count"]
        <= before_summary["catastrophic_100cm_count"],
        "raw_precision_non_degraded": after_summary["raw_gt_precision_percent"]
        + 0.05
        >= before_summary["raw_gt_precision_percent"],
    }
    gate["meaningful_improvement"] = bool(
        after_summary["raw_gt_precision_percent"]
        >= before_summary["raw_gt_precision_percent"] + 0.01
        or after_summary["mean_te_cm"] <= 0.995 * before_summary["mean_te_cm"]
        or after_summary["p90_te_cm"] <= 0.995 * before_summary["p90_te_cm"]
        or after_summary["cvar95_te_cm"] <= 0.995 * before_summary["cvar95_te_cm"]
    )
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--complete-positive-teacher", required=True)
    parser.add_argument("--canonical-map", required=True)
    parser.add_argument("--base-positive-teacher", required=True)
    parser.add_argument("--function-graph", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--raster-provenance", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--matching-rows-target", type=int, required=True)
    parser.add_argument("--ransac-reprojection-px", type=float, required=True)
    parser.add_argument("--clean-reprojection-px", type=float, required=True)
    parser.add_argument("--maximum-swaps", type=int, default=16)
    parser.add_argument("--minimum-clean-replacements", type=int, default=2)
    parser.add_argument("--crossfit-blocks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    map_path = Path(args.map).resolve()
    metric_path = Path(args.metric_state).resolve()
    canonical_path = Path(args.canonical_map).resolve()
    payload_path = Path(args.track_payload).resolve()
    state = torch.load(map_path, map_location="cpu", weights_only=False)
    metric_state = torch.load(metric_path, map_location="cpu", weights_only=False)
    canonical = torch.load(canonical_path, map_location="cpu", weights_only=False)
    current_teacher = torch.load(
        args.complete_positive_teacher, map_location="cpu", weights_only=False
    )
    base_teacher = torch.load(
        args.base_positive_teacher, map_location="cpu", weights_only=False
    )
    graph = torch.load(args.function_graph, map_location="cpu", weights_only=False)
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    query_cache = torch.load(args.query_cache, map_location="cpu", weights_only=False)
    provenance = torch.load(
        args.raster_provenance, map_location="cpu", weights_only=False
    )
    device = torch.device(args.device)
    selection_queries, gate_queries, split_report = temporal_crossfit_split(
        list(current_teacher["query_names"]), args.crossfit_blocks
    )
    swaps, proposal_report, candidate_payload = propose_swaps(
        state=state,
        metric_state_path=metric_path,
        current_teacher=current_teacher,
        canonical=canonical,
        base_teacher=base_teacher,
        graph=graph,
        payload=payload,
        query_cache=query_cache,
        selection_queries=selection_queries,
        device=device,
        maximum_swaps=args.maximum_swaps,
        minimum_clean_replacements=args.minimum_clean_replacements,
        matching_rows_target=args.matching_rows_target,
    )
    revised, revised_metric = materialize_swaps(
        state=state,
        metric_state=metric_state,
        canonical=canonical,
        payload=payload,
        candidate_payload=candidate_payload,
        swaps=swaps,
        canonical_path=canonical_path,
        payload_path=payload_path,
        device=device,
    )
    revised_map_path = output / "revised_anchor_map.pt"
    revised_metric_path = output / "revised_metric_state.pt"
    revised_teacher_path = output / "revised_complete_positive_teacher.pt"
    revised_metric["map_path"] = str(revised_map_path)
    torch.save(revised, revised_map_path)
    torch.save(revised_metric, revised_metric_path)
    teacher_config = current_teacher["config"]
    revised_teacher = build_teacher(
        anchor_map=revised,
        query_cache=query_cache,
        provenance=provenance,
        track_payload=payload,
        device=device,
        strong_radius_px=float(teacher_config["strong_radius_px"]),
        ambiguous_radius_px=float(teacher_config["ambiguous_radius_px"]),
        depth_abs_tolerance_m=float(teacher_config["depth_abs_tolerance_m"]),
        depth_rel_tolerance=float(teacher_config["depth_rel_tolerance"]),
        alpha_minimum=float(teacher_config["alpha_minimum"]),
        contribution_minimum=float(teacher_config["contribution_minimum"]),
    )
    revised_teacher.update(
        {
            "anchor_map": str(revised_map_path),
            "query_cache": str(Path(args.query_cache).resolve()),
            "raster_provenance": str(Path(args.raster_provenance).resolve()),
            "track_payload": str(payload_path),
        }
    )
    torch.save(revised_teacher, revised_teacher_path)

    common = {
        "query_cache": query_cache,
        "device": device,
        "ransac_reprojection_px": args.ransac_reprojection_px,
        "clean_reprojection_px": args.clean_reprojection_px,
        "task_translation_m": float(
            state["track_centric_reconstruction"]["calibration"]["parameters"][
                "task_translation_m"
            ]
        ),
        "task_rotation_deg": float(
            state["track_centric_reconstruction"]["calibration"]["parameters"][
                "task_rotation_deg"
            ]
        ),
        "seed": args.seed,
    }
    before_selection = collect_deployment_statistics(
        state=state,
        metric_state_path=metric_path,
        teacher=current_teacher,
        query_indices=selection_queries,
        progress_label="before_selection_replay",
        **common,
    )
    after_selection = collect_deployment_statistics(
        state=revised,
        metric_state_path=revised_metric_path,
        teacher=revised_teacher,
        query_indices=selection_queries,
        progress_label="after_selection_replay",
        **common,
    )
    before_gate = collect_deployment_statistics(
        state=state,
        metric_state_path=metric_path,
        teacher=current_teacher,
        query_indices=gate_queries,
        progress_label="before_gate_replay",
        **common,
    )
    after_gate = collect_deployment_statistics(
        state=revised,
        metric_state_path=revised_metric_path,
        teacher=revised_teacher,
        query_indices=gate_queries,
        progress_label="after_gate_replay",
        **common,
    )
    gate = _heldout_gate(before_gate, after_gate)
    accepted = bool(swaps) and all(gate.values())
    report = {
        "schema": "lafgs_crossfit_swap_revision",
        "version": 1,
        "uses_test_queries": False,
        "source_map": str(map_path),
        "revised_map": str(revised_map_path),
        "revised_metric_state": str(revised_metric_path),
        "revised_teacher": str(revised_teacher_path),
        "split": split_report,
        "proposal": proposal_report,
        "swaps": [
            {
                "remove_universe_id": int(value[0]),
                "add_universe_id": int(value[1]),
                "assignment_gain_score": float(value[2]),
                "clean_replacement_rows": int(value[3]),
                "mean_score_margin": float(value[4]),
            }
            for value in swaps
        ],
        "selection_before": before_selection["summary"],
        "selection_after": after_selection["summary"],
        "gate_before": before_gate["summary"],
        "gate_after": after_gate["summary"],
        "gate": gate,
        "accepted": accepted,
    }
    (output / "crossfit_swap_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
