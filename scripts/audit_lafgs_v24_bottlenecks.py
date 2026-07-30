#!/usr/bin/env python3
"""Audit positive coverage, rescue value, metric collateral, and OOF pair scope."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from localization_training.positive_retrieval import (
    RANK_BUCKET_NAMES,
    csr_candidate_positive_mask,
    strict_positive_rank_buckets,
)
from localization_training.shared_metric import SharedLowRankMetric


def _records_by_name(payload: dict) -> dict[str, dict]:
    return {
        str(name): record
        for name, record in zip(payload["query_names"], payload["records"])
    }


def _dynamic_by_name(payload: dict) -> dict[str, dict]:
    return {
        str(record["query_name"]): record for record in payload["records"]
    }


def _replay_by_name(path: str) -> dict[str, dict]:
    payload = json.loads(Path(path).read_text())
    return {str(record["query"]): record for record in payload["results"]}


def _sha256_tensor(value: torch.Tensor) -> str:
    value = torch.as_tensor(value).detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _atomic_torch(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _positive_sets(record: dict) -> list[set[int]]:
    offsets = torch.as_tensor(record["positive_offsets"]).long()
    indices = torch.as_tensor(record["positive_indices"]).long()
    return [
        set(indices[offsets[row] : offsets[row + 1]].tolist())
        for row in range(offsets.numel() - 1)
    ]


def _nested_counter(counter: dict) -> dict:
    return {
        str(group): dict(sorted(values.items()))
        for group, values in sorted(counter.items(), key=lambda value: str(value[0]))
    }


def _te_bucket(te_cm: float, median: float, p90: float) -> str:
    if te_cm <= median:
        return "te_le_median"
    if te_cm < p90:
        return "te_median_to_p90"
    return "te_ge_p90"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--active-positive-teacher", required=True)
    parser.add_argument("--canonical-positive-teacher", required=True)
    parser.add_argument("--baseline-topk", required=True)
    parser.add_argument("--baseline-dynamic", required=True)
    parser.add_argument("--baseline-replay", required=True)
    parser.add_argument("--candidate-topk", required=True)
    parser.add_argument("--oracle-topk", required=True)
    parser.add_argument("--oracle-dynamic", required=True)
    parser.add_argument("--oracle-replay", required=True)
    parser.add_argument("--edge-rescue-topk", required=True)
    parser.add_argument("--confusion-graph", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--output-oof-oracle-topk", required=True)
    parser.add_argument("--topk-audit", type=int, default=64)
    parser.add_argument("--minimum-oof-events", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    metric_payload = torch.load(
        args.metric_state, map_location="cpu", weights_only=False
    )
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    active = torch.load(
        args.active_positive_teacher, map_location="cpu", weights_only=False
    )
    canonical = torch.load(
        args.canonical_positive_teacher,
        map_location="cpu",
        weights_only=False,
    )
    baseline = torch.load(
        args.baseline_topk, map_location="cpu", weights_only=False
    )
    baseline_dynamic = torch.load(
        args.baseline_dynamic, map_location="cpu", weights_only=False
    )
    candidate = torch.load(
        args.candidate_topk, map_location="cpu", weights_only=False
    )
    oracle = torch.load(
        args.oracle_topk, map_location="cpu", weights_only=False
    )
    oracle_dynamic = torch.load(
        args.oracle_dynamic, map_location="cpu", weights_only=False
    )
    proposal = torch.load(
        args.edge_rescue_topk, map_location="cpu", weights_only=False
    )
    graph = torch.load(
        args.confusion_graph, map_location="cpu", weights_only=False
    )
    track = torch.load(
        args.track_payload, map_location="cpu", weights_only=False
    )
    names = [str(value) for value in baseline["query_names"]]
    for label, payload in (
        ("active teacher", active),
        ("canonical teacher", canonical),
        ("candidate", candidate),
        ("oracle", oracle),
        ("proposal", proposal),
    ):
        if [str(value) for value in payload["query_names"]] != names:
            raise ValueError(f"{label} query registry does not align")
    anchor_count = int(torch.as_tensor(state["anchor_xyz"]).shape[0])
    if int(active["anchor_count"]) != anchor_count:
        raise ValueError("active teacher does not align with map")

    bank = F.normalize(
        torch.as_tensor(state["anchor_features"]).float(), dim=1
    ).to(device)
    metric = SharedLowRankMetric(**metric_payload["metric_config"]).to(device)
    metric.load_state_dict(metric_payload["metric_state_dict"])
    metric.eval()
    active_by_name = _records_by_name(active)
    canonical_by_name = _records_by_name(canonical)
    baseline_by_name = _records_by_name(baseline)
    candidate_by_name = _records_by_name(candidate)
    oracle_by_name = _records_by_name(oracle)
    proposal_by_name = _records_by_name(proposal)
    dynamic_by_name = _dynamic_by_name(baseline_dynamic)
    oracle_dynamic_by_name = _dynamic_by_name(oracle_dynamic)
    baseline_replay = _replay_by_name(args.baseline_replay)
    oracle_replay = _replay_by_name(args.oracle_replay)
    query_bins = {
        str(name): int(value)
        for name, value in zip(
            track["query_names"],
            torch.as_tensor(track["query_bins"]).tolist(),
        )
    }
    te_values = np.asarray(
        [float(dynamic_by_name[name]["te_cm"]) for name in names]
    )
    te_median = float(np.median(te_values))
    te_p90 = float(np.percentile(te_values, 90))

    rank_total = Counter()
    rank_by_sequence: dict[str, Counter] = defaultdict(Counter)
    rank_by_view_bin: dict[int, Counter] = defaultdict(Counter)
    rank_by_te: dict[str, Counter] = defaultdict(Counter)
    collateral_total = Counter()
    collateral_by_sequence: dict[str, Counter] = defaultdict(Counter)
    per_query_rank = []
    oof_records = []
    oof_accepted = Counter()

    edge_by_pair = {
        (int(edge["correct_anchor"]), int(edge["confusing_anchor"])): int(
            edge["edge_index"]
        )
        for edge in graph["edges"]
    }
    event_trajectories: dict[int, list[str]] = defaultdict(list)
    for event in graph["events"]:
        event_trajectories[int(event["edge_index"])].append(
            str(event["trajectory"])
        )

    dependency = torch.as_tensor(
        state.get("coarse_dependency_group_ids", state["dependency_group_ids"])
    ).long()
    rescue_value = Counter()
    rescue_query_names = set()
    with torch.inference_mode():
        for query_index, name in enumerate(names):
            active_record = active_by_name[name]
            canonical_record = canonical_by_name[name]
            base_record = baseline_by_name[name]
            candidate_record = candidate_by_name[name]
            oracle_record = oracle_by_name[name]
            proposal_record = proposal_by_name[name]
            rows = torch.as_tensor(base_record["query_rows"]).long()
            for label, record in (
                ("active", active_record),
                ("canonical", canonical_record),
                ("candidate", candidate_record),
                ("oracle", oracle_record),
                ("proposal", proposal_record),
            ):
                if not torch.equal(
                    rows, torch.as_tensor(record["query_rows"]).long()
                ):
                    raise ValueError(f"{label} rows do not align for {name}")
            base_indices = torch.as_tensor(
                base_record["topk_anchor_indices"]
            ).long()
            active_offsets = torch.as_tensor(
                active_record["positive_offsets"]
            ).long()
            active_indices = torch.as_tensor(
                active_record["positive_indices"]
            ).long()
            top16_positive = csr_candidate_positive_mask(
                base_indices, active_offsets, active_indices
            )
            cached = cache[name]
            descriptor = F.normalize(
                torch.as_tensor(cached["native_descriptors"]).float()[
                    rows
                ].to(device),
                dim=1,
            )
            transformed, _ = metric(descriptor)
            top64 = torch.topk(
                transformed @ bank.T,
                k=min(int(args.topk_audit), anchor_count),
                dim=1,
            ).indices.cpu()
            top64_positive = csr_candidate_positive_mask(
                top64, active_offsets, active_indices
            )
            bucket = strict_positive_rank_buckets(
                active_offsets=active_offsets,
                canonical_offsets=torch.as_tensor(
                    canonical_record["positive_offsets"]
                ).long(),
                top16_positive_mask=top16_positive,
                top64_positive_mask=top64_positive,
            )
            sequence = name.split("/", 1)[0]
            view_bin = query_bins[name]
            te_bucket = _te_bucket(
                float(dynamic_by_name[name]["te_cm"]), te_median, te_p90
            )
            query_counter = Counter()
            for bucket_id, count in zip(
                *torch.unique(bucket, return_counts=True)
            ):
                label = RANK_BUCKET_NAMES[int(bucket_id)]
                value = int(count)
                rank_total[label] += value
                rank_by_sequence[sequence][label] += value
                rank_by_view_bin[view_bin][label] += value
                rank_by_te[te_bucket][label] += value
                query_counter[label] += value
            per_query_rank.append(
                {
                    "query": name,
                    "sequence": sequence,
                    "view_bin": view_bin,
                    "te_cm": float(dynamic_by_name[name]["te_cm"]),
                    "counts": dict(query_counter),
                }
            )

            positive_sets = _positive_sets(active_record)
            candidate_indices = torch.as_tensor(
                candidate_record["topk_anchor_indices"]
            ).long()[:, 0]
            base_top1 = base_indices[:, 0]
            changed = candidate_indices != base_top1
            for slot in torch.nonzero(changed, as_tuple=False).reshape(-1).tolist():
                positives = positive_sets[slot]
                base_positive = int(base_top1[slot]) in positives
                new_positive = int(candidate_indices[slot]) in positives
                if not positives:
                    category = "neutral_no_top16_positive"
                elif base_positive:
                    category = "protected_positive"
                elif bool(top16_positive[slot].any()):
                    category = "rescue_eligible"
                else:
                    category = "positive_beyond_top16"
                transition = (
                    "positive_to_wrong"
                    if base_positive and not new_positive
                    else "wrong_to_positive"
                    if not base_positive and new_positive
                    else "positive_to_positive"
                    if base_positive and new_positive
                    else "wrong_to_wrong"
                )
                key = f"{category}:{transition}"
                collateral_total[key] += 1
                collateral_by_sequence[sequence][key] += 1

            oracle_indices = torch.as_tensor(
                oracle_record["topk_anchor_indices"]
            ).long()[:, 0]
            oracle_changed = oracle_indices != base_top1
            oracle_outcome = oracle_dynamic_by_name[name]
            oracle_inlier = torch.as_tensor(
                oracle_outcome["ransac_inlier_mask"]
            ).bool()
            oracle_error = torch.as_tensor(
                oracle_outcome["gt_reprojection_errors_px"]
            ).float()
            base_outcome = dynamic_by_name[name]
            base_clean_inlier = torch.as_tensor(
                base_outcome["clean_inlier_mask"]
            ).bool()
            clean_groups = set(
                dependency[base_top1[base_clean_inlier]].tolist()
            )
            if bool(oracle_changed.any()):
                rescue_query_names.add(name)
            for slot in torch.nonzero(
                oracle_changed, as_tuple=False
            ).reshape(-1).tolist():
                rescue_value["accepted"] += 1
                rescue_value["gt_clean_2px"] += int(
                    float(oracle_error[slot]) <= 2.0
                )
                rescue_value["ransac_inlier"] += int(oracle_inlier[slot])
                rescue_value["redundant_dependency_group"] += int(
                    int(dependency[oracle_indices[slot]]) in clean_groups
                )

            proposed_indices = torch.as_tensor(
                proposal_record["topk_anchor_indices"]
            ).long()
            proposed_scores = torch.as_tensor(
                proposal_record["topk_scores"]
            ).float()
            output_top1 = base_top1.clone()
            output_score = torch.as_tensor(
                base_record["topk_scores"]
            ).float()[:, 0].clone()
            heldout_trajectory = sequence
            proposed_changed = proposed_indices[:, 0] != base_top1
            for slot in torch.nonzero(
                proposed_changed, as_tuple=False
            ).reshape(-1).tolist():
                correct = int(proposed_indices[slot, 0])
                confusing = int(base_top1[slot])
                edge = edge_by_pair.get((correct, confusing))
                if edge is None:
                    continue
                support = sum(
                    trajectory != heldout_trajectory
                    for trajectory in event_trajectories[edge]
                )
                if support < int(args.minimum_oof_events):
                    continue
                positives = positive_sets[slot]
                if confusing in positives or correct not in positives:
                    continue
                output_top1[slot] = correct
                output_score[slot] = proposed_scores[slot, 0]
                oof_accepted["total"] += 1
                oof_accepted[sequence] += 1
            oof_records.append(
                {
                    "query_name": name,
                    "query_rows": rows,
                    "topk_anchor_indices": output_top1[:, None],
                    "topk_scores": output_score[:, None],
                }
            )
            if (query_index + 1) % 50 == 0:
                print(f"audit {query_index + 1}/{len(names)}", flush=True)

    for name in rescue_query_names:
        before = float(baseline_replay[name]["te_cm"])
        after = float(oracle_replay[name]["te_cm"])
        rescue_value["queries"] += 1
        rescue_value["query_pose_improved"] += int(after < before)
        rescue_value["query_pose_worsened"] += int(after > before)
        rescue_value["query_pose_equal"] += int(after == before)
    oof_path = Path(args.output_oof_oracle_topk).resolve()
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch(
        oof_path,
        {
            "schema": "lafgs_exact_topk_outcomes",
            "version": 1,
            "query_names": names,
            "query_start": 0,
            "topk": 1,
            "anchor_count": anchor_count,
            "anchor_ids_sha256": _sha256_tensor(state["anchor_ids"]),
            "method": "cross_trajectory_oof_pair_strict_oracle",
            "records": oof_records,
            "provenance": {
                "baseline_topk": str(Path(args.baseline_topk).resolve()),
                "edge_rescue_topk": str(
                    Path(args.edge_rescue_topk).resolve()
                ),
                "confusion_graph": str(
                    Path(args.confusion_graph).resolve()
                ),
                "gt_on_off_only": True,
                "minimum_oof_events": int(args.minimum_oof_events),
            },
        },
    )
    report = {
        "schema": "lafgs_v24_bottleneck_audit",
        "version": 1,
        "query_count": len(names),
        "row_count": int(sum(rank_total.values())),
        "top16_no_positive_decomposition": {
            "total": dict(rank_total),
            "by_sequence": _nested_counter(rank_by_sequence),
            "by_view_bin": _nested_counter(rank_by_view_bin),
            "by_te_bucket": _nested_counter(rank_by_te),
            "te_median_cm": te_median,
            "te_p90_cm": te_p90,
            "per_query": per_query_rank,
        },
        "strict_oracle_rescue_value": dict(rescue_value),
        "v23_global_metric_collateral": {
            "total": dict(collateral_total),
            "by_sequence": _nested_counter(collateral_by_sequence),
        },
        "cross_trajectory_oof_pair_oracle": {
            "accepted": dict(oof_accepted),
            "topk": str(oof_path),
            "gt_used_only_for_pair_on_off": True,
        },
    }
    report_path = Path(args.output_report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(report_path, report)
    print(json.dumps({**report, "top16_no_positive_decomposition": {
        **report["top16_no_positive_decomposition"],
        "per_query": f"{len(per_query_rank)} records",
    }}, indent=2))


if __name__ == "__main__":
    main()
