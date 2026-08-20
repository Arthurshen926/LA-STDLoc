#!/usr/bin/env python3
"""Audit view-bin descriptor headroom with mapping-query-local LOO replay.

This command consumes frozen V4 artifacts, never opens the official test split,
and does not materialize a deployable map.  Correct identities are supplied by
the existing Track observations.  Geometry and Anchor selection stay frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from evidence.tracks import (
    LeaveOneQueryOutTrackDescriptorBank,
    robust_fuse_track_descriptors,
)
from evidence.view_mixture import build_view_mixture, mixture_scores
from map_learning.trainer import track_descriptor_payload_for_loo


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "p10": None, "median": None, "p90": None}
    value = torch.tensor(values)
    return {
        "mean": float(value.mean()),
        "p10": float(torch.quantile(value, 0.1)),
        "median": float(value.median()),
        "p90": float(torch.quantile(value, 0.9)),
    }


def _track_observations(replay, local_row: int, excluded_query: int | None = None):
    track = int(replay.track_indices[int(local_row)])
    observations = replay.observation_by_track[track]
    queries = torch.as_tensor(replay.tracks["query_index"])[observations].long()
    keypoints = torch.as_tensor(replay.tracks["keypoint_index"])[observations].long()
    valid = torch.tensor([
        replay.cached_validity[replay.query_names[int(q)]] is None
        or bool(replay.cached_validity[replay.query_names[int(q)]][int(k)])
        for q, k in zip(queries.tolist(), keypoints.tolist())
    ])
    keep = torch.tensor([
        replay.cached_descriptor_keep[replay.query_names[int(q)]] is None
        or bool(replay.cached_descriptor_keep[replay.query_names[int(q)]][int(k)])
        for q, k in zip(queries.tolist(), keypoints.tolist())
    ])
    if bool(valid.any()):
        observations, queries, keypoints, keep = (
            observations[valid], queries[valid], keypoints[valid], keep[valid]
        )
    observations, queries, keypoints = observations[keep], queries[keep], keypoints[keep]
    if excluded_query is not None:
        retained = queries != int(excluded_query)
        observations, queries, keypoints = (
            observations[retained], queries[retained], keypoints[retained]
        )
    if queries.numel() == 0:
        return None
    descriptors = F.normalize(torch.stack([
        replay.cached_descriptors[replay.query_names[int(q)]][int(k)].float()
        for q, k in zip(queries.tolist(), keypoints.tolist())
    ]), dim=1)
    confidence = torch.as_tensor(replay.tracks["confidence"])[observations].float()
    reliability = torch.tensor([
        1.0 if replay.cached_reliability[replay.query_names[int(q)]] is None
        else float(replay.cached_reliability[replay.query_names[int(q)]][int(k)])
        for q, k in zip(queries.tolist(), keypoints.tolist())
    ]).clamp(0, 1)
    return descriptors, replay.query_bins[queries], confidence * reliability, queries, keypoints


def _stream_rank_metrics(
    *, query: torch.Tensor, correct: torch.Tensor, correct_score: torch.Tensor,
    anchor_count: int, chunk_size: int, score_chunk,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rank = torch.ones_like(correct)
    false_max = torch.full_like(correct_score, -torch.inf)
    query_rows = torch.arange(query.shape[0], device=query.device)
    for start in range(0, int(anchor_count), int(chunk_size)):
        stop = min(start + int(chunk_size), int(anchor_count))
        score = score_chunk(start, stop)
        anchors = torch.arange(start, stop, device=query.device)[None]
        better = ((score > correct_score[:, None]) | (
            (score == correct_score[:, None]) & (anchors < correct[:, None])
        ))
        in_chunk = (correct >= start) & (correct < stop)
        if bool(in_chunk.any()):
            better[query_rows[in_chunk], correct[in_chunk] - start] = False
            score[query_rows[in_chunk], correct[in_chunk] - start] = -torch.inf
        rank += better.sum(dim=1)
        false_max = torch.maximum(false_max, score.max(dim=1).values)
    return rank, correct_score - false_max, false_max


def _group_macro(success: torch.Tensor, group_ids: torch.Tensor) -> float:
    _, inverse = torch.unique(group_ids, sorted=True, return_inverse=True)
    totals = torch.zeros(int(inverse.max()) + 1, dtype=torch.float32)
    totals.scatter_add_(0, inverse, success.float())
    counts = torch.bincount(inverse, minlength=totals.numel()).float()
    return float((totals / counts).mean())


@torch.inference_mode()
def run(args) -> dict:
    torch.set_num_threads(int(args.cpu_threads))
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    payload = torch.load(args.track_payload, map_location="cpu", weights_only=False)
    cache = torch.load(args.query_cache, map_location="cpu", weights_only=False)
    if cache.get("uses_test_queries") is not False or payload.get("rendered_rgb_only") is not True:
        raise ValueError("audit requires rendered mapping-only artifacts")
    features = torch.as_tensor(state["anchor_features"]).float()
    if not torch.equal(features, torch.as_tensor(state.get("v7_metric_raw_features", features)).float()):
        raise ValueError("audit requires the identity descriptor bank")
    track_rows_in_unified_map = torch.nonzero(
        torch.as_tensor(state["track_cluster_ids"]).long() >= 0, as_tuple=False
    ).reshape(-1)
    track_features = features[track_rows_in_unified_map]
    replay = LeaveOneQueryOutTrackDescriptorBank(
        payload=track_descriptor_payload_for_loo(payload),
        query_cache=cache,
        track_indices=torch.as_tensor(state["track_cluster_ids"])[track_rows_in_unified_map],
        reference_features=track_features,
        trim_fraction=float(args.trim_fraction),
        validate_reference=False,
    )
    # The exact replay above benefits from bounded parallel fusion.  The audit
    # below consists of thousands of tiny spherical reductions, for which a
    # threaded BLAS launch is substantially slower than one host thread.
    torch.set_num_threads(1)
    track_replay = replay
    anchor_count, dim = track_features.shape
    bin_count = max(1, int(torch.as_tensor(payload["query_bins"]).max()) + 1)
    oracle = torch.zeros((anchor_count, bin_count, dim), dtype=torch.float32)
    oracle_valid = torch.zeros((anchor_count, bin_count), dtype=torch.bool)
    selective = torch.zeros((anchor_count, 2, dim), dtype=torch.float32)
    selective_prior = torch.zeros((anchor_count, 2), dtype=torch.float32)
    selective[:, 0] = F.normalize(track_features, dim=1)
    selective_prior[:, 0] = 1
    oracle[:, 0] = selective[:, 0]
    oracle_valid[:, 0] = True

    full_observations = {}
    eligible = []
    diagnostics = {}
    for local_row in range(anchor_count):
        global_row = local_row
        item = _track_observations(track_replay, local_row)
        full_observations[local_row] = item
        descriptors, bins, weights, _, _ = item
        oracle[global_row].zero_(); oracle_valid[global_row].zero_()
        for view_bin in torch.unique(bins, sorted=True).tolist():
            selected = bins == int(view_bin)
            oracle[global_row, int(view_bin)] = F.normalize(
                (descriptors[selected] * weights[selected, None].clamp_min(1e-8)).sum(0), dim=0
            )
            oracle_valid[global_row, int(view_bin)] = True
        mixture = build_view_mixture(
            descriptors, bins, weights,
            minimum_cluster_observations=args.minimum_cluster_observations,
            minimum_cluster_view_bins=args.minimum_cluster_view_bins,
            minimum_angle_degrees=args.minimum_angle_degrees,
            minimum_loss_improvement=args.minimum_loss_improvement,
        )
        diagnostics[local_row] = mixture
        if mixture.eligible:
            eligible.append((mixture.loss_improvement, global_row, local_row))

    budget_extra = int(math.floor(float(args.maximum_prototype_ratio - 1.0) * anchor_count))
    eligible.sort(key=lambda row: (-row[0], row[1]))
    selected_local = {row[2] for row in eligible[:max(0, budget_extra)]}
    base_eligible_local = {row[2] for row in eligible}
    if len(base_eligible_local) > budget_extra:
        raise ValueError(
            "strict query-local eligibility audit requires an unsaturated prototype budget"
        )
    for local_row in selected_local:
        global_row = int(local_row)
        mixture = diagnostics[local_row]
        selective[global_row] = mixture.prototypes
        selective_prior[global_row] = mixture.priors

    if bool(getattr(args, "sensitivity_only", False)):
        fallback_pairs = []
        for local_row in sorted(base_eligible_local):
            descriptors, bins, weights, queries, _ = full_observations[local_row]
            for query_index in torch.unique(queries, sorted=True).tolist():
                retained = queries != int(query_index)
                eligible_without_query = False
                if bool(retained.any()):
                    eligible_without_query = build_view_mixture(
                        descriptors[retained], bins[retained], weights[retained],
                        minimum_cluster_observations=args.minimum_cluster_observations,
                        minimum_cluster_view_bins=args.minimum_cluster_view_bins,
                        minimum_angle_degrees=args.minimum_angle_degrees,
                        minimum_loss_improvement=args.minimum_loss_improvement,
                    ).eligible
                if not eligible_without_query:
                    fallback_pairs.append((int(query_index), int(local_row)))
        result = {
            "schema": "lafgs_view_mixture_k2_fallback_sensitivity",
            "version": 1,
            "uses_test_queries": False,
            "audit_only": True,
            "inputs": {
                "map": str(args.map.resolve()),
                "track_payload": str(args.track_payload.resolve()),
                "query_cache": str(args.query_cache.resolve()),
            },
            "input_sha256": {
                "map": sha256_file(args.map),
                "track_payload": sha256_file(args.track_payload),
                "query_cache": sha256_file(args.query_cache),
            },
            "configuration": vars(args) | {
                "map": str(args.map), "track_payload": str(args.track_payload),
                "query_cache": str(args.query_cache), "output": str(args.output),
            },
            "base_eligible_k2_count": len(base_eligible_local),
            "query_local_k2_to_k1_fallback_count": len(fallback_pairs),
            "affected_query_count": len({row[0] for row in fallback_pairs}),
            "affected_anchor_count": len({row[1] for row in fallback_pairs}),
            "fallback_pairs": fallback_pairs,
        }
        _atomic_json(args.output, result)
        return result

    device = torch.device(args.device)
    single_base = F.normalize(track_features, dim=1).to(device)
    single = single_base.clone()
    oracle_base, oracle_valid_base = oracle.to(device), oracle_valid.to(device)
    oracle_bank, oracle_mask = oracle_base.clone(), oracle_valid_base.clone()
    selective_base, prior_base = selective.to(device), selective_prior.to(device)
    selective_bank, prior_bank = selective_base.clone(), prior_base.clone()
    previous_rows = torch.empty(0, dtype=torch.long, device=device)
    maximum_query_local_eligible = len(base_eligible_local)
    records = {name: {"rank": [], "margin": [], "false": []} for name in ("single", "view_bin_oracle", "selective_k2")}
    query_rows = []
    positive_query_ids: list[int] = []
    positive_anchor_rows: list[int] = []
    latency = {name: [] for name in records}
    progress_path = args.output.with_name(f".{args.output.name}.progress.pt")
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    requested_start = int(args.query_start)
    query_limit = (
        len(replay.query_names) if int(args.query_end) <= 0
        else min(int(args.query_end), len(replay.query_names))
    )
    if not 0 <= requested_start < query_limit:
        raise ValueError("invalid mapping-query audit range")
    selected_queries = list(range(requested_start, query_limit, int(args.query_stride)))
    if not selected_queries:
        raise ValueError("mapping-query audit sampling is empty")
    query_position = 0
    if progress_path.is_file():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("schema") != "lafgs_view_mixture_audit_progress":
            raise ValueError("unsupported view-mixture progress sidecar")
        if int(progress.get("version", 0)) != 2:
            raise ValueError("progress sidecar predates query-sampling contract")
        if progress.get("selected_queries") != selected_queries:
            raise ValueError("progress sidecar query sampling differs")
        query_position = int(progress["next_query_position"])
        records = progress["records"]
        query_rows = progress["query_rows"]
        positive_query_ids = progress["positive_query_ids"]
        positive_anchor_rows = progress["positive_anchor_rows"]
        latency = progress["latency"]
    for position in range(query_position, len(selected_queries)):
        query_index = selected_queries[position]
        if position % 100 == 0:
            print(
                f"view-mixture mapping LOO query {position}/{len(selected_queries)} "
                f"(global {query_index})",
                flush=True,
            )
        if previous_rows.numel():
            single[previous_rows] = single_base[previous_rows]
            oracle_bank[previous_rows] = oracle_base[previous_rows]
            oracle_mask[previous_rows] = oracle_valid_base[previous_rows]
            selective_bank[previous_rows] = selective_base[previous_rows]
            prior_bank[previous_rows] = prior_base[previous_rows]
        affected_local = track_replay.rows_by_query[query_index]
        changed_rows_list = []
        changed_features_list = []
        for local_row in affected_local:
            descriptors_r, bins_r, weights_r, queries_r, _ = full_observations[int(local_row)]
            retained = queries_r != int(query_index)
            if not bool(retained.any()):
                continue
            changed_rows_list.append(int(local_row))
            changed_features_list.append(
                robust_fuse_track_descriptors(
                    descriptors_r[retained], bins_r[retained], weights_r[retained],
                    trim_fraction=float(args.trim_fraction),
                )
            )
        changed_rows = torch.tensor(changed_rows_list, dtype=torch.long)
        changed_features = (
            torch.stack(changed_features_list)
            if changed_features_list
            else track_features.new_empty((0, dim))
        )
        changed_device = changed_rows.to(device)
        if changed_rows.numel():
            normalized = F.normalize(changed_features.float(), dim=1).to(device)
            single[changed_device] = normalized
            # Surface/non-selected-K2 rows remain exact single-descriptor LOO.
            oracle_bank[changed_device] = 0; oracle_mask[changed_device] = False
            oracle_bank[changed_device, 0] = normalized; oracle_mask[changed_device, 0] = True
            selective_bank[changed_device] = 0; prior_bank[changed_device] = 0
            selective_bank[changed_device, 0] = normalized; prior_bank[changed_device, 0] = 1
        positive_descriptors, positive_rows = [], []
        query_local_eligible = len(base_eligible_local) - sum(
            int(int(row) in base_eligible_local) for row in affected_local
        )
        for local_row in affected_local:
            global_row = int(local_row)
            full = full_observations[int(local_row)]
            descriptors, _, _, queries, _ = full
            selected = queries == int(query_index)
            if bool(selected.any()):
                positive_descriptors.extend(descriptors[selected])
                positive_rows.extend([global_row] * int(selected.sum()))
            retained = queries != int(query_index)
            if not bool(retained.any()):
                continue
            descriptors_r, bins_r, weights_r = descriptors[retained], full[1][retained], full[2][retained]
            oracle_bank[global_row] = 0; oracle_mask[global_row] = False
            for view_bin in torch.unique(bins_r, sorted=True).tolist():
                chosen = bins_r == int(view_bin)
                proto = F.normalize((descriptors_r[chosen] * weights_r[chosen, None].clamp_min(1e-8)).sum(0), dim=0)
                oracle_bank[global_row, int(view_bin)] = proto.to(device)
                oracle_mask[global_row, int(view_bin)] = True
            mixture = build_view_mixture(
                descriptors_r, bins_r, weights_r,
                minimum_cluster_observations=args.minimum_cluster_observations,
                minimum_cluster_view_bins=args.minimum_cluster_view_bins,
                minimum_angle_degrees=args.minimum_angle_degrees,
                minimum_loss_improvement=args.minimum_loss_improvement,
            )
            if mixture.eligible:
                query_local_eligible += 1
                selective_bank[global_row, :2] = mixture.prototypes.to(device)
                prior_bank[global_row, :2] = mixture.priors.to(device)
        if query_local_eligible > budget_extra:
            raise ValueError(
                "query-local K2 eligibility saturates the preregistered prototype budget"
            )
        maximum_query_local_eligible = max(
            maximum_query_local_eligible, query_local_eligible
        )
        previous_rows = changed_device
        if not positive_rows:
            continue
        query = torch.stack(positive_descriptors).to(device)
        correct = torch.tensor(positive_rows, dtype=torch.long, device=device)
        positive_query_ids.extend([int(query_index)] * len(positive_rows))
        positive_anchor_rows.extend(int(row) for row in positive_rows)
        query = F.normalize(query, dim=1)
        score_sets = {}
        correct_single = (query * single[correct]).sum(dim=1)
        correct_oracle_raw = torch.einsum("qd,qbd->qb", query, F.normalize(oracle_bank[correct], dim=2))
        correct_oracle = torch.where(oracle_mask[correct], correct_oracle_raw, torch.full_like(correct_oracle_raw, -torch.inf)).max(1).values
        correct_k2 = mixture_scores(query, selective_bank[correct], prior_bank[correct], temperature=args.temperature).diagonal()
        scoring = {
            "single": (correct_single, lambda a, b: query @ single[a:b].T),
            "view_bin_oracle": (correct_oracle, lambda a, b: torch.where(
                oracle_mask[None, a:b],
                torch.einsum("qd,nbd->qnb", query, F.normalize(oracle_bank[a:b], dim=2)),
                torch.full((query.shape[0], b-a, bin_count), -torch.inf, device=device),
            ).max(2).values),
            "selective_k2": (correct_k2, lambda a, b: mixture_scores(
                query, selective_bank[a:b], prior_bank[a:b], temperature=args.temperature
            )),
        }
        for name, (correct_score, scorer) in scoring.items():
            if device.type == "cuda": torch.cuda.synchronize(device)
            started = time.perf_counter()
            score_sets[name] = _stream_rank_metrics(
                query=query, correct=correct, correct_score=correct_score,
                anchor_count=anchor_count, chunk_size=args.anchor_chunk_size,
                score_chunk=scorer,
            )
            if device.type == "cuda": torch.cuda.synchronize(device)
            latency[name].append(1000 * (time.perf_counter() - started))
        query_record = {"query_index": query_index, "positive_count": len(positive_rows)}
        for name, (rank, margin, false) in score_sets.items():
            records[name]["rank"].extend(rank.cpu().tolist())
            records[name]["margin"].extend(margin.cpu().tolist())
            records[name]["false"].extend(false.cpu().tolist())
            query_record[name] = {"r1": float((rank <= 1).float().mean()), "zero_correct_top1": bool((rank > 1).all())}
        query_rows.append(query_record)
        if (position + 1) % 100 == 0:
            temporary = progress_path.with_name(f".{progress_path.name}.{os.getpid()}.tmp")
            torch.save({
                "schema": "lafgs_view_mixture_audit_progress",
                "version": 2,
                "selected_queries": selected_queries,
                "next_query_position": position + 1,
                "records": records,
                "query_rows": query_rows,
                "positive_query_ids": positive_query_ids,
                "positive_anchor_rows": positive_anchor_rows,
                "latency": latency,
            }, temporary)
            os.replace(temporary, progress_path)

    summary = {}
    baseline_rank = torch.tensor(records["single"]["rank"])
    query_ids = torch.tensor(positive_query_ids, dtype=torch.long)
    anchor_rows = torch.tensor(positive_anchor_rows, dtype=torch.long)
    for name, values in records.items():
        rank = torch.tensor(values["rank"])
        query_macro = {}
        track_macro = {}
        for cutoff in (1, 4, 16):
            success = (rank <= cutoff).float()
            query_macro[str(cutoff)] = _group_macro(success, query_ids)
            track_macro[str(cutoff)] = _group_macro(success, anchor_rows)
        summary[name] = {
            "positive_observation_count": int(rank.numel()),
            "recall_at_1": float((rank <= 1).float().mean()),
            "recall_at_4": float((rank <= 4).float().mean()),
            "recall_at_16": float((rank <= 16).float().mean()),
            "mapping_query_macro_recall": query_macro,
            "track_macro_recall": track_macro,
            "correct_best_wrong_margin": _quantiles(values["margin"]),
            "false_maximum": _quantiles(values["false"]),
            "benefited_fraction_vs_single": float((rank < baseline_rank).float().mean()),
            "harmed_fraction_vs_single": float((rank > baseline_rank).float().mean()),
            "retrieval_proxy_zero_correct_top1_query_count": sum(row[name]["zero_correct_top1"] for row in query_rows),
            "matching_latency_ms": _quantiles(latency[name]),
        }
    result = {
        "schema": "lafgs_view_mixture_mapping_loo_headroom", "version": 1,
        "uses_test_queries": False, "audit_only": True,
        "one_anchor_at_most_one_pnp_vote": True,
        "candidate_registry": "selected_track_anchors_only",
        "surface_completion_policy": "excluded_from_low_cost_track_appearance_headroom_audit",
        "inputs": {"map": str(args.map.resolve()), "track_payload": str(args.track_payload.resolve()), "query_cache": str(args.query_cache.resolve())},
        "input_sha256": {"map": sha256_file(args.map), "track_payload": sha256_file(args.track_payload), "query_cache": sha256_file(args.query_cache)},
        "configuration": vars(args) | {"map": str(args.map), "track_payload": str(args.track_payload), "query_cache": str(args.query_cache), "output": str(args.output)},
        "query_range": {"start": requested_start, "end": query_limit, "stride": int(args.query_stride), "selected_query_count": len(selected_queries), "full_query_count": len(replay.query_names)},
        "prototype_budget": {"anchor_count": anchor_count, "eligible_k2_count": len(eligible), "selected_k2_count": len(selected_local), "prototype_count": anchor_count + len(selected_local), "prototype_ratio": (anchor_count + len(selected_local)) / anchor_count, "query_local_eligibility_recomputed": True, "maximum_query_local_eligible_k2_count": maximum_query_local_eligible, "budget_unsaturated_for_every_query": maximum_query_local_eligible <= budget_extra},
        "summary": summary,
        "query_guard": {"mapping_query_count": len(query_rows), "queries": query_rows},
    }
    records_path = args.output.with_name(f"{args.output.stem}_records.pt")
    temporary_records = records_path.with_name(f".{records_path.name}.{os.getpid()}.tmp")
    torch.save({
        "schema": "lafgs_view_mixture_audit_records", "version": 1,
        "query_range": result["query_range"], "records": records,
        "query_rows": query_rows, "positive_query_ids": positive_query_ids,
        "positive_anchor_rows": positive_anchor_rows, "latency": latency,
    }, temporary_records)
    os.replace(temporary_records, records_path)
    result["records"] = str(records_path.resolve())
    result["records_sha256"] = sha256_file(records_path)
    _atomic_json(args.output, result)
    progress_path.unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--track-payload", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--trim-fraction", type=float, default=0.2)
    parser.add_argument("--minimum-cluster-observations", type=int, default=2)
    parser.add_argument("--minimum-cluster-view-bins", type=int, default=2)
    parser.add_argument("--minimum-angle-degrees", type=float, default=12.0)
    parser.add_argument("--minimum-loss-improvement", type=float, default=0.015)
    parser.add_argument("--maximum-prototype-ratio", type=float, default=1.2)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--anchor-chunk-size", type=int, default=2048)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-end", type=int, default=0)
    parser.add_argument("--query-stride", type=int, default=1)
    parser.add_argument("--sensitivity-only", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result.get("summary", result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
