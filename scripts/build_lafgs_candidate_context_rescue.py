#!/usr/bin/env python3
"""Build frozen-O3 candidate rescue and strict pair-oracle assignments."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from localization_training.candidate_context_rescue import (
    candidate_conditioned_rescue,
    candidate_conditioned_rescue_from_edge_keys,
    directed_edge_keys,
    events_by_query_row,
    oracle_acceptance_mask,
)
from localization_training.contextual_descriptor import BoundedContextProjector
from localization_training.relational_context import (
    relational_sparse_query_context,
)
from localization_training.shared_metric import SharedLowRankMetric


def _atomic_torch(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _sha256_tensor(value: torch.Tensor) -> str:
    value = torch.as_tensor(value).detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _project_errors(
    xyz: torch.Tensor,
    keypoints: torch.Tensor,
    K: torch.Tensor,
    pose_w2c: torch.Tensor,
) -> torch.Tensor:
    camera = xyz @ pose_w2c[:3, :3].T + pose_w2c[:3, 3]
    projected = camera[:, :2] / camera[:, 2:3].clamp_min(1e-8)
    projected = projected @ K[:2, :2].T + K[:2, 2]
    errors = torch.linalg.vector_norm(projected - keypoints, dim=1)
    errors[camera[:, 2] <= 1e-6] = torch.inf
    return errors


def _restore_rejected(
    rescued_indices: torch.Tensor,
    rescued_scores: torch.Tensor,
    baseline_indices: torch.Tensor,
    baseline_scores: torch.Tensor,
    accepted: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = baseline_indices.clone()
    scores = baseline_scores.clone()
    indices[accepted] = rescued_indices[accepted]
    scores[accepted] = rescued_scores[accepted]
    return indices, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--baseline-topk", required=True)
    parser.add_argument("--confusion-graph", required=True)
    parser.add_argument("--context-state", required=True)
    parser.add_argument(
        "--observed-context-families",
        default="",
        help="Optional OOF view-conditioned prototypes replacing static map context.",
    )
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--context-weight", type=float, default=0.25)
    parser.add_argument("--maximum-score-delta", type=float, default=0.05)
    parser.add_argument("--oracle-threshold-px", type=float, default=2.0)
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
    baseline = torch.load(
        args.baseline_topk, map_location="cpu", weights_only=False
    )
    graph = torch.load(
        args.confusion_graph, map_location="cpu", weights_only=False
    )
    context_state = torch.load(
        args.context_state, map_location="cpu", weights_only=False
    )
    if baseline.get("schema") != "lafgs_exact_topk_outcomes":
        raise ValueError("unsupported baseline top-K state")
    if graph.get("schema") != "lafgs_anchor_family_confusion_graph":
        raise ValueError("unsupported confusion graph")
    if context_state.get("schema") != "lafgs_fixed_3d_context_state":
        raise ValueError("unsupported context state")
    anchor_ids = torch.as_tensor(state["anchor_ids"]).long()
    if baseline["anchor_ids_sha256"] != _sha256_tensor(anchor_ids):
        raise ValueError("baseline top-K does not align with map")
    if not torch.equal(
        torch.as_tensor(context_state["anchor_ids"]).long(), anchor_ids
    ):
        raise ValueError("context state does not align with map")

    metric = SharedLowRankMetric(**metric_payload["metric_config"]).to(device)
    metric.load_state_dict(metric_payload["metric_state_dict"])
    metric.eval()
    projector = BoundedContextProjector(
        **context_state["query_projector_config"]
    ).to(device)
    projector.load_state_dict(context_state["query_projector_state_dict"])
    projector.eval()
    observed_families = (
        torch.load(
            args.observed_context_families,
            map_location="cpu",
            weights_only=False,
        )
        if args.observed_context_families
        else None
    )
    if observed_families is not None:
        if (
            observed_families.get("schema")
            != "lafgs_observed_context_families"
        ):
            raise ValueError("unsupported observed context family state")
        if not torch.equal(
            torch.as_tensor(observed_families["anchor_ids"]).long(),
            anchor_ids,
        ):
            raise ValueError("observed context families do not align with map")
        family_trajectories = list(observed_families["trajectories"])
        family_trajectory_index = {
            trajectory: index
            for index, trajectory in enumerate(family_trajectories)
        }
        family_context = torch.as_tensor(
            observed_families["prototype_context"]
        ).float()
        family_context = F.normalize(
            family_context, dim=family_context.ndim - 1
        ).to(device)
        family_counts = torch.as_tensor(
            observed_families["observation_counts"]
        ).to(device)
        context_map = None
    else:
        family_trajectories = []
        family_trajectory_index = {}
        family_context = None
        family_counts = None
        context_map = F.normalize(
            torch.as_tensor(context_state["anchor_context"]).float(), dim=1
        ).to(device)
    context_config = dict(context_state["config"])
    if context_config.get("query_representation") != "relational_sparse_2d_v1":
        raise ValueError("candidate rescue currently requires relational query context")

    global_edge_keys = directed_edge_keys(
        graph["edges"], anchor_count=len(anchor_ids)
    )
    query_events = events_by_query_row(graph["events"])
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    variants = {
        "edge": [],
        "event": [],
        "edge_oracle2px": [],
        "event_oracle2px": [],
    }
    totals = {
        name: {
            "eligible_rows": 0,
            "eligible_pairs": 0,
            "changed_rows": 0,
            "accepted_rows": 0,
        }
        for name in variants
    }
    context_seconds = 0.0
    rerank_seconds = 0.0

    with torch.inference_mode():
        for record in tqdm(baseline["records"], desc="Candidate rescue"):
            name = str(record["query_name"])
            cached = cache[name]
            rows = torch.as_tensor(record["query_rows"]).long()
            candidates = torch.as_tensor(
                record["topk_anchor_indices"]
            ).long()
            local_scores = torch.as_tensor(record["topk_scores"]).float()
            all_descriptors = F.normalize(
                torch.as_tensor(cached["native_descriptors"]).float().to(device),
                dim=1,
            )

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            transformed, _ = metric(all_descriptors)
            query_context = relational_sparse_query_context(
                transformed,
                torch.as_tensor(cached["native_keypoints"]).float().to(device),
                torch.as_tensor(cached["native_scores"]).float().to(device),
                neighbor_count=int(context_config["query_neighbor_count"]),
                chunk_size=int(context_config.get("context_chunk_size", 256)),
            )[rows.to(device)]
            query_context, _ = projector(query_context)
            if family_context is None:
                candidate_context = torch.einsum(
                    "nd,nkd->nk",
                    query_context,
                    context_map[candidates.to(device)],
                )
            else:
                trajectory = str(name).split("/", 1)[0]
                if trajectory not in family_trajectory_index:
                    raise ValueError(
                        f"observed families do not contain trajectory {trajectory}"
                    )
                candidate_ids = candidates.to(device)
                if family_context.ndim == 3:
                    candidate_prototypes = family_context[:, candidate_ids]
                    family_scores = torch.einsum(
                        "nd,tnkd->ntk",
                        query_context,
                        candidate_prototypes,
                    )
                    valid_family = family_counts[:, candidate_ids].permute(
                        1, 0, 2
                    ) > 0
                    valid_family[
                        :,
                        family_trajectory_index[trajectory],
                        :,
                    ] = False
                elif family_context.ndim == 4:
                    candidate_prototypes = family_context[
                        :, :, candidate_ids
                    ]
                    family_scores = torch.einsum(
                        "nd,tbnkd->ntbk",
                        query_context,
                        candidate_prototypes,
                    ).flatten(1, 2)
                    valid_family = family_counts[
                        :, :, candidate_ids
                    ].permute(2, 0, 1, 3)
                    valid_family[
                        :,
                        family_trajectory_index[trajectory],
                        :,
                        :,
                    ] = False
                    valid_family = valid_family.flatten(1, 2) > 0
                else:
                    raise ValueError(
                        "observed context prototypes must be TxAxD or TxBxAxD"
                    )
                family_scores = family_scores.masked_fill(
                    ~valid_family, -torch.inf
                )
                candidate_context = family_scores.max(dim=1).values
                candidate_context[
                    ~torch.isfinite(candidate_context)
                ] = 0.0
            candidate_context = candidate_context.cpu()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            context_seconds += time.perf_counter() - started

            event_row_edges = [
                query_events.get((name, int(row)), ())
                for row in rows.tolist()
            ]
            started = time.perf_counter()
            rescued = {}
            diagnostics = {}
            result_indices, result_scores, result_diagnostics = (
                candidate_conditioned_rescue_from_edge_keys(
                    candidates,
                    local_scores,
                    candidate_context,
                    global_edge_keys,
                    anchor_count=len(anchor_ids),
                    context_weight=float(args.context_weight),
                    maximum_score_delta=float(args.maximum_score_delta),
                )
            )
            rescued["edge"] = (result_indices, result_scores)
            diagnostics["edge"] = result_diagnostics
            result_indices, result_scores, result_diagnostics = (
                candidate_conditioned_rescue(
                    candidates,
                    local_scores,
                    candidate_context,
                    event_row_edges,
                    context_weight=float(args.context_weight),
                    maximum_score_delta=float(args.maximum_score_delta),
                )
            )
            rescued["event"] = (result_indices, result_scores)
            diagnostics["event"] = result_diagnostics

            keypoints = (
                torch.as_tensor(cached["native_keypoints"]).float()[rows]
                + float(cached.get("pixel_center_offset", 0.5))
            )
            K = torch.as_tensor(cached["native_K"]).float()
            pose = torch.as_tensor(cached["pose_w2c"]).float()
            baseline_errors = _project_errors(
                xyz[candidates[:, 0]], keypoints, K, pose
            )
            for mode in ("edge", "event"):
                result_indices, result_scores = rescued[mode]
                rescued_errors = _project_errors(
                    xyz[result_indices[:, 0]], keypoints, K, pose
                )
                accepted = oracle_acceptance_mask(
                    baseline_errors,
                    rescued_errors,
                    diagnostics[mode]["changed_mask"],
                    strict_threshold_px=float(args.oracle_threshold_px),
                )
                oracle_indices, oracle_scores = _restore_rejected(
                    result_indices,
                    result_scores,
                    candidates,
                    local_scores,
                    accepted,
                )
                rescued[f"{mode}_oracle2px"] = (
                    oracle_indices,
                    oracle_scores,
                )
                totals[f"{mode}_oracle2px"]["accepted_rows"] += int(
                    accepted.sum()
                )
            rerank_seconds += time.perf_counter() - started

            for mode, (result_indices, result_scores) in rescued.items():
                base_mode = mode.removesuffix("_oracle2px")
                mode_diagnostics = diagnostics[base_mode]
                changed = result_indices[:, 0] != candidates[:, 0]
                totals[mode]["eligible_rows"] += int(
                    mode_diagnostics["eligible_row_count"]
                )
                totals[mode]["eligible_pairs"] += int(
                    mode_diagnostics["eligible_pair_count"]
                )
                totals[mode]["changed_rows"] += int(changed.sum())
                variants[mode].append(
                    {
                        "query_name": name,
                        "query_rows": rows,
                        "topk_anchor_indices": result_indices,
                        "topk_scores": result_scores,
                    }
                )

    output_prefix = Path(args.output_prefix).resolve()
    provenance = {
        "map": str(Path(args.map).resolve()),
        "metric_state": str(Path(args.metric_state).resolve()),
        "query_cache": str(Path(args.query_cache).resolve()),
        "baseline_topk": str(Path(args.baseline_topk).resolve()),
        "confusion_graph": str(Path(args.confusion_graph).resolve()),
        "context_state": str(Path(args.context_state).resolve()),
        "observed_context_families": str(
            Path(args.observed_context_families).resolve()
        )
        if args.observed_context_families
        else None,
        "context_weight": float(args.context_weight),
        "maximum_score_delta": float(args.maximum_score_delta),
        "oracle_threshold_px": float(args.oracle_threshold_px),
        "candidate_context_ms_per_query": float(
            1000.0 * context_seconds / max(len(baseline["records"]), 1)
        ),
        "candidate_rerank_ms_per_query": float(
            1000.0 * rerank_seconds / max(len(baseline["records"]), 1)
        ),
    }
    for mode, records in variants.items():
        output = output_prefix.with_name(
            f"{output_prefix.name}_{mode}.pt"
        )
        _atomic_torch(
            output,
            {
                "schema": "lafgs_exact_topk_outcomes",
                "version": 2,
                "query_names": list(baseline["query_names"]),
                "query_start": int(baseline.get("query_start", 0)),
                "topk": int(baseline["topk"]),
                "anchor_count": int(baseline["anchor_count"]),
                "anchor_ids_sha256": baseline["anchor_ids_sha256"],
                "records": records,
                "method": (
                    "candidate_conditioned_observed_context_"
                    if observed_families is not None
                    else "candidate_conditioned_context_"
                )
                + mode,
                "summary": totals[mode],
                "provenance": provenance,
            },
        )
        print({"output": str(output), **totals[mode]})


if __name__ == "__main__":
    main()
