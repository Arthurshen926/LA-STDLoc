#!/usr/bin/env python3
"""Train-only retrieved-view reciprocal Track localization.

Global rendered-image descriptors propose training views.  Every query/reference
pair is then matched independently with reciprocal local descriptors; retained
reference keypoints inherit their ray-triangulated Track xyz.  Pair matches are
merged to at most one winner per query keypoint before one standard PoseLib
solve.  Matching never uses the held pose, source mapping RGB, or test queries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from localization.pose_solver import pose_error, solve_absolute_pose
from map_learning.trainer import _pose_error_cm, _project_errors
from scripts.evaluate_rendered_track_crossfit import _sequence_name
from scripts.evaluate_rendered_track_retrieval_crossfit import (
    _atomic_json,
    pooled_image_descriptor,
)


def reciprocal_pair_matches(
    query_descriptors: torch.Tensor,
    reference_descriptors: torch.Tensor,
    *,
    minimum_similarity: float,
    minimum_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = F.normalize(torch.as_tensor(query_descriptors).float(), dim=1)
    reference = F.normalize(torch.as_tensor(reference_descriptors).float(), dim=1)
    if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1]:
        raise ValueError("query/reference descriptors must be aligned matrices")
    if min(query.shape[0], reference.shape[0]) < 2:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty, torch.empty(0)
    similarity = query @ reference.T
    values_q, indices_q = torch.topk(similarity, k=2, dim=1)
    values_r, indices_r = torch.topk(similarity, k=2, dim=0)
    query_row = torch.arange(query.shape[0])
    reference_row = indices_q[:, 0]
    mutual = indices_r[0, reference_row] == query_row
    valid = (
        mutual
        & (values_q[:, 0] >= float(minimum_similarity))
        & ((values_q[:, 0] - values_q[:, 1]) >= float(minimum_margin))
        & (
            (values_r[0, reference_row] - values_r[1, reference_row])
            >= float(minimum_margin)
        )
    )
    selected = torch.nonzero(valid, as_tuple=False).reshape(-1)
    return selected, reference_row[selected], values_q[selected, 0]


def merge_pair_matches(
    query_rows: list[torch.Tensor],
    anchor_rows: list[torch.Tensor],
    scores: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not query_rows:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty, torch.empty(0)
    query = torch.cat(query_rows).long()
    anchor = torch.cat(anchor_rows).long()
    score = torch.cat(scores).float()
    if not (query.numel() == anchor.numel() == score.numel()):
        raise ValueError("pair-match rows differ")
    score_order = torch.argsort(score, descending=True, stable=True)
    query_order = torch.argsort(query[score_order], stable=True)
    grouped = score_order[query_order]
    grouped_query = query[grouped]
    first = torch.ones(grouped.numel(), dtype=torch.bool)
    first[1:] = grouped_query[1:] != grouped_query[:-1]
    selected = grouped[first]
    output_order = torch.argsort(query[selected], stable=True)
    selected = selected[output_order]
    return query[selected], anchor[selected], score[selected]


def _summary(rows: list[dict]) -> dict:
    te = np.asarray([row["te_cm"] for row in rows], dtype=np.float64)
    ae = np.asarray([row["ae_deg"] for row in rows], dtype=np.float64)
    tail = max(int(np.ceil(0.05 * len(rows))), 1)
    return {
        "query_count": len(rows),
        "median_te_cm": float(np.median(te)),
        "mean_te_cm": float(np.mean(te)),
        "p90_te_cm": float(np.percentile(te, 90)),
        "cvar95_te_cm": float(np.sort(te)[-tail:].mean()),
        "median_ae_deg": float(np.median(ae)),
        "mean_ae_deg": float(np.mean(ae)),
        "p90_ae_deg": float(np.percentile(ae, 90)),
        "recall_5cm_5deg_percent": float(100 * np.mean((te < 5) & (ae < 5))),
        "catastrophic_100cm_count": int(np.count_nonzero(te >= 100)),
        "mean_inliers": float(np.mean([row["inliers"] for row in rows])),
        "mean_clean_inliers": float(np.mean([row["clean_inliers"] for row in rows])),
        "mean_pair_match_count": float(
            np.mean([row["pair_match_count"] for row in rows])
        ),
        "mean_merged_match_count": float(
            np.mean([row["merged_match_count"] for row in rows])
        ),
        "mean_hypotheses": float(np.mean([row["hypotheses"] for row in rows])),
    }


@torch.inference_mode()
def run(args) -> dict:
    state = torch.load(args.anchor_map, map_location="cpu", weights_only=False)
    payload = torch.load(args.track_payload, map_location="cpu", weights_only=False)
    teacher = torch.load(args.teacher, map_location="cpu", weights_only=False)
    cache_payload = torch.load(args.query_cache, map_location="cpu", weights_only=False)
    if (
        cache_payload.get("uses_source_mapping_rgb") is not False
        or cache_payload.get("uses_test_queries") is not False
    ):
        raise ValueError("pair retrieval requires rendered mapping-only cache")
    cache = cache_payload.get("queries", cache_payload)
    names = list(teacher["query_names"])
    if names != list(payload["query_names"]):
        raise ValueError("teacher and Track query names differ")
    tracks = payload["tracks"]
    selected_track_ids = torch.as_tensor(state["track_cluster_ids"]).long()
    observation_track = torch.as_tensor(tracks["track_index"]).long()
    total_tracks = int(torch.as_tensor(tracks["track_level"]).numel())
    track_to_anchor = torch.full((total_tracks,), -1, dtype=torch.long)
    track_to_anchor[selected_track_ids] = torch.arange(selected_track_ids.numel())
    observation_anchor = track_to_anchor[observation_track]
    observation_query = torch.as_tensor(tracks["query_index"]).long()
    observation_keypoint = torch.as_tensor(tracks["keypoint_index"]).long()
    per_query_observations = []
    eligible_reference = torch.zeros(len(names), dtype=torch.bool)
    for query_index in range(len(names)):
        rows = torch.nonzero(
            (observation_query == query_index) & (observation_anchor >= 0),
            as_tuple=False,
        ).reshape(-1)
        per_query_observations.append(rows)
        eligible_reference[query_index] = bool(rows.numel())
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    device = torch.device(args.device)
    global_descriptors = torch.stack(
        [pooled_image_descriptor(cache[name]["native_descriptors"]) for name in names]
    ).to(device)
    folds = []
    all_rows = []
    for held_sequence in sorted({_sequence_name(name) for name in names}):
        if args.held_sequence and held_sequence != args.held_sequence:
            continue
        train_queries = torch.as_tensor(
            [
                index
                for index, name in enumerate(names)
                if _sequence_name(name) != held_sequence
                and bool(eligible_reference[index])
            ]
        ).long()
        if train_queries.numel() < args.reference_count:
            raise RuntimeError("held fold has too few eligible reference views")
        held_queries = [
            index
            for index, name in enumerate(names)
            if _sequence_name(name) == held_sequence
        ]
        query_rows = []
        for query_index in held_queries:
            cached = cache[names[query_index]]
            deployment_rows = torch.as_tensor(
                teacher["records"][query_index]["query_rows"]
            ).long()
            query_descriptors = torch.as_tensor(cached["native_descriptors"])[
                deployment_rows
            ].float()
            retrieval = (
                global_descriptors[query_index] @ global_descriptors[train_queries].T
            )
            references = train_queries[
                torch.topk(retrieval, k=args.reference_count).indices.cpu()
            ]
            matched_queries = []
            matched_anchors = []
            matched_scores = []
            for reference in references.tolist():
                observation_rows = per_query_observations[int(reference)]
                keypoint_rows = observation_keypoint[observation_rows]
                reference_descriptors = torch.as_tensor(
                    cache[names[int(reference)]]["native_descriptors"]
                )[keypoint_rows].float()
                source, target, score = reciprocal_pair_matches(
                    query_descriptors.to(device),
                    reference_descriptors.to(device),
                    minimum_similarity=args.minimum_similarity,
                    minimum_margin=args.minimum_margin,
                )
                if source.numel():
                    matched_queries.append(source.cpu())
                    matched_anchors.append(
                        observation_anchor[observation_rows[target.cpu()]]
                    )
                    matched_scores.append(score.cpu())
            merged_query, merged_anchor, _ = merge_pair_matches(
                matched_queries, matched_anchors, matched_scores
            )
            if merged_query.numel() < 4:
                raise RuntimeError(
                    f"query {names[query_index]} has fewer than four reciprocal Track matches"
                )
            keypoints = torch.as_tensor(cached["native_keypoints"])[
                deployment_rows[merged_query]
            ].float()
            keypoints += float(cached.get("pixel_center_offset", 0.5))
            intrinsic = torch.as_tensor(cached["native_K"]).float()
            estimate = solve_absolute_pose(
                keypoints.numpy(),
                xyz[merged_anchor].numpy(),
                intrinsic.numpy(),
                reprojection_error_px=args.ransac_reprojection_px,
                confidence=0.99999,
                max_iterations=args.max_iterations,
                min_iterations=args.min_iterations,
                seed=args.seed,
            )
            inliers = torch.as_tensor(estimate.inliers).long().reshape(-1)
            clean = 0
            if inliers.numel():
                errors = _project_errors(
                    xyz[merged_anchor[inliers]],
                    keypoints[inliers],
                    intrinsic,
                    torch.as_tensor(cached["pose_w2c"]).float(),
                )
                clean = int((errors <= args.clean_reprojection_px).sum())
            ae_deg, _ = pose_error(
                estimate.pose_w2c,
                torch.as_tensor(cached["pose_w2c"]).cpu().numpy(),
            )
            query_rows.append(
                {
                    "query_index": query_index,
                    "query_name": names[query_index],
                    "te_cm": float(
                        _pose_error_cm(
                            estimate.pose_w2c, torch.as_tensor(cached["pose_w2c"])
                        )
                    ),
                    "ae_deg": float(ae_deg),
                    "inliers": int(inliers.numel()),
                    "clean_inliers": clean,
                    "hypotheses": int(estimate.diagnostics.get("iterations", 0)),
                    "pair_match_count": int(
                        sum(value.numel() for value in matched_queries)
                    ),
                    "merged_match_count": int(merged_query.numel()),
                    "reference_queries": references.tolist(),
                }
            )
            if len(query_rows) % 25 == 0:
                print(
                    json.dumps(
                        {
                            "event": f"pair_retrieval_{held_sequence}",
                            "queries_complete": len(query_rows),
                        }
                    ),
                    flush=True,
                )
        all_rows.extend(query_rows)
        folds.append(
            {
                "held_sequence": held_sequence,
                "summary": _summary(query_rows),
                "queries": query_rows,
            }
        )
        print(
            json.dumps(
                {key: value for key, value in folds[-1].items() if key != "queries"},
                sort_keys=True,
            ),
            flush=True,
        )
    if not folds:
        raise RuntimeError("requested held mapping sequence was not found")
    report = {
        "schema": "lafgs_rendered_track_reciprocal_view_retrieval_crossfit",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "held_sequence_excluded_from_retrieval": True,
        "matching_uses_query_pose": False,
        "one_poselib_call_per_query": True,
        "reference_count": args.reference_count,
        "minimum_similarity": args.minimum_similarity,
        "minimum_margin": args.minimum_margin,
        "folds": folds,
        "combined_summary": _summary(all_rows),
    }
    _atomic_json(report, args.output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-map", type=Path, required=True)
    parser.add_argument("--track-payload", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--held-sequence", default="")
    parser.add_argument("--reference-count", type=int, default=8)
    parser.add_argument("--minimum-similarity", type=float, default=0.65)
    parser.add_argument("--minimum-margin", type=float, default=0.01)
    parser.add_argument("--ransac-reprojection-px", type=float, default=12.0)
    parser.add_argument("--clean-reprojection-px", type=float, default=4.0)
    parser.add_argument("--min-iterations", type=int, default=1000)
    parser.add_argument("--max-iterations", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for field in ("anchor_map", "track_payload", "teacher", "query_cache", "output"):
        setattr(args, field, getattr(args, field).resolve())
    if args.reference_count < 1:
        raise ValueError("reference_count must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
