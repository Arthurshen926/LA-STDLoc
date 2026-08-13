#!/usr/bin/env python3
"""Train-only rendered-view retrieval plus Track-observation localization.

The method keeps the ray-triangulated Track geometry but avoids collapsing all
view appearances into one descriptor.  A held mapping query first retrieves
rendered training views using a fixed pooled SuperPoint descriptor.  Its local
features then match only raw Track observations from those views, followed by
one standard PoseLib solve.  The held sequence is excluded from both retrieval
and the observation bank; source mapping RGB and test queries are forbidden.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from localization.matcher import global_cosine_top1
from localization.pose_solver import pose_error, solve_absolute_pose
from map_learning.trainer import _pose_error_cm, _project_errors
from scripts.evaluate_rendered_track_crossfit import _sequence_name


def pooled_image_descriptor(descriptors: torch.Tensor) -> torch.Tensor:
    """Fixed first/second-moment global descriptor for rendered view retrieval."""
    descriptors = F.normalize(torch.as_tensor(descriptors).float(), dim=1)
    if descriptors.ndim != 2 or descriptors.shape[0] == 0:
        raise ValueError("image descriptors must be a non-empty matrix")
    mean = descriptors.mean(dim=0)
    rms = descriptors.square().mean(dim=0).clamp_min(0).sqrt()
    return F.normalize(torch.cat((mean, rms)), dim=0)


def observation_bank_for_references(
    *,
    reference_queries: torch.Tensor,
    track_to_anchor: torch.Tensor,
    track_query: torch.Tensor,
    track_keypoint: torch.Tensor,
    names: list[str],
    cache: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect raw descriptor rows and target anchors for retrieved views."""
    reference_queries = torch.as_tensor(reference_queries).long().reshape(-1)
    track_to_anchor = torch.as_tensor(track_to_anchor).long().reshape(-1)
    track_query = torch.as_tensor(track_query).long().reshape(-1)
    track_keypoint = torch.as_tensor(track_keypoint).long().reshape(-1)
    if track_query.shape != track_keypoint.shape:
        raise ValueError("Track query/keypoint observation rows differ")
    is_reference = torch.zeros(len(names), dtype=torch.bool)
    is_reference[reference_queries] = True
    valid_track = (track_to_anchor >= 0) & is_reference[track_query]
    observations = torch.nonzero(valid_track, as_tuple=False).reshape(-1)
    if observations.numel() == 0:
        raise RuntimeError("retrieved rendered views contain no selected Track rows")
    queries = track_query[observations]
    keypoints = track_keypoint[observations]
    descriptors = torch.stack(
        [
            torch.as_tensor(cache[names[int(query)]]["native_descriptors"])[
                int(keypoint)
            ]
            for query, keypoint in zip(queries.tolist(), keypoints.tolist())
        ]
    ).float()
    return descriptors, track_to_anchor[observations], observations


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
        "mean_reference_count": float(
            np.mean([row["reference_count"] for row in rows])
        ),
        "mean_observation_bank_size": float(
            np.mean([row["observation_bank_size"] for row in rows])
        ),
        "mean_hypotheses": float(np.mean([row["hypotheses"] for row in rows])),
    }


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
        raise ValueError("retrieval crossfit requires rendered mapping-only cache")
    cache = cache_payload.get("queries", cache_payload)
    names = list(teacher["query_names"])
    if names != list(payload["query_names"]):
        raise ValueError("teacher and Track query names differ")
    tracks = payload["tracks"]
    selected_track_ids = torch.as_tensor(state["track_cluster_ids"]).long()
    observation_track = torch.as_tensor(tracks["track_index"]).long()
    if "track_level" in tracks:
        total_tracks = int(torch.as_tensor(tracks["track_level"]).numel())
    elif observation_track.numel():
        total_tracks = int(observation_track.max()) + 1
    else:
        raise ValueError("Track payload has no observations or Track registry")
    track_to_anchor = torch.full((total_tracks,), -1, dtype=torch.long)
    track_to_anchor[selected_track_ids] = torch.arange(selected_track_ids.numel())
    observation_query = torch.as_tensor(tracks["query_index"]).long()
    observation_keypoint = torch.as_tensor(tracks["keypoint_index"]).long()
    observation_anchor = track_to_anchor[observation_track]
    eligible_reference_query = torch.zeros(len(names), dtype=torch.bool)
    eligible_reference_query[observation_query[observation_anchor >= 0]] = True
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    device = torch.device(args.device)
    global_descriptors = torch.stack(
        [pooled_image_descriptor(cache[name]["native_descriptors"]) for name in names]
    ).to(device)
    sequences = sorted({_sequence_name(name) for name in names})
    fold_reports = []
    all_rows = []
    for held_sequence in sequences:
        if args.held_sequence and held_sequence != args.held_sequence:
            continue
        train_queries = torch.as_tensor(
            [
                index
                for index, name in enumerate(names)
                if _sequence_name(name) != held_sequence
                and bool(eligible_reference_query[index])
            ]
        ).long()
        if train_queries.numel() < args.reference_count:
            raise RuntimeError(
                f"held fold {held_sequence} has fewer eligible rendered references "
                f"than requested: {int(train_queries.numel())} < {args.reference_count}"
            )
        held_queries = [
            index
            for index, name in enumerate(names)
            if _sequence_name(name) == held_sequence
        ]
        query_rows = []
        for query_index in held_queries:
            record = teacher["records"][query_index]
            cached = cache[names[query_index]]
            retrieval_scores = (
                global_descriptors[query_index] @ global_descriptors[train_queries].T
            )
            reference_count = min(args.reference_count, int(train_queries.numel()))
            references = train_queries[
                torch.topk(retrieval_scores, k=reference_count).indices.cpu()
            ]
            observation_descriptors, candidate_anchor, _ = (
                observation_bank_for_references(
                    reference_queries=references,
                    track_to_anchor=observation_anchor,
                    track_query=observation_query,
                    track_keypoint=observation_keypoint,
                    names=names,
                    cache=cache,
                )
            )
            rows = torch.as_tensor(record["query_rows"]).long()
            query_descriptors = torch.as_tensor(cached["native_descriptors"])[
                rows
            ].float()
            matches = global_cosine_top1(
                query_descriptors.to(device), observation_descriptors.to(device)
            )
            winners = candidate_anchor[matches.anchor_indices.cpu()]
            keypoints = torch.as_tensor(cached["native_keypoints"])[rows].float()
            keypoints = keypoints + float(cached.get("pixel_center_offset", 0.5))
            intrinsic = torch.as_tensor(cached["native_K"]).float()
            estimate = solve_absolute_pose(
                keypoints.numpy(),
                xyz[winners].numpy(),
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
                    xyz[winners[inliers]],
                    keypoints[inliers],
                    intrinsic,
                    torch.as_tensor(cached["pose_w2c"]).float(),
                )
                clean = int((errors <= args.clean_reprojection_px).sum())
            ae_deg, _ = pose_error(
                estimate.pose_w2c,
                torch.as_tensor(cached["pose_w2c"]).cpu().numpy(),
            )
            te_cm = _pose_error_cm(
                estimate.pose_w2c, torch.as_tensor(cached["pose_w2c"])
            )
            query_rows.append(
                {
                    "query_index": query_index,
                    "query_name": names[query_index],
                    "te_cm": float(te_cm),
                    "ae_deg": float(ae_deg),
                    "inliers": int(inliers.numel()),
                    "clean_inliers": clean,
                    "hypotheses": int(estimate.diagnostics.get("iterations", 0)),
                    "reference_count": reference_count,
                    "reference_queries": references.tolist(),
                    "observation_bank_size": int(observation_descriptors.shape[0]),
                }
            )
            if len(query_rows) % 25 == 0:
                print(
                    json.dumps(
                        {
                            "event": f"retrieval_crossfit_{held_sequence}",
                            "queries_complete": len(query_rows),
                        }
                    ),
                    flush=True,
                )
        all_rows.extend(query_rows)
        fold_reports.append(
            {
                "held_sequence": held_sequence,
                "train_query_count": int(train_queries.numel()),
                "summary": _summary(query_rows),
                "queries": query_rows,
            }
        )
        print(
            json.dumps(
                {
                    key: value
                    for key, value in fold_reports[-1].items()
                    if key != "queries"
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if not fold_reports:
        raise RuntimeError("requested held mapping sequence was not found")
    report = {
        "schema": "lafgs_rendered_track_view_retrieval_mapping_crossfit",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "held_sequence_excluded_from_retrieval": True,
        "one_poselib_call_per_query": True,
        "reference_count": args.reference_count,
        "folds": fold_reports,
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
