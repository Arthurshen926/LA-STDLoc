#!/usr/bin/env python3
"""Evaluate one-shot Top-K candidate PnP on held mapping sequences.

Each query row contributes its K highest-ranked 3D candidates to one standard
PoseLib solve.  Fold descriptors were built without the held mapping sequence.
This is a train-only geometry-consensus diagnostic, not a test evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from localization.localizer import load_shared_metric
from localization.pose_solver import pose_error, solve_absolute_pose
from map_learning.trainer import _pose_error_cm, _project_errors
from scripts.evaluate_rendered_track_crossfit import _sequence_name


def flatten_topk_correspondences(
    keypoints: torch.Tensor,
    anchor_xyz: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    keypoints = torch.as_tensor(keypoints).float()
    anchor_xyz = torch.as_tensor(anchor_xyz).float()
    indices = torch.as_tensor(indices).long()
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError("keypoints must have shape [N,2]")
    if indices.ndim != 2 or indices.shape[0] != keypoints.shape[0]:
        raise ValueError("Top-K indices and keypoint rows differ")
    if indices.numel() and (
        int(indices.min()) < 0 or int(indices.max()) >= anchor_xyz.shape[0]
    ):
        raise ValueError("Top-K indices reference an unknown anchor")
    return (
        keypoints[:, None, :].expand(-1, indices.shape[1], -1).reshape(-1, 2),
        anchor_xyz[indices.reshape(-1)],
    )


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
        "mean_candidate_correspondences": float(
            np.mean([row["correspondences"] for row in rows])
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
    cache_payload = torch.load(args.query_cache, map_location="cpu", weights_only=False)
    if (
        cache_payload.get("uses_source_mapping_rgb") is not False
        or cache_payload.get("uses_test_queries") is not False
    ):
        raise ValueError("Top-K PnP requires rendered mapping-only cache")
    cache = cache_payload.get("queries", cache_payload)
    device = torch.device(args.device)
    fold_reports = []
    all_rows = []
    for fold_dir in sorted(
        path for path in args.crossfit_dir.iterdir() if path.is_dir()
    ):
        held_sequence = fold_dir.name
        if args.held_sequence and held_sequence != args.held_sequence:
            continue
        state = torch.load(
            fold_dir / "anchor_map.pt", map_location="cpu", weights_only=False
        )
        teacher = torch.load(
            fold_dir / "positive_teacher.pt", map_location="cpu", weights_only=False
        )
        metric = load_shared_metric(
            fold_dir / "metric_state.pt",
            anchor_ids=torch.as_tensor(state["anchor_ids"]).long(),
            device=device,
        )
        bank = F.normalize(
            torch.as_tensor(state["anchor_features"]).float().to(device), dim=1
        )
        xyz = torch.as_tensor(state["anchor_xyz"]).float()
        effective_topk = min(args.topk, int(bank.shape[0]))
        query_rows = []
        for query_index, (name, record) in enumerate(
            zip(teacher["query_names"], teacher["records"])
        ):
            if _sequence_name(name) != held_sequence:
                continue
            cached = cache[name]
            rows = torch.as_tensor(record["query_rows"]).long()
            descriptor = F.normalize(
                torch.as_tensor(cached["native_descriptors"])[rows].float(), dim=1
            ).to(device)
            adapted, _ = metric(descriptor)
            indices = torch.topk(
                adapted @ bank.T, k=effective_topk, dim=1
            ).indices.cpu()
            keypoints = torch.as_tensor(cached["native_keypoints"])[rows].float()
            keypoints = keypoints + float(cached.get("pixel_center_offset", 0.5))
            candidate_keypoints, candidate_xyz = flatten_topk_correspondences(
                keypoints, xyz, indices
            )
            intrinsic = torch.as_tensor(cached["native_K"]).float()
            estimate = solve_absolute_pose(
                candidate_keypoints.numpy(),
                candidate_xyz.numpy(),
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
                    candidate_xyz[inliers],
                    candidate_keypoints[inliers],
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
                    "query_name": name,
                    "te_cm": float(te_cm),
                    "ae_deg": float(ae_deg),
                    "inliers": int(inliers.numel()),
                    "clean_inliers": clean,
                    "correspondences": int(candidate_xyz.shape[0]),
                    "hypotheses": int(estimate.diagnostics.get("iterations", 0)),
                }
            )
            if len(query_rows) % 25 == 0:
                print(
                    json.dumps(
                        {
                            "event": f"topk_pnp_{held_sequence}",
                            "queries_complete": len(query_rows),
                        }
                    ),
                    flush=True,
                )
        if not query_rows:
            raise RuntimeError(f"no held queries evaluated for {held_sequence}")
        all_rows.extend(query_rows)
        fold_reports.append(
            {
                "held_sequence": held_sequence,
                "topk": effective_topk,
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
        "schema": "lafgs_rendered_track_topk_pnp_mapping_crossfit",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "one_poselib_call_per_query": True,
        "candidate_topk": args.topk,
        "folds": fold_reports,
        "combined_summary": _summary(all_rows),
    }
    _atomic_json(report, args.output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crossfit-dir", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--held-sequence", default="")
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--ransac-reprojection-px", type=float, default=12.0)
    parser.add_argument("--clean-reprojection-px", type=float, default=4.0)
    parser.add_argument("--min-iterations", type=int, default=1000)
    parser.add_argument("--max-iterations", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.crossfit_dir = args.crossfit_dir.resolve()
    args.query_cache = args.query_cache.resolve()
    args.output = args.output.resolve()
    if args.topk < 1:
        raise ValueError("topk must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
