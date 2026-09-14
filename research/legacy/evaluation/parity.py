"""Exact sparse-runtime parity against the frozen golden fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from data.datasets import ColmapDataset
from evaluation.golden import verify_fixture
from localization.localizer import SparseLocalizer


def run_sparse_parity(
    *,
    fixture: str | Path,
    dataset_root: str | Path,
    map_path: str | Path,
    metric_state_path: str | Path,
    device: str = "cuda",
    deployment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fixture = Path(fixture).resolve()
    manifest = verify_fixture(fixture)
    dataset = ColmapDataset(dataset_root)
    deployment = deployment or {}
    localizer = SparseLocalizer(
        map_path,
        metric_state_path,
        device=device,
        keypoint_count=deployment.get("keypoints", 2048),
        reprojection_error_px=deployment.get("reprojection_error_px", 12.0),
        confidence=deployment.get("confidence", 0.99999),
        max_iterations=deployment.get("maximum_iterations", 100000),
        min_iterations=deployment.get("minimum_iterations", 1000),
        seed=2026,
    )
    totals = {
        "query_count": 0,
        "keypoints_exact": 0,
        "top1_exact": 0,
        "scores_close": 0,
        "inliers_exact": 0,
        "poses_close": 0,
    }
    maximum_score_error = 0.0
    maximum_pose_error = 0.0
    failures = []
    for record in manifest["queries"]:
        with np.load(fixture / record["file"], allow_pickle=False) as expected:
            camera = dataset.camera(record["image_name"])
            result = localizer.localize(
                dataset.load_image(camera),
                fov_x=camera.fov_x,
                fov_y=camera.fov_y,
                valid_mask=dataset.valid_mask(camera),
            )
            score_error = float(
                np.max(
                    np.abs(
                        result.matches.scores.cpu().numpy()
                        - expected["topk_scores"][:, 0]
                    )
                )
            )
            pose_error = float(
                np.max(np.abs(result.pose.pose_w2c - expected["pred_pose_w2c"]))
            )
            maximum_score_error = max(maximum_score_error, score_error)
            maximum_pose_error = max(maximum_pose_error, pose_error)
            checks = {
                "keypoints_exact": np.array_equal(
                    result.sparse_features.keypoints.cpu().numpy(),
                    expected["keypoint_xy"],
                ),
                "top1_exact": np.array_equal(
                    result.matches.anchor_indices.cpu().numpy(),
                    expected["topk_landmark_idx"][:, 0],
                ),
                "scores_close": np.allclose(
                    result.matches.scores.cpu().numpy(),
                    expected["topk_scores"][:, 0],
                    rtol=0.0,
                    atol=5e-7,
                ),
                "inliers_exact": np.array_equal(
                    result.pose.inliers,
                    expected["hard_post_inliers"],
                ),
                "poses_close": np.allclose(
                    result.pose.pose_w2c,
                    expected["pred_pose_w2c"],
                    rtol=0.0,
                    atol=1e-6,
                ),
            }
            totals["query_count"] += 1
            for name, passed in checks.items():
                totals[name] += int(passed)
            if not all(checks.values()):
                failures.append({"image_name": record["image_name"], **checks})
    return {
        "schema": "lafgs_sparse_parity_report",
        "version": 1,
        "totals": totals,
        "maximum_absolute_score_error": maximum_score_error,
        "maximum_absolute_pose_error": maximum_pose_error,
        "failures": failures,
        "passed": not failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default="paper_baseline/golden_fixture")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--config", default="configs/paper_mainline.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from common.config import load_mainline_config

    report = run_sparse_parity(
        fixture=args.fixture,
        dataset_root=args.dataset_root,
        map_path=args.map,
        metric_state_path=args.metric_state,
        device=args.device,
        deployment=load_mainline_config(args.config).values["deployment"],
    )
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
