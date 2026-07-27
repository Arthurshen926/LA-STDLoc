#!/usr/bin/env python
"""Evaluate dependency-aware RANSAC proposal pools with full-set pose scoring."""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_discrete_decision_oracles import (
    pose_error,
    select_candidates,
    summarize_pose_errors,
)
from utils.pose_utils import solve_pose


def _reprojection_errors(pose_w2c, p2d, p3d, K):
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    camera = np.asarray(p3d, dtype=np.float64) @ pose[:3, :3].T
    camera += pose[:3, 3]
    depth = camera[:, 2]
    projected = np.full_like(np.asarray(p2d, dtype=np.float64), np.nan)
    valid = np.isfinite(camera).all(axis=1) & (depth > 1e-8)
    projected[valid, 0] = (
        K[0, 0] * camera[valid, 0] / depth[valid] + K[0, 2]
    )
    projected[valid, 1] = (
        K[1, 1] * camera[valid, 1] / depth[valid] + K[1, 2]
    )
    return np.linalg.norm(projected - p2d, axis=1)


def _pose_score(pose, p2d, p3d, K, threshold):
    errors = _reprojection_errors(pose, p2d, p3d, K)
    inliers = np.flatnonzero(errors <= float(threshold))
    truncated = np.minimum(errors, float(threshold))
    return int(inliers.size), -float(np.mean(truncated)), inliers


def _refine_with_full_inliers(
    pose, p2d, p3d, K, score_threshold, refine_threshold=6.0
):
    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    initial_pose = pose.copy()
    initial_score = _pose_score(
        initial_pose, p2d, p3d, K, score_threshold
    )
    inliers = np.flatnonzero(
        _reprojection_errors(pose, p2d, p3d, K)
        <= float(refine_threshold)
    )
    for _ in range(2):
        if inliers.size < 4:
            break
        rvec = cv2.Rodrigues(pose[:3, :3])[0]
        tvec = pose[:3, 3].reshape(3, 1)
        success, rvec, tvec = cv2.solvePnP(
            p3d[inliers],
            p2d[inliers],
            K,
            np.zeros((4, 1), dtype=np.float64),
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            break
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = cv2.Rodrigues(rvec)[0]
        pose[:3, 3] = tvec.reshape(3)
        inliers = np.flatnonzero(
            _reprojection_errors(pose, p2d, p3d, K)
            <= float(refine_threshold)
        )
    refined_score = _pose_score(pose, p2d, p3d, K, score_threshold)
    if refined_score[:2] < initial_score[:2]:
        return initial_pose, initial_score[2], False
    return pose, refined_score[2], True


def _distinct_family_rows(source_ids, scores):
    source_ids = np.asarray(source_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    seen = set()
    selected = []
    for row in order:
        source = int(source_ids[row])
        if source not in seen:
            seen.add(source)
            selected.append(int(row))
    return np.asarray(selected, dtype=np.int64)


def _cell_balanced_rows(p2d, source_ids, scores, width, height, budget):
    unique_rows = _distinct_family_rows(source_ids, scores)
    xy = np.asarray(p2d)[unique_rows]
    col = np.clip((xy[:, 0] * 4 / max(width, 1)).astype(int), 0, 3)
    row = np.clip((xy[:, 1] * 4 / max(height, 1)).astype(int), 0, 3)
    cells = row * 4 + col
    queues = []
    for cell in range(16):
        local = unique_rows[cells == cell]
        queues.append(
            local[np.argsort(-np.asarray(scores)[local], kind="stable")].tolist()
        )
    selected = []
    while len(selected) < int(budget):
        progress = False
        for queue in queues:
            if queue and len(selected) < int(budget):
                selected.append(queue.pop(0))
                progress = True
        if not progress:
            break
    return np.asarray(selected, dtype=np.int64)


def _proposal_rows(
    mode,
    p2d,
    source_ids,
    dependency_ids,
    scores,
    width,
    height,
    budget,
):
    if mode == "uniform":
        return np.arange(len(scores), dtype=np.int64)
    if mode == "family":
        return _distinct_family_rows(source_ids, scores)
    if mode == "family_cell":
        return _cell_balanced_rows(
            p2d, source_ids, scores, width, height, budget
        )
    if mode == "dependency":
        return _distinct_family_rows(dependency_ids, scores)
    if mode == "dependency_cell":
        return _cell_balanced_rows(
            p2d, dependency_ids, scores, width, height, budget
        )
    raise ValueError(mode)


def evaluate(dump_dir, output, modes, seed, proposal_budget):
    dump_dir = Path(dump_dir)
    manifest = json.loads((dump_dir / "manifest.json").read_text())
    with np.load(dump_dir / manifest["landmark_bank"]) as bank_file:
        xyz = np.asarray(bank_file["landmark_xyz"], dtype=np.float64)
        source = np.asarray(
            bank_file["source_gaussian_idx"], dtype=np.int64
        )
        dependency = (
            np.asarray(bank_file["dependency_group_id"], dtype=np.int64)
            if "dependency_group_id" in bank_file.files
            else source
        )
    records = []
    errors = {
        mode: {"ae": [], "te": [], "time": [], "hypotheses": []}
        for mode in modes
    }
    for query_index, filename in enumerate(manifest["query_files"]):
        with np.load(dump_dir / filename, allow_pickle=False) as loaded:
            query = {key: np.asarray(loaded[key]) for key in loaded.files}
        keypoints = np.asarray(query["keypoint_xy"], dtype=np.float64) + 0.5
        raw_rows = np.asarray(
            query["matcher_raw_keypoint_idx"], dtype=np.int64
        )
        raw_lm = np.asarray(
            query["matcher_raw_landmark_idx"], dtype=np.int64
        )
        raw_scores = np.asarray(
            query["matcher_raw_scores"], dtype=np.float64
        )
        selected = select_candidates(
            raw_rows,
            raw_lm,
            raw_scores,
            threshold=float(query["candidate_threshold"]),
            max_matches_per_keypoint=int(
                query["max_matches_per_keypoint"]
            ),
            max_matches_per_landmark=int(
                query["max_matches_per_landmark"]
            ),
            min_match_count=int(query["min_candidate_matches"]),
            refill_trigger_count=int(
                query["candidate_refill_trigger_count"]
            ),
        )
        p2d = keypoints[selected.keypoint_idx]
        p3d = xyz[selected.landmark_idx]
        scores = selected.scores
        families = source[selected.landmark_idx]
        dependencies = dependency[selected.landmark_idx]
        K = np.asarray(query["K"], dtype=np.float64)
        query_record = {"image_name": str(query["image_name"].item())}
        for mode in modes:
            proposal = _proposal_rows(
                mode,
                p2d,
                families,
                dependencies,
                scores,
                int(query["width"]),
                int(query["height"]),
                proposal_budget,
            )
            start = time.perf_counter()
            pose, _, diagnostics = solve_pose(
                p2d[proposal],
                p3d[proposal],
                K,
                str(query["solver"].item()),
                float(query["reprojection_error"]),
                float(query["confidence"]),
                int(query["max_iterations"]),
                int(query["min_iterations"]),
                ransac_seed=int(seed),
                return_diagnostics=True,
            )
            pose, inliers, refinement_accepted = _refine_with_full_inliers(
                pose,
                p2d,
                p3d,
                K,
                float(query["reprojection_error"]),
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            ae, te = pose_error(pose, query["gt_pose_w2c"])
            errors[mode]["ae"].append(ae)
            errors[mode]["te"].append(te)
            errors[mode]["time"].append(elapsed_ms)
            hypotheses = diagnostics.get("ransac_actual_hypotheses")
            if hypotheses is not None:
                errors[mode]["hypotheses"].append(int(hypotheses))
            query_record[mode] = {
                "proposal_count": int(proposal.size),
                "full_score_count": int(p2d.shape[0]),
                "full_inlier_count": int(inliers.size),
                "full_refinement_accepted": bool(refinement_accepted),
                "hypotheses": hypotheses,
                "solver_and_refine_ms": elapsed_ms,
                "ae_deg": ae,
                "te_cm": te,
            }
        records.append(query_record)
    summary = {}
    for mode, values in errors.items():
        summary[mode] = {
            **summarize_pose_errors(values["ae"], values["te"]),
            "solver_and_refine_ms_mean": float(np.mean(values["time"])),
            "solver_and_refine_ms_median": float(np.median(values["time"])),
            "hypotheses_mean": (
                float(np.mean(values["hypotheses"]))
                if values["hypotheses"]
                else None
            ),
            "hypotheses_median": (
                float(np.median(values["hypotheses"]))
                if values["hypotheses"]
                else None
            ),
        }
    report = {
        "schema": "lafgs_dependency_aware_ransac",
        "version": 2,
        "seed": int(seed),
        "proposal_budget": int(proposal_budget),
        "full_correspondences_used_for_scoring_and_refinement": True,
        "summary": summary,
        "queries": records,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[
            "uniform",
            "family",
            "family_cell",
            "dependency",
            "dependency_cell",
        ],
        default=[
            "uniform",
            "family",
            "family_cell",
            "dependency",
            "dependency_cell",
        ],
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--proposal_budget", type=int, default=1024)
    args = parser.parse_args()
    report = evaluate(
        args.dump_dir,
        args.output,
        args.modes,
        args.seed,
        args.proposal_budget,
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
