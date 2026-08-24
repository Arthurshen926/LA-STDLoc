#!/usr/bin/env python3
"""Audit exact-identity and bounded joint-correction PoseLib headroom for V6."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from common.hashing import sha256_file
from common.v6_contracts import (
    DESCRIPTOR_SPLIT_SCHEMA,
    FEEDBACK_SCHEMA,
    RENDER_OBSERVATION_SCHEMA,
    require_mapping_only,
    require_schema,
)
from common.v6_pipeline_contract import validate_v6_feedback_scene_calibration
from evidence.observation_provider import GaussianRenderObservationProvider
from evidence.projective_loo import LeaveOneQueryOutProjectiveMap
from localization.pose_solver import pose_error, solve_absolute_pose
from map_learning.pose_set_oracle import PoseSetAction, normalized_pose_risk
from map_learning.v6_feedback_evaluator import _pose_neighborhoods
from map_learning.v6_pose_set_oracle import (
    apply_swaps,
    bounded_minimum_success_set,
    serialize_actions,
    unique_anchor_rows,
)
from topology.layered_sufficiency import visibility_image_cells


_SOURCE_PATHS = (
    "scripts/audit_v6_pose_set_oracle.py",
    "map_learning/v6_pose_set_oracle.py",
    "map_learning/pose_set_oracle.py",
    "map_learning/v6_feedback_evaluator.py",
    "evidence/projective_loo.py",
    "localization/pose_solver.py",
)


def _require(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"{label} SHA differs")
    return actual


def _producer() -> dict:
    root = Path(__file__).resolve().parents[1]
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("V6 pose-set oracle requires a clean worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "git_commit": commit,
        "worktree_clean": True,
        "source_sha256": {path: sha256_file(root / path) for path in _SOURCE_PATHS},
        "torch_version": torch.__version__,
    }


def _project_error(
    keypoints: torch.Tensor,
    anchors: torch.Tensor,
    *,
    xyz: torch.Tensor,
    intrinsics: torch.Tensor,
    pose_w2c: torch.Tensor,
) -> torch.Tensor:
    camera = xyz[anchors].float() @ pose_w2c[:3, :3].float().T
    camera = camera + pose_w2c[:3, 3].float()
    projected_h = camera @ intrinsics.float().T
    projected = projected_h[:, :2] / projected_h[:, 2:].clamp_min(1e-8)
    error = torch.linalg.norm(projected - keypoints.float(), dim=1)
    return torch.where(
        torch.isfinite(error) & (camera[:, 2] > 0),
        error,
        torch.full_like(error, float("inf")),
    )


def _summarize(rows: list[dict]) -> dict:
    full_success = [bool(row["full_identity"]["success"]) for row in rows]
    correction = [row["correction_set"] for row in rows]
    found = [row for row in correction if row["available"]]
    return {
        "baseline_failed_query_count": len(rows),
        "full_identity_success_count": int(sum(full_success)),
        "full_identity_success_fraction": float(np.mean(full_success)) if rows else 0.0,
        "bounded_correction_success_count": len(found),
        "bounded_correction_success_fraction": (
            float(len(found) / len(rows)) if rows else 0.0
        ),
        "bounded_correction_depth_median": (
            float(np.median([row["depth"] for row in found])) if found else None
        ),
        "bounded_correction_depth_maximum": (
            max((int(row["depth"]) for row in found), default=None)
        ),
        "full_identity_te_cm_median": (
            float(np.median([row["full_identity"]["te_cm"] for row in rows]))
            if rows
            else None
        ),
        "full_identity_ae_deg_median": (
            float(np.median([row["full_identity"]["ae_deg"] for row in rows]))
            if rows
            else None
        ),
    }


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict:
    if int(args.cpu_threads) < 1:
        raise ValueError("CPU thread count must be positive")
    if int(args.maximum_actions) < 1 or int(args.maximum_depth) < 1:
        raise ValueError("oracle action and depth limits must be positive")
    if int(args.beam_width) < 1:
        raise ValueError("oracle beam width must be positive")
    torch.set_num_threads(int(args.cpu_threads))
    os.environ["OMP_NUM_THREADS"] = str(int(args.cpu_threads))
    os.environ["MKL_NUM_THREADS"] = str(int(args.cpu_threads))
    paths = {
        "map": args.map.resolve(),
        "observation_cache": args.observation_cache.resolve(),
        "feedback": args.feedback.resolve(),
        "mapping_split": args.mapping_split.resolve(),
        "scene_calibration": args.scene_calibration.resolve(),
    }
    hashes = {
        name: _require(paths[name], getattr(args, f"expected_{name}_sha256"), name)
        for name in paths
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    state = torch.load(paths["map"], map_location="cpu", weights_only=False)
    cache = torch.load(
        paths["observation_cache"], map_location="cpu", weights_only=False
    )
    feedback_payload = torch.load(
        paths["feedback"], map_location="cpu", weights_only=False
    )
    feedback = feedback_payload.get("feedback", feedback_payload)
    split = json.loads(paths["mapping_split"].read_text())
    calibration = json.loads(paths["scene_calibration"].read_text())
    require_mapping_only(state.get("provenance", {}), label="V6 oracle map")
    require_schema(cache, RENDER_OBSERVATION_SCHEMA, label="V6 oracle observations")
    require_schema(feedback, FEEDBACK_SCHEMA, label="V6 oracle feedback")
    require_schema(split, DESCRIPTOR_SPLIT_SCHEMA, label="V6 oracle mapping split")
    if split.get("source_feedback_sha256") != hashes["feedback"]:
        raise ValueError("mapping split is not bound to the oracle feedback")
    input_hashes = feedback_payload.get("input_sha256", {})
    for key in ("map", "observation_cache", "scene_calibration"):
        if input_hashes.get(key) != hashes[key]:
            raise ValueError(f"feedback is not bound to the oracle {key}")
    names = list(feedback.get("query_names", ()))
    cache_names = list(cache.get("query_names", cache.get("queries", {})))
    if names != cache_names or names != list(state.get("v6_mapping_query_names", ())):
        raise ValueError("oracle map/cache/feedback query registries differ")
    validation = torch.as_tensor(split.get("validation_query_indices", ())).long()
    if validation.numel() == 0 or int(validation.min()) < 0 or int(validation.max()) >= len(names):
        raise ValueError("oracle validation registry is invalid")
    parameters = calibration.get("parameters", {})
    ransac_px = validate_v6_feedback_scene_calibration(
        calibration, query_count=len(names)
    )
    task_translation_m = float(parameters.get("task_translation_m", 0.05))
    task_rotation_deg = float(parameters.get("task_rotation_deg", 5.0))
    if not task_translation_m > 0 or not task_rotation_deg > 0:
        raise ValueError("oracle task thresholds must be positive")
    observations = GaussianRenderObservationProvider(cache, query_names=names)
    neighborhoods = _pose_neighborhoods(observations, int(args.loo_pose_neighbors))
    replay = LeaveOneQueryOutProjectiveMap(
        state, observations, affected_anchor_policy="rebuild"
    )
    base_xyz = torch.as_tensor(state["anchor_xyz"]).float()
    validation_set = set(validation.tolist())
    selected_queries = [
        index
        for index, record in enumerate(feedback["records"])
        if index in validation_set and not bool(record.get("pose_success"))
    ]
    if not selected_queries:
        raise ValueError("validation split contains no failed baseline queries")
    started = time.perf_counter()
    output_rows = []
    for completed, query_index in enumerate(selected_queries, start=1):
        record = feedback["records"][query_index]
        expected_excluded = torch.as_tensor(record.get("excluded_query_indices")).long()
        if not torch.equal(expected_excluded, neighborhoods[query_index]):
            raise ValueError("oracle LOO neighborhood differs from feedback")
        update = replay.query_update(
            query_index, excluded_queries=neighborhoods[query_index]
        )
        xyz = base_xyz.clone()
        active = torch.ones(base_xyz.shape[0], dtype=torch.bool)
        affected = torch.as_tensor(update["anchor_rows"]).long()
        if affected.numel():
            xyz[affected] = torch.as_tensor(update["anchor_xyz"]).float()
            active[affected] = torch.as_tensor(update["valid"]).bool()
        view = observations.build_view(query_index)
        keypoints = view.physical_keypoints.float()
        intrinsics = view.intrinsics.float()
        gt_pose = view.pose_w2c.float()
        winners = torch.as_tensor(record["winner_anchor_ids"]).long()
        winner_scores = torch.as_tensor(record["winner_scores"]).float()
        if winners.shape != winner_scores.shape or winners.shape[0] != keypoints.shape[0]:
            raise ValueError("oracle feedback winner rows are not aligned")
        if not bool(active[winners].all()):
            raise ValueError("oracle feedback winner is not LOO active")
        exact_pairs = torch.as_tensor(
            record.get("exact_identity_positive_pairs", ())
        ).long().reshape(-1, 2)
        if exact_pairs.numel() and not bool(active[exact_pairs[:, 1]].all()):
            raise ValueError("oracle exact identity is not LOO active")
        exact_pairs = unique_anchor_rows(
            exact_pairs,
            keypoints=keypoints,
            anchor_xyz=xyz,
            intrinsics=intrinsics,
            pose_w2c=gt_pose,
        )

        solve_cache: dict[bytes, dict] = {}

        def solve(assignments: np.ndarray, rows: np.ndarray | None = None) -> dict:
            assignments = np.asarray(assignments, dtype=np.int64)
            if rows is None:
                rows = np.arange(assignments.shape[0], dtype=np.int64)
            rows = np.asarray(rows, dtype=np.int64)
            cache_key = assignments.tobytes() + rows.tobytes()
            if cache_key in solve_cache:
                return solve_cache[cache_key]
            estimate = solve_absolute_pose(
                keypoints[torch.from_numpy(rows)].numpy(),
                xyz[torch.from_numpy(assignments)].numpy(),
                intrinsics.numpy(),
                reprojection_error_px=float(ransac_px),
                confidence=0.99999,
                max_iterations=100000,
                min_iterations=1000,
                seed=int(args.seed),
            )
            ae_deg, te_cm = pose_error(estimate.pose_w2c, gt_pose.numpy())
            failed = int(np.asarray(estimate.inliers).size) < 4
            result = {
                "te_cm": float(te_cm),
                "ae_deg": float(ae_deg),
                "success": bool(te_cm < 100.0 * task_translation_m and ae_deg < task_rotation_deg),
                "solver_failed": bool(failed),
                "inlier_count": int(np.asarray(estimate.inliers).size),
                "hypotheses": int(estimate.diagnostics.get("iterations", 0)),
                "risk": normalized_pose_risk(
                    translation_cm=te_cm,
                    rotation_deg=ae_deg,
                    translation_scale_m=task_translation_m,
                    rotation_scale_deg=task_rotation_deg,
                    failed=failed,
                ),
            }
            solve_cache[cache_key] = result
            return result

        full_identity = solve(
            exact_pairs[:, 1].numpy(), exact_pairs[:, 0].numpy()
        )
        full_identity["correspondence_count"] = int(exact_pairs.shape[0])
        full_identity["image_cell_count"] = int(
            torch.unique(
                visibility_image_cells(
                    keypoints[exact_pairs[:, 0]], image_hw=view.image_hw
                )
            ).numel()
        )

        confusion = torch.as_tensor(record.get("confusion_pairs", ())).long().reshape(-1, 3)
        positive_by_row = {int(row): int(anchor) for row, anchor in exact_pairs.tolist()}
        inlier_rows = torch.as_tensor(record.get("inlier_query_rows", ())).long()
        inlier_clean = torch.as_tensor(record.get("inlier_clean_mask", ())).bool()
        harmful = set(inlier_rows[~inlier_clean].tolist())
        current_error = _project_error(
            keypoints, winners, xyz=xyz, intrinsics=intrinsics, pose_w2c=gt_pose
        )
        actions = []
        for row, negative, positive in confusion.tolist():
            if positive_by_row.get(int(row)) != int(positive):
                continue
            positive_error = _project_error(
                keypoints[row : row + 1],
                torch.tensor([positive]),
                xyz=xyz,
                intrinsics=intrinsics,
                pose_w2c=gt_pose,
            )[0]
            gain = float(current_error[row] - positive_error)
            priority = 1000.0 * float(row in harmful) + min(max(gain, -100.0), 100.0)
            priority += 1e-6 * float(winner_scores[row])
            actions.append(PoseSetAction("swap", int(row), int(positive), priority))
        actions = sorted(actions, key=lambda action: (-action.priority, action))[
            : int(args.maximum_actions)
        ]

        def evaluate(selected: tuple[PoseSetAction, ...]) -> dict:
            revised = apply_swaps(winners.numpy(), selected)
            return solve(revised)

        selected, selected_outcome, trace = bounded_minimum_success_set(
            actions,
            evaluate,
            maximum_depth=int(args.maximum_depth),
            beam_width=int(args.beam_width),
        )
        selected_rows = torch.tensor(
            [] if selected is None else [action.row for action in selected],
            dtype=torch.long,
        )
        correction = {
            "available": selected is not None,
            "depth": None if selected is None else len(selected),
            "actions": serialize_actions(selected),
            "candidate_action_count": len(actions),
            "trace": trace,
            "image_cell_count": (
                0
                if selected_rows.numel() == 0
                else int(
                    torch.unique(
                        visibility_image_cells(
                            keypoints[selected_rows], image_hw=view.image_hw
                        )
                    ).numel()
                )
            ),
            "outcome": selected_outcome,
        }
        output_rows.append(
            {
                "query_index": query_index,
                "image_name": names[query_index],
                "baseline": {
                    "te_cm": float(record["te_cm"]),
                    "ae_deg": float(record["ae_deg"]),
                    "pose_success": bool(record["pose_success"]),
                    "correspondence_count": int(winners.numel()),
                },
                "full_identity": full_identity,
                "correction_set": correction,
            }
        )
        print(
            json.dumps(
                {
                    "event": "v6_pose_set_oracle",
                    "query": completed,
                    "query_count": len(selected_queries),
                    "full_identity_success": full_identity["success"],
                    "correction_depth": correction["depth"],
                }
            ),
            flush=True,
        )

    report = {
        "schema": "lafgs_v6_exact_identity_pose_set_oracle",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "scope": "mapping_action_holdout_baseline_pose_failures",
        "summary": _summarize(output_rows),
        "queries": output_rows,
        "configuration": {
            "loo_pose_neighbors": int(args.loo_pose_neighbors),
            "loo_affected_anchor_policy": "rebuild",
            "maximum_actions": int(args.maximum_actions),
            "maximum_depth": int(args.maximum_depth),
            "beam_width": int(args.beam_width),
            "seed": int(args.seed),
            "ransac_reprojection_px": float(ransac_px),
            "task_translation_m": task_translation_m,
            "task_rotation_deg": task_rotation_deg,
            "correction_search_is_globally_exact": False,
            "full_identity_one_vote_per_anchor": True,
        },
        "input_sha256": hashes,
        "producer": _producer(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("map", "observation_cache", "feedback", "mapping_split", "scene_calibration"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
        parser.add_argument(f"--expected-{name.replace('_', '-')}-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loo-pose-neighbors", type=int, default=3)
    parser.add_argument("--maximum-actions", type=int, default=20)
    parser.add_argument("--maximum-depth", type=int, default=12)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"output": str(args.output), **report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
