#!/usr/bin/env python3
"""Inference-aligned Structure M-step for LaFGS alternating reconstruction.

The script matches every mapping query against the current active map once,
then proposes batches from an inactive candidate pool.  A batch is accepted
only after replaying the exact hard top-1 and fixed-seed PnP path on affected
queries and passing global pose-risk and protected-clean gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from localization_training.alternating_map import (
    PoseRiskConfig,
    evaluate_structure_proposal,
    summarize_pose_risk,
)
from localization_training.ulf_initializer import sample_mask_at_grid_uv
from utils.pose_utils import cal_pose_error, solve_pose


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_map(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema") != "lafgs_materialized_anchor_map":
        raise ValueError(f"Unsupported anchor map: {path}")
    return state


def _deployment_valid_mask(
    cached: dict, name: str, deployment_masks: dict | None
) -> torch.Tensor:
    keypoints = torch.as_tensor(cached["native_keypoints"]).float()
    valid = torch.ones(keypoints.shape[0], dtype=torch.bool)
    if cached.get("native_valid_mask") is not None:
        valid &= sample_mask_at_grid_uv(
            torch.as_tensor(cached["native_valid_mask"]), keypoints
        ).cpu()
    if deployment_masks is None or name not in deployment_masks:
        return valid
    channels = deployment_masks[name]
    if len(channels) < 3:
        raise ValueError(f"deployment mask for {name!r} needs three channels")
    target_hw = tuple(
        int(value) for value in cached.get("native_input_hw", ())
    )
    if len(target_hw) != 2:
        raise ValueError("native_input_hw is required for deployment masks")
    resized = []
    for channel in channels[:3]:
        mask = torch.as_tensor(channel).detach().cpu().float()
        while mask.ndim > 2:
            mask = mask.squeeze(0)
        resized.append(
            F.interpolate(
                mask[None, None], size=target_hw, mode="nearest"
            )[0, 0].bool()
        )
    deployment_valid = resized[0] & resized[1] & resized[2]
    valid &= sample_mask_at_grid_uv(deployment_valid, keypoints).cpu()
    return valid


def _pose_error_m(predicted: np.ndarray, target: torch.Tensor) -> float:
    _, translation_cm = cal_pose_error(
        np.asarray(predicted), torch.as_tensor(target).cpu().numpy()
    )
    return float(translation_cm) / 100.0


def _run_query_pose(
    cached: dict,
    landmark_xyz: torch.Tensor,
    landmark_indices: torch.Tensor,
    scores: torch.Tensor,
    keypoint_rows: torch.Tensor,
    *,
    seed: int,
) -> tuple[float, float, float]:
    keypoints = (
        torch.as_tensor(cached["native_keypoints"]).float()[keypoint_rows]
        + float(cached.get("pixel_center_offset", 0.5))
    )
    start = time.perf_counter()
    pose, _, diagnostics = solve_pose(
        keypoints.cpu().numpy(),
        landmark_xyz[landmark_indices].cpu().numpy(),
        torch.as_tensor(cached["native_K"]).cpu().numpy(),
        solver="poselib",
        reprojection_error=12.0,
        confidence=0.99999,
        max_iterations=100000,
        min_iterations=1000,
        scores=scores.cpu().numpy(),
        ransac_seed=int(seed),
        return_diagnostics=True,
    )
    elapsed = time.perf_counter() - start
    hypotheses = diagnostics.get("ransac_actual_hypotheses")
    return (
        _pose_error_m(pose, cached["pose_w2c"]),
        float(hypotheses) if hypotheses is not None else float("nan"),
        elapsed,
    )


def _materialize(
    base: dict,
    pool: dict,
    candidate_rows: torch.Tensor,
    *,
    report: dict,
    provenance: dict,
) -> dict:
    candidate_rows = torch.as_tensor(candidate_rows, dtype=torch.long)
    row_fields = (
        "source_primitive_ids",
        "track_cluster_ids",
        "anchor_xyz",
        "anchor_features",
        "anchor_type",
    )
    output = dict(base)
    for key in row_fields:
        output[key] = torch.cat(
            (
                torch.as_tensor(base[key]),
                torch.as_tensor(pool[key])[candidate_rows],
            )
        )
    output["anchor_ids"] = torch.arange(
        output["anchor_xyz"].shape[0], dtype=torch.long
    )
    output["canonical_anchor_count"] = int(base["anchor_xyz"].shape[0])
    output["micro_anchor_count"] = int(
        output["anchor_xyz"].shape[0] - int(output["base_anchor_count"])
    )
    output["requested_micro_anchor_budget"] = int(
        output["micro_anchor_count"]
    )
    output["alternating_structure_update"] = report
    output["provenance"] = {
        **base.get("provenance", {}),
        **provenance,
    }
    return output


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-map", required=True)
    parser.add_argument("--candidate-pool-map", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument(
        "--deployment-mask-cache",
        default="",
        help="masks.pkl used by deployment keypoint filtering.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Limit the ordered inactive pool; zero uses every available row.",
    )
    parser.add_argument("--max-active-change-fraction", type=float, default=0.05)
    parser.add_argument("--translation-scale-m", type=float, default=0.10)
    parser.add_argument("--cvar-fraction", type=float, default=0.20)
    parser.add_argument("--cvar-weight", type=float, default=0.35)
    parser.add_argument("--complexity-weight", type=float, default=0.002)
    parser.add_argument("--max-median-regression-m", type=float, default=0.0005)
    parser.add_argument("--max-r5-regression", type=float, default=0.002)
    parser.add_argument("--max-protected-regressions", type=int, default=0)
    parser.add_argument("--minimum-objective-gain", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Structure M-step requires CUDA matching")
    if int(args.batch_size) <= 0:
        raise ValueError("batch size must be positive")

    base_path = Path(args.base_map).resolve()
    pool_path = Path(args.candidate_pool_map).resolve()
    cache_path = Path(args.query_cache).resolve()
    mask_path = (
        Path(args.deployment_mask_cache).resolve()
        if args.deployment_mask_cache
        else None
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base = _load_map(base_path)
    pool = _load_map(pool_path)
    base_count = int(base["anchor_xyz"].shape[0])
    if int(pool["anchor_xyz"].shape[0]) <= base_count:
        raise ValueError("candidate pool must extend the base map")
    if not torch.equal(
        torch.as_tensor(base["source_primitive_ids"]),
        torch.as_tensor(pool["source_primitive_ids"])[:base_count],
    ):
        raise ValueError("candidate pool does not preserve the base map prefix")
    pool_rows = torch.arange(
        base_count, int(pool["anchor_xyz"].shape[0]), dtype=torch.long
    )
    max_changes = max(
        1, int(round(base_count * float(args.max_active_change_fraction)))
    )
    pool_rows = pool_rows[:max_changes]
    if int(args.max_candidates) > 0:
        pool_rows = pool_rows[: int(args.max_candidates)]
    device = torch.device("cuda")
    base_features = F.normalize(
        torch.as_tensor(base["anchor_features"]).float(), dim=1
    ).to(device)
    candidate_features = F.normalize(
        torch.as_tensor(pool["anchor_features"])[pool_rows].float(), dim=1
    ).to(device)
    all_xyz = torch.cat(
        (
            torch.as_tensor(base["anchor_xyz"]).float(),
            torch.as_tensor(pool["anchor_xyz"])[pool_rows].float(),
        )
    )

    print(f"Loading query cache: {cache_path}", flush=True)
    query_payload = torch.load(
        cache_path, map_location="cpu", weights_only=False
    )
    query_cache = query_payload.get("queries", query_payload)
    deployment_masks = None
    if mask_path is not None:
        with mask_path.open("rb") as handle:
            deployment_masks = pickle.load(handle)
    names = list(query_cache)
    match_state = []
    for query_index, name in enumerate(names):
        cached = query_cache[name]
        descriptors = F.normalize(
            torch.as_tensor(cached["native_descriptors"]).float(), dim=1
        )
        query_rows = torch.arange(descriptors.shape[0], dtype=torch.long)
        valid = _deployment_valid_mask(cached, name, deployment_masks)
        query_rows = query_rows[valid]
        descriptors = descriptors[valid]
        descriptors = descriptors.to(device)
        base_scores, base_indices = (
            descriptors @ base_features.T
        ).max(dim=1)
        candidate_scores = descriptors @ candidate_features.T
        match_state.append(
            {
                "base_scores": base_scores.half().cpu(),
                "base_indices": base_indices.int().cpu(),
                "candidate_scores": candidate_scores.half().cpu(),
                "query_rows": query_rows,
            }
        )
        if (query_index + 1) % 25 == 0 or query_index + 1 == len(names):
            print(
                f"Matched {query_index + 1}/{len(names)} mapping queries",
                flush=True,
            )

    config = PoseRiskConfig(
        translation_scale_m=float(args.translation_scale_m),
        cvar_fraction=float(args.cvar_fraction),
        cvar_weight=float(args.cvar_weight),
        complexity_weight=float(args.complexity_weight),
        reference_anchor_count=base_count,
        max_median_regression_m=float(args.max_median_regression_m),
        max_r5_regression=float(args.max_r5_regression),
        max_protected_regressions=int(args.max_protected_regressions),
        minimum_objective_gain=float(args.minimum_objective_gain),
    )
    active = torch.zeros(pool_rows.numel(), dtype=torch.bool)
    current_scores = [
        state["base_scores"].float().clone() for state in match_state
    ]
    current_indices = [
        state["base_indices"].long().clone() for state in match_state
    ]
    errors = np.zeros(len(names), dtype=np.float64)
    hypotheses = np.zeros(len(names), dtype=np.float64)
    runtimes = np.zeros(len(names), dtype=np.float64)
    for query_index, name in enumerate(names):
        errors[query_index], hypotheses[query_index], runtimes[query_index] = (
            _run_query_pose(
                query_cache[name],
                all_xyz,
                current_indices[query_index],
                current_scores[query_index],
                match_state[query_index]["query_rows"],
                seed=int(args.seed) + query_index,
            )
        )
    baseline_summary = summarize_pose_risk(
        errors,
        anchor_count=base_count,
        config=config,
        hypotheses=hypotheses,
        runtime_seconds=runtimes,
    )
    print("Baseline: " + json.dumps(baseline_summary, sort_keys=True), flush=True)

    operations = []
    for start in range(0, pool_rows.numel(), int(args.batch_size)):
        local_rows = torch.arange(
            start,
            min(start + int(args.batch_size), pool_rows.numel()),
            dtype=torch.long,
        )
        proposal_active = active.clone()
        proposal_active[local_rows] = True
        proposal_scores = []
        proposal_indices = []
        affected = []
        active_rows = torch.nonzero(
            proposal_active, as_tuple=False
        ).reshape(-1)
        for query_index, state in enumerate(match_state):
            candidate_score, candidate_local = state[
                "candidate_scores"
            ][:, active_rows].float().max(dim=1)
            wins = candidate_score > state["base_scores"].float()
            score = state["base_scores"].float().clone()
            index = state["base_indices"].long().clone()
            score[wins] = candidate_score[wins]
            index[wins] = base_count + active_rows[candidate_local[wins]]
            proposal_scores.append(score)
            proposal_indices.append(index)
            affected.append(
                not torch.equal(index, current_indices[query_index])
            )
        affected_indices = np.flatnonzero(np.asarray(affected, dtype=bool))
        proposal_errors = errors.copy()
        proposal_hypotheses = hypotheses.copy()
        proposal_runtimes = runtimes.copy()
        for query_index in affected_indices.tolist():
            (
                proposal_errors[query_index],
                proposal_hypotheses[query_index],
                proposal_runtimes[query_index],
            ) = _run_query_pose(
                query_cache[names[query_index]],
                all_xyz,
                proposal_indices[query_index],
                proposal_scores[query_index],
                match_state[query_index]["query_rows"],
                seed=int(args.seed) + query_index,
            )
        decision = evaluate_structure_proposal(
            errors,
            proposal_errors,
            current_anchor_count=base_count + int(active.sum()),
            proposal_anchor_count=base_count + int(proposal_active.sum()),
            config=config,
            current_hypotheses=hypotheses,
            proposal_hypotheses=proposal_hypotheses,
            current_runtime_seconds=runtimes,
            proposal_runtime_seconds=proposal_runtimes,
        )
        operation = {
            "operation": "add_batch",
            "candidate_local_rows": local_rows.tolist(),
            "candidate_pool_rows": pool_rows[local_rows].tolist(),
            "affected_query_count": int(affected_indices.size),
            **decision,
        }
        operations.append(operation)
        print(
            f"Batch {start}:{start + local_rows.numel()} "
            f"affected={affected_indices.size} accepted={decision['accepted']} "
            f"gain={decision['objective_gain']:.6g} "
            f"gates={json.dumps(decision['gates'], sort_keys=True)}",
            flush=True,
        )
        if decision["accepted"]:
            active = proposal_active
            current_scores = proposal_scores
            current_indices = proposal_indices
            errors = proposal_errors
            hypotheses = proposal_hypotheses
            runtimes = proposal_runtimes
        partial_report = {
            "completed_candidate_count": int(start + local_rows.numel()),
            "accepted_candidate_local_rows": torch.nonzero(
                active, as_tuple=False
            ).reshape(-1).tolist(),
            "operations": operations,
        }
        temporary = output_dir / "alternating_report.partial.json.tmp"
        temporary.write_text(
            json.dumps(partial_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir / "alternating_report.partial.json")

    accepted_local = torch.nonzero(active, as_tuple=False).reshape(-1)
    accepted_pool_rows = pool_rows[accepted_local]
    final_summary = summarize_pose_risk(
        errors,
        anchor_count=base_count + int(active.sum()),
        config=config,
        hypotheses=hypotheses,
        runtime_seconds=runtimes,
    )
    report = {
        "schema": "lafgs_alternating_structure_report",
        "version": 1,
        "base_anchor_count": base_count,
        "candidate_count": int(pool_rows.numel()),
        "accepted_candidate_count": int(active.sum()),
        "accepted_candidate_pool_rows": accepted_pool_rows.tolist(),
        "batch_size": int(args.batch_size),
        "baseline": baseline_summary,
        "final": final_summary,
        "operations": operations,
        "stop_reason": (
            "no_positive_structure_batch"
            if not bool(active.any())
            else "candidate_pool_exhausted"
        ),
    }
    provenance = {
        "alternating_base_map_path": str(base_path),
        "alternating_base_map_sha256": _sha256(base_path),
        "alternating_candidate_pool_path": str(pool_path),
        "alternating_candidate_pool_sha256": _sha256(pool_path),
        "alternating_query_cache_path": str(cache_path),
        "alternating_query_cache_signature": query_payload.get("signature"),
        "alternating_deployment_mask_path": (
            str(mask_path) if mask_path is not None else None
        ),
        "alternating_deployment_mask_sha256": (
            _sha256(mask_path) if mask_path is not None else None
        ),
        "alternating_statistics_split": "all_895_mapping_train",
        "alternating_seed": int(args.seed),
    }
    state = _materialize(
        base,
        pool,
        accepted_pool_rows,
        report=report,
        provenance=provenance,
    )
    torch.save(state, output_dir / "active_map.pt")
    (output_dir / "alternating_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["final"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
