#!/usr/bin/env python3
"""Dynamic LaFGS active-map E/M steps over one unified anchor universe."""

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
from scripts.run_lafgs_alternating_structure import (
    _deployment_valid_mask,
    _run_query_pose,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema") != "lafgs_materialized_anchor_map":
        raise ValueError(f"Unsupported anchor map: {path}")
    return state


def _initial_active_rows(initial: dict, universe: dict) -> torch.Tensor:
    count = int(universe["anchor_xyz"].shape[0])
    active = torch.zeros(count, dtype=torch.bool)
    canonical = int(initial.get("canonical_anchor_count", 0))
    if canonical <= 0 or canonical > count:
        raise ValueError("initial map has an invalid canonical anchor count")
    active[:canonical] = True
    update = initial.get("alternating_structure_update", {})
    accepted = torch.as_tensor(
        update.get("accepted_candidate_pool_rows", []), dtype=torch.long
    )
    if accepted.numel():
        active[accepted] = True
    expected = int(initial["anchor_xyz"].shape[0])
    if int(active.sum()) != expected:
        raise ValueError(
            f"Cannot map initial rows into universe: {int(active.sum())} "
            f"versus {expected}"
        )
    return active


def _overlay_initial_features(
    initial: dict, universe: dict, active: torch.Tensor
) -> torch.Tensor:
    features = torch.as_tensor(universe["anchor_features"]).float().clone()
    rows = torch.nonzero(active, as_tuple=False).reshape(-1)
    source = torch.as_tensor(initial["source_primitive_ids"]).long()
    target = torch.as_tensor(universe["source_primitive_ids"])[rows].long()
    track_source = torch.as_tensor(initial["track_cluster_ids"]).long()
    track_target = torch.as_tensor(universe["track_cluster_ids"])[rows].long()
    if not torch.equal(source, target) or not torch.equal(
        track_source, track_target
    ):
        raise ValueError("initial-map identities do not align with universe rows")
    features[rows] = torch.as_tensor(initial["anchor_features"]).float()
    return F.normalize(features, dim=1)


def _project_error_px(
    points: torch.Tensor,
    keypoints: torch.Tensor,
    K: torch.Tensor,
    pose_w2c: torch.Tensor,
) -> torch.Tensor:
    points = points.float()
    pose = pose_w2c.float()
    camera = points @ pose[:3, :3].T + pose[:3, 3]
    projected = camera @ K.float().T
    uv = projected[:, :2] / projected[:, 2:].clamp_min(1e-8)
    error = torch.linalg.norm(uv - keypoints.float(), dim=1)
    return torch.where(camera[:, 2] > 1e-6, error, torch.full_like(error, 1e6))


@torch.no_grad()
def _mine(
    query_cache: dict,
    names: list[str],
    xyz: torch.Tensor,
    features: torch.Tensor,
    active: torch.Tensor,
    deployment_masks,
    device: torch.device,
) -> tuple[
    list[dict], torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray
]:
    active_rows = torch.nonzero(active, as_tuple=False).reshape(-1)
    inactive_rows = torch.nonzero(~active, as_tuple=False).reshape(-1)
    active_features = features[active_rows].to(device)
    inactive_features = (
        features[inactive_rows].to(device) if inactive_rows.numel() else None
    )
    winner_count = torch.zeros(active.numel(), dtype=torch.long)
    clean_count = torch.zeros(active.numel(), dtype=torch.long)
    add_benefit = torch.zeros(active.numel(), dtype=torch.long)
    add_harm = torch.zeros(active.numel(), dtype=torch.long)
    matching_runtime = np.zeros(len(names), dtype=np.float64)
    states = []
    for query_index, name in enumerate(names):
        cached = query_cache[name]
        valid = _deployment_valid_mask(cached, name, deployment_masks)
        rows = torch.nonzero(valid, as_tuple=False).reshape(-1)
        descriptors = F.normalize(
            torch.as_tensor(cached["native_descriptors"])[rows].float(),
            dim=1,
        ).to(device)
        torch.cuda.synchronize(device)
        matching_start = time.perf_counter()
        active_score, active_local = (
            descriptors @ active_features.T
        ).max(dim=1)
        torch.cuda.synchronize(device)
        matching_runtime[query_index] = time.perf_counter() - matching_start
        active_index = active_rows[active_local.cpu()]
        keypoints = (
            torch.as_tensor(cached["native_keypoints"])[rows].float()
            + float(cached.get("pixel_center_offset", 0.5))
        )
        current_error = _project_error_px(
            xyz[active_index],
            keypoints,
            torch.as_tensor(cached["native_K"]),
            torch.as_tensor(cached["pose_w2c"]),
        )
        winner_count.index_add_(
            0, active_index, torch.ones_like(active_index)
        )
        clean_rows = torch.unique(active_index[current_error <= 4.0])
        clean_count.index_add_(
            0, clean_rows, torch.ones_like(clean_rows)
        )
        state = {
            "query_rows": rows,
            "indices": active_index,
            "scores": active_score.half().cpu(),
        }
        if inactive_features is not None:
            inactive_score, inactive_local = (
                descriptors @ inactive_features.T
            ).max(dim=1)
            inactive_index = inactive_rows[inactive_local.cpu()]
            wins = inactive_score.cpu() > active_score.cpu()
            candidate_error = _project_error_px(
                xyz[inactive_index],
                keypoints,
                torch.as_tensor(cached["native_K"]),
                torch.as_tensor(cached["pose_w2c"]),
            )
            beneficial = wins & (candidate_error <= 4.0) & (
                current_error > 4.0
            )
            harmful = wins & (candidate_error > 4.0) & (
                current_error <= 4.0
            )
            add_benefit.index_add_(
                0,
                inactive_index[beneficial],
                torch.ones_like(inactive_index[beneficial]),
            )
            add_harm.index_add_(
                0,
                inactive_index[harmful],
                torch.ones_like(inactive_index[harmful]),
            )
            state.update(
                {
                    "inactive_indices": inactive_index,
                    "inactive_scores": inactive_score.half().cpu(),
                }
            )
        states.append(state)
        if (query_index + 1) % 25 == 0 or query_index + 1 == len(names):
            print(
                f"E-step matched {query_index + 1}/{len(names)} queries",
                flush=True,
            )
    return (
        states,
        winner_count,
        clean_count,
        add_benefit - add_harm,
        matching_runtime,
    )


def _proposal_assignments(
    states: list[dict],
    proposal_active: torch.Tensor,
    features: torch.Tensor,
    query_cache: dict,
    names: list[str],
    device: torch.device,
) -> tuple[
    list[torch.Tensor], list[torch.Tensor], np.ndarray, np.ndarray
]:
    active_rows = torch.nonzero(proposal_active, as_tuple=False).reshape(-1)
    active_features = features[active_rows].to(device)
    indices, scores, affected = [], [], []
    matching_runtime = np.zeros(len(names), dtype=np.float64)
    with torch.no_grad():
        for query_index, (name, old) in enumerate(zip(names, states)):
            cached = query_cache[name]
            descriptors = F.normalize(
                torch.as_tensor(cached["native_descriptors"])[
                    old["query_rows"]
                ].float(),
                dim=1,
            ).to(device)
            torch.cuda.synchronize(device)
            matching_start = time.perf_counter()
            score, local = (descriptors @ active_features.T).max(dim=1)
            torch.cuda.synchronize(device)
            matching_runtime[query_index] = (
                time.perf_counter() - matching_start
            )
            index = active_rows[local.cpu()]
            indices.append(index)
            scores.append(score.half().cpu())
            affected.append(not torch.equal(index, old["indices"]))
    return (
        indices,
        scores,
        np.flatnonzero(np.asarray(affected)),
        matching_runtime,
    )


def _evaluate_assignments(
    query_cache,
    names,
    xyz,
    indices,
    scores,
    query_rows,
    seed,
    subset=None,
    errors=None,
    hypotheses=None,
    runtimes=None,
):
    if errors is None:
        errors = np.zeros(len(names), dtype=np.float64)
        hypotheses = np.zeros(len(names), dtype=np.float64)
        runtimes = np.zeros(len(names), dtype=np.float64)
        subset = np.arange(len(names))
    else:
        errors, hypotheses, runtimes = (
            errors.copy(),
            hypotheses.copy(),
            runtimes.copy(),
        )
    subset_list = np.asarray(subset).tolist()
    for completed, query_index in enumerate(subset_list, start=1):
        errors[query_index], hypotheses[query_index], runtimes[query_index] = (
            _run_query_pose(
                query_cache[names[query_index]],
                xyz,
                indices[query_index],
                scores[query_index].float(),
                query_rows[query_index],
                seed=seed + query_index,
            )
        )
        if completed % 25 == 0 or completed == len(subset_list):
            print(
                f"PnP replay {completed}/{len(subset_list)} queries",
                flush=True,
            )
    return errors, hypotheses, runtimes


def _materialize(universe: dict, active: torch.Tensor, report: dict) -> dict:
    rows = torch.nonzero(active, as_tuple=False).reshape(-1)
    output = {}
    universe_count = int(active.numel())
    for key, value in universe.items():
        if torch.is_tensor(value) and value.ndim and value.shape[0] == universe_count:
            output[key] = value[rows]
        elif not key.startswith("full_prior_source_group_"):
            output[key] = value
    output["anchor_ids"] = torch.arange(rows.numel(), dtype=torch.long)
    output["canonical_anchor_count"] = int(rows.numel())
    output["base_anchor_count"] = int(
        (torch.as_tensor(output["anchor_type"]) == 0).sum()
    )
    output["micro_anchor_count"] = int(
        (torch.as_tensor(output["anchor_type"]) != 0).sum()
    )
    output["requested_micro_anchor_budget"] = output["micro_anchor_count"]
    output["active_map_v2"] = report
    output["provenance"] = {
        **universe.get("provenance", {}),
        "active_map_v2_universe_rows": rows,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-map", required=True)
    parser.add_argument("--universe-map", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--deployment-mask-cache", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--retire-fraction-per-round", type=float, default=0.04)
    parser.add_argument("--operation-batch-size", type=int, default=64)
    parser.add_argument("--min-clean-support", type=int, default=2)
    parser.add_argument("--max-add-per-round", type=int, default=64)
    parser.add_argument("--complexity-weight", type=float, default=0.01)
    parser.add_argument("--worst-group-weight", type=float, default=0.15)
    parser.add_argument("--hypotheses-weight", type=float, default=0.002)
    parser.add_argument("--runtime-weight", type=float, default=0.002)
    parser.add_argument(
        "--frontend-runtime-seconds",
        type=float,
        default=0.30,
        help="Measured invariant native SuperPoint cost included in total time.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Active Map V2 requires CUDA")

    initial_path, universe_path = Path(args.initial_map), Path(args.universe_map)
    initial, universe = _load(initial_path), _load(universe_path)
    active = _initial_active_rows(initial, universe)
    features = _overlay_initial_features(initial, universe, active)
    xyz = torch.as_tensor(universe["anchor_xyz"]).float()
    payload = torch.load(args.query_cache, map_location="cpu", weights_only=False)
    query_cache = payload.get("queries", payload)
    names = list(query_cache)
    deployment_masks = None
    if args.deployment_mask_cache:
        with Path(args.deployment_mask_cache).open("rb") as handle:
            deployment_masks = pickle.load(handle)
    device = torch.device("cuda")
    config = PoseRiskConfig(
        complexity_weight=args.complexity_weight,
        reference_anchor_count=int(active.sum()),
        hypotheses_weight=args.hypotheses_weight,
        runtime_weight=args.runtime_weight,
        worst_group_weight=args.worst_group_weight,
    )
    group_ids = np.asarray([name.split("/", 1)[0] for name in names])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    operations, initial_summary = [], None
    errors = hypotheses = pnp_runtimes = runtimes = None

    for round_index in range(args.rounds):
        print(
            f"Round {round_index + 1}: re-mining {int(active.sum())} "
            f"active / {active.numel()} universe anchors",
            flush=True,
        )
        (
            states,
            winner_count,
            clean_count,
            add_utility,
            matching_runtimes,
        ) = _mine(
            query_cache,
            names,
            xyz,
            features,
            active,
            deployment_masks,
            device,
        )
        indices = [state["indices"] for state in states]
        scores = [state["scores"] for state in states]
        query_rows = [state["query_rows"] for state in states]
        errors, hypotheses, pnp_runtimes = _evaluate_assignments(
            query_cache,
            names,
            xyz,
            indices,
            scores,
            query_rows,
            args.seed,
        )
        runtimes = (
            pnp_runtimes
            + matching_runtimes
            + float(args.frontend_runtime_seconds)
        )
        current_summary = summarize_pose_risk(
            errors,
            anchor_count=int(active.sum()),
            config=config,
            group_ids=group_ids,
            hypotheses=hypotheses,
            runtime_seconds=runtimes,
        )
        if initial_summary is None:
            initial_summary = current_summary

        protected = active & (clean_count >= args.min_clean_support)
        zero_winner = torch.nonzero(
            active & ~protected & (winner_count == 0), as_tuple=False
        ).reshape(-1)
        retire_limit = int(
            max(1, round(int(active.sum()) * args.retire_fraction_per_round))
        )
        retire_rows = zero_winner[:retire_limit]

        inactive_rank = torch.argsort(add_utility, descending=True)
        add_rows = inactive_rank[
            (~active[inactive_rank]) & (add_utility[inactive_rank] > 0)
        ][: args.max_add_per_round]
        operation_specs = []
        if add_rows.numel() and retire_rows.numel() >= add_rows.numel():
            operation_specs.append(
                (
                    "swap",
                    add_rows,
                    retire_rows[: add_rows.numel()],
                )
            )
            retire_rows = retire_rows[add_rows.numel() :]
        elif add_rows.numel():
            operation_specs.append(("add", add_rows, torch.empty(0, dtype=torch.long)))
        if retire_rows.numel():
            operation_specs.append(
                ("retire", torch.empty(0, dtype=torch.long), retire_rows)
            )

        accepted_in_round = 0
        for operation_name, rows_to_add, rows_to_retire in operation_specs:
            proposal_active = active.clone()
            proposal_active[rows_to_add] = True
            proposal_active[rows_to_retire] = False
            (
                proposal_indices,
                proposal_scores,
                affected,
                proposal_matching_runtimes,
            ) = _proposal_assignments(
                states,
                proposal_active,
                features,
                query_cache,
                names,
                device,
            )
            (
                proposal_errors,
                proposal_hypotheses,
                proposal_pnp_runtimes,
            ) = (
                _evaluate_assignments(
                    query_cache,
                    names,
                    xyz,
                    proposal_indices,
                    proposal_scores,
                    query_rows,
                    args.seed,
                    subset=affected,
                    errors=errors,
                    hypotheses=hypotheses,
                    runtimes=pnp_runtimes,
                )
            )
            proposal_runtimes = (
                proposal_pnp_runtimes
                + proposal_matching_runtimes
                + float(args.frontend_runtime_seconds)
            )
            decision = evaluate_structure_proposal(
                errors,
                proposal_errors,
                current_anchor_count=int(active.sum()),
                proposal_anchor_count=int(proposal_active.sum()),
                config=config,
                group_ids=group_ids,
                current_hypotheses=hypotheses,
                proposal_hypotheses=proposal_hypotheses,
                current_runtime_seconds=runtimes,
                proposal_runtime_seconds=proposal_runtimes,
            )
            operation = {
                "round": round_index + 1,
                "operation": operation_name,
                "add_rows": rows_to_add.tolist(),
                "retire_rows": rows_to_retire.tolist(),
                "affected_query_count": int(affected.size),
                "protected_anchor_count": int(protected.sum()),
                **decision,
            }
            operations.append(operation)
            print(
                f"{operation_name}: add={rows_to_add.numel()} "
                f"retire={rows_to_retire.numel()} affected={affected.size} "
                f"accepted={decision['accepted']} "
                f"gain={decision['objective_gain']:.6g}",
                flush=True,
            )
            if decision["accepted"]:
                active = proposal_active
                errors, hypotheses, pnp_runtimes, runtimes = (
                    proposal_errors,
                    proposal_hypotheses,
                    proposal_pnp_runtimes,
                    proposal_runtimes,
                )
                accepted_in_round += 1
                break
        if accepted_in_round == 0:
            print("No positive operation; stopping.", flush=True)
            break

    final_summary = summarize_pose_risk(
        errors,
        anchor_count=int(active.sum()),
        config=config,
        group_ids=group_ids,
        hypotheses=hypotheses,
        runtime_seconds=runtimes,
    )
    report = {
        "schema": "lafgs_dynamic_active_map",
        "version": 2,
        "initial_anchor_count": int(_initial_active_rows(initial, universe).sum()),
        "universe_anchor_count": int(active.numel()),
        "final_anchor_count": int(active.sum()),
        "initial": initial_summary,
        "final": final_summary,
        "operations": operations,
        "config": vars(args),
        "provenance": {
            "initial_map": str(initial_path.resolve()),
            "initial_map_sha256": _sha256(initial_path),
            "universe_map": str(universe_path.resolve()),
            "universe_map_sha256": _sha256(universe_path),
            "query_cache": str(Path(args.query_cache).resolve()),
            "statistics_split": "all_895_mapping_train",
        },
    }
    state = _materialize(universe, active, report)
    state["anchor_features"] = features[active]
    torch.save(state, output_dir / "active_map_v2.pt")
    (output_dir / "active_map_v2_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["final"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
