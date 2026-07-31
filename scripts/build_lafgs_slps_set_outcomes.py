#!/usr/bin/env python3
"""Build exact PoseLib set outcomes for scene-specific SLPS training."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
import torch

from localization_training.pose_sufficient_selector import (
    build_pose_sufficient_features,
    constrained_pose_sufficient_mask,
)
from localization_training.slps_selector import (
    SLPS_FEATURE_NAMES,
    beta_track_stability,
    build_relation_groups,
    build_slps_features,
)
from scripts.train_lafgs_basis_core_reserve_selector import (
    _cross_validation_folds,
)
from scripts.train_lafgs_pose_sufficient_selector import (
    _anchor_statistics,
    _records_by_name,
    _sha256_tensor,
    _trajectory,
)
from utils.pose_utils import cal_pose_error, solve_pose


_WORKER_QUERIES: list[dict] = []


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _stable_seed(name: str, seed: int) -> int:
    digest = hashlib.sha256(f"{name}:{int(seed)}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def _project_errors(
    xyz: torch.Tensor,
    keypoints: torch.Tensor,
    K: torch.Tensor,
    pose_w2c: torch.Tensor,
) -> torch.Tensor:
    camera = xyz @ pose_w2c[:3, :3].T + pose_w2c[:3, 3]
    projected = torch.empty_like(keypoints)
    projected[:, 0] = (
        K[0, 0] * camera[:, 0] / camera[:, 2].clamp_min(1e-8) + K[0, 2]
    )
    projected[:, 1] = (
        K[1, 1] * camera[:, 1] / camera[:, 2].clamp_min(1e-8) + K[1, 2]
    )
    return torch.linalg.norm(projected - keypoints, dim=1)


def _stratified_query_names(
    names: list[str],
    dynamic_by_name: dict[str, dict],
    *,
    maximum_queries: int,
) -> list[str]:
    if int(maximum_queries) <= 0:
        return list(names)
    maximum = min(max(int(maximum_queries), 1), len(names))
    if maximum >= len(names):
        return list(names)
    selected: set[str] = set()
    trajectories: dict[str, list[str]] = defaultdict(list)
    for name in names:
        trajectories[_trajectory(name)].append(name)
    uniform_budget = maximum // 2
    for trajectory_names in trajectories.values():
        share = max(
            1,
            round(uniform_budget * len(trajectory_names) / len(names)),
        )
        indices = np.linspace(
            0, len(trajectory_names) - 1, min(share, len(trajectory_names))
        ).round().astype(int)
        selected.update(trajectory_names[index] for index in indices)

    def difficulty(name: str) -> tuple[float, float, str]:
        record = dynamic_by_name[name]
        harmful = torch.as_tensor(record["harmful_inlier_mask"]).float()
        anchors = torch.as_tensor(record["top1_anchor_indices"]).long()
        repeated = 1.0 - float(torch.unique(anchors).numel()) / max(len(anchors), 1)
        return (
            float(record["te_cm"]),
            float(harmful.mean()) + repeated,
            name,
        )

    difficult = sorted(names, key=difficulty, reverse=True)
    for name in difficult:
        if len(selected) >= maximum:
            break
        selected.add(name)
    return [name for name in names if name in selected][:maximum]


def _query_contribution(
    record: dict, anchor_count: int
) -> dict[str, torch.Tensor]:
    anchors = torch.as_tensor(record["top1_anchor_indices"]).long()
    output = {}
    values = {
        "attempts": torch.ones(len(anchors)),
        "clean": (
            torch.as_tensor(record["gt_reprojection_errors_px"]).float()
            <= 2.0
        ).float(),
        "clean_inlier": torch.as_tensor(
            record["clean_inlier_mask"]
        ).float(),
        "harmful_inlier": torch.as_tensor(
            record["harmful_inlier_mask"]
        ).float(),
    }
    for name, value in values.items():
        folded = torch.zeros(anchor_count)
        folded.index_add_(0, anchors, value)
        output[name] = folded
    return output


def _ranked_unique_mask(
    order: torch.Tensor,
    *,
    budget: int,
    dependency_groups: torch.Tensor | None = None,
    maximum_per_dependency: int = 0,
) -> torch.Tensor:
    order = torch.as_tensor(order).long().reshape(-1)
    target = min(max(int(budget), 4), len(order))
    if target >= len(order):
        return torch.ones(len(order), dtype=torch.bool)
    selected = []
    selected_set = set()
    group_count: dict[int, int] = defaultdict(int)
    groups = (
        torch.as_tensor(dependency_groups).long().reshape(-1).tolist()
        if dependency_groups is not None
        else None
    )
    if groups is not None and maximum_per_dependency > 0:
        for index in order.tolist():
            group = groups[index]
            if group_count[group] >= int(maximum_per_dependency):
                continue
            selected.append(index)
            selected_set.add(index)
            group_count[group] += 1
            if len(selected) >= target:
                break
    for index in order.tolist():
        if len(selected) >= target:
            break
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)
    mask = torch.zeros(len(order), dtype=torch.bool)
    mask[torch.as_tensor(selected).long()] = True
    return mask


def _make_subsets(
    query: dict,
    *,
    budgets: tuple[int, ...],
    seed: int,
) -> list[dict]:
    features = query["features"].float()
    strict = query["strict_clean"].bool()
    harmful = query["harmful"].bool()
    score = features[:, 0]
    margin = features[:, 1]
    keypoint_score = features[:, 5]
    quality = (
        1.2 * features[:, 3]
        + 1.0 * features[:, 4]
        + 0.5 * keypoint_score
        + 1.5 * features[:, 11]
        - 2.0 * features[:, 12]
        - 0.5 * features[:, 14]
    )
    count = len(features)
    masks: list[tuple[str, torch.Tensor]] = [
        ("all", torch.ones(count, dtype=torch.bool))
    ]
    for budget in budgets:
        target = min(max(int(budget), 4), count)
        orders = {
            "score": torch.argsort(score, descending=True, stable=True),
            "margin": torch.argsort(margin, descending=True, stable=True),
            "quality": torch.argsort(quality, descending=True, stable=True),
        }
        for label, order in orders.items():
            masks.append(
                (
                    f"{label}_{target}",
                    _ranked_unique_mask(order, budget=target),
                )
            )
        balanced = constrained_pose_sufficient_mask(
            torch.sigmoid(quality),
            image_cells=query["relation_groups"][:, 0],
            dependency_groups=query["relation_groups"][:, 2],
            source_groups=query["relation_groups"][:, 3],
            xyz=query["xyz"],
            budget=target,
            minimum_per_image_cell=max(target // 128, 1),
            minimum_per_spatial_bin=max(target // 256, 1),
            maximum_per_dependency=3,
            maximum_per_source=2,
        )
        masks.append((f"balanced_{target}", balanced))
        masks.append(
            (
                f"dependency_unique_{target}",
                _ranked_unique_mask(
                    orders["quality"],
                    budget=target,
                    dependency_groups=query["relation_groups"][:, 2],
                    maximum_per_dependency=1,
                ),
            )
        )
        generator = torch.Generator().manual_seed(
            _stable_seed(query["query_name"], seed + target)
        )
        random_order = torch.randperm(count, generator=generator)
        masks.append(
            (
                f"random_{target}",
                _ranked_unique_mask(random_order, budget=target),
            )
        )
        oracle_order = torch.argsort(
            strict.float() * 100.0 + quality,
            descending=True,
            stable=True,
        )
        masks.append(
            (
                f"strict_oracle_{target}",
                _ranked_unique_mask(oracle_order, budget=target),
            )
        )
    hard_order = torch.argsort(
        harmful.float() * 100.0
        + (~strict).float() * 10.0
        + features[:, 14]
        + score,
        descending=True,
        stable=True,
    )
    masks.append(
        (
            f"catastrophic_hard_{min(512, count)}",
            _ranked_unique_mask(hard_order, budget=min(512, count)),
        )
    )
    unique = {}
    for label, mask in masks:
        signature = hashlib.sha256(mask.numpy().tobytes()).hexdigest()
        unique.setdefault(
            signature,
            {
                "name": label,
                "indices": torch.where(mask)[0].to(torch.int32),
                "count": int(mask.sum()),
            },
        )
    return list(unique.values())


def _solve_task(task: tuple[int, int, int]) -> tuple[int, int, int, dict]:
    query_index, subset_index, seed = task
    query = _WORKER_QUERIES[query_index]
    subset = query["subsets"][subset_index]
    indices = torch.as_tensor(subset["indices"]).long()
    keypoints = query["keypoints"][indices].numpy()
    xyz = query["xyz"][indices].numpy()
    start = time.perf_counter()
    pose, inliers, diagnostics = solve_pose(
        keypoints,
        xyz,
        query["K"].numpy(),
        solver="poselib",
        reprojection_error=12.0,
        confidence=0.99999,
        max_iterations=100000,
        min_iterations=1000,
        scores=query["top1_scores"][indices].numpy(),
        ransac_seed=int(seed),
        return_diagnostics=True,
    )
    runtime_ms = 1000.0 * (time.perf_counter() - start)
    re_deg, te_cm = cal_pose_error(pose, query["pose_w2c"].numpy())
    inliers = torch.as_tensor(inliers).long().reshape(-1)
    gt_errors = query["gt_errors"][indices]
    inlier_precision = (
        float((gt_errors[inliers] <= 2.0).float().mean())
        if len(inliers)
        else 0.0
    )
    hypotheses = diagnostics.get("ransac_actual_hypotheses")
    outcome = {
        "te_cm": float(te_cm),
        "re_deg": float(re_deg),
        "r5_success": bool(te_cm <= 5.0 and re_deg <= 5.0),
        "catastrophic": bool(te_cm > 100.0 or re_deg > 10.0),
        "inlier_count": int(len(inliers)),
        "inlier_ratio": float(len(inliers) / max(len(indices), 1)),
        "inlier_gt_precision_2px": inlier_precision,
        "hypotheses": int(hypotheses) if hypotheses is not None else None,
        "runtime_ms": runtime_ms,
        "seed": int(seed),
    }
    return query_index, subset_index, int(seed), outcome


def _annotate_relative_targets(query: dict) -> None:
    by_seed: dict[int, dict[str, dict]] = defaultdict(dict)
    for subset in query["subsets"]:
        for outcome in subset["outcomes"]:
            by_seed[int(outcome["seed"])][subset["name"]] = outcome
    for seed, outcomes in by_seed.items():
        baseline = outcomes.get("all")
        if baseline is None:
            raise ValueError(f"query {query['query_name']} misses all-set outcome")
        for outcome in outcomes.values():
            te_limit = max(
                float(baseline["te_cm"]) * 1.15,
                float(baseline["te_cm"]) + 0.5,
            )
            re_limit = max(
                float(baseline["re_deg"]) * 1.15,
                float(baseline["re_deg"]) + 0.25,
            )
            safe = (
                not outcome["catastrophic"]
                and float(outcome["te_cm"]) <= te_limit
                and float(outcome["re_deg"]) <= re_limit
                and (
                    not baseline["r5_success"]
                    or outcome["r5_success"]
                )
            )
            hypotheses = max(int(outcome.get("hypotheses") or 100000), 1)
            set_count = next(
                subset["count"]
                for subset in query["subsets"]
                if outcome in subset["outcomes"]
            )
            pose_cost = (
                4.0 * float(outcome["catastrophic"])
                + 1.5 * float(not outcome["r5_success"])
                + np.log1p(float(outcome["te_cm"]) / 5.0)
                + 0.25 * np.log1p(float(outcome["re_deg"]) / 5.0)
                + 0.08 * np.log1p(hypotheses / 1000.0)
                + 0.03 * set_count / 512.0
            )
            outcome["safe_relative_all"] = bool(safe)
            outcome["target_utility"] = -float(pose_cost)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--topk-outcomes", required=True)
    parser.add_argument("--dynamic-outcomes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budgets", default="256,384,512,768")
    parser.add_argument("--maximum-queries", type=int, default=384)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--secondary-seed", type=int, default=2027)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    partial_output = output.with_suffix(output.suffix + ".partial.pt")

    state = torch.load(args.map, map_location="cpu", weights_only=False)
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    topk = torch.load(
        args.topk_outcomes, map_location="cpu", weights_only=False
    )
    dynamic = torch.load(
        args.dynamic_outcomes, map_location="cpu", weights_only=False
    )
    if topk.get("schema") != "lafgs_exact_topk_outcomes":
        raise ValueError("unsupported top-K outcomes")
    if dynamic.get("schema") != "lafgs_dynamic_self_localization_outcomes":
        raise ValueError("unsupported dynamic outcomes")
    if topk["provenance"].get("family_prototype_state_sha256") is not None:
        raise ValueError("SLPS requires a single-descriptor candidate graph")
    if list(topk["query_names"]) != list(dynamic["query_names"]):
        raise ValueError("SLPS top-K and dynamic query ordering differ")
    anchor_ids = torch.as_tensor(state["anchor_ids"]).long()
    if (
        int(topk["anchor_count"]) != len(anchor_ids)
        or topk["anchor_ids_sha256"] != _sha256_tensor(anchor_ids)
    ):
        raise ValueError("SLPS top-K does not align with the map")

    names = list(topk["query_names"])
    topk_by_name = _records_by_name(topk)
    dynamic_by_name = _records_by_name(dynamic)
    selected_names = _stratified_query_names(
        names,
        dynamic_by_name,
        maximum_queries=int(args.maximum_queries),
    )
    fold_names, fold_lookup, fold_contract = _cross_validation_folds(
        names, single_trajectory_fold_count=5
    )
    fold_indices = torch.as_tensor(
        [fold_lookup[name] for name in names]
    ).long()
    source = torch.as_tensor(state["source_primitive_ids"]).long()
    dependency = torch.as_tensor(
        state.get("coarse_dependency_group_ids", state["dependency_group_ids"])
    ).long()
    track = torch.as_tensor(
        state.get("track_cluster_ids", state["dependency_group_ids"])
    ).long()
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    anchor_type = torch.as_tensor(
        state.get("anchor_type", torch.zeros(len(anchor_ids)))
    ).long()
    tie_reassigned_rows = 0
    tie_reassigned_queries = 0
    for name in names:
        topk_record = topk_by_name[name]
        original = dynamic_by_name[name]
        current_top1 = torch.as_tensor(
            topk_record["topk_anchor_indices"]
        ).long()[:, 0]
        original_top1 = torch.as_tensor(
            original["top1_anchor_indices"]
        ).long()
        mismatch = current_top1 != original_top1
        if not bool(mismatch.any()):
            continue
        scores = torch.as_tensor(topk_record["topk_scores"]).float()
        if bool(((scores[mismatch, 0] - scores[mismatch, 1]).abs() > 1e-6).any()):
            raise ValueError(
                f"non-tied single-descriptor top-1 differs from A1 for {name}"
            )
        cached = cache[name]
        rows = torch.as_tensor(topk_record["query_rows"]).long()
        observed = (
            torch.as_tensor(cached["native_keypoints"]).float()[rows]
            + float(cached.get("pixel_center_offset", 0.5))
        )
        gt_errors = torch.as_tensor(
            original["gt_reprojection_errors_px"]
        ).float().clone()
        gt_errors[mismatch] = _project_errors(
            xyz[current_top1[mismatch]],
            observed[mismatch],
            torch.as_tensor(cached["native_K"]).float(),
            torch.as_tensor(cached["pose_w2c"]).float(),
        )
        inlier = torch.as_tensor(
            original["ransac_inlier_mask"]
        ).bool().clone()
        # The old inlier bit refers to the other tied anchor identity.
        inlier[mismatch] = False
        corrected = dict(original)
        corrected.update(
            {
                "top1_anchor_indices": current_top1,
                "top1_scores": scores[:, 0],
                "top1_margins": scores[:, 0] - scores[:, 1],
                "gt_reprojection_errors_px": gt_errors,
                "ransac_inlier_mask": inlier,
                "clean_inlier_mask": inlier & (gt_errors <= 4.0),
                "harmful_inlier_mask": inlier & (gt_errors > 4.0),
            }
        )
        dynamic_by_name[name] = corrected
        tie_reassigned_rows += int(mismatch.sum())
        tie_reassigned_queries += 1
    dynamic_records = [dynamic_by_name[name] for name in names]
    statistics = _anchor_statistics(
        dynamic_records,
        fold_indices,
        trajectory_count=len(fold_names),
        anchor_count=len(anchor_ids),
    )
    full_statistics = {
        name: value.sum(dim=0) for name, value in statistics.items()
    }
    full_stability = beta_track_stability(
        attempts_by_fold=statistics["attempts"],
        clean_inlier_by_fold=statistics["clean_inlier"],
        track_groups=track,
    )
    budgets = tuple(
        sorted(
            {
                int(value)
                for value in str(args.budgets).split(",")
                if value.strip()
            }
        )
    )

    queries = []
    for name in selected_names:
        topk_record = topk_by_name[name]
        dynamic_record = dynamic_by_name[name]
        rows = torch.as_tensor(topk_record["query_rows"]).long()
        if not torch.equal(
            rows, torch.as_tensor(dynamic_record["query_rows"]).long()
        ):
            raise ValueError(f"SLPS row contract differs for {name}")
        topk_indices = torch.as_tensor(
            topk_record["topk_anchor_indices"]
        ).long()
        topk_scores = torch.as_tensor(topk_record["topk_scores"]).float()
        dynamic_top1 = torch.as_tensor(
            dynamic_record["top1_anchor_indices"]
        ).long()
        if not torch.equal(topk_indices[:, 0], dynamic_top1):
            raise AssertionError("tie-corrected A1 top-1 still differs")
        contribution = _query_contribution(
            dynamic_record, len(anchor_ids)
        )
        folded = {
            key: (full_statistics[key] - contribution[key]).clamp_min(0.0)
            for key in full_statistics
        }
        cached = cache[name]
        selector_keypoints = torch.as_tensor(
            cached["native_keypoints"]
        ).float()[rows]
        keypoints = selector_keypoints + float(
            cached.get("pixel_center_offset", 0.5)
        )
        top1 = topk_indices[:, 0]
        base = build_pose_sufficient_features(
            topk_scores,
            topk_indices,
            keypoints=selector_keypoints,
            keypoint_scores=torch.as_tensor(
                dynamic_record["keypoint_scores"]
            ).float(),
            image_hw=cached["native_input_hw"],
            source_groups=source,
            dependency_groups=dependency,
            anchor_statistics=folded,
        )
        query_features = build_slps_features(
            base,
            xyz=xyz[top1],
            anchor_type=anchor_type[top1],
            track_groups=track[top1],
            track_stability=full_stability[top1],
            anchor_map_support=folded["attempts"][top1],
        )
        relation_groups = build_relation_groups(
            keypoints=selector_keypoints,
            image_hw=cached["native_input_hw"],
            xyz=xyz[top1],
            dependency_groups=dependency[top1],
            source_groups=source[top1],
            track_groups=track[top1],
        )
        query = {
            "query_name": name,
            "fold_index": int(fold_lookup[name]),
            "query_rows": rows.to(torch.int32),
            "features": query_features.to(torch.float16),
            "relation_groups": relation_groups.to(torch.int64),
            "strict_clean": (
                torch.as_tensor(
                    dynamic_record["gt_reprojection_errors_px"]
                ).float()
                <= 2.0
            ),
            "solver_clean": torch.as_tensor(
                dynamic_record["clean_inlier_mask"]
            ).bool(),
            "harmful": torch.as_tensor(
                dynamic_record["harmful_inlier_mask"]
            ).bool(),
            "keypoints": keypoints,
            "xyz": xyz[top1],
            "K": torch.as_tensor(cached["native_K"]).float(),
            "pose_w2c": torch.as_tensor(cached["pose_w2c"]).float(),
            "top1_scores": topk_scores[:, 0],
            "gt_errors": torch.as_tensor(
                dynamic_record["gt_reprojection_errors_px"]
            ).float(),
        }
        query["subsets"] = _make_subsets(
            query, budgets=budgets, seed=int(args.seed)
        )
        for subset in query["subsets"]:
            subset["outcomes"] = []
        queries.append(query)

    global _WORKER_QUERIES
    _WORKER_QUERIES = queries
    secondary_profiles = {
        "all",
        "score_512",
        "quality_512",
        "balanced_512",
        "dependency_unique_512",
        "strict_oracle_512",
        "catastrophic_hard_512",
    }
    tasks = []
    for query_index, query in enumerate(queries):
        for subset_index, subset in enumerate(query["subsets"]):
            tasks.append((query_index, subset_index, int(args.seed)))
            if subset["name"] in secondary_profiles:
                tasks.append(
                    (
                        query_index,
                        subset_index,
                        int(args.secondary_seed),
                    )
                )
    partial_identity = {
        "map_sha256": _sha256_file(args.map),
        "topk_outcomes_sha256": _sha256_file(args.topk_outcomes),
        "dynamic_outcomes_sha256": _sha256_file(args.dynamic_outcomes),
        "query_names": selected_names,
        "budgets": list(budgets),
        "seed": int(args.seed),
        "secondary_seed": int(args.secondary_seed),
    }
    completed_results = []
    completed_keys = set()
    if partial_output.is_file():
        partial = torch.load(
            partial_output, map_location="cpu", weights_only=False
        )
        if partial.get("identity") != partial_identity:
            raise ValueError("SLPS corpus partial identity differs")
        completed_results = list(partial.get("results", ()))
        for query_index, subset_index, solve_seed, outcome in completed_results:
            queries[query_index]["subsets"][subset_index][
                "outcomes"
            ].append(outcome)
            completed_keys.add(
                (int(query_index), int(subset_index), int(solve_seed))
            )
        tasks = [
            task for task in tasks if tuple(map(int, task)) not in completed_keys
        ]
    print(
        json.dumps(
            {
                "query_count": len(queries),
                "set_solve_count": len(tasks) + len(completed_results),
                "set_solve_remaining": len(tasks),
                "set_solve_resumed": len(completed_results),
                "workers": int(args.workers),
            }
        ),
        flush=True,
    )
    context = mp.get_context("fork")
    completed = len(completed_results)
    total_solve_count = len(tasks) + completed
    with context.Pool(processes=max(int(args.workers), 1)) as pool:
        for query_index, subset_index, solve_seed, outcome in pool.imap_unordered(
            _solve_task, tasks, chunksize=1
        ):
            queries[query_index]["subsets"][subset_index][
                "outcomes"
            ].append(outcome)
            completed_results.append(
                (query_index, subset_index, solve_seed, outcome)
            )
            completed += 1
            if completed % 250 == 0:
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "total": total_solve_count,
                            "percent": (
                                100.0 * completed / total_solve_count
                            ),
                        }
                    ),
                    flush=True,
                )
            if completed % 500 == 0:
                _atomic_torch(
                    partial_output,
                    {
                        "schema": "lafgs_slps_set_outcomes_partial",
                        "identity": partial_identity,
                        "results": completed_results,
                    },
                )
    for query in queries:
        _annotate_relative_targets(query)
        for key in (
            "keypoints",
            "xyz",
            "K",
            "pose_w2c",
            "top1_scores",
            "gt_errors",
        ):
            query.pop(key)

    feature_rows = torch.cat(
        [query["features"].float() for query in queries], dim=0
    )
    feature_mean = feature_rows.mean(dim=0)
    feature_scale = feature_rows.std(dim=0, unbiased=False).clamp_min(1e-4)
    summary = {
        "query_count": len(queries),
        "set_solve_count": total_solve_count,
        "set_count": int(sum(len(query["subsets"]) for query in queries)),
        "row_count": int(sum(len(query["features"]) for query in queries)),
        "safe_rate": float(
            np.mean(
                [
                    outcome["safe_relative_all"]
                    for query in queries
                    for subset in query["subsets"]
                    for outcome in subset["outcomes"]
                ]
            )
        ),
        "catastrophic_rate": float(
            np.mean(
                [
                    outcome["catastrophic"]
                    for query in queries
                    for subset in query["subsets"]
                    for outcome in subset["outcomes"]
                ]
            )
        ),
        "tie_reassigned_rows": tie_reassigned_rows,
        "tie_reassigned_queries": tie_reassigned_queries,
    }
    _atomic_torch(
        output,
        {
            "schema": "lafgs_slps_set_outcomes",
            "version": 1,
            "query_names": selected_names,
            "queries": queries,
            "feature_names": list(SLPS_FEATURE_NAMES),
            "feature_mean": feature_mean,
            "feature_scale": feature_scale,
            "fold_names": fold_names,
            "fold_contract": fold_contract,
            "anchor_count": len(anchor_ids),
            "anchor_ids_sha256": _sha256_tensor(anchor_ids),
            "anchor_statistics": full_statistics,
            "anchor_track_stability": full_stability,
            "candidate_graph_contract": dict(topk["provenance"]),
            "retrieval_topk": int(topk["topk"]),
            "source": {
                "map": str(Path(args.map).resolve()),
                "map_sha256": _sha256_file(args.map),
                "query_cache": str(Path(args.query_cache).resolve()),
                "query_cache_sha256": _sha256_file(args.query_cache),
                "topk_outcomes": str(Path(args.topk_outcomes).resolve()),
                "topk_outcomes_sha256": _sha256_file(args.topk_outcomes),
                "dynamic_outcomes": str(
                    Path(args.dynamic_outcomes).resolve()
                ),
                "dynamic_outcomes_sha256": _sha256_file(
                    args.dynamic_outcomes
                ),
            },
            "config": vars(args),
            "summary": summary,
        },
    )
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    partial_output.unlink(missing_ok=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
