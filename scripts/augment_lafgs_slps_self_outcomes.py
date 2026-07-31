#!/usr/bin/env python3
"""Augment SLPS supervision with exact outcomes of its own learned sets."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
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
)
from localization_training.slps_selector import (
    SLPS_BIAS_AWARE_FEATURE_NAMES,
    build_relation_groups,
    build_slps_features,
    slps_from_state,
)
from scripts.build_lafgs_slps_set_outcomes import (
    _annotate_relative_targets,
    _project_errors,
    _records_by_name,
    _sha256_file,
    _sha256_tensor,
    _stable_seed,
)
from utils.pose_utils import cal_pose_error, solve_pose


_WORKER_QUERIES: list[dict] = []


def _atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _signature(indices: torch.Tensor, count: int) -> str:
    mask = torch.zeros(int(count), dtype=torch.bool)
    mask[torch.as_tensor(indices).long()] = True
    return hashlib.sha256(mask.numpy().tobytes()).hexdigest()


def _indices(mask: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.as_tensor(mask).bool())[0].to(torch.int32)


def _replace(
    base: torch.Tensor,
    remove: torch.Tensor,
    add: torch.Tensor,
) -> torch.Tensor:
    output = torch.as_tensor(base).bool().clone()
    output[torch.as_tensor(remove).long()] = False
    output[torch.as_tensor(add).long()] = True
    return output


def _dependency_repair(
    base: torch.Tensor,
    ordering: torch.Tensor,
    dependency: torch.Tensor,
    *,
    replacement_count: int,
) -> torch.Tensor:
    selected = torch.where(base)[0]
    by_group: dict[int, list[int]] = defaultdict(list)
    for index in selected.tolist():
        by_group[int(dependency[index])].append(index)
    redundant = [
        index
        for values in by_group.values()
        for index in values[2:]
    ]
    if not redundant:
        return base.clone()
    selected_groups = {
        int(dependency[index]) for index in selected.tolist()
    }
    candidates = [
        index
        for index in ordering.tolist()
        if not bool(base[index])
        and int(dependency[index]) not in selected_groups
    ]
    count = min(
        int(replacement_count), len(redundant), len(candidates)
    )
    return _replace(
        base,
        torch.as_tensor(redundant[:count]),
        torch.as_tensor(candidates[:count]),
    )


def _learned_subsets(
    query: dict,
    model,
    *,
    budgets: tuple[int, ...],
    replacement_count: int,
    seed: int,
) -> list[dict]:
    device = model.feature_mean.device
    features = query["features"].to(device=device, dtype=torch.float32)
    groups = query["relation_groups"].to(device=device)
    with torch.inference_mode():
        encoded = model.encode(features, groups)
        maximum = min(
            len(features),
            max(budgets) + max(int(replacement_count), 1),
        )
        ordering = model.greedy_order(
            encoded, groups, maximum_count=maximum
        ).cpu()
        harmful_probability = encoded["harmful_probability"].cpu()
    strict = torch.as_tensor(query["strict_clean"]).bool()
    dependency = torch.as_tensor(query["relation_groups"])[:, 2].long()
    count = len(features)
    subsets = []
    generator = torch.Generator().manual_seed(
        _stable_seed(query["query_name"], seed + 7103)
    )
    for budget in budgets:
        target = min(max(int(budget), 4), count)
        base = torch.zeros(count, dtype=torch.bool)
        base[ordering[:target]] = True
        variants: list[tuple[str, torch.Tensor]] = [
            (f"learned_nested_{target}", base)
        ]
        swap_count = min(
            max(int(replacement_count), 1),
            max(target // 4, 1),
            count - target,
        )
        selected = torch.where(base)[0]
        outside = ordering[target:]
        harmful_order = selected[
            torch.argsort(
                harmful_probability[selected],
                descending=True,
                stable=True,
            )
        ]
        if swap_count > 0 and len(outside) >= swap_count:
            variants.append(
                (
                    f"learned_risk_swap_{target}",
                    _replace(
                        base,
                        harmful_order[:swap_count],
                        outside[:swap_count],
                    ),
                )
            )
            drop = base.clone()
            drop[harmful_order[:swap_count]] = False
            variants.append((f"learned_drop_{target}", drop))
            add = base.clone()
            add[outside[:swap_count]] = True
            variants.append((f"learned_add_{target}", add))

            oracle_remove = selected[
                torch.argsort(
                    strict[selected].float(),
                    descending=False,
                    stable=True,
                )
            ]
            oracle_add = outside[strict[outside]]
            oracle_count = min(
                swap_count, len(oracle_remove), len(oracle_add)
            )
            if oracle_count > 0:
                variants.append(
                    (
                        f"learned_oracle_swap_{target}",
                        _replace(
                            base,
                            oracle_remove[:oracle_count],
                            oracle_add[:oracle_count],
                        ),
                    )
                )
            random_remove = selected[
                torch.randperm(len(selected), generator=generator)
            ][:swap_count]
            random_add = outside[
                torch.randperm(len(outside), generator=generator)
            ][:swap_count]
            variants.append(
                (
                    f"learned_random_swap_{target}",
                    _replace(base, random_remove, random_add),
                )
            )
        variants.append(
            (
                f"learned_dependency_swap_{target}",
                _dependency_repair(
                    base,
                    ordering,
                    dependency,
                    replacement_count=max(swap_count, 1),
                ),
            )
        )
        for name, mask in variants:
            subsets.append(
                {
                    "name": name,
                    "indices": _indices(mask),
                    "count": int(mask.sum()),
                    "outcomes": [],
                }
            )
    unique = {}
    for subset in subsets:
        unique.setdefault(
            _signature(subset["indices"], count), subset
        )
    return list(unique.values())


def _solve_task(task: tuple[int, int, int]) -> tuple[int, int, int, dict]:
    query_index, subset_index, seed = task
    query = _WORKER_QUERIES[query_index]
    subset = query["subsets"][subset_index]
    indices = torch.as_tensor(subset["indices"]).long()
    start = time.perf_counter()
    pose, inliers, diagnostics = solve_pose(
        query["keypoints"][indices].numpy(),
        query["xyz"][indices].numpy(),
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
    errors = query["gt_errors"][indices]
    precision = (
        float((errors[inliers] <= 2.0).float().mean())
        if len(inliers)
        else 0.0
    )
    hypotheses = diagnostics.get("ransac_actual_hypotheses")
    return query_index, subset_index, int(seed), {
        "te_cm": float(te_cm),
        "re_deg": float(re_deg),
        "r5_success": bool(te_cm <= 5.0 and re_deg <= 5.0),
        "catastrophic": bool(te_cm > 100.0 or re_deg > 10.0),
        "inlier_count": int(len(inliers)),
        "inlier_ratio": float(len(inliers) / max(len(indices), 1)),
        "inlier_gt_precision_2px": precision,
        "hypotheses": int(hypotheses) if hypotheses is not None else None,
        "runtime_ms": runtime_ms,
        "seed": int(seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--topk-outcomes", required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--budgets", default="256,384,512,768")
    parser.add_argument("--replacement-count", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--secondary-seed", type=int, default=2027)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    partial_output = output.with_suffix(output.suffix + ".partial.pt")
    corpus = torch.load(
        args.corpus, map_location="cpu", weights_only=False
    )
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    topk = torch.load(
        args.topk_outcomes, map_location="cpu", weights_only=False
    )
    selector_state = torch.load(
        args.selector, map_location="cpu", weights_only=False
    )
    selector_sha256 = _sha256_file(args.selector)
    if (
        corpus.get("schema") != "lafgs_slps_set_outcomes"
        or selector_state.get("schema") != "lafgs_slps_selector"
        or topk.get("schema") != "lafgs_exact_topk_outcomes"
    ):
        raise ValueError("unsupported SLPS self-mining input")
    if (
        dict(corpus["candidate_graph_contract"])
        != dict(topk["provenance"])
        or dict(selector_state["candidate_graph_contract"])
        != dict(topk["provenance"])
    ):
        raise ValueError("SLPS self-mining candidate graph differs")
    anchor_ids = torch.as_tensor(state["anchor_ids"]).long()
    if (
        corpus["anchor_ids_sha256"] != _sha256_tensor(anchor_ids)
        or selector_state["anchor_ids_sha256"]
        != _sha256_tensor(anchor_ids)
    ):
        raise ValueError("SLPS self-mining map identity differs")

    model = slps_from_state(selector_state, device=args.device)
    budgets = tuple(
        sorted(
            {
                int(value)
                for value in str(args.budgets).split(",")
                if value.strip()
            }
        )
    )
    topk_by_name = _records_by_name(topk)
    xyz_map = torch.as_tensor(state["anchor_xyz"]).float()
    source = torch.as_tensor(state["source_primitive_ids"]).long()
    dependency = torch.as_tensor(
        state.get(
            "coarse_dependency_group_ids",
            state["dependency_group_ids"],
        )
    ).long()
    track = torch.as_tensor(
        state.get("track_cluster_ids", state["dependency_group_ids"])
    ).long()
    anchor_type = torch.as_tensor(
        state.get("anchor_type", torch.zeros(len(anchor_ids)))
    ).long()
    statistics = {
        name: torch.as_tensor(value).float()
        for name, value in selector_state["anchor_statistics"].items()
    }
    stability = torch.as_tensor(
        selector_state["anchor_track_stability"]
    ).float()
    bias_aware_features = list(corpus.get("feature_names", ())) == list(
        SLPS_BIAS_AWARE_FEATURE_NAMES
    )
    if bias_aware_features and selector_state.get(
        "residual_signature_state"
    ) is None:
        raise ValueError("bias-aware self-mining misses residual signatures")
    queries = copy.deepcopy(corpus["queries"])
    for query in queries:
        for subset in query["subsets"]:
            subset["deployment_calibration"] = False
    solve_queries = []
    for query in queries:
        name = query["query_name"]
        record = topk_by_name[name]
        rows = torch.as_tensor(record["query_rows"]).long()
        topk_indices = torch.as_tensor(
            record["topk_anchor_indices"]
        ).long()
        top1 = topk_indices[:, 0]
        cached = cache[name]
        keypoints = (
            torch.as_tensor(cached["native_keypoints"]).float()[rows]
            + float(cached.get("pixel_center_offset", 0.5))
        )
        selector_keypoints = torch.as_tensor(
            cached["native_keypoints"]
        ).float()[rows]
        gt_errors = _project_errors(
            xyz_map[top1],
            keypoints,
            torch.as_tensor(cached["native_K"]).float(),
            torch.as_tensor(cached["pose_w2c"]).float(),
        )
        topk_scores = torch.as_tensor(record["topk_scores"]).float()
        if bias_aware_features:
            # The augmented corpus already contains leave-query-out residual
            # evidence. Rebuilding from the deployment state would expose the
            # query's own signed GT residual to its learned-set proposal.
            query["features"] = query["features"].float()
        else:
            base_features = build_pose_sufficient_features(
                topk_scores,
                topk_indices,
                keypoints=selector_keypoints,
                keypoint_scores=torch.as_tensor(
                    cached["native_scores"]
                ).float()[rows],
                image_hw=cached["native_input_hw"],
                source_groups=source,
                dependency_groups=dependency,
                anchor_statistics=statistics,
                entropy_temperature=float(
                    selector_state.get("entropy_temperature", 0.05)
                ),
                prior_strength=float(
                    selector_state.get("prior_strength", 12.0)
                ),
            )
            query["features"] = build_slps_features(
                base_features,
                xyz=xyz_map[top1],
                anchor_type=anchor_type[top1],
                track_groups=track[top1],
                track_stability=stability[top1],
                anchor_map_support=statistics["attempts"][top1],
            ).to(torch.float32)
        query["relation_groups"] = build_relation_groups(
            keypoints=selector_keypoints,
            image_hw=cached["native_input_hw"],
            xyz=xyz_map[top1],
            dependency_groups=dependency[top1],
            source_groups=source[top1],
            track_groups=track[top1],
        )
        query["strict_clean"] = gt_errors <= 2.0
        new_subsets = _learned_subsets(
            query,
            model,
            budgets=budgets,
            replacement_count=int(args.replacement_count),
            seed=int(args.seed),
        )
        for subset in new_subsets:
            subset["self_mining_selector_sha256"] = selector_sha256
            subset["deployment_calibration"] = str(
                subset["name"]
            ).startswith("learned_nested_")
        existing = {
            _signature(subset["indices"], len(query["features"])): subset
            for subset in query["subsets"]
        }
        first_new_index = len(query["subsets"])
        for subset in new_subsets:
            signature = _signature(
                subset["indices"], len(query["features"])
            )
            if signature in existing:
                if bool(subset["deployment_calibration"]):
                    existing[signature]["deployment_calibration"] = True
                    existing[signature][
                        "self_mining_selector_sha256"
                    ] = selector_sha256
                continue
            query["subsets"].append(subset)
            existing[signature] = subset
        solve_queries.append(
            {
                "subsets": query["subsets"],
                "first_new_index": first_new_index,
                "keypoints": keypoints,
                "xyz": xyz_map[top1],
                "K": torch.as_tensor(cached["native_K"]).float(),
                "pose_w2c": torch.as_tensor(cached["pose_w2c"]).float(),
                "top1_scores": torch.as_tensor(
                    record["topk_scores"]
                ).float()[:, 0],
                "gt_errors": gt_errors,
            }
        )

    global _WORKER_QUERIES
    _WORKER_QUERIES = solve_queries
    tasks = []
    for query_index, query in enumerate(solve_queries):
        for subset_index in range(
            query["first_new_index"], len(query["subsets"])
        ):
            tasks.append((query_index, subset_index, int(args.seed)))
            name = query["subsets"][subset_index]["name"]
            if name.startswith("learned_nested_"):
                tasks.append(
                    (
                        query_index,
                        subset_index,
                        int(args.secondary_seed),
                    )
                )
    identity = {
        "corpus_sha256": _sha256_file(args.corpus),
        "map_sha256": _sha256_file(args.map),
        "topk_sha256": _sha256_file(args.topk_outcomes),
        "selector_sha256": _sha256_file(args.selector),
        "budgets": list(budgets),
        "replacement_count": int(args.replacement_count),
        "seed": int(args.seed),
        "secondary_seed": int(args.secondary_seed),
    }
    completed_results = []
    completed_keys = set()
    if partial_output.is_file():
        partial = torch.load(
            partial_output, map_location="cpu", weights_only=False
        )
        if partial.get("identity") != identity:
            raise ValueError("SLPS self-mining partial identity differs")
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
    total = len(tasks) + len(completed_results)
    print(
        json.dumps(
            {
                "query_count": len(queries),
                "solve_count": total,
                "remaining": len(tasks),
            }
        ),
        flush=True,
    )
    completed = len(completed_results)
    # The selector has already initialized CUDA and PyTorch's OpenMP pool.
    # Forking PoseLib workers with that large inherited pool can deadlock all
    # children on futexes. Pose solving itself is process-parallel here.
    torch.set_num_threads(1)
    context = mp.get_context("fork")
    with context.Pool(processes=max(int(args.workers), 1)) as pool:
        for result in pool.imap_unordered(_solve_task, tasks, chunksize=1):
            query_index, subset_index, solve_seed, outcome = result
            queries[query_index]["subsets"][subset_index][
                "outcomes"
            ].append(outcome)
            completed_results.append(result)
            completed += 1
            if completed % 250 == 0:
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "total": total,
                            "percent": 100.0 * completed / max(total, 1),
                        }
                    ),
                    flush=True,
                )
            if completed % 500 == 0:
                _atomic_torch(
                    partial_output,
                    {
                        "schema": "lafgs_slps_self_mining_partial",
                        "identity": identity,
                        "results": completed_results,
                    },
                )
    for query in queries:
        _annotate_relative_targets(query)

    augmented = dict(corpus)
    augmented.update(
        {
            "version": int(corpus.get("version", 1)) + 1,
            "queries": queries,
            "self_mining": {
                "source_corpus": str(Path(args.corpus).resolve()),
                "source_corpus_sha256": _sha256_file(args.corpus),
                "selector": str(Path(args.selector).resolve()),
                "selector_sha256": selector_sha256,
                "solve_count": total,
                "budgets": list(budgets),
                "replacement_count": int(args.replacement_count),
                "feature_history_contract": (
                    "leave_query_out_residual_signatures"
                    if bias_aware_features
                    else "deployment_full_map_history"
                ),
            },
        }
    )
    all_outcomes = [
        outcome
        for query in queries
        for subset in query["subsets"]
        for outcome in subset["outcomes"]
    ]
    augmented["summary"] = {
        **dict(corpus["summary"]),
        "set_count": int(
            sum(len(query["subsets"]) for query in queries)
        ),
        "set_solve_count": int(len(all_outcomes)),
        "safe_rate": float(
            np.mean(
                [outcome["safe_relative_all"] for outcome in all_outcomes]
            )
        ),
        "catastrophic_rate": float(
            np.mean([outcome["catastrophic"] for outcome in all_outcomes])
        ),
        "self_mined_solve_count": total,
    }
    _atomic_torch(output, augmented)
    summary = {
        "output": str(output),
        **augmented["summary"],
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    partial_output.unlink(missing_ok=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
