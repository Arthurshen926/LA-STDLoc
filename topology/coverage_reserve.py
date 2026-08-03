#!/usr/bin/env python3
"""Add a compact pose-sufficient reserve to a Track-centric core map."""

from __future__ import annotations

import argparse
import heapq
import json
import os
from collections import defaultdict
from pathlib import Path

import torch

from topology.pose_information import (
    conditional_add_gain,
    fisher_contributions,
    pose_jacobian_analytic,
    task_scaled_pose_jacobian,
    translation_schur_complement,
)
from map_learning.observations import _query_index_remap
from topology.track_core import _materialize


def _expanded_positive_pairs(record: dict) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.as_tensor(record["query_rows"]).long()
    offsets = torch.as_tensor(record["positive_offsets"]).long()
    indices = torch.as_tensor(record["positive_indices"]).long()
    counts = offsets[1:] - offsets[:-1]
    nonempty = counts > 0
    return torch.repeat_interleave(rows[nonempty], counts[nonempty]), indices


def _information_contributions(
    xyz: torch.Tensor, K: torch.Tensor, pose: torch.Tensor
) -> torch.Tensor:
    jacobian = pose_jacobian_analytic(
        xyz.double(), K.double(), pose.double()
    )
    jacobian = task_scaled_pose_jacobian(
        jacobian,
        translation_scale=0.07160573943725686,
        rotation_scale=torch.deg2rad(torch.tensor(2.0)).item(),
    )
    return fisher_contributions(jacobian)


def _cell_ids(rows: torch.Tensor, keypoints: torch.Tensor, hw) -> torch.Tensor:
    height, width = (int(value) for value in hw)
    xy = keypoints[rows]
    x = (xy[:, 0] / max(width, 1) * 4).floor().long().clamp(0, 3)
    y = (xy[:, 1] / max(height, 1) * 4).floor().long().clamp(0, 3)
    return y * 4 + x


def _depth_bins(xyz: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype)
    depth = (pose.float() @ torch.cat((xyz.float(), ones), dim=1).T)[2]
    return (torch.log2(depth.clamp_min(0.25)) * 2).floor().long()


def greedy_pose_reserve(
    query_candidates: list[list[tuple[int, float]]],
    source_ids: torch.Tensor,
    voxel_ids: torch.Tensor,
    *,
    budget: int,
    minimum_queries_per_anchor_set: int = 3,
    maximum_per_source: int = 1,
    maximum_per_voxel: int = 3,
    query_weights: torch.Tensor | None = None,
    force_fill: bool = True,
) -> torch.Tensor:
    """Greedy query-balanced selection over conditional set gains."""
    if query_weights is None:
        query_weights = torch.ones(len(query_candidates))
    query_weights = torch.as_tensor(query_weights).float().reshape(-1)
    if query_weights.numel() != len(query_candidates):
        raise ValueError("query weights must align with query candidates")
    deficits = [
        int(minimum_queries_per_anchor_set) if float(weight) > 0 else 0
        for weight in query_weights
    ]
    by_anchor: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for query, candidates in enumerate(query_candidates):
        for anchor, score in candidates:
            if score > 0:
                by_anchor[int(anchor)].append((query, float(score)))

    heap = []
    for anchor, entries in by_anchor.items():
        gain = sum(
            score * float(query_weights[query])
            for query, score in entries
            if deficits[query] > 0
        )
        if gain > 0:
            heapq.heappush(heap, (-gain, -len(entries), anchor))

    selected = []
    selected_set = set()
    source_count: dict[int, int] = defaultdict(int)
    voxel_count: dict[int, int] = defaultdict(int)
    while heap and len(selected) < int(budget):
        negative_gain, _, anchor = heapq.heappop(heap)
        if anchor in selected_set:
            continue
        source = int(source_ids[anchor])
        voxel = int(voxel_ids[anchor])
        if (
            source_count[source] >= int(maximum_per_source)
            or voxel_count[voxel] >= int(maximum_per_voxel)
        ):
            continue
        gain = sum(
            score * float(query_weights[query])
            for query, score in by_anchor[anchor]
            if deficits[query] > 0
        )
        if gain <= 0:
            break
        if not torch.isclose(
            torch.tensor(gain),
            torch.tensor(-negative_gain),
            rtol=0,
            atol=1e-7,
        ):
            heapq.heappush(
                heap, (-gain, -len(by_anchor[anchor]), anchor)
            )
            continue
        selected.append(anchor)
        selected_set.add(anchor)
        source_count[source] += 1
        voxel_count[voxel] += 1
        for query, _ in by_anchor[anchor]:
            deficits[query] = max(0, deficits[query] - 1)

    if force_fill and len(selected) < int(budget):
        global_order = sorted(
            (
                (sum(score for _, score in entries), anchor)
                for anchor, entries in by_anchor.items()
            ),
            reverse=True,
        )
        for _, anchor in global_order:
            if len(selected) >= int(budget):
                break
            if anchor in selected_set:
                continue
            source = int(source_ids[anchor])
            voxel = int(voxel_ids[anchor])
            if (
                source_count[source] >= int(maximum_per_source)
                or voxel_count[voxel] >= int(maximum_per_voxel)
            ):
                continue
            selected.append(anchor)
            selected_set.add(anchor)
            source_count[source] += 1
            voxel_count[voxel] += 1
    return torch.as_tensor(selected, dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-map", required=True)
    parser.add_argument("--canonical-map", required=True)
    parser.add_argument("--function-graph", required=True)
    parser.add_argument("--complete-positive-teacher", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reserve-additions", default="1000,2000,3000")
    parser.add_argument("--top-candidates-per-query", type=int, default=64)
    parser.add_argument("--maximum-harmful-rate", type=float, default=0.10)
    parser.add_argument("--dependency-voxel-size", type=float, default=0.5)
    parser.add_argument("--query-outcomes", default="")
    parser.add_argument(
        "--pose-scoring-cache",
        default="",
        help="Reuse a completed, identity-checked scoring cache from another reserve objective.",
    )
    parser.add_argument(
        "--reserve-mode",
        choices=("all", "precision", "robustness"),
        default="all",
    )
    parser.add_argument(
        "--force-fill",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    core_path = Path(args.core_map).resolve()
    canonical_path = Path(args.canonical_map).resolve()
    graph_path = Path(args.function_graph).resolve()
    teacher_path = Path(args.complete_positive_teacher).resolve()
    payload_path = Path(args.track_payload).resolve()
    query_path = Path(args.query_cache).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    core = torch.load(core_path, map_location="cpu", weights_only=False)
    canonical = torch.load(
        canonical_path, map_location="cpu", weights_only=False
    )
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    teacher = torch.load(
        teacher_path, map_location="cpu", weights_only=False
    )
    payload = torch.load(
        payload_path, map_location="cpu", weights_only=False
    )
    query_payload = torch.load(
        query_path, map_location="cpu", weights_only=False
    )
    query_cache = query_payload.get("queries", query_payload)
    outcome_by_name = {}
    if args.query_outcomes:
        outcomes = json.loads(Path(args.query_outcomes).read_text())
        outcome_by_name = {
            record["query"]: float(record["te_cm"])
            for record in outcomes["results"]
        }
        if set(outcome_by_name) != set(teacher["query_names"]):
            raise ValueError("query outcomes do not align with teacher queries")

    reconstruction = core["track_centric_reconstruction"]
    track_indices = torch.as_tensor(
        reconstruction["track_indices"]
    ).long()
    base_rows = torch.as_tensor(
        reconstruction["base_canonical_rows"]
    ).long()
    track_count = int(track_indices.numel())
    track_features = torch.as_tensor(core["anchor_features"])[
        :track_count
    ].float()
    canonical_xyz = torch.as_tensor(canonical["anchor_xyz"]).float()
    source_ids = torch.as_tensor(
        canonical["source_primitive_ids"]
    ).long()
    base_count = int(canonical_xyz.shape[0])
    selected_base = torch.zeros(base_count, dtype=torch.bool)
    selected_base[base_rows] = True

    opportunity = torch.as_tensor(
        graph["provenance_opportunity_count"]
    ).float()
    harmful = torch.as_tensor(
        graph["provenance_harmful_solver_inlier_count"]
    ).float()
    harmful_rate = harmful / opportunity.clamp_min(1)
    eligible = (
        torch.as_tensor(graph["provenance_legal_hit_2px_count"]) > 0
    ) & (harmful_rate <= float(args.maximum_harmful_rate))
    eligible &= ~selected_base

    voxel_coordinates = torch.floor(
        canonical_xyz / float(args.dependency_voxel_size)
    ).long()
    voxel_ids = torch.unique(
        torch.cat((source_ids[:, None], voxel_coordinates), dim=1),
        dim=0,
        return_inverse=True,
    )[1]

    payload_to_teacher = _query_index_remap(
        payload["query_names"], teacher["query_names"]
    )
    track_observations: dict[int, list[tuple[int, int]]] = defaultdict(list)
    tracks = payload["tracks"]
    selected_track_mask = torch.zeros(
        int(torch.as_tensor(payload["track_geometry"]["triangulated"]).numel()),
        dtype=torch.bool,
    )
    selected_track_mask[track_indices] = True
    observation_track = torch.as_tensor(tracks["track_index"]).long()
    observation_query = torch.as_tensor(tracks["query_index"]).long()
    observation_row = torch.as_tensor(tracks["keypoint_index"]).long()
    keep_observation = selected_track_mask[observation_track]
    for query, row, track in zip(
        observation_query[keep_observation].tolist(),
        observation_row[keep_observation].tolist(),
        observation_track[keep_observation].tolist(),
    ):
        track_observations[int(payload_to_teacher[query])].append((row, track))
    track_xyz = torch.as_tensor(
        payload["track_geometry"]["triangulated_xyz"]
    ).float()

    query_candidates: list[list[tuple[int, float]]] = []
    query_diagnostics = []
    scoring_identity = {
        "core_map": str(core_path),
        "canonical_map": str(canonical_path),
        "function_graph": str(graph_path),
        "positive_teacher": str(teacher_path),
        "query_cache": str(query_path),
        "dependency_voxel_size": float(args.dependency_voxel_size),
        "maximum_harmful_rate": float(args.maximum_harmful_rate),
    }
    scoring_partial = output_dir / "pose_reserve_scoring.partial.pt"
    local_scoring_cache = output_dir / "pose_reserve_scoring.pt"
    scoring_cache = (
        Path(args.pose_scoring_cache).resolve()
        if args.pose_scoring_cache
        else local_scoring_cache
    )
    if args.pose_scoring_cache and not scoring_cache.is_file():
        raise FileNotFoundError(scoring_cache)
    if scoring_cache.is_file():
        saved_scoring = torch.load(
            scoring_cache, map_location="cpu", weights_only=False
        )
        if saved_scoring["identity"] != scoring_identity:
            raise ValueError("pose reserve scoring cache identity mismatch")
        query_candidates = saved_scoring["query_candidates"]
        query_diagnostics = saved_scoring["query_diagnostics"]
    elif scoring_partial.is_file():
        saved_scoring = torch.load(
            scoring_partial, map_location="cpu", weights_only=False
        )
        if saved_scoring["identity"] != scoring_identity:
            raise ValueError("pose reserve partial identity mismatch")
        query_candidates = saved_scoring["query_candidates"]
        query_diagnostics = saved_scoring["query_diagnostics"]
    eye = torch.eye(6, dtype=torch.float64) * 1e-6
    for completed, record in enumerate(teacher["records"], start=1):
        if completed <= len(query_candidates):
            continue
        query_index = int(record["query_index"])
        name = teacher["query_names"][query_index]
        cached = query_cache[name]
        K = torch.as_tensor(cached["native_K"]).float()
        pose = torch.as_tensor(cached["pose_w2c"]).float()
        keypoints = torch.as_tensor(cached["native_keypoints"]).float()
        rows, anchors = _expanded_positive_pairs(record)
        valid = (anchors >= 0) & (anchors < base_count)
        rows, anchors = rows[valid], anchors[valid]

        current_mask = selected_base[anchors]
        current_rows = rows[current_mask]
        current_xyz = canonical_xyz[anchors[current_mask]]
        observed_tracks = track_observations.get(query_index, [])
        if observed_tracks:
            track_rows = torch.as_tensor(
                [item[0] for item in observed_tracks], dtype=torch.long
            )
            track_points = track_xyz[
                torch.as_tensor(
                    [item[1] for item in observed_tracks], dtype=torch.long
                )
            ]
            current_rows = torch.cat((current_rows, track_rows))
            current_xyz = torch.cat((current_xyz, track_points))
        if current_xyz.shape[0] >= 4:
            current_information = _information_contributions(
                current_xyz, K, pose
            ).sum(dim=0) + eye
        else:
            current_information = eye.clone()

        candidate_mask = eligible[anchors]
        candidate_rows = rows[candidate_mask]
        candidate_anchors = anchors[candidate_mask]
        if candidate_anchors.numel() == 0:
            query_candidates.append([])
            query_diagnostics.append(
                {"query": name, "candidate_count": 0}
            )
            continue
        candidate_xyz = canonical_xyz[candidate_anchors]
        contributions = _information_contributions(
            candidate_xyz, K, pose
        )
        unique_anchors, inverse = torch.unique(
            candidate_anchors, sorted=False, return_inverse=True
        )
        grouped = torch.zeros(
            (unique_anchors.numel(), 6, 6), dtype=torch.float64
        )
        grouped.index_add_(0, inverse, contributions)
        gains = conditional_add_gain(
            current_information[None], grouped, objective="translation"
        ).clamp_min(0)

        current_cells = set(
            _cell_ids(
                current_rows, keypoints, cached["native_input_hw"]
            ).tolist()
        )
        current_depth = set(_depth_bins(current_xyz, pose).tolist())
        current_voxels = set(
            torch.unique(
                torch.floor(
                    current_xyz / float(args.dependency_voxel_size)
                ).long(),
                dim=0,
            )
            .matmul(torch.tensor([73856093, 19349663, 83492791]))
            .tolist()
        )
        pair_cells = _cell_ids(
            candidate_rows, keypoints, cached["native_input_hw"]
        )
        pair_depth = _depth_bins(candidate_xyz, pose)
        pair_voxels = (
            torch.floor(
                candidate_xyz / float(args.dependency_voxel_size)
            )
            .long()
            .matmul(torch.tensor([73856093, 19349663, 83492791]))
        )
        novelty = torch.ones(unique_anchors.numel(), dtype=torch.float64)
        for pair_index, anchor_group in enumerate(inverse.tolist()):
            novelty[anchor_group] = max(
                float(novelty[anchor_group]),
                1.0
                + 0.25 * (int(pair_cells[pair_index]) not in current_cells)
                + 0.25 * (int(pair_depth[pair_index]) not in current_depth)
                + 0.25
                * (int(pair_voxels[pair_index]) not in current_voxels),
            )
        scores = gains * novelty
        finite = torch.isfinite(scores)
        if bool(finite.any()):
            scale = torch.quantile(scores[finite], 0.95).clamp_min(1e-8)
            scores = (scores / scale).clamp_max(4.0)
        scores[~finite] = 0
        topk = min(
            int(args.top_candidates_per_query), int(unique_anchors.numel())
        )
        values, order = torch.topk(scores, k=topk)
        query_candidates.append(
            [
                (int(unique_anchors[index]), float(value))
                for value, index in zip(values.tolist(), order.tolist())
                if value > 0
            ]
        )
        translation = translation_schur_complement(current_information)
        eigenvalues = torch.linalg.eigvalsh(translation)
        query_diagnostics.append(
            {
                "query": name,
                "current_pair_count": int(current_xyz.shape[0]),
                "candidate_count": int(unique_anchors.numel()),
                "translation_min_eigenvalue": float(eigenvalues[0]),
                "translation_condition": float(
                    eigenvalues[-1] / eigenvalues[0].clamp_min(1e-12)
                ),
            }
        )
        if completed % 50 == 0:
            temporary = scoring_partial.with_suffix(".pt.tmp")
            torch.save(
                {
                    "identity": scoring_identity,
                    "query_candidates": query_candidates,
                    "query_diagnostics": query_diagnostics,
                },
                temporary,
            )
            os.replace(temporary, scoring_partial)
            print(f"pose reserve scoring: {completed}/{len(teacher['records'])}")
    if len(query_candidates) != len(teacher["records"]):
        raise RuntimeError("pose reserve scoring did not cover every query")
    temporary = local_scoring_cache.with_suffix(".pt.tmp")
    torch.save(
        {
            "identity": scoring_identity,
            "query_candidates": query_candidates,
            "query_diagnostics": query_diagnostics,
        },
        temporary,
    )
    os.replace(temporary, local_scoring_cache)
    scoring_partial.unlink(missing_ok=True)

    additions = sorted(
        {int(value) for value in args.reserve_additions.split(",")}
    )
    query_weights = torch.ones(len(query_candidates))
    if args.reserve_mode != "all":
        errors = torch.as_tensor(
            [outcome_by_name[name] for name in teacher["query_names"]]
        )
        if args.reserve_mode == "precision":
            query_weights = ((errors > 5.0) & (errors <= 15.0)).float()
        else:
            query_weights = (errors > 50.0).float()
    selected = greedy_pose_reserve(
        query_candidates,
        source_ids,
        voxel_ids,
        budget=max(additions),
        query_weights=query_weights,
        force_fill=bool(args.force_fill),
    )
    summary = {
        "schema": "lafgs_v10_pose_sufficient_map_build",
        "core_map": str(core_path),
        "selection_split": "all_mapping_train",
        "selection_query_count": int(query_weights.numel()),
        "candidate_harmful_rate_max": float(args.maximum_harmful_rate),
        "reserve_mode": args.reserve_mode,
        "target_query_count": int((query_weights > 0).sum()),
        "force_fill": bool(args.force_fill),
        "natural_reserve_count": int(selected.numel()),
        "query_diagnostics": query_diagnostics,
        "maps": {},
    }
    realized_additions = {
        value for value in additions if value <= selected.numel()
    }
    if selected.numel():
        realized_additions.add(int(selected.numel()))
    for addition in sorted(realized_additions):
        added = selected[:addition]
        reserve = torch.cat((base_rows, added))
        state = _materialize(
            canonical,
            payload,
            track_indices,
            track_features,
            reserve,
            budget=int(track_count + reserve.numel()),
            quality_tier="strict_pose_sufficient",
            source_map=canonical_path,
            payload_path=payload_path,
            dependency_voxel_size=float(args.dependency_voxel_size),
        )
        state["track_centric_reconstruction"].update(
            {
                "schema": "lafgs_v10_pose_sufficient_map",
                "core_map": str(core_path),
                "pose_reserve_count": int(addition),
                "pose_reserve_canonical_rows": added,
            }
        )
        path = output_dir / f"pose_sufficient_add{addition:04d}.pt"
        torch.save(state, path)
        summary["maps"][f"add{addition:04d}"] = {
            "path": str(path),
            "anchor_count": int(track_count + reserve.numel()),
            "pose_reserve_count": int(addition),
        }
    report = output_dir / "pose_sufficient_build.json"
    report.write_text(json.dumps(summary, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
