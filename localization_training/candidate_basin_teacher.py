"""Candidate-aware P3P Basin teacher over deployed family retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from localization_training.basin_distillation import (
    GOOD_SET,
    HARMFUL_SET,
    NEAR_MISS_SET,
    as_basin_record,
    basin_risk,
    evaluate_p3p_triplet,
    expanded_positive_lookup,
    image_cell_ids,
    is_diverse_triplet,
    pack_basin_records,
)
from localization_training.shared_metric import SharedLowRankMetric


@dataclass(frozen=True)
class CandidateBasinConfig:
    topk: int = 16
    good_sets_per_query: int = 8
    harmful_sets_per_query: int = 6
    repairs_per_query: int = 8
    legal_candidates_per_row: int = 4
    two_edge_beam: int = 8
    adaptive_budget_factor: int = 2
    high_tail_threshold_cm: float = 0.0
    clean_threshold_px: float = 4.0
    harmful_threshold_px: float = 12.0
    minimum_harmful_inliers: int = 6
    seed: int = 2026
    query_start: int = 0
    query_limit: int = 0


def family_topk(
    query: torch.Tensor,
    bank: torch.Tensor,
    family: dict,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retrieve unique geometry anchors and retain each winning family mode."""
    primary_scores = query @ bank.T
    prototypes = F.normalize(
        torch.as_tensor(family["prototype_features"]).float(), dim=1
    ).to(query.device)
    parents = torch.as_tensor(
        family["prototype_anchor_indices"], device=query.device
    ).long()
    bias = torch.as_tensor(
        family.get("prototype_bias", torch.zeros(len(prototypes))),
        device=query.device,
    ).float()
    if bool((bias > 0).any()):
        raise ValueError("family prototype bias must be non-positive")
    temperature = torch.as_tensor(
        family.get("prototype_temperature", torch.ones(len(prototypes))),
        device=query.device,
    ).float()
    if bool((temperature <= 0).any()):
        raise ValueError("family prototype temperature must be positive")
    prototype_scores = query @ prototypes.T
    prototype_scores = prototype_scores / temperature[None] + bias[None]
    parent_winners: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for parent in parents.unique().tolist():
        modes = torch.nonzero(
            parents == int(parent), as_tuple=False
        ).reshape(-1)
        best, local = prototype_scores[:, modes].max(dim=1)
        improve = best > primary_scores[:, int(parent)]
        primary_scores[improve, int(parent)] = best[improve]
        winners = torch.full(
            (query.shape[0],), -1, device=query.device, dtype=torch.long
        )
        winners[improve] = modes[local[improve]]
        parent_winners[int(parent)] = (best, winners)
    scores, anchors = torch.topk(
        primary_scores,
        k=min(int(topk), primary_scores.shape[1]),
        dim=1,
    )
    modes = torch.full_like(anchors, -1)
    for column in range(anchors.shape[1]):
        for parent in torch.unique(anchors[:, column]).tolist():
            winner = parent_winners.get(int(parent))
            if winner is None:
                continue
            selected_rows = torch.nonzero(
                anchors[:, column] == int(parent), as_tuple=False
            ).reshape(-1)
            modes[selected_rows, column] = winner[1][selected_rows]
    return scores, anchors, modes, prototype_scores


def pose_basin_level(outcome: dict) -> int:
    if not bool(outcome["valid"]):
        return 0
    if float(outcome["te_cm"]) <= 5.0 and float(outcome["re_deg"]) <= 5.0:
        return 3
    if float(outcome["te_cm"]) <= 15.0 and float(outcome["re_deg"]) <= 2.0:
        return 2
    if float(outcome["te_cm"]) <= 50.0 and float(outcome["re_deg"]) <= 5.0:
        return 1
    return 0


def pack_candidate_basin_records(records: list[dict]) -> dict:
    packed = pack_basin_records(records)
    packed["set_mode_indices"] = (
        torch.as_tensor(
            [record["mode_indices"] for record in records],
            dtype=torch.long,
        )
        if records
        else torch.empty((0, 3), dtype=torch.long)
    )
    packed["repair_order"] = (
        torch.as_tensor(
            [record.get("repair_order", 0) for record in records],
            dtype=torch.int8,
        )
        if records
        else torch.empty(0, dtype=torch.int8)
    )
    packed["basin_level"] = (
        torch.as_tensor(
            [record.get("basin_level", 0) for record in records],
            dtype=torch.int8,
        )
        if records
        else torch.empty(0, dtype=torch.int8)
    )
    return packed


def candidate_basin_record(
    *,
    query_rows,
    anchors,
    modes,
    set_type,
    outcome,
    parent=-1,
    repair_order=0,
    replaced_position=-1,
    blame=0.0,
) -> dict:
    result = as_basin_record(
        query_rows=query_rows,
        anchors=anchors,
        set_type=set_type,
        outcome=outcome,
        propensity=1.0,
        proposal_attempts=1,
        parent=parent,
        replaced_position=(
            int(replaced_position) if int(repair_order) == 1 else -1
        ),
        blame=blame,
    )
    result["mode_indices"] = [int(value) for value in modes]
    result["repair_order"] = int(repair_order)
    result["basin_level"] = pose_basin_level(outcome)
    return result


def project_errors(
    xyz: np.ndarray,
    points2d: np.ndarray,
    K: np.ndarray,
    pose: np.ndarray,
) -> np.ndarray:
    camera = xyz @ pose[:3, :3].T + pose[:3, 3]
    valid = camera[:, 2] > 1e-8
    projected = np.empty_like(points2d)
    projected[:, 0] = K[0, 0] * camera[:, 0] / np.maximum(
        camera[:, 2], 1e-8
    ) + K[0, 2]
    projected[:, 1] = K[1, 1] * camera[:, 1] / np.maximum(
        camera[:, 2], 1e-8
    ) + K[1, 2]
    errors = np.linalg.norm(projected - points2d, axis=1)
    errors[~valid] = np.inf
    return errors


def build_candidate_basin_teacher(
    *,
    state: dict,
    metric: SharedLowRankMetric,
    family: dict,
    dynamic: dict,
    positives: dict,
    cache: dict,
    config: CandidateBasinConfig,
    device: torch.device,
    progress=None,
) -> dict:
    """Build good/harmful and one/two-edge counterfactual Basin sets."""
    names = list(dynamic["query_names"])
    if names != list(positives["query_names"]):
        raise ValueError("teacher query registries differ")
    anchor_count = int(torch.as_tensor(state["anchor_xyz"]).shape[0])
    if int(dynamic["anchor_count"]) != anchor_count:
        raise ValueError("dynamic outcomes do not align with map")
    if int(positives["anchor_count"]) != anchor_count:
        raise ValueError("positive teacher does not align with map")
    if not torch.equal(
        torch.as_tensor(family["landmark_indices"]).long(),
        torch.arange(anchor_count),
    ):
        raise ValueError("family state does not align with map")
    bank = F.normalize(
        torch.as_tensor(state["anchor_features"]).float(), dim=1
    ).to(device)
    xyz = torch.as_tensor(state["anchor_xyz"]).double().numpy()
    dependency = (
        torch.as_tensor(state["dependency_group_ids"]).long().numpy()
    )
    surfaces = (
        torch.as_tensor(state["source_primitive_ids"]).long().numpy()
    )
    family_parents = torch.as_tensor(
        family["prototype_anchor_indices"]
    ).long()
    family_modes_by_parent = {
        int(parent): torch.nonzero(
            family_parents == int(parent), as_tuple=False
        ).reshape(-1)
        for parent in family_parents.unique().tolist()
    }
    metric = metric.to(device).eval()
    dynamic_te = np.asarray(
        [float(record["te_cm"]) for record in dynamic["records"]]
    )
    high_tail = (
        float(config.high_tail_threshold_cm)
        if float(config.high_tail_threshold_cm) > 0
        else float(np.quantile(dynamic_te, 0.9))
    )
    start = max(int(config.query_start), 0)
    stop = len(names)
    if int(config.query_limit) > 0:
        stop = min(stop, start + int(config.query_limit))
    totals = {
        "good": 0,
        "harmful": 0,
        "one_edge_repair": 0,
        "two_edge_repair": 0,
        "strict": 0,
        "precision": 0,
        "coarse": 0,
        "queries_with_strict": 0,
        "queries_with_repair": 0,
    }
    output_records = []
    rng = np.random.default_rng(int(config.seed))
    for query_index in range(start, stop):
        name = names[query_index]
        cached = cache[name]
        rows = torch.as_tensor(
            dynamic["records"][query_index]["query_rows"]
        ).long()
        raw = F.normalize(
            torch.as_tensor(cached["native_descriptors"]).float()[rows],
            dim=1,
        ).to(device)
        with torch.no_grad():
            query, _ = metric(raw)
            top_scores, top_anchors, top_modes, prototype_scores = family_topk(
                query, bank, family, config.topk
            )
        top_scores = top_scores.cpu()
        top_anchors = top_anchors.cpu()
        top_modes = top_modes.cpu()
        prototype_scores = prototype_scores.cpu()
        primary_scores = (query @ bank.T).cpu()
        points2d = (
            torch.as_tensor(cached["native_keypoints"]).double()[rows].numpy()
            + float(cached.get("pixel_center_offset", 0.5))
        )
        K = torch.as_tensor(cached["native_K"]).double().numpy()
        gt_pose = torch.as_tensor(cached["pose_w2c"]).double().numpy()
        height, width = cached["native_input_hw"]
        cells = image_cell_ids(points2d, int(width), int(height))
        current_anchors = top_anchors[:, 0].numpy()
        current_modes = top_modes[:, 0].numpy()
        points3d = xyz[current_anchors]
        errors = project_errors(points3d, points2d, K, gt_pose)
        clean_pool = np.flatnonzero(
            errors <= float(config.clean_threshold_px)
        )
        harmful_pool = np.flatnonzero(
            errors > float(config.harmful_threshold_px)
        )
        positive_lookup = expanded_positive_lookup(
            positives["records"][query_index]
        )
        legal_candidate_cache: dict[
            int, list[tuple[int, int, float]]
        ] = {}

        def legal_candidates(
            edge_index: int,
        ) -> list[tuple[int, int, float]]:
            cached_candidates = legal_candidate_cache.get(int(edge_index))
            if cached_candidates is not None:
                return cached_candidates
            row = int(rows[edge_index])
            legal = set(positive_lookup.get(row, []))
            candidates = []
            for score, anchor, mode in zip(
                top_scores[edge_index].tolist(),
                top_anchors[edge_index].tolist(),
                top_modes[edge_index].tolist(),
            ):
                if int(anchor) in legal:
                    candidates.append((int(anchor), int(mode), float(score)))
            for anchor in legal:
                if any(value[0] == int(anchor) for value in candidates):
                    continue
                score = float(primary_scores[edge_index, int(anchor)])
                mode = -1
                family_modes = family_modes_by_parent.get(
                    int(anchor), torch.empty(0, dtype=torch.long)
                )
                if family_modes.numel():
                    values = prototype_scores[edge_index, family_modes]
                    value, local = values.max(dim=0)
                    if float(value) > score:
                        score = float(value)
                        mode = int(family_modes[int(local)])
                candidates.append((int(anchor), mode, score))
            candidates.sort(key=lambda value: -value[2])
            selected = candidates[: int(config.legal_candidates_per_row)]
            legal_candidate_cache[int(edge_index)] = selected
            return selected

        repairable = np.asarray(
            [
                edge
                for edge in harmful_pool.tolist()
                if legal_candidates(edge)
            ],
            dtype=np.int64,
        )
        adaptive = (
            int(config.adaptive_budget_factor)
            if dynamic_te[query_index] >= high_tail
            else 1
        )
        good_budget = int(config.good_sets_per_query) * adaptive
        harmful_budget = int(config.harmful_sets_per_query) * adaptive
        repair_budget = int(config.repairs_per_query) * adaptive
        records: list[dict] = []
        seen = set()
        good_rows = np.asarray(
            [
                edge
                for edge in range(len(rows))
                if legal_candidates(edge)
            ],
            dtype=np.int64,
        )
        for _ in range(good_budget * 12):
            if (
                sum(
                    record["set_type"] == GOOD_SET for record in records
                )
                >= good_budget
            ):
                break
            if good_rows.size < 3:
                break
            sample = rng.choice(good_rows, 3, replace=False)
            choices = [legal_candidates(int(edge))[0] for edge in sample]
            anchors = np.asarray([value[0] for value in choices])
            modes = np.asarray([value[1] for value in choices])
            identity = tuple(
                sorted(
                    (int(rows[edge]), int(anchor))
                    for edge, anchor in zip(sample, anchors)
                )
            )
            if identity in seen:
                continue
            outcome = evaluate_p3p_triplet(
                sample,
                points2d,
                points3d,
                K,
                gt_pose,
                basis_points3d=xyz[anchors],
            )
            if pose_basin_level(outcome) == 0:
                continue
            seen.add(identity)
            records.append(
                candidate_basin_record(
                    query_rows=rows[sample],
                    anchors=anchors,
                    modes=modes,
                    set_type=GOOD_SET,
                    outcome=outcome,
                )
            )
        harmful_records = []
        for _ in range(harmful_budget * 16):
            if len(harmful_records) >= harmful_budget:
                break
            if clean_pool.size < 2 or repairable.size < 1:
                break
            sample = np.concatenate(
                (
                    rng.choice(clean_pool, 2, replace=False),
                    rng.choice(repairable, 1, replace=False),
                )
            )
            rng.shuffle(sample)
            if not is_diverse_triplet(
                sample,
                points3d,
                dependency[current_anchors],
                cells,
                surfaces[current_anchors],
            ):
                continue
            outcome = evaluate_p3p_triplet(
                sample, points2d, points3d, K, gt_pose
            )
            if pose_basin_level(outcome) > 0 or int(
                outcome["inlier_count"]
            ) < int(config.minimum_harmful_inliers):
                continue
            record = candidate_basin_record(
                query_rows=rows[sample],
                anchors=current_anchors[sample],
                modes=current_modes[sample],
                set_type=HARMFUL_SET,
                outcome=outcome,
            )
            record["_sample"] = sample
            harmful_records.append(record)
            records.append(record)
        repairs = 0
        for parent in harmful_records:
            if repairs >= repair_budget:
                break
            sample = parent["_sample"]
            parent_index = records.index(parent)
            parent_risk = basin_risk(parent)
            one_edge = []
            for position, edge in enumerate(sample.tolist()):
                for anchor, mode, _ in legal_candidates(edge):
                    child_anchors = current_anchors[sample].copy()
                    child_modes = current_modes[sample].copy()
                    child_anchors[position] = anchor
                    child_modes[position] = mode
                    outcome = evaluate_p3p_triplet(
                        sample,
                        points2d,
                        points3d,
                        K,
                        gt_pose,
                        basis_points3d=xyz[child_anchors],
                    )
                    one_edge.append(
                        (
                            pose_basin_level(outcome),
                            parent_risk - basin_risk(outcome),
                            position,
                            child_anchors,
                            child_modes,
                            outcome,
                        )
                    )
            one_edge.sort(
                key=lambda value: (value[0], value[1]), reverse=True
            )
            best = one_edge[0] if one_edge else None
            if best is not None and best[0] > 0 and best[1] > 0:
                (
                    _,
                    blame,
                    position,
                    child_anchors,
                    child_modes,
                    outcome,
                ) = best
                records.append(
                    candidate_basin_record(
                        query_rows=rows[sample],
                        anchors=child_anchors,
                        modes=child_modes,
                        set_type=NEAR_MISS_SET,
                        outcome=outcome,
                        parent=parent_index,
                        repair_order=1,
                        replaced_position=position,
                        blame=blame,
                    )
                )
                repairs += 1
                continue
            two_edge = []
            for first in one_edge[: int(config.two_edge_beam)]:
                _, _, first_position, first_anchors, first_modes, _ = first
                for second_position, edge in enumerate(sample.tolist()):
                    if second_position == first_position:
                        continue
                    for anchor, mode, _ in legal_candidates(edge)[:2]:
                        child_anchors = first_anchors.copy()
                        child_modes = first_modes.copy()
                        child_anchors[second_position] = anchor
                        child_modes[second_position] = mode
                        outcome = evaluate_p3p_triplet(
                            sample,
                            points2d,
                            points3d,
                            K,
                            gt_pose,
                            basis_points3d=xyz[child_anchors],
                        )
                        two_edge.append(
                            (
                                pose_basin_level(outcome),
                                parent_risk - basin_risk(outcome),
                                child_anchors,
                                child_modes,
                                outcome,
                            )
                        )
            two_edge.sort(
                key=lambda value: (value[0], value[1]), reverse=True
            )
            if two_edge and two_edge[0][0] > 0 and two_edge[0][1] > 0:
                _, blame, child_anchors, child_modes, outcome = two_edge[0]
                records.append(
                    candidate_basin_record(
                        query_rows=rows[sample],
                        anchors=child_anchors,
                        modes=child_modes,
                        set_type=NEAR_MISS_SET,
                        outcome=outcome,
                        parent=parent_index,
                        repair_order=2,
                        blame=blame,
                    )
                )
                repairs += 1
        for record in records:
            record.pop("_sample", None)
        packed = pack_candidate_basin_records(records)
        packed.update({"query_index": query_index, "query_name": name})
        output_records.append(packed)
        types = packed["set_types"]
        levels = packed["basin_level"]
        counts = {
            "good": int((types == GOOD_SET).sum()),
            "harmful": int((types == HARMFUL_SET).sum()),
            "one_edge_repair": int((packed["repair_order"] == 1).sum()),
            "two_edge_repair": int((packed["repair_order"] == 2).sum()),
            "strict": int((levels == 3).sum()),
            "precision": int((levels >= 2).sum()),
            "coarse": int((levels >= 1).sum()),
        }
        for key, value in counts.items():
            totals[key] += value
        totals["queries_with_strict"] += int(counts["strict"] > 0)
        totals["queries_with_repair"] += int(
            counts["one_edge_repair"] + counts["two_edge_repair"] > 0
        )
        if progress is not None:
            progress(query_index - start + 1, stop - start, dict(totals))
    return {
        "schema": "lafgs_candidate_aware_basin_teacher",
        "version": 3,
        "query_names": names[start:stop],
        "query_start": start,
        "query_stop": stop,
        "anchor_count": anchor_count,
        "prototype_count": int(
            torch.as_tensor(family["prototype_features"]).shape[0]
        ),
        "records": output_records,
        "summary": {**totals, "query_count": len(output_records)},
        "config": asdict(config),
    }

