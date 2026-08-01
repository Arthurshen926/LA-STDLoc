"""Training-only candidate repairs supported by independent mapping trajectories."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from localization_training.appearance_family import trajectory_id
from localization_training.shared_metric import SharedLowRankMetric


@dataclass(frozen=True)
class TrajectoryStableCandidateConfig:
    topk: int = 16
    minimum_observations: int = 2
    minimum_trajectories: int = 2
    minimum_view_bins: int = 2
    single_trajectory_minimum_observations: int = 3
    single_trajectory_minimum_view_bins: int = 3
    maximum_score_gap: float = 0.20
    promotion_weight: float = 2.0
    harmful_inlier_multiplier: float = 1.5


def _positive_entries(record: dict) -> list[tuple[torch.Tensor, torch.Tensor]]:
    offsets = torch.as_tensor(record["positive_offsets"]).long()
    indices = torch.as_tensor(record["positive_indices"]).long()
    return [
        (
            indices[offsets[row] : offsets[row + 1]],
            torch.arange(offsets[row], offsets[row + 1], dtype=torch.long),
        )
        for row in range(offsets.numel() - 1)
    ]


def _is_stable(
    proposals: list[dict],
    *,
    multi_trajectory_scene: bool,
    config: TrajectoryStableCandidateConfig,
) -> bool:
    trajectories = {value["trajectory"] for value in proposals}
    view_bins = {int(value["view_bin"]) for value in proposals}
    if multi_trajectory_scene:
        return bool(
            len(proposals) >= int(config.minimum_observations)
            and len(trajectories) >= int(config.minimum_trajectories)
            and len(view_bins) >= int(config.minimum_view_bins)
        )
    return bool(
        len(proposals) >= int(config.single_trajectory_minimum_observations)
        and len(view_bins) >= int(config.single_trajectory_minimum_view_bins)
    )


def build_trajectory_stable_candidate_teacher(
    *,
    state: dict,
    metric: SharedLowRankMetric,
    positives: dict,
    dynamic: dict,
    cache: dict,
    query_bins: dict[str, int],
    config: TrajectoryStableCandidateConfig,
    device: torch.device,
    progress=None,
) -> dict:
    """Mine rank promotion/recall targets from the current deployed matcher.

    The output never changes deployment. It only promotes an existing legal
    anchor when the same false-attractor-to-target relation repeats across
    independent trajectory or view-bin support.
    """
    names = [str(value) for value in positives["query_names"]]
    if names != [str(value) for value in dynamic["query_names"]]:
        raise ValueError("candidate teacher query registries differ")
    anchor_count = int(torch.as_tensor(state["anchor_xyz"]).shape[0])
    if int(positives["anchor_count"]) != anchor_count:
        raise ValueError("candidate teacher positives do not align with map")
    if int(dynamic["anchor_count"]) != anchor_count:
        raise ValueError("candidate teacher outcomes do not align with map")
    if set(names) != set(cache):
        raise ValueError("candidate teacher cache query registry differs")
    missing_bins = sorted(set(names) - set(query_bins))
    if missing_bins:
        raise ValueError(f"candidate teacher is missing query bins: {missing_bins[:3]}")

    bank = F.normalize(torch.as_tensor(state["anchor_features"]).float(), dim=1).to(
        device
    )
    coarse_groups = torch.as_tensor(
        state.get(
            "coarse_dependency_group_ids",
            state.get("dependency_group_ids", torch.arange(anchor_count)),
        )
    ).long()
    metric = metric.to(device).eval()
    proposals = []
    generated_top1_mismatch = 0
    compared_top1 = 0
    with torch.no_grad():
        for query_index, name in enumerate(names):
            teacher_record = positives["records"][query_index]
            outcome = dynamic["records"][query_index]
            rows = torch.as_tensor(teacher_record["query_rows"]).long()
            if not torch.equal(rows, torch.as_tensor(outcome["query_rows"]).long()):
                raise ValueError("candidate teacher rows do not align with outcomes")
            positive_rows = _positive_entries(teacher_record)
            raw = F.normalize(
                torch.as_tensor(cache[name]["native_descriptors"]).float()[rows], dim=1
            ).to(device)
            query, _ = metric(raw)
            scores = query @ bank.T
            top_scores, top_indices = torch.topk(
                scores, k=min(int(config.topk), anchor_count), dim=1
            )
            deployed_top1 = torch.as_tensor(outcome["top1_anchor_indices"]).long()
            compared_top1 += int(rows.numel())
            generated_top1_mismatch += int(
                (top_indices[:, 0].cpu() != deployed_top1).sum()
            )
            gt_errors = torch.as_tensor(outcome["gt_reprojection_errors_px"]).float()
            inliers = torch.as_tensor(outcome["ransac_inlier_mask"]).bool()
            trajectory = trajectory_id(name)
            view_bin = int(query_bins[name])
            for row_index, (legal, flattened_positions) in enumerate(positive_rows):
                if not legal.numel():
                    continue
                legal_device = legal.to(device)
                false_anchor = int(top_indices[row_index, 0])
                if bool((legal == false_anchor).any()):
                    continue
                retrieved = top_indices[row_index]
                legal_retrieved = torch.isin(retrieved, legal_device)
                if bool(legal_retrieved.any()):
                    first_rank = int(torch.nonzero(legal_retrieved)[0])
                    target_anchor = int(retrieved[first_rank])
                    repair_type = 1
                    positive_rank = first_rank + 1
                else:
                    legal_scores = scores[row_index, legal_device]
                    target_anchor = int(legal_device[int(legal_scores.argmax())])
                    repair_type = 2
                    positive_rank = -1
                target_column = int(torch.nonzero(legal == target_anchor)[0])
                flattened_position = int(flattened_positions[target_column])
                score_gap = float(
                    scores[row_index, false_anchor]
                    - scores[row_index, target_anchor]
                )
                if score_gap > float(config.maximum_score_gap):
                    continue
                harmful_inlier = bool(inliers[row_index] and gt_errors[row_index] > 4.0)
                proposals.append(
                    {
                        "query_index": query_index,
                        "query_name": name,
                        "teacher_row": row_index,
                        "query_row": int(rows[row_index]),
                        "trajectory": trajectory,
                        "view_bin": view_bin,
                        "false_anchor": false_anchor,
                        "false_group": int(coarse_groups[false_anchor]),
                        "target_anchor": target_anchor,
                        "positive_flat_position": flattened_position,
                        "repair_type": repair_type,
                        "positive_rank": positive_rank,
                        "score_gap": score_gap,
                        "harmful_inlier": harmful_inlier,
                    }
                )
            if progress is not None:
                progress(query_index + 1, len(names))

    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for proposal in proposals:
        grouped[(proposal["false_group"], proposal["target_anchor"])].append(
            proposal
        )
    scene_trajectories = {trajectory_id(name) for name in names}
    multi_trajectory_scene = len(scene_trajectories) > 1
    stable_keys = {
        key
        for key, values in grouped.items()
        if _is_stable(
            values,
            multi_trajectory_scene=multi_trajectory_scene,
            config=config,
        )
    }
    accepted = [
        value
        for key, values in grouped.items()
        if key in stable_keys
        for value in values
    ]
    accepted_by_query_row = {
        (int(value["query_index"]), int(value["teacher_row"])): value
        for value in accepted
    }
    output_records = []
    for query_index, record in enumerate(positives["records"]):
        rows = torch.as_tensor(record["query_rows"]).long()
        pair_count = int(torch.as_tensor(record["positive_indices"]).numel())
        positive_weights = torch.ones(pair_count, dtype=torch.float32)
        row_weights = torch.ones(rows.numel(), dtype=torch.float32)
        promotion_positive = torch.full((rows.numel(),), -1, dtype=torch.long)
        promotion_negative = torch.full((rows.numel(),), -1, dtype=torch.long)
        promotion_weights = torch.zeros(rows.numel(), dtype=torch.float32)
        promotion_types = torch.zeros(rows.numel(), dtype=torch.uint8)
        for row_index in range(rows.numel()):
            proposal = accepted_by_query_row.get((query_index, row_index))
            if proposal is None:
                continue
            weight = float(config.promotion_weight)
            if proposal["harmful_inlier"]:
                weight *= float(config.harmful_inlier_multiplier)
            positive_weights[proposal["positive_flat_position"]] = 1.0 + weight
            row_weights[row_index] = 1.0 + weight
            promotion_positive[row_index] = int(proposal["target_anchor"])
            promotion_negative[row_index] = int(proposal["false_anchor"])
            promotion_weights[row_index] = weight
            promotion_types[row_index] = int(proposal["repair_type"])
        output_records.append(
            {
                "query_index": query_index,
                "query_name": names[query_index],
                "query_rows": rows,
                "positive_weights": positive_weights,
                "row_weights": row_weights,
                "promotion_positive_anchor": promotion_positive,
                "promotion_negative_anchor": promotion_negative,
                "promotion_weights": promotion_weights,
                "promotion_types": promotion_types,
            }
        )

    stable_support = []
    for key in sorted(stable_keys):
        values = grouped[key]
        stable_support.append(
            {
                "false_group": int(key[0]),
                "target_anchor": int(key[1]),
                "observation_count": len(values),
                "trajectory_count": len({value["trajectory"] for value in values}),
                "view_bin_count": len({value["view_bin"] for value in values}),
                "promotion_count": sum(value["repair_type"] == 1 for value in values),
                "recall_count": sum(value["repair_type"] == 2 for value in values),
                "harmful_inlier_count": sum(value["harmful_inlier"] for value in values),
            }
        )
    return {
        "schema": "lafgs_trajectory_stable_pose_guided_candidate_teacher",
        "version": 1,
        "anchor_count": anchor_count,
        "query_names": names,
        "records": output_records,
        "stable_support": stable_support,
        "summary": {
            "query_count": len(names),
            "scene_trajectory_count": len(scene_trajectories),
            "proposal_count": len(proposals),
            "stable_relation_count": len(stable_keys),
            "accepted_row_count": len(accepted),
            "promotion_row_count": sum(value["repair_type"] == 1 for value in accepted),
            "recall_row_count": sum(value["repair_type"] == 2 for value in accepted),
            "harmful_inlier_row_count": sum(value["harmful_inlier"] for value in accepted),
            "unmatchable_rows_ignored": sum(
                (torch.as_tensor(record["positive_offsets"])[1:]
                 == torch.as_tensor(record["positive_offsets"])[:-1]).sum().item()
                for record in positives["records"]
            ),
            "generated_top1_mismatch_rate": (
                generated_top1_mismatch / max(compared_top1, 1)
            ),
        },
        "config": asdict(config),
    }
