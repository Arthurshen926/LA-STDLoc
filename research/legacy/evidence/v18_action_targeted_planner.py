"""Action-targeted intervention/necessity/global confirmation selection."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from evidence.v7_query_planner import camera_centers


def _visible_rows(
    xyz: torch.Tensor,
    pose: torch.Tensor,
    intrinsic: torch.Tensor,
    image_hw: torch.Tensor,
) -> torch.Tensor:
    camera = xyz @ pose[:3, :3].T + pose[:3, 3]
    projected = camera @ intrinsic.T
    uv = projected[:, :2] / projected[:, 2:].clamp_min(1e-8)
    height, width = map(int, image_hw.tolist())
    return (
        torch.isfinite(uv).all(1)
        & torch.isfinite(camera[:, 2])
        & (camera[:, 2] > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )


def _subset_plan(plan: Mapping, selected: torch.Tensor) -> dict:
    count = int(plan["query_count"])
    output = {}
    for key, value in plan.items():
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == count:
            output[key] = value[selected].clone()
        elif isinstance(value, list) and len(value) == count:
            output[key] = [value[index] for index in selected.tolist()]
        else:
            output[key] = value
    output["query_count"] = int(selected.numel())
    return output


def select_action_targeted_queries(
    *,
    candidate_plan: Mapping,
    anchor_xyz: torch.Tensor,
    harmful_anchor_rows: torch.Tensor,
    reactivated_anchor_rows: torch.Tensor,
    backup_offsets: torch.Tensor,
    backup_anchor_rows: torch.Tensor,
    anchor_observation_offsets: torch.Tensor,
    observation_query_indices: torch.Tensor,
    mapping_poses_w2c: torch.Tensor,
    maximum_queries: int,
    intervention_fraction: float = 0.40,
    necessity_fraction: float = 0.30,
    minimum_observation_direction_cosine: float = 0.50,
) -> dict:
    """Select targeted views from an already fresh continuous-SE(3) pool.

    Intervention views jointly see a harmful Anchor and one certified backup,
    or a reactivated Anchor, from a direction supported by the action Anchor's
    original mapping observations. Necessity views see the harmful Anchor from
    the same support cone. Remaining slots are global collateral views.
    The function never changes poses and never consumes test metadata.
    """

    if candidate_plan.get("uses_test_queries") is not False:
        raise ValueError("action-targeted planning requires no-test candidates")
    count = int(candidate_plan["query_count"])
    poses = torch.as_tensor(candidate_plan["pose_w2c"]).double()
    intrinsics = torch.as_tensor(candidate_plan["intrinsics"]).double()
    image_hw = torch.as_tensor(candidate_plan["image_hw"]).long()
    if poses.shape != (count, 4, 4) or intrinsics.shape != (count, 3, 3):
        raise ValueError("action-targeted candidate camera registry differs")
    xyz = torch.as_tensor(anchor_xyz).double()
    harmful = torch.as_tensor(harmful_anchor_rows).long().reshape(-1)
    reactivated = torch.as_tensor(reactivated_anchor_rows).long().reshape(-1)
    backup_offsets = torch.as_tensor(backup_offsets).long().reshape(-1)
    backups = torch.as_tensor(backup_anchor_rows).long().reshape(-1)
    if backup_offsets.shape != (harmful.numel() + 1,) or int(backup_offsets[-1]) != backups.numel():
        raise ValueError("harmful/backup CSR does not align")
    action_rows = torch.cat((harmful, reactivated))
    if action_rows.numel() and (
        int(action_rows.min()) < 0
        or int(action_rows.max()) >= xyz.shape[0]
        or (
            backups.numel()
            and (int(backups.min()) < 0 or int(backups.max()) >= xyz.shape[0])
        )
    ):
        raise ValueError("targeted action references an Anchor outside the map")
    observation_offsets = torch.as_tensor(anchor_observation_offsets).long()
    observation_queries = torch.as_tensor(observation_query_indices).long()
    mapping_poses = torch.as_tensor(mapping_poses_w2c).double()
    if observation_offsets.shape != (xyz.shape[0] + 1,):
        raise ValueError("targeted planner Anchor observation CSR differs")
    mapping_centers = camera_centers(mapping_poses)
    candidate_centers = camera_centers(poses)

    intervention_score = torch.zeros(count)
    necessity_score = torch.zeros(count)
    action_direction_score = torch.full((count,), -1.0)
    for candidate in range(count):
        targeted_rows = torch.unique(torch.cat((harmful, reactivated, backups)))
        visible = _visible_rows(
            xyz[targeted_rows],
            poses[candidate],
            intrinsics[candidate],
            image_hw[candidate],
        )
        visibility = {
            int(anchor): bool(value)
            for anchor, value in zip(targeted_rows.tolist(), visible.tolist())
        }
        supported_direction: dict[int, float] = {}
        for anchor in action_rows.tolist():
            observation_start = int(observation_offsets[anchor])
            observation_stop = int(observation_offsets[anchor + 1])
            queries = observation_queries[observation_start:observation_stop]
            if queries.numel() == 0:
                supported_direction[anchor] = -1.0
                continue
            action_direction = torch.nn.functional.normalize(
                candidate_centers[candidate] - xyz[anchor], dim=0
            )
            observed_directions = torch.nn.functional.normalize(
                mapping_centers[queries] - xyz[anchor], dim=1
            )
            supported_direction[anchor] = float(
                (observed_directions @ action_direction).max()
            )
        action_direction_score[candidate] = max(
            supported_direction.values(), default=-1.0
        )
        intervention = sum(
            visibility.get(anchor, False)
            and supported_direction.get(anchor, -1.0)
            >= float(minimum_observation_direction_cosine)
            for anchor in reactivated.tolist()
        )
        necessity_values = []
        for local, anchor in enumerate(harmful.tolist()):
            start, stop = int(backup_offsets[local]), int(backup_offsets[local + 1])
            local_backups = backups[start:stop].tolist()
            if (
                visibility.get(anchor, False)
                and supported_direction.get(anchor, -1.0)
                >= float(minimum_observation_direction_cosine)
                and any(visibility.get(backup, False) for backup in local_backups)
            ):
                intervention += 1
            if not visibility.get(anchor, False):
                continue
            necessity_values.append(supported_direction.get(anchor, -1.0))
        intervention_score[candidate] = intervention
        necessity_score[candidate] = max(necessity_values, default=-1.0)

    maximum = min(max(int(maximum_queries), 1), count)
    intervention_quota = (
        min(round(maximum * float(intervention_fraction)), maximum)
        if action_rows.numel()
        else 0
    )
    necessity_quota = (
        min(round(maximum * float(necessity_fraction)), maximum - intervention_quota)
        if harmful.numel()
        else 0
    )
    global_quota = maximum - intervention_quota - necessity_quota
    selected: list[int] = []
    kinds: dict[int, str] = {}

    def take(order: torch.Tensor, quota: int, kind: str, predicate) -> None:
        for index in order.tolist():
            if len([value for value in selected if kinds[value] == kind]) >= quota:
                break
            if index in kinds or not predicate(index):
                continue
            selected.append(index)
            kinds[index] = kind

    intervention_order = torch.argsort(intervention_score, descending=True, stable=True)
    take(
        intervention_order,
        intervention_quota,
        "intervention",
        lambda index: float(intervention_score[index]) > 0.0,
    )
    necessity_order = torch.argsort(necessity_score, descending=True, stable=True)
    take(
        necessity_order,
        necessity_quota,
        "necessity",
        lambda index: float(necessity_score[index]) > 0.0,
    )
    global_score = torch.as_tensor(
        candidate_plan.get("visible_cell_count", torch.ones(count))
    ).float()
    global_order = torch.argsort(global_score, descending=True, stable=True)
    take(global_order, global_quota, "global_collateral", lambda _index: True)
    # If a target category has too few feasible views, fill only from the
    # independent global pool and disclose the realized mix.
    for index in global_order.tolist():
        if len(selected) >= maximum:
            break
        if index not in kinds:
            selected.append(index)
            kinds[index] = "global_collateral"
    selected_tensor = torch.tensor(selected, dtype=torch.long)
    output = _subset_plan(candidate_plan, selected_tensor)
    output["query_kinds"] = [kinds[index] for index in selected]
    output["target_intervention_score"] = intervention_score[selected_tensor]
    output["target_necessity_direction_score"] = necessity_score[selected_tensor]
    output["target_action_direction_score"] = action_direction_score[selected_tensor]
    realized = {
        kind: output["query_kinds"].count(kind)
        for kind in ("intervention", "necessity", "global_collateral")
    }
    output["action_targeted_planner"] = {
        "schema": "lafgs_v18_action_targeted_confirmation",
        "version": 1,
        "uses_test_queries": False,
        "loo_used": False,
        "target_action_anchor_count": int(harmful.numel()),
        "target_reactivated_anchor_count": int(reactivated.numel()),
        "target_backup_anchor_count": int(backups.numel()),
        "candidate_query_count": count,
        "selected_query_count": int(selected_tensor.numel()),
        "requested_mix": {
            "intervention": intervention_quota,
            "necessity": necessity_quota,
            "global_collateral": global_quota,
        },
        "realized_mix": realized,
        "selection_policy": "joint_visibility_then_original_view_direction_then_global",
        "minimum_observation_direction_cosine": float(
            minimum_observation_direction_cosine
        ),
    }
    return output


__all__ = ["select_action_targeted_queries"]
