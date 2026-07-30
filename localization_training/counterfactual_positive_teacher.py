"""Build pose-validated targets for selection-aware descriptor repair."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from localization_training.pose_information import pose_jacobian_analytic
from localization_training.selection_aware_reconstruction import (
    ModeTable,
    winning_representation,
)


ROUTE_PRIMARY = 0
ROUTE_FAMILY = 1
ROUTE_REJECT = 2
ROUTE_NAMES = ("primary", "family", "reject")


@dataclass(frozen=True)
class CounterfactualTeacherConfig:
    translation_scale_m: float = 0.07160573943725686
    rotation_scale_degrees: float = 2.0
    residual_scale_px: float = 4.0
    residual_clip_px: float = 12.0
    minimum_translation_logdet_gain: float = -1e-10
    minimum_primary_trajectory_support: int = 4
    minimum_family_trajectory_support: int = 2


def _trajectory(name: str) -> str:
    return name.split("/", 1)[0]


def _positive_lookup(record: dict) -> dict[int, torch.Tensor]:
    rows = torch.as_tensor(record["query_rows"]).long().reshape(-1)
    offsets = torch.as_tensor(record["positive_offsets"]).long().reshape(-1)
    indices = torch.as_tensor(record["positive_indices"]).long().reshape(-1)
    return {
        int(row): indices[int(offsets[index]) : int(offsets[index + 1])]
        for index, row in enumerate(rows.tolist())
    }


def _pack(values: list[torch.Tensor], dtype) -> tuple[torch.Tensor, torch.Tensor]:
    counts = torch.as_tensor([len(value) for value in values], dtype=torch.long)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    if int(offsets[-1]) == 0:
        return offsets, torch.empty(0, dtype=dtype)
    return offsets, torch.cat(
        [torch.as_tensor(value, dtype=dtype).reshape(-1) for value in values]
    )


def build_anchor_cross_trajectory_support(
    positive_teacher: dict,
    anchor_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count positive observations and distinct mapping trajectories."""

    observations = torch.zeros(anchor_count, dtype=torch.long)
    trajectories = [set() for _ in range(anchor_count)]
    for name, record in zip(
        positive_teacher["query_names"], positive_teacher["records"]
    ):
        values = torch.as_tensor(record["positive_indices"]).long().unique()
        values = values[(values >= 0) & (values < anchor_count)]
        observations[values] += 1
        trajectory = _trajectory(name)
        for anchor in values.tolist():
            trajectories[int(anchor)].add(trajectory)
    trajectory_count = torch.as_tensor(
        [len(value) for value in trajectories], dtype=torch.long
    )
    return observations, trajectory_count


def _pose_terms(
    points_2d: torch.Tensor,
    points_3d: torch.Tensor,
    K: torch.Tensor,
    pose_w2c: torch.Tensor,
    config: CounterfactualTeacherConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    points_2d = torch.as_tensor(points_2d, dtype=torch.float64)
    points_3d = torch.as_tensor(points_3d, dtype=torch.float64)
    K = torch.as_tensor(K, dtype=torch.float64)
    pose = torch.as_tensor(pose_w2c, dtype=torch.float64)
    camera = points_3d @ pose[:3, :3].T + pose[:3, 3]
    depth = camera[:, 2]
    projected = camera @ K.T
    uv = projected[:, :2] / depth[:, None].clamp_min(1e-12)
    residual = points_2d - uv
    residual_norm = torch.linalg.norm(residual, dim=1)
    finite = (
        torch.isfinite(camera).all(dim=1)
        & (depth > 1e-8)
        & torch.isfinite(residual_norm)
    )
    clip = torch.clamp(
        float(config.residual_clip_px) / residual_norm.clamp_min(1e-12),
        max=1.0,
    )
    residual = residual * clip[:, None]
    weight = torch.exp(
        -residual_norm.square() / (2.0 * float(config.residual_scale_px) ** 2)
    )
    weight[~finite] = 0.0
    residual[~finite] = 0.0
    jacobian = pose_jacobian_analytic(points_3d, K, pose).double()
    scales = torch.tensor(
        [
            config.translation_scale_m,
            config.translation_scale_m,
            config.translation_scale_m,
            torch.deg2rad(torch.tensor(config.rotation_scale_degrees)).item(),
            torch.deg2rad(torch.tensor(config.rotation_scale_degrees)).item(),
            torch.deg2rad(torch.tensor(config.rotation_scale_degrees)).item(),
        ],
        dtype=torch.float64,
    )
    jacobian = jacobian * scales
    information = weight[:, None, None] * torch.einsum(
        "nai,naj->nij", jacobian, jacobian
    )
    gradient = weight[:, None] * torch.einsum(
        "nai,na->ni", jacobian, residual
    )
    return information, gradient


def _pose_metrics(
    information: torch.Tensor,
    gradient: torch.Tensor,
    translation_scale_m: float,
) -> tuple[float, float]:
    information = 0.5 * (information + information.T)
    try:
        delta = -torch.linalg.solve(information, gradient)
    except torch.linalg.LinAlgError:
        delta = -torch.linalg.pinv(information) @ gradient
    h_tt = information[:3, :3]
    h_tr = information[:3, 3:]
    h_rr = information[3:, 3:]
    translation = h_tt - h_tr @ torch.linalg.pinv(h_rr) @ h_tr.T
    translation = 0.5 * (translation + translation.T)
    eigenvalues = torch.linalg.eigvalsh(translation).clamp_min(1e-12)
    return (
        float(torch.linalg.norm(delta[:3]) * float(translation_scale_m)),
        float(eigenvalues.log().sum()),
    )


def choose_candidate_targets(
    *,
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    reprojection_errors: torch.Tensor,
    trajectory_support: torch.Tensor,
    bias_gain_m2: torch.Tensor,
    translation_logdet_gain: torch.Tensor,
    strict_counterfactual: torch.Tensor,
) -> dict[str, int]:
    count = len(candidate_ids)
    if not (
        len(scores)
        == len(reprojection_errors)
        == len(trajectory_support)
        == len(bias_gain_m2)
        == len(translation_logdet_gain)
        == len(strict_counterfactual)
        == count
    ):
        raise ValueError("counterfactual candidate evidence does not align")
    if count == 0:
        return {
            "score_best": -1,
            "reprojection_best": -1,
            "track_stable": -1,
            "counterfactual_pose_best": -1,
        }
    score_best = int(torch.argmax(scores))
    reprojection_best = int(torch.argmin(reprojection_errors))
    stable_order = torch.argsort(
        trajectory_support.float() * 1e4 - reprojection_errors,
        descending=True,
        stable=True,
    )
    strict = torch.where(strict_counterfactual)[0]
    pose_best = -1
    if len(strict):
        base = bias_gain_m2[strict].clamp_min(0)
        utility = base + 0.05 * translation_logdet_gain[strict].clamp_min(0)
        pose_best = int(strict[int(torch.argmax(utility))])
    return {
        "score_best": score_best,
        "reprojection_best": reprojection_best,
        "track_stable": int(stable_order[0]),
        "counterfactual_pose_best": pose_best,
    }


def build_counterfactual_positive_teacher(
    *,
    state: dict,
    selected_outcomes: dict,
    triage: dict,
    source_positive_teacher: dict,
    query_cache: dict,
    metric,
    mode_table: ModeTable,
    config: CounterfactualTeacherConfig,
    device: torch.device,
    progress=None,
) -> tuple[dict, dict]:
    """Select verified repair targets with a one-row pose-bias audit."""

    names = list(selected_outcomes["query_names"])
    if names != list(triage["query_names"]) or names != list(
        source_positive_teacher["query_names"]
    ):
        raise ValueError("counterfactual teacher query registries differ")
    xyz = torch.as_tensor(state["anchor_xyz"]).double()
    anchor_count = len(xyz)
    _, trajectory_support = build_anchor_cross_trajectory_support(
        source_positive_teacher, anchor_count
    )
    cache = query_cache.get("queries", query_cache)
    totals = Counter()
    audit_records = []
    teacher_records = []

    for query_index, name in enumerate(names):
        selected_record = selected_outcomes["records"][query_index]
        triage_record = triage["records"][query_index]
        source_record = source_positive_teacher["records"][query_index]
        rows = torch.as_tensor(selected_record["query_rows"]).long()
        selected = torch.as_tensor(selected_record["selected_row_mask"]).bool()
        top1 = torch.as_tensor(
            selected_record["topk_anchor_indices"]
        ).long()[:, 0]
        selected_positions = torch.where(selected)[0]
        selected_lookup = torch.full((len(rows),), -1, dtype=torch.long)
        selected_lookup[selected_positions] = torch.arange(len(selected_positions))
        cached = cache[name]
        all_keypoints = (
            torch.as_tensor(cached["native_keypoints"]).double()[rows]
            + float(cached.get("pixel_center_offset", 0.5))
        )
        K = torch.as_tensor(cached["native_K"]).double()
        pose = torch.as_tensor(cached["pose_w2c"]).double()
        base_info, base_gradient = _pose_terms(
            all_keypoints[selected_positions],
            xyz[top1[selected_positions]],
            K,
            pose,
            config,
        )
        H = torch.eye(6, dtype=torch.float64) * 1e-4 + base_info.sum(0)
        g = base_gradient.sum(0)
        base_bias, base_logdet = _pose_metrics(
            H, g, config.translation_scale_m
        )
        raw_query = F.normalize(
            torch.as_tensor(cached["native_descriptors"]).float()[rows].to(device),
            dim=1,
        )
        with torch.no_grad():
            query, _ = metric(raw_query)
            query = F.normalize(query, dim=1)

        candidate_offsets = torch.as_tensor(
            triage_record["active_positive_offsets"]
        ).long()
        candidate_ids_packed = torch.as_tensor(
            triage_record["active_positive_indices"]
        ).long()
        candidate_errors_packed = torch.as_tensor(
            triage_record["active_positive_reprojection_errors_px"]
        ).float()
        harmful_positions = torch.as_tensor(
            triage_record["selected_row_positions"]
        ).long()
        category = torch.as_tensor(triage_record["category"]).long()
        repairable = (category == 0) | (category == 1)
        output_candidate_ids = []
        output_scores = []
        output_errors = []
        output_support = []
        output_bias_gain = []
        output_logdet_gain = []
        output_strict = []
        choice_values = {
            name: [] for name in (
                "score_best",
                "reprojection_best",
                "track_stable",
                "counterfactual_pose_best",
            )
        }
        routes = []
        target_by_row = {}
        for local, position in enumerate(harmful_positions.tolist()):
            start = int(candidate_offsets[local])
            end = int(candidate_offsets[local + 1])
            candidate_ids = candidate_ids_packed[start:end]
            candidate_errors = candidate_errors_packed[start:end]
            if not bool(repairable[local]) or not len(candidate_ids):
                output_candidate_ids.append(torch.empty(0, dtype=torch.long))
                output_scores.append(torch.empty(0))
                output_errors.append(torch.empty(0))
                output_support.append(torch.empty(0, dtype=torch.long))
                output_bias_gain.append(torch.empty(0))
                output_logdet_gain.append(torch.empty(0))
                output_strict.append(torch.empty(0, dtype=torch.bool))
                for values in choice_values.values():
                    values.append(-1)
                routes.append(ROUTE_REJECT)
                totals[ROUTE_NAMES[ROUTE_REJECT]] += 1
                continue
            repeated = query[position : position + 1].expand(
                len(candidate_ids), -1
            )
            _, candidate_scores = winning_representation(
                repeated, candidate_ids, mode_table
            )
            candidate_info, candidate_gradient = _pose_terms(
                all_keypoints[position : position + 1].expand(
                    len(candidate_ids), -1
                ),
                xyz[candidate_ids],
                K,
                pose,
                config,
            )
            selected_local = int(selected_lookup[position])
            if selected_local < 0:
                raise ValueError("triage repair row is not in the selected set")
            bias_gains = []
            logdet_gains = []
            strict_values = []
            for candidate_index in range(len(candidate_ids)):
                candidate_H = (
                    H
                    - base_info[selected_local]
                    + candidate_info[candidate_index]
                )
                candidate_g = (
                    g
                    - base_gradient[selected_local]
                    + candidate_gradient[candidate_index]
                )
                bias, logdet = _pose_metrics(
                    candidate_H,
                    candidate_g,
                    config.translation_scale_m,
                )
                bias_gain = base_bias**2 - bias**2
                logdet_gain = logdet - base_logdet
                bias_gains.append(bias_gain)
                logdet_gains.append(logdet_gain)
                strict_values.append(
                    bias_gain > 0
                    and logdet_gain
                    >= float(config.minimum_translation_logdet_gain)
                )
            bias_gains = torch.as_tensor(bias_gains).float()
            logdet_gains = torch.as_tensor(logdet_gains).float()
            strict_values = torch.as_tensor(strict_values).bool()
            support = trajectory_support[candidate_ids]
            choices = choose_candidate_targets(
                candidate_ids=candidate_ids,
                scores=candidate_scores,
                reprojection_errors=candidate_errors,
                trajectory_support=support,
                bias_gain_m2=bias_gains,
                translation_logdet_gain=logdet_gains,
                strict_counterfactual=strict_values,
            )
            for choice_name, candidate_index in choices.items():
                choice_values[choice_name].append(
                    int(candidate_ids[candidate_index])
                    if candidate_index >= 0
                    else -1
                )
                if candidate_index >= 0:
                    totals[f"{choice_name}_rows"] += 1
                    totals[f"{choice_name}_score_sum"] += float(
                        candidate_scores[candidate_index]
                    )
                    totals[f"{choice_name}_reprojection_sum"] += float(
                        candidate_errors[candidate_index]
                    )
                    totals[f"{choice_name}_support_sum"] += int(
                        support[candidate_index]
                    )
                    totals[f"{choice_name}_strict_rows"] += int(
                        strict_values[candidate_index]
                    )
            valid_choices = [
                value for value in choices.values() if int(value) >= 0
            ]
            if valid_choices and len(set(valid_choices)) == 1:
                totals["all_target_variants_agree"] += 1
            if (
                choices["counterfactual_pose_best"] >= 0
                and choices["counterfactual_pose_best"]
                != choices["score_best"]
            ):
                totals["pose_target_differs_from_score_best"] += 1
            pose_index = choices["counterfactual_pose_best"]
            route = ROUTE_REJECT
            if pose_index >= 0:
                support_count = int(support[pose_index])
                if support_count >= int(
                    config.minimum_primary_trajectory_support
                ):
                    route = ROUTE_PRIMARY
                elif support_count >= int(
                    config.minimum_family_trajectory_support
                ):
                    route = ROUTE_FAMILY
                if route != ROUTE_REJECT:
                    target_by_row[int(rows[position])] = int(
                        candidate_ids[pose_index]
                    )
            routes.append(route)
            totals[ROUTE_NAMES[route]] += 1
            output_candidate_ids.append(candidate_ids)
            output_scores.append(candidate_scores)
            output_errors.append(candidate_errors)
            output_support.append(support)
            output_bias_gain.append(bias_gains)
            output_logdet_gain.append(logdet_gains)
            output_strict.append(strict_values)
        packed_offsets, packed_ids = _pack(output_candidate_ids, torch.long)
        _, packed_scores = _pack(output_scores, torch.float32)
        _, packed_errors = _pack(output_errors, torch.float32)
        _, packed_support = _pack(output_support, torch.long)
        _, packed_bias = _pack(output_bias_gain, torch.float32)
        _, packed_logdet = _pack(output_logdet_gain, torch.float32)
        _, packed_strict = _pack(output_strict, torch.bool)
        audit_records.append(
            {
                "query_index": query_index,
                "query_name": name,
                "query_rows": torch.as_tensor(
                    triage_record["query_rows"]
                ).long(),
                "candidate_offsets": packed_offsets,
                "candidate_anchor_indices": packed_ids,
                "candidate_scores": packed_scores,
                "candidate_reprojection_errors_px": packed_errors,
                "candidate_trajectory_support": packed_support,
                "candidate_bias_gain_m2": packed_bias,
                "candidate_translation_logdet_gain": packed_logdet,
                "candidate_strict_counterfactual": packed_strict,
                **{
                    key: torch.as_tensor(value, dtype=torch.long)
                    for key, value in choice_values.items()
                },
                "route": torch.as_tensor(routes, dtype=torch.int8),
                "base_bias_m": base_bias,
                "base_translation_logdet": base_logdet,
            }
        )
        source_lookup = _positive_lookup(source_record)
        positive_values = []
        for row in rows.tolist():
            if int(row) in set(
                torch.as_tensor(triage_record["query_rows"]).long().tolist()
            ):
                target = target_by_row.get(int(row), -1)
                positive_values.append(
                    torch.tensor([target], dtype=torch.long)
                    if target >= 0
                    else torch.empty(0, dtype=torch.long)
                )
            else:
                positive_values.append(
                    source_lookup.get(int(row), torch.empty(0, dtype=torch.long))
                )
        positive_offsets, positive_indices = _pack(
            positive_values, torch.long
        )
        teacher_records.append(
            {
                **source_record,
                "positive_offsets": positive_offsets,
                "positive_indices": positive_indices,
            }
        )
        totals["repairable_rows"] += int(repairable.sum())
        totals["strict_candidate_rows"] += int(
            sum(bool(torch.as_tensor(value).any()) for value in output_strict)
        )
        if progress is not None:
            progress(query_index + 1, len(names), dict(totals))

    teacher_positive_rows = sum(
        int(
            (
                torch.as_tensor(record["positive_offsets"])[1:]
                - torch.as_tensor(record["positive_offsets"])[:-1]
                > 0
            ).sum()
        )
        for record in teacher_records
    )
    teacher_positive_pairs = sum(
        len(record["positive_indices"]) for record in teacher_records
    )
    audit = {
        "schema": "lafgs_counterfactual_positive_audit",
        "version": 1,
        "query_names": names,
        "anchor_count": anchor_count,
        "route_names": list(ROUTE_NAMES),
        "records": audit_records,
        "summary": {
            "repairable_rows": int(totals["repairable_rows"]),
            "strict_counterfactual_rows": int(totals["strict_candidate_rows"]),
            "route_counts": {
                name: int(totals[name]) for name in ROUTE_NAMES
            },
            "target_variant_summary": {
                variant: {
                    "rows": int(totals[f"{variant}_rows"]),
                    "mean_score": (
                        float(totals[f"{variant}_score_sum"])
                        / max(int(totals[f"{variant}_rows"]), 1)
                    ),
                    "mean_reprojection_error_px": (
                        float(totals[f"{variant}_reprojection_sum"])
                        / max(int(totals[f"{variant}_rows"]), 1)
                    ),
                    "mean_trajectory_support": (
                        float(totals[f"{variant}_support_sum"])
                        / max(int(totals[f"{variant}_rows"]), 1)
                    ),
                    "strict_counterfactual_rate": (
                        int(totals[f"{variant}_strict_rows"])
                        / max(int(totals[f"{variant}_rows"]), 1)
                    ),
                }
                for variant in (
                    "score_best",
                    "reprojection_best",
                    "track_stable",
                    "counterfactual_pose_best",
                )
            },
            "all_target_variants_agree": int(
                totals["all_target_variants_agree"]
            ),
            "pose_target_differs_from_score_best": int(
                totals["pose_target_differs_from_score_best"]
            ),
        },
        "config": asdict(config),
    }
    teacher = {
        **source_positive_teacher,
        "schema": "lafgs_counterfactual_routed_positive_teacher",
        "version": 1,
        "records": teacher_records,
        "diagnostics": {
            **dict(source_positive_teacher.get("diagnostics", {})),
            "positive_rows": int(teacher_positive_rows),
            "strong_pair_count": int(teacher_positive_pairs),
            "counterfactual_routing": audit["summary"],
            "source_diagnostics": dict(
                source_positive_teacher.get("diagnostics", {})
            ),
        },
        "counterfactual_audit_summary": audit["summary"],
        "config": {
            **dict(source_positive_teacher.get("config", {})),
            "counterfactual_positive_teacher": asdict(config),
        },
    }
    return audit, teacher
