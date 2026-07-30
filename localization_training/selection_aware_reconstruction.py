"""Local descriptor reconstruction from pose-sufficient selection outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from localization_training.pose_sufficient_selector import (
    basis_aware_core_reserve_mask,
    image_grid_cells,
)


ROLE_NEUTRAL = 0
ROLE_HARMFUL = 1
ROLE_CRITICAL = 2


@dataclass(frozen=True)
class SelectionAwareTrainingData:
    query_features: torch.Tensor
    positive_representation: torch.Tensor
    negative_representation: torch.Tensor
    role: torch.Tensor
    weight: torch.Tensor
    baseline_margin: torch.Tensor
    diagnostics: dict[str, float]
    positive_trainable: torch.Tensor | None = None
    negative_trainable: torch.Tensor | None = None

    def validate(self, representation_count: int) -> None:
        count = len(self.query_features)
        aligned = (
            self.positive_representation,
            self.negative_representation,
            self.role,
            self.weight,
            self.baseline_margin,
        )
        if self.query_features.ndim != 2 or any(
            len(value) != count for value in aligned
        ):
            raise ValueError("selection-aware training tensors do not align")
        if count and (
            int(self.positive_representation.min()) < 0
            or int(self.negative_representation.min()) < 0
            or int(self.positive_representation.max()) >= representation_count
            or int(self.negative_representation.max()) >= representation_count
        ):
            raise ValueError("selection-aware representation index is invalid")
        if not torch.isfinite(self.query_features.float()).all():
            raise ValueError("selection-aware query features must be finite")
        if not torch.isfinite(self.weight).all() or bool(
            (self.weight <= 0).any()
        ):
            raise ValueError("selection-aware weights must be positive")
        for trainable in (
            self.positive_trainable,
            self.negative_trainable,
        ):
            if trainable is not None and len(trainable) != count:
                raise ValueError(
                    "selection-aware trainable masks must align"
                )


@dataclass(frozen=True)
class ModeTable:
    features: torch.Tensor
    bias: torch.Tensor
    temperature: torch.Tensor
    representation_ids: torch.Tensor
    valid: torch.Tensor
    anchor_count: int
    prototype_count: int


@dataclass(frozen=True)
class SelectionAwareOptimizationConfig:
    steps: int = 150
    batch_size: int = 4096
    learning_rate: float = 0.02
    maximum_descriptor_delta: float = 0.03
    maximum_bias_delta: float = 0.02
    ranking_margin: float = 0.02
    preserve_tolerance: float = 0.01
    ranking_temperature: float = 0.05
    descriptor_replay_weight: float = 0.001
    bias_replay_weight: float = 0.02
    seed: int = 2026


def build_mode_table(
    anchor_features: torch.Tensor,
    prototype_features: torch.Tensor,
    prototype_parents: torch.Tensor,
    prototype_bias: torch.Tensor,
    prototype_temperature: torch.Tensor,
) -> ModeTable:
    """Pack primary and family modes into a per-anchor padded table."""

    anchors = F.normalize(torch.as_tensor(anchor_features).float(), dim=1)
    prototypes = F.normalize(
        torch.as_tensor(prototype_features).float(), dim=1
    )
    parents = torch.as_tensor(prototype_parents).long().reshape(-1)
    bias = torch.as_tensor(prototype_bias).float().reshape(-1)
    temperature = torch.as_tensor(
        prototype_temperature
    ).float().reshape(-1)
    if (
        prototypes.ndim != 2
        or prototypes.shape[1] != anchors.shape[1]
        or len(prototypes) != len(parents)
        or len(prototypes) != len(bias)
        or len(prototypes) != len(temperature)
    ):
        raise ValueError("family mode tensors do not align")
    if len(parents) and (
        int(parents.min()) < 0 or int(parents.max()) >= len(anchors)
    ):
        raise ValueError("family mode parent is outside the anchor map")
    if bool((bias > 1e-8).any()) or bool((temperature <= 0).any()):
        raise ValueError("family mode calibration is invalid")

    counts = torch.bincount(parents, minlength=len(anchors))
    width = int(counts.max().item()) + 1 if len(counts) else 1
    features = anchors.new_zeros((len(anchors), width, anchors.shape[1]))
    table_bias = anchors.new_full((len(anchors), width), float("-inf"))
    table_temperature = anchors.new_ones((len(anchors), width))
    representation_ids = torch.full(
        (len(anchors), width), -1, dtype=torch.long
    )
    valid = torch.zeros((len(anchors), width), dtype=torch.bool)
    features[:, 0] = anchors
    table_bias[:, 0] = 0.0
    representation_ids[:, 0] = torch.arange(len(anchors))
    valid[:, 0] = True
    offsets = torch.zeros(len(anchors), dtype=torch.long)
    for prototype_index, parent in enumerate(parents.tolist()):
        column = int(offsets[parent]) + 1
        offsets[parent] += 1
        features[parent, column] = prototypes[prototype_index]
        table_bias[parent, column] = bias[prototype_index]
        table_temperature[parent, column] = temperature[prototype_index]
        representation_ids[parent, column] = len(anchors) + prototype_index
        valid[parent, column] = True
    return ModeTable(
        features=features,
        bias=table_bias,
        temperature=table_temperature,
        representation_ids=representation_ids,
        valid=valid,
        anchor_count=len(anchors),
        prototype_count=len(prototypes),
    )


def winning_representation(
    query_features: torch.Tensor,
    anchor_indices: torch.Tensor,
    table: ModeTable,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact primary/family mode that scores each anchor."""

    query = F.normalize(torch.as_tensor(query_features).float(), dim=1)
    anchors = torch.as_tensor(anchor_indices).long().reshape(-1)
    if len(query) != len(anchors):
        raise ValueError("query and anchor rows must align")
    if len(anchors) and (
        int(anchors.min()) < 0 or int(anchors.max()) >= table.anchor_count
    ):
        raise ValueError("winner anchor is outside the mode table")
    features = table.features[anchors].to(query.device)
    bias = table.bias[anchors].to(query.device)
    temperature = table.temperature[anchors].to(query.device)
    valid = table.valid[anchors].to(query.device)
    scores = (
        torch.einsum("bd,bmd->bm", query, features)
        / temperature
        + bias
    ).masked_fill(~valid, float("-inf"))
    best_score, column = scores.max(dim=1)
    representation = torch.gather(
        table.representation_ids[anchors].to(query.device),
        1,
        column[:, None],
    ).reshape(-1)
    return representation.cpu(), best_score.cpu()


def selection_role_masks(
    *,
    selected: torch.Tensor,
    strict_clean: torch.Tensor,
    solver_clean: torch.Tensor,
    harmful: torch.Tensor,
    reserve_gain: torch.Tensor,
    maximum_critical: int,
) -> dict[str, torch.Tensor]:
    """Split one query into protected, harmful, and basis-critical rows."""

    selected = torch.as_tensor(selected).bool().reshape(-1)
    strict_clean = torch.as_tensor(strict_clean).bool().reshape(-1)
    solver_clean = torch.as_tensor(solver_clean).bool().reshape(-1)
    harmful = torch.as_tensor(harmful).bool().reshape(-1)
    gain = torch.as_tensor(reserve_gain).float().reshape(-1)
    if not all(
        len(value) == len(selected)
        for value in (strict_clean, solver_clean, harmful, gain)
    ):
        raise ValueError("selection role masks must align")
    selected_harmful = selected & harmful
    protected = selected & (strict_clean | solver_clean) & ~selected_harmful
    critical_candidates = (
        ~selected
        & (strict_clean | solver_clean)
        & torch.isfinite(gain)
        & (gain > 0)
    )
    critical = torch.zeros_like(selected)
    candidates = torch.where(critical_candidates)[0]
    if len(candidates):
        order = candidates[
            torch.argsort(gain[candidates], descending=True, stable=True)
        ]
        critical[order[: max(int(maximum_critical), 0)]] = True
    return {
        "protected": protected,
        "harmful": selected_harmful,
        "critical": critical,
    }


def bounded_representations(
    base: torch.Tensor,
    raw_delta: torch.Tensor,
    *,
    maximum_delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a bounded residual and return normalized descriptors."""

    base = F.normalize(torch.as_tensor(base).float(), dim=1)
    raw_delta = torch.as_tensor(raw_delta).float()
    if raw_delta.shape != base.shape:
        raise ValueError("representation residual must align with its base")
    raw = raw_delta
    maximum = max(float(maximum_delta), 0.0)
    norm = torch.linalg.norm(raw, dim=1, keepdim=True)
    scale = maximum * torch.tanh(norm) / norm.clamp_min(1e-8)
    scale = torch.where(
        norm > 0,
        scale,
        torch.full_like(scale, maximum),
    )
    bounded = raw * scale
    return F.normalize(base + bounded, dim=1), bounded


def selection_aware_ranking_loss(
    positive_score: torch.Tensor,
    negative_score: torch.Tensor,
    *,
    role: torch.Tensor,
    baseline_margin: torch.Tensor,
    weight: torch.Tensor,
    margin: float,
    preserve_tolerance: float,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine protected replay with harmful/critical ranking correction."""

    positive = torch.as_tensor(positive_score).float().reshape(-1)
    negative = torch.as_tensor(negative_score).float().reshape(-1)
    role = torch.as_tensor(role).long().reshape(-1)
    baseline = torch.as_tensor(baseline_margin).float().reshape(-1)
    weight = torch.as_tensor(weight).float().reshape(-1)
    if not all(
        len(value) == len(positive)
        for value in (negative, role, baseline, weight)
    ):
        raise ValueError("selection-aware ranking inputs must align")
    current_margin = positive - negative
    neutral = role == ROLE_NEUTRAL
    active = ~neutral
    preserve_loss = torch.where(
        neutral,
        F.relu(baseline - float(preserve_tolerance) - current_margin),
        torch.zeros_like(current_margin),
    )
    scale = max(float(temperature), 1e-6)
    active_loss = torch.where(
        active,
        F.softplus(
            (float(margin) - current_margin) / scale
        )
        * scale,
        torch.zeros_like(current_margin),
    )
    total = ((preserve_loss + active_loss) * weight).sum() / weight.sum().clamp_min(
        1e-8
    )
    return total, {
        "preserve": preserve_loss[neutral].mean()
        if bool(neutral.any())
        else total.new_zeros(()),
        "active": active_loss[active].mean()
        if bool(active.any())
        else total.new_zeros(()),
        "margin": current_margin.mean(),
    }


def recompute_basis_roles(
    *,
    selected_record: dict,
    dynamic_record: dict,
    keypoints: torch.Tensor,
    image_hw: tuple[int, int] | list[int],
    source_groups: torch.Tensor,
    dependency_groups: torch.Tensor,
    xyz: torch.Tensor,
    selector_config: dict,
    maximum_critical: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Recompute and verify the exact OOF selector before making teachers."""

    anchors = torch.as_tensor(
        selected_record["topk_anchor_indices"]
    ).long()[:, 0]
    points = torch.as_tensor(keypoints).float()
    selected, _, details = basis_aware_core_reserve_mask(
        selected_record["strict_probability"],
        selected_record["solver_probability"],
        selected_record["harmful_probability"],
        image_points=points,
        image_hw=image_hw,
        image_cells=image_grid_cells(points, image_hw),
        dependency_groups=torch.as_tensor(dependency_groups).long()[anchors],
        source_groups=torch.as_tensor(source_groups).long()[anchors],
        xyz=torch.as_tensor(xyz).float()[anchors],
        return_details=True,
        **selector_config,
    )
    expected = torch.as_tensor(
        selected_record["selected_row_mask"]
    ).bool()
    if not torch.equal(selected, expected):
        raise ValueError("selection-aware teacher does not reproduce OOF selection")
    strict = (
        torch.as_tensor(
            dynamic_record["gt_reprojection_errors_px"]
        ).float()
        <= 2.0
    )
    roles = selection_role_masks(
        selected=selected,
        strict_clean=strict,
        solver_clean=dynamic_record["clean_inlier_mask"],
        harmful=dynamic_record["harmful_inlier_mask"],
        reserve_gain=details["reserve_gain"],
        maximum_critical=maximum_critical,
    )
    return roles, details


def _take_highest(
    mask: torch.Tensor,
    score: torch.Tensor,
    maximum: int,
) -> torch.Tensor:
    indices = torch.where(torch.as_tensor(mask).bool())[0]
    maximum = max(int(maximum), 0)
    if not len(indices) or maximum == 0:
        return indices[:0]
    order = torch.argsort(
        torch.as_tensor(score).float()[indices],
        descending=True,
        stable=True,
    )
    return indices[order[:maximum]]


def _take_evenly(mask: torch.Tensor, maximum: int) -> torch.Tensor:
    indices = torch.where(torch.as_tensor(mask).bool())[0]
    maximum = max(int(maximum), 0)
    if len(indices) <= maximum:
        return indices
    positions = torch.linspace(
        0, len(indices) - 1, steps=maximum
    ).round().long()
    return indices[positions]


def _positive_anchor_lookup(record: dict) -> dict[int, torch.Tensor]:
    rows = torch.as_tensor(record["query_rows"]).long().reshape(-1)
    offsets = torch.as_tensor(record["positive_offsets"]).long().reshape(-1)
    indices = torch.as_tensor(record["positive_indices"]).long().reshape(-1)
    if len(offsets) != len(rows) + 1 or int(offsets[-1]) != len(indices):
        raise ValueError("complete-positive CSR is malformed")
    return {
        int(row): indices[int(offsets[index]) : int(offsets[index + 1])]
        for index, row in enumerate(rows.tolist())
    }


@torch.no_grad()
def build_selection_aware_training_data(
    *,
    state: dict,
    family: dict,
    selected_outcomes: dict,
    dynamic_outcomes: dict,
    complete_positive_teacher: dict,
    cache: dict,
    metric,
    selector_config: dict,
    device: torch.device,
    maximum_protected_per_query: int = 96,
    maximum_neutral_per_query: int = 64,
    maximum_harmful_per_query: int = 64,
    maximum_critical_per_query: int = 32,
    counterfactual_audit: dict | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> SelectionAwareTrainingData:
    """Build detached local reconstruction evidence from OOF selections."""

    names = list(selected_outcomes["query_names"])
    if names != list(dynamic_outcomes["query_names"]) or names != list(
        complete_positive_teacher["query_names"]
    ):
        raise ValueError("selection-aware query registries differ")
    if int(selected_outcomes["anchor_count"]) != len(state["anchor_ids"]):
        raise ValueError("selection-aware outcomes do not align with the map")
    if int(complete_positive_teacher["anchor_count"]) != len(
        state["anchor_ids"]
    ):
        raise ValueError("selection-aware positives do not align with the map")
    if counterfactual_audit is not None and names != list(
        counterfactual_audit["query_names"]
    ):
        raise ValueError("selection-aware counterfactual audit differs")
    family_indices = torch.as_tensor(family["landmark_indices"]).long()
    if not torch.equal(
        family_indices.reshape(-1),
        torch.as_tensor(state["anchor_ids"]).long().reshape(-1),
    ):
        raise ValueError("selection-aware family does not align with the map")
    missing_cache = [name for name in names if name not in cache]
    if missing_cache:
        raise ValueError(f"selection-aware cache misses {missing_cache[:3]}")

    table = build_mode_table(
        state["anchor_features"],
        family["prototype_features"],
        family["prototype_anchor_indices"],
        family.get(
            "prototype_bias",
            torch.zeros(len(family["prototype_features"])),
        ),
        family.get(
            "prototype_temperature",
            torch.ones(len(family["prototype_features"])),
        ),
    )
    table = ModeTable(
        features=table.features.to(device),
        bias=table.bias.to(device),
        temperature=table.temperature.to(device),
        representation_ids=table.representation_ids.to(device),
        valid=table.valid.to(device),
        anchor_count=table.anchor_count,
        prototype_count=table.prototype_count,
    )
    flat_features = torch.cat(
        (
            F.normalize(torch.as_tensor(state["anchor_features"]).float(), dim=1),
            F.normalize(
                torch.as_tensor(family["prototype_features"]).float(), dim=1
            ),
        )
    ).to(device)
    flat_bias = torch.cat(
        (
            torch.zeros(table.anchor_count),
            torch.as_tensor(
                family.get(
                    "prototype_bias",
                    torch.zeros(table.prototype_count),
                )
            ).float(),
        )
    ).to(device)
    flat_temperature = torch.cat(
        (
            torch.ones(table.anchor_count),
            torch.as_tensor(
                family.get(
                    "prototype_temperature",
                    torch.ones(table.prototype_count),
                )
            ).float(),
        )
    ).to(device)
    source = torch.as_tensor(state["source_primitive_ids"]).long()
    dependency = torch.as_tensor(
        state.get("coarse_dependency_group_ids", state["dependency_group_ids"])
    ).long()
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    query_blocks = []
    positive_blocks = []
    negative_blocks = []
    role_blocks = []
    weight_blocks = []
    margin_blocks = []
    positive_trainable_blocks = []
    negative_trainable_blocks = []
    role_counts = {
        "neutral": 0,
        "protected": 0,
        "harmful": 0,
        "critical": 0,
        "harmful_without_positive": 0,
        "counterfactual_family_without_prototype": 0,
        "strict_basin_protected": 0,
    }
    active_representations: set[int] = set()

    for query_index, name in enumerate(names):
        selected_record = selected_outcomes["records"][query_index]
        dynamic_record = dynamic_outcomes["records"][query_index]
        positive_record = complete_positive_teacher["records"][query_index]
        counterfactual_record = (
            counterfactual_audit["records"][query_index]
            if counterfactual_audit is not None
            else None
        )
        if (
            selected_record["query_name"] != name
            or dynamic_record["query_name"] != name
            or positive_record["query_name"] != name
        ):
            raise ValueError("selection-aware record order differs")
        rows = torch.as_tensor(selected_record["query_rows"]).long()
        dynamic_rows = torch.as_tensor(dynamic_record["query_rows"]).long()
        if not torch.equal(rows, dynamic_rows):
            raise ValueError("selection-aware row order differs")
        cached = cache[name]
        keypoints = torch.as_tensor(
            cached["native_keypoints"]
        ).float()[rows]
        roles, details = recompute_basis_roles(
            selected_record=selected_record,
            dynamic_record=dynamic_record,
            keypoints=keypoints,
            image_hw=cached["native_input_hw"],
            source_groups=source,
            dependency_groups=dependency,
            xyz=xyz,
            selector_config=selector_config,
            maximum_critical=maximum_critical_per_query,
        )
        raw_query = F.normalize(
            torch.as_tensor(
                cached["native_descriptors"]
            ).float()[rows].to(device),
            dim=1,
        )
        query, _ = metric(raw_query)
        query = F.normalize(query, dim=1)
        anchors = torch.as_tensor(
            selected_record["topk_anchor_indices"]
        ).long()
        top1_representation, top1_score = winning_representation(
            query, anchors[:, 0], table
        )
        top2_representation, top2_score = winning_representation(
            query, anchors[:, 1], table
        )

        protected = _take_highest(
            roles["protected"],
            selected_record["strict_probability"],
            maximum_protected_per_query,
        )
        excluded = roles["harmful"] | roles["critical"] | roles["protected"]
        neutral = _take_evenly(
            ~excluded, maximum_neutral_per_query
        )
        replay = torch.cat((protected, neutral))
        if len(replay):
            query_blocks.append(query[replay].half().cpu())
            positive_blocks.append(top1_representation[replay])
            negative_blocks.append(top2_representation[replay])
            role_blocks.append(
                torch.full((len(replay),), ROLE_NEUTRAL, dtype=torch.long)
            )
            replay_weight = torch.cat(
                (
                    torch.full((len(protected),), 2.0),
                    torch.full((len(neutral),), 0.5),
                )
            )
            weight_blocks.append(replay_weight)
            margin_blocks.append(
                top1_score[replay] - top2_score[replay]
            )
            positive_trainable_blocks.append(
                torch.ones(len(replay), dtype=torch.bool)
            )
            negative_trainable_blocks.append(
                torch.ones(len(replay), dtype=torch.bool)
            )
            if (
                counterfactual_record is not None
                and 3.0 <= float(dynamic_record["te_cm"]) <= 8.0
                and len(protected)
            ):
                replay_weight[: len(protected)] = 4.0
                role_counts["strict_basin_protected"] += len(protected)
            role_counts["protected"] += len(protected)
            role_counts["neutral"] += len(neutral)
            active_representations.update(
                top1_representation[replay].tolist()
            )
            active_representations.update(
                top2_representation[replay].tolist()
            )

        critical = _take_highest(
            roles["critical"],
            details["reserve_gain"],
            maximum_critical_per_query,
        )
        if len(critical):
            gain = details["reserve_gain"][critical]
            gain_weight = gain / gain.mean().clamp_min(1e-8)
            gain_weight = gain_weight.clamp(0.5, 2.0)
            query_blocks.append(query[critical].half().cpu())
            positive_blocks.append(top1_representation[critical])
            negative_blocks.append(top2_representation[critical])
            role_blocks.append(
                torch.full((len(critical),), ROLE_CRITICAL, dtype=torch.long)
            )
            weight_blocks.append(gain_weight.cpu())
            margin_blocks.append(
                top1_score[critical] - top2_score[critical]
            )
            positive_trainable_blocks.append(
                torch.ones(len(critical), dtype=torch.bool)
            )
            negative_trainable_blocks.append(
                torch.ones(len(critical), dtype=torch.bool)
            )
            role_counts["critical"] += len(critical)
            active_representations.update(
                top1_representation[critical].tolist()
            )
            active_representations.update(
                top2_representation[critical].tolist()
            )

        harmful = _take_highest(
            roles["harmful"],
            selected_record["harmful_probability"],
            maximum_harmful_per_query,
        )
        positives = _positive_anchor_lookup(positive_record)
        route_by_row = {}
        target_representation_by_row = {}
        if counterfactual_record is not None:
            route_by_row = {
                int(row): int(route)
                for row, route in zip(
                    torch.as_tensor(
                        counterfactual_record["query_rows"]
                    ).long().tolist(),
                    torch.as_tensor(
                        counterfactual_record["route"]
                    ).long().tolist(),
                )
            }
            if "target_representation" in counterfactual_record:
                target_representation_by_row = {
                    int(row): int(representation)
                    for row, representation in zip(
                        torch.as_tensor(
                            counterfactual_record["query_rows"]
                        ).long().tolist(),
                        torch.as_tensor(
                            counterfactual_record[
                                "target_representation"
                            ]
                        ).long().tolist(),
                    )
                }
        for row_index in harmful.tolist():
            legal = positives.get(int(rows[row_index]))
            if legal is None or not len(legal):
                role_counts["harmful_without_positive"] += 1
                continue
            legal = legal[legal != anchors[row_index, 0]]
            if not len(legal):
                role_counts["harmful_without_positive"] += 1
                continue
            repeated_query = query[row_index : row_index + 1].expand(
                len(legal), -1
            )
            legal_representation, legal_score = winning_representation(
                repeated_query, legal, table
            )
            best = int(torch.argmax(legal_score))
            positive_representation = legal_representation[best]
            route = route_by_row.get(int(rows[row_index]), -1)
            if counterfactual_record is not None:
                routed_representation = target_representation_by_row.get(
                    int(rows[row_index]), -1
                )
                if routed_representation >= 0:
                    positive_representation = torch.tensor(
                        routed_representation, dtype=torch.long
                    )
                    legal_score = (
                        (
                            query[row_index]
                            * flat_features[routed_representation]
                        ).sum()
                        / flat_temperature[routed_representation]
                        + flat_bias[routed_representation]
                    ).reshape(1).cpu()
                    best = 0
                elif route == 0:
                    positive_representation = legal_representation[best]
                elif route == 1:
                    if int(positive_representation) < table.anchor_count:
                        role_counts[
                            "counterfactual_family_without_prototype"
                        ] += 1
                        continue
                else:
                    role_counts["harmful_without_positive"] += 1
                    continue
            negative_representation = top1_representation[row_index]
            query_blocks.append(query[row_index : row_index + 1].half().cpu())
            positive_blocks.append(positive_representation.reshape(1))
            negative_blocks.append(negative_representation.reshape(1))
            role_blocks.append(torch.tensor([ROLE_HARMFUL]))
            weight_blocks.append(
                torch.tensor(
                    [
                        1.0
                        + float(
                            selected_record["harmful_probability"][row_index]
                        )
                    ]
                )
            )
            margin_blocks.append(
                (
                    legal_score[best]
                    - top1_score[row_index]
                ).reshape(1)
            )
            positive_trainable_blocks.append(torch.ones(1, dtype=torch.bool))
            negative_trainable_blocks.append(
                torch.tensor(
                    [counterfactual_record is None], dtype=torch.bool
                )
            )
            role_counts["harmful"] += 1
            active_representations.add(int(positive_representation))
            if counterfactual_record is None:
                active_representations.add(int(negative_representation))
        if progress is not None:
            progress(query_index + 1, len(names))

    if not query_blocks:
        raise ValueError("selection-aware teacher produced no training pairs")
    data = SelectionAwareTrainingData(
        query_features=torch.cat(query_blocks).detach(),
        positive_representation=torch.cat(positive_blocks).long(),
        negative_representation=torch.cat(negative_blocks).long(),
        role=torch.cat(role_blocks).long(),
        weight=torch.cat(weight_blocks).float(),
        baseline_margin=torch.cat(margin_blocks).float(),
        diagnostics={
            **{key: float(value) for key, value in role_counts.items()},
            "pair_count": float(sum(len(value) for value in query_blocks)),
            "active_representation_count": float(len(active_representations)),
            "active_anchor_representation_count": float(
                sum(value < table.anchor_count for value in active_representations)
            ),
            "active_prototype_representation_count": float(
                sum(value >= table.anchor_count for value in active_representations)
            ),
        },
        positive_trainable=torch.cat(positive_trainable_blocks).bool(),
        negative_trainable=torch.cat(negative_trainable_blocks).bool(),
    )
    data.validate(table.anchor_count + table.prototype_count)
    return data


def optimize_selection_aware_representations(
    *,
    state: dict,
    family: dict,
    data: SelectionAwareTrainingData,
    config: SelectionAwareOptimizationConfig,
    device: torch.device,
    progress: Callable[[dict[str, float]], None] | None = None,
) -> tuple[dict, dict, list[dict[str, float]]]:
    """Run one bounded local reconstruction macro-round."""

    base_anchor = F.normalize(
        torch.as_tensor(state["anchor_features"]).detach().float(), dim=1
    ).detach()
    base_prototype = F.normalize(
        torch.as_tensor(family["prototype_features"]).detach().float(), dim=1
    ).detach()
    base = torch.cat((base_anchor, base_prototype)).to(device).detach()
    anchor_count = len(base_anchor)
    prototype_count = len(base_prototype)
    data.validate(len(base))
    base_bias = torch.as_tensor(
        family.get("prototype_bias", torch.zeros(prototype_count))
    ).detach().float().to(device)
    temperature = torch.cat(
        (
            torch.ones(anchor_count),
            torch.as_tensor(
                family.get(
                    "prototype_temperature", torch.ones(prototype_count)
                )
            ).float(),
        )
    ).to(device).detach()
    raw_delta = torch.nn.Parameter(torch.zeros_like(base))
    raw_bias_delta = torch.nn.Parameter(
        torch.zeros(prototype_count, device=device)
    )
    optimizer = torch.optim.Adam(
        (raw_delta, raw_bias_delta), lr=float(config.learning_rate)
    )
    positive_trainable = (
        torch.ones(len(data.query_features), dtype=torch.bool)
        if data.positive_trainable is None
        else torch.as_tensor(data.positive_trainable).bool()
    )
    negative_trainable = (
        torch.ones(len(data.query_features), dtype=torch.bool)
        if data.negative_trainable is None
        else torch.as_tensor(data.negative_trainable).bool()
    )
    active_ids = torch.unique(
        torch.cat(
            (
                data.positive_representation[positive_trainable],
                data.negative_representation[negative_trainable],
            )
        )
    ).to(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config.seed))
    history = []
    pair_count = len(data.query_features)
    batch_size = min(max(int(config.batch_size), 1), pair_count)

    for step in range(1, int(config.steps) + 1):
        indices = torch.randint(
            pair_count,
            (batch_size,),
            generator=generator,
        )
        query = data.query_features[indices].float().to(device)
        positive_ids = data.positive_representation[indices].to(device)
        negative_ids = data.negative_representation[indices].to(device)
        role = data.role[indices].to(device)
        weight = data.weight[indices].to(device)
        baseline_margin = data.baseline_margin[indices].to(device)
        positive_trainable_batch = positive_trainable[indices].to(device)
        negative_trainable_batch = negative_trainable[indices].to(device)
        transformed, bounded = bounded_representations(
            base,
            raw_delta,
            maximum_delta=float(config.maximum_descriptor_delta),
        )
        prototype_bias = torch.clamp(
            base_bias
            + float(config.maximum_bias_delta) * torch.tanh(raw_bias_delta),
            max=0.0,
        )
        representation_bias = torch.cat(
            (torch.zeros(anchor_count, device=device), prototype_bias)
        )

        def score(
            representation: torch.Tensor,
            trainable: torch.Tensor,
        ) -> torch.Tensor:
            feature = transformed[representation]
            feature = torch.where(
                trainable[:, None], feature, feature.detach()
            )
            bias = representation_bias[representation]
            bias = torch.where(trainable, bias, bias.detach())
            return (
                (
                    query
                    * feature
                ).sum(dim=1)
                / temperature[representation]
                + bias
            )

        positive_score = score(positive_ids, positive_trainable_batch)
        negative_score = score(negative_ids, negative_trainable_batch)
        ranking, parts = selection_aware_ranking_loss(
            positive_score,
            negative_score,
            role=role,
            baseline_margin=baseline_margin,
            weight=weight,
            margin=float(config.ranking_margin),
            preserve_tolerance=float(config.preserve_tolerance),
            temperature=float(config.ranking_temperature),
        )
        descriptor_replay = (
            torch.linalg.norm(bounded[active_ids], dim=1)
            / max(float(config.maximum_descriptor_delta), 1e-8)
        ).square().mean()
        bias_shift = prototype_bias - base_bias
        bias_replay = (
            bias_shift
            / max(float(config.maximum_bias_delta), 1e-8)
        ).square().mean() if prototype_count else ranking.new_zeros(())
        loss = (
            ranking
            + float(config.descriptor_replay_weight) * descriptor_replay
            + float(config.bias_replay_weight) * bias_replay
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        raw_delta.grad[
            torch.ones(len(base), dtype=torch.bool, device=device).scatter_(
                0, active_ids, False
            )
        ] = 0
        optimizer.step()
        if step == 1 or step % 10 == 0 or step == int(config.steps):
            record = {
                "step": float(step),
                "loss": float(loss.detach()),
                "ranking_loss": float(ranking.detach()),
                "preserve_loss": float(parts["preserve"].detach()),
                "active_loss": float(parts["active"].detach()),
                "batch_margin": float(parts["margin"].detach()),
                "descriptor_replay": float(descriptor_replay.detach()),
                "bias_replay": float(bias_replay.detach()),
            }
            history.append(record)
            if progress is not None:
                progress(record)

    with torch.no_grad():
        transformed, bounded = bounded_representations(
            base,
            raw_delta,
            maximum_delta=float(config.maximum_descriptor_delta),
        )
        prototype_bias = torch.clamp(
            base_bias
            + float(config.maximum_bias_delta) * torch.tanh(raw_bias_delta),
            max=0.0,
        )
    updated_state = dict(state)
    updated_state["anchor_features"] = transformed[:anchor_count].cpu()
    updated_family = dict(family)
    updated_family["prototype_features"] = transformed[anchor_count:].cpu()
    updated_family["prototype_bias"] = prototype_bias.cpu()
    active_delta = torch.linalg.norm(bounded[active_ids], dim=1)
    metadata = {
        "schema": "lafgs_selection_aware_reconstruction",
        "version": 1,
        "config": {
            key: value
            for key, value in config.__dict__.items()
        },
        "training_data": dict(data.diagnostics),
        "active_representation_count": int(len(active_ids)),
        "descriptor_delta_mean": float(active_delta.mean()),
        "descriptor_delta_p95": float(torch.quantile(active_delta, 0.95)),
        "descriptor_delta_max": float(active_delta.max()),
        "prototype_bias_shift_abs_mean": float(
            (prototype_bias - base_bias).abs().mean()
        ) if prototype_count else 0.0,
        "prototype_bias_shift_abs_max": float(
            (prototype_bias - base_bias).abs().max()
        ) if prototype_count else 0.0,
        "history": history,
    }
    updated_state["selection_aware_reconstruction"] = metadata
    updated_family["selection_aware_reconstruction"] = metadata
    return updated_state, updated_family, history
