"""Self-localization-guided pose-sufficient correspondence set learning.

The selector operates on the unchanged single-descriptor global top-1 graph.
It learns a query-level, approximately submodular utility from exact PoseLib
set outcomes and produces one nested correspondence ordering for one PnP solve.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from localization_training.pose_sufficient_selector import (
    FEATURE_NAMES,
    image_grid_cells,
)
from localization_training.slps_residual_signatures import (
    RESIDUAL_SIGNATURE_FEATURE_NAMES,
)


SLPS_FEATURE_NAMES = FEATURE_NAMES + (
    "xyz_robust_x",
    "xyz_robust_y",
    "xyz_robust_z",
    "xyz_robust_radius",
    "anchor_type",
    "query_track_multiplicity",
    "anchor_track_stability",
    "anchor_map_support",
)

SLPS_BIAS_AWARE_FEATURE_NAMES = (
    SLPS_FEATURE_NAMES + RESIDUAL_SIGNATURE_FEATURE_NAMES
)


def slps_feature_names(input_dim: int) -> tuple[str, ...]:
    if int(input_dim) == len(SLPS_FEATURE_NAMES):
        return SLPS_FEATURE_NAMES
    if int(input_dim) == len(SLPS_BIAS_AWARE_FEATURE_NAMES):
        return SLPS_BIAS_AWARE_FEATURE_NAMES
    raise ValueError(f"unsupported SLPS input dimension: {int(input_dim)}")

RELATION_NAMES = (
    "image",
    "spatial_3d",
    "dependency",
    "source",
    "track",
)


def _multiplicity(values: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(values).long().reshape(-1)
    if not len(values):
        return torch.empty(0, dtype=torch.float32, device=values.device)
    groups = _local_group_ids(values)
    return torch.bincount(groups)[groups].float()


def _local_group_ids(values: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(values).long().reshape(-1)
    if not len(values):
        return values
    # Negative IDs denote unknown identity, not a shared relation. Treat each
    # unknown row as a singleton so unrelated base anchors are never pooled.
    output = torch.empty_like(values)
    known = values >= 0
    known_count = 0
    if bool(known.any()):
        _, inverse = torch.unique(
            values[known], sorted=True, return_inverse=True
        )
        output[known] = inverse
        known_count = int(inverse.max()) + 1
    unknown = ~known
    if bool(unknown.any()):
        output[unknown] = known_count + torch.arange(
            int(unknown.sum()), device=values.device
        )
    return output


def normalize_relation_groups(
    relation_groups: torch.Tensor,
) -> torch.Tensor:
    groups = torch.as_tensor(relation_groups).long()
    if groups.ndim != 2:
        raise ValueError("SLPS relation groups must be a matrix")
    return torch.stack(
        [
            _local_group_ids(groups[:, relation])
            for relation in range(groups.shape[1])
        ],
        dim=1,
    )


def robust_xyz_coordinates(xyz: torch.Tensor) -> torch.Tensor:
    """Normalize candidate geometry within a query without scene-scale leakage."""

    xyz = torch.as_tensor(xyz).float().reshape(-1, 3)
    if not len(xyz):
        return xyz
    center = torch.quantile(xyz, 0.5, dim=0)
    absolute = (xyz - center).abs()
    scale = torch.quantile(absolute, 0.5, dim=0)
    fallback = torch.quantile(absolute, 0.75, dim=0).clamp_min(1e-4)
    scale = torch.where(scale > 1e-4, scale, fallback)
    return ((xyz - center) / scale).clamp(-6.0, 6.0) / 6.0


def robust_spatial_bins(xyz: torch.Tensor) -> torch.Tensor:
    """Assign robustly normalized geometry to 27 query-local spatial cells."""

    normalized = robust_xyz_coordinates(xyz)
    trit = (normalized > (-0.5 / 6.0)).long()
    trit = trit + (normalized > (0.5 / 6.0)).long()
    return trit[:, 0] + 3 * trit[:, 1] + 9 * trit[:, 2]


def build_relation_groups(
    *,
    keypoints: torch.Tensor,
    image_hw: tuple[int, int] | list[int],
    xyz: torch.Tensor,
    dependency_groups: torch.Tensor,
    source_groups: torch.Tensor,
    track_groups: torch.Tensor,
) -> torch.Tensor:
    """Build the five deployment-visible typed relation graphs."""

    length = len(torch.as_tensor(keypoints).reshape(-1, 2))
    aligned = (
        torch.as_tensor(xyz).reshape(-1, 3),
        torch.as_tensor(dependency_groups).reshape(-1),
        torch.as_tensor(source_groups).reshape(-1),
        torch.as_tensor(track_groups).reshape(-1),
    )
    if any(len(value) != length for value in aligned):
        raise ValueError("SLPS relation inputs must align")
    raw_groups = torch.stack(
        (
            image_grid_cells(keypoints, image_hw, rows=8, cols=8),
            robust_spatial_bins(xyz),
            torch.as_tensor(dependency_groups).long().reshape(-1),
            torch.as_tensor(source_groups).long().reshape(-1),
            torch.as_tensor(track_groups).long().reshape(-1),
        ),
        dim=1,
    )
    return normalize_relation_groups(raw_groups)


def build_slps_features(
    base_features: torch.Tensor,
    *,
    xyz: torch.Tensor,
    anchor_type: torch.Tensor,
    track_groups: torch.Tensor,
    track_stability: torch.Tensor,
    anchor_map_support: torch.Tensor,
    residual_signature_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extend the established train/deploy row contract with local geometry."""

    base = torch.as_tensor(base_features).float()
    xyz = torch.as_tensor(xyz).float().reshape(-1, 3)
    anchor_type = torch.as_tensor(anchor_type).float().reshape(-1)
    track = torch.as_tensor(track_groups).long().reshape(-1)
    stability = torch.as_tensor(track_stability).float().reshape(-1)
    support = torch.as_tensor(anchor_map_support).float().reshape(-1)
    length = len(base)
    if (
        base.ndim != 2
        or base.shape[1] != len(FEATURE_NAMES)
        or any(
            len(value) != length
            for value in (xyz, anchor_type, track, stability, support)
        )
    ):
        raise ValueError("SLPS row feature inputs must align")
    normalized_xyz = robust_xyz_coordinates(xyz)
    radius = torch.linalg.norm(normalized_xyz, dim=1)
    anchor_type = anchor_type / max(float(anchor_type.max()), 1.0)
    output = torch.cat(
        (
            base,
            normalized_xyz,
            radius[:, None],
            anchor_type[:, None],
            torch.log1p(_multiplicity(track))[:, None],
            stability.clamp(0.0, 1.0)[:, None],
            torch.log1p(support.clamp_min(0.0))[:, None],
        ),
        dim=1,
    )
    if residual_signature_features is None:
        return output
    residual = torch.as_tensor(residual_signature_features).float()
    if residual.shape != (length, len(RESIDUAL_SIGNATURE_FEATURE_NAMES)):
        raise ValueError("SLPS residual signature features must align")
    return torch.cat((output, residual), dim=1)


def beta_track_stability(
    *,
    attempts_by_fold: torch.Tensor,
    clean_inlier_by_fold: torch.Tensor,
    track_groups: torch.Tensor,
    prior_strength: float = 4.0,
) -> torch.Tensor:
    """Estimate cross-fold track stability and broadcast it to anchor rows."""

    attempts = torch.as_tensor(attempts_by_fold).float()
    clean = torch.as_tensor(clean_inlier_by_fold).float()
    track = torch.as_tensor(track_groups).long().reshape(-1)
    if attempts.shape != clean.shape or attempts.ndim != 2:
        raise ValueError("track stability statistics must be [fold, anchor]")
    if attempts.shape[1] != len(track):
        raise ValueError("track stability groups must align with anchors")
    prior = max(float(prior_strength), 0.0)
    global_rate = clean.sum() / attempts.sum().clamp_min(1.0)
    observed = attempts > 0
    output = torch.zeros(len(track))
    unknown = track < 0
    if bool(unknown.any()):
        unknown_attempts = attempts[:, unknown].sum(dim=0)
        unknown_clean = clean[:, unknown].sum(dim=0)
        output[unknown] = (
            unknown_clean + prior * global_rate
        ) / (unknown_attempts + prior).clamp_min(1e-6)
    known = ~unknown
    if not bool(known.any()):
        return output.clamp(0.0, 1.0)
    _, inverse = torch.unique(
        track[known], sorted=False, return_inverse=True
    )
    track_count = int(inverse.max()) + 1
    known_indices = torch.where(known)[0]
    for track_index in range(track_count):
        anchors = known_indices[inverse == track_index]
        fold_observed = observed[:, anchors].any(dim=1)
        if not bool(fold_observed.any()):
            output[anchors] = float(global_rate)
            continue
        weighted_attempts = attempts[:, anchors].sum(dim=1)
        weighted_clean = clean[:, anchors].sum(dim=1)
        fold_rate = (
            weighted_clean + prior * global_rate
        ) / (weighted_attempts + prior).clamp_min(1e-6)
        values = fold_rate[fold_observed]
        stability = values.mean() - values.std(unbiased=False)
        output[anchors] = stability.clamp(0.0, 1.0)
    return output


@dataclass(frozen=True)
class SLPSModelConfig:
    input_dim: int = len(SLPS_FEATURE_NAMES)
    hidden_dim: int = 32
    relation_layers: int = 2
    complementarity_dim: int = 8
    relation_count: int = len(RELATION_NAMES)
    utility_scale: float = 128.0
    coverage_scale: float = 16.0
    greedy_block_size: int = 32
    quality_utility_heads: bool = False
    relative_outcome_heads: bool = False
    bias_aware_utility: bool = False
    decoupled_risk_encoder: bool = False
    bounded_residual_utility_fraction: float = 0.0


@dataclass(frozen=True)
class SLPSSelection:
    selected_mask: torch.Tensor
    ordering: torch.Tensor
    selected_budget: int
    used_fallback: bool
    safe_probability: float
    catastrophic_probability: float
    expected_hypotheses: float
    diagnostics: dict[str, float]
    relative_utility_lcb: float = float("nan")
    relative_utility_median: float = float("nan")


def _segment_mean(
    values: torch.Tensor, groups: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    groups = torch.as_tensor(groups).long().reshape(-1)
    inverse = groups
    counts = torch.bincount(inverse)
    pooled = values.new_zeros((len(counts), values.shape[1]))
    pooled.index_add_(0, inverse, values)
    pooled = pooled / counts.to(values.dtype).clamp_min(1.0)[:, None]
    return pooled[inverse], counts[inverse].to(values.dtype)


def _relation_inverse(groups: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(groups).long().reshape(-1)


def _relation_group_count(groups: torch.Tensor) -> int:
    groups = torch.as_tensor(groups).long().reshape(-1)
    if not len(groups):
        return 0
    return int(torch.bincount(groups).count_nonzero())


class _TypedRelationLayer(nn.Module):
    def __init__(self, hidden_dim: int, relation_count: int):
        super().__init__()
        input_dim = hidden_dim * (2 + relation_count) + relation_count
        self.update = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.relation_count = int(relation_count)

    def forward(
        self, hidden: torch.Tensor, relation_groups: torch.Tensor
    ) -> torch.Tensor:
        contexts = []
        counts = []
        for relation in range(self.relation_count):
            context, count = _segment_mean(
                hidden, relation_groups[:, relation]
            )
            contexts.append(context)
            counts.append(torch.log1p(count)[:, None])
        global_context = hidden.mean(dim=0, keepdim=True).expand_as(hidden)
        update = self.update(
            torch.cat(
                (hidden, global_context, *contexts, *counts),
                dim=1,
            )
        )
        return self.norm(hidden + update)


class SLPSSelector(nn.Module):
    """Typed relation encoder with an approximately submodular set utility."""

    def __init__(
        self,
        config: SLPSModelConfig,
        *,
        feature_mean: torch.Tensor | None = None,
        feature_scale: torch.Tensor | None = None,
    ):
        super().__init__()
        self.config = config
        self.feature_names = slps_feature_names(config.input_dim)
        mean = (
            torch.zeros(config.input_dim)
            if feature_mean is None
            else torch.as_tensor(feature_mean).float()
        )
        scale = (
            torch.ones(config.input_dim)
            if feature_scale is None
            else torch.as_tensor(feature_scale).float()
        )
        if mean.numel() != config.input_dim or scale.numel() != config.input_dim:
            raise ValueError("SLPS normalization dimensions differ")
        self.register_buffer("feature_mean", mean.reshape(-1))
        self.register_buffer(
            "feature_scale", scale.reshape(-1).clamp_min(1e-6)
        )
        self.row_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )
        self.relation_layers = nn.ModuleList(
            [
                _TypedRelationLayer(
                    config.hidden_dim, config.relation_count
                )
                for _ in range(config.relation_layers)
            ]
        )
        if config.decoupled_risk_encoder:
            self.risk_row_encoder = nn.Sequential(
                nn.Linear(config.input_dim, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
            )
            self.risk_relation_layers = nn.ModuleList(
                [
                    _TypedRelationLayer(
                        config.hidden_dim, config.relation_count
                    )
                    for _ in range(config.relation_layers)
                ]
            )
        else:
            self.risk_row_encoder = None
            self.risk_relation_layers = nn.ModuleList()
        self.additive_head = nn.Linear(config.hidden_dim, 1)
        if float(config.bounded_residual_utility_fraction) > 0.0:
            self.residual_utility_head = nn.Linear(config.hidden_dim, 1)
            nn.init.zeros_(self.residual_utility_head.weight)
            nn.init.zeros_(self.residual_utility_head.bias)
        else:
            self.residual_utility_head = None
        self.strict_head = nn.Linear(config.hidden_dim, 1)
        self.solver_head = nn.Linear(config.hidden_dim, 1)
        self.harmful_head = nn.Linear(config.hidden_dim, 1)
        self.coverage_head = nn.Linear(
            config.hidden_dim, config.relation_count
        )
        self.complementarity_head = nn.Linear(
            config.hidden_dim, config.complementarity_dim
        )
        if config.bias_aware_utility:
            self.bias_head = nn.Linear(config.hidden_dim, 2)
            self.bias_weight_raw = nn.Parameter(torch.tensor(-2.0))
        else:
            self.bias_head = None
            self.register_parameter("bias_weight_raw", None)
        risk_input_dim = (
            3 * config.hidden_dim
            + 8
            + (3 if config.bias_aware_utility else 0)
        )
        self.set_outcome_head = nn.Sequential(
            nn.Linear(risk_input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 3),
        )
        relative_input_dim = (
            6 * config.hidden_dim
            + 2 * config.relation_count
            + 20
        )
        self.relative_outcome_head = nn.Sequential(
            nn.Linear(relative_input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 2),
        )
        self.harmful_weight_raw = nn.Parameter(torch.tensor(0.0))
        self.strict_weight_raw = nn.Parameter(torch.tensor(-1.0))
        self.solver_weight_raw = nn.Parameter(torch.tensor(-1.0))
        self.coverage_alpha_raw = nn.Parameter(
            torch.zeros(config.relation_count)
        )
        self.logdet_beta_raw = nn.Parameter(torch.tensor(-1.0))
        self.size_cost_raw = nn.Parameter(torch.tensor(-1.0))
        nn.init.constant_(self.additive_head.bias, -1.0)
        nn.init.constant_(self.coverage_alpha_raw, -3.0)
        nn.init.constant_(self.logdet_beta_raw, -3.0)
        nn.init.constant_(self.size_cost_raw, -0.5)
        if self.bias_head is not None:
            nn.init.zeros_(self.bias_head.weight)
            nn.init.zeros_(self.bias_head.bias)
        nn.init.zeros_(self.relative_outcome_head[-1].weight)
        nn.init.zeros_(self.relative_outcome_head[-1].bias)
        with torch.no_grad():
            self.relative_outcome_head[-1].bias[1] = -2.0
        self._outcome_atlas: dict | None = None

    def export_config(self) -> dict:
        return asdict(self.config)

    def attach_outcome_atlas(self, atlas: dict | None) -> None:
        """Attach scene-specific exact self-localization support outcomes."""

        if atlas is None:
            self._outcome_atlas = None
            return
        if atlas.get("schema") != "lafgs_slps_outcome_atlas":
            raise ValueError("unsupported SLPS outcome atlas")
        support = torch.as_tensor(atlas["support_anchor_mask"]).bool()
        safe = torch.as_tensor(atlas["safe_probability_targets"]).float()
        catastrophic = torch.as_tensor(
            atlas["catastrophic_probability_targets"]
        ).float()
        utility = torch.as_tensor(atlas["relative_utility_targets"]).float()
        budgets = tuple(int(value) for value in atlas["budgets"])
        query_names = tuple(str(value) for value in atlas["support_query_names"])
        if (
            support.ndim != 2
            or safe.shape != (len(support), len(budgets))
            or catastrophic.shape != safe.shape
            or utility.shape != safe.shape
            or len(query_names) != len(support)
        ):
            raise ValueError("SLPS outcome atlas tensors do not align")
        context_support = atlas.get("support_context_mask")
        if context_support is not None:
            context_support = torch.as_tensor(context_support).bool()
            if context_support.ndim != 2 or len(context_support) != len(
                support
            ):
                raise ValueError("SLPS context atlas tensors do not align")
        set_support = atlas.get("support_set_anchor_mask")
        if set_support is not None:
            set_support = torch.as_tensor(set_support).bool()
            if set_support.shape[:2] != (len(support), len(budgets)):
                raise ValueError("SLPS set atlas tensors do not align")
            if set_support.shape[2] != support.shape[1]:
                raise ValueError("SLPS set atlas anchor dimension differs")
        set_context_support = atlas.get("support_set_context_mask")
        if set_context_support is not None:
            set_context_support = torch.as_tensor(
                set_context_support
            ).bool()
            if (
                set_support is None
                or set_context_support.shape[:2] != set_support.shape[:2]
            ):
                raise ValueError("SLPS set-context atlas tensors do not align")
        device = self.feature_mean.device
        self._outcome_atlas = {
            "support_anchor_mask": support.to(device=device),
            "support_anchor_count": support.sum(dim=1).float().to(device=device),
            "safe_probability_targets": safe.to(device=device),
            "catastrophic_probability_targets": catastrophic.to(device=device),
            "relative_utility_targets": utility.to(device=device),
            "budgets": budgets,
            "support_query_names": query_names,
            "neighbor_count": max(int(atlas.get("neighbor_count", 8)), 1),
            "similarity_power": max(
                float(atlas.get("similarity_power", 8.0)), 0.0
            ),
            "support_context_mask": (
                context_support.to(device=device)
                if context_support is not None
                else None
            ),
            "support_context_count": (
                context_support.sum(dim=1).float().to(device=device)
                if context_support is not None
                else None
            ),
            "context_grid_size": max(
                int(atlas.get("context_grid_size", 0)), 0
            ),
            "context_weight": min(
                max(float(atlas.get("context_weight", 0.0)), 0.0), 1.0
            ),
            "support_set_anchor_mask": (
                set_support.to(device=device)
                if set_support is not None
                else None
            ),
            "support_set_anchor_count": (
                set_support.sum(dim=2).float().to(device=device)
                if set_support is not None
                else None
            ),
            "support_set_context_mask": (
                set_context_support.to(device=device)
                if set_context_support is not None
                else None
            ),
            "support_set_context_count": (
                set_context_support.sum(dim=2).float().to(device=device)
                if set_context_support is not None
                else None
            ),
            "set_query_context_weight": min(
                max(
                    float(atlas.get("set_query_context_weight", 0.0)),
                    0.0,
                ),
                1.0,
            ),
        }

    @torch.no_grad()
    def predict_atlas_outcomes(
        self,
        anchor_indices: torch.Tensor,
        *,
        features: torch.Tensor | None = None,
        budget_masks: dict[int, torch.Tensor] | None = None,
        query_name: str | None = None,
    ) -> dict[int, dict[str, float]]:
        """Transfer exact outcomes from view- and selected-set-aligned support."""

        if self._outcome_atlas is None:
            return {}
        atlas = self._outcome_atlas
        support = atlas["support_anchor_mask"]
        rows = torch.as_tensor(
            anchor_indices, device=support.device
        ).long().reshape(-1)
        if not len(rows):
            return {}
        anchors = torch.unique(rows)
        if int(anchors.min()) < 0 or int(anchors.max()) >= support.shape[1]:
            raise ValueError("SLPS atlas anchor indices are out of bounds")
        feature_rows = None
        cells = None
        context_support = atlas["support_context_mask"]
        grid_size = int(atlas["context_grid_size"])
        if (
            grid_size > 0 and features is not None
        ):
            feature_rows = torch.as_tensor(
                features, device=support.device
            ).float()
            if len(rows) != len(feature_rows):
                raise ValueError("SLPS context atlas rows do not align")
            x_index = self.feature_names.index("keypoint_x")
            y_index = self.feature_names.index("keypoint_y")
            x = (
                feature_rows[:, x_index] * grid_size
            ).floor().long().clamp(0, grid_size - 1)
            y = (
                feature_rows[:, y_index] * grid_size
            ).floor().long().clamp(0, grid_size - 1)
            cells = x + grid_size * y

        def incidence_similarity(
            support_mask: torch.Tensor,
            support_count: torch.Tensor,
            selected_rows: torch.Tensor,
            selected_cells: torch.Tensor | None,
            context_mask: torch.Tensor | None,
            context_count: torch.Tensor | None,
        ) -> torch.Tensor:
            selected_anchors = torch.unique(selected_rows)
            overlap = support_mask[:, selected_anchors].sum(dim=1).float()
            similarity = overlap / torch.sqrt(
                support_count.clamp_min(1.0)
                * float(max(len(selected_anchors), 1))
            )
            if (
                context_mask is not None
                and context_count is not None
                and selected_cells is not None
            ):
                token_count = grid_size * grid_size
                tokens = torch.unique(
                    selected_rows * token_count + selected_cells
                )
                context_overlap = context_mask[:, tokens].sum(dim=1).float()
                context_similarity = context_overlap / torch.sqrt(
                    context_count.clamp_min(1.0)
                    * float(max(len(tokens), 1))
                )
                context_weight = float(atlas["context_weight"])
                similarity = (
                    (1.0 - context_weight) * similarity
                    + context_weight * context_similarity
                )
            return similarity

        query_similarity = incidence_similarity(
            support,
            atlas["support_anchor_count"],
            rows,
            cells,
            context_support,
            atlas["support_context_count"],
        )
        set_support = atlas["support_set_anchor_mask"]
        set_context_support = atlas["support_set_context_mask"]
        budget_lookup = {
            int(budget): column
            for column, budget in enumerate(atlas["budgets"])
        }
        output = {}
        for budget, column in budget_lookup.items():
            selected_mask = None
            if budget_masks is not None:
                selected_mask = budget_masks.get(int(budget))
            if selected_mask is not None:
                selected_mask = torch.as_tensor(
                    selected_mask, device=support.device
                ).bool().reshape(-1)
                if len(selected_mask) != len(rows):
                    raise ValueError("SLPS atlas budget mask does not align")
            if set_support is not None and selected_mask is not None:
                selected_rows = rows[selected_mask]
                selected_cells = (
                    cells[selected_mask] if cells is not None else None
                )
                similarity = incidence_similarity(
                    set_support[:, column],
                    atlas["support_set_anchor_count"][:, column],
                    selected_rows,
                    selected_cells,
                    (
                        set_context_support[:, column]
                        if set_context_support is not None
                        else None
                    ),
                    (
                        atlas["support_set_context_count"][:, column]
                        if atlas["support_set_context_count"] is not None
                        else None
                    ),
                )
                query_weight = float(atlas["set_query_context_weight"])
                similarity = (
                    (1.0 - query_weight) * similarity
                    + query_weight * query_similarity
                )
            else:
                similarity = query_similarity.clone()
            if query_name is not None:
                for index, support_name in enumerate(
                    atlas["support_query_names"]
                ):
                    if support_name == str(query_name):
                        similarity[index] = float("-inf")
            finite = torch.isfinite(similarity)
            available = int(finite.sum())
            if available < 1:
                continue
            neighbor_count = min(int(atlas["neighbor_count"]), available)
            values, indices = torch.topk(
                similarity, k=neighbor_count, largest=True, sorted=True
            )
            values = values.clamp_min(0.0)
            if not bool((values > 0).any()):
                continue
            maximum = values.max().clamp_min(1e-8)
            weights = (values / maximum).pow(
                float(atlas["similarity_power"])
            )
            weights = weights / weights.sum().clamp_min(1e-8)
            output[int(budget)] = {
                "safe_probability": float(
                    (
                        weights
                        * atlas["safe_probability_targets"][indices, column]
                    ).sum()
                ),
                "catastrophic_probability": float(
                    (
                        weights
                        * atlas["catastrophic_probability_targets"][
                            indices, column
                        ]
                    ).sum()
                ),
                "relative_utility": float(
                    (
                        weights
                        * atlas["relative_utility_targets"][indices, column]
                    ).sum()
                ),
                "maximum_similarity": float(values.max()),
                "effective_neighbors": float(
                    1.0 / weights.square().sum().clamp_min(1e-8)
                ),
            }
        return output

    def encode(
        self, features: torch.Tensor, relation_groups: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        features = torch.as_tensor(
            features, device=self.feature_mean.device
        ).float()
        groups = torch.as_tensor(
            relation_groups, device=features.device
        ).long()
        if (
            features.ndim != 2
            or features.shape[1] != self.config.input_dim
            or groups.shape != (len(features), self.config.relation_count)
        ):
            raise ValueError("SLPS model inputs differ from its contract")
        normalized = (features - self.feature_mean) / self.feature_scale
        hidden = self.row_encoder(normalized)
        for layer in self.relation_layers:
            hidden = layer(hidden, groups)
        complementarity = F.normalize(
            self.complementarity_head(hidden), dim=1
        )
        additive_base = F.softplus(self.additive_head(hidden).reshape(-1))
        utility_residual = additive_base.new_zeros(len(additive_base))
        utility_residual_unit = additive_base.new_zeros(len(additive_base))
        if self.residual_utility_head is not None:
            utility_residual_unit = torch.tanh(
                self.residual_utility_head(hidden).reshape(-1)
            )
            robust_scale = (
                torch.quantile(additive_base.detach(), 0.90)
                - torch.quantile(additive_base.detach(), 0.10)
            ).clamp_min(0.1)
            utility_residual = (
                float(self.config.bounded_residual_utility_fraction)
                * robust_scale
                * utility_residual_unit
            )
        output = {
            "hidden": hidden,
            "additive": additive_base + utility_residual,
            "additive_base": additive_base,
            "utility_residual": utility_residual,
            "utility_residual_unit": utility_residual_unit,
            "strict_probability": torch.sigmoid(
                self.strict_head(hidden).reshape(-1)
            ),
            "solver_probability": torch.sigmoid(
                self.solver_head(hidden).reshape(-1)
            ),
            "harmful_probability": torch.sigmoid(
                self.harmful_head(hidden).reshape(-1)
            ),
            "coverage": F.softplus(self.coverage_head(hidden)),
            "complementarity": complementarity,
        }
        if self.risk_row_encoder is not None:
            risk_hidden = self.risk_row_encoder(normalized)
            for layer in self.risk_relation_layers:
                risk_hidden = layer(risk_hidden, groups)
            output["risk_hidden"] = risk_hidden
        if self.bias_head is not None:
            output["bias_vector"] = torch.tanh(self.bias_head(hidden))
        return output

    def _coverage_score(
        self,
        coverage: torch.Tensor,
        relation_groups: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        alpha = F.softplus(self.coverage_alpha_raw)
        score = coverage.new_zeros(())
        for relation in range(self.config.relation_count):
            groups = relation_groups[mask, relation]
            inverse = _relation_inverse(groups)
            group_sum = coverage.new_zeros(len(mask))
            if len(inverse):
                group_sum.index_add_(
                    0, inverse, coverage[mask, relation]
                )
                score = score + alpha[relation] * torch.log1p(
                    group_sum
                ).sum()
        return score / float(self.config.coverage_scale)

    def score_set(
        self,
        encoded: dict[str, torch.Tensor],
        relation_groups: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.score_sets(
            encoded,
            relation_groups,
            torch.as_tensor(mask).reshape(1, -1),
        )[0]

    def score_sets(
        self,
        encoded: dict[str, torch.Tensor],
        relation_groups: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized utility for multiple candidate sets from one query."""

        masks = torch.as_tensor(
            masks, device=encoded["hidden"].device
        ).bool()
        if masks.ndim != 2:
            raise ValueError("SLPS set masks must be a matrix")
        groups = torch.as_tensor(
            relation_groups, device=masks.device
        ).long()
        if (
            masks.shape[1] != len(encoded["hidden"])
            or bool((masks.sum(dim=1) < 1).any())
        ):
            raise ValueError("SLPS set masks are empty or misaligned")
        weights = masks.to(encoded["hidden"].dtype)
        scale = float(self.config.utility_scale)
        row_utility = encoded["additive"]
        if self.config.quality_utility_heads:
            row_utility = (
                row_utility
                + F.softplus(self.strict_weight_raw)
                * encoded["strict_probability"]
                + F.softplus(self.solver_weight_raw)
                * encoded["solver_probability"]
            )
        additive = (weights * row_utility[None]).sum(dim=1) / scale
        harmful = (
            F.softplus(self.harmful_weight_raw)
            * (
                weights
                * encoded["harmful_probability"][None]
            ).sum(dim=1)
            / scale
        )
        alpha = F.softplus(self.coverage_alpha_raw)
        coverage_score = additive.new_zeros(len(masks))
        for relation in range(self.config.relation_count):
            relation_ids = groups[:, relation]
            group_count = (
                int(relation_ids.max()) + 1 if len(relation_ids) else 0
            )
            group_sum = additive.new_zeros(
                (len(masks), group_count)
            )
            if group_count:
                source = (
                    weights
                    * encoded["coverage"][:, relation][None]
                )
                group_sum.index_add_(1, relation_ids, source)
                coverage_score = coverage_score + alpha[
                    relation
                ] * torch.log1p(group_sum).sum(dim=1)
        coverage_score = coverage_score / float(
            self.config.coverage_scale
        )
        z = encoded["complementarity"]
        information = torch.einsum(
            "sn,nd,ne->sde", weights, z, z
        )
        identity = torch.eye(
            self.config.complementarity_dim,
            dtype=z.dtype,
            device=z.device,
        )
        logdet = torch.linalg.slogdet(
            information + identity[None]
        ).logabsdet
        size_cost = (
            F.softplus(self.size_cost_raw)
            * weights.sum(dim=1)
            / scale
        )
        bias_penalty = additive.new_zeros(len(masks))
        if self.bias_head is not None:
            bias_reliability = encoded["solver_probability"].clamp_min(0.05)
            bias_mass = weights * bias_reliability[None]
            bias_sum = bias_mass @ encoded["bias_vector"]
            bias_mean = bias_sum / bias_mass.sum(dim=1).clamp_min(1e-6)[:, None]
            bias_penalty = F.softplus(self.bias_weight_raw) * bias_mean.square().sum(
                dim=1
            )
        return (
            additive
            - harmful
            + coverage_score
            + F.softplus(self.logdet_beta_raw) * logdet
            - size_cost
            - bias_penalty
        )

    def _score_set_reference(
        self,
        encoded: dict[str, torch.Tensor],
        relation_groups: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Scalar reference retained for utility parity tests."""

        mask = torch.as_tensor(
            mask, device=encoded["hidden"].device
        ).bool().reshape(-1)
        groups = torch.as_tensor(
            relation_groups, device=mask.device
        ).long()
        if len(mask) != len(encoded["hidden"]) or int(mask.sum()) < 1:
            raise ValueError("SLPS set mask is empty or misaligned")
        scale = float(self.config.utility_scale)
        row_utility = encoded["additive"]
        if self.config.quality_utility_heads:
            row_utility = (
                row_utility
                + F.softplus(self.strict_weight_raw)
                * encoded["strict_probability"]
                + F.softplus(self.solver_weight_raw)
                * encoded["solver_probability"]
            )
        additive = row_utility[mask].sum() / scale
        harmful = (
            F.softplus(self.harmful_weight_raw)
            * encoded["harmful_probability"][mask].sum()
            / scale
        )
        coverage = self._coverage_score(
            encoded["coverage"], groups, mask
        )
        z = encoded["complementarity"][mask]
        matrix = torch.eye(
            self.config.complementarity_dim,
            dtype=z.dtype,
            device=z.device,
        ) + z.T @ z
        logdet = torch.linalg.slogdet(matrix).logabsdet
        size_cost = (
            F.softplus(self.size_cost_raw) * mask.sum().to(z.dtype) / scale
        )
        bias_penalty = z.new_zeros(())
        if self.bias_head is not None:
            bias_reliability = encoded["solver_probability"][mask].clamp_min(
                0.05
            )
            bias_mean = (
                bias_reliability[:, None] * encoded["bias_vector"][mask]
            ).sum(dim=0) / bias_reliability.sum().clamp_min(1e-6)
            bias_penalty = (
                F.softplus(self.bias_weight_raw) * bias_mean.square().sum()
            )
        return (
            additive
            - harmful
            + coverage
            + F.softplus(self.logdet_beta_raw) * logdet
            - size_cost
            - bias_penalty
        )

    def predict_set_outcome(
        self,
        encoded: dict[str, torch.Tensor],
        relation_groups: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mask = torch.as_tensor(
            mask, device=encoded["hidden"].device
        ).bool().reshape(-1)
        groups = torch.as_tensor(
            relation_groups, device=mask.device
        ).long()
        hidden = encoded.get("risk_hidden", encoded["hidden"])
        selected = hidden[mask]
        if not len(selected):
            raise ValueError("SLPS cannot predict an empty set")
        selected_mean = selected.mean(dim=0)
        selected_max = selected.max(dim=0).values
        all_mean = hidden.mean(dim=0)
        distinct_ratios = []
        for relation in range(self.config.relation_count):
            selected_groups = _relation_group_count(
                groups[mask, relation]
            )
            all_groups = max(
                _relation_group_count(groups[:, relation]), 1
            )
            distinct_ratios.append(float(selected_groups) / float(all_groups))
        structural_values = [
                torch.log1p(mask.sum()).item() / 8.0,
                float(mask.float().mean()),
                float(
                    encoded["strict_probability"][mask].mean().detach()
                ),
                float(
                    encoded["solver_probability"][mask].mean().detach()
                ),
                float(
                    encoded["harmful_probability"][mask].mean().detach()
                ),
                float(sum(distinct_ratios) / len(distinct_ratios)),
                float(
                    encoded["additive"][mask].mean().detach()
                ),
                float(
                    encoded["coverage"][mask].mean().detach()
                ),
            ]
        if self.bias_head is not None:
            reliability = encoded["solver_probability"][mask].clamp_min(0.05)
            bias_mean = (
                reliability[:, None] * encoded["bias_vector"][mask]
            ).sum(dim=0) / reliability.sum().clamp_min(1e-6)
            structural_values.extend(
                (
                    float(bias_mean[0].detach()),
                    float(bias_mean[1].detach()),
                    float(torch.linalg.norm(bias_mean).detach()),
                )
            )
        structural = hidden.new_tensor(structural_values)
        raw = self.set_outcome_head(
            torch.cat(
                (selected_mean, selected_max, all_mean, structural), dim=0
            )
        )
        return {
            "safe_logit": raw[0],
            "catastrophic_logit": raw[1],
            "log_hypotheses": raw[2],
        }

    def predict_relative_outcome(
        self,
        encoded: dict[str, torch.Tensor],
        relation_groups: torch.Tensor,
        mask: torch.Tensor,
        *,
        selected_utility: torch.Tensor | None = None,
        all_utility: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict conservative utility gain relative to the all-row solve."""

        mask = torch.as_tensor(
            mask, device=encoded["hidden"].device
        ).bool().reshape(-1)
        groups = torch.as_tensor(
            relation_groups, device=mask.device
        ).long()
        hidden = encoded.get("risk_hidden", encoded["hidden"])
        selected = hidden[mask]
        if not len(selected):
            raise ValueError("SLPS cannot predict an empty set")
        excluded = hidden[~mask]
        selected_std = selected.std(dim=0, unbiased=False)
        excluded_mean = (
            excluded.mean(dim=0) if len(excluded) else hidden.mean(dim=0)
        )
        hidden_summary = torch.cat(
            (
                selected.mean(dim=0),
                selected_std,
                selected.amin(dim=0),
                selected.amax(dim=0),
                hidden.mean(dim=0),
                excluded_mean,
            ),
            dim=0,
        )

        quality_rows = (
            encoded["strict_probability"],
            encoded["solver_probability"],
            encoded["harmful_probability"],
            encoded["additive"],
            encoded["coverage"].mean(dim=1),
        )
        quality_summary = []
        for values in quality_rows:
            selected_mean = values[mask].mean()
            all_mean = values.mean()
            quality_summary.extend(
                (selected_mean, all_mean, selected_mean - all_mean)
            )

        relation_summary = []
        selected_count = mask.sum().to(hidden.dtype).clamp_min(1.0)
        for relation in range(self.config.relation_count):
            selected_groups = _relation_group_count(
                groups[mask, relation]
            )
            all_groups = max(
                _relation_group_count(groups[:, relation]), 1
            )
            relation_summary.extend(
                (
                    hidden.new_tensor(
                        float(selected_groups) / float(all_groups)
                    ),
                    hidden.new_tensor(float(selected_groups))
                    / selected_count,
                )
            )

        all_mask = torch.ones_like(mask)
        selected_score = (
            self.score_set(encoded, groups, mask)
            if selected_utility is None
            else selected_utility
        )
        all_score = (
            self.score_set(encoded, groups, all_mask)
            if all_utility is None
            else all_utility
        )
        structural = (
            torch.log1p(selected_count) / 8.0,
            mask.to(hidden.dtype).mean(),
            selected_score,
            all_score,
            selected_score - all_score,
        )
        summary = torch.cat(
            (
                hidden_summary,
                torch.stack(quality_summary),
                torch.stack(relation_summary),
                torch.stack(structural),
            ),
            dim=0,
        )
        raw = self.relative_outcome_head(summary)
        median = raw[0]
        lower_quantile = median - F.softplus(raw[1])
        return {
            "relative_utility_median": median,
            "relative_utility_lcb": lower_quantile,
        }

    @torch.no_grad()
    def greedy_order(
        self,
        encoded: dict[str, torch.Tensor],
        relation_groups: torch.Tensor,
        *,
        maximum_count: int,
    ) -> torch.Tensor:
        """Greedily maximize learned diminishing-return marginal utility."""

        groups = torch.as_tensor(
            relation_groups, device=encoded["hidden"].device
        ).long()
        count = len(groups)
        target = min(max(int(maximum_count), 0), count)
        if target == 0:
            return torch.empty(0, dtype=torch.long, device=groups.device)
        scale = float(self.config.utility_scale)
        marginal_base = (
            encoded["additive"]
            - F.softplus(self.harmful_weight_raw)
            * encoded["harmful_probability"]
            - F.softplus(self.size_cost_raw)
        )
        if self.config.quality_utility_heads:
            marginal_base = (
                marginal_base
                + F.softplus(self.strict_weight_raw)
                * encoded["strict_probability"]
                + F.softplus(self.solver_weight_raw)
                * encoded["solver_probability"]
            )
        marginal_base = marginal_base / scale
        coverage_alpha = (
            F.softplus(self.coverage_alpha_raw)
            / float(self.config.coverage_scale)
        )
        relation_inverse = []
        relation_sums = []
        for relation in range(self.config.relation_count):
            inverse = _relation_inverse(groups[:, relation])
            relation_inverse.append(inverse)
            relation_sums.append(
                encoded["coverage"].new_zeros(count)
            )
        z = encoded["complementarity"]
        inverse_information = torch.eye(
            self.config.complementarity_dim,
            device=z.device,
            dtype=z.dtype,
        )
        selected = torch.zeros(count, dtype=torch.bool, device=z.device)
        bias_sum = z.new_zeros(2)
        bias_mass = z.new_zeros(())
        ordering = []
        block_size = max(int(self.config.greedy_block_size), 1)
        while sum(len(value) for value in ordering) < target:
            marginal = marginal_base.clone()
            for relation in range(self.config.relation_count):
                inverse = relation_inverse[relation]
                current = relation_sums[relation][inverse]
                weight = encoded["coverage"][:, relation]
                marginal = marginal + coverage_alpha[relation] * (
                    torch.log1p(current + weight) - torch.log1p(current)
                )
            projected = z @ inverse_information
            leverage = (projected * z).sum(dim=1).clamp_min(0.0)
            marginal = marginal + F.softplus(
                self.logdet_beta_raw
            ) * torch.log1p(leverage)
            if self.bias_head is not None:
                reliability = encoded["solver_probability"].clamp_min(0.05)
                current_bias = bias_sum / bias_mass.clamp_min(1e-6)
                current_penalty = current_bias.square().sum()
                candidate_sum = (
                    bias_sum[None]
                    + reliability[:, None] * encoded["bias_vector"]
                )
                candidate_mass = bias_mass + reliability
                candidate_bias = candidate_sum / candidate_mass[:, None]
                marginal = marginal + F.softplus(self.bias_weight_raw) * (
                    current_penalty - candidate_bias.square().sum(dim=1)
                )
            marginal[selected] = float("-inf")
            remaining = target - sum(len(value) for value in ordering)
            selected_block = torch.topk(
                marginal,
                k=min(block_size, remaining),
                largest=True,
                sorted=True,
            ).indices
            ordering.append(selected_block)
            selected[selected_block] = True
            for relation in range(self.config.relation_count):
                relation_sums[relation].index_add_(
                    0,
                    relation_inverse[relation][selected_block],
                    encoded["coverage"][selected_block, relation],
                )
            block_z = z[selected_block]
            right = inverse_information @ block_z.T
            middle = torch.eye(
                len(selected_block), device=z.device, dtype=z.dtype
            ) + block_z @ right
            inverse_information = inverse_information - (
                right @ torch.linalg.solve(middle, right.T)
            )
            if self.bias_head is not None:
                block_reliability = encoded["solver_probability"][
                    selected_block
                ].clamp_min(0.05)
                bias_sum = bias_sum + (
                    block_reliability[:, None]
                    * encoded["bias_vector"][selected_block]
                ).sum(dim=0)
                bias_mass = bias_mass + block_reliability.sum()
        return torch.cat(ordering, dim=0)

    @torch.no_grad()
    def select(
        self,
        features: torch.Tensor,
        relation_groups: torch.Tensor,
        *,
        budgets: Iterable[int] = (256, 384, 512, 768),
        safe_probability_threshold: float = 0.75,
        catastrophic_probability_threshold: float = 0.15,
        minimum_probability_margin: float = 0.1,
        relative_utility_lcb_threshold: float | None = None,
        risk_gate_mode: str = "neural",
        atlas_safe_probability_threshold: float = 0.75,
        atlas_catastrophic_probability_threshold: float = 0.15,
        atlas_relative_utility_threshold: float = 0.0,
        atlas_minimum_similarity: float = 0.0,
        anchor_indices: torch.Tensor | None = None,
        query_name: str | None = None,
        encoded: dict[str, torch.Tensor] | None = None,
    ) -> SLPSSelection:
        if encoded is None:
            encoded = self.encode(features, relation_groups)
        risk_gate_mode = str(risk_gate_mode)
        if risk_gate_mode not in {"neural", "atlas", "intersection"}:
            raise ValueError(f"unsupported SLPS risk gate mode: {risk_gate_mode}")
        count = len(features)
        groups = torch.as_tensor(
            relation_groups, device=encoded["hidden"].device
        ).long()
        neural_infeasible = (
            float(safe_probability_threshold) > 1.0
            or float(catastrophic_probability_threshold) < 0.0
        )
        if neural_infeasible and risk_gate_mode != "atlas":
            selected = torch.ones(count, dtype=torch.bool)
            return SLPSSelection(
                selected_mask=selected,
                ordering=torch.empty(0, dtype=torch.long),
                selected_budget=count,
                used_fallback=True,
                safe_probability=float("nan"),
                catastrophic_probability=float("nan"),
                expected_hypotheses=float("nan"),
                diagnostics={
                    "selected_budget": float(count),
                    "fallback": 1.0,
                    "fallback_infeasible_calibration": 1.0,
                },
            )
        nested_budgets = sorted(
            {
                min(max(int(value), 4), count)
                for value in budgets
                if int(value) > 0
            }
        )
        if count not in nested_budgets:
            nested_budgets.append(count)
        maximum_compact = max(
            (value for value in nested_budgets if value < count),
            default=0,
        )
        ordering = self.greedy_order(
            encoded,
            relation_groups,
            maximum_count=maximum_compact,
        )
        chosen_mask = torch.ones(
            count, dtype=torch.bool, device=ordering.device
        )
        chosen_budget = count
        chosen_safe = 1.0
        chosen_catastrophic = 0.0
        chosen_hypotheses = float("nan")
        chosen_relative_lcb = float("nan")
        chosen_relative_median = float("nan")
        fallback = True
        candidate_diagnostics = {}
        candidate_diagnostics.update(
            {
                "row_solver_probability_mean": float(
                    encoded["solver_probability"].mean()
                ),
                "row_solver_probability_p10": float(
                    torch.quantile(encoded["solver_probability"], 0.10)
                ),
                "row_solver_probability_p90": float(
                    torch.quantile(encoded["solver_probability"], 0.90)
                ),
                "row_strict_probability_mean": float(
                    encoded["strict_probability"].mean()
                ),
                "row_harmful_probability_mean": float(
                    encoded["harmful_probability"].mean()
                ),
            }
        )
        all_utility = None
        if self.config.relative_outcome_heads:
            all_utility = self.score_set(
                encoded,
                groups,
                torch.ones(count, dtype=torch.bool, device=groups.device),
            )
        atlas_budget_masks = {}
        for budget in nested_budgets:
            if budget >= count:
                continue
            budget_mask = torch.zeros(
                count, dtype=torch.bool, device=ordering.device
            )
            budget_mask[ordering[:budget]] = True
            atlas_budget_masks[int(budget)] = budget_mask
        atlas_outcomes = (
            self.predict_atlas_outcomes(
                anchor_indices,
                features=features,
                budget_masks=atlas_budget_masks,
                query_name=query_name,
            )
            if anchor_indices is not None
            else {}
        )
        accepted_candidates = []
        for budget in nested_budgets:
            mask = torch.ones(
                count, dtype=torch.bool, device=ordering.device
            )
            if budget < count:
                mask.zero_()
                mask[ordering[:budget]] = True
            outcome = self.predict_set_outcome(
                encoded, relation_groups, mask
            )
            safe = float(torch.sigmoid(outcome["safe_logit"]))
            catastrophic = float(
                torch.sigmoid(outcome["catastrophic_logit"])
            )
            hypotheses = float(
                torch.expm1(outcome["log_hypotheses"].clamp(0.0, 20.0))
            )
            relative_lcb = float("nan")
            relative_median = float("nan")
            if self.config.relative_outcome_heads:
                selected_utility = self.score_set(
                    encoded, groups, mask
                )
                relative = self.predict_relative_outcome(
                    encoded,
                    relation_groups,
                    mask,
                    selected_utility=selected_utility,
                    all_utility=all_utility,
                )
                relative_lcb = float(
                    relative["relative_utility_lcb"]
                )
                relative_median = float(
                    relative["relative_utility_median"]
                )
            candidate_diagnostics[f"safe_{budget}"] = safe
            candidate_diagnostics[f"catastrophic_{budget}"] = catastrophic
            candidate_diagnostics[f"hypotheses_{budget}"] = hypotheses
            candidate_diagnostics[f"relative_lcb_{budget}"] = relative_lcb
            candidate_diagnostics[
                f"relative_median_{budget}"
            ] = relative_median
            atlas = atlas_outcomes.get(int(budget))
            atlas_safe = float("nan")
            atlas_catastrophic = float("nan")
            atlas_utility = float("nan")
            atlas_similarity = float("nan")
            atlas_effective_neighbors = float("nan")
            if atlas is not None:
                atlas_safe = float(atlas["safe_probability"])
                atlas_catastrophic = float(
                    atlas["catastrophic_probability"]
                )
                atlas_utility = float(atlas["relative_utility"])
                atlas_similarity = float(atlas["maximum_similarity"])
                atlas_effective_neighbors = float(
                    atlas["effective_neighbors"]
                )
            candidate_diagnostics[f"atlas_safe_{budget}"] = atlas_safe
            candidate_diagnostics[
                f"atlas_catastrophic_{budget}"
            ] = atlas_catastrophic
            candidate_diagnostics[
                f"atlas_relative_utility_{budget}"
            ] = atlas_utility
            candidate_diagnostics[
                f"atlas_similarity_{budget}"
            ] = atlas_similarity
            candidate_diagnostics[
                f"atlas_effective_neighbors_{budget}"
            ] = atlas_effective_neighbors
            confident = (
                abs(safe - 0.5) >= float(minimum_probability_margin)
                and abs(catastrophic - 0.5)
                >= float(minimum_probability_margin)
            )
            neural_accepted = (
                not neural_infeasible
                and confident
                and safe >= float(safe_probability_threshold)
                and catastrophic
                <= float(catastrophic_probability_threshold)
                and (
                    not self.config.relative_outcome_heads
                    or (
                        relative_utility_lcb_threshold is not None
                        and relative_lcb
                        >= float(relative_utility_lcb_threshold)
                    )
                )
            )
            atlas_accepted = (
                atlas is not None
                and atlas_safe
                >= float(atlas_safe_probability_threshold)
                and atlas_catastrophic
                <= float(atlas_catastrophic_probability_threshold)
                and atlas_utility
                >= float(atlas_relative_utility_threshold)
                and atlas_similarity >= float(atlas_minimum_similarity)
            )
            accepted = (
                neural_accepted
                if risk_gate_mode == "neural"
                else atlas_accepted
                if risk_gate_mode == "atlas"
                else neural_accepted and atlas_accepted
            )
            if budget < count and accepted:
                accepted_candidates.append(
                    (
                        atlas_utility
                        if risk_gate_mode in {"atlas", "intersection"}
                        else -float(budget),
                        -int(budget),
                        mask,
                        int(budget),
                        atlas_safe
                        if risk_gate_mode == "atlas"
                        else safe,
                        atlas_catastrophic
                        if risk_gate_mode == "atlas"
                        else catastrophic,
                        hypotheses,
                        relative_lcb,
                        relative_median,
                    )
                )
            if budget == count:
                chosen_safe = safe
                chosen_catastrophic = catastrophic
                chosen_hypotheses = hypotheses
                chosen_relative_lcb = relative_lcb
                chosen_relative_median = relative_median
        if accepted_candidates:
            (
                _,
                _,
                chosen_mask,
                chosen_budget,
                chosen_safe,
                chosen_catastrophic,
                chosen_hypotheses,
                chosen_relative_lcb,
                chosen_relative_median,
            ) = max(accepted_candidates, key=lambda value: value[:2])
            fallback = False
        candidate_diagnostics["selected_budget"] = float(chosen_budget)
        candidate_diagnostics["fallback"] = float(fallback)
        return SLPSSelection(
            selected_mask=chosen_mask.cpu(),
            ordering=ordering.cpu(),
            selected_budget=chosen_budget,
            used_fallback=fallback,
            safe_probability=chosen_safe,
            catastrophic_probability=chosen_catastrophic,
            expected_hypotheses=chosen_hypotheses,
            diagnostics=candidate_diagnostics,
            relative_utility_lcb=chosen_relative_lcb,
            relative_utility_median=chosen_relative_median,
        )


def slps_from_state(
    state: dict, *, device: torch.device | str = "cpu"
) -> SLPSSelector:
    if state.get("schema") != "lafgs_slps_selector":
        raise ValueError("unsupported SLPS selector state")
    model_config = SLPSModelConfig(**state["model_config"])
    expected_feature_names = slps_feature_names(model_config.input_dim)
    if list(state.get("feature_names", ())) != list(expected_feature_names):
        raise ValueError("SLPS serialized feature contract differs")
    if list(state.get("relation_names", ())) != list(RELATION_NAMES):
        raise ValueError("SLPS serialized relation contract differs")
    model = SLPSSelector(
        model_config,
        feature_mean=state["feature_mean"],
        feature_scale=state["feature_scale"],
    ).to(device)
    incompatible = model.load_state_dict(
        state["model_state_dict"], strict=False
    )
    allowed_missing = {
        "strict_weight_raw",
        "solver_weight_raw",
    }
    allowed_missing.update(
        key
        for key in incompatible.missing_keys
        if key.startswith("relative_outcome_head.")
    )
    if (
        set(incompatible.missing_keys) - allowed_missing
        or incompatible.unexpected_keys
    ):
        raise ValueError(
            "SLPS serialized parameters differ from the model contract"
        )
    model.eval()
    model.attach_outcome_atlas(state.get("outcome_atlas"))
    return model
