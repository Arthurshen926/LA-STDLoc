"""Risk-controlled certification for an already fixed correspondence set.

This module deliberately does not learn a correspondence ordering.  It only
decides whether a fixed, previously validated compact set is safe enough to
use for a query; otherwise the caller must fall back to all correspondences.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F


CERTIFIER_FEATURE_NAMES = (
    "input_count_log_fraction",
    "solver_probability_mean",
    "solver_probability_p10",
    "solver_probability_p90",
    "strict_probability_mean",
    "harmful_probability_mean",
    "strict_lcb_per_selected",
    "log_expected_basis_normalized",
    "dependency_group_fraction",
    "selected_2d_occupancy_fraction",
    "selected_2d_entropy_normalized",
    "selected_2d_max_cell_fraction",
    "selected_3d_voxel_per_match",
    "selected_3d_max_voxel_fraction",
    "strict_to_solver_ratio",
    "harmful_to_solver_ratio",
    "solver_probability_spread",
)


_SPARSE_KEYS = (
    "sparse_diag_pose_sufficient_input_count",
    "sparse_diag_pose_sufficient_selected_count",
    "sparse_diag_pose_sufficient_probability_mean",
    "sparse_diag_pose_sufficient_probability_p10",
    "sparse_diag_pose_sufficient_probability_p90",
    "sparse_diag_pose_sufficient_strict_probability_mean",
    "sparse_diag_pose_sufficient_harmful_probability_mean",
    "sparse_diag_pose_sufficient_strict_lcb",
    "sparse_diag_pose_sufficient_log_expected_basis",
    "sparse_diag_pose_sufficient_dependency_group_count",
    "sparse_diag_all_2d_occupancy_frac",
    "sparse_diag_all_2d_entropy_norm",
    "sparse_diag_all_2d_max_cell_frac",
    "sparse_diag_all_3d_voxel_per_match",
    "sparse_diag_all_3d_max_voxel_frac",
)


def fixed_budget_certifier_features(
    sparse_diagnostics: Mapping[str, object],
) -> torch.Tensor:
    """Build scene-independent features available before the PnP solve.

    The ``all_2d`` and ``all_3d`` fields refer to the already fixed compact
    set.  Reprojection, inlier, GT, pose, runtime, and identity/history fields
    are intentionally excluded from this contract.
    """

    missing = [key for key in _SPARSE_KEYS if key not in sparse_diagnostics]
    if missing:
        raise KeyError(f"fixed-budget diagnostics miss keys: {missing}")
    values = {
        key: float(sparse_diagnostics[key]) for key in _SPARSE_KEYS
    }
    if not all(torch.isfinite(torch.tensor(list(values.values())))):
        raise ValueError("fixed-budget diagnostics must be finite")

    input_count = max(
        values["sparse_diag_pose_sufficient_input_count"], 1.0
    )
    selected_count = max(
        values["sparse_diag_pose_sufficient_selected_count"], 1.0
    )
    solver_mean = max(
        values["sparse_diag_pose_sufficient_probability_mean"], 1e-6
    )
    solver_p10 = values[
        "sparse_diag_pose_sufficient_probability_p10"
    ]
    solver_p90 = values[
        "sparse_diag_pose_sufficient_probability_p90"
    ]
    strict_mean = values[
        "sparse_diag_pose_sufficient_strict_probability_mean"
    ]
    harmful_mean = values[
        "sparse_diag_pose_sufficient_harmful_probability_mean"
    ]
    output = torch.tensor(
        (
            torch.log1p(torch.tensor(input_count)).item()
            / torch.log1p(torch.tensor(2048.0)).item(),
            solver_mean,
            solver_p10,
            solver_p90,
            strict_mean,
            harmful_mean,
            values["sparse_diag_pose_sufficient_strict_lcb"]
            / selected_count,
            values["sparse_diag_pose_sufficient_log_expected_basis"]
            / 16.0,
            values[
                "sparse_diag_pose_sufficient_dependency_group_count"
            ]
            / selected_count,
            values["sparse_diag_all_2d_occupancy_frac"],
            values["sparse_diag_all_2d_entropy_norm"],
            values["sparse_diag_all_2d_max_cell_frac"],
            values["sparse_diag_all_3d_voxel_per_match"],
            values["sparse_diag_all_3d_max_voxel_frac"],
            strict_mean / solver_mean,
            harmful_mean / solver_mean,
            solver_p90 - solver_p10,
        ),
        dtype=torch.float32,
    )
    if output.shape != (len(CERTIFIER_FEATURE_NAMES),):
        raise AssertionError("fixed-budget feature contract drifted")
    return output


@dataclass(frozen=True)
class LinearRiskCertifierState:
    feature_names: tuple[str, ...]
    center: list[float]
    scale: list[float]
    weight: list[float]
    bias: float
    unsafe_probability_threshold: float

    def to_dict(self) -> dict:
        return asdict(self)


def _robust_normalizer(features: torch.Tensor) -> tuple[torch.Tensor, ...]:
    center = torch.quantile(features, 0.5, dim=0)
    absolute = (features - center).abs()
    scale = torch.quantile(absolute, 0.75, dim=0)
    standard = features.std(dim=0, unbiased=False)
    scale = torch.where(scale >= 1e-4, scale, standard)
    scale = scale.clamp_min(1e-4)
    normalized = ((features - center) / scale).clamp(-8.0, 8.0)
    return normalized, center, scale


def fit_linear_risk_certifier(
    features: torch.Tensor,
    unsafe_labels: torch.Tensor,
    *,
    steps: int = 800,
    learning_rate: float = 0.03,
    weight_decay: float = 0.02,
    seed: int = 2026,
    device: str | torch.device = "cpu",
) -> LinearRiskCertifierState:
    """Fit the intentionally minimal unsafe-set logistic model."""

    features = torch.as_tensor(features).float()
    labels = torch.as_tensor(unsafe_labels).float().reshape(-1)
    if features.ndim != 2 or features.shape[1] != len(
        CERTIFIER_FEATURE_NAMES
    ):
        raise ValueError("fixed-budget certifier feature shape is invalid")
    if len(features) != len(labels) or len(features) < 2:
        raise ValueError("fixed-budget certifier labels must align")
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("unsafe labels must be binary")
    if int(labels.sum()) in (0, len(labels)):
        raise ValueError("risk fitting requires both label classes")

    normalized, center, scale = _robust_normalizer(features)
    target_device = torch.device(device)
    normalized = normalized.to(target_device)
    labels = labels.to(target_device)
    torch.manual_seed(int(seed))
    model = nn.Linear(features.shape[1], 1).to(target_device)
    nn.init.zeros_(model.weight)
    unsafe_count = labels.sum().clamp_min(1.0)
    safe_count = (1.0 - labels).sum().clamp_min(1.0)
    positive_weight = (safe_count / unsafe_count).detach()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    for _ in range(max(int(steps), 1)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(normalized).reshape(-1)
        loss = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=positive_weight
        )
        loss.backward()
        optimizer.step()
    return LinearRiskCertifierState(
        feature_names=CERTIFIER_FEATURE_NAMES,
        center=center.tolist(),
        scale=scale.tolist(),
        weight=model.weight.detach().cpu().reshape(-1).tolist(),
        bias=float(model.bias.detach().cpu()),
        unsafe_probability_threshold=float("-inf"),
    )


def predict_unsafe_probability(
    state: LinearRiskCertifierState | Mapping[str, object],
    features: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(state, LinearRiskCertifierState):
        state = LinearRiskCertifierState(**state)
    if tuple(state.feature_names) != CERTIFIER_FEATURE_NAMES:
        raise ValueError("fixed-budget certifier feature names differ")
    values = torch.as_tensor(features).float()
    one_row = values.ndim == 1
    values = values.reshape(-1, len(CERTIFIER_FEATURE_NAMES))
    center = values.new_tensor(state.center)
    scale = values.new_tensor(state.scale).clamp_min(1e-4)
    weight = values.new_tensor(state.weight)
    normalized = ((values - center) / scale).clamp(-8.0, 8.0)
    probability = torch.sigmoid(normalized @ weight + float(state.bias))
    return probability[0] if one_row else probability


def certify_fixed_budget(
    state: LinearRiskCertifierState | Mapping[str, object],
    features: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(state, LinearRiskCertifierState):
        state = LinearRiskCertifierState(**state)
    return predict_unsafe_probability(state, features) <= float(
        state.unsafe_probability_threshold
    )


def one_sided_binomial_upper_bound(
    failures: int,
    count: int,
    *,
    confidence: float = 0.95,
) -> float:
    """Conservative one-sided Wilson bound for selective false-safe risk."""

    failures = int(failures)
    count = int(count)
    if count <= 0:
        return 1.0
    if failures < 0 or failures > count:
        raise ValueError("binomial failure count is invalid")
    if not 0.5 < float(confidence) < 1.0:
        raise ValueError("confidence must lie in (0.5, 1)")
    probability = failures / count
    z = NormalDist().inv_cdf(float(confidence))
    z2 = z * z
    denominator = 1.0 + z2 / count
    center = probability + z2 / (2.0 * count)
    radius = z * (
        probability * (1.0 - probability) / count
        + z2 / (4.0 * count * count)
    ) ** 0.5
    return min((center + radius) / denominator, 1.0)


@dataclass(frozen=True)
class RiskCalibration:
    threshold: float
    accepted_count: int
    failure_count: int
    empirical_false_safe_rate: float
    false_safe_upper_bound: float
    feasible: bool


def calibrate_selective_risk(
    unsafe_probabilities: torch.Tensor,
    unsafe_labels: torch.Tensor,
    *,
    risk_limit: float = 0.02,
    confidence: float = 0.95,
) -> RiskCalibration:
    """Choose the largest tied score prefix satisfying the risk bound."""

    probabilities = torch.as_tensor(unsafe_probabilities).double().reshape(-1)
    labels = torch.as_tensor(unsafe_labels).bool().reshape(-1)
    if len(probabilities) != len(labels) or not len(probabilities):
        raise ValueError("risk calibration inputs must be non-empty and align")
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("risk probabilities must be finite")
    if not 0.0 < float(risk_limit) < 1.0:
        raise ValueError("risk limit must lie in (0, 1)")

    best: RiskCalibration | None = None
    for threshold in torch.unique(probabilities, sorted=True).tolist():
        accepted = probabilities <= float(threshold)
        count = int(accepted.sum())
        failures = int(labels[accepted].sum())
        upper = one_sided_binomial_upper_bound(
            failures, count, confidence=confidence
        )
        if upper <= float(risk_limit):
            best = RiskCalibration(
                threshold=float(threshold),
                accepted_count=count,
                failure_count=failures,
                empirical_false_safe_rate=failures / count,
                false_safe_upper_bound=upper,
                feasible=True,
            )
    if best is not None:
        return best
    return RiskCalibration(
        threshold=float("-inf"),
        accepted_count=0,
        failure_count=0,
        empirical_false_safe_rate=0.0,
        false_safe_upper_bound=1.0,
        feasible=False,
    )
