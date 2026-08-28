"""Query-only causal action gate for the render-trained scene detector."""

from __future__ import annotations

import torch
from torch import nn


FEATURE_NAMES = (
    "mean_native_reliability",
    "p10_native_reliability",
    "std_native_reliability",
    "fraction_native_reliability_ge_05",
    "mean_native_keypoint_score",
    "std_native_keypoint_score",
    "detector_native_overlap",
)


class SceneActionGate(nn.Module):
    """A standardized linear abstention gate with an auditable decision."""

    def __init__(self, feature_mean: torch.Tensor, feature_std: torch.Tensor) -> None:
        super().__init__()
        mean = torch.as_tensor(feature_mean).float().reshape(-1)
        std = torch.as_tensor(feature_std).float().reshape(-1)
        if mean.numel() != len(FEATURE_NAMES) or std.shape != mean.shape:
            raise ValueError("scene action gate statistics differ from its feature contract")
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_std", std.clamp_min(1e-6))
        self.linear = nn.Linear(len(FEATURE_NAMES), 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(features).float()
        if value.shape[-1] != len(FEATURE_NAMES):
            raise ValueError("scene action gate feature dimension differs")
        normalized = (value - self.feature_mean) / self.feature_std
        return self.linear(normalized)[..., 0]


def feature_tensor(record: dict) -> torch.Tensor:
    return torch.tensor([float(record[name]) for name in FEATURE_NAMES])


def query_action_features(
    *,
    native_keypoints: torch.Tensor,
    native_scores: torch.Tensor,
    detector_keypoints: torch.Tensor,
    detector_logits: torch.Tensor,
) -> torch.Tensor:
    points = torch.as_tensor(native_keypoints).float().reshape(-1, 2)
    scores = torch.as_tensor(native_scores, device=points.device).float().reshape(-1)
    detector_points = torch.as_tensor(detector_keypoints, device=points.device).float().reshape(-1, 2)
    logits = torch.as_tensor(detector_logits, device=points.device).float()
    while logits.ndim > 2 and logits.shape[0] == 1:
        logits = logits.squeeze(0)
    if logits.ndim != 2 or points.shape[0] != scores.numel():
        raise ValueError("scene action gate query evidence does not align")
    native_xy = points.round().long()
    detector_xy = detector_points.round().long()
    for xy in (native_xy, detector_xy):
        xy[:, 0].clamp_(0, logits.shape[1] - 1)
        xy[:, 1].clamp_(0, logits.shape[0] - 1)
    reliability = torch.sigmoid(logits[native_xy[:, 1], native_xy[:, 0]])
    native_linear = native_xy[:, 1] * logits.shape[1] + native_xy[:, 0]
    detector_linear = detector_xy[:, 1] * logits.shape[1] + detector_xy[:, 0]
    overlap = torch.isin(detector_linear, native_linear).float().mean()
    return torch.stack((
        reliability.mean(),
        torch.quantile(reliability, 0.1),
        reliability.std(),
        (reliability >= 0.5).float().mean(),
        scores.mean(),
        scores.std(),
        overlap,
    ))


def load_scene_action_gate(
    checkpoint: dict, *, map_sha256: str, detector_sha256: str
) -> SceneActionGate:
    if not (
        checkpoint.get("schema") == "lafgs_v12_scene_action_gate"
        and checkpoint.get("loo_used") is False
        and checkpoint.get("uses_test_queries") is False
        and checkpoint.get("feature_names") == list(FEATURE_NAMES)
        and checkpoint.get("map_sha256") == map_sha256
        and checkpoint.get("detector_sha256") == detector_sha256
    ):
        raise ValueError("scene action gate lineage differs")
    model = SceneActionGate(checkpoint["feature_mean"], checkpoint["feature_std"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval()
