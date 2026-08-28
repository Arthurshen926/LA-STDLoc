"""Minimal scene-specific keypoint detector for the clean-Anchor mainline.

The head consumes the frozen stride-8 SuperPoint descriptor map.  It changes
only keypoint allocation: descriptors are still sampled from the unmodified
SuperPoint map and localization remains one global Top-1 plus one PoseLib call.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


CHECKPOINT_SCHEMA = "lafgs_scene_specific_detector"
CHECKPOINT_VERSION = 1


class SceneSpecificDetector(nn.Module):
    """Small residual heatmap head over frozen SuperPoint features."""

    def __init__(self, feature_dim: int = 256, hidden_dim: int = 64) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, 1, 1),
        )

    def forward(
        self, features: torch.Tensor, *, output_hw: tuple[int, int] | None = None
    ) -> torch.Tensor:
        value = torch.as_tensor(features)
        if value.ndim != 4 or value.shape[1] != self.feature_dim:
            raise ValueError(
                f"features must have shape [B,{self.feature_dim},H,W]"
            )
        logits = self.net(value)
        if output_hw is not None and tuple(logits.shape[-2:]) != tuple(output_hw):
            logits = F.interpolate(
                logits, size=tuple(map(int, output_hw)), mode="bilinear",
                align_corners=False,
            )
        return logits[:, 0]


def tri_state_detector_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    positive_weight: float | None = None,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Balanced BCE for labels in ``{-1 ignore, 0 negative, 1 positive}``."""

    prediction = torch.as_tensor(logits).float()
    target = torch.as_tensor(labels, device=prediction.device)
    if prediction.shape != target.shape:
        raise ValueError("detector logits and tri-state labels must align")
    if not bool(((target == -1) | (target == 0) | (target == 1)).all()):
        raise ValueError("detector labels must be -1, 0, or 1")
    valid = target >= 0
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    binary = target[valid].float()
    if positive_weight is None:
        positives = binary.sum()
        negatives = binary.numel() - positives
        weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 50.0)
    else:
        if float(positive_weight) <= 0:
            raise ValueError("positive_weight must be positive")
        weight = prediction.new_tensor(float(positive_weight))
    loss = F.binary_cross_entropy_with_logits(
        prediction[valid], binary, pos_weight=weight, reduction="none"
    )
    if sample_weight is not None:
        contribution = torch.as_tensor(
            sample_weight, device=prediction.device, dtype=prediction.dtype
        )
        if contribution.shape != prediction.shape or bool((contribution < 0).any()):
            raise ValueError("detector contribution weights must align and be non-negative")
        contribution = contribution[valid]
        return (loss * contribution).sum() / contribution.sum().clamp_min(1e-8)
    return loss.mean()


def detector_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Threshold-free ranking metrics on supervised pixels."""

    scores = torch.sigmoid(torch.as_tensor(logits).float()).reshape(-1)
    target = torch.as_tensor(labels, device=scores.device).reshape(-1)
    valid = target >= 0
    positive = target == 1
    negative = target == 0
    if not bool(valid.any()) or not bool(positive.any()) or not bool(negative.any()):
        return {"positive_mean": float("nan"), "negative_mean": float("nan"), "separation": float("nan")}
    positive_mean = scores[positive].mean()
    negative_mean = scores[negative].mean()
    return {
        "positive_mean": float(positive_mean),
        "negative_mean": float(negative_mean),
        "separation": float(positive_mean - negative_mean),
    }


def fuse_scene_reliability(
    native_scores: torch.Tensor,
    detector_logits: torch.Tensor,
    *,
    strength: float = 1.0,
) -> torch.Tensor:
    """Boundedly modulate native peaks without inventing new extrema."""

    scores = torch.as_tensor(native_scores).float()
    logits = torch.as_tensor(detector_logits, device=scores.device).float()
    if scores.shape != logits.shape:
        raise ValueError("native scores and scene reliability logits must align")
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("scene reliability strength must be in [0,1]")
    reliability = torch.sigmoid(logits)
    multiplier = (1.0 - float(strength)) + float(strength) * reliability
    return scores * multiplier


def protected_scene_candidate_indices(
    *,
    keypoints: torch.Tensor,
    native_scores: torch.Tensor,
    detector_logits: torch.Tensor,
    output_count: int,
    protected_native_count: int,
) -> torch.Tensor:
    """Keep a native core and use learned reliability only for the tail budget.

    ``keypoints`` and ``native_scores`` are already native-NMS candidates in
    descending score order.  Consequently this policy runs no second detector
    or NMS and has an explicit maximum replacement count.
    """

    points = torch.as_tensor(keypoints).float().reshape(-1, 2)
    scores = torch.as_tensor(native_scores, device=points.device).float().reshape(-1)
    logits = torch.as_tensor(detector_logits, device=points.device).float()
    while logits.ndim > 2 and logits.shape[0] == 1:
        logits = logits.squeeze(0)
    total = int(output_count)
    protected = int(protected_native_count)
    if logits.ndim != 2 or points.shape[0] != scores.numel():
        raise ValueError("protected scene candidates do not align")
    if not 0 <= protected <= total <= points.shape[0]:
        raise ValueError("protected scene candidate budget is invalid")
    xy = points.round().long()
    xy[:, 0].clamp_(0, logits.shape[1] - 1)
    xy[:, 1].clamp_(0, logits.shape[0] - 1)
    reliability = torch.sigmoid(logits[xy[:, 1], xy[:, 0]])
    tail_score = scores[protected:] * reliability[protected:]
    fill = total - protected
    tail = torch.argsort(tail_score, descending=True, stable=True)[:fill] + protected
    return torch.cat(
        (torch.arange(protected, device=points.device, dtype=torch.long), tail)
    )


def mean_candidate_reliability(
    keypoints: torch.Tensor, detector_logits: torch.Tensor
) -> torch.Tensor:
    """Query-level confidence from already selected native candidates."""

    points = torch.as_tensor(keypoints).float().reshape(-1, 2)
    logits = torch.as_tensor(detector_logits, device=points.device).float()
    while logits.ndim > 2 and logits.shape[0] == 1:
        logits = logits.squeeze(0)
    if logits.ndim != 2:
        raise ValueError("detector logits must be a heatmap")
    if points.shape[0] == 0:
        return logits.new_tensor(0.0)
    xy = points.round().long()
    xy[:, 0].clamp_(0, logits.shape[1] - 1)
    xy[:, 1].clamp_(0, logits.shape[0] - 1)
    return torch.sigmoid(logits[xy[:, 1], xy[:, 0]]).mean()


def load_scene_detector_checkpoint(
    checkpoint: dict, *, map_sha256: str | None = None
) -> SceneSpecificDetector:
    """Load a detector while enforcing its clean-map lineage."""

    if checkpoint.get("schema") != CHECKPOINT_SCHEMA or int(checkpoint.get("version", -1)) != CHECKPOINT_VERSION:
        raise ValueError("invalid scene-specific detector checkpoint")
    lineage = checkpoint.get("lineage", {})
    if lineage.get("uses_test_rgb") is not False:
        raise ValueError("scene detector lineage must explicitly exclude test RGB")
    if map_sha256 is not None and lineage.get("anchor_map_sha256") != str(map_sha256):
        raise ValueError("scene detector was trained for a different Anchor map")
    model = SceneSpecificDetector(
        feature_dim=int(checkpoint["model"]["feature_dim"]),
        hidden_dim=int(checkpoint["model"]["hidden_dim"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model
