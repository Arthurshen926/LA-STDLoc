"""Calibrated confidence for an unchanged native sparse candidate graph.

The native ULF-parity frontend deliberately emits one cosine top-1 landmark
per SuperPoint keypoint.  This module predicts a *sampling confidence* for
that already-selected pair.  It never selects, suppresses, or replaces a
candidate; callers may only use the returned value to order a robust solver.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch


STATE_VERSION = 1
FEATURE_NAMES = (
    "top1_cosine",
    "top1_top2_margin",
    "topk_entropy",
    "false_attractor_rate",
    "log_incoming_count",
    "keypoint_score",
)


def _as_vector(value, *, name, device, dtype):
    value = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def build_native_matchability_features(
    topk_scores,
    topk_indices,
    *,
    false_attractor_rate,
    incoming_count,
    keypoint_scores,
    entropy_temperature=0.05,
):
    """Build fixed, candidate-preserving confidence features.

    ``topk_indices[:, 0]`` must already be the deployed top-1 landmark.  The
    extra retrieval ranks are only context for confidence estimation; they are
    not exposed to candidate selection or PnP as alternate correspondences.
    """
    scores = torch.as_tensor(topk_scores)
    indices = torch.as_tensor(topk_indices, device=scores.device, dtype=torch.long)
    if scores.ndim != 2 or indices.shape != scores.shape:
        raise ValueError("topk_scores and topk_indices must be equal 2D tensors")
    if scores.shape[1] < 2:
        raise ValueError("native matchability requires at least top-2 retrieval")
    if not torch.isfinite(scores).all():
        raise ValueError("topk_scores must be finite")
    temperature = float(entropy_temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("entropy_temperature must be finite and positive")

    dtype = scores.dtype if scores.is_floating_point() else torch.float32
    scores = scores.to(dtype=dtype)
    false_rate = _as_vector(
        false_attractor_rate,
        name="false_attractor_rate",
        device=scores.device,
        dtype=dtype,
    )
    incoming = _as_vector(
        incoming_count,
        name="incoming_count",
        device=scores.device,
        dtype=dtype,
    )
    keypoint = _as_vector(
        keypoint_scores,
        name="keypoint_scores",
        device=scores.device,
        dtype=dtype,
    )
    if keypoint.numel() != scores.shape[0]:
        raise ValueError("keypoint_scores must have one value per retrieved row")
    top1_indices = indices[:, 0]
    if top1_indices.numel() and (
        int(top1_indices.min()) < 0
        or int(top1_indices.max()) >= false_rate.numel()
        or int(top1_indices.max()) >= incoming.numel()
    ):
        raise ValueError("top-1 landmark indices exceed matchability statistics")

    scaled = (scores - scores[:, :1]) / temperature
    probability = torch.softmax(scaled, dim=1)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=1)
    entropy = entropy / math.log(float(scores.shape[1]))
    return torch.stack(
        (
            scores[:, 0],
            scores[:, 0] - scores[:, 1],
            entropy,
            false_rate[top1_indices].clamp(0.0, 1.0),
            torch.log1p(incoming[top1_indices].clamp_min(0.0)),
            keypoint,
        ),
        dim=1,
    )


def validate_native_matchability_state(state, *, landmark_count=None):
    """Validate the serializable logistic-confidence state schema."""
    if not isinstance(state, Mapping):
        raise ValueError("native matchability state must be a mapping")
    if int(state.get("version", -1)) != STATE_VERSION:
        raise ValueError(
            f"unsupported native matchability state version: {state.get('version')!r}"
        )
    names = tuple(state.get("feature_names", ()))
    if names != FEATURE_NAMES:
        raise ValueError(
            "native matchability feature schema mismatch: "
            f"expected={FEATURE_NAMES!r} got={names!r}"
        )
    expected = len(FEATURE_NAMES)
    for name in ("feature_mean", "feature_std", "weights"):
        value = torch.as_tensor(state.get(name, []), dtype=torch.float32).reshape(-1)
        if value.numel() != expected or not torch.isfinite(value).all():
            raise ValueError(f"native matchability {name} must be finite [{expected}]")
        if name == "feature_std" and bool((value <= 0.0).any()):
            raise ValueError("native matchability feature_std must be positive")
    bias = torch.as_tensor(state.get("bias", float("nan")), dtype=torch.float32)
    if bias.numel() != 1 or not torch.isfinite(bias).all():
        raise ValueError("native matchability bias must be finite scalar")
    topk = int(state.get("topk", 0))
    if topk < 2:
        raise ValueError("native matchability topk must be at least 2")
    temperature = float(state.get("entropy_temperature", float("nan")))
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("native matchability entropy_temperature must be positive")
    floor = float(state.get("confidence_floor", float("nan")))
    if not math.isfinite(floor) or not 0.0 <= floor < 1.0:
        raise ValueError("native matchability confidence_floor must be in [0, 1)")
    rates = torch.as_tensor(state.get("landmark_false_attractor_rate", []))
    incoming = torch.as_tensor(state.get("landmark_incoming_count", []))
    if rates.ndim != 1 or incoming.ndim != 1 or rates.shape != incoming.shape:
        raise ValueError("native matchability landmark statistics must be equal vectors")
    if landmark_count is not None and rates.numel() != int(landmark_count):
        raise ValueError(
            "native matchability landmark statistic count does not match active bank"
        )
    if not torch.isfinite(rates).all() or not torch.isfinite(incoming).all():
        raise ValueError("native matchability landmark statistics must be finite")
    return state


def calibrated_native_matchability(features, state, *, confidence_floor=None):
    """Return monotonic solver confidences from the validated logistic model."""
    validate_native_matchability_state(state)
    features = torch.as_tensor(features)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            "native matchability features must have shape "
            f"[N, {len(FEATURE_NAMES)}]"
        )
    dtype = features.dtype if features.is_floating_point() else torch.float32
    device = features.device
    mean = torch.as_tensor(state["feature_mean"], device=device, dtype=dtype)
    std = torch.as_tensor(state["feature_std"], device=device, dtype=dtype)
    weights = torch.as_tensor(state["weights"], device=device, dtype=dtype)
    bias = torch.as_tensor(state["bias"], device=device, dtype=dtype).reshape(())
    logits = ((features.to(dtype=dtype) - mean) / std) @ weights + bias
    probability = torch.sigmoid(logits)
    floor = float(state["confidence_floor"] if confidence_floor is None else confidence_floor)
    if not math.isfinite(floor) or not 0.0 <= floor < 1.0:
        raise ValueError("confidence_floor must be in [0, 1)")
    # The floor deliberately preserves every candidate and ranking order.
    return floor + (1.0 - floor) * probability
