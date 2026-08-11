"""Official-compatible SuperPoint FeatureBooster inference utilities.

The network structure is adapted from SJTU-ViSYS/FeatureBooster under the
Apache License 2.0.  We intentionally preserve the upstream module names so
the official ``SuperPoint+Boost-F.pth`` state dictionary loads strictly.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


FEATUREBOOSTER_ENVIRONMENT_VARIABLE = "LAFGS_FEATUREBOOSTER_WEIGHTS"
FEATUREBOOSTER_DEFAULT_PATH = Path("~/.cache/lafgs/SuperPoint+Boost-F.pth")
SUPERPOINT_BOOST_F_SHA256 = (
    "5334d9aa861e877a2b99baff0d682e1ac8a749cdd65eb1d4b8bd0a8bb8bf0359"
)
SUPERPOINT_BOOST_F_CONFIG = {
    "keypoint_dim": 3,
    "keypoint_encoder": [32, 64, 128, 256],
    "descriptor_encoder": [256, 256],
    "descriptor_dim": 256,
    "Attentional_layers": 9,
    "last_activation": None,
    "l2_normalization": True,
    "output_dim": 256,
}


def _mlp(channels: Sequence[int], do_bn: bool = False) -> nn.Module:
    layers: list[nn.Module] = []
    for index in range(1, len(channels)):
        layers.append(nn.Linear(channels[index - 1], channels[index]))
        if index < len(channels) - 1:
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[index]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class KeypointEncoder(nn.Module):
    """Encode normalized keypoint geometry with the upstream MLP."""

    def __init__(
        self,
        keypoint_dim: int,
        feature_dim: int,
        layers: Sequence[int],
        dropout: bool = False,
        p: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = _mlp([keypoint_dim, *layers, feature_dim])
        self.use_dropout = dropout
        self.dropout = nn.Dropout(p=p)

    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(keypoints)
        return self.dropout(encoded) if self.use_dropout else encoded


class DescriptorEncoder(nn.Module):
    """Apply the upstream residual descriptor MLP."""

    def __init__(
        self,
        feature_dim: int,
        layers: Sequence[int],
        dropout: bool = False,
        p: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = _mlp([feature_dim, *layers, feature_dim])
        self.use_dropout = dropout
        self.dropout = nn.Dropout(p=p)

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(descriptors)
        if self.use_dropout:
            encoded = self.dropout(encoded)
        return descriptors + encoded


class AFTAttention(nn.Module):
    """Attention-free attention used by the official Boost-F checkpoint."""

    def __init__(
        self, d_model: int, dropout: bool = False, p: float = 0.1
    ) -> None:
        super().__init__()
        self.dim = d_model
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.use_dropout = dropout
        self.dropout = nn.Dropout(p=p)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features
        query = torch.sigmoid(self.query(features))
        key = torch.softmax(self.key(features).T, dim=-1).T
        value = self.value(features)
        features = query * (key * value).sum(dim=-2, keepdim=True)
        features = self.proj(features)
        if self.use_dropout:
            features = self.dropout(features)
        return self.layer_norm(features + residual)


class PositionwiseFeedForward(nn.Module):
    def __init__(
        self, feature_dim: int, dropout: bool = False, p: float = 0.1
    ) -> None:
        super().__init__()
        self.mlp = _mlp([feature_dim, feature_dim * 2, feature_dim])
        self.layer_norm = nn.LayerNorm(feature_dim, eps=1e-6)
        self.use_dropout = dropout
        self.dropout = nn.Dropout(p=p)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features
        features = self.mlp(features)
        if self.use_dropout:
            features = self.dropout(features)
        return self.layer_norm(features + residual)


class AttentionalLayer(nn.Module):
    def __init__(
        self, feature_dim: int, dropout: bool = False, p: float = 0.1
    ) -> None:
        super().__init__()
        self.attn = AFTAttention(feature_dim, dropout=dropout, p=p)
        self.ffn = PositionwiseFeedForward(feature_dim, dropout=dropout, p=p)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.attn(features))


class AttentionalNN(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        layer_num: int,
        dropout: bool = False,
        p: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                AttentionalLayer(feature_dim, dropout=dropout, p=p)
                for _ in range(layer_num)
            ]
        )

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            descriptors = layer(descriptors)
        return descriptors


class FeatureBooster(nn.Module):
    """FeatureBooster with parameter names compatible with upstream weights."""

    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        self.config = {**SUPERPOINT_BOOST_F_CONFIG, **(config or {})}
        descriptor_dim = int(self.config["descriptor_dim"])
        self.use_kenc = True
        self.use_cross = True
        self.kenc = KeypointEncoder(
            int(self.config["keypoint_dim"]),
            descriptor_dim,
            self.config["keypoint_encoder"],
        )
        self.denc = DescriptorEncoder(
            descriptor_dim, self.config["descriptor_encoder"]
        )
        self.attn_proj = AttentionalNN(
            descriptor_dim, int(self.config["Attentional_layers"])
        )
        self.final_proj = nn.Linear(descriptor_dim, int(self.config["output_dim"]))
        self.use_dropout = False
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(descriptor_dim, eps=1e-6)
        self.last_activation = None

    def forward(
        self, descriptors: torch.Tensor, keypoint_properties: torch.Tensor
    ) -> torch.Tensor:
        descriptors = self.denc(descriptors)
        descriptors = descriptors + self.kenc(keypoint_properties)
        descriptors = self.attn_proj(self.layer_norm(descriptors))
        descriptors = self.final_proj(descriptors)
        return F.normalize(descriptors, dim=-1)


def normalize_keypoint_properties(
    keypoints: torch.Tensor,
    scores: torch.Tensor,
    image_hw: Sequence[int],
) -> torch.Tensor:
    """Create the official normalized ``(x, y, score)`` property tensor."""
    keypoints = torch.as_tensor(keypoints).float()
    scores = torch.as_tensor(scores).float().reshape(-1)
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError("keypoints must have shape [N, 2]")
    if keypoints.shape[0] != scores.numel():
        raise ValueError("keypoints and scores must have the same row count")
    if len(image_hw) != 2:
        raise ValueError("image_hw must contain height and width")
    height, width = (int(value) for value in image_hw)
    if height <= 0 or width <= 0:
        raise ValueError("image height and width must be positive")
    center = keypoints.new_tensor([width / 2.0, height / 2.0])
    scale = float(max(height, width)) * 0.7
    return torch.cat(((keypoints - center) / scale, scores[:, None]), dim=1)


def resolve_featurebooster_weights(path: str | Path | None = None) -> Path:
    """Resolve and verify the official SuperPoint Boost-F checkpoint."""
    if path is None:
        configured = os.environ.get(FEATUREBOOSTER_ENVIRONMENT_VARIABLE)
        path = configured if configured else FEATUREBOOSTER_DEFAULT_PATH
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"official FeatureBooster weights not found at {resolved}; set "
            f"{FEATUREBOOSTER_ENVIRONMENT_VARIABLE} or pass an explicit path"
        )
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if digest != SUPERPOINT_BOOST_F_SHA256:
        raise ValueError(
            "unexpected SuperPoint+Boost-F.pth SHA256: "
            f"expected {SUPERPOINT_BOOST_F_SHA256}, got {digest}"
        )
    return resolved


def load_superpoint_boost_f(
    path: str | Path | None,
    *,
    device: torch.device | str,
) -> FeatureBooster:
    """Load the official Boost-F checkpoint with strict key validation."""
    resolved = resolve_featurebooster_weights(path)
    model = FeatureBooster()
    try:
        state = torch.load(resolved, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(resolved, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)
