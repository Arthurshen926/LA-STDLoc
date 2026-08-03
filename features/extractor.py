"""Frozen native SuperPoint frontend used throughout the paper pipeline."""

from __future__ import annotations

import torch
from torch import nn

from features.superpoint import SuperPoint


class FeatureExtractor(nn.Module):
    feature_dim = 256

    def __init__(self, feature_type: str = "sp") -> None:
        super().__init__()
        if str(feature_type).lower() not in {"sp", "superpoint"}:
            raise ValueError("The LaFGS release supports only native SuperPoint")
        self.feature_type = "sp"
        self.model = SuperPoint().eval()

    @torch.no_grad()
    def forward(self, image):
        features, scores = self.model(image)
        return {"feature_map": features, "scores": scores}

    @torch.no_grad()
    def detectAndCompute(self, image, top_k=None, detection_threshold=None):
        return self.model.detectAndCompute(
            image, top_k=top_k, detection_threshold=detection_threshold
        )

    @torch.no_grad()
    def detectAndComputeDense(self, image):
        return self.model.detectAndComputeDense(image)

    @torch.no_grad()
    def detectAndComputeWithDense(self, image, top_k=None, detection_threshold=None):
        return self.model.detectAndComputeWithDense(
            image, top_k=top_k, detection_threshold=detection_threshold
        )
