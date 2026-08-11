"""Native SuperPoint frontend with the frozen pixel-center convention."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from features.superpoint import SuperPoint
from map_learning.context_metric import (
    MapConsistentContextAdapter,
    dense_context_tokens,
)
from map_learning.metric import SharedLowRankMetric


@dataclass(frozen=True)
class SparseFeatures:
    keypoints: torch.Tensor
    scores: torch.Tensor
    descriptors: torch.Tensor
    image_hw: tuple[int, int]


def sample_mask(mask: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
    mask = torch.as_tensor(mask, device=keypoints.device).bool()
    if mask.ndim != 2:
        raise ValueError("valid mask must be two-dimensional")
    xy = keypoints.round().long()
    valid = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < mask.shape[1])
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < mask.shape[0])
    )
    result = torch.zeros_like(valid)
    result[valid] = mask[xy[valid, 1], xy[valid, 0]]
    return result


class NativeSuperPointFrontend:
    def __init__(
        self,
        *,
        device: torch.device | str = "cuda",
        keypoint_count: int = 2048,
        metric: SharedLowRankMetric | None = None,
        context_adapter: MapConsistentContextAdapter | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.keypoint_count = int(keypoint_count)
        self.model = SuperPoint().to(self.device).eval()
        self.metric = metric.to(self.device).eval() if metric is not None else None
        self.context_adapter = (
            context_adapter.to(self.device).eval()
            if context_adapter is not None
            else None
        )
        if self.metric is not None and self.context_adapter is not None:
            raise ValueError(
                "shared metric and context adapter are separate descriptor protocols"
            )

    @torch.inference_mode()
    def __call__(
        self, image: torch.Tensor, *, valid_mask: torch.Tensor | None = None
    ) -> SparseFeatures:
        image = torch.as_tensor(image, device=self.device).float()
        if image.ndim == 3:
            image = image[None]
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
            raise ValueError("image must have shape [3,H,W] or [1,3,H,W]")
        height, width = map(int, image.shape[-2:])
        resized_mask = None
        if valid_mask is not None:
            resized_mask = torch.as_tensor(
                valid_mask, device=self.device, dtype=torch.float32
            )
            while resized_mask.ndim > 2:
                resized_mask = resized_mask.squeeze(0)
            if resized_mask.shape != (height, width):
                resized_mask = F.interpolate(
                    resized_mask[None, None],
                    size=(height, width),
                    mode="nearest",
                )[0, 0]
            resized_mask = resized_mask.bool()
            image = image * resized_mask[None, None].to(image.dtype)

        dense_descriptors = None
        if self.context_adapter is None:
            sparse = self.model.detectAndCompute(
                image, top_k=self.keypoint_count
            )[0]
        else:
            sparse_batch, dense_outputs = self.model.detectAndComputeWithDense(
                image, top_k=self.keypoint_count
            )
            sparse = sparse_batch[0]
            dense_descriptors = dense_outputs[0][0]
        keypoints = sparse["keypoints"]
        scores = sparse["keypoint_scores"]
        descriptors = F.normalize(sparse["descriptors"], dim=1)
        if resized_mask is not None:
            keep = sample_mask(resized_mask, keypoints)
            keypoints = keypoints[keep]
            scores = scores[keep]
            descriptors = descriptors[keep]
        if self.context_adapter is not None and descriptors.numel():
            if resized_mask is None:
                dense_valid = torch.ones(
                    dense_descriptors.shape[-2:],
                    dtype=torch.bool,
                    device=self.device,
                )
            else:
                dense_valid = F.interpolate(
                    resized_mask[None, None].float(),
                    size=dense_descriptors.shape[-2:],
                    mode="nearest",
                )[0, 0].bool()
            dense_descriptors = dense_descriptors * dense_valid[None]
            context = dense_context_tokens(
                dense_descriptors,
                keypoints,
                (height, width),
                valid_mask=dense_valid,
                kernels=self.context_adapter.context_kernels,
            )
            if self.context_adapter.context_mode == "local_only":
                context[:, -1] = 0.0
            elif self.context_adapter.context_mode == "global_only":
                context[:, :-1] = 0.0
            elif self.context_adapter.context_mode == "zero":
                context.zero_()
            descriptors, _ = self.context_adapter(descriptors, context)
        if self.metric is not None and descriptors.numel():
            descriptors, _ = self.metric(descriptors)
        return SparseFeatures(keypoints, scores, descriptors, (height, width))
