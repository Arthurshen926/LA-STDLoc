"""Native SuperPoint frontend with the frozen pixel-center convention."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from features.superpoint import SuperPoint, quadratic_subpixel_keypoints
from features.scene_specific_detector import (
    SceneSpecificDetector,
    fuse_scene_reliability,
    mean_candidate_reliability,
)
from features.photometric import canonicalize_image, validate_photometric_contract
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
    detector_ranks: torch.Tensor | None = None


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
        nms_radius: int = 4,
        subpixel_keypoints: bool = False,
        subpixel_geometry_only: bool = False,
        subpixel_maximum_offset: float = 0.5,
        metric: SharedLowRankMetric | None = None,
        context_adapter: MapConsistentContextAdapter | None = None,
        photometric_contract: dict | None = None,
        scene_detector: SceneSpecificDetector | None = None,
        scene_detector_strength: float = 1.0,
        scene_detector_abstain_threshold: float | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.keypoint_count = int(keypoint_count)
        if int(nms_radius) < 0:
            raise ValueError("SuperPoint NMS radius must be non-negative")
        self.nms_radius = int(nms_radius)
        self.subpixel_keypoints = bool(subpixel_keypoints)
        self.subpixel_geometry_only = bool(subpixel_geometry_only)
        self.subpixel_maximum_offset = float(subpixel_maximum_offset)
        if self.subpixel_keypoints and self.subpixel_geometry_only:
            raise ValueError("select one SuperPoint subpixel mode")
        if not 0.0 <= self.subpixel_maximum_offset <= 0.5:
            raise ValueError("SuperPoint subpixel maximum offset is invalid")
        self.model = SuperPoint().to(self.device).eval()
        self.model.nms_radius = self.nms_radius
        self.metric = metric.to(self.device).eval() if metric is not None else None
        self.context_adapter = (
            context_adapter.to(self.device).eval()
            if context_adapter is not None
            else None
        )
        self.photometric_contract = (
            validate_photometric_contract(photometric_contract)
            if photometric_contract is not None
            else None
        )
        self.scene_detector = (
            scene_detector.to(self.device).eval()
            if scene_detector is not None
            else None
        )
        if not 0.0 <= float(scene_detector_strength) <= 1.0:
            raise ValueError("scene detector strength must be in [0,1]")
        self.scene_detector_strength = float(scene_detector_strength)
        if scene_detector_abstain_threshold is not None and not 0.0 <= float(scene_detector_abstain_threshold) <= 1.0:
            raise ValueError("scene detector abstain threshold must be in [0,1]")
        self.scene_detector_abstain_threshold = (
            None if scene_detector_abstain_threshold is None
            else float(scene_detector_abstain_threshold)
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
        if self.photometric_contract is not None:
            image = canonicalize_image(image, self.photometric_contract)
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
        subpixel_score_map = None
        if self.scene_detector is not None:
            dense_descriptors, native_scores = self.model._dense_outputs(image)
            detector_logits = self.scene_detector(
                dense_descriptors, output_hw=(height, width)
            )
            native_sparse = None
            activate = True
            if self.scene_detector_abstain_threshold is not None:
                native_sparse = self.model._sparse_from_dense(
                    dense_descriptors, native_scores, top_k=self.keypoint_count,
                    detection_threshold=0.0,
                    subpixel_refinement=self.subpixel_keypoints,
                )[0]
                activate = bool(
                    mean_candidate_reliability(
                        native_sparse["keypoints"], detector_logits[0]
                    ) >= self.scene_detector_abstain_threshold
                )
            if activate:
                detector_scores = fuse_scene_reliability(
                    native_scores, detector_logits,
                    strength=self.scene_detector_strength,
                )
                sparse = self.model._sparse_from_dense(
                    dense_descriptors, detector_scores,
                    top_k=self.keypoint_count, detection_threshold=0.0,
                    subpixel_refinement=self.subpixel_keypoints,
                )[0]
                subpixel_score_map = detector_scores[0]
            else:
                sparse = native_sparse
                subpixel_score_map = native_scores[0]
        elif self.context_adapter is None:
            if self.subpixel_geometry_only:
                sparse_batch, dense_outputs = self.model.detectAndComputeWithDense(
                    image,
                    top_k=self.keypoint_count,
                    subpixel_refinement=False,
                )
                sparse = sparse_batch[0]
                subpixel_score_map = dense_outputs[1][0, 0]
            else:
                sparse = self.model.detectAndCompute(
                    image,
                    top_k=self.keypoint_count,
                    subpixel_refinement=self.subpixel_keypoints,
                )[0]
        else:
            sparse_batch, dense_outputs = self.model.detectAndComputeWithDense(
                image,
                top_k=self.keypoint_count,
                subpixel_refinement=self.subpixel_keypoints,
            )
            sparse = sparse_batch[0]
            dense_descriptors = dense_outputs[0][0]
            subpixel_score_map = dense_outputs[1][0, 0]
        keypoints = sparse["keypoints"]
        if self.subpixel_geometry_only:
            if subpixel_score_map is None:
                raise RuntimeError("subpixel geometry score map is unavailable")
            keypoints = quadratic_subpixel_keypoints(
                keypoints,
                subpixel_score_map,
                maximum_offset=self.subpixel_maximum_offset,
            )
        scores = sparse["keypoint_scores"]
        descriptors = F.normalize(sparse["descriptors"], dim=1)
        detector_ranks = torch.arange(
            keypoints.shape[0], device=keypoints.device, dtype=torch.long
        )
        if resized_mask is not None:
            keep = sample_mask(resized_mask, keypoints)
            keypoints = keypoints[keep]
            scores = scores[keep]
            descriptors = descriptors[keep]
            detector_ranks = detector_ranks[keep]
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
        return SparseFeatures(
            keypoints,
            scores,
            descriptors,
            (height, width),
            detector_ranks,
        )
