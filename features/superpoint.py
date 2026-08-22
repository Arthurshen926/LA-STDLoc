from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


SUPERPOINT_WEIGHT_SHA256 = (
    "52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e"
)
SUPERPOINT_UPSTREAM_URL = (
    "https://github.com/magicleap/SuperPointPretrainedNetwork/"
    "raw/master/superpoint_v1.pth"
)


def resolve_superpoint_weights() -> Path:
    """Resolve user-supplied upstream weights and enforce frozen parity."""
    environment = os.environ.get("LAFGS_SUPERPOINT_WEIGHTS")
    candidates = (
        [Path(environment).expanduser()] if environment else []
    ) + [
        Path.home() / ".cache/lafgs/superpoint_v1.pth",
        Path(__file__).parent / "weights/superpoint_v1.pth",
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            "SuperPoint weights are not redistributed by LaFGS. Download "
            f"{SUPERPOINT_UPSTREAM_URL}, then set LAFGS_SUPERPOINT_WEIGHTS "
            "or place the file at ~/.cache/lafgs/superpoint_v1.pth."
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SUPERPOINT_WEIGHT_SHA256:
        raise RuntimeError(
            f"SuperPoint weight SHA256 mismatch for {path}: {digest} != "
            f"{SUPERPOINT_WEIGHT_SHA256}"
        )
    return path.resolve()


def batched_nms(scores, nms_radius: int):
    """Apply SuperPoint's two-pass local non-maximum suppression."""
    if nms_radius < 0:
        raise ValueError("nms_radius must be non-negative")

    def max_pool(value):
        return F.max_pool2d(
            value,
            kernel_size=nms_radius * 2 + 1,
            stride=1,
            padding=nms_radius,
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        suppression_mask = max_pool(max_mask.float()) > 0
        suppressed_scores = torch.where(suppression_mask, zeros, scores)
        new_max_mask = suppressed_scores == max_pool(suppressed_scores)
        max_mask = max_mask | (new_max_mask & (~suppression_mask))
    return torch.where(max_mask, scores, zeros)


def render_validity_mask_from_alpha(
    alpha: torch.Tensor,
    *,
    minimum_alpha: float,
    neighborhood_radius: int = 0,
) -> torch.Tensor:
    """Gate detector rows whose support neighborhood leaves rendered content."""

    value = torch.as_tensor(alpha)
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    elif value.ndim == 2:
        value = value[None]
    if value.ndim != 3:
        raise ValueError("alpha must have shape [H,W], [B,H,W], or [B,1,H,W]")
    if not 0.0 <= float(minimum_alpha) <= 1.0:
        raise ValueError("minimum_alpha must be in [0,1]")
    radius = int(neighborhood_radius)
    if radius < 0:
        raise ValueError("neighborhood_radius must be non-negative")
    invalid = (~torch.isfinite(value)) | (value < float(minimum_alpha))
    if radius:
        invalid = F.max_pool2d(
            invalid[:, None].float(),
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
        )[:, 0].bool()
    return ~invalid


def select_top_k_keypoints(keypoints, scores, k):
    if k is None or int(k) >= int(keypoints.shape[0]):
        return keypoints, scores
    scores, indices = torch.topk(scores, int(k), dim=0, sorted=True)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, stride: int = 8):
    """Bilinearly sample a SuperPoint descriptor grid at image-grid indices.

    Sparse SuperPoint keypoints come from the full-resolution score grid, not
    from the stride-8 descriptor grid.  Their physical pixel centers are
    ``index + 0.5``; this is the convention used by the sparse PnP path and
    by ULF-Loc's original sparse frontend.
    """
    batch, channels, height, width = descriptors.shape
    if keypoints.shape[0] != batch:
        raise ValueError("keypoints and descriptors must have the same batch size")
    keypoints = (keypoints + 0.5) / (
        keypoints.new_tensor([width, height]) * float(stride)
    )
    grid = keypoints.mul(2.0).sub(1.0).view(batch, 1, -1, 2)
    sampled = F.grid_sample(
        descriptors,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return F.normalize(sampled.reshape(batch, channels, -1), p=2, dim=1)


def quadratic_subpixel_keypoints(
    keypoints: torch.Tensor,
    score_map: torch.Tensor,
    *,
    maximum_offset: float = 0.5,
) -> torch.Tensor:
    """Refine fixed detector rows by a bounded 3x3 quadratic peak fit.

    The candidate identities and scores stay unchanged.  Only their continuous
    coordinates move, so this can be evaluated as a geometry factor without
    changing detector count, NMS, top-k ordering, or the Track row registry.
    """
    points = torch.as_tensor(keypoints)
    scores = torch.as_tensor(score_map)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("keypoints must have shape [N,2]")
    if scores.ndim != 2:
        raise ValueError("score_map must have shape [H,W]")
    if not (0.0 <= float(maximum_offset) <= 0.5):
        raise ValueError("subpixel maximum offset must be in [0,0.5]")
    refined = points.float().clone()
    if refined.numel() == 0 or float(maximum_offset) == 0.0:
        return refined
    integer = points.long()
    x = integer[:, 0]
    y = integer[:, 1]
    interior = (
        (x > 0)
        & (x + 1 < scores.shape[1])
        & (y > 0)
        & (y + 1 < scores.shape[0])
    )
    rows = torch.nonzero(interior, as_tuple=False).reshape(-1)
    if rows.numel() == 0:
        return refined
    x = x[rows]
    y = y[rows]
    center = scores[y, x].float()
    horizontal = scores[y, x - 1].float() - 2.0 * center + scores[y, x + 1].float()
    vertical = scores[y - 1, x].float() - 2.0 * center + scores[y + 1, x].float()
    dx = torch.zeros_like(center)
    dy = torch.zeros_like(center)
    valid_x = torch.isfinite(horizontal) & (horizontal < -1e-12)
    valid_y = torch.isfinite(vertical) & (vertical < -1e-12)
    dx[valid_x] = 0.5 * (
        scores[y[valid_x], x[valid_x] - 1]
        - scores[y[valid_x], x[valid_x] + 1]
    ) / horizontal[valid_x]
    dy[valid_y] = 0.5 * (
        scores[y[valid_y] - 1, x[valid_y]]
        - scores[y[valid_y] + 1, x[valid_y]]
    ) / vertical[valid_y]
    offset = torch.stack((dx, dy), dim=1).clamp(
        -float(maximum_offset), float(maximum_offset)
    )
    offset = torch.where(torch.isfinite(offset), offset, torch.zeros_like(offset))
    refined[rows] += offset
    return refined


class SuperPoint(nn.Module):
    def __init__(self):
        super().__init__()
        out_channels = 256

        # Kept here rather than in FeatureExtractor so the direct sparse API
        # has the same defaults as ULF-Loc's SuperPoint frontend.
        self.nms_radius = 4
        self.max_num_keypoints = None
        self.detection_threshold = 0.0
        self.remove_borders = 4

        self.transform = transforms.Grayscale(num_output_channels=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, out_channels,
            kernel_size=1, stride=1, padding=0)

        path = resolve_superpoint_weights()
        self.load_state_dict(torch.load(str(path), map_location="cpu"), strict=False)

        print('Loaded SuperPoint model')

    def _dense_outputs(self, x):
        """Return the native stride-8 descriptor map and full-resolution scores."""
        x = self.transform(x)
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        score_logits = self.convPb(self.relu(self.convPa(x)))
        scores = F.softmax(score_logits, dim=1)[:, :-1]
        batch, _, height, width = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(batch, height, width, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(
            batch, height * 8, width * 8
        )

        descriptors = self.convDb(self.relu(self.convDa(x)))
        descriptors = F.normalize(descriptors, p=2, dim=1)
        return descriptors, scores

    def _sparse_from_dense(
        self,
        descriptors_dense,
        scores,
        *,
        top_k=None,
        detection_threshold=None,
        subpixel_refinement: bool = False,
        validity_mask: torch.Tensor | None = None,
    ):
        threshold = (
            self.detection_threshold
            if detection_threshold is None
            else float(detection_threshold)
        )
        all_valid = True
        if validity_mask is not None:
            validity_mask = torch.as_tensor(validity_mask, device=scores.device).bool()
            if validity_mask.ndim == 4 and validity_mask.shape[1] == 1:
                validity_mask = validity_mask[:, 0]
            elif validity_mask.ndim == 2:
                validity_mask = validity_mask[None]
            if validity_mask.shape != scores.shape:
                raise ValueError(
                    "validity_mask must align with the full-resolution score map"
                )
            all_valid = bool(validity_mask.all())
        nms_scores = (
            scores if all_valid else torch.where(validity_mask, scores, -torch.inf)
        )
        suppressed = batched_nms(nms_scores, self.nms_radius)
        if not all_valid:
            suppressed = torch.where(validity_mask, suppressed, -torch.inf)
        if self.remove_borders:
            pad = int(self.remove_borders)
            suppressed[:, :pad] = -1
            suppressed[:, :, :pad] = -1
            suppressed[:, -pad:] = -1
            suppressed[:, :, -pad:] = -1

        result = []
        for batch_index in range(suppressed.shape[0]):
            y, x_coord = torch.where(suppressed[batch_index] > threshold)
            keypoints = torch.stack([x_coord, y], dim=-1).float()
            keypoint_scores = suppressed[batch_index, y, x_coord]
            keypoints, keypoint_scores = select_top_k_keypoints(
                keypoints,
                keypoint_scores,
                self.max_num_keypoints if top_k is None else top_k,
            )
            if bool(subpixel_refinement):
                keypoints = quadratic_subpixel_keypoints(
                    keypoints, scores[batch_index]
                )
            descriptors = sample_descriptors(
                keypoints[None], descriptors_dense[batch_index : batch_index + 1]
            )[0].transpose(0, 1)
            result.append(
                {
                    "keypoints": keypoints,
                    "keypoint_scores": keypoint_scores,
                    "descriptors": descriptors,
                }
            )
        return result

    @torch.inference_mode()
    def detectAndCompute(
        self,
        x,
        top_k=None,
        detection_threshold=None,
        *,
        subpixel_refinement: bool = False,
        validity_mask: torch.Tensor | None = None,
    ):
        """Return native sparse SuperPoint descriptors for every input image.

        This intentionally does not sample the resized deployment feature
        pyramid.  It is the API used by the ULF-compatible initializer and by
        the sparse frontend parity audit.
        """
        device = next(self.parameters()).device
        descriptors_dense, scores = self._dense_outputs(x.to(device))
        return self._sparse_from_dense(
            descriptors_dense,
            scores,
            top_k=top_k,
            detection_threshold=detection_threshold,
            subpixel_refinement=subpixel_refinement,
            validity_mask=validity_mask,
        )

    @torch.inference_mode()
    def detectAndComputeWithDense(
        self,
        x,
        top_k=None,
        detection_threshold=None,
        *,
        subpixel_refinement: bool = False,
        validity_mask: torch.Tensor | None = None,
    ):
        """Return native sparse and dense outputs from one encoder forward."""

        device = next(self.parameters()).device
        descriptors, scores = self._dense_outputs(x.to(device))
        sparse = self._sparse_from_dense(
            descriptors,
            scores,
            top_k=top_k,
            detection_threshold=detection_threshold,
            subpixel_refinement=subpixel_refinement,
            validity_mask=validity_mask,
        )
        return sparse, (descriptors, scores.unsqueeze(1))

    @torch.inference_mode()
    def detectAndComputeDense(self, x):
        """Return the native stride-8 descriptor map and score map."""
        device = next(self.parameters()).device
        descriptors, scores = self._dense_outputs(x.to(device))
        return descriptors, scores.unsqueeze(1)

    def forward(self, x):
        """ Compute keypoints, scores, descriptors for image """
        return self._dense_outputs(x)
