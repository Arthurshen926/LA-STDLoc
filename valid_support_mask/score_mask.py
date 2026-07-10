from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


def _score_tensor(value):
    if hasattr(value, "score_map"):
        value = value.score_map
    score = torch.as_tensor(value, dtype=torch.float32).detach()
    if score.dim() == 3:
        if score.shape[0] == 1:
            score = score[0]
        elif score.shape[-1] == 1:
            score = score[..., 0]
        else:
            score = score.mean(dim=0)
    if score.dim() != 2:
        raise ValueError(f"Expected a 2D score map, got shape {tuple(score.shape)}")
    return torch.nan_to_num(score, nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _erode_binary(mask, radius):
    radius = int(radius or 0)
    if radius <= 0 or mask.numel() == 0:
        return mask.bool()
    kernel = 2 * radius + 1
    invalid = (~mask.bool()).float()[None, None]
    near_invalid = F.max_pool2d(invalid, kernel_size=kernel, stride=1, padding=radius)[0, 0] > 0
    return mask.bool() & ~near_invalid


def _label_components(mask, min_area=1, min_area_frac=0.0, max_components=0):
    mask_cpu = mask.detach().cpu().bool()
    height, width = mask_cpu.shape
    min_pixels = max(int(min_area or 1), int(round(float(min_area_frac or 0.0) * height * width)))
    try:
        import cv2
        import numpy as np

        array = mask_cpu.numpy().astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(array, connectivity=4)
        components = []
        for label in range(1, int(count)):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_pixels:
                continue
            components.append(
                {
                    "label": label,
                    "area": area,
                    "bbox_xyxy": [x, y, x + w, y + h],
                }
            )
        components.sort(key=lambda item: item["area"], reverse=True)
        if int(max_components or 0) > 0:
            components = components[: int(max_components)]
        filtered_np = np.zeros((height, width), dtype=bool)
        public = []
        total = float(max(height * width, 1))
        for component in components:
            filtered_np[labels == component.pop("label")] = True
            component["area_frac"] = float(component["area"] / total)
            public.append(component)
        return torch.from_numpy(filtered_np).to(device=mask.device), public
    except Exception:
        pass

    visited = torch.zeros((height, width), dtype=torch.bool)
    components = []
    for y in range(height):
        for x in range(width):
            if visited[y, x] or not mask_cpu[y, x]:
                continue
            stack = [(x, y)]
            visited[y, x] = True
            pixels = []
            while stack:
                px, py = stack.pop()
                pixels.append((px, py))
                for nx, ny in ((px - 1, py), (px + 1, py), (px, py - 1), (px, py + 1)):
                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue
                    if visited[ny, nx] or not mask_cpu[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((nx, ny))
            if len(pixels) < min_pixels:
                continue
            xs = [item[0] for item in pixels]
            ys = [item[1] for item in pixels]
            components.append(
                {
                    "area": int(len(pixels)),
                    "bbox_xyxy": [int(min(xs)), int(min(ys)), int(max(xs) + 1), int(max(ys) + 1)],
                    "_pixels": pixels,
                }
            )

    components.sort(key=lambda item: item["area"], reverse=True)
    if int(max_components or 0) > 0:
        components = components[: int(max_components)]
    filtered = torch.zeros((height, width), dtype=torch.bool)
    public = []
    total = float(max(height * width, 1))
    for component in components:
        for x, y in component.pop("_pixels"):
            filtered[y, x] = True
        component["area_frac"] = float(component["area"] / total)
        public.append(component)
    return filtered.to(device=mask.device), public


@dataclass
class ScoreValidMaskConfig:
    max_artifact_score: float = 0.45
    erosion_radius: int = 3
    min_component_area: int = 64
    min_component_area_frac: float = 0.0
    max_components: int = 0
    min_valid_frac: float = 0.0
    min_alpha: Optional[float] = None


@dataclass
class ScoreValidMask:
    mask: torch.Tensor
    score_map: torch.Tensor
    components: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    @property
    def confidence_map(self):
        return (1.0 - self.score_map).clamp(0.0, 1.0)

    def valid_points(self, points_xy):
        points = torch.as_tensor(points_xy, dtype=torch.float32, device=self.mask.device)
        if points.numel() == 0:
            return torch.zeros(points.shape[:-1], dtype=torch.bool, device=self.mask.device)
        if points.shape[-1] != 2:
            raise ValueError(f"Expected point coordinates shaped (..., 2), got {tuple(points.shape)}")
        flat = points.reshape(-1, 2)
        x = torch.floor(flat[:, 0]).long()
        y = torch.floor(flat[:, 1]).long()
        height, width = self.mask.shape
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        keep = torch.zeros(flat.shape[0], dtype=torch.bool, device=self.mask.device)
        if inside.any():
            keep[inside] = self.mask[y[inside], x[inside]].bool()
        return keep.reshape(points.shape[:-1])

    def to_feature_mask(self, output_hw, min_valid_fraction=0.5):
        height, width = int(output_hw[0]), int(output_hw[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid output feature-map size: {output_hw}")
        pooled = F.interpolate(self.mask.float()[None, None], size=(height, width), mode="area")[0, 0]
        return pooled >= float(min_valid_fraction)


class ScoreValidMaskBuilder:
    """Turns generic artifact scores into clean connected local regions."""

    def __init__(self, config=None):
        self.config = config or ScoreValidMaskConfig()

    def build(self, evidence_or_score, alpha=None):
        cfg = self.config
        score = _score_tensor(evidence_or_score)
        valid = score <= float(cfg.max_artifact_score)
        if alpha is not None and cfg.min_alpha is not None:
            alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32, device=score.device).squeeze()
            if alpha_tensor.shape != score.shape:
                alpha_tensor = F.interpolate(alpha_tensor[None, None], size=score.shape, mode="bilinear", align_corners=False)[0, 0]
            valid = valid & (alpha_tensor >= float(cfg.min_alpha))
        valid = _erode_binary(valid, cfg.erosion_radius)
        valid, components = _label_components(
            valid,
            min_area=cfg.min_component_area,
            min_area_frac=cfg.min_component_area_frac,
            max_components=cfg.max_components,
        )
        valid_frac = float(valid.float().mean().item()) if valid.numel() else 0.0
        if valid_frac < float(cfg.min_valid_frac):
            valid = torch.zeros_like(valid, dtype=torch.bool)
            components = []
            valid_frac = 0.0
        largest = max((item["area_frac"] for item in components), default=0.0)
        summary = {
            "valid_frac": valid_frac,
            "invalid_frac": float(1.0 - valid_frac),
            "component_count": int(len(components)),
            "largest_component_frac": float(largest),
            "max_artifact_score": float(cfg.max_artifact_score),
            "erosion_radius": int(cfg.erosion_radius),
            "min_component_area": int(cfg.min_component_area),
        }
        return ScoreValidMask(mask=valid.bool(), score_map=score, components=components, summary=summary)


def save_score_valid_mask_png(valid_mask, path):
    import numpy as np
    from PIL import Image

    mask = valid_mask.mask if hasattr(valid_mask, "mask") else valid_mask
    array = torch.as_tensor(mask).detach().cpu().bool().numpy().astype(np.uint8) * 255
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="L").save(path)
    return path


ArtifactValidMaskConfig = ScoreValidMaskConfig
ArtifactValidMask = ScoreValidMask
ArtifactValidMaskBuilder = ScoreValidMaskBuilder
save_valid_mask_png = save_score_valid_mask_png
