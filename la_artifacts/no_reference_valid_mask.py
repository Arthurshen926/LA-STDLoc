from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


def _as_chw_float(image):
    tensor = torch.as_tensor(image, dtype=torch.float32).detach()
    if tensor.dim() == 2:
        tensor = tensor[None].repeat(3, 1, 1)
    elif tensor.dim() == 3 and tensor.shape[-1] == 3 and tensor.shape[0] != 3:
        tensor = tensor.permute(2, 0, 1)
    if tensor.dim() != 3:
        raise ValueError(f"Expected image as HxW, CxHxW, or HxWxC, got {tuple(tensor.shape)}")
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    if tensor.shape[0] != 3:
        tensor = tensor[:3]
    if tensor.max().item() > 2.0:
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0)


def _gray(image):
    return (0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]).clamp(0.0, 1.0)


def _odd(value):
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def _avg_pool(map2d, kernel):
    kernel = _odd(kernel)
    pad = kernel // 2
    padded = F.pad(map2d[None, None], (pad, pad, pad, pad), mode="replicate")
    return F.avg_pool2d(padded, kernel_size=kernel, stride=1, padding=0)[0, 0]


def _dilate(mask, radius):
    radius = int(radius or 0)
    if radius <= 0:
        return mask.bool()
    kernel = 2 * radius + 1
    return F.max_pool2d(mask.float()[None, None], kernel_size=kernel, stride=1, padding=radius)[0, 0] > 0


def _erode(mask, radius):
    radius = int(radius or 0)
    if radius <= 0:
        return mask.bool()
    kernel = 2 * radius + 1
    eroded_invalid = F.max_pool2d((~mask.bool()).float()[None, None], kernel_size=kernel, stride=1, padding=radius)[0, 0] > 0
    return mask.bool() & ~eroded_invalid


def _gradient(gray):
    gx_kernel = gray.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
    gy_kernel = gx_kernel.t()
    x = F.pad(gray[None, None], (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(x, gx_kernel[None, None], padding=0)[0, 0]
    gy = F.conv2d(x, gy_kernel[None, None], padding=0)[0, 0]
    return torch.sqrt(gx.square() + gy.square()).clamp_min(0.0)


def _normalize_by_quantile(value, quantile=0.95):
    flat = value.detach().reshape(-1)
    if flat.numel() == 0:
        return value
    if flat.numel() > 65536:
        stride = int(torch.ceil(torch.tensor(flat.numel() / 65536.0)).item())
        flat = flat[::stride]
    scale = torch.quantile(flat, float(quantile)).clamp_min(1e-6)
    return (value / scale).clamp(0.0, 1.0)


def _connected_components(mask, min_area=1):
    mask_cpu = mask.detach().cpu().bool()
    height, width = mask_cpu.shape
    min_area = int(max(1, min_area))
    try:
        import cv2
        import numpy as np

        array = mask_cpu.numpy().astype("uint8")
        count, labels, stats, _ = cv2.connectedComponentsWithStats(array, connectivity=4)
        filtered = np.zeros((height, width), dtype=bool)
        components = []
        total = float(max(height * width, 1))
        for label in range(1, int(count)):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            filtered[labels == label] = True
            components.append({"area": area, "area_frac": float(area / total), "bbox_xyxy": [x, y, x + w, y + h]})
        components.sort(key=lambda item: item["area"], reverse=True)
        return torch.from_numpy(filtered).to(mask.device), components
    except Exception:
        return mask.bool(), []


@dataclass
class NoReferenceValidMaskConfig:
    structure_window: int = 9
    support_threshold: float = 0.22
    support_dilate_radius: int = 5
    support_min_area: int = 24
    invalid_min_area: int = 96
    invalid_dilate_radius: int = 1
    invalid_erode_radius: int = 0
    dark_threshold: float = 0.035
    bright_threshold: float = 0.985
    blank_variance_threshold: float = 0.03
    low_structure_threshold: float = 0.035
    low_variance_threshold: float = 0.004
    border_invalid_width: int = 0
    use_saturation_invalid: bool = True


@dataclass
class NoReferenceValidMask:
    valid_mask: torch.Tensor
    support_mask: torch.Tensor
    invalid_score: torch.Tensor
    support_score: torch.Tensor
    channel_maps: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)

    @property
    def mask(self):
        return self.valid_mask

    @property
    def score_map(self):
        return self.invalid_score

    def valid_points(self, points_xy):
        return _points_in_mask(self.valid_mask, points_xy)

    def support_points(self, points_xy):
        return _points_in_mask(self.support_mask, points_xy)

    def to_feature_mask(self, output_hw, min_valid_fraction=0.5, kind="valid"):
        mask = self.support_mask if str(kind) == "support" else self.valid_mask
        height, width = int(output_hw[0]), int(output_hw[1])
        pooled = F.interpolate(mask.float()[None, None], size=(height, width), mode="area")[0, 0]
        return pooled >= float(min_valid_fraction)


def _points_in_mask(mask, points_xy):
    points = torch.as_tensor(points_xy, dtype=torch.float32, device=mask.device)
    if points.numel() == 0:
        return torch.zeros(points.shape[:-1], dtype=torch.bool, device=mask.device)
    flat = points.reshape(-1, 2)
    x = torch.floor(flat[:, 0]).long()
    y = torch.floor(flat[:, 1]).long()
    height, width = mask.shape
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    keep = torch.zeros(flat.shape[0], dtype=torch.bool, device=mask.device)
    if inside.any():
        keep[inside] = mask[y[inside], x[inside]].bool()
    return keep.reshape(points.shape[:-1])


class NoReferenceValidMaskBuilder:
    """Builds synthetic RGB masks without a reference image.

    valid_mask is conservative and only removes obvious image defects such as
    blank saturated borders. support_mask is a separate localization prior.
    """

    def __init__(self, config: Optional[NoReferenceValidMaskConfig] = None):
        self.config = config or NoReferenceValidMaskConfig()

    def build(self, image):
        cfg = self.config
        rgb = _as_chw_float(image)
        gray = _gray(rgb)
        grad = _gradient(gray)
        local_mean = _avg_pool(gray, cfg.structure_window)
        local_var = _avg_pool((gray - local_mean).square(), cfg.structure_window)
        local_grad = _avg_pool(grad, cfg.structure_window)

        grad_score = _normalize_by_quantile(local_grad, 0.97)
        var_score = _normalize_by_quantile(torch.sqrt(local_var.clamp_min(0.0)), 0.97)
        support_score = torch.maximum(grad_score, var_score).clamp(0.0, 1.0)
        support = support_score >= float(cfg.support_threshold)
        support = _dilate(support, cfg.support_dilate_radius)
        support, support_components = _connected_components(support, min_area=cfg.support_min_area)

        low_structure = (support_score <= float(cfg.low_structure_threshold)) & (local_var <= float(cfg.low_variance_threshold))
        flat_blank = local_var <= float(cfg.blank_variance_threshold)
        dark_blank = (gray <= float(cfg.dark_threshold)) & flat_blank
        bright_blank = (gray >= float(cfg.bright_threshold)) & flat_blank
        invalid = dark_blank | bright_blank
        if cfg.use_saturation_invalid:
            channel_range = rgb.max(dim=0).values - rgb.min(dim=0).values
            saturated_flat = ((rgb.max(dim=0).values >= 0.995) | (rgb.min(dim=0).values <= 0.005)) & (channel_range <= 0.02)
            invalid = invalid | (saturated_flat & low_structure)
        if int(cfg.border_invalid_width or 0) > 0:
            b = int(cfg.border_invalid_width)
            invalid[:b, :] = True
            invalid[-b:, :] = True
            invalid[:, :b] = True
            invalid[:, -b:] = True

        invalid, invalid_components = _connected_components(invalid, min_area=cfg.invalid_min_area)
        invalid = _dilate(invalid, cfg.invalid_dilate_radius)
        valid = ~invalid
        valid = _erode(valid, cfg.invalid_erode_radius)
        support = support & valid

        invalid_score = torch.maximum(dark_blank.float(), bright_blank.float())
        if cfg.use_saturation_invalid:
            invalid_score = torch.maximum(invalid_score, (invalid.float() * 0.75).clamp(0.0, 1.0))

        total = float(max(gray.numel(), 1))
        summary = {
            "valid_frac": float(valid.float().mean().item()),
            "invalid_frac": float((~valid).float().mean().item()),
            "support_frac": float(support.float().mean().item()),
            "support_score_mean": float(support_score.float().mean().item()),
            "invalid_score_mean": float(invalid_score.float().mean().item()),
            "support_component_count": int(len(support_components)),
            "invalid_component_count": int(len(invalid_components)),
            "largest_support_component_frac": float(max((item["area"] / total for item in support_components), default=0.0)),
            "largest_invalid_component_frac": float(max((item["area"] / total for item in invalid_components), default=0.0)),
        }
        return NoReferenceValidMask(
            valid_mask=valid.bool(),
            support_mask=support.bool(),
            invalid_score=invalid_score.clamp(0.0, 1.0),
            support_score=support_score.clamp(0.0, 1.0),
            channel_maps={
                "gray": gray,
                "gradient": grad,
                "local_gradient": local_grad,
                "local_variance": local_var,
                "low_structure": low_structure.float(),
            },
            summary=summary,
        )


def _save_mask(mask, path):
    import numpy as np
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = torch.as_tensor(mask).detach().cpu().bool().numpy().astype("uint8") * 255
    Image.fromarray(array, mode="L").save(path)
    return str(path)


def _save_score(score, path):
    import numpy as np
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (torch.as_tensor(score).detach().cpu().float().clamp(0.0, 1.0).numpy() * 255.0).round().astype("uint8")
    Image.fromarray(array, mode="L").save(path)
    return str(path)


def save_no_reference_valid_mask_pngs(result, path_prefix):
    prefix = Path(path_prefix)
    return {
        "valid_mask": _save_mask(result.valid_mask, prefix.with_suffix(".valid_mask.png")),
        "support_mask": _save_mask(result.support_mask, prefix.with_suffix(".support_mask.png")),
        "invalid_score": _save_score(result.invalid_score, prefix.with_suffix(".invalid_score.png")),
        "support_score": _save_score(result.support_score, prefix.with_suffix(".support_score.png")),
    }
