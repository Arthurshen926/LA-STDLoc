"""Mapping/deployment mask handling at native SuperPoint keypoints."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from features.multiview_fusion import sample_mask_at_grid_uv


def deployment_valid_mask(
    cached: dict, name: str, deployment_masks: dict | None
) -> torch.Tensor:
    keypoints = torch.as_tensor(cached["native_keypoints"]).float()
    valid = torch.ones(keypoints.shape[0], dtype=torch.bool)
    if cached.get("native_valid_mask") is not None:
        valid &= sample_mask_at_grid_uv(
            torch.as_tensor(cached["native_valid_mask"]), keypoints
        ).cpu()
    if deployment_masks is None or name not in deployment_masks:
        return valid
    channels = deployment_masks[name]
    if len(channels) < 3:
        raise ValueError(f"deployment mask for {name!r} needs three channels")
    target_hw = tuple(int(value) for value in cached.get("native_input_hw", ()))
    if len(target_hw) != 2:
        raise ValueError("native_input_hw is required for deployment masks")
    resized = []
    for channel in channels[:3]:
        mask = torch.as_tensor(channel).detach().cpu().float()
        while mask.ndim > 2:
            mask = mask.squeeze(0)
        resized.append(
            F.interpolate(mask[None, None], size=target_hw, mode="nearest")[
                0, 0
            ].bool()
        )
    valid &= sample_mask_at_grid_uv(
        resized[0] & resized[1] & resized[2], keypoints
    ).cpu()
    return valid
