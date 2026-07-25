import torch
import torch.nn.functional as F
from torch import nn


def pool_local_query_context(
    query_feature_map,
    keypoint_xy,
    image_hw,
    *,
    radius=2,
    step_px=8.0,
):
    feature_map = torch.as_tensor(query_feature_map)
    if feature_map.ndim == 4:
        if feature_map.shape[0] != 1:
            raise ValueError("query feature map batch size must be one")
        feature_map = feature_map[0]
    if feature_map.ndim != 3:
        raise ValueError("query feature map must have shape CxHxW")
    keypoint_xy = torch.as_tensor(
        keypoint_xy, device=feature_map.device, dtype=feature_map.dtype
    ).reshape(-1, 2)
    radius = max(int(radius), 0)
    axis = torch.arange(
        -radius,
        radius + 1,
        device=feature_map.device,
        dtype=feature_map.dtype,
    )
    dy, dx = torch.meshgrid(axis, axis, indexing="ij")
    offsets = torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=1)
    sample_xy = (
        keypoint_xy[:, None, :]
        + 0.5
        + offsets[None] * float(step_px)
    )
    height, width = map(int, image_hw)
    grid = sample_xy.clone()
    grid[..., 0] = 2.0 * grid[..., 0] / max(float(width), 1.0) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / max(float(height), 1.0) - 1.0
    sampled = F.grid_sample(
        feature_map[None],
        grid[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0].permute(1, 2, 0)
    sampled = F.normalize(sampled, dim=2)
    return F.normalize(sampled.mean(dim=1), dim=1)


class LocalContextMetricAdapter(nn.Module):
    def __init__(
        self,
        descriptor_dim=256,
        rank=16,
        max_residual_norm=0.10,
    ):
        super().__init__()
        self.descriptor_dim = int(descriptor_dim)
        self.rank = int(rank)
        self.max_residual_norm = float(max_residual_norm)
        self.down = nn.Linear(2 * self.descriptor_dim, self.rank)
        self.up = nn.Linear(self.rank, self.descriptor_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, sparse_descriptor, local_context):
        sparse_descriptor = F.normalize(
            torch.as_tensor(sparse_descriptor), dim=1
        )
        local_context = F.normalize(torch.as_tensor(local_context), dim=1)
        if sparse_descriptor.shape != local_context.shape:
            raise ValueError("sparse descriptors and local context must align")
        residual = self.up(
            F.silu(self.down(torch.cat([sparse_descriptor, local_context], dim=1)))
        )
        residual_norm = torch.linalg.norm(residual, dim=1, keepdim=True)
        bounded_scale = torch.clamp(
            self.max_residual_norm / residual_norm.clamp_min(1e-8),
            max=1.0,
        )
        bounded_residual = residual * bounded_scale
        adapted = F.normalize(sparse_descriptor + bounded_residual, dim=1)
        return adapted, bounded_residual

    def export_config(self):
        return {
            "descriptor_dim": self.descriptor_dim,
            "rank": self.rank,
            "max_residual_norm": self.max_residual_norm,
        }
