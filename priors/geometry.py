from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _quaternion_to_rotation(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = F.normalize(quaternion, dim=-1)
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


@dataclass(frozen=True)
class GaussianPriorGeometry:
    gaussian_type: str
    xyz: torch.Tensor
    rotation: torch.Tensor
    scaling: torch.Tensor

    def __post_init__(self):
        gaussian_type = str(self.gaussian_type).lower()
        if gaussian_type not in {"2dgs", "3dgs"}:
            raise ValueError(f"Unsupported Gaussian prior: {gaussian_type}")
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError("xyz must be [N, 3]")
        if self.rotation.shape != (self.xyz.shape[0], 4):
            raise ValueError("rotation must be [N, 4]")
        expected_scale_dim = 2 if gaussian_type == "2dgs" else 3
        if self.scaling.shape != (self.xyz.shape[0], expected_scale_dim):
            raise ValueError(
                f"{gaussian_type} scaling must be [N, {expected_scale_dim}]"
            )

    @property
    def frame(self) -> torch.Tensor:
        return _quaternion_to_rotation(self.rotation)

    @property
    def planarity(self) -> torch.Tensor:
        if str(self.gaussian_type).lower() == "2dgs":
            return torch.zeros(
                self.xyz.shape[0],
                device=self.xyz.device,
                dtype=self.xyz.dtype,
            )
        ordered = torch.sort(self.scaling.clamp_min(1e-8), dim=1).values
        return ordered[:, 0] / ordered[:, 1].clamp_min(1e-8)

    @property
    def proxy_normals(self) -> torch.Tensor:
        frame = self.frame
        if str(self.gaussian_type).lower() == "2dgs":
            return frame[:, :, 2]
        minimum_axis = self.scaling.argmin(dim=1)
        gather_index = minimum_axis[:, None, None].expand(-1, 3, 1)
        return frame.gather(2, gather_index).squeeze(2)
