from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable

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

    def anchor_local_coordinates(self, anchor_xyz: torch.Tensor) -> torch.Tensor:
        """Express world-space anchors in each Gaussian's local frame."""
        if anchor_xyz.shape != self.xyz.shape:
            raise ValueError("anchor_xyz must match xyz")
        displacement = anchor_xyz - self.xyz
        return torch.einsum("nji,nj->ni", self.frame, displacement)

    def surface_residual_components(
        self,
        anchor_xyz: torch.Tensor,
        *,
        minimum_scale_m: float = 1e-4,
    ) -> dict[str, torch.Tensor]:
        """Measure anchor support relative to the Gaussian surface/volume."""
        local = self.anchor_local_coordinates(anchor_xyz)
        if str(self.gaussian_type).lower() == "2dgs":
            tangent = local[:, :2]
            tangent_distance = torch.linalg.norm(tangent, dim=1)
            normal_distance = local[:, 2].abs()
            tangent_normalized = torch.linalg.norm(
                tangent / self.scaling.clamp_min(minimum_scale_m),
                dim=1,
            )
        else:
            minimum_axis = self.scaling.argmin(dim=1)
            normal_distance = local.gather(
                1, minimum_axis[:, None]
            ).squeeze(1).abs()
            tangent_mask = torch.ones_like(local, dtype=torch.bool)
            tangent_mask.scatter_(1, minimum_axis[:, None], False)
            tangent = local[tangent_mask].reshape(-1, 2)
            tangent_scale = self.scaling[tangent_mask].reshape(-1, 2)
            tangent_distance = torch.linalg.norm(tangent, dim=1)
            tangent_normalized = torch.linalg.norm(
                tangent / tangent_scale.clamp_min(minimum_scale_m),
                dim=1,
            )
        return {
            "local_coordinates_m": local,
            "tangent_distance_m": tangent_distance,
            "normal_distance_m": normal_distance,
            "tangent_normalized": tangent_normalized,
        }

    def anchor_axis_bounds(
        self,
        *,
        tangent_bound_m: float,
        normal_bound_m: float,
        covariance_scale: float,
        absolute_bound_m: float,
        minimum_bound_m: float = 1e-4,
    ) -> torch.Tensor:
        if str(self.gaussian_type).lower() == "2dgs":
            return self.xyz.new_tensor(
                [tangent_bound_m, tangent_bound_m, normal_bound_m]
            )[None].expand_as(self.xyz)
        covariance_bounds = (
            self.scaling.detach().clamp_min(minimum_bound_m)
            * float(covariance_scale)
        )
        return covariance_bounds.clamp(
            min=float(minimum_bound_m),
            max=float(absolute_bound_m),
        )

    def materialize_anchor(
        self,
        raw_offset: torch.Tensor,
        *,
        tangent_bound_m: float,
        normal_bound_m: float,
        covariance_scale: float,
        absolute_bound_m: float,
    ) -> torch.Tensor:
        if raw_offset.shape != self.xyz.shape:
            raise ValueError("raw_offset must match xyz")
        bounds = self.anchor_axis_bounds(
            tangent_bound_m=tangent_bound_m,
            normal_bound_m=normal_bound_m,
            covariance_scale=covariance_scale,
            absolute_bound_m=absolute_bound_m,
        )
        if str(self.gaussian_type).lower() == "2dgs":
            tangent_raw = raw_offset[:, :2]
            tangent_norm = torch.linalg.norm(
                tangent_raw, dim=1, keepdim=True
            )
            tangent_radius = (
                torch.tanh(tangent_norm) * float(tangent_bound_m)
            )
            tangent_scale = torch.where(
                tangent_norm > 1e-8,
                tangent_radius / tangent_norm.clamp_min(1e-8),
                torch.full_like(tangent_norm, float(tangent_bound_m)),
            )
            local_offset = torch.cat(
                (
                    tangent_raw * tangent_scale,
                    torch.tanh(raw_offset[:, 2:3])
                    * float(normal_bound_m),
                ),
                dim=1,
            )
        else:
            local_offset = bounds * torch.tanh(raw_offset)
        world_offset = torch.einsum("nij,nj->ni", self.frame, local_offset)
        return self.xyz + world_offset

    def mahalanobis_anchor_prior(
        self,
        anchor_xyz: torch.Tensor,
        *,
        minimum_scale_m: float = 1e-4,
    ) -> torch.Tensor:
        local = self.anchor_local_coordinates(anchor_xyz)
        if str(self.gaussian_type).lower() == "2dgs":
            scale = torch.cat(
                (
                    self.scaling,
                    torch.full_like(self.scaling[:, :1], minimum_scale_m),
                ),
                dim=1,
            )
        else:
            scale = self.scaling
        normalized = local / scale.clamp_min(minimum_scale_m)
        return normalized.square().sum(dim=1)


def tensor_sha256(tensors: Iterable[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = torch.as_tensor(tensor).detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_gaussian_anchor_resume(
    initial_state,
    *,
    gaussian_type: str,
    tangent_bound_m: float,
    normal_bound_m: float,
    covariance_scale: float,
    absolute_bound_m: float,
):
    if not isinstance(initial_state, dict):
        return
    raw_offset = initial_state.get("raw_anchor_offset")
    if raw_offset is None:
        return
    raw_offset = torch.as_tensor(raw_offset, dtype=torch.float32)
    if (
        raw_offset.numel() == 0
        or not bool(torch.isfinite(raw_offset).all().item())
        or float(raw_offset.abs().max().item()) <= 1e-12
    ):
        return
    config = initial_state.get("config")
    if not isinstance(config, dict):
        raise ValueError(
            "Nonzero localization anchor offset has no parameterization metadata"
        )
    gaussian_type = str(gaussian_type).lower()
    expected_parameterization = (
        "radial_tanh_tangent_plane_v1"
        if gaussian_type == "2dgs"
        else "covariance_bounded_tanh_v1"
    )
    if config.get("surface_anchor_parameterization") != expected_parameterization:
        raise ValueError(
            "Cannot reinterpret a saved localization anchor with a different "
            "Gaussian prior parameterization"
        )
    expected = (
        {
            "tangent_bound_m": tangent_bound_m,
            "normal_bound_m": normal_bound_m,
        }
        if gaussian_type == "2dgs"
        else {
            "covariance_anchor_scale": covariance_scale,
            "covariance_anchor_absolute_bound_m": absolute_bound_m,
        }
    )
    for key, value in expected.items():
        try:
            previous = float(config[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Saved localization anchor is missing {key}"
            ) from exc
        if not math.isclose(
            previous, float(value), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"Saved localization anchor uses {key}={previous:g}, "
                f"requested {float(value):g}"
            )
