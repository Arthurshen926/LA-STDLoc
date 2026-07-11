import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class PairMeasurementOutput:
    inlier_logits: torch.Tensor
    offset: torch.Tensor
    cholesky: torch.Tensor

    @property
    def covariance(self):
        return self.cholesky @ self.cholesky.transpose(-1, -2)


def build_pair_geometry_features(
    keypoint_xy,
    candidate_landmark_xyz,
    all_landmark_xyz,
    image_size,
):
    """Build bounded image/map coordinates for query-set calibration."""
    candidate_landmark_xyz = torch.as_tensor(candidate_landmark_xyz)
    device = candidate_landmark_xyz.device
    dtype = candidate_landmark_xyz.dtype
    keypoint_xy = torch.as_tensor(
        keypoint_xy, device=device, dtype=dtype
    ).reshape(-1, 2)
    candidate_landmark_xyz = candidate_landmark_xyz.reshape(-1, 3)
    all_landmark_xyz = torch.as_tensor(
        all_landmark_xyz, device=device, dtype=dtype
    ).reshape(-1, 3)
    if keypoint_xy.shape[0] != candidate_landmark_xyz.shape[0]:
        raise ValueError("keypoint and candidate landmark counts must match")
    if len(image_size) != 2:
        raise ValueError("image_size must be (height, width)")
    height, width = int(image_size[0]), int(image_size[1])
    image_scale = candidate_landmark_xyz.new_tensor(
        [max(width, 1), max(height, 1)]
    )
    image_coordinates = 2.0 * keypoint_xy / image_scale - 1.0
    if all_landmark_xyz.shape[0] > 0:
        map_center = all_landmark_xyz.mean(dim=0)
        map_scale = all_landmark_xyz.std(dim=0, unbiased=False).clamp_min(1e-3)
        map_coordinates = torch.tanh(
            (candidate_landmark_xyz - map_center) / map_scale
        )
    else:
        map_coordinates = candidate_landmark_xyz.new_zeros(
            candidate_landmark_xyz.shape
        )
    return torch.cat([image_coordinates, map_coordinates], dim=1)


def sample_local_correlation_patch(
    query_feature_map,
    keypoint_xy,
    landmark_descriptors,
    radius=2,
):
    """Sample pair-conditioned local descriptor correlation around each keypoint."""
    if query_feature_map.dim() != 3:
        raise ValueError("query_feature_map must have shape CxHxW")
    keypoint_xy = torch.as_tensor(
        keypoint_xy,
        device=query_feature_map.device,
        dtype=query_feature_map.dtype,
    ).reshape(-1, 2)
    landmark_descriptors = torch.as_tensor(
        landmark_descriptors,
        device=query_feature_map.device,
        dtype=query_feature_map.dtype,
    ).reshape(keypoint_xy.shape[0], -1)
    if landmark_descriptors.shape[1] != query_feature_map.shape[0]:
        raise ValueError(
            "landmark descriptor dimension must match query feature channels"
        )
    radius = max(int(radius), 0)
    diameter = 2 * radius + 1
    offsets_y, offsets_x = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=query_feature_map.device),
        torch.arange(-radius, radius + 1, device=query_feature_map.device),
        indexing="ij",
    )
    offsets = torch.stack([offsets_x, offsets_y], dim=-1).reshape(1, -1, 2)
    sample_xy = keypoint_xy[:, None] + offsets.to(dtype=query_feature_map.dtype)
    height, width = query_feature_map.shape[-2:]
    grid = sample_xy.clone()
    grid[..., 0] = (
        2.0 * grid[..., 0] / max(int(width) - 1, 1) - 1.0
    )
    grid[..., 1] = (
        2.0 * grid[..., 1] / max(int(height) - 1, 1) - 1.0
    )
    sampled = F.grid_sample(
        query_feature_map[None],
        grid[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0]
    sampled = F.normalize(sampled, dim=0)
    landmarks = F.normalize(landmark_descriptors, dim=1)
    correlation = torch.einsum("cpk,pc->pk", sampled, landmarks)
    return correlation.reshape(keypoint_xy.shape[0], diameter * diameter)


def gaussian_measurement_nll(target_offset, output, valid_mask=None):
    if output.offset.numel() == 0:
        return output.offset.sum() * 0.0
    target_offset = torch.as_tensor(
        target_offset,
        device=output.offset.device,
        dtype=output.offset.dtype,
    ).reshape_as(output.offset)
    residual = target_offset - output.offset
    whitened = torch.linalg.solve_triangular(
        output.cholesky,
        residual.unsqueeze(-1),
        upper=False,
    ).squeeze(-1)
    nll = 0.5 * whitened.square().sum(dim=1)
    nll = nll + torch.log(
        torch.diagonal(output.cholesky, dim1=-2, dim2=-1)
    ).sum(dim=1)
    if valid_mask is not None:
        valid_mask = torch.as_tensor(
            valid_mask, device=nll.device, dtype=torch.bool
        ).reshape(-1)
        nll = nll[valid_mask]
    return nll.mean() if nll.numel() else output.offset.sum() * 0.0


class PairMeasurementHead(nn.Module):
    def __init__(
        self,
        descriptor_dim,
        pair_feature_dim=6,
        patch_radius=2,
        hidden_dim=64,
        max_offset=2.0,
        covariance_floor=0.1,
        initial_sigma=1.0,
        cosine_bias=0.65,
        cosine_scale=10.0,
        use_set_context=False,
        use_geometry_context=False,
    ):
        super().__init__()
        self.descriptor_dim = int(descriptor_dim)
        self.pair_feature_dim = int(pair_feature_dim)
        self.patch_radius = int(patch_radius)
        self.hidden_dim = int(hidden_dim)
        self.max_offset = float(max_offset)
        self.covariance_floor = float(covariance_floor)
        self.use_set_context = bool(use_set_context)
        self.use_geometry_context = bool(use_geometry_context)
        if self.use_geometry_context and not self.use_set_context:
            raise ValueError("geometry context requires query-set context")
        self.geometry_feature_dim = 5
        self.patch_dim = (2 * self.patch_radius + 1) ** 2
        if self.descriptor_dim <= 0:
            raise ValueError("descriptor_dim must be positive")
        if initial_sigma <= self.covariance_floor:
            raise ValueError("initial_sigma must exceed covariance_floor")

        self.cosine_bias = nn.Parameter(torch.tensor(float(cosine_bias)))
        self.log_cosine_scale = nn.Parameter(
            torch.tensor(math.log(max(float(cosine_scale), 1e-6)))
        )
        self.context_network = nn.Sequential(
            nn.Linear(self.pair_feature_dim + self.patch_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.descriptor_network = nn.Sequential(
            nn.Linear(2 * self.descriptor_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.output_network = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 6),
        )
        nn.init.zeros_(self.output_network[-1].weight)
        nn.init.zeros_(self.output_network[-1].bias)
        diagonal_target = float(initial_sigma) - self.covariance_floor
        inverse_softplus = math.log(math.expm1(diagonal_target))
        with torch.no_grad():
            self.output_network[-1].bias[3] = inverse_softplus
            self.output_network[-1].bias[5] = inverse_softplus
        self.set_network = None
        if self.use_set_context:
            self.set_network = nn.Sequential(
                nn.Linear(6 * self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 6),
            )
            nn.init.zeros_(self.set_network[-1].weight)
            nn.init.zeros_(self.set_network[-1].bias)
        self.geometry_token_network = None
        self.geometry_set_network = None
        if self.use_geometry_context:
            self.geometry_token_network = nn.Sequential(
                nn.Linear(
                    2 * self.hidden_dim + self.geometry_feature_dim + 6,
                    self.hidden_dim,
                ),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
            )
            self.geometry_set_network = nn.Sequential(
                nn.Linear(3 * self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 6),
            )
            nn.init.zeros_(self.geometry_set_network[-1].weight)
            nn.init.zeros_(self.geometry_set_network[-1].bias)

    def forward(
        self,
        pair_features,
        local_correlation_patch,
        query_descriptors,
        landmark_descriptors,
        geometry_features=None,
    ):
        pair_features = pair_features.reshape(-1, self.pair_feature_dim)
        local_correlation_patch = local_correlation_patch.reshape(
            pair_features.shape[0], self.patch_dim
        )
        query_descriptors = F.normalize(
            query_descriptors.reshape(pair_features.shape[0], self.descriptor_dim),
            dim=1,
        )
        landmark_descriptors = F.normalize(
            landmark_descriptors.reshape(
                pair_features.shape[0], self.descriptor_dim
            ),
            dim=1,
        )
        context = self.context_network(
            torch.cat([pair_features, local_correlation_patch], dim=1)
        )
        descriptor = self.descriptor_network(
            torch.cat(
                [
                    query_descriptors * landmark_descriptors,
                    torch.abs(query_descriptors - landmark_descriptors),
                ],
                dim=1,
            )
        )
        pair_latent = torch.cat([context, descriptor], dim=1)
        raw = self.output_network(pair_latent)
        if self.set_network is not None and pair_latent.shape[0] > 0:
            set_mean = pair_latent.mean(dim=0, keepdim=True)
            set_max = pair_latent.max(dim=0, keepdim=True).values
            set_context = torch.cat(
                [
                    pair_latent,
                    set_mean.expand_as(pair_latent),
                    set_max.expand_as(pair_latent),
                ],
                dim=1,
            )
            raw = raw + self.set_network(set_context)
        if self.geometry_token_network is not None:
            if geometry_features is None:
                raise ValueError(
                    "geometry_features are required by the geometry set calibrator"
                )
            geometry_features = torch.as_tensor(
                geometry_features,
                device=pair_latent.device,
                dtype=pair_latent.dtype,
            ).reshape(pair_latent.shape[0], self.geometry_feature_dim)
            if pair_latent.shape[0] > 0:
                geometry_token = self.geometry_token_network(
                    torch.cat([pair_latent, geometry_features, raw], dim=1)
                )
                geometry_mean = geometry_token.mean(dim=0, keepdim=True)
                geometry_max = geometry_token.max(dim=0, keepdim=True).values
                geometry_context = torch.cat(
                    [
                        geometry_token,
                        geometry_mean.expand_as(geometry_token),
                        geometry_max.expand_as(geometry_token),
                    ],
                    dim=1,
                )
                raw = raw + self.geometry_set_network(geometry_context)
        scale = self.log_cosine_scale.exp().clamp(1e-3, 1e3)
        inlier_logits = scale * (pair_features[:, 0] - self.cosine_bias)
        inlier_logits = inlier_logits + raw[:, 0]
        offset = torch.tanh(raw[:, 1:3]) * self.max_offset
        l11 = self.covariance_floor + F.softplus(raw[:, 3])
        l21 = torch.tanh(raw[:, 4]) * l11
        l22 = self.covariance_floor + F.softplus(raw[:, 5])
        cholesky = raw.new_zeros((raw.shape[0], 2, 2))
        cholesky[:, 0, 0] = l11
        cholesky[:, 1, 0] = l21
        cholesky[:, 1, 1] = l22
        return PairMeasurementOutput(inlier_logits, offset, cholesky)

    def export_config(self):
        return {
            "descriptor_dim": self.descriptor_dim,
            "pair_feature_dim": self.pair_feature_dim,
            "patch_radius": self.patch_radius,
            "hidden_dim": self.hidden_dim,
            "max_offset": self.max_offset,
            "covariance_floor": self.covariance_floor,
            "use_set_context": self.use_set_context,
            "use_geometry_context": self.use_geometry_context,
        }
