#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from utils.general_utils import (build_rotation, build_scaling_rotation,
                                 get_expon_lr_func, inverse_sigmoid,
                                 strip_symmetric)
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p
from localization_training.lafgs_reconstruction import pose_aware_split_score


def _rotation_matrix_from_quaternion(rotation):
    norm = torch.linalg.norm(rotation, dim=1, keepdim=True).clamp_min(1e-12)
    q = rotation / norm
    matrix = torch.zeros((q.shape[0], 3, 3), dtype=q.dtype, device=q.device)
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    matrix[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[:, 0, 1] = 2 * (x * y - r * z)
    matrix[:, 0, 2] = 2 * (x * z + r * y)
    matrix[:, 1, 0] = 2 * (x * y + r * z)
    matrix[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[:, 1, 2] = 2 * (y * z - r * x)
    matrix[:, 2, 0] = 2 * (x * z - r * y)
    matrix[:, 2, 1] = 2 * (y * z + r * x)
    matrix[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def split_localization_child_features(parent_features, parent_prototypes=None, prototype_counts=None, repeat=2):
    child_features = parent_features.repeat(repeat, *([1] * (parent_features.dim() - 1)))
    if repeat < 2 or parent_prototypes is None or prototype_counts is None or parent_features.numel() == 0:
        return child_features

    parent_count = parent_features.shape[0]
    flat_dim = parent_features.reshape(parent_count, -1).shape[1]
    prototypes = torch.as_tensor(parent_prototypes, device=parent_features.device, dtype=parent_features.dtype)
    prototypes = prototypes.reshape(parent_count, -1)
    if prototypes.shape[1] < flat_dim:
        return child_features
    prototypes = prototypes[:, :flat_dim]
    counts = torch.as_tensor(prototype_counts, device=parent_features.device).reshape(parent_count)
    valid = (counts > 0) & torch.isfinite(prototypes).all(dim=1) & (torch.linalg.norm(prototypes, dim=1) > 0)
    if not valid.any():
        return child_features

    prototype_features = F.normalize(prototypes, p=2, dim=1).reshape_as(parent_features)
    second_child = child_features[parent_count : 2 * parent_count]
    second_child[valid] = prototype_features[valid]
    child_features[parent_count : 2 * parent_count] = second_child
    return child_features


class GaussianModel_2dgs(nn.Module):
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        super(GaussianModel_2dgs, self).__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self._loc_feature = torch.empty(0)
        self._loc_opacity = torch.empty(0)
        self._loc_anchor_offset = torch.empty(0)
        self.surfel_loc_tangent_bound = 0.0
        self.surfel_loc_normal_bound = 0.0
        self.surfel_loc_radius_floor = 0.0
        self.detach_loc_anchor_base = False

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._loc_feature,
            self._loc_opacity,
            self._loc_anchor_offset,
            float(getattr(self, "surfel_loc_tangent_bound", 0.0) or 0.0),
            float(getattr(self, "surfel_loc_normal_bound", 0.0) or 0.0),
            float(getattr(self, "surfel_loc_radius_floor", 0.0) or 0.0),
        )
    
    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            self._loc_feature,
            *loc_extra,
        ) = model_args
        if len(loc_extra) >= 1 and torch.is_tensor(loc_extra[0]):
            self._loc_opacity = nn.Parameter(loc_extra[0].detach().clone().requires_grad_(True))
        if len(loc_extra) >= 2 and torch.is_tensor(loc_extra[1]):
            self._loc_anchor_offset = nn.Parameter(loc_extra[1].detach().clone().requires_grad_(True))
        if len(loc_extra) >= 3:
            self.surfel_loc_tangent_bound = float(loc_extra[2] or 0.0)
        if len(loc_extra) >= 4:
            self.surfel_loc_normal_bound = float(loc_extra[3] or 0.0)
        if len(loc_extra) >= 5:
            self.surfel_loc_radius_floor = float(loc_extra[4] or 0.0)
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        expected = len(self.optimizer.param_groups)
        actual = len(opt_dict.get("param_groups", [])) if opt_dict else 0
        if actual == expected:
            self.optimizer.load_state_dict(opt_dict)
        elif opt_dict:
            print(
                f"[LaFGS-2DGS] Skip optimizer restore: checkpoint has {actual} "
                f"param groups, current model has {expected}."
            )

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) #.clamp(max=1)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_loc_xyz(self):
        self._ensure_loc_anchor_state()
        tangent_bound = float(getattr(self, "surfel_loc_tangent_bound", 0.0) or 0.0)
        normal_bound = float(getattr(self, "surfel_loc_normal_bound", 0.0) or 0.0)
        if tangent_bound <= 0.0 and normal_bound <= 0.0:
            return self.get_xyz
        detach_base = bool(getattr(self, "detach_loc_anchor_base", False))
        xyz = self.get_xyz.detach() if detach_base else self.get_xyz
        if xyz.numel() == 0:
            return xyz
        raw = self._loc_anchor_offset.to(device=xyz.device, dtype=xyz.dtype)
        rotation_input = self.get_rotation.to(device=xyz.device, dtype=xyz.dtype)
        scales = self.get_scaling.to(device=xyz.device, dtype=xyz.dtype)
        if detach_base:
            rotation_input = rotation_input.detach()
            scales = scales.detach()
        rotation = _rotation_matrix_from_quaternion(rotation_input)
        if scales.shape[1] >= 2:
            tangent_scales = scales[:, :2]
            radius = tangent_scales.mean(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            radius = scales.reshape(scales.shape[0], -1).mean(dim=1, keepdim=True).clamp_min(1e-8)
        radius_floor = float(getattr(self, "surfel_loc_radius_floor", 0.0) or 0.0)
        if radius_floor > 0.0:
            radius = radius.clamp_min(radius_floor)
        tangent_delta = torch.tanh(raw[:, :2]) * (tangent_bound * radius)
        normal_delta = torch.tanh(raw[:, 2:3]) * (normal_bound * radius)
        return (
            xyz
            + rotation[:, :, 0] * tangent_delta[:, 0:1]
            + rotation[:, :, 1] * tangent_delta[:, 1:2]
            + rotation[:, :, 2] * normal_delta
        )
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_loc_feature(self):
        return self._loc_feature

    def materialized_loc_feature(self, indices=None):
        features = self.get_loc_feature
        if indices is not None:
            indices = torch.as_tensor(indices, dtype=torch.long, device=features.device)
            features = features[indices]
        return features

    def _loc_feature_dim(self):
        if self._loc_feature.numel() == 0:
            return 0
        return self._loc_feature.reshape(self._loc_feature.shape[0], -1).shape[1]

    def _localization_buffer_names(self):
        return [
            "loc_grad_accum",
            "loc_grad_denom",
            "loc_observation_count",
            "loc_repeatability_ema",
            "loc_positive_prob_ema",
            "loc_margin_ema",
            "loc_entropy_ema",
            "loc_outlier_ema",
            "loc_reproj_error_ema",
            "loc_information_ema",
            "loc_redundancy_ema",
            "loc_prototype",
            "loc_prototype_count",
            "loc_birth_iteration",
            "last_topology_iteration",
            "loc_node_id",
            "loc_parent_node_id",
            "loc_source_index",
            "loc_source_xyz",
        ]

    def init_localization_state(self, from_rgb_opacity=True, birth_iteration=0):
        n = self.get_xyz.shape[0]
        device = self.get_xyz.device
        feature_dim = self._loc_feature_dim()
        if from_rgb_opacity and torch.is_tensor(self._opacity) and self._opacity.numel() == n:
            loc_opacity = self._opacity.detach().clone()
        else:
            loc_opacity = inverse_sigmoid(torch.full((n, 1), 0.1, dtype=torch.float32, device=device))
        self._loc_opacity = nn.Parameter(loc_opacity.to(device=device).detach().clone().requires_grad_(True))
        self.loc_grad_accum = torch.zeros((n, 1), dtype=torch.float32, device=device)
        self.loc_grad_denom = torch.zeros((n, 1), dtype=torch.float32, device=device)
        self.loc_observation_count = torch.zeros(n, dtype=torch.long, device=device)
        self.loc_repeatability_ema = torch.zeros(n, dtype=torch.float32, device=device)
        self.loc_positive_prob_ema = torch.zeros(n, dtype=torch.float32, device=device)
        self.loc_margin_ema = torch.zeros(n, dtype=torch.float32, device=device)
        self.loc_entropy_ema = torch.zeros(n, dtype=torch.float32, device=device)
        self.loc_outlier_ema = torch.zeros(n, dtype=torch.float32, device=device)
        self.loc_reproj_error_ema = torch.zeros(n, dtype=torch.float32, device=device)
        self.loc_information_ema = torch.zeros(n, dtype=torch.float32, device=device)
        self.loc_redundancy_ema = torch.zeros(n, dtype=torch.float32, device=device)
        self.loc_prototype = torch.zeros((n, feature_dim), dtype=torch.float32, device=device)
        self.loc_prototype_count = torch.zeros(n, dtype=torch.float32, device=device)
        self.loc_birth_iteration = torch.full((n,), birth_iteration, dtype=torch.long, device=device)
        self.last_topology_iteration = torch.full((n,), birth_iteration, dtype=torch.long, device=device)
        self.loc_node_id = torch.arange(n, dtype=torch.long, device=device)
        self.loc_parent_node_id = torch.full((n,), -1, dtype=torch.long, device=device)
        self.loc_source_index = torch.arange(n, dtype=torch.long, device=device)
        self.loc_source_xyz = self.get_xyz.detach().clone().to(device=device, dtype=torch.float32)
        self._ensure_loc_anchor_state()

    def _default_localization_buffer(self, name, birth_iteration=0):
        n = self.get_xyz.shape[0]
        device = self.get_xyz.device
        feature_dim = self._loc_feature_dim()
        if name in ("loc_grad_accum", "loc_grad_denom"):
            return torch.zeros((n, 1), dtype=torch.float32, device=device)
        if name == "loc_observation_count":
            return torch.zeros(n, dtype=torch.long, device=device)
        if name == "loc_prototype":
            return torch.zeros((n, feature_dim), dtype=torch.float32, device=device)
        if name == "loc_prototype_count":
            return torch.zeros(n, dtype=torch.float32, device=device)
        if name in ("loc_birth_iteration", "last_topology_iteration"):
            return torch.full((n,), birth_iteration, dtype=torch.long, device=device)
        if name in ("loc_source_index", "loc_node_id"):
            return torch.arange(n, dtype=torch.long, device=device)
        if name == "loc_parent_node_id":
            return torch.full((n,), -1, dtype=torch.long, device=device)
        if name == "loc_source_xyz":
            return self.get_xyz.detach().clone().to(device=device, dtype=torch.float32)
        return torch.zeros(n, dtype=torch.float32, device=device)

    def _ensure_localization_state(self, birth_iteration=0):
        n = self.get_xyz.shape[0]
        device = self.get_xyz.device
        loc_opacity = getattr(self, "_loc_opacity", None)
        if not torch.is_tensor(loc_opacity) or loc_opacity.shape[0] != n:
            if torch.is_tensor(self._opacity) and self._opacity.numel() == n:
                loc_opacity = self._opacity.detach().clone()
            else:
                loc_opacity = inverse_sigmoid(torch.full((n, 1), 0.1, dtype=torch.float32, device=device))
            self._loc_opacity = nn.Parameter(loc_opacity.to(device=device).detach().clone().requires_grad_(True))
        for name in self._localization_buffer_names():
            value = getattr(self, name, None)
            if not torch.is_tensor(value) or value.shape[0] != n:
                setattr(self, name, self._default_localization_buffer(name, birth_iteration=birth_iteration))
        self._ensure_loc_anchor_state()

    def _ensure_loc_anchor_state(self):
        n = self.get_xyz.shape[0]
        device = self.get_xyz.device
        dtype = self.get_xyz.dtype if self.get_xyz.is_floating_point() else torch.float32
        value = getattr(self, "_loc_anchor_offset", None)
        if torch.is_tensor(value) and value.shape == (n, 3):
            if value.device != device or value.dtype != dtype:
                self._loc_anchor_offset = nn.Parameter(
                    value.to(device=device, dtype=dtype).detach().clone().requires_grad_(True)
                )
            return
        self._loc_anchor_offset = nn.Parameter(torch.zeros((n, 3), dtype=dtype, device=device).requires_grad_(True))

    def loc_anchor_offset_regularization(self):
        self._ensure_loc_anchor_state()
        if self._loc_anchor_offset.numel() == 0:
            return self.get_xyz.new_tensor(0.0)
        return torch.tanh(self._loc_anchor_offset).square().mean()

    def _ensure_screen_radius_state(self):
        n = self.get_xyz.shape[0]
        radii = getattr(self, "max_radii2D", None)
        if not torch.is_tensor(radii) or radii.shape[0] != n:
            self.max_radii2D = torch.zeros((n,), dtype=torch.float32, device=self.get_xyz.device)

    @property
    def get_loc_opacity(self):
        if not torch.is_tensor(getattr(self, "_loc_opacity", None)) or self._loc_opacity.numel() == 0:
            return self.get_opacity
        return self.opacity_activation(self._loc_opacity)

    def update_screen_radii(self, point_selector, radii):
        if point_selector is None or radii is None:
            return
        self._ensure_screen_radius_state()
        selector = point_selector.to(device=self.get_xyz.device)
        radii = torch.as_tensor(radii, device=self.get_xyz.device, dtype=torch.float32).reshape(-1)
        if selector.dtype == torch.bool:
            selector = selector.reshape(-1)
            if selector.shape[0] != self.get_xyz.shape[0]:
                return
            count = int(selector.sum().item())
            if count == 0:
                return
            values = radii[selector] if radii.numel() == selector.numel() else radii[:count]
            self.max_radii2D[selector] = torch.maximum(self.max_radii2D[selector], values)
            return
        idx = selector.to(dtype=torch.long).reshape(-1)
        if idx.numel() == 0:
            return
        values = radii[idx] if radii.numel() == self.get_xyz.shape[0] else radii[: idx.numel()]
        self.max_radii2D[idx] = torch.maximum(self.max_radii2D[idx], values)

    def add_localization_stats(self, full_idx, means2d_grad=None, radii=None, episode_stats=None, ema_decay=0.95):
        self._ensure_localization_state()
        full_idx = torch.as_tensor(full_idx, device=self.get_xyz.device, dtype=torch.long).reshape(-1)
        if full_idx.numel() == 0:
            return
        self.update_screen_radii(full_idx, radii)
        if means2d_grad is not None:
            grad = means2d_grad.detach().reshape(full_idx.numel(), -1)
            self.loc_grad_accum[full_idx] += torch.linalg.norm(grad[:, :2], dim=-1, keepdim=True)
            self.loc_grad_denom[full_idx] += 1
        self.loc_observation_count[full_idx] += 1
        if episode_stats is None:
            return
        update_mask = torch.as_tensor(
            episode_stats.get("update_mask", torch.ones(full_idx.numel(), device=self.get_xyz.device)),
            device=self.get_xyz.device,
            dtype=torch.bool,
        ).reshape(-1)
        if update_mask.numel() == 1:
            update_mask = update_mask.expand(full_idx.numel())
        update_mask = update_mask[: full_idx.numel()]

        def as_vector(key):
            value = episode_stats.get(key)
            if value is None:
                return None
            value = torch.as_tensor(value, device=self.get_xyz.device, dtype=torch.float32).reshape(-1)
            if value.numel() == 1:
                value = value.expand(full_idx.numel())
            return value[: full_idx.numel()]

        stat_map = {
            "repeatability": "loc_repeatability_ema",
            "positive_prob": "loc_positive_prob_ema",
            "margin": "loc_margin_ema",
            "entropy": "loc_entropy_ema",
            "outlier": "loc_outlier_ema",
            "reproj_error": "loc_reproj_error_ema",
            "information": "loc_information_ema",
            "redundancy": "loc_redundancy_ema",
        }
        for key, attr in stat_map.items():
            value = as_vector(key)
            if value is not None and bool(update_mask.any().item()):
                target_idx = full_idx[update_mask]
                old = getattr(self, attr)[target_idx]
                getattr(self, attr)[target_idx] = float(ema_decay) * old + (1.0 - float(ema_decay)) * value[update_mask]
        prototype = episode_stats.get("prototype")
        if prototype is not None and self.loc_prototype.shape[1] > 0 and bool(update_mask.any().item()):
            prototype = torch.as_tensor(prototype, device=self.get_xyz.device, dtype=torch.float32)
            prototype = prototype.reshape(full_idx.numel(), -1)[:, : self.loc_prototype.shape[1]]
            target_idx = full_idx[update_mask]
            old = self.loc_prototype[target_idx]
            self.loc_prototype[target_idx] = (
                float(ema_decay) * old + (1.0 - float(ema_decay)) * F.normalize(prototype[update_mask], p=2, dim=-1)
            )
            self.loc_prototype_count[target_idx] += 1

    def _robust_z(self, value, mask):
        out = torch.zeros_like(value, dtype=torch.float32)
        if mask.sum() == 0:
            return out
        data = value[mask].float()
        median = data.median()
        mad = (data - median).abs().median().clamp_min(1e-6)
        out[mask] = ((value[mask].float() - median) / (1.4826 * mad)).clamp(-5.0, 5.0)
        return out

    def _observed_localization_mask(self, min_observations=8):
        self._ensure_localization_state()
        return self.loc_observation_count >= min_observations

    def compute_landmark_reliability(self, min_observations=8):
        self._ensure_localization_state()
        observed = self._observed_localization_mask(min_observations)
        reliability = (
            self._robust_z(self.loc_repeatability_ema, observed)
            + self._robust_z(self.loc_positive_prob_ema, observed)
            + self._robust_z(self.loc_margin_ema, observed)
            - self._robust_z(self.loc_entropy_ema, observed)
            - self._robust_z(self.loc_outlier_ema, observed)
            - self._robust_z(self.loc_reproj_error_ema, observed)
        )
        reliability[~torch.isfinite(reliability)] = 0.0
        reliability[~observed] = 0.0
        return reliability

    def compute_pose_geometry_value(self, min_observations=8):
        self._ensure_localization_state()
        observed = self._observed_localization_mask(min_observations)
        geometry = self._robust_z(self.loc_information_ema, observed)
        geometry[~torch.isfinite(geometry)] = 0.0
        geometry[~observed] = 0.0
        return geometry

    def compute_localization_utility(self, min_observations=8):
        self._ensure_localization_state()
        observed = self._observed_localization_mask(min_observations)
        utility = (
            self.compute_landmark_reliability(min_observations)
            + self.compute_pose_geometry_value(min_observations)
            - self._robust_z(self.loc_redundancy_ema, observed)
        )
        utility[~torch.isfinite(utility)] = 0.0
        utility[~observed] = 0.0
        return utility

    def capture_localization_state(self):
        self._ensure_localization_state()
        state = {
            "version": 4,
            "loc_opacity": self._loc_opacity.detach(),
            "loc_current_xyz": self.get_loc_xyz.detach(),
            "loc_anchor_offset": self._loc_anchor_offset.detach(),
            "surfel_loc_tangent_bound": float(getattr(self, "surfel_loc_tangent_bound", 0.0) or 0.0),
            "surfel_loc_normal_bound": float(getattr(self, "surfel_loc_normal_bound", 0.0) or 0.0),
            "surfel_loc_radius_floor": float(getattr(self, "surfel_loc_radius_floor", 0.0) or 0.0),
            "detach_loc_anchor_base": bool(getattr(self, "detach_loc_anchor_base", False)),
        }
        for name in self._localization_buffer_names():
            state[name] = getattr(self, name).detach()
        return state

    def restore_localization_state(self, state):
        if state is None:
            self.init_localization_state(from_rgb_opacity=True)
            return
        self._ensure_localization_state()
        device = self.get_xyz.device
        loc_opacity = state.get("loc_opacity", self._opacity.detach().clone()).to(device=device).detach().clone()
        if self.optimizer is not None:
            optimizable_tensors = self.replace_tensor_to_optimizer(loc_opacity, "loc_opacity")
            self._loc_opacity = optimizable_tensors.get("loc_opacity", nn.Parameter(loc_opacity.requires_grad_(True)))
        else:
            self._loc_opacity = nn.Parameter(loc_opacity.requires_grad_(True))
        for name in self._localization_buffer_names():
            if name in state:
                setattr(self, name, state[name].to(device=device).detach().clone())
        self.surfel_loc_tangent_bound = float(state.get("surfel_loc_tangent_bound", getattr(self, "surfel_loc_tangent_bound", 0.0)) or 0.0)
        self.surfel_loc_normal_bound = float(state.get("surfel_loc_normal_bound", getattr(self, "surfel_loc_normal_bound", 0.0)) or 0.0)
        self.surfel_loc_radius_floor = float(state.get("surfel_loc_radius_floor", getattr(self, "surfel_loc_radius_floor", 0.0)) or 0.0)
        self.detach_loc_anchor_base = bool(state.get("detach_loc_anchor_base", getattr(self, "detach_loc_anchor_base", False)))
        loc_anchor_offset = state.get("loc_anchor_offset", None)
        if loc_anchor_offset is not None:
            loc_anchor_offset = loc_anchor_offset.to(device=device, dtype=self.get_xyz.dtype).detach().clone()
            if self.optimizer is not None:
                optimizable_tensors = self.replace_tensor_to_optimizer(loc_anchor_offset, "loc_anchor_offset")
                self._loc_anchor_offset = optimizable_tensors.get(
                    "loc_anchor_offset",
                    nn.Parameter(loc_anchor_offset.requires_grad_(True)),
                )
            else:
                self._loc_anchor_offset = nn.Parameter(loc_anchor_offset.requires_grad_(True))
        self._ensure_localization_state()

    def save_localization_state(self, path):
        mkdir_p(os.path.dirname(path))
        torch.save(self.capture_localization_state(), path)

    def load_localization_state(self, path):
        if os.path.exists(path):
            self.restore_localization_state(torch.load(path, map_location=self.get_xyz.device))
        else:
            self.init_localization_state(from_rgb_opacity=True)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, loc_feature_size : int, speedup: bool):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        _loc_feature = torch.randn(fused_point_cloud.shape[0], loc_feature_size, 1).float().cuda() 
        _loc_feature = F.normalize(_loc_feature, p=2, dim=1)

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._loc_feature = nn.Parameter(_loc_feature.transpose(1, 2).contiguous().requires_grad_(True))
        self.init_localization_state(from_rgb_opacity=True)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self._ensure_localization_state()
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        loc_opacity_lr = getattr(training_args, "loc_opacity_lr", training_args.opacity_lr * 0.1)
        loc_anchor_lr = getattr(training_args, "loc_anchor_lr", 0.0)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._loc_feature], 'lr':training_args.loc_feature_lr, "name": "loc_feature"},
            {'params': [self._loc_opacity], 'lr': loc_opacity_lr, "name": "loc_opacity"},
            {'params': [self._loc_anchor_offset], 'lr': loc_anchor_lr, "name": "loc_anchor_offset"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        # Add loc features
        for i in range(self._loc_feature.shape[1]*self._loc_feature.shape[2]):  
            l.append('loc_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        loc_feature = self._loc_feature.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy() 

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, loc_feature), axis=1)

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, loc_feature_dim=None):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        count = sum(1 for name in plydata.elements[0].data.dtype.names if name.startswith("loc_"))
        if count > 0:
            loc_feature = np.stack([np.asarray(plydata.elements[0][f"loc_{i}"]) for i in range(count)], axis=1)
            loc_feature = np.expand_dims(loc_feature, axis=-1)
        else:
            feature_dim = int(loc_feature_dim or 256)
            if feature_dim <= 0:
                raise ValueError(f"loc_feature_dim must be positive for RGB-only PLY loading, got {feature_dim}")
            rng = np.random.default_rng(0)
            loc_feature = rng.standard_normal((xyz.shape[0], feature_dim, 1)).astype(np.float32)
            norm = np.linalg.norm(loc_feature, axis=1, keepdims=True)
            loc_feature = loc_feature / np.clip(norm, 1e-12, None)
            print(
                "Loaded RGB-only Gaussian PLY without loc_* fields; "
                f"initialized LaFGS loc_feature with dim={feature_dim}"
            )

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self._loc_feature = nn.Parameter(torch.tensor(loc_feature, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.init_localization_state(from_rgb_opacity=True)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                old_param = group["params"][0]
                stored_state = self.optimizer.state.get(old_param, None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[old_param]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                if stored_state is not None:
                    self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._loc_feature = optimizable_tensors["loc_feature"]
        if "loc_opacity" in optimizable_tensors:
            self._loc_opacity = optimizable_tensors["loc_opacity"]
        if "loc_anchor_offset" in optimizable_tensors:
            self._loc_anchor_offset = optimizable_tensors["loc_anchor_offset"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self._prune_localization_buffers(valid_points_mask)

    def _new_localization_node_ids(self, count, existing_node_ids=None):
        count = int(count)
        device = self.get_xyz.device
        if count <= 0:
            return torch.empty((0,), dtype=torch.long, device=device)
        if existing_node_ids is None:
            existing_node_ids = getattr(self, "loc_node_id", None)
        if torch.is_tensor(existing_node_ids) and existing_node_ids.numel() > 0:
            start = int(existing_node_ids.to(dtype=torch.long).max().item()) + 1
        else:
            start = 0
        return torch.arange(start, start + count, dtype=torch.long, device=device)

    def _cat_localization_buffers(self, parent_mask, repeat=1, birth_iteration=None):
        if parent_mask is None:
            n = self.get_xyz.shape[0]
            self.init_localization_state(from_rgb_opacity=True)
            assert self.get_xyz.shape[0] == n
            return
        parent_mask = torch.as_tensor(parent_mask, device=self.get_xyz.device, dtype=torch.bool)
        originals = {name: getattr(self, name) for name in self._localization_buffer_names()}
        for name in self._localization_buffer_names():
            value = originals[name]
            if value.shape[0] != parent_mask.shape[0]:
                raise RuntimeError(
                    f"Cannot extend localization buffer {name}: "
                    f"buffer has {value.shape[0]} rows, parent mask has {parent_mask.shape[0]}."
                )
            if name == "loc_node_id":
                parent_count = int(parent_mask.sum().item())
                extension = self._new_localization_node_ids(
                    parent_count * int(repeat),
                    existing_node_ids=originals.get("loc_node_id"),
                )
            elif name == "loc_parent_node_id":
                parent_node_id = originals.get("loc_node_id")
                if not torch.is_tensor(parent_node_id) or parent_node_id.shape[0] != parent_mask.shape[0]:
                    parent_node_id = torch.arange(parent_mask.shape[0], dtype=torch.long, device=value.device)
                extension = parent_node_id[parent_mask].to(device=value.device, dtype=value.dtype)
                if int(repeat) != 1:
                    extension = extension.repeat(int(repeat))
            else:
                extension = value[parent_mask]
                if int(repeat) != 1:
                    extension = extension.repeat(*([int(repeat)] + [1] * (extension.dim() - 1)))
                if birth_iteration is not None and name in ("loc_birth_iteration", "last_topology_iteration"):
                    extension = torch.full_like(extension, int(birth_iteration))
            setattr(self, name, torch.cat([value, extension], dim=0))

    def _prune_localization_buffers(self, valid_points_mask):
        for name in self._localization_buffer_names():
            value = getattr(self, name, None)
            if torch.is_tensor(value) and value.shape[0] == valid_points_mask.shape[0]:
                setattr(self, name, value[valid_points_mask])

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_loc_feature,
        new_loc_opacity=None,
        new_loc_anchor_offset=None,
        loc_parent_mask=None,
        loc_repeat=1,
        loc_birth_iteration=None,
    ):
        if new_loc_opacity is None:
            new_loc_opacity = new_opacities.detach().clone()
        if new_loc_anchor_offset is None:
            new_loc_anchor_offset = torch.zeros((new_xyz.shape[0], 3), dtype=self._loc_anchor_offset.dtype, device=self._loc_anchor_offset.device)
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "loc_feature": new_loc_feature,
        "loc_opacity": new_loc_opacity,
        "loc_anchor_offset": new_loc_anchor_offset} 

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._loc_feature = optimizable_tensors["loc_feature"] 
        self._loc_opacity = optimizable_tensors["loc_opacity"]
        self._loc_anchor_offset = optimizable_tensors["loc_anchor_offset"]
        self._cat_localization_buffers(
            loc_parent_mask,
            repeat=loc_repeat,
            birth_iteration=loc_birth_iteration,
        )

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, loc_birth_iteration=None):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_loc_feature = self._loc_feature[selected_pts_mask].repeat(N,1,1) 
        new_loc_opacity = self._loc_opacity[selected_pts_mask].repeat(N, 1)
        new_loc_anchor_offset = self._loc_anchor_offset[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_loc_feature,
            new_loc_opacity,
            new_loc_anchor_offset,
            selected_pts_mask,
            N,
            loc_birth_iteration,
        ) 
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, loc_birth_iteration=None):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_loc_feature = self._loc_feature[selected_pts_mask] 
        new_loc_opacity = self._loc_opacity[selected_pts_mask]
        new_loc_anchor_offset = self._loc_anchor_offset[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_loc_feature,
            new_loc_opacity,
            new_loc_anchor_offset,
            selected_pts_mask,
            1,
            loc_birth_iteration,
        ) 

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, loc_birth_iteration=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, loc_birth_iteration=loc_birth_iteration)
        self.densify_and_split(grads, max_grad, extent, loc_birth_iteration=loc_birth_iteration)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats_gsplat(self, viewspace_point_tensor, update_filter, width, height):
        grad = viewspace_point_tensor.grad.squeeze(0) # [N, 2]
        # Normalize the gradient to [-1, 1] screen size
        grad[:, 0] *= width * 0.5
        grad[:, 1] *= height * 0.5
        self.xyz_gradient_accum[update_filter] += torch.norm(grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

class GaussianModel(nn.Module):
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        super(GaussianModel, self).__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self._loc_feature = torch.empty(0)
        self._loc_opacity = torch.empty(0)
        self._loc_overlay_feature = torch.empty(0)
        self._loc_overlay_active_logit = torch.empty(0)
        self.loc_overlay_source_index = torch.empty(0, dtype=torch.long)
        self.loc_overlay_max_residual_norm = 0.0
        self.loc_overlay_normalize = False

    def _loc_feature_dim(self):
        if self._loc_feature.numel() == 0:
            return 0
        return self._loc_feature.reshape(self._loc_feature.shape[0], -1).shape[1]

    def _has_descriptor_overlay(self):
        overlay_feature = getattr(self, "_loc_overlay_feature", None)
        overlay_active = getattr(self, "_loc_overlay_active_logit", None)
        overlay_source = getattr(self, "loc_overlay_source_index", None)
        if not torch.is_tensor(overlay_feature) or overlay_feature.numel() == 0:
            return False
        if not torch.is_tensor(overlay_active) or overlay_active.shape[0] != overlay_feature.shape[0]:
            return False
        if not torch.is_tensor(overlay_source) or overlay_source.numel() != overlay_feature.shape[0]:
            return False
        return True

    def clear_descriptor_overlay(self):
        device = self._loc_feature.device if torch.is_tensor(self._loc_feature) else torch.device("cpu")
        self._loc_overlay_feature = nn.Parameter(torch.empty(0, device=device), requires_grad=True)
        self._loc_overlay_active_logit = nn.Parameter(torch.empty(0, device=device), requires_grad=True)
        self.loc_overlay_source_index = torch.empty(0, dtype=torch.long, device=device)
        self.loc_overlay_max_residual_norm = 0.0
        self.loc_overlay_normalize = False

    def init_descriptor_overlay(
        self,
        source_indices,
        init_active_logit=0.0,
        reset=False,
        max_residual_norm=0.0,
        normalize=False,
    ):
        if self._loc_feature.numel() == 0:
            raise ValueError("descriptor overlay requires initialized localization features")
        device = self._loc_feature.device
        dtype = self._loc_feature.dtype
        source_indices = torch.as_tensor(source_indices, dtype=torch.long, device=device).reshape(-1)
        if source_indices.numel() == 0:
            self.clear_descriptor_overlay()
            return False
        source_indices = torch.unique(source_indices, sorted=True)
        feature_shape = (source_indices.numel(),) + tuple(self._loc_feature.shape[1:])
        active_shape = (source_indices.numel(),) + (1,) * max(self._loc_feature.dim() - 1, 1)
        overlay_feature = torch.zeros(feature_shape, dtype=dtype, device=device)
        overlay_active = torch.full(active_shape, float(init_active_logit), dtype=dtype, device=device)

        if self._has_descriptor_overlay() and not reset:
            old_sources = self.loc_overlay_source_index.to(device=device, dtype=torch.long)
            old_feature = self._loc_overlay_feature.detach().to(device=device, dtype=dtype)
            old_active = self._loc_overlay_active_logit.detach().to(device=device, dtype=dtype)
            old_pos = torch.searchsorted(old_sources, source_indices)
            old_pos_clamped = old_pos.clamp(max=max(old_sources.numel() - 1, 0))
            has_old = (old_pos < old_sources.numel()) & (old_sources[old_pos_clamped] == source_indices)
            if bool(has_old.any()):
                overlay_feature[has_old] = old_feature[old_pos[has_old]]
                overlay_active[has_old] = old_active[old_pos[has_old]]

        self._loc_overlay_feature = nn.Parameter(overlay_feature.requires_grad_(True))
        self._loc_overlay_active_logit = nn.Parameter(overlay_active.requires_grad_(True))
        self.loc_overlay_source_index = source_indices.detach().clone()
        self.loc_overlay_max_residual_norm = float(max_residual_norm)
        self.loc_overlay_normalize = bool(normalize)
        return True

    def add_descriptor_overlay_to_optimizer(self, lr=None):
        if not self._has_descriptor_overlay() or self.optimizer is None:
            return False
        if lr is None or float(lr) <= 0.0:
            lr = None
            for group in self.optimizer.param_groups:
                if group.get("name") == "loc_feature":
                    lr = group.get("la_base_lr", group.get("lr", 0.0))
                    break
            if lr is None:
                lr = 0.0
        lr = float(lr)
        params = {
            "loc_overlay_feature": self._loc_overlay_feature,
            "loc_overlay_active_logit": self._loc_overlay_active_logit,
        }
        existing = {group.get("name"): group for group in self.optimizer.param_groups}
        for name, param in params.items():
            group = existing.get(name)
            if group is None:
                self.optimizer.add_param_group({"params": [param], "lr": lr, "name": name, "la_base_lr": lr})
            else:
                group["params"] = [param]
                group["lr"] = lr
                group["la_base_lr"] = lr
        return True

    def materialized_loc_feature(self, indices=None):
        features = self.get_loc_feature
        if indices is not None:
            indices = torch.as_tensor(indices, dtype=torch.long, device=features.device)
            features = features[indices]
        return features

    def _has_localization_state(self):
        n = self.get_xyz.shape[0]
        loc_opacity = getattr(self, "_loc_opacity", None)
        if not torch.is_tensor(loc_opacity) or loc_opacity.shape[0] != n:
            return False
        for name in self._localization_buffer_names():
            value = getattr(self, name, None)
            if not torch.is_tensor(value) or value.shape[0] != n:
                return False
        return True

    def _localization_buffer_names(self):
        return [
            "loc_grad_accum",
            "loc_grad_denom",
            "loc_observation_count",
            "loc_repeatability_ema",
            "loc_positive_prob_ema",
            "loc_margin_ema",
            "loc_entropy_ema",
            "loc_outlier_ema",
            "loc_reproj_error_ema",
            "loc_information_ema",
            "loc_redundancy_ema",
            "loc_prototype",
            "loc_prototype_count",
            "loc_birth_iteration",
            "last_topology_iteration",
            "loc_node_id",
            "loc_parent_node_id",
            "loc_source_index",
            "loc_source_xyz",
        ]

    def init_localization_state(self, from_rgb_opacity=True, birth_iteration=0):
        n = self.get_xyz.shape[0]
        device = self.get_xyz.device
        feature_dim = self._loc_feature_dim()
        if from_rgb_opacity and self._opacity.numel() == n:
            loc_opacity = self._opacity.detach().clone()
        else:
            loc_opacity = inverse_sigmoid(torch.full((n, 1), 0.1, dtype=torch.float32, device=device))
        self._loc_opacity = nn.Parameter(loc_opacity.requires_grad_(True))

        self.loc_grad_accum = torch.zeros((n, 1), dtype=torch.float32, device=device)
        self.loc_grad_denom = torch.zeros((n, 1), dtype=torch.float32, device=device)
        self.loc_observation_count = torch.zeros(n, dtype=torch.long, device=device)
        self.loc_repeatability_ema = torch.zeros(n, dtype=torch.float32, device=device)
        self.loc_positive_prob_ema = torch.zeros(n, dtype=torch.float32, device=device)
        self.loc_margin_ema = torch.zeros(n, dtype=torch.float32, device=device)
        self.loc_entropy_ema = torch.zeros(n, dtype=torch.float32, device=device)
        self.loc_outlier_ema = torch.zeros(n, dtype=torch.float32, device=device)
        self.loc_reproj_error_ema = torch.zeros(n, dtype=torch.float32, device=device)
        self.loc_information_ema = torch.zeros(n, dtype=torch.float32, device=device)
        self.loc_redundancy_ema = torch.zeros(n, dtype=torch.float32, device=device)
        self.loc_prototype = torch.zeros((n, feature_dim), dtype=torch.float32, device=device)
        self.loc_prototype_count = torch.zeros(n, dtype=torch.float32, device=device)
        self.loc_birth_iteration = torch.full((n,), birth_iteration, dtype=torch.long, device=device)
        self.last_topology_iteration = torch.full((n,), birth_iteration, dtype=torch.long, device=device)
        self.loc_node_id = torch.arange(n, dtype=torch.long, device=device)
        self.loc_parent_node_id = torch.full((n,), -1, dtype=torch.long, device=device)
        self.loc_source_index = torch.arange(n, dtype=torch.long, device=device)
        self.loc_source_xyz = self.get_xyz.detach().clone().to(device=device, dtype=torch.float32)

    def _default_localization_buffer(self, name, birth_iteration=0):
        n = self.get_xyz.shape[0]
        device = self.get_xyz.device
        feature_dim = self._loc_feature_dim()
        if name in ("loc_grad_accum", "loc_grad_denom"):
            return torch.zeros((n, 1), dtype=torch.float32, device=device)
        if name == "loc_observation_count":
            return torch.zeros(n, dtype=torch.long, device=device)
        if name == "loc_prototype":
            return torch.zeros((n, feature_dim), dtype=torch.float32, device=device)
        if name == "loc_prototype_count":
            return torch.zeros(n, dtype=torch.float32, device=device)
        if name in ("loc_birth_iteration", "last_topology_iteration"):
            return torch.full((n,), birth_iteration, dtype=torch.long, device=device)
        if name in ("loc_source_index", "loc_node_id"):
            return torch.arange(n, dtype=torch.long, device=device)
        if name == "loc_parent_node_id":
            return torch.full((n,), -1, dtype=torch.long, device=device)
        if name == "loc_source_xyz":
            return self.get_xyz.detach().clone().to(device=device, dtype=torch.float32)
        return torch.zeros(n, dtype=torch.float32, device=device)

    def _ensure_localization_state(self, birth_iteration=0):
        n = self.get_xyz.shape[0]
        device = self.get_xyz.device
        loc_opacity = getattr(self, "_loc_opacity", None)
        if not torch.is_tensor(loc_opacity) or loc_opacity.shape[0] != n:
            if self._opacity.numel() == n:
                loc_opacity = self._opacity.detach().clone()
            else:
                loc_opacity = inverse_sigmoid(torch.full((n, 1), 0.1, dtype=torch.float32, device=device))
            self._loc_opacity = nn.Parameter(loc_opacity.to(device=device).detach().clone().requires_grad_(True))

        for name in self._localization_buffer_names():
            value = getattr(self, name, None)
            if not torch.is_tensor(value) or value.shape[0] != n:
                setattr(self, name, self._default_localization_buffer(name, birth_iteration=birth_iteration))

    def _ensure_screen_radius_state(self):
        n = self.get_xyz.shape[0]
        radii = getattr(self, "max_radii2D", None)
        if not torch.is_tensor(radii) or radii.shape[0] != n:
            self.max_radii2D = torch.zeros((n,), dtype=torch.float32, device=self.get_xyz.device)

    def update_screen_radii(self, point_selector, radii):
        if point_selector is None or radii is None:
            return
        self._ensure_screen_radius_state()
        n = self.get_xyz.shape[0]
        device = self.get_xyz.device
        selector = point_selector.to(device=device)
        radii = torch.as_tensor(radii, device=device, dtype=torch.float32).reshape(-1)

        if selector.dtype == torch.bool:
            selector = selector.reshape(-1)
            if selector.shape[0] != n:
                raise RuntimeError(
                    f"Cannot update screen radii: visibility mask has {selector.shape[0]} rows, model has {n} points."
                )
            count = int(selector.sum().item())
            if count == 0:
                return
            if radii.numel() == n:
                values = radii[selector]
            elif radii.numel() == count:
                values = radii
            elif radii.numel() == 1:
                values = radii.expand(count)
            else:
                raise RuntimeError(
                    f"Cannot update screen radii: got {radii.numel()} radii for {count} selected points."
                )
            self.max_radii2D[selector] = torch.maximum(self.max_radii2D[selector], values)
            return

        idx = selector.to(dtype=torch.long).reshape(-1)
        if idx.numel() == 0:
            return
        if radii.numel() == n:
            values = radii[idx]
        elif radii.numel() == idx.numel():
            values = radii
        elif radii.numel() == 1:
            values = radii.expand(idx.numel())
        else:
            raise RuntimeError(
                f"Cannot update screen radii: got {radii.numel()} radii for {idx.numel()} selected points."
            )
        self.max_radii2D[idx] = torch.maximum(self.max_radii2D[idx], values)

    def _maybe_load_optimizer_state(self, opt_dict):
        if not opt_dict:
            return
        expected = len(self.optimizer.param_groups)
        actual = len(opt_dict.get("param_groups", []))
        if actual == expected:
            self.optimizer.load_state_dict(opt_dict)
        else:
            print(
                f"[LA-STDLoc] Skip optimizer restore: checkpoint has {actual} "
                f"param groups, current model has {expected}."
            )

    def _new_localization_node_ids(self, count, existing_node_ids=None):
        count = int(count)
        device = self.get_xyz.device
        if count <= 0:
            return torch.empty((0,), dtype=torch.long, device=device)
        if existing_node_ids is None:
            existing_node_ids = getattr(self, "loc_node_id", None)
        if torch.is_tensor(existing_node_ids) and existing_node_ids.numel() > 0:
            start = int(existing_node_ids.to(dtype=torch.long).max().item()) + 1
        else:
            start = 0
        return torch.arange(start, start + count, dtype=torch.long, device=device)

    def _cat_localization_buffers(self, parent_mask, repeat=1, birth_iteration=None):
        if parent_mask is None:
            n = self.get_xyz.shape[0]
            self.init_localization_state(from_rgb_opacity=True)
            assert self.get_xyz.shape[0] == n
            return
        originals = {name: getattr(self, name) for name in self._localization_buffer_names()}
        for name in self._localization_buffer_names():
            value = originals[name]
            if value.shape[0] != parent_mask.shape[0]:
                raise RuntimeError(
                    f"Cannot extend localization buffer {name}: "
                    f"buffer has {value.shape[0]} rows, parent mask has {parent_mask.shape[0]}."
                )
            if name == "loc_node_id":
                parent_count = int(parent_mask.to(dtype=torch.bool).sum().item())
                extension = self._new_localization_node_ids(
                    parent_count * int(repeat),
                    existing_node_ids=originals.get("loc_node_id"),
                )
            elif name == "loc_parent_node_id":
                parent_node_id = originals.get("loc_node_id")
                if not torch.is_tensor(parent_node_id) or parent_node_id.shape[0] != parent_mask.shape[0]:
                    parent_node_id = torch.arange(parent_mask.shape[0], dtype=torch.long, device=value.device)
                extension = parent_node_id[parent_mask].to(device=value.device, dtype=value.dtype)
                if repeat != 1:
                    extension = extension.repeat(int(repeat))
            else:
                extension = value[parent_mask]
            if repeat != 1 and name not in ("loc_node_id", "loc_parent_node_id"):
                reps = [repeat] + [1] * (extension.dim() - 1)
                extension = extension.repeat(*reps)
            if birth_iteration is not None and name in ("loc_birth_iteration", "last_topology_iteration"):
                extension = torch.full_like(extension, int(birth_iteration))
            setattr(self, name, torch.cat([value, extension], dim=0))

    def _prune_localization_buffers(self, valid_points_mask):
        for name in self._localization_buffer_names():
            value = getattr(self, name)
            if value.shape[0] != valid_points_mask.shape[0]:
                raise RuntimeError(
                    f"Cannot prune localization buffer {name}: "
                    f"buffer has {value.shape[0]} rows, prune mask has {valid_points_mask.shape[0]}."
                )
            setattr(self, name, value[valid_points_mask])

    @property
    def get_loc_opacity(self):
        if not hasattr(self, "_loc_opacity") or self._loc_opacity.numel() == 0:
            return self.get_opacity
        return self.opacity_activation(self._loc_opacity)

    def add_localization_stats(
        self,
        full_idx,
        means2d_grad=None,
        radii=None,
        episode_stats=None,
        ema_decay=0.95,
    ):
        self._ensure_localization_state()
        if full_idx is None:
            return
        full_idx = full_idx.to(device=self.get_xyz.device, dtype=torch.long).reshape(-1)
        if full_idx.numel() == 0:
            return
        self.update_screen_radii(full_idx, radii)

        if means2d_grad is not None:
            grad = means2d_grad.detach()
            if grad.dim() == 3:
                grad = grad.squeeze(0)
            grad = grad.reshape(full_idx.numel(), -1)
            grad_norm = torch.linalg.norm(grad[:, :2], dim=-1, keepdim=True)
            self.loc_grad_accum[full_idx] += grad_norm
            self.loc_grad_denom[full_idx] += 1

        self.loc_observation_count[full_idx] += 1
        if episode_stats is None:
            return

        update_mask = episode_stats.get("update_mask")
        if update_mask is None:
            update_mask = torch.ones(full_idx.numel(), device=self.get_xyz.device, dtype=torch.bool)
        else:
            update_mask = torch.as_tensor(update_mask, device=self.get_xyz.device, dtype=torch.bool).reshape(-1)
            if update_mask.numel() == 1:
                update_mask = update_mask.expand(full_idx.numel())
            update_mask = update_mask[: full_idx.numel()]

        def as_vector(key, default=None):
            value = episode_stats.get(key, default)
            if value is None:
                return None
            value = torch.as_tensor(value, device=self.get_xyz.device, dtype=torch.float32).reshape(-1)
            if value.numel() == 1:
                value = value.expand(full_idx.numel())
            return value[: full_idx.numel()]

        stat_map = {
            "repeatability": "loc_repeatability_ema",
            "positive_prob": "loc_positive_prob_ema",
            "margin": "loc_margin_ema",
            "entropy": "loc_entropy_ema",
            "outlier": "loc_outlier_ema",
            "reproj_error": "loc_reproj_error_ema",
            "information": "loc_information_ema",
            "redundancy": "loc_redundancy_ema",
        }
        for key, attr in stat_map.items():
            value = as_vector(key)
            if value is not None and update_mask.any():
                target_idx = full_idx[update_mask]
                old = getattr(self, attr)[target_idx]
                getattr(self, attr)[target_idx] = ema_decay * old + (1.0 - ema_decay) * value[update_mask]

        prototype = episode_stats.get("prototype")
        if prototype is not None and self.loc_prototype.shape[1] > 0 and update_mask.any():
            prototype = torch.as_tensor(prototype, device=self.get_xyz.device, dtype=torch.float32)
            prototype = prototype.reshape(full_idx.numel(), -1)[:, : self.loc_prototype.shape[1]]
            target_idx = full_idx[update_mask]
            old = self.loc_prototype[target_idx]
            self.loc_prototype[target_idx] = (
                ema_decay * old + (1.0 - ema_decay) * F.normalize(prototype[update_mask], p=2, dim=-1)
            )
            self.loc_prototype_count[target_idx] += 1

    def add_sparse_match_label_stats(
        self,
        full_idx,
        visible_count,
        matched_count,
        correct_count,
        inlier_count,
        ema_decay=0.95,
    ):
        self._ensure_localization_state()
        full_idx = torch.as_tensor(full_idx, device=self.get_xyz.device, dtype=torch.long).reshape(-1)
        if full_idx.numel() == 0:
            return

        def count_tensor(value):
            return torch.as_tensor(value, device=self.get_xyz.device, dtype=torch.float32).reshape(-1)[: full_idx.numel()]

        visible = count_tensor(visible_count)
        matched = count_tensor(matched_count)
        correct = count_tensor(correct_count)
        inlier = count_tensor(inlier_count)
        valid_visible = visible > 0
        valid_matched = matched > 0
        if valid_visible.any():
            self.loc_observation_count[full_idx[valid_visible]] += visible[valid_visible].to(dtype=torch.long)

        repeatability = torch.zeros_like(visible)
        repeatability[valid_visible] = correct[valid_visible] / visible[valid_visible].clamp_min(1.0)
        positive_prob = torch.zeros_like(matched)
        positive_prob[valid_matched] = correct[valid_matched] / matched[valid_matched].clamp_min(1.0)
        inlier_rate = torch.zeros_like(matched)
        inlier_rate[valid_matched] = inlier[valid_matched] / matched[valid_matched].clamp_min(1.0)
        outlier_rate = torch.zeros_like(matched)
        outlier_rate[valid_matched] = 1.0 - inlier_rate[valid_matched]
        entropy = -(positive_prob * positive_prob.clamp_min(1e-6).log())
        margin = 2.0 * positive_prob - 1.0

        updates = {
            "loc_repeatability_ema": repeatability,
            "loc_positive_prob_ema": positive_prob,
            "loc_information_ema": inlier_rate,
            "loc_outlier_ema": outlier_rate,
            "loc_entropy_ema": entropy,
            "loc_margin_ema": margin,
        }
        valid_any = valid_visible | valid_matched
        for attr, value in updates.items():
            if valid_any.any():
                old = getattr(self, attr)[full_idx[valid_any]]
                getattr(self, attr)[full_idx[valid_any]] = (
                    ema_decay * old + (1.0 - ema_decay) * value[valid_any]
                )

    def _robust_z(self, value, mask):
        out = torch.zeros_like(value, dtype=torch.float32)
        if mask.sum() == 0:
            return out
        data = value[mask].float()
        median = data.median()
        mad = (data - median).abs().median().clamp_min(1e-6)
        out[mask] = ((value[mask].float() - median) / (1.4826 * mad)).clamp(-5.0, 5.0)
        return out

    def _observed_localization_mask(self, min_observations=8):
        self._ensure_localization_state()
        return self.loc_observation_count >= min_observations

    def compute_landmark_reliability(self, min_observations=8):
        self._ensure_localization_state()
        observed = self._observed_localization_mask(min_observations)
        reliability = (
            self._robust_z(self.loc_repeatability_ema, observed)
            + self._robust_z(self.loc_positive_prob_ema, observed)
            + self._robust_z(self.loc_margin_ema, observed)
            - self._robust_z(self.loc_entropy_ema, observed)
            - self._robust_z(self.loc_outlier_ema, observed)
            - self._robust_z(self.loc_reproj_error_ema, observed)
        )
        reliability[~torch.isfinite(reliability)] = 0.0
        reliability[~observed] = 0.0
        return reliability

    def compute_pose_geometry_value(self, min_observations=8):
        self._ensure_localization_state()
        observed = self._observed_localization_mask(min_observations)
        geometry = self._robust_z(self.loc_information_ema, observed)
        geometry[~torch.isfinite(geometry)] = 0.0
        geometry[~observed] = 0.0
        return geometry

    def compute_split_necessity(self, min_observations=8, min_radius=0.0, min_repeatability=0.25):
        self._ensure_localization_state()
        self._ensure_screen_radius_state()
        observed = self._observed_localization_mask(min_observations)
        large = self.max_radii2D >= min_radius
        repeatable = self.loc_repeatability_ema >= min_repeatability
        eligible = observed & large & repeatable
        grad = self.loc_grad_accum.squeeze(-1) / self.loc_grad_denom.squeeze(-1).clamp_min(1.0)
        grad_score = self._robust_z(grad, eligible).clamp_min(0.0)
        entropy_score = (1.0 + self._robust_z(self.loc_entropy_ema, eligible).clamp_min(0.0)) * self.loc_entropy_ema.float().clamp_min(0.0)
        repeatability = self.loc_repeatability_ema.float().clamp(0.0, 1.0)
        raw_confidence = self.loc_positive_prob_ema.float()
        teacher_confidence = torch.where(raw_confidence > 0, raw_confidence.clamp(0.0, 1.0), torch.ones_like(raw_confidence))
        pose_effective = self.loc_information_ema.float().clamp(0.0, 1.0)
        if not bool((pose_effective[eligible] > 0).any()):
            pose_effective = torch.ones_like(pose_effective)
        pnp_residual = self.loc_reproj_error_ema.float().clamp_min(0.0)
        if not bool((pnp_residual[eligible] > 0).any()):
            pnp_residual = 1.0 + grad_score
        else:
            pnp_residual = pnp_residual * (1.0 + grad_score)
        split_score = pose_aware_split_score(
            footprint=self.max_radii2D.float(),
            ambiguity=entropy_score,
            pnp_residual=pnp_residual,
            repeatability=repeatability,
            positive_prob=teacher_confidence,
            pose_information=pose_effective,
            min_footprint=min_radius,
            min_repeatability=min_repeatability,
        )
        split_score[~torch.isfinite(split_score)] = 0.0
        split_score[~eligible] = 0.0
        return split_score

    def compute_localization_utility(self, min_observations=8):
        self._ensure_localization_state()
        observed = self._observed_localization_mask(min_observations)
        utility = (
            self.compute_landmark_reliability(min_observations)
            + self.compute_pose_geometry_value(min_observations)
            - self._robust_z(self.loc_redundancy_ema, observed)
        )
        utility[~torch.isfinite(utility)] = 0.0
        utility[~observed] = 0.0
        return utility

    def capture_localization_state(self):
        self._ensure_localization_state()
        state = {
            "version": 2,
            "loc_opacity": self._loc_opacity.detach(),
            "loc_current_xyz": self.get_loc_xyz.detach(),
        }
        for name in self._localization_buffer_names():
            state[name] = getattr(self, name).detach()
        if self._has_descriptor_overlay():
            state["loc_overlay_source_index"] = self.loc_overlay_source_index.detach()
            state["loc_overlay_feature"] = self._loc_overlay_feature.detach()
            state["loc_overlay_active_logit"] = self._loc_overlay_active_logit.detach()
            state["loc_overlay_max_residual_norm"] = float(getattr(self, "loc_overlay_max_residual_norm", 0.0))
            state["loc_overlay_normalize"] = bool(getattr(self, "loc_overlay_normalize", False))
        return state

    def restore_localization_state(self, state):
        if state is None:
            self.init_localization_state(from_rgb_opacity=True)
            return
        device = self.get_xyz.device
        loc_opacity = state.get("loc_opacity", None)
        if loc_opacity is None:
            loc_opacity = self._opacity.detach().clone()
        loc_opacity = loc_opacity.to(device=device).detach().clone()
        if self.optimizer is not None:
            optimizable_tensors = self.replace_tensor_to_optimizer(loc_opacity, "loc_opacity")
            self._loc_opacity = optimizable_tensors.get(
                "loc_opacity",
                nn.Parameter(loc_opacity.requires_grad_(True)),
            )
        else:
            self._loc_opacity = nn.Parameter(loc_opacity.requires_grad_(True))
        for name in self._localization_buffer_names():
            if name in state:
                setattr(self, name, state[name].to(device=device).detach().clone())
        self._ensure_localization_state()
        overlay_feature = state.get("loc_overlay_feature", None)
        overlay_source = state.get("loc_overlay_source_index", None)
        overlay_active = state.get("loc_overlay_active_logit", None)
        if overlay_feature is not None and overlay_source is not None:
            overlay_feature = overlay_feature.to(device=device, dtype=self._loc_feature.dtype).detach().clone()
            if overlay_active is None:
                overlay_active = torch.zeros(
                    (overlay_feature.shape[0],) + (1,) * max(overlay_feature.dim() - 1, 1),
                    dtype=overlay_feature.dtype,
                    device=device,
                )
            else:
                overlay_active = overlay_active.to(device=device, dtype=overlay_feature.dtype).detach().clone()
            self._loc_overlay_feature = nn.Parameter(overlay_feature.requires_grad_(True))
            self._loc_overlay_active_logit = nn.Parameter(overlay_active.requires_grad_(True))
            self.loc_overlay_source_index = overlay_source.to(device=device, dtype=torch.long).detach().clone()
            self.loc_overlay_max_residual_norm = float(state.get("loc_overlay_max_residual_norm", 0.0))
            self.loc_overlay_normalize = bool(state.get("loc_overlay_normalize", False))
            self.add_descriptor_overlay_to_optimizer()

    def save_localization_state(self, path):
        mkdir_p(os.path.dirname(path))
        torch.save(self.capture_localization_state(), path)

    def load_localization_state(self, path):
        if os.path.exists(path):
            self.restore_localization_state(torch.load(path, map_location=self.get_xyz.device))
        else:
            self.init_localization_state(from_rgb_opacity=True)

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._loc_feature, 
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self._loc_feature) = model_args 
        self.init_localization_state(from_rgb_opacity=True)
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self._maybe_load_optimizer_state(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_loc_xyz(self):
        return self.get_xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    @property
    def get_loc_feature(self):
        if not self._has_descriptor_overlay():
            return self._loc_feature
        base = self._loc_feature
        n = base.shape[0]
        source_index = getattr(self, "loc_source_index", None)
        if not torch.is_tensor(source_index) or source_index.numel() != n:
            source_index = torch.arange(n, dtype=torch.long, device=base.device)
        else:
            source_index = source_index.to(device=base.device, dtype=torch.long).reshape(-1)
        overlay_source = self.loc_overlay_source_index.to(device=base.device, dtype=torch.long)
        if overlay_source.numel() == 0:
            return base
        pos = torch.searchsorted(overlay_source, source_index)
        pos_clamped = pos.clamp(max=overlay_source.numel() - 1)
        valid = (pos < overlay_source.numel()) & (overlay_source[pos_clamped] == source_index)
        if not bool(valid.any()):
            return base
        overlay = torch.zeros_like(base)
        active = torch.sigmoid(self._loc_overlay_active_logit.to(device=base.device, dtype=base.dtype))
        residual_table = self._loc_overlay_feature.to(device=base.device, dtype=base.dtype) * active
        max_norm = float(getattr(self, "loc_overlay_max_residual_norm", 0.0))
        if max_norm > 0.0 and residual_table.numel() > 0:
            flat = residual_table.reshape(residual_table.shape[0], -1)
            norm = torch.linalg.norm(flat, dim=1, keepdim=True)
            scale = torch.clamp(max_norm / norm.clamp_min(1e-12), max=1.0)
            residual_table = (flat * scale).reshape_as(residual_table)
        overlay[valid] = residual_table[pos[valid]]
        features = base + overlay
        if bool(getattr(self, "loc_overlay_normalize", False)):
            features = F.normalize(features.reshape(n, -1), p=2, dim=1).reshape_as(features)
        return features

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, loc_feature_size : int, speedup: bool):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        
        _loc_feature = torch.randn(fused_point_cloud.shape[0], loc_feature_size, 1).float().cuda() 
        _loc_feature = F.normalize(_loc_feature, p=2, dim=1)

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._loc_feature = nn.Parameter(_loc_feature.transpose(1, 2).contiguous().requires_grad_(True))
        self.init_localization_state(from_rgb_opacity=True)


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self._ensure_localization_state()
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        loc_opacity_lr = getattr(training_args, "loc_opacity_lr", training_args.opacity_lr * 0.1)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._loc_feature], 'lr':training_args.loc_feature_lr, "name": "loc_feature"},
            {'params': [self._loc_opacity], 'lr': loc_opacity_lr, "name": "loc_opacity"},
        ]
        if self._has_descriptor_overlay():
            overlay_lr = getattr(training_args, "loc_overlay_lr", 0.0)
            if float(overlay_lr) <= 0.0:
                overlay_lr = training_args.loc_feature_lr
            l.extend(
                [
                    {'params': [self._loc_overlay_feature], 'lr': overlay_lr, "name": "loc_overlay_feature"},
                    {'params': [self._loc_overlay_active_logit], 'lr': overlay_lr, "name": "loc_overlay_active_logit"},
                ]
            )

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        # Add loc features
        for i in range(self._loc_feature.shape[1]*self._loc_feature.shape[2]):  
            l.append('loc_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        loc_feature = self._loc_feature.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy() 

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, loc_feature), axis=1)
            
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, loc_feature_dim=None):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        count = sum(1 for name in plydata.elements[0].data.dtype.names if name.startswith("loc_"))
        if count > 0:
            loc_feature = np.stack([np.asarray(plydata.elements[0][f"loc_{i}"]) for i in range(count)], axis=1)
            loc_feature = np.expand_dims(loc_feature, axis=-1)
        else:
            feature_dim = int(loc_feature_dim or 256)
            if feature_dim <= 0:
                raise ValueError(f"loc_feature_dim must be positive for RGB-only PLY loading, got {feature_dim}")
            rng = np.random.default_rng(0)
            loc_feature = rng.standard_normal((xyz.shape[0], feature_dim, 1)).astype(np.float32)
            norm = np.linalg.norm(loc_feature, axis=1, keepdims=True)
            loc_feature = loc_feature / np.clip(norm, 1e-12, None)
            print(
                "Loaded RGB-only Gaussian PLY without loc_* fields; "
                f"initialized LaFGS loc_feature with dim={feature_dim}"
            )

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._loc_feature = nn.Parameter(torch.tensor(loc_feature, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree
        self.init_localization_state(from_rgb_opacity=True)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                old_param = group["params"][0]
                stored_state = self.optimizer.state.get(old_param, None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[old_param]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                if stored_state is not None:
                    self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._loc_feature = optimizable_tensors["loc_feature"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_loc_feature):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "loc_feature": new_loc_feature} 

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._loc_feature = optimizable_tensors["loc_feature"] 

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_loc_feature = self._loc_feature[selected_pts_mask].repeat(N,1,1) 

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_loc_feature) 
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_loc_feature = self._loc_feature[selected_pts_mask] 

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_loc_feature) 

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, loc_birth_iteration=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, loc_birth_iteration=loc_birth_iteration)
        self.densify_and_split(grads, max_grad, extent, loc_birth_iteration=loc_birth_iteration)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()


    def add_densification_stats_gsplat(self, viewspace_point_tensor, update_filter, width, height):
        grad = viewspace_point_tensor.grad.squeeze(0) # [N, 2]
        # Normalize the gradient to [-1, 1] screen size
        grad[:, 0] *= width * 0.5
        grad[:, 1] *= height * 0.5
        self.xyz_gradient_accum[update_filter] += torch.norm(grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._loc_feature = optimizable_tensors["loc_feature"]
        self._loc_opacity = optimizable_tensors["loc_opacity"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self._prune_localization_buffers(valid_points_mask)

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_loc_feature,
        new_loc_opacity=None,
        loc_parent_mask=None,
        loc_repeat=1,
        loc_birth_iteration=None,
    ):
        if new_loc_opacity is None:
            new_loc_opacity = new_opacities.detach().clone()
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "loc_feature": new_loc_feature,
            "loc_opacity": new_loc_opacity,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._loc_feature = optimizable_tensors["loc_feature"]
        self._loc_opacity = optimizable_tensors["loc_opacity"]
        self._cat_localization_buffers(
            loc_parent_mask,
            repeat=loc_repeat,
            birth_iteration=loc_birth_iteration,
        )

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split_selected(self, selected_mask, scene_extent, N=2, loc_birth_iteration=None):
        selected_pts_mask = torch.as_tensor(
            selected_mask,
            device=self.get_xyz.device,
            dtype=torch.bool,
        ).reshape(-1)
        if selected_pts_mask.numel() != self.get_xyz.shape[0]:
            raise RuntimeError(
                "Cannot split selected Gaussians: "
                f"mask has {selected_pts_mask.numel()} rows, model has {self.get_xyz.shape[0]} points."
            )
        if selected_pts_mask.sum() == 0:
            return 0
        split_count = int(selected_pts_mask.sum().item())

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=self.get_xyz.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_loc_feature = split_localization_child_features(
            self._loc_feature[selected_pts_mask],
            self.loc_prototype[selected_pts_mask],
            self.loc_prototype_count[selected_pts_mask],
            repeat=N,
        )
        new_loc_opacity = self._loc_opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_loc_feature,
            new_loc_opacity,
            selected_pts_mask,
            N,
            loc_birth_iteration,
        )
        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device=self.get_xyz.device, dtype=bool),
            )
        )
        self.prune_points(prune_filter)
        return split_count

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, loc_birth_iteration=None):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_loc_feature = self._loc_feature[selected_pts_mask].repeat(N, 1, 1)
        new_loc_opacity = self._loc_opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_loc_feature,
            new_loc_opacity,
            selected_pts_mask,
            N,
            loc_birth_iteration,
        )
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, loc_birth_iteration=None):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_loc_feature = self._loc_feature[selected_pts_mask]
        new_loc_opacity = self._loc_opacity[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_loc_feature,
            new_loc_opacity,
            selected_pts_mask,
            1,
            loc_birth_iteration,
        )
