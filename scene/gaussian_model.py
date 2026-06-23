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
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

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

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._loc_feature], 'lr':training_args.loc_feature_lr, "name": "loc_feature"},
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

    def load_ply(self, path):
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
        loc_feature = np.stack([np.asarray(plydata.elements[0][f"loc_{i}"]) for i in range(count)], axis=1) 
        loc_feature = np.expand_dims(loc_feature, axis=-1) 

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

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

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

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

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

    def _loc_feature_dim(self):
        if self._loc_feature.numel() == 0:
            return 0
        return self._loc_feature.reshape(self._loc_feature.shape[0], -1).shape[1]

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

    def _cat_localization_buffers(self, parent_mask, repeat=1):
        if parent_mask is None:
            n = self.get_xyz.shape[0]
            self.init_localization_state(from_rgb_opacity=True)
            assert self.get_xyz.shape[0] == n
            return
        for name in self._localization_buffer_names():
            value = getattr(self, name)
            if value.shape[0] != parent_mask.shape[0]:
                raise RuntimeError(
                    f"Cannot extend localization buffer {name}: "
                    f"buffer has {value.shape[0]} rows, parent mask has {parent_mask.shape[0]}."
                )
            extension = value[parent_mask]
            if repeat != 1:
                reps = [repeat] + [1] * (extension.dim() - 1)
                extension = extension.repeat(*reps)
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
            if value is not None:
                old = getattr(self, attr)[full_idx]
                getattr(self, attr)[full_idx] = ema_decay * old + (1.0 - ema_decay) * value

        prototype = episode_stats.get("prototype")
        if prototype is not None and self.loc_prototype.shape[1] > 0:
            prototype = torch.as_tensor(prototype, device=self.get_xyz.device, dtype=torch.float32)
            prototype = prototype.reshape(full_idx.numel(), -1)[:, : self.loc_prototype.shape[1]]
            old = self.loc_prototype[full_idx]
            self.loc_prototype[full_idx] = ema_decay * old + (1.0 - ema_decay) * F.normalize(prototype, p=2, dim=-1)
            self.loc_prototype_count[full_idx] += 1

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
        entropy_score = self._robust_z(self.loc_entropy_ema, eligible).clamp_min(0.0)
        repeatability = self.loc_repeatability_ema.float().clamp(0.0, 1.0)
        split_score = grad_score * entropy_score * repeatability
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
        state = {"version": 1, "loc_opacity": self._loc_opacity.detach()}
        for name in self._localization_buffer_names():
            state[name] = getattr(self, name).detach()
        return state

    def restore_localization_state(self, state):
        if state is None:
            self.init_localization_state(from_rgb_opacity=True)
            return
        device = self.get_xyz.device
        loc_opacity = state.get("loc_opacity", None)
        if loc_opacity is None:
            loc_opacity = self._opacity.detach().clone()
        self._loc_opacity = nn.Parameter(loc_opacity.to(device=device).detach().clone().requires_grad_(True))
        for name in self._localization_buffer_names():
            if name in state:
                setattr(self, name, state[name].to(device=device).detach().clone())
        self._ensure_localization_state()

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

    def load_ply(self, path):
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
        loc_feature = np.stack([np.asarray(plydata.elements[0][f"loc_{i}"]) for i in range(count)], axis=1) 
        loc_feature = np.expand_dims(loc_feature, axis=-1) 

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
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

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

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

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
        self._cat_localization_buffers(loc_parent_mask, repeat=loc_repeat)

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split_selected(self, selected_mask, scene_extent, N=2):
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
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent,
        )
        if selected_pts_mask.sum() == 0:
            return

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
        )
        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device=self.get_xyz.device, dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
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
        )
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
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
        )
