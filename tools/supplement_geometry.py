"""Geometry controls with exactly shared per-anchor observation indices."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch
from plyfile import PlyData

from figure_cache_reader import read_cache
from gaussian_center_control import projected_centers
from supplement_experiments import array, tensor, scene_data, save_map, write
from anygsloc.features.raster_sampling import sample_raster_at_grid_uv
from anygsloc.mapping.triangulation import _camera_rays, _weighted_ray_intersection, _project
from anygsloc.artifacts import sha256


def refit_ray(uv, K, poses):
    """Equal base weights; robust perpendicular-ray estimator on ALL input rows."""
    uv, K, poses = (torch.as_tensor(x, dtype=torch.float64) for x in (uv, K, poses))
    centers, directions = _camera_rays(uv, K, poses)
    point, _, _ = _weighted_ray_intersection(centers, directions, torch.ones(len(uv), dtype=torch.float64))
    for _ in range(3):
        projected, z = _project(point, K, poses)
        residual = torch.linalg.norm(projected-uv, dim=1)
        weight = torch.minimum(torch.ones_like(residual), 2./residual.clamp_min(1e-8)) * (z > 0)
        point, _, _ = _weighted_ray_intersection(centers, directions, weight)
    _, z = _project(point, K, poses)
    if not torch.isfinite(point).all() or int((z > 0).sum()) < 3:
        raise ValueError('unusable common ray geometry')
    return point.numpy().astype(np.float32)


def build(root, scene):
    torch.set_num_threads(1)
    s = scene_data(root, scene)
    out = root / 'geometry' / scene
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    src = Path(s['compact_research']) / 'projective_anchor_map.pt'
    state = read_cache(src)
    cache = read_cache(s['cache'])['queries']
    vertices = PlyData.read(s['ply'])['vertex']
    xyz = np.column_stack([vertices[k] for k in ['x', 'y', 'z']]).astype(np.float32)
    centers, depths, uvs, intrinsics, poses = [], [], [], [], []
    names = state['v6_mapping_query_names']
    for index, name in enumerate(names):
        c = cache[name]
        uv = tensor(c['native_keypoints']).float()
        K, pose = tensor(c['native_K']).float(), tensor(c['pose_w2c']).float()
        raster = tensor(c['native_depth'])
        h, w = raster.shape[-2:]
        z = sample_raster_at_grid_uv(raster, uv).float()
        alpha = sample_raster_at_grid_uv(tensor(c['native_alpha']).float(), uv)
        valid = torch.isfinite(z) & (z > 0) & torch.isfinite(alpha) & (alpha >= .05)
        pixel = uv+.5
        camera = torch.stack(((pixel[:, 0]-K[0, 2])/K[0, 0]*z,
                              (pixel[:, 1]-K[1, 2])/K[1, 1]*z, z), 1)
        dp = ((camera-pose[:3, 3]) @ pose[:3, :3]).numpy()
        cp = projected_centers(xyz, K.numpy(), pose.numpy(), w, h, uv.numpy())
        dp[~valid.numpy()] = np.nan
        cp[~valid.numpy()] = np.nan
        depths.append(dp); centers.append(cp); uvs.append(pixel.numpy())
        intrinsics.append(K.numpy()); poses.append(pose.numpy())
        if index % 100 == 0:
            print('geometry views', scene, index+1, '/', len(names), flush=True)
    intrinsics, poses = np.stack(intrinsics), np.stack(poses)
    csr = state['projective_anchor_observations']
    offsets, qs, ks = (array(csr[k]).astype(np.int64) for k in ['observation_offsets', 'query_indices', 'keypoint_indices'])
    retained, cp_xyz, dp_xyz, ray_xyz, common_q, common_k, common_offsets = [], [], [], [], [], [], [0]
    ray_rejected = 0
    common_eligible = 0
    for i, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
        cp = np.stack([centers[q][k] for q, k in zip(qs[a:b], ks[a:b])])
        dp = np.stack([depths[q][k] for q, k in zip(qs[a:b], ks[a:b])])
        valid = np.isfinite(cp).all(1) & np.isfinite(dp).all(1)
        if valid.sum() < 3:
            continue
        common_eligible += 1
        q, k = qs[a:b][valid], ks[a:b][valid]
        assert len(np.unique(q)) == len(q), 'duplicate camera in frozen anchor'
        uv = np.stack([uvs[qi][ki] for qi, ki in zip(q, k)])
        try:
            ray = refit_ray(uv, intrinsics[q], poses[q])
        except (ValueError, RuntimeError):
            ray_rejected += 1
            continue
        # Match the previous PyTorch lower-median definition for even counts.
        cp_xyz.append(torch.from_numpy(cp[valid]).median(0).values.numpy())
        dp_xyz.append(torch.from_numpy(dp[valid]).median(0).values.numpy())
        ray_xyz.append(ray)
        retained.append(i)
        common_q.extend(q.tolist()); common_k.extend(k.tolist()); common_offsets.append(len(common_q))
        if i % 10000 == 0:
            print('geometry refit', scene, i, '/', len(offsets)-1, flush=True)
    if not retained:
        raise ValueError('empty shared geometry set')
    np.savez(out / 'common_observations.npz', source_rows=retained, offsets=common_offsets,
             query_indices=common_q, keypoint_indices=common_k)
    features = array(state['anchor_features'])[retained]
    provenance = dict(experiment='strict_common_observation_geometry', source_map_sha256=sha256(src),
        common_observations_sha256=sha256(out/'common_observations.npz'),
        ray_base_weights='uniform', observation_cap=None)
    maps = {}
    for kind, values in [('ray', ray_xyz), ('depth', dp_xyz), ('center', cp_xyz)]:
        maps[kind] = save_map(out / kind / 'deployment_map.pt', np.stack(values), features,
                             np.arange(len(retained)), dict(provenance, geometry=kind))
    write(out / 'summary.json', dict(scene=scene, source_count=len(offsets)-1,
        common_eligible=common_eligible, ray_rejected_shared_across_arms=ray_rejected,
        common_count=len(retained), common_observations=len(common_q), maps=maps,
        wall_seconds=time.perf_counter()-start, uses_test_queries=False,
        caveat='Equal base weights control; not a replay of unavailable historical edge weights.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--scene', required=True)
    a = p.parse_args()
    build(a.root, a.scene)
