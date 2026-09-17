"""Paired XYZ estimator control, with explicitly sparse center projection.

Center visibility is a one-pixel z-buffer, not Gaussian alpha compositing.
This does NOT remove the Track association used to define the common anchors.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import torch
from plyfile import PlyData
from anygsloc.artifacts import sha256, export_map, validate_identity
from anygsloc.features.raster_sampling import sample_raster_at_grid_uv
from anygsloc.mapping.subset import subset


def projected_centers(xyz, K, pose, width, height, keypoints, radius=2.0):
    camera = xyz @ pose[:3, :3].T + pose[:3, 3]
    keep = np.isfinite(camera).all(1) & (camera[:, 2] > 1e-6)
    ids = np.flatnonzero(keep)
    uvh = camera[ids] @ K.T
    uv = uvh[:, :2] / uvh[:, 2:3] - 0.5
    valid = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    uv, ids = uv[valid], ids[valid]
    result = np.full((len(keypoints), 3), np.nan, dtype=np.float32)
    if not len(ids):
        return result
    pixels = np.floor(uv).astype(np.int64)
    bins = pixels[:, 1] * width + pixels[:, 0]
    order = np.lexsort((ids, camera[ids, 2], bins))
    _, first = np.unique(bins[order], return_index=True)
    winners = order[first]
    distance, nearest = cKDTree(uv[winners]).query(keypoints, distance_upper_bound=radius)
    matched = np.isfinite(distance)
    result[matched] = xyz[ids[winners[nearest[matched]]]]
    return result


def main():
    p = argparse.ArgumentParser()
    for name in ['map-dir', 'cache', 'ply', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(4)
    source = a.map_dir / 'projective_anchor_map.pt'
    state = torch.load(source, map_location='cpu', weights_only=False)
    metric = torch.load(a.map_dir / 'identity_metric.pt', map_location='cpu', weights_only=False)
    validate_identity(metric, state, source)
    if state['provenance'].get('uses_test_queries') is not False:
        raise ValueError('mapping-only input required')
    cache = torch.load(a.cache, map_location='cpu', weights_only=False)['queries']
    vertices = PlyData.read(a.ply)['vertex']
    xyz = np.column_stack([vertices[k] for k in ['x', 'y', 'z']]).astype(np.float32)
    centers, depths = [], []
    for name in state['v6_mapping_query_names']:
        c = cache[name]
        uv = c['native_keypoints'].float()
        K, pose = c['native_K'].float(), c['pose_w2c'].float()
        raster = c['native_depth']
        h, w = raster.shape[-2:]
        z = sample_raster_at_grid_uv(raster, uv).float()
        alpha = sample_raster_at_grid_uv(c['native_alpha'].float(), uv)
        valid = torch.isfinite(z) & (z > 0) & torch.isfinite(alpha) & (alpha >= 0.05)
        pixel = uv + 0.5
        camera = torch.stack(((pixel[:, 0]-K[0, 2])/K[0, 0]*z,
                              (pixel[:, 1]-K[1, 2])/K[1, 1]*z, z), 1)
        depth = (camera-pose[:3, 3]) @ pose[:3, :3]
        center = torch.from_numpy(projected_centers(xyz, K.numpy(), pose.numpy(), w, h, uv.numpy()))
        depth[~valid] = float('nan')
        center[~valid] = float('nan')
        depths.append(depth)
        centers.append(center)
    csr = state['projective_anchor_observations']
    off = csr['observation_offsets'].tolist()
    qs, ks = csr['query_indices'].tolist(), csr['keypoint_indices'].tolist()
    rows, center_xyz, depth_xyz = [], [], []
    for i, (start, end) in enumerate(zip(off[:-1], off[1:])):
        cp = torch.stack([centers[qs[j]][ks[j]] for j in range(start, end)])
        dp = torch.stack([depths[qs[j]][ks[j]] for j in range(start, end)])
        valid = torch.isfinite(cp).all(1) & torch.isfinite(dp).all(1)
        if int(valid.sum()) >= 3:
            rows.append(i)
            center_xyz.append(cp[valid].median(0).values)
            depth_xyz.append(dp[valid].median(0).values)
    if not rows:
        raise ValueError('No three-observation common support')
    base = subset(state, torch.tensor(rows))
    a.output.mkdir(parents=True)
    for kind, replacement in [('ray', None), ('depth', depth_xyz), ('center', center_xyz)]:
        revised = dict(base)
        if replacement is not None:
            revised['anchor_xyz'] = torch.stack(replacement)
        revised['provenance'] = dict(base['provenance'], center_control=dict(
            kind=kind, source_sha256=sha256(source), ply_sha256=sha256(a.ply),
            cache_sha256=sha256(a.cache), fixed_common_tracks=True,
            projection='one_pixel_zbuffer_then_nearest_projected_center_within_2px',
            aggregation='coordinatewise_median_same_valid_observations', uses_test_queries=False))
        folder = a.output / kind
        folder.mkdir()
        path = folder / 'projective_anchor_map.pt'
        torch.save(revised, path)
        identity = folder / 'identity_metric.pt'
        torch.save(dict(metric, landmark_indices=revised['anchor_ids'], map_path=str(path),
                        map_sha256=sha256(path)), identity)
        export_map(path, identity, folder / 'deployment_map.pt')
    (a.output / 'report.json').write_text(json.dumps(dict(source_count=len(state['anchor_ids']),
        common_count=len(rows), retained_source_rows=rows, radius_px=2,
        minimum_common_observations=3, whole_geometry_pipeline_ablation=False), indent=2))


if __name__ == '__main__':
    main()
