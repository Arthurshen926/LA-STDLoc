"""Rendered RGB -> ordinary feature tracks -> fixed-camera triangulation.

No alpha/depth/semantic evidence enters feature extraction, matching, fitting,
fusion, or selection. Native auxiliary buffers are inspected only for renderer
parity diagnostics. Mapping RGB is never loaded.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from figure_cache_reader import read_cache
from supplement_experiments import array, tensor, read, write, scene_data, spatial_rows, save_map
from anygsloc.artifacts import sha256


@torch.inference_mode()
def render(root, scene):
    from anygsloc.priors.models import GaussianModel2D
    from anygsloc.priors.rendering import render_from_pose_gsplat
    from anygsloc.features.extractor import FeatureExtractor
    from anygsloc.features.superpoint import render_validity_mask_from_alpha
    s = scene_data(root, scene)
    out = root / 'classic' / scene / 'render'
    out.mkdir(parents=True, exist_ok=False)
    original = read_cache(s['cache'])
    assert original['uses_test_queries'] is False and original['uses_source_mapping_rgb'] is False
    cfg = original['configuration']
    assert cfg['gaussian_type'] == '2dgs' and cfg['background'] == 'black'
    assert cfg['keypoints'] == 2048 and cfg['nms_radius'] == 4
    assert cfg.get('mapping_render_sampling', {}).get('supersampling_factor', 1) == 1
    torch.set_num_threads(4)
    model = GaussianModel2D(3)
    model.load_ply(s['ply'], loc_feature_dim=0)
    model = model.cuda().eval()
    extractor = FeatureExtractor('sp', nms_radius=4).cuda().eval()
    records = {}
    parity = []
    start = time.perf_counter()
    for i, (name, old) in enumerate(original['queries'].items()):
        K, pose = tensor(old['native_K']).float(), tensor(old['pose_w2c']).float()
        h, w = array(old['native_input_hw']).astype(int).tolist()
        fx, fy = 2*math.atan(w/(2*float(K[0, 0]))), 2*math.atan(h/(2*float(K[1, 1])))
        package = render_from_pose_gsplat(model, pose.cuda(), fx, fy, w, h,
            bg_color=torch.zeros(3, device='cuda'), render_mode='RGB+ED', rgb_only=True,
            rasterize_mode='antialiased')
        rgb = package['render'].float()[:3].clamp(0, 1)
        # Construction reads RGB only. No mask is passed to SuperPoint.
        sparse = extractor.detectAndCompute(rgb[None], top_k=2048, detection_threshold=0.)[0]
        records[name] = dict(native_keypoints=sparse['keypoints'].cpu().float(),
            native_descriptors=F.normalize(sparse['descriptors'].cpu().float(), dim=1),
            native_scores=sparse['keypoint_scores'].cpu().float(), native_K=K,
            pose_w2c=pose, native_input_hw=torch.tensor([h, w]))
        if i in {0, len(original['queries'])//2, len(original['queries'])-1}:
            # Audit separately against the old gated detector; its outputs never enter the baseline.
            alpha = package.get('alphas', package.get('rend_alpha'))
            policy = cfg['render_valid_observation_extraction']
            valid = render_validity_mask_from_alpha(alpha, minimum_alpha=policy['minimum_alpha'],
                                                    neighborhood_radius=policy['neighborhood_radius'])
            gated, _ = extractor.detectAndComputeWithValidityAudit(rgb[None], top_k=2048,
                                    detection_threshold=0., validity_mask=valid)
            old_xy = array(old['native_keypoints'])
            new_xy = gated[0]['keypoints'].cpu().numpy()
            old_desc = array(old['native_descriptors'])
            new_desc = F.normalize(gated[0]['descriptors'].cpu().float(), dim=1).numpy()
            lookup = {tuple(x): j for j, x in enumerate(new_xy.tolist())}
            pairs = [(j, lookup[tuple(x)]) for j, x in enumerate(old_xy.tolist()) if tuple(x) in lookup]
            shared = len(pairs)/max(1, len(old_xy))
            cosines = [float(old_desc[a] @ new_desc[b]) for a, b in pairs]
            check = dict(name=name, shared_keypoint_fraction=shared,
                         mean_descriptor_cosine=float(np.mean(cosines)) if cosines else 0.)
            parity.append(check)
            write(out / 'renderer_parity.json', parity)
            if shared < .99 or check['mean_descriptor_cosine'] < .999:
                raise ValueError(f'RGB renderer/feature parity failed; do not promote baseline: {check}')
        if i % 25 == 0:
            print('classic RGB', scene, i+1, '/', len(original['queries']), flush=True)
            write(out / 'progress.json', dict(completed=i+1, total=len(original['queries'])))
    torch.save(dict(queries=records, uses_test_queries=False, uses_source_mapping_rgb=False,
                    construction_modalities=['RGB', 'known_mapping_cameras'],
                    no_alpha_depth_semantic_filter=True), out / 'rgb_features.pt')
    write(out / 'summary.json', dict(complete=True, views=len(records), ply_sha256=s['ply_sha256'],
        reference_cache_sha256=s['cache_sha256'], feature_cache_sha256=sha256(out/'rgb_features.pt'),
        renderer_parity=parity, wall_seconds=time.perf_counter()-start))


def assemble_tracks(pairs, lengths, minimum_views=3):
    """Conventional confidence-sorted union; reject multi-observation camera conflicts."""
    offsets = np.r_[0, np.cumsum(lengths)]
    camera_of = np.repeat(np.arange(len(lengths)), lengths)
    parent = np.arange(offsets[-1], dtype=np.int64)
    sizes = np.ones(len(parent), dtype=np.int32)
    views = {}
    active = set()
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i
    edges = []
    for q1, q2, a, b, score in pairs:
        edges.extend(zip((offsets[q1]+a).tolist(), (offsets[q2]+b).tolist(), score.tolist()))
    edges.sort(key=lambda e: (-e[2], e[0], e[1]))
    conflicts = 0
    for a, b, _ in edges:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        va, vb = views.get(ra, {int(camera_of[ra])}), views.get(rb, {int(camera_of[rb])})
        if va & vb:
            conflicts += 1
            continue
        if sizes[ra] < sizes[rb]:
            ra, rb, va, vb = rb, ra, vb, va
        parent[rb] = ra
        sizes[ra] += sizes[rb]
        views[ra] = va | vb
        views.pop(rb, None)
        active.update((a, b))
    groups = {}
    for node in sorted(active):
        groups.setdefault(find(node), []).append(node)
    tracks = []
    for nodes in groups.values():
        if len(nodes) >= minimum_views:
            qq = camera_of[nodes]
            assert len(np.unique(qq)) == len(nodes)
            tracks.append((qq.astype(np.int64), np.asarray(nodes)-offsets[qq]))
    return tracks, dict(edges=len(edges), conflicting_unions_rejected=conflicts, tracks=len(tracks))


@torch.inference_mode()
def match(root, scene):
    from anygsloc.mapping.camera_pair_policy import candidate_camera_pairs
    from anygsloc.mapping.triangulation import reciprocal_epipolar_matches
    out = root / 'classic' / scene / 'match'
    out.mkdir(parents=True, exist_ok=False)
    data = torch.load(root/'classic'/scene/'render/rgb_features.pt', map_location='cpu')
    assert data['no_alpha_depth_semantic_filter']
    records = list(data['queries'].values())
    poses = torch.stack([r['pose_w2c'] for r in records])
    K = torch.stack([r['native_K'] for r in records])
    config = read(root/'protocol.json')['classic']
    pairs = candidate_camera_pairs(poses, neighbors=config['pair_neighbors'],
        minimum_baseline_m=config['min_baseline_m'], maximum_baseline_m=config['max_baseline_m'],
        maximum_axis_angle_deg=config['max_axis_angle_deg'], policy='nearest')
    descriptors = [r['native_descriptors'].cuda() for r in records]
    matches = []
    start = time.perf_counter()
    for i, (a, b) in enumerate(pairs):
        source, target, confidence = reciprocal_epipolar_matches(descriptors[a], descriptors[b],
            records[a]['native_keypoints']+.5, records[b]['native_keypoints']+.5,
            K[a], poses[a], K[b], poses[b], minimum_similarity=config['cosine_min'],
            minimum_margin=config['cosine_margin'], maximum_epipolar_error_px=config['epipolar_px'],
            epipolar_candidate_topk=1)
        matches.append((a, b, source.cpu().numpy(), target.cpu().numpy(), confidence.cpu().numpy()))
        if i % 250 == 0:
            print('classic matches', scene, i+1, '/', len(pairs), flush=True)
    del descriptors
    torch.cuda.empty_cache()
    tracks, stats = assemble_tracks(matches, [len(r['native_keypoints']) for r in records])
    torch.save(dict(tracks=tracks, camera_names=list(data['queries']), stats=stats), out/'tracks.pt')
    write(out/'summary.json', dict(complete=True, pairs=len(pairs), **stats, wall_seconds=time.perf_counter()-start,
        features_sha256=sha256(root/'classic'/scene/'render/rgb_features.pt'), tracks_sha256=sha256(out/'tracks.pt')))


def fit_point(uv, K, poses):
    """Normalized DLT, fixed-camera point-only Huber refinement, standard pruning."""
    from scipy.optimize import least_squares
    uv, K, poses = (np.asarray(x, dtype=np.float64) for x in (uv, K, poses))
    normalized = np.linalg.solve(K, np.c_[uv, np.ones(len(uv))][..., None])[..., 0]
    normalized /= normalized[:, 2:3]
    projection = poses[:, :3]
    A = np.stack((normalized[:, 0, None]*projection[:, 2]-projection[:, 0],
                  normalized[:, 1, None]*projection[:, 2]-projection[:, 1]), axis=1).reshape(-1, 4)
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    if abs(vt[-1, 3]) < 1e-12:
        return None
    point = vt[-1, :3]/vt[-1, 3]
    def refine(point, ids):
        P = np.einsum('nij,njk->nik', K[ids], projection[ids])
        def fun(x):
            h = P[:, :, :3] @ x + P[:, :, 3]
            z = np.where(np.abs(h[:, 2]) < 1e-12, 1e-12, h[:, 2])
            return (h[:, :2]/z[:, None]-uv[ids]).ravel()
        def jac(x):
            h = P[:, :, :3] @ x + P[:, :, 3]
            z = np.where(np.abs(h[:, 2]) < 1e-12, 1e-12, h[:, 2])
            return ((P[:, :2, :3]*z[:, None, None]-h[:, :2, None]*P[:, 2:3, :3])/
                    z[:, None, None]**2).reshape(-1, 3)
        return least_squares(fun, point, jac=jac, loss='huber', f_scale=2., max_nfev=30).x
    def errors(x):
        camera = poses[:, :3, :3] @ x + poses[:, :3, 3]
        h = np.einsum('nij,nj->ni', K, camera)
        z = np.where(np.abs(h[:, 2]) < 1e-12, 1e-12, h[:, 2])
        return np.linalg.norm(h[:, :2]/z[:, None]-uv, axis=1), camera[:, 2]
    point = refine(point, np.arange(len(uv)))
    error, depth = errors(point)
    keep = np.flatnonzero(np.isfinite(error) & (error <= 4.) & (depth > 0))
    if len(keep) < 3:
        return None
    point = refine(point, keep)
    error, depth = errors(point)
    # Keep a consistent final observation set for both coordinates and descriptor averaging.
    revised = keep[(error[keep] <= 4.) & (depth[keep] > 0) & np.isfinite(error[keep])]
    if len(revised) != len(keep):
        if len(revised) < 3:
            return None
        keep = revised
        point = refine(point, keep)
        error, depth = errors(point)
    if (not np.isfinite(point).all() or np.median(error[keep]) > 2. or
            np.any(error[keep] > 4.) or np.any(depth[keep] <= 0)):
        return None
    centers = -np.einsum('nji,nj->ni', poses[keep, :3, :3], poses[keep, :3, 3])
    rays = point-centers
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    angles = np.degrees(np.arccos(np.clip(rays @ rays.T, -1, 1)))[np.triu_indices(len(keep), 1)]
    if np.quantile(angles, .75) < 1.:
        return None
    return point.astype(np.float32), keep, float(np.median(error[keep]))


_FIT = None


def _fit_track(index):
    tracks, records = _FIT
    q, k = tracks[index]
    uv = np.stack([records[qi]['native_keypoints'][ki].numpy()+.5 for qi, ki in zip(q, k)])
    K = np.stack([records[qi]['native_K'].numpy() for qi in q])
    poses = np.stack([records[qi]['pose_w2c'].numpy() for qi in q])
    try:
        result = fit_point(uv, K, poses)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return None
    if result is None:
        return None
    xyz, keep, error = result
    descriptors = np.stack([records[q[j]]['native_descriptors'][k[j]].numpy() for j in keep])
    descriptors /= np.maximum(np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-12)
    feature = descriptors.mean(0)
    norm = np.linalg.norm(feature)
    if norm < 1e-12:
        return None
    return index, xyz, (feature/norm).astype(np.float32), q[keep], k[keep], error


def fit(root, scene, workers=4):
    global _FIT
    torch.set_num_threads(1)
    s = scene_data(root, scene)
    out = root / 'classic' / scene / 'fit'
    out.mkdir(parents=True, exist_ok=False)
    data = torch.load(root/'classic'/scene/'render/rgb_features.pt', map_location='cpu')
    graph = torch.load(root/'classic'/scene/'match/tracks.pt', map_location='cpu')
    assert list(data['queries']) == graph['camera_names']
    _FIT = graph['tracks'], list(data['queries'].values())
    start = time.perf_counter()
    accepted = []
    # This is a fresh CPU-only process; it has not initialized CUDA.
    assert not torch.cuda.is_initialized()
    with mp.get_context('fork').Pool(workers) as pool:
        for i, result in enumerate(pool.imap(_fit_track, range(len(graph['tracks'])), chunksize=64)):
            if result is not None:
                accepted.append(result)
            if i % 2000 == 0:
                write(out/'progress.json', dict(processed=i+1, tracks=len(graph['tracks']), accepted=len(accepted)))
                print('classic triangulation', scene, i+1, '/', len(graph['tracks']), 'accepted', len(accepted), flush=True)
    if not accepted:
        raise ValueError('No valid classical points; retain failure and diagnose')
    xyz, features = np.stack([r[1] for r in accepted]), np.stack([r[2] for r in accepted])
    actual_k = min(s['K'], len(xyz))
    provenance = dict(experiment='rendered_RGB_known_pose_classical', uses_depth=False,
                      uses_alpha=False, uses_semantics=False, fusion='normalized_arithmetic_mean',
                      features_sha256=sha256(root/'classic'/scene/'render/rgb_features.pt'))
    full = save_map(out/'full/deployment_map.pt', xyz, features, np.arange(len(xyz)), provenance)
    rows = spatial_rows(xyz, actual_k)
    compact = save_map(out/'spatial/deployment_map.pt', xyz, features, rows, dict(provenance, strategy='spatial'))
    # Both construction methods use the same generic selector and attainable capacity.
    ours = read_cache(s['full_map'])
    ours_xyz, ours_features = array(ours['anchor_xyz']), array(ours['anchor_features'])
    ours_rows = spatial_rows(ours_xyz, actual_k)
    ours_map = save_map(out/'ours_spatial/deployment_map.pt', ours_xyz, ours_features, ours_rows,
                        dict(experiment='construction_control_ours', source_sha256=s['full_map_sha256'], strategy='spatial'))
    offsets = np.r_[0, np.cumsum([len(r[3]) for r in accepted])]
    np.savez(out/'tracks.npz', source_track=[r[0] for r in accepted], offsets=offsets,
             query_indices=np.concatenate([r[3] for r in accepted]), keypoint_indices=np.concatenate([r[4] for r in accepted]),
             median_reprojection_px=[r[5] for r in accepted])
    write(out/'summary.json', dict(complete=True, scene=scene, tracks=len(graph['tracks']),
        valid_points=len(xyz), requested_K=s['K'], actual_K=actual_k, budget_shortfall=s['K']-actual_k,
        full_map=full, classic_map=compact, ours_map=ours_map, wall_seconds=time.perf_counter()-start,
        cameras_fixed=True, ordinary_mean_descriptors=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['render', 'match', 'fit'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--scene', required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if a.action == 'fit':
        fit(a.root, a.scene, a.workers)
    else:
        {'render': render, 'match': match}[a.action](a.root, a.scene)
