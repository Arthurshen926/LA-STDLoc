"""Explicit, separately versioned Cambridge supplementary experiments.

Never changes FROZEN or an existing experiment directory. Run in g4splat.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from anygsloc.artifacts import SCHEMA, sha256, validate
from anygsloc.config import FROZEN

REPO = Path(__file__).resolve().parents[1]
POOL = Path('/mnt/pool/sqy')
SCENES = ['ShopFacade', 'OldHospital', 'KingsCollege', 'GreatCourt', 'StMarysChurch']
BUDGETS = {'A': (.999, 250, 30000), 'B': (.999, 250, 100000),
           'C': (.99999, 250, 100000), 'D': (.99999, 1000, 100000)}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    from anygsloc.evaluation.runner import json_safe
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False))
    tmp.replace(path)


def array(value):
    # Only trusted, locally generated Torch ZIP tensors are accepted here.
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy()
    return value.numpy()


def tensor(value):
    return torch.from_numpy(array(value).copy())


def spatial_rows(xyz, k, voxel_m=.5):
    """Voxel round-robin, exact K, independent of scores and query labels.

    Within a voxel use distance to its geometric center, then source row.
    At a partially filled round use evenly spaced occupied-voxel ranks.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if not 0 < k <= len(xyz) or not np.isfinite(xyz).all() or voxel_m <= 0:
        raise ValueError('invalid spatial selection input')
    cells = np.floor(xyz / voxel_m).astype(np.int64)
    _, group = np.unique(cells, axis=0, return_inverse=True)
    d2 = ((xyz - (cells + .5) * voxel_m) ** 2).sum(1)
    order = np.lexsort((np.arange(len(xyz)), d2, group))
    counts = np.bincount(group)
    starts = np.r_[0, counts.cumsum()[:-1]]
    chosen = []
    left = k
    for level in range(int(counts.max())):
        active = np.flatnonzero(counts > level)
        if len(active) > left:
            active = active[np.floor((np.arange(left) + .5) * len(active) / left).astype(int)]
        chosen.extend(order[starts[active] + level].tolist())
        left -= len(active)
        if left == 0:
            break
    result = np.sort(np.asarray(chosen, dtype=np.int64))
    assert len(result) == k and len(np.unique(result)) == k
    return result


def quality_rows(quality, k):
    quality = np.asarray(quality)
    if not 0 < k <= len(quality) or not np.isfinite(quality).all():
        raise ValueError('invalid quality selection input')
    return np.sort(np.lexsort((np.arange(len(quality)), -quality))[:k])


def save_map(path, xyz, features, rows, provenance):
    rows = np.asarray(rows, dtype=np.int64)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    row_path = path.with_suffix('.source_rows.npy')
    if row_path.exists():
        raise FileExistsError(row_path)
    np.save(row_path, rows)
    state = dict(schema=SCHEMA, anchor_ids=torch.arange(len(rows)),
                 anchor_xyz=torch.as_tensor(np.asarray(xyz)[rows].copy(), dtype=torch.float32),
                 anchor_features=torch.as_tensor(np.asarray(features)[rows].copy(), dtype=torch.float32),
                 provenance={**provenance, 'retained_source_rows_sha256': sha256(row_path), 'uses_test_queries': False})
    validate(state)
    torch.save(state, path)
    return dict(path=str(path), count=len(rows), bytes=path.stat().st_size, sha256=sha256(path))


def scene_data(root, scene):
    return read(Path(root) / 'protocol.json')['scenes'][scene]


def prepare(root):
    from figure_cache_reader import read_cache
    root.mkdir(parents=True, exist_ok=False)
    scenes = {}
    geo_jobs = read(POOL / 'anygsloc_center_control_20260915/gpu0/jobs.json')
    for scene in SCENES:
        command = next(j['command'] for j in geo_jobs if j['id'] == scene + '_build')
        arg = lambda key: command[command.index(key)+1]
        fact = POOL / f'anygsloc_observation_factorial_20260914/maps/{scene}_v1_f1_c1'
        full = fact / 'full/projective_anchor_map.pt'
        compact = POOL / f'anygsloc_release_validation_20260913/maps/Cambridge/{scene}/deployment_map.pt'
        reference = POOL / f'anygsloc_protocol_v2_primary_20260914/runs/Cambridge/{scene}'
        manifest = read(reference / 'manifest.json')
        assert sha256(compact) == manifest['map_sha256']
        meta = read_cache(full)
        cache = Path(arg('--cache'))
        expected_cache_hash = meta['provenance']['v8_observation_cache_sha256']
        actual_cache_hash = sha256(cache)
        assert actual_cache_hash == expected_cache_hash
        count = read(fact / 'report.json')['compact_count']
        scenes[scene] = dict(dataset=str(POOL / 'Cambridge_stdloc' / scene),
            full_map=str(full), full_map_sha256=sha256(full), compact_map=str(compact),
            compact_map_sha256=manifest['map_sha256'], K=count,
            selector_rows=str(fact / 'report.json'), compact_research=arg('--map-dir'),
            cache=str(cache), cache_sha256=actual_cache_hash,
            ply=arg('--ply'), ply_sha256=sha256(arg('--ply')), reference=str(reference))
        print('Frozen inputs', scene, 'K', count, flush=True)
    write(root / 'protocol.json', dict(version=1, date='2026-10-09', scenes=scenes,
        seed=2026, scene_order=SCENES, native_config=asdict(FROZEN), ransac_budgets=BUDGETS,
        selector=dict(quality='anchor_matchability descending; source row tie break',
                      spatial='0.5 m world voxels; round-robin; geometric-center ordering; evenly spaced last round',
                      exact_K=True, row_order='sorted source rows', random='not in minimum campaign'),
        classic=dict(rgb_only=True, pair_neighbors=6, min_baseline_m=.03, max_baseline_m=5.,
            max_axis_angle_deg=75., mutual_top1=True, cosine_min=.65, cosine_margin=.01,
            epipolar_px=2., cycle_filter=False, conflict_policy='reject conflicting union, strongest edges first',
            minimum_views=3, minimum_parallax_deg=1., maximum_median_reprojection_px=2.,
            maximum_observation_reprojection_px=4., triangulation='normalized DLT + fixed-camera Huber point refinement',
            fusion='L2 normalize each observation, arithmetic mean, L2 normalize',
            capacity='same scene K; if insufficient use common reachable K and report shortfall',
            comparison='classic spatial vs ours spatial at same K; ours full selector as system reference'),
        geometry=dict(common_observations='finite depth and projected center, at least three views',
            base_weights='equal across common observations; archived CSR does not retain edge confidence',
            ray='perpendicular-ray LS + 3 Huber reprojection reweightings, delta 2 px; no observation cap',
            validity='all three coordinates finite; ray positive depth in >=3 common views; shared rows in all arms',
            descriptors='same source descriptor in every coordinate arm; never rerun selector'),
        reporting=dict(all_registered_conditions=True, test_tuned_development=True,
            primary='scene equal TE/RE/R5', extra='R15/P90/failures/map bytes/iterations/solver time',
            timing='accuracy telemetry is not an isolated benchmark; capture contention',
            no_existing_artifact_overwrites=True)))


def select(root, scene):
    from figure_cache_reader import read_cache
    s = scene_data(root, scene)
    output = root / 'selector' / scene
    output.mkdir(parents=True, exist_ok=False)
    state = read_cache(s['full_map'])
    xyz, features, quality = (array(state[k]) for k in ['anchor_xyz', 'anchor_features', 'anchor_matchability'])
    original_rows = np.asarray(read(s['selector_rows'])['retained_source_rows'])
    archived = torch.load(s['compact_map'], map_location='cpu', weights_only=True)
    assert np.array_equal(xyz[original_rows], archived['anchor_xyz'].numpy())
    assert np.array_equal(features[original_rows], archived['anchor_features'].numpy())
    assert len(original_rows) == s['K']
    maps = {}
    for name, fn in [('quality', lambda: quality_rows(quality, s['K'])),
                     ('spatial', lambda: spatial_rows(xyz, s['K']))]:
        begin = time.perf_counter()
        rows = fn()
        seconds = time.perf_counter()-begin
        maps[name] = save_map(output / name / 'deployment_map.pt', xyz, features, rows,
            dict(experiment='selector_equal_K', strategy=name, source_sha256=s['full_map_sha256']))
        maps[name]['selection_seconds'] = seconds
    write(output / 'summary.json', dict(scene=scene, N=len(xyz), K=s['K'], maps=maps,
          archived_full_selector_numerically_reproduced=True))


def capture(root, scene):
    from anygsloc.data.datasets import ColmapDataset
    from anygsloc.data.geometric_validity import SameKRectificationSupport
    from anygsloc.data.calibration import resolve_scene_calibration
    from anygsloc.evaluation.metrics import pose_error
    from anygsloc.evaluation.runner import summarize
    import anygsloc.online.runtime as runtime
    s = scene_data(root, scene)
    output = root / 'ransac' / scene / 'capture'
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    dataset = ColmapDataset(s['dataset'], load_masks=False)
    support = SameKRectificationSupport(dataset)
    calibration = resolve_scene_calibration(dataset.split('mapping'))
    engine = runtime.Localizer(s['compact_map'], keypoint_count=calibration['keypoint_count'],
        reprojection_error_px=calibration['reprojection_error_px'], mask_policy='geometric_validity', seed=2026)
    reference = read(Path(s['reference']) / 'results.json')
    saved, rows = [], []
    real_solve = runtime.solve_absolute_pose
    def recording(p2, p3, K, **kwargs):
        begin = time.perf_counter()
        result = real_solve(p2, p3, K, **kwargs)
        saved.append(dict(points2=p2.copy(), points3=p3.copy(), K=K.copy(),
                          options=kwargs, diagnostics=result.diagnostics, solver_ms=(time.perf_counter()-begin)*1000))
        return result
    runtime.solve_absolute_pose = recording
    start = time.perf_counter()
    try:
        for i, camera in enumerate(dataset.split('test')):
            result = engine.localize(dataset.load_image(camera), fov_x=camera.fov_x,
                                    fov_y=camera.fov_y, valid_mask=support(camera))
            re, te = pose_error(result['pose'].pose_w2c, camera.pose_w2c) if result['success'] else (None, None)
            saved[-1].update(image_name=camera.image_name, gt_pose=camera.pose_w2c,
                             capture_pose=result['pose'].pose_w2c)
            old = reference[i]
            assert camera.image_name == old['image_name'] and result['success'] == old['success']
            if result['success']:
                if abs(te-old['translation_error_cm']) > .001 or abs(re-old['rotation_error_deg']) > .001:
                    raise ValueError(f'Frozen A parity failed at {camera.image_name}: {te}, {re}')
            rows.append(dict(image_name=camera.image_name, success=result['success'],
                rotation_error_deg=re, translation_error_cm=te, runtime_ms=result['runtime_ms'],
                inlier_count=len(result['pose'].inliers), pose_w2c=result['pose'].pose_w2c.tolist()))
            if i % 25 == 0:
                write(output / 'progress.json', dict(completed=i+1, total=len(reference)))
                print('capture', scene, i+1, '/', len(reference), flush=True)
    finally:
        runtime.solve_absolute_pose = real_solve
    torch.save(saved, output / 'correspondences.pt')
    write(output / 'results.json', rows)
    write(output / 'summary.json', dict(**summarize(rows, time.perf_counter()-start),
        frozen_A_parity=True, correspondences_sha256=sha256(output / 'correspondences.pt'),
        reference=s['reference'], map_sha256=s['compact_map_sha256']))


def replay(root, scene):
    from anygsloc.online.pose import solve_absolute_pose
    from anygsloc.evaluation.metrics import pose_error
    from anygsloc.evaluation.runner import summarize
    source = root / 'ransac' / scene / 'capture'
    assert read(source / 'summary.json')['frozen_A_parity']
    data = torch.load(source / 'correspondences.pt', map_location='cpu', weights_only=False)
    output = root / 'ransac' / scene / 'replay'
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    rows = {name: [] for name in BUDGETS}
    for i, record in enumerate(data):
        # Round-robin order reduces systematic warm-cache/time-order bias.
        names = list(BUDGETS)
        names = names[i % 4:] + names[:i % 4]
        for name in names:
            probability, minimum, maximum = BUDGETS[name]
            options = dict(record['options'], confidence=probability, min_iterations=minimum, max_iterations=maximum)
            begin = time.perf_counter()
            result = solve_absolute_pose(record['points2'], record['points3'], record['K'], **options)
            ms = (time.perf_counter()-begin)*1000
            success = bool(len(result.inliers))
            re, te = pose_error(result.pose_w2c, record['gt_pose']) if success else (None, None)
            if name == 'A':
                assert np.array_equal(result.pose_w2c, record['capture_pose']), record['image_name']
            info = {k: v for k, v in result.diagnostics.items() if k != 'inliers'}
            rows[name].append(dict(image_name=record['image_name'], success=success,
                translation_error_cm=te, rotation_error_deg=re, runtime_ms=ms,
                inlier_count=len(result.inliers), correspondence_count=len(record['points2']),
                pose_w2c=result.pose_w2c.tolist(), diagnostics=info,
                hit_maximum=int(info.get('iterations', 0)) >= maximum))
        if i % 100 == 0:
            print('replay', scene, i+1, '/', len(data), flush=True)
    for name, records in rows.items():
        summary = summarize(records, sum(r['runtime_ms'] for r in records)/1000)
        summary.update(timing_protocol='solver_only_replay_telemetry_not_end_to_end',
            budget=BUDGETS[name], actual_iterations_mean=float(np.mean([r['diagnostics'].get('iterations', 0) for r in records])),
            hit_maximum_percent=100*sum(r['hit_maximum'] for r in records)/len(records),
            correspondence_sha256=read(source/'summary.json')['correspondences_sha256'])
        write(output / name / 'results.json', records)
        write(output / name / 'summary.json', summary)
    write(output / 'summary.json', dict(complete=True, queries=len(data), budgets=list(BUDGETS), A_exact_capture_parity=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'select', 'capture', 'replay'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--scene', choices=SCENES)
    a = p.parse_args()
    if a.action == 'prepare':
        prepare(a.root)
    else:
        if a.scene is None:
            p.error('--scene is required')
        {'select': select, 'capture': capture, 'replay': replay}[a.action](a.root, a.scene)


if __name__ == '__main__':
    main()
