"""Audit a frozen F0 map with spatially held-out mapping-ray geometry."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from common.hashing import sha256_file
from evidence.geometry_crossfit import crossfit_geometry
from features.multiview_fusion import PIXEL_CENTER_OFFSET
from features.raster_sampling import sample_raster_at_grid_uv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--maximum-anchors', type=int, default=2048)
    parser.add_argument('--surface-evidence', action='store_true',
                        help='Report rendered depth conflict without treating it as truth or a gate.')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.maximum_anchors < 1:
        parser.error('maximum-anchors must be positive')
    torch.set_num_threads(4)
    started = time.perf_counter()
    state = torch.load(args.map, map_location='cpu', weights_only=False)
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    provenance = state['provenance']
    if any(provenance.get(k) is not False or cache.get(k) is not False
           for k in ('uses_source_mapping_rgb', 'uses_test_queries')):
        raise ValueError('mapping-only inputs required')
    cache_sha = sha256_file(args.cache)
    if provenance.get('v8_observation_cache_sha256') != cache_sha:
        raise ValueError('map/cache lineage mismatch')
    names = state['v6_mapping_query_names']
    if names != list(cache['queries']):
        raise ValueError('mapping registry mismatch')
    csr = state['projective_anchor_observations']
    if csr.get('keypoint_index_semantics') != 'original_unfiltered_observation_cache_row':
        raise ValueError('audit requires original source row IDs')
    records = [cache['queries'][name] for name in names]
    for record in records:
        if float(record.get('pixel_center_offset', PIXEL_CENTER_OFFSET)) != float(PIXEL_CENTER_OFFSET):
            raise ValueError('pixel convention mismatch')
    K = np.stack([np.asarray(r['native_K']) for r in records])
    T = np.stack([np.asarray(r['pose_w2c']) for r in records])
    points = [np.asarray(r['native_keypoints']) for r in records]
    depth_samples = None
    if args.surface_evidence:
        depth_samples = [
            np.asarray(sample_raster_at_grid_uv(r['native_depth'], r['native_keypoints']))
            if r.get('native_depth') is not None else np.full(len(r['native_keypoints']), np.nan)
            for r in records]
    offsets, query, keypoint = [np.asarray(csr[k]) for k in
                               ('observation_offsets', 'query_indices', 'keypoint_indices')]
    xyz = np.asarray(state['anchor_xyz'])
    count = len(xyz)
    if offsets.shape != (count + 1,) or offsets[0] != 0 or offsets[-1] != len(query) or np.any(np.diff(offsets) <= 0) or query.shape != keypoint.shape:
        raise ValueError('invalid observation CSR')
    selected = np.unique(np.linspace(0, count - 1, min(count, args.maximum_anchors), dtype=int))
    rows = []
    for anchor in selected:
        q, k = query[offsets[anchor]:offsets[anchor+1]], keypoint[offsets[anchor]:offsets[anchor+1]]
        if len(np.unique(q)) != len(q) or np.any(q < 0) or np.any(q >= len(records)):
            raise ValueError('Track camera IDs must be unique and in range')
        if any(j < 0 or j >= len(points[i]) for i, j in zip(q, k)):
            raise ValueError('Track source keypoint outside cache')
        # Bound diagnostic cost without claiming to replay the production IRLS
        # subset. All sampled observations are original, distinct cameras.
        take = np.unique(np.linspace(0, len(q)-1, min(32, len(q)), dtype=int))
        qs, ks = q[take], k[take]
        uv = np.stack([points[i][j] for i, j in zip(qs, ks)]) + PIXEL_CENTER_OFFSET
        depth = None if depth_samples is None else np.array([depth_samples[i][j] for i, j in zip(qs, ks)])
        result = crossfit_geometry(uv, K[qs], T[qs], xyz[anchor], rendered_depth=depth)
        result.update(anchor_row=int(anchor), full_observation_count=len(q),
                      short_track_loo10_budget_zero=bool(len(q) < 10),
                      candidate_kind=state['anchor_candidate_kind'][anchor])
        rows.append(result)
    report = {
        'schema': 'anygsloc_mapping_geometry_crossfit_audit', 'version': 1,
        'uses_source_mapping_rgb': False, 'uses_test_queries': False,
        'map_mutation_count': 0, 'absolute_accuracy_certified': False,
        'input': {'map': str(args.map.resolve()), 'map_sha256': sha256_file(args.map),
                  'cache': str(args.cache.resolve()), 'cache_sha256': cache_sha},
        'implementation_sha256': {
            'audit': sha256_file(Path(__file__)),
            'geometry': sha256_file(Path(__file__).resolve().parents[1]/'evidence/geometry_crossfit.py')},
        'policy': {'split': 'spatial_PCA_half', 'maximum_observations': 32,
                   'maximum_heldout_median_px': 2., 'minimum_group_parallax_p75_deg': 1.,
                   'sampling': 'uniform_anchor_row_not_physical_identity_paired',
                   'fit': 'unweighted_ray_LS_not_production_robust_replay',
                   'surface_evidence': 'diagnostic_only' if args.surface_evidence else 'not_loaded_in_this_audit'},
        'anchor_count': count, 'sampled_anchor_count': len(rows),
        'status_counts': dict(Counter(r['status'] for r in rows)),
        'reason_counts': dict(Counter(r['reason'] for r in rows)),
        'short_track_loo10_budget_zero_count': sum(r['short_track_loo10_budget_zero'] for r in rows),
        'timing_seconds': time.perf_counter()-started, 'rows': rows,
    }
    # JSON has no portable infinity representation. Preserve failed-depth
    # reason codes, replacing nonfinite diagnostics with explicit nulls.
    def clean(value):
        if isinstance(value, float) and not np.isfinite(value): return None
        if isinstance(value, list): return [clean(x) for x in value]
        if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
        return value
    report = clean(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
