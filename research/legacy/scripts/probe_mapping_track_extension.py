"""Geometry-only local extension pilot using frozen, V2-valid mapping rows.

Original descriptors, Anchor IDs and descriptor observation CSR stay fixed.
Additional GEOMETRY observations are stored separately and explicitly. This
experimental output is not a promoted deployment map.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from common.hashing import sha256_file
from evidence.mapping_track_extension import extend_geometry
from features.multiview_fusion import PIXEL_CENTER_OFFSET
from map_learning.v24_anchor_view_support import build_anchor_view_support
from topology.v6_anchor_map import identity_metric_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    torch.set_num_threads(4)
    state = torch.load(args.map, weights_only=False, map_location='cpu')
    cache = torch.load(args.cache, weights_only=False, map_location='cpu')
    audit = json.loads(args.geometry.read_text())
    map_sha, cache_sha = sha256_file(args.map), sha256_file(args.cache)
    p = state['provenance']
    if audit['input']['map_sha256'] != map_sha or p['v8_observation_cache_sha256'] != cache_sha or audit['input']['cache_sha256'] != cache_sha:
        raise ValueError('source identities differ')
    for payload in (p, cache, audit):
        if payload.get('uses_source_mapping_rgb') is not False or payload.get('uses_test_queries') is not False:
            raise ValueError('all construction inputs must be mapping-only')
    names = state['v6_mapping_query_names']
    if names != list(cache['queries']): raise ValueError('camera registry differs')
    validity = {}
    for filename, digest in zip(p['v8_v2_audit_shards'], p['v8_v2_audit_shard_sha256']):
        if sha256_file(Path(filename)) != digest: raise ValueError('V2 audit changed')
        shard = torch.load(filename, weights_only=False, map_location='cpu')
        if shard['input']['observation_cache_sha256'] != cache_sha: raise ValueError('V2/cache differs')
        for row in shard['records']:
            i = row['query_index']
            if i in validity or names[i] != row['query_name']: raise ValueError('V2 registry differs')
            validity[i] = np.asarray(row['row_valid'], dtype=bool)
    if sorted(validity) != list(range(len(names))): raise ValueError('incomplete V2 rows')
    records = [cache['queries'][n] for n in names]
    for r in records:
        if float(r.get('pixel_center_offset', PIXEL_CENTER_OFFSET)) != PIXEL_CENTER_OFFSET:
            raise ValueError('pixel convention differs')
    K = np.stack([np.asarray(r['native_K'], dtype=float) for r in records])
    T = np.stack([np.asarray(r['pose_w2c'], dtype=float) for r in records])
    centers = -np.einsum('nji,nj->ni', T[:, :3, :3], T[:, :3, 3])
    uv = [np.asarray(r['native_keypoints'], dtype=float) + PIXEL_CENTER_OFFSET for r in records]
    descriptors = [np.asarray(r['native_descriptors']) for r in records]
    features = np.asarray(state['anchor_features'])
    csr = state['projective_anchor_observations']
    if csr.get('keypoint_index_semantics') != 'original_unfiltered_observation_cache_row': raise ValueError('wrong row semantics')
    offsets, query, keypoint = [np.asarray(csr[k]) for k in ('observation_offsets', 'query_indices', 'keypoint_indices')]
    targets = [r for r in audit['rows'] if r['status'] == 'unstable' and 7 <= r['full_observation_count'] <= 19 and r['candidate_kind']=='v2_projective_track']
    changes, candidate_count = [], 0
    for entry in targets:
        a = entry['anchor_row']
        xyz = np.asarray(state['anchor_xyz'][a], dtype=float)
        q, k = query[offsets[a]:offsets[a+1]], keypoint[offsets[a]:offsets[a+1]]
        if len(q) != entry['full_observation_count']: raise ValueError('geometry rows differ')
        cam = np.einsum('nij,j->ni', T[:, :3, :3], xyz) + T[:, :3, 3]
        h = np.einsum('nij,nj->ni', K, cam)
        projected = h[:, :2] / np.maximum(h[:, 2:], 1e-8)
        existing = set(q.tolist())
        possible = []
        descriptor = features[a] / max(np.linalg.norm(features[a]), 1e-8)
        for i, rec in enumerate(records):
            if i in existing or cam[i, 2] <= 0: continue
            height, width = np.asarray(rec['native_input_hw'])
            if not (8 <= projected[i, 0] < width-8 and 8 <= projected[i, 1] < height-8): continue
            separation = np.linalg.norm(centers[q]-centers[i], axis=1).min()
            if separation < .25: continue
            near = np.flatnonzero(validity[i] & (np.sum((uv[i]-projected[i])**2, axis=1) <= 64))
            if len(near) == 0: continue
            bank = descriptors[i][near]
            scores = (bank @ descriptor) / np.maximum(np.linalg.norm(bank, axis=1), 1e-8)
            order = np.argsort(-scores, kind='stable')
            if scores[order[0]] < .8 or (len(order)>1 and scores[order[0]]-scores[order[1]] < .04): continue
            possible.append((float(scores[order[0]]), float(separation), i, int(near[order[0]])))
        possible.sort(reverse=True)
        additions = []
        for score, separation, i, j in possible:
            if any(np.linalg.norm(centers[i]-centers[v[0]]) < .25 for v in additions): continue
            additions.append((i, j))
            if len(additions)==2: break
        if not additions: continue
        candidate_count += 1
        aq, ak = np.array(additions).T
        proposal = extend_geometry(np.stack([uv[i][j] for i,j in zip(q,k)]), K[q], T[q], xyz,
                                   np.stack([uv[i][j] for i,j in additions]), K[aq], T[aq])
        if proposal is None: continue
        state['anchor_xyz'][a] = torch.as_tensor(proposal['xyz'], dtype=state['anchor_xyz'].dtype)
        state['anchor_position_covariance'][a] = torch.as_tensor(proposal['covariance'], dtype=state['anchor_position_covariance'].dtype)
        changes.append({'anchor_row': a, 'additional_geometry_query_indices': aq.tolist(),
                        'additional_geometry_keypoint_indices': ak.tolist(), 'shift_m': proposal['shift_m'],
                        'before': proposal['before'], 'after': proposal['after']})
    report = {'schema': 'anygsloc_mapping_track_extension_pilot', 'uses_source_mapping_rgb': False,
              'uses_test_queries': False, 'promoted': False, 'geometry_only_ablation': True,
              'source_map_sha256': map_sha, 'source_cache_sha256': cache_sha,
              'source_geometry_audit_sha256': sha256_file(args.geometry),
              'target_count': len(targets), 'with_added_measurements': candidate_count,
              'accepted_count': len(changes), 'changes': changes,
              'policy': {'track_length': [7,19], 'roi_radius_px': 8., 'minimum_cosine': .8,
                         'minimum_local_margin': .04, 'minimum_camera_separation_m': .25,
                         'maximum_added_views': 2, 'maximum_relative_xyz_update': .02},
              'implementation_sha256': {'script': sha256_file(Path(__file__)),
                'extension': sha256_file(Path(__file__).resolve().parents[1]/'evidence/mapping_track_extension.py')}}
    args.output.mkdir(parents=True, exist_ok=False)
    if changes:
        # Descriptor support and geometry-fit support are intentionally separate.
        state['geometry_extension_observations'] = changes
        state['provenance'] = {**p, 'geometry_extension_pilot': report,
                               'geometry_fit_has_additional_mapping_observations': True,
                               'descriptor_observations_unchanged': True}
        state['anchor_view_support'] = build_anchor_view_support(
            anchor_xyz=state['anchor_xyz'], observation_offsets=csr['observation_offsets'],
            observation_query_indices=csr['query_indices'], mapping_pose_w2c=torch.from_numpy(T).float())
        path = args.output/'projective_anchor_map.pt'
        torch.save(state, path)
        torch.save(identity_metric_state(state, map_path=str(path.resolve()), map_sha256=sha256_file(path)), args.output/'identity_metric.pt')
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='changes'}, indent=2))


if __name__=='__main__': main()
