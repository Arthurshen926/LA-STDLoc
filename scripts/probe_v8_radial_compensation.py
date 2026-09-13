"""Fixed-track, offline compensation for pixel-attested double undistortion."""
import argparse
import json
from pathlib import Path
import torch
from common.hashing import sha256_file
from data.colmap import read_extrinsics_binary, read_intrinsics_binary
from evidence.radial_compensation import undo_extra_undistortion, refit_fixed_tracks
from map_learning.v24_anchor_view_support import build_anchor_view_support
from topology.v6_anchor_map import identity_metric_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['map', 'cache', 'dataset', 'pixel-audit', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    audit = json.loads(args.pixel_audit.read_text())
    if audit['dataset'] != str(args.dataset) or audit['uses_test_queries'] or len(audit['rows']) < 12:
        raise ValueError('matching mapping-only pixel audit required')
    if not all(r['processed_vs_raw_clahe_undistort']['bit_exact'] and r['prior_vs_undistort_processed']['bit_exact'] for r in audit['rows']):
        raise ValueError('double undistortion not pixel attested')
    torch.set_num_threads(4)
    state = torch.load(args.map, map_location='cpu', weights_only=False)
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    if sha256_file(args.cache) != state['provenance']['v8_observation_cache_sha256']: raise ValueError('cache lineage mismatch')
    for payload in [state['provenance'], cache]:
        if any(payload.get(k) is not False for k in ['uses_source_mapping_rgb', 'uses_test_queries']): raise ValueError('render mapping-only required')
    if list(cache['queries']) != state['v6_mapping_query_names']: raise ValueError('registry mismatch')
    cameras = read_intrinsics_binary(str(args.dataset / 'sparse/0/cameras.bin'))
    images = {r.name: r for r in read_extrinsics_binary(str(args.dataset / 'sparse/0/images.bin')).values()}
    records = list(cache['queries'].values())
    radial = []
    for name in cache['queries']:
        camera = cameras[images[name].camera_id]
        if camera.model != 'SIMPLE_RADIAL': raise ValueError('expected original radial calibration')
        radial.append(float(camera.params[3]))
    csr = state['projective_anchor_observations']
    if csr.get('keypoint_index_semantics') != 'original_unfiltered_observation_cache_row': raise ValueError('source rows required')
    offsets = csr['observation_offsets'].long(); q = csr['query_indices'].long(); k = csr['keypoint_indices'].long()
    raw = torch.cat([torch.zeros(1, dtype=torch.long), torch.tensor([len(r['native_keypoints']) for r in records]).cumsum(0)])
    uv = torch.cat([r['native_keypoints'].double() for r in records])[raw[q] + k]
    allK = torch.stack([torch.as_tensor(r['native_K']).double() for r in records])
    allT = torch.stack([torch.as_tensor(r['pose_w2c']).double() for r in records])
    K, T = allK[q], allT[q]
    revised = undo_extra_undistortion(uv, K, torch.tensor(radial).double()[q])
    row = torch.repeat_interleave(torch.arange(len(state['anchor_xyz'])), offsets[1:] - offsets[:-1])
    original = state['anchor_xyz'].clone()
    report = {'schema': 'anygsloc_attested_extra_rectification_compensation_v1', 'promoted': False,
              'uses_source_mapping_rgb': False, 'uses_test_queries': False,
              'audit_uses_source_mapping_rgb': True, 'pixel_audit_sha256': sha256_file(args.pixel_audit),
              'source_map_sha256': sha256_file(args.map), 'source_cache_sha256': sha256_file(args.cache),
              'descriptor_ids_csr_capacity_unchanged': True, 'online_cost_added': False,
              'scope': 'Approximate frozen-map salvage; cannot undo prior training or erroneous Track selection',
              'uv_shift_px_p50_p90': torch.quantile(torch.linalg.vector_norm(revised-uv, dim=1), torch.tensor([.5,.9], dtype=torch.double)).tolist(),
              'implementation_sha256': sha256_file(Path(__file__)),
              'geometry_implementation_sha256': sha256_file(Path(__file__).resolve().parents[1] / 'evidence/radial_compensation.py')}
    args.output.mkdir(parents=True)
    for mode, points in [('refit_control', uv), ('radial_compensated', revised)]:
        xyz, valid = refit_fixed_tracks(original, row, points + .5, K, T)
        state['anchor_xyz'] = xyz
        state.pop('anchor_position_covariance', None)
        state['anchor_view_support'] = build_anchor_view_support(anchor_xyz=xyz, observation_offsets=offsets, observation_query_indices=q, mapping_pose_w2c=allT.float())
        detail = {**report, 'mode': mode, 'fallback_count': int((~valid).sum()),
                  'xyz_shift_cm_p50_p90': torch.quantile(torch.linalg.vector_norm(xyz-original, dim=1)*100, torch.tensor([.5,.9])).tolist()}
        state['provenance'] = {**state['provenance'], 'radial_compensation': detail}
        state['projective_anchor_construction'] = {**state['projective_anchor_construction'], 'final_xyz_source': mode + '_fixed_camera_huber2_10_steps'}
        directory = args.output / mode; directory.mkdir()
        path = directory / 'projective_anchor_map.pt'; torch.save(state, path)
        torch.save(identity_metric_state(state, map_path=str(path.resolve()), map_sha256=sha256_file(path)), directory / 'identity_metric.pt')
        torch.save({'physical_uv': points + .5, 'refit_valid': valid}, directory / 'observation_override.pt')
        (directory / 'report.json').write_text(json.dumps(detail, indent=2))
        print(json.dumps(detail), flush=True)


if __name__ == '__main__': main()
