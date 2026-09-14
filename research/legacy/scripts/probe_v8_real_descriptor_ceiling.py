"""Real-mapping-RGB diagnostic, with bit-exact source fusion and frozen XYZ.

Not a deployable source-image-free method. Real RGB is never used to choose
tracks, keypoints, weights, XYZ, or query-time gates in this diagnostic.
"""
import argparse
import json
from pathlib import Path
import time
import torch
from common.hashing import sha256_file
from data.datasets import ColmapDataset
from evidence.v8_fixed_descriptor_replay import prepare_replay, fuse_replay
from features.extractor import FeatureExtractor
from features.superpoint import sample_descriptors, SUPERPOINT_WEIGHT_SHA256
from scripts.materialize_mapping_rgb_descriptors import resolve_attested_mapping_cameras
from topology.v6_anchor_map import identity_metric_state


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--association', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    started = time.perf_counter()
    torch.set_num_threads(4)
    state = torch.load(args.map, weights_only=False, map_location='cpu')
    cache = torch.load(args.cache, weights_only=False, map_location='cpu')
    association = torch.load(args.association, weights_only=False, map_location='cpu')
    p = state['provenance']
    for payload in (p, cache, association):
        if payload.get('uses_source_mapping_rgb') is not False or payload.get('uses_test_queries') is not False:
            raise ValueError('source inputs must be frozen render-only mapping artifacts')
    cache_sha, graph_sha = sha256_file(args.cache), sha256_file(args.association)
    if p['v8_observation_cache_sha256'] != cache_sha or p['v8_association_graph_sha256'] != graph_sha:
        raise ValueError('source lineage mismatch')
    replay = prepare_replay(state, cache, association)
    descriptors = [r['native_descriptors'].float() for r in cache['queries'].values()]
    source = fuse_replay(descriptors, replay)
    if not torch.equal(source, state['anchor_features']):
        raise ValueError(f'original fusion not bit-exact: max delta {float((source-state["anchor_features"]).abs().max())}')
    print(f'Exact source fusion PASS: {len(source)} Anchors', flush=True)
    dataset = ColmapDataset(args.dataset, images='processed')
    cameras, registry = resolve_attested_mapping_cameras(dataset.split('mapping'), cache, list(cache['queries']))
    extractor = FeatureExtractor('sp', nms_radius=4).cuda().eval()
    rgb_descriptors, image_hashes = [], []
    for i, camera in enumerate(cameras):
        record = cache['queries'][camera.image_name]
        image = dataset.load_image(camera).cuda()
        dense, _ = extractor.detectAndComputeDense(image[None])
        points = record['native_keypoints'].float().cuda()
        rgb_descriptors.append(sample_descriptors(points[None], dense)[0].T.cpu())
        image_hashes.append({'name': camera.image_name, 'sha256': sha256_file(camera.image_path)})
        if (i+1)%100==0 or i+1==len(cameras): print(f'Real descriptor sampling {i+1}/{len(cameras)}', flush=True)
    fused = fuse_replay(rgb_descriptors, replay)
    state['anchor_features'] = fused
    state['anchor_observation_features'] = fused.clone()
    state['anchor_descriptor_residual'] = torch.zeros_like(fused)
    report = {'schema': 'anygsloc_v8_fixed_geometry_real_descriptor_diagnostic',
              'uses_source_mapping_rgb': True, 'uses_test_queries': False, 'deployment_eligible': False,
              'source_fusion_bit_exact': True, 'xyz_identity_csr_unchanged': True,
              'fusion_weights_and_bins_unchanged': True, 'fixed_render_keypoint_positions': True,
              'source_map_sha256': sha256_file(args.map), 'source_cache_sha256': cache_sha,
              'source_association_sha256': graph_sha, 'camera_registry': registry,
              'superpoint_weight_sha256': SUPERPOINT_WEIGHT_SHA256,
              'images': image_hashes, 'anchor_count': len(fused),
              'implementation_sha256': {'script': sha256_file(Path(__file__)),
                'fusion': sha256_file(Path(__file__).resolve().parents[1]/'evidence/v8_fixed_descriptor_replay.py')},
              'seconds': time.perf_counter()-started}
    state['provenance'] = {**p, 'uses_source_mapping_rgb': True, 'deployment_eligible': False,
                           'real_descriptor_diagnostic': report}
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output/'diagnostic_map.pt'
    torch.save(state, path)
    torch.save(identity_metric_state(state, map_path=str(path.resolve()), map_sha256=sha256_file(path)), args.output/'identity_metric.pt')
    torch.save({'uses_source_mapping_rgb': True, 'uses_test_queries': False,
                'query_names': list(cache['queries']), 'descriptors': rgb_descriptors,
                'source_cache_sha256': cache_sha}, args.output/'fixed_row_real_descriptors.pt')
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    print(f'Completed real descriptor diagnostic at {args.output}', flush=True)


if __name__=='__main__': main()
