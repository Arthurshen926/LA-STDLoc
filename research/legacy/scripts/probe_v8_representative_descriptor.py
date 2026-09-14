"""Fixed-map render-only single-observation descriptor ablations."""
import argparse
import json
from pathlib import Path
import torch
from common.hashing import sha256_file
from evidence.v8_fixed_descriptor_replay import prepare_replay, fuse_replay
from evidence.representative_descriptor import representatives
from topology.v6_anchor_map import identity_metric_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['map', 'cache', 'association', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    state, cache, graph = [torch.load(path, map_location='cpu', weights_only=False)
                           for path in [args.map, args.cache, args.association]]
    for payload in [state['provenance'], cache, graph]:
        if any(payload.get(k) is not False for k in ['uses_source_mapping_rgb', 'uses_test_queries']):
            raise ValueError('mapping render-only required')
    cache_sha, graph_sha = sha256_file(args.cache), sha256_file(args.association)
    if cache_sha != state['provenance']['v8_observation_cache_sha256'] or graph_sha != state['provenance']['v8_association_graph_sha256']:
        raise ValueError('lineage mismatch')
    replay = prepare_replay(state, cache, graph)
    desc = [r['native_descriptors'].float() for r in cache['queries'].values()]
    source = fuse_replay(desc, replay)
    if not torch.equal(source, state['anchor_features']):
        raise ValueError('source fusion not bit exact')
    print('Exact source fusion PASS', flush=True)
    gathered = torch.cat(desc)[replay['source_index']]
    features, indices = [], []
    for a, (s, e) in enumerate(zip(replay['offsets'][:-1], replay['offsets'][1:])):
        f, i = representatives(gathered[s:e], replay['bins'][s:e], replay['weights'][s:e], source[a])
        features.append(f); indices.append(i + s)
    features, indices = torch.stack(features), torch.stack(indices)
    report = {'schema': 'anygsloc_single_observation_descriptor_v1',
              'uses_source_mapping_rgb': False, 'uses_test_queries': False, 'promoted': False,
              'source_map_sha256': sha256_file(args.map), 'source_cache_sha256': cache_sha,
              'source_association_sha256': graph_sha, 'source_fusion_bit_exact': True,
              'xyz_ids_csr_capacity_unchanged': True, 'online_descriptor_count_per_anchor': 1,
              'implementation_sha256': sha256_file(Path(__file__)),
              'selection_implementation_sha256': sha256_file(Path(__file__).resolve().parents[1] / 'evidence/representative_descriptor.py')}
    args.output.mkdir(parents=True)
    for column, mode in enumerate(['balanced_medoid', 'nearest_f0']):
        fused = features[:, column].contiguous()
        detail = {**report, 'mode': mode, 'changed_anchor_count': int((fused != source).any(1).sum()),
                  'cosine_p10_p50': torch.quantile((source * fused).sum(1), torch.tensor([.1, .5])).tolist()}
        state['anchor_features'] = fused
        state['anchor_observation_features'] = fused.clone()
        state['anchor_descriptor_residual'] = torch.zeros_like(fused)
        state['provenance'] = {**state['provenance'], 'representative_descriptor': detail}
        directory = args.output / mode; directory.mkdir()
        path = directory / 'projective_anchor_map.pt'; torch.save(state, path)
        torch.save(identity_metric_state(state, map_path=str(path.resolve()), map_sha256=sha256_file(path)), directory / 'identity_metric.pt')
        torch.save({'selected_csr_observation': indices[:, column]}, directory / 'selection.pt')
        (directory / 'report.json').write_text(json.dumps(detail, indent=2))
        print(json.dumps(detail), flush=True)


if __name__ == '__main__':
    main()
