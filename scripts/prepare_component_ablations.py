"""Fixed-Fast mapping component ablations; never mutate the input map."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import torch
import torch.nn.functional as F
from scripts.materialize_purity_candidates import subset


def materialize(scene, variant, output, map_dir=None, cache_path=None):
    root = Path('/mnt/pool/sqy/lafgs_v24_cambridge_high_capacity_online_20260901') / scene
    map_dir = map_dir or root / 'projective_map'
    state = torch.load(map_dir / 'projective_anchor_map.pt', map_location='cpu', weights_only=False)
    metric = torch.load(map_dir / 'identity_metric.pt', map_location='cpu', weights_only=False)
    assert state['provenance']['uses_test_queries'] is False
    if variant == 'no_completion':
        rows = torch.tensor([i for i, kind in enumerate(state['anchor_candidate_kind']) if kind != 'depth_proposed_projective_completion'])
        revised = subset(state, rows)
        assert len(rows) < len(state['anchor_ids']), 'No completion Anchors identified'
    else:
        cache = torch.load(cache_path or root / 'observations/rendered_rgb_feature_cache.pt', map_location='cpu', weights_only=False)
        names = state['v6_mapping_query_names']
        records = cache['queries']
        features = [F.normalize(records[name]['native_descriptors'].float(), dim=1) for name in names]
        csr = state['projective_anchor_observations']
        offsets = csr['observation_offsets'].tolist()
        query = csr['query_indices'].tolist(); keypoint = csr['keypoint_indices'].tolist()
        result = []
        for i in range(len(state['anchor_ids'])):
            observations = [features[query[j]][keypoint[j]] for j in range(offsets[i], offsets[i+1])]
            if variant == 'mean_descriptor':
                descriptor = torch.stack(observations).mean(0)
            else:
                # Deterministic mapping-only representative, not chosen by test performance.
                j = min(range(offsets[i], offsets[i+1]), key=lambda j: (names[query[j]], keypoint[j]))
                descriptor = features[query[j]][keypoint[j]]
            result.append(F.normalize(descriptor, dim=0))
        revised = dict(state)
        revised['anchor_features'] = torch.stack(result)
        revised['anchor_observation_features'] = revised['anchor_features'].clone()
        revised['anchor_descriptor_residual'] = torch.zeros_like(revised['anchor_features'])
        assert torch.equal(revised['anchor_xyz'], state['anchor_xyz'])
    revised['provenance'] = dict(state['provenance'], component_ablation=variant)
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'projective_anchor_map.pt'
    torch.save(revised, path)
    with path.open('rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    torch.save(dict(metric, landmark_indices=revised['anchor_ids'], map_path=str(path), map_sha256=sha), output / 'identity_metric.pt')
    print(scene, variant, len(revised['anchor_ids']), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path)
    p.add_argument('--scene'); p.add_argument('--variant', choices=['no_completion','mean_descriptor','single_descriptor'])
    p.add_argument('--output', type=Path)
    p.add_argument('--map-dir', type=Path)
    p.add_argument('--cache-path', type=Path)
    a = p.parse_args()
    torch.set_num_threads(4)
    if a.scene:
        return materialize(a.scene, a.variant, a.output, a.map_dir, a.cache_path)
    a.root.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1]
    code = a.root / 'code'
    for directory in ['common','data','evidence','features','localization','evaluation','priors','map_learning','topology','scripts','configs']:
        shutil.copytree(source / directory, code / directory, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    original = json.loads(Path('/mnt/pool/sqy/anygsloc_core65_budget_20260910/gpu0/jobs.json').read_text())
    jobs = []
    for scene in ['ShopFacade','StMarysChurch','OldHospital','GreatCourt','KingsCollege']:
        ref = next(j for j in original if j['id']==f'{scene}_core65_min250_conf999_max30k_seed2026')
        for variant in ['full_reference','no_completion','mean_descriptor','single_descriptor']:
            command = list(ref['command']); command[command.index('--config')+1] = str(code/'configs/anygsloc_mainline.yaml')
            result = a.root / 'runs' / scene / variant
            command[command.index('--output')+1] = str(result)
            deps = []
            env = dict(ref['env'], PYTHONPATH=str(code))
            if variant != 'full_reference':
                output = a.root/'maps'/scene/variant; key = f'{scene}_{variant}_map'; deps=[key]
                jobs.append(dict(id=key, command=[command[0],'-u','-m','scripts.prepare_component_ablations','--scene',scene,'--variant',variant,'--output',str(output)], cwd=str(code),env=env,deps=[],outputs=[str(output/'projective_anchor_map.pt'),str(output/'identity_metric.pt')]))
                for flag,name in [('--map','projective_anchor_map.pt'),('--metric-state','identity_metric.pt')]:
                    command[command.index(flag)+1]=str(output/name)
            jobs.append(dict(ref,id=f'{scene}_{variant}_eval',command=command,cwd=str(code),env=env,deps=deps,outputs=[str(result/'summary.json'),str(result/'results.json')]))
    (a.root/'jobs.json').write_text(json.dumps(jobs,indent=2))
    (a.root/'protocol.json').write_text(json.dumps(dict(phase=1,scenes=5,seed=2026,online='frozen full Fast args identical for all cells',variants=['full_reference','no_completion','mean_descriptor','single_descriptor'],pending=['no_V2_requires_reassociation','association_rules','robust_vs_plain_triangulation'],claim='Mean/single descriptor contrasts change fusion only; removal of completion also changes capacity'),indent=2))
    for path in code.rglob('*'):
        if path.is_file(): path.chmod(0o444)
    print(len(jobs),'jobs registered',flush=True)


if __name__ == '__main__':
    main()
