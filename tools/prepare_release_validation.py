"""Create an explicit, restart-safe E1 registry from verified existing assets.

Uses the archived durable queue as infrastructure, not as a localization method.
No prior is reconstructed and no test query is used to select map points.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
POOL = Path('/mnt/pool/sqy')
OLD = POOL / 'anygsloc_paper_experiments_20260902'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--refresh-unstarted', action='store_true')
    p.add_argument('--priors', action='store_true')
    a = p.parse_args()
    out = a.output.resolve()
    if (out / 'registry.json').exists() and not a.refresh_unstarted:
        raise FileExistsError(out / 'registry.json')
    if list(out.glob('gpu*/state.json')):
        raise RuntimeError('cannot change an already started registry')
    cells = json.loads((OLD / 'primary_24_scene_all_input_audit.json').read_text())['cells']
    evidence = json.loads((POOL/'anygsloc_paper_tables_20260910/results_inventory.json').read_text())['evaluations']
    completed = {(r['family'], r['scene']): Path(r['map']).parent for r in evidence
                 if r['condition'] == 'Base' and r['seed'] == 2026}
    if a.priors:
        conditions = {'vanilla_2dgs','vanilla_3dgs','long_lrm_raw','long_lrm_1k','i3dgs_raw','i3dgs_1k'}
        cells = [dict(family='CambridgePrior', scene=r['scene']+'/'+r['condition'],
                      dataset=str(POOL/'Cambridge_stdloc'/r['scene']), gaussian_ply=None,
                      prior=r['condition'], source_map=str(Path(r['map']).parent))
                 for r in evidence if r['family']=='Cambridge' and r['condition'] in conditions and r['seed']==2026]
        if len(cells) != 30:
            raise ValueError('expected all five scenes and six alternative priors')
    known = {
        'ShopFacade': POOL / 'anygsloc_full_selector_shop_20260911/maps/full_selector',
        'OldHospital': POOL / 'anygsloc_full_selector_validation_20260911/OldHospital/maps/full_selector',
        'KingsCollege': POOL / 'anygsloc_full_selector_validation_20260911/KingsCollege/maps/full_selector',
        'GreatCourt': POOL / 'anygsloc_missing_cache_downstream_20260912/GreatCourt/selector_full/maps/full_selector',
        'StMarysChurch': POOL / 'anygsloc_missing_cache_downstream_20260912/StMarysChurch/selector_full/maps/full_selector',
    }
    lanes = [[], [], []]
    registry = []
    for index, cell in enumerate(cells):
        family, scene = cell['family'], cell['scene']
        key = family + '/' + scene
        lane = index % 3
        base = OLD / key
        compact = known.get(scene)
        full = Path(cell['source_map']) if 'source_map' in cell else completed.get((family, scene), base / 'projective_map')
        cache = full.parent / 'observations/rendered_rgb_feature_cache.pt' if a.priors else base / 'observations/rendered_rgb_feature_cache.pt'
        if not cache.exists():
            cache = POOL / 'anygsloc_execution_queue_20260909/runs' / ('primary_'+scene) / 'observations/rendered_rgb_feature_cache.pt'
        target = out / 'maps' / key
        exported = target / 'deployment_map.pt'
        missing = []
        inputs = [Path(cell['dataset']) / 'processed',
                  (compact or full) / 'projective_anchor_map.pt',
                  (compact or full) / 'identity_metric.pt']
        if not compact:
            inputs.append(cache)
        missing = [str(path) for path in inputs if not path.exists()]
        record = dict(key=key, dataset=cell['dataset'], prior=cell.get('prior', 'MAtCHA' if family == 'Cambridge' else 'Vanilla2DGS'),
                      prior_ply=cell['gaussian_ply'], source_map=str(compact or full),
                      cache=str(cache) if not compact else None,
                      deployment_map=str(exported), gpu=lane, missing_inputs=missing,
                      status='blocked_inputs' if missing else 'queued',
                      evidence_status='provisional_until_full_E0_driver_and_input_audit')
        registry.append(record)
        if missing:
            continue
        prefix = (family + '_' + scene).replace('/', '_')
        mask_policy = 'explicit_valid_mask' if family.startswith('Cambridge') else 'none'
        record['mask_policy'] = mask_policy
        if compact:
            command = [sys.executable, '-m', 'anygsloc', 'export-map', '--source', str(compact / 'projective_anchor_map.pt'),
                       '--metric', str(compact / 'identity_metric.pt'), '--output', str(exported)]
        else:
            command = [sys.executable, '-m', 'anygsloc', 'compact-map', '--map-dir', str(full),
                       '--cache', str(cache), '--output', str(target)]
        lanes[lane].append(dict(id=prefix+'_map', command=command, cwd=str(ROOT), outputs=[str(exported)]))
        for seed in (2026, 2027, 2028):
            result = out / 'runs' / key / str(seed)
            command = [sys.executable, '-m', 'anygsloc', 'evaluate', '--dataset', cell['dataset'],
                       '--map', str(exported), '--output', str(result), '--image-domain', 'rectified',
                       '--mask-policy', mask_policy, '--device', 'cuda:0', '--seed', str(seed)]
            lanes[lane].append(dict(id=prefix+'_'+str(seed), deps=[prefix+'_map'], command=command,
                                   cwd=str(ROOT), outputs=[str(result/'results.json'), str(result/'summary.json')]))
    out.mkdir(parents=True, exist_ok=True)
    (out/'registry.json').write_text(json.dumps(dict(schema='release_validation_v1', cells=registry,
        formal_timing=False, self_implemented_traditional_baselines=False), indent=2))
    hashes = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
              for f in sorted((ROOT/'anygsloc').rglob('*.py'))}
    (out/'source_hashes.json').write_text(json.dumps(hashes, indent=2))
    for lane, jobs in enumerate(lanes):
        folder = out / f'gpu{lane}'
        folder.mkdir(exist_ok=True)
        (folder/'jobs.json').write_text(json.dumps(jobs, indent=2))
    print(json.dumps(dict(cells=len(registry), queued=sum(not r['missing_inputs'] for r in registry),
                         blocked=[r for r in registry if r['missing_inputs']], jobs=[len(x) for x in lanes]), indent=2))


if __name__ == '__main__':
    main()
