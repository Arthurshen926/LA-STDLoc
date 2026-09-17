"""Inventory original cost evidence; never turn missing timing into zero."""
import argparse
import csv
import json
from pathlib import Path

SCENES = ['ShopFacade', 'KingsCollege', 'OldHospital', 'StMarysChurch', 'GreatCourt']
PRIORS = ['MAtCHA', 'vanilla_2dgs', 'vanilla_3dgs', 'long_lrm_raw', 'long_lrm_1k', 'i3dgs_raw', 'i3dgs_1k']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False, parents=True)
    pool = Path('/mnt/pool/sqy')
    evidence = []
    for root in ['anygsloc_three_gpu_20260910', 'anygsloc_corrected_cambridge_matrix_20260909',
                 'anygsloc_execution_queue_20260909', 'anygsloc_registered_matrix_20260909']:
        for path in (pool / root).glob('gpu*/state.json'):
            for key, row in json.loads(path.read_text()).get('jobs', {}).items():
                if any(scene in key for scene in SCENES):
                    evidence.append(dict(source=str(path), id=key, **row))
    rows = []
    conversions = []
    cells = json.loads((pool / 'anygsloc_release_priors_20260913/registry.json').read_text())['cells']
    for cell in cells:
        _, scene, prior = cell['key'].split('/')
        logs = Path(cell['source_map']).parent / 'logs'
        stages = []
        for name in ['observations', 'v2_audit_000', 'v2_audit_001', 'projective_map']:
            f = logs / (name + '.timing.json')
            command = logs / (name + '.command.json')
            if f.is_file():
                measured = json.loads(f.read_text())
                stages.append(dict(stage=name, evidence=str(f), **measured))
            if name == 'observations' and command.is_file():
                cmd = json.loads(command.read_text())['command']
                if '--gaussian-ply' in cmd:
                    stages.append(dict(stage='prior_asset_reference', ply=cmd[cmd.index('--gaussian-ply') + 1],
                                       command_evidence=str(command)))
        conversions.append(dict(scene=scene, prior=prior, source_map=cell['source_map'], stages=stages,
            compact_selection_seconds=None, complete_conversion_seconds=None,
            note='Cached/reused stages are not zero build time; selector and asset hashes require separate reconciliation.'))
    for scene in SCENES:
        for prior in PRIORS:
            native = prior.replace('_raw', '').replace('_1k', '')
            stages = ['reconstruct'] if prior.startswith('vanilla') else ['reconstruct', 'export']
            if prior.endswith('_1k'):
                stages.append('ft1000')
            matches = []
            for stage in stages:
                # Do not choose arbitrarily between duplicate reconstructions.
                candidates = [x for x in evidence if x['id'] == f'{scene}_{native}_{stage}'
                              and x.get('status') == 'complete' and x.get('returncode') == 0]
                if len(candidates) == 1:
                    matches.append(candidates[0])
            complete = prior != 'MAtCHA' and len(matches) == len(stages)
            rows.append(dict(scene=scene, prior=prior,
                historical_prior_job_wall_seconds=sum(x['seconds'] for x in matches) if complete else None,
                prior_status='historical_queue_timing_requires_exact_asset_lineage_audit' if complete else 'missing_or_ambiguous',
                evidence=json.dumps([dict(source=x['source'], id=x['id'], seconds=x['seconds']) for x in matches]),
                conversion_seconds=None, total_seconds=None,
                semantic_preparation_seconds=None, input_preparation_seconds=None,
                note='Queue timing includes polling and startup; not isolated kernel time. Total withheld pending matched lineage and conversion measurement.'))
    (a.output / 'raw_stage_evidence.json').write_text(json.dumps(evidence, indent=2))
    (a.output / 'conversion_stage_evidence.json').write_text(json.dumps(conversions, indent=2))
    with (a.output / 'prior_costs.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('cells', len(rows), 'historical_subtotals', sum(x['historical_prior_job_wall_seconds'] is not None for x in rows))


if __name__ == '__main__':
    main()
