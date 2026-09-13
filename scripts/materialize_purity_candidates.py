"""Materialize audited mapping-only deduplication as new maps and register paired tests."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def subset(state, rows):
    n = len(state['anchor_ids'])
    result = dict(state)
    for key, value in state.items():
        if isinstance(value, torch.Tensor) and value.ndim and len(value) == n and key != 'v6_mapping_query_bins':
            result[key] = value[rows].clone()
    result['anchor_candidate_kind'] = [state['anchor_candidate_kind'][i] for i in rows.tolist()]
    result['anchor_ids'] = torch.arange(len(rows))
    csr = state['projective_anchor_observations']
    offsets = csr['observation_offsets']
    selected = torch.cat([torch.arange(int(offsets[i]), int(offsets[i+1])) for i in rows.tolist()])
    counts = (offsets[1:] - offsets[:-1])[rows]
    result['projective_anchor_observations'] = dict(csr,
        observation_offsets=torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)]),
        query_indices=csr['query_indices'][selected], keypoint_indices=csr['keypoint_indices'][selected])
    result['anchor_view_support'] = {
        k: v[rows].clone() if isinstance(v, torch.Tensor) and v.ndim and len(v) == n else v
        for k, v in state['anchor_view_support'].items()}
    result['canonical_anchor_count'] = len(rows)
    result['micro_anchor_count'] = int((result['anchor_type'] == 1).sum())
    result['base_anchor_count'] = len(rows) - result['micro_anchor_count']
    assert int(result['projective_anchor_observations']['observation_offsets'][-1]) == len(selected)
    assert len(result['anchor_view_support']['mode_count']) == len(rows)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    audit = json.loads((a.root / 'audit.json').read_text())
    reference = Path('/mnt/pool/sqy/anygsloc_core65_budget_20260910')
    original = json.loads((reference / 'gpu0/jobs.json').read_text())
    jobs = []
    for scene, report in audit['reports'].items():
        folder = Path('/mnt/pool/sqy/lafgs_v24_cambridge_high_capacity_online_20260901') / scene / 'projective_map'
        state = torch.load(folder / 'projective_anchor_map.pt', map_location='cpu', weights_only=False)
        metric = torch.load(folder / 'identity_metric.pt', map_location='cpu', weights_only=False)
        for variant, proposal in report['duplicate_proposals'].items():
            output = a.root / 'maps' / scene / variant
            output.mkdir(parents=True, exist_ok=False)
            keep = torch.ones(len(state['anchor_ids']), dtype=torch.bool)
            for removed, survivor in proposal['removed_to_survivor']:
                keep[removed] = False
            assert all(bool(keep[survivor]) for _, survivor in proposal['removed_to_survivor'])
            rows = torch.nonzero(keep, as_tuple=False).flatten()
            revised = subset(state, rows)
            revised['provenance'] = dict(state['provenance'], mapping_only_deduplication=dict(
                variant=variant, uses_test_queries=False, source_map=str(folder / 'projective_anchor_map.pt'),
                retained_source_rows=rows, source_count=len(keep)))
            map_path = output / 'projective_anchor_map.pt'
            torch.save(revised, map_path)
            with map_path.open('rb') as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            revised_metric = dict(metric, landmark_indices=revised['anchor_ids'], map_path=str(map_path), map_sha256=sha)
            torch.save(revised_metric, output / 'identity_metric.pt')
            for policy in ['base', 'core65_min250_conf999_max30k']:
                job = next(j for j in original if j['id'] == f'{scene}_{policy}_seed2026')
                command = list(job['command'])
                result = a.root / 'runs' / scene / variant / policy
                for option, value in [('--map', map_path), ('--metric-state', output / 'identity_metric.pt'), ('--output', result)]:
                    command[command.index(option) + 1] = str(value)
                jobs.append(dict(job, id=f'{scene}_{variant}_{policy}', command=command,
                                 outputs=[str(result / n) for n in ['summary.json', 'results.json']]))
    (a.root / 'jobs.json').write_text(json.dumps(jobs, indent=2))
    print(len(jobs), 'candidate evaluations registered', flush=True)


if __name__ == '__main__':
    main()
