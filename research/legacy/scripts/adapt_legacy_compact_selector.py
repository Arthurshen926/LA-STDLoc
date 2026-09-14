"""Reuse legacy matching-coverage selection on current mapping-only F0 Anchors."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from topology.sufficiency_selector import select_precision_compatibility
from scripts.materialize_purity_candidates import subset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=False)
    original = json.loads(Path('/mnt/pool/sqy/anygsloc_core65_budget_20260910/gpu0/jobs.json').read_text())
    jobs, reports = [], {}
    for scene in ['StMarysChurch', 'OldHospital', 'GreatCourt', 'KingsCollege', 'ShopFacade']:
        folder = Path('/mnt/pool/sqy/lafgs_v24_cambridge_high_capacity_online_20260901') / scene / 'projective_map'
        source_map = folder / 'projective_anchor_map.pt'
        state = torch.load(source_map, map_location='cpu', weights_only=False)
        assert state['provenance']['uses_test_queries'] is False
        metric = torch.load(folder / 'identity_metric.pt', map_location='cpu', weights_only=False)
        csr = state['projective_anchor_observations']
        offsets, queries, points = [csr[k].tolist() for k in ['observation_offsets', 'query_indices', 'keypoint_indices']]
        n = len(state['anchor_ids'])
        edges = []
        for i in range(n):
            edge = {}
            for j in range(offsets[i], offsets[i+1]):
                edge.setdefault(queries[j], []).append(points[j])
            edges.append(edge)
        # Explicit adaptation: existing mapping matchability, then independent-view support.
        # No query-derived counts or learned metric are introduced.
        reliability = state['anchor_matchability'].float()
        order = torch.tensor(sorted(range(n), key=lambda i: (-float(reliability[i]), -len(edges[i]), i)))
        reports[scene] = {}
        for target in [256, 512, 1024]:
            chosen, report = select_precision_compatibility(edges, order,
                len(state['v6_mapping_query_names']), target,
                minimum_count=1024, maximum_count=n, check_interval=512)
            rows = chosen.sort().values
            output = a.root / 'maps' / scene / f'target{target}'
            output.mkdir(parents=True)
            revised = subset(state, rows)
            revised['provenance'] = dict(state['provenance'], compact_selector=dict(
                source_map=str(source_map), retained_source_rows=rows,
                uses_test_queries=False, offline_feedback=False, report=report))
            path = output / 'projective_anchor_map.pt'
            torch.save(revised, path)
            with path.open('rb') as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            torch.save(dict(metric, landmark_indices=revised['anchor_ids'], map_path=str(path),
                            map_sha256=sha), output / 'identity_metric.pt')
            reports[scene][str(target)] = report
            print(scene, target, len(rows), report['matching_rank_p10'], flush=True)
            for policy in ['base', 'fast_no_gate']:
                reference_policy = 'base' if policy == 'base' else 'core65_min250_conf999_max30k'
                job = next(j for j in original if j['id'] == f'{scene}_{reference_policy}_seed2026')
                command = list(job['command'])
                if '--match-retention-fraction' in command:
                    i = command.index('--match-retention-fraction')
                    del command[i:i+2]
                result = a.root / 'runs' / scene / f'target{target}' / policy
                for option, value in [('--map', path), ('--metric-state', output / 'identity_metric.pt'), ('--output', result)]:
                    command[command.index(option)+1] = str(value)
                jobs.append(dict(job, id=f'{scene}_target{target}_{policy}', command=command,
                                 outputs=[str(result / name) for name in ['summary.json', 'results.json']]))
        # Full-map Fast-without-gate is the missing budget-matched control.
        job = next(j for j in original if j['id'] == f'{scene}_core65_min250_conf999_max30k_seed2026')
        command = list(job['command']); i = command.index('--match-retention-fraction'); del command[i:i+2]
        result = a.root / 'runs' / scene / 'full' / 'fast_no_gate'
        command[command.index('--output')+1] = str(result)
        jobs.append(dict(job, id=f'{scene}_full_fast_no_gate', command=command,
                         outputs=[str(result / name) for name in ['summary.json', 'results.json']]))
    (a.root / 'selection_report.json').write_text(json.dumps(reports, indent=2))
    (a.root / 'jobs.json').write_text(json.dumps(jobs, indent=2))
    print(len(jobs), 'evaluations registered', flush=True)


if __name__ == '__main__':
    main()
