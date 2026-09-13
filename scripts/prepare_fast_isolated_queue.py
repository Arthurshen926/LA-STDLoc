"""Repeat fixed Fast candidates serially, with paired full-map references."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = Path('/mnt/pool/sqy/anygsloc_core65_budget_20260910')
    original = json.loads((source / 'gpu0/jobs.json').read_text())
    variants = ['base', 'core65', 'core65_min250_conf999_max30k']
    jobs = []
    for repeat in range(2):
        for job in original:
            if not any(job['id'].endswith('_' + v + '_seed2026') for v in variants):
                continue
            command = list(job['command'])
            output = root / 'runs' / f'repeat{repeat}' / job['id']
            command[command.index('--output') + 1] = str(output)
            jobs.append(dict(job, id=f"repeat{repeat}_{job['id']}", command=command,
                             outputs=[str(output / n) for n in
                                      ['summary.json', 'results.json', 'deployment_contract.json']]))
    (root / 'jobs.json').write_text(json.dumps(jobs, indent=2))
    (root / 'protocol.json').write_text(json.dumps(dict(
        source_protocol=str(source / 'protocol.json'), repeats=2, seed=2026,
        variants=variants, gpu=0, serial=True, offline_feedback=False,
        timing_scope='Existing localize timer, excludes IO and map loading. '
                     'Repeat runs are timing replicates, not independent accuracy seeds.',
        promotion='Fast remains candidate; report all scenes and tail metrics.'), indent=2))
    print(len(jobs), 'serial evaluations registered', root)


if __name__ == '__main__':
    main()
