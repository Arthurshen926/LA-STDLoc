"""Register three-seed one-factor RANSAC budget tests against frozen first-pose code."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    old = Path('/mnt/pool/sqy/anygsloc_first_pose_efficiency_20260910')
    previous = json.loads((old / 'protocol.json').read_text())
    # Fail before creating a queue if either the code or maps changed.
    for registry, prefix in [(previous['code_sha256'], old / 'code'),
                             (previous['input_sha256'], Path('/'))]:
        for name, expected in registry.items():
            path = prefix / name
            with path.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != expected:
                raise ValueError(f'Frozen input changed: {path}')
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    variants = {
        'base': [],
        'core65': [],
        'core65_min250': ['--ransac-minimum-iterations-override', '250'],
        'core65_conf999': ['--ransac-confidence-override', '0.999'],
        'core65_max30k': ['--ransac-maximum-iterations-override', '30000'],
        'core65_min250_conf999': ['--ransac-minimum-iterations-override', '250',
                                '--ransac-confidence-override', '0.999'],
        'core65_min250_conf999_max30k': ['--ransac-minimum-iterations-override', '250',
                                       '--ransac-confidence-override', '0.999',
                                       '--ransac-maximum-iterations-override', '30000'],
    }
    base_jobs = [j for j in json.loads((old / 'jobs.json').read_text())
                 if j['id'].endswith('_base')]
    for gpu, seed in enumerate([2026, 2027, 2028]):
        lane = root / f'gpu{gpu}'
        lane.mkdir()
        jobs = []
        for original in base_jobs:
            scene = original['id'].removesuffix('_base')
            for variant, flags in variants.items():
                output = root / 'runs' / scene / f'{variant}_seed{seed}'
                command = list(original['command'])
                command[command.index('--output') + 1] = str(output)
                command[command.index('--seed') + 1] = str(seed)
                if variant != 'base':
                    command += ['--match-retention-fraction', '0.65']
                command += flags
                jobs.append(dict(original, id=f'{scene}_{variant}_seed{seed}',
                                 command=command,
                                 env=dict(original['env'], CUDA_VISIBLE_DEVICES=str(gpu)),
                                 outputs=[str(output / n) for n in
                                          ['summary.json', 'results.json', 'deployment_contract.json']]))
        (lane / 'jobs.json').write_text(json.dumps(jobs, indent=2))
    protocol = dict(previous, seeds=[2026, 2027, 2028], variants=variants,
                    frozen_code=str(old / 'code'), seed=None,
                    timing='One process per GPU, seed bound to GPU; shared CPU contention. '
                           'Compare paired baseline per lane. Final isolated timing still required.',
                    decision='No automatic promotion. Report every scene and seed, median TE/RE, '
                             'R5/R10/R15/R50, >=1m failures, mean/P50/P90 latency. '
                             'A lower maximum budget is not a reliability guarantee.')
    (root / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    print(root, 3 * len(base_jobs) * len(variants), 'registered evaluations')


if __name__ == '__main__':
    main()
