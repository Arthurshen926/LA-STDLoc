"""Prepare immutable 2x2x2x2 Cambridge factorial; explicit --launch required."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

POOL = Path('/mnt/pool/sqy')
REPO = Path(__file__).resolve().parents[1]
SCENES = ['ShopFacade', 'KingsCollege', 'OldHospital', 'StMarysChurch', 'GreatCourt']


def inputs(scene, v2):
    late = scene in ['StMarysChurch', 'GreatCourt']
    cache_root = POOL / ('anygsloc_cache_recovery_v2_20260912' if late else
                        'lafgs_v24_cambridge_high_capacity_online_20260901') / scene
    if v2:
        root = POOL / ('anygsloc_missing_cache_downstream_20260912' if late else
                       'lafgs_v24_cambridge_high_capacity_online_20260901') / scene
    else:
        root = (POOL / 'anygsloc_final_ablation_20260913/maps' / scene / 'no_v2' if late else
                POOL / 'anygsloc_core_geometry_20260911' / scene / 'no_v2')
    return root / 'projective_map', cache_root / 'observations/rendered_rgb_feature_cache.pt'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--launch', action='store_true')
    a = p.parse_args()
    for scene, v2 in itertools.product(SCENES, [1, 0]):
        mp, cache = inputs(scene, v2)
        for path in [mp / 'projective_anchor_map.pt', mp / 'identity_metric.pt', cache]:
            if not path.is_file():
                raise FileNotFoundError(path)
    a.output.mkdir(parents=True, exist_ok=False)
    code = a.output / 'code'
    for folder in ['anygsloc', 'tools']:
        shutil.copytree(REPO / folder, code / folder, ignore=shutil.ignore_patterns('__pycache__'))
    (code / 'scripts').mkdir()
    shutil.copy2(REPO / 'research/legacy/scripts/durable_experiment_queue.py', code / 'scripts')
    shutil.copy2(REPO / 'docs/release/factorial_ablation_protocol.md', a.output / 'protocol.md')
    env = dict(PYTHONPATH=str(code), PATH=str(Path(sys.executable).parent) + ':' + os.environ['PATH'],
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    jobs = []
    registry = []
    def add(key, command, outputs, deps=()):
        jobs.append(dict(id=key, command=list(map(str, command)), outputs=list(map(str, outputs)),
                         deps=list(deps), cwd=str(code), env=env))
    for scene, v2, fusion, completion in itertools.product(SCENES, [1, 0], [1, 0], [1, 0]):
        name = f'{scene}_v{v2}_f{fusion}_c{completion}'
        target = a.output / 'maps' / name
        mp, cache = inputs(scene, v2)
        add(name + '_build', [sys.executable, code / 'tools/observation_factorial.py',
            '--map-dir', mp, '--cache', cache, '--output', target,
            '--fusion', fusion, '--completion', completion],
            [target / 'report.json', target / 'full/deployment_map.pt', target / 'compact/deployment_map.pt'])
        for selection in ['full', 'compact']:
            key = name + '_' + selection
            result = a.output / 'runs' / key
            deployment = target / selection / 'deployment_map.pt'
            add(key, [sys.executable, '-m', 'anygsloc', 'evaluate', '--dataset',
                POOL / 'Cambridge_stdloc' / scene, '--map', deployment, '--output', result,
                '--image-domain', 'rectified', '--mask-policy', 'geometric_validity',
                '--rectification-contract', 'opencv_same_k', '--seed', '2026', '--device', 'cuda:0'],
                [result / 'summary.json', result / 'results.json'], [name + '_build'])
            registry.append(dict(id=key, scene=scene, v2=v2, fusion=fusion, completion=completion,
                                 selection=selection, map=str(deployment), result=str(result)))
    # Timing rounds intentionally run after ALL accuracy/map tasks, never concurrently.
    for row in registry:
        target = a.output / 'benchmarks' / row['id']
        add(row['id'] + '_benchmark', [sys.executable, code / 'tools/benchmark_release.py',
            '--dataset', POOL / 'Cambridge_stdloc' / row['scene'], '--map', row['map'],
            '--output', target], [target / 'summary.json'], [row['id']])
    lane = a.output / 'gpu0'
    lane.mkdir()
    (lane / 'jobs.json').write_text(json.dumps(jobs, indent=2))
    (a.output / 'registry.json').write_text(json.dumps(registry, indent=2))
    (a.output / 'source_hashes.json').write_text(json.dumps({str(f.relative_to(code)):
        hashlib.sha256(f.read_bytes()).hexdigest() for f in code.rglob('*.py')}, indent=2))
    if a.launch:
        with (lane / 'watchdog.log').open('ab', buffering=0) as log:
            child = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.durable_experiment_queue',
                '--root', str(lane), '--gpu', '0', '--watchdog'], cwd=code,
                env=dict(os.environ, **env), stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        print('launched', child.pid)
    print('accuracy', len(registry), 'jobs', len(jobs), 'root', a.output)


if __name__ == '__main__':
    main()
