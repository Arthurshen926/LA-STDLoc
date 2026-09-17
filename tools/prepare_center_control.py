"""Build five CPU-only geometry controls; evaluate after baseline GPU lane ends."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    repo = Path(__file__).resolve().parents[1]
    pool = Path('/mnt/pool/sqy')
    out = pool / 'anygsloc_center_control_20260915'
    out.mkdir(exist_ok=False)
    code = out / 'code'
    for folder in ['anygsloc', 'tools']:
        shutil.copytree(repo / folder, code / folder, ignore=shutil.ignore_patterns('__pycache__'))
    (code / 'scripts').mkdir()
    for name in ['durable_experiment_queue.py', 'wait_queue_lane_finished.py']:
        shutil.copy2(repo / 'research/legacy/scripts' / name, code / 'scripts' / name)
    cells = [x for x in json.loads((pool / 'anygsloc_release_validation_20260913/registry.json').read_text())['cells']
             if x['key'].startswith('Cambridge/')]
    jobs = []
    def add(key, cmd, outputs, deps=()):
        jobs.append(dict(id=key, command=list(map(str, cmd)), outputs=list(map(str, outputs)),
                         deps=list(deps), cwd=str(code), env=dict(PYTHONPATH=str(code))))
    for cell in cells:
        scene = cell['key'].split('/')[1]
        late = scene in ['GreatCourt', 'StMarysChurch']
        cache = pool / ('anygsloc_cache_recovery_v2_20260912' if late else
                        'lafgs_v24_cambridge_high_capacity_online_20260901') / scene / 'observations/rendered_rgb_feature_cache.pt'
        for path in [cache, Path(cell['prior_ply']), Path(cell['source_map']) / 'projective_anchor_map.pt']:
            if not path.is_file():
                raise FileNotFoundError(path)
        target = out / 'maps' / scene
        add(scene + '_build', [sys.executable, code / 'tools/gaussian_center_control.py',
            '--map-dir', cell['source_map'], '--cache', cache, '--ply', cell['prior_ply'], '--output', target],
            [target / 'report.json'] + [target / k / 'deployment_map.pt' for k in ['ray', 'depth', 'center']])
    add('wait_baselines', [sys.executable, '-m', 'scripts.wait_queue_lane_finished', '--state',
        pool / 'anygsloc_cambridge_recovery_v3_20260915/gpu0/state.json'], [])
    for cell in cells:
        scene = cell['key'].split('/')[1]
        for kind in ['ray', 'depth', 'center']:
            target = out / 'runs' / scene / kind
            add(scene + '_' + kind, [sys.executable, '-m', 'anygsloc', 'evaluate', '--dataset', cell['dataset'],
                '--map', out / 'maps' / scene / kind / 'deployment_map.pt', '--output', target,
                '--image-domain', 'rectified', '--mask-policy', 'geometric_validity',
                '--rectification-contract', 'opencv_same_k', '--seed', '2026', '--device', 'cuda:0'],
                [target / 'summary.json', target / 'results.json'], [scene + '_build', 'wait_baselines'])
    lane = out / 'gpu0'
    lane.mkdir()
    (lane / 'jobs.json').write_text(json.dumps(jobs, indent=2))
    with (lane / 'watchdog.log').open('ab', buffering=0) as log:
        child = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.durable_experiment_queue',
            '--root', str(lane), '--gpu', '0', '--watchdog'], cwd=code,
            env=dict(os.environ, PYTHONPATH=str(code)), stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
    print('launched', child.pid, 'jobs', len(jobs), out)


if __name__ == '__main__':
    main()
