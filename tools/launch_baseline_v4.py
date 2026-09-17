"""Finalize audited v4 compatibility recovery and resume preserved ULF assets."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    root = Path('/mnt/pool/sqy/anygsloc_baseline_recovery_v4_20260915')
    lane = root / 'gpu0'
    if (lane / 'state.json').exists():
        raise FileExistsError('Queue already launched')
    jobs = json.loads((lane / 'jobs.json').read_text())
    key = 'ULFLoc_ShopFacade_mapping_semantics_off'
    old = Path('/mnt/pool/sqy/anygsloc_cambridge_recovery_v3_20260915/baseline_models') / key
    ply = old / 'point_cloud/iteration_30000/point_cloud.ply'
    indices = old / 'test/keypoints_sampled_idx.pkl'
    assert ply.is_file() and indices.is_file()
    target = root / 'baseline_models' / key / 'point_cloud/iteration_30000/point_cloud.ply'
    target.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(ply, target)
    job = next(j for j in jobs if j['id'] == key + '_train')
    job['env'].update(ULFLOC_POSTPROCESS_PLY=str(target), ULFLOC_POSTPROCESS_INDICES=str(indices))
    (lane / 'jobs.json').write_text(json.dumps(jobs, indent=2))
    paths = [root / 'baselines/STDLoc/gaussian_renderer/__init__.py',
             root / 'baselines/ULFLoc/train.py', root / 'code/tools/baseline_protocol_v2.py', ply, indices]
    hashes = {}
    for path in paths:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 << 20), b''):
                h.update(block)
        hashes[str(path)] = h.hexdigest()
    (root / 'compatibility_recovery.json').write_text(json.dumps(dict(
        hashes=hashes, stdloc='scalar radius=max(x_radius,y_radius)',
        ulfloc='fusion_batch_size=1; unchanged image resolution and interpolation',
        reused_gaussian_and_indices=True,
        timing_caveat='ShopFacade ULF train receipt measures only recovery; do not report as complete construction cost'), indent=2))
    with (lane / 'watchdog.log').open('ab', buffering=0) as log:
        child = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.durable_experiment_queue',
            '--root', str(lane), '--gpu', '0', '--watchdog'], cwd=root / 'code',
            env=dict(os.environ, PYTHONPATH=str(root / 'code')), stdout=log,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    print('launched', child.pid, len(jobs), 'tasks')


if __name__ == '__main__':
    main()
