"""Delegate a not-started task pair without restarting the live GPU0 worker."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path('/mnt/pool/sqy/anygsloc_baseline_recovery_v4_20260915')
    source = root / 'gpu0'
    lane = root / 'gpu1'
    jobs = json.loads((source / 'jobs.json').read_text())
    state = json.loads((source / 'state.json').read_text())['jobs']
    keys = ['STDLoc_GreatCourt_mapping_semantics_on_train',
            'STDLoc_GreatCourt_mapping_semantics_on_eval']
    if any(k in state for k in keys):
        raise RuntimeError('Cannot delegate a task already touched by GPU0')
    selected = []
    delegation = {}
    for key in keys:
        job = dict(next(j for j in jobs if j['id'] == key))
        for dep in job.get('deps', []):
            if dep not in keys and state.get(dep, {}).get('status') != 'complete':
                raise RuntimeError('Prerequisite incomplete: ' + dep)
        job['deps'] = [dep for dep in job.get('deps', []) if dep in keys]
        job['env'] = dict(job['env'], ANYGSLOC_DELEGATE_EXECUTOR='1', CUDA_VISIBLE_DEVICES='1')
        receipt = job['command'][job['command'].index('--receipt') + 1]
        if Path(receipt).exists() or any(Path(p).exists() for p in job['outputs']):
            raise FileExistsError('Delegated outputs already exist')
        delegation[receipt] = dict(job=key, state=str(lane / 'state.json'), gpu=1)
        selected.append(job)
    lane.mkdir(exist_ok=False)
    (lane / 'jobs.json').write_text(json.dumps(selected, indent=2))
    # Atomic publication; the existing worker reads this when it reaches these commands.
    temporary = root / 'delegation.tmp'
    temporary.write_text(json.dumps(delegation, indent=2))
    temporary.replace(root / 'delegation.json')
    adapter = root / 'code/tools/baseline_protocol_v2.py'
    (root / 'parallel_handoff.json').write_text(json.dumps(dict(
        gpu0_uninterrupted=True, gpu1_jobs=keys,
        adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest(),
        timing='benchmark jobs remain on GPU0 after all training/evaluation pairs, including delegated completion'), indent=2))
    with (lane / 'watchdog.log').open('ab', buffering=0) as log:
        child = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.durable_experiment_queue',
            '--root', str(lane), '--gpu', '1', '--watchdog'], cwd=root / 'code',
            env=dict(os.environ, PYTHONPATH=str(root / 'code')), stdout=log,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    print('GPU1 watchdog', child.pid, 'jobs', keys)


if __name__ == '__main__':
    main()
