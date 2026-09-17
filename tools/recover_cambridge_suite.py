"""Recover failed v2 jobs into fresh outputs; keep successful evidence untouched."""
import json
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('/mnt/pool/sqy/anygsloc_cambridge_suite_v2_20260914'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--copy-baselines', action='store_true')
    args = parser.parse_args()
    old, out = args.source, args.output
    state = json.loads((old / 'gpu0/state.json').read_text())['jobs']
    source = json.loads((old / 'gpu0/jobs.json').read_text())
    out.mkdir(exist_ok=False)
    shutil.copytree(old / 'code', out / 'code', ignore=shutil.ignore_patterns('__pycache__'))
    if args.copy_baselines:
        shutil.copytree(old / 'baselines', out / 'baselines', ignore=shutil.ignore_patterns('__pycache__'))
    else:
        (out / 'baselines').symlink_to(old / 'baselines', target_is_directory=True)
    data = out / 'cambridge_data'
    data.mkdir()
    for scene in ['ShopFacade', 'KingsCollege', 'OldHospital', 'StMarysChurch', 'GreatCourt']:
        (data / scene).symlink_to(Path('/mnt/pool/sqy/Cambridge_stdloc') / scene, target_is_directory=True)
    jobs = []
    keys = {j['id'] for j in source if state.get(j['id'], {}).get('status') in ['failed', 'blocked']}
    def replace(value):
        return str(value).replace(str(old), str(out)).replace('/mnt/pool/sqy/Cambridge_stdloc/', str(data) + '/')
    for job in source:
        if job['id'] not in keys:
            continue
        job = dict(job)
        job['command'] = [replace(x) for x in job['command']]
        job['outputs'] = [replace(x) for x in job['outputs']]
        job['cwd'] = replace(job['cwd'])
        job['deps'] = [x for x in job.get('deps', []) if x in keys]
        # Run a complete representative build first, avoiding repeated startup failures.
        for method in ['STDLoc', 'ULFLoc']:
            representative = method + '_ShopFacade_mapping_semantics_off'
            if job['id'].startswith(method + '_') and job['id'].endswith('_train') and job['id'] != representative + '_train':
                job['deps'].append(representative + '_eval')
        if job['id'].startswith('conversion_') and job['id'] != 'conversion_ShopFacade':
            job['deps'].append('conversion_ShopFacade')
        binary = Path(job['command'][0]).parent
        job['env'] = dict(job.get('env', {}), PYTHONPATH=str(out / 'code'),
                          PATH=str(binary) + ':/usr/local/cuda/bin:' + os.environ['PATH'],
                          LD_LIBRARY_PATH=str(binary.parent / 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', ''),
                          CC='/usr/bin/gcc', CXX='/usr/bin/g++', MAX_JOBS='4', TORCH_CUDA_ARCH_LIST='8.6',
                          ULFLOC_FUSION_BATCH_SIZE='1')
        jobs.append(job)
    # Stable topological order ensures representative gates run before dependents.
    ordered = []
    done = set()
    while jobs:
        ready = next((j for j in jobs if set(j['deps']) <= done), None)
        if ready is None:
            raise ValueError('cyclic recovery dependencies')
        jobs.remove(ready)
        ordered.append(ready)
        done.add(ready['id'])
    lane = out / 'gpu0'
    lane.mkdir()
    (lane / 'jobs.json').write_text(json.dumps(ordered, indent=2))
    (out / 'recovery.json').write_text(json.dumps(dict(source=str(old), jobs=len(ordered),
        changes=['prepend interpreter bin to PATH for ninja', 'lowercase Cambridge dataset alias'],
        original_outputs_preserved=True), indent=2))
    if args.prepare_only:
        print('prepared', len(ordered), out)
        return
    with (lane / 'watchdog.log').open('ab', buffering=0) as log:
        p = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.durable_experiment_queue',
            '--root', str(lane), '--gpu', '0', '--watchdog'], cwd=out / 'code',
            env=dict(os.environ, PYTHONPATH=str(out / 'code')), stdout=log,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    print('launched', p.pid, 'jobs', len(ordered), out)


if __name__ == '__main__':
    main()
