"""Frozen, resumable experiment DAG with one GPU lane and one CPU lane."""
from __future__ import annotations

import argparse
import csv
import fcntl
import importlib.metadata
import json
import os
import signal
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from supplement_experiments import SCENES, read, write, REPO
from anygsloc.artifacts import sha256


FILES = ['supplement_experiments.py', 'supplement_classic.py', 'supplement_geometry.py',
         'supplement_evaluate.py', 'run_supplement_campaign.py', 'figure_cache_reader.py',
         'gaussian_center_control.py', 'verify_supplement_campaign.py']


def prepare_jobs(root, gpu):
    protocol = read(root/'protocol.json')
    assert not protocol.get('smoke_only')
    code = root/'code'
    code.mkdir(exist_ok=False)
    shutil.copytree(REPO/'anygsloc', code/'anygsloc', ignore=shutil.ignore_patterns('__pycache__'))
    (code/'tools').mkdir()
    for name in FILES:
        shutil.copy2(REPO/'tools'/name, code/'tools'/name)
    from anygsloc.features.superpoint import resolve_superpoint_weights
    (code/'weights').mkdir()
    shutil.copy2(resolve_superpoint_weights(), code/'weights/superpoint_v1.pth')
    jobs = []
    def add(key, script, args, resource, output, deps=()):
        jobs.append(dict(id=key, command=[sys.executable, '-u', str(code/'tools'/script), *map(str, args)],
                         resource=resource, output=str(output), deps=list(deps), cwd=str(code)))
        return key
    def evaluate(key, scene, map_path, deps, budget='A', benchmark=False):
        output = root/('benchmarks' if benchmark else 'evaluations')/key
        args = ['--dataset', protocol['scenes'][scene]['dataset'], '--map', map_path,
                '--output', output, '--budget', budget]
        if benchmark:
            args += ['--warmup', '50', '--rounds', '3']
        return add(key, 'supplement_evaluate.py', args, 'gpu', output/'summary.json', deps)
    accuracy = []
    build_jobs = []
    benchmark_specs = []
    for scene in SCENES:
        s = protocol['scenes'][scene]
        common = ['--root', root, '--scene', scene]
        selection = add(scene+'_select', 'supplement_experiments.py', ['select', *common],
                        'cpu', root/'selector'/scene/'summary.json')
        capture = add(scene+'_capture', 'supplement_experiments.py', ['capture', *common],
                      'gpu', root/'ransac'/scene/'capture/summary.json')
        replay = add(scene+'_replay', 'supplement_experiments.py', ['replay', *common],
                     'cpu', root/'ransac'/scene/'replay/summary.json', [capture])
        accuracy += [capture, replay]
        for method in ['quality', 'spatial']:
            path = root/'selector'/scene/method/'deployment_map.pt'
            key = scene+'_selector_'+method
            accuracy.append(evaluate(key, scene, path, [selection]))
            benchmark_specs.append((key, scene, path, 'A'))
        render = add(scene+'_classic_render', 'supplement_classic.py', ['render', *common],
                     'gpu', root/'classic'/scene/'render/summary.json')
        match = add(scene+'_classic_match', 'supplement_classic.py', ['match', *common],
                    'gpu', root/'classic'/scene/'match/summary.json', [render])
        fit = add(scene+'_classic_fit', 'supplement_classic.py', ['fit', *common, '--workers', '4'],
                  'cpu', root/'classic'/scene/'fit/summary.json', [match])
        build_jobs += [selection, render, match, fit]
        for method in ['spatial', 'ours_spatial']:
            path = root/'classic'/scene/'fit'/method/'deployment_map.pt'
            key = scene+'_classic_'+method
            accuracy.append(evaluate(key, scene, path, [fit]))
            benchmark_specs.append((key, scene, path, 'A'))
        geometry = add(scene+'_geometry_build', 'supplement_geometry.py', common,
                       'cpu', root/'geometry'/scene/'summary.json')
        build_jobs.append(geometry)
        for method in ['ray', 'depth', 'center']:
            key = scene+'_geometry_'+method
            path = root/'geometry'/scene/method/'deployment_map.pt'
            accuracy.append(evaluate(key, scene, path, [geometry]))
        key = scene+'_native_D'
        accuracy.append(evaluate(key, scene, Path(s['compact_map']), [replay], budget='D'))
        benchmark_specs += [(scene+'_native_A', scene, Path(s['compact_map']), 'A'),
                            (key, scene, Path(s['compact_map']), 'D')]
    # Benchmark starts only after all construction/accuracy jobs; no CPU lane competition.
    prerequisites = accuracy+build_jobs
    check = add('verify_controls', 'verify_supplement_campaign.py', ['--root', root],
                'cpu', root/'verification.json', prerequisites)
    for key, scene, path, budget in benchmark_specs:
        evaluate('timing_'+key, scene, path, [check], budget=budget, benchmark=True)
    execution = root/'execution'
    execution.mkdir(exist_ok=False)
    write(execution/'jobs.json', jobs)
    write(execution/'frozen.json', dict(protocol_sha256=sha256(root/'protocol.json'),
        source_hashes={str(p.relative_to(code)): sha256(p) for p in sorted(code.rglob('*')) if p.is_file()},
        jobs_sha256=sha256(execution/'jobs.json'), gpu=gpu, python=sys.executable,
        package_versions={name: importlib.metadata.version(name)
                          for name in ['torch', 'numpy', 'scipy', 'gsplat', 'poselib']},
        gpu_lane='one active task', cpu_lane='one active task, up to four triangulation workers',
        benchmark_policy='no campaign CPU work during timings; external GPU/host activity disclosed'))
    print('Registered', len(jobs), 'tasks; GPU', gpu, flush=True)


def collect(root):
    rows = []
    protocol = read(root/'protocol.json')
    for path in sorted((root/'ransac').glob('*/capture/summary.json')):
        value = read(path)
        scene = path.parents[1].name
        rows.append(dict(condition=scene+'_selector_full', stage='accuracy', queries=value['query_count'],
            TE_cm=value['median_te_cm'], RE_deg=value['median_ae_deg'], R5_pct=value['recall_5cm_5deg_percent'],
            R15_pct=value['recall_15cm_5deg_percent'], P90_TE_cm=value['p90_te_cm'],
            failures=value['failed_query_count'], map_bytes=Path(protocol['scenes'][scene]['compact_map']).stat().st_size,
            mean_ms=value['runtime_ms_mean'], p95_ms=value['runtime_ms_p95'], source=str(path)))
    for path in sorted((root/'evaluations').glob('*/summary.json')):
        value = read(path)
        rows.append(dict(condition=path.parent.name, stage='accuracy', queries=value['query_count'],
            TE_cm=value['median_te_cm'], RE_deg=value['median_ae_deg'], R5_pct=value['recall_5cm_5deg_percent'],
            R15_pct=value['recall_15cm_5deg_percent'], P90_TE_cm=value['p90_te_cm'],
            failures=value['failed_query_count'], map_bytes=value['map_bytes'],
            mean_ms=value['runtime_ms_mean'], p95_ms=value['runtime_ms_p95'], source=str(path)))
    for path in sorted((root/'ransac').glob('*/replay/*/summary.json')):
        value = read(path)
        rows.append(dict(condition=path.parents[2].name+'_ransac_'+path.parent.name, stage='solver_replay',
            queries=value['query_count'], TE_cm=value['median_te_cm'], RE_deg=value['median_ae_deg'],
            R5_pct=value['recall_5cm_5deg_percent'], R15_pct=value['recall_15cm_5deg_percent'],
            P90_TE_cm=value['p90_te_cm'], failures=value['failed_query_count'], map_bytes=None,
            mean_ms=value['runtime_ms_mean'], p95_ms=value['runtime_ms_p95'], source=str(path)))
    for path in sorted((root/'benchmarks').glob('*/summary.json')):
        value = read(path)
        rounds = value['rounds']
        rows.append(dict(condition=path.parent.name, stage='warmed_timing', queries=value['query_count'],
            TE_cm=value['median_te_cm'], RE_deg=value['median_ae_deg'], R5_pct=value['recall_5cm_5deg_percent'],
            R15_pct=value['recall_15cm_5deg_percent'], P90_TE_cm=value['p90_te_cm'],
            failures=value['failed_query_count'], map_bytes=value['map_bytes'],
            mean_ms=statistics.mean(r['runtime_ms_mean'] for r in rounds),
            p95_ms=statistics.mean(r['runtime_ms_p95'] for r in rounds), source=str(path)))
    if rows:
        with (root/'results.csv.tmp').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        (root/'results.csv.tmp').replace(root/'results.csv')
        grouped = {}
        for row in rows:
            condition = row['condition'].removeprefix('timing_')
            scene = next((s for s in SCENES if condition.startswith(s+'_')), None)
            if scene:
                method = condition[len(scene)+1:]
                grouped.setdefault((row['stage'], method), {})[scene] = row
        macros = []
        for (stage, method), values in sorted(grouped.items()):
            if set(values) != set(SCENES):
                continue
            metrics = ['TE_cm', 'RE_deg', 'R5_pct', 'R15_pct', 'P90_TE_cm', 'mean_ms', 'p95_ms']
            macros.append(dict(stage=stage, method=method, scenes=5,
                **{key: (statistics.mean(v[key] for v in values.values())
                         if all(v[key] is not None for v in values.values()) else None) for key in metrics}))
        write(root/'macro_results.json', dict(complete_scene_groups_only=True, results=macros,
            caution='Accuracy-run timing and solver-only replay are not formal end-to-end benchmarks.'))
    jobs = read(root/'execution/jobs.json')
    state = read(root/'execution/state.json') if (root/'execution/state.json').exists() else {'jobs': {}}
    statuses = [state['jobs'].get(j['id'], {}).get('status', 'pending') for j in jobs]
    write(root/'status.json', dict(registered=len(jobs), counts={s: statuses.count(s) for s in set(statuses)},
        all_complete=all(s == 'complete' for s in statuses), result_rows=len(rows),
        active=[j['id'] for j in jobs if state['jobs'].get(j['id'], {}).get('status') == 'running']))


def gpu_available(gpu):
    result = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True)
    return result.returncode == 0 and not result.stdout.strip()


def worker(root):
    def interrupted(signum, frame):
        raise KeyboardInterrupt('worker interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    execution = root/'execution'
    lock = (execution/'worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    freeze = read(execution/'frozen.json')
    assert sha256(root/'protocol.json') == freeze['protocol_sha256']
    assert sha256(execution/'jobs.json') == freeze['jobs_sha256']
    for rel, digest in freeze['source_hashes'].items():
        assert sha256(root/'code'/rel) == digest, rel
    jobs = read(execution/'jobs.json')
    state = read(execution/'state.json') if (execution/'state.json').exists() else dict(jobs={})
    if any(v.get('status') == 'running' for v in state['jobs'].values()):
        raise RuntimeError('Interrupted worker requires explicit inspection; refusing to duplicate running jobs')
    active = {}
    env = dict(os.environ, PYTHONPATH=str(root/'code')+':'+str(root/'code/tools'),
               PATH=str(Path(sys.executable).parent)+':/usr/local/cuda-11.8/bin:'+os.environ.get('PATH', ''),
               CUDA_HOME='/usr/local/cuda-11.8',
               LAFGS_SUPERPOINT_WEIGHTS=str(root/'code/weights/superpoint_v1.pth'),
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
               LD_LIBRARY_PATH=str(Path(sys.executable).parent.parent/'lib')+':'+os.environ.get('LD_LIBRARY_PATH', ''),
               TMPDIR=str(root/'temporary'))
    (root/'temporary').mkdir(exist_ok=True)
    (execution/'logs').mkdir(exist_ok=True)
    try:
        while True:
            for resource, (process, job, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                output = Path(job['output'])
                success = code == 0 and output.is_file()
                state['jobs'][job['id']].update(status='complete' if success else 'failed',
                    finished=time.time(), exit_code=code, output_sha256=sha256(output) if output.is_file() else None)
                del active[resource]
                print(job['id'], state['jobs'][job['id']]['status'], flush=True)
            for job in jobs:
                if state['jobs'].get(job['id'], {}).get('status') in {'running', 'complete', 'failed', 'blocked'}:
                    continue
                dependencies = [state['jobs'].get(d, {}).get('status', 'pending') for d in job['deps']]
                if any(d in {'failed', 'blocked'} for d in dependencies):
                    state['jobs'][job['id']] = dict(status='blocked', reason='dependency failed')
                    continue
                if not all(d == 'complete' for d in dependencies) or job['resource'] in active:
                    continue
                if job['resource'] == 'gpu' and not gpu_available(freeze['gpu']):
                    continue
                log = (execution/'logs'/(job['id']+'.log')).open('ab', buffering=0)
                child_env = dict(env, CUDA_VISIBLE_DEVICES=str(freeze['gpu']) if job['resource'] == 'gpu' else '')
                process = subprocess.Popen(job['command'], cwd=job['cwd'], env=child_env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state['jobs'][job['id']] = dict(status='running', started=time.time(), pid=process.pid,
                                              resource=job['resource'])
                active[job['resource']] = (process, job, log)
                print('Started', job['id'], 'pid', process.pid, flush=True)
            state['updated'] = time.time()
            write(execution/'state.json', state)
            collect(root)
            statuses = [state['jobs'].get(j['id'], {}).get('status', 'pending') for j in jobs]
            if not active and all(s in {'complete', 'failed', 'blocked'} for s in statuses):
                break
            time.sleep(5)
    finally:
        # Include triangulation subprocesses, without touching unrelated processes.
        for process, job, log in active.values():
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
            log.close()
            state['jobs'][job['id']].update(status='interrupted', finished=time.time())
        write(execution/'state.json', state)
        collect(root)


def launch(root):
    execution = root/'execution'
    if (execution/'launch.json').exists():
        raise FileExistsError('campaign already launched')
    script = root/'code/tools/run_supplement_campaign.py'
    env = dict(os.environ, PYTHONPATH=str(root/'code')+':'+str(root/'code/tools'))
    with (execution/'worker.log').open('ab', buffering=0) as log:
        child = subprocess.Popen([sys.executable, '-u', str(script), 'worker', '--root', str(root)],
            cwd=root/'code', env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    write(execution/'launch.json', dict(pid=child.pid, started=time.time(), script=str(script)))
    print('Launched durable worker', child.pid, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'launch', 'worker', 'status'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=2)
    a = p.parse_args()
    {'prepare': lambda: prepare_jobs(a.root, a.gpu), 'launch': lambda: launch(a.root),
     'worker': lambda: worker(a.root), 'status': lambda: collect(a.root)}[a.action]()
