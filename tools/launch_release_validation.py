"""Freeze executable sources and launch three durable lanes after kernel E0."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--e0-root', type=Path)
    p.add_argument('--after', type=Path, help='wait for all three predecessor lanes to finish')
    a = p.parse_args()
    root = a.root.resolve()
    if a.after:
        while True:
            ready = True
            for gpu in range(3):
                path = a.after/f'gpu{gpu}'/'state.json'
                ready = ready and path.exists() and json.loads(path.read_text()).get('queue_finished', False)
            if ready:
                break
            time.sleep(10)
    e0_root = a.e0_root or root
    for scene in ('ShopFacade', 'StMarysChurch'):
        evidence = json.loads((e0_root/'e0'/f'{scene}.json').read_text())
        if not evidence.get('complete') or not evidence.get('exact'):
            raise RuntimeError('E0 kernel gate failed')
    code = root/'code'
    if code.exists():
        raise FileExistsError('Launch already prepared; resume individual queue lanes explicitly')
    code.mkdir()
    source = e0_root/'code/anygsloc' if a.e0_root else ROOT/'anygsloc'
    shutil.copytree(source, code/'anygsloc', ignore=shutil.ignore_patterns('__pycache__'))
    scripts = code/'scripts'
    scripts.mkdir()
    shutil.copy2(ROOT/'research/legacy/scripts/durable_experiment_queue.py', scripts/'durable_experiment_queue.py')
    hashes = {str(f.relative_to(code)): hashlib.sha256(f.read_bytes()).hexdigest()
              for f in code.rglob('*.py')}
    (root/'execution_source_hashes.json').write_text(json.dumps(hashes, indent=2))
    environment = dict(python=sys.version, executable=sys.executable,
        pip_freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'], text=True),
        gpu=subprocess.check_output(['nvidia-smi'], text=True),
        git_branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True),
        source_tree_is_uncommitted=True)
    (root/'environment.json').write_text(json.dumps(environment, indent=2))
    workers = []
    for gpu in range(3):
        lane = root/f'gpu{gpu}'
        jobs = json.loads((lane/'jobs.json').read_text())
        for job in jobs:
            job['cwd'] = str(code)
            job['env'] = dict(PYTHONPATH=str(code))
        (lane/'jobs.json').write_text(json.dumps(jobs, indent=2))
        env = dict(os.environ, PYTHONPATH=str(code), PYTHONUNBUFFERED='1')
        with (lane/'watchdog.log').open('ab', buffering=0) as log:
            child = subprocess.Popen([sys.executable,'-u','-m','scripts.durable_experiment_queue',
                '--root',str(lane),'--gpu',str(gpu),'--watchdog'],cwd=code,env=env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append(dict(gpu=gpu,pid=child.pid))
    (root/'launch.json').write_text(json.dumps(workers, indent=2))
    print(json.dumps(workers))


if __name__ == '__main__':
    main()
