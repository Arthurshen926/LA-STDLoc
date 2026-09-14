"""Single-seed, same-map Cambridge comparison without any query mask loading."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/mnt/pool/sqy/anygsloc_release_validation_20260913')
OUT = Path('/mnt/pool/sqy/anygsloc_cambridge_no_query_mask_20260913')


def main():
    OUT.mkdir(exist_ok=False)
    code = OUT/'code'
    shutil.copytree(ROOT/'anygsloc', code/'anygsloc', ignore=shutil.ignore_patterns('__pycache__'))
    (code/'scripts').mkdir()
    shutil.copy2(ROOT/'research/legacy/scripts/durable_experiment_queue.py', code/'scripts/durable_experiment_queue.py')
    hashes = {str(p.relative_to(code)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code.rglob('*.py')}
    (OUT/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    jobs=[]
    # Small scenes first give an early sanity check; every scene is evaluated.
    for scene in ['ShopFacade','KingsCollege','OldHospital','StMarysChurch','GreatCourt']:
        original=BASE/'runs/Cambridge'/scene/'2026/manifest.json'
        manifest=json.loads(original.read_text())
        deployment=BASE/'maps/Cambridge'/scene/'deployment_map.pt'
        if hashlib.sha256(deployment.read_bytes()).hexdigest()!=manifest['map_sha256']:
            raise ValueError('comparison map changed')
        output=OUT/'runs'/scene
        command=[sys.executable,'-m','anygsloc','evaluate','--dataset',manifest['dataset'],
                 '--images',manifest['images'],'--map',str(deployment),'--output',str(output),
                 '--image-domain','rectified','--mask-policy','none','--device','cuda:0','--seed','2026']
        jobs.append(dict(id=scene,command=command,cwd=str(code),env=dict(PYTHONPATH=str(code)),
                         outputs=[str(output/'summary.json'),str(output/'results.json')]))
    lane=OUT/'gpu0';lane.mkdir()
    (lane/'jobs.json').write_text(json.dumps(jobs,indent=2))
    (OUT/'protocol.json').write_text(json.dumps(dict(seed=2026,query_masks='none_not_loaded',
        maps='identical_to_masked_baseline',prior_reconstruction='unchanged',
        distortion_border_mask='also_disabled',timing='not_isolated_formal_benchmark'),indent=2))
    with (lane/'watchdog.log').open('ab',buffering=0) as log:
        child=subprocess.Popen([sys.executable,'-u','-m','scripts.durable_experiment_queue',
            '--root',str(lane),'--gpu','0','--watchdog'],cwd=code,
            env=dict(os.environ,PYTHONPATH=str(code)),stdout=log,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True)
    print(json.dumps(dict(pid=child.pid,root=str(OUT),evaluations=len(jobs))))


if __name__=='__main__':
    main()
