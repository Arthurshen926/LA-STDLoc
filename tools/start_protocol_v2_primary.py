"""Launch fresh single-seed main-table evaluations under query protocol v2."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import hashlib

REPO = Path(__file__).resolve().parents[1]
OLD = Path('/mnt/pool/sqy/anygsloc_release_validation_20260913')
OUT = Path('/mnt/pool/sqy/anygsloc_protocol_v2_primary_20260914')


def main():
    OUT.mkdir(exist_ok=False)
    code=OUT/'code'
    shutil.copytree(REPO/'anygsloc', code/'anygsloc', ignore=shutil.ignore_patterns('__pycache__'))
    (code/'scripts').mkdir()
    shutil.copy2(REPO/'research/legacy/scripts/durable_experiment_queue.py',code/'scripts/durable_experiment_queue.py')
    shutil.copy2(REPO/'docs/release/semantic_mask_protocol_v2.md',OUT/'protocol.md')
    (OUT/'source_hashes.json').write_text(json.dumps({str(p.relative_to(code)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in code.rglob('*.py')},indent=2))
    jobs=[]
    cells=json.loads((OLD/'registry.json').read_text())['cells']
    for cell in cells:
        key=cell['key'];cambridge=key.startswith('Cambridge/')
        deployment=Path(cell['deployment_map'])
        if not deployment.is_file():raise FileNotFoundError(deployment)
        output=OUT/'runs'/key
        command=[sys.executable,'-m','anygsloc','evaluate','--dataset',cell['dataset'],
            '--map',str(deployment),'--output',str(output),'--image-domain','rectified',
            '--mask-policy','geometric_validity' if cambridge else 'none','--seed','2026','--device','cuda:0']
        if cambridge:command+=['--rectification-contract','opencv_same_k']
        jobs.append(dict(id=key.replace('/','_'),command=command,cwd=str(code),env=dict(PYTHONPATH=str(code)),
            outputs=[str(output/'summary.json'),str(output/'results.json')]))
    # One GPU avoids displacing unrelated jobs currently using the other GPUs.
    lane=OUT/'gpu0';lane.mkdir()
    (lane/'jobs.json').write_text(json.dumps(jobs,indent=2))
    (OUT/'registry.json').write_text(json.dumps(dict(seed=2026,cells=cells,
        scope='primary24_existing_priors_not_dual_build_complete',query_semantics=False),indent=2))
    with (lane/'watchdog.log').open('ab',buffering=0) as log:
        child=subprocess.Popen([sys.executable,'-u','-m','scripts.durable_experiment_queue','--root',str(lane),
            '--gpu','0','--watchdog'],cwd=code,env=dict(os.environ,PYTHONPATH=str(code)),
            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    print(json.dumps(dict(pid=child.pid,evaluations=len(jobs),root=str(OUT))))


if __name__=='__main__':main()
