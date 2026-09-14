"""Durable staged full-selector validation, without changing frozen selection policy."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run-child',action='store_true')
    a=p.parse_args()
    if a.run_child:
        subprocess.run([sys.executable,'-m','scripts.durable_experiment_queue','--root',str(a.root),'--gpu','0'],check=True)
        state=json.loads((a.root/'state.json').read_text())
        if not state.get('all_complete'):raise RuntimeError('Child evaluation incomplete; inspect state')
        return
    a.root.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().parents[1];code=a.root/'code'
    for directory in ['common','data','evidence','features','localization','evaluation','priors','map_learning','topology','scripts','configs']:
        shutil.copytree(source/directory,code/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    hashes={}
    for path in code.rglob('*'):
        if path.is_file():
            hashes[str(path.relative_to(code))]=hashlib.sha256(path.read_bytes()).hexdigest()
            path.chmod(0o444)
    env=dict(PYTHONPATH=str(code),CUDA_VISIBLE_DEVICES='0',LD_LIBRARY_PATH='/root/miniconda3/envs/g4splat/lib')
    python='/root/miniconda3/envs/g4splat/bin/python'
    jobs=[]
    shop=a.root/'ShopFacade';shop.mkdir()
    old=Path('/mnt/pool/sqy/anygsloc_full_selector_shop_20260911')
    expanded=[]
    for seed in [2026,2027,2028]:
        for job in json.loads((old/'jobs.json').read_text()):
            command=list(job['command']);result=shop/'runs'/f"{job['id']}_seed{seed}"
            command[command.index('--seed')+1]=str(seed)
            command[command.index('--output')+1]=str(result)
            expanded.append(dict(job,id=f"{job['id']}_seed{seed}",command=command,outputs=[str(result/'summary.json'),str(result/'results.json')]))
    (shop/'jobs.json').write_text(json.dumps(expanded,indent=2))
    for scene in ['ShopFacade','OldHospital','KingsCollege']:
        output=a.root/scene;deps=[]
        if scene!='ShopFacade':
            deps=[scene+'_build']
            jobs.append(dict(id=deps[0],command=[python,'-u','-m','scripts.run_full_selector_pilot','--root',str(output),'--scene',scene,'--seeds','2026','2027','2028'],env=env,cwd=str(code),deps=[],outputs=[str(output/'jobs.json'),str(output/'protocol.json')]))
        jobs.append(dict(id=scene+'_evaluate',command=[python,'-u','-m','scripts.prepare_full_selector_validation','--root',str(output),'--run-child'],env=env,cwd=str(code),deps=deps,outputs=[str(output/'state.json')]))
    (a.root/'jobs.json').write_text(json.dumps(jobs,indent=2))
    (a.root/'protocol.json').write_text(json.dumps(dict(code_sha256=hashes,
        scenes=['ShopFacade','OldHospital','KingsCollege'],seeds=[2026,2027,2028],
        variants=['core','matching','full_selector','equal_size_quality','full_map'],
        total_evaluations=45,selection_unchanged=True,offline_feedback=False,
        pending_scenes_reason='GreatCourt/StMarysChurch raw mapping cache unavailable; not silently reconstructed',
        promotion='No automatic promotion. Report per-scene and all seeds, equal-size control, latency and failures.'),indent=2))
    print(a.root,'45 evaluations registered',flush=True)


if __name__=='__main__':main()
