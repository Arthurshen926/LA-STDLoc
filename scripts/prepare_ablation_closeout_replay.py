"""Uniform rerun of registered core/selection tables; no test-based variant selection."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=False)
    geometry=Path('/mnt/pool/sqy/anygsloc_core_geometry_20260911')
    selector=Path('/mnt/pool/sqy/anygsloc_full_selector_validation_20260911')
    code=geometry/'code';python='/root/miniconda3/envs/g4splat/bin/python'
    registry=[]
    for gpu,scene in enumerate(['ShopFacade','OldHospital','KingsCollege']):
        lane=a.root/f'gpu{gpu}';lane.mkdir()
        env=dict(PYTHONPATH=str(code),CUDA_VISIBLE_DEVICES=str(gpu),LD_LIBRARY_PATH='/root/miniconda3/envs/g4splat/lib')
        jobs=[dict(id='wait_construction',command=[python,'-m','scripts.wait_queue_lane_finished','--state',str(geometry/f'gpu{gpu}/state.json')],outputs=[],deps=[],cwd=str(code),env=env)]
        groups={
            'geometry':[j for j in json.loads((geometry/f'gpu{gpu}/jobs.json').read_text()) if '_eval' in j['id']],
            'selection':json.loads((selector/scene/'jobs.json').read_text())}
        for group,items in groups.items():
            for old in items:
                command=list(old['command']);name=group+'_'+old['id'];out=a.root/'runs'/scene/name
                command[command.index('--output')+1]=str(out)
                command[command.index('--config')+1]=str(code/'configs/anygsloc_mainline.yaml')
                jobs.append(dict(id=name,command=command,outputs=[str(out/'summary.json'),str(out/'results.json')],deps=['wait_construction'],cwd=str(code),env=env))
                registry.append(dict(scene=scene,group=group,condition=old['id'],output=str(out),
                    map=command[command.index('--map')+1],seed=int(command[command.index('--seed')+1])))
        (lane/'jobs.json').write_text(json.dumps(jobs,indent=2))
    assert len(registry)==81
    (a.root/'registry.json').write_text(json.dumps(registry,indent=2))
    print(len(registry),'evaluations registered')


if __name__=='__main__':main()
