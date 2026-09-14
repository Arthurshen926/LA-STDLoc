"""Queue geometry-source pairs after the existing GPU lanes release their devices."""
import argparse,json,shutil
from pathlib import Path
from scripts.closeout_remaining_ablations import POOL,PYTHON


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--preceding',type=Path,required=True);a=p.parse_args()
    a.root.mkdir(exist_ok=False);code=a.root/'code'
    shutil.copytree(a.preceding/'code',code)
    shutil.copyfile(Path(__file__).with_name('materialize_depth_ablation.py'),code/'scripts/materialize_depth_ablation.py')
    registry=json.loads((a.preceding/'registry.json').read_text());new=[]
    for gpu,scenes in enumerate([['GreatCourt'],['StMarysChurch'],['ShopFacade','OldHospital','KingsCollege']]):
        lane=a.root/f'gpu{gpu}';lane.mkdir();jobs=[]
        env=dict(PYTHONPATH=str(code),CUDA_VISIBLE_DEVICES=str(gpu),PATH='/root/miniconda3/envs/g4splat/bin:/usr/local/cuda/bin:/usr/bin:/bin',LD_LIBRARY_PATH='/root/miniconda3/envs/g4splat/lib')
        def add(key,cmd,outputs,deps=()):jobs.append(dict(id=key,command=list(map(str,cmd)),outputs=list(map(str,outputs)),deps=list(deps),cwd=str(code),env=env))
        add('wait',[PYTHON,'-m','scripts.wait_queue_lane_finished','--state',a.preceding/f'gpu{gpu}/state.json'],[])
        for scene in scenes:
            ref=next(r for r in registry if r['scene']==scene and r['condition']=='descriptor_reference' and r['seed']==2026)
            folder=Path(ref['map']);source=POOL/('anygsloc_cache_recovery_v2_20260912' if scene in ['GreatCourt','StMarysChurch'] else 'lafgs_v24_cambridge_high_capacity_online_20260901')/scene
            out=a.root/'maps'/scene;key=scene+'_build'
            add(key,[PYTHON,'-u','-m','scripts.materialize_depth_ablation','--map-dir',folder,'--cache',source/'observations/rendered_rgb_feature_cache.pt','--output',out],[out/'report.json'],['wait'])
            original=next(j for j in json.loads((a.preceding/'gpu2/jobs.json').read_text()) if j['id']==scene+'_descriptor_reference_2026')
            for kind in ['ray_common','depth_common']:
                for seed in [2026,2027,2028]:
                    result=a.root/'runs'/scene/f'{kind}_seed{seed}';cmd=list(original['command'])
                    for flag,val in [('--map',out/kind/'projective_anchor_map.pt'),('--metric-state',out/kind/'identity_metric.pt'),('--output',result),('--config',code/'configs/anygsloc_mainline.yaml'),('--seed',seed)]:cmd[cmd.index(flag)+1]=str(val)
                    add(f'{scene}_{kind}_{seed}',cmd,[result/'summary.json',result/'results.json'],[key]);new.append(dict(scene=scene,condition=kind,seed=seed,output=str(result)))
        (lane/'jobs.json').write_text(json.dumps(jobs,indent=2))
    (a.root/'registry.json').write_text(json.dumps(new,indent=2))


if __name__=='__main__':main()
