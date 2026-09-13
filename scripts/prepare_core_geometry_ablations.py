"""Freeze and queue component rebuilds followed by unchanged Compact+Fast."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=False)
    src=Path(__file__).resolve().parents[1];code=a.root/'code'
    for d in ['common','data','evidence','features','localization','evaluation','priors','map_learning','topology','scripts','configs']:
        shutil.copytree(src/d,code/d,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    hashes={str(x.relative_to(code)):hashlib.sha256(x.read_bytes()).hexdigest() for x in code.rglob('*') if x.is_file()}
    old=Path('/mnt/pool/sqy/anygsloc_full_selector_validation_20260911')
    variants={'rebuild_reference':[], 'no_v2':['--disable-v2-row-filter'],
              'no_cycle':['--disable-cycle-consistency'],
              'no_robust_ray':['--disable-robust-ray-reweighting']}
    for gpu,scene in enumerate(['ShopFacade','OldHospital','KingsCollege']):
        lane=a.root/f'gpu{gpu}';lane.mkdir();jobs=[]
        source=Path('/mnt/pool/sqy/lafgs_v24_cambridge_high_capacity_online_20260901')/scene
        cache=source/'observations/rendered_rgb_feature_cache.pt'
        shards=sorted((source/'v2_audit').glob('*.pt'))
        if not cache.is_file() or not shards:raise FileNotFoundError(f'{scene}: missing cache/audits')
        refs=json.loads((old/scene/'jobs.json').read_text());ref=next(j for j in refs if j['id']=='full_selector_seed2026')
        python=ref['command'][0]
        env=dict(PYTHONPATH=str(code),CUDA_VISIBLE_DEVICES=str(gpu),LD_LIBRARY_PATH='/root/miniconda3/envs/g4splat/lib')
        def add(key,command,outputs,deps=()):
            jobs.append(dict(id=key,command=list(map(str,command)),outputs=list(map(str,outputs)),deps=list(deps),cwd=str(code),env=env))
        for variant,flags in variants.items():
            base=a.root/scene/variant;mp=base/'projective_map';compact=base/'compact'
            add(variant+'_map',[python,'-u','-m','scripts.materialize_v8_v2_projective_map','--observation-cache',cache,'--v2-audit-shards',*shards,'--output-dir',mp,'--device','cuda:0','--triangulation-workers','4',*flags],[mp/'projective_anchor_map.pt',mp/'identity_metric.pt'])
            add(variant+'_select',[python,'-u','-m','scripts.run_full_selector_pilot','--scene',scene,'--root',compact,'--input-map-dir',mp,'--seeds','2026','2027','2028'],[compact/'protocol.json',compact/'maps/full_selector/projective_anchor_map.pt'],[variant+'_map'])
            for seed in [2026,2027,2028]:
                command=list(ref['command']);out=base/'evaluation'/f'seed{seed}';folder=compact/'maps/full_selector'
                for flag,val in [('--map',folder/'projective_anchor_map.pt'),('--metric-state',folder/'identity_metric.pt'),('--output',out),('--seed',seed)]:command[command.index(flag)+1]=str(val)
                add(variant+f'_eval{seed}',command,[out/'summary.json',out/'results.json'],[variant+'_select'])
        (lane/'jobs.json').write_text(json.dumps(jobs,indent=2))
    (a.root/'protocol.json').write_text(json.dumps(dict(code_sha256=hashes,variants=variants,seeds=[2026,2027,2028],total_evaluations=36,
        frozen_online='Compact+Fast; one solve, no offline feedback',
        no_robust_ray='Remove iterative ray residual reweighting only; retain confidence weighting and all acceptance thresholds',
        no_cycle='Remove cycle-based track constraint/graded chain routing; retain reciprocal and epipolar pair tests',
        limitations=['Three cache-available Cambridge scenes only','Depth geometry substitution and V2-only validity floor not covered','Descriptor ablations tracked separately'],
        integrity='Full rebuilt reference detects drift; no silent promotion'),indent=2))
    for x in code.rglob('*'):
        if x.is_file():x.chmod(0o444)
    print('36 evaluations + 12 map rebuilds + 12 selector builds registered',flush=True)


if __name__=='__main__':main()
