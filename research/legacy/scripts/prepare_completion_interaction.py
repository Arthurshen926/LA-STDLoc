"""Register completion ablations around a fixed Compact+Fast baseline."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import torch
from scripts.materialize_purity_candidates import subset


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=False)
    src=Path(__file__).resolve().parents[1];code=a.root/'code'
    for directory in ['common','data','evidence','features','localization','evaluation','priors','map_learning','topology','scripts','configs']:
        shutil.copytree(src/directory,code/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    hashes={str(x.relative_to(code)):hashlib.sha256(x.read_bytes()).hexdigest() for x in code.rglob('*') if x.is_file()}
    old=Path('/mnt/pool/sqy/anygsloc_full_selector_validation_20260911')
    reports={}
    for gpu,scene in enumerate(['OldHospital','ShopFacade']):
        lane=a.root/f'gpu{gpu}';lane.mkdir();jobs=[]
        refs=json.loads((old/scene/'jobs.json').read_text())
        ref=next(j for j in refs if j['id']=='full_selector_seed2026')
        cmd=ref['command'];map_path=Path(cmd[cmd.index('--map')+1]);metric_path=Path(cmd[cmd.index('--metric-state')+1])
        state=torch.load(map_path,map_location='cpu',weights_only=False)
        metric=torch.load(metric_path,map_location='cpu',weights_only=False)
        rows=torch.tensor([i for i,k in enumerate(state['anchor_candidate_kind']) if k!='depth_proposed_projective_completion'])
        revised=subset(state,rows);revised['provenance']=dict(state['provenance'],postselection_completion_removed=True)
        out=a.root/scene/'post_remove';out.mkdir(parents=True)
        path=out/'projective_anchor_map.pt';torch.save(revised,path)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        torch.save(dict(metric,landmark_indices=revised['anchor_ids'],map_path=str(path),map_sha256=digest),out/'identity_metric.pt')
        reports[scene]=dict(compact_count=len(state['anchor_ids']),retained_count=len(rows),removed_count=len(state['anchor_ids'])-len(rows))
        env=dict(PYTHONPATH=str(code),CUDA_VISIBLE_DEVICES=str(gpu),LD_LIBRARY_PATH='/root/miniconda3/envs/g4splat/lib')
        for seed in [2026,2027,2028]:
            for variant,mp,mt in [('compact_reference',map_path,metric_path),('post_remove',path,out/'identity_metric.pt')]:
                command=list(cmd);result=a.root/scene/'runs'/f'{variant}_seed{seed}'
                for flag,value in [('--map',mp),('--metric-state',mt),('--output',result),('--seed',seed)]:command[command.index(flag)+1]=str(value)
                jobs.append(dict(id=f'{variant}_{seed}',command=command,cwd=ref['cwd'],env=dict(ref['env'],CUDA_VISIBLE_DEVICES=str(gpu)),deps=[],outputs=[str(result/'summary.json'),str(result/'results.json')]))
        rebuilt=a.root/scene/'reselect'
        jobs.append(dict(id='reselect_build',command=[cmd[0],'-u','-m','scripts.run_full_selector_pilot','--root',str(rebuilt),'--scene',scene,'--seeds','2026','2027','2028','--exclude-candidate-completion'],cwd=str(code),env=env,deps=[],outputs=[str(rebuilt/'protocol.json'),str(rebuilt/'jobs.json')]))
        for seed in [2026,2027,2028]:
            command=list(cmd);result=a.root/scene/'runs'/f'reselect_seed{seed}';folder=rebuilt/'maps/full_selector'
            for flag,value in [('--map',folder/'projective_anchor_map.pt'),('--metric-state',folder/'identity_metric.pt'),('--output',result),('--seed',seed)]:command[command.index(flag)+1]=str(value)
            jobs.append(dict(id=f'reselect_{seed}',command=command,cwd=ref['cwd'],env=dict(ref['env'],CUDA_VISIBLE_DEVICES=str(gpu)),deps=['reselect_build'],outputs=[str(result/'summary.json'),str(result/'results.json')]))
        (lane/'jobs.json').write_text(json.dumps(jobs,indent=2))
    (a.root/'protocol.json').write_text(json.dumps(dict(code_sha256=hashes,counts=reports,seeds=[2026,2027,2028],
        total_evaluations=18,online='frozen Fast identical across arms',offline_feedback=False,
        main_ablation='Exclude candidate completion before recomputing all selector stages',
        diagnostic='Remove completion after selection, without reselection',
        note='Original F0 base tracks unchanged; no-completion construction reused by exact candidate exclusion'),indent=2))
    for x in code.rglob('*'):
        if x.is_file():x.chmod(0o444)
    print(reports,flush=True)


if __name__=='__main__':main()
