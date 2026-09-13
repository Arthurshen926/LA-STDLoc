"""Freeze local implementation and prepare safe single-GPU queue handoff."""
import argparse
import json
from pathlib import Path
import shutil
from common.hashing import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--office-pid',type=int,required=True);a=p.parse_args()
    if a.root.exists():raise FileExistsError(a.root)
    source=Path(__file__).resolve().parents[1];snap=a.root/'code';snap.mkdir(parents=True)
    for directory in ['common','data','evidence','features','localization','evaluation','priors','map_learning','topology','scripts','configs']:
        shutil.copytree(source/directory,snap/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    manifest={str(p.relative_to(snap)):sha256_file(p) for p in snap.rglob('*') if p.is_file()}
    (a.root/'code_manifest.json').write_text(json.dumps(manifest,indent=2))
    for path in snap.rglob('*'):
        if path.is_file():path.chmod(0o444)
    old=Path('/mnt/pool/sqy/anygsloc_execution_queue_20260909')
    jobs=json.loads((old/'jobs.json').read_text());byid={j['id']:j for j in jobs}
    def convert(j):
        # Only rewrite local code and new output roots. External priors/data stay fixed.
        text=json.dumps(j).replace(str(source),str(snap)).replace(str(old),str(a.root))
        return json.loads(text)
    def add(key,cmd,deps=(),outputs=()):
        return {'id':key,'command':list(map(str,cmd)),'deps':list(deps),'outputs':list(map(str,outputs)),
                'cwd':str(snap),'env':{'PYTHONPATH':str(snap),'LD_LIBRARY_PATH':'/root/miniconda3/envs/g4splat/lib',
                'PATH':'/root/miniconda3/envs/g4splat/bin:/usr/local/cuda/bin:/usr/bin:/bin'}}
    py='/root/miniconda3/envs/g4splat/bin/python'
    first=convert(byid['ShopFacade_vanilla_2dgs_reconstruct'])
    cmd=first['command'];first['command']=[cmd[0],'-u',str(snap/'scripts/run_with_cuda_budget.py'),'--fraction','0.55',*cmd[2:]]
    first['allocator_budget_note']='55% allocator budget only while overlapping existing office map; objective and iterations unchanged'
    pending=[first]
    office=old/'runs/primary_office/projective_map/report.json'
    identity=Path(f'/proc/{a.office_pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    pending.append(add('primary_office_map',[py,'-u','-m','scripts.wait_external_map','--pid',a.office_pid,
        '--identity',identity,'--report',office],outputs=[office]))
    for scene in ['office','redkitchen']:
        key='primary_'+scene
        if scene=='redkitchen':
            pending.append(add(key+'_map',[py,'-u','-m','scripts.audit_stage_recovery',
                '--command',old/f'runs/{key}/logs/projective_map.command.json','--snapshot',snap,
                '--output',a.root/f'audits/{scene}.json','--map-report',old/f'runs/{key}/projective_map/report.json'],
                outputs=[a.root/f'audits/{scene}.json']))
        for seed in [2026,2027]:
            j=convert(byid[key+f'_eval{seed}'])
            for flag in ['--map','--metric-state']:
                ix=j['command'].index(flag)+1;j['command'][ix]=j['command'][ix].replace(str(a.root),str(old))
            pending.append(j)
    for scene in ['stairs','pumpkin']:
        key='primary_'+scene;orig=old/'runs'/key;out=a.root/'runs'/key
        audit=key+'_cache_audit'
        pending.append(add(audit,[py,'-u','-m','scripts.audit_stage_recovery','--command',orig/'logs/observations.command.json',
                                 '--snapshot',snap,'--output',a.root/f'audits/{scene}.json'],outputs=[a.root/f'audits/{scene}.json']))
        obs=orig/'observations/rendered_rgb_feature_cache.pt'
        dataset=Path('/mnt/pool/sqy/datasets/7Scenes_pgt_full_reference_v5')/scene
        shards=[];deps=[]
        for index in range(2):
            shard=out/f'v2_audit/shard_{index}.pt';shards.append(shard);dep=key+f'_audit{index}';deps.append(dep)
            pending.append(add(dep,[py,'-u','-m','scripts.audit_v7_mapping_render_quality_shard','--dataset',dataset,
                '--images','processed','--gaussian-ply',dataset/'lafgs_prior_v1/point_cloud/iteration_30000/point_cloud.ply',
                '--gaussian-type','2dgs','--sh-degree','3','--observation-cache',obs,'--output',shard,
                '--shard-index',index,'--shard-count','2','--device','cuda:0'],[audit],[shard]))
        pending.append(add(key+'_map',[py,'-u','-m','scripts.materialize_v8_v2_projective_map',
            '--observation-cache',obs,'--v2-audit-shards',*shards,'--output-dir',out/'projective_map',
            '--device','cuda:0','--cpu-threads','4','--triangulation-workers','4'],deps,[out/'projective_map/report.json']))
        pending.extend(convert(byid[key+f'_eval{seed}']) for seed in [2026,2027])
    pending.extend(convert(j) for j in jobs if not j['id'].startswith('primary_') and j['id']!=first['id'])
    ids=[]
    for j in pending:
        if j['id'] in ids or not set(j['deps'])<=set(ids):raise ValueError('invalid DAG')
        ids.append(j['id'])
    (a.root/'jobs.json').write_text(json.dumps(pending,indent=2))
    print('snapshot',snap,'files',len(manifest),'jobs',len(pending),flush=True)


if __name__=='__main__':main()
