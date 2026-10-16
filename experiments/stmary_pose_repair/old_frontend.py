"""Run the byte-verified historical alpha frontend/full mapper on repaired inputs."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time
from anygsloc.artifacts import sha256
from .prepare import ROOT, JOBS, json_new
from .train import read, bound

OLD=Path('/mnt/pool/sqy/anygsloc_three_gpu_20260910')
OLD_RUN=OLD/'runs/StMarysChurch_vanilla_3dgs'


def source_audit():
    historical=read(OLD_RUN/'logs/observations.command.json')['input_contract']['code']
    for p,h in historical.items():
        if sha256(OLD/'code'/p)!=h:raise ValueError('historical mapper source changed: '+p)
    return historical

def plan(root,gpu):
    root=Path(root);accepted=read(root/'training_acceptance.json')
    if accepted.get('complete') is not True or sha256(accepted['ply']['path'])!=accepted['ply']['sha256']:
        raise ValueError('trained prior must be accepted and hash-bound')
    original=next(x for x in read(JOBS) if x['id']=='StMarysChurch_vanilla_3dgs_map')
    job=copy.deepcopy(original);argv=job['command']
    for flag,val in {'--dataset':root/'dataset/StMarysChurch','--gaussian-ply':accepted['ply']['path'],
                     '--output':root/'old_frontend/StMarysChurch_vanilla_3dgs','--gpu':gpu}.items():
        argv[argv.index(flag)+1]=str(val)
    if '--mapping-only' not in argv:raise ValueError('historical mapper would run evaluations before B freeze')
    job['id']='StMarysChurch_pose_repaired_old_frontend_full'
    job['env']['CUDA_VISIBLE_DEVICES']=str(gpu)
    job['outputs']=[str(root/'old_frontend/StMarysChurch_vanilla_3dgs/anygsloc_scene_manifest.json')]
    env=os.environ.copy();env.update(job['env'])
    response=subprocess.run(argv+['--dry-run'],cwd=job['cwd'],env=env,capture_output=True,text=True,check=True)
    generated=json.loads(response.stdout)
    old=read(OLD_RUN/'anygsloc_scene_manifest.json')
    if [s['name'] for s in generated['stages']]!=[s['name'] for s in old['stages']]:
        raise ValueError('historical stage list changed')
    replacements={old['dataset']:str(root/'dataset/StMarysChurch'),old['prior']['ply']:accepted['ply']['path'],str(OLD_RUN):str(root/'old_frontend/StMarysChurch_vanilla_3dgs')}
    for before,after in zip(old['stages'],generated['stages']):
        expected=[]
        for token in before['command']:
            for src,dst in replacements.items():token=token.replace(src,dst)
            expected.append(token)
        if expected!=after['command']:raise ValueError('historical stage command changed: '+before['name'])
    if generated['mapping_detection']!=old['mapping_detection'] or generated['config']!=old['config']:
        raise ValueError('historical mapping parameters changed')
    job.update(training_acceptance_sha256=sha256(root/'training_acceptance.json'),
        generated_plan=generated,historical_source_sha256=source_audit(),
        historical_manifest=bound(OLD_RUN/'anygsloc_scene_manifest.json'))
    return job

def launch(root,gpu):
    root=Path(root);job=plan(root,gpu)
    out=root/'old_frontend/StMarysChurch_vanilla_3dgs'
    if out.exists() or (root/'old_frontend_job.json').exists():raise FileExistsError('existing mapper attempt requires explicit reconciliation')
    json_new(root/'old_frontend_job.json',job)
    env=os.environ.copy();env.update(job['env']);started=time.time()
    print(json.dumps({'started_unix':started,'command':job['command']}),flush=True)
    with (root/'old_frontend.log').open('x') as log:
        proc=subprocess.run(job['command'],cwd=job['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT)
    outcome={'returncode':proc.returncode,'started_unix':started,'ended_unix':time.time(),'wall_seconds':time.time()-started}
    json_new(root/'old_frontend_outcome.json',outcome)
    if proc.returncode!=0:raise RuntimeError('historical mapper failed; inspect exact stage logs')
    source_audit()
    m=read(out/'anygsloc_scene_manifest.json');accepted=read(root/'training_acceptance.json')
    if m.get('complete') is not True or m['prior']['ply_sha256']!=accepted['ply']['sha256']:
        raise ValueError('mapper completion/prior identity mismatch')
    if set(m['results'])!={'observations','v2_audit_000','v2_audit_001','projective_map'}:
        raise ValueError('unexpected mapper stage results')
    if m['scientific_scope'].get('mapping_uses_test_queries') is not False or m['scientific_scope'].get('mapping_uses_source_rgb') is not False:
        raise ValueError('mapper crossed source-free training boundary')
    for result in m['results'].values():
        for p,h in result['artifact_sha256'].items():
            if not Path(p).resolve().is_relative_to(out.resolve()) or sha256(p)!=h:
                raise ValueError('mapper output identity/hash mismatch')
    report={'complete':True,'producer_manifest':bound(out/'anygsloc_scene_manifest.json'),
        'full_map':bound(out/'projective_map/projective_anchor_map.pt'),
        'observation_cache':{'path':str(out/'observations/rendered_rgb_feature_cache.pt'),
                             'sha256':m['results']['observations']['output_sha256'],
                             'bytes':(out/'observations/rendered_rgb_feature_cache.pt').stat().st_size},
        'training_acceptance_sha256':sha256(root/'training_acceptance.json'),
        'old_frontend_job_sha256':sha256(root/'old_frontend_job.json'),'recipe_matches_original_except_input_output_gpu':True}
    json_new(root/'old_frontend_acceptance.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['plan','launch']);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--gpu',type=int,required=True)
    a=p.parse_args()
    if a.gpu not in (0,1,2):p.error('one assigned GPU required')
    if a.stage=='plan':print(json.dumps(plan(a.root,a.gpu),indent=2))
    else:launch(a.root,a.gpu)
