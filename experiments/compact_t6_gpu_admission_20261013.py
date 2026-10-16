"""Execution-only per-GPU admission; frozen Compact T6 science is unchanged."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from experiments import compact_t6_20261013 as D

ADDENDUM = dict(schema='compact_t6_per_gpu_admission_20261013_v1',
 reason='External jobs occupy physical GPUs 1/2; admit available GPUs independently.',
 invariant='Original scientific protocol, maps, inputs, frontend, seed, backend, thresholds, and population unchanged.',
 scheduling='GPU0 qualifies first and may run accuracy alone; idle GPU1/2 qualify against GPU0 before joining; maximum three workers.',
 qualification='Original 16-prediction pilot per GPU; production replay and semantic-mask checks unchanged; cross-GPU exact matches/inliers and pose tolerance unchanged.',
 pose_atol=1e-7,pose_rtol=1e-6,peak_reserved_limit_bytes=20*1024**3,
 formal_timing=False,foreign_gpu_processes='reject; never terminate or share assigned GPU')

def freeze(root):
    D.cpu_only();root=Path(root);D.protocol(root)
    D.write(root/'execution_admission_protocol.json',dict(design=ADDENDUM,
        scientific_protocol=D.ref(root/'protocol.json'),driver=D.ref(Path(__file__).resolve()),
        tests=D.ref(D.C.REPO/'tests/test_compact_t6_gpu_admission_20261013.py')))
    print('EXECUTION_ADDENDUM_FROZEN',flush=True)

def execution(root):
    root=Path(root);D.protocol(root);a=D.read(root/'execution_admission_protocol.json')
    if a['design']!=ADDENDUM:raise ValueError('execution design changed')
    for k in ('scientific_protocol','driver','tests'):D.check(a[k])
    r=D.read(root/'independent_execution_review.json')
    if r.get('verdict')!='PASS' or r.get('protocol_sha256')!=D.sha256(root/'protocol.json') or r.get('execution_addendum_sha256')!=D.sha256(root/'execution_admission_protocol.json'):
        raise ValueError('execution amendment not independently reviewed')
    return a

def validate_record(r,gpu,protocol_sha,sealed_sha,cases):
    if not r['complete'] or r['protocol_sha256']!=protocol_sha or r['sealed_sha256']!=sealed_sha or r['actual_cpu_threads']!=4 or not r['no_ground_truth_metrics']:
        raise ValueError('pilot identity')
    if r['hardware']['index']!=gpu or r['hardware']['uuid']!=D.GPU_UUIDS[gpu] or r['hardware']['name']!='NVIDIA GeForce RTX 3090':raise ValueError('hardware identity')
    if r['peak_reserved_bytes']>=ADDENDUM['peak_reserved_limit_bytes']:raise ValueError('memory admission')
    if len(r['outputs'])!=len(cases):raise ValueError('case count')
    for output,case in zip(r['outputs'],cases):
        if (output['scene'],output['mode'],output['map_sha256'])!=(case['scene'],case['mode'],case['map']['sha256']):raise ValueError('pilot cases')
        if [x['name'] for x in output['rows']]!=case['names']:raise ValueError('pilot population')

def same_predictions(a,b):
    for k in ('python','torch','cuda'):
        if a['hardware'][k]!=b['hardware'][k]:raise ValueError('software differs '+k)
    for ao,bo in zip(a['outputs'],b['outputs']):
        for x,y in zip(ao['rows'],bo['rows']):
            for k in ('name','success','inliers','matched_keypoints_sha256','matched_anchors_sha256'):
                if x[k]!=y[k]:raise ValueError('cross-GPU prediction mismatch '+k)
            if not np.allclose(x['pose'],y['pose'],atol=ADDENDUM['pose_atol'],rtol=ADDENDUM['pose_rtol']):raise ValueError('cross-GPU pose mismatch')

def qualify(root,gpu):
    D.cpu_only();root=Path(root);execution(root);p,s=D.sealed(root);D.require_review(root,'maps')
    ps=D.sha256(root/'protocol.json');ss=D.sha256(root/'sealed.json');pilot=root/'pilot'/f'gpu{gpu}.json';r=D.read(pilot)
    validate_record(r,gpu,ps,ss,s['pilot_cases'])
    if gpu==0:
        D.write(root/'pilot_pass.json',dict(complete=True,verdict='PASS',protocol_sha256=ps,sealed_sha256=ss,
            records=[D.ref(pilot)],maximum_concurrency=3,predictions_only=True,no_gt_selection=True,
            scheduling='per-GPU qualification required by execution addendum',execution_addendum=D.ref(root/'execution_admission_protocol.json')))
    else:
        g=D.read(root/'gpu_admission/gpu0.json')
        if g['verdict']!='PASS':raise ValueError('reference GPU0 not qualified')
        D.check(g['pilot']);base=D.read(g['pilot']['path']);validate_record(base,0,ps,ss,s['pilot_cases']);same_predictions(base,r)
    D.write(root/'gpu_admission'/f'gpu{gpu}.json',dict(verdict='PASS',gpu=gpu,protocol_sha256=ps,sealed_sha256=ss,
        pilot=D.ref(pilot),execution_addendum=D.ref(root/'execution_admission_protocol.json'),
        baseline_pilot=D.ref(root/'pilot/gpu0.json'),production_replay_verified=True,no_gt_selection=True))
    print('GPU_ADMISSION_PASS',gpu,flush=True)

def evaluate(root,scene,mode,gpu):
    root=Path(root);execution(root);a=D.read(root/'gpu_admission'/f'gpu{gpu}.json')
    if a['verdict']!='PASS' or a['gpu']!=gpu or a['protocol_sha256']!=D.sha256(root/'protocol.json') or a['sealed_sha256']!=D.sha256(root/'sealed.json'):raise ValueError('GPU not admitted')
    for k in ('pilot','execution_addendum','baseline_pilot'):D.check(a[k])
    D.evaluate(root,scene,mode,gpu)
    folder=root/'evaluations'/scene/mode
    D.write(folder/'execution_admission.json',dict(gpu_admission=D.ref(root/'gpu_admission'/f'gpu{gpu}.json'),
        execution_addendum=D.ref(root/'execution_admission_protocol.json'),attestation=D.ref(folder/'attestation.json'),formal_timing=False))

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('stage',choices=('freeze','qualify','evaluate'));a.add_argument('--root',type=Path,default=D.ROOT)
    a.add_argument('--gpu',type=int,choices=(0,1,2));a.add_argument('--scene',choices=D.C.SCENES);a.add_argument('--mode',choices=D.NEW);x=a.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    if x.stage=='freeze':freeze(x.root)
    elif x.stage=='qualify':qualify(x.root,x.gpu)
    else:evaluate(x.root,x.scene,x.mode,x.gpu)
if __name__=='__main__':main()
