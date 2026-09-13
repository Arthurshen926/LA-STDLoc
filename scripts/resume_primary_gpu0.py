"""Explicit legacy-cache replay for two interrupted primary scenes on GPU 0."""
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    from common.hashing import sha256_file
    import torch
    repo = Path(__file__).resolve().parents[1]
    root = Path('/mnt/pool/sqy/anygsloc_primary_resume_20260909')
    root.mkdir(exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', PYTHONPATH=str(repo),
               LD_LIBRARY_PATH='/root/miniconda3/envs/g4splat/lib')
    env['PATH'] = '/root/miniconda3/envs/g4splat/bin:' + env.get('PATH','')
    py = '/root/miniconda3/envs/g4splat/bin/python'
    for family, scene in [('12Scenes','office2_5a'), ('7Scenes','office')]:
        old = Path('/mnt/pool/sqy/anygsloc_paper_experiments_20260902')/family/scene
        dst = root/family/scene
        if dst.exists(): raise FileExistsError(dst)
        dst.mkdir(parents=True)
        dataset = Path('/mnt/pool/sqy/datasets')/(family+'_pgt_full_reference_v5')/scene
        prior = dataset/'lafgs_prior_v1/point_cloud/iteration_30000/point_cloud.ply'
        shards = sorted((old/'v2_audit').glob('*.pt'))
        assert len(shards)==2
        prior_hash = sha256_file(prior)
        for p in shards:
            shard = torch.load(p,map_location='cpu',weights_only=False)
            if shard['input']['gaussian_ply_sha256'] != prior_hash:
                raise ValueError('legacy shard/prior mismatch')
        command = json.loads((old/'logs/projective_map.command.json').read_text())['command']
        command[command.index('--output-dir')+1] = str(dst/'projective_map')
        command[command.index('--cpu-threads')+1] = '4'
        stages = [('map',command)]
        for seed in [2026,2027]:
            stages.append((f'evaluate_{seed}',[py,'-u','-m','scripts.evaluate','--dataset',str(dataset),
                '--images','processed','--map',str(dst/'projective_map/projective_anchor_map.pt'),
                '--metric-state',str(dst/'projective_map/identity_metric.pt'), '--output',str(dst/f'evaluation/base_seed{seed}'),
                '--config',str(repo/'configs/anygsloc_mainline.yaml'),'--split','test','--device','cuda:0',
                '--seed',str(seed),'--deployment-mode']))
        record = {'legacy_unbound_cache_replay':True,'prior_sha256':prior_hash,
                  'note':'Materializer verifies cache SHA, query registry and shard scope. Legacy cache itself lacks prior binding.',
                  'complete':False,'stages':[]}
        for name,cmd in stages:
            row={'name':name,'command':cmd,'status':'running','start':time.time()}
            record['stages'].append(row)
            (dst/'replay.json').write_text(json.dumps(record,indent=2))
            print(family,scene,name,'START',flush=True)
            with (dst/(name+'.log')).open('xb') as log:
                result=subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
            row.update(returncode=result.returncode,status='complete' if result.returncode==0 else 'failed',seconds=time.time()-row['start'])
            (dst/'replay.json').write_text(json.dumps(record,indent=2))
            if result.returncode: raise RuntimeError(f'{scene} {name} failed')
        record['complete']=True
        (dst/'replay.json').write_text(json.dumps(record,indent=2))


if __name__=='__main__': main()
