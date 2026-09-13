"""Same-budget official Vanilla replay from hash-attested corrected inputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from common.hashing import sha256_file
from scripts.run_anygsloc_scene import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['input', 'dataset', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--gpu', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--allocator-max-split-mb', type=int, default=0,
                   help='Optional allocator-only retry; does not change training objective or resolution.')
    p.add_argument('--checkpoint-iterations', nargs='*', type=int, default=[])
    args = p.parse_args()
    args.input, args.dataset, args.output = [x.resolve() for x in [args.input,args.dataset,args.output]]
    if args.output.exists(): raise FileExistsError(args.output)
    manifest_path = args.input / 'rectification_replay_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['uses_test_queries'] or not manifest['mapping_rgb_only'] or Path(manifest['dataset']) != args.dataset:
        raise ValueError('corrected mapping-only input required')
    for r in manifest['images']:
        if sha256_file(args.input / 'images' / r['name']) != r['sha256']: raise ValueError('input image changed')
    for name, digest in manifest['sparse_sha256'].items():
        if sha256_file(args.input / 'sparse/0' / name) != digest: raise ValueError('sparse initialization changed')
    repo = Path(__file__).resolve().parents[1]
    official = Path('/mnt/pool/sqy/third_party_lafgs_priors/2d-gaussian-splatting')
    commit = subprocess.check_output(['git','-C',str(official),'rev-parse','HEAD'],text=True).strip()
    if commit != '335ad612f2e783a4e57b9cbc4d1e167bd599fc98': raise ValueError('official trainer revision changed')
    train_py = '/root/miniconda3/envs/cybersim_agent/bin/python'
    local_py = '/root/miniconda3/envs/g4splat/bin/python'
    native = args.output / 'native'; exported = args.output / 'stdloc_model'; scene = args.output / 'anygsloc'
    train_env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    train_env['PYTHONPATH'] = ':'.join(str(official / x) for x in ['submodules/diff-surfel-rasterization','submodules/simple-knn','.'])
    train_env['LD_LIBRARY_PATH'] = '/root/miniconda3/envs/cybersim_agent/lib'
    if args.allocator_max_split_mb:
        if args.allocator_max_split_mb < 21:
            raise ValueError('allocator max split size must exceed 20 MB')
        train_env['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{args.allocator_max_split_mb}'
    local_env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONPATH=str(repo))
    local_env['LD_LIBRARY_PATH'] = '/root/miniconda3/envs/g4splat/lib'
    local_env['PATH'] = '/root/miniconda3/envs/g4splat/bin:' + os.environ.get('PATH','')
    stages = [('train', [train_py,str(official/'train.py'),'--source_path',str(args.input),'--model_path',str(native),
                         '--images','images','--resolution','1','--data_device','cpu','--sh_degree','3',
                         '--iterations','30000','--test_iterations','30000','--save_iterations','30000','--quiet','--port',str(args.port)], train_env),
              ('export', [local_py,'-m','priors.gaussian_io','--input_ply',str(native/'point_cloud/iteration_30000/point_cloud.ply'),
                          '--output_model',str(exported),'--gaussian_type','2dgs','--sh_degree','3',
                          '--source_path',str(args.dataset),'--images','processed','--longest_edge','0','--iteration','30000',
                          '--prior_kind','rgb_only','--black_background','--source_method','vanilla_2dgs','--source_commit',commit], local_env),
              ('mapping', [local_py,'-u','-m','scripts.run_anygsloc_scene','--scene',args.dataset.name+'_vanilla_rectified',
                           '--dataset',str(args.dataset),'--images','processed','--prior-manifest',str(exported/'rgb_prior_manifest.json'),
                           '--output',str(scene),'--gpu',args.gpu,'--cpu-threads','4','--triangulation-workers','4','--mapping-only'],local_env)]
    if args.checkpoint_iterations:
        if any(step < 1 or step > 30000 for step in args.checkpoint_iterations):
            raise ValueError('checkpoint iteration outside training budget')
        stages[0][1].extend(['--checkpoint_iterations', *map(str, args.checkpoint_iterations)])
    for seed in [2026,2027]:
        stages.append((f'evaluate_{seed}',[local_py,'-u','-m','scripts.evaluate','--dataset',str(args.dataset),'--images','processed',
                      '--map',str(scene/'projective_map/projective_anchor_map.pt'),'--metric-state',str(scene/'projective_map/identity_metric.pt'),
                      '--output',str(scene/f'evaluation/base_seed{seed}'),'--config',str(repo/'configs/anygsloc_mainline.yaml'),
                      '--split','test','--device','cuda:0','--seed',str(seed),'--deployment-mode','--audit-correspondences'],local_env))
    args.output.mkdir(parents=True)
    record = {'schema':'anygsloc_corrected_vanilla_retrain_replay_v1','input_manifest_sha256':sha256_file(manifest_path),
              'training_seed':0,'evaluation_seeds':[2026,2027],'iterations':30000,'official_commit':commit,
              'allocator_config':train_env.get('PYTORCH_CUDA_ALLOC_CONF'),
              'checkpoint_iterations':args.checkpoint_iterations,
              'train_py_sha256':sha256_file(official/'train.py'),'runner_sha256':sha256_file(Path(__file__)),
              'source_mapping_images_used_for_prior_training_only':True,'test_used_for_training':False,
              'offline_feedback':False,'stages':[],'complete':False}
    for name,command,env in stages:
        entry = {'name':name,'command':command,'status':'running','started_unix':time.time()}
        record['stages'].append(entry); atomic_json(args.output/'replay.json',record)
        print(name,'START',flush=True)
        with (args.output/(name+'.log')).open('xb') as log:
            result = subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
        entry.update(status='complete' if result.returncode == 0 else 'failed', seconds=time.time()-entry['started_unix'], returncode=result.returncode)
        atomic_json(args.output/'replay.json',record)
        if result.returncode: raise subprocess.CalledProcessError(result.returncode,command)
        print(name,'COMPLETE',flush=True)
    record['complete'] = True; atomic_json(args.output/'replay.json',record)


if __name__ == '__main__': main()
