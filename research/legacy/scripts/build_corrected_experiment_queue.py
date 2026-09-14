"""Materialize executable primary + corrected Cambridge reconstruction DAG."""
import argparse
import json
from pathlib import Path
from scripts.durable_experiment_queue import atomic


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=True)
    if (a.root/'jobs.json').exists():raise FileExistsError(a.root/'jobs.json')
    py='/root/miniconda3/envs/g4splat/bin/python';repo=Path('/root/STDLoc')
    baseenv={'PYTHONPATH':str(repo),'LD_LIBRARY_PATH':'/root/miniconda3/envs/g4splat/lib',
             'PATH':'/root/miniconda3/envs/g4splat/bin:/usr/local/cuda/bin:/usr/bin:/bin'}
    jobs=[]
    def add(key,cmd,deps=(),outputs=(),env=None,cwd=None):
        jobs.append({'id':key,'command':list(map(str,cmd)),'deps':list(deps),'outputs':list(map(str,outputs)),
                     'env':{**baseenv,**(env or {})},'cwd':str(cwd or repo)})
        return key
    def evaluate(key,dataset,out,dep):
        for seed in [2026,2027]:
            ev=out/f'evaluation/base_seed{seed}'
            add(key+f'_eval{seed}',[py,'-u','-m','scripts.evaluate','--dataset',dataset,'--images','processed',
                '--map',out/'projective_map/projective_anchor_map.pt','--metric-state',out/'projective_map/identity_metric.pt',
                '--output',ev,'--config',repo/'configs/anygsloc_mainline.yaml','--split','test','--device','cuda:0',
                '--seed',seed,'--deployment-mode'],[dep],[ev/'summary.json',ev/'results.json'])
    def mapping(key,dataset,ply,typ,white,dep=None):
        out=a.root/'runs'/key
        cmd=[py,'-u','-m','scripts.run_anygsloc_scene','--scene',key,'--dataset',dataset,'--images','processed',
             '--gaussian-ply',ply,'--gaussian-type',typ,'--sh-degree','3','--output',out,'--gpu','0',
             '--cpu-threads','4','--triangulation-workers','4','--mapping-only']
        if white:cmd+=['--white-background']
        k=add(key+'_map',cmd,[dep] if dep else [],[out/'projective_map/report.json'])
        evaluate(key,dataset,out,k)
    for family,scene in [('12Scenes','office2_5b'),('7Scenes','stairs'),('7Scenes','pumpkin'),('7Scenes','redkitchen'),('7Scenes','office')]:
        d=Path('/mnt/pool/sqy/datasets')/(family+'_pgt_full_reference_v5')/scene
        key='primary_'+scene
        if scene=='office':
            old=Path('/mnt/pool/sqy/anygsloc_paper_experiments_20260902/7Scenes/office')
            cmd=json.loads((old/'logs/projective_map.command.json').read_text())['command']
            out=a.root/'runs'/key
            cmd[cmd.index('--output-dir')+1]=str(out/'projective_map');cmd[cmd.index('--cpu-threads')+1]='4'
            dep=add(key+'_map',cmd,outputs=[out/'projective_map/report.json'])
            evaluate(key,d,out,dep)
        else:mapping(key,d,d/'lafgs_prior_v1/point_cloud/iteration_30000/point_cloud.ply','2dgs',False)
    prepared=Path('/mnt/pool/sqy/anygsloc_corrected_cambridge_matrix_20260909/inputs')
    registered=Path('/mnt/pool/sqy/anygsloc_registered_matrix_20260909')
    for scene in ['ShopFacade','OldHospital','KingsCollege','StMarysChurch','GreatCourt']:
        d=registered/'datasets/GreatCourt' if scene=='GreatCourt' else Path('/mnt/pool/sqy/Cambridge_stdloc')/scene
        inp=prepared/scene;results=a.root/'priors'/scene
        # Independent initializers are followed by raw and separately initialized +1k evaluations.
        for method in ['long_lrm','i3dgs']:
            key=scene+'_'+method;native=results/method/'native';world=results/method/'world.ply'
            if method=='long_lrm':
                train=add(key+'_reconstruct',['/root/miniconda3/envs/long_lrm22/bin/python','-u','-m','scripts.run_long_lrm_cambridge',
                    '--long-lrm-root','/root/third_party_lafgs_priors/Long-LRM','--prepared',inp/'long_lrm_input_pose32',
                    '--checkpoint','/mnt/pool/sqy/long_lrm_checkpoints/dl3dv_i540_32input_8target_aug/checkpoint_000010000.pt',
                    '--output',native,'--seed','2026','--cpu-threads','4'],outputs=[native/'gaussians.ply'],
                    env={'LD_LIBRARY_PATH':'/root/miniconda3/envs/long_lrm22/lib'})
                exp=add(key+'_export',[py,'-m','scripts.export_long_lrm_prior','--source-ply',native/'gaussians.ply',
                    '--prepared-manifest',inp/'long_lrm_input_pose32/anygsloc_long_lrm_input.json',
                    '--output-ply',world,'--output-manifest',world.parent/'export.json'],[train],[world])
            else:
                iroot=Path('/root/third_party_lafgs_priors/i3dgs')
                train=add(key+'_reconstruct',['/root/miniconda3/envs/i3dgs/bin/python','-u',iroot/'train.py','-s',inp/'i3dgs_input_d2',
                    '-m',native,'--use_colmap_poses','--fix_focal','--lr_poses','0','--downsampling','1',
                    '--num_loader_threads','4','--max_active_keyframes','250','--no-hierarchy','--no-use_rotated_descriptors',
                    '--sh_degree','3','--num_iterations','30','--save_at_finetune_epoch','1','3','--display_runtimes'],
                    outputs=[native/'3/hierarchy.ply'],cwd=iroot,env={'LD_LIBRARY_PATH':'/root/miniconda3/envs/i3dgs/lib',
                    'CUDA_HOME':'/usr/local/cuda-11.8','PATH':'/usr/local/cuda-11.8/bin:/root/miniconda3/envs/i3dgs/bin:/usr/bin:/bin',
                    'I3DGS_XFEAT_ROOT':str(iroot/'submodules/xfeat'),'I3DGS_MEGALOC_ROOT':'/root/third_party_lafgs_priors/MegaLoc',
                    'HF_HOME':'/mnt/pool/sqy/huggingface_cache','TORCH_HOME':'/mnt/pool/sqy/torch_cache'})
                exp=add(key+'_export',[py,'-m','scripts.export_i3dgs_prior','--hierarchy-ply',native/'3/hierarchy.ply',
                    '--prepared-manifest',inp/'i3dgs_input_d2/anygsloc_i3dgs_input.json','--output-ply',world,
                    '--output-manifest',world.parent/'export.json'],[train],[world])
            mapping(key+'_raw',d,world,'3dgs',method=='long_lrm',exp)
            ft=results/method/'ft1000';groot=Path('/root/third_party_lafgs_priors/gaussian-splatting')
            cmd=[py,'-u',groot/'train.py','-s',inp/'i3dgs_input_d1','-m',ft,'--init_ply',world,
                '--images','images','--resolution','1','--sh_degree','3','--iterations','1000','--data_device','cpu',
                '--position_lr_init','0.00008','--position_lr_final','0.0000008','--position_lr_max_steps','1000',
                '--scaling_lr','0.0025','--rotation_lr','0.0005','--opacity_lr','0.0125','--feature_lr','0.0025',
                '--exposure_lr_init','0','--exposure_lr_final','0','--depth_l1_weight_init','0','--depth_l1_weight_final','0',
                '--densify_until_iter','0','--lambda_dssim','0.2','--test_iterations','1001','--save_iterations','1000','--disable_viewer','--quiet']
            ply=ft/'point_cloud/iteration_1000/point_cloud.ply'
            k=add(key+'_ft1000',cmd,[exp],[ply],{'PYTORCH_CUDA_ALLOC_CONF':'max_split_size_mb:128'},groot)
            mapping(key+'_1k',d,ply,'3dgs',False,k)
        for typ in ['2dgs','3dgs']:
            if typ=='2dgs' and scene in ['OldHospital','KingsCollege']:continue
            key=scene+'_vanilla_'+typ
            vinp=(Path('/mnt/pool/sqy/anygsloc_weak_prior_protocol_20260907/corrected_prior_inputs_20260908')/scene
                  if scene in ['OldHospital','KingsCollege'] else registered/'vanilla_inputs'/scene)
            native=results/('vanilla_'+typ)/'native';groot=Path('/mnt/pool/sqy/third_party_lafgs_priors')/('2d-gaussian-splatting' if typ=='2dgs' else 'gaussian-splatting')
            cmd=['/root/miniconda3/envs/cybersim_agent/bin/python','-u',groot/'train.py','--source_path',vinp,
                '--model_path',native,'--images','images','--resolution','1','--data_device','cpu','--sh_degree','3',
                '--iterations','30000','--test_iterations','30000','--save_iterations','30000','--quiet',
                '--checkpoint_iterations','10000','20000']
            if typ=='3dgs':cmd+=['--depth_l1_weight_init','0','--depth_l1_weight_final','0','--disable_viewer']
            else:cmd+=['--port','6040']
            paths=[groot/'submodules'/('diff-surfel-rasterization' if typ=='2dgs' else 'diff-gaussian-rasterization'),groot/'submodules/simple-knn',groot/'submodules/fused-ssim',groot]
            k=add(key+'_reconstruct',cmd,outputs=[native/'point_cloud/iteration_30000/point_cloud.ply'],env={
                'PYTHONPATH':':'.join(map(str,paths)),'LD_LIBRARY_PATH':'/root/miniconda3/envs/cybersim_agent/lib',
                'PYTORCH_CUDA_ALLOC_CONF':'max_split_size_mb:128'},cwd=groot)
            mapping(key,d,native/'point_cloud/iteration_30000/point_cloud.ply',typ,False,k)
    ids=[j['id'] for j in jobs]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate job')
    for n,j in enumerate(jobs):
        if not set(j['deps'])<=set(ids[:n]):raise ValueError('non-topological dependency')
    atomic(a.root/'jobs.json',jobs)
    print(len(jobs),'jobs',sum(j['id'].endswith('_reconstruct') for j in jobs),'reconstructions',flush=True)


if __name__=='__main__':main()
