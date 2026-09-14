"""Register and launch the complete Cambridge downstream experiment DAG.

All commands are explicit. Baselines are native pinned code with protocol-only
adapters, not new traditional localization implementations.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

REPO=Path(__file__).resolve().parents[1]
POOL=Path('/mnt/pool/sqy')
OUT=POOL/'anygsloc_cambridge_suite_v2_20260914'
PY=sys.executable
BPY='/root/miniconda3/envs/ulfloc_repro/bin/python'
SCENES=['ShopFacade','KingsCollege','OldHospital','StMarysChurch','GreatCourt']


def snapshot_git(repo,ref,target):
    target.mkdir(parents=True)
    archive=target.parent/(target.name+'.tar')
    with archive.open('wb') as stream:
        subprocess.run(['git','-C',str(repo),'archive',ref],stdout=stream,check=True)
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            if member.issym() or member.islnk() or not (target/member.name).resolve().is_relative_to(target.resolve()):
                raise ValueError('unsafe source archive member')
        stream.extractall(target)
    return subprocess.check_output(['git','-C',str(repo),'rev-parse',ref],text=True).strip()


def main():
    OUT.mkdir(exist_ok=False)
    code=OUT/'code'
    shutil.copytree(REPO/'anygsloc',code/'anygsloc',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(REPO/'tools',code/'tools',ignore=shutil.ignore_patterns('__pycache__'))
    (code/'scripts').mkdir()
    for name in ['durable_experiment_queue.py','wait_queue_lane_finished.py']:
        shutil.copy2(REPO/'research/legacy/scripts'/name,code/'scripts'/name)
    commits={}
    commits['STDLoc']=snapshot_git(REPO,'upstream/main',OUT/'baselines/STDLoc')
    commits['ULFLoc']=snapshot_git(Path('/tmp/ULF-Loc-repro-b28d532'),'HEAD',OUT/'baselines/ULFLoc')
    (OUT/'baseline_commits.json').write_text(json.dumps(commits,indent=2))
    jobs=[];registry=[]
    def add(key,command,outputs=(),deps=(),cwd=code,group=''):
        jobs.append(dict(id=key,command=[str(x) for x in command],outputs=[str(x) for x in outputs],
                         deps=list(deps),cwd=str(cwd),env=dict(PYTHONPATH=str(code),OMP_NUM_THREADS='4',
                         OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')))
        registry.append(dict(id=key,group=group,dependencies=list(deps)))
        return key
    wait=add('wait_primary24',[PY,'-m','scripts.wait_queue_lane_finished','--state',
        POOL/'anygsloc_protocol_v2_primary_20260914/gpu0/state.json'])
    def evaluate(key,scene,deployment,deps=()):
        result=OUT/'runs'/key
        return add(key,[PY,'-m','anygsloc','evaluate','--dataset',POOL/'Cambridge_stdloc'/scene,
            '--map',deployment,'--output',result,'--image-domain','rectified',
            '--mask-policy','geometric_validity','--rectification-contract','opencv_same_k',
            '--seed','2026','--device','cuda:0'],[result/'summary.json',result/'results.json'],[wait]+list(deps),group='accuracy')
    def export(key,source):
        source=Path(source)
        if source.is_file():source=source.parent
        for filename in ['projective_anchor_map.pt','identity_metric.pt']:
            if not (source/filename).is_file():raise FileNotFoundError(source/filename)
        target=OUT/'maps'/key/'deployment_map.pt'
        job=add(key+'_export',[PY,'-m','anygsloc','export-map','--source',source/'projective_anchor_map.pt',
            '--metric',source/'identity_metric.pt','--output',target],[target],[wait],group='map_export')
        return target,job
    # 30 alternative prior maps; original MAtCHA five cells live in primary24.
    prior_cells=json.loads((POOL/'anygsloc_release_priors_20260913/registry.json').read_text())['cells']
    for cell in prior_cells:
        _,scene,prior=cell['key'].split('/')
        key='prior_'+scene+'_'+prior
        existing=Path(cell['deployment_map'])
        if existing.is_file():
            target=existing;deps=[]
        else:
            target=OUT/'maps'/key/'deployment_map.pt'
            job=add(key+'_compact',[PY,'-m','anygsloc','compact-map','--map-dir',cell['source_map'],
                '--cache',cell['cache'],'--output',target.parent],[target],[wait],group='prior_compaction')
            deps=[job]
        evaluate(key,scene,target,deps)
    # Reuse mapping-only ablation assets, not old query results. One seed only.
    ablations=[]
    for folder in ['anygsloc_ablation_closeout_20260911','anygsloc_final_ablation_20260913']:
        for row in json.loads((POOL/folder/'registry.json').read_text()):
            if row['seed']!=2026:continue
            label=row.get('group','')+'_'+row['condition']
            key='ablation_'+row['scene']+'_'+label
            ablations.append((key,row['scene'],row['map']))
    for scene in SCENES:
        for condition in ['depth_common','ray_common']:
            ablations.append(('ablation_'+scene+'_'+condition,scene,
                POOL/'anygsloc_depth_ablation_20260913/maps'/scene/condition))
    for key,scene,source in ablations:
        target,job=export(key,source)
        evaluate(key,scene,target,[job])
    # Native baseline rebuilds: 2 methods x 2 mapping protocols x 5 scenes.
    baseline_eval=[]
    for scene in SCENES:
        for method in ['ULFLoc','STDLoc']:
            repo=OUT/'baselines'/method
            cfg=repo/'configs'/('ulfloc_cambridge.yaml' if method=='ULFLoc' else 'stdloc_cambridge.yaml')
            for semantics in ['off','on']:
                key=method+'_'+scene+'_mapping_semantics_'+semantics
                model=OUT/'baseline_models'/key
                source=POOL/'Cambridge_stdloc'/scene
                receipt=OUT/'receipts'/(key+'_train.json')
                common=[BPY,code/'tools/baseline_protocol_v2.py','--repo',repo,'--source',source,
                    '--method',method,'--mapping-semantics',semantics]
                train=[*common,'--phase','train','--receipt',receipt,'--','-s',source,'-m',model,
                    '-r','1','-f','sp','-g','3dgs','--iterations','30000','--data_device','cpu',
                    '--images','processed','--densify_grad_threshold','0.0004',
                    '--position_lr_init','0.000016','--scaling_lr','0.001']
                train+=['--sample_kpts','--cfg',cfg] if method=='ULFLoc' else ['--train_detector']
                train_job=add(key+'_train',train,[receipt,model/'point_cloud/iteration_30000/point_cloud.ply'],[wait],repo,'baseline_build')
                receipt=OUT/'receipts'/(key+'_eval.json')
                native=['-s',source,'-m',model,'--images',source/'processed','--data_device','cpu','--cfg',cfg,'--longest_edge','640']
                eval_job=add(key+'_eval',[*common,'--phase','evaluate','--receipt',receipt,'--',*native],
                    [receipt],[train_job],repo,'baseline_coarse_and_fine')
                baseline_eval.append((key,common,native,eval_job,model,repo))
    # Fresh conversion timing: identical fixed upstream prior, no reconstruction charged downstream.
    cells=json.loads((POOL/'anygsloc_release_validation_20260913/registry.json').read_text())['cells']
    for cell in cells:
        if not cell['key'].startswith('Cambridge/'):continue
        scene=cell['key'].split('/')[1];target=OUT/'conversion_cost'/scene
        receipt=OUT/'receipts'/('conversion_'+scene+'.json')
        add('conversion_'+scene,[PY,code/'tools/experiment_task.py','--receipt',receipt,'--',PY,
            '-m','anygsloc','build-map','--dataset',cell['dataset'],'--gaussian-ply',cell['prior_ply'],
            '--gaussian-type','2dgs','--sh-degree','3','--image-domain','rectified','--output',target,'--gpu','0'],
            [receipt,target/'compact/deployment_map.pt'],[wait],group='offline_conversion_cost')
    # Formal cost measurements are downstream of accuracy, not inferred from parallel logs.
    idle=add('wait_idle_for_benchmarks',[PY,code/'tools/wait_gpu_idle.py'],deps=[wait],group='benchmark_isolation')
    for scene in SCENES:
        target=OUT/'deployment_benchmark'/scene
        deployment=POOL/'anygsloc_release_validation_20260913/maps/Cambridge'/scene/'deployment_map.pt'
        add('benchmark_ours_'+scene,[PY,code/'tools/benchmark_release.py','--dataset',POOL/'Cambridge_stdloc'/scene,
            '--map',deployment,'--output',target],[target/'summary.json'],[idle],group='deployment_benchmark')
        receipt=OUT/'receipts'/('storage_ours_'+scene+'.json')
        add('storage_ours_'+scene,[PY,code/'tools/experiment_task.py','--receipt',receipt,'--asset',deployment],
            [receipt],[wait],group='storage')
    for key,common,native,dependency,model,repo in baseline_eval:
        for repeat in range(3):
            receipt=OUT/'receipts'/(key+'_benchmark'+str(repeat)+'.json')
            add(key+'_benchmark'+str(repeat),[*common,'--phase','evaluate','--receipt',receipt,'--warmup','50',
                '--run-tag','benchmark'+str(repeat),'--',*native],[receipt],[dependency,idle],repo,'baseline_deployment_benchmark')
        receipt=OUT/'receipts'/('storage_'+key+'.json')
        add('storage_'+key,[PY,code/'tools/experiment_task.py','--receipt',receipt,'--asset',model,
            '--asset',repo/'encoders/sp_encoder/weights/superpoint_v1.pth'],[receipt],[dependency],group='baseline_storage_inventory')
    (OUT/'registry.json').write_text(json.dumps(dict(seed=2026,query_semantics=False,scope='Cambridge',
        jobs=registry,baseline_commits=commits,upstream_prior_semantics='as-recorded; not inferred from labels'),indent=2))
    lane=OUT/'gpu0';lane.mkdir();(lane/'jobs.json').write_text(json.dumps(jobs,indent=2))
    (OUT/'source_hashes.json').write_text(json.dumps({str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in [code,OUT/'baselines'] for p in folder.rglob('*.py')},indent=2))
    with (lane/'watchdog.log').open('ab',buffering=0) as log:
        child=subprocess.Popen([PY,'-u','-m','scripts.durable_experiment_queue','--root',str(lane),
            '--gpu','0','--watchdog'],cwd=code,env=dict(os.environ,PYTHONPATH=str(code)),
            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    print(json.dumps(dict(jobs=len(jobs),ablations=len(ablations),prior_evaluations=len(prior_cells),
        baseline_builds=len(baseline_eval),pid=child.pid,root=str(OUT))))


if __name__=='__main__':main()
