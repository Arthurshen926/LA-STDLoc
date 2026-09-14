"""Register frozen remaining Cambridge ablations with three independent GPU lanes."""
import argparse
import json
import shutil
from pathlib import Path

POOL = Path('/mnt/pool/sqy')
PYTHON = '/root/miniconda3/envs/g4splat/bin/python'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    a.root.mkdir(exist_ok=False)
    code = a.root/'code'
    for name in ['scripts','common','data','evidence','features','localization','evaluation','priors','map_learning','topology','configs']:
        shutil.copytree(Path(__file__).resolve().parents[1]/name, code/name,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    lanes = [[], [], []]
    registry = []
    refs = json.loads((POOL/'anygsloc_core65_budget_20260910/gpu0/jobs.json').read_text())
    def add(gpu, key, cmd, outputs, deps=()):
        lanes[gpu].append(dict(id=key, command=list(map(str,cmd)), outputs=list(map(str,outputs)),
            deps=list(deps),cwd=str(code),env=dict(PYTHONPATH=str(code),CUDA_VISIBLE_DEVICES=str(gpu),
            PATH='/root/miniconda3/envs/g4splat/bin:/usr/local/cuda/bin:/usr/bin:/bin',
            LD_LIBRARY_PATH='/root/miniconda3/envs/g4splat/lib',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')))
    def evaluate(gpu, scene, name, folder, deps=()):
        ref = next(j for j in refs if j['id']==f'{scene}_core65_min250_conf999_max30k_seed2026')
        for seed in [2026,2027,2028]:
            out=a.root/'runs'/scene/f'{name}_seed{seed}'
            cmd=list(ref['command'])
            for flag,value in [('--map',folder/'projective_anchor_map.pt'),('--metric-state',folder/'identity_metric.pt'),
                               ('--config',code/'configs/anygsloc_mainline.yaml'),('--output',out),('--seed',seed)]:
                cmd[cmd.index(flag)+1]=str(value)
            add(gpu,f'{scene}_{name}_{seed}',cmd,[out/'summary.json',out/'results.json'],deps)
            registry.append(dict(scene=scene,condition=name,seed=seed,output=str(out),map=str(folder)))
    for gpu,scene in enumerate(['GreatCourt','StMarysChurch']):
        source=POOL/'anygsloc_cache_recovery_v2_20260912'/scene
        old=POOL/'anygsloc_missing_cache_downstream_20260912'/scene
        for name,folder in [('full',old/'projective_map')]+[(n,old/'selector_full/maps'/n) for n in ['core','matching','full_selector','equal_size_quality']]:
            assert (folder/'projective_anchor_map.pt').is_file(),folder
            evaluate(gpu,scene,'selection_'+name,folder)
        for variant,flag in [('no_v2','--disable-v2-row-filter'),('no_cycle','--disable-cycle-consistency'),('no_robust_ray','--disable-robust-ray-reweighting')]:
            base=a.root/'maps'/scene/variant; mp=base/'projective_map'; compact=base/'compact'
            key=f'{scene}_{variant}_build'
            shards=sorted((source/'v2_audit').glob('*.pt'))
            add(gpu,key,[PYTHON,'-u','-m','scripts.materialize_v8_v2_projective_map','--observation-cache',source/'observations/rendered_rgb_feature_cache.pt',
                '--v2-audit-shards',*shards,'--output-dir',mp,'--device','cuda:0','--triangulation-workers','4',flag],[mp/'report.json'])
            select=key+'_select'
            add(gpu,select,[PYTHON,'-u','-m','scripts.run_full_selector_pilot','--scene',scene,'--root',compact,
                '--source-scene-root',source,'--input-map-dir',mp],[compact/'protocol.json'],[key])
            evaluate(gpu,scene,'geometry_'+variant,compact/'maps/full_selector',[select])
    # Descriptor comparisons hold the Compact point identities and XYZ fixed.
    for scene in ['ShopFacade','OldHospital','KingsCollege','GreatCourt','StMarysChurch']:
        if scene in ['GreatCourt','StMarysChurch']:
            compact=POOL/'anygsloc_missing_cache_downstream_20260912'/scene/'selector_full/maps/full_selector'
            source=POOL/'anygsloc_cache_recovery_v2_20260912'/scene
        else:
            source=POOL/'lafgs_v24_cambridge_high_capacity_online_20260901'/scene
            compact=POOL/'anygsloc_full_selector_validation_20260911'/scene/'maps/full_selector'
            if not compact.exists():
                compact=POOL/'anygsloc_full_selector_shop_20260911/maps/full_selector'
        assert (compact/'projective_anchor_map.pt').is_file(),compact
        evaluate(2,scene,'descriptor_reference',compact)
        for variant in ['mean_descriptor','single_descriptor','no_completion']:
            out=a.root/'maps'/scene/variant;key=f'{scene}_{variant}_build'
            add(2,key,[PYTHON,'-u','-m','scripts.prepare_component_ablations','--scene',scene,'--variant',variant,
                '--map-dir',compact,'--cache-path',source/'observations/rendered_rgb_feature_cache.pt','--output',out],
                [out/'projective_anchor_map.pt',out/'identity_metric.pt'])
            evaluate(2,scene,variant,out,[key])
    for gpu,jobs in enumerate(lanes):
        lane=a.root/f'gpu{gpu}';lane.mkdir();(lane/'jobs.json').write_text(json.dumps(jobs,indent=2))
    (a.root/'registry.json').write_text(json.dumps(registry,indent=2))
    print('registered',len(registry),'evaluations', [len(j) for j in lanes],flush=True)


if __name__=='__main__':main()
