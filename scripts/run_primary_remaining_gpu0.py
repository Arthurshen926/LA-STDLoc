"""Fresh GPU-0 runs for primary scenes without reusable observation caches."""
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    root=Path('/mnt/pool/sqy/anygsloc_primary_resume_20260909')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    env['LD_LIBRARY_PATH']='/root/miniconda3/envs/g4splat/lib'
    env['PATH']='/root/miniconda3/envs/g4splat/bin:'+env.get('PATH','')
    for family,scene in [('12Scenes','office2_5b'),('7Scenes','stairs'),('7Scenes','pumpkin'),('7Scenes','redkitchen')]:
        if shutil.disk_usage(root).free < 200*1024**3:
            raise RuntimeError('shared disk reserve reached')
        d=Path('/mnt/pool/sqy/datasets')/(family+'_pgt_full_reference_v5')/scene
        out=root/family/scene
        if out.exists(): raise FileExistsError(out)
        cmd=[sys.executable,'-u','-m','scripts.run_anygsloc_scene','--scene',scene,'--dataset',str(d),
             '--gaussian-ply',str(d/'lafgs_prior_v1/point_cloud/iteration_30000/point_cloud.ply'),
             '--gaussian-type','2dgs','--sh-degree','3','--output',str(out),'--gpu','0',
             '--cpu-threads','4','--triangulation-workers','4']
        with (root/(family+'_'+scene+'.log')).open('xb') as log:
            subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__': main()
