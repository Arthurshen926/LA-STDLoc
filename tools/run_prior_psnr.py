"""Run the frozen five-scene / five-prior PSNR evaluation on three local GPUs."""
from pathlib import Path
import os, subprocess, sys

ROOT=Path(__file__).resolve().parents[1]
PYTHON=Path('/root/miniconda3/envs/g4splat/bin/python')
SCRIPT=ROOT/'tools/evaluate_prior_psnr.py'
OUT=ROOT/'artifacts/prior_psnr_20261008'
OUT.mkdir(parents=True,exist_ok=True)
env=os.environ.copy()
env['PATH']=str(PYTHON.parent)+os.pathsep+env.get('PATH','')
env['LD_LIBRARY_PATH']=str(PYTHON.parent.parent/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH','')
subprocess.run([str(PYTHON),str(SCRIPT),'prepare'],cwd=ROOT,env=env,check=True)
processes=[]
for index in range(3):
    local=env.copy();local['CUDA_VISIBLE_DEVICES']=str(index)
    stream=(OUT/f'worker_{index}.log').open('a',buffering=1)
    process=subprocess.Popen([str(PYTHON),str(SCRIPT),'worker','--worker',str(index),'--workers','3'],
                             cwd=ROOT,env=local,stdout=stream,stderr=subprocess.STDOUT)
    processes.append((process,stream))
codes=[]
for process,stream in processes:
    codes.append(process.wait());stream.close()
if any(codes):
    raise RuntimeError(f'PSNR worker exit codes {codes}; inspect worker logs. Completed views are resumable.')
subprocess.run([str(PYTHON),str(SCRIPT),'summarize'],cwd=ROOT,env=env,check=True)
