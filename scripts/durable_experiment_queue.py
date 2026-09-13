"""Serial GPU queue with atomic state, heartbeat, dependency isolation and watchdog.

Completed jobs are not rerun. Interrupted/orphaned jobs are reported, not silently
overwritten. Independent jobs continue after failures. No process on GPU 1/2 is touched.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def atomic(path, value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def identity(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except (FileNotFoundError,ProcessLookupError):return None


def run(root, gpu='0'):
    if gpu not in {'0','1','2'}:raise ValueError('invalid reserved GPU')
    root.mkdir(parents=True,exist_ok=True)
    with (root/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=json.loads((root/'jobs.json').read_text())
        path=root/'state.json'
        state=json.loads(path.read_text()) if path.exists() else {'jobs':{}}
        for job in jobs:
            r=state['jobs'].get(job['id'],{})
            if r.get('status')=='running':
                if r.get('process_identity') and identity(r['pid'])==r['process_identity']:
                    # Surviving child must finish before another GPU task starts.
                    while identity(r['pid'])==r['process_identity']:
                        state['heartbeat']=time.time();atomic(path,state);time.sleep(10)
                r.update(status='interrupted',reason='worker exited without collecting child completion; manual artifact audit required')
        atomic(path,state)
        for job in jobs:
            state['heartbeat']=time.time()
            if state['jobs'].get(job['id'],{}).get('status') in {'complete','failed','interrupted'}:continue
            if any(state['jobs'].get(dep,{}).get('status')!='complete' for dep in job.get('deps',[])):
                state['jobs'][job['id']]={'status':'blocked','reason':'dependency not complete'};atomic(path,state);continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
            env.update(job.get('env',{}))
            if env['CUDA_VISIBLE_DEVICES']!=gpu:raise ValueError('job GPU differs from assigned lane')
            r={'status':'running','started':time.time(),'command':job['command']}
            state['jobs'][job['id']]=r
            with (root/(job['id']+'.log')).open('ab',buffering=0) as log:
                try:
                    child=subprocess.Popen(job['command'],cwd=job.get('cwd','/root/STDLoc'),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    r.update(pid=child.pid,process_identity=identity(child.pid));atomic(path,state)
                    while child.poll() is None:
                        state['heartbeat']=time.time();atomic(path,state);time.sleep(10)
                    missing=[p for p in job.get('outputs',[]) if not Path(p).is_file()]
                    r.update(returncode=child.returncode,status='complete' if child.returncode==0 and not missing else 'failed',missing_outputs=missing)
                except Exception as exc:r.update(status='failed',error=repr(exc))
            r['seconds']=time.time()-r['started'];atomic(path,state)
        state.update(queue_finished=True,all_complete=all(state['jobs'].get(j['id'],{}).get('status')=='complete' for j in jobs));atomic(path,state)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--watchdog',action='store_true')
    p.add_argument('--gpu',choices=['0','1','2'],default='0');a=p.parse_args()
    if not a.watchdog:return run(a.root,a.gpu)
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    for attempt in range(4):
        result=subprocess.run([sys.executable,'-u','-m','scripts.durable_experiment_queue','--root',str(a.root),'--gpu',a.gpu])
        atomic(a.root/'watchdog.json',{'attempt':attempt,'returncode':result.returncode,'updated':time.time(),'pid':os.getpid()})
        if result.returncode==0:return
        time.sleep(5)
    raise RuntimeError('worker repeatedly failed; inspect state and watchdog log')


if __name__=='__main__':main()
