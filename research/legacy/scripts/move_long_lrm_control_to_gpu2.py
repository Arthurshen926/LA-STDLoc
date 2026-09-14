"""Move only not-started supplemental controls to the newly idle GPU 2."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    pool=Path('/mnt/pool/sqy');old=pool/'anygsloc_long_lrm_two_epoch_control_20260910'
    root=pool/'anygsloc_long_lrm_two_epoch_gpu2_control_20260910'
    shop=json.loads((pool/'anygsloc_shop_tile_recovery_20260910/gpu2/state.json').read_text())
    if not shop.get('all_complete'):raise RuntimeError('GPU 2 predecessor not complete')
    for gpu in [0,1]:
        state=json.loads((old/f'gpu{gpu}/state.json').read_text())
        if set(state['jobs'])!={'wait_preceding_lane'}:raise RuntimeError('control computation already began')
    targets=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:args=(proc/'cmdline').read_bytes().split(b'\0');args=[x.decode() for x in args if x]
        except (OSError,UnicodeDecodeError):continue
        if 'scripts.durable_experiment_queue' in args and any(str(old/f'gpu{g}') in args for g in [0,1]):
            targets.append((int(proc.name),args,'watchdog' if '--watchdog' in args else 'worker'))
    for role in ['watchdog','worker']:
        for pid,args,r in targets:
            if r==role:os.kill(pid,signal.SIGTERM);print('stopped waiting control',role,pid,flush=True)
        time.sleep(.2)
    for gpu in [0,1]:
        state=json.loads((old/f'gpu{gpu}/state.json').read_text());pid=state['jobs']['wait_preceding_lane']['pid']
        args=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if b'scripts.wait_queue_lane_finished' not in args:raise RuntimeError('wait process identity mismatch')
        os.kill(pid,signal.SIGTERM)
    root.mkdir(exist_ok=False);lane=root/'gpu2';lane.mkdir();jobs=[]
    for gpu in [1,0]:
        for original in json.loads((old/f'gpu{gpu}/jobs.json').read_text()):
            if original['id']=='wait_preceding_lane':continue
            j=json.loads(json.dumps(original).replace(str(old/'runs'),str(root/'runs')).replace(str(old/'priors'),str(root/'priors')))
            j['deps']=[d for d in j['deps'] if d!='wait_preceding_lane'];j['env']['CUDA_VISIBLE_DEVICES']='2'
            if '--gpu' in j['command']:j['command'][j['command'].index('--gpu')+1]='2'
            jobs.append(j)
    (lane/'jobs.json').write_text(json.dumps(jobs,indent=2))
    (root/'handoff.json').write_text(json.dumps({'source':str(old),'immutable_code':str(old/'code'),'source_code_manifest':str(old/'code_manifest.json'),
        'reason':'GPU 2 finished base matrix; supplemental controls had only wait jobs running','stopped_waiting_queue_pids':[x[0] for x in targets]},indent=2))
    with (lane/'launcher.log').open('ab') as log:
        p=subprocess.Popen(['/root/miniconda3/envs/g4splat/bin/python','-m','scripts.durable_experiment_queue','--root',str(lane),'--gpu','2','--watchdog'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print('GPU 2 control watchdog',p.pid,flush=True)
    with (root/'collector.log').open('ab') as log:
        p=subprocess.Popen(['/root/miniconda3/envs/g4splat/bin/python','-m','scripts.collect_three_gpu_results','--root',str(root),'--follow'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print('collector',p.pid,flush=True)


if __name__=='__main__':main()
