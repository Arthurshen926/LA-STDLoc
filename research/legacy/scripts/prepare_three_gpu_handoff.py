"""Distribute the frozen experiment DAG into three independently executable lanes."""
import argparse
import json
from pathlib import Path
import shutil
from common.hashing import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    old=Path('/mnt/pool/sqy/anygsloc_frozen_execution_20260910')
    if a.root.exists():raise FileExistsError(a.root)
    snap=a.root/'code';snap.parent.mkdir(parents=True)
    shutil.copytree(old/'code',snap,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    worker=snap/'scripts/durable_experiment_queue.py';worker.chmod(0o644)
    shutil.copyfile(Path(__file__).with_name('durable_experiment_queue.py'),worker)
    manifest={str(p.relative_to(snap)):sha256_file(p) for p in snap.rglob('*') if p.is_file()}
    (a.root/'code_manifest.json').write_text(json.dumps(manifest,indent=2))
    for p in snap.rglob('*'):
        if p.is_file():p.chmod(0o444)
    jobs=json.loads((old/'jobs.json').read_text())
    lanes={0:[],1:[],2:[]}
    # Prioritize StMary for the requested canopy comparison; retain dependencies
    # within each scene so no lane waits for another lane's pending reconstruction.
    for original in jobs:
        key=original['id']
        if key.startswith('primary_') or key.startswith('GreatCourt_'):gpu=0
        elif key.startswith(('StMarysChurch_','OldHospital_')):gpu=1
        else:gpu=2
        j=json.loads(json.dumps(original).replace(str(old),str(a.root)))
        j['env']['CUDA_VISIBLE_DEVICES']=str(gpu)
        if '--gpu' in j['command']:j['command'][j['command'].index('--gpu')+1]=str(gpu)
        if '--snapshot' in j['command']:
            # Audit against the exact original immutable implementation, which
            # is byte-identical to this snapshot except for GPU lane dispatch.
            j['command'][j['command'].index('--snapshot')+1]=str(old/'code')
        if key=='ShopFacade_vanilla_2dgs_reconstruct':
            # Fresh full-budget retry. Do not retain the failed 55% concurrency cap.
            c=j['command'];j['command']=[c[0],'-u',*c[c.index('--fraction')+2:]]
            j.pop('allocator_budget_note',None)
            j['retry_note']='exclusive GPU, fresh initialization; previous capped failure preserved'
        lanes[gpu].append(j)
    def priority(j):
        key=j['id']
        if key.startswith('primary_redkitchen'):return 0
        if key.startswith('primary_office'):return 1
        if key.startswith('primary_'):return 2
        if key.startswith('StMarysChurch_'):return 0
        if key.startswith('ShopFacade_'):return 0
        return 3
    for gpu,items in lanes.items():
        # Stable scene ordering preserves each scene's existing topological order.
        items.sort(key=priority)
        seen=set()
        for j in items:
            if not set(j['deps'])<=seen:raise ValueError(f'cross-lane or out-of-order dependency: {j["id"]}')
            seen.add(j['id'])
        lane=a.root/f'gpu{gpu}';lane.mkdir()
        (lane/'jobs.json').write_text(json.dumps(items,indent=2))
        print('GPU',gpu,len(items),'jobs','first',items[0]['id'])
    (a.root/'handoff.json').write_text(json.dumps({'previous_queue':str(old),'active_lanes':[0,1,2],
        'existing_office_pid':25984,'code_snapshot':str(snap),'all_expected_jobs':len(jobs),
        'scope':'remaining primary scenes + full corrected Cambridge matrix; no AnySplat'},indent=2))


if __name__=='__main__':main()
