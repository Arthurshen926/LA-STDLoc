"""Continuously publish truthful lane status and newly completed full-test metrics."""
import argparse
import json
from pathlib import Path
import time
from common.hashing import sha256_file


def save(path,payload):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(payload,indent=2));tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--follow',action='store_true');a=p.parse_args()
    checked={}
    while True:
        lanes=[];job_states={}
        for gpu in range(3):
            if not (a.root/f'gpu{gpu}/jobs.json').is_file():continue
            path=a.root/f'gpu{gpu}/state.json'
            state=json.loads(path.read_text()) if path.exists() else {}
            job_states.update(state.get('jobs',{}))
            jobs=json.loads((a.root/f'gpu{gpu}/jobs.json').read_text())
            counts={}
            for job in jobs:
                status=state.get('jobs',{}).get(job['id'],{}).get('status','pending')
                counts[status]=counts.get(status,0)+1
            lanes.append({'gpu':gpu,'counts':counts,'heartbeat_age_seconds':time.time()-state.get('heartbeat',time.time()),
                'finished':state.get('queue_finished',False),'all_complete':state.get('all_complete',False),
                'failed_or_blocked':{k:v for k,v in state.get('jobs',{}).items() if v['status'] in ['failed','blocked','interrupted']}})
        for path in (a.root/'runs').glob('*/evaluation/*/summary.json'):
            if str(path) in checked:continue
            job_id=path.parts[-4]+'_eval'+path.parent.name.removeprefix('base_seed')
            if job_states.get(job_id,{}).get('status')!='complete':continue
            results=path.parent/'results.json'
            if not results.is_file():continue
            try:
                summary=json.loads(path.read_text());rows=json.loads(results.read_text())
                mapfile=Path(summary['input_map_path'])
                if sha256_file(mapfile)!=summary['input_map_sha256']:raise ValueError('map checksum mismatch')
                names=[r['image_name'] for r in rows]
                if len(names)!=len(set(names)):raise ValueError('duplicate test query')
                checked[str(path)]={'condition':path.parts[-4],'seed':path.parent.name,'query_count':len(rows),
                    'metrics':{k:summary.get(k) for k in ['median_te_cm','median_ae_deg','mean_te_cm','p90_te_cm','catastrophic_100cm_count','total_ms_mean']},
                    'summary_sha256':sha256_file(path),'results_sha256':sha256_file(results),'map_sha256':summary['input_map_sha256']}
            except (json.JSONDecodeError,KeyError):continue
        payload={'updated':time.time(),'lanes':lanes,'new_evaluations':checked,
                 'all_queued_jobs_complete':all(l['all_complete'] for l in lanes),
                 'scope':'outputs from this execution root only; previously completed references are not recounted',
                 'timing_caveat':'shared-host pipeline timings, not isolated real-time certification'}
        save(a.root/'live_results.json',payload)
        if not a.follow or all(l['finished'] for l in lanes):break
        time.sleep(60)


if __name__=='__main__':main()
