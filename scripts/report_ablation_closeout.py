"""Report registered completion and all available paired results without cherry-picking."""
import argparse,json,time
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--watch',action='store_true');a=p.parse_args()
    if a.watch:
        import subprocess,sys
        while True:
            subprocess.run([sys.executable,'-m','scripts.report_ablation_closeout','--output',str(a.output)],check=True)
            lanes=[Path('/mnt/pool/sqy')/root/f'gpu{gpu}/state.json' for root in ['anygsloc_final_ablation_20260913','anygsloc_depth_ablation_20260913'] for gpu in range(3)]
            if all(path.is_file() and json.loads(path.read_text()).get('queue_finished') for path in lanes):
                return
            time.sleep(60)
    roots=[Path('/mnt/pool/sqy')/n for n in ['anygsloc_ablation_closeout_20260911','anygsloc_final_ablation_20260913','anygsloc_depth_ablation_20260913']]
    groups={};missing=[];complete=0;registered=0
    for root in roots:
        for r in json.loads((root/'registry.json').read_text()):
            registered+=1;out=Path(r['output']);summary=out/'summary.json';results=out/'results.json'
            if not summary.is_file() or not results.is_file():missing.append(r);continue
            s=json.loads(summary.read_text());rows=json.loads(results.read_text())
            if len(rows)!=s['query_count']:raise ValueError(f'query count mismatch: {out}')
            names=[x['image_name'] for x in rows]
            if len(set(names))!=len(names):raise ValueError(f'duplicate test query: {out}')
            te=np.array([x['translation_error_cm'] for x in rows]);re=np.array([x['rotation_error_deg'] for x in rows])
            metrics={k:s[k] for k in ['median_te_cm','median_ae_deg','p90_te_cm','catastrophic_100cm_count','runtime_ms_mean']}
            for cm in [5,10,15,50]:metrics[f'R{cm}']=100*float(np.mean((te<cm)&(re<5)))
            condition=(r.get('group','')+'_'+r['condition']).strip('_')
            import re as regex
            condition=regex.sub(r'_?(seed|eval)202[678]$','',condition)
            key=(root.name,r['scene'],condition);groups.setdefault(key,[]).append(dict(seed=r['seed'],metrics=metrics,output=str(out),query_names=names));complete+=1
    lines=['# AnyGSLoc ablation closeout status','',f'Completed {complete}/{registered} registered evaluations. Missing {len(missing)}.','',
           'Timing is concurrent throughput telemetry; formal latency needs isolated measurement. Three seeds measure solver randomness, not independent test sets.','',
           '| Source | Scene | Condition | Seeds | TE cm | RE deg | R5 % | R15 % | R50 % | P90 TE | >=1m |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    report=[]
    for key,runs in sorted(groups.items()):
        if any(x['query_names']!=runs[0]['query_names'] for x in runs):raise ValueError('query registry drift')
        values={k:float(np.mean([r['metrics'][k] for r in runs])) for k in runs[0]['metrics']}
        report.append(dict(source=key[0],scene=key[1],condition=key[2],seeds=[r['seed'] for r in runs],metrics=values))
        nums=[values[k] for k in ['median_te_cm','median_ae_deg','R5','R15','R50','p90_te_cm','catastrophic_100cm_count']]
        lines.append('| '+' | '.join([*key,str(len(runs)),*[f'{n:.3f}' for n in nums]])+' |')
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'status.json').write_text(json.dumps(dict(complete=complete,registered=registered,missing=missing,results=report),indent=2))
    (a.output/'tables.md').write_text('\n'.join(lines)+'\n')
    print(f'{complete}/{registered}; {a.output}/tables.md')


if __name__=='__main__':main()
