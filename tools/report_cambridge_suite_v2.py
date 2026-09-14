"""Generate separate accuracy/cost evidence tables, preserving pending cells."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def stage_success(row, stage):
    def count(pose):
        value=pose['inliers']
        return value if isinstance(value,(int,float)) else len(value)
    previous=row['sparse'];valid=count(previous)>0
    if stage=='coarse':return valid
    for current in row['dense']:
        # Native refinement may deliberately return the preceding valid pose
        # with zero NEW inliers. Do not count that fallback as a pose failure.
        valid=count(current)>0 or (valid and np.array_equal(current['pose_w2c'],previous['pose_w2c']))
        previous=current
    return valid


def report(root):
    state=json.loads((root/'gpu0/state.json').read_text()) if (root/'gpu0/state.json').exists() else {'jobs':{}}
    jobs=json.loads((root/'gpu0/jobs.json').read_text())
    accuracy=[]
    primary=Path('/mnt/pool/sqy/anygsloc_protocol_v2_primary_20260914/runs/Cambridge')
    for p in primary.glob('*/summary.json'):
        summary=json.loads(p.read_text())
        manifest=json.loads((p.parent/'manifest.json').read_text())
        rows=json.loads((p.parent/'results.json').read_text())
        if [r['image_name'] for r in rows]!=manifest['queries']:
            raise ValueError('primary query coverage mismatch')
        accuracy.append(dict(condition='AnyGSLoc_'+p.parent.name,stage='single',query_count=len(rows),
            median_te_cm=summary['median_te_cm'],median_re_deg=summary['median_ae_deg'],
            R5_percent=summary['recall_5cm_5deg_percent']))
    for p in (root/'runs').glob('*/summary.json'):
        manifest=json.loads((p.parent/'manifest.json').read_text())
        rows=json.loads((p.parent/'results.json').read_text())
        if [r['image_name'] for r in rows]!=manifest['queries']:
            raise ValueError('incomplete query coverage: '+str(p))
        summary=json.loads(p.read_text())
        accuracy.append(dict(condition=p.parent.name,stage='single',query_count=len(rows),
            median_te_cm=summary['median_te_cm'],median_re_deg=summary['median_ae_deg'],
            R5_percent=summary['recall_5cm_5deg_percent']))
    baseline_errors=[]
    for p in (root/'receipts').glob('*_eval.json'):
        receipt=json.loads(p.read_text())
        if receipt['status']!='complete':continue
        output=Path(receipt['native_output'])
        if not output.is_absolute():output=root/'baselines'/receipt['method']/output
        rows=json.loads((output/'results.json').read_text())
        if not rows:raise ValueError('empty native results')
        key=p.stem.removesuffix('_eval')
        scene=key.split('_mapping_semantics_')[0].split('_',1)[1]
        reference=Path('/mnt/pool/sqy/anygsloc_protocol_v2_primary_20260914/runs/Cambridge')/scene
        manifest=json.loads((reference/'manifest.json').read_text())
        if len(rows)!=len(manifest['queries']):
            baseline_errors.append(dict(condition=key,error='query_count_mismatch'));continue
        # Pinned native readers sort test images by name. Validate their GT order.
        from anygsloc.data.datasets import ColmapDataset
        data=ColmapDataset(manifest['dataset'],load_masks=False)
        cameras=data.split('test')
        if not all(np.allclose(row['gt_pose_w2c'],cam.pose_w2c,atol=1e-4,rtol=0)
                   for row,cam in zip(rows,cameras)):
            baseline_errors.append(dict(condition=key,error='native_GT_order_mismatch'));continue
        for stage,te_key,re_key in [('coarse','sparse_TE','sparse_AE'),('fine','dense_TE','dense_AE')]:
            te=[];re=[]
            for row in rows:
                valid=stage_success(row,stage)
                te.append(row[te_key] if valid else float('inf'))
                re.append(row[re_key] if valid else float('inf'))
            accuracy.append(dict(condition=key,stage=stage,query_count=len(rows),
                median_te_cm=statistics.median(te),median_re_deg=statistics.median(re),
                R5_percent=100*sum(t<5 and r<5 for t,r in zip(te,re))/len(rows)))
    costs=[]
    for p in (root/'receipts').glob('*.json'):
        value=json.loads(p.read_text())
        if value.get('status')!='complete':continue
        row=dict(condition=p.stem,source=str(p))
        for key in ['wall_seconds','peak_process_tree_rss_bytes','peak_rss_kib','peak_torch_gpu_bytes']:
            if key in value:row[key]=value[key]
        if value.get('build_stages'):
            row['mapping_postprocessing_seconds']=sum(x['wall_seconds'] for x in value['build_stages'])
            row['build_cost_scope']=value['build_cost_scope']
            row['segmentation_generation_cost']=value['segmentation_generation_cost']
        if 'assets' in value:
            row['supplied_assets_bytes']=sum(x['bytes'] for x in value['assets'])
            row['storage_scope']='native_directory_inventory_not_minimal_deployment'
        timings=value.get('timings',[])
        if timings:
            for key in ['coarse_prefix_ms','total_ms','fine_increment_ms']:
                row[key+'_mean']=statistics.mean(x[key] for x in timings)
            row['warmup']=value.get('warmup_queries',0)
        costs.append(row)
    for p in (root/'deployment_benchmark').glob('*/summary.json'):
        value=json.loads(p.read_text())
        if not value.get('complete'):continue
        costs.append(dict(condition='ours_'+p.parent.name,source=str(p),
            runtime_ms_mean=statistics.mean(r['runtime_ms_mean'] for r in value['rounds']),
            runtime_ms_p95=statistics.mean(r['runtime_ms_p95'] for r in value['rounds']),
            map_bytes=value['map_bytes'],initialization_seconds=value['initialization_seconds'],
            peak_torch_gpu_bytes=max(r['peak_torch_gpu_allocated_bytes'] for r in value['rounds']),warmup=50))
    for p in (root/'conversion_cost').glob('*/logs/*.json'):
        value=json.loads(p.read_text())
        if value.get('status')=='complete':
            costs.append(dict(condition='conversion_'+p.parent.parent.name+'_'+p.stem,
                wall_seconds=value['wall_seconds'],source=str(p),build_cost_scope='given_fixed_gaussian_prior'))
    for name,rows in [('accuracy',accuracy),('costs',costs)]:
        if rows:
            fields=list(dict.fromkeys(k for row in rows for k in row))
            with (root/(name+'.csv')).open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    statuses={job['id']:state['jobs'].get(job['id'],{}).get('status','pending') for job in jobs}
    status=dict(registered=len(jobs),counts={s:list(statuses.values()).count(s) for s in set(statuses.values())},
        queue_finished=state.get('queue_finished',False),all_complete=all(v=='complete' for v in statuses.values()),
        baseline_result_errors=baseline_errors,accuracy_rows=len(accuracy),cost_rows=len(costs))
    tmp=root/'report_status.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(root/'report_status.json')
    return status


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--watch',action='store_true')
    a=p.parse_args()
    while True:
        result=report(a.root);print(json.dumps(result),flush=True)
        if not a.watch or result['queue_finished']:break
        time.sleep(30)


if __name__=='__main__':main()
