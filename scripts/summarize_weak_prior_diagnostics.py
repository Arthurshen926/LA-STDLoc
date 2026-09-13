"""Rebuild the bounded weak-prior experiment ledger from completed artifacts."""
import argparse
import json
from pathlib import Path
import numpy as np
from common.hashing import sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--include-no-v2',action='store_true')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    rows=[];missing=[]
    for arm in ['OldHospital_vanilla_2dgs','OldHospital_matcha']:
        directory=args.root/arm/'evaluation'
        base=json.loads((directory/'base_seed2026/results.json').read_text())
        names=[x['image_name'] for x in base];x=np.array([v['translation_error_cm'] for v in base])
        variants=['base_seed2026','real_descriptor_diagnostic_seed2026',
                  'geometry_descriptor_within_bin_seed2026','geometry_descriptor_across_bin_seed2026',
                  'depth_geometry_control_seed2026','surface_geometry_gate_seed2026']
        if args.include_no_v2:variants+=['no_v2_control_seed2026']
        if arm.endswith('vanilla_2dgs'):
            variants+=['base_seed2027','depth_geometry_control_seed2027',
                       'multiview_keypoint_adjustment_panel2048_seed2026',
                       'multiview_keypoint_adjustment_full_seed2026']
        for condition in variants:
            path=directory/condition
            if not (path/'summary.json').is_file() or not (path/'results.json').is_file():
                missing.append(str(path));continue
            summary=json.loads((path/'summary.json').read_text());results=json.loads((path/'results.json').read_text())
            if names!=[r['image_name'] for r in results]:raise ValueError('test registry differs')
            if sha256_file(Path(summary['input_map_path']))!=summary['input_map_sha256']:raise ValueError('evaluated map changed')
            y=np.array([r['translation_error_cm'] for r in results]);reference=x
            reference_condition='base_seed2026'
            if condition.endswith('seed2027'):
                ref=json.loads((directory/'base_seed2027/results.json').read_text())
                if names!=[r['image_name'] for r in ref]:raise ValueError('seed registry differs')
                reference=np.array([r['translation_error_cm'] for r in ref]);reference_condition='base_seed2027'
            rng=np.random.default_rng(20260907);index=rng.integers(0,len(y),(2000,len(y)))
            delta=np.median(y[index],axis=1)-np.median(reference[index],axis=1)
            rows.append({'arm':arm,'condition':condition,'reference_condition':reference_condition,
                'query_count':len(results),'summary_sha256':sha256_file(path/'summary.json'),
                'results_sha256':sha256_file(path/'results.json'),
                'map_sha256':summary['input_map_sha256'],
                'metrics':{k:summary[k] for k in ['median_te_cm','mean_te_cm','p90_te_cm','median_ae_deg',
                    'mean_ae_deg','catastrophic_100cm_count','matching_ms_mean','total_ms_mean']},
                'source_rgb_diagnostic':summary.get('source_rgb_diagnostic_opt_in',False),
                'te_improved_worsened_unchanged':[int(sum(y<reference)),int(sum(y>reference)),int(sum(y==reference))],
                'descriptive_paired_bootstrap_median_te_delta_cm_95':np.percentile(delta,[2.5,97.5]).tolist()})
    report={'schema':'anygsloc_weak_prior_controlled_diagnostics_ledger','version':1,
            'development_is_test_adaptive':True,'independent_generalization_claim':False,
            'bootstrap_caveat':'descriptive iid query resampling, no sequence-dependence or selection correction',
            'timing_caveat':'shared machine, Base protocol, not dedicated 10 Hz certification',
            'all_expected_evaluations_completed':not missing,'missing':missing,'rows':rows}
    with args.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print('Completed evaluations:',len(rows),'Missing:',len(missing),flush=True)
    if missing:print('\n'.join(missing))


if __name__=='__main__':main()
