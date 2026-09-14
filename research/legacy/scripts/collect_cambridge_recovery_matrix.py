"""Merge original and recovery results, requiring both seeds and exact test lists."""
import argparse
import json
import time
from pathlib import Path
from common.hashing import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--follow',action='store_true');a=p.parse_args()
    pool=Path('/mnt/pool/sqy');output=pool/'anygsloc_invalid_primitive_recovery_20260910/cambridge_matrix.json'
    roots=[pool/n for n in ['anygsloc_three_gpu_20260910','anygsloc_invalid_primitive_recovery_20260910','anygsloc_shop_tile_recovery_20260910']]
    scenes=['GreatCourt','KingsCollege','OldHospital','ShopFacade','StMarysChurch']
    methods=['vanilla_2dgs','vanilla_3dgs','long_lrm_raw','long_lrm_1k','i3dgs_raw','i3dgs_1k'];cache={}
    while True:
        certified={}
        for root in roots:
            ledger=root/'live_results.json'
            if ledger.exists():certified.update(json.loads(ledger.read_text()).get('new_evaluations',{}))
        rows=[]
        for scene in scenes:
            expected={line.split()[0] for line in (pool/f'Cambridge_stdloc/{scene}/dataset_test.txt').read_text().splitlines() if line.strip() and line.split()[0].endswith('.png')}
            if not expected:raise ValueError('empty test split')
            for method in methods:
                condition=scene+'_'+method;row={'condition':condition,'expected_test_queries':len(expected),'seeds':{}}
                for seed in [2026,2027]:
                    candidates=[Path(path) for path,v in certified.items() if v['condition']==condition and v['seed']==f'base_seed{seed}']
                    if method=='vanilla_2dgs' and scene in ['OldHospital','KingsCollege']:
                        folder='OldHospital' if scene=='OldHospital' else 'KingsCollege_allocator_retry'
                        candidates.append(pool/f'anygsloc_rectified_prior_retrain_20260908/{folder}/anygsloc/evaluation/base_seed{seed}/summary.json')
                    if not candidates:continue
                    path=candidates[-1]
                    if not path.is_file():continue
                    if str(path) not in cache:
                        summary=json.loads(path.read_text());results=path.parent/'results.json'
                        queries=json.loads(results.read_text());names=[q['image_name'] for q in queries]
                        if len(names)!=len(expected) or set(names)!=expected:raise ValueError(f'incomplete or duplicate test queries: {path}')
                        mapfile=Path(summary['input_map_path'])
                        if sha256_file(mapfile)!=summary['input_map_sha256']:raise ValueError('map hash mismatch')
                        cache[str(path)]={'summary':str(path),'summary_sha256':sha256_file(path),'results_sha256':sha256_file(results),
                          'query_count':len(names),'median_te_cm':summary['median_te_cm'],'median_re_deg':summary['median_ae_deg'],
                          'catastrophic_100cm_count':summary['catastrophic_100cm_count'],'mean_ms':summary['total_ms_mean']}
                    row['seeds'][str(seed)]=cache[str(path)]
                row['complete']=len(row['seeds'])==2;rows.append(row)
        payload={'updated':time.time(),'complete_cells':sum(r['complete'] for r in rows),'expected_cells':30,
                 'missing_cells':[r['condition'] for r in rows if not r['complete']],'rows':rows,
                 'scope':'fixed corrected Cambridge six-prior-condition matrix, both seeds; excludes supplemental two-epoch control'}
        tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(payload,indent=2));tmp.replace(output)
        print(payload['complete_cells'],'/30',payload['missing_cells'],flush=True)
        if not a.follow or payload['complete_cells']==30:return
        time.sleep(60)


if __name__=='__main__':main()
