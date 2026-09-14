"""Compare paired seed-2026 results; refuse changed maps or query lists."""
import json
from pathlib import Path

BASE = Path('/mnt/pool/sqy/anygsloc_release_validation_20260913/runs/Cambridge')
ROOT = Path('/mnt/pool/sqy/anygsloc_cambridge_no_query_mask_20260913')


def main():
    comparisons=[]
    for scene in ['GreatCourt','KingsCollege','OldHospital','ShopFacade','StMarysChurch']:
        new=ROOT/'runs'/scene
        if not (new/'summary.json').exists():
            print(scene, 'pending')
            continue
        old=BASE/scene/'2026'
        a=json.loads((old/'manifest.json').read_text())
        b=json.loads((new/'manifest.json').read_text())
        for key in ['map_sha256','queries','calibration','seed','images','dataset','image_domain']:
            assert a[key]==b[key], (scene,key)
        ca=dict(a['config']);cb=dict(b['config'])
        assert ca.pop('mask_policy')=='explicit_valid_mask'
        assert cb.pop('mask_policy')=='none'
        assert ca==cb, scene
        for path in [old,new]:
            results=json.loads((path/'results.json').read_text())
            assert [r['image_name'] for r in results]==a['queries']
        sa=json.loads((old/'summary.json').read_text())
        sb=json.loads((new/'summary.json').read_text())
        row=dict(scene=scene,queries=len(a['queries']),seed=2026)
        for key in ['median_te_cm','median_ae_deg','recall_5cm_5deg_percent',
                    'recall_15cm_5deg_percent','recall_50cm_5deg_percent',
                    'catastrophic_100cm_count','runtime_ms_mean']:
            row[key]=dict(masked=sa[key],unmasked=sb[key],delta=sb[key]-sa[key])
        comparisons.append(row)
    report=dict(complete=len(comparisons)==5,scenes=comparisons,
                timing='telemetry_not_formal',intervention='disable_all_query_masks_including_distortion_border')
    if comparisons:
        report['macro']={key:{mode:sum(row[key][mode] for row in comparisons)/len(comparisons)
                             for mode in ['masked','unmasked','delta']}
                         for key in ['median_te_cm','median_ae_deg','recall_5cm_5deg_percent']}
    (ROOT/'comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
