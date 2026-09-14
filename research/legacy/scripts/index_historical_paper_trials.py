"""Index historical trials separately; these summaries are not newly certified tests."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    roots=[(Path('/mnt/pool/sqy/lafgs_v24_cambridge_high_capacity_online_20260901'),'historical_mAtCHA_online_exploration'),
           (Path('/mnt/pool/sqy/anygsloc_paper_experiments_20260902/CambridgePrior'),'historical_weak_prior_mixed_protocol_not_main_table')]
    rows=[]
    for root,scope in roots:
        for path in sorted(root.glob('*/evaluation/*/summary.json')):
            d=json.loads(path.read_text());rows.append({'scope':scope,'condition':path.parent.name,'scene':path.parents[2].name,
                'summary':str(path),'query_count':d.get('query_count'),'median_te_cm':d.get('median_te_cm'),
                'median_re_deg':d.get('median_ae_deg'),'mean_te_cm':d.get('mean_te_cm'),'p90_te_cm':d.get('p90_te_cm'),
                'recall_5cm5deg':d.get('recall_5cm_5deg_percent'),'cat100':d.get('catastrophic_100cm_count'),
                'recorded_mean_ms':d.get('total_ms_mean'),'newly_verified_full_test':False,
                'warning':'Exploratory or obsolete protocol; may use bug-affected priors. Do not mix with corrected main tables or select best trial.'})
    with a.output.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('historical summaries indexed',len(rows),flush=True)


if __name__=='__main__':main()
