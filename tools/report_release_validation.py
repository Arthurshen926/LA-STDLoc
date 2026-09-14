"""Audit full query coverage before including a release run in a result table."""
import argparse
import csv
import json
from pathlib import Path


def checked_run(folder):
    manifest = json.loads((folder/'manifest.json').read_text())
    rows = json.loads((folder/'results.json').read_text())
    names = [r['image_name'] for r in rows]
    if names != manifest['queries'] or len(names) != len(set(names)):
        raise ValueError('query coverage/order differs from manifest')
    summary = json.loads((folder/'summary.json').read_text())
    if summary['query_count'] != len(names):
        raise ValueError('summary denominator differs from results')
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    registry = json.loads((a.root/'registry.json').read_text())
    rows = []
    for cell in registry['cells']:
        for seed in (2026, 2027, 2028):
            folder = a.root/'runs'/cell['key']/str(seed)
            row = dict(scene=cell['key'], prior=cell['prior'], seed=seed, status='pending')
            if (folder/'summary.json').exists():
                try:
                    summary = checked_run(folder)
                    row.update(status='complete', **{k: summary[k] for k in (
                        'query_count','median_te_cm','median_ae_deg','p90_te_cm',
                        'recall_5cm_5deg_percent','catastrophic_100cm_count','runtime_ms_mean')})
                except (ValueError, KeyError, OSError) as exc:
                    row.update(status='invalid', error=str(exc))
            elif (folder/'progress.jsonl').exists():
                row.update(status='partial', queries_written=sum(1 for _ in (folder/'progress.jsonl').open()))
            rows.append(row)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (a.root/'results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    status = dict(registered=len(rows), complete=sum(r['status']=='complete' for r in rows),
                  invalid=sum(r['status']=='invalid' for r in rows),
                  timing='parallel_unwarmed_telemetry_not_formal',
                  release_evidence='provisional_pending_historical_driver_and_input_hash_audit', rows=rows)
    tmp = a.root/'report_status.tmp'
    tmp.write_text(json.dumps(status, indent=2))
    tmp.replace(a.root/'report_status.json')
    print(json.dumps({k:v for k,v in status.items() if k!='rows'}, indent=2))


if __name__ == '__main__':
    main()
