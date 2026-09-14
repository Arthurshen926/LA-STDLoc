"""Relate frozen mapping geometry evidence to post-solve test diagnostics.

GT projection proximity is NOT verified physical feature identity. No output
of this script is suitable for source-image-free map construction.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize_matches(errors, inliers):
    errors, inliers = np.asarray(errors), np.asarray(inliers, dtype=bool)
    def rate(x):
        return float(np.mean(x) * 100) if len(x) else None
    return {'matches': len(errors), 'inliers': int(inliers.sum()),
            'gt_within_4px_percent': rate(errors <= 4),
            'inlier_gt_within_4px_percent': rate(errors[inliers] <= 4),
            'correct_inliers_4px': int(np.sum(inliers & (errors <= 4)))}


def analyze(geometry, summary, audit):
    if geometry['input']['map_sha256'] != summary['input_map_sha256']:
        raise ValueError('geometry/query map identity differs')
    if str(audit['schema']) != 'anygsloc_postsolve_gt_diagnostic_v1' or not bool(audit['uses_test_gt']):
        raise ValueError('unknown query diagnostic schema')
    if len(audit['query_names']) != summary['query_count']:
        raise ValueError('query registry length differs')
    owners = audit['anchor_rows']
    error, inlier = audit['gt_reprojection_px'], audit['inlier']
    count = geometry['anchor_count']
    if owners.ndim != 1 or error.shape != owners.shape or inlier.shape != owners.shape or np.any(owners < 0) or np.any(owners >= count):
        raise ValueError('invalid match owner rows')
    labels = np.full(count, 'not_sampled', dtype=object)
    lengths = np.zeros(count, dtype=int)
    seen = set()
    for row in geometry['rows']:
        anchor = row['anchor_row']
        if anchor in seen or not 0 <= anchor < count:
            raise ValueError('invalid geometry row registry')
        seen.add(anchor)
        labels[anchor] = row['status']
        lengths[anchor] = row['full_observation_count']
    output = {'uses_test_gt': True, 'deployment_eligible': False,
              'physical_identity_verified': False, 'map_sha256': summary['input_map_sha256'],
              'groups': {}, 'length_strata': {}}
    for status in ['internally_consistent', 'unstable', 'unknown', 'not_sampled']:
        mask = labels[owners] == status
        output['groups'][status] = summarize_matches(error[mask], inlier[mask])
    for lo, hi in [(3, 6), (6, 10), (10, 20), (20, 1000000)]:
        strata = {}
        for status in ['internally_consistent', 'unstable', 'unknown']:
            mask = (labels[owners] == status) & (lengths[owners] >= lo) & (lengths[owners] < hi)
            strata[status] = summarize_matches(error[mask], inlier[mask])
        output['length_strata'][f'{lo}-{hi-1}'] = strata
    # Paired within-query rates avoid conflating a category with hard queries.
    offsets = audit['offsets']
    if len(offsets) != len(audit['query_names']) + 1 or offsets[0] != 0 or offsets[-1] != len(owners) or np.any(np.diff(offsets) < 0):
        raise ValueError('invalid query CSR')
    differences = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        label = labels[owners[start:end]]
        good, bad = label == 'internally_consistent', label == 'unstable'
        if good.sum() >= 5 and bad.sum() >= 5:
            ok = error[start:end] <= 4
            differences.append(float(ok[bad].mean() - ok[good].mean()))
    output['within_query_comparison'] = {
        'minimum_matches_per_group': 5, 'query_count': len(differences),
        'mean_unstable_minus_consistent_precision_pp': float(np.mean(differences)*100) if differences else None}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    geometry = json.loads(args.geometry.read_text())
    summary = json.loads((args.evaluation/'summary.json').read_text())
    with np.load(args.evaluation/'correspondence_audit.npz', allow_pickle=False) as audit:
        result = analyze(geometry, summary, audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
