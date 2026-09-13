"""Attested full-test ledger for the bounded 2026-09-08 follow-up."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from common.hashing import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-root', type=Path, required=True)
    p.add_argument('--kings-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--include-repeat-seed', action='store_true')
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    arms = [(args.old_root / 'OldHospital_vanilla_2dgs', ['representative_balanced_medoid', 'representative_nearest_f0', 'refit_control', 'radial_compensated']),
            (args.old_root / 'OldHospital_matcha', ['representative_balanced_medoid', 'representative_nearest_f0']),
            (args.kings_root, ['depth_geometry_control', 'refit_control', 'radial_compensated'])]
    rows = []
    for directory, variants in arms:
        base = json.loads((directory / 'evaluation/base_seed2026/results.json').read_text())
        names = [r['image_name'] for r in base]
        x = np.array([r['translation_error_cm'] for r in base])
        conditions = ['base_seed2026'] + [v + '_20260908_seed2026' for v in variants]
        if args.include_repeat_seed and directory.name == 'OldHospital_vanilla_2dgs':
            conditions += ['base_seed2027', 'radial_compensated_20260908_seed2027']
        if args.include_repeat_seed and directory == args.kings_root:
            conditions += [v + '_20260908_seed2027' for v in ['base', 'radial_compensated', 'depth_geometry_control']]
        for condition in conditions:
            path = directory / 'evaluation' / condition
            summary = json.loads((path / 'summary.json').read_text())
            results = json.loads((path / 'results.json').read_text())
            if [r['image_name'] for r in results] != names: raise ValueError('query registry differs')
            if sha256_file(Path(summary['input_map_path'])) != summary['input_map_sha256']: raise ValueError('map changed since evaluation')
            y = np.array([r['translation_error_cm'] for r in results])
            reference = x
            if condition.endswith('seed2027'):
                reference_name = 'base_20260908_seed2027' if directory == args.kings_root else 'base_seed2027'
                reference_rows = json.loads((directory / 'evaluation' / reference_name / 'results.json').read_text())
                if names != [r['image_name'] for r in reference_rows]: raise ValueError('repeat seed registry differs')
                reference = np.array([r['translation_error_cm'] for r in reference_rows])
            ix = np.random.default_rng(20260908).integers(len(x), size=(2000, len(x)))
            delta = np.median(y[ix], axis=1) - np.median(reference[ix], axis=1)
            rows.append({'arm': directory.name, 'condition': condition, 'query_count': len(x),
                         'source_summary': str(path / 'summary.json'), 'summary_sha256': sha256_file(path / 'summary.json'),
                         'results_sha256': sha256_file(path / 'results.json'), 'map_sha256': summary['input_map_sha256'],
                         'metrics': {k: summary[k] for k in ['median_te_cm','median_ae_deg','mean_te_cm','p90_te_cm','catastrophic_100cm_count','matching_ms_mean','total_ms_mean']},
                         'te_improved_worsened_unchanged': [int((y < reference).sum()), int((y > reference).sum()), int((y == reference).sum())],
                         'descriptive_paired_bootstrap_median_te_delta_95_cm': np.quantile(delta, [.025,.975]).tolist()})
    report = {'schema': 'anygsloc_rectification_followup_20260908', 'new_full_evaluations': 13 if args.include_repeat_seed else 9,
              'development_is_test_adaptive': True, 'independent_generalization_claim': False,
              'all_expected_evaluations_completed': True,
              'timing_caveat': 'shared hardware, Base protocol, not 10 Hz certification',
              'bootstrap_caveat': 'descriptive iid query resampling, no trajectory dependence or selection correction', 'rows': rows}
    geometry_audits = []
    for directory, _ in [arms[0], arms[2]]:
        original = torch.load(directory / 'projective_map/projective_anchor_map.pt', map_location='cpu', weights_only=False)
        revised = torch.load(directory / 'radial_compensation_20260908/radial_compensated/projective_anchor_map.pt', map_location='cpu', weights_only=False)
        def exact(a, b):
            if isinstance(a, torch.Tensor): return torch.equal(a,b)
            if isinstance(a, dict): return a.keys() == b.keys() and all(exact(a[k],b[k]) for k in a)
            return a == b
        invariants = {k: exact(original[k], revised[k]) for k in ['anchor_ids','anchor_features','anchor_observation_features','projective_anchor_observations']}
        if not all(invariants.values()): raise ValueError('fixed-map invariant failed')
        detail = {'arm': directory.name, 'invariants': invariants}
        before_name = 'geometry_diagnostic_seed2026' if directory == arms[0][0] else 'base_20260908_seed2027'
        before_path = directory / 'evaluation' / before_name / 'correspondence_audit.npz'
        if before_path.exists():
            before = np.load(before_path)
            after = np.load(directory / 'evaluation/radial_compensated_20260908_seed2026/correspondence_audit.npz')
            equal = {k: np.array_equal(before[k],after[k]) for k in ['query_names','offsets','anchor_rows']}
            if not all(equal.values()): raise ValueError('raw correspondences differ')
            m = before['inlier'].astype(bool)
            detail.update(raw_matches_unchanged=True, raw_match_count=len(m), reference_inlier_count=int(m.sum()),
                          reference_audit_condition=before_name,
                          fixed_inlier_gt_px_before_p50_p90=np.quantile(before['gt_reprojection_px'][m],[.5,.9]).tolist(),
                          fixed_inlier_gt_px_after_p50_p90=np.quantile(after['gt_reprojection_px'][m],[.5,.9]).tolist(),
                          fixed_inlier_within4_before_after=[int((before['gt_reprojection_px'][m]<=4).sum()),int((after['gt_reprojection_px'][m]<=4).sum())],
                          uses_test_gt_postsolve_only=True)
        geometry_audits.append(detail)
    report['fixed_geometry_change_audits'] = geometry_audits
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    for r in rows: print(r['arm'], r['condition'], r['metrics'], r['te_improved_worsened_unchanged'], flush=True)


if __name__ == '__main__': main()
