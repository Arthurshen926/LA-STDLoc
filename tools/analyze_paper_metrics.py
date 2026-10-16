"""Read-only, post-hoc metric sensitivity analysis of frozen Cambridge runs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from report_cambridge_suite_v2 import stage_success

POOL = Path('/mnt/pool/sqy')
SCENES = ['ShopFacade', 'OldHospital', 'KingsCollege', 'GreatCourt', 'StMarysChurch']


def read(path):
    return json.loads(path.read_text())


def error_metrics(te, re):
    te, re = np.asarray(te, dtype=float), np.asarray(re, dtype=float)
    valid = np.isfinite(te) & np.isfinite(re)
    def percentile(values, q):
        values = np.sort(values)
        index = (len(values)-1)*q/100
        lo, hi = int(np.floor(index)), int(np.ceil(index))
        if np.isinf(values[hi]):
            return float('inf')
        return float(values[lo]+(values[hi]-values[lo])*(index-lo))
    result = dict(median_TE_cm=percentile(te, 50), mean_TE_cm=float(te.mean()),
        P90_TE_cm=percentile(te, 90), P95_TE_cm=percentile(te, 95),
        median_RE_deg=percentile(re, 50), mean_RE_deg=float(re.mean()),
        P90_RE_deg=percentile(re, 90), P95_RE_deg=percentile(re, 95),
        failures=int((~valid).sum()), failure_pct=float(100*np.mean(~valid)),
        catastrophic_100cm_pct=float(100*np.mean(te >= 100)),
        successful_only_mean_TE_cm=float(te[valid].mean()) if valid.any() else float('inf'),
        successful_only_mean_RE_deg=float(re[valid].mean()) if valid.any() else float('inf'))
    for threshold in [1, 2, 5, 10, 15, 20, 25, 30, 50, 100]:
        result[f'R{threshold}_pct'] = float(100*np.mean((te < threshold) & (re < 5)))
    for t, angle in [(1,1),(2,2),(200,5),(500,10)]:
        result[f'R{t}cm_{angle}deg_pct'] = float(100*np.mean((te < t) & (re < angle)))
    return result


def analyze(output, all_conditions=False):
    output.mkdir(parents=True, exist_ok=False)
    supplement = POOL/'anygsloc_supplement_20261009'
    suite = POOL/'anygsloc_cambridge_suite_v2_20260914/runs'
    factorial = read(POOL/'anygsloc_observation_factorial_20260914/registry.json')
    baseline = POOL/'anygsloc_baseline_recovery_v4_20260915'
    rows, curves, sources = [], {}, {}
    for scene in SCENES:
        original = supplement/'ransac'/scene/'capture/results.json'
        jobs = [('complete', original, None)]
        for name in ['selector_quality', 'selector_spatial', 'classic_spatial',
                     'geometry_ray', 'geometry_depth', 'geometry_center', 'native_D']:
            jobs.append((name, supplement/'evaluations'/(scene+'_'+name)/'results.json', None))
        for label, suffix in [('mean_descriptor', '__mean_descriptor'),
                              ('single_descriptor', '__single_descriptor')]:
            jobs.append((label, suite/('ablation_'+scene+suffix)/'results.json', None))
        control = read(Path(__file__).resolve().parents[1]/'paper/AnyGSLoc_Transactions_v7/data/selector_controls'/f'{scene}.json')
        quality_id = control['evaluated_maps']['equal_size_quality']['condition']
        jobs.append(('quality_completion', suite/quality_id/'results.json', None))
        for label, config in [('full_map', 'v1_f1_c1_full'), ('no_screening', 'v0_f1_c1_compact'),
                              ('no_completion', 'v1_f1_c0_compact'), ('observation_all_off', 'v0_f0_c0_compact')]:
            entry = next(e for e in factorial if e['id'] == scene+'_'+config)
            jobs.append((label, Path(entry['result'])/'results.json', None))
        if all_conditions:
            for label in ['core','matching']:
                jobs.append(('selection_'+label, suite/control['evaluated_maps'][label]['condition']/'results.json', None))
            for budget in ['B','C']:
                jobs.append(('ransac_'+budget, supplement/'ransac'/scene/'replay'/budget/'results.json', None))
            for prior in ['vanilla_2dgs','vanilla_3dgs','i3dgs_raw','i3dgs_1k','long_lrm_raw','long_lrm_1k']:
                jobs.append(('prior_'+prior,suite/('prior_'+scene+'_'+prior)/'results.json',None))
            for entry in factorial:
                if entry['scene'] == scene:
                    jobs.append(('factorial_'+entry['id'][len(scene)+1:],Path(entry['result'])/'results.json',None))
        for method in ['STDLoc', 'ULFLoc']:
            for setting in ['on', 'off']:
                receipt = read(baseline/'receipts'/f'{method}_{scene}_mapping_semantics_{setting}_eval.json')
                native = Path(receipt['native_output'])
                if not native.is_absolute():
                    native = baseline/'baselines'/method/native
                for stage in ['coarse', 'fine']:
                    jobs.append((f'{method}_{setting}_{stage}', native/'results.json', stage))
        names = [r['image_name'] for r in read(original)]
        for label, path, stage in jobs:
            data = read(path)
            assert len(data) == len(names)
            if stage is None:
                assert [r['image_name'] for r in data] == names
                ok = np.array([r['success'] for r in data])
                te = np.array([r['translation_error_cm'] if valid else np.inf for r, valid in zip(data, ok)])
                re = np.array([r['rotation_error_deg'] if valid else np.inf for r, valid in zip(data, ok)])
            else:
                prefix = 'sparse' if stage == 'coarse' else 'dense'
                ok = np.array([stage_success(r, stage) for r in data])
                te = np.array([r[prefix+'_TE'] if valid else np.inf for r, valid in zip(data, ok)])
                re = np.array([r[prefix+'_AE'] if valid else np.inf for r, valid in zip(data, ok)])
            assert not np.isnan(te).any() and not np.isnan(re).any()
            metrics = error_metrics(te, re)
            assert metrics['failures'] == int((~ok).sum())
            rows.append(dict(condition=label, scene=scene, queries=len(data), **metrics))
            grid = np.arange(1, 101)
            curve = [float(100*np.mean((te < t) & (re < 5))) for t in grid]
            curves.setdefault(label, []).append(curve)
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    macros = []
    for label in dict.fromkeys(r['condition'] for r in rows):
        group = [r for r in rows if r['condition'] == label]
        assert {r['scene'] for r in group} == set(SCENES)
        macros.append(dict(condition=label, **{key: float(np.mean([r[key] for r in group]))
            for key in metrics if key != 'failures'}, total_failures=sum(r['failures'] for r in group)))
    for name, content in [('per_scene.csv', rows), ('macro.csv', macros)]:
        with (output/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(content[0])); writer.writeheader(); writer.writerows(content)
    (output/'recall_curves.json').write_text(json.dumps(dict(translation_thresholds_cm=list(range(1, 101)),
        rotation_threshold_deg=5, comparison='strict less-than',
        macros={k: np.mean(v, axis=0).tolist() for k, v in curves.items()}), indent=2))
    (output/'provenance.json').write_text(json.dumps(dict(
        purpose='Post-hoc descriptive metric sensitivity; no threshold optimization or primary metric replacement',
        aggregation='Equal mean of five scene statistics, including scene means and scene medians separately',
        failure_policy='Primary error metrics retain infinite errors for failures; recall includes all queries. Explicit successful_only means exclude failures and must be shown with failure counts.',
        source_hashes=sources), indent=2))
    for row in macros:
        print(row['condition'], ' '.join(f'{key}={row[key]:.3f}' for key in
            ['median_TE_cm','mean_TE_cm','P90_TE_cm','R5_pct','R10_pct','R15_pct','R50_pct']),
            'failures='+str(row['total_failures']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--all-conditions', action='store_true')
    args = parser.parse_args()
    analyze(args.output, args.all_conditions)
