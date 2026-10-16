"""Read-only audit of the frozen paper evidence; never launches experiments.

Run with the existing g4splat interpreter (NumPy/OpenCV required).
Writes only a new audit directory. Raw experiment outputs remain unchanged.
"""
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POOL = Path('/mnt/pool/sqy')
SCENES = ['GreatCourt', 'KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch']


def read(path):
    return json.loads(Path(path).read_text())


def table(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(te, re):
    return dict(TE_cm=float(np.median(te)), RE_deg=float(np.median(re)),
                R5_pct=float(100 * np.mean((te < 5) & (re < 5))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    evidence = ROOT / 'docs/paper/evidence_20260917'
    baseline = POOL / 'anygsloc_baseline_recovery_v4_20260915'
    spec = importlib.util.spec_from_file_location('reporter', ROOT / 'tools/report_cambridge_suite_v2.py')
    reporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reporter)
    from anygsloc.data.datasets import ColmapDataset
    from anygsloc.evaluation.metrics import pose_error

    mismatches = []
    source_manifest = read(evidence / 'source_manifest.json')
    for name, expected in source_manifest.items():
        path = Path(name)
        if not path.is_file() or digest(path) != expected:
            mismatches.append(name)
    references = {}
    for scene in SCENES:
        manifest = read(POOL / f'anygsloc_protocol_v2_primary_20260914/runs/Cambridge/{scene}/manifest.json')
        cameras = ColmapDataset(manifest['dataset'], load_masks=False).split('test')
        assert [c.image_name for c in cameras] == manifest['queries']
        references[scene] = (manifest, cameras)

    original = table(evidence / 'baseline_accuracy.csv')
    baseline_rows = []
    arrays = {}
    max_te_difference, max_re_difference = 0.0, 0.0
    cfg_differences = []
    for method in ['STDLoc', 'ULFLoc']:
        for scene in SCENES:
            cfgs = {}
            for semantics in ['off', 'on']:
                key = f'{method}_{scene}_mapping_semantics_{semantics}'
                receipt = read(baseline / f'receipts/{key}_eval.json')
                assert receipt['status'] == 'complete' and receipt['query_semantics'] is False
                assert receipt['seed'] == 2026 and receipt['mapping_semantics'] == semantics
                native_output = Path(receipt['native_output'])
                if not native_output.is_absolute():
                    native_output = baseline / 'baselines' / method / native_output
                result = read(native_output / 'results.json')
                manifest, cameras = references[scene]
                assert len(result) == len(cameras)
                assert all(np.allclose(row['gt_pose_w2c'], cam.pose_w2c, atol=1e-4, rtol=0)
                           for row, cam in zip(result, cameras))
                cfg = ast.parse((baseline / f'baseline_models/{key}/cfg_args').read_text(), mode='eval').body
                cfgs[semantics] = {kw.arg: ast.literal_eval(kw.value) for kw in cfg.keywords}
                for stage, prefix in [('coarse', 'sparse'), ('fine', 'dense')]:
                    valid = np.array([reporter.stage_success(row, stage) for row in result])
                    te = np.array([row[prefix + '_TE'] if ok else np.inf for row, ok in zip(result, valid)])
                    re = np.array([row[prefix + '_AE'] if ok else np.inf for row, ok in zip(result, valid)])
                    computed = metrics(te, re)
                    old = next(r for r in original if r['condition'] == key and r['stage'] == stage)
                    for new_key, old_key in [('TE_cm', 'median_te_cm'), ('RE_deg', 'median_re_deg'), ('R5_pct', 'R5_percent')]:
                        assert np.isclose(computed[new_key], float(old[old_key]), atol=1e-10, rtol=1e-10)
                    # Recompute errors from stored transforms, independently of stored error fields.
                    for row, ok in zip(result, valid):
                        if not ok:
                            continue
                        pose = row['sparse'] if stage == 'coarse' else row['dense'][-1]
                        re_check, te_check = pose_error(np.asarray(pose['pose_w2c'], dtype=np.float32),
                                                        np.asarray(row['gt_pose_w2c'], dtype=np.float32))
                        max_te_difference = max(max_te_difference, abs(te_check - row[prefix + '_TE']))
                        max_re_difference = max(max_re_difference, abs(re_check - row[prefix + '_AE']))
                    arrays[(method, scene, semantics, stage)] = (te, re)
                    baseline_rows.append(dict(method=method, scene=scene, semantics=semantics,
                                              stage=stage, queries=len(result), failures=int((~valid).sum()), **computed))
            difference = {k: [cfgs['off'].get(k), cfgs['on'].get(k)] for k in cfgs['off'].keys() | cfgs['on'].keys()
                          if cfgs['off'].get(k) != cfgs['on'].get(k)}
            cfg_differences.append(dict(method=method, scene=scene, differences=difference))

    paired = []
    for method in ['STDLoc', 'ULFLoc']:
        for scene in SCENES:
            for stage in ['coarse', 'fine']:
                off_te, off_re = arrays[(method, scene, 'off', stage)]
                on_te, on_re = arrays[(method, scene, 'on', stage)]
                off = metrics(off_te, off_re)
                on = metrics(on_te, on_re)
                off_ok, on_ok = (off_te < 5) & (off_re < 5), (on_te < 5) & (on_re < 5)
                paired.append(dict(method=method, scene=scene, stage=stage,
                                   TE_off=off['TE_cm'], TE_on=on['TE_cm'], delta_TE=on['TE_cm']-off['TE_cm'],
                                   R5_off=off['R5_pct'], R5_on=on['R5_pct'], delta_R5=on['R5_pct']-off['R5_pct'],
                                   R5_gained_queries=int((on_ok & ~off_ok).sum()),
                                   R5_lost_queries=int((off_ok & ~on_ok).sum())))

    # Every current factorial and geometry cell: query coverage, protocol, raw reduction, and map hash.
    cells = []
    factorial_root = POOL / 'anygsloc_observation_factorial_20260914'
    for entry in read(factorial_root / 'registry.json'):
        cells.append((entry['id'], entry['scene'], Path(entry['result']), Path(entry['map'])))
    geometry_root = POOL / 'anygsloc_center_control_20260915'
    jobs = read(geometry_root / 'gpu0/jobs.json')
    for job in jobs:
        cmd = job['command']
        if 'evaluate' in cmd:
            result = Path(cmd[cmd.index('--output')+1])
            scene = Path(cmd[cmd.index('--dataset')+1]).name
            cells.append((job['id'], scene, result, Path(cmd[cmd.index('--map')+1])))
    checks = []
    for name, scene, result, map_path in cells:
        manifest, summary, rows = (read(result / x) for x in ['manifest.json', 'summary.json', 'results.json'])
        ref = references[scene][0]
        assert [r['image_name'] for r in rows] == manifest['queries'] == ref['queries']
        assert manifest['seed'] == 2026 and manifest['config'] == ref['config']
        assert manifest['image_domain'] == 'rectified' and manifest['rectification_contract'] == 'opencv_same_k'
        assert digest(map_path) == manifest['map_sha256']
        te = np.array([r['translation_error_cm'] if r['success'] else np.inf for r in rows])
        re = np.array([r['rotation_error_deg'] if r['success'] else np.inf for r in rows])
        computed = metrics(te, re)
        for k, old_key in [('TE_cm', 'median_te_cm'), ('RE_deg', 'median_ae_deg'), ('R5_pct', 'recall_5cm_5deg_percent')]:
            assert np.isclose(computed[k], summary[old_key], atol=1e-10, rtol=1e-10)
        checks.append(dict(condition=name, scene=scene, queries=len(rows), **computed))
    assert len(baseline_rows) == 40 and len(checks) == 95

    # Factor effects are conditional comparisons, not claims of significance.
    fm = table(evidence / 'factorial_macro.csv')
    effects = []
    for factor in ['V2', 'fusion', 'completion']:
        others = [x for x in ['V2', 'fusion', 'completion', 'selection'] if x != factor]
        for off in [r for r in fm if r[factor] == '0']:
            on = next(r for r in fm if r[factor] == '1' and all(r[k] == off[k] for k in others))
            effects.append(dict(factor=factor, **{k: off[k] for k in others},
                                delta_TE_cm=float(on['TE_cm'])-float(off['TE_cm']),
                                delta_R5_pp=float(on['R5_pct'])-float(off['R5_pct']),
                                anchors_off=int(off['total_anchors']), anchors_on=int(on['total_anchors'])))
    write_csv(args.output / 'baseline_recomputed.csv', baseline_rows)
    write_csv(args.output / 'baseline_paired.csv', paired)
    write_csv(args.output / 'ablation_recomputed.csv', checks)
    # Conditional columns differ by factor; JSON preserves this without fabricating empty cells.
    (args.output / 'factorial_effects.json').write_text(json.dumps(effects, indent=2))
    result = dict(source_files=len(source_manifest), source_hash_mismatches=mismatches,
                  baseline_stage_cells=len(baseline_rows), ablation_cells=len(checks),
                  baseline_errors_recomputed_from_poses=dict(max_TE_difference_cm=max_te_difference,
                                                            max_RE_difference_deg=max_re_difference),
                  paired_cfg_args_differences=cfg_differences,
                  solver_iterations_saved_in_original_results=False,
                  scope='Existing frozen artifacts only; no new localization or training runs; no statistical significance claim.')
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'paired_cfg_args_differences'}, indent=2))


if __name__ == '__main__':
    main()
