"""Materialize compact, reproducible paper evidence inside the Git checkout."""
import csv
import hashlib
import json
from pathlib import Path
import statistics as stats

POOL = Path('/mnt/pool/sqy')
OUT = Path(__file__).resolve().parents[1] / 'docs/paper/evidence_20260917'
SOURCES = {}


def read(path):
    data = path.read_bytes()
    SOURCES[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def csv_read(path):
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return list(csv.DictReader(path.open()))


def write_csv(name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / (name + '.csv')).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def table(rows, fields):
    lines = ['| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join(['---'] * len(fields)) + ' |']
    for row in rows:
        values = [f'{row[k]:.4f}' if isinstance(row[k], float) else str(row[k]) for k in fields]
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def accuracy(summary):
    return dict(queries=summary['query_count'], TE_cm=summary['median_te_cm'],
        RE_deg=summary['median_ae_deg'], R5_pct=summary['recall_5cm_5deg_percent'],
        R15_pct=summary['recall_15cm_5deg_percent'], P90_TE_cm=summary['p90_te_cm'],
        failures=summary['failed_query_count'])


def macro(rows):
    return {k: stats.mean(float(x[k]) for x in rows) for k in ['TE_cm', 'RE_deg', 'R5_pct']}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    primary_root = POOL / 'anygsloc_protocol_v2_primary_20260914'
    baseline_root = POOL / 'anygsloc_baseline_recovery_v4_20260915'
    factorial_root = POOL / 'anygsloc_observation_factorial_20260914'
    center_root = POOL / 'anygsloc_center_control_20260915'
    for root in [primary_root, baseline_root, factorial_root, center_root]:
        state = read(root / 'gpu0/state.json')
        assert state['all_complete'], root
    primary = []
    for f in sorted((primary_root / 'runs').glob('*/*/summary.json')):
        summary = read(f)
        manifest = read(f.parent / 'manifest.json')
        results = read(f.parent / 'results.json')
        assert [x['image_name'] for x in results] == manifest['queries']
        assert len(results) == summary['query_count']
        primary.append(dict(dataset=f.parent.parent.name, scene=f.parent.name, **accuracy(summary)))
    assert len(primary) == 24
    write_csv('primary_24', primary)
    datasets = [dict(dataset=d, **macro([x for x in primary if x['dataset'] == d]))
                for d in ['Cambridge', '7Scenes', '12Scenes']]
    baseline = csv_read(baseline_root / 'accuracy.csv')
    assert len([x for x in baseline if x['stage'] in ['coarse', 'fine']]) == 40
    write_csv('baseline_accuracy', baseline)
    baseline_macro = []
    for method in ['STDLoc', 'ULFLoc']:
        for semantics in ['off', 'on']:
            for stage in ['coarse', 'fine']:
                rows = [x for x in baseline if x['condition'].startswith(method + '_')
                        and x['condition'].endswith('semantics_' + semantics) and x['stage'] == stage]
                assert len(rows) == 5
                baseline_macro.append(dict(method=method, mapping_mask=semantics, stage=stage,
                    **macro([dict(TE_cm=x['median_te_cm'], RE_deg=x['median_re_deg'], R5_pct=x['R5_percent']) for x in rows])))
    write_csv('baseline_macro', baseline_macro)
    suite = csv_read(POOL / 'anygsloc_cambridge_suite_v2_20260914/accuracy.csv')
    write_csv('earlier_73_ablations_and_30_priors', suite)
    priors = []
    for prior in ['vanilla_2dgs', 'vanilla_3dgs', 'long_lrm_raw', 'long_lrm_1k', 'i3dgs_raw', 'i3dgs_1k']:
        rows = [x for x in suite if x['condition'].startswith('prior_') and x['condition'].endswith('_' + prior)]
        assert len(rows) == 5
        priors.append(dict(prior=prior, **macro([dict(TE_cm=x['median_te_cm'], RE_deg=x['median_re_deg'], R5_pct=x['R5_percent']) for x in rows])))
    write_csv('prior_macro', priors)
    factorial, timing = [], []
    for cell in read(factorial_root / 'registry.json'):
        summary = read(Path(cell['result']) / 'summary.json')
        build = read(Path(cell['map']).parent.parent / 'report.json')
        benchmark = read(factorial_root / 'benchmarks' / cell['id'] / 'summary.json')
        assert benchmark['complete'] and len(benchmark['rounds']) == 3
        count = build['compact_count' if cell['selection'] == 'compact' else 'full_count']
        row = {k: cell[k] for k in ['id', 'scene', 'v2', 'fusion', 'completion', 'selection']}
        row.update(accuracy(summary), anchors=count, map_bytes=benchmark['map_bytes'],
            core_mean_ms=stats.mean(x['runtime_ms_mean'] for x in benchmark['rounds']),
            core_p95_ms_round_mean=stats.mean(x['runtime_ms_p95'] for x in benchmark['rounds']),
            loop_fps_round_mean=stats.mean(x['loop_queries_per_second'] for x in benchmark['rounds']),
            selection_seconds=build['selection_seconds'])
        factorial.append(row)
    assert len(factorial) == 80
    write_csv('factorial_80', factorial)
    factors = []
    for v in [0, 1]:
        for f in [0, 1]:
            for c in [0, 1]:
                for selection in ['full', 'compact']:
                    rows = [x for x in factorial if (x['v2'], x['fusion'], x['completion'], x['selection']) == (v, f, c, selection)]
                    assert len(rows) == 5
                    factors.append(dict(V2=v, fusion=f, completion=c, selection=selection, **macro(rows),
                        total_anchors=sum(x['anchors'] for x in rows), total_map_MB=sum(x['map_bytes'] for x in rows)/1e6,
                        core_mean_ms=stats.mean(x['core_mean_ms'] for x in rows),
                        loop_fps_scene_mean=stats.mean(x['loop_fps_round_mean'] for x in rows)))
    write_csv('factorial_macro', factors)
    geometry = []
    for path in sorted((center_root / 'runs').glob('*/*/summary.json')):
        scene, kind = path.parent.parent.name, path.parent.name
        support = read(center_root / 'maps' / scene / 'report.json')
        geometry.append(dict(scene=scene, geometry=kind, **accuracy(read(path)),
            common_count=support['common_count'], source_count=support['source_count']))
    assert len(geometry) == 15
    write_csv('geometry_15', geometry)
    geometry_macro = [dict(geometry=k, **macro([x for x in geometry if x['geometry'] == k])) for k in ['ray', 'depth', 'center']]
    costs = csv_read(baseline_root / 'costs.csv')
    write_csv('baseline_costs_raw', costs)
    for method in ['STDLoc', 'ULFLoc']:
        for sem in ['off', 'on']:
            rows = [x for x in costs if x['condition'].startswith(method + '_') and 'semantics_' + sem + '_benchmark' in x['condition']]
            assert len(rows) == 15
            timing.append(dict(method=method, mapping_mask=sem,
                coarse_prefix_ms=stats.mean(float(x['coarse_prefix_ms_mean']) for x in rows),
                coarse_plus_fine_ms=stats.mean(float(x['total_ms_mean']) for x in rows)))
    write_csv('baseline_timing_macro', timing)
    conversions = []
    for path in sorted((POOL / 'anygsloc_cambridge_recovery_v3_20260915/receipts').glob('conversion_*.json')):
        x = read(path)
        assert x['status'] == 'complete'
        conversions.append(dict(scene=path.stem.removeprefix('conversion_'), minutes=x['wall_seconds']/60,
                                scope='given_MAtCHA_PLY_conversion_only'))
    write_csv('matcha_conversion_cost', conversions)
    write_csv('prior_costs_PROVISIONAL', csv_read(POOL / 'anygsloc_build_cost_audit_v2_20260915/prior_costs.csv'))
    sections = [('# 数值附录（2026-09-17）', '所有宏平均均为五场景或对应数据集场景等权平均；不是 pooled median。TE 单位 cm，RE 单位度。'),
        ('## 24 场景主表', table(primary, ['dataset', 'scene', 'queries', 'TE_cm', 'RE_deg', 'R5_pct'])),
        ('## 数据集汇总', table(datasets, ['dataset', 'TE_cm', 'RE_deg', 'R5_pct'])),
        ('## 原生基线复现', table(baseline_macro, ['method', 'mapping_mask', 'stage', 'TE_cm', 'RE_deg', 'R5_pct'])),
        ('## 高斯先验', table(priors, ['prior', 'TE_cm', 'RE_deg', 'R5_pct'])),
        ('## 完整组合消融（无筛选删行）', table(factors, ['V2', 'fusion', 'completion', 'selection', 'TE_cm', 'RE_deg', 'R5_pct', 'total_anchors', 'core_mean_ms', 'loop_fps_scene_mean'])),
        ('## 固定共同点几何对照', table(geometry_macro, ['geometry', 'TE_cm', 'RE_deg', 'R5_pct'])),
        ('## 原生基线预热耗时', table(timing, ['method', 'mapping_mask', 'coarse_prefix_ms', 'coarse_plus_fine_ms'])),
        ('## 已有 MAtCHA PLY 的转换成本', table(conversions, ['scene', 'minutes', 'scope']))]
    (OUT / 'TABLES.md').write_text('\n\n'.join(title+'\n\n'+body for title, body in sections)+'\n')
    (OUT / 'source_manifest.json').write_text(json.dumps(SOURCES, indent=2))
    print('Evidence exported to', OUT, 'source files', len(SOURCES))


if __name__ == '__main__':
    main()
