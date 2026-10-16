"""Assemble reviewable experiment tables without editing the paper or raw runs."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = Path('/mnt/pool/sqy/anygsloc_supplement_20261009')
EVIDENCE = REPO/'docs/paper/evidence_20260917'
METRICS = REPO/'artifacts/metric_sensitivity_20261009'


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    inputs = {}
    def track(path):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    def read(path):
        return json.loads(track(path).read_text())
    def csv_read(path):
        with track(path).open() as f:
            return list(csv.DictReader(f))
    def save(name, rows):
        with (output/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    def copy(path, name):
        shutil.copy2(track(path), output/name)
    metrics = {r['condition']: r for r in csv_read(METRICS/'macro.csv')}
    timing = read(CAMPAIGN/'macro_results.json')['results']
    native_timing = csv_read(EVIDENCE/'baseline_timing_macro.csv')
    protocol = read(CAMPAIGN/'protocol.json')
    verification = read(CAMPAIGN/'verification.json')
    assert read(CAMPAIGN/'status.json')['all_complete'] and verification['complete']
    selected_k = sum(s['K'] for s in protocol['scenes'].values())
    def select(label, condition, fields):
        return dict(condition=label, **{k: metrics[condition][k] for k in fields})
    main_fields = ['median_TE_cm','median_RE_deg','R5_pct','R15_pct']
    comparison = []
    for method in ['STDLoc','ULFLoc']:
        for stage in ['coarse','fine']:
            row = select(method+' '+stage, method+'_on_'+stage, main_fields)
            t = next(r for r in native_timing if r['method'] == method and r['mapping_mask'] == 'on')
            row.update(mapping_semantics='enabled during baseline mapping; query semantics disabled',
                online_rendering=stage == 'fine',
                mean_pipeline_ms=float(t['coarse_prefix_ms' if stage == 'coarse' else 'coarse_plus_fine_ms']),
                timing_cohort='20260915 native baseline timers')
            comparison.append(row)
    for label, condition, timed in [('Rendered-RGB triangulation','classic_spatial','classic_spatial'),
                                     ('AnyGSLoc','complete','native_A')]:
        row = select(label, condition, main_fields)
        t = next(r for r in timing if r['stage'] == 'warmed_timing' and r['method'] == timed)
        row.update(mapping_semantics='no additional conversion masks; supplied MAtCha prior has upstream semantics',
            online_rendering=False, mean_pipeline_ms=t['mean_ms'], timing_cohort='20261009 warmed three-round timers')
        comparison.append(row)
    save('T1_local_comparison.csv', comparison)
    common_k = sum(r['geometry_K'] for r in verification['checks'])
    save('T2_geometry.csv', [dict(select(kind, 'geometry_'+kind, main_fields+['P90_TE_cm']),
        total_points=common_k, shared_observations=True, shared_descriptors=True) for kind in ['center','depth','ray']])
    factorial = csv_read(EVIDENCE/'factorial_macro.csv')
    observations = []
    for label, condition, v, f, c in [('All observation factors off','observation_all_off',0,0,0),
            ('Screening off','no_screening',0,1,1), ('Descriptor fusion off','single_descriptor',1,0,1),
            ('Candidate completion off','no_completion',1,1,0), ('Complete','complete',1,1,1)]:
        source = next(r for r in factorial if (r['V2'],r['fusion'],r['completion'],r['selection']) == (str(v),str(f),str(c),'compact'))
        observations.append(dict(select(label, condition, main_fields+['mean_TE_cm']),
            total_points=int(source['total_anchors']), control='same selection parameters; natural point count'))
    save('T3a_observation_controls.csv', observations)
    save('T3b_descriptor_controls.csv', [dict(select(label, condition, main_fields+['mean_TE_cm']),
        total_points=selected_k, fixed_XYZ=True) for label,condition in
        [('Single observation','single_descriptor'),('Ordinary multi-view mean','mean_descriptor'),('Proposed fusion','complete')]])
    selection = []
    full = next(r for r in factorial if (r['V2'],r['fusion'],r['completion'],r['selection']) == ('1','1','1','full'))
    for label, condition in [('Full reference','full_map'), ('Spatial','selector_spatial'),
                             ('Quality Top-K','selector_quality'), ('Complete selector','complete')]:
        if condition == 'full_map':
            points, size = int(full['total_anchors']), float(full['total_map_MB'])
        else:
            paths = [Path(s['compact_map']) if condition == 'complete' else
                CAMPAIGN/'selector'/scene/condition.removeprefix('selector_')/'deployment_map.pt'
                for scene, s in protocol['scenes'].items()]
            points, size = selected_k, sum(p.stat().st_size for p in paths)/1e6
        selection.append(dict(select(label, condition, main_fields), total_points=points,
            total_map_MB=size, equal_K_arm=condition != 'full_map'))
    save('T4_selection_with_full_reference.csv', selection)
    save('S1_pose_completion.csv', [dict(select(label, condition, main_fields+['mean_TE_cm','P90_TE_cm']),
        total_points=selected_k, added_points_per_scene=4096) for label,condition in
        [('Shared prefix + quality','quality_completion'),('Shared prefix + pose','complete')]])
    ransac = []
    for budget in 'ABCD':
        s = next(r for r in timing if r['stage'] == 'solver_replay' and r['method'] == 'ransac_'+budget)
        t = next((r for r in timing if r['stage'] == 'warmed_timing' and r['method'] == 'native_'+budget), None)
        ransac.append(dict(budget=budget, confidence=protocol['ransac_budgets'][budget][0],
            min_iterations=protocol['ransac_budgets'][budget][1],max_iterations=protocol['ransac_budgets'][budget][2],
            median_TE_cm=s['TE_cm'],R5_pct=s['R5_pct'],R15_pct=s['R15_pct'],
            mean_pipeline_ms=t['mean_ms'] if t else None))
    save('S2_ransac.csv', ransac)
    priors = [dict(prior='MAtCha', TE_cm=metrics['complete']['median_TE_cm'],
        RE_deg=metrics['complete']['median_RE_deg'], R5_pct=metrics['complete']['R5_pct'])]
    priors += csv_read(EVIDENCE/'prior_macro.csv')
    save('T5_priors.csv', priors)
    copy(EVIDENCE/'primary_24.csv','T0_primary_24.csv')
    copy(METRICS/'macro.csv','S3_all_metrics.csv')
    copy(METRICS/'per_scene.csv','S4_per_scene_metrics.csv')
    copy(METRICS/'recall_curves.json','S5_recall_curves.json')
    copy(METRICS/'recall_sensitivity.pdf','S5_recall_sensitivity.pdf')
    copy(EVIDENCE/'factorial_macro.csv','S6_factorial_all_16.csv')
    copy(EVIDENCE/'factorial_80.csv','S7_factorial_all_80_scenes.csv')
    copy(EVIDENCE/'baseline_timing_macro.csv','S8_native_baseline_times_both_semantic_settings.csv')
    manifest = dict(status='reviewable material package; manuscript not changed',
        main_order=['Setup','Accuracy and native inference cost','Geometry','Observation and descriptor controls',
                    'Selection and compactness','Prior compatibility'],
        budget_control='Same K within every scene for three selection strategies; full map is a larger-budget reference. File headers need not match byte-for-byte.',
        metric_policy='Module-specific explanatory metrics plus shared localization anchors; all methods in each comparison share metric definitions; newly inspected metrics are post-hoc descriptive analyses.',
        primary_config='A, seed 2026; no query semantic masks; scene-equal aggregation',
        notes=['Baseline main rows use mapping mask on for both methods; off rows remain in S3/S4/S8.',
            'Do not mix mask-on accuracy with mask-off latency or deployment inventory.',
            'Existing cross-method asset inventory was collected for mask-off, so it is not inserted into T1.',
            'Native timing boundaries/configurations differ; these are system operating points, not an equal-compute algorithm comparison.',
            'Mean and median TE are separately averaged over five scenes. Infinite errors are retained for solver failures; S3 may contain inf.',
            'No statistical significance follows from three timing repeats or post-hoc metric selection.',
            'T3a changes natural capacity; T3b fixes points and coordinates. All-off observations is not the independent classical baseline.',
            'S1 isolates the final addition stage and is distinct from global quality Top-K.',
            'T5 contains all seven prior conditions, including poor-quality inputs; it is not a controlled prior-quality ranking.',
            'A multi-budget selector curve has not been run. Current data establish one equal-budget operating point per scene.'],
        source_hashes=inputs)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(output)
    print('files', len(list(output.iterdir())), 'frozen inputs',len(inputs))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
