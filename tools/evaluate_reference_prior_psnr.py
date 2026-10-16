"""Evaluate frozen E4 priors on common registered train / mapping views.

Reuse the test evaluator's rendering and per-image metric code unchanged. Write
reference results to a separate directory and compare only after full coverage.
Run with the g4splat environment (including its lib directory in LD_LIBRARY_PATH).
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import copy
import csv
import hashlib
import io
import json
import math
import shutil
import sys

import numpy as np
from PIL import Image

import evaluate_prior_psnr as base

ROOT = base.ROOT
OUT = ROOT / 'artifacts/prior_psnr_reference_20261008'
TEST = ROOT / 'artifacts/prior_psnr_20261008'
base.OUT = OUT
SCENES, KEYS = base.SCENES, base.KEYS
load_json, write_json, sha = base.load_json, base.write_json, base.sha
NAMES = {'StMarysChurch': "St. Mary's Church", 'OldHospital': 'Old Hospital',
         'GreatCourt': 'Great Court', 'ShopFacade': 'Shop Facade',
         'KingsCollege': "King's College"}


def split_names(path):
    names = [line.split()[0] for line in path.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith('#')
             and Path(line.split()[0]).suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}]
    assert len(names) == len(set(names)), path
    return set(names)


def prepare():
    from anygsloc.data.datasets import ColmapDataset

    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'manifest.json').exists():
        check_frozen_inputs()
        print('Using existing frozen reference-view manifest.', flush=True)
        return
    test = load_json(TEST / 'manifest.json')
    protocol = copy.deepcopy(test['protocol'])
    protocol.update({
        'split': 'reference / mapping (registered official train views)',
        'evaluation_views': 'Every calibrated ColmapDataset mapping view, sorted lexically; '
                            'verified equal to the registered official dataset_train.txt subset; '
                            'identical views for all five priors',
        'pooled_mean': 'Also report the mean over all 5,364 reference images; '
                       'the headline macro gives each of the five scenes equal weight',
        'interpretation': 'Input-prior rendering quality at map-construction reference cameras; '
                          'not held-out novel-view rendering quality',
        'train_membership': 'The common benchmark reference pool is used for every prior, '
                            'regardless of each prior producer\'s actual training/input subset',
    })
    scenes = {}
    for scene in SCENES:
        root = Path('/mnt/pool/sqy/Cambridge_stdloc') / scene
        ds = ColmapDataset(root, load_masks=False)
        cams = ds.split('mapping')
        names = {c.image_name for c in cams}
        official_train = split_names(root / 'dataset_train.txt')
        official_test = split_names(root / 'dataset_test.txt')
        registered = {c.image_name for c in ds.split('all')}
        assert names == official_train & registered
        assert not names & official_test
        assert official_test == {v['name'] for v in test['scenes'][scene]['views']}
        assert not official_train & official_test
        previous = test['scenes'][scene]
        assert sha(root / 'dataset_test.txt') == previous['test_split_sha256']
        for path, digest in previous['calibration_sources'].items():
            assert sha(path) == digest, path
        for prior in previous['priors'].values():
            assert sha(prior['ply']) == prior['ply_sha256'], prior['ply']

        def record(camera):
            data = camera.image_path.read_bytes()
            with Image.open(io.BytesIO(data)) as image:
                assert image.size == (camera.width, camera.height) == (1920, 1080)
                assert image.mode == 'RGB', (scene, camera.image_name)
            K = [[camera.width / (2 * math.tan(camera.fov_x / 2)), 0, camera.width / 2],
                 [0, camera.height / (2 * math.tan(camera.fov_y / 2)), camera.height / 2],
                 [0, 0, 1]]
            return {'name': camera.image_name, 'path': str(camera.image_path),
                    'sha256': hashlib.sha256(data).hexdigest(),
                    'width': camera.width, 'height': camera.height,
                    'K': K, 'pose_w2c': camera.pose_w2c.tolist()}

        with ThreadPoolExecutor(max_workers=8) as pool:
            views = list(pool.map(record, cams))
        scenes[scene] = {
            'views': views, 'n_views': len(views), 'priors': previous['priors'],
            'calibration_sources': previous['calibration_sources'],
            'split_sources': {str(root / f'dataset_{split}.txt'): sha(root / f'dataset_{split}.txt')
                              for split in ['train', 'test']},
            'official_train_count': len(official_train),
            'unregistered_train_views': sorted(official_train - registered),
            'registered_train_equals_mapping': True, 'reference_test_disjoint': True,
        }
        print(f'Prepared {scene}: {len(views)} common reference views', flush=True)
    total = sum(scene['n_views'] for scene in scenes.values())
    assert total == 5364
    # Preserve both the imported evaluator and this wrapper for exact reproduction.
    code = OUT / 'code'
    code.mkdir(exist_ok=True)
    for path in [Path(base.__file__), Path(__file__).resolve()]:
        shutil.copy2(path, code / path.name)
    test_hashes = {str(path.relative_to(ROOT)): sha(path)
                   for path in sorted(TEST.rglob('*')) if path.is_file()}
    manifest = {
        'protocol': protocol, 'source': test['source'], 'source_sha256': test['source_sha256'],
        'scenes': scenes, 'scene_order': SCENES, 'prior_order': KEYS,
        'prior_labels': test['prior_labels'], 'n_reference_images': total,
        'n_rendered_comparisons': total * len(KEYS),
        'test_manifest': str(TEST / 'manifest.json'),
        'test_manifest_sha256': sha(TEST / 'manifest.json'),
        'test_artifact_hashes': test_hashes,
        'evaluator_sources': {str(Path(base.__file__).resolve()): sha(base.__file__),
                              str(Path(__file__).resolve()): sha(__file__)},
    }
    write_json(OUT / 'manifest.json', manifest)
    lines = ['Reference-view Gaussian prior PSNR — 2026-10-08', '',
             f'{total:,} reference images per prior; {total * len(KEYS):,} comparisons.',
             'Same 25 frozen E4 priors and renderer/metric protocol as the existing test run.',
             'Great Court has one official train image without a registered camera; '
             'it is excluded consistently for all priors and listed in manifest.json.', '',
             *[f'{key}: {value}' for key, value in protocol.items()], '',
             'All actual reference views are evaluated, not a producer-specific training subset.',
             'Reproduce with g4splat Python and its lib directory in LD_LIBRARY_PATH:',
             'python tools/evaluate_reference_prior_psnr.py prepare',
             'CUDA_VISIBLE_DEVICES=0 python tools/evaluate_reference_prior_psnr.py worker --workers 1',
             'python tools/evaluate_reference_prior_psnr.py summarize',
             'CUDA_VISIBLE_DEVICES=0 python tools/evaluate_reference_prior_psnr.py verify',
             'python tools/evaluate_reference_prior_psnr.py report',
             '', 'RGB ground truth is used only for this metric; priors, maps, poses, figures '
             'and the manuscript are not modified.']
    (OUT / 'protocol.txt').write_text('\n'.join(lines) + '\n')


def check_frozen_inputs():
    manifest = load_json(OUT / 'manifest.json')
    for path, digest in manifest['evaluator_sources'].items():
        assert sha(path) == digest, ('Evaluator changed after manifest freeze', path)
    assert sha(manifest['source']) == manifest['source_sha256']
    assert sha(manifest['test_manifest']) == manifest['test_manifest_sha256']
    return manifest


def summarize():
    manifest = check_frozen_inputs()
    binding = sha(OUT / 'manifest.json')
    results, all_rows, checks = {}, [], []
    for scene in SCENES:
        data = manifest['scenes'][scene]
        views = data['views']
        results[scene] = {}
        for path, digest in {**data['calibration_sources'], **data['split_sources']}.items():
            assert sha(path) == digest, path
        for key in KEYS:
            job = OUT / 'jobs' / f'{scene}__{key}'
            summary = load_json(job / 'summary.json')
            rows = [json.loads(line) for line in (job / 'records.jsonl').read_text().splitlines()]
            rows.sort(key=lambda row: row['index'])
            assert summary['manifest_sha256'] == binding
            assert summary['records_sha256'] == sha(job / 'records.jsonl')
            assert summary['script_sha256'] == sha(base.__file__)
            assert len(rows) == len(views) == summary['n_views']
            assert [r['index'] for r in rows] == list(range(len(views)))
            assert [r['name'] for r in rows] == [v['name'] for v in views]
            for row, view in zip(rows, views):
                assert row['GT_sha256'] == view['sha256']
                assert row['manifest_sha256'] == binding
                assert math.isfinite(row['mse']) and row['mse'] > 0
                assert math.isfinite(row['psnr_db'])
                assert abs(row['psnr_db'] + 10 * math.log10(row['mse'])) < 1e-10
                assert row['pixels'] == view['width'] * view['height']
                all_rows.append({'scene': scene, 'prior': key, **row})
            assert abs(np.mean([r['psnr_db'] for r in rows]) - summary['mean_psnr_db']) < 1e-10
            cpu_error = abs(rows[0]['mse'] - rows[0]['independent_CPU_mse'])
            assert cpu_error < 1e-12
            source = data['priors'][key]
            assert sha(source['ply']) == source['ply_sha256']
            results[scene][key] = summary
            checks.append({'scene': scene, 'prior': key, 'n_views': len(rows),
                           'all_expected_views_complete': True, 'same_GT_and_cameras': True,
                           'metric_recomputed': True, 'prior_hash_unchanged': True,
                           'first_view_CPU_GPU_MSE_difference': cpu_error,
                           'tile32_view_count': len(summary['tile32_views'])})
    assert len(all_rows) == manifest['n_rendered_comparisons']
    for path, digest in manifest['test_artifact_hashes'].items():
        assert sha(ROOT / path) == digest, ('Existing test artifact changed', path)
    macro = {key: float(np.mean([results[s][key]['mean_psnr_db'] for s in SCENES])) for key in KEYS}
    pooled = {key: float(np.mean([r['psnr_db'] for r in all_rows if r['prior'] == key])) for key in KEYS}
    result = {'protocol': manifest['protocol'], 'scene_order': SCENES, 'prior_order': KEYS,
              'prior_labels': manifest['prior_labels'],
              'scene_means': {s: {k: results[s][k]['mean_psnr_db'] for k in KEYS} for s in SCENES},
              'scene_view_counts': {s: manifest['scenes'][s]['n_views'] for s in SCENES},
              'five_scene_macro_psnr_db': macro, 'pooled_image_mean_psnr_db': pooled,
              'n_reference_images': manifest['n_reference_images'],
              'n_rendered_comparisons': len(all_rows), 'manifest_sha256': binding,
              'job_summaries': results}
    write_json(OUT / 'summary.json', result)
    with (OUT / 'scene_psnr.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['scene', 'n_reference_views'] + base.LABELS)
        for scene in SCENES:
            writer.writerow([scene, result['scene_view_counts'][scene]] +
                            [result['scene_means'][scene][key] for key in KEYS])
        writer.writerow(['Five-scene equal-weight mean', manifest['n_reference_images']] + list(macro.values()))
        writer.writerow(['All-reference-image mean', manifest['n_reference_images']] + list(pooled.values()))
    columns = ['scene', 'prior', 'index', 'name', 'mse', 'psnr_db', 'pixels', 'tile_size',
               'seconds', 'GT_sha256', 'manifest_sha256']
    with (OUT / 'per_image_psnr.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)
    write_json(OUT / 'verification.json', {'status': 'passed', 'checks': checks,
               'rendered_comparisons': len(all_rows), 'all_reference_test_sets_disjoint': True,
               'all_existing_test_artifact_hashes_unchanged': True,
               'test_files_checked': len(manifest['test_artifact_hashes'])})
    print(json.dumps(result['scene_means'], indent=2))
    print('Five-scene reference macro:', macro, flush=True)


def verify_renderer():
    import torch
    from anygsloc.priors.models import FrozenGaussianModel
    from anygsloc.priors.rendering import render_from_pose_gsplat

    manifest = check_frozen_inputs()
    torch.set_num_threads(4)
    checks = []
    for key in ['matcha', 'vanilla_3dgs']:
        scene = 'StMarysChurch'
        source = manifest['scenes'][scene]['priors'][key]
        view = manifest['scenes'][scene]['views'][0]
        assert sha(source['ply']) == source['ply_sha256']
        model = FrozenGaussianModel(source['sh_degree'], device='cuda')
        model.load_ply(source['ply'], loc_feature_dim=0)
        K = np.asarray(view['K'])
        w, h = view['width'], view['height']
        with torch.inference_mode():
            result = render_from_pose_gsplat(
                model, torch.tensor(view['pose_w2c'], dtype=torch.float32, device='cuda'),
                2 * math.atan(w / (2 * K[0, 0])), 2 * math.atan(h / (2 * K[1, 1])), w, h,
                bg_color=torch.zeros(3, device='cuda'), render_mode='RGB', rgb_only=True,
                rasterize_mode='antialiased')
            rgb = result['render'].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        gt = base.read_rgb(view).astype(np.float64) / 255.
        mse = float(np.mean((rgb.astype(np.float64) - gt) ** 2))
        job = OUT / 'jobs' / f'{scene}__{key}'
        row = next(json.loads(line) for line in (job / 'records.jsonl').read_text().splitlines()
                   if json.loads(line)['index'] == 0)
        assert abs(mse - row['mse']) < 1e-12
        assert np.array_equal(np.uint8(np.round(rgb * 255)), np.array(Image.open(job / 'first_view_render.png')))
        checks.append({'scene': scene, 'prior': key, 'name': view['name'],
                       'mse_difference': abs(mse - row['mse']),
                       'PSNR_difference_dB': abs(-10 * math.log10(mse) - row['psnr_db']),
                       'saved_render_pixels_exact_match': True})
        del model, result
        torch.cuda.empty_cache()
    write_json(OUT / 'production_renderer_verification.json', {'status': 'passed', 'checks': checks})
    print(json.dumps(checks, indent=2))


def report():
    manifest = check_frozen_inputs()
    data = load_json(OUT / 'summary.json')
    test = load_json(TEST / 'summary.json')
    verification = load_json(OUT / 'verification.json')
    native = load_json(OUT / 'production_renderer_verification.json')
    assert verification['status'] == native['status'] == 'passed'
    assert data['n_rendered_comparisons'] == 26820
    comparison = []
    for scene in SCENES + ['Five-scene equal-weight mean']:
        is_macro = scene == 'Five-scene equal-weight mean'
        reference = data['five_scene_macro_psnr_db'] if is_macro else data['scene_means'][scene]
        heldout = test['five_scene_macro_psnr_db'] if is_macro else test['scene_means'][scene]
        for key in KEYS:
            comparison.append({'scene': scene, 'prior': key,
                               'reference_psnr_db': reference[key], 'test_psnr_db': heldout[key],
                               'reference_minus_test_db': reference[key] - heldout[key]})
    write_json(OUT / 'reference_vs_test.json', {'metric': 'Mean per-image PSNR (dB)',
               'reference_manifest_sha256': sha(OUT / 'manifest.json'),
               'test_manifest_sha256': sha(TEST / 'manifest.json'), 'rows': comparison})
    with (OUT / 'reference_vs_test.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    lines = ['Reference-view Gaussian prior PSNR — completed 2026-10-08',
             '25 / 25 scene-prior combinations; 5,364 common reference views per prior; 26,820 renders.',
             '1920 x 1080, calibrated GT cameras, original priors, full RGB image without masks or fitted exposure.',
             'Per-image PSNR averaged within each scene; headline macro gives all five scenes equal weight.', '',
             'Scene                    N      MAtCha     2DGS     3DGS   i3DGS raw   i3DGS post-optimized']
    for scene in SCENES:
        lines.append(f'{NAMES[scene]:24s} {data["scene_view_counts"][scene]:4d}  ' +
                     '  '.join(f'{data["scene_means"][scene][key]:9.4f}' for key in KEYS))
    lines.append(f'{"Five-scene macro":24s} 5364  ' +
                 '  '.join(f'{data["five_scene_macro_psnr_db"][key]:9.4f}' for key in KEYS))
    lines.append(f'{"Pooled image mean":24s} 5364  ' +
                 '  '.join(f'{data["pooled_image_mean_psnr_db"][key]:9.4f}' for key in KEYS))
    lines += ['', 'Comparison with the preserved test evaluation:',
              'Prior                         Reference      Test       Difference (dB)']
    for row in comparison[-len(KEYS):]:
        lines.append(f'{data["prior_labels"][row["prior"]]:28s} {row["reference_psnr_db"]:9.4f} '
                     f'{row["test_psnr_db"]:9.4f} {row["reference_minus_test_db"]:+12.4f}')
    higher = sum(row['reference_minus_test_db'] > 0 for row in comparison[:-len(KEYS)])
    deltas = {scene: data['scene_means'][scene]['i3dgs_1k'] - data['scene_means'][scene]['i3dgs_raw']
              for scene in SCENES}
    lines += ['', f'Reference mean exceeds test mean in {higher} / 25 scene-prior combinations.',
              'i3DGS post-optimization gains (dB): ' + json.dumps(deltas),
              '', 'Official train views without registered cameras (excluded for every prior): ' +
              json.dumps({s: manifest['scenes'][s]['unregistered_train_views'] for s in SCENES
                          if manifest['scenes'][s]['unregistered_train_views']}),
              'The common registered train/reference pool is not a claim that every prior producer used all these views.',
              'All existing test artifacts and model hashes are preserved; all per-image records checked.',
              'CPU/GPU metric checks passed for the first view of every job; native 2D/3D production-renderer checks passed.',
              'E4-ready reference scene means and caption text: e4_psnr_rows.json and e4_psnr_caption.txt.',
              'Paper and figure artwork have not been changed. See protocol.txt for reproduction.']
    (OUT / 'results.txt').write_text('\n'.join(lines) + '\n')
    e4 = {'label': 'Reference-view PSNR ↑ (dB)',
          'split': 'All registered mapping/reference views of the indicated scene',
          'resolution': [1920, 1080], 'keys': KEYS,
          'labels': [data['prior_labels'][key] for key in KEYS],
          'scenes': {scene: {'n_views': data['scene_view_counts'][scene],
                            'PSNR_dB': data['scene_means'][scene],
                            'display_values_2dp': [f'{data["scene_means"][scene][key]:.2f}' for key in KEYS]}
                     for scene in ['StMarysChurch', 'ShopFacade']},
          'global_values': {'label': 'Five-scene mean reference-view PSNR (dB)',
                            'aggregation': 'Equal-weight mean of per-scene mean per-image PSNR',
                            'PSNR_dB': data['five_scene_macro_psnr_db']},
          'source': 'summary.json', 'figure_integration_done': False}
    write_json(OUT / 'e4_psnr_rows.json', e4)
    caption = ('Reference-view PSNR (dB, higher is better) is averaged over all calibrated '
               'mapping views of each scene, using the same reference images and cameras for every prior. '
               'We evaluate full-resolution RGB renders of the input Gaussian priors against the '
               'corresponding reference images at ground-truth poses, without image masks or exposure fitting. '
               'This measures input-prior rendering quality at the reference cameras.\n')
    (OUT / 'e4_psnr_caption.txt').write_text(caption)
    latex = ['% Scene & reference views & MAtCha & 2DGS & 3DGS & i3DGS raw & i3DGS post-optimized']
    for scene in SCENES:
        latex.append(NAMES[scene] + ' & ' + str(data['scene_view_counts'][scene]) + ' & ' +
                     ' & '.join(f'{data["scene_means"][scene][key]:.2f}' for key in KEYS) + r' \\')
    latex.append('Five-scene mean & 5364 & ' +
                 ' & '.join(f'{data["five_scene_macro_psnr_db"][key]:.2f}' for key in KEYS) + r' \\')
    (OUT / 'psnr_table_rows.tex').write_text('\n'.join(latex) + '\n')
    verification.update(production_renderer_cross_checks=native['checks'], figure_integration_done=False)
    write_json(OUT / 'verification.json', verification)
    files = [path for path in OUT.rglob('*') if path.is_file() and path.name != 'output_manifest.sha256.json']
    write_json(OUT / 'output_manifest.sha256.json', {str(path.relative_to(ROOT)): sha(path) for path in sorted(files)})
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'worker', 'summarize', 'verify', 'report'])
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--scene', choices=SCENES)
    parser.add_argument('--prior', choices=KEYS)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare()
    elif args.phase == 'worker':
        check_frozen_inputs()
        assert 0 <= args.worker < args.workers
        assert bool(args.scene) == bool(args.prior), 'Supply both --scene and --prior, or neither.'
        assert args.limit >= 0 and (not args.limit or args.scene), 'Limit requires one scene-prior job.'
        base.worker(args.worker, args.workers, args.scene, args.prior, args.limit)
    elif args.phase == 'summarize':
        summarize()
    elif args.phase == 'verify':
        verify_renderer()
    else:
        report()
