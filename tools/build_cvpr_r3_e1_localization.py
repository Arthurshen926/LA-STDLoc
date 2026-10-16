"""R3 E1: archived coarse and final poses, rendered only for paper inspection.

Reuses the R2 examples and all final-pose render pixels. Adds only six offline
renders from already archived coarse poses. No localization is run.
"""
from pathlib import Path
import argparse, copy, gc, hashlib, json, shutil, sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'paper/experiments_cvpr_r3_20261003'
ASSETS = OUT / 'assets/e1'
PREVIOUS = ROOT / 'paper/experiments_cvpr_r2_20261003'
PROV = OUT / 'E1_final_localization.provenance.json'
METHODS = ['STDLoc_coarse', 'STDLoc', 'ULFLoc_coarse', 'ULFLoc', 'AnyGSLoc']
GT, EST, INK, GRAY = '#219653', '#cc2f83', '#233047', '#697585'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(p):
    PROV.write_text(json.dumps(p, indent=2))


def prepare():
    ASSETS.mkdir(parents=True, exist_ok=True)
    prev_path = PREVIOUS / 'E1_final_localization.provenance.json'
    p = copy.deepcopy(read(prev_path))
    p['revision'] = 3
    p['scope'] = 'Archived coarse and final baseline poses and single-stage compact AnyGSLoc poses; selected favorable examples from R2; offline visualization only.'
    p['previous_revision'] = {'path': str(prev_path), 'sha256': digest(prev_path)}
    p['selection']['revision_3_policy'] = 'Exactly the same three query images and native 320x240 ROI coordinates as R2. No additional query or ROI screening. Refinement improves both baselines in rows a/b; ULF-Loc gets worse in row c and is shown unchanged.'
    # Keep complete selection audit portable without loading any localization data.
    for name in ['all_query_candidates.json', 'screen_displacement_candidates.json']:
        shutil.copy2(PREVIOUS / 'assets/e1' / name, ASSETS / name)
    for row in p['rows']:
        for method in ['STDLoc', 'ULFLoc']:
            m = row['methods'][method]
            row['methods'][method + '_coarse'] = {
                'TE_cm': m['coarse_TE_cm'], 'RE_deg': m['coarse_RE_deg'],
                'pose_w2c': m['coarse_pose_w2c'], 'native_result_index': m['native_result_index'],
                'native_pose_key': 'sparse.pose_w2c', 'stage': 'coarse'}
            m['stage'] = 'fine (final)'
            m['native_pose_key'] = 'dense[-1].pose_w2c'
        row['methods']['AnyGSLoc']['stage'] = 'single'
        for method, item in row['renders'].items():
            src = Path(item['path'])
            dst = ASSETS / src.name
            assert digest(src) == item['sha256']
            shutil.copy2(src, dst)
            item['path'] = str(dst)
            item['reuse'] = 'Byte-identical to R2; no rerendering.'
        src = PREVIOUS / 'assets/e1' / f'{row["stem"]}_query.png'
        shutil.copy2(src, ASSETS / src.name)
        row['local_query_image'] = str(ASSETS / src.name)
        row['local_query_sha256'] = digest(ASSETS / src.name)
        row['local_query_pixels_match_original'] = bool(np.array_equal(
            np.asarray(Image.open(row['query_path']).convert('RGB')),
            np.asarray(Image.open(ASSETS / src.name).convert('RGB'))))
        assert row['local_query_pixels_match_original']
    p['visualization_contract']['reference'] = 'The exact same MAtCHA Gaussian reference is used for GT and all five estimated-pose panels within each row.'
    for key in ['half_alternative', 'outputs', 'validation', 'display', 'caption']:
        p.pop(key, None)
    save(p)
    print('Prepared R3; retained rows and ROIs:', [(r['scene'], r['image'], r['crop']) for r in p['rows']], flush=True)


def render():
    import torch
    from gsplat import rasterization_2dgs
    from gsplat.cuda._torch_impl import _spherical_harmonics
    from anygsloc.priors.models import FrozenGaussianModel
    torch.set_num_threads(4)
    p = read(PROV)
    for row in p['rows']:
        width, height = row['render_metadata']['width'], row['render_metadata']['height']
        cK = np.array(row['render_metadata']['K'], dtype=np.float64)
        K = torch.tensor(cK, dtype=torch.float32)
        pending = [m for m in METHODS if m not in row['renders'] or not Path(row['renders'][m]['path']).exists()]
        if not pending:
            continue
        assert digest(row['reference_ply']) == row['reference_sha256']
        model = FrozenGaussianModel(row['sh_degree'], device='cpu')
        model.load_ply(row['reference_ply'], loc_feature_dim=0)
        scales = model.get_scaling
        if scales.shape[1] == 2:
            scales = torch.cat([scales, torch.ones(len(scales), 1)], dim=1)
        geom = dict(means=model.get_xyz.cuda(), quats=model.get_rotation.cuda(),
                    scales=scales.cuda(), opacities=model.get_opacity[:, 0].cuda())
        print('Loaded', row['scene'], len(model.get_xyz), 'Gaussians; pending', pending, flush=True)
        with torch.inference_mode():
            for method in pending:
                pose = torch.tensor(row['methods'][method]['pose_w2c'], dtype=torch.float32)
                campos = torch.linalg.inv(pose)[:3, 3]
                rgb = []
                for start in range(0, len(model.get_xyz), 65536):
                    coeff = torch.cat([model._features_dc[start:start+65536], model._features_rest[start:start+65536]], dim=1)
                    rgb.append((_spherical_harmonics(row['sh_degree'], model.get_xyz[start:start+65536]-campos, coeff)+.5).clamp_min(0))
                rgb = torch.cat(rgb)
                torch.cuda.reset_peak_memory_stats()
                kwargs = dict(**geom, colors=rgb[None].cuda(), viewmats=pose[None].cuda(),
                              Ks=K[None].cuda(), width=width, height=height, packed=False,
                              backgrounds=torch.zeros(1, 3, device='cuda'), near_plane=.01,
                              far_plane=10000., render_mode='RGB')
                result = rasterization_2dgs(**kwargs)
                arr = result[0][0].clamp(0, 1).cpu().numpy()
                dest = ASSETS / f'{row["stem"]}_{method}_render.png'
                Image.fromarray(np.uint8(np.round(arr*255))).save(dest)
                row['renders'][method] = {'path': str(dest), 'sha256': digest(dest),
                                         'cuda_peak_own_bytes': torch.cuda.max_memory_allocated(),
                                         'stage': 'archived coarse pose, offline render only'}
                print('Rendered', row['scene'], method, 'peak', round(torch.cuda.max_memory_allocated()/2**20), 'MiB', flush=True)
                del kwargs, result, rgb
                gc.collect()
                torch.cuda.empty_cache()
                save(p)
        del geom, model, scales
        gc.collect()
        torch.cuda.empty_cache()


def draw():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox
    p = read(PROV)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'text.color': INK,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    figw, figh = 7.1, 3.9
    fig = plt.figure(figsize=(figw, figh))
    left, gap, groupgap = .017, .011, .021
    # A little more space between methods than between each method's stages.
    w = (1-2*left-2*gap-3*groupgap)/6
    xs = [left]
    for j in range(5):
        xs.append(xs[-1]+w+(gap if j in [1, 3] else groupgap))
    for a, b, title in [(0, 0, 'Real query'), (1, 2, 'STDLoc'), (3, 4, 'ULF-Loc'), (5, 5, 'AnyGSLoc')]:
        fig.text((xs[a]+xs[b]+w)/2, .983, title, ha='center', va='top', fontsize=8.1, weight='bold')
        if b > a:
            fig.add_artist(Line2D([xs[a], xs[b]+w], [.932, .932], transform=fig.transFigure, color='#aab3be', lw=.6))
    for j, title in enumerate(['', 'Coarse', 'Fine (final)', 'Coarse', 'Fine (final)', 'Single stage']):
        fig.text(xs[j]+w/2, .917, title, ha='center', va='top', fontsize=7, color=GRAY)
    for a in [1, 3]:
        fig.text((xs[a]+w+xs[a+1])/2, .917, '→', ha='center', va='top', fontsize=7, color=GRAY)
    legend_parts = [('Shared reference · ', GRAY), ('GT', GT), (' / ', GRAY), ('Estimated pose', EST), (' · alternating bands', GRAY)]
    legend = HPacker(children=[TextArea(t, textprops={'size': 7, 'color': c}) for t, c in legend_parts], align='baseline', pad=0, sep=0)
    fig.add_artist(AnchoredOffsetbox(loc='center', child=legend, frameon=False, bbox_to_anchor=(.5, .858), bbox_transform=fig.transFigure, pad=0, borderpad=0))
    validations = []
    for i, row in enumerate(p['rows']):
        row.pop('query_SHA256_matches_local_copy', None)
        row['local_query_sha256'] = digest(row['local_query_image'])
        row['local_query_pixels_match_original'] = bool(np.array_equal(
            np.asarray(Image.open(row['query_path']).convert('RGB')),
            np.asarray(Image.open(row['local_query_image']).convert('RGB'))))
        assert row['local_query_pixels_match_original']
        archived_stage_checks = []
        for base in ['STDLoc', 'ULFLoc']:
            source = row['result_sources'][base]
            assert digest(source) == p['source_sha256'][source]['sha256']
            native = read(source)[row['methods'][base]['native_result_index']]
            for method, stage, tekey, rekey in [(base+'_coarse', native['sparse'], 'sparse_TE', 'sparse_AE'),
                                              (base, native['dense'][-1], 'dense_TE', 'dense_AE')]:
                record = row['methods'][method]
                assert np.array_equal(np.array(record['pose_w2c']), np.array(stage['pose_w2c']))
                assert record['TE_cm'] == native[tekey] and record['RE_deg'] == native[rekey]
                archived_stage_checks.append(method)
        ims = {m: np.array(Image.open(row['renders'][m]['path']).convert('RGB')) for m in ['GT']+METHODS}
        x0, y0, x1, y1 = row['crop']
        ch, cw = y1-y0, x1-x0
        gt = ims['GT'][y0:y1, x0:x1]
        top = .815-i*.265
        h = w*figw/figh*ch/cw
        bottom = top-h
        ax = fig.add_axes([xs[0], bottom, w, h])
        ax.imshow(Image.open(row['local_query_image']))
        ax.axis('off')
        native = np.array(row['crop'])*row['camera']['width']/row['render_metadata']['width']
        ax.add_patch(Rectangle(native[:2], native[2]-native[0], native[3]-native[1], fill=False, ec='white', lw=.8))
        fig.text(xs[0]+w/2, bottom-.016, f'({chr(97+i)}) '+row['scene'], ha='center', va='top', fontsize=7, weight='bold')
        edges = np.linspace(0, ch, 5).astype(int)
        exact, te_errors, re_errors = [], [], []
        for j, method in enumerate(METHODS, 1):
            est = ims[method][y0:y1, x0:x1]
            composed = gt.copy()
            for b in [1, 3]:
                composed[edges[b]:edges[b+1]] = est[edges[b]:edges[b+1]]
            Image.fromarray(composed).save(ASSETS / f'{row["stem"]}_{method}_bands.png')
            ax = fig.add_axes([xs[j], bottom, w, h])
            ax.imshow(composed)
            ax.set_xlim(0, cw)
            ax.set_ylim(ch, 0)
            ax.axis('off')
            for b in range(4):
                ax.plot([-3, -3], edges[b:b+2], color=GT if b%2 == 0 else EST, lw=1.6, clip_on=False)
                sl = slice(edges[b], edges[b+1])
                exact.append(np.array_equal(composed[sl], (gt if b%2 == 0 else est)[sl]))
            metric = row['methods'][method]
            # Compact per-example errors remain readable at the two-column print width.
            fig.text(xs[j]+w/2, bottom-.016, f'{metric["TE_cm"]:.2f} cm / {metric["RE_deg"]:.3f}°', ha='center', va='top', fontsize=7)
            g, e = np.array(row['camera']['gt_pose_w2c']), np.array(metric['pose_w2c'])
            te = np.linalg.norm(np.linalg.inv(e)[:3, 3]-np.linalg.inv(g)[:3, 3])*100
            re = np.rad2deg(np.arccos(np.clip((np.trace(e[:3, :3]@g[:3, :3].T)-1)/2, -1, 1)))
            te_errors.append(abs(te-metric['TE_cm']))
            re_errors.append(abs(re-metric['RE_deg']))
        assert all(exact)
        assert max(te_errors) < .02
        assert max(re_errors) < .005  # Archived float32 rotation matrices permit small arccos differences.
        validations.append({'scene': row['scene'], 'exact_unwarped_source_pixels_all_bands': all(exact),
                            'recomputed_TE_max_error_cm': max(te_errors), 'recomputed_RE_max_error_deg': max(re_errors),
                            'native_results_hash_pose_and_TE_RE_exact_match': archived_stage_checks})
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / f'E1_final_localization.{ext}', dpi=400, pad_inches=0)
    plt.close(fig)
    caption = ('Selected localization examples showing the coarse and final refinement stages of STDLoc and ULF-Loc alongside single-stage compact-map AnyGSLoc. The real query identifies the shared detail region. Each comparison panel alternates four horizontal bands: a fixed reference Gaussian model rendered at the ground-truth pose in bands 1 and 3, and the same model rendered at the archived estimated pose in bands 2 and 4. Structural continuity across band boundaries indicates pose alignment. The intrinsics, reference model, native resolution, RGB processing, and crop are identical for all stages and methods within each row. Labels report translation / rotation errors. Refinement improves both baselines in the first two examples; the last example also shows that refinement need not improve every query. AnyGSLoc has lower translation and rotation errors than both final baselines in these deliberately selected favorable examples. The queries and detail regions are identical to the previous final-stage comparison, and all selection records are available. Gaussian rendering is used solely for offline visualization and is absent from AnyGSLoc online localization. No displacement is exaggerated; aggregate performance is reported separately.')
    limitations = [
        'Selected favorable examples do not establish average accuracy or uniform superiority.',
        'GT bands are reference-model renderings at the GT camera pose, not the real query image.',
        'Each baseline fine panel is the last archived dense refinement pose; it is not a separate independent prediction.',
        'View-dependent SH color is reevaluated per actual camera pose, so appearance changes also reflect the shared renderer.',
        'Same-reference renderings share the model\'s geometry and appearance limitations.']
    p.update(validation=validations, caption=caption, limitations=limitations)
    p['display'] = {'figure_inches': [figw, figh], 'minimum_ordinary_font_pt': 7,
                    'native_crop_pixels': [320, 240], 'same_R2_queries_ROIs_and_final_renders': True,
                    'layout': '3 rows × 6 columns, baseline coarse/final grouped by method',
                    'GT_bands_1indexed': [1, 3], 'estimated_bands_1indexed': [2, 4]}
    p['source_script'] = {'path': str(Path(__file__).resolve()), 'sha256': digest(__file__)}
    p['outputs'] = {ext: {'path': str(OUT/f'E1_final_localization.{ext}'), 'sha256': digest(OUT/f'E1_final_localization.{ext}')} for ext in ['pdf', 'svg', 'png']}
    (OUT / 'E1_final_localization.caption.txt').write_text(caption+'\n\nLimitations:\n'+'\n'.join('- '+x for x in limitations)+'\n')
    save(p)
    print('Wrote E1 coarse/final comparison:', validations, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'render', 'draw'])
    args = ap.parse_args()
    {'prepare': prepare, 'render': render, 'draw': draw}[args.mode]()
