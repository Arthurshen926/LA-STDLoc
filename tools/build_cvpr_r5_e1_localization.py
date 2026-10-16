"""R5 E1: restrained, shared-position structural boundary guides.

Reuses every R4 query, render, ROI and archived metric exactly. No rendering,
localization, training or sample selection is performed by this script.
"""
from pathlib import Path
import argparse, copy, hashlib, json, shutil, sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'paper/experiments_cvpr_r5_20261003'
ASSETS = OUT / 'assets/e1'
PREVIOUS = ROOT / 'paper/experiments_cvpr_r4_20261003'
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
    p['revision'] = 5
    p['scope'] = 'Readability only: a single shared-position, open-corner structural guide per row and concise publication caption. All R4 source RGB pixels, crops, samples, poses, and errors are preserved.'
    p['previous_detailed_caption'] = p['caption']
    p['previous_revision'] = {'path': str(prev_path), 'sha256': digest(prev_path)}
    p['selection']['revision_5_policy'] = 'Exactly the same queries, native 320x240 ROIs, reference and pose renderings, poses, and metrics as R4. The same one structural-boundary guide is used for every method within a row. No additional query or ROI screening.'
    for name in ['all_query_candidates.json', 'screen_displacement_candidates.json']:
        shutil.copy2(PREVIOUS / 'assets/e1' / name, ASSETS / name)
    for row in p['rows']:
        row['display_name'] = {'GreatCourt': 'Great Court', 'StMarysChurch': "St. Mary's Church", 'ShopFacade': 'Shop Facade'}[row['scene']]
        row['structural_guide'] = {
            'GreatCourt': {'box_xyxy_crop_pixels': [254, 98, 304, 142], 'structure': 'right column at the second band boundary'},
            'StMarysChurch': {'box_xyxy_crop_pixels': [125, 101, 168, 139], 'structure': 'vertical architectural frame at the second band boundary'},
            'ShopFacade': {'box_xyxy_crop_pixels': [104, 40, 143, 80], 'structure': 'vertical window frame at the first band boundary'},
        }[row['scene']]
        row['structural_guide'].update(identical_position_for_all_methods=True, arm_length_crop_pixels=6, fill='none', vector_overlay=True)
        for method, item in row['renders'].items():
            src = Path(item['path'])
            dst = ASSETS / src.name
            assert digest(src) == item['sha256']
            shutil.copy2(src, dst)
            item['path'] = str(dst)
            item['reuse'] = 'Byte-identical to R4; no rerendering.'
        src = Path(row['local_query_image'])
        dst = ASSETS / src.name
        shutil.copy2(src, dst)
        row['local_query_image'] = str(dst)
        row['local_query_sha256'] = digest(dst)
        row['local_query_pixels_match_original'] = bool(np.array_equal(
            np.asarray(Image.open(row['query_path']).convert('RGB')),
            np.asarray(Image.open(dst).convert('RGB'))))
        assert row['local_query_pixels_match_original']
    for key in ['outputs', 'validation', 'display', 'caption']:
        p.pop(key, None)
    save(p)
    print('Prepared R5; byte-identical R4 source renders and query images.', flush=True)


def source_band_legend(fig):
    """Four neutral image strips; edge colors identify source, not accuracy."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    # Physical schematic size is 16 x 16 points, independent of panel contents.
    sx, sy, sw, sh = .407, .8262, 16/(72*7.1), 16/(72*3.9)
    ax = fig.add_axes([sx, sy, sw, sh])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor='#eef0f2', edgecolor='#aab3be', linewidth=.35))
    # A small neutral window frame ties the schematic to the architectural panels.
    for x in [.29, .74]:
        ax.plot([x, x], [.05, .95], color='#8793a0', linewidth=.65)
    ax.plot([.29, .74], [.53, .53], color='#8793a0', linewidth=.65)
    for b in range(4):
        ax.add_patch(Rectangle((0, 1-(b+1)/4), .075, .25,
                               facecolor=GT if b%2 == 0 else EST, edgecolor='none'))
        if b:
            ax.plot([.09, 1], [1-b/4, 1-b/4], color='white', linewidth=.5)
    fig.text(sx-.019, sy+sh/2, 'Shared reference', ha='right', va='center', fontsize=7, color=GRAY)
    tx = sx+sw+.022
    for y, label, color in [(sy+sh*.74, 'Render at GT pose', GT),
                            (sy+sh*.25, 'Render at estimated pose', EST)]:
        fig.add_artist(Line2D([tx-.013, tx-.005], [y, y], transform=fig.transFigure, color=color, lw=1.6))
        fig.text(tx, y, label, ha='left', va='center', fontsize=7, color=INK)


def structural_guide(ax, box):
    """Four short transparent corner brackets; structure remains unobscured."""
    import matplotlib.patheffects as pe
    x0, y0, x1, y1 = box
    arm = 6
    artists = []
    for x, y, dx, dy in [(x0,y0,1,1), (x1,y0,-1,1), (x0,y1,1,-1), (x1,y1,-1,-1)]:
        line, = ax.plot([x+dx*arm,x,x], [y,y,y+dy*arm], color='white',
                        linewidth=.48, solid_capstyle='butt', solid_joinstyle='miter', zorder=5,
                        path_effects=[pe.Stroke(linewidth=.95, foreground='#25313c', alpha=.72), pe.Normal()])
        artists.append(line)
    return artists

def draw():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
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
    source_band_legend(fig)
    validations = []
    guide_artists = []
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
        fig.text(xs[0]+w/2, bottom-.016, f'({chr(97+i)}) '+row['display_name'], ha='center', va='top', fontsize=7, weight='bold')
        edges = np.linspace(0, ch, 5).astype(int)
        exact, te_errors, re_errors = [], [], []
        for j, method in enumerate(METHODS, 1):
            est = ims[method][y0:y1, x0:x1]
            composed = gt.copy()
            for b in [1, 3]:
                composed[edges[b]:edges[b+1]] = est[edges[b]:edges[b+1]]
            band_path = ASSETS / f'{row["stem"]}_{method}_bands.png'
            Image.fromarray(composed).save(band_path)
            previous_band = PREVIOUS / 'assets/e1' / band_path.name
            assert digest(band_path) == digest(previous_band)
            ax = fig.add_axes([xs[j], bottom, w, h])
            ax.imshow(composed)
            ax.set_xlim(0, cw)
            ax.set_ylim(ch, 0)
            ax.axis('off')
            for b in range(4):
                ax.plot([-3, -3], edges[b:b+2], color=GT if b%2 == 0 else EST, lw=1.6, clip_on=False)
                sl = slice(edges[b], edges[b+1])
                exact.append(np.array_equal(composed[sl], (gt if b%2 == 0 else est)[sl]))
            guide_artists.extend(structural_guide(ax, row['structural_guide']['box_xyxy_crop_pixels']))
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
                            'all_band_PNGs_byte_identical_to_R4': True,
                            'shared_structural_guide': row['structural_guide'],
                            'recomputed_TE_max_error_cm': max(te_errors), 'recomputed_RE_max_error_deg': max(re_errors),
                            'native_results_hash_pose_and_TE_RE_exact_match': archived_stage_checks})
    for artist in guide_artists:
        artist.set_visible(False)
    unmarked_path = ASSETS / 'E1_unmarked_reference.png'
    fig.savefig(unmarked_path, dpi=400, pad_inches=0)
    assert np.array_equal(np.array(Image.open(unmarked_path)), np.array(Image.open(PREVIOUS / 'E1_final_localization.png')))
    for artist in guide_artists:
        artist.set_visible(True)
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / f'E1_final_localization.{ext}', dpi=400, pad_inches=0)
    plt.close(fig)
    caption = ('Selected favorable localization examples comparing STDLoc and ULF-Loc coarse/final poses with single-stage AnyGSLoc. All panels in a row share one reference Gaussian model, intrinsics, and crop. Bands 1/3 render the GT pose; bands 2/4 render the estimated pose. Source colors indicate GT/estimated rendering, and identical corner guides highlight a structural boundary for comparison. Continuity across boundaries indicates alignment. Numbers report translation error (cm) / rotation error (degrees). The shared reference is used only for offline visualization and is absent from AnyGSLoc online localization. Aggregate accuracy is reported separately.')
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
                    'GT_bands_1indexed': [1, 3], 'estimated_bands_1indexed': [2, 4],
                    'source_schematic': 'Four neutral grey bands with a window-frame schematic; green/magenta edge bars denote GT/estimated render sources. No plotted data are altered.',
                    'unchanged_R4_panel_coordinates_and_underlying_source_pixels': True,
                    'new_vector_overlay': 'One open-corner guide per row, at exactly the same crop-pixel coordinates in every method panel. No arrow, magnification, displacement scaling, or image modification.',
                    'unmarked_export_pixels_identical_to_R4': True,
                    'unmarked_reference': {'path': str(unmarked_path), 'sha256': digest(unmarked_path)},
                    'scene_display_names': {r['scene']: r['display_name'] for r in p['rows']}}
    p['source_script'] = {'path': str(Path(__file__).resolve()), 'sha256': digest(__file__)}
    p['outputs'] = {ext: {'path': str(OUT/f'E1_final_localization.{ext}'), 'sha256': digest(OUT/f'E1_final_localization.{ext}')} for ext in ['pdf', 'svg', 'png']}
    (OUT / 'E1_final_localization.caption.txt').write_text(caption+'\n')
    save(p)
    print('Wrote R5 E1 boundary-guide refinement:', validations, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'draw', 'all'])
    args = ap.parse_args()
    if args.mode in ['prepare', 'all']:
        prepare()
    if args.mode in ['draw', 'all']:
        draw()
