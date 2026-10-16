"""Readability-only revision of E4; reuse every R3 query, pose, render and PCA."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / 'paper/experiments_cvpr_r3_20261003'
OUT = ROOT / 'paper/experiments_cvpr_r4_20261003'
A = OUT / 'assets/e4'
P = OUT / 'E4_gaussian_prior.provenance.json'
KEYS = ['matcha', 'vanilla_2dgs', 'vanilla_3dgs', 'i3dgs_raw', 'i3dgs_1k']
LABELS = ['MAtCHA', '2DGS', '3DGS', 'i3DGS raw', 'i3DGS +1k']
SCENE_LABELS = {'StMarysChurch': "St. Mary's Church", 'ShopFacade': 'Shop Facade'}
INK, MUTED = '#243349', '#607083'
GREEN, MAGENTA = '#219653', '#cc2f83'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def prepare():
    """Copy only selected R3 assets; do not invoke rendering or estimation."""
    A.mkdir(parents=True, exist_ok=True)
    d = read(SOURCE / 'E4_gaussian_prior.provenance.json')
    filenames = ['shared_pca.npz']
    for cid in d['selected']:
        filenames += [cid + s for s in ['_real_query.png', '_gt_reference_crop.png', '_gt_reference_full_2880.png']]
        filenames += [cid + '_' + key + suffix for key in KEYS for suffix in
                      ['_prior.png', '_map.npz', '_alignment.png', '_reference_full_2880.png']]
    for name in filenames:
        shutil.copy2(SOURCE / 'assets/e4' / name, A / name)
    d['r4_revision'] = {
        'scope': 'Readability-only revision; no changes to cases, poses, renders, crops, descriptors, PCA or map points.',
        'source_provenance': str(SOURCE / 'E4_gaussian_prior.provenance.json'),
        'source_provenance_sha256': digest(SOURCE / 'E4_gaussian_prior.provenance.json'),
        'new_render_calls': 0,
        'source_asset_sha256': {name: digest(A / name) for name in filenames},
        'changes': ['Human-readable scene names', 'Explicit prior/map/alignment row labels',
                    'One global four-band key shared by both cases',
                    'Render at GT pose replaces GT reference',
                    'Caption clarifies shared display model and total scene landmark count']
    }
    P.write_text(json.dumps(d, indent=2))


def draw():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    d = read(P)
    fw, fh = 7.1, 6.3
    fig = plt.figure(figsize=(fw, fh), facecolor='white')
    margin, qwidth, gap = .1, .92, .065
    pw = (fw - 2 * margin - qwidth - 5 * gap) / 5
    xs = [margin] + [margin + qwidth + gap + i * (pw + gap) for i in range(5)]
    widths = [qwidth] + [pw] * 5

    def text_in(x, y, text, **kw):
        return fig.text(x / fw, 1 - y / fh, text, **kw)

    def panel(x, top, width, height):
        return fig.add_axes([x / fw, 1 - (top + height) / fh, width / fw, height / fh])

    for j, label in enumerate(['Real query'] + LABELS):
        text_in(xs[j] + widths[j] / 2, .075, label, ha='center', va='top',
                weight='bold', fontsize=7.8, color=INK)

    for ci, cid in enumerate(d['selected']):
        item = next(x for x in d['candidates'] if x['id'] == cid)
        scene = item['scene']
        offset = ci * 3.1
        text_in(margin, .34 + offset, f"({chr(97 + ci)})  {SCENE_LABELS[scene]}",
                fontsize=8, weight='bold', color=INK)
        text_in(xs[1], .50 + offset, 'Prior render · same GT camera',
                ha='left', va='bottom', fontsize=7, color=MUTED)
        tops = [.55 + offset, 1.31 + offset, 2.24 + offset]
        ih = pw * 9 / 16

        photo = Image.open(A / (cid + '_real_query.png'))
        ax = panel(xs[0], tops[0] + (ih - qwidth * 9 / 16) / 2, qwidth, qwidth * 9 / 16)
        ax.imshow(photo)
        ax.axis('off')
        roi = item['alignment']['crop_native_px']
        ax.add_patch(Rectangle((roi[0], roi[1]), roi[2] - roi[0], roi[3] - roi[1],
                               fill=False, ec='white', lw=.7))

        text_in(xs[0] + qwidth / 2, tops[1] + .035, 'Compact\nlandmarks',
                ha='center', va='top', linespacing=1.25, weight='bold', color=INK)
        text_in(xs[0] + qwidth / 2, tops[1] + .355, 'Shared\ndescriptor PCA',
                ha='center', va='top', linespacing=1.25, color=MUTED)
        text_in(xs[0] + qwidth / 2, tops[2] - .01, 'Pose alignment',
                ha='center', va='bottom', weight='bold', color=INK)

        gt = Image.open(A / (cid + '_gt_reference_crop.png'))
        ah = pw * 150 / 240
        ax = panel(xs[0], tops[2] + (ah - qwidth * 150 / 240) / 2, qwidth, qwidth * 150 / 240)
        ax.imshow(gt)
        ax.axis('off')
        ax.plot([0, 0], [0, gt.height], color=GREEN, lw=2, clip_on=False)
        text_in(xs[0] + qwidth / 2, tops[2] + ah + .095, 'Render at GT pose',
                ha='center', va='top', color=INK, fontsize=7)

        for j, key in enumerate(KEYS, 1):
            src, x = d['scenes'][scene]['priors'][key], xs[j]
            ax = panel(x, tops[0], pw, ih)
            ax.imshow(Image.open(A / (cid + '_' + key + '_prior.png')))
            ax.axis('off')
            ax = panel(x, tops[1], pw, ih)
            m = np.load(A / (cid + '_' + key + '_map.npz'))
            order = np.argsort(m['depth'])[::-1]
            ax.set_facecolor('#f6f7f9')
            ax.scatter(m['uv'][order, 0], m['uv'][order, 1], c=m['colors'][order],
                       s=.08, lw=0, rasterized=True)
            ax.set_xlim(0, item['width'])
            ax.set_ylim(item['height'], 0)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            text_in(x + pw / 2, tops[1] + ih + .075, f"N = {src['landmark_count'] / 1000:.1f}k",
                    ha='center', va='top', color=MUTED, fontsize=7)

            im = Image.open(A / (cid + '_' + key + '_alignment.png'))
            ax = panel(x, tops[2], pw, ah)
            ax.imshow(im, interpolation='nearest')
            ax.axis('off')
            cuts = np.linspace(0, im.height, 5)
            for band in range(4):
                ax.plot([0, 0], [cuts[band], cuts[band + 1]],
                        color=GREEN if band % 2 == 0 else MAGENTA, lw=2, clip_on=False)
            r = item['results'][key]
            text_in(x + pw / 2, tops[2] + ah + .095,
                    f"{r['translation_error_cm']:.2f} cm / {r['rotation_error_deg']:.3f}°",
                    ha='center', va='top', fontsize=7, color=INK)

        if ci == 0:
            fig.add_artist(plt.Line2D([margin / fw, (fw - margin) / fw], [1 - 3.22 / fh] * 2,
                                     transform=fig.transFigure, color='#dbe0e6', lw=.6))

    # Four-band neutral diagram: color indicates render source, never correctness.
    # This uses the same visual grammar and wording as E1.
    legend_top = .275
    text_in(4.39, legend_top + .111, 'Shared reference', ha='right', va='center', color=MUTED)
    ax = panel(4.52, legend_top, 16 / 72, 16 / 72)
    ax.set(xlim=(0, 1), ylim=(1, 0))
    ax.axis('off')
    ax.add_patch(Rectangle((0, 0), 1, 1, fc='#eef0f2', ec='#aab3be', lw=.35))
    for x in [.29, .74]:
        ax.plot([x, x], [.05, .95], color='#8793a0', lw=.65)
    ax.plot([.29, .74], [.47, .47], color='#8793a0', lw=.65)
    for band in range(4):
        ax.add_patch(Rectangle((0, band / 4), .075, .25,
                               fc=GREEN if band % 2 == 0 else MAGENTA, ec='none'))
    for y in [.25, .5, .75]:
        ax.plot([.09, 1], [y, y], color='white', lw=.5)
    for y, color, label in [(legend_top + .058, GREEN, 'Render at GT pose'),
                             (legend_top + .167, MAGENTA, 'Render at estimated pose')]:
        fig.add_artist(plt.Line2D([4.82 / fw, 4.88 / fw], [1 - y / fh] * 2,
                                 transform=fig.transFigure, color=color, lw=1.6))
        text_in(4.92, y, label, ha='left', va='center', color=INK, fontsize=7)

    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / ('E4_gaussian_prior.' + ext), dpi=400, facecolor='white')
    plt.close(fig)

    caption = (
        'Gaussian-prior compatibility in two illustrative cases. Each case shows prior RGB rendered at '
        'the same calibrated GT camera (top), compact landmarks with shared descriptor PCA colors (middle), '
        'and pose alignment in one common display model (bottom). The appearance and landmark rows use '
        'each indicated prior. In the alignment row, all five recorded estimates and GT are instead '
        'rendered with the same frozen MAtCHA reference model, changing only the camera pose. Four '
        'horizontal bands alternate render at GT pose (green edge) and render at estimated pose '
        '(magenta edge); the edge colors identify the render source, not correctness. This common-model '
        'comparison exposes pose disagreement without attributing reference-render appearance to the '
        'input localization prior. White rectangles locate the same crop used for all five estimates '
        'in each case. The four non-raw estimates align closely; raw i3DGS gives visible contour '
        'discontinuities. Labels report recorded translation/rotation errors; N is the total landmark '
        'count in the scene deployment map, not the number visible in the panel. '
        "The examples, St. Mary's Church seq13/frame00161 and Shop Facade seq3/frame00047, were deliberately "
        'selected from the archived shortlist so that MAtCHA, 2DGS, 3DGS, and i3DGS+1k each satisfy '
        'TE < 5 cm and RE < 0.3 degrees, while raw disagreement is visually interpretable; they are not '
        'representative error samples. Each common crop covers 240 x 150 original-camera pixels; '
        'reference images were supersampled equally for print quality. No pose displacement or camera '
        'alignment is altered. A single PCA basis and normalization are fitted to equal samples from '
        'all ten scene/prior maps; colors encode descriptor directions, not semantics. MAtCHA uses the '
        'factorial v1_f1_c1_compact map and other producers use archived prior-suite compact maps. '
        'The reference model is an offline visualization tool, not ground-truth appearance and not an '
        'input to AnyGSLoc online localization. This revision reuses all R3 images and data; no rendering, '
        'training, reconstruction, or localization was rerun.'
    )
    (OUT / 'E4_gaussian_prior.caption.txt').write_text(caption + '\n')
    d['outputs'] = {e: {'path': str(OUT / ('E4_gaussian_prior.' + e)),
                        'sha256': digest(OUT / ('E4_gaussian_prior.' + e))} for e in ['pdf', 'svg', 'png']}
    d['builder'] = {'path': str(Path(__file__).resolve()), 'sha256': digest(__file__),
                    'phases': ['prepare', 'draw', 'validate'],
                    'python': '/root/miniconda3/envs/g4splat/bin/python', 'GPU_used': False}
    d['asset_audit'] = {'shared_pca_sha256': digest(A / 'shared_pca.npz'),
                        'selected': {cid: {'assets': {path.name: digest(path)
                                     for path in sorted(A.glob(cid + '_*')) if path.is_file()}}
                                     for cid in d['selected']}}
    d['r4_revision']['layout'] = {'size_inches': [fw, fh], 'minimum_font_pt': 7,
                                  'shared_legend': 'One four-band source diagram for both cases',
                                  'scene_display_names': SCENE_LABELS}
    P.write_text(json.dumps(d, indent=2))


def validate():
    from PIL import Image
    from anygsloc.evaluation.metrics import pose_error
    sys.path.insert(0, '/root/.cache/anygsloc_figure_deps')
    import fitz

    d = read(P)
    report = {'new_localization_runs': 0, 'new_training_runs': 0, 'new_render_calls': 0,
              'cases': {}, 'figure': {}}
    for name, expected in d['r4_revision']['source_asset_sha256'].items():
        assert digest(A / name) == expected == digest(SOURCE / 'assets/e4' / name)
    report['all_selected_assets_bit_identical_to_R3'] = True
    for cid in d['selected']:
        item = next(x for x in d['candidates'] if x['id'] == cid)
        x, y, x1, y1 = item['alignment']['crop_render_px']
        gt = np.asarray(Image.open(A / (cid + '_gt_reference_full_2880.png')))[y:y1, x:x1]
        assert np.array_equal(gt, np.asarray(Image.open(A / (cid + '_gt_reference_crop.png'))))
        stats = {}
        for key in KEYS:
            src = d['scenes'][item['scene']]['priors'][key]
            assert digest(src['map']) == src['map_sha256']
            assert digest(src['results']) == src['results_sha256']
            re, te = pose_error(np.asarray(item['results'][key]['pose_w2c']), np.asarray(item['gt_pose_w2c']))
            assert abs(te - item['results'][key]['translation_error_cm']) < .002
            assert abs(re - item['results'][key]['rotation_error_deg']) < .0002
            pred = np.asarray(Image.open(A / (cid + '_' + key + '_reference_full_2880.png')))[y:y1, x:x1]
            mix, cuts = gt.copy(), np.linspace(0, len(gt), 5, dtype=int)
            for b in [1, 3]:
                mix[cuts[b]:cuts[b + 1]] = pred[cuts[b]:cuts[b + 1]]
            assert np.array_equal(mix, np.asarray(Image.open(A / (cid + '_' + key + '_alignment.png'))))
            stats[key] = {'recorded_TE_RE_verified': True, 'map_and_result_hashes_verified': True,
                           'exact_unwarped_band_composition_verified': True, 'TE_cm': float(te), 'RE_deg': float(re)}
        report['cases'][cid] = stats
    doc = fitz.open(OUT / 'E4_gaussian_prior.pdf')
    p = doc[0]
    spans = [s for b in p.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
    assert len(doc) == 1 and min(s['size'] for s in spans) >= 6.99
    assert all(p.rect.contains(fitz.Rect(s['bbox'])) for s in spans)
    fonts = p.get_fonts(full=True)
    assert all(f[2] != 'Type3' for f in fonts)
    assert all(len(doc.extract_font(f[0])[3]) > 0 for f in fonts)
    pw = (7.1 - .2 - .92 - 5 * .065) / 5
    report['figure'] = {'size_inches': [p.rect.width / 72, p.rect.height / 72],
                        'PNG_dimensions': list(Image.open(OUT / 'E4_gaussian_prior.png').size),
                        'minimum_font_pt': min(s['size'] for s in spans), 'all_text_inside_page': True,
                        'fonts_embedded': True, 'no_type3_fonts': True,
                        'alignment_native_crop_PPI': 360 / pw, 'prior_RGB_native_PPI': 640 / pw}
    report['files'] = {ext: digest(OUT / ('E4_gaussian_prior.' + ext))
                       for ext in ['pdf', 'svg', 'png', 'caption.txt', 'provenance.json']}
    (OUT / 'E4_gaussian_prior.validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report['figure'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'draw', 'validate', 'all'], nargs='?', default='all')
    phase = parser.parse_args().phase
    if phase == 'all':
        prepare()
        draw()
        validate()
    else:
        globals()[phase]()
