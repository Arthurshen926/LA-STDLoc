"""Build E2 from frozen geometry controls and archived, independently matched query.

No feature extraction, matching, map construction, or localization is performed.
The only computations are artifact checks, projections, and descriptive statistics.
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/experiments_cvpr_20261003'
A = OUT / 'assets/e2'
CONTROL = Path('/mnt/pool/sqy/anygsloc_center_control_20260915')
ARCHIVE = ROOT / 'paper/fig1_geometry_sources_v2_20261002'
TABLE = ROOT / 'docs/paper/evidence_20260917/geometry_15.csv'
SCENES = ['GreatCourt', 'KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch']
KINDS = ['center', 'depth', 'ray']
COLORS = {'center': '#C28A37', 'depth': '#427AB0', 'ray': '#087F72'}
MARKERS = {'center': 's', 'depth': '^', 'ray': 'o'}
LABELS = {'center': 'Gaussian-center aggregate', 'depth': 'Rendered-depth aggregate', 'ray': 'Multiview rays'}
INK = '#233447'
MUTED = '#637181'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def project(x, pose, K):
    pose, K = np.asarray(pose), np.asarray(K)
    p = (np.asarray(x) @ pose[:3, :3].T + pose[:3, 3]) @ K.T
    assert p[2] > 0
    return p[:2] / p[2] - .5  # native feature-grid coordinates


def audit():
    torch.set_num_threads(4)
    records = list(csv.DictReader(TABLE.open()))
    checks = []
    for scene in SCENES:
        maps = {kind: load(CONTROL / f'maps/{scene}/{kind}/deployment_map.pt') for kind in KINDS}
        check = {'scene': scene, 'landmarks': len(maps['ray']['anchor_ids']), 'controls': {}}
        for kind in KINDS:
            path = CONTROL / f'maps/{scene}/{kind}/deployment_map.pt'
            run = CONTROL / f'runs/{scene}/{kind}'
            manifest = json.loads((run / 'manifest.json').read_text())
            results = json.loads((run / 'results.json').read_text())
            assert torch.equal(maps[kind]['anchor_ids'], maps['ray']['anchor_ids'])
            assert torch.equal(maps[kind]['anchor_features'], maps['ray']['anchor_features'])
            digest = sha(path)
            assert manifest['map_sha256'] == digest
            te = np.array([r['translation_error_cm'] for r in results])
            re = np.array([r['rotation_error_deg'] for r in results])
            row = next(r for r in records if r['scene'] == scene and r['geometry'] == kind)
            assert len(results) == int(row['queries'])
            assert np.isclose(np.median(te), float(row['TE_cm']), atol=1e-8)
            assert np.isclose(np.mean((te < 5) & (re < 5)) * 100, float(row['R5_pct']), atol=1e-8)
            check['controls'][kind] = {'map': str(path), 'map_sha256': digest,
                'run': str(run), 'results_sha256': sha(run/'results.json'),
                'ids_bitwise_equal_to_ray': True, 'descriptors_bitwise_equal_to_ray': True,
                'median_TE_cm': float(np.median(te)), 'R5_pct': float(np.mean((te < 5) & (re < 5))*100)}
        checks.append(check)
        del maps

    e = json.loads((ARCHIVE / 'evidence.json').read_text())
    source = load(e['compact_map'])
    maps = {k: load(CONTROL / f'maps/KingsCollege/{k}/deployment_map.pt') for k in KINDS}
    srcrow = int(e['landmark_row'])
    equal = torch.all(maps['ray']['anchor_xyz'] == source['anchor_xyz'][srcrow], dim=1)
    rows = torch.where(equal)[0]
    assert len(rows) == 1
    row = int(rows[0])
    assert torch.equal(maps['ray']['anchor_features'][row], source['anchor_features'][srcrow])
    track = load(CONTROL / 'maps/KingsCollege/ray/projective_anchor_map.pt')
    assert e['query']['name'] not in track['v6_mapping_query_names']
    obs = track['projective_anchor_observations']
    start, end = map(int, obs['observation_offsets'][row:row+2])
    observed = set(zip(obs['query_indices'][start:end].tolist(), obs['keypoint_indices'][start:end].tolist()))
    for v in e['views']:
        pair = (track['v6_mapping_query_names'].index(v['name']), v['keypoint'])
        assert pair in observed
    xyz = {k: maps[k]['anchor_xyz'][row].numpy().tolist() for k in KINDS}
    views = []
    for i, v in enumerate(e['views']):
        im = Image.open(ARCHIVE / f'assets/render_{i}.png').convert('RGB')
        u = np.asarray(v['u'])
        box = [int(u[0]-64), int(u[1]-64), int(u[0]+64), int(u[1]+64)]
        path = A / f'mapping_{i}.png'
        im.crop(box).save(path)
        predictions = {k: project(xyz[k], v['pose'], v['K']).tolist() for k in KINDS}
        views.append({'name': v['name'], 'image': str(path), 'observation': v['u'],
            'crop': box, 'predictions': predictions, 'pose_w2c': v['pose'], 'K': v['K'],
            'residual_px': {k: float(np.linalg.norm(np.array(predictions[k])-u)) for k in KINDS}})
    q = e['query']
    u = np.asarray(e['query_feature_grid_uv'])
    im = Image.open(q['path']).convert('RGB')
    box = [int(u[0]-64), int(u[1]-64), int(u[0]+64), int(u[1]+64)]
    path = A / 'independent_query.png'
    im.crop(box).save(path)
    predictions = {k: project(xyz[k], q['pose'], q['K']).tolist() for k in KINDS}
    views.append({'name': q['name'], 'image': str(path), 'observation': u.tolist(), 'crop': box,
        'predictions': predictions, 'pose_w2c': q['pose'], 'K': q['K'],
        'residual_px': {k: float(np.linalg.norm(np.array(predictions[k])-u)) for k in KINDS}})
    np.savez_compressed(A/'track_geometry.npz', **{f'xyz_{k}': np.array(xyz[k]) for k in KINDS},
        mapping_camera_centers=np.array([v['C'] for v in e['views']]))
    limits = [
        'Local example was inherited from archived Fig1: a descriptor top-1 query correspondence verified with the dataset reference pose (<3 px). It is an illustrative, selected correspondence, not an unbiased query-reprojection statistic.',
        'The plus marker is an archived detected feature, not a manually annotated ground-truth landmark. Independent-query residuals use the dataset reference pose and are distinct from estimated-pose accuracy below.',
        'All coordinate controls have bitwise-identical deployed anchor IDs and descriptors within each scene. Center/depth are coordinatewise medians of the same usable observations (at least three). Ray XYZ retains the original track support; this is an XYZ estimator control, not a whole association-pipeline ablation.',
        'The Gaussian-center control selects the nearest z-buffered projected primitive center within 2 px, then aggregates across observations. It is not an individual primitive center and does not use full Gaussian alpha compositing for visibility.',
        'Mapping patches are previously rendered Gaussian-prior RGB. The query patch is a real processed query image and is absent from mapping support. No new matching, map construction, or localization runs were performed.',
        'Projection uses feature-grid coordinates pi(TX)-0.5 consistently; all residual panels have identical pixel limits. Magnified residual plots are separate from 128x128 image patches.',
        'The five-scene result includes KingsCollege, where ray TE is worse than depth and ray R5 is worse than center. No monotonic improvement is claimed.'
    ]
    provenance = {'figure': 'E2_geometry_source_evidence', 'artifact_only': True,
        'data_table': str(TABLE), 'data_table_sha256': sha(TABLE), 'control_audit': checks,
        'archived_local_evidence': str(ARCHIVE/'evidence.json'),
        'archived_local_evidence_sha256': sha(ARCHIVE/'evidence.json'),
        'scene': 'KingsCollege', 'source_compact_row': srcrow, 'control_common_row': row,
        'row_join': 'unique bitwise-equal ray XYZ, with bitwise-equal descriptor verified',
        'mapping_support': end-start, 'query_absent_from_mapping_support': True,
        'xyz': xyz, 'views': views, 'limitations': limits,
        'query_match_selection': q['policy'],
        'source_control_code': str(ROOT/'tools/gaussian_center_control.py')}
    shutil.copyfile(TABLE, A/'geometry_15.csv')
    return records, provenance


def plot(records, p):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
        'axes.labelsize': 8, 'axes.titlesize': 8, 'text.color': INK,
        'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    fig = plt.figure(figsize=(7.10, 5.30))
    # Dedicated positions keep labels out of the photographs and plotting areas.
    xs = [.058, .296, .534, .772]
    width = .176
    fig.text(.058, .985, '(a) One landmark: mapping support and independent-query evidence',
             va='top', fontsize=9, weight='bold')
    fig.text(.058, .950, 'KingsCollege  |  fixed landmark identity + descriptor  |  only XYZ changes',
             va='top', fontsize=7.5, color=MUTED)
    for i, v in enumerate(p['views']):
        ax = fig.add_axes([xs[i], .654, width, .252])
        ax.imshow(Image.open(v['image']), interpolation='nearest')
        ax.plot(64, 64, marker='+', ms=8, color='white', mew=1.6)
        ax.add_patch(Rectangle((56, 56), 16, 16, fill=False, edgecolor='white', lw=.65, ls=':'))
        ax.set_axis_off()
        title = f'Mapping {i+1}' if i < 3 else 'Real query (held out)'
        ax.set_title(title, pad=4, fontsize=8, color=INK, weight='bold')
        ax.text(.5, -.055, v['name'], ha='center', va='top', transform=ax.transAxes, fontsize=7, color=MUTED)
        ax = fig.add_axes([xs[i], .438, width, .162])
        ax.set_facecolor('#FAFBFC')
        ax.axhline(0, lw=.6, color='#CDD3D9'); ax.axvline(0, lw=.6, color='#CDD3D9')
        ax.scatter(0, 0, marker='+', s=58, color=INK, linewidths=1.25, zorder=8)
        for kind in KINDS:
            delta = np.array(v['predictions'][kind])-v['observation']
            ax.annotate('', xy=delta, xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=COLORS[kind], lw=1, shrinkA=1, shrinkB=1))
            ax.scatter(*delta, marker=MARKERS[kind], s=28, facecolors='white', edgecolors=COLORS[kind], linewidths=1.1, zorder=6)
        ax.set(xlim=(-4.5, 1.5), ylim=(3, -1.6), xticks=[-4, -2, 0], yticks=[0, 2])
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(labelsize=7, length=2, pad=1.5)
        ax.set_xlabel('Horizontal offset (px)', fontsize=7, labelpad=1)
        if i == 0: ax.set_ylabel('Vertical offset (px)', fontsize=7, labelpad=2)
        for j, kind in enumerate(KINDS):
            ax.text((j+.5)/3, -.46, f"{v['residual_px'][kind]:.2f}",
                transform=ax.transAxes, ha='center', color=COLORS[kind], fontsize=7, weight='bold')
    fig.text(.058, .616, 'Offsets from the feature (+); identical pixel axes. Colored values: residual norm (px).',
        fontsize=7, color=MUTED)
    handles = [Line2D([], [], color=COLORS[k], marker=MARKERS[k], ms=5, lw=0,
               markerfacecolor='white', label=LABELS[k]) for k in KINDS]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .309),
               ncol=3, frameon=False, fontsize=7.4, columnspacing=1.3, handletextpad=.25)
    for pi, (key, title, ylab, ylim) in enumerate([
        ('TE_cm', '(b) Held-out localization error', 'Median TE (cm)  ↓', (0,24)),
        ('R5_pct', '(c) Held-out localization recall', r'$R_{5\,\mathrm{cm},\,5^\circ}$ (%)  ↑', (0,100))]):
        ax = fig.add_axes([.068 + pi*.490, .092, .398, .186])
        ax.axvspan(.5, 1.5, color='#FFF2E7', zorder=0)
        for j, kind in enumerate(KINDS):
            val = [float(next(r for r in records if r['scene']==s and r['geometry']==kind)[key]) for s in SCENES]
            x = np.arange(5) + (j-1)*.20
            ax.vlines(x, 0, val, colors=COLORS[kind], alpha=.18, linewidth=.7)
            ax.scatter(x, val, s=24, marker=MARKERS[kind], color=COLORS[kind], zorder=3)
        ax.set(xticks=np.arange(5), xticklabels=['GC','KC','OH','SF','SM'], ylim=ylim, xlim=(-.5,4.5))
        ax.set_ylabel(ylab, labelpad=2, fontsize=7.8)
        ax.set_title(title, loc='left', fontsize=8.4, weight='bold', pad=6)
        ax.grid(axis='y', color='#DFE4E8', lw=.5)
        ax.set_axisbelow(True); ax.tick_params(axis='both', labelsize=7, length=2, pad=2)
    fig.text(.5, .018, 'All five scenes retained; KingsCollege (shaded) is a counterexample to uniform improvement.',
        ha='center', color=MUTED, fontsize=7)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    text_objects=[t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    outside=[]
    for t in text_objects:
        b=t.get_window_extent(renderer)
        if b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.width+.5 or b.y1>fig.bbox.height+.5:
            outside.append(t.get_text())
    assert not outside, outside
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/f'E2_geometry_source_evidence.{ext}', dpi=320)
    plt.close(fig)
    return {'visible_text_items':len(text_objects),'text_outside_canvas':outside,
        'minimum_declared_font_pt':min(t.get_fontsize() for t in text_objects),
        'figure_inches':[7.1,5.3],'visual_inspection':'PNG inspected; title and residual labels separated.'}


def main():
    A.mkdir(parents=True, exist_ok=True)
    records, p = audit()
    p['render_validation']=plot(records, p)
    caption = (
        'Geometry-source evidence with frozen landmark identities and descriptors. '
        '(a) Three archived Gaussian-rendered mapping patches and one real held-out query show the same KingsCollege landmark. '
        'The plus denotes the recorded image feature. Offset plots use identical native-pixel axes; colored numbers give Euclidean reprojection residuals. '
        'The query correspondence is archived, independently matched to the compact map, and verified to have identical XYZ/descriptor in the control map; its residual uses the dataset reference pose. '
        '(b,c) Existing test results for all five Cambridge scenes compare Gaussian-center aggregation, rendered-depth aggregation, and multiview-ray coordinates. '
        'All 15 map/run bindings and bitwise equality of deployed identities/descriptors were checked. '
        'Center/depth aggregate the same usable observations, while ray geometry retains its original track support. '
        'The selected local example does not estimate average query behavior: KingsCollege is retained as an aggregate counterexample (depth has lower TE and center higher recall than rays). '
        'GC/KC/OH/SF/SM denote GreatCourt, KingsCollege, OldHospital, ShopFacade, and StMarysChurch. '
        'No new localization experiments were run.'
    )
    (OUT/'E2_geometry_source_evidence.caption.txt').write_text(caption+'\n\nLimitations and provenance:\n'+'\n'.join('- '+x for x in p['limitations'])+'\n')
    p['outputs'] = {ext: {'path': str(OUT/f'E2_geometry_source_evidence.{ext}'),
                         'sha256': sha(OUT/f'E2_geometry_source_evidence.{ext}')} for ext in ['pdf','svg','png']}
    (OUT/'E2_geometry_source_evidence.provenance.json').write_text(json.dumps(p, indent=2)+'\n')
    print(json.dumps({'outputs': list(p['outputs']), 'audited_scenes': len(p['control_audit']),
        'query_residual_px': p['views'][-1]['residual_px'], 'common_row': p['control_common_row']}, indent=2))


if __name__ == '__main__':
    main()
