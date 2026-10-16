"""R5: native-stage operating points with explicit timing scope.

Consumes archived metrics only. Fine time is an increment, not a third pose.
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[1]
PREV = ROOT / 'paper/experiments_cvpr_20261003'
OUT = ROOT / 'paper/experiments_cvpr_r5_20261003'
OUT.mkdir(parents=True, exist_ok=True)
COL = {'STDLoc': '#497fa6', 'ULFLoc': '#896897', 'AnyGSLoc': '#cb7327'}
INK, GRAY = '#263341', '#6c7782'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label(ax, xy, lines, color, offset, align=(0, .5), size=7.1):
    children = [TextArea(lines[0], textprops=dict(size=7.9, color=color, weight='semibold'))]
    children += [TextArea(line, textprops=dict(size=size, color=INK)) for line in lines[1:]]
    box = VPacker(children=children, pad=0, sep=2.0,
                  align='left' if align[0] == 0 else 'right' if align[0] == 1 else 'center')
    artist = AnnotationBbox(box, xy, xybox=offset, xycoords='data', boxcoords='offset points',
                           box_alignment=align, frameon=False, pad=0)
    ax.add_artist(artist)
    return artist


def main():
    old = json.loads((PREV / 'E5_deployment.json').read_text())
    points = old['points']
    final = [p for p in points if p['stage'] in ['final', 'single']]
    stages = []
    for p in final:
        coarse = next((q for q in points if q['method'] == p['method'] and q['stage'] == 'coarse'), None)
        stages.append(dict(method=p['method'], coarse_prefix_ms=coarse['latency_ms'] if coarse else None,
                           fine_increment_ms=p['latency_ms'] - coarse['latency_ms'] if coarse else None,
                           total_latency_ms=p['latency_ms'], equivalent_FPS=1000 / p['latency_ms'],
                           coarse_R5_pct=coarse['R5_pct'] if coarse else None,
                           final_R5_pct=p['R5_pct'], TE_cm=p['TE_cm'], RE_deg=p['RE_deg'],
                           storage_MB=p['storage_MB']))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'text.color': INK,
                         'axes.labelcolor': INK, 'axes.edgecolor': '#9aa4ae',
                         'xtick.color': GRAY, 'ytick.color': GRAY, 'axes.linewidth': .65,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.65), gridspec_kw={'width_ratios': [1.13, 1]})
    fig.subplots_adjust(left=.095, right=.980, bottom=.24, top=.89, wspace=.39)
    specs = [
        ((18, 45.5), (43, 2650), [20, 25, 30, 35, 40, 45], [50, 100, 250, 500, 1000, 2000],
         '(a) Coarse-to-fine latency', 'Mean pipeline latency (ms, log) ↓'),
        ((36, 44.5), (105, 3800), [36, 38, 40, 42, 44], [125, 250, 500, 1000, 2000],
         '(b) Deployment storage', 'Five-scene storage (MB, log) ↓')]
    for ax, (xlim, ylim, xticks, yticks, title, ylabel) in zip(axs, specs):
        ax.set(xlim=xlim, ylim=ylim, yscale='log', xlabel='Recall @ 5 cm / 5° (%) ↑', ylabel=ylabel)
        ax.set_title(title, fontsize=9, weight='semibold', loc='left', pad=13)
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        ax.yaxis.set_major_locator(FixedLocator(yticks))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{v:,.0f}'))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(labelsize=7.1, length=2.8, pad=4, width=.65)
        ax.grid(axis='y', color='#dce2e7', lw=.55, alpha=.65)
        ax.set_axisbelow(True)
    # Scope labels sit between the title and axes, without changing the data region.
    for ax, subtitle in zip(axs, ['Native configurations · RTX 3090', 'Final outputs · five Cambridge scenes']):
        ax.text(0, 1.014, subtitle, transform=ax.transAxes, fontsize=7, color=GRAY,
                ha='left', va='bottom')
    boxes = []
    ax = axs[0]
    for method in ['STDLoc', 'ULFLoc']:
        c = next(p for p in points if p['method'] == method and p['stage'] == 'coarse')
        f = next(p for p in points if p['method'] == method and p['stage'] == 'final')
        color = COL[method]
        ax.annotate('', xy=(f['R5_pct'], f['latency_ms']), xytext=(c['R5_pct'], c['latency_ms']),
                    arrowprops=dict(arrowstyle='-|>', lw=1.35, color=color, mutation_scale=9,
                                    shrinkA=5, shrinkB=6), zorder=2)
        ax.scatter(c['R5_pct'], c['latency_ms'], s=47, marker='o', facecolor='white',
                   edgecolor=color, lw=1.4, zorder=4)
        ax.scatter(f['R5_pct'], f['latency_ms'], s=48, marker='o', facecolor=color,
                   edgecolor='white', lw=.6, zorder=4)
        name = method.replace('ULFLoc', 'ULF-Loc†')
        if method == 'STDLoc':
            coarse_offset, coarse_align = (-4, -11), (1, 1)
            final_offset, final_align = (10, -5), (0, .5)
            fine_xy, fine_offset = (33.5, 325), (0, -13)
        else:
            coarse_offset, coarse_align = (0, -13), (0, 1)
            final_offset, final_align = (-1, 6), (1, 0)
            fine_xy, fine_offset = (29.8, 825), (0, 10)
        boxes.append(label(ax, (c['R5_pct'], c['latency_ms']),
                           [name + ' · coarse', f'{c["latency_ms"]:,.1f} ms'],
                           color, coarse_offset, coarse_align))
        boxes.append(label(ax, (f['R5_pct'], f['latency_ms']),
                           [name + ' · final', f'{f["latency_ms"]:,.1f} ms total'],
                           color, final_offset, final_align))
        ax.annotate(f'Fine +{f["latency_ms"] - c["latency_ms"]:,.1f} ms', fine_xy,
                    xytext=fine_offset, textcoords='offset points', ha='center', va='center',
                    color=color, fontsize=7.0,
                    bbox=dict(boxstyle='round,pad=.16', fc='white', ec='none', alpha=.94))
    ours = next(p for p in points if p['method'] == 'AnyGSLoc')
    ax.scatter(ours['R5_pct'], ours['latency_ms'], s=102, marker='*', c=COL['AnyGSLoc'],
               edgecolor='white', lw=.65, zorder=4)
    boxes.append(label(ax, (ours['R5_pct'], ours['latency_ms']),
                       ['AnyGSLoc', f'{ours["latency_ms"]:.1f} ms total'],
                       COL['AnyGSLoc'], (-11, 4), (1, .5)))
    offsets = {'STDLoc': ((9, 0), (0, .5)), 'ULFLoc': ((-10, 5), (1, .5)),
               'AnyGSLoc': ((-10, -3), (1, .5))}
    for p in final:
        ax = axs[1]
        ours_flag = p['method'] == 'AnyGSLoc'
        ax.scatter(p['R5_pct'], p['storage_MB'], s=102 if ours_flag else 48,
                   marker='*' if ours_flag else 'o', c=COL[p['method']], edgecolor='white', lw=.65, zorder=4)
        offset, align = offsets[p['method']]
        boxes.append(label(ax, (p['R5_pct'], p['storage_MB']),
                           [p['method'].replace('ULFLoc', 'ULF-Loc'), f'{p["storage_MB"]:,.1f} MB'],
                           COL[p['method']], offset, align))
    handles = [Line2D([], [], marker='o', ls='', mfc='white', mec='#697785', mew=1.1, ms=6, label='Coarse'),
               Line2D([], [], marker='o', ls='', mfc='#697785', mec='white', ms=6.5, label='Final (coarse + fine)'),
               Line2D([], [], marker='*', ls='', mfc=COL['AnyGSLoc'], mec='white', ms=9, label='Single stage')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.535, .025), ncol=3,
               frameon=False, fontsize=7.3, columnspacing=1.5, handletextpad=.4)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for box in boxes:
        b = box.get_window_extent(renderer)
        if b.x0 < 0 or b.y0 < 0 or b.x1 > fig.bbox.width or b.y1 > fig.bbox.height:
            outside.append(str(b))
    assert not outside, outside
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / f'E5_deployment.{ext}', dpi=400, facecolor='white')
    plt.close(fig)
    caption = (
        'Accuracy and deployment cost under native configurations on five Cambridge Landmarks scenes. '
        '(a) Mean latency to the coarse and final outputs; arrows label the additional fine-stage time. '
        'Final latency includes the coarse prefix and shared query precomputation. '
        '(b) Native scene-specific inference storage, summed over five scenes, with a tighter recall range. '
        'Recall and latency are scene-equal means; cost axes are logarithmic. '
        'Measurements use RTX 3090 GPUs with native feature grids, solver budgets and timing boundaries. '
        '† ULF-Loc latency retains its reference-camera inlier diagnostic. '
        'Required scene models, descriptors, scene-specific detectors and indices are included in storage; '
        'shared weights, query inputs and intermediate training artifacts are excluded. '
        'These are measured operating points; no equal-accuracy parameter sweep is implied. '
        'The paper timing and visualization protocol specifies the detailed boundaries.')
    (OUT / 'E5_deployment.caption.txt').write_text(caption + '\n')
    provenance = {'figure': 'E5_deployment', 'revision': 5, 'operating_points': points,
                  'stage_cost_breakdown': stages, 'original_point_source': str(PREV / 'E5_deployment.json'),
                  'original_source_sha256': sha(PREV / 'E5_deployment.json'),
                  'previous_revision': {'path': str(ROOT / 'paper/experiments_cvpr_r4_20261003/E5_deployment.provenance.json'), 'sha256': sha(ROOT / 'paper/experiments_cvpr_r4_20261003/E5_deployment.provenance.json')},
                  'r5_operating_points_unchanged': points == json.loads((ROOT / 'paper/experiments_cvpr_r4_20261003/E5_deployment.provenance.json').read_text())['operating_points'],
                  'design': {'x': 'R5 recall; panel-specific explicit ranges',
                             'y': ['mean native pipeline latency to the displayed stage', 'five-scene native storage'],
                             'cost_scale': 'logarithmic',
                             'stage_encoding': 'two baseline operating points: coarse prefix and cumulative final; fine is a labeled increment',
                             'equivalent_FPS': 'Removed from the figure to reduce redundant annotations. Conversion remains in source CSV: 1000 / scene-macro mean latency; not measured throughput.',
                             'changes_from_R4': 'Frozen operating points, scales and stage increments; native-configuration/hardware subtitle, final-output storage scope and dagger on both ULF-Loc latency points.'},
                  'caption': caption, 'limitations': old['limitations'],
                  'source_script': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
                  'outputs': {ext: {'sha256': sha(OUT / f'E5_deployment.{ext}')} for ext in ['pdf', 'svg', 'png']}}
    (OUT / 'E5_deployment.provenance.json').write_text(json.dumps(provenance, indent=2))
    with (OUT / 'E5_cost_breakdown.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(stages[0])); w.writeheader(); w.writerows(stages)
    for name in ['E5_storage_audit.json', 'E5_storage_audit.txt', 'E5_storage_by_scene.csv']:
        shutil.copy2(PREV / name, OUT / name)
    print(json.dumps(stages, indent=2))


if __name__ == '__main__':
    main()
