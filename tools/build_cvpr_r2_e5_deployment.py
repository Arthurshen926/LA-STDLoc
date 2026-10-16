"""Revision 2: final operating points, accuracy on x, deployment cost on y.

Existing measurements and the completed native-file audit are reused verbatim.
No timing, inference, training, or storage repacking is performed.
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
OUT = ROOT / 'paper/experiments_cvpr_r2_20261003'
OUT.mkdir(parents=True, exist_ok=True)
COL = {'STDLoc': '#497fa6', 'ULFLoc': '#896897', 'AnyGSLoc': '#cb7327'}
INK, GRAY = '#263341', '#6c7782'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def label(ax, point, key, offset, align):
    method = point['method']
    val = point[key]
    unit = 'ms total' if key == 'latency_ms' else 'MB'
    name = method.replace('ULFLoc', 'ULF-Loc')
    text = VPacker(children=[
        TextArea(name, textprops=dict(size=8, color=COL[method], weight='semibold')),
        TextArea(f'{val:,.1f} {unit}', textprops=dict(size=7.4, color=INK))],
        pad=0, sep=2.5, align='left' if align == 0 else 'right')
    artist=AnnotationBbox(text,(point['R5_pct'],val),xybox=offset,xycoords='data',
        boxcoords='offset points',box_alignment=(align,.5),frameon=False,pad=0)
    ax.add_artist(artist)
    return artist

def main():
    old=json.loads((PREV/'E5_deployment.json').read_text())
    final=[p for p in old['points'] if p['stage'] in ['final','single']]
    stages=[]
    for p in final:
        coarse=next((q for q in old['points'] if q['method']==p['method'] and q['stage']=='coarse'),None)
        stages.append(dict(method=p['method'],coarse_prefix_ms=coarse['latency_ms'] if coarse else None,
            refinement_increment_ms=p['latency_ms']-coarse['latency_ms'] if coarse else None,
            total_latency_ms=p['latency_ms'],R5_pct=p['R5_pct'],TE_cm=p['TE_cm'],RE_deg=p['RE_deg'],storage_MB=p['storage_MB']))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'text.color':INK,
        'axes.labelcolor':INK,'axes.edgecolor':'#9aa4ae','xtick.color':GRAY,'ytick.color':GRAY,
        'axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig,axs=plt.subplots(1,2,figsize=(7.1,3.42),sharex=True)
    fig.subplots_adjust(left=.097,right=.978,bottom=.24,top=.88,wspace=.31)
    specifications=[('latency_ms','(a) Native pipeline latency','Total latency (ms, log) ↓',
        (44,2400),[50,100,250,500,1000,2000]),
        ('storage_MB','(b) Five-scene storage','Five-scene storage (MB, log) ↓',
        (105,3800),[125,250,500,1000,2000])]
    offsets={
        'latency_ms':{'STDLoc':((11,6),0),'ULFLoc':((-10,6),1),'AnyGSLoc':((-10,14),1)},
        'storage_MB':{'STDLoc':((10,0),0),'ULFLoc':((-11,6),1),'AnyGSLoc':((-11,-4),1)}}
    boxes=[]
    for ax,(key,title,ylab,ylim,ticks) in zip(axs,specifications):
        ax.set(xlim=(36,44.3),ylim=ylim,yscale='log',xlabel='Recall @ 5 cm / 5° (%) ↑',ylabel=ylab)
        ax.set_title(title,fontsize=9,weight='semibold',loc='left',pad=12)
        ax.xaxis.set_major_locator(FixedLocator([36,38,40,42,44]))
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:,.0f}'))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(labelsize=7.1,length=2.8,pad=4,width=.65)
        ax.grid(axis='y',color='#dce2e7',lw=.55,alpha=.65)
        ax.set_axisbelow(True)
        for p in final:
            ours=p['method']=='AnyGSLoc';color=COL[p['method']]
            # Final operating points only; no coarse accuracy is paired with final cost.
            ax.scatter(p['R5_pct'],p[key],s=105 if ours else 49,
                marker='*' if ours else 'o',c=color,ec='white',lw=.65,zorder=4)
            off,align=offsets[key][p['method']]
            boxes.append(label(ax,p,key,off,align))
    legend=[Line2D([],[],marker='o',ls='',mfc='#697785',mec='white',ms=6.5,
                   label='Baselines: coarse + refinement'),
            Line2D([],[],marker='*',ls='',mfc=COL['AnyGSLoc'],mec='white',ms=10,
                   label='AnyGSLoc: single stage')]
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.535,.013),ncol=2,
        frameon=False,fontsize=7.4,columnspacing=2.4,handletextpad=.45)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    outside=[]
    for box in boxes:
        b=box.get_window_extent(renderer)
        if b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height:outside.append(str(b))
    assert not outside,outside
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/f'E5_deployment.{ext}',dpi=400,facecolor='white')
    plt.close(fig)
    caption=(
        'Final localization accuracy versus deployment cost on Cambridge Landmarks. '
        'Both panels share the horizontal recall axis; lower and farther right is preferable. '
        '(a) The baseline points use total warmed query time, including both the native coarse prefix and dense refinement; AnyGSLoc uses one sparse localization stage. '
        'Only final operating points are shown. Baseline coarse-prefix and refinement-increment times are retained in the accompanying data. '
        '(b) Native scene-specific inference representation storage, summed over the same five scenes in decimal MB. '
        'Recall and latency are equal-weight scene means. Cost axes are logarithmic. '
        'Storage includes required Gaussian, landmark, descriptor, and scene-detector assets, and excludes shared pretrained weights, intermediate checkpoints, training logs, duplicate benchmark files, and query RGB/calibration inputs. '
        'Native model formats and overhead are preserved; this is a dependency-based asset audit, not an optimal-compression lower bound. '
        'Native resolutions and configurations differ. Timings exclude outer I/O and pose-error evaluation but retain ULF-Loc’s native reference-camera inlier diagnostic. '
        'The two-stage coarse prefix includes shared query precomputation. No new timing or localization experiments were run.')
    (OUT/'E5_deployment.caption.txt').write_text(caption+'\n')
    provenance={'figure':'E5_deployment','revision':2,'final_operating_points':final,
        'stage_cost_breakdown':stages,'original_point_source':str(PREV/'E5_deployment.json'),
        'original_source_sha256':sha(PREV/'E5_deployment.json'),
        'design':{'x':'R5 recall, same axis range across panels','y':['total query latency','native scene-specific storage'],
            'cost_scale':'logarithmic','markers':'final poses only; baseline circles include both stages; star is single stage',
            'removed_from_main_plot':'coarse operating points and repeated TE/RE labels; retained in data'},
        'caption':caption,'limitations':old['limitations'],
        'outputs':{ext:{'sha256':sha(OUT/f'E5_deployment.{ext}')} for ext in ['pdf','svg','png']}}
    (OUT/'E5_deployment.provenance.json').write_text(json.dumps(provenance,indent=2))
    with (OUT/'E5_cost_breakdown.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(stages[0]));w.writeheader();w.writerows(stages)
    for name in ['E5_storage_audit.json','E5_storage_audit.txt','E5_storage_by_scene.csv']:
        shutil.copy2(PREV/name,OUT/name)
    print(json.dumps(stages,indent=2))

if __name__=='__main__':main()
