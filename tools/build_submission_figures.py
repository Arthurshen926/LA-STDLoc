"""Assemble an evidence-bound, editable candidate-final figure suite.

Run after materialize_mechanism_samples.py render, draw_mechanism_figures.py,
and build_layout_candidates.py. No experiments or scientific images are invented.
"""
from pathlib import Path
import csv,json,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageOps,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'paper/figures_submission_candidate';OUT.mkdir(exist_ok=True)
V5=ROOT/'paper/figures_v5';DATA=ROOT/'docs/paper/evidence_20260917'
TEAL='#187b70';BLUE='#336f9c';GOLD='#b87b38';INK='#243349';GRAY='#8794a2'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
    'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'svg.fonttype':'none','text.color':INK,'axes.labelcolor':INK})
MAN={}
def rows(name):
    with (DATA/name).open() as f:return list(csv.DictReader(f))
def finish(fig,name,caption,limits=()):
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',dpi=300,bbox_inches='tight',pad_inches=.07)
    plt.close(fig);MAN[name]={'caption':caption,'limitations':list(limits)}
def inherit(src,name,entry):
    for ext in ['pdf','svg','png']:shutil.copyfile(src.with_suffix('.'+ext),OUT/(name+'.'+ext))
    MAN[name]=entry
def clean(ax):ax.grid(axis='x',alpha=.14);ax.set_axisbelow(True)

def main():
    old=json.loads((V5/'figure_manifest.json').read_text())
    inherit(V5/'D03_overview_landmark_centric','F02_overview',{
        'caption':'AnyGSLoc separates rendering primitives from localization landmarks. Offline, native Gaussian render evidence screens detected observations; cross-view tracks support fixed-camera multi-ray geometry and descriptor fusion; compact selection retains complementary localization constraints. Online, only the sparse map is used for feature matching and one robust pose-solver call.',
        'limitations':['Real StMarysChurch evidence and track, OldHospital maps; panels do not represent one cross-scene anchor trajectory.','Descriptor glyphs are schematic. Gaussian ellipses are sampled projected covariances.','Validity overlay is visualization only, not semantic masking or preprocessing of feature-extractor input.']})
    for name in ['F03_render_validity','F04_landmark_geometry','F05_compact_selection','S01_five_scene_sample_board']:
        inherit(V5/name,name,old['figures'][name])
    hero=old['validity_hero'];mm=np.load(V5/'assets'/(hero['stem']+'_mask.npz'))
    fig,axs=plt.subplots(2,3,figsize=(7.1,4.1))
    for ax,key,title in zip(axs.flat,['support','depth_continuity','structure','border_ok','distortion_ok','valid'],
            ['Alpha + finite depth','Depth continuity','RGB structure support','Image border support','Distortion support','Intersection: V2 validity']):
        ax.imshow(mm[key],cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([]);ax.set_title(title,fontsize=8)
        for spine in ax.spines.values():spine.set_visible(True);spine.set_color('#d6dde3');spine.set_linewidth(.6)
    fig.text(.5,.025,'White: passes this predicate  ·  Black: fails  ·  All predicates evaluated before cropping',ha='center',fontsize=7)
    fig.subplots_adjust(wspace=.10,hspace=.18,bottom=.09,top=.93)
    finish(fig,'S03_validity_components','Individual V2 predicates on the same StMarysChurch crop; their intersection produces the displayed validity mask. Masks are evaluated on native full-frame buffers, then cropped. Sampling the intersection at all detected keypoints exactly reproduces archived filtering decisions.',[
        'This interior crop need not illustrate every failure mode: image-border and distortion predicates can pass the whole region.','Predicates overlap; panel areas cannot be summed into a rejection count.','No semantic segmentation, no nonblack-pixel test and no mask interpolation.'])
    macro=rows('factorial_macro.csv')
    full,compact=[next(r for r in macro if r['selection']==s and all(r[k]=='1' for k in ['V2','fusion','completion'])) for s in ['full','compact']]
    accuracy=rows('baseline_macro.csv');timing=rows('baseline_timing_macro.csv')
    names=[];ts=[];tes=[];colors=[]
    for method,col in [('STDLoc',GOLD),('ULFLoc',BLUE)]:
        t=next(r for r in timing if r['method']==method and r['mapping_mask']=='off')
        for stage,key in [('coarse','coarse_prefix_ms'),('fine','coarse_plus_fine_ms')]:
            names.append(method.replace('ULFLoc','ULF-Loc')+' '+stage);ts.append(float(t[key]));colors.append(col)
            tes.append(float(next(r for r in accuracy if r['method']==method and r['mapping_mask']=='off' and r['stage']==stage)['TE_cm']))
    names.append('AnyGSLoc');ts.append(float(compact['core_mean_ms']));tes.append(float(compact['TE_cm']));colors.append(TEAL)
    # A compact route illustration uses glyphs instead of text-filled boxes.
    fig=plt.figure(figsize=(7.1,3.2));gs=fig.add_gridspec(1,2,width_ratios=[1.2,1])
    ax=fig.add_subplot(gs[0,0]);ax.set(xlim=(0,10),ylim=(0,6));ax.axis('off')
    for y,label,col in [(4.5,'Feature-Gaussian routes',BLUE),(1.8,'AnyGSLoc',TEAL)]:
        ax.text(0,y+1,label,color=col,weight='bold',fontsize=10)
        # Representation glyph: overlapping ellipses vs discrete landmark nodes.
        from matplotlib.patches import Ellipse
        for j,(x0,y0) in enumerate([(0.4,-.2),(.85,.3),(1.3,-.1),(.9,-.55)]):
            if col==BLUE:ax.add_patch(Ellipse((x0,y+y0),.8,.36,angle=j*32,fc=col,alpha=.3))
            else:ax.plot(x0,y+y0,'o',ms=4,color=col)
        ax.annotate('',(4.0,y),(2.0,y),arrowprops={'arrowstyle':'->','color':GRAY})
        ax.text(4.1,y,'Sparse pose',va='center',fontsize=8,color=col)
        if col==BLUE:
            ax.annotate('',(8.0,y),(7.3,y),arrowprops={'arrowstyle':'->','color':GRAY})
            ax.text(8.25,y,'Dense\nrefinement',va='center',fontsize=8)
        else:ax.text(8,y,'R, t',va='center',fontsize=14,color=col,weight='bold')
        ax.text(0,y-1.05,'Feature-bearing primitives' if col==BLUE else 'Compact XYZ + descriptors',fontsize=8,color=GRAY)
    ax=fig.add_subplot(gs[0,1]);
    for i,(label,t,te,col) in enumerate(zip(names,ts,tes,colors)):
        ax.scatter(t,te,s=35,c=col,marker='o' if i%2==0 else 's')
        ax.annotate(label,(t,te),xytext=(5,5 if i!=4 else -13),textcoords='offset points',fontsize=7)
    for i in [0,2]:ax.plot(ts[i:i+2],tes[i:i+2],color=colors[i],lw=.8,alpha=.5)
    ax.set(xscale='log',xlim=(45,2500),ylim=(7.9,13),xlabel='Mean algorithm latency (ms; log)',ylabel='Mean scene-median TE (cm)');clean(ax)
    fig.subplots_adjust(wspace=.43,bottom=.19,top=.95)
    finish(fig,'F01_problem_and_position','Representation and deployment choices in Gaussian localization. Coarse/fine measurements are paired within each reproduced baseline; fine latency includes coarse processing. AnyGSLoc uses sparse localization without an additional dense refinement stage.',[
        'Conceptual route abstraction, not an assertion of identical STDLoc and ULF-Loc training.','Five-scene macro results, native method configurations; not an equal-resolution timing control.','Connecting segments link algorithm stages, not a continuous accuracy-speed trade-off.'])
    fig,axs=plt.subplots(1,3,figsize=(7.1,3.1),sharey=True)
    conf=[(v,f,c) for v,f in [(0,0),(1,0),(0,1),(1,1)] for c in [0,1]]
    rr=[next(r for r in macro if r['selection']=='compact' and tuple(int(r[k]) for k in ['V2','fusion','completion'])==cfg) for cfg in conf]
    labels=[f'{v} / {f} / {c}' for v,f,c in conf]
    for ax,key,title,scale in zip(axs,['TE_cm','R5_pct','total_anchors'],['Median TE (cm) ↓','Recall @ 5 cm / 5° (%) ↑','Landmarks (×1,000)'],[1,1,1000]):
        vals=np.array([float(r[key])/scale for r in rr]);ax.scatter(vals,np.arange(8),c=[TEAL if cfg==(1,1,1) else BLUE for cfg in conf],s=24)
        ax.set_title(title,fontsize=8);ax.set_yticks(range(8),labels);clean(ax)
        span=max(np.ptp(vals),.1);ax.set_xlim(min(vals)-span*.25,max(vals)+span*.55)
        for y,val in enumerate(vals):ax.annotate(f'{val:.2f}' if scale==1 else f'{val:.1f}',(val,y),xytext=(5,0),textcoords='offset points',va='center',fontsize=6.5)
    axs[0].invert_yaxis();axs[0].set_ylabel('V2 / fusion / completion\n0: off    1: on')
    fig.subplots_adjust(wspace=.17,left=.15,bottom=.12,top=.88)
    finish(fig,'F06_observation_ablation','All eight observation configurations with compact selection, averaged equally across five Cambridge scenes. Full configuration is highlighted; none of the measured configurations is omitted.',[
        'Each variant reselects its map. This is a pipeline ablation, not a fixed-map descriptor-only control.','Horizontal ranges are local to each metric; use annotated values, not apparent displacement, to judge effect size.'])
    fig,axs=plt.subplots(1,3,figsize=(7.1,3.0),gridspec_kw={'width_ratios':[1.25,1,1]})
    for ax,vals,title in [(axs[0],ts,'Algorithm latency (ms) ↓'),(axs[1],tes,'Median TE (cm) ↓')]:
        ax.barh(range(5),vals,color=colors,height=.55);ax.invert_yaxis();ax.set_yticks(range(5),names if ax is axs[0] else ['']*5);ax.set_title(title,fontsize=8);clean(ax)
        ax.set_xlim(0,max(vals)*1.34)
        for y,val in enumerate(vals):ax.text(val+max(vals)*.025,y,f'{val:.1f}',va='center',fontsize=7)
    vals=[float(r['total_map_MB']) for r in [full,compact]]
    axs[2].barh([0,1],vals,color=[GRAY,TEAL],height=.5);axs[2].set_yticks([0,1],['Full','Compact']);axs[2].invert_yaxis();axs[2].set_title('Our map storage (MB) ↓',fontsize=8);axs[2].set_xlim(0,max(vals)*1.38);clean(axs[2])
    for y,val in enumerate(vals):axs[2].text(val+10,y,f'{val:.1f}',va='center',fontsize=7)
    fig.subplots_adjust(left=.19,right=.98,wspace=.43,bottom=.14,top=.87)
    finish(fig,'F07_deployment','Five-scene accuracy and mean algorithm latency of reproduced methods, including both coarse and cumulative fine stages. Storage compares AnyGSLoc full and compact deployment maps, summed over five scenes (decimal MB).',[
        'No query semantic masks; baseline mapping semantic masks off. Upstream MAtCHA prior preparation is a separate cost.','Algorithm timing excludes disk I/O and ground-truth evaluation; no tail-latency claim.','Baseline minimal deployment assets and offline construction costs are not audited here. Own-map compression is not a baseline storage comparison.','Native input/configuration differences remain; selector compression itself has no measured speedup.'])
    priors=rows('prior_macro.csv');pkeys=['vanilla_2dgs','vanilla_3dgs','i3dgs_raw','i3dgs_1k','long_lrm_raw','long_lrm_1k'];pn=['MAtCHA','Vanilla 2DGS','Vanilla 3DGS','i3DGS raw','i3DGS +1k','Long-LRM raw','Long-LRM +1k'];pr=[compact]+[next(r for r in priors if r['prior']==p) for p in pkeys]
    fig,axs=plt.subplots(1,2,figsize=(7.1,3.2),sharey=True)
    for ax,key,title in zip(axs,['TE_cm','R5_pct'],['Median TE (cm; log) ↓','Recall @ 5 cm / 5° (%) ↑']):
        vals=[float(r[key]) for r in pr];ax.scatter(vals,range(7),s=30,c=[TEAL]+[BLUE]*2+[GOLD]*4);ax.set_yticks(range(7),pn);ax.set_title(title,fontsize=9);clean(ax)
        if key=='TE_cm':ax.set_xscale('log');ax.set_xlim(5,2500)
        else:ax.set_xlim(-2,58)
        for y,val in enumerate(vals):ax.annotate(f'{val:.2f}',(val,y),xytext=(5,0),textcoords='offset points',va='center',fontsize=7)
    axs[0].invert_yaxis();fig.subplots_adjust(left=.20,wspace=.22,top=.88,bottom=.13)
    finish(fig,'F08_prior_sensitivity','Localization across all evaluated Gaussian-prior families, including degraded feed-forward priors. Values are means of scene metrics, not rendering-quality scores.',[
        'Upstream inputs and training conditions differ; this is not evidence of invariance to prior quality.','Long-LRM failures retain upstream-quality/integration uncertainty.'])
    old4=json.loads((ROOT/'paper/figures_v4/manifest.json').read_text())
    entry=old4['figures']['F06_query_cases']
    fig,axs=plt.subplots(2,3,figsize=(7.1,4.1),gridspec_kw={'height_ratios':[1.2,1]})
    for i,sample in enumerate(entry['selection']):
        picture=Path('/mnt/pool/sqy/Cambridge_stdloc/ShopFacade/processed')/sample['image']
        axs[0,i].imshow(Image.open(picture));axs[0,i].axis('off');axs[0,i].set_title(sample['rule']+'\n'+sample['image'],fontsize=7)
        ax=axs[1,i];ax.axis('off');values=[]
        for label,record in sample['methods'].items():
            te=record.get('TE_cm',record.get('translation_error_cm'));re=record.get('RE_deg',record.get('rotation_error_deg'))
            values.append([label.replace('ULFLoc','ULF-Loc'),f'{te:.2f}',f'{re:.3f}'])
        table=ax.table(cellText=values,colLabels=['Method','TE (cm)','RE (°)'],loc='center',colWidths=[.48,.26,.26],cellLoc='left')
        table.auto_set_font_size(False);table.set_fontsize(6.7);table.scale(1,1.3)
        for (r,c),cell in table.get_celld().items():
            cell.set_edgecolor('white');cell.set_facecolor('#edf4f2' if r==5 else ('#edf0f4' if r==0 else 'white'))
            if r==0:cell.set_text_props(weight='bold')
    fig.subplots_adjust(wspace=.20,hspace=.04,bottom=.08,top=.94)
    finish(fig,'S02_query_cases',entry['caption']+' C: coarse; F: fine.',entry['limitations'])
    MAN['S02_query_cases']['selection']=entry['selection']
    source_files=list(DATA.glob('*.csv'))+[V5/'selection.json',V5/'figure_manifest.json',ROOT/'paper/figures_v4/manifest.json']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    (OUT/'manifest.json').write_text(json.dumps({'status':'candidate_final_requires_author_review','figures':MAN,'source_sha256':hashes},indent=2))
    captions=[];includes=[]
    for name,entry in sorted(MAN.items()):
        captions+=['## '+name,'',entry['caption'],'']+['- '+x for x in entry.get('limitations',[])]+['']
        includes.append('\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{figures_submission_candidate/'+name+'.pdf}\n\\caption{'+entry['caption'].replace('%','\\%').replace('&','\\&')+'}\n\\label{fig:'+name.lower()+'}\n\\end{figure*}\n')
    (OUT/'CAPTIONS.md').write_text('\n'.join(captions));(OUT/'figures.tex').write_text('\n'.join(includes))
    cards=[]
    for name in sorted(MAN):
        im=Image.open(OUT/(name+'.png')).convert('RGB');im.thumbnail((850,510))
        card=Image.new('RGB',(890,560),'#f1f3f5');card.paste(im,((890-im.width)//2,35+(510-im.height)//2));ImageDraw.Draw(card).text((20,12),name,fill=INK);cards.append(card)
    board=Image.new('RGB',(1780,560*((len(cards)+1)//2)),'white')
    for i,card in enumerate(cards):board.paste(card,((i%2)*890,(i//2)*560))
    board.save(OUT/'contact_sheet.jpg',quality=94)
    print('Candidate suite:',OUT,flush=True)

if __name__=='__main__':main()
