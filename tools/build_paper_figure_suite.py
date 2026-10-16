"""Reproducible v4 figure drafts; CPU only, measured data or labelled schematics.

Run with the g4splat environment from the repository root. Does not run evaluation
or alter maps. PNG previews and editable SVG/PDF outputs accompany a provenance
manifest. Draft figures are deliberately not represented as submission-ready.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anygsloc.data.datasets import ColmapDataset

OUT = ROOT / 'paper/figures_v4'
DATA = ROOT / 'docs/paper/evidence_20260917'
POOL = Path('/mnt/pool/sqy')
FACTOR = POOL / 'anygsloc_observation_factorial_20260914'
BASE = POOL / 'anygsloc_baseline_recovery_v4_20260915'
PRIMARY = POOL / 'anygsloc_protocol_v2_primary_20260914/runs/Cambridge'
SCENES = ['GreatCourt', 'KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch']
SHORT = ['GC', 'KC', 'OH', 'SF', 'SM']
BLUE, ORANGE, GREEN, INK = '#377EB8', '#D78532', '#25836C', '#233047'
SOURCES, FIGURES = {}, {}
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
    'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42,
    'svg.fonttype':'none', 'axes.titleweight':'bold', 'savefig.facecolor':'white'})


def source(p):
    p = Path(p)
    if str(p) not in SOURCES:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
        SOURCES[str(p)] = {'sha256':h.hexdigest(), 'bytes':p.stat().st_size}
    return p


def csvrows(name):
    with source(DATA / name).open() as f: return list(csv.DictReader(f))


def js(p): return json.loads(source(p).read_text())


def save(fig, name, caption, limitations=(), selection=None):
    fig.text(.015, .012, 'AnyGSLoc | evidence-backed draft | 2026-09-25', fontsize=7, color='#687587')
    for suffix in ('pdf','svg','png'):
        fig.savefig(OUT / f'{name}.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    FIGURES[name] = dict(caption=caption, limitations=list(limitations), selection=selection,
                        sources=list(SOURCES))


def card(ax, xy, w, h, title, subtitle='', color=BLUE):
    x,y = xy
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.014',
                 edgecolor=color,facecolor=color+'12',linewidth=1.3))
    ax.text(x+w/2,y+h*.67,title,ha='center',va='center',weight='bold',color=INK,fontsize=11)
    if subtitle: ax.text(x+w/2,y+h*.26,subtitle,ha='center',va='center',fontsize=9,color=INK)


def arrow(ax, a, b, color=INK):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=color,lw=1.5))


def schematic_rays(ax):
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    point=np.array([.55,.7])
    for x in [.15,.45,.88]:
        ax.plot([x,point[0]],[.15,point[1]],c=BLUE,lw=1.3)
        ax.scatter([x],[.15],marker='^',s=75,c=BLUE)
    ax.add_patch(Ellipse((.67,.67),.3,.14,angle=25,facecolor=ORANGE+'30',edgecolor=ORANGE))
    ax.scatter(*point,c=GREEN,s=55,zorder=4)
    ax.scatter(.67,.67,c=ORANGE,marker='x',s=40)
    ax.text(.5,.94,'Track-constrained position',ha='center',fontsize=10)
    ax.text(.5,.02,'Schematic; not measured geometry',ha='center',fontsize=8,color='#687587')


def loadmap(selection):
    p=FACTOR / f'maps/ShopFacade_v1_f1_c1/{selection}/deployment_map.pt'
    return torch.load(source(p),map_location='cpu',weights_only=True)


def project(xyz, camera):
    pose=camera.pose_w2c
    q=xyz @ pose[:3,:3].T + pose[:3,3]
    fx=camera.width/(2*np.tan(camera.fov_x/2));fy=camera.height/(2*np.tan(camera.fov_y/2))
    z=np.maximum(q[:,2],1e-8)
    uv=np.column_stack((fx*q[:,0]/z+camera.width/2,fy*q[:,1]/z+camera.height/2))
    valid=(q[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<camera.width)&(uv[:,1]>=0)&(uv[:,1]<camera.height)
    return uv,valid


def photo(ax, camera):
    ax.imshow(Image.open(source(camera.image_path)).convert('RGB'))
    ax.axis('off')


def mapview(ax, state, camera, color=GREEN):
    photo(ax,camera)
    uv,good=project(state['anchor_xyz'].numpy(),camera)
    ax.scatter(uv[good,0],uv[good,1],s=.5,c=color,alpha=.45,rasterized=True)
    return int(good.sum())


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    factor=csvrows('factorial_80.csv'); macro=csvrows('factorial_macro.csv')
    geometry=csvrows('geometry_15.csv'); primary=csvrows('primary_24.csv')
    def fm(selection):
        return next(r for r in macro if all(r[k]=='1' for k in ['V2','fusion','completion']) and r['selection']==selection)
    fullrow,comprow=fm('full'),fm('compact')
    dataset=ColmapDataset(str(POOL/'Cambridge_stdloc/ShopFacade'),load_masks=False)
    for p in (dataset.root/'sparse/0').iterdir():
        if p.name in ('cameras.bin','images.bin','cameras.txt','images.txt'):source(p)
    source(dataset.root/'dataset_test.txt')
    results=js(PRIMARY/'ShopFacade/results.json')
    ordered=sorted(results,key=lambda r:r['translation_error_cm'] if r['success'] else float('inf'))
    selected=[ordered[round(q*(len(ordered)-1))] for q in [.5,.9,1.0]]
    camera=dataset.camera(selected[0]['image_name'])
    full,compact=loadmap('full'),loadmap('compact')

    fig=plt.figure(figsize=(13,6.5));gs=fig.add_gridspec(2,3,width_ratios=[1.0,1.4,1.1])
    ax=fig.add_subplot(gs[:,0]);photo(ax,camera);ax.set_title('A real localization query')
    ax=fig.add_subplot(gs[0,1:]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    card(ax,(.02,.3),.27,.4,'Gaussian landmarks','Learned / fused features',ORANGE)
    card(ax,(.37,.3),.22,.4,'Initial pose','Sparse matching',ORANGE)
    card(ax,(.68,.3),.29,.4,'Refined pose','Rendering + dense matching',ORANGE)
    arrow(ax,(.29,.5),(.37,.5));arrow(ax,(.59,.5),(.68,.5))
    ax.set_title('Related Gaussian localization routes',loc='left')
    ax=fig.add_subplot(gs[1,1:]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    card(ax,(.02,.3),.27,.4,'Reconstruct landmarks','Geometry + observations',GREEN)
    card(ax,(.37,.3),.22,.4,'Compact map','Complementary constraints',GREEN)
    card(ax,(.68,.3),.29,.4,'Single-stage pose','Sparse matching + PnP',GREEN)
    arrow(ax,(.29,.5),(.37,.5));arrow(ax,(.59,.5),(.68,.5))
    ax.set_title('AnyGSLoc: keep the evidence, simplify deployment',loc='left')
    fig.subplots_adjust(wspace=.25,hspace=.3,bottom=.12)
    save(fig,'F01_routes','Conceptual route comparison with an actual median-error ShopFacade query.',
         ['STDLoc learns feature Gaussians; ULF-Loc directly fuses landmark features. The top row abstracts inference, not identical training.',
          'This figure does not prove that all dense refinement is caused by a specific geometry defect.'])

    fig=plt.figure(figsize=(13,7));gs=fig.add_gridspec(2,4,height_ratios=[1.4,1])
    ax=fig.add_subplot(gs[0,0]);schematic_rays(ax);ax.set_title('1  Reconstruct geometry')
    ax=fig.add_subplot(gs[0,1]);ax.axis('off')
    descriptors=full['anchor_features'][:40].numpy()
    ax.imshow(descriptors,aspect='auto',cmap='coolwarm',vmin=-.2,vmax=.2)
    ax.set_title('2  One descriptor / point')
    ax=fig.add_subplot(gs[0,2]);mapview(ax,full,camera,BLUE);ax.set_title(f'3  Candidate map\n{len(full["anchor_xyz"]):,} points')
    ax=fig.add_subplot(gs[0,3]);mapview(ax,compact,camera);ax.set_title(f'4  Compact map\n{len(compact["anchor_xyz"]):,} points')
    ax=fig.add_subplot(gs[1,0]);photo(ax,camera);ax.set_title('Real query')
    ax=fig.add_subplot(gs[1,1:]);ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    for x,title,sub in [(.02,'SuperPoint','Extract'),(.36,'Exact sparse matching','Filter correspondences'),(.7,'Robust PnP','Estimate pose')]:
        card(ax,(x,.3),.27,.4,title,sub,GREEN)
    arrow(ax,(.29,.5),(.36,.5));arrow(ax,(.63,.5),(.7,.5))
    ax.text(.5,.08,'ONLINE: no scene renderer and no additional dense refinement',ha='center',color=GREEN)
    fig.subplots_adjust(wspace=.24,hspace=.4,bottom=.1)
    save(fig,'F02_overview','Offline landmark construction and online sparse localization. Actual map descriptors and projected maps.',
         ['Ray inset is a schematic; rendered observations and actual Track patches remain to be added.',
          'Projected map points are not visibility-tested matches; first 40 stored descriptor rows shown without selection by appearance.'])

    fig=plt.figure(figsize=(13,7));gs=fig.add_gridspec(2,3)
    reference=None
    for j,g in enumerate(['center','depth','ray']):
        state=torch.load(source(POOL/f'anygsloc_center_control_20260915/maps/ShopFacade/{g}/deployment_map.pt'),map_location='cpu',weights_only=True)
        if reference is not None:
            assert torch.equal(state['anchor_ids'],reference['anchor_ids'])
            assert torch.equal(state['anchor_features'],reference['anchor_features'])
        reference=state
        ax=fig.add_subplot(gs[0,j]);mapview(ax,state,camera,[ORANGE,BLUE,GREEN][j]);ax.set_title(f'{g.capitalize()} coordinates | common IDs')
    for j,metric in enumerate(['TE_cm','RE_deg','R5_pct']):
        ax=fig.add_subplot(gs[1,j])
        for g,col in zip(['center','depth','ray'],[ORANGE,BLUE,GREEN]):
            vals=[float(next(r for r in geometry if r['scene']==s and r['geometry']==g)[metric]) for s in SCENES]
            ax.plot(SHORT,vals,'o-',c=col,label=g,lw=1.4,ms=4)
        ax.set_ylabel({'TE_cm':'Median translation error (cm)','RE_deg':'Median rotation error (deg)','R5_pct':'Recall @ 5 cm / 5 deg (%)'}[metric]);ax.grid(alpha=.15)
    ax.legend(frameon=False,ncol=3,fontsize=8)
    fig.subplots_adjust(wspace=.35,hspace=.3,bottom=.13)
    save(fig,'F03_geometry','Coordinate-source control on common anchor IDs; five-scene results and one actual query projection.',
         ['Projection overlays are not correspondence residuals or independent geometric ground truth.',
          'Ray coordinates retain original Track support; depth/center use common available observations. Common coverage is 65–85%.'])

    fig,axs=plt.subplots(1,3,figsize=(13,4.8))
    configurations=[('0','0'),('1','0'),('0','1'),('1','1')]
    labels=['V2 off\nfusion off','V2 on\nfusion off','V2 off\nfusion on','V2 on\nfusion on']
    for ax,metric in zip(axs,['TE_cm','R5_pct','total_anchors']):
        for c,col,offset in [('0',BLUE,-.17),('1',GREEN,.17)]:
            vals=[float(next(r for r in macro if r['selection']=='compact' and r['V2']==v and r['fusion']==f and r['completion']==c)[metric]) for v,f in configurations]
            ax.bar(np.arange(4)+offset,vals,.32,label=f'completion {"on" if c=="1" else "off"}',color=col)
        ax.set_xticks(range(4),labels,fontsize=8)
        ax.set_ylabel({'TE_cm':'Mean scene-median TE (cm)','R5_pct':'Mean scene recall @ 5 cm / 5 deg (%)','total_anchors':'Total retained landmarks'}[metric])
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axs[0].legend(frameon=False,fontsize=8,loc='lower left',bbox_to_anchor=(0,1.02),ncol=2)
    fig.subplots_adjust(wspace=.32,bottom=.2,top=.85)
    save(fig,'F04_observations','All eight compact-map observation combinations; identical macro aggregation, no configurations suppressed.',
         ['Quantitative draft; actual per-observation V2 buffer/patch and descriptor-ranking illustrations remain to be extracted.',
          'Each configuration reselects its map; point counts differ. This is a pipeline ablation, not a fixed-map descriptor-only test.'])

    fig=plt.figure(figsize=(13,7));gs=fig.add_gridspec(2,3)
    for i,(state,label,col) in enumerate([(full,'Full',BLUE),(compact,'Compact',GREEN)]):
        ax=fig.add_subplot(gs[0,i]);mapview(ax,state,camera,col);ax.set_title(f'{label}: {len(state["anchor_xyz"]):,} points')
    ax=fig.add_subplot(gs[0,2]);ax.axis('off')
    ax.text(.05,.78,'Five-scene deployment maps',weight='bold',fontsize=12)
    for y,key,label in [(.57,'total_anchors','Points'),(.36,'total_map_MB','Map MB'),(.15,'TE_cm','Mean median TE (cm)')]:
        a,b=float(fullrow[key]),float(comprow[key]);ax.text(.05,y,f'{label}\n{a:,.2f}  →  {b:,.2f}',fontsize=12)
    for j,metric in enumerate(['anchors','TE_cm','R5_pct']):
        ax=fig.add_subplot(gs[1,j])
        for sel,col in [('full',BLUE),('compact',GREEN)]:
            vals=[float(next(r for r in factor if r['scene']==s and r['selection']==sel and all(r[k]=='1' for k in ['v2','fusion','completion']))[metric]) for s in SCENES]
            ax.plot(SHORT,vals,'o-',c=col,label=sel,lw=1.4,ms=4)
        ax.set_ylabel({'anchors':'Map points','TE_cm':'Median TE (cm)','R5_pct':'Recall @ 5 cm / 5 deg (%)'}[metric]);ax.grid(alpha=.15)
    ax.legend(frameon=False)
    fig.subplots_adjust(wspace=.35,hspace=.4,bottom=.12)
    save(fig,'F05_selection','Measured full/compact maps and all five-scene accuracy changes; no claim of lossless compression or runtime gain.',
         ['Overlay shows map projection, not occlusion-filtered visible landmarks.',
          'Matching/pose-information mechanistic insets require selector trace audits; this draft shows established compression evidence.'])

    # Baseline queries are aligned by recorded GT pose, not assumed array order.
    baseline=[]
    for method in ['STDLoc','ULFLoc']:
        receipt=js(BASE/f'receipts/{method}_ShopFacade_mapping_semantics_off_eval.json')
        p=Path(receipt['native_output'])
        if not p.is_absolute():p=BASE/'baselines'/method/p
        baseline.append((method,js(p/'results.json')))
    fig=plt.figure(figsize=(13,7));gs=fig.add_gridspec(2,3,height_ratios=[1.6,1])
    sample_records=[]
    for j,(record,kind) in enumerate(zip(selected,['Median TE','90th-percentile TE','Largest TE'])):
        cam=dataset.camera(record['image_name']);ax=fig.add_subplot(gs[0,j]);photo(ax,cam)
        ax.set_title(f'{kind}\n{record["image_name"]}',fontsize=10)
        values=[];names=[];sample={'image':record['image_name'],'rule':kind,'methods':{}}
        for method,rows in baseline:
            distances=[np.max(np.abs(np.asarray(r['gt_pose_w2c'])-cam.pose_w2c)) for r in rows]
            idx=int(np.argmin(distances));assert distances[idx]<1e-5
            assert sum(d<1e-5 for d in distances)==1, 'ambiguous GT-pose association'
            r=rows[idx]
            for prefix,stage in [('sparse','C'),('dense','F')]:
                values.append(float(r[prefix+'_TE']));names.append(f'{method}\n{stage}')
                sample['methods'][method+' '+stage]={'TE_cm':float(r[prefix+'_TE']),'RE_deg':float(r[prefix+'_AE'])}
        values.append(record['translation_error_cm']);names.append('Ours')
        sample['methods']['AnyGSLoc']={k:record[k] for k in ['translation_error_cm','rotation_error_deg']}
        ax=fig.add_subplot(gs[1,j]);ax.bar(names,values,color=[ORANGE,ORANGE,BLUE,BLUE,GREEN])
        ax.tick_params(axis='x',labelsize=8);ax.set_ylabel('Translation error (cm)');ax.grid(axis='y',alpha=.15)
        sample_records.append(sample)
    fig.subplots_adjust(wspace=.35,hspace=.16,bottom=.13)
    save(fig,'F06_query_cases','Real queries with coarse (C), fine (F), and AnyGSLoc error; samples chosen solely from ours TE ranks.',
         ['ShopFacade only; not evidence of five-scene qualitative generalization.',
          'This panel shows input queries and measured errors, not an overlay proving correspondence correctness.'],sample_records)

    timing=csvrows('baseline_timing_macro.csv');accuracy=csvrows('baseline_macro.csv')
    fig,axs=plt.subplots(1,3,figsize=(13,4.7))
    names=[];times=[];tes=[]
    for method in ['STDLoc','ULFLoc']:
        t=next(r for r in timing if r['method']==method and r['mapping_mask']=='off')
        for stage,col in [('coarse','coarse_prefix_ms'),('fine','coarse_plus_fine_ms')]:
            names.append(method+'\n'+stage);times.append(float(t[col]))
            tes.append(float(next(r for r in accuracy if r['method']==method and r['mapping_mask']=='off' and r['stage']==stage)['TE_cm']))
    names.append('Ours\nsingle');times.append(float(comprow['core_mean_ms']));tes.append(float(comprow['TE_cm']))
    colors=[ORANGE,ORANGE,BLUE,BLUE,GREEN]
    for ax,vals,label in [(axs[0],times,'Mean algorithm latency (ms)'),(axs[1],tes,'Mean scene-median TE (cm)')]:
        ax.bar(names,vals,color=colors);ax.tick_params(axis='x',labelsize=8);ax.set_ylabel(label)
        for k,v in enumerate(vals):ax.text(k,v,f'{v:.1f}',ha='center',va='bottom',fontsize=8)
        ax.margins(y=.16)
    axs[2].bar(['Ours full','Ours compact'],[float(fullrow['total_map_MB']),float(comprow['total_map_MB'])],color=[BLUE,GREEN]);axs[2].set_ylabel('Five-scene map storage (MB)')
    axs[2].set_ylim(0,float(fullrow['total_map_MB'])*1.3)
    axs[2].text(.5,.93,'Baseline deployment assets:\naudit pending; not plotted',transform=axs[2].transAxes,ha='center',va='top',fontsize=9)
    fig.subplots_adjust(wspace=.35,bottom=.18,top=.93)
    save(fig,'F07_cost_accuracy','Current reproduced configurations: algorithm mean latency and accuracy; own-map storage in decimal MB.',
         ['Native configurations differ; this is not an equal-resolution controlled algorithm speedup.',
          'Baselines mapping mask off; MAtCHA upstream semantic preparation disclosed separately. Baseline minimal asset sizes are not yet audited.',
          'No disk I/O or GT error evaluation in algorithm timing; no tail-latency panel.'])

    priors=csvrows('prior_macro.csv')
    pnames=['MAtCHA','Vanilla 2DGS','Vanilla 3DGS','i3DGS raw','i3DGS +1k','Long-LRM raw','Long-LRM +1k']
    pkeys=['vanilla_2dgs','vanilla_3dgs','i3dgs_raw','i3dgs_1k','long_lrm_raw','long_lrm_1k']
    prows=[comprow]+[next(r for r in priors if r['prior']==k) for k in pkeys]
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,metric,label in [(axs[0],'TE_cm','Mean scene-median TE (cm; log scale)'),(axs[1],'R5_pct','Mean scene recall @ 5 cm / 5 deg (%)')]:
        vals=[float(r[metric]) for r in prows]
        ax.barh(pnames,vals,color=[GREEN,BLUE,BLUE,ORANGE,ORANGE,'#8C6BB1','#8C6BB1']);ax.invert_yaxis();ax.set_xlabel(label)
        if metric=='TE_cm':ax.set_xscale('log');ax.set_xlim(.8,1500)
        else:ax.set_xlim(0,55)
        for y,v in enumerate(vals):ax.text(v*1.04 if metric=='TE_cm' else v+.5,y,f'{v:.2f}',va='center',fontsize=9)
    fig.subplots_adjust(wspace=.5,left=.14,bottom=.14,top=.95)
    save(fig,'F08_priors','All available prior-family macro results, including degraded feed-forward priors; no quality-insensitivity claim.',
         ['Prior training inputs and optimization conditions differ.',
          'Long-LRM failures retain upstream-quality/integration uncertainty.',
          'Matched-view prior renderings are not included in this draft; numbers are localization results, not image-quality scores.'])

    manifest={'status':'review_draft_not_submission_ready','figures':FIGURES,'sources':SOURCES,
              'generator':str(Path(__file__).resolve()),'aggregation':'scene-equal means of per-scene metrics',
              'runtime':'CPU visualization only; no new evaluations',
              'metadata':{'matplotlib':matplotlib.__version__,'torch':torch.__version__}}
    source(Path(__file__))
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    tiles=[]
    for name in FIGURES:
        im=Image.open(OUT/f'{name}.png').convert('RGB');im.thumbnail((740,430))
        tile=Image.new('RGB',(760,465),'#f2f4f7');tile.paste(im,((760-im.width)//2,22))
        ImageDraw.Draw(tile).text((14,443),name,fill=INK);tiles.append(tile)
    board=Image.new('RGB',(1520,465*4),'white')
    for i,tile in enumerate(tiles):board.paste(tile,((i%2)*760,(i//2)*465))
    board.save(OUT/'contact_sheet.jpg',quality=92)
    lines=['# AnyGSLoc v4 figure suite — review draft','',
           'Generated from frozen evidence; not submission-ready. Schematics and measurement limitations are explicit.',
           'Rebuild: `/root/miniconda3/envs/g4splat/bin/python tools/build_paper_figure_suite.py`','',
           '![All figures](contact_sheet.jpg)','']
    for name,r in FIGURES.items():
        lines.extend([f'## {name}','',r['caption'],'',f'[PDF]({name}.pdf) · [SVG]({name}.svg) · [PNG]({name}.png)',''])
        lines.extend('- '+s for s in r['limitations']);lines.append('')
    (OUT/'README.md').write_text('\n'.join(lines))
    print(json.dumps({'output':str(OUT),'figures':len(FIGURES),'sources':len(SOURCES)}))


if __name__=='__main__':main()
