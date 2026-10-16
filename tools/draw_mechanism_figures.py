"""Data-first publication figures with explicit sample-selection boundaries."""
from pathlib import Path
import json,csv,sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Ellipse
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'paper/figures_v5';A=OUT/'assets';D=ROOT/'docs/paper/evidence_20260917'
POOL=Path('/mnt/pool/sqy');TEAL='#187b70';BLUE='#336f9c';GOLD='#b87b38';RED='#bd5954';INK='#243349'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.titleweight':'bold',
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none',
    'text.color':INK,'axes.labelcolor':INK,'savefig.facecolor':'white'})
FIGS={}
def rows(name):
    with (D/name).open() as f:return list(csv.DictReader(f))
def save(fig,name,caption,limitations):
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',dpi=240,bbox_inches='tight',pad_inches=.06)
    plt.close(fig);FIGS[name]={'caption':caption,'limitations':limitations}
def crop(ax,v,marks=False):
    ax.imshow(Image.open(A/(v['stem']+'_crop.png')));ax.axis('off')
    if marks:
        p=np.load(A/(v['stem']+'.npz'));x0,y0,x1,y1=v['crop'];xy=p['xy']-[x0,y0]
        good=(xy[:,0]>=0)&(xy[:,0]<x1-x0)&(xy[:,1]>=0)&(xy[:,1]<y1-y0)
        for keep,col,marker,label in [(True,TEAL,'o','retained'),(False,RED,'x','rejected')]:
            q=xy[good&(p['keep']==keep)]
            cells=np.floor(q/32).astype(int)
            _,first=np.unique(cells,axis=0,return_index=True);q=q[np.sort(first)]
            if keep:ax.scatter(q[:,0],q[:,1],s=16,marker=marker,facecolors='none',edgecolors=col,linewidths=.8,label=label)
            else:ax.scatter(q[:,0],q[:,1],s=16,marker=marker,c=col,linewidths=.8,label=label)
def context(ax,v):
    ax.imshow(Image.open(A/(v['stem']+'_rgb.png')));x0,y0,x1,y1=v['crop']
    ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec=GOLD,lw=1.7));ax.axis('off')
def validity(ax,v,binary=False):
    valid=np.load(A/(v['stem']+'_mask.npz'))['valid']
    if binary:ax.imshow(valid,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
    else:
        rgb=np.asarray(Image.open(A/(v['stem']+'_crop.png')).convert('RGB'))/255.
        gray=rgb.mean(axis=2,keepdims=True)
        muted=np.repeat(.20+.25*gray,3,axis=2)
        ax.imshow(np.where(valid[...,None],rgb,muted),interpolation='nearest')
    ax.axis('off')
def buffers(ax,v,key):
    a=np.load(A/(v['stem']+'_buffers.npz'))[key]
    cmap='cividis' if key=='depth' else 'magma'
    lim=np.percentile(a[np.isfinite(a)], [2,98]) if key=='depth' else [0,1]
    im=ax.imshow(a,cmap=cmap,vmin=lim[0],vmax=lim[1]);ax.axis('off');return im
def gaussians(ax,v):
    a=np.load(A/(v['stem']+'_gaussians.npz'));x0,y0,x1,y1=v['crop']
    for uv,cov in zip(a['uv'],a['cov']):
        val,vec=np.linalg.eigh(cov);val=np.maximum(val,0);order=np.argsort(val)[::-1];val=val[order];vec=vec[:,order]
        angle=np.degrees(np.arctan2(vec[1,0],vec[0,0]));sizes=2*np.sqrt(val)
        ax.add_patch(Ellipse(uv,*sizes,angle=angle,fc=BLUE,ec=BLUE,alpha=.19,lw=.4,clip_on=True))
    ax.set(xlim=(0,x1-x0),ylim=(y1-y0,0),aspect='equal');ax.axis('off')

def main():
    m=json.loads((OUT/'selection.json').read_text());vv=[v for v in m['views'] if 'thresholds' in v]
    # All scenes remain visible in the sample board, including weak examples.
    fig,axs=plt.subplots(5,4,figsize=(10,10.6))
    for i,v in enumerate(vv):
        context(axs[i,0],v);validity(axs[i,1],v);validity(axs[i,2],v,True);crop(axs[i,3],v,True)
        axs[i,0].set_title(v['scene']+' | '+v['name'],loc='left',fontsize=7)
        axs[i,1].set_title(f"ROI: {v['crop_accepted']} retained / {v['crop_rejected']} rejected",fontsize=7)
    for ax,t in zip(axs[0],['Scene & selected region','Validity overlay','Binary validity mask','Archived point decisions']):
        ax.text(.5,1.23,t,transform=ax.transAxes,ha='center',fontsize=9,weight='bold')
    fig.subplots_adjust(wspace=.1,hspace=.32,bottom=.03,top=.94)
    save(fig,'S01_five_scene_sample_board','Five-scene audit board: actual MAtCHA renders, dense V2 validity predicates, and archived point decisions. Color/white denotes admissible support; gray/black denotes excluded support.',
         ['Views are selected at the 90th-percentile rejection rate, not representative random samples.','Point display retains one marker per 32 px cell per decision class; reported counts use all ROI observations.','Binary masks reproduce all archived point decisions exactly; these are non-semantic render-validity predicates, not correctness ground truth.'])
    # Strong local examples are allowed, but the rule is explicit and all candidates remain available.
    v=max(vv,key=lambda r:min(r['crop_accepted'],r['crop_rejected']))
    fig,axs=plt.subplots(2,3,figsize=(8.0,4.1))
    context(axs[0,0],v);gaussians(axs[0,1],v);crop(axs[0,2],v)
    imd=buffers(axs[1,0],v,'depth');validity(axs[1,1],v,True);validity(axs[1,2],v)
    titles=['(a) Gaussian-rendered view','(b) Gaussian support','(c) Rendered RGB region',
            '(d) Native depth evidence','(e) V2 validity mask','(f) Validity overlay']
    for ax,t in zip(axs.flat,titles):ax.set_title(t,loc='left',fontsize=8)
    fig.colorbar(imd,ax=axs[1,0],fraction=.025,pad=.02)
    axs[1,0].set_title('(d) Depth evidence (m)',loc='left',fontsize=8)
    axs[1,1].text(.5,-.08,'White: admissible  ·  Black: excluded',transform=axs[1,1].transAxes,ha='center',fontsize=7)
    axs[1,2].text(.5,-.08,'Color: admissible  ·  Gray: excluded',transform=axs[1,2].transAxes,ha='center',fontsize=7)
    fig.text(.5,.015,f"{v['scene']}  ·  Same region throughout  ·  {v['crop_accepted']} retained / {v['crop_rejected']} rejected observations",ha='center',fontsize=8,color=TEAL)
    fig.subplots_adjust(wspace=.18,hspace=.18,bottom=.13,top=.93)
    save(fig,'F03_render_validity','Real Gaussian primitives and rendered evidence in StMarysChurch. The binary mask visualizes the full V2 render-validity predicate; the RGB overlay preserves admissible regions in color and mutes excluded regions in gray. The same crop is used throughout the local panels.',
         ['Gaussian panel: one nearest primitive center per 24 px cell, first-order covariance projection, not an alpha-composited rendering.',
          'The mask combines alpha/finite-depth support, depth continuity, RGB structure support, image borders and extreme distortion; no pixel-nonblack or semantic gate is added.',
          'Retained/rejected are algorithm decisions, not ground-truth correct/incorrect labels.',
          'Dense predicate is evaluated at native full-frame resolution then cropped; sampled decisions exactly match archived keypoints. No label interpolation or semantic segmentation.',
          'Muted RGB is a display overlay only; V2 filters detected observations, not the input RGB before feature extraction.',
          'Selected strongest balanced ROI among the five archived candidates; no test error was used.'])
    # Geometry figure: three recorded observations of one anchor plus held-out results.
    # Both candidate tracks are retained in selection.json; choose the one whose
    # mapping projections most clearly distinguish the coordinate sources.
    def contrast(g):
        view_records=[next(v for v in m['views'] if v['stem']==s) for s in g['views']]
        return np.median([max(np.linalg.norm(np.array(v['predicted_xy'][x])-v['predicted_xy']['ray']) for x in ['center','depth']) for v in view_records])
    geo=max(m['geometry'],key=contrast)
    views=[next(v for v in m['views'] if v['stem']==s) for s in geo['views']]
    fig=plt.figure(figsize=(8,5.1));gs=fig.add_gridspec(2,3,height_ratios=[1.2,1])
    for i,vv_i in enumerate(views):
        ax=fig.add_subplot(gs[0,i]);crop(ax,vv_i)
        origin=np.array(vv_i['crop'][:2]);u=np.array(vv_i['observed_xy'])-origin
        ax.scatter(*u,marker='+',s=110,c='white',linewidths=2,zorder=5)
        for g,col,marker in [('center',GOLD,'s'),('depth',BLUE,'^'),('ray',TEAL,'o')]:
            p=np.array(vv_i['predicted_xy'][g])-origin
            ax.plot([u[0],p[0]],[u[1],p[1]],color=col,lw=.8,alpha=.8)
            ax.scatter(*p,marker=marker,s=65,facecolors='none',edgecolors=col,linewidths=1.8,label=g)
        ax.set_title(f"View {i+1} · {vv_i['name']}",fontsize=7)
        ax.set_xlim(28,164);ax.set_ylim(164,28)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,-.22),ncol=3,frameon=False,fontsize=7)
    table=rows('geometry_15.csv');scenes=['GreatCourt','KingsCollege','OldHospital','ShopFacade','StMarysChurch'];short=['GC','KC','OH','SF','SM']
    for j,key,label in [(0,'TE_cm','Test median TE (cm)'),(1,'R5_pct','Test recall @ 5 cm / 5° (%)')]:
        ax=fig.add_subplot(gs[1,j]);x=np.arange(5)
        for g,col,marker in [('center',GOLD,'s'),('depth',BLUE,'^'),('ray',TEAL,'o')]:
            vals=[float(next(r for r in table if r['scene']==s and r['geometry']==g)[key]) for s in scenes]
            ax.plot(x,vals,marker=marker,color=col,lw=1,ms=3.5,label=g)
        ax.set_xticks(x,short);ax.set_ylabel(label);ax.grid(axis='y',alpha=.15)
    ax=fig.add_subplot(gs[1,2]);ax.axis('off')
    ax.text(0,.95,'Re-estimate the landmark',weight='bold',fontsize=9,va='top')
    ax.text(0,.72,'Same anchor identity\nSame deployed descriptor\nDifferent coordinate sources',fontsize=8,linespacing=1.6,va='top')
    ax.text(0,.27,'White + : cached observation\nColors: projected coordinates\nNot independent 3D ground truth',fontsize=7,color='#5d6c7c',linespacing=1.5,va='top')
    fig.subplots_adjust(wspace=.34,hspace=.58,bottom=.09,top=.96)
    save(fig,'F04_landmark_geometry',f"A recorded {geo['scene']} track illustrated across three rendered mapping views; coordinate-source controls evaluated on all five test scenes.",
         ['Local observations are mapping evidence; reduced mapping reprojection discrepancy is not an independent accuracy test.',
          'Ray uses original Track support while center/depth use common usable observations; all coordinate controls share IDs and descriptors.',
          'Center control is the existing aggregated projected-center definition, not necessarily an individual Gaussian center.',
          'Figure crop is centered on the recorded observation; out-of-crop projections are clipped, not moved.'])
    # Two selection examples: favorable OldHospital and a counterexample KingsCollege.
    from anygsloc.data.datasets import ColmapDataset
    fact=rows('factorial_80.csv');scene_data=[]
    for scene in ['OldHospital','KingsCollege']:
        view=next(v for v in m['views'] if v['scene']==scene and 'thresholds' in v)
        data=ColmapDataset(str(POOL/'Cambridge_stdloc'/scene),load_masks=False);cam=data.camera(view['name'])
        maps={g:torch.load(POOL/f'anygsloc_observation_factorial_20260914/maps/{scene}_v1_f1_c1/{g}/deployment_map.pt',map_location='cpu',weights_only=True) for g in ['full','compact']}
        # IDs are renumbered on subset export; the source-row registry is authoritative.
        registry=json.loads((POOL/f'anygsloc_observation_factorial_20260914/maps/{scene}_v1_f1_c1/report.json').read_text())
        indices=np.array(registry['retained_source_rows'],dtype=int)
        assert len(np.unique(indices))==len(indices)
        assert torch.equal(maps['full']['anchor_xyz'][indices],maps['compact']['anchor_xyz'])
        assert torch.allclose(maps['full']['anchor_features'][indices],maps['compact']['anchor_features'],atol=1e-6)
        retained=np.zeros(len(maps['full']['anchor_xyz']),dtype=bool);retained[indices]=True;xyz=maps['full']['anchor_xyz'].numpy()
        p=cam.pose_w2c;q=xyz@p[:3,:3].T+p[:3,3];k=np.array(view['native_K']);uv=q@k.T;uv=uv[:,:2]/np.maximum(uv[:,2:],1e-8)
        h,w=view['native_input_hw'];valid=(q[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
        scene_data.append((scene,view,maps,retained,uv,valid,q))
        np.savez_compressed(A/f'{scene}_selection_projection.npz',uv=uv,keep=retained,valid=valid,
                            hw=view['native_input_hw'],full_count=len(retained),compact_count=int(retained.sum()))
    fig,axs=plt.subplots(2,3,figsize=(8.2,4.8))
    for row,(scene,view,maps,keep,uv,good,q) in enumerate(scene_data):
        h,w=view['native_input_hw'];fulln=len(keep);n=int(keep.sum())
        for col,mask,title in [(0,good,'Candidates: retained / removed'),(1,good&keep,'Compact map')]:
            ax=axs[row,col]
            if col==0:
                ax.scatter(uv[good&~keep,0],uv[good&~keep,1],s=.08,c='#bfc7ce',rasterized=True)
            ax.scatter(uv[good&keep,0],uv[good&keep,1],s=.08,c=TEAL,rasterized=True)
            ax.set(xlim=(0,w),ylim=(h,0),aspect='equal');ax.axis('off');ax.set_title(scene+'\n'+('Retained / removed' if col==0 else title),fontsize=8,loc='left')
            ax.text(.02,-.07,f'{fulln:,} → {n:,} landmarks  (−{100*(1-n/fulln):.1f}%)' if col==0 else 'Same camera • identical point size',transform=ax.transAxes,fontsize=7,color=TEAL)
        ax=axs[row,2]
        z=q[good,2];lo,hi=np.percentile(z,[2,98]);edges=np.linspace(lo,hi,11)
        ax.hist(q[good&~keep,2],bins=edges,color='#bfc7ce',label='removed',histtype='step',lw=1.2,density=True)
        ax.hist(q[good&keep,2],bins=edges,color=TEAL,label='retained',histtype='step',lw=1.5,density=True)
        ax.set_xlabel('Camera depth (m)');ax.set_ylabel('Normalized density');ax.set_title('Retained depth distribution',fontsize=8)
        r={sel:next(r for r in fact if r['scene']==scene and r['selection']==sel and all(r[k]=='1' for k in ['v2','fusion','completion'])) for sel in ['full','compact']}
        ax.text(0,-.37,f"Test TE: {float(r['full']['TE_cm']):.2f} → {float(r['compact']['TE_cm']):.2f} cm\nTest R5: {float(r['full']['R5_pct']):.2f} → {float(r['compact']['R5_pct']):.2f}%",transform=ax.transAxes,fontsize=7,va='top')
    axs[0,2].legend(frameon=False,fontsize=7,ncol=2)
    fig.subplots_adjust(wspace=.35,hspace=1.05,bottom=.20,top=.92)
    save(fig,'F05_compact_selection','Actual selected and removed landmarks with matched plotting parameters; a favorable OldHospital example and KingsCollege counterexample.',
         ['Depth histograms normalize each group separately and use full-map projected-depth 2–98% bounds, excluding displayed tails.',
          'Map projection is not an occlusion test. Depth distribution describes the result, not causal proof of the pose-information objective.',
          'Mechanistic matching-reassignment or log-det gains are not depicted before trace verification.'])
    # A visual storyboard, based on these real assets, not a replacement for mechanistic panels.
    chosen_scene=v['scene'];chosen_geo=views[0]
    fig=plt.figure(figsize=(10,4));gs=fig.add_gridspec(2,5,height_ratios=[1.5,.6])
    ax=fig.add_subplot(gs[0,0]);gaussians(ax,v);ax.set_title('Gaussian prior',loc='left')
    ax=fig.add_subplot(gs[0,1]);crop(ax,v,True);ax.set_title('Admissible\nobservations',loc='left')
    ax=fig.add_subplot(gs[0,2]);crop(ax,chosen_geo);origin=np.array(chosen_geo['crop'][:2])
    for g,col,mk in [('center',GOLD,'s'),('depth',BLUE,'^'),('ray',TEAL,'o')]:
        p=np.array(chosen_geo['predicted_xy'][g])-origin;ax.scatter(*p,s=60,marker=mk,facecolors='none',edgecolors=col,lw=2)
    ax.set_title('Reconstructed\nlandmark',loc='left')
    scene,view,maps,keep,uv,good,q=scene_data[0]
    ax=fig.add_subplot(gs[0,3]);ax.scatter(uv[good&keep,0],uv[good&keep,1],s=.1,c=TEAL,rasterized=True);ax.set_xlim(0,view['native_input_hw'][1]);ax.set_ylim(view['native_input_hw'][0],0);ax.set_aspect('equal');ax.axis('off');ax.set_title('Compact map',loc='left')
    ax=fig.add_subplot(gs[0,4]);ax.axis('off');ax.text(.5,.65,'2D → 3D',ha='center',fontsize=19,color=TEAL);ax.text(.5,.36,'one robust\npose solve',ha='center',fontsize=11);ax.set_title('Sparse localization',loc='left')
    descriptions=[chosen_scene+'\nProjected primitives','Actual V2 row decisions\nNative rendering evidence',geo['scene']+'\nCoordinate-source distinction','OldHospital\nSelected actual landmarks','No online renderer\nNo dense refinement']
    for i,t in enumerate(descriptions):
        ax=fig.add_subplot(gs[1,i]);ax.axis('off');ax.text(.5,.8,t,ha='center',va='top',fontsize=8,color='#5d6c7c')
    fig.subplots_adjust(wspace=.2,hspace=.04,bottom=.07,top=.87)
    save(fig,'D01_overview_storyboard','Asset-driven alternative overview storyboard; components explicitly use different scenes.',
         ['Layout candidate, not a submission-ready overview; descriptor fusion and selection-objective graphics still require integration.',
          'Panels are illustrative examples, not one continuous scene or one anchor traced across all stages.'])
    (OUT/'figure_manifest.json').write_text(json.dumps({'figures':FIGS,'validity_hero':v,'geometry_hero':geo},indent=2))
    tiles=[]
    for name in FIGS:
        if name.startswith('S01'):continue
        im=Image.open(OUT/(name+'.png')).convert('RGB');im.thumbnail((1050,650))
        tile=Image.new('RGB',(1080,700),'#f5f7f9');tile.paste(im,((1080-im.width)//2,15));ImageDraw.Draw(tile).text((15,675),name,fill=INK);tiles.append(tile)
    board=Image.new('RGB',(2160,1400),'white')
    for i,t in enumerate(tiles):board.paste(t,((i%2)*1080,(i//2)*700))
    board.save(OUT/'contact_sheet.jpg',quality=94)
    print('figures:',len(FIGS),flush=True)

if __name__=='__main__':main()
