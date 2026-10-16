"""Frozen-map compactness plate with one shared descriptor PCA and explicit ROI counts."""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'paper/experiments_cvpr_20261003'
A = OUT/'assets/e3'
MAPROOT = Path('/mnt/pool/sqy/anygsloc_observation_factorial_20260914/maps')
TABLE = ROOT/'docs/paper/evidence_20260917/factorial_80.csv'
SCENES = ['OldHospital', 'KingsCollege']
ALLSCENES = ['GreatCourt','KingsCollege','OldHospital','ShopFacade','StMarysChurch']
INK, MUTED, TEAL = '#22354A', '#647385', '#087F72'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20), b''): h.update(b)
    return h.hexdigest()


def read_map(scene, kind):
    return torch.load(MAPROOT/f'{scene}_v1_f1_c1/{kind}/deployment_map.pt', map_location='cpu', weights_only=False)


def prepare():
    torch.set_num_threads(4)
    A.mkdir(parents=True, exist_ok=True)
    table = list(csv.DictReader(TABLE.open()))
    rows = [r for r in table if r['v2']==r['fusion']==r['completion']=='1']
    totals = {k: dict(landmarks=0, serialized_bytes=0) for k in ['full','compact']}
    scene_counts = {}
    for scene in ALLSCENES:
        scene_counts[scene]={}
        report = json.loads((MAPROOT/f'{scene}_v1_f1_c1/report.json').read_text())
        for kind in ['full','compact']:
            r = next(r for r in rows if r['scene']==scene and r['selection']==kind)
            path = MAPROOT/f'{scene}_v1_f1_c1/{kind}/deployment_map.pt'
            assert int(r['anchors'])==report[f'{kind}_count']
            assert int(r['map_bytes'])==path.stat().st_size
            totals[kind]['landmarks']+=int(r['anchors'])
            totals[kind]['serialized_bytes']+=int(r['map_bytes'])
            scene_counts[scene][kind]={'landmarks':int(r['anchors']), 'serialized_bytes':int(r['map_bytes']),
                'median_TE_cm':float(r['TE_cm']), 'R5_pct':float(r['R5_pct'])}
    assert totals['full']['landmarks']==535840 and totals['compact']['landmarks']==176329
    selection=json.loads((ROOT/'paper/figures_v5/selection.json').read_text())
    first=json.loads((ROOT/'paper/fig1_geometry_sources_20261002/evidence.json').read_text())['views'][0]
    cameras={}
    for scene in SCENES:
        if scene=='KingsCollege':
            cameras[scene]={'name':first['name'], 'K':first['K'], 'pose':first['pose'], 'hw':[1080,1920]}
        else:
            v=next(v for v in selection['views'] if v['scene']==scene and 'thresholds' in v)
            cameras[scene]={'name':v['name'],'K':v['native_K'],'pose':v['pose_w2c'],'hw':v['native_input_hw']}
    data={}
    for scene in SCENES:
        full,compact=read_map(scene,'full'),read_map(scene,'compact')
        source=MAPROOT/f'{scene}_v1_f1_c1'
        report=json.loads((source/'report.json').read_text())
        rows_kept=np.array(report['retained_source_rows'],dtype=np.int64)
        assert len(np.unique(rows_kept))==len(rows_kept)
        assert torch.equal(full['anchor_xyz'][rows_kept],compact['anchor_xyz'])
        assert torch.allclose(full['anchor_features'][rows_kept],compact['anchor_features'],atol=1e-6,rtol=0)
        data[scene]={'xyz':full['anchor_xyz'].numpy(), 'features':full['anchor_features'].float(),
            'keep':np.zeros(len(full['anchor_xyz']),dtype=bool), 'rows_kept':rows_kept,
            'feature_max_abs_error':float((full['anchor_features'][rows_kept]-compact['anchor_features']).abs().max()),
            'full_sha256':sha(source/'full/deployment_map.pt'), 'compact_sha256':sha(source/'compact/deployment_map.pt'),
            'registry_sha256':sha(source/'report.json')}
        data[scene]['keep'][rows_kept]=True
        del compact,full
    # Exact covariance PCA over every Full-map descriptor in both displayed scenes.
    # Compact descriptors never participate in fitting or normalization.
    features=torch.cat([data[s]['features'] for s in SCENES],dim=0).double()
    mean=features.mean(0)
    features-=mean
    cov=features.T@features/(len(features)-1)
    eigenvalues,basis=torch.linalg.eigh(cov)
    basis=basis[:,-3:].flip(1)
    for k in range(3):
        if basis[basis[:,k].abs().argmax(),k] < 0: basis[:,k]*=-1
    pc=(features@basis).numpy()
    lo,hi=np.percentile(pc,[1,99],axis=0)
    np.savez_compressed(A/'shared_full_descriptor_pca.npz',mean=mean.numpy(),components=basis.numpy(),
        rgb_low=lo,rgb_high=hi,eigenvalues=eigenvalues.numpy())
    del features,pc,cov
    scene_meta={}
    for scene in SCENES:
        d=data[scene];c=cameras[scene]
        features=d.pop('features')
        rgb=np.clip(((features.double()-mean)@basis).numpy()-lo,0,hi-lo)/(hi-lo)
        xyz=d['xyz'];P=np.array(c['pose']);K=np.array(c['K']);h,w=c['hw']
        cam=xyz@P[:3,:3].T+P[:3,3];q=cam@K.T
        uv=q[:,:2]/np.maximum(q[:,2:],1e-8)-.5
        visible=(cam[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
        # Densest fixed-size full-map ROI, chosen without compact membership or pose errors.
        candidates=[]
        rw,rh=440,280
        for y in range(100,h-rh+1,60):
            for x in range(0,w-rw+1,60):
                roi=visible&(uv[:,0]>=x)&(uv[:,0]<x+rw)&(uv[:,1]>=y)&(uv[:,1]<y+rh)
                candidates.append((int(roi.sum()),x,y))
        count,x,y=max(candidates)
        roi=visible&(uv[:,0]>=x)&(uv[:,0]<x+rw)&(uv[:,1]>=y)&(uv[:,1]<y+rh)
        # The main crop is fixed per scene and reused verbatim in all three panels.
        crop=[0,0,w,h] if scene=='OldHospital' else [0,180,w,940]
        draw=visible&(uv[:,1]>=crop[1])&(uv[:,1]<crop[3])
        np.savez_compressed(A/f'{scene}_display.npz',uv=uv,rgb=rgb.astype(np.float32),keep=d['keep'],visible=draw,
            roi=roi,depth=cam[:,2],retained_source_rows=d['rows_kept'])
        scene_meta[scene]={'camera':c,'display_crop':crop,'roi_box':[x,y,x+rw,y+rh],
            'roi_full_count':int(roi.sum()),'roi_retained_count':int((roi&d['keep']).sum()),
            'roi_retention':float((roi&d['keep']).sum()/roi.sum()),
            'visible_full_count':int(draw.sum()),'visible_compact_count':int((draw&d['keep']).sum()),
            'full_map_path':str(MAPROOT/f'{scene}_v1_f1_c1/full/deployment_map.pt'),
            'compact_map_path':str(MAPROOT/f'{scene}_v1_f1_c1/compact/deployment_map.pt'),
            'registry_path':str(MAPROOT/f'{scene}_v1_f1_c1/report.json'),
            'source_hashes':{k:d[k] for k in ['full_sha256','compact_sha256','registry_sha256']},
            'compact_xyz_bitwise_equal_to_retained_full':True,
            'compact_descriptor_max_abs_difference':d['feature_max_abs_error']}
    p={'figure':'E3_compact_map','artifact_only':True,'table':str(TABLE),'table_sha256':sha(TABLE),
        'totals_all_five_scenes':totals,'all_scene_results':scene_counts,'displayed_scenes':scene_meta,
        'pca':{'fit_scenes':SCENES,'fit_population':'All Full-map descriptors in both displayed scenes, concatenated',
            'fit_count':sum(scene_counts[s]['full']['landmarks'] for s in SCENES),
            'method':'Exact covariance eigendecomposition in float64; leading three components; deterministic component sign',
            'rgb_scaling':'1st and 99th percentiles over the same Full descriptor population; clip identically for all panels',
            'compact_refit':False,'semantic_coloring':False,
            'explained_variance_ratio':(eigenvalues[-3:].flip(0)/eigenvalues.sum()).numpy().tolist()},
        'roi_selection':'Highest Full-map projected count in a fixed 440x280 native-pixel window, scanned every 60px from x=0,y=100. No compact membership or query accuracy used.',
        'limitations':['PCA colors visualize frozen SuperPoint descriptor structure, not semantic categories.',
            'All shown points are projections of real stored XYZ. Frustum tests do not perform occlusion or depth-surface visibility checks.',
            'Full and Compact use exactly the same camera, crop, marker size, PCA basis, and RGB scaling; the Selection overlay uses 1.5x larger retained markers to identify membership.',
            'ROI counts are descriptive statistics derived from frozen maps. The main count banners report whole-scene maps, not the number of points visible in the displayed camera.',
            'Both scenes are retained: selection improves OldHospital test error/recall but worsens KingsCollege. These examples do not establish uniform benefit.',
            'The headline compression and serialized deployment-map sizes sum all five scenes; they are not the totals for the two displayed scenes.',
            'No map building, optimization, feature extraction, or localization was rerun.']}
    (A/'prepared.json').write_text(json.dumps(p,indent=2)+'\n')
    return p


def panel(ax,arr,kind,crop,zoom=False):
    uv,rgb,keep=arr['uv'],arr['rgb'],arr['keep']
    valid=arr['visible']
    inside=valid&(uv[:,0]>=crop[0])&(uv[:,0]<crop[2])&(uv[:,1]>=crop[1])&(uv[:,1]<crop[3])
    size=.24 if not zoom else .70
    if kind=='selection':
        removed=inside&~keep
        ax.scatter(uv[removed,0],uv[removed,1],s=size*.75,c='#536479',alpha=.15,lw=0,rasterized=True)
        retained=inside&keep
        ax.scatter(uv[retained,0],uv[retained,1],s=size*1.5,c=rgb[retained],alpha=1,lw=0,rasterized=True)
    else:
        shown=inside if kind=='full' else inside&keep
        ax.scatter(uv[shown,0],uv[shown,1],s=size,c=rgb[shown],alpha=1,lw=0,rasterized=True)
    ax.set(xlim=(crop[0],crop[2]),ylim=(crop[3],crop[1]),aspect='equal')
    ax.set_axis_off()


def draw(p):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'text.color':INK,
        'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    fig=plt.figure(figsize=(7.1,5.25))
    xs=[.025,.359,.693];width=.284
    for x,title in zip(xs,['(a) Full map','(b) Selection overlay','(c) Compact map']):
        fig.text(x+width/2,.982,title,ha='center',va='top',fontsize=9,weight='bold')
    for row,scene in enumerate(SCENES):
        m=p['displayed_scenes'][scene];arr=np.load(A/f'{scene}_display.npz')
        stats=p['all_scene_results'][scene];full,compact=stats['full'],stats['compact']
        top=.919-row*.436
        fig.text(.025,top,scene,fontsize=8.5,weight='bold',va='top')
        fig.text(.215,top,f"{full['landmarks']:,} → {compact['landmarks']:,} landmarks  (−{100*(1-compact['landmarks']/full['landmarks']):.1f}%)",
            fontsize=8,va='top',color=TEAL)
        fig.text(.980,top-.033,f"Test TE {full['median_TE_cm']:.2f} → {compact['median_TE_cm']:.2f} cm  |  R₅ {full['R5_pct']:.2f} → {compact['R5_pct']:.2f}%",
            fontsize=7,ha='right',va='top',color=MUTED)
        for col,kind in enumerate(['full','selection','compact']):
            ax=fig.add_axes([xs[col],top-.236,width,.174])
            panel(ax,arr,kind,m['display_crop'])
            x0,y0,x1,y1=m['roi_box']
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec=INK,lw=.7))
            ax.text(x0+9,y0+42,'ROI',fontsize=7,color=INK,
                bbox=dict(fc='white',ec='none',pad=.2,alpha=.8))
            ax=fig.add_axes([xs[col],top-.395,width,.139])
            panel(ax,arr,kind,m['roi_box'],zoom=True)
            for spine in ax.spines.values():spine.set_visible(True)
            # An explicit frame distinguishes the enlarged area from the full view.
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec='#D3DBE2',lw=.75))
            if col==0: label=f"ROI: {m['roi_full_count']:,} landmarks"
            elif col==1: label='Gray: removed  ·  Color: retained'
            else: label=f"ROI: {m['roi_retained_count']:,}  ({100*m['roi_retention']:.1f}% retained)"
            fig.text(xs[col]+width/2,top-.401,label,ha='center',va='top',fontsize=7,color=MUTED)
    total=p['totals_all_five_scenes'];fn,cn=total['full']['landmarks'],total['compact']['landmarks']
    fig.text(.5,.027,f'All five scenes:  {fn:,} → {cn:,} landmarks   |   −{100*(1-cn/fn):.1f}%   |   {total["full"]["serialized_bytes"]/1e6:.2f} → {total["compact"]["serialized_bytes"]/1e6:.2f} MB',
        ha='center',va='center',fontsize=8.1,color=TEAL,weight='bold',
        bbox=dict(boxstyle='round,pad=.40,rounding_size=.18',fc='#EDF7F4',ec='none'))
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    text_objects=[t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    outside=[]
    for t in text_objects:
        b=t.get_window_extent(renderer)
        if b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.width+.5 or b.y1>fig.bbox.height+.5:
            outside.append(t.get_text())
    assert not outside, outside
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'E3_compact_map.{ext}',dpi=400)
    plt.close(fig)
    return {'visible_text_items':len(text_objects),'text_outside_canvas':outside,
        'minimum_declared_font_pt':min(t.get_fontsize() for t in text_objects),
        'visual_inspection':'PNG inspected at final composition after reducing marker sizes to avoid saturated Full/Compact plots.'}


def main():
    p=json.loads((A/'prepared.json').read_text()) if '--redraw' in sys.argv else prepare()
    p['render_validation']=draw(p)
    p['pca']['basis_path']=str(A/'shared_full_descriptor_pca.npz')
    p['pca']['basis_sha256']=sha(A/'shared_full_descriptor_pca.npz')
    p['plot_parameters']={'figure_inches':[7.1,5.25],'main_point_area_pt2':.24,'roi_point_area_pt2':.70,
        'selection_retained_area_multiplier':1.5,'selection_removed_alpha':.15,'raster_dpi':400}
    p['outputs']={ext:{'path':str(OUT/f'E3_compact_map.{ext}'),'sha256':sha(OUT/f'E3_compact_map.{ext}')} for ext in ['pdf','svg','png']}
    (OUT/'E3_compact_map.provenance.json').write_text(json.dumps(p,indent=2)+'\n')
    caption=('Compact-map selection on frozen artifacts. Full maps, selection overlays, and compact maps use matched cameras and crops. '
        'Points are colored by one shared three-component PCA projection of frozen SuperPoint descriptors, fitted once on every Full-map descriptor in the two displayed scenes; fixed Full-population 1–99 percentile bounds are reused in every panel. '
        'Gray points in the overlays are removed landmarks. Full/Compact have identical point sizes; overlay retained points are slightly enlarged to identify selection membership. '
        'The lower strips magnify the same densest Full-map ROI; their labels report actual local counts and retention. '
        'Whole-scene counts and held-out test scores above each row preserve a favorable OldHospital example and an unfavorable KingsCollege example. '
        'The bottom banner sums all five Cambridge scenes: 535,840 to 176,329 landmarks, a 67.1% reduction, with serialized deployment maps decreasing from 559.43 to 184.10 MB. '
        'Map projections are not occlusion tests, and PCA colors do not denote semantic categories. All statistics derive from existing artifacts; no localization or mapping experiment was rerun.')
    (OUT/'E3_compact_map.caption.txt').write_text(caption+'\n\nLimitations:\n'+'\n'.join('- '+s for s in p['limitations'])+'\n')
    print(json.dumps({'totals':p['totals_all_five_scenes'],'rois':{s:{k:v for k,v in m.items() if k.startswith('roi_')} for s,m in p['displayed_scenes'].items()}},indent=2))


if __name__=='__main__':main()
