"""E1: archived final poses projected against a shared dataset reference cloud.

CPU-only paper visualization. No method is rerun and no map is altered.
The COLMAP cloud supplies isolated landmarks, NOT invented structural edges.
"""
from pathlib import Path
import hashlib
import json
import struct
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from scipy.ndimage import sobel, uniform_filter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from anygsloc.data.datasets import ColmapDataset
POOL=Path('/mnt/pool/sqy')
OUT=ROOT/'paper/experiments_cvpr_20261003'
ASSETS=OUT/'assets/e1'
BASE=POOL/'anygsloc_baseline_recovery_v4_20260915'
FACTOR=POOL/'anygsloc_observation_factorial_20260914'
SCENES=['GreatCourt','KingsCollege','OldHospital','ShopFacade','StMarysChurch']
GT='#219653';EST='#cc2f83';INK='#233047';MUTED='#627083'
SOURCES={}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':8,
 'pdf.fonttype':42,'svg.fonttype':'none','text.color':INK,'savefig.facecolor':'white'})


def source(path):
    path=Path(path)
    if str(path) not in SOURCES:
        SOURCES[str(path)]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    return path


def read_sfm_points(path):
    """Read neutral dataset cloud, preserving stored point IDs and track lengths."""
    with Path(path).open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        xyz=np.empty((n,3));rgb=np.empty((n,3),np.uint8);ids=np.empty(n,np.uint64)
        error=np.empty(n);track_length=np.empty(n,np.int64)
        for i in range(n):
            r=struct.unpack('<QdddBBBd',f.read(43))
            ids[i]=r[0];xyz[i]=r[1:4];rgb[i]=r[4:7];error[i]=r[7]
            count=struct.unpack('<Q',f.read(8))[0];track_length[i]=count;f.seek(8*count,1)
    return {'xyz':xyz,'rgb':rgb,'id':ids,'error':error,'track_length':track_length}


def camera_K(camera):
    return np.array([[camera.width/(2*np.tan(camera.fov_x/2)),0,camera.width/2],
                     [0,camera.height/(2*np.tan(camera.fov_y/2)),camera.height/2],[0,0,1.]])


def project_points(xyz,pose,K):
    q=np.asarray(xyz)@np.asarray(pose)[:3,:3].T+np.asarray(pose)[:3,3]
    p=q@np.asarray(K).T
    with np.errstate(divide='ignore',invalid='ignore'):uv=p[:,:2]/p[:,2,None]
    return uv,q[:,2]


def _image_axes(fig,box,photo):
    ax=fig.add_axes(box)
    ax.imshow(photo);ax.set_xticks([]);ax.set_yticks([])
    for s in ax.spines.values():s.set_color('#d4dbe1');s.set_linewidth(.35)
    return ax


def load_candidates():
    candidates=[];datasets={};results={};sources={};invalid=[];association=[]
    for scene in SCENES:
        root=POOL/'Cambridge_stdloc'/scene
        for name in ['cameras.bin','images.bin']:source(root/'sparse/0'/name)
        source(root/'dataset_test.txt')
        dataset=ColmapDataset(root,load_masks=False);datasets[scene]=dataset
        ours_path=source(FACTOR/f'runs/{scene}_v1_f1_c1_compact/results.json')
        source(ours_path.with_name('manifest.json'))
        ours=json.loads(ours_path.read_text())
        methods={};src={'AnyGSLoc':str(ours_path)}
        for method in ['STDLoc','ULFLoc']:
            receipt_path=source(BASE/f'receipts/{method}_{scene}_mapping_semantics_off_eval.json')
            receipt=json.loads(receipt_path.read_text());p=Path(receipt['native_output'])
            if not p.is_absolute():p=BASE/'baselines'/method/p
            p=source(p/'results.json');r=json.loads(p.read_text());gt=np.asarray([x['gt_pose_w2c'] for x in r])
            methods[method]=(r,cKDTree(gt.reshape(len(gt),-1)));src[method]=str(p)
        results[scene]={};sources[scene]=src
        for o in ours:
            name=o['image_name'];cam=dataset.camera(name)
            if not o.get('success') or o.get('pose_w2c') is None:
                invalid.append([scene,name,'AnyGSLoc invalid pose']);continue
            r={'scene':scene,'image':name,'ours_TE':o['translation_error_cm'],'ours_RE':o['rotation_error_deg']}
            records={'AnyGSLoc':{'TE_cm':o['translation_error_cm'],'RE_deg':o['rotation_error_deg'],'pose_w2c':o['pose_w2c']}}
            valid=True
            for method,(rows,tree) in methods.items():
                distance,index=tree.query(cam.pose_w2c.flatten());assert distance<1e-5,(scene,name,method,distance)
                assert len(tree.query_ball_point(cam.pose_w2c.flatten(),r=1e-5))==1
                association.append(float(distance));b=rows[int(index)]
                if not b.get('dense') or b.get('dense_TE') is None:valid=False;continue
                records[method]={'TE_cm':b['dense_TE'],'RE_deg':b['dense_AE'],
                    'pose_w2c':b['dense'][-1]['pose_w2c'],'coarse_pose_w2c':b['sparse']['pose_w2c'],
                    'coarse_TE_cm':b['sparse_TE'],'coarse_RE_deg':b['sparse_AE'],'native_result_index':int(index)}
                r[method+'_TE']=b['dense_TE'];r[method+'_RE']=b['dense_AE']
            if not valid or not np.isfinite([r.get(k,np.nan) for k in ['ours_TE','ours_RE','STDLoc_TE','STDLoc_RE','ULFLoc_TE','ULFLoc_RE']]).all():
                invalid.append([scene,name,'At least one nonfinite pose/error']);continue
            r['delta']=r['ours_TE']-min(r['STDLoc_TE'],r['ULFLoc_TE'])
            candidates.append(r);results[scene][name]=records
    return candidates,datasets,results,sources,invalid,max(association)


def select_cases(candidates):
    median=float(np.median([r['ours_TE'] for r in candidates]))
    typical_pool=[r for r in candidates if all(r[k]<15 for k in ['ours_TE','STDLoc_TE','ULFLoc_TE']) and all(r[k]<5 for k in ['ours_RE','STDLoc_RE','ULFLoc_RE'])]
    typical=min(typical_pool,key=lambda r:(abs(r['ours_TE']-median),r['scene'],r['image']))
    quantiles=np.quantile([r['delta'] for r in candidates],[.02,.98])
    chosen=[typical]+[min(candidates,key=lambda r:(abs(r['delta']-target),r['scene'],r['image'])) for target in quantiles]
    return chosen,{'candidate_count':len(candidates),'typical_success_threshold':'all final TE < 15 cm and RE < 5 degrees',
      'typical_eligible_count':len(typical_pool),'pooled_ours_median_TE_cm':median,
      'typical_rule':'Closest AnyGSLoc TE to pooled median, within the stated common-success set.',
      'favorable_adverse_rule':'Nearest query to the 2nd and 98th percentiles of delta = AnyGSLoc TE - min(STDLoc final TE, ULF-Loc final TE), using the identical common finite-pose candidate set.',
      'delta_percentiles':[.02,.98],'delta_targets_cm':quantiles.tolist(),
      'tie_break':'Distance, scene name, image name; geometry and images do not enter query selection.'}


def choose_roi(photo,uv,z,width,height):
    """Choose a central textured dense reference-cloud region; uses GT only."""
    gray=np.asarray(photo.resize((width//4,height//4))).astype(float).mean(2)/255
    texture=np.hypot(sobel(gray,axis=0),sobel(gray,axis=1))
    rw,rh=480,120
    rows=[]
    for cy in np.arange(height*.34,height*.78+1,60):
        for cx in np.arange(width*.25,width*.75+1,60):
            x0=int(cx-rw/2);y0=int(cy-rh/2)
            mask=(z>0)&(uv[:,0]>=x0)&(uv[:,0]<x0+rw)&(uv[:,1]>=y0)&(uv[:,1]<y0+rh)
            n=int(mask.sum())
            if n<80:continue
            tex=float(texture[y0//4:(y0+rh)//4,x0//4:(x0+rw)//4].mean())
            score=np.log1p(n)*tex
            rows.append((score,x0,y0,n,tex))
    assert rows,'No GT-visible textured neutral-cloud ROI'
    score,x0,y0,n,texture=max(rows)
    return [x0,y0,x0+rw,y0+rh],{'criterion':'argmax log(1 + projected point count) * mean RGB gradient, central candidate grid; GT-only and method-independent','point_count':n,'score':float(score),'mean_gradient':texture}


def sparse_ids(uv,z,bounds,cell,quality=None):
    """Shared GT pixel grid; choose closest-to-cell-center reference point."""
    x0,y0,x1,y1=bounds
    valid=np.flatnonzero((z>0)&np.isfinite(uv).all(1)&(uv[:,0]>=x0)&(uv[:,0]<x1)&(uv[:,1]>=y0)&(uv[:,1]<y1))
    if quality is not None:valid=valid[quality[valid]>0]
    grid=np.floor((uv[valid]-[x0,y0])/cell).astype(int)
    dist=np.linalg.norm(uv[valid]-(grid+.5)*cell-[x0,y0],axis=1)
    order=np.lexsort((valid,dist if quality is None else -quality[valid]));chosen={}
    for j in order:chosen.setdefault(tuple(grid[j]),int(valid[j]))
    return np.array(list(chosen.values()),int)


def overlay(ax,xyz,gt_pose,est_pose,K,ids,origin=(0,0),zoom=False):
    g,gz=project_points(xyz[ids],gt_pose,K);e,ez=project_points(xyz[ids],est_pose,K)
    g-=origin;e-=origin
    # Corresponding landmark projections; these segments are displacement aids, not scene edges.
    valid=np.isfinite(e).all(1)&(ez>0)
    for a,b in zip(g[valid],e[valid]):
        ax.plot([a[0],b[0]],[a[1],b[1]],color='#ffffff',lw=.6 if zoom else .3,alpha=.72,zorder=2,clip_on=True)
    ax.scatter(g[:,0],g[:,1],s=15 if zoom else 6,marker='o',facecolors='none',edgecolors=GT,linewidths=.65 if zoom else .4,zorder=3)
    ax.scatter(e[valid,0],e[valid,1],s=8 if zoom else 2.2,marker='x',c=EST,linewidths=.65 if zoom else .45,zorder=4)
    return {'positive_depth':int(valid.sum()),'total':len(ids),'median_GT_est_displacement_px':float(np.median(np.linalg.norm(g[valid]-e[valid],axis=1))) if valid.any() else None}


def main():
    ASSETS.mkdir(parents=True,exist_ok=True)
    candidates,datasets,results,paths,invalid,gt_tol=load_candidates()
    (ASSETS/'query_scores.json').write_text(json.dumps(candidates,indent=2))
    cases,selection=select_cases(candidates)
    fig=plt.figure(figsize=(7.1,5.3))
    margin=.025;gap=.012;colw=(1-2*margin-3*gap)/4
    xs=[margin+i*(colw+gap) for i in range(4)]
    names=['Real query','STDLoc final','ULF-Loc final','AnyGSLoc']
    sub=['','Two-stage','Two-stage','Single-stage · compact']
    for x,label,s in zip(xs,names,sub):
        fig.text(x+colw/2,.978,label,ha='center',va='top',fontsize=8.6,weight='bold')
        if s:fig.text(x+colw/2,.950,s,ha='center',va='top',fontsize=7,color=MUTED)
    cloud_cache={};records=[];te_deltas=[]
    labels=['Typical','Ours-favorable','Adverse']
    for i,(sample,label) in enumerate(zip(cases,labels)):
        scene=sample['scene'];name=sample['image'];cam=datasets[scene].camera(name);K=camera_K(cam)
        photo=Image.open(source(cam.image_path)).convert('RGB');width,height=photo.size
        assert (width,height)==(cam.width,cam.height)
        cloud_path=source(POOL/'Cambridge_stdloc'/scene/'sparse/0/points3D.bin')
        if scene not in cloud_cache:cloud_cache[scene]=read_sfm_points(cloud_path)
        cloud=cloud_cache[scene];xyz=cloud['xyz'];uv,z=project_points(xyz,cam.pose_w2c,K)
        roi,roi_audit=choose_roi(photo,uv,z,width,height)
        # Remove isolated projected sky/outlier samples, using reference density and RGB texture only.
        on=np.flatnonzero((z>0)&np.isfinite(uv).all(1)&(uv[:,0]>=2)&(uv[:,0]<width-2)&(uv[:,1]>=2)&(uv[:,1]<height-2))
        density=cKDTree(uv[on]).query_ball_point(uv[on],r=16,return_length=True)
        gray=np.asarray(photo).astype(float).mean(2)/255
        gradient=np.hypot(sobel(gray,axis=0),sobel(gray,axis=1))
        coords=np.round(uv[on]).astype(int)
        texture=gradient[coords[:,1],coords[:,0]]
        quality=np.zeros(len(xyz));quality[on]=texture*np.log1p(density)
        quality[on[(density<12)|(texture<.10)]]=0
        main_ids=sparse_ids(uv,z,[60,60,width-60,height-60],240,quality)
        roi_ids=sparse_ids(uv,z,roi,72,quality)
        panel_records=results[scene][name]
        ytop=.911-i*.290
        fig.text(xs[0],ytop+.001,f'({chr(97+i)}) {label}',fontsize=7.7,weight='bold',va='bottom')
        main_bottom=ytop-.176;zoom_bottom=main_bottom-.085;zoomheight=colw*7.1/5.3*.25
        for j in range(4):
            ax=_image_axes(fig,[xs[j],main_bottom,colw,colw*7.1/5.3*.5625],photo)
            ax.set_xlim(0,width);ax.set_ylim(height,0)
            ax.add_patch(Rectangle((roi[0],roi[1]),roi[2]-roi[0],roi[3]-roi[1],fill=False,ec='white',lw=.6,zorder=5))
            zoom=_image_axes(fig,[xs[j],zoom_bottom,colw,zoomheight],photo.crop(tuple(roi)))
            zoom.set_xlim(0,roi[2]-roi[0]);zoom.set_ylim(roi[3]-roi[1],0)
            if j:
                key=['STDLoc','ULFLoc','AnyGSLoc'][j-1];r=panel_records[key]
                cm=np.linalg.norm(np.linalg.inv(np.asarray(r['pose_w2c']))[:3,3]-np.linalg.inv(cam.pose_w2c)[:3,3])*100
                te_deltas.append(abs(cm-r['TE_cm']))
                overlay(ax,xyz,cam.pose_w2c,r['pose_w2c'],K,main_ids)
                r['ROI_projection']=overlay(zoom,xyz,cam.pose_w2c,r['pose_w2c'],K,roi_ids,origin=roi[:2],zoom=True)
                # Annotated errors remain archived values, never recomputed substitutes.
                fig.text(xs[j]+colw/2,ytop+.001,f'{r["TE_cm"]:.2f} cm / {r["RE_deg"]:.3f}°',ha='center',va='bottom',fontsize=7.2)
            else:
                ax.text(.025,.965,scene,transform=ax.transAxes,ha='left',va='top',fontsize=7,color='white',bbox=dict(facecolor='#15202c',alpha=.68,pad=2,edgecolor='none'))
                zoom.text(.025,.075,'4× crop',transform=zoom.transAxes,ha='left',va='bottom',fontsize=7,color='white',bbox=dict(facecolor='#15202c',alpha=.72,pad=1.5,edgecolor='none'))
        stem=f'{i+1}_{scene}_{name.replace("/","_").replace(".png","")}'
        photo.save(ASSETS/f'{stem}_query.png')
        np.savez_compressed(ASSETS/f'{stem}_geometry.npz',xyz_main=xyz[main_ids],xyz_roi=xyz[roi_ids],id_main=cloud['id'][main_ids],id_roi=cloud['id'][roi_ids],K=K,gt_pose_w2c=cam.pose_w2c,ROI=np.array(roi))
        records.append({'row':label,**sample,'query_path':str(cam.image_path),'geometry_path':str(cloud_path),
            'native_width_height':[width,height],'K':K.tolist(),'gt_pose_w2c':cam.pose_w2c.tolist(),
            'ROI':roi,'ROI_selection':roi_audit,'main_landmark_count':len(main_ids),'ROI_landmark_count':len(roi_ids),
            'point_display_selection':'GT-only pixel grid; highest query RGB gradient times log(1 + 16px reference-point density); at least 12 neighbors and gradient >= 0.10; no predicted-pose information.',
            'main_point_ids':cloud['id'][main_ids].tolist(),'ROI_point_ids':cloud['id'][roi_ids].tolist(),
            'methods':panel_records,'result_sources':paths[scene]})
    handles=[Line2D([],[],color=GT,marker='o',mfc='none',linestyle='none',markersize=4,label='GT projection'),
             Line2D([],[],color=EST,marker='x',linestyle='none',markersize=4,label='Estimated-pose projection')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.009),ncol=2,frameon=False,fontsize=7,columnspacing=2)
    fig.text(.5,.048,'Shared reference-cloud landmarks · identical RGB and crop per row · TE / RE',ha='center',fontsize=7,color=MUTED)
    name='E1_final_localization'
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',dpi=350,bbox_inches=None,pad_inches=0)
    plt.close(fig)
    assert max(te_deltas)<.02,('Pose / archived TE mismatch',max(te_deltas))
    caption=('Final localization on three automatically selected Cambridge queries. Each method panel overlays the same neutral dataset COLMAP reference-cloud landmarks on the same real query: green circles use ground-truth poses and magenta crosses use archived final poses. White connectors indicate the displacement of corresponding projections; they are not structural edges. The lower strip is the same 4× crop for all methods. Annotations report archived translation/rotation errors (cm/degrees). STDLoc and ULF-Loc use their final refinement result; AnyGSLoc uses its single-stage compact-map result. The typical query is closest to pooled median AnyGSLoc TE among queries with all final poses below 15 cm / 5 degrees. Favorable/adverse rows are nearest to the symmetric 2nd/98th percentiles of AnyGSLoc TE minus the best baseline-final TE over the same 1,918 finite-pose queries. Geometry projection is offline visualization only.')
    limitations=['The reference cloud is sparse dataset geometry, not a mesh or verified architectural-edge model. No structural edges are invented.',
       'Projected cloud landmarks are not query-method matches or inlier labels. The figure is a pose-alignment visualization, not a correspondence correctness test.',
       'No occlusion test is applied. Positive depth and common GT image bounds define visible candidates; hidden reference points can remain.',
       'Query selection spans all five Cambridge scenes but three displayed queries do not establish five-scene qualitative generalization.',
       'The same neutral XYZ and point IDs are used across all methods in each row; display thinning and crop choice depend only on GT geometry and RGB.',
       'Near-perfect projection overlap may be below print resolution; archived TE / RE retain the quantitative difference.',
       'Tiny camera insets are omitted to keep the full-width 3 × 4 figure legible; archived coarse poses remain in the manifest.']
    manifest={'figure':name,'status':'paper_figure_with_disclosed_sparse_projection_limitations','caption':caption,
       'selection':selection,'rows':records,'excluded_queries':invalid,'limitations':limitations,
       'validation':{'max_baseline_GT_pose_association_l2':gt_tol,'max_archived_vs_projected_pose_TE_delta_cm':max(te_deltas),
       'no_algorithm_experiments':True,'all_poses_from_archived_results':True,'same_K_RGB_XYZ_and_crop_per_row':True},
       'source_sha256':SOURCES,'output_width_inches':7.1,'output_height_inches':5.3,'colors':{'GT':GT,'estimate':EST},
       'regenerate':'/root/miniconda3/envs/g4splat/bin/python tools/build_cvpr_e1_localization.py'}
    (OUT/f'{name}.json').write_text(json.dumps(manifest,indent=2))
    (OUT/f'{name}_caption.txt').write_text(caption+'\n\nDisclosure notes:\n'+'\n'.join('- '+x for x in limitations)+'\n')
    print(json.dumps({'outputs':str(OUT/name),'selected':cases,'max_TE_delta_cm':max(te_deltas)},indent=2))

if __name__=='__main__':main()
