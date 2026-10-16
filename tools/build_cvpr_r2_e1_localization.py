"""Round-2 E1: shared-Gaussian GT/estimated-pose alignment from archived poses.

Prepare/draw are CPU only; render uses CPU SH plus one native GPU rasterization
at a time. This is paper-only visualization, never a localization experiment.
"""
from pathlib import Path
import argparse, gc, hashlib, json, sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from build_cvpr_e1_localization import load_candidates,camera_K,read_sfm_points,project_points,SOURCES
POOL=Path('/mnt/pool/sqy');OUT=ROOT/'paper/experiments_cvpr_r2_20261003';ASSETS=OUT/'assets/e1'
PROV=OUT/'E1_final_localization.provenance.json'
METHODS=['STDLoc','ULFLoc','AnyGSLoc'];GT='#219653';EST='#cc2f83';INK='#233047';GRAY='#697585'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def save(p):PROV.write_text(json.dumps(p,indent=2))

def prepare():
    ASSETS.mkdir(parents=True,exist_ok=True)
    allrows,ds,results,paths,invalid,tol=load_candidates()
    eligible=[]
    for r in allrows:
        ok=(r['ours_TE']<10 and r['ours_RE']<.2 and
            r['ours_TE']<min(r['STDLoc_TE'],r['ULFLoc_TE']) and
            r['ours_RE']<min(r['STDLoc_RE'],r['ULFLoc_RE']) and
            max(r['STDLoc_TE'],r['ULFLoc_TE'])<100 and max(r['STDLoc_RE'],r['ULFLoc_RE'])<2)
        r['eligible']=bool(ok)
        if ok:eligible.append(r)
    from scipy.spatial import cKDTree
    screened=[];geometry_sources={}
    for scene in sorted(set(r['scene'] for r in eligible)):
        sfm=POOL/'Cambridge_stdloc'/scene/'sparse/0/points3D.bin'
        xyz=read_sfm_points(sfm)['xyz'];xyz=xyz[np.linspace(0,len(xyz)-1,min(30000,len(xyz)),dtype=int)]
        geometry_sources[str(sfm)]=digest(sfm)
        for r in [r for r in eligible if r['scene']==scene]:
            cam=ds[scene].camera(r['image']);K=camera_K(cam);uv,z=project_points(xyz,cam.pose_w2c,K)
            idx=np.flatnonzero((z>0)&(uv[:,0]>96)&(uv[:,0]<cam.width-96)&(uv[:,1]>96)&(uv[:,1]<cam.height-96))
            density=cKDTree(uv[idx]).query_ball_point(uv[idx],r=24,return_length=True);idx=idx[density>=6]
            cells=np.floor(uv[idx]/120).astype(int);_,which=np.unique(cells[:,0]+16*cells[:,1],return_index=True);idx=idx[which]
            dis=[]
            for m in METHODS:
                est,zz=project_points(xyz[idx],results[scene][r['image']][m]['pose_w2c'],K)
                dd=np.linalg.norm(est-uv[idx],axis=1);dis.append(dd)
                r[m+'_median_screen_px']=float(np.median(dd));r[m+'_p80_screen_px']=float(np.quantile(dd,.8))
            d=np.array(dis);r['screen_score']=float(np.quantile(np.minimum(d[0],d[1])-d[2],.8));r['screen_samples']=len(idx);screened.append(r)
    screen_eligible=[r for r in screened if r['AnyGSLoc_median_screen_px']<1.0]
    winners=[max([r for r in screen_eligible if r['scene']==sc],key=lambda r:(r['screen_score'],r['image'])) for sc in sorted(set(r['scene'] for r in screen_eligible))]
    winners=sorted(winners,key=lambda r:(-r['screen_score'],r['scene'],r['image']))[:3]
    (ASSETS/'screen_displacement_candidates.json').write_text(json.dumps(screened,indent=2))
    p={'figure':'E1_final_localization','revision':2,'scope':'Selected illustrative favorable examples. Existing compact AnyGSLoc and final baseline poses; no new localization experiments.',
       'selection':{'all_common_finite_queries':len(allrows),'eligible_queries':len(eligible),
         'query_filter':'AnyGSLoc TE < 10 cm, RE < 0.2 deg, and both TE and RE strictly lower than each baseline final. Both baseline finals TE < 100 cm and RE < 2 deg for readable alignment.',
         'ranking':'Require AnyGSLoc median screen displacement < 1 native px. Within each scene maximize p80 over common reference points of min(STDLoc displacement, ULF-Loc displacement) - AnyGSLoc displacement. Select three scene winners with largest scores; lexical scene/query tie break.',
         'screen_projection':'Use at most 30000 deterministically index-sampled dataset COLMAP points. GT positive depth, 96px image border excluded, at least 6 reference points in a 24px neighborhood, one first-ID point per 120px GT grid. Same XYZ and K for all predicted poses; statistic is display selection only, not query correspondence accuracy.',
         'screen_candidates':'assets/e1/screen_displacement_candidates.json','screen_geometry_sources':geometry_sources,
         'bias_disclosure':'Deliberately selected favorable examples, not random or representative outcomes; aggregate results are reported separately.',
         'all_candidates':'assets/e1/all_query_candidates.json','scene_winners':winners},
       'gt_association_max_l2':tol,'rows':[],'source_sha256':SOURCES,
       'visualization_contract':{'reference':'The exact same MAtCHA Gaussian model is used for GT and all three estimated poses within each row.',
         'real_query':'Real RGB shown only in the first column. Comparison panels are shared-reference renders, not real RGB or each method\'s own map.',
         'camera':'Same calibrated K, image size, black background, clamp to [0,1], and crop for every pose in each row.',
         'online':'Rendering is offline figure production only; AnyGSLoc online still uses only its compact sparse map.',
         'displacement':'No pose, geometry, pixel displacement, or relative alignment is exaggerated.'}}
    (ASSETS/'all_query_candidates.json').write_text(json.dumps(allrows,indent=2))
    for i,r in enumerate(winners):
        cam=ds[r['scene']].camera(r['image']);stem=f'{i+1}_{r["scene"]}'
        ply=POOL/f'stdloc_lafgs_v1_frozen_multiscene_20260731/{r["scene"]}/prior/rgb_matcha_2dgs/point_cloud/iteration_30000/point_cloud.ply'
        Image.open(cam.image_path).convert('RGB').save(ASSETS/f'{stem}_query.png')
        p['rows'].append({**r,'stem':stem,'camera':{'width':cam.width,'height':cam.height,'K':camera_K(cam).tolist(),'gt_pose_w2c':cam.pose_w2c.tolist()},
          'query_path':str(cam.image_path),'query_sha256':digest(cam.image_path),'reference_ply':str(ply),'reference_sha256':digest(ply),
          'gaussian_type':'2dgs','sh_degree':3,'methods':results[r['scene']][r['image']],'result_sources':paths[r['scene']]})
    if PROV.exists():
        old=read(PROV)
        for row in p['rows']:
            prev=next((r for r in old['rows'] if r['scene']==row['scene'] and r['image']==row['image'] and r['reference_sha256']==row['reference_sha256']),None)
            if prev:
                for key in ['renders','render_metadata','crop','crop_selection']: 
                    if key in prev:row[key]=prev[key]
    save(p);print('Selected:',[(r['scene'],r['image']) for r in p['rows']],flush=True)

def render(width=1280):
    import torch
    from gsplat import rasterization_2dgs
    from gsplat.cuda._torch_impl import _spherical_harmonics
    from anygsloc.priors.models import FrozenGaussianModel
    torch.set_num_threads(4);p=read(PROV)
    for row in p['rows']:
        h=round(width*row['camera']['height']/row['camera']['width']);cK=np.array(row['camera']['K'],dtype=np.float64);cK[:2]*=width/row['camera']['width'];K=torch.tensor(cK,dtype=torch.float32)
        poses={'GT':row['camera']['gt_pose_w2c'],**{m:row['methods'][m]['pose_w2c'] for m in METHODS}}
        if all((ASSETS/f'{row["stem"]}_{m}_render.png').exists() for m in poses) and row.get('render_metadata',{}).get('width')==width:continue
        assert digest(row['reference_ply'])==row['reference_sha256']
        model=FrozenGaussianModel(row['sh_degree'],device='cpu');model.load_ply(row['reference_ply'],loc_feature_dim=0)
        print('Loaded reference',row['scene'],len(model.get_xyz),'Gaussians',flush=True)
        scales=model.get_scaling
        if scales.shape[1]==2:scales=torch.cat([scales,torch.ones(len(scales),1)],dim=1)
        geom=dict(means=model.get_xyz.cuda(),quats=model.get_rotation.cuda(),scales=scales.cuda(),opacities=model.get_opacity[:,0].cuda())
        row['renders']={}
        with torch.inference_mode():
            for method,po in poses.items():
                dest=ASSETS/f'{row["stem"]}_{method}_render.png';pose=torch.tensor(po,dtype=torch.float32);campos=torch.linalg.inv(pose)[:3,3]
                rgb=[]
                for start in range(0,len(model.get_xyz),65536):
                    stop=start+65536;coeff=torch.cat([model._features_dc[start:stop],model._features_rest[start:stop]],dim=1)
                    rgb.append((_spherical_harmonics(row['sh_degree'],model.get_xyz[start:stop]-campos,coeff)+.5).clamp_min(0))
                rgb=torch.cat(rgb)
                torch.cuda.reset_peak_memory_stats()
                kwargs=dict(**geom,colors=rgb[None].cuda(),viewmats=pose[None].cuda(),Ks=K[None].cuda(),width=width,height=h,packed=False,backgrounds=torch.zeros(1,3,device='cuda'),near_plane=.01,far_plane=10000.,render_mode='RGB')
                result=rasterization_2dgs(**kwargs)
                arr=result[0][0].clamp(0,1).cpu().numpy();Image.fromarray(np.uint8(np.round(arr*255))).save(dest)
                alpha=result[1][0].cpu().numpy();np.save(ASSETS/f'{row["stem"]}_{method}_alpha.npy',alpha)
                row['renders'][method]={'path':str(dest),'sha256':digest(dest),'cuda_peak_own_bytes':torch.cuda.max_memory_allocated()}
                print('Rendered',row['scene'],method,'peak',round(torch.cuda.max_memory_allocated()/2**20),'MiB',flush=True)
                del kwargs,result,rgb;gc.collect();torch.cuda.empty_cache()
        row['render_metadata']={'width':width,'height':h,'K':cK.tolist(),'primitives':len(model.get_xyz),'all_primitives':True,
           'renderer':'gsplat rasterization_2dgs, CPU spherical harmonics evaluated independently at each archived camera center, black background, RGB clamped identically, 8-bit rounded PNG.',
           'camera_poses':'Archived W2C matrices, unchanged','offline_visualization_only':True,'gpu':'CUDA_VISIBLE_DEVICES=1; existing user processes left untouched'}
        save(p);del geom,model,scales;gc.collect();torch.cuda.empty_cache()

def choose_crop(gt,ests,alpha):
    """Select a facade crop for reading real pose misalignment; no image warping."""
    from scipy.ndimage import sobel,uniform_filter
    im=gt.astype(float)/255;gray=im.mean(2);gy=sobel(gray,axis=0);gx=sobel(gray,axis=1);grad=np.hypot(gx,gy)
    H,W=gray.shape;cw,ch=320,240
    rows=[]
    # Candidate windows stay off the image border and in the lower 80%.
    for y in range(int(H*.15),H-ch-16,24):
        for x in range(24,W-cw-24,24):
            g=grad[y:y+ch,x:x+cw];a=alpha[y:y+ch,x:x+cw]
            coverage=float((a>.9).mean())
            if coverage<.98:continue
            # Seams should cross real high-contrast facade structure.
            stripe=np.s_[y+ch//2-20:y+ch//2+20,x:x+cw]
            vertical_edge=float(np.abs(gx[stripe]).mean())
            texture=float(g.mean())
            # Rank shared crop by actual baseline-vs-GT discrepancy, requiring ours close.
            diffs=[np.mean(np.abs(e[y:y+ch,x:x+cw].astype(float)/255-im[y:y+ch,x:x+cw])) for e in ests]
            margin=min(diffs[:2])-diffs[2]
            score=max(margin,0)*np.sqrt(texture)*np.sqrt(vertical_edge)
            rows.append((score,x,y,coverage,texture,margin,diffs))
    assert rows
    z=max(rows,key=lambda z:(z[0],-z[2],-z[1]));score,x,y,coverage,texture,margin,diffs=z
    return [x,y,x+cw,y+ch],{'rule':'Max positive RGB-discrepancy margin min(baseline−GT)−(AnyGSLoc−GT), weighted by GT image gradient and GT seam structure, among fixed 320×240 windows with >98% alpha>0.9. Display-only favorable ROI selection; no pixel displacement is changed.',
        'score':float(score),'coverage':coverage,'gradient':texture,'raw_RGB_MAE':[float(d) for d in diffs], 'candidate_count':len(rows)}

def draw(style="bands"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox
    p=read(PROV);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'text.color':INK,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    # Main: full real query as context, half-image GT/estimate pairs in one shared detail crop.
    fig=plt.figure(figsize=(7.1,5.2));left=.026;gap=.015;w=(1-2*left-3*gap)/4;xs=[left+j*(w+gap) for j in range(4)]
    for j,title in enumerate(['Real query','STDLoc final','ULF-Loc final','AnyGSLoc']):
        fig.text(xs[j]+w/2,.972,title,ha='center',va='top',fontsize=8.4,weight='bold')
    legend_parts=[('Shared reference rendering · ',GRAY),('GT',GT),(' / ',GRAY),('Estimated pose',EST),(' · '+('top / bottom' if style=='half' else 'alternating bands'),GRAY)]
    legend=HPacker(children=[TextArea(t,textprops={'size':7,'color':c}) for t,c in legend_parts],align='baseline',pad=0,sep=0)
    fig.add_artist(AnchoredOffsetbox(loc='center',child=legend,frameon=False,bbox_to_anchor=(.5,.926),bbox_transform=fig.transFigure,pad=0,borderpad=0))
    for i,row in enumerate(p['rows']):
        ims={m:np.array(Image.open(row['renders'][m]['path']).convert('RGB')) for m in ['GT']+METHODS}
        alpha=np.load(ASSETS/f'{row["stem"]}_GT_alpha.npy').squeeze()
        crop,audit=choose_crop(ims['GT'],[ims[m] for m in METHODS],alpha);row['crop']=crop;row['crop_selection']=audit
        x0,y0,x1,y1=crop;ch=y1-y0;cw=x1-x0
        gt=ims['GT'][y0:y1,x0:x1]
        top=.887-i*.280;h=w*7.1/5.2*.75;bottom=top-h
        # Real query gets the same 4:3 outer cell; full-frame photo preserves context.
        ax=fig.add_axes([xs[0],bottom,w,h]);ax.imshow(Image.open(row['query_path']));ax.axis('off')
        native_scale=row['camera']['width']/row['render_metadata']['width'];native=np.array(crop)*native_scale
        ax.add_patch(Rectangle(native[:2],native[2]-native[0],native[3]-native[1],fill=False,ec='white',lw=.85))
        for j,m in enumerate(METHODS,1):
            est=ims[m][y0:y1,x0:x1];half=np.concatenate([gt[:ch//2],est[ch//2:]],axis=0)
            Image.fromarray(half).save(ASSETS/f'{row["stem"]}_{m}_half.png')
            # Alternative uses alternating horizontal GT/estimated bands, all exact pixels.
            bands=gt.copy()
            band_edges=np.linspace(0,ch,5).astype(int)
            for b in [1,3]:bands[band_edges[b]:band_edges[b+1]]=est[band_edges[b]:band_edges[b+1]]
            Image.fromarray(bands).save(ASSETS/f'{row["stem"]}_{m}_bands.png')
            ax=fig.add_axes([xs[j],bottom,w,h]);ax.imshow(half if style=='half' else bands);ax.set_xlim(0,cw);ax.set_ylim(ch,0);ax.axis('off')
            
            if style=='half':ax.plot([0,cw],[ch/2,ch/2],c='white',lw=.4)
            # Short outside edge bars make source identity explicit without tinting imagery.
            
            if style=='half':
                ax.plot([-3,-3],[0,ch/2],c=GT,lw=2,clip_on=False)
                ax.plot([-3,-3],[ch/2,ch],c=EST,lw=2,clip_on=False)
            else:
                for b in range(4):ax.plot([-3,-3],band_edges[b:b+2],c=GT if b%2==0 else EST,lw=2,clip_on=False)
            metric=row['methods'][m]
            fig.text(xs[j]+w/2,bottom-.018,f'{metric["TE_cm"]:.2f} cm / {metric["RE_deg"]:.3f}°',ha='center',va='top',fontsize=7.2)
        fig.text(xs[0]+w/2,bottom-.018,f'({chr(97+i)}) {row["scene"]}',ha='center',va='top',fontsize=7.3,weight='bold')
    output_name='E1_final_localization' if style=='bands' else 'E1_final_localization_half_alternative'
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{output_name}.{ext}',dpi=400,pad_inches=0)
    plt.close(fig)
    caption=('Selected illustrative favorable localization examples from three Cambridge scenes. The first column shows the real query and the displayed detail region. Each comparison panel alternates four equal horizontal bands: ground-truth-pose rendering in bands 1 and 3, and the archived final estimated-pose rendering in bands 2 and 4. Continuity of the shared scene structure across the three boundaries indicates alignment. Within each row all methods use exactly the same fixed MAtCHA Gaussian reference model, calibrated intrinsics, resolution, tone mapping and crop. Thus the renderings compare camera poses, not each localization method\'s own scene representation. Green/magenta edge bars identify GT/estimated bands; annotations give archived translation and rotation errors. AnyGSLoc uses its evaluated compact map and one localization stage; both baselines use their final refinement poses. Examples are deliberately selected where AnyGSLoc has lower translation and rotation error than both baselines, with one query per scene; among favorable candidates, common-reference screen displacement identifies readable local differences; the complete candidate lists and deterministic selection rules are archived. Detail crops favor visible alignment differences and are identical across methods. No pose or pixel displacement is magnified relative to the shared crop. Gaussian rendering is solely an offline figure-production tool and is absent from AnyGSLoc online inference. Aggregate test performance is reported separately.')
    if style!='bands':
        p['half_alternative']={'outputs':{ext:{'path':str(OUT/f'{output_name}.{ext}'),'sha256':digest(OUT/f'{output_name}.{ext}')} for ext in ['pdf','svg','png']},'same_scientific_inputs_and_ROI':True}
        save(p);return
    validations=[]
    for row in p['rows']:
        gt_pose=np.asarray(row['camera']['gt_pose_w2c']);te_error=[]
        for method in METHODS:
            record=row['methods'][method];pose=np.asarray(record['pose_w2c'])
            te=np.linalg.norm(np.linalg.inv(pose)[:3,3]-np.linalg.inv(gt_pose)[:3,3])*100
            te_error.append(abs(te-record['TE_cm']))
        assert max(te_error)<.02
        gt_render=np.array(Image.open(row['renders']['GT']['path']))
        x0,y0,x1,y1=row['crop'];gt_crop=gt_render[y0:y1,x0:x1];edges=np.linspace(0,y1-y0,5).astype(int)
        exact=[]
        for method in METHODS:
            composed=np.array(Image.open(ASSETS/f'{row["stem"]}_{method}_bands.png'))
            estimated=np.array(Image.open(row['renders'][method]['path']))[y0:y1,x0:x1]
            for band in range(4):
                sl=slice(edges[band],edges[band+1]);exact.append(np.array_equal(composed[sl],(gt_crop if band%2==0 else estimated)[sl]))
        assert all(exact)
        validations.append({'scene':row['scene'],'archived_TE_recomputed_max_error_cm':max(te_error),'all_bands_are_exact_unwarped_source_pixels':all(exact)})
    p['validation']=validations
    p['display']={'primary':'Four alternating horizontal GT/estimated bands with source-colored edge bars; no artificial seam lines.','bands':4,'GT_bands_1indexed':[1,3],'estimated_bands_1indexed':[2,4],'figure_inches':[7.1,5.2],'minimum_ordinary_font_pt':7,'reference_render_resolution':[1920,1080]}
    p['source_script']={'path':str(Path(__file__).resolve()),'sha256':digest(__file__)}
    p['caption']=caption;p['outputs']={ext:{'path':str(OUT/f'E1_final_localization.{ext}'),'sha256':digest(OUT/f'E1_final_localization.{ext}')} for ext in ['pdf','svg','png']}
    p['limitations']=['These are selected favorable examples; they do not imply uniform superiority or representative test accuracy.',
      'The GT bands are Gaussian renderings at the reference pose, not a photometric ground-truth image. The real query is shown separately.',
      'Small pose errors may produce subtle seam differences depending on scene depth; raw TE/RE remain the quantitative reference.',
      'View-dependent SH appearance is reevaluated for each actual pose; no color matching, optical-flow warp, edge displacement or pose exaggeration is applied.',
      'Both GT and all estimated renderings inherit the same reference-model geometric and appearance limitations.']
    save(p);(OUT/'E1_final_localization.caption.txt').write_text(caption+'\n\nLimitations:\n'+'\n'.join('- '+x for x in p['limitations'])+'\n')
    print('Wrote E1 shared-reference alignment figure',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','render','draw']);ap.add_argument('--width',type=int,default=1920);ap.add_argument('--style',choices=['half','bands'],default='bands');args=ap.parse_args()
    {'prepare':prepare,'render':lambda:render(args.width),'draw':lambda:draw(args.style)}[args.mode]()
