"""Build the camera-ready-format overview from current maps and real renders.

Rendering / one-query localization are for figure materialization only. No map,
method configuration, or experiment result is changed. SVG is the master file.
"""
from pathlib import Path
import sys, json, hashlib, base64, io, html
import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'paper/figures_publication'
ASSETS=OUT/'assets'
POOL=Path('/mnt/pool/sqy')
OLD=POOL/'lafgs_v24_cambridge_high_capacity_online_20260901/ShopFacade'
FACTOR=POOL/'anygsloc_observation_factorial_20260914/maps/ShopFacade_v1_f1_c1'
PRIOR=POOL/'stdloc_lafgs_v1_frozen_multiscene_20260731/ShopFacade/prior/rgb_matcha_2dgs/point_cloud/iteration_30000/point_cloud.ply'


def materialize():
    from anygsloc.data.datasets import ColmapDataset
    from anygsloc.priors.models import GaussianModel2D
    from anygsloc.priors.rendering import render_from_pose_gsplat
    from anygsloc.online.runtime import Localizer
    from anygsloc.data.calibration import resolve_scene_calibration
    from anygsloc.data.geometric_validity import SameKRectificationSupport
    torch.set_num_threads(4)
    ASSETS.mkdir(parents=True,exist_ok=True)
    if (ASSETS/'materialized.json').exists():return json.loads((ASSETS/'materialized.json').read_text())
    print('Loading current source map and observation cache',flush=True)
    state=torch.load(OLD/'projective_map/projective_anchor_map.pt',map_location='cpu',weights_only=False)
    cache=torch.load(OLD/'observations/rendered_rgb_feature_cache.pt',map_location='cpu',weights_only=False)
    records=cache['queries'];names=state['v6_mapping_query_names']
    obs=state['projective_anchor_observations'];offset=obs['observation_offsets']
    counts=offset[1:]-offset[:-1]
    # Highest-support track; index tie broken by the stored order. No test error selection.
    track=int(counts.argmax());a,b=int(offset[track]),int(offset[track+1])
    entries=sorted(zip(obs['query_indices'][a:b].tolist(),obs['keypoint_indices'][a:b].tolist()))
    entries=[entries[i] for i in [0,len(entries)//2,len(entries)-1]]
    data=ColmapDataset(str(POOL/'Cambridge_stdloc/ShopFacade'),load_masks=False)
    model=GaussianModel2D(3);model.load_ply(str(PRIOR),loc_feature_dim=0);model=model.cuda().eval()
    views=[]
    with torch.inference_mode():
        for i,(qi,ki) in enumerate(entries):
            name=names[qi];cam=data.camera(name);record=records[name]
            assert name in {c.image_name for c in data.split('mapping')}
            assert np.allclose(record['pose_w2c'].numpy(),cam.pose_w2c,atol=1e-5)
            package=render_from_pose_gsplat(model,record['pose_w2c'].cuda().float(),cam.fov_x,cam.fov_y,cam.width,cam.height,
                bg_color=torch.zeros(3,device='cuda'),render_mode='RGB+ED',rgb_only=True,rasterize_mode='antialiased')
            print('Render keys',list(package),flush=True)
            rgb=package['render'].detach().cpu().clamp(0,1).numpy()
            if rgb.shape[0]==3:rgb=rgb.transpose(1,2,0)
            image=Image.fromarray((rgb*255).round().astype(np.uint8));image.save(ASSETS/f'render_{i}.png')
            uv=record['native_keypoints'][ki].numpy();x,y=map(float,uv)
            half=64;box=(round(x)-half,round(y)-half,round(x)+half,round(y)+half)
            assert box[0]>=0 and box[1]>=0 and box[2]<=cam.width and box[3]<=cam.height, 'track crop crosses image boundary'
            image.crop(box).save(ASSETS/f'patch_{i}.png')
            views.append({'name':name,'keypoint_index':ki,'xy':uv.tolist(),'crop':box,'pose_w2c':cam.pose_w2c.tolist()})
    del model,cache
    torch.cuda.empty_cache()
    query='seq1/frame00024.png';cam=data.camera(query)
    Image.open(cam.image_path).convert('RGB').save(ASSETS/'query.png')
    cal=resolve_scene_calibration(data.split('mapping'))
    engine=Localizer(FACTOR/'compact/deployment_map.pt',keypoint_count=cal['keypoint_count'],reprojection_error_px=cal['reprojection_error_px'],mask_policy='geometric_validity',device='cuda:0',seed=2026)
    with torch.inference_mode():
        result=engine.localize(data.load_image(cam),fov_x=cam.fov_x,fov_y=cam.fov_y,valid_mask=SameKRectificationSupport(data)(cam))
    matches=result['matches'];xy=result['keypoints'][matches.keypoint_indices].cpu().numpy()
    xyz=engine.anchor_xyz[matches.anchor_indices].cpu().numpy()
    inliers=np.asarray(result['pose'].inliers,dtype=int)
    # One inlier per 4x3 query grid cell, in original matching order.
    chosen=[];occupied=set()
    for idx in inliers:
        cell=(min(3,int(xy[idx,0]/cam.width*4)),min(2,int(xy[idx,1]/cam.height*3)))
        if cell not in occupied:occupied.add(cell);chosen.append(int(idx))
    arrays={};map_counts={}
    for selection in ['full','compact']:
        mp=torch.load(FACTOR/selection/'deployment_map.pt',map_location='cpu',weights_only=True)
        map_counts[selection]=len(mp['anchor_xyz'])
        points=mp['anchor_xyz'].numpy();pose=cam.pose_w2c
        q=points@pose[:3,:3].T+pose[:3,3]
        k=result['intrinsic'];uv=q@k.T;uv=uv[:,:2]/np.maximum(uv[:,2:3],1e-8)
        good=(q[:,2]>0)&(uv[:,0]>0)&(uv[:,0]<cam.width)&(uv[:,1]>0)&(uv[:,1]<cam.height)
        arrays[selection+'_uv']=uv[good]/[cam.width,cam.height]
        arrays[selection+'_depth']=q[good,2]
    arrays['match_xy']=xy[chosen]/[cam.width,cam.height]
    p=result['pose'].pose_w2c
    if hasattr(p,'cpu'):p=p.cpu().numpy()
    qp=xyz[chosen]@np.asarray(p)[:3,:3].T+np.asarray(p)[:3,3]
    uv=qp@result['intrinsic'].T;arrays['match_map_uv']=(uv[:,:2]/uv[:,2:3])/[cam.width,cam.height]
    np.savez_compressed(ASSETS/'points.npz',**arrays)
    meta={'scene':'ShopFacade','track_id':int(state['anchor_ids'][track]),'track_support':int(counts[track]),'track_selection':'highest support, three observations at sorted-index endpoints and midpoint',
          'views':views,'query':query,'query_selection':'fixed median-TE example from previous audited figure suite',
          'match_selection':'one solver inlier per 4x3 image grid cell, original matching order','match_count':len(chosen),
          'full_count':map_counts['full'],'compact_count':map_counts['compact'],'point_projection':'GT query camera; no occlusion test; illustration only',
          'prior':str(PRIOR),'source_map':str(OLD/'projective_map/projective_anchor_map.pt'),
          'cache':str(OLD/'observations/rendered_rgb_feature_cache.pt'),
          'rendering':'Original mapping resolution, black background, gsplat RGB+ED; actual Gaussian RGB, not captured mapping images',
          'schematic_elements':['multi-ray geometry diagram','descriptor fusion glyph','camera frustum'],
          'counts_scope':'ShopFacade, not five-scene aggregate'}
    (ASSETS/'materialized.json').write_text(json.dumps(meta,indent=2));return meta


INK='#243349';MUTED='#5d6c7c';BLUE='#336f9c';TEAL='#187b70';GOLD='#b87b38';LIGHT='#dce4ea'
S=[]
def add(s):S.append(s)
def text(x,y,s,size=22,color=INK,weight=400,anchor='start'):
    add(f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" font-family="DejaVu Sans, sans-serif">{html.escape(s)}</text>')
def line(x1,y1,x2,y2,col=LIGHT,width=1.5,dash='',arrow=False):
    add(f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{col}" stroke-width="{width}"'+(f' stroke-dasharray="{dash}"' if dash else '')+(' marker-end="url(#arrow)"' if arrow else '')+'/>')
def rect(x,y,w,h,fill='white',stroke=LIGHT,r=8):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}"/>')
def image(name,x,y,w,h):
    # Keep native source assets intact; embed only the resolution needed at >500 dpi.
    bitmap=Image.open(ASSETS/name).convert('RGB')
    bitmap.thumbnail((round(w*2),round(h*2)),Image.Resampling.LANCZOS)
    buffer=io.BytesIO();bitmap.save(buffer,format='PNG')
    uri=base64.b64encode(buffer.getvalue()).decode()
    add(f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid meet" href="data:image/png;base64,{uri}"/>')
def circle(x,y,r,col,fill=None):
    add(f'<circle cx="{x}" cy="{y}" r="{r}" stroke="{col}" fill="{fill or col}"/>')
def cloud(points,x,y,w,h,col,size=.95):
    for a,b in points:circle(round(x+a*w,2),round(y+b*h,2),size,col)
def camera(x,y,col=BLUE):
    add(f'<path d="M{x-14},{y+13} L{x-14},{y-13} L{x+7},{y-13} L{x+7},{y+13} Z M{x+7},{y-8} L{x+20},{y-15} L{x+20},{y+15} L{x+7},{y+8}" fill="white" stroke="{col}" stroke-width="2"/>')


def draw(meta):
    points=np.load(ASSETS/'points.npz')
    S.clear();add('<svg xmlns="http://www.w3.org/2000/svg" width="180mm" height="108mm" viewBox="0 0 1800 1080">')
    add('<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10" fill="#738393"/></marker></defs>')
    rect(0,0,1800,1080,'white','white',0)
    text(35,40,'OFFLINE',21,TEAL,700);text(170,40,'Construct a compact map for sparse localization',27,INK,600)
    text(35,78,'Gaussian prior + calibrated mapping cameras',21,MUTED)
    # Three quiet panels; content rather than nested boxes carries the hierarchy.
    for x,w,label,title,col in [(35,510,'a','Validate observations',BLUE),(575,520,'b','Reconstruct landmarks',GOLD),(1125,640,'c','Retain complementary constraints',TEAL)]:
        rect(x,104,w,486,'#fbfcfd',LIGHT,12)
        circle(x+29,137,15,col);text(x+29,145,label,22,'white',600,'middle')
        text(x+58,145,title,24,INK,600)
    # Actual Gaussian images stacked, one track highlighted in every patch.
    for i,(x,y) in enumerate([(61,183),(147,204),(233,225)]):
        rect(x-3,y-3,276,157,'white',LIGHT,3);image(f'render_{i}.png',x,y,270,151)
    text(62,408,'Gaussian-rendered observations',21,BLUE,600)
    text(62,441,'Observation validity screening',22)
    text(62,473,'Mutual matching • epipolar checks',20,MUTED)
    text(62,503,'Cycle-consistent multi-view tracks',20,MUTED)
    text(62,557,'RGB + native validity cues • fixed cameras',18,MUTED)
    # Real patches from one shared track, not arbitrary repeated textures.
    for i,x in enumerate([608,768,928]):
        image(f'patch_{i}.png',x,186,126,126);rect(x,186,126,126,'none',LIGHT,0)
        line(x+54,249,x+72,249,GOLD,2);line(x+63,240,x+63,258,GOLD,2)
        text(x+63,337,f'View {i+1}',18,MUTED,anchor='middle')
    # Geometric principle is explicitly schematic; measurements are the patches above.
    for x,y in [(652,438),(820,461),(990,428)]:
        camera(x,y,BLUE);line(x+2,y-16,822,374,BLUE,1.8)
    circle(822,374,6,GOLD);text(845,380,'X',22,GOLD,600)
    text(606,503,'Fixed-camera multi-ray triangulation',21,INK,600)
    text(606,533,'Robust multi-view descriptor fusion',21,INK)
    text(606,563,'Real track patches above; ray layout schematic',17,MUTED)
    line(549,352,571,352,width=2,arrow=True)
    line(1099,352,1121,352,width=2,arrow=True)
    # Actual map projection with uniform-size points, same camera/framing.
    cloud(points['full_uv'],1151,195,268,173,'#8097aa',.65)
    cloud(points['compact_uv'],1468,195,268,173,TEAL,.9)
    line(1426,282,1457,282,width=2,arrow=True)
    text(1285,403,f'{meta["full_count"]:,} candidates',22,INK,600,'middle')
    text(1602,403,f'{meta["compact_count"]:,} retained',22,TEAL,600,'middle')
    text(1153,444,'Matching coverage',22,INK,600)
    text(1153,476,'+ complementary pose information',21,INK)
    text(1153,516,'One 3D point + one 256-D descriptor',21,TEAL,600)
    text(1153,558,'Actual ShopFacade maps • same projection',18,MUTED)
    # Deployment border makes the offline/online dependency explicit.
    line(35,630,1325,630,LIGHT,1.5,'6 6')
    line(1648,630,1765,630,LIGHT,1.5,'6 6')
    rect(1340,606,291,47,'#edf5f3','#bed7d1',7)
    text(1486,637,'Deploy only the sparse map',19,TEAL,600,'middle')
    line(1486,592,1486,604,TEAL,2)
    add('<path d="M1486,654 L1486,683 L1050,683 L1050,735" fill="none" stroke="#187b70" stroke-width="2" marker-end="url(#arrow)"/>')
    text(35,685,'ONLINE',21,TEAL,700);text(170,685,'A single sparse localization stage',27,INK,600)
    # Query and correspondences are materialized by the frozen localizer.
    image('query.png',38,734,343,343*9/16)
    for x,y in points['match_xy']:circle(38+x*343,734+y*343*9/16,3.0,GOLD,'none')
    text(38,977,'Real query → SuperPoint',23,INK,600)
    line(401,829,452,829,width=2,arrow=True)
    image('query.png',477,749,235,235*9/16)
    cloud(points['compact_uv'],903,749,235,235*9/16,'#bccdc9',.55)
    colors=[BLUE,TEAL,GOLD,'#886f9e']
    for i,(uv,q) in enumerate(zip(points['match_xy'],points['match_map_uv'])):
        col=colors[i%4];a=477+uv[0]*235;b=749+uv[1]*235*9/16;c=903+q[0]*235;d=749+q[1]*235*9/16
        line(a,b,c,d,col,1.1);circle(a,b,2.5,col);circle(c,d,2.5,col)
    text(590,924,'2D query features',18,MUTED,anchor='middle');text(1020,924,'3D landmarks',18,MUTED,anchor='middle')
    text(805,977,'Exact matching + confidence filtering',23,INK,600,'middle')
    line(1164,829,1218,829,width=2,arrow=True)
    # Camera glyph: symbolic output, not a measured 3D error plot.
    camera(1330,821,TEAL)
    for dx,dy,col in [(85,0,GOLD),(-26,57,TEAL),(-26,-48,BLUE)]:
        line(1373,821,1373+dx,821+dy,col,2)
    text(1535,821,'R, t',36,TEAL,600,'middle')
    text(1470,872,'one PoseLib solve',23,INK,anchor='middle')
    text(1470,977,'Robust pose estimation',23,INK,600,'middle')
    text(900,1042,'No online scene rendering  ·  No additional dense refinement  ·  No query semantic model',22,TEAL,500,'middle')
    add('</svg>')
    path=OUT/'AnyGSLoc_method_overview.svg';path.write_text('\n'.join(S))
    sys.path.insert(0,'/root/.cache/anygsloc_figure_deps')
    import cairosvg
    cairosvg.svg2pdf(url=str(path),write_to=str(path.with_suffix('.pdf')))
    cairosvg.svg2png(url=str(path),write_to=str(path.with_suffix('.png')),output_width=3000,output_height=1800)
    cairosvg.svg2png(url=str(path),write_to=str(OUT/'overview_print_size.png'),output_width=1417,output_height=850)
    # Verify and archive the actual asset bytes used by the SVG.
    paths=[PRIOR,OLD/'projective_map/projective_anchor_map.pt',FACTOR/'full/deployment_map.pt',FACTOR/'compact/deployment_map.pt',Path(__file__)]
    paths+=list(ASSETS.glob('*'))
    manifest=dict(meta);manifest['sources']={}
    for p in paths:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        manifest['sources'][str(p)]={'sha256':h.hexdigest(),'bytes':p.stat().st_size}
    manifest['matching_visualization']='A spatially distributed subset of solver inliers, not all matches; pose-projected right endpoints, not independent GT validation.'
    manifest['status']='submission-format figure; author scientific/caption review still required'
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(path,flush=True)


if __name__=='__main__':draw(materialize())
