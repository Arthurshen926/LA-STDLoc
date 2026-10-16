"""Select auditable figure samples (CPU mmap) and replay RGB/buffer renders (GPU).

Selection: i3dgs Python (torch >= 2.1 for mmap). Rendering: frozen g4splat Python.
Neither phase mutates maps or changes any experiment results.
"""
from pathlib import Path
import sys,json,argparse,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from figure_cache_reader import read_cache
POOL=Path('/mnt/pool/sqy');OUT=ROOT/'paper/figures_v5';ASSETS=OUT/'assets'
SCENES=['GreatCourt','StMarysChurch','OldHospital','KingsCollege','ShopFacade']

def load(p):return torch.load(p,map_location='cpu',weights_only=False,mmap=True)
def cache_root(scene):
    return POOL/('anygsloc_cache_recovery_v2_20260912' if scene in SCENES[:2] else 'lafgs_v24_cambridge_high_capacity_online_20260901')/scene
def source_root(scene):
    return POOL/('anygsloc_missing_cache_downstream_20260912' if scene in SCENES[:2] else 'lafgs_v24_cambridge_high_capacity_online_20260901')/scene
def project(xyz,record):
    p=np.asarray(record['pose_w2c']);k=np.asarray(record['native_K']);q=np.asarray(xyz)@p[:3,:3].T+p[:3,3]
    uv=q@k.T;return uv[...,:2]/np.maximum(uv[...,2:],1e-8)
def serial_record(record):
    return {k:record[k].numpy().tolist() for k in ['native_K','pose_w2c','native_input_hw']}

def select():
    torch.set_num_threads(4);ASSETS.mkdir(parents=True,exist_ok=True)
    manifest={'selection_policy':{'validity':'Within each scene, 90th-percentile mapping-view rejection rate; crop maximizes balanced accepted/rejected support, no test error used.',
        'geometry':'GreatCourt and StMarysChurch: 75th-percentile center/depth-vs-ray displacement among supported common anchors in the selected mapping view; three recorded observations.',
        'limitations':'Illustrative high-contrast samples, not an unbiased estimate of average benefit. Local projections do not supply independent 3D ground truth.'},'views':[],'geometry':[],'audit_summary':[]}
    for scene in SCENES:
        print('select',scene,flush=True)
        root=cache_root(scene);records=[];sources=[]
        for path in sorted((root/'v2_audit').glob('*.pt')):
            d=load(path);records+=d['records'];sources.append(str(path));binding=d['input'];thresholds=d['thresholds']
        ranked=sorted(records,key=lambda r:(float((~r['row_valid']).float().mean()),r['query_name']))
        pick=ranked[round(.9*(len(ranked)-1))]
        cache=read_cache(root/'observations/rendered_rgb_feature_cache.pt');record=cache['queries'][pick['query_name']]
        xy=record['native_keypoints'].numpy();keep=pick['row_valid'].numpy();h,w=map(int,record['native_input_hw'])
        # Fixed-size, interior sliding windows. Max-min counts exposes the decision boundary.
        cw,ch=min(640,w//2),min(480,h//2);best=None
        for y in range(round(h*.05),h-ch-round(h*.05)+1,max(1,ch//4)):
            for x in range(round(w*.05),w-cw-round(w*.05)+1,max(1,cw//4)):
                inside=(xy[:,0]>=x)&(xy[:,0]<x+cw)&(xy[:,1]>=y)&(xy[:,1]<y+ch)
                a=int((inside&keep).sum());b=int((inside&~keep).sum());score=min(a,b)
                if best is None or score>best[0]:best=(score,[x,y,x+cw,y+ch],inside,a,b)
        stem=scene+'_validity';arrays={'xy':xy.copy(),'keep':keep.copy()}
        arrays.update({k:v.numpy().copy() for k,v in pick['row_reasons'].items()})
        np.savez_compressed(ASSETS/(stem+'.npz'),**arrays)
        item=dict(scene=scene,stem=stem,name=pick['query_name'],prior=binding['gaussian_ply'],
            prior_sha256=binding['gaussian_ply_sha256'],thresholds=thresholds,crop=best[1],
            crop_accepted=best[3],crop_rejected=best[4],audit_sources=sources,
            rejected_fraction=float((~keep).mean()),**serial_record(record))
        manifest['views'].append(item)
        manifest['audit_summary'].append(dict(scene=scene,mapping_views=len(records),
            rejection_median=float(np.median([float((~r['row_valid']).float().mean()) for r in records])),
            selected_view=item['name'],selected_rejection=item['rejected_fraction']))
        if scene not in SCENES[:2]:continue
        geo={g:load(POOL/f'anygsloc_center_control_20260915/maps/{scene}/{g}/deployment_map.pt') for g in ['center','depth','ray']}
        for g in ['center','depth']:
            assert torch.equal(geo[g]['anchor_ids'],geo['ray']['anchor_ids'])
            assert torch.equal(geo[g]['anchor_features'],geo['ray']['anchor_features'])
        # Subsetting renumbers anchor_ids. Use the common-map CSR directly,
        # never join its IDs to the unrelated full-map integer namespace.
        src=load(POOL/f'anygsloc_center_control_20260915/maps/{scene}/ray/projective_anchor_map.pt')
        assert torch.equal(src['anchor_xyz'],geo['ray']['anchor_xyz'])
        names=src['v6_mapping_query_names'];qi=names.index(item['name']);obs=src['projective_anchor_observations'];off=obs['observation_offsets']
        source_rows=torch.searchsorted(off[1:],torch.where(obs['query_indices']==qi)[0],right=True).unique().numpy()
        id_to_common={int(v):i for i,v in enumerate(geo['ray']['anchor_ids'])}
        candidates=[]
        for row in source_rows:
            anchor=int(src['anchor_ids'][row]);common=id_to_common.get(anchor)
            if common is None or int(off[row+1]-off[row])<6:continue
            pts={g:project(geo[g]['anchor_xyz'][common].numpy(),record) for g in geo}
            u=pts['ray']
            if not (100<u[0]<w-100 and 100<u[1]<h-100):continue
            delta=max(float(np.linalg.norm(pts[g]-u)) for g in ['center','depth'])
            if 1<delta<70:candidates.append((delta,int(row),common,anchor))
        if not candidates:raise ValueError('No valid geometry candidates: '+scene)
        candidates.sort();delta,row,common,anchor=candidates[round(.75*(len(candidates)-1))]
        a,b=int(off[row]),int(off[row+1]);entries=sorted(zip(obs['query_indices'][a:b].tolist(),obs['keypoint_indices'][a:b].tolist()))
        eligible=[]
        for q,kp in entries:
            rec=cache['queries'][names[q]];uv=rec['native_keypoints'].numpy()[kp];hh,ww=map(int,rec['native_input_hw'])
            if 100<uv[0]<ww-100 and 100<uv[1]<hh-100:eligible.append((q,kp))
        assert len(eligible)>=3
        ids=[0,len(eligible)//2,len(eligible)-1];views=[]
        for j,idx in enumerate(ids):
            q,kp=eligible[idx];rec=cache['queries'][names[q]];uv=rec['native_keypoints'].numpy()[kp]
            stem=f'{scene}_geometry_{j}';crop=[int(round(uv[0]))-96,int(round(uv[1]))-96,int(round(uv[0]))+96,int(round(uv[1]))+96]
            predictions={g:project(geo[g]['anchor_xyz'][common].numpy(),rec).tolist() for g in geo}
            vi=dict(scene=scene,stem=stem,name=names[q],prior=item['prior'],prior_sha256=item['prior_sha256'],crop=crop,
                observed_xy=uv.tolist(),predicted_xy=predictions,**serial_record(rec))
            manifest['views'].append(vi);views.append(stem)
        manifest['geometry'].append(dict(scene=scene,anchor_id=anchor,common_index=common,source_row=row,
            support=b-a,ranking_displacement_px=delta,candidate_count=len(candidates),views=views,
            xyz={g:geo[g]['anchor_xyz'][common].numpy().tolist() for g in geo}))
    (OUT/'selection.json').write_text(json.dumps(manifest,indent=2))


def render():
    from PIL import Image
    from anygsloc.priors.models import GaussianModel2D
    from anygsloc.priors.rendering import render_from_pose_gsplat
    from anygsloc.mapping.render_certificate import CertificateThresholds,certify_v7_render,extreme_distortion_row_mask,rgb_structure_support_mask
    from anygsloc.mapping.render_certificate import _sample_rows, _depth_discontinuity, _border_mask
    torch.set_num_threads(4)
    m=json.loads((OUT/'selection.json').read_text());prior=None;model=None;replay=[]
    for v in m['views']:
        target=ASSETS/(v['stem']+'_rgb.png')
        if target.exists() and (ASSETS/(v['stem']+'_replay.json')).exists() and ('thresholds' not in v or (ASSETS/(v['stem']+'_mask.npz')).exists()):continue
        if v['prior']!=prior:
            if model is not None:del model;torch.cuda.empty_cache()
            prior=v['prior'];h=hashlib.sha256()
            with open(prior,'rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            assert h.hexdigest()==v['prior_sha256'],'prior content differs from archived audit'
            model=GaussianModel2D(3);model.load_ply(prior,loc_feature_dim=0);model=model.cuda().eval()
        h,w=v['native_input_hw'];k=np.array(v['native_K']);pose=torch.tensor(v['pose_w2c'],device='cuda',dtype=torch.float32)
        print('render',v['scene'],v['name'],flush=True)
        with torch.inference_mode():
            pkg=render_from_pose_gsplat(model,pose,2*np.arctan(w/(2*k[0,0])),2*np.arctan(h/(2*k[1,1])),w,h,
                bg_color=torch.zeros(3,device='cuda'),render_mode='RGB+ED',rgb_only=True,rasterize_mode='antialiased')
            rgb=pkg['render'][:3].clamp(0,1)
            image=Image.fromarray((rgb.cpu().numpy().transpose(1,2,0)*255).round().astype(np.uint8));image.save(target)
            image.crop(v['crop']).save(ASSETS/(v['stem']+'_crop.png'))
            record={'scene':v['scene'],'name':v['name'],'prior_sha256':v['prior_sha256']}
            if 'thresholds' in v:
                arr=np.load(ASSETS/(v['stem']+'.npz'));xy=torch.tensor(arr['xy'],device='cuda');th=CertificateThresholds(**v['thresholds'])
                distortion=extreme_distortion_row_mask(pkg['rend_dist'],xy,mad_multiplier=th.distortion_mad_multiplier,tail_quantile=th.distortion_tail_quantile)
                cert=certify_v7_render(rgb=rgb,alpha=pkg['rend_alpha'],depth=pkg['depth'],keypoints=xy,
                    nearest_mapping_distance_m=0.,median_adjacent_baseline_m=1.,source_family_support=1,artifact_row_mask=distortion,thresholds=th)
                structure,score=rgb_structure_support_mask(rgb,support_threshold=th.rgb_structure_support_threshold,
                    window_reference_px=th.rgb_structure_window_reference_px,dilate_reference_px=th.rgb_structure_dilate_reference_px,
                    reference_short_side=th.rgb_structure_reference_short_side)
                record['archived_row_decision_agreement']=float(np.mean(cert['row_valid'].cpu().numpy()==arr['keep']))
                x0,y0,x1,y1=v['crop']
                # Evaluate the real predicate at native full-frame resolution before
                # cropping. This is not an interpolation of sparse keypoint labels.
                dep=pkg['depth'].squeeze();alp=pkg['rend_alpha'].squeeze()
                value=pkg['rend_dist'].squeeze().float().abs();finite=torch.isfinite(value);samples=value[finite]
                if samples.numel():
                    median=samples.median();mad=(samples-median).abs().median().clamp_min(1e-8)
                    threshold=torch.maximum(median+th.distortion_mad_multiplier*mad,torch.quantile(samples,th.distortion_tail_quantile))
                    extreme=(~finite)|(value>threshold)
                else:extreme=torch.ones_like(finite)
                assert torch.equal(_sample_rows(extreme,xy),distortion)
                support=torch.isfinite(dep)&(dep>0)&torch.isfinite(alp)&(alp>=th.alpha_minimum)
                edges=_depth_discontinuity(dep,relative=th.depth_discontinuity_relative)
                border=_border_mask(h,w,th.border_fraction,dep.device)
                valid=support&~edges&~border&~extreme&structure
                assert torch.equal(_sample_rows(valid,xy),cert['row_valid'])
                assert np.array_equal(_sample_rows(valid,xy).cpu().numpy(),arr['keep'])
                masks={'valid':valid,'support':support,'depth_continuity':~edges,'structure':structure,'distortion_ok':~extreme,'border_ok':~border}
                np.savez_compressed(ASSETS/(v['stem']+'_mask.npz'),**{key:val.cpu().numpy()[y0:y1,x0:x1] for key,val in masks.items()})
                Image.fromarray(valid.cpu().numpy().astype(np.uint8)*255).save(ASSETS/(v['stem']+'_mask_full.png'))
                record['dense_predicate_matches_all_archived_keypoints']=True
                np.savez_compressed(ASSETS/(v['stem']+'_buffers.npz'),structure=score.cpu().numpy()[y0:y1,x0:x1],
                    depth=pkg['depth'].squeeze().cpu().numpy()[y0:y1,x0:x1],alpha=pkg['rend_alpha'].squeeze().cpu().numpy()[y0:y1,x0:x1])
                # Real projected Gaussian ellipses: one nearest center per 24 px cell.
                # This is a first-order projection of primitive covariance, not alpha compositing.
                centers=model.get_xyz@pose[:3,:3].T+pose[:3,3]
                kk=torch.tensor(k,device='cuda',dtype=torch.float32)
                uv=centers@kk.T;uv=uv[:,:2]/uv[:,2:].clamp_min(1e-8)
                inside=(centers[:,2]>0)&(uv[:,0]>x0)&(uv[:,0]<x1)&(uv[:,1]>y0)&(uv[:,1]<y1)
                ids=torch.where(inside)[0];ordered=ids[torch.argsort(centers[ids,2])].cpu().numpy()
                coords=uv.cpu().numpy();chosen=[];seen=set()
                for idx in ordered:
                    cell=(int((coords[idx,0]-x0)//24),int((coords[idx,1]-y0)//24))
                    if cell not in seen:seen.add(cell);chosen.append(int(idx))
                qi=model.get_rotation[chosen].cpu().numpy();s=model.get_scaling[chosen].cpu().numpy()
                xyz=centers[chosen].cpu().numpy();world_r=pose[:3,:3].cpu().numpy();covs=[]
                for q,scale,xyz_i in zip(qi,s,xyz):
                    qw,qx,qy,qz=q/np.linalg.norm(q)
                    rr=np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw)],
                        [2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw)],
                        [2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy)]])
                    ss=np.r_[scale,1e-8] if len(scale)==2 else scale
                    cov=world_r@rr@np.diag(ss**2)@rr.T@world_r.T;x,y,z=xyz_i
                    jac=np.array([[k[0,0]/z,0,-k[0,0]*x/z**2],[0,k[1,1]/z,-k[1,1]*y/z**2]])
                    covs.append(jac@cov@jac.T)
                np.savez_compressed(ASSETS/(v['stem']+'_gaussians.npz'),uv=coords[chosen]-[x0,y0],cov=np.array(covs),primitive_ids=chosen)
            (ASSETS/(v['stem']+'_replay.json')).write_text(json.dumps(record,indent=2));replay.append(record)
    print('Rendered assets complete',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('phase',choices=['select','render']);args=a.parse_args()
    select() if args.phase=='select' else render()
