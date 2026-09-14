"""Mapping-render-only, bounded featuremetric observation correction pilot.

Not independent score-peak refinement; targets exclude the current view.
Not Pixel-Perfect SfM: uses frozen stride-8 SuperPoint fields and a single
consensus step, followed by fixed-camera triangulation and source-view guards.
"""
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
import torch
from common.hashing import sha256_file
from data.datasets import ColmapDataset
from evidence.geometry_crossfit import crossfit_geometry
from evidence.multiview_keypoint_adjustment import adjust_keypoints,spatial_prediction_improves
from evidence.v8_fixed_descriptor_replay import prepare_replay
from evidence.tracks import fuse_projective_anchor_observations
from features.extractor import FeatureExtractor
from features.superpoint import sample_descriptors
from features.raster_sampling import sample_raster_at_grid_uv
from priors.models import GaussianModel2D
from priors.rendering import render_from_pose_gsplat
from map_learning.v24_anchor_view_support import build_anchor_view_support
from topology.v6_anchor_map import identity_metric_state
from scripts.materialize_mapping_rgb_descriptors import resolve_attested_mapping_cameras


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['map','cache','association','geometry','prior','dataset','output']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(4)
    state=torch.load(args.map,map_location='cpu',weights_only=False)
    cache=torch.load(args.cache,map_location='cpu',weights_only=False)
    graph=torch.load(args.association,map_location='cpu',weights_only=False)
    audit=json.loads(args.geometry.read_text())
    for payload in [state['provenance'],cache,graph,audit]:
        if any(payload.get(k) is not False for k in ['uses_source_mapping_rgb','uses_test_queries']):raise ValueError('render mapping-only required')
    map_sha,cache_sha,graph_sha=sha256_file(args.map),sha256_file(args.cache),sha256_file(args.association)
    if audit['input']['map_sha256']!=map_sha or audit['input']['cache_sha256']!=cache_sha or state['provenance']['v8_observation_cache_sha256']!=cache_sha or state['provenance']['v8_association_graph_sha256']!=graph_sha:raise ValueError('lineage differs')
    if cache['configuration']['gaussian_type']!='2dgs' or cache['configuration']['background']!='black' or cache['configuration']['mapping_render_sampling']['supersampling_factor']!=1:raise ValueError('unsupported render contract')
    replay=prepare_replay(state,cache,graph)
    rec=list(cache['queries'].values());csr=state['projective_anchor_observations']
    cameras,registry=resolve_attested_mapping_cameras(ColmapDataset(args.dataset,images='processed').split('mapping'),cache,list(cache['queries']))
    offsets,q,k=[np.asarray(csr[x]) for x in ['observation_offsets','query_indices','keypoint_indices']]
    K=np.stack([np.asarray(r['native_K'],float) for r in rec]);T=np.stack([np.asarray(r['pose_w2c'],float) for r in rec])
    targets=[r['anchor_row'] for r in audit['rows'] if r['status']=='unstable' and 7<=r['full_observation_count']<=19 and r['candidate_kind']=='v2_projective_track']
    by_view=defaultdict(list);data={}
    for a in targets:
        s,e=offsets[a:a+2];qs,ks=q[s:e],k[s:e]
        desc=torch.stack([rec[i]['native_descriptors'][j] for i,j in zip(qs,ks)]).float()
        uv=torch.stack([rec[i]['native_keypoints'][j] for i,j in zip(qs,ks)]).float()
        bins=replay['bins'][s:e];weights=replay['weights'][s:e]
        baseline=fuse_projective_anchor_observations(desc,bins,detector_weight=weights,trim_fraction=.2)
        if not torch.equal(baseline,state['anchor_features'][a]):raise ValueError('target source fusion not exact')
        templates=[]
        for j in range(len(qs)):
            other=torch.arange(len(qs))!=j
            templates.append(fuse_projective_anchor_observations(desc[other],bins[other],detector_weight=weights[other],trim_fraction=.2))
            by_view[int(qs[j])].append((a,j))
        data[a]={'uv':uv,'new_uv':uv.clone(),'desc':desc,'new_desc':desc.clone(),'templates':torch.stack(templates)}
    print('Targets/views',len(targets),len(by_view),flush=True)
    model=GaussianModel2D(3);model.load_ply(args.prior,loc_feature_dim=0);model=model.cuda().eval()
    extractor=FeatureExtractor('sp',nms_radius=4).cuda().eval();extractor.requires_grad_(False)
    max_replay_delta=0.; shifts=[]
    for count,(i,entries) in enumerate(sorted(by_view.items())):
        height,width=map(int,rec[i]['native_input_hw'])
        with torch.no_grad():
            package=render_from_pose_gsplat(model,torch.from_numpy(T[i]).float().cuda(),
                cameras[i].fov_x,cameras[i].fov_y,width,height,
                bg_color=torch.zeros(3,device='cuda'),render_mode='RGB+ED',rgb_only=True,rasterize_mode='antialiased')
            dense,_=extractor.detectAndComputeDense(package['render'][:3].clamp(0,1)[None])
        points=torch.stack([data[a]['uv'][j] for a,j in entries]).cuda()
        template=torch.stack([data[a]['templates'][j] for a,j in entries]).cuda()
        with torch.no_grad():
            expected=torch.stack([data[a]['desc'][j] for a,j in entries]).cuda()
            replayed=sample_descriptors(points[None],dense)[0].T
            error=float((expected-replayed).abs().max());max_replay_delta=max(max_replay_delta,error)
            if error>2e-5:raise ValueError(f'render descriptor replay mismatch {error}')
        revised,descriptors,_,_=adjust_keypoints(dense,points,template)
        revised,descriptors=revised.cpu(),descriptors.cpu()
        valid=sample_raster_at_grid_uv(rec[i]['native_valid_mask'],revised).bool()
        for row,(a,j) in enumerate(entries):
            if not valid[row]:continue
            data[a]['new_uv'][j]=revised[row];data[a]['new_desc'][j]=descriptors[row]
            shifts.append(float(torch.linalg.vector_norm(revised[row]-data[a]['uv'][j])))
        if (count+1)%50==0 or count+1==len(by_view):print('Adjusted mapping views',count+1,'/',len(by_view),flush=True)
    accepted=[];rejected=Counter()
    for a in targets:
        s,e=offsets[a:a+2];qs=q[s:e];uv=np.asarray(data[a]['new_uv'],float)+.5;original=np.asarray(data[a]['uv'],float)+.5
        xyz=np.asarray(state['anchor_xyz'][a],float)
        before=crossfit_geometry(original,K[qs],T[qs],xyz);after=crossfit_geometry(uv,K[qs],T[qs],xyz)
        if not spatial_prediction_improves(before,after):
            rejected['no_spatial_group_improvement']+=1;continue
        def residual(x):
            cam=np.einsum('nij,j->ni',T[qs,:3,:3],x)+T[qs,:3,3];h=np.einsum('nij,nj->ni',K[qs],cam)
            return (h[:,:2]/np.maximum(h[:,2:],1e-8)-uv).ravel()
        fit=least_squares(residual,xyz,loss='huber',f_scale=2.,max_nfev=30)
        cam=np.einsum('nij,j->ni',T[qs,:3,:3],fit.x)+T[qs,:3,3]
        olderr=np.linalg.norm(residual(fit.x).reshape(-1,2)+uv-original,axis=1)
        normal=fit.jac.T@fit.jac
        if not fit.success or not np.isfinite(fit.x).all() or np.any(cam[:,2]<=0) or np.linalg.norm(fit.x-xyz)>.02*np.median(cam[:,2]) or np.median(olderr)>before['original_reprojection_median_px']+.25 or np.linalg.cond(normal)>1e6:
            rejected['fit_guard']+=1;continue
        state['anchor_xyz'][a]=torch.from_numpy(fit.x).float()
        state['anchor_position_covariance'][a]=torch.from_numpy(np.linalg.inv(normal)*max(1.,np.median(olderr)**2)).float()
        state['anchor_features'][a]=fuse_projective_anchor_observations(data[a]['new_desc'],replay['bins'][s:e],detector_weight=replay['weights'][s:e],trim_fraction=.2)
        accepted.append({'anchor_row':a,'query_indices':qs.tolist(),'source_keypoint_indices':k[s:e].tolist(),
                         'adjusted_grid_uv':data[a]['new_uv'].tolist(),'xyz_shift_cm':float(np.linalg.norm(fit.x-xyz)*100),
                         'before':before,'after':after})
    report={'schema':'anygsloc_multiview_featuremetric_observation_pilot','uses_source_mapping_rgb':False,
            'uses_test_queries':False,'promoted':False,'source_map_sha256':map_sha,'source_cache_sha256':cache_sha,
            'source_association_sha256':graph_sha,'source_geometry_sha256':sha256_file(args.geometry),
            'prior_sha256':sha256_file(args.prior),'target_count':len(targets),'accepted_count':len(accepted),
            'camera_registry':registry,
            'rejected':dict(rejected),'render_descriptor_max_abs_replay_delta':max_replay_delta,
            'observation_shift_px_p50_p90':np.percentile(shifts,[50,90]).tolist() if shifts else [],
            'policy':{'maximum_displacement_px':1.5,'trust':.002,'adam_steps':40,'target':'other-view descriptor consensus excluding self',
                      'acceptance':'both spatial heldout groups nonworse, maximum at least 5% better, original measurement fit protected',
                      'crossfit_caveat':'descriptor targets span the groups, so post-adjustment prediction is a stability check, not statistically independent validation',
                      'limitations':'one pass of stride-8 SuperPoint fields; internally correlated evidence, not verified true correspondence'},
            'implementation_sha256':{'script':sha256_file(Path(__file__)),
                'adjustment':sha256_file(Path(__file__).resolve().parents[1]/'evidence/multiview_keypoint_adjustment.py')},'changes':accepted}
    state['anchor_observation_features']=state['anchor_features'].clone();state['anchor_descriptor_residual']=torch.zeros_like(state['anchor_features'])
    state['geometry_and_descriptor_adjusted_observations']=accepted
    state['provenance']={**state['provenance'],'multiview_keypoint_adjustment':report}
    state['anchor_view_support']=build_anchor_view_support(anchor_xyz=state['anchor_xyz'],observation_offsets=csr['observation_offsets'],observation_query_indices=csr['query_indices'],mapping_pose_w2c=torch.from_numpy(T).float())
    args.output.mkdir(parents=True,exist_ok=False);path=args.output/'projective_anchor_map.pt';torch.save(state,path)
    torch.save(identity_metric_state(state,map_path=str(path.resolve()),map_sha256=sha256_file(path)),args.output/'identity_metric.pt')
    (args.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['changes','camera_registry']},indent=2),flush=True)


if __name__=='__main__':main()
