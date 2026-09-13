"""Fixed-correspondence depth-lift geometry baseline, not a new F0 method.

Same rendered keypoints, descriptors, Track identities and map capacity.
Backproject the cached axial render depths, then take a coordinate median.
This is a conditional depth-vs-ray representation ablation, not a full direct
Gaussian-primitive baseline and not an RGB-invariant prior perturbation test.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from common.hashing import sha256_file
from features.raster_sampling import sample_raster_at_grid_uv
from map_learning.v24_anchor_view_support import build_anchor_view_support
from topology.v6_anchor_map import identity_metric_state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['map','cache','output']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(4)
    state=torch.load(args.map,map_location='cpu',weights_only=False)
    cache=torch.load(args.cache,map_location='cpu',weights_only=False)
    cache_sha=sha256_file(args.cache)
    if state['provenance']['v8_observation_cache_sha256']!=cache_sha:raise ValueError('map/cache mismatch')
    for payload in [state['provenance'],cache]:
        if any(payload.get(k) is not False for k in ['uses_source_mapping_rgb','uses_test_queries']):raise ValueError('mapping render-only required')
    if state['v6_mapping_query_names']!=list(cache['queries']):raise ValueError('registry mismatch')
    world=[]; poses=[]
    for r in cache['queries'].values():
        depth=np.asarray(sample_raster_at_grid_uv(r['native_depth'],r['native_keypoints']),float)
        uv=np.asarray(r['native_keypoints'],float)+.5
        K=np.asarray(r['native_K'],float);T=np.asarray(r['pose_w2c'],float)
        camera=np.linalg.solve(K,np.c_[uv,np.ones(len(uv))].T).T*depth[:,None]
        X=(camera-T[:3,3])@T[:3,:3]
        X[~np.isfinite(depth)|(depth<=0)]=np.nan
        world.append(X);poses.append(T)
    csr=state['projective_anchor_observations']
    if csr.get('keypoint_index_semantics')!='original_unfiltered_observation_cache_row':raise ValueError('source row mismatch')
    offset,q,k=[np.asarray(csr[x]) for x in ['observation_offsets','query_indices','keypoint_indices']]
    old=state['anchor_xyz'].clone(); changed=0;fallback=0
    for a in range(len(old)):
        points=np.stack([world[i][j] for i,j in zip(q[offset[a]:offset[a+1]],k[offset[a]:offset[a+1]])])
        points=points[np.isfinite(points).all(1)]
        if len(points)<3:
            fallback+=1;continue
        state['anchor_xyz'][a]=torch.from_numpy(np.median(points,axis=0)).to(old.dtype)
        changed+=1
    shift=torch.linalg.vector_norm(state['anchor_xyz']-old,dim=1)*100
    report={'schema':'anygsloc_fixed_track_depth_geometry_control','uses_source_mapping_rgb':False,
            'uses_test_queries':False,'promoted':False,'source_map_sha256':sha256_file(args.map),
            'source_cache_sha256':cache_sha,'implementation_sha256':sha256_file(Path(__file__)),
            'descriptor_ids_observations_unchanged':True,'replaced_count':changed,'fallback_ray_count':fallback,
            'xyz_shift_cm_p50_p90':torch.quantile(shift,torch.tensor([.5,.9])).tolist(),
            'covariance':'removed because ray-fit covariance is not valid for depth-lifted XYZ',
            'depth_sampling':'cached axial depth nearest raster cell; coordinate-wise multiview median',
            'scope':'conditional on F0 selected observations and identities, not a complete primitive baseline'}
    state.pop('anchor_position_covariance',None)
    state['anchor_view_support']=build_anchor_view_support(anchor_xyz=state['anchor_xyz'],
        observation_offsets=csr['observation_offsets'],observation_query_indices=csr['query_indices'],
        mapping_pose_w2c=torch.from_numpy(np.stack(poses)).float())
    state['projective_anchor_construction']={**state['projective_anchor_construction'],
        'final_xyz_source':'diagnostic_render_depth_lift_coordinate_median_with_explicit_ray_fallback'}
    state['provenance']={**state['provenance'],'depth_geometry_control':report}
    args.output.mkdir(parents=True,exist_ok=False)
    path=args.output/'projective_anchor_map.pt';torch.save(state,path)
    torch.save(identity_metric_state(state,map_path=str(path.resolve()),map_sha256=sha256_file(path)),args.output/'identity_metric.pt')
    (args.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
