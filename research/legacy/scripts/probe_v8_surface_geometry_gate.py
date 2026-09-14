"""Mapping-only expected/median-depth plus normal-conflict gate ablation."""
import argparse
import json
from pathlib import Path
import torch
from common.hashing import sha256_file
from data.datasets import ColmapDataset
from evidence.surface_geometry_evidence import surface_row_evidence
from features.raster_sampling import sample_raster_at_grid_uv
from priors.models import GaussianModel2D
from priors.rendering import render_from_pose_gsplat
from scripts.materialize_mapping_rgb_descriptors import resolve_attested_mapping_cameras
from topology.v6_anchor_map import identity_metric_state,subset_projective_anchor_map


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['map','cache','dataset','prior','output']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(4)
    state=torch.load(args.map,map_location='cpu',weights_only=False);cache=torch.load(args.cache,map_location='cpu',weights_only=False)
    cache_sha=sha256_file(args.cache)
    if state['provenance']['v8_observation_cache_sha256']!=cache_sha:raise ValueError('map/cache mismatch')
    for payload in [state['provenance'],cache]:
        if any(payload.get(k) is not False for k in ['uses_source_mapping_rgb','uses_test_queries']):raise ValueError('mapping renders only')
    cameras,registry=resolve_attested_mapping_cameras(ColmapDataset(args.dataset,images='processed').split('mapping'),cache,state['v6_mapping_query_names'])
    model=GaussianModel2D(3);model.load_ply(args.prior,loc_feature_dim=0);model=model.cuda().eval()
    samples=[];depth_replay=0.
    for i,c in enumerate(cameras):
        r=cache['queries'][c.image_name];points=r['native_keypoints'].cuda()
        out=render_from_pose_gsplat(model,torch.as_tensor(c.pose_w2c).float().cuda(),c.fov_x,c.fov_y,c.width,c.height,
            bg_color=torch.zeros(3,device='cuda'),render_mode='RGB+ED',rgb_only=True,rasterize_mode='antialiased')
        def plane(key):return sample_raster_at_grid_uv(out[key].squeeze(),points).float()
        def normal(key):
            raster=out[key].squeeze()
            if raster.shape!=(c.height,c.width,3):raise ValueError('normal layout differs')
            return torch.stack([sample_raster_at_grid_uv(raster[:,:,j],points) for j in range(3)],dim=1)
        depth=plane('depth');old=sample_raster_at_grid_uv(r['native_depth'].cuda(),points)
        error=float((depth-old).abs().max());depth_replay=max(depth_replay,error)
        if error>1e-5:raise ValueError(f'depth replay differs {error}')
        result=surface_row_evidence(depth,plane('rend_median'),plane('rend_alpha'),normal('rend_normal'),normal('surf_normal'))
        samples.append({k:v.cpu() for k,v in result.items()})
        if (i+1)%100==0 or i+1==len(cameras):print('Surface evidence views',i+1,'/',len(cameras),flush=True)
    csr=state['projective_anchor_observations']
    if csr.get('keypoint_index_semantics')!='original_unfiltered_observation_cache_row':raise ValueError('CSR semantics differ')
    offsets=csr['observation_offsets'];query=csr['query_indices'];key=csr['keypoint_indices'];bins=state['v6_mapping_query_bins']
    labels=[];rejected=[]
    for a in range(len(state['anchor_ids'])):
        s,e=offsets[a:a+2];q,k=query[s:e],key[s:e]
        valid=torch.tensor([samples[i]['valid'][j] for i,j in zip(q,k)],dtype=torch.bool)
        conflict=torch.tensor([samples[i]['conflict'][j] for i,j in zip(q,k)],dtype=torch.bool)
        bad=int(conflict.sum())>=3 and int(conflict.sum())>=.5*len(q) and len(torch.unique(bins[q[conflict]]))>=2
        if bad:rejected.append(a)
        labels.append('conflicting' if bad else ('unknown' if int(valid.sum())<3 else 'no_strong_conflict'))
    keep=torch.ones(len(labels),dtype=torch.bool);keep[rejected]=False
    result=subset_projective_anchor_map(state,torch.nonzero(keep).flatten())
    report={'schema':'anygsloc_surface_geometry_gate_ablation','uses_source_mapping_rgb':False,'uses_test_queries':False,
        'absolute_geometry_certified':False,'promoted':False,'map_sha256':sha256_file(args.map),'cache_sha256':cache_sha,
        'prior_sha256':sha256_file(args.prior),'implementation_sha256':{'script':sha256_file(Path(__file__)),
            'evidence':sha256_file(Path(__file__).resolve().parents[1]/'evidence/surface_geometry_evidence.py')},
        'camera_registry':registry,'maximum_depth_replay_abs_delta':depth_replay,'input_anchor_count':len(labels),
        'rejected_count':len(rejected),'retained_count':int(keep.sum()),'labels':labels,'rejected_source_rows':rejected,
        'policy':{'row_valid':'finite positive depths, alpha>=.5, normal norms>.1',
                  'row_conflict':'relative expected/median depth difference>.02 AND normal-axis disagreement>45 degrees',
                  'anchor_reject':'at least 3 conflicts, at least half ALL observations, at least 2 conflicting camera bins',
                  'unknown_policy':'preserved','limitations':'both channels come from same prior; cannot expose common stable world bias'}}
    result['provenance']={**result['provenance'],'surface_geometry_gate':{k:v for k,v in report.items() if k not in ['labels','camera_registry']}}
    args.output.mkdir(parents=True,exist_ok=False)
    torch.save({'map_sha256':report['map_sha256'],'cache_sha256':cache_sha,'uses_source_mapping_rgb':False,
                'uses_test_queries':False,'query_names':list(cache['queries']),'records':samples},args.output/'surface_observation_evidence.pt')
    path=args.output/'projective_anchor_map.pt';torch.save(result,path)
    torch.save(identity_metric_state(result,map_path=str(path.resolve()),map_sha256=sha256_file(path)),args.output/'identity_metric.pt')
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['labels','rejected_source_rows','camera_registry']},indent=2),flush=True)


if __name__=='__main__':main()
