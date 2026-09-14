"""Paired geometry-source control on identical Compact rows and descriptors."""
import argparse
import json
from pathlib import Path
import torch
from common.hashing import sha256_file
from features.raster_sampling import sample_raster_at_grid_uv
from scripts.materialize_purity_candidates import subset


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--map-dir',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    state=torch.load(a.map_dir/'projective_anchor_map.pt',map_location='cpu',weights_only=False)
    metric=torch.load(a.map_dir/'identity_metric.pt',map_location='cpu',weights_only=False)
    cache=torch.load(a.cache,map_location='cpu',weights_only=False)
    assert state['provenance']['uses_test_queries'] is False
    names=state['v6_mapping_query_names'];assert names==list(cache['queries'])
    points=[]
    for name in names:
        c=cache['queries'][name];uv=c['native_keypoints'].float();z=sample_raster_at_grid_uv(c['native_depth'],uv).float()
        alpha=sample_raster_at_grid_uv(c['native_alpha'].float(),uv)
        K=c['native_K'].float();pose=c['pose_w2c'].float();uv=uv+0.5
        xyz=torch.stack(((uv[:,0]-K[0,2])/K[0,0]*z,(uv[:,1]-K[1,2])/K[1,1]*z,z),1)
        xyz=(xyz-pose[:3,3])@pose[:3,:3]
        xyz[~(torch.isfinite(z)&(z>0)&torch.isfinite(alpha)&(alpha>=0.05))]=float('nan')
        points.append(xyz)
    csr=state['projective_anchor_observations'];off=csr['observation_offsets'].tolist()
    qs=csr['query_indices'].tolist();ks=csr['keypoint_indices'].tolist();keep=[];depth=[]
    for i in range(len(state['anchor_ids'])):
        xyz=torch.stack([points[qs[j]][ks[j]] for j in range(off[i],off[i+1])])
        valid=xyz[torch.isfinite(xyz).all(1)]
        if len(valid)>=3:
            keep.append(i);depth.append(valid.median(dim=0).values)
    if not keep:raise ValueError('No common valid geometry rows')
    base=subset(state,torch.tensor(keep));a.output.mkdir(parents=True,exist_ok=False)
    for kind in ['ray_common','depth_common']:
        revised=dict(base)
        if kind=='depth_common':revised['anchor_xyz']=torch.stack(depth)
        revised['provenance']=dict(base['provenance'],geometry_source_ablation=kind,source_map_sha256=sha256_file(a.map_dir/'projective_anchor_map.pt'),observation_cache_sha256=sha256_file(a.cache))
        folder=a.output/kind;folder.mkdir();path=folder/'projective_anchor_map.pt';torch.save(revised,path)
        torch.save(dict(metric,landmark_indices=revised['anchor_ids'],map_path=str(path),map_sha256=sha256_file(path)),folder/'identity_metric.pt')
    (a.output/'report.json').write_text(json.dumps(dict(source_count=len(state['anchor_ids']),common_count=len(keep),excluded_count=len(state['anchor_ids'])-len(keep),minimum_valid_depth_observations=3,aggregation='coordinatewise median',fixed_descriptors=True,uses_test_queries=False),indent=2))


if __name__=='__main__':main()
