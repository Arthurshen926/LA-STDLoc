"""Fixed-XYZ render-only descriptor consistency ablation; no query feedback."""
import argparse
import json
from pathlib import Path
import torch
from common.hashing import sha256_file
from evidence.v8_fixed_descriptor_replay import prepare_replay, fuse_replay, fuse_geometry_balanced
from topology.v6_anchor_map import identity_metric_state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['map','cache','association','output']: p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(4)
    state=torch.load(args.map,map_location='cpu',weights_only=False)
    cache=torch.load(args.cache,map_location='cpu',weights_only=False)
    graph=torch.load(args.association,map_location='cpu',weights_only=False)
    for payload in [state['provenance'],cache,graph]:
        if any(payload.get(k) is not False for k in ['uses_source_mapping_rgb','uses_test_queries']): raise ValueError('render mapping-only required')
    cache_sha,graph_sha=sha256_file(args.cache),sha256_file(args.association)
    if cache_sha!=state['provenance']['v8_observation_cache_sha256'] or graph_sha!=state['provenance']['v8_association_graph_sha256']:raise ValueError('lineage mismatch')
    replay=prepare_replay(state,cache,graph)
    desc=[r['native_descriptors'].float() for r in cache['queries'].values()]
    source=fuse_replay(desc,replay)
    if not torch.equal(source,state['anchor_features']):raise ValueError('source fusion not bit exact')
    print('Exact source descriptor fusion PASS',flush=True)
    csr=state['projective_anchor_observations']; q=csr['query_indices'].long(); k=csr['keypoint_indices'].long()
    rec=list(cache['queries'].values())
    K=torch.stack([torch.as_tensor(r['native_K']).double() for r in rec]); T=torch.stack([torch.as_tensor(r['pose_w2c']).double() for r in rec])
    uv=torch.cat([r['native_keypoints'].double() for r in rec])[replay['source_index']]+.5
    row=torch.repeat_interleave(torch.arange(len(source)),replay['offsets'][1:]-replay['offsets'][:-1])
    X=state['anchor_xyz'].double()[row]
    cam=(T[q,:3,:3]@X[...,None])[...,0]+T[q,:3,3]
    h=(K[q]@cam[...,None])[...,0]
    residual=torch.linalg.vector_norm(h[:,:2]/h[:,2:].clamp_min(1e-8)-uv,dim=1)
    reliability=(2./residual.clamp_min(2.)).float()*(cam[:,2]>0).float()
    report={'schema':'anygsloc_fixed_xyz_geometry_descriptor_alignment',
            'uses_source_mapping_rgb':False,'uses_test_queries':False,'promoted':False,
            'geometry_weight_role':'Huber2 evaluated at frozen final XYZ on ALL source rows, not a production IRLS subset replay',
            'source_fusion_bit_exact':True,'source_map_sha256':sha256_file(args.map),
            'source_cache_sha256':cache_sha,'source_association_sha256':graph_sha,
            'xyz_identity_csr_unchanged':True,'source_observations':len(q),
            'residual_over_2px_percent':float((residual>2).float().mean()*100),
            'residual_over_4px_percent':float((residual>4).float().mean()*100),
            'legacy_weight_mass_over_4px_percent':float(replay['weights'][residual>4].sum()/replay['weights'].sum()*100),
            'implementation_sha256':{'script':sha256_file(Path(__file__)),
              'fusion':sha256_file(Path(__file__).resolve().parents[1]/'evidence/v8_fixed_descriptor_replay.py')}}
    args.output.mkdir(parents=True,exist_ok=False)
    gathered=torch.cat(desc)[replay['source_index']]
    for mode in ['within_bin','across_bin']:
        if mode=='within_bin':
            fused=fuse_replay(desc,{**replay,'weights':replay['weights']*reliability})
        else:
            fused=torch.stack([fuse_geometry_balanced(gathered[s:e],replay['bins'][s:e],replay['weights'][s:e],reliability[s:e])
                               for s,e in zip(replay['offsets'][:-1],replay['offsets'][1:])])
        state['anchor_features']=fused; state['anchor_observation_features']=fused.clone(); state['anchor_descriptor_residual']=torch.zeros_like(fused)
        detail={**report,'mode':mode,'changed_anchor_count':int(torch.any(fused!=source,dim=1).sum()),
                'descriptor_cosine_p10_p50':torch.quantile((source*fused).sum(1),torch.tensor([.1,.5])).tolist()}
        state['provenance']={**state['provenance'],'geometry_descriptor_alignment':detail}
        directory=args.output/mode;directory.mkdir()
        path=directory/'projective_anchor_map.pt';torch.save(state,path)
        torch.save(identity_metric_state(state,map_path=str(path.resolve()),map_sha256=sha256_file(path)),directory/'identity_metric.pt')
        (directory/'report.json').write_text(json.dumps(detail,indent=2))
        print(mode,detail,flush=True)


if __name__=='__main__':main()
