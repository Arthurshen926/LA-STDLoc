"""Descriptor-only interventions on immutable native landmarks and supports."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from anygsloc.mapping.tracks import fuse_projective_anchor_observations
from anygsloc.mapping.triangulation import _project
from anygsloc.mapping.observation_provider import PIXEL_CENTER_OFFSET

BASE_KIND="v2_projective_track"
COMPLETION_KIND="depth_proposed_projective_completion"
MAXIMUM_REPROJECTION_PX=4.0
MINIMUM_SUPPORT=3
TRIM_FRACTION=.2
ZERO_NORM=1e-12

def tensor(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if hasattr(value,"numpy"):return torch.from_numpy(value.numpy())
    return torch.as_tensor(value)

def pack_selected(full,cache,association,rows,mapping_names,test_names):
    rows=torch.as_tensor(rows,dtype=torch.long)
    if (rows.ndim!=1 or rows.numel()==0 or len(torch.unique(rows))!=len(rows) or
        int(rows.min())<0 or int(rows.max())>=len(full["anchor_xyz"])):
        raise ValueError("invalid immutable native rows")
    provenance=full["provenance"]
    if (provenance.get("uses_source_mapping_rgb") is not False or provenance.get("uses_test_queries") is not False or
        provenance.get("joint_track_purification",{}).get("enabled") is not False):
        raise ValueError("source scope or purified association confidence unsupported")
    if (list(full["v6_mapping_query_names"])!=mapping_names or list(cache["queries"])!=mapping_names or
        list(association["query_names"])!=mapping_names or set(mapping_names)&set(test_names) or
        association.get("uses_source_mapping_rgb") is not False or association.get("uses_test_queries") is not False):
        raise ValueError("mapping-only camera identity mismatch")
    bins=tensor(full["v6_mapping_query_bins"]).long()
    if not torch.equal(bins,tensor(association["query_bins"]).long()):
        raise ValueError("original view bins differ")
    csr=full["projective_anchor_observations"]
    offsets=tensor(csr["observation_offsets"]).long()
    cq=tensor(csr["query_indices"]).long();ck=tensor(csr["keypoint_indices"]).long()
    if (len(offsets)!=len(full["anchor_xyz"])+1 or int(offsets[0])!=0 or
        int(offsets[-1])!=len(cq) or len(cq)!=len(ck) or bool((offsets[1:]<offsets[:-1]).any())):
        raise ValueError("invalid source CSR")
    lengths=offsets[rows+1]-offsets[rows]
    if bool((lengths<MINIMUM_SUPPORT).any()):raise ValueError("native support unexpectedly smaller than original minimum")
    index=torch.cat([torch.arange(int(offsets[r]),int(offsets[r+1])) for r in rows])
    query=cq[index];keypoint=ck[index]
    if bool(((query<0)|(query>=len(mapping_names))).any()):raise ValueError("CSR camera index outside mapping registry")
    packed_offsets=torch.cat((torch.zeros(1,dtype=torch.long),lengths.cumsum(0)))
    kinds=[full["anchor_candidate_kind"][i] for i in rows.tolist()]
    if not set(kinds)<=set((BASE_KIND,COMPLETION_KIND)):raise ValueError("unsupported candidate family")
    count=len(query);desc=torch.empty(count,256);scores=torch.empty(count);uv=torch.empty(count,2)
    K=torch.empty(len(mapping_names),3,3);poses=torch.empty(len(mapping_names),4,4)
    qorder=torch.argsort(query,stable=True);qoffset=torch.cat((torch.zeros(1,dtype=torch.long),torch.bincount(query,minlength=len(mapping_names)).cumsum(0)))
    row_counts=[]
    for q,name in enumerate(mapping_names):
        record=cache["queries"][name];d=tensor(record["native_descriptors"]).float()
        xy=tensor(record["native_keypoints"]).float();score=tensor(record["native_scores"]).float()
        if d.shape!=(len(xy),256) or score.shape!=(len(xy),):raise ValueError("source sparse rows differ")
        ids=qorder[qoffset[q]:qoffset[q+1]];kp=keypoint[ids]
        if len(kp) and (int(kp.min())<0 or int(kp.max())>=len(xy)):raise ValueError("CSR source row outside original cache")
        desc[ids]=d[kp];scores[ids]=score[kp];uv[ids]=xy[kp]+PIXEL_CENTER_OFFSET
        K[q]=tensor(record["native_K"]);poses[q]=tensor(record["pose_w2c"])
        row_counts.append(len(xy))
    if not all(torch.isfinite(x).all() for x in (desc,scores,uv,K,poses)) or bool((scores<0).any()):
        raise ValueError("nonfinite/negative source observations")
    tracks=association["tracks"];aq=tensor(tracks["query_index"]).long();ak=tensor(tracks["keypoint_index"]).long()
    confidence=tensor(tracks["confidence"]).float();track=tensor(tracks["track_index"]).long()
    source=[tensor(r).long() for r in association["source_keypoint_indices_by_query"]]
    if len(source)!=len(mapping_names):raise ValueError("V2 source-row registry differs")
    sizes=torch.tensor([len(x) for x in source]);starts=torch.cat((torch.zeros(1,dtype=torch.long),sizes.cumsum(0)))
    if (bool(((aq<0)|(aq>=len(source))).any()) or bool(((ak<0)|(ak>=sizes[aq])).any()) or
        len(confidence)!=len(aq) or not torch.isfinite(confidence).all() or bool((confidence<0).any())):
        raise ValueError("invalid association/source-row mapping")
    for q,ids in enumerate(source):
        if len(torch.unique(ids))!=len(ids) or (len(ids) and (int(ids.min())<0 or int(ids.max())>=row_counts[q])):
            raise ValueError("invalid V2 original source rows")
    original=torch.cat(source)[starts[aq]+ak]
    stride=max(row_counts)+1;codes=aq*stride+original
    order=torch.argsort(codes,stable=True);sorted_codes=codes[order]
    if len(torch.unique(sorted_codes))!=len(sorted_codes):raise ValueError("association repeats a physical observation")
    desired=query*stride+keypoint;positions=torch.searchsorted(sorted_codes,desired)
    found=positions<len(sorted_codes);found[found.clone()]=sorted_codes[positions[found]]==desired[found]
    conf=torch.empty(count)
    for i,kind in enumerate(kinds):
        a,b=int(packed_offsets[i]),int(packed_offsets[i+1]);q=query[a:b]
        if len(torch.unique(q))!=len(q):raise ValueError("same camera voted twice in an anchor")
        if kind==BASE_KIND:
            if not bool(found[a:b].all()):raise ValueError("base observation has no authentic association confidence")
            graph_ids=order[positions[a:b]]
            if len(torch.unique(track[graph_ids]))!=1:raise ValueError("base CSR mixes association components")
            conf[a:b]=confidence[graph_ids]
        else:
            if bool(found[a:b].any()):raise ValueError("completion illegally reuses base association observation")
            conf[a:b]=scores[a:b].clamp(0,1)
    return {"rows":rows,"offsets":packed_offsets,"query":query,"keypoint":keypoint,"descriptors":desc,
            "detector_confidence":scores*conf,"bins":bins,"K":K,"poses":poses,"uv":uv,"kinds":kinds}

def descriptor_variants(full,packed):
    rows=packed["rows"];old=full["anchor_features"][rows].float()
    mean_all=old.clone();reproj_robust=old.clone();replayed=old.clone()
    offsets=packed["offsets"];support=torch.empty(len(rows),dtype=torch.long)
    keep_count=torch.empty_like(support);fallback_mean=torch.zeros(len(rows),dtype=torch.bool)
    fallback_reproj=torch.zeros_like(fallback_mean);reproj_zero=torch.zeros_like(fallback_mean)
    for i,row in enumerate(rows.tolist()):
        a,b=int(offsets[i]),int(offsets[i+1]);q=packed["query"][a:b]
        d=packed["descriptors"][a:b];weight=packed["detector_confidence"][a:b];bins=packed["bins"][q]
        replayed[i]=fuse_projective_anchor_observations(d,bins,detector_weight=weight,trim_fraction=TRIM_FRACTION)
        feature=F.normalize(d,dim=1).mean(0);norm=torch.linalg.vector_norm(feature)
        if norm<ZERO_NORM:fallback_mean[i]=True
        else:mean_all[i]=feature/norm
        projected,depth=_project(full["anchor_xyz"][row].double(),packed["K"][q].double(),packed["poses"][q].double())
        error=torch.linalg.vector_norm(projected-packed["uv"][a:b].double(),dim=1)
        keep=torch.isfinite(error)&torch.isfinite(depth)&(error<=MAXIMUM_REPROJECTION_PX)&(depth>0)
        support[i]=b-a;keep_count[i]=int(keep.sum())
        if keep_count[i]<MINIMUM_SUPPORT:
            fallback_reproj[i]=True;continue
        fused=fuse_projective_anchor_observations(d[keep],bins[keep],detector_weight=weight[keep],trim_fraction=TRIM_FRACTION)
        if torch.linalg.vector_norm(fused)<ZERO_NORM:
            fallback_reproj[i]=True;reproj_zero[i]=True
        else:reproj_robust[i]=fused
    delta=(replayed-old).abs()
    replay={"bitwise_equal":torch.equal(replayed,old),"different_rows":int((delta.max(1).values>0).sum()),
            "max_abs":float(delta.max()),"native_count":len(rows)}
    stats={"original_support":support,"retained_support":keep_count,"mean_zero_fallback":fallback_mean,
           "reproj_fallback":fallback_reproj,"reproj_zero_fallback":reproj_zero}
    return {"mean_all":mean_all,"reproj4_robust":reproj_robust},replay,stats
