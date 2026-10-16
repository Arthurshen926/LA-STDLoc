import copy
import numpy as np
import pytest
import torch
from experiments.cambridge_mainline_ablations import t6
from experiments.cambridge_input_matrix.contracts import MatrixObservationProvider
from experiments.cambridge_input_matrix import projective_association as P

def payload():
 names=["view_b","view_a","view_c"];records={}
 pts=torch.tensor([[-.3,0.,5.],[-.1,.2,5.],[.2,-.2,5.],[.4,.1,5.]])
 for i,name in enumerate(names):
  pose=torch.eye(4);pose[0,3]=[-.2,0.,.2][i];K=torch.tensor([[500.,0.,320.],[0.,500.,240.],[0.,0.,1.]])
  cam=pts+pose[:3,3];pix=cam@K.T;uv=pix[:,:2]/pix[:,2:]-.5
  records[name]={"native_keypoints":uv,"native_descriptors":torch.eye(256)[:4],
   "native_scores":torch.ones(4),"native_K":K,"pose_w2c":pose,"native_input_hw":torch.tensor([480,640]),
   "row_source":torch.ones(4,dtype=torch.long),"source_row":torch.arange(4),"v2_valid":torch.ones(4,dtype=torch.bool),
   "native_valid_keypoint_mask":torch.ones(4,dtype=torch.bool),
   "native_depth_at_keypoints":torch.full((4,),5.),"native_alpha_at_keypoints":torch.ones(4)}
 return {"input_mode":"pure_gs","mapping_names":names,"test_names":["query"],"queries":records,
  "uses_source_mapping_rgb":False,"uses_test_queries":False}

# These three sidecar-only diagnostics are deliberately undefined before
# reconstruction (no scene-point overlap input / no triangulation yet).
# They never enter XYZ, descriptors, solver acceptance or common support.
NAN_DIAGNOSTICS={"mapping_point_overlap_jaccard","mapping_point_parallax_median_deg",
                 "actual_triangulation_parallax_median_deg"}
def equal(a,b,path=()):
 if torch.is_tensor(a):
  allow_nan=bool(path) and path[-1] in NAN_DIAGNOSTICS
  if not allow_nan:assert torch.isfinite(a).all() and torch.isfinite(b).all(),path
  else:assert not torch.isinf(a).any() and not torch.isinf(b).any(),path
  torch.testing.assert_close(a,b,rtol=0,atol=0,equal_nan=allow_nan)
 elif isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:equal(a[k],b[k],path+(k,))
 elif isinstance(a,(tuple,list)):
  assert len(a)==len(b)
  for j,(x,y) in enumerate(zip(a,b)):equal(x,y,path+(j,))
 else:assert a==b

def test_nonfinite_numerical_output_cannot_pass_comparator():
 for field in ("anchor_xyz","anchor_features","confidence","solver_point"):
  with pytest.raises(AssertionError):equal({field:torch.tensor([float("nan")])},{field:torch.tensor([float("nan")])})


def test_unfiltered_association_preserves_mainline_math_on_valid_input():
 data=payload();provider=MatrixObservationProvider(data)
 a=P.build_projective_association_graph(provider,device="cpu")
 b=t6.unfiltered_association(provider,"cpu")
 for key in ("tracks","component_statistics","query_bins","pair_sidecar","diagnostics"):equal(a[key],b[key])
 assert b["uses_source_mapping_rgb"] is False and b["uses_test_queries"] is False
 assert b["contract"]["render_valid_observations_required"] is False

def test_off_filter_only_does_not_rewrite_or_drop_evidence():
 data=payload();original=copy.deepcopy(data)
 a=t6.unfiltered_association(MatrixObservationProvider(data),"cpu")
 for r in data["queries"].values():
  r["v2_valid"][:]=False
  r["native_valid_keypoint_mask"][:]=False
 b=t6.unfiltered_association(MatrixObservationProvider(data),"cpu")
 for key in ("tracks","component_statistics","query_bins"):equal(a[key],b[key])
 for name in data["mapping_names"]:
  for key in ("native_keypoints","native_descriptors","native_depth_at_keypoints","native_alpha_at_keypoints"):
   assert torch.equal(data["queries"][name][key],original["queries"][name][key])

def test_single_observation_lexicographic_rule_is_exact():
 data=payload();names=data["mapping_names"]
 csr={"observation_offsets":torch.tensor([0,3]),"query_indices":torch.tensor([0,1,2]),"keypoint_indices":torch.tensor([2,1,0])}
 features,chosen=t6.single_descriptors(csr,names,data["queries"])
 assert chosen.tolist()==[[1,1]]
 assert torch.equal(features[0],data["queries"]["view_a"]["native_descriptors"][1])

@pytest.mark.parametrize("field",["uses_test_queries","uses_source_mapping_rgb","input_mode","registry","overlap","rows"])
def test_invalid_cache_rejected(field):
 data=payload();names=list(data["mapping_names"])
 if field in ("uses_test_queries","uses_source_mapping_rgb"):data[field]=True
 elif field=="input_mode":data[field]="raw_only"
 elif field=="registry":data["mapping_names"]=names[::-1]
 elif field=="overlap":data["test_names"]=[names[0]]
 elif field=="rows":data["queries"][names[0]]["v2_valid"]=torch.ones(1,dtype=torch.bool)
 with pytest.raises(ValueError):t6.validate_cache(data,names)

def test_gt_values_not_used_by_association():
 data=payload();a=t6.unfiltered_association(MatrixObservationProvider(data),"cpu")
 data["ignored_test_pose"]=torch.rand(4,4)
 b=t6.unfiltered_association(MatrixObservationProvider(data),"cpu")
 equal(a,b)

def test_build_review_binds_prepared(tmp_path):
 t6.C.write_new(tmp_path/"protocol.json",{"a":1});t6.C.write_new(tmp_path/"prepared.json",{"complete":True})
 t6.C.write_new(tmp_path/"independent_pre_build_review.json",{"verdict":"PASS",
  "protocol_sha256":t6.sha256(tmp_path/"protocol.json"),"prepared_sha256":t6.sha256(tmp_path/"prepared.json")})
 t6.build_review(tmp_path)
 (tmp_path/"prepared.json").write_text("{}")
 with pytest.raises(ValueError):t6.build_review(tmp_path)

def test_cpu_only():
 assert not torch.cuda.is_initialized()
