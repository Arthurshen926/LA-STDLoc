import copy
import numpy as np
import pytest
import torch
from experiments.cambridge_mainline_ablations import t7

def fixture():
 state={
  "anchor_xyz":torch.arange(12,dtype=torch.float32).reshape(4,3),
  "anchor_features":torch.eye(4),"anchor_matchability":torch.tensor([.5,.9,.9,.1]),
  "v6_mapping_query_names":["a","b","c","d"],
  "provenance":{"uses_test_queries":False,"uses_source_mapping_rgb":False},
  "projective_anchor_observations":{"observation_offsets":torch.tensor([0,2,5,8,12]),
    "query_indices":torch.tensor([0,1,0,1,2,0,1,2,0,1,2,3]),
    "keypoint_indices":torch.arange(12)}}
 trace={"selected_universe_ids":torch.tensor([0,1,2]),
  "final_source_ids":torch.tensor([0,2]),
  "trace":[{"candidate_universe_id":0,"primary_reason":"precision"},
           {"candidate_universe_id":1,"primary_reason":"matching_completion"},
           {"candidate_universe_id":2,"primary_reason":"observability_completion"}]}
 return state,trace,np.array([0,2])

def test_stage_and_budget_rows():
 s,t,n=fixture();o=t7.derive_rows(s,t,n)
 assert o["quality_stage"].tolist()==[0]
 assert o["coverage_stage"].tolist()==[0,1]
 assert o["quality_topk"].tolist()==[1,2]
 assert o["tracklength_topk"].tolist()==[1,3]
 assert all(len(o[k])==len(n) for k in ("quality_topk","tracklength_topk"))

def test_input_immutability_and_no_gt_dependency():
 s,t,n=fixture();old=copy.deepcopy(s);first=t7.derive_rows(s,t,n)
 s["ignored_test_poses"]=torch.rand(99,4,4)
 second=t7.derive_rows(s,t,n)
 for k in first:assert np.array_equal(first[k],second[k])
 for k in ("anchor_xyz","anchor_features","anchor_matchability"):assert torch.equal(old[k],s[k])

@pytest.mark.parametrize("bad",["test","source","duplicate_camera","offset","quality","trace","final","unknown_stage"])
def test_invalid_contracts_rejected(bad):
 s,t,n=fixture()
 if bad=="test":s["provenance"]["uses_test_queries"]=True
 elif bad=="source":s["provenance"]["uses_source_mapping_rgb"]=True
 elif bad=="duplicate_camera":s["projective_anchor_observations"]["query_indices"][1]=0
 elif bad=="offset":s["projective_anchor_observations"]["observation_offsets"][-1]=99
 elif bad=="quality":s["anchor_matchability"][0]=float("nan")
 elif bad=="trace":t["selected_universe_ids"]=torch.tensor([0,1,3])
 elif bad=="final":n=np.array([0,1])
 elif bad=="unknown_stage":t["trace"][0]["primary_reason"]="new_rule"
 with pytest.raises(ValueError):t7.derive_rows(s,t,n)

def test_subset_drift_rejected():
 s,t,n=fixture();sub={k:s[k][n].clone() for k in ("anchor_xyz","anchor_features")}
 t7.check_subset(s,sub,n)
 sub["anchor_xyz"][0,0]+=1
 with pytest.raises(ValueError):t7.check_subset(s,sub,n)

def test_failures_stay_in_metrics():
 rows=[{"image_name":"a","success":True,"translation_error_cm":1.,"rotation_error_deg":1.},
       {"image_name":"b","success":False,"translation_error_cm":None,"rotation_error_deg":None}]
 v=t7.independent_metrics(rows,["a","b"])
 assert v["R5_pct"]==50 and v["R15_pct"]==50 and v["failures"]==1
 assert np.isinf(v["mean_TE_cm"])
 with pytest.raises(ValueError):t7.independent_metrics(rows,["b","a"])

def test_snapshot_changes_rejected():
 a={"mapping_registry":[],"calibration":{},"mask_sha256":"x","queries":[{"pose":1}]}
 t7.same_snapshot(a,copy.deepcopy(a))
 b=copy.deepcopy(a);b["queries"][0]["pose"]=2
 with pytest.raises(ValueError):t7.same_snapshot(a,b)

def test_cpu_only():
 assert not torch.cuda.is_initialized()


def test_full_review_binds_actual_protocol_and_report(tmp_path):
 t7.write_new(tmp_path/"protocol.json",{"a":1})
 t7.write_new(tmp_path/"report.json",{"complete":True})
 t7.write_new(tmp_path/"independent_final_review.json",{"complete":True,"verdict":"PASS",
  "protocol_sha256":t7.sha256(tmp_path/"protocol.json"),"report_sha256":t7.sha256(tmp_path/"report.json")})
 t7.check_full_review(tmp_path)
 (tmp_path/"report.json").write_text("{}")
 with pytest.raises(ValueError):t7.check_full_review(tmp_path)

def test_actual_evidence_path_and_exact_names_bound(tmp_path):
 folder=tmp_path/"output";folder.mkdir()
 (folder/"results.json").write_text("[]")
 other=tmp_path/"other";other.mkdir();(other/"results.json").write_text("[]")
 refs={"results.json":t7.evidence(folder/"results.json")}
 t7.checked_files(folder,refs,("results.json",))
 with pytest.raises(ValueError):t7.checked_files(folder,refs,("results.json","summary.json"))
 refs["results.json"]=t7.evidence(other/"results.json")
 with pytest.raises(ValueError):t7.checked_files(folder,refs,("results.json",))

def test_manifest_and_query_snapshot_inventory_hashes(tmp_path):
 ready={"map_manifests":{},"query_snapshots":{}}
 for scene in t7.SCENES:
  maps={}
  for config in t7.NEW:
   dest=tmp_path/"maps"/scene/config/"deployment_map.pt"
   maps[config]={"path":str(dest),"sha256":"a","rows":{"path":str(dest.with_suffix(".source_rows.npy")),"sha256":"b"}}
  path=tmp_path/"maps"/scene/"manifest.json";t7.write_new(path,maps)
  ready["map_manifests"][scene]=t7.evidence(path)
  path=tmp_path/"query_inputs"/(scene+".json");t7.write_new(path,{"queries":[]})
  ready["query_snapshots"][scene]=t7.evidence(path)
 t7.check_frozen_inventory(tmp_path,ready)
 bad=copy.deepcopy(ready);bad["map_manifests"].pop(t7.SCENES[0])
 with pytest.raises(ValueError):t7.check_frozen_inventory(tmp_path,bad)
 (tmp_path/"maps"/t7.SCENES[0]/"manifest.json").write_text("{}")
 with pytest.raises(ValueError):t7.check_frozen_inventory(tmp_path,ready)

def test_review_binds_preflight_too(tmp_path):
 t7.write_new(tmp_path/"protocol.json",{"a":1})
 t7.write_new(tmp_path/"preflight.json",{"complete":True})
 t7.write_new(tmp_path/"independent_pre_gpu_review.json",{"verdict":"PASS",
  "protocol_sha256":t7.sha256(tmp_path/"protocol.json"),"preflight_sha256":t7.sha256(tmp_path/"preflight.json")})
 t7.require_review(tmp_path)
 (tmp_path/"preflight.json").write_text("{}")
 with pytest.raises(ValueError):t7.require_review(tmp_path)
