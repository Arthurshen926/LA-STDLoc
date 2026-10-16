import copy,types
import numpy as np
import pytest
from experiments.cambridge_mainline_ablations import matcha_t5 as M,t5 as K

def test_original_numeric_function_objects_and_globals_preserved():
 assert M.CONFIGS==K.CONFIGS==("ray","depth","center")
 for name in ("build","evaluate"):
  assert getattr(M,name).__code__ is getattr(K,name).__code__
  assert getattr(M,name).__globals__ is not getattr(K,name).__globals__
 for name in ("depth_xyz","coordinate_group","project_residual","projected_centers","refit_ray"):
  assert M._scope[name] is getattr(K,name)
 assert M._scope["protocol"] is M.protocol and K.protocol is not M.protocol
 assert M._scope["checked_geometry"] is M.checked_geometry
 assert M._scope["DESIGN"] is M.DESIGN and K.DESIGN["prior"]!=M.DESIGN["prior"]
def test_depth_kernel_bitwise_identical():
 uv=np.array([[3.,2.],[0.,0.]],dtype=np.float32)
 z=np.array([4.,2.],dtype=np.float32)
 camera=np.array([[100.,0.,2.],[0.,100.,2.],[0.,0.,1.]])
 pose=np.eye(4);pose[:3,3]=[1.,2.,3.]
 assert np.array_equal(M._scope["depth_xyz"](uv,z,camera,pose),K.depth_xyz(uv,z,camera,pose))
def source_index():
 return dict(schema="matcha_t5_source_index_v1",prior_label="MAtCha",condition=M.MATCHA_CONDITION,
  cases={s:"/tmp/test_only/"+s for s in M.C.SCENES},source_index_review=dict(path="/tmp/test_only/review.json",sha256="x"),
  upstream_source_review=dict(path="/tmp/test_only/upstream.json",sha256="y"),
  downstream_reviews={s:{} for s in M.C.SCENES})
def test_fixed_five_scene_index_and_full_scope():
 v=source_index();M.validate_index(v)
 assert M.DESIGN["new_evaluations"]==15 and "none" in M.DESIGN["selection"]
@pytest.mark.parametrize("mutate",[
 lambda x:x.update(prior_label="vanilla3D"),
 lambda x:x.update(condition="long_lrm_raw"),
 lambda x:x["cases"].pop("StMarysChurch"),
 lambda x:x["cases"].update(StMarysChurch=x["cases"]["ShopFacade"]),
 lambda x:x.update(source_index_review={})])
def test_invalid_source_contract_rejected(mutate):
 v=source_index();mutate(v)
 with pytest.raises(ValueError):M.validate_index(v)
def test_cpu_preparation_gate(monkeypatch):
 monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0")
 with pytest.raises(ValueError):M.cpu_only()
 monkeypatch.setenv("CUDA_VISIBLE_DEVICES","");monkeypatch.setattr(M.torch.cuda,"is_initialized",lambda:False);M.cpu_only()
def test_adapter_import_does_not_replace_original_protocol():
 assert K.build.__globals__["protocol"] is K.protocol
 assert K.evaluate.__globals__["checked_geometry"] is K.checked_geometry

def case_fixture(tmp_path,background="white"):
 import json
 case=tmp_path/"case";case.mkdir()
 def file(rel,data):
  p=case/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);return p
 def write(rel,data):return file(rel,json.dumps(data).encode())
 state=file("selector_state.pt",b"state");cache=file("observations/pure_gs.pt",b"cache")
 full=file("maps/Full/deployment_map.pt",b"full");ply=file("prior.ply",b"ply")
 snapshot=write("snapshot.json",{})
 prior=dict(ply=str(ply),ply_sha256=M.C.sha256(ply),gaussian_type="2dgs",sh_degree=3,background=background)
 source=dict(verdict="PASS",complete=True,scene="ShopFacade",condition=M.MATCHA_CONDITION,prior=prior,
  evidence_sha256={str(ply):M.C.sha256(ply)})
 upstream=write("upstream.json",dict(schema=M.UPSTREAM_CASE_SCHEMA,scope="single_scene",scene_names=["ShopFacade"],count=1,verdict="PASS",complete=True,cases={M.MATCHA_CONDITION+"/ShopFacade":source}))
 upstream_ref=dict(path=str(upstream),sha256=M.C.sha256(upstream))
 p=dict(scene_name="ShopFacade",condition=M.MATCHA_CONDITION,mapping_semantics=False,query_semantics=True,
  prior=prior,evaluation_snapshot_reference=str(snapshot),upstream_review=upstream_ref)
 protocol=write("protocol.json",p)
 summary=dict(complete=True,protocol_sha256=M.C.sha256(protocol),
  selector_state_sha256=M.C.sha256(state),cache_sha256=M.C.sha256(cache),
  Full=dict(path=str(full),sha256=M.C.sha256(full),count=1))
 maps=write("maps/summary.json",summary)
 review=dict(verdict="PASS",complete=True,scene="ShopFacade",condition=M.MATCHA_CONDITION,
  protocol_sha256=M.C.sha256(protocol),
  evidence_sha256={str(f):M.C.sha256(f) for f in (protocol,maps,state,cache,full)})
 write("independent_final_review.json",review)
 return case,review,upstream_ref

def test_every_artifact_has_direct_independent_binding(tmp_path):
 case,review,upstream=case_fixture(tmp_path)
 p,refs=M.accepted_case(case,"ShopFacade",M.MATCHA_CONDITION,upstream)
 assert refs["ply"]["path"] not in review["evidence_sha256"]
 source=M.C.read(upstream["path"])["cases"][M.MATCHA_CONDITION+"/ShopFacade"]
 assert source["evidence_sha256"][refs["ply"]["path"]]==refs["ply"]["sha256"]

@pytest.mark.parametrize("rel",["selector_state.pt","observations/pure_gs.pt","maps/Full/deployment_map.pt"])
def test_downstream_artifact_review_binding_required(tmp_path,rel):
 import json
 case,review,upstream=case_fixture(tmp_path);review["evidence_sha256"].pop(str(case/rel))
 (case/"independent_final_review.json").write_text(json.dumps(review))
 with pytest.raises(ValueError,match="bind source artifact"):M.accepted_case(case,"ShopFacade",M.MATCHA_CONDITION,upstream)

def test_actual_ply_must_match_reviewed_protocol(tmp_path):
 case,_,upstream=case_fixture(tmp_path);(case/"prior.ply").write_bytes(b"changed")
 with pytest.raises(ValueError,match="Source hash differs"):M.accepted_case(case,"ShopFacade",M.MATCHA_CONDITION,upstream)

@pytest.mark.parametrize("change",["missing_ply_evidence","wrong_ply_path","wrong_type","wrong_sh","wrong_background","not_pass","wrong_case","wrong_schema","upstream_not_pass","upstream_incomplete","case_incomplete","wrong_scope","wrong_scene_names","wrong_count"])
def test_upstream_direct_ply_review_is_mandatory(tmp_path,change):
 import json
 case,review,upstream=case_fixture(tmp_path)
 u=M.C.read(upstream["path"]);source=u["cases"][M.MATCHA_CONDITION+"/ShopFacade"]
 if change=="missing_ply_evidence":source["evidence_sha256"]={}
 elif change=="wrong_ply_path":source["prior"]["ply"]=str(case/"another.ply")
 elif change=="wrong_type":source["prior"]["gaussian_type"]="other"
 elif change=="wrong_sh":source["prior"]["sh_degree"]=0
 elif change=="wrong_background":source["prior"]["background"]="black"
 elif change=="not_pass":source["verdict"]="PENDING"
 elif change=="wrong_case":source["scene"]="KingsCollege"
 elif change=="wrong_schema":u["schema"]="other"
 elif change=="upstream_not_pass":u["verdict"]="PENDING"
 elif change=="upstream_incomplete":u["complete"]=False
 elif change=="case_incomplete":source["complete"]=False
 elif change=="wrong_scope":u["scope"]="all_scenes"
 elif change=="wrong_scene_names":u["scene_names"]=["KingsCollege"]
 else:u["count"]=5
 from pathlib import Path
 Path(upstream["path"]).write_text(json.dumps(u));upstream["sha256"]=M.C.sha256(upstream["path"])
 p=M.C.read(case/"protocol.json");p["upstream_review"]=upstream
 (case/"protocol.json").write_text(json.dumps(p));h=M.C.sha256(case/"protocol.json")
 maps=M.C.read(case/"maps/summary.json");maps["protocol_sha256"]=h;(case/"maps/summary.json").write_text(json.dumps(maps))
 review["protocol_sha256"]=h;review["evidence_sha256"][str(case/"protocol.json")]=h
 review["evidence_sha256"][str(case/"maps/summary.json")]=M.C.sha256(case/"maps/summary.json")
 (case/"independent_final_review.json").write_text(json.dumps(review))
 with pytest.raises(ValueError):M.accepted_case(case,"ShopFacade",M.MATCHA_CONDITION,upstream)

def test_substituted_upstream_receipt_rejected(tmp_path):
 case,_,upstream=case_fixture(tmp_path)
 duplicate=case/"other_upstream.json";duplicate.write_bytes((case/"upstream.json").read_bytes())
 fake=dict(path=str(duplicate),sha256=M.C.sha256(duplicate))
 with pytest.raises(ValueError,match="substituted"):M.accepted_case(case,"ShopFacade",M.MATCHA_CONDITION,fake)

def test_wrong_production_background_rejected(tmp_path):
 case,_,upstream=case_fixture(tmp_path,background="black")
 with pytest.raises(ValueError,match="MAt prior"):M.accepted_case(case,"ShopFacade",M.MATCHA_CONDITION,upstream)

def scene_index():
 v=source_index();scene="ShopFacade"
 v.update(schema="matcha_t5_scene_source_index_v1",scope="single_scene",scene=scene)
 v["cases"]={scene:v["cases"][scene]};v["downstream_reviews"]={scene:v["downstream_reviews"][scene]}
 return v

def test_single_scene_index_does_not_require_other_sources():
 v=scene_index()
 assert M.validate_index(v)==("ShopFacade",)
 assert "StMarysChurch" not in v["cases"]

@pytest.mark.parametrize("mutate",[
 lambda x:x.update(scope="five_scenes"),
 lambda x:x.update(scene="StMarysChurch"),
 lambda x:x["cases"].update(KingsCollege="/tmp/not_accepted"),
 lambda x:x["downstream_reviews"].update(KingsCollege={})])
def test_single_scene_scope_cannot_expand(mutate):
 v=scene_index();mutate(v)
 with pytest.raises(ValueError):M.validate_index(v)

@pytest.mark.parametrize("mode",["missing_scene","duplicate_root"])
def test_final_collection_requires_five_unique_scenes(tmp_path,monkeypatch,mode):
 import json
 monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
 index=dict(schema="matcha_t5_case_collection_v1",cases={s:dict(root=str(tmp_path/s)) for s in M.C.SCENES})
 if mode=="missing_scene":index["cases"].pop("StMarysChurch")
 else:index["cases"]["StMarysChurch"]["root"]=index["cases"]["ShopFacade"]["root"]
 path=tmp_path/"collection.json";path.write_text(json.dumps(index))
 with pytest.raises(ValueError):M.report_collection(path,tmp_path/"out")
 assert not (tmp_path/"out").exists()

def geometry_fixture(tmp_path):
 import json
 torch=M.torch;scene="ShopFacade";root=tmp_path/scene;root.mkdir()
 features=torch.ones((1,256),dtype=torch.float32)/16
 def write_map(path,xyz,provenance):
  path.parent.mkdir(parents=True,exist_ok=True)
  torch.save(dict(schema=K.SCHEMA,anchor_ids=torch.tensor([0]),anchor_xyz=torch.tensor([xyz],dtype=torch.float32),
   anchor_features=features.clone(),provenance=provenance),path)
 full=root/"source_full.pt";write_map(full,[0,0,1],{})
 source=M.C.evidence(full);folder=root/"geometry"/scene;folder.mkdir(parents=True)
 common=folder/"common_observations.npz"
 np.savez(common,source_rows=np.array([0]),offsets=np.array([0,3]),query_indices=np.array([0,1,2]),keypoint_indices=np.array([0,0,0]))
 cref=M.C.evidence(common);maps={}
 for i,mode in enumerate(M.CONFIGS):
  path=folder/mode/"deployment_map.pt"
  write_map(path,[i,0,1],dict(common_observations_sha256=cref["sha256"],source_full_sha256=source["sha256"]))
  maps[mode]=dict(M.C.evidence(path),count=1)
 (folder/"manifest.json").write_text(json.dumps(dict(complete=True,maps=maps,common_observations=cref,common_count=1)))
 (root/"protocol.json").write_text("{}")
 p=dict(scene_scope=[scene],sources={scene:dict(full=source)})
 return root,p

def test_single_scene_seal_has_exactly_three_maps_and_no_global_mutation(tmp_path,monkeypatch):
 root,p=geometry_fixture(tmp_path);monkeypatch.setattr(M,"protocol",lambda root:p)
 M.seal(root)
 ready=M.C.read(root/"preflight.json")
 assert ready["new_maps"]==3 and ready["scene_scope"]==["ShopFacade"]
 for mode in M.CONFIGS:assert M.checked_geometry(root,"ShopFacade",mode)["count"]==1
 assert len(K.C.SCENES)==5 and len(M.C.SCENES)==5
 with pytest.raises(ValueError):M.checked_geometry(root,"StMarysChurch","ray")

def test_seal_rejects_descriptor_changed_from_full_source(tmp_path,monkeypatch):
 root,p=geometry_fixture(tmp_path);monkeypatch.setattr(M,"protocol",lambda root:p)
 folder=root/"geometry"/"ShopFacade";info=M.C.read(folder/"manifest.json")
 path=folder/"depth"/"deployment_map.pt";state=M.C.load(path)
 state["anchor_features"]=M.torch.zeros((1,256));state["anchor_features"][0,0]=1
 M.torch.save(state,path);info["maps"]["depth"]=dict(M.C.evidence(path),count=1)
 import json
 (folder/"manifest.json").write_text(json.dumps(info))
 with pytest.raises(ValueError,match="descriptor differs"):M.seal(root)
 assert not (root/"preflight.json").exists()

