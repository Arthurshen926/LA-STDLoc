import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.cambridge_mainline_ablations import accuracy_scheduler as S

def result(start=0,end=10):
 return {"rows":[{"name":"q","success":True,"pose":np.eye(4).tolist(),"inliers":[1,2],
  "matched_keypoints_sha256":"a","matched_anchors_sha256":"b"}],
  "start":start,"end":end,"elapsed":end-start,"torch_max_reserved_bytes":4*1024**3}
def decision(**kw):
 args=dict(serial=[result(0,10),result(10,20)],parallel=[result(0,12),result(0,12)],
  peak_bytes=10*1024**3,baseline_bytes=0,free_cores=20,ram=32*1024**3,covered=True)
 args.update(kw);return S.decide(**args)
def test_parallel_pass_requires_all_frozen_gates():
 r=decision();assert r["concurrency"]==2 and all(r["gates"].values())
@pytest.mark.parametrize("override",[
 {"peak_bytes":20*1024**3},{"baseline_bytes":13*1024**3},{"free_cores":11},
 {"ram":15*1024**3},{"covered":False},
 {"parallel":[result(0,19),result(0,19)]}])
def test_resource_or_speed_failure_is_serial(override):
 assert decision(**override)["concurrency"]==1
@pytest.mark.parametrize("field",["success","inliers","matched_keypoints_sha256","matched_anchors_sha256"])
def test_discrete_prediction_change_is_serial(field):
 a=result();b=copy.deepcopy(a);b["rows"][0][field]={"success":False,"inliers":[],"matched_keypoints_sha256":"x","matched_anchors_sha256":"x"}[field]
 assert not S.same_predictions(a,b)
def test_nonfinite_and_material_pose_difference_fail():
 a=result();b=copy.deepcopy(a);b["rows"][0]["pose"][0][3]=float("nan")
 assert not S.same_predictions(a,b)
 b=copy.deepcopy(a);b["rows"][0]["pose"][0][3]=1e-4
 assert not S.same_predictions(a,b)
def test_query_subset_uses_order_and_dimensions_only():
 cameras=[SimpleNamespace(image_name="q"+str(i),ignored_gt=i*999) for i in range(12)]
 sizes={c.image_name:[100,100] for c in cameras};sizes["q11"]=[200,100]
 assert S.choose_names(cameras,sizes)==["q"+str(i) for i in range(7)]+["q11"]
 for c in cameras:c.ignored_gt=-123
 assert S.choose_names(cameras,sizes)==["q"+str(i) for i in range(7)]+["q11"]
def test_finite_15_cells_and_no_classic():
 for table in ("T5","T6"):
  cells=S.finite_cell_order(table)
  assert len(cells)==15 and len(set(cells))==15
  assert all(scene in S.C.SCENES and "classic" not in mode for mode,scene in cells)
def test_child_command_keeps_frozen_evaluator():
 command=S.query_command("T5","/tmp/source","ray","ShopFacade")
 assert command[2:4]==["-m","experiments.cambridge_mainline_ablations.t5"]
 assert command[-2:]==["--config","ray"]
 with pytest.raises(ValueError):S.query_command("T5","/tmp/source","arbitrary","ShopFacade")
def test_environment_isolated_four_threads():
 env=S.worker_env()
 assert env["CUDA_VISIBLE_DEVICES"]=="0"
 for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):assert env[k]=="4"
def test_empty_parallel_gates_cannot_authorize(tmp_path,monkeypatch):
 monkeypatch.setattr(S,"source",lambda root:{})
 monkeypatch.setattr(S,"require_review",lambda root:None)
 S.C.write_new(tmp_path/"protocol.json",{})
 S.C.write_new(tmp_path/"pilot_result.json",{"complete":True,"protocol_sha256":S.C.sha256(tmp_path/"protocol.json"),"concurrency":2,"gates":{}})
 with pytest.raises(ValueError,match="failed parallel gate"):S.checked_policy(tmp_path)
def test_execution_review_binds_pilot_receipt(tmp_path,monkeypatch):
 monkeypatch.setattr(S,"source",lambda root:{})
 monkeypatch.setattr(S,"require_review",lambda root:None)
 S.C.write_new(tmp_path/"protocol.json",{})
 S.C.write_new(tmp_path/"pilot_result.json",{"complete":True,"protocol_sha256":S.C.sha256(tmp_path/"protocol.json"),"concurrency":1})
 S.C.write_new(tmp_path/"independent_execution_review.json",{"verdict":"PASS","protocol_sha256":S.C.sha256(tmp_path/"protocol.json"),"pilot_result_sha256":S.C.sha256(tmp_path/"pilot_result.json")})
 assert S.checked_policy(tmp_path)["concurrency"]==1
 (tmp_path/"pilot_result.json").write_text((tmp_path/"pilot_result.json").read_text()+" ")
 with pytest.raises(ValueError,match="execution review"):S.checked_policy(tmp_path)
def test_managed_children_use_distinct_processes_and_are_waited(tmp_path,monkeypatch):
 launched=[]
 class Child:
  def __init__(self,command,**kwargs):
   self.folder=S.Path(command[-1]);self.returncode=None;self.waited=False
   S.C.write_new(self.folder/"ready.json",{"pid":100+len(launched)})
   S.C.write_new(self.folder/"result.json",result());launched.append(self)
  def poll(self):return self.returncode
  def wait(self,timeout=None):
   assert (self.folder/"go").exists();self.waited=True;self.returncode=0;return 0
  def terminate(self):self.returncode=-15
  def kill(self):self.returncode=-9
 monkeypatch.setattr(S.subprocess,"Popen",Child)
 rows=S.managed_group(tmp_path,[tmp_path/"p0",tmp_path/"p1"])
 assert len(rows)==2 and len(launched)==2 and all(c.waited for c in launched)
def pilot_fixture():
 names=["q"+str(i) for i in range(8)]
 p={"pilot":{"configuration":"ray","scene":"ShopFacade","map":{"sha256":"map"},
  "query_names":names,"expected_config":{"a":1},"expected_calibration":{"b":2}},
  "query_snapshots":{"ShopFacade":{"sha256":"snapshot"}}}
 r=result();r["rows"]=[dict(r["rows"][0],name=n) for n in names*2]
 r.update(protocol_sha256="protocol",map_sha256="map",query_inputs_sha256="snapshot",
  configuration="ray",scene="ShopFacade",config={"a":1},calibration={"b":2},
  query_names=names,seed=2026,torch_threads=4,queries=16)
 return p,r
def test_empty_rows_are_never_deterministic_evidence():
 a=result();a["rows"]=[]
 assert not S.same_predictions(a,a)
 p,r=pilot_fixture();r["rows"]=[]
 with pytest.raises(ValueError,match="coverage"):S.validate_pilot_record(r,p,"protocol")
@pytest.mark.parametrize("field",["map_sha256","scene","configuration","query_inputs_sha256","protocol_sha256"])
def test_wrong_pilot_identity_rejected(field):
 p,r=pilot_fixture();r[field]="wrong"
 with pytest.raises(ValueError,match="identity"):S.validate_pilot_record(r,p,"protocol")
def test_pilot_order_and_count_exact():
 p,r=pilot_fixture();S.validate_pilot_record(r,p,"protocol")
 r["rows"]=r["rows"][::-1]
 with pytest.raises(ValueError,match="coverage"):S.validate_pilot_record(r,p,"protocol")
def test_replaced_raw_pilot_evidence_rejected(tmp_path):
 p,r=pilot_fixture();path=tmp_path/"pilot/serial0/result.json";S.C.write_new(path,r)
 receipt={"concurrency":1,"evidence":{"pilot/serial0/result.json":S.C.evidence(path)}}
 path.write_text("{}")
 with pytest.raises(ValueError,match="hash"):S.checked_pilot_evidence(tmp_path,p,receipt)
def test_pilot_evidence_wrong_target_rejected(tmp_path):
 p,r=pilot_fixture();path=tmp_path/"another.json";S.C.write_new(path,r)
 receipt={"concurrency":1,"evidence":{"pilot/serial0/result.json":S.C.evidence(path)}}
 with pytest.raises(ValueError,match="path"):S.checked_pilot_evidence(tmp_path,p,receipt)
def test_current_resources_lower_concurrency_and_foreign_gpu_pauses(monkeypatch):
 monkeypatch.setattr(S,"gpu_process_ids",lambda:set())
 monkeypatch.setattr(S,"cpu_free",lambda:20)
 monkeypatch.setattr(S,"memory_available",lambda:32*1024**3)
 assert S.current_admission(2)["concurrency"]==2
 monkeypatch.setattr(S,"cpu_free",lambda:5)
 assert S.current_admission(2)["concurrency"]==1
 monkeypatch.setattr(S,"gpu_process_ids",lambda:{123})
 with pytest.raises(RuntimeError,match="foreign"):S.current_admission(2)
 assert S.current_admission(2,owned={123})["concurrency"]==1
def test_no_cuda_on_import():
 assert not torch.cuda.is_initialized()
