"""CPU-only benchmark contract tests; never instantiate a CUDA Localizer."""
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments import cambridge_deployment_benchmark as B

class Clock:
    def __init__(self): self.t=0.
    def __call__(self): return self.t
    def add(self,x): self.t+=x
class Data:
    def __init__(self,c): self.c=c;self.images=[];self.masks=[]
    def load_image(self,cam): self.images.append(cam.image_name);self.c.add(.002);return cam.image_name
    def valid_mask(self,cam): self.masks.append(cam.image_name);self.c.add(.003);return np.ones((2,2))
class Engine:
    def __init__(self,c):self.c=c;self.calls=[]
    def localize(self,image,**kwargs):
        self.calls.append((image,kwargs));self.c.add(.005)
        return {"runtime_ms":5.,"success":image!="b","pose":SimpleNamespace(pose_w2c=np.eye(4))}
def cameras():
    return [SimpleNamespace(image_name=n,fov_x=1.,fov_y=1.,pose_w2c=np.eye(4)) for n in ("a","b")]
def test_round_real_calls_boundaries_and_failures(monkeypatch):
    c=Clock();d=Data(c);e=Engine(c)
    def metric(*args):c.add(1.);return 0.,0.
    monkeypatch.setattr(B,"pose_error",metric)
    r=B.benchmark_round(d,cameras(),e,clock=c,sync=lambda:c.add(.007))
    assert len(e.calls)==52 and len(d.images)==52 and len(d.masks)==52
    assert r["warmup_names"]==["a","b"]*25
    assert [v["image_name"] for v in r["queries"]]==["a","b"]
    assert r["accuracy"]["failed_query_count"]==1
    for row in r["queries"]:
        assert row["runtime_ms"]==5
        assert row["frame_ms"]==pytest.approx(17)
        assert row["rgb_decode_preprocess_ms"]==pytest.approx(2)
        assert row["mask_compose_ms"]==pytest.approx(3)
    assert r["timing"]["runtime_ms"]["count"]==2
    assert all(call[1]["valid_mask"] is not None for call in e.calls)
def test_missing_mask_fails_closed():
    c=Clock();d=Data(c);d.valid_mask=lambda cam:None
    with pytest.raises(ValueError,match="mask missing"):
        B.benchmark_round(d,cameras(),Engine(c),warmup=0,clock=c)
def test_empty_query_and_bad_timing_rejected():
    with pytest.raises(ValueError):B.benchmark_round(None,[],None)
    with pytest.raises(ValueError):B.stats([])
    with pytest.raises(ValueError):B.stats([{k:float("nan") for k in B.TIMINGS}])
def release():
    return {"approved":True,"scope":"cambridge_full_compact_formal_timing","protocol_sha256":"bound",
            "gpu_uuid":"GPU-actual","approval_reference":"parent authorization","valid_from_unix":100,"valid_until_unix":200}
@pytest.mark.parametrize("key,value",[("approved",False),("scope","different"),("protocol_sha256","other"),
 ("gpu_uuid","cuda:0"),("approval_reference",""),("valid_from_unix",151),("valid_until_unix",149)])
def test_release_is_bounded(key,value):
    r=release();r[key]=value
    with pytest.raises(ValueError):B.check_release(r,"bound",150)
def test_valid_release_and_no_unbounded_window():
    assert B.check_release(release(),"bound",150)=="GPU-actual"
    r=release();r["valid_until_unix"]=10000
    with pytest.raises(ValueError):B.check_release(r,"bound",150)
def test_target_gpu_only_and_monitor_contamination(monkeypatch):
    sample={"compute_apps_csv":"GPU-a, 7, python, 100\nGPU-b, 8, python, 200"}
    assert B.target_pids(sample,"GPU-a")=={7}
    m=B.Monitor("GPU-a",{6})
    def probe():m.stop.set();return sample
    monkeypatch.setattr(B,"hardware",probe)
    m.work()
    with pytest.raises(RuntimeError,match="isolation lost"):m.check()
def test_changed_artifact_detected(tmp_path):
    p=tmp_path/"map";p.write_bytes(b"old");ref={"path":str(p),"bytes":3,"sha256":B.sha256(p)}
    B.check_ref(ref);p.write_bytes(b"new")
    with pytest.raises(ValueError):B.check_ref(ref)
def test_unchanged_localizer_and_cpu_only_import():
    from anygsloc.online.runtime import Localizer
    assert B.Localizer is Localizer
    assert B.DESIGN["warmup_per_round"]==50 and B.DESIGN["rounds"]==3
    assert B.DESIGN["query_semantics"] is True
    assert not torch.cuda.is_initialized()
def fixture_cells(tmp_path,monkeypatch):
    names=["a","b"];scenes=("one","two");info={}
    monkeypatch.setattr(B.F,"SCENES",scenes)
    (tmp_path/"protocol.json").write_text("{}");psha=B.sha256(tmp_path/"protocol.json")
    for scene in scenes:
        B.write(tmp_path/"query_inputs"/(scene+".json"),{})
        qsha=B.sha256(tmp_path/"query_inputs"/(scene+".json"))
        info[scene]={"scene_info":{"test_names":names},"maps":{}}
        for cfg in B.CONFIGS:
            ref={"path":scene+cfg,"sha256":"maphash","bytes":20 if scene=="one" else 40,"count":3}
            info[scene]["maps"][cfg]=ref
            out=tmp_path/"runs"/scene/cfg;round_refs={};allrows=[]
            for i in range(3):
                rows=[{"image_name":n,**{k:10. if scene=="one" else 30. for k in B.TIMINGS}} for n in names]
                B.write(out/f"round{i}.json",{"round":i,"queries":rows,"warmup_names":names*25})
                round_refs[str(i)]=B.S.evidence(out/f"round{i}.json");allrows+=rows
            rel={**release(),"protocol_sha256":psha,"gpu_uuid":"GPU-a"}
            B.write(out/"release_used.json",rel);rref=B.S.evidence(out/"release_used.json")
            identity={"protocol_sha256":psha,"scene":scene,"configuration":cfg,"query_inputs_sha256":qsha,"map":ref}
            B.write(out/"initialization.json",{**identity,"before":{"unix":150},"release":rref})
            B.write(out/"telemetry.json",[])
            B.write(out/"summary.json",{"complete":True,**identity,
              "round_files":round_refs,"gpu_uuid":"GPU-a","sample_count":6,"timing":B.stats(allrows),
              "initialization":B.S.evidence(out/"initialization.json"),"telemetry":B.S.evidence(out/"telemetry.json"),
              "release":rref})
    p={"scenes":info};monkeypatch.setattr(B,"protocol",lambda root:p);return p
def test_three_round_report_uses_scene_equal_and_pooled(tmp_path,monkeypatch):
    fixture_cells(tmp_path,monkeypatch);B.report(tmp_path);r=B.A.read(tmp_path/"report.json")
    for cfg in B.CONFIGS:
        assert r["results"][cfg]["scene_equal_mean_ms"]["frame_ms"]==20
        assert r["results"][cfg]["pooled"]["frame_ms"]["count"]==12
        assert r["results"][cfg]["total_map_bytes"]==60
@pytest.mark.parametrize("key,value",[("scene","two"),("configuration","compact"),("query_inputs_sha256","wrong")])
def test_wrong_cell_identity_rejected(tmp_path,monkeypatch,key,value):
    p=fixture_cells(tmp_path,monkeypatch);path=tmp_path/"runs/one/full/summary.json";s=B.A.read(path)
    s[key]=value;path.write_text(__import__("json").dumps(s))
    with pytest.raises(ValueError):B.checked_cell(tmp_path,"one","full",p)
def test_verified_round_cannot_point_elsewhere(tmp_path,monkeypatch):
    p=fixture_cells(tmp_path,monkeypatch);path=tmp_path/"runs/one/full/summary.json";s=B.A.read(path)
    s["round_files"]["0"]=B.S.evidence(tmp_path/"runs/two/full/round0.json");path.write_text(__import__("json").dumps(s))
    with pytest.raises(ValueError,match="path differs"):B.checked_cell(tmp_path,"one","full",p)
def test_exact_subset_and_corruption_rejected():
    f={"anchor_xyz":torch.arange(9).reshape(3,3),"anchor_features":torch.arange(12).reshape(3,4)}
    c={k:v[[0,2]].clone() for k,v in f.items()};B.verify_subset(f,c,np.array([0,2]))
    c["anchor_features"][0,0]=100
    with pytest.raises(ValueError):B.verify_subset(f,c,np.array([0,2]))
    with pytest.raises(ValueError):B.verify_subset(f,c,np.array([0,0]))
@pytest.mark.parametrize("existing",["summary.json","failure.json"])
def test_existing_output_never_mutated(tmp_path,monkeypatch,existing):
    (tmp_path/"protocol.json").write_text("{}")
    rel={**release(),"protocol_sha256":B.sha256(tmp_path/"protocol.json"),"valid_from_unix":B.time.time()-10,
         "valid_until_unix":B.time.time()+100}
    B.write(tmp_path/"release.json",rel);out=tmp_path/"runs/one/full";out.mkdir(parents=True)
    (out/existing).write_text("untouched")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES",rel["gpu_uuid"])
    monkeypatch.setattr(B,"protocol",lambda root:{"scenes":{"one":{}}})
    with pytest.raises(FileExistsError):B.evaluate(tmp_path,"one","full",tmp_path/"release.json")
    assert {x.name:x.read_text() for x in out.iterdir()}=={existing:"untouched"}
    assert not (tmp_path/"benchmark.lock").exists()
