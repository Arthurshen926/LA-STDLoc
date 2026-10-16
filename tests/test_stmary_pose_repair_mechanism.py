import copy,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import experiments.stmary_pose_repair.mechanism as S

def accepted():
    return ({"complete":True,"seed":0,"iterations":30000,
             "recipe_matches_original_except_source_model_gpu":True,
             "ply":{"path":"/future/accepted.ply","sha256":"bound","bytes":10}},
            {"complete":True,"cuda_initialized":False,"mapping_count":1487,"test_count":530},
            {"complete":True,"verdict":"PASS"})

def test_release_gates_and_no_placeholder():
    t,c,r=accepted();S.check_acceptance(t,c,r)
    for key,value in [("complete",False),("seed",2026),("iterations",10000),
                      ("recipe_matches_original_except_source_model_gpu",False),("ply",{})]:
        bad=copy.deepcopy(t);bad[key]=value
        with pytest.raises(ValueError):S.check_acceptance(bad,c,r)
    with pytest.raises(ValueError):S.check_acceptance(t,c,{"complete":True,"verdict":"FAIL"})
    with pytest.raises(ValueError):S.check_acceptance(t,{**c,"test_count":529},r)

def test_missing_prior_does_not_freeze_protocol(tmp_path):
    with pytest.raises(FileNotFoundError):S.freeze(tmp_path)
    assert not (tmp_path/"mechanism").exists()

def test_training_acceptance_binds_review_cpu_outcome_and_sources(tmp_path):
    reviewed=tmp_path/"reviewed.txt";reviewed.write_text("canonical cameras")
    trainer=tmp_path/"trainer.py";trainer.write_text("seed0")
    review={"reviewed_paths_sha256":{str(reviewed):S.sha256(reviewed)}}
    (tmp_path/"training_acceptance.json").write_text("{}")
    names={"cpu_validation.json":"cpu_validation_sha256","independent_review.json":"independent_review_sha256",
           "training_launch.json":"training_launch_sha256","training_outcome.json":"training_outcome_sha256"}
    training={"source_hashes":{str(trainer):S.sha256(trainer)}}
    for name,key in names.items():
        p=tmp_path/name;p.write_text(json.dumps({"returncode":0}) if name=="training_outcome.json" else "{}")
        training[key]=S.sha256(p)
    refs=S.bind_release_evidence(tmp_path,training,review)
    assert "reviewed:"+str(reviewed) in refs
    reviewed.write_text("changed cameras")
    with pytest.raises(ValueError):S.bind_release_evidence(tmp_path,training,review)
    reviewed.write_text("canonical cameras")
    (tmp_path/"independent_review.json").write_text("changed PASS")
    with pytest.raises(ValueError):S.bind_release_evidence(tmp_path,training,review)

def test_file_binding_detects_change(tmp_path):
    p=tmp_path/"x";p.write_bytes(b"old");ref=S.evidence(p)
    assert S.unchanged(ref,True)==p
    p.write_bytes(b"different")
    with pytest.raises(ValueError):S.unchanged(ref,True)

def test_query_result_blocks_budget_freeze(tmp_path):
    d=tmp_path/"evaluations/old_native";d.mkdir(parents=True);(d/"results.json").write_text("[]")
    with pytest.raises(ValueError):S.no_query_outputs(tmp_path)

def test_old_budget_comes_from_actual_native_rows(tmp_path,monkeypatch):
    (tmp_path/"protocol.json").write_text("{}")
    state={"anchor_xyz":torch.arange(24).reshape(8,3).float(),
           "anchor_features":torch.nn.functional.normalize(torch.ones(8,256),dim=1)}
    seen=[]
    def selector(full,cache):
        seen.append((full,cache));return torch.tensor([2,5]),{"selected":torch.tensor([2,5])}
    monkeypatch.setattr(S,"protocol",lambda root:{})
    monkeypatch.setattr(S,"old_inputs",lambda root:({"full":{"sha256":"fresh-repaired-full"}},state,{"mapping":"only"}))
    monkeypatch.setattr(S.A,"select_compact_rows_v2",selector)
    S.select_old(tmp_path)
    b=json.loads((tmp_path/"budget.json").read_text())
    assert b["B"]==2 and b["query_results_present"] is False
    assert seen[0][0] is state
    assert np.array_equal(np.load(tmp_path/"maps/old_native/deployment_map.source_rows.npy"),[2,5])
    assert S.budget(tmp_path)==2
    with pytest.raises(FileExistsError):S.select_old(tmp_path)
    b["B"]=3;(tmp_path/"budget.json").write_text(json.dumps(b))
    with pytest.raises(ValueError):S.budget(tmp_path)

def test_scalar_record_bitwise_matches_v3_pure_gs_extraction(tmp_path,monkeypatch):
    import anygsloc.features.extractor as ext
    import anygsloc.priors.models as models
    import anygsloc.priors.rendering as rendering
    import anygsloc.mapping.virtual_camera_registry as registry
    M=S.M
    camera=SimpleNamespace(image_name="mapping.png",pose_w2c=np.eye(4,dtype=np.float32),
                           fov_x=1.,fov_y=1.,height=16,width=16)
    K=torch.tensor([[10.,0,8],[0,10.,8],[0,0,1.]])
    package={"render":torch.linspace(0,1,3*16*16).reshape(3,16,16),
             "alphas":torch.ones(1,16,16),"depth":torch.ones(1,16,16)*2.}
    sparse={"keypoints":torch.tensor([[4.,4.],[8.,8.],[11.,11.]]),
            "descriptors":torch.arange(1,769).reshape(3,256).float(),
            "keypoint_scores":torch.tensor([.8,.6,.4])}
    class Model:
        def __init__(self,*a,**k):pass
        def load_ply(self,*a,**k):return self
        def to(self,*a,**k):return self
        def eval(self):return self
    class Detector(Model):
        def detectAndCompute(self,image,top_k,detection_threshold):
            assert top_k==2048 and detection_threshold==0.
            return [{k:v.clone() for k,v in sparse.items()}]
    class Dataset:
        def __init__(self,*a,**k):pass
        def split(self,name):
            assert name=="mapping";return [camera]
        def load_image(self,*a):raise AssertionError("unexpected actual image read")
    monkeypatch.setattr(models,"GaussianModel3D",Model)
    monkeypatch.setattr(ext,"FeatureExtractor",lambda *a,**k:Detector())
    monkeypatch.setattr(rendering,"render_from_pose_gsplat",lambda *a,**k:package)
    monkeypatch.setattr(registry,"camera_intrinsic",lambda c:K.clone())
    monkeypatch.setattr(M,"ColmapDataset",Dataset)
    monkeypatch.setattr(M,"resolve_virtual_camera_registry",lambda cameras,reg:cameras)
    monkeypatch.setattr(M,"safe_mapping_image",lambda *a:torch.zeros(3,16,16))
    ply=tmp_path/"prior.ply";ply.write_bytes(b"trusted-mock-prior")
    root=tmp_path/"v3";root.mkdir();(root/"protocol.json").write_text("{}")
    s={"dataset":"unused","registry":{},"mapping_names":[camera.image_name],"test_names":["held_out.png"],
       "prior":{"ply":str(ply),"ply_sha256":S.sha256(ply),"gaussian_type":"3dgs"}}
    monkeypatch.setattr(M,"protocol",lambda root:{"scenes":{S.SCENE:s}})
    before=torch.cuda.is_initialized()
    M._extract(root,S.SCENE,torch.device("cpu"))
    expected=S.A.load(root/"observations"/S.SCENE/"pure_gs.pt")["queries"][camera.image_name]
    actual=S.scalar_record(camera,torch.from_numpy(camera.pose_w2c).float(),package,sparse)
    assert actual.keys()==expected.keys()
    for k in actual:assert torch.equal(actual[k],expected[k]),k
    assert actual["native_alpha_at_keypoints"].dtype==torch.float32
    assert actual["row_source"].eq(1).all()
    assert torch.cuda.is_initialized()==before

def test_only_six_frozen_cells_and_pure_gs():
    assert len(S.CONFIGS)==6
    assert S.CHUNK_SIZE==256
    assert S.CONFIGS[-1]=="classic_spatial_B"
    assert "raw_only" not in S.DESIGN["mapping_input"] and "hybrid" not in S.DESIGN["mapping_input"]

def test_invalid_chunk_never_initializes_cuda(tmp_path,monkeypatch):
    monkeypatch.setattr(S,"protocol",lambda root:{"scenes":{S.SCENE:{"mapping_names":["m"]}}})
    with pytest.raises(ValueError):S.extract_chunk(tmp_path,0,"cpu")

def test_evaluation_reference_matches_runner_dry_run(tmp_path,monkeypatch,capsys):
    from anygsloc.evaluation import runner
    mapping=[SimpleNamespace(width=1280,height=720,fov_x=.9,fov_y=.7)]
    names=["first.png","second.png"]
    tests=[SimpleNamespace(image_name=n) for n in names]
    class Dataset:
        def __init__(self,root,**kwargs):self.root=Path(root).resolve()
        def split(self,split):return mapping if split=="mapping" else tests
    monkeypatch.setattr(runner,"ColmapDataset",Dataset)
    map_file=tmp_path/"map.pt";map_file.write_bytes(b"dry-run-only")
    runner.main(["--dataset",str(tmp_path),"--map",str(map_file),"--output",str(tmp_path/"unused"),
                 "--image-domain","rectified","--images","processed","--mask-policy","explicit_valid_mask",
                 "--rectification-contract","opencv_same_k","--seed","2026","--dry-run"])
    actual=json.loads(capsys.readouterr().out)
    expected=S.evaluation_reference(tmp_path,mapping,names)
    assert expected["queries"]==names
    assert all(actual[k]==v for k,v in expected.items())
    assert expected["config"]["keypoint_count"]==1024
    assert not (tmp_path/"unused").exists()

def test_chunk_publication_is_atomic_and_exclusive(tmp_path,monkeypatch):
    path=tmp_path/"000000.pt"
    original=S.torch.save
    def interrupted(value,target):
        Path(target).write_bytes(b"partial")
        raise RuntimeError("interrupted")
    monkeypatch.setattr(S.torch,"save",interrupted)
    with pytest.raises(RuntimeError):S.atomic_save_new({"complete":True},path)
    assert not path.exists() and not list(tmp_path.glob("*.tmp"))
    monkeypatch.setattr(S.torch,"save",original)
    S.atomic_save_new({"complete":True},path)
    h=S.sha256(path)
    with pytest.raises(FileExistsError):S.atomic_save_new({"complete":False},path)
    assert S.sha256(path)==h and S.A.load(path)["complete"] is True

def test_chunk_identity_and_incomplete_chunk_sets_are_rejected(tmp_path,monkeypatch):
    path=tmp_path/"chunk.pt";names=["mapping.png"]
    data={"protocol_sha256":"current","start":0,"names":names,"queries":{names[0]:{"row":1}}}
    S.atomic_save_new(data,path)
    assert S.load_chunk(path,names,0,"current")==data["queries"]
    for expected_names,start,ph in [(["test.png"],0,"current"),(names,256,"current"),(names,0,"old")]:
        with pytest.raises(ValueError):S.load_chunk(path,expected_names,start,ph)
    corrupt=tmp_path/"corrupt.pt";corrupt.write_bytes(b"incomplete torch serialization")
    with pytest.raises(Exception):S.load_chunk(corrupt,names,0,"current")
    monkeypatch.setattr(S,"protocol",lambda root:{"scenes":{S.SCENE:{"mapping_names":names}}})
    with pytest.raises(ValueError,match="missing or extra"):S.finalize_cache(tmp_path)
    folder=tmp_path/"new_observations/chunks";folder.mkdir(parents=True)
    S.atomic_save_new(data,folder/"000000.pt");S.atomic_save_new(data,folder/"000256.pt")
    with pytest.raises(ValueError,match="missing or extra"):S.finalize_cache(tmp_path)
    assert not (tmp_path/"new_observations/pure_gs.pt").exists()

def test_full_map_requires_repaired_protocol_cache_and_constructor():
    p={"protocol_sha256":"repaired","cache_sha256":"new-cache","method":"anygsloc",
       "input_mode":"pure_gs","uses_source_mapping_rgb":False,"uses_test_queries":False}
    S.require_full_binding({"provenance":p},"anygsloc","repaired","new-cache")
    for key,value in [("protocol_sha256","old-v3"),("cache_sha256","old-cache"),("method","classic"),
                      ("input_mode","hybrid"),("uses_source_mapping_rgb",True),("uses_test_queries",True)]:
        with pytest.raises(ValueError):S.require_full_binding({"provenance":{**p,key:value}},"anygsloc","repaired","new-cache")

