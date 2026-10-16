"""Small deterministic CPU contracts; no real caches, PLY, CUDA, or dataset access."""
import copy
from types import SimpleNamespace
import pytest
import torch
from experiments.cambridge_gs_assisted_real import contracts as C, core
from experiments.cambridge_input_matrix.contracts import ASSOCIATION_GRAPH_SCHEMA

def example():
    names=["a","b","c"];records={};aux={}
    K=torch.tensor([[100.,0,50.],[0,100.,50.],[0,0,1.]])
    points=torch.tensor([[.2,.3,6.],[.125,.125,5.125],[.4,.4,7.]])
    for name,tx in zip(names,[-1.,0.,1.]):
        pose=torch.eye(4);pose[0,3]=-tx
        camera=points+pose[:3,3]
        xy=camera[:,:2]/camera[:,2:]*100+50-.5
        r=dict(native_keypoints=xy,native_descriptors=torch.eye(3),native_scores=torch.ones(3),
            native_K=K.clone(),pose_w2c=pose,native_input_hw=torch.tensor([100,100]),
            row_source=torch.zeros(3,dtype=torch.long),source_row=torch.arange(3))
        records[name]=r
        aux[name]={**{k:r[k].clone() for k in C.GEOMETRY},"native_keypoints":xy.clone(),
            "depth":points[:,2].clone(),"alpha":torch.ones(3),"certificate_valid":torch.tensor([False,True,False])}
    raw=dict(input_mode="raw_only",uses_source_mapping_rgb=True,uses_test_queries=False,
        mapping_names=names,test_names=["heldout"],queries=records)
    auxiliary=dict(mapping_names=names,uses_test_queries=False,queries=aux)
    csr=dict(observation_offsets=torch.tensor([0,3]),query_indices=torch.arange(3),
        keypoint_indices=torch.zeros(3,dtype=torch.long),keypoint_index_semantics="original_unfiltered_observation_cache_row")
    base=dict(anchor_ids=torch.arange(1),anchor_xyz=points[:1].clone(),anchor_features=torch.eye(3)[:1].clone(),
        anchor_position_covariance=torch.eye(3)[None]*.01,anchor_matchability=torch.tensor([.8]),
        v6_mapping_query_names=names,v6_mapping_query_bins=torch.arange(3),
        projective_anchor_observations=csr,provenance=dict(input_mode="raw_only",uses_source_mapping_rgb=True,uses_test_queries=False))
    association=dict(schema=ASSOCIATION_GRAPH_SCHEMA,input_mode="raw_only",
        uses_source_mapping_rgb=True,uses_test_queries=False,experimental_extension=True,
        query_names=names,query_bins=torch.arange(3),tracks=dict(query_index=torch.arange(3),
        keypoint_index=torch.zeros(3,dtype=torch.long)))
    full={k:base[k].clone() for k in ("anchor_xyz","anchor_features")}
    return raw,auxiliary,association,base,full

def test_all_invalid_exact_fallback_and_unfiltered_rows():
    raw,aux,graph,base,full=example()
    for r in aux["queries"].values():r["certificate_valid"].fill_(False)
    before=copy.deepcopy(raw)
    state,ev=core.construct(raw,aux,graph,base,full,"test","cpu")
    assert ev["completion"] is None
    for key in C.SCIENCE_FIELDS:core.assert_equal(base[key],state[key],key)
    core.assert_equal(before,raw,"raw input")
    payload=C.attach_auxiliary(raw,aux,"test")
    provider=C.AssistedObservationProvider(payload)
    for name in raw["mapping_names"]:
        assert len(payload["queries"][name]["native_keypoints"])==3
        assert int(payload["queries"][name]["native_alpha_at_keypoints"].sum())==0
        for key in ("native_keypoints","native_descriptors","native_scores"):
            assert torch.equal(raw["queries"][name][key],payload["queries"][name][key])

def test_invalid_aux_native_compact_identical():
    from anygsloc.mapping.compact_v2 import select_compact_rows_v2
    raw,aux,graph,base,full=example()
    for r in aux["queries"].values():r["certificate_valid"].fill_(False)
    state,_=core.construct(raw,aux,graph,base,full,"test","cpu")
    before,_=select_compact_rows_v2(base,raw["queries"])
    after,_=select_compact_rows_v2(state,raw["queries"])
    assert torch.equal(before,after)

def test_actual_completion_real_features_ray_and_prefix():
    raw,aux,graph,base,full=example()
    state,ev=core.construct(raw,aux,graph,base,full,"test","cpu")
    assert ev["completion"] is not None
    assert len(state["anchor_xyz"])==2
    assert ev["actual_budget"]==dict(raw_rows=9,eligible_unused_rows=3,capped_proposal_rows=3)
    core.assert_base_prefix(base,state)
    assert torch.allclose(state["anchor_xyz"][1],torch.tensor([.125,.125,5.125]),atol=1e-4)
    assert torch.allclose(state["anchor_features"][1],torch.tensor([0.,1.,0.]),atol=1e-6)
    assert state["projective_anchor_observations"]["keypoint_indices"].tolist()==[0,0,0,1,1,1]
    assert state["provenance"]["full_is_append_only"] is True
    assert state["provenance"]["compact_is_append_only"] is False
    assert ev["completion"]["contract"]["gaussian_depth_used_for_final_xyz"] is False

@pytest.mark.parametrize("kind",["render_row","source_free","test_input","test_name","same_camera","local_row","raw_gaussian"])
def test_reject_bad_raw(kind):
    raw,aux,*_=example()
    if kind=="render_row":raw["queries"]["a"]["row_source"][0]=1
    if kind=="source_free":raw["uses_source_mapping_rgb"]=False
    if kind=="test_input":raw["uses_test_queries"]=True
    if kind=="test_name":raw["test_names"]=["a"]
    if kind=="same_camera":raw["queries"]["a"]["pose_w2c"]=raw["queries"]["b"]["pose_w2c"].clone()
    if kind=="local_row":raw["queries"]["a"]["source_row"]=torch.tensor([1,0,2])
    if kind=="raw_gaussian":raw["queries"]["a"]["v2_valid"]=torch.ones(3,dtype=torch.bool)
    with pytest.raises(ValueError):C.attach_auxiliary(raw,aux,"test")

@pytest.mark.parametrize("kind",["camera","xy","length","mask_type","order","test_input"])
def test_reject_bad_aux(kind):
    raw,aux,*_=example()
    if kind=="camera":aux["queries"]["a"]["pose_w2c"][0,3]+=1
    if kind=="xy":aux["queries"]["a"]["native_keypoints"][0,0]+=1
    if kind=="length":aux["queries"]["a"]["depth"]=torch.ones(2)
    if kind=="mask_type":aux["queries"]["a"]["certificate_valid"]=torch.ones(3)
    if kind=="order":aux["mapping_names"]=list(reversed(aux["mapping_names"]))
    if kind=="test_input":aux["uses_test_queries"]=True
    with pytest.raises(ValueError):C.attach_auxiliary(raw,aux,"test")

@pytest.mark.parametrize("kind",["provenance","v2","valid_mask","drop_row"])
def test_provider_rejects_disguised_source_and_hard_filter(kind):
    raw,aux,*_=example();p=C.attach_auxiliary(raw,aux,"test")
    if kind=="provenance":p["uses_render_descriptors"]=True
    if kind=="v2":p["queries"]["a"]["v2_valid"]=torch.ones(3,dtype=torch.bool)
    if kind=="valid_mask":p["queries"]["a"]["native_valid_keypoint_mask"]=torch.ones(3,dtype=torch.bool)
    if kind=="drop_row":
        for key in ("native_keypoints","native_descriptors","native_scores","row_source","source_row"):
            p["queries"]["a"][key]=p["queries"]["a"][key][1:]
    with pytest.raises(ValueError):C.AssistedObservationProvider(p)

@pytest.mark.parametrize("key",["anchor_xyz","anchor_features","anchor_position_covariance","anchor_matchability"])
def test_base_scientific_prefix_mutation_rejected(key):
    *_,base,full=example();changed=copy.deepcopy(base);changed[key].reshape(-1)[0]+=1
    with pytest.raises(ValueError):core.assert_base_prefix(base,changed)

@pytest.mark.parametrize("kind",["used_row","duplicate_camera","bad_offset","bad_row","nan_quality","bad_cov","render_provenance"])
def test_bad_appended_evidence_rejected(kind):
    raw,aux,graph,base,full=example()
    _,ev=core.construct(raw,aux,graph,base,full,"test","cpu")
    c=copy.deepcopy(ev["completion"])
    if kind=="used_row":c["projective_anchor_observations"]["keypoint_indices"][0]=0
    if kind=="duplicate_camera":c["projective_anchor_observations"]["query_indices"][1]=0
    if kind=="bad_offset":c["projective_anchor_observations"]["observation_offsets"][-1]=2
    if kind=="bad_row":c["projective_anchor_observations"]["keypoint_indices"][0]=100
    if kind=="nan_quality":c["identity_reliability"][0]=float("nan")
    if kind=="bad_cov":c["anchor_position_covariance"][0,0,0]=-1
    if kind=="render_provenance":c["uses_render_descriptors"]=True
    with pytest.raises(ValueError):core.append_candidate(base,c,raw,graph,"test")

def test_invalid_aux_does_not_zero_raw_quality():
    raw,aux,graph,base,full=example()
    for r in aux["queries"].values():r["depth"].fill_(float("nan"))
    state,_=core.construct(raw,aux,graph,base,full,"test","cpu")
    assert torch.equal(state["anchor_matchability"],base["anchor_matchability"])

@pytest.mark.parametrize("kind",["visible","device","count","current","uuid"])
def test_resource_guard_fails_closed_without_cuda(kind,monkeypatch):
    from experiments.cambridge_gs_assisted_real import resource
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES",resource.PHYSICAL_UUID if kind!="visible" else "0,1,2")
    monkeypatch.setattr(torch.cuda,"device_count",lambda:2 if kind=="count" else 1)
    monkeypatch.setattr(torch.cuda,"current_device",lambda:1 if kind=="current" else 0)
    monkeypatch.setattr(resource,"physical_context_uuid",lambda:"wrong" if kind=="uuid" else resource.PHYSICAL_UUID)
    with pytest.raises(ValueError):resource.assert_gpu("cuda:1" if kind=="device" else "cuda:0")

def test_cpu_never_initialized_cuda():
    assert not torch.cuda.is_initialized()

def _write_json(path,value):
    import json
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

def _artifact_fixture(root):
    from experiments.cambridge_gs_assisted_real import artifacts as A
    from experiments.cambridge_gs_assisted_real.launch import reference
    (root/"protocol.json").write_text("{}")
    for selection,stage in (("full","build"),("native_compact","select")):
        mp=root/"maps"/"ShopFacade"/selection/"deployment_map.pt"
        mp.parent.mkdir(parents=True);mp.write_text("identical-content-on-purpose")
        mi=reference(mp);mi["count"]=1
        summary=dict(**A.identity(root,"ShopFacade",stage),complete=True,
            **{"full" if selection=="full" else "map":mi})
        sp=A.stage_path(root,"ShopFacade",stage);_write_json(sp,summary)
        _write_json(root/"execution"/(A.stage_name("ShopFacade",stage)+".json"),
            dict(**A.identity(root,"ShopFacade",stage),success=True,returncode=0,summary=reference(sp)))
        files={}
        for name in ("results.json","manifest.json","summary.json"):
            fp=root/"evaluations"/"ShopFacade"/selection/name
            _write_json(fp,{})
            files[name]=reference(fp)
        cell=dict(**A.identity(root,"ShopFacade","evaluate",selection),complete=True,map=mi,files=files)
        cp=A.stage_path(root,"ShopFacade","evaluate",selection);_write_json(cp,cell)
        _write_json(root/"execution"/(A.stage_name("ShopFacade","evaluate",selection)+".json"),
            dict(**A.identity(root,"ShopFacade","evaluate",selection),success=True,returncode=0,summary=reference(cp)))
    return A

def test_complete_cell_identity_ok(tmp_path):
    A=_artifact_fixture(tmp_path)
    assert A.checked_cell(tmp_path,"ShopFacade","full")["selection"]=="full"

@pytest.mark.parametrize("kind",["swap_cell","wrong_same_content_path","wrong_map","missing_receipt","failed_receipt"])
def test_stage_and_cell_identity_negative(tmp_path,kind):
    from experiments.cambridge_gs_assisted_real.launch import reference
    A=_artifact_fixture(tmp_path)
    cp=A.stage_path(tmp_path,"ShopFacade","evaluate","full")
    receipt=tmp_path/"execution"/"evaluate_ShopFacade_full.json"
    cell=A.read(cp)
    if kind=="swap_cell":cell=A.read(A.stage_path(tmp_path,"ShopFacade","evaluate","native_compact"))
    if kind=="wrong_same_content_path":cell["files"]["results.json"]=A.read(A.stage_path(tmp_path,"ShopFacade","evaluate","native_compact"))["files"]["results.json"]
    if kind=="wrong_map":cell["map"]=A.read(A.stage_path(tmp_path,"ShopFacade","evaluate","native_compact"))["map"]
    _write_json(cp,cell)
    r=A.read(receipt);r["summary"]=reference(cp)
    if kind=="failed_receipt":r["success"]=False
    _write_json(receipt,r)
    if kind=="missing_receipt":receipt.unlink()
    with pytest.raises((ValueError,FileNotFoundError)):A.checked_cell(tmp_path,"ShopFacade","full")

@pytest.mark.parametrize("kind",["prior","raw","scene","protocol"])
def test_auxiliary_identity_negative(tmp_path,kind):
    from experiments.cambridge_gs_assisted_real import artifacts as A
    (tmp_path/"protocol.json").write_text("{}")
    source={"raw_cache":{"sha256":"raw"}}
    aux=dict(**A.identity(tmp_path,"ShopFacade","sample"),raw_cache_sha256="raw",prior_sha256="gs")
    key={"prior":"prior_sha256","raw":"raw_cache_sha256","scene":"scene","protocol":"protocol_sha256"}[kind]
    aux[key]="wrong"
    with pytest.raises(ValueError):A.check_aux_lineage(aux,tmp_path,"ShopFacade",source,"gs")

def test_supervisor_timeout_drains_owned_child(tmp_path):
    import sys
    from experiments.cambridge_gs_assisted_real.launch import supervise
    result=supervise([sys.executable,"-c","import time;time.sleep(10)"],tmp_path/"sleep.log",seconds=.05)
    assert result["error"] is not None
    assert result["owned_worker_waited"] is True
    assert result["returncode"]!=0

def test_supervisor_signal_during_launch_drains(tmp_path,monkeypatch):
    import os,signal,sys
    from experiments.cambridge_gs_assisted_real import launch
    original=launch.subprocess.Popen
    def spawn(*args,**kwargs):
        child=original(*args,**kwargs)
        os.kill(os.getpid(),signal.SIGTERM)
        return child
    monkeypatch.setattr(launch.subprocess,"Popen",spawn)
    result=launch.supervise([sys.executable,"-c","import time;time.sleep(10)"],tmp_path/"signal.log",seconds=3)
    assert "pending supervisor signal" in result["error"]
    assert result["owned_worker_waited"] is True

def test_exclusive_stage_lock(tmp_path,monkeypatch):
    from experiments.cambridge_gs_assisted_real import launch
    monkeypatch.setattr(launch,"MIRROR",tmp_path)
    with launch.exclusive_lock("same"):
        with pytest.raises(BlockingIOError):
            with launch.exclusive_lock("same"):pass

def test_parent_exit_child_retains_lock(tmp_path):
    import fcntl,os,signal,subprocess,sys,time
    lock=tmp_path/"orphan_safe.lock"
    # The short child inherits the same flock open-file description. Parent exits abruptly.
    script=("import fcntl,os,subprocess,sys;"
        "f=open(sys.argv[1],'a+');fcntl.flock(f,fcntl.LOCK_EX);"
        "c=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.7)'],pass_fds=(f.fileno(),),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);"
        "os.write(1,str(c.pid).encode());os._exit(0)")
    p=subprocess.run([sys.executable,"-c",script,str(lock)],capture_output=True,text=True,timeout=3)
    pid=int(p.stdout)
    try:
        with lock.open("a+") as f:
            with pytest.raises(BlockingIOError):fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            until=time.monotonic()+3
            while True:
                try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    if time.monotonic()>until:raise
                    time.sleep(.02)
    finally:
        try:os.kill(pid,signal.SIGKILL)
        except ProcessLookupError:pass
