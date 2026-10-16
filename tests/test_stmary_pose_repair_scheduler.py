"""CPU-only scheduling parity: no rendering, training, or new evaluator."""
import copy, importlib.util, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.stmary_pose_repair import scheduler as T
S=T.S

def fixture(root,monkeypatch):
    spec=importlib.util.spec_from_file_location("stage_a_fixture",Path(__file__).with_name("test_cambridge_mechanism_stage_a.py"))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    data,proof,small=module.example()
    root.mkdir();(root/"protocol.json").write_text("{}")
    p={"code_sha256":S.code_hashes(),"prior":{"sha256":"fixture-prior"}}
    monkeypatch.setattr(S,"protocol",lambda r:p)
    cache=root/"new_observations/pure_gs.pt";cache.parent.mkdir()
    torch.save(data,cache)
    S.A.write_new(cache.parent/"summary.json",{"complete":True,"protocol_sha256":S.sha256(root/"protocol.json"),
        "prior_sha256":"fixture-prior","cache":S.evidence(cache)})
    folder=root/"new_full/anygsloc";folder.mkdir(parents=True)
    full={**small,"provenance":{"protocol_sha256":S.sha256(root/"protocol.json"),"cache_sha256":S.sha256(cache),
        "method":"anygsloc","input_mode":"pure_gs","uses_source_mapping_rgb":False,"uses_test_queries":False}}
    torch.save(full,folder/"deployment_map.pt")
    info=S.A.checked(folder/"deployment_map.pt")
    state=S.A.adapt_new_state(proof,data,full)
    state["provenance"]["mechanism_binding"]={"protocol_sha256":S.sha256(root/"protocol.json"),
        "cache_sha256":S.sha256(cache),"full_sha256":info["sha256"]}
    torch.save(state,folder/"selector_state.pt")
    S.A.write_new(folder/"summary.json",{"complete":True,"method":"anygsloc","protocol_sha256":S.sha256(root/"protocol.json"),
        "cache_sha256":S.sha256(cache),"full":info,"selector_state":S.evidence(folder/"selector_state.pt")})
    T.freeze(root)
    return state,data,info

def test_real_selector_precompute_publish_equals_frozen_selection(tmp_path,monkeypatch):
    root=tmp_path/"case";state,data,info=fixture(root,monkeypatch)
    expected,trace=S.A.select_compact_rows_v2(state,data["queries"])
    before=torch.cuda.is_initialized()
    T.precompute(root)
    assert not (root/"budget.json").exists()
    saved=S.A.load(root/"scheduler/new_native/selection.pt")
    assert torch.equal(saved["rows"],expected)
    assert torch.equal(saved["trace"]["final_source_ids"],trace["final_source_ids"])
    with pytest.raises(FileExistsError):T.precompute(root)
    with pytest.raises(FileNotFoundError):T.publish(root)
    monkeypatch.setattr(S,"old_inputs",lambda r:({"full":info},state,data["queries"]))
    S.select_old(root)
    T.publish(root)
    for config,rows in [("new_native",expected.numpy()),
                        ("new_spatial_B",S.spatial_rows(state["anchor_xyz"].numpy(),S.budget(root))),
                        ("new_spatial_N",S.spatial_rows(state["anchor_xyz"].numpy(),len(expected)))]:
        sel=S.checked_selection(root,config);published=S.A.load(sel["path"])
        assert torch.equal(published["anchor_xyz"],state["anchor_xyz"][rows])
        assert torch.equal(published["anchor_features"],state["anchor_features"][rows])
        assert np.array_equal(np.load(Path(sel["path"]).with_suffix(".source_rows.npy")),rows)
    assert torch.equal(S.A.load(root/"new_native_trace.pt")["final_source_ids"],expected)
    with pytest.raises(FileExistsError):T.publish(root)
    assert torch.cuda.is_initialized()==before

def test_precompute_binding_cannot_be_replaced(tmp_path,monkeypatch):
    root=tmp_path/"case";state,data,info=fixture(root,monkeypatch);T.precompute(root)
    monkeypatch.setattr(S,"old_inputs",lambda r:({"full":info},state,data["queries"]));S.select_old(root)
    path=root/"scheduler/new_native/summary.json";meta=json.loads(path.read_text())
    for key in ("cache_sha256","full_sha256","selector_state_sha256","scheduler_protocol_sha256"):
        bad=copy.deepcopy(meta);bad["binding"][key]="different";path.write_text(json.dumps(bad))
        with pytest.raises(ValueError,match="source binding"):T.publish(root)
    path.write_text(json.dumps(meta))
    payload=Path(meta["payload"]["path"]);payload.write_bytes(b"changed")
    with pytest.raises(ValueError):T.publish(root)
    assert not (root/"maps/new_native").exists()

def test_sidecar_does_not_change_base_and_rejects_code_drift(tmp_path,monkeypatch):
    root=tmp_path/"case";fixture(root,monkeypatch)
    assert (root/"protocol.json").read_text()=="{}"
    T.protocol(root)
    monkeypatch.setattr(T,"code_hashes",lambda:{"changed":"source"})
    with pytest.raises(ValueError,match="code changed"):T.protocol(root)

@pytest.mark.parametrize("rows",[torch.tensor([],dtype=torch.long),torch.tensor([0,0]),
    torch.tensor([-1]),torch.tensor([3]),torch.tensor([1.]),torch.tensor([[1]])])
def test_invalid_precomputed_rows_rejected(rows):
    with pytest.raises(ValueError):T.valid_rows(rows,3)

def test_ready_cell_evaluation_matches_original_runner_argv(tmp_path,monkeypatch):
    import anygsloc.data.datasets as datasets
    import anygsloc.evaluation.runner as runner
    root=tmp_path/"eval";root.mkdir()
    for name in ("protocol.json","budget.json"):(root/name).write_text("{}")
    (root/"scheduler").mkdir();(root/"scheduler/protocol.json").write_text("{}")
    scene={"dataset":str(tmp_path/"dataset"),"test_names":["held-out.png"]}
    p={"scenes":{S.SCENE:scene}}
    monkeypatch.setattr(T,"protocol",lambda root:p);monkeypatch.setattr(S,"protocol",lambda root:p)
    monkeypatch.setattr(S,"budget",lambda root:2)
    seen=[];verified=[];argv=[]
    selection={"path":str(tmp_path/"map.pt"),"sha256":"map"}
    monkeypatch.setattr(S,"checked_selection",lambda root,c:(seen.append(c) or selection))
    monkeypatch.setattr(S,"verify_cell",lambda root,c,p:verified.append(c))
    class Dataset:
        def __init__(self,*a,**k):pass
        def split(self,name):assert name=="test";return [SimpleNamespace(image_name="held-out.png")]
        def valid_mask(self,c):return torch.ones(1)
    monkeypatch.setattr(datasets,"ColmapDataset",Dataset)
    monkeypatch.setattr(S.M,"evaluation_snapshot",lambda *a:"same-frozen-snapshot")
    def run(args):
        argv.append(args);folder=Path(args[args.index("--output")+1]);folder.mkdir(parents=True)
        for name in ("results.json","manifest.json","summary.json"):(folder/name).write_text("{}")
    monkeypatch.setattr(runner,"main",run)
    T.evaluate(root,"old_native","cuda:1")
    assert seen==["old_native"] and verified==["old_native"]
    att=json.loads((root/"evaluations/old_native/attestation.json").read_text())
    assert att["query_inputs_sha256"]=="same-frozen-snapshot"
    assert att["budget_sha256"]==S.sha256(root/"budget.json")
    # Original path is invoked in a separate root only to compare arguments.
    original=tmp_path/"original";original.mkdir();(original/"protocol.json").write_text("{}");(original/"budget.json").write_text("{}")
    S.evaluate(original,"old_native","cuda:1")
    normalize=lambda args:[x.replace(str(original),str(root)) for x in args]
    assert argv[0]==normalize(argv[1])
    with pytest.raises(FileExistsError):T.evaluate(root,"old_native","cuda:1")

def test_exclusive_locks_reject_duplicate_live_stage(tmp_path):
    with T.lock(tmp_path,"native"):
        with pytest.raises(BlockingIOError):
            with T.lock(tmp_path,"native"):pass
    with T.lock(tmp_path,"native"):pass
