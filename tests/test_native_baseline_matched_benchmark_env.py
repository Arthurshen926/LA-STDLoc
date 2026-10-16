"""Bounded GPU0 ULF environment-alias regressions; no GPU calls."""
import ast
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace
import pytest
from experiments.native_baseline_matched_benchmark import isolation
from experiments.native_baseline_matched_benchmark.common import ref, read
from experiments.native_baseline_matched_benchmark.native import transform_native, transform_wrapper
from experiments.native_baseline_matched_benchmark.results import verify_contexts
from tests.test_native_baseline_matched_benchmark import BASE

UUID = isolation.ULF_ENV_ALIAS_GPU0_UUID

def original_env_nodes(method):
    path = BASE / "baselines" / method / (method.lower() + ".py")
    original = ast.parse(path.read_text())
    nodes = [n for n in original.body if isinstance(n, ast.Assign)
             and "CUDA_VISIBLE_DEVICES" in ast.dump(n)]
    transformed = ast.parse(transform_native(path.read_text(), method))
    assert [ast.dump(n) for n in nodes] == [ast.dump(n) for n in transformed.body
        if isinstance(n, ast.Assign) and "CUDA_VISIBLE_DEVICES" in ast.dump(n)]
    return path, original, nodes

def fixture(tmp_path, monkeypatch, method="ULFLoc", family="native"):
    cp = tmp_path / "contract.json"
    c = dict(case=method + "_test", round=0, cell=dict(method=method, family=family))
    cp.write_text(json.dumps(c))
    lp = tmp_path / "lease.json"
    lp.write_text(json.dumps(dict(valid_until_unix=time.time()+9999, contract_sha256=ref(cp)["sha256"])))
    runtime = dict(contract_path=str(cp), contract_ref=ref(cp), lease_ref=ref(lp), gpu_uuid=UUID)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", UUID)
    cuda = SimpleNamespace(device_count=lambda:1, current_device=lambda:0)
    proof = isolation.context_proof(runtime, torch_module=SimpleNamespace(cuda=cuda), driver_uuid=lambda:UUID)
    return c, runtime, proof

def execute_actual_ulf_assignment():
    import os
    path, tree, nodes = original_env_nodes("ULFLoc")
    assert len(nodes) == 1 and nodes[0].lineno == 26
    assert ast.literal_eval(nodes[0].value) == "0"
    assert any(isinstance(n, ast.Import) and any(a.name == "torch" for a in n.names)
               and n.lineno < nodes[0].lineno for n in tree.body)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), {"os":os})
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"

def test_real_ulf_assignment_strict_start_and_proven_gpu0_alias(tmp_path, monkeypatch):
    c, runtime, proof = fixture(tmp_path, monkeypatch)
    isolation.runtime_gate(c, runtime)  # launch/worker/execute start remains UUID-only
    execute_actual_ulf_assignment()
    with pytest.raises(ValueError):
        isolation.runtime_gate(c, runtime)
    isolation.runtime_gate(c, runtime, initialized_context=proof)
    monitor = isolation.Monitor(runtime)
    monitor.context_proofs["initialization"] = proof
    monkeypatch.setattr(isolation, "sample", lambda: dict(compute_apps_csv=UUID+",42,python,1"))
    monitor.observe()
    assert monitor.samples[-1]["cuda_visible_devices_environment"] == "0"

@pytest.mark.parametrize("fault", ["no_init", "wrong_uuid", "count", "current",
 "case", "round", "contract", "STD", "Any", "family", "literal_one", "two_devices",
 "other_leased_gpu", "expired", "changed_contract"])
def test_alias_never_bypasses_wrong_proof_family_device_lease_or_contract(tmp_path, monkeypatch, fault):
    method = "STDLoc" if fault == "STD" else ("AnyGSLoc" if fault == "Any" else "ULFLoc")
    family = "any" if fault in ("Any","family") else "native"
    c, runtime, proof = fixture(tmp_path, monkeypatch, method, family)
    execute_actual_ulf_assignment()
    if fault == "no_init": proof = None
    elif fault in ("wrong_uuid","count","current","case","round","contract"):
        field, value = {"wrong_uuid":("actual_context_uuid","GPU-wrong"),"count":("visible_device_count",2),
                        "current":("logical_device",1),"case":("case","other"),"round":("round",1),
                        "contract":("contract",{})}[fault]
        proof[field] = value
    elif fault == "literal_one": monkeypatch.setenv("CUDA_VISIBLE_DEVICES","1")
    elif fault == "two_devices": monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0,1")
    elif fault == "other_leased_gpu":
        runtime["gpu_uuid"]="GPU-another";proof["actual_context_uuid"]="GPU-another"
    elif fault == "expired":
        path=Path(runtime["lease_ref"]["path"])
        value=read(path);value["valid_until_unix"]=0;path.write_text(json.dumps(value));runtime["lease_ref"]=ref(path)
    elif fault == "changed_contract":
        path=Path(runtime["contract_path"]);path.write_text(path.read_text()+" ")
    with pytest.raises(ValueError):
        isolation.runtime_gate(c,runtime,initialized_context=proof)

@pytest.mark.parametrize("fault", [None,"missing","wrong_uuid","wrong_contract","wrong_case","wrong_round","extra_context_key"])
def test_ulf_result_preserves_exact_two_context_keys(tmp_path,monkeypatch,fault):
    c,runtime,proof=fixture(tmp_path,monkeypatch)
    row=dict(monitor_closed_clean=True,gpu_uuid=UUID,
             context_proofs=dict(initialization=copy.deepcopy(proof),completion=copy.deepcopy(proof)))
    if fault is None:
        verify_contexts(row,c,ref(runtime["contract_path"]))
        assert set(row["context_proofs"])=={"initialization","completion"}
        return
    if fault=="missing":del row["context_proofs"]["completion"]
    elif fault=="extra_context_key":row["context_proofs"]["native_entry"]=proof
    else:
        field,value={"wrong_uuid":("actual_context_uuid","bad"),"wrong_contract":("contract",{}),
                     "wrong_case":("case","wrong"),"wrong_round":("round",2)}[fault]
        row["context_proofs"]["initialization"][field]=value
    with pytest.raises(ValueError):verify_contexts(row,c,ref(runtime["contract_path"]))

def test_real_v5_std_success_result_envelopes_still_verify():
    root=Path("/mnt/pool/sqy/anygsloc_native_baseline_stmary_repair_20261012_v1/matched_timing_preparation/v5")
    for semantic in ("off","on"):
        case="STDLoc_ShopFacade_mapping_semantics_"+semantic+"_eval"
        cp=root/"contracts"/case/"round0.json"
        c=read(cp);row=read(root/"runs"/case/"round0/round.json")
        verify_contexts(row,c,ref(cp))
        assert len(row["queries"])==100 and len(row["warmup_queries"])==50

def test_actual_wrapper_and_controlled_entry_order_without_scientific_transform():
    path=BASE/"code/tools/baseline_protocol_v2.py"
    text=transform_wrapper(path.read_text())
    assert text.index("torch.cuda.reset_peak_memory_stats()") < text.index("namespace=_CONTROLLED_NATIVE_RUN(")
    path=Path(__file__).parents[1]/"experiments/native_baseline_matched_benchmark/native.py"
    source=path.read_text()
    controlled=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=="controlled")
    # reset_peak ordering is not proof that a Driver context is active.
    # No new CUDA operation/proof may be inserted before original model loading.
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                   and n.func.id=="context_proof" for n in ast.walk(controlled))
    assert "native_entry_context" not in source
    assert not original_env_nodes("STDLoc")[2]

def test_first_frame_context_failure_rejects_before_alias_observation(tmp_path, monkeypatch):
    c,runtime,proof=fixture(tmp_path,monkeypatch)
    execute_actual_ulf_assignment()
    monitor=isolation.Monitor(runtime)
    def fail(runtime):
        raise ValueError("no actual CUDA context")
    monkeypatch.setattr(isolation,"context_proof",fail)
    monkeypatch.setattr(isolation,"sample",lambda: (_ for _ in ()).throw(AssertionError("proof must come first")))
    with pytest.raises(ValueError,match="no actual CUDA context"):
        monitor.check()
    assert monitor.context_proofs=={} and monitor.thread is None

def test_any_real_entry_device_loader_and_env_text_have_no_native_alias_path():
    root=Path(__file__).parents[1]
    paths=["experiments/native_baseline_matched_benchmark/any_adapter.py",
           "experiments/cambridge_deployment_benchmark.py","anygsloc/online/runtime.py",
           "anygsloc/data/datasets.py","anygsloc/features/superpoint.py"]
    for name in paths:
        tree=ast.parse((root/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,(ast.Assign,ast.AugAssign,ast.AnnAssign)):
                targets=getattr(node,"targets",[getattr(node,"target",None)])
                assert not any(t is not None and "CUDA_VISIBLE_DEVICES" in ast.dump(t) for t in targets)
            if isinstance(node,ast.Call):
                assert not(isinstance(node.func,ast.Name) and node.func.id=="DataLoader")
                assert not(isinstance(node.func,ast.Attribute) and node.func.attr=="set_device")
    adapter=ast.parse((root/paths[0]).read_text())
    calls=[n for n in ast.walk(adapter) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="Localizer"]
    assert len(calls)==1 and next(ast.literal_eval(k.value) for k in calls[0].keywords if k.arg=="device")=="cuda:0"

