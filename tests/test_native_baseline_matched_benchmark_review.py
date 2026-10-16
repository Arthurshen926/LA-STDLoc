"""Adversarial evidence and shutdown regressions requested by independent review."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from tests.test_native_baseline_matched_benchmark import selected, filled
from experiments.native_baseline_matched_benchmark.common import ref, validate_result
from experiments.native_baseline_matched_benchmark.results import verify_terminal, verify_contexts, check_accuracy_evidence
from experiments.native_baseline_matched_benchmark import isolation

IDENTITY = [[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]

def strict_native():
    recorder, _ = filled()
    value = recorder.result()
    for row in value["queries"] + value["warmup_queries"]:
        row["stages"].update(camera_materialize_ms=1, mask_compose_ms=None)
        row["localization"] = dict(success=None, sparse_pose_w2c=IDENTITY, dense_pose_w2c=IDENTITY,
                                   sparse_inliers=10, dense_inliers=10, matches=None)
    return value

@pytest.mark.parametrize("field", ["rgb_decode_ms", "camera_materialize_ms", "mask_preprocess_ms",
                                   "coarse_prefix_ms", "dense_increment_ms", "mask_compose_ms"])
def test_required_native_boundaries_cannot_disappear(field):
    value = strict_native()
    validate_result(value, selected(), 0, "native")
    del value["queries"][0]["stages"][field]
    with pytest.raises(ValueError):
        validate_result(value, selected(), 0, "native")

@pytest.mark.parametrize("change", ["wrong_sum", "invent_mask", "missing_pose", "init_nan", "negative_peak", "float_peak", "missing_peak", "phase_peak"])
def test_invalid_native_boundaries_peak_summaries_and_pose_fail_closed(change):
    value = strict_native()
    if change == "wrong_sum":
        value["queries"][0]["stages"]["dense_increment_ms"] = 4
    elif change == "invent_mask":
        value["queries"][0]["stages"]["mask_compose_ms"] = 0
    elif change == "missing_pose":
        del value["queries"][0]["localization"]["dense_pose_w2c"]
    elif change == "init_nan":
        value["initialization"]["total_ms"] = float("nan")
    elif change == "negative_peak":
        value["warmup_queries"][0]["torch_peak"]["allocated_bytes"] = -1
    elif change == "float_peak":
        value["queries"][0]["torch_peak"]["reserved_bytes"] = 10.5
    elif change == "missing_peak":
        value["initialization"]["torch_peak"] = {}
    elif change == "phase_peak":
        value["measured_torch_peak"]["allocated_bytes"] = 100
    with pytest.raises(ValueError):
        validate_result(value, selected(), 0, "native")

def terminal_fixture(tmp_path):
    result = tmp_path / "round.json"
    launch = tmp_path / "launch.json"
    result.write_text(json.dumps(dict(round=0, timing=1)))
    launch.write_text("{}")
    evidence = dict(terminal=True, exit_code=0, evidence_source="WebCodex observe_jobs", job_id="wc_job_test",
                    contract_sha256="c", case="test", round=0, result=ref(result), launch=ref(launch))
    return result, launch, evidence

@pytest.mark.parametrize("change", ["changed_timing", "same_content_wrong_path", "cross_round", "changed_launch"])
def test_terminal_binds_canonical_paths_and_actual_bytes(tmp_path, change):
    result, launch, evidence = terminal_fixture(tmp_path)
    verify_terminal(evidence, dict(sha256="c"), "test", 0, ref(result), ref(launch))
    if change == "changed_timing":
        result.write_text(json.dumps(dict(round=0, timing=999)))
    elif change == "same_content_wrong_path":
        other = tmp_path / "elsewhere.json"
        other.write_bytes(result.read_bytes())
        evidence["result"] = ref(other)
    elif change == "cross_round":
        other = tmp_path / "round1.json"
        other.write_text(json.dumps(dict(round=1, timing=1)))
        evidence["result"] = ref(other)
    elif change == "changed_launch":
        launch.write_text('{"gpu":"other"}')
    with pytest.raises(ValueError):
        verify_terminal(evidence, dict(sha256="c"), "test", 0, ref(result), ref(launch))

def test_monitor_join_time_error_prevents_clean_close(monkeypatch):
    monitor = isolation.Monitor({})
    class Thread:
        def join(self, timeout):
            monitor.error = "external PID observed while closing"
        def is_alive(self):
            return False
    monitor.thread = Thread()
    with pytest.raises(RuntimeError, match="during close"):
        monitor.__exit__(None, None, None)
    assert not monitor.closed_clean

def test_monitor_final_scan_rejects_external_pid(monkeypatch):
    monitor = isolation.Monitor(dict(gpu_uuid="GPU-test"))
    monitor.observed_pids = {("GPU-test",42)}
    monkeypatch.setattr(isolation, "sample", lambda: dict(compute_apps_csv="GPU-test,42,python,1\nGPU-other,99,python,1"))
    with pytest.raises(RuntimeError, match="competing"):
        monitor.__exit__(None,None,None)
    assert not monitor.closed_clean

def test_monitor_clean_close_records_final_context(monkeypatch):
    monitor = isolation.Monitor({})
    calls = []
    monkeypatch.setattr(monitor, "observe", lambda: calls.append("last_scan"))
    monkeypatch.setattr(isolation, "context_proof", lambda runtime: {"proof":"final"})
    monitor.__exit__(None,None,None)
    assert calls == ["last_scan"]
    assert monitor.closed_clean
    assert monitor.context_proofs == {"completion":{"proof":"final"}}

@pytest.mark.parametrize("module", ["native.py", "any_adapter.py"])
def test_success_artifact_written_only_after_with_monitor_exit(module):
    folder = Path(__file__).parents[1] / "experiments/native_baseline_matched_benchmark"
    tree = ast.parse((folder / module).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=="execute")
    blocks = [n for n in fn.body if isinstance(n, ast.With)]
    assert len(blocks) == 1
    writes = lambda node: [n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="write_new"]
    assert not writes(blocks[0])
    assert len(writes(fn)) == 1
    assert isinstance(fn.body[-1], ast.Expr) and fn.body[-1].value.func.id == "write_new"

@pytest.mark.parametrize("count,current,actual", [(2,0,"GPU-test"), (1,1,"GPU-test"), (1,0,"GPU-wrong")])
def test_actual_cuda_context_mismatch_rejected_without_gpu(tmp_path,count,current,actual):
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(dict(case="test",round=0)))
    runtime = dict(gpu_uuid="GPU-test",contract_path=str(path),contract_ref=ref(path))
    mock = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:count,current_device=lambda:current))
    with pytest.raises(ValueError):
        isolation.context_proof(runtime,torch_module=mock,driver_uuid=lambda:actual)

def test_context_proofs_bind_case_round_and_contract(tmp_path):
    path = tmp_path / "contract.json"
    c = dict(case="test",round=0)
    path.write_text(json.dumps(c))
    runtime = dict(gpu_uuid="GPU-test",contract_path=str(path),contract_ref=ref(path))
    mock = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:1,current_device=lambda:0))
    proof = isolation.context_proof(runtime,torch_module=mock,driver_uuid=lambda:"GPU-test")
    value = dict(monitor_closed_clean=True,gpu_uuid="GPU-test",context_proofs=dict(initialization=proof,completion=copy.deepcopy(proof)))
    verify_contexts(value,c,ref(path))
    value["context_proofs"]["completion"]["round"] = 1
    with pytest.raises(ValueError):
        verify_contexts(value,c,ref(path))

def test_any_failure_evidence_is_counted_not_filtered(tmp_path):
    old = tmp_path / "old_full_accuracy.json"
    names = selected()["measured_query_order"]
    old.write_text(json.dumps(dict(queries=[dict(image_name=n,success=True,pose_w2c=IDENTITY) for n in names])))
    cell = dict(family="any",accuracy_reference=ref(old))
    current = [dict(name=n,localization=dict(success=i!=0,pose_w2c=IDENTITY,matches_count=100,inliers_count=0 if i==0 else 20)) for i,n in enumerate(names)]
    value = dict(accuracy_reference=ref(old),queries=current,process_peak_rss_kib=100)
    report = check_accuracy_evidence(value,dict(cell=cell),{})
    assert report["reported_failures"] == 1
    assert report["reference_failures"] == 0
    assert report["changed_success_count"] == 1
    assert report["timing_queries_filtered"] == 0

@pytest.mark.parametrize("field", ["rgb_decode_preprocess_ms", "mask_compose_ms", "coarse_prefix_ms", "dense_increment_ms"])
def test_any_required_and_unavailable_fields_cannot_disappear(field):
    value = strict_native()
    for row in value["queries"] + value["warmup_queries"]:
        row["stages"] = dict(core_ms=2,frame_ms=5,rgb_decode_preprocess_ms=1,mask_compose_ms=1,
                             coarse_prefix_ms=None,dense_increment_ms=None)
        row["localization"] = dict(success=True,pose_w2c=IDENTITY,matches_count=100,inliers_count=10)
    validate_result(value, selected(), 0, "any")
    del value["queries"][0]["stages"][field]
    with pytest.raises(ValueError):
        validate_result(value, selected(), 0, "any")

def test_canonical_result_ref_rejects_symlink_redirection(tmp_path):
    from experiments.native_baseline_matched_benchmark.results import canonical_ref
    original = tmp_path / "round.json"
    other = tmp_path / "other_round.json"
    other.write_text("{}")
    original.symlink_to(other)
    with pytest.raises(ValueError):
        canonical_ref(original)

def test_native_evidence_persists_same_round_results_after_frame_clock():
    from experiments.native_baseline_matched_benchmark.native import NativeRecorder
    from tests.test_native_baseline_matched_benchmark import Clock
    recorder = NativeRecorder(selected(),0,clock=Clock())
    name = selected()["warmup_query_order"][0]
    recorder.begin(name)
    recorder.core(dict(total_ms=2,coarse_prefix_ms=1,fine_increment_ms=1))
    result = dict(sparse=dict(pose_w2c=IDENTITY,inliers=12),
                  dense=[dict(pose_w2c=IDENTITY,inliers=14)])
    recorder.finish(name,result)
    evidence = recorder.warmup_rows[0]["localization"]
    assert evidence["dense_pose_w2c"] == IDENTITY
    assert evidence["sparse_inliers"] == 12 and evidence["dense_inliers"] == 14
    assert evidence["success"] is None and evidence["matches"] is None

def native_accuracy_fixture(tmp_path):
    from experiments.native_baseline_matched_benchmark.plan import native_contract
    from experiments.native_baseline_matched_benchmark.common import native_evidence
    names = selected()["measured_query_order"]
    snap = tmp_path / "snapshot.json"
    snap.write_text(json.dumps(dict(queries=[dict(name=n,pose_w2c=IDENTITY) for n in names])))
    original = dict(method="STDLoc",model_view="old_model",receipt="old_receipt",native_output="old_output",
                    query_names=names,query_snapshot=ref(snap),
                    wrapper_argv=["--receipt","old_receipt","--run-tag","query_semantic_on","--warmup","0","--","-m","old"])
    cp = tmp_path / "accuracy_contract.json"
    cp.write_text(json.dumps(original))
    oldrows = [dict(sparse=dict(pose_w2c=IDENTITY,inliers=12),dense=[dict(pose_w2c=IDENTITY,inliers=14)],
                    gt_pose_w2c=IDENTITY) for _ in names]
    oldpath = tmp_path / "old_results.json"
    oldpath.write_text(json.dumps(oldrows))
    verification = tmp_path / "verification.json"
    verification.write_text("{}")
    cell = dict(family="native",method="STDLoc",mapping_semantics="off",accuracy_contract=ref(cp),
                accuracy_reference=ref(verification),accuracy_results=ref(oldpath))
    c = dict(case="native_test",round=0,cell=cell,output=str(tmp_path/"new"))
    nc = native_contract(c)
    results = Path(nc["native_output"]) / "results.json"
    results.parent.mkdir(parents=True)
    results.write_text(json.dumps(oldrows))
    receipt = Path(nc["receipt"])
    receipt.write_text(json.dumps(dict(status="complete",method="STDLoc",phase="evaluate",mapping_semantics="off",
                         query_semantics=True,seed=2026,warmup_queries=0,native_output=nc["native_output"],
                         timings=[dict(total_ms=2) for _ in range(150)])))
    value = dict(accuracy_reference=ref(verification),native_results=ref(results),native_receipt=ref(receipt),
                 process_peak_rss_kib=100,queries=[dict(name=n,localization=native_evidence(r),stages=dict(core_ms=2))
                                               for n,r in zip(names,oldrows)])
    terminal = dict(native_results=ref(results),native_receipt=ref(receipt))
    return c,value,terminal

@pytest.mark.parametrize("change", [None,"wrong_name","changed_result","wrong_path","timing_mismatch"])
def test_native_accuracy_results_receipts_and_names_bound(tmp_path,change):
    c,value,terminal = native_accuracy_fixture(tmp_path)
    if change is None:
        report = check_accuracy_evidence(value,c,terminal)
        assert report["measured_query_count"] == 100 and report["timing_queries_filtered"] == 0
        return
    if change == "wrong_name":
        value["queries"][0]["name"] = "other"
    elif change == "changed_result":
        path = Path(value["native_results"]["path"])
        rows = json.loads(path.read_text())
        rows[0]["dense"][0]["inliers"] = 1
        path.write_text(json.dumps(rows))
    elif change == "wrong_path":
        path = Path(value["native_results"]["path"])
        other = tmp_path / "same_content.json"
        other.write_bytes(path.read_bytes())
        value["native_results"] = ref(other)
    elif change == "timing_mismatch":
        value["queries"][0]["stages"]["core_ms"] = 1
    with pytest.raises((ValueError,KeyError)):
        check_accuracy_evidence(value,c,terminal)

def test_native_frozen_helper_search_priority_and_preloaded_shadow_rejection(tmp_path):
    from experiments.native_baseline_matched_benchmark.native import frozen_helper_search_path
    wrapper = tmp_path / "frozen/code/tools/wrapper.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("")
    code = wrapper.resolve().parents[1]
    root = str(tmp_path / "mainrepo")
    assert frozen_helper_search_path(wrapper,[root,str(code)],{}) == [str(code),root]
    good = {"anygsloc":SimpleNamespace(__file__=str(code/"anygsloc/__init__.py"))}
    assert frozen_helper_search_path(wrapper,[root],good)[0] == str(code)
    wrong = {"anygsloc.data.colmap":SimpleNamespace(__file__=root+"/anygsloc/data/colmap.py")}
    with pytest.raises(ValueError):
        frozen_helper_search_path(wrapper,[root],wrong)



