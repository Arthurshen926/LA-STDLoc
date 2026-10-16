import ast
import copy
import hashlib
import json
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
import pytest
from experiments.native_baseline_matched_benchmark.common import (
    Recorder, SCENES, selection, check_selection, schedule, validate_result, ref, verify)
from experiments.native_baseline_matched_benchmark.native import (
    NativeRecorder, fresh_camera, transform_native, transform_wrapper)
from experiments.native_baseline_matched_benchmark.any_adapter import run_round
from experiments.native_baseline_matched_benchmark.isolation import check_lease, applications, assert_idle
from experiments.native_baseline_matched_benchmark.launch import check_review, verify_scene_images
from experiments.native_baseline_matched_benchmark.results import stats, verify_terminal
from experiments.native_baseline_matched_benchmark.plan import native_contract, DESIGN

BASE = Path("/mnt/pool/sqy/anygsloc_baseline_recovery_v4_20260915")
NAMES = ["seq/frame%05d.png" % i for i in range(140)]

def selected():
    return selection(SCENES[0], NAMES)

class Clock:
    def __init__(self):
        self.value = 0
    def __call__(self):
        self.value += .001
        return self.value

def filled(round_id=0):
    calls = []
    recorder = Recorder(selected(), round_id, clock=Clock(), sync=lambda: calls.append("sync"),
                        reset_peak=lambda: calls.append("reset"), memory=lambda: dict(allocated_bytes=7, reserved_bytes=11))
    for warmup, order, name in recorder.schedule:
        recorder.begin(name)
        recorder.timed("rgb_decode_ms", lambda: calls.append(name))
        recorder.start_mask()
        recorder.end_mask()
        recorder.core(dict(total_ms=2, coarse_prefix_ms=1, fine_increment_ms=1))
        recorder.finish(name)
    return recorder, calls

@pytest.mark.parametrize("scene", SCENES)
def test_selection_reproduces_independent_hash_and_order(scene):
    record = selection(scene, NAMES)
    rank = sorted(NAMES, key=lambda n: (hashlib.sha256(
        ("native_baseline_timing_subset_v1\x002026\x00" + scene + "\x00" + n).encode()).hexdigest(), n))[:100]
    assert record["selection_rank"] == rank
    assert record["measured_query_order"] == [n for n in NAMES if n in rank]
    assert record["warmup_query_order"] == record["measured_query_order"][:50]
    assert check_selection(scene, record) == record

@pytest.mark.parametrize("field", ["full_query_count", "full_query_order", "selection_rank", "measured_query_order", "warmup_query_order"])
def test_selection_tampering_rejected(field):
    record = selected()
    if field == "full_query_count":
        record[field] -= 1
    else:
        record[field] = list(reversed(record[field]))
    with pytest.raises(ValueError):
        check_selection(SCENES[0], record)

@pytest.mark.parametrize("names", [NAMES[:99], NAMES + NAMES[:1], ["../bad"] + NAMES, ["/bad"] + NAMES, ["bad\x00"] + NAMES])
def test_invalid_query_population_rejected(names):
    with pytest.raises(ValueError):
        selection(SCENES[0], names)

@pytest.mark.parametrize("round_id", [0, 1, 2])
def test_round_exact150_calls_distinct50_warmups100_measured_reset_and_exclusion(round_id):
    recorder, calls = filled(round_id)
    result = recorder.result()
    validate_result(result, selected(), round_id)
    assert len(result["warmup_names"]) == len(set(result["warmup_names"])) == 50
    assert len(result["queries"]) == 100
    assert calls.count("reset") == 150
    assert [c for c in calls if c in NAMES] == selected()["warmup_query_order"] + selected()["measured_query_order"]
    assert stats(result["queries"])["core_ms"]["count"] == 100
    assert all(row["torch_peak"] == dict(allocated_bytes=7, reserved_bytes=11) for row in result["queries"])

@pytest.mark.parametrize("round_id", [-1, 3, True, "0"])
def test_invalid_rounds_rejected(round_id):
    with pytest.raises(ValueError):
        schedule(selected(), round_id)

def test_frame_state_and_incomplete_round_rejected():
    recorder = Recorder(selected(), 0, clock=Clock())
    with pytest.raises(ValueError):
        recorder.begin("wrong")
    recorder.begin(selected()["warmup_query_order"][0])
    with pytest.raises(ValueError):
        recorder.begin(selected()["warmup_query_order"][0])
    with pytest.raises(ValueError):
        recorder.finish(selected()["warmup_query_order"][0])
    with pytest.raises(ValueError):
        recorder.result()

@pytest.mark.parametrize("number", [-1, float("nan"), float("inf")])
def test_invalid_core_clock_rejected(number):
    recorder = Recorder(selected(), 0, clock=Clock())
    recorder.begin(selected()["warmup_query_order"][0])
    with pytest.raises(ValueError):
        recorder.core(dict(total_ms=number, coarse_prefix_ms=1, fine_increment_ms=1))

@pytest.mark.parametrize("change", ["name", "round", "warmup", "query_order", "missing"])
def test_result_identity_tampering_rejected(change):
    recorder, _ = filled()
    value = recorder.result()
    if change == "missing":
        value["queries"].pop()
    else:
        value["queries"][0][change] = "wrong"
    with pytest.raises(ValueError):
        validate_result(value, selected(), 0)

@pytest.mark.parametrize("method", ["STDLoc", "ULFLoc"])
def test_actual_native_classes_functions_and_solver_calls_ast_unchanged(method):
    path = BASE / "baselines" / method / (method.lower() + ".py")
    assert path.is_file(), "real native source is required for this review"
    source = path.read_text()
    before = ast.parse(source)
    after = ast.parse(transform_native(source, method))
    definitions = lambda tree: [ast.dump(n) for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    assert definitions(before) == definitions(after)
    assert source.count("preload_cameras=False") == 1
    transformed = transform_native(source, method)
    assert transformed.count("_MATCHED.cameras(") == 1
    assert transformed.count("_MATCHED.finish(camera_info.image_name, loc_res)") == 1
    assert transformed.count("_MATCHED.start_mask()") == transformed.count("_MATCHED.end_mask()") == 1
    assert "continue\n\n        # evaluation" in transformed
    localize_calls = lambda tree: [ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.Call)
                                   and isinstance(n.func, ast.Attribute) and n.func.attr == "localize"]
    assert localize_calls(before) == localize_calls(after)

def test_actual_native_wrapper_warmup_disabled_and_instrumentation_additive():
    from experiments.native_baseline_repair.query_semantic_adapter import wrapper_text
    original = (BASE / "code/tools/baseline_protocol_v2.py").read_text()
    transformed = transform_wrapper(original)
    assert "if a.warmup != 0: raise" in transformed
    assert transformed.count("_MATCHED.core(timings[-1])") == 1
    fn = lambda t, name: next(n for n in ast.walk(ast.parse(t)) if isinstance(n, ast.FunctionDef) and n.name == name)
    for name in ["pose", "seed", "coarse_call", "camera_init"]:
        assert ast.dump(fn(wrapper_text(original), name)) == ast.dump(fn(transformed, name))
    call = fn(transformed, "localize_call")
    call.body = [n for n in call.body if not (
        isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
        and isinstance(n.value.func.value, ast.Name) and n.value.func.value.id == "_MATCHED")]
    assert ast.dump(call) == ast.dump(fn(wrapper_text(original), "localize_call"))

def cpu_camera_code(method):
    import torch
    import numpy as np
    import math
    root = BASE / "baselines" / method
    env = dict(torch=torch, np=np, math=math, nn=torch.nn)
    def definitions(path, names):
        tree = ast.parse(path.read_text())
        nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        assert {node.name for node in nodes} == set(names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
    definitions(root / "utils/graphics_utils.py", ["getWorld2View2", "getProjectionMatrix"])
    definitions(root / "scene/cameras.py", ["Camera"])
    definitions(root / "utils/camera_utils.py", ["loadCam"])
    return env

@pytest.mark.parametrize("method", ["STDLoc", "ULFLoc"])
def test_real_native_fresh_decode_pixels_cameras_and_original_metadata_equal(tmp_path, monkeypatch, method):
    import torch
    import numpy as np
    from PIL import Image
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: (_ for _ in ()).throw(AssertionError("CPU tests must not initialize CUDA")))
    code = cpu_camera_code(method)
    Info = namedtuple("Info", "uid R T FovY FovX image image_path image_name width height")
    pixels = np.random.RandomState(2026).randint(0,256,(13,19,3),dtype=np.uint8)
    path = tmp_path / "query.png"
    Image.fromarray(pixels).save(path)
    original = Image.open(path)
    info = Info(4, np.eye(3), np.array([1.,2.,3.]), .9, 1.1, original, str(path), "query.png", 19, 13)
    args = SimpleNamespace(resolution=1, data_device="cpu")
    class Dataset:
        split = "test"
        scene = SimpleNamespace(scene_info=SimpleNamespace(test_cameras=[info]))
        def __getitem__(self, index):
            return code["loadCam"](args, index, self.scene.scene_info.test_cameras[index], 1.0)
    dataset = Dataset()
    baseline = dataset[0]
    for i in range(2):
        fresh = fresh_camera(dataset, 0, lambda op: op(), lambda op: op())
        assert dataset.scene.scene_info.test_cameras[0] is info
        for key in ["original_image", "world_view_transform", "projection_matrix", "full_proj_transform", "camera_center"]:
            assert torch.equal(getattr(baseline, key), getattr(fresh, key)), key
        for key in ["image_name", "image_width", "image_height", "FoVx", "FoVy", "uid"]:
            assert getattr(baseline, key) == getattr(fresh, key)
    original.close()
    assert not torch.cuda.is_initialized()

def test_native_sequence_calls_materializer150_times_without_caching(monkeypatch):
    import experiments.native_baseline_matched_benchmark.native as module
    from torch.utils.data import DataLoader, Dataset
    infos = [SimpleNamespace(image_name=n) for n in NAMES]
    class SceneDataset(Dataset):
        __module__ = "scene"
        split = "test"
        scene = SimpleNamespace(scene_info=SimpleNamespace(test_cameras=infos))
        def __len__(self):
            return len(infos)
    dataset = SceneDataset()
    loader = DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=True)
    calls = []
    def materialize(ds, index, decode, camera):
        calls.append(index)
        return SimpleNamespace(image_name=infos[index].image_name)
    monkeypatch.setattr(module, "fresh_camera", materialize)
    recorder = NativeRecorder(selected(), 0, clock=Clock())
    for camera in recorder.cameras(loader, "ULFLoc"):
        recorder.core(dict(total_ms=2, coarse_prefix_ms=1, fine_increment_ms=1))
        recorder.finish(camera.image_name)
    assert len(calls) == 150
    assert [NAMES[i] for i in calls] == selected()["warmup_query_order"] + selected()["measured_query_order"]
    assert len(recorder.result()["queries"]) == 100

def test_any_same_localizer_call150_times_no_feature_cache_and_unavailable_stages():
    selected_record = selected()
    cameras = [SimpleNamespace(image_name=n, fov_x=1, fov_y=2) for n in NAMES]
    calls = []
    class Data:
        def load_image(self, camera):
            calls.append(("decode", camera.image_name))
            return camera.image_name
        def valid_mask(self, camera):
            calls.append(("mask", camera.image_name))
            return "mask"
    class Engine:
        def localize(self, image, **kwargs):
            calls.append(("features-localize", image))
            assert kwargs == dict(fov_x=1, fov_y=2, valid_mask="mask")
            return dict(runtime_ms=1, success=True, pose=SimpleNamespace(pose_w2c=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], inliers=[0]))
    recorder = Recorder(selected_record, 1, clock=Clock(), memory=lambda: dict(allocated_bytes=0, reserved_bytes=0))
    value = run_round(Data(), cameras, Engine(), selected_record, 1, recorder)
    assert [name for stage, name in calls if stage == "features-localize"] == selected_record["warmup_query_order"] + selected_record["measured_query_order"]
    assert len(calls) == 450
    assert stats(value["queries"])["coarse_prefix_ms"] is None
    assert stats(value["queries"])["core_ms"]["count"] == 100

def lease():
    return dict(approved=True, scope="cambridge_matched100_formal_timing", contract_sha256="c",
                protocol_sha256="p", all_training_accuracy_finished=True, cpu_io_quiet=True,
                approval_reference="owner explicit release", owner_handover="all load ended",
                gpu_uuid="GPU-test", valid_from_unix=0, valid_until_unix=5000)

@pytest.mark.parametrize("change", [dict(approved=False), dict(scope="old_full_timing"),
    dict(contract_sha256="wrong"), dict(protocol_sha256="wrong"), dict(cpu_io_quiet=False),
    dict(all_training_accuracy_finished=False), dict(owner_handover=""), dict(valid_until_unix=1000),
    dict(valid_until_unix=8000), dict(gpu_uuid="0")])
def test_lease_rejects_old_reused_partial_or_expiring_release(change):
    value = lease()
    assert check_lease(value, "c", "p", now=10) == "GPU-test"
    value.update(change)
    with pytest.raises(ValueError):
        check_lease(value, "c", "p", now=10)

def test_physical_gpu_gate_rejects_any_competing_gpu():
    value = dict(gpu_csv="GPU-test, name, 0, 0", compute_apps_csv="")
    assert_idle(value, "GPU-test")
    value["compute_apps_csv"] = "GPU-other, 42, python, 100"
    with pytest.raises(RuntimeError):
        assert_idle(value, "GPU-test")

def test_review_requires_new_protocol_not_old_telemetry():
    good = dict(complete=True, verdict="PASS", scope="matched_timing_cpu", protocol_sha256="p", no_gpu_run_performed=True)
    check_review(good, dict(sha256="p"))
    for key in good:
        bad = dict(good)
        bad.pop(key)
        with pytest.raises(ValueError):
            check_review(bad, dict(sha256="p"))

def test_file_hash_and_rgb_tamper_rejected(tmp_path):
    path = tmp_path / "processed/q.png"
    path.parent.mkdir()
    path.write_bytes(b"frozen")
    value = ref(path)
    snapshot = dict(queries=[dict(name="q.png", image_sha256=value["sha256"])])
    verify_scene_images(snapshot, tmp_path, ["q.png"])
    path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        verify(value)
    with pytest.raises(ValueError):
        verify_scene_images(snapshot, tmp_path, ["q.png"])

def test_terminal_evidence_requires_exact_job_round_and_contract(tmp_path):
    result = tmp_path / "round.json"
    launch = tmp_path / "launch.json"
    result.write_text("{}")
    launch.write_text("{}")
    value = dict(terminal=True, exit_code=0, evidence_source="WebCodex observe_jobs",
                 job_id="wc_job_test", contract_sha256="c", case="STDLoc", round=0,
                 result=ref(result), launch=ref(launch))
    verify_terminal(value, dict(sha256="c"), "STDLoc", 0, ref(result), ref(launch))
    for key, replacement in [("round",1), ("terminal",False), ("exit_code",-1), ("contract_sha256","old"), ("job_id","old")]:
        bad = dict(value)
        bad[key] = replacement
        with pytest.raises(ValueError):
            verify_terminal(bad, dict(sha256="c"), "STDLoc", 0, ref(result), ref(launch))

def test_native_output_derivation_changes_only_routing_and_disables_old_warmup(tmp_path):
    base = dict(model_view="old", receipt="old_receipt", native_output="old_output", method="STDLoc",
                wrapper_argv=["--receipt","old_receipt","--run-tag","query_semantic_on","--warmup","0","--","-m","old","--cfg","frozen","--dense","1"],
                model_assets={"same":"value"}, seed=2026)
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(base))
    c = dict(output=str(tmp_path / "new"), cell=dict(accuracy_contract=ref(path)))
    result = native_contract(c)
    assert result["model_assets"] == base["model_assets"]
    assert result["wrapper_argv"][-4:] == base["wrapper_argv"][-4:]
    assert result["wrapper_argv"][result["wrapper_argv"].index("--warmup") + 1] == "0"
    assert result["seed"] == 2026

def test_documented_boundaries_do_not_claim_gt_free_ulf_or_free_segmentation():
    assert "INCLUDE GT" in DESIGN["ULFLoc"]
    assert "no GT passed" not in DESIGN["ULFLoc"]
    assert "excluded" in DESIGN["segmentation"]
    assert DESIGN["unavailable"]["native_mask_compose_ms"] is None

def test_any_camera_gt_dimensions_intrinsics_and_order_tampering():
    import math
    import numpy as np
    from experiments.native_baseline_matched_benchmark.launch import verify_any_cameras
    camera = SimpleNamespace(image_name="q", pose_w2c=np.eye(4, dtype=np.float32),
                             width=20, height=10, fov_x=2*math.atan(1), fov_y=2*math.atan(.5))
    snapshot = dict(queries=[dict(name="q", pose_w2c=camera.pose_w2c.tolist(), image_hw=[10,20],
                                 intrinsic=[[10,0,10],[0,10,5],[0,0,1]])])
    verify_any_cameras([camera], snapshot, ["q"])
    for key, value in [("pose_w2c", np.zeros((4,4))), ("width",21), ("fov_x",.5), ("image_name","wrong")]:
        changed = copy.copy(camera)
        setattr(changed,key,value)
        with pytest.raises(ValueError):
            verify_any_cameras([changed], snapshot, ["q"])

def test_missing_final_model_contracts_abort_binding_without_partial_assets(tmp_path, monkeypatch):
    from experiments.native_baseline_matched_benchmark import plan
    qp = tmp_path / "query_protocol.json"
    ap = tmp_path / "any_protocol.json"
    qp.write_text(json.dumps(dict(cells={"STDLoc_StMarysChurch_mapping_semantics_off_eval":{}})))
    ap.write_text("{}")
    monkeypatch.setattr(plan, "ROOT", tmp_path)
    monkeypatch.setattr(plan, "protocol", lambda root: dict(accuracy_protocol=ref(qp), any_protocol=ref(ap)))
    with pytest.raises(ValueError, match="no partial binding"):
        plan.bind(tmp_path / "preparation")
    assert not (tmp_path / "preparation/assets.json").exists()

def test_three_round_recorders_never_share_state():
    all_rows = []
    for round_id in range(3):
        recorder, _ = filled(round_id)
        result = recorder.result()
        assert {row["round"] for row in result["queries"]} == {round_id}
        all_rows.extend(result["queries"])
    assert len(all_rows) == 300
    assert stats(all_rows)["core_ms"]["count"] == 300
