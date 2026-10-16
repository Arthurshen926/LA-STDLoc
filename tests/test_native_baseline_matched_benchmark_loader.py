"""Real frozen native loader/SceneDataset/Camera CPU regressions for v5."""
import ast
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
import pytest
from tests.test_native_baseline_matched_benchmark import BASE, Clock, selected, NAMES, cpu_camera_code
from experiments.native_baseline_matched_benchmark.native import NativeRecorder, native_test_dataset, transform_native
from experiments.native_baseline_matched_benchmark.common import sha

SCENE_SHA = {
    "STDLoc": "6746ea1b208396b04093a199090b559e41dad17fdceec9c333e95a58842f7280",
    "ULFLoc": "7a41bdc43c7d9b974d0ccc3aa2fa324696edf0c007fa2bdab4b42f19eafe0701",
}

def actual_loader(tmp_path, monkeypatch, method):
    import torch
    import numpy as np
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from torch.utils.data._utils.collate import default_convert
    from torch.utils.data._utils.pin_memory import pin_memory
    def forbidden(*args, **kwargs):
        raise AssertionError("DataLoader iteration/worker/prefetch or CUDA is forbidden in CPU fixture")
    monkeypatch.setattr(DataLoader, "__iter__", forbidden)
    monkeypatch.setattr(DataLoader, "_get_iterator", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.Tensor, "pin_memory", forbidden)
    source = BASE / "baselines" / method / "scene/__init__.py"
    assert sha(source) == SCENE_SHA[method]
    tree = ast.parse(source.read_text())
    scene_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scene")
    factory = next(n for n in scene_class.body if isinstance(n, ast.FunctionDef) and n.name == "getTestCameras")
    dataset_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SceneDataset")
    code = cpu_camera_code(method)
    scope = dict(__name__="scene", Scene=type("Scene", (), {}), Dataset=Dataset,
                 DataLoader=DataLoader, loadCam=code["loadCam"])
    exec(compile(ast.Module(body=[dataset_class, factory], type_ignores=[]), str(source), "exec"), scope)
    pixels = np.random.RandomState(2026).randint(0, 256, (13, 19, 3), dtype=np.uint8)
    image_path = tmp_path / "query.png"
    Image.fromarray(pixels).save(image_path)
    original = Image.open(image_path)
    Info = namedtuple("Info", "uid R T FovY FovX image image_path image_name width height")
    infos = [Info(i, np.eye(3), np.array([float(i), 2., 3.]), .9, 1.1,
                  original, str(image_path), name, 19, 13) for i, name in enumerate(NAMES)]
    scene = SimpleNamespace(preload_cameras=False, shuffle=False,
        args=SimpleNamespace(resolution=1, data_device="cpu"), resolution_scales=[1.0],
        scene_info=SimpleNamespace(test_cameras=infos, train_cameras=[]))
    loader = scope["getTestCameras"](scene)
    assert type(loader) is DataLoader and type(loader.dataset).__module__ == "scene"
    assert type(loader.dataset).__name__ == "SceneDataset" and loader.dataset.split == "test"
    assert loader.num_workers == (4 if method == "STDLoc" else 0)
    assert loader.batch_size is None and loader.pin_memory is True and loader._iterator is None
    # Default no-batch collation and native pin-memory dispatch are no-ops for
    # the actual native Camera object, not merely configuration assertions.
    camera = loader.dataset[0]
    assert default_convert(camera) is camera
    assert not hasattr(camera, "pin_memory")
    assert pin_memory(camera) is camera
    assert loader.collate_fn(camera) is camera
    assert not torch.cuda.is_initialized()
    return loader, original

@pytest.mark.parametrize("method", ["STDLoc", "ULFLoc"])
def test_real_factory_dataset_camera150_sync_calls_without_loader_iteration(tmp_path, monkeypatch, method):
    import torch
    loader, original = actual_loader(tmp_path, monkeypatch, method)
    dataset = native_test_dataset(loader, method)
    originals = list(dataset.scene.scene_info.test_cameras)
    calls = []
    getitem = type(dataset).__getitem__
    def counted(self, index):
        calls.append(index)
        return getitem(self, index)
    monkeypatch.setattr(type(dataset), "__getitem__", counted)
    recorder = NativeRecorder(selected(), 0, clock=Clock())
    try:
        for camera in recorder.cameras(loader, method):
            index = NAMES.index(camera.image_name)
            expected = getitem(dataset, index)
            for key in ("original_image", "world_view_transform", "projection_matrix",
                        "full_proj_transform", "camera_center"):
                assert torch.equal(getattr(camera, key), getattr(expected, key)), key
            for key in ("image_name", "image_width", "image_height", "FoVx", "FoVy", "uid"):
                assert getattr(camera, key) == getattr(expected, key)
            assert dataset.scene.scene_info.test_cameras[index] is originals[index]
            recorder.core(dict(total_ms=2, coarse_prefix_ms=1, fine_increment_ms=1))
            recorder.finish(camera.image_name)
        assert len(calls) == 150
        assert [NAMES[i] for i in calls] == selected()["warmup_query_order"] + selected()["measured_query_order"]
        assert len(recorder.result()["queries"]) == 100
        assert loader._iterator is None and not torch.cuda.is_initialized()
    finally:
        original.close()

@pytest.mark.parametrize("method", ["STDLoc", "ULFLoc"])
@pytest.mark.parametrize("fault", ["wrong_method_workers", "batch", "split", "dataset", "loader",
                                   "collation", "pin_memory", "sampler", "wrong_name"])
def test_real_native_loader_wrong_contract_or_camera_rejected(tmp_path, monkeypatch, method, fault):
    from torch.utils.data import RandomSampler
    loader, original = actual_loader(tmp_path, monkeypatch, method)
    try:
        if fault == "wrong_method_workers":
            object.__setattr__(loader, "num_workers", 0 if method == "STDLoc" else 4)
        elif fault == "batch":
            object.__setattr__(loader, "batch_size", 1)
        elif fault == "split":
            loader.dataset.split = "train"
        elif fault == "dataset":
            object.__setattr__(loader, "dataset", SimpleNamespace(split="test"))
        elif fault == "loader":
            loader = SimpleNamespace(dataset=loader.dataset, num_workers=loader.num_workers, batch_size=None)
        elif fault == "collation":
            object.__setattr__(loader, "collate_fn", lambda x: x)
        elif fault == "pin_memory":
            object.__setattr__(loader, "pin_memory", False)
        elif fault == "sampler":
            object.__setattr__(loader, "sampler", RandomSampler(loader.dataset))
        else:
            original_get = type(loader.dataset).__getitem__
            def wrong_name(self, index):
                camera = original_get(self, index)
                camera.image_name = "wrong"
                return camera
            monkeypatch.setattr(type(loader.dataset), "__getitem__", wrong_name)
        recorder = NativeRecorder(selected(), 0, clock=Clock())
        with pytest.raises(ValueError):
            next(recorder.cameras(loader, method))
        assert len(recorder.rows) == len(recorder.warmup_rows) == 0
    finally:
        original.close()

@pytest.mark.parametrize("method", ["STDLoc", "ULFLoc"])
def test_actual_entry_ast_binds_method_and_preserves_scene_loader_recipe(method):
    source = (BASE / "baselines" / method / (method.lower() + ".py")).read_text()
    original = ast.parse(source)
    transformed = ast.parse(transform_native(source, method))
    scene_calls = lambda t: [n for n in ast.walk(t) if isinstance(n, ast.Call)
                             and isinstance(n.func, ast.Name) and n.func.id == "Scene"]
    assert [ast.dump(n) for n in scene_calls(original)] == [ast.dump(n) for n in scene_calls(transformed)]
    assert len(scene_calls(original)) == 1
    keywords = {k.arg: k.value for k in scene_calls(original)[0].keywords}
    assert ast.literal_eval(keywords["shuffle"]) is False
    assert ast.literal_eval(keywords["preload_cameras"]) is False
    calls = [n for n in ast.walk(transformed) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == "_MATCHED" and n.func.attr == "cameras"]
    assert len(calls) == 1 and ast.literal_eval(calls[0].args[1]) == method
    assert isinstance(calls[0].args[0].func, ast.Attribute) and calls[0].args[0].func.attr == "getTestCameras"

