"""Native source-only instrumentation. The scientific classes are byte/AST unchanged."""
import argparse
import ast
import os
from pathlib import Path
import sys
import time
from .common import Recorder, read, ref, verify, write_new, validate_result, native_evidence
from experiments.native_baseline_repair.query_semantic_adapter import once, std_text, wrapper_text, mask_rgb

def frozen_helper_search_path(wrapper_path, search_path, modules):
    """Keep original frozen Any data helpers ahead of the worker package repository."""
    frozen = Path(wrapper_path).resolve().parents[1]
    for name, module in modules.items():
        if name == "anygsloc" or name.startswith("anygsloc."):
            source = getattr(module, "__file__", None)
            if source is None or os.path.commonpath((str(frozen), str(Path(source).resolve()))) != str(frozen):
                raise ValueError("native data helper already imported from outside frozen wrapper tree")
    return [str(frozen)] + [p for p in search_path if p != str(frozen)]

def transform_native(original, method):
    text = std_text(original) if method == "STDLoc" else original
    text = once(text, "    test_cameras = scene.getTestCameras()\n",
                "    test_cameras = _MATCHED.cameras(scene.getTestCameras(), %r)\n" % method)
    if method == "STDLoc":
        text = once(text,
                    "        query_image = _QUERY_MASK_RGB(camera_info.image_name, query_image)\n",
                    "        _MATCHED.start_mask()\n"
                    "        query_image = _QUERY_MASK_RGB(camera_info.image_name, query_image)\n"
                    "        _MATCHED.end_mask()\n")
        call = "        loc_res = stdloc.localize(query_image, fovx, fovy)\n"
    elif method == "ULFLoc":
        text = once(text, "        if masks is not None:\n",
                    "        _MATCHED.start_mask()\n        if masks is not None:\n")
        text = once(text, "        fovx = camera_info.FoVx\n",
                    "        _MATCHED.end_mask()\n        fovx = camera_info.FoVx\n")
        call = "        loc_res = ulfloc.localize(camera_info, query_image, fovx, fovy)\n"
    else:
        raise ValueError("unknown native method")
    text = once(text, call, call +
                "        if _MATCHED.finish(camera_info.image_name, loc_res):\n"
                "            continue\n")
    ast.parse(text)
    before, after = ast.parse(original), ast.parse(text)
    bclasses = [ast.dump(n) for n in before.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    aclasses = [ast.dump(n) for n in after.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    if bclasses != aclasses:
        raise ValueError("native scientific definitions changed")
    return text

def transform_wrapper(original):
    text = wrapper_text(original)
    text = once(text,
        "                    native_stage_timing=result.get('timing_ms')))\n",
        "                    native_stage_timing=result.get('timing_ms')))\n"
        "                _MATCHED.core(timings[-1])\n")
    text = once(text, "            raw=original_load(stream,*args,**kwargs)\n",
                "            raw=_MATCHED.archive_load(original_load,stream,*args,**kwargs)\n")
    text = once(text, "                query_masks=TrainingMasks(original_load(query_stream))\n",
                "                query_masks=TrainingMasks(_MATCHED.archive_load(original_load,query_stream))\n")
    # Fail closed rather than re-entering an unrelated delegated job or repeating frame0.
    text = once(text, "    delegation = a.receipt.parent.parent / 'delegation.json'\n",
                "    if a.warmup != 0: raise ValueError('legacy repeated-frame warmup forbidden')\n"
                "    delegation = a.receipt.parent.parent / 'delegation.json'\n")
    text = once(text, "    if delegation.is_file():\n",
                "    if delegation.is_file(): raise ValueError('legacy delegation forbidden')\n"
                "    if False:\n")
    ast.parse(text)
    return text

def fresh_camera(dataset, index, decode, materialize):
    """Fresh PIL decode, then the exact native dataset __getitem__ and original index."""
    from PIL import Image
    infos = dataset.scene.scene_info.test_cameras
    old = infos[index]
    def open_decode():
        image = Image.open(old.image_path)
        try:
            image.load()
        except BaseException:
            image.close()
            raise
        return image
    fresh = decode(open_decode)
    try:
        infos[index] = old._replace(image=fresh)
        try:
            return materialize(lambda: dataset[index])
        finally:
            infos[index] = old
    finally:
        fresh.close()

def native_test_dataset(loader, method):
    """Validate original loader configuration without starting its iterator/workers."""
    from torch.utils.data import DataLoader, Dataset, SequentialSampler
    from torch.utils.data._utils.collate import default_convert
    workers = {"STDLoc": 4, "ULFLoc": 0}
    if method not in workers or type(loader) is not DataLoader:
        raise ValueError("requires original native DataLoader and known method")
    dataset = loader.dataset
    if (not isinstance(dataset, Dataset) or type(dataset).__name__ != "SceneDataset"
            or type(dataset).__module__ != "scene" or getattr(dataset, "split", None) != "test"):
        raise ValueError("requires original native SceneDataset test split")
    if (loader.num_workers != workers[method] or loader.batch_size is not None
            or loader.batch_sampler is not None or loader.collate_fn is not default_convert
            or type(loader.sampler) is not SequentialSampler or loader.pin_memory is not True
            or loader.drop_last is not False or loader.persistent_workers is not False):
        raise ValueError("native test loader configuration differs from frozen method")
    # The matched synchronous frame intentionally materializes dataset[index].
    # Constructing a loader does not launch workers. Never iterate it or prefetch.
    return dataset

class NativeRecorder(Recorder):
    def finish(self, name, result=None):
        warmup = super().finish(name)
        if result is not None:
            (self.warmup_rows if warmup else self.rows)[-1]["localization"] = native_evidence(result)
        return warmup

    def archive_load(self, loader, *args, **kwargs):
        begin = self.clock()
        result = loader(*args, **kwargs)
        self.archive_loads_ms.append((self.clock() - begin) * 1000)
        return result

    def cameras(self, loader, method):
        dataset = native_test_dataset(loader, method)
        infos = dataset.scene.scene_info.test_cameras
        indexes = {c.image_name: i for i, c in enumerate(infos)}
        if len(indexes) != len(infos):
            raise ValueError("duplicate native query metadata")
        # No camera/feature cache; call native materialization again for every frame.
        for warmup, order, name in self.schedule:
            if name not in indexes:
                raise ValueError("frozen query absent from native metadata")
            self.begin(name)
            camera = fresh_camera(dataset, indexes[name],
                    lambda op: self.timed("rgb_decode_ms", op),
                    lambda op: self.timed("camera_materialize_ms", op))
            if camera.image_name != name:
                raise ValueError("native camera name mismatch")
            yield camera
            if self.active is not None:
                raise ValueError("native frame was not completed")

def execute(c, runtime):
    import torch
    from .plan import native_contract
    from .isolation import Monitor, runtime_gate
    runtime_gate(c, runtime)
    nc = native_contract(c)
    out = Path(c["output"])
    model = Path(nc["model_view"])
    model.mkdir(parents=True, exist_ok=False)
    for rel, value in nc["model_assets"].items():
        target = model / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(value["path"])
    def memory():
        return dict(allocated_bytes=torch.cuda.max_memory_allocated(),
                    reserved_bytes=torch.cuda.max_memory_reserved())
    with Monitor(runtime) as monitor:
        recorder = NativeRecorder(c["selection"], c["round"], sync=torch.cuda.synchronize,
                                  reset_peak=torch.cuda.reset_peak_memory_stats,
                                  memory=memory, guard=monitor.check)
        def controlled(path, method, masks):
            if Path(path).resolve() != Path(nc["native_entry"]["path"]).resolve() or method != c["cell"]["method"]:
                raise ValueError("native execution target changed")
            scope = dict(__name__="__main__", __file__=path, __package__="",
                         _MATCHED=recorder, _CONTROLLED_OUTPUT=nc["native_output"],
                         _QUERY_MASK_RGB=lambda name, image: mask_rgb(name, image, masks, set(nc["query_names"])))
            source = Path(path).read_text()
            exec(compile(transform_native(source, method), path, "exec"), scope)
            return scope
        scope = dict(__name__="__main__", __file__=nc["wrapper"]["path"], __package__="",
                     _MATCHED=recorder, _CONTROLLED_NATIVE_RUN=controlled)
        sys.path[:] = frozen_helper_search_path(nc["wrapper"]["path"], sys.path, sys.modules)
        sys.argv = [nc["wrapper"]["path"]] + nc["wrapper_argv"]
        exec(compile(transform_wrapper(Path(nc["wrapper"]["path"]).read_text()), nc["wrapper"]["path"], "exec"), scope)
        result = recorder.result()
        native_rows = read(Path(nc["native_output"]) / "results.json")
        if len(native_rows) != len(result["queries"]):
            raise ValueError("native result count differs from measured queries")
        for row, native_row in zip(result["queries"], native_rows):
            row["localization"] = native_evidence(native_row)
        for row in result["warmup_queries"] + result["queries"]:
            row["stages"]["mask_compose_ms"] = None
        result["boundaries"] = dict(core="native unchanged; ULF GT diagnostic inclusive" if nc["method"] == "ULFLoc" else "native unchanged; no GT passed to core",
                                   rgb_decode_ms="PIL reopen+load; no resize", camera_materialize_ms="native SceneDataset __getitem__; camera tensors/resize/geometry/H2D",
                                   mask_preprocess_ms="native synchronized compose/device-transfer/application",
                                   frame_ms="continuous decode through native core return; includes native pre-core bookkeeping",
                                   segmentation_network_ms=None)
        monitor.check()
        weights = Path(nc["native_entry"]["path"]).parent / "encoders/sp_encoder/weights/superpoint_v1.pth"
        result.update(contract=ref(runtime["contract_path"]), lease=runtime["lease_ref"],
                      method=nc["method"], case=c["case"], storage_bytes=c["cell"]["storage_bytes"],
                      shared_feature_weights_excluded_from_map_storage=ref(weights),
                      versions=dict(python=__import__("platform").python_version(), torch=torch.__version__,
                                    cuda=torch.version.cuda, torch_threads=torch.get_num_threads()),
                      native_receipt=ref(nc["receipt"]), native_results=ref(Path(nc["native_output"]) / "results.json"),
                      accuracy_reference=c["cell"]["accuracy_reference"], host_samples=monitor.samples,
                      process_peak_rss_kib=__import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss,
                      gpu_uuid=runtime["gpu_uuid"])
        validate_result(result, c["selection"], c["round"], "native")
    result["monitor_closed_clean"] = monitor.closed_clean
    result["context_proofs"] = monitor.context_proofs
    write_new(out / "round.json", result)
