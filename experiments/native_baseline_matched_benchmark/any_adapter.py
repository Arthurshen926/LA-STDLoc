"""New subset runs reuse the accepted Any dataset and unchanged Localizer."""
from pathlib import Path
import resource
from .common import Recorder, read, ref, write_new, validate_result, plain

def run_round(data, cameras, engine, selected, round_id, recorder):
    by_name = {camera.image_name: camera for camera in cameras}
    if len(by_name) != len(cameras):
        raise ValueError("duplicate Any camera")
    for warmup, order, name in recorder.schedule:
        camera = by_name[name]
        recorder.begin(name)
        image = recorder.timed("rgb_decode_preprocess_ms", lambda: data.load_image(camera))
        mask = recorder.timed("mask_compose_ms", lambda: data.valid_mask(camera))
        if mask is None:
            raise ValueError("semantic-on mask missing")
        result = engine.localize(image, fov_x=camera.fov_x, fov_y=camera.fov_y, valid_mask=mask)
        recorder.core(dict(total_ms=result["runtime_ms"], coarse_prefix_ms=None, fine_increment_ms=None))
        is_warmup = recorder.finish(name)
        pose = result.get("pose")
        matches = result.get("matches")
        rotation_error = translation_error = None
        if result.get("success") and hasattr(camera, "pose_w2c"):
            from anygsloc.evaluation.metrics import pose_error
            rotation_error, translation_error = pose_error(pose.pose_w2c, camera.pose_w2c)
            rotation_error, translation_error = float(rotation_error), float(translation_error)
        (recorder.warmup_rows if is_warmup else recorder.rows)[-1]["localization"] = dict(
            pose_w2c=plain(getattr(pose, "pose_w2c", None)), success=result.get("success"),
            rotation_error_deg=rotation_error, translation_error_cm=translation_error,
            matches_count=len(matches.keypoint_indices) if matches is not None else None,
            inliers_count=len(pose.inliers) if pose is not None and hasattr(pose, "inliers") else None)
    value = recorder.result()
    validate_result(value, selected, round_id, "any")
    return value

def execute(c, runtime):
    from .isolation import Monitor, runtime_gate
    runtime_gate(c, runtime)
    import torch
    from experiments import cambridge_deployment_benchmark as B
    p = read(c["protocol"]["path"])
    old = read(p["any_protocol"]["path"])
    item = old["scenes"][c["cell"]["scene"]]
    def memory():
        return dict(allocated_bytes=torch.cuda.max_memory_allocated(),
                    reserved_bytes=torch.cuda.max_memory_reserved())
    with Monitor(runtime) as monitor:
        recorder = Recorder(c["selection"], c["round"], sync=torch.cuda.synchronize,
                            reset_peak=torch.cuda.reset_peak_memory_stats, memory=memory, guard=monitor.check)
        torch.set_num_threads(4)
        start = recorder.clock()
        data = B.ColmapDataset(item["scene_info"]["dataset"], load_masks=True)
        dataset_ms = (recorder.clock() - start) * 1000
        cameras = data.split("test")
        if [camera.image_name for camera in cameras] != item["scene_info"]["test_names"]:
            raise ValueError("Any frozen full query order changed")
        cal = item["calibration"]
        engine = B.Localizer(c["cell"]["map"]["path"], keypoint_count=2048,
                            reprojection_error_px=cal["reprojection_error_px"],
                            mask_policy="explicit_valid_mask", device="cuda:0", seed=2026)
        expected = B.A.read((B.BASE / "evaluations/anygsloc/manifest.json") if c["cell"]["scene"] == B.S.SCENE else
                           (B.A.MATRIX / "evaluations" / c["cell"]["scene"] / "pure_gs/anygsloc/full/manifest.json"))["config"]
        for key, value in expected.items():
            if key in engine.config and engine.config[key] != value:
                raise ValueError("Any Localizer config differs: " + key)
        from anygsloc.features.superpoint import resolve_superpoint_weights
        weights = ref(resolve_superpoint_weights())
        value = run_round(data, cameras, engine, c["selection"], c["round"], recorder)
        value.update(contract=ref(runtime["contract_path"]), lease=runtime["lease_ref"],
                     case=c["case"], method="AnyGSLoc", variant=c["cell"]["variant"],
                     accuracy_reference=c["cell"]["accuracy_reference"],
                     storage_bytes=c["cell"]["storage_bytes"], gpu_uuid=runtime["gpu_uuid"],
                     shared_feature_weights_excluded_from_map_storage=weights,
                     versions=dict(python=__import__("platform").python_version(), torch=torch.__version__,
                                   cuda=torch.version.cuda, torch_threads=torch.get_num_threads()),
                     dataset_including_mask_archive_ms=dataset_ms,
                     process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     boundaries=dict(core_ms="unchanged Localizer.runtime_ms: H2D, semantic application, features, matching, PnP",
                                     frame_ms="decode/preprocess+CPU mask composition+localize; synchronized",
                                     coarse_prefix_ms=None, dense_increment_ms=None,
                                     segmentation_network_ms=None),
                     host_samples=monitor.samples)
        monitor.check()
    value["monitor_closed_clean"] = monitor.closed_clean
    value["context_proofs"] = monitor.context_proofs
    write_new(Path(c["output"]) / "round.json", value)
