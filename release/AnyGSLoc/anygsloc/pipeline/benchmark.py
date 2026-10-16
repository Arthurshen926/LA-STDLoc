"""Three fresh timing rounds, fixed 100 queries, 50 excluded warmup calls."""

import hashlib
import math
import resource
import time
from pathlib import Path

SEED = 2026
NAMESPACE = "native_baseline_timing_subset_v1"
SCENES = ("ShopFacade", "OldHospital", "KingsCollege", "GreatCourt", "StMarysChurch")
ROUNDS = (0, 1, 2)
MEASURED = 100
WARMUP = 50


def selection(scene, full_names):
    names = list(full_names)
    if scene not in SCENES or len(names) < MEASURED or len(set(names)) != len(names):
        raise ValueError("invalid frozen scene/query names")
    if any(
        not isinstance(n, str)
        or not n
        or "\x00" in n
        or n.startswith("/")
        or ".." in Path(n).parts
        for n in names
    ):
        raise ValueError("unsafe complete query name")

    def key(name):
        return hashlib.sha256(
            ("\x00".join((NAMESPACE, str(SEED), scene, name))).encode()
        ).hexdigest(), name

    ranked = sorted(names, key=key)[:MEASURED]
    chosen = set(ranked)
    measured = [name for name in names if name in chosen]
    return dict(
        full_query_count=len(names),
        full_query_order=names,
        selection_rank=ranked,
        measured_query_order=measured,
        warmup_query_order=measured[:WARMUP],
    )


def schedule(record, round_id):
    if type(round_id) is not int or round_id not in ROUNDS:
        raise ValueError("round must be 0, 1 or 2")
    measured = record["measured_query_order"]
    warmup = record["warmup_query_order"]
    if (
        len(measured) != MEASURED
        or len(set(measured)) != MEASURED
        or warmup != measured[:WARMUP]
    ):
        raise ValueError("requires 100 unique measured and first 50 distinct warmups")
    return [(True, i, n) for i, n in enumerate(warmup)] + [
        (False, i, n) for i, n in enumerate(measured)
    ]


def finite_ms(value):
    return (
        isinstance(value, (float, int))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


class Recorder:
    """One fresh process/round; no feature caching or scientific methods live here."""

    def __init__(
        self,
        selected,
        round_id,
        clock=time.perf_counter,
        sync=lambda: None,
        reset_peak=lambda: None,
        memory=lambda: {},
        guard=lambda: None,
    ):
        self.schedule = schedule(selected, round_id)
        self.round_id = round_id
        self.clock, self.sync, self.reset_peak = clock, sync, reset_peak
        self.memory, self.guard = memory, guard
        self.rows, self.warmup_rows, self.active = [], [], None
        self.index = 0
        self.initialization_started = clock()
        self.initialization = None
        self.archive_loads_ms = []

    def begin(self, name):
        if self.active is not None or self.index >= len(self.schedule):
            raise ValueError("invalid frame state")
        warmup, order, expected = self.schedule[self.index]
        if name != expected:
            raise ValueError("query execution order changed")
        self.sync()
        if self.initialization is None:
            self.initialization = dict(
                total_ms=(self.clock() - self.initialization_started) * 1000,
                torch_peak=self.memory(),
            )
        self.guard()
        self.sync()
        self.reset_peak()
        self.active = dict(
            name=name,
            warmup=warmup,
            query_order=order,
            round=self.round_id,
            stages={},
            peaks=[self.memory()],
        )
        self.started = self.clock()

    def timed(self, label, operation):
        if self.active is None or label in self.active["stages"]:
            raise ValueError("duplicate or out-of-frame stage")
        self.sync()
        start = self.clock()
        result = operation()
        self.sync()
        self.active["stages"][label] = (self.clock() - start) * 1000
        self.active["peaks"].append(self.memory())
        return result

    def start_mask(self):
        if self.active is None or hasattr(self, "mask_started"):
            raise ValueError("invalid mask boundary")
        self.sync()
        self.mask_started = self.clock()

    def end_mask(self):
        if not hasattr(self, "mask_started"):
            raise ValueError("mask start missing")
        self.sync()
        self.active["stages"]["mask_preprocess_ms"] = (
            self.clock() - self.mask_started
        ) * 1000
        del self.mask_started
        self.active["peaks"].append(self.memory())

    def core(self, timing):
        if self.active is None or "core_ms" in self.active["stages"]:
            raise ValueError("core missing/duplicated")
        for target, source in (
            ("core_ms", "total_ms"),
            ("coarse_prefix_ms", "coarse_prefix_ms"),
            ("dense_increment_ms", "fine_increment_ms"),
        ):
            value = timing[source]
            if value is not None and not finite_ms(value):
                raise ValueError("invalid native timing")
            self.active["stages"][target] = value
        self.active["native_stage_timing"] = timing.get("native_stage_timing")
        self.active["peaks"].append(self.memory())

    def finish(self, name):
        if (
            self.active is None
            or self.active["name"] != name
            or hasattr(self, "mask_started")
        ):
            raise ValueError("mismatched frame end")
        self.sync()
        self.active["stages"]["frame_ms"] = (self.clock() - self.started) * 1000
        row = self.active
        if "core_ms" not in row["stages"]:
            raise ValueError("native core did not run")
        for value in row["stages"].values():
            if value is not None and not finite_ms(value):
                raise ValueError("nonfinite frame stage")
        row["torch_peak"] = {
            key: max(p.get(key, 0) for p in row["peaks"])
            for key in ("allocated_bytes", "reserved_bytes")
        }
        del row["peaks"]
        (self.warmup_rows if row["warmup"] else self.rows).append(row)
        self.active = None
        self.index += 1
        self.guard()
        return row["warmup"]

    def result(self):
        if (
            self.active is not None
            or self.index != len(self.schedule)
            or len(self.rows) != MEASURED
            or len(self.warmup_rows) != WARMUP
        ):
            raise ValueError("incomplete matched round")

        def peaks(rows):
            return {
                key: max(row["torch_peak"][key] for row in rows)
                for key in ("allocated_bytes", "reserved_bytes")
            }

        return dict(
            round=self.round_id,
            warmup_names=[r["name"] for r in self.warmup_rows],
            queries=self.rows,
            warmup_queries=self.warmup_rows,
            initialization=self.initialization,
            measured_torch_peak=peaks(self.rows),
            warmup_torch_peak=peaks(self.warmup_rows),
            mask_archive_loads_ms=self.archive_loads_ms,
            warmup_excluded_from_statistics=True,
        )


def plain(value):
    return value.tolist() if hasattr(value, "tolist") else value


def run_round(data, cameras, engine, selected, round_id, recorder):
    by_name = {camera.image_name: camera for camera in cameras}
    if len(by_name) != len(cameras):
        raise ValueError("duplicate Any camera")
    for warmup, order, name in recorder.schedule:
        camera = by_name[name]
        recorder.begin(name)
        image = recorder.timed(
            "rgb_decode_preprocess_ms", lambda: data.load_image(camera)
        )
        mask = recorder.timed("mask_compose_ms", lambda: data.valid_mask(camera))
        if mask is None:
            raise ValueError("semantic-on mask missing")
        result = engine.localize(
            image, fov_x=camera.fov_x, fov_y=camera.fov_y, valid_mask=mask
        )
        recorder.core(
            dict(
                total_ms=result["runtime_ms"],
                coarse_prefix_ms=None,
                fine_increment_ms=None,
            )
        )
        is_warmup = recorder.finish(name)
        pose = result.get("pose")
        matches = result.get("matches")
        rotation_error = translation_error = None
        if result.get("success") and hasattr(camera, "pose_w2c"):
            from anygsloc.evaluation.metrics import pose_error

            rotation_error, translation_error = pose_error(
                pose.pose_w2c, camera.pose_w2c
            )
            rotation_error, translation_error = (
                float(rotation_error),
                float(translation_error),
            )
        (recorder.warmup_rows if is_warmup else recorder.rows)[-1]["localization"] = (
            dict(
                pose_w2c=plain(getattr(pose, "pose_w2c", None)),
                success=result.get("success"),
                rotation_error_deg=rotation_error,
                translation_error_cm=translation_error,
                matches_count=len(matches.keypoint_indices)
                if matches is not None
                else None,
                inliers_count=len(pose.inliers)
                if pose is not None and hasattr(pose, "inliers")
                else None,
            )
        )
    value = recorder.result()
    return value


def run(spec, scene, map_path, output, device, round_id):
    import torch

    from anygsloc.data.calibration import resolve_scene_calibration
    from anygsloc.data.datasets import ColmapDataset
    from anygsloc.online.runtime import Localizer
    from anygsloc.pipeline.io import complete, prepare_output, reference, write_json

    out = prepare_output(
        output, {"scene": scene, "round": round_id, "map": reference(map_path)}
    )
    if spec.get("query_mask") != "explicit_valid_mask":
        raise ValueError("paper timing requires Cambridge query semantic masks")
    with torch.cuda.device(device):
        torch.set_num_threads(4)

        def memory():
            return {
                "allocated_bytes": torch.cuda.max_memory_allocated(),
                "reserved_bytes": torch.cuda.max_memory_reserved(),
            }

        # Dataset loading belongs to initialization, before warmup.
        start = time.perf_counter()
        data = ColmapDataset(spec["dataset"], load_masks=True)
        cameras = data.split("test")
        selected = selection(scene, [c.image_name for c in cameras])
        cal = resolve_scene_calibration(data.split("mapping"))
        recorder = Recorder(
            selected,
            round_id,
            sync=torch.cuda.synchronize,
            reset_peak=torch.cuda.reset_peak_memory_stats,
            memory=memory,
        )
        recorder.initialization_started = start
        engine = Localizer(
            map_path,
            keypoint_count=cal["keypoint_count"],
            reprojection_error_px=cal["reprojection_error_px"],
            mask_policy="explicit_valid_mask",
            device=device,
            seed=2026,
        )
        value = run_round(data, cameras, engine, selected, round_id, recorder)
        value.update(
            selection=selected,
            scene=scene,
            storage_bytes=Path(map_path).stat().st_size,
            method="AnyGSLoc",
            map_name=out.parent.name,
            map_sha256=reference(map_path)["sha256"],
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            versions={"torch": torch.__version__, "cuda": torch.version.cuda},
            device=torch.cuda.get_device_name(device),
            segmentation_inference_included=False,
        )
    write_json(out / "round.json", value)
    complete(out, ["round.json"])
