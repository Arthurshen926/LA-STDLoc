"""Pure CPU contracts and state machine for paired timing. Compatible with Python 3.8."""
import hashlib
import json
import math
from pathlib import Path
import time

SEED = 2026
NAMESPACE = "native_baseline_timing_subset_v1"
SCENES = ("ShopFacade", "OldHospital", "KingsCollege", "GreatCourt", "StMarysChurch")
ROUNDS = (0, 1, 2)
MEASURED = 100
WARMUP = 50
ROOT = Path("/mnt/pool/sqy/anygsloc_native_baseline_stmary_repair_20261012_v1")
PREPARATION = ROOT / "matched_timing_preparation"
ANY_ROOT = Path("/mnt/pool/sqy/anygsloc_cambridge_deployment_benchmark_20261011_v1")
REPO = Path(__file__).resolve().parents[2]
PENDING_SHA = "d974b57c1e80d70cc068010be2e04e63f02f207ec9a48c694cab00da3c10357c"
QUERY_SHA = "54bfd40d4c9a19e8a423b9a2fd35e448a8c2f2d16f2f72d924a1d42c459a46ad"
ANY_SHA = "cf2ff778c23acaeabeb66d004349248c5f6503f14984d85c8e5b4f752afe43f4"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def ref(path):
    path = Path(path)
    return dict(path=str(path.resolve()), sha256=sha(path), bytes=path.stat().st_size)

def verify(value):
    if sha(value["path"]) != value["sha256"] or ("bytes" in value and Path(value["path"]).stat().st_size != value["bytes"]):
        raise ValueError("frozen input changed: " + value["path"])

def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)

def selection(scene, full_names):
    names = list(full_names)
    if scene not in SCENES or len(names) < MEASURED or len(set(names)) != len(names):
        raise ValueError("invalid frozen scene/query names")
    if any(not isinstance(n, str) or not n or "\x00" in n or n.startswith("/") or ".." in Path(n).parts for n in names):
        raise ValueError("unsafe complete query name")
    def key(name):
        return hashlib.sha256(("\x00".join((NAMESPACE, str(SEED), scene, name))).encode()).hexdigest(), name
    ranked = sorted(names, key=key)[:MEASURED]
    chosen = set(ranked)
    measured = [name for name in names if name in chosen]
    return dict(full_query_count=len(names), full_query_order=names, selection_rank=ranked,
                measured_query_order=measured, warmup_query_order=measured[:WARMUP])

def check_selection(scene, record):
    expected = selection(scene, record["full_query_order"])
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError("selection/order tampered: " + key)
    return expected

def schedule(record, round_id):
    if type(round_id) is not int or round_id not in ROUNDS:
        raise ValueError("round must be 0, 1 or 2")
    measured = record["measured_query_order"]
    warmup = record["warmup_query_order"]
    if len(measured) != MEASURED or len(set(measured)) != MEASURED or warmup != measured[:WARMUP]:
        raise ValueError("requires 100 unique measured and first 50 distinct warmups")
    return [(True, i, n) for i, n in enumerate(warmup)] + [(False, i, n) for i, n in enumerate(measured)]

def finite_ms(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0

class Recorder:
    """One fresh process/round; no feature caching or scientific methods live here."""
    def __init__(self, selected, round_id, clock=time.perf_counter, sync=lambda: None,
                 reset_peak=lambda: None, memory=lambda: {}, guard=lambda: None):
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
            self.initialization = dict(total_ms=(self.clock() - self.initialization_started) * 1000,
                                       torch_peak=self.memory())
        self.guard()
        self.sync()
        self.reset_peak()
        self.active = dict(name=name, warmup=warmup, query_order=order, round=self.round_id,
                           stages={}, peaks=[self.memory()])
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
        self.active["stages"]["mask_preprocess_ms"] = (self.clock() - self.mask_started) * 1000
        del self.mask_started
        self.active["peaks"].append(self.memory())

    def core(self, timing):
        if self.active is None or "core_ms" in self.active["stages"]:
            raise ValueError("core missing/duplicated")
        for target, source in (("core_ms", "total_ms"), ("coarse_prefix_ms", "coarse_prefix_ms"),
                               ("dense_increment_ms", "fine_increment_ms")):
            value = timing[source]
            if value is not None and not finite_ms(value):
                raise ValueError("invalid native timing")
            self.active["stages"][target] = value
        self.active["native_stage_timing"] = timing.get("native_stage_timing")
        self.active["peaks"].append(self.memory())

    def finish(self, name):
        if self.active is None or self.active["name"] != name or hasattr(self, "mask_started"):
            raise ValueError("mismatched frame end")
        self.sync()
        self.active["stages"]["frame_ms"] = (self.clock() - self.started) * 1000
        row = self.active
        if "core_ms" not in row["stages"]:
            raise ValueError("native core did not run")
        for value in row["stages"].values():
            if value is not None and not finite_ms(value):
                raise ValueError("nonfinite frame stage")
        row["torch_peak"] = {key: max(p.get(key, 0) for p in row["peaks"]) for key in ("allocated_bytes", "reserved_bytes")}
        del row["peaks"]
        (self.warmup_rows if row["warmup"] else self.rows).append(row)
        self.active = None
        self.index += 1
        self.guard()
        return row["warmup"]

    def result(self):
        if self.active is not None or self.index != len(self.schedule) or len(self.rows) != MEASURED or len(self.warmup_rows) != WARMUP:
            raise ValueError("incomplete matched round")
        peaks = lambda rows: {key: max(row["torch_peak"][key] for row in rows) for key in ("allocated_bytes", "reserved_bytes")}
        return dict(round=self.round_id, warmup_names=[r["name"] for r in self.warmup_rows],
                    queries=self.rows, warmup_queries=self.warmup_rows, initialization=self.initialization,
                    measured_torch_peak=peaks(self.rows), warmup_torch_peak=peaks(self.warmup_rows),
                    mask_archive_loads_ms=self.archive_loads_ms, warmup_excluded_from_statistics=True)

def validate_peak(peak):
    if set(peak) != {"allocated_bytes", "reserved_bytes"} or any(type(v) is not int or v < 0 for v in peak.values()):
        raise ValueError("invalid or missing peak memory")

def validate_pose(pose):
    if not isinstance(pose, list) or len(pose) != 4 or any(not isinstance(r, list) or len(r) != 4 for r in pose):
        raise ValueError("missing/invalid pose evidence")
    if any(not isinstance(v, (float, int)) or isinstance(v, bool) or not math.isfinite(v) for r in pose for v in r):
        raise ValueError("nonfinite pose evidence")

def plain(value):
    return value.tolist() if hasattr(value, "tolist") else value

def native_evidence(result):
    sparse, dense = result["sparse"], result["dense"][-1]
    return dict(success=result.get("success"), sparse_pose_w2c=plain(sparse["pose_w2c"]),
                dense_pose_w2c=plain(dense["pose_w2c"]), sparse_inliers=plain(sparse.get("inliers")),
                dense_inliers=plain(dense.get("inliers")), matches=plain(result.get("matches")),
                errors=dict(sparse_rotation_deg=result.get("sparse_AE"), sparse_translation_cm=result.get("sparse_TE"),
                            dense_rotation_deg=result.get("dense_AE"), dense_translation_cm=result.get("dense_TE")))

def validate_result(value, selected, round_id, family=None):
    schedule(selected, round_id)
    if value.get("round") != round_id or value.get("warmup_excluded_from_statistics") is not True:
        raise ValueError("wrong round identity")
    rows = value["queries"]
    if [r["name"] for r in rows] != selected["measured_query_order"] or value["warmup_names"] != selected["warmup_query_order"]:
        raise ValueError("result query sequence differs")
    for warm, records in ((False, rows), (True, value["warmup_queries"])):
        expected = selected["warmup_query_order"] if warm else selected["measured_query_order"]
        if [r["name"] for r in records] != expected:
            raise ValueError("warmup result sequence differs")
        for i, row in enumerate(records):
            if row["warmup"] is not warm or row["round"] != round_id or row["query_order"] != i:
                raise ValueError("result query identity changed")
            for val in row["stages"].values():
                if val is not None and not finite_ms(val):
                    raise ValueError("invalid result stage")
            if not finite_ms(row["stages"].get("core_ms")) or not finite_ms(row["stages"].get("frame_ms")):
                raise ValueError("required measurement missing")
            if family is not None:
                stages = row["stages"]
                required = (("rgb_decode_ms", "camera_materialize_ms", "mask_preprocess_ms", "coarse_prefix_ms", "dense_increment_ms")
                            if family == "native" else ("rgb_decode_preprocess_ms", "mask_compose_ms"))
                unavailable = ("mask_compose_ms",) if family == "native" else ("coarse_prefix_ms", "dense_increment_ms")
                if any(not finite_ms(stages.get(k)) for k in required) or any(k not in stages or stages[k] is not None for k in unavailable):
                    raise ValueError("missing family timing boundary or invented unavailable timing")
                if family == "native" and not math.isclose(stages["core_ms"], stages["coarse_prefix_ms"] + stages["dense_increment_ms"], rel_tol=1e-9, abs_tol=1e-6):
                    raise ValueError("native coarse+dense does not equal core")
                validate_peak(row["torch_peak"])
                evidence = row.get("localization", {})
                if family == "any":
                    if type(evidence.get("success")) is not bool:
                        raise ValueError("Any failure/success evidence missing")
                    validate_pose(evidence.get("pose_w2c"))
                    for k in ("matches_count", "inliers_count"):
                        if k not in evidence or (evidence[k] is not None and (type(evidence[k]) is not int or evidence[k] < 0)):
                            raise ValueError("Any match/inlier evidence invalid")
                else:
                    validate_pose(evidence.get("sparse_pose_w2c"))
                    validate_pose(evidence.get("dense_pose_w2c"))
    if family is not None:
        if family not in ("native", "any") or not finite_ms(value["initialization"].get("total_ms")):
            raise ValueError("initialization or family missing")
        validate_peak(value["initialization"]["torch_peak"])
        for key, records in (("measured_torch_peak", rows), ("warmup_torch_peak", value["warmup_queries"])):
            validate_peak(value[key])
            expected = {k: max(r["torch_peak"][k] for r in records) for k in value[key]}
            if value[key] != expected:
                raise ValueError("phase peak summary differs from per-frame peaks")
    return value
