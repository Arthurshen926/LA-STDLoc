"""CPU-only, fail-closed acceptance of completed staged native STDLoc training."""
import argparse
import ast
from collections import OrderedDict
from datetime import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import re
import sys
import tempfile

# This executable never obtains a CUDA device, including while loading products.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

PROTOCOL_SHA = "75bb2ff18d072c8893c12818678bd0dc82f77ca0e8c74c2cf5e912d53645487a"
BUILD_SHA = "de52a7d507fa68695d35f680b1fd88bcb2f67845fa3c029285aeaa4805e9cbb7"
BASE = Path("/mnt/pool/sqy/anygsloc_baseline_recovery_v4_20260915")
STAGES = (("gaussian_1", 0, 10000), ("gaussian_2", 10000, 20000),
          ("gaussian_3", 20000, 30000), ("detector", 30000, 30000))
STATE_KEYS = {"schema", "identity", "iteration", "scene", "options", "dataset", "masks",
              "gaussian_capture", "gaussian_extra", "gaussian_constants", "optimizer_binding",
              "feature_state", "feature_extra", "background", "remaining_stack",
              "ema_loss_for_log", "runtime_flags", "rng"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ref(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise ValueError("missing, empty or symlinked product: " + str(path))
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def verify(reference):
    path = Path(reference["path"])
    if path.stat().st_size != reference["bytes"] or sha(path) != reference["sha256"]:
        raise ValueError("artifact drift: " + str(path))


def read(path):
    return json.loads(Path(path).read_text())


def require_values(value, expected, label):
    for key, wanted in expected.items():
        actual = value.get(key)
        if type(wanted) is bool:
            valid = actual is wanted
        else:
            valid = type(actual) is type(wanted) and actual == wanted
        if not valid:
            raise ValueError(label + " mismatch: " + key)


def manifest_hash(manifest):
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def primitive_manifest(value):
    if isinstance(value, dict):
        return {"dict": [[primitive_manifest(k), primitive_manifest(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {type(value).__name__: [primitive_manifest(v) for v in value]}
    if value is None or isinstance(value, (bool, int, str)):
        return {type(value).__name__: value}
    if isinstance(value, float):
        return {"float": value.hex()}
    raise ValueError("unsupported primitive manifest")


def manifest_fields(value):
    if not isinstance(value, dict) or set(value) != {"dict"}:
        raise ValueError("checkpoint root manifest is not a typed dictionary")
    result = {}
    for item in value["dict"]:
        if not isinstance(item, list) or len(item) != 2 or set(item[0]) != {"str"}:
            raise ValueError("malformed checkpoint root key")
        key = item[0]["str"]
        if key in result:
            raise ValueError("duplicate checkpoint manifest key")
        result[key] = item[1]
    return result


def check_terminal(value, case, semantic, stage, config_path, launch_path):
    require_values(value, {"case": case, "stage": stage, "mapping_semantics": semantic,
                          "terminal": True, "exit_code": 0,
                          "evidence_source": "WebCodex observe_jobs",
                          "config_sha256": sha(config_path), "launch_sha256": sha(launch_path)},
                   "official Runner terminal")
    if not re.fullmatch(r"wc_job_[A-Za-z0-9_-]+", value.get("job_id", "")):
        raise ValueError("missing official Runner job ID")
    try:
        observed = datetime.fromisoformat(value["observed_utc"].replace("Z", "+00:00"))
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError("invalid Runner observation timestamp") from exc
    if observed.tzinfo is None:
        raise ValueError("Runner observation timestamp needs timezone")
    return value["job_id"]


def check_checkpoint(directory, iteration, identity, verify_blob=True):
    directory = Path(directory)
    record = read(directory / "complete.json")
    require_values(record, {"schema": "std_native_exact_state_v1", "complete": True,
                            "iteration": iteration, "identity": identity, "blob": "state.pt"},
                   "checkpoint")
    if manifest_hash(record["state_manifest"]) != record["state_digest"]:
        raise ValueError("checkpoint immutable manifest digest mismatch")
    fields = manifest_fields(record["state_manifest"])
    if set(fields) != STATE_KEYS:
        raise ValueError("incomplete checkpoint state coverage")
    if fields["iteration"] != primitive_manifest(iteration) or fields["identity"] != primitive_manifest(identity):
        raise ValueError("checkpoint manifest identity mismatch")
    if fields["schema"] != primitive_manifest("std_native_exact_state_v1"):
        raise ValueError("checkpoint manifest schema mismatch")
    proof = read(directory / "roundtrip_verified.json")
    require_values(proof, {"complete": True, "iteration": iteration,
                           "state_digest": record["state_digest"], "blob_sha256": record["blob_sha256"],
                           "roundtrip_location": "cpu", "original_storage_device_metadata_verified": True},
                   "save/load roundtrip")
    blob = directory / "state.pt"
    if blob.is_symlink() or not blob.is_file() or blob.stat().st_size != record["blob_bytes"]:
        raise ValueError("checkpoint blob size/path mismatch")
    if verify_blob and sha(blob) != record["blob_sha256"]:
        raise ValueError("checkpoint blob hash mismatch")
    return record, manifest_hash(fields["rng"])


def check_restore(value, checkpoint, rng_digest, iteration):
    require_values(value, {"complete": True, "iteration": iteration,
                           "state_digest": checkpoint["state_digest"], "rng_digest": rng_digest},
                   "restore_verified")


class RestrictedTensorUnpickler(pickle.Unpickler):
    """Native tensor pickle only; nested torch storage is weights-only and CPU."""
    def find_class(self, module, name):
        import torch
        allowed = {
            ("torch._utils", "_rebuild_tensor"): torch._utils._rebuild_tensor,
            ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
            ("collections", "OrderedDict"): OrderedDict,
        }
        if (module, name) == ("torch.storage", "_load_from_bytes"):
            return lambda data: torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        if (module, name) in allowed:
            return allowed[(module, name)]
        raise pickle.UnpicklingError("unapproved native pickle global: " + module + "." + name)

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("unexpected outer pickle persistent ID")


def load_indices(path):
    import torch
    with Path(path).open("rb") as stream:
        value = RestrictedTensorUnpickler(stream).load()
        if stream.read(1):
            raise ValueError("trailing native indices pickle data")
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise ValueError("native indices must be a CPU-loaded tensor")
    return value


def check_indices(value, point_count):
    import torch
    if value.dtype not in (torch.int32, torch.int64) or value.ndim != 1:
        raise ValueError("indices must be one-dimensional native integers")
    if value.numel() != 16384 or value.unique().numel() != 16384:
        raise ValueError("expected exactly 16384 unique native landmarks")
    if int(value.min()) < 0 or int(value.max()) >= point_count:
        raise ValueError("landmark index outside Gaussian PLY")
    return {"landmarks": 16384, "indices_dtype": str(value.dtype),
            "minimum_index": int(value.min()), "maximum_index": int(value.max())}


def check_detector(path):
    import torch
    with Path(path).open("rb") as stream:
        state = torch.load(stream, map_location="cpu", weights_only=True)
    expected = {}
    for layer, inp, out in ((0, 256, 128), (2, 128, 64), (4, 64, 32), (6, 32, 1)):
        expected["cnn." + str(layer) + ".weight"] = (out, inp, 3, 3)
        expected["cnn." + str(layer) + ".bias"] = (out,)
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("terminal native KpDetector keys mismatch")
    for key, shape in expected.items():
        tensor = state[key]
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                or tensor.dtype != torch.float32 or tuple(tensor.shape) != shape
                or not bool(torch.isfinite(tensor).all())):
            raise ValueError("invalid terminal detector tensor: " + key)
    return {"detector_tensor_count": len(state), "detector_feature_dim": 256,
            "detector_all_finite": True}


def expected_ply_fields():
    return (["x", "y", "z", "nx", "ny", "nz"]
            + ["f_dc_" + str(i) for i in range(3)]
            + ["f_rest_" + str(i) for i in range(45)] + ["opacity"]
            + ["scale_" + str(i) for i in range(3)]
            + ["rot_" + str(i) for i in range(4)]
            + ["loc_" + str(i) for i in range(256)])


def check_ply(path):
    import numpy as np
    from plyfile import PlyData
    ply = PlyData.read(str(path), mmap="r")
    if len(ply.elements) != 1 or ply.elements[0].name != "vertex":
        raise ValueError("unexpected native Gaussian PLY elements")
    vertex = ply["vertex"].data
    if len(vertex) < 16384 or list(vertex.dtype.names or ()) != expected_ply_fields():
        raise ValueError("native full FeatureGaussian PLY schema/count mismatch")
    for name in vertex.dtype.names:
        if vertex.dtype[name].kind != "f" or vertex.dtype[name].itemsize != 4:
            raise ValueError("unexpected native PLY field dtype: " + name)
        # Column chunks bound working memory even for multi-million-point PLY.
        for start in range(0, len(vertex), 262144):
            if not np.isfinite(vertex[name][start:start + 262144]).all():
                raise ValueError("nonfinite PLY field: " + name)
    return {"point_count": len(vertex), "ply_field_count": len(vertex.dtype.names),
            "ply_feature_dim": 256, "ply_all_fields_finite": True}


def parse_cfg(path):
    try:
        root = ast.parse(Path(path).read_text(), mode="eval").body
    except SyntaxError as exc:
        raise ValueError("invalid native cfg_args syntax") from exc
    if (not isinstance(root, ast.Call) or not isinstance(root.func, ast.Name)
            or root.func.id != "Namespace" or root.args):
        raise ValueError("unexpected native cfg_args")
    result = {}
    for keyword in root.keywords:
        if keyword.arg is None or keyword.arg in result:
            raise ValueError("duplicate or expanded cfg_args keyword")
        result[keyword.arg] = ast.literal_eval(keyword.value)
    return result


def check_cfg(path, model, source):
    cfg = parse_cfg(path)
    expected = {"source_path": source, "model_path": str(model), "sh_degree": 3,
                "data_device": "cpu", "eval": False, "feature_type": "sp",
                "gaussian_type": "3dgs", "images": "processed", "longest_edge": 640,
                "norm_before_render": True, "resolution": 1, "speedup": False,
                "white_background": True,
                "render_items": ["RGB", "Depth", "Edge", "Normal", "Curvature", "Feature Map"]}
    require_values(cfg, expected, "native cfg_args")
    if set(cfg) != set(expected):
        raise ValueError("unexpected native cfg_args fields")
    return cfg


def check_cameras(path, source, readers=None):
    import numpy as np
    if readers is None:
        sys.path.insert(0, str(BASE / "code"))
        from anygsloc.data.colmap import read_intrinsics_binary, read_extrinsics_binary, qvec2rotmat
    else:
        read_intrinsics_binary, read_extrinsics_binary, qvec2rotmat = readers
    source = Path(source)
    intrinsics = read_intrinsics_binary(str(source / "sparse/0/cameras.bin"))
    extrinsics = read_extrinsics_binary(str(source / "sparse/0/images.bin"))
    by_name = {entry.name: entry for entry in extrinsics.values()}
    train = {line.split()[0] for line in (source / "dataset_train.txt").read_text().splitlines()[3:] if line.strip()}
    test = {line.split()[0] for line in (source / "dataset_test.txt").read_text().splitlines()[3:] if line.strip()}
    if len(train) != 1487 or len(test) != 530 or train & test or set(by_name) != train | test:
        raise ValueError("native camera split/source coverage changed")
    names = sorted(test) + sorted(train)  # native cameras.json is written before shuffle
    rows = read(path)
    if not isinstance(rows, list) or len(rows) != len(names):
        raise ValueError("native cameras.json count mismatch")
    for index, (row, name) in enumerate(zip(rows, names)):
        extrinsic = by_name[name]
        intrinsic = intrinsics[extrinsic.camera_id]
        require_values(row, {"id": index, "img_name": name,
                             "width": int(intrinsic.width), "height": int(intrinsic.height)},
                       "camera")
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = qvec2rotmat(extrinsic.qvec)
        world_to_camera[:3, 3] = extrinsic.tvec
        camera_to_world = np.linalg.inv(world_to_camera)
        for key, expected in (("position", camera_to_world[:3, 3]),
                              ("rotation", camera_to_world[:3, :3])):
            actual = np.asarray(row[key], dtype=np.float64)
            if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=1e-12, atol=1e-10):
                raise ValueError("camera pose differs from frozen source: " + name)
        if intrinsic.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            fx = fy = float(intrinsic.params[0])
        elif intrinsic.model in ("PINHOLE", "OPENCV"):
            fx, fy = map(float, intrinsic.params[:2])
        else:
            raise ValueError("unsupported original native camera model")
        for key, focal in (("fx", fx), ("fy", fy)):
            if not math.isfinite(row[key]) or not math.isclose(row[key], focal, rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError("camera intrinsics differ from frozen source: " + name)
    return {"camera_count": len(rows), "mapping_cameras": 1487, "query_cameras": 530,
            "camera_pose_and_intrinsics_verified": True}


def check_trace(rows, start, stop):
    if not isinstance(rows, list) or not rows:
        raise ValueError("missing bounded Gaussian trace")
    steps = [row["iteration"] for row in rows]
    if steps != sorted(set(steps)) or steps[0] != start + 1 or steps[-1] != stop:
        raise ValueError("Gaussian trace boundary/order mismatch")
    for row in rows:
        if not start < row["iteration"] <= stop or not row.get("camera"):
            raise ValueError("Gaussian trace identity mismatch")
        for key in ("loss", "l1", "feature_l1"):
            if not math.isfinite(float(row[key])):
                raise ValueError("nonfinite Gaussian trace")
        if any(not math.isfinite(float(lr)) or lr < 0 for lr in row["optimizer_lr"]):
            raise ValueError("invalid Gaussian trace learning rate")
    return len(rows)


def check_stage_chain(root, semantic, protocol, build_hash, build):
    root = Path(root)
    case = "STDLoc_StMarysChurch_mapping_semantics_" + semantic + "_train"
    spec = protocol["jobs"][case]
    model = Path(spec["fresh_model_path"])
    identity = {"protocol_sha256": sha(root / "training_protocol_v4.json"),
                "build_sha256": build_hash, "mapping_semantics": semantic,
                "model_path": str(model), "source_alias": protocol["source"], "probe_only": False}
    base = root / "std_execution/production" / semantic
    previous = None
    evidence = {}
    job_ids = []
    stages = []
    for stage, start, stop in STAGES:
        directory = base / stage
        config = read(directory / "config.json")
        require_values(config, {"stage": stage, "start": start, "stop": stop,
                                "identity": identity, "probe_only": False, "run_dir": str(directory),
                                "runner_timeout_seconds": 3400,
                                "stage_receipt": str(directory / "native_receipt.json"),
                                "final_receipt": spec["fresh_receipt_path"]}, "stage config")
        expected_command = list(spec["new_job"]["command"])
        expected_command[expected_command.index("--receipt") + 1] = str(directory / "native_receipt.json")
        if config["wrapper_command"] != expected_command:
            raise ValueError("stage changed native scientific command")
        expected_code = {**protocol["native_code_sha256"], **build["file_sha256"]}
        if (config["code_sha256"] != expected_code
                or config["generated_trainer"] != build["generated_trainer"]
                or config["native_trainer"] != str(Path(spec["new_job"]["cwd"]) / "train.py")
                or config["checkpoint_at"] != (-1 if stage == "detector" else stop)
                or config["checkpoint_out"] != str(directory / ("checkpoint_" + str(stop)))
                or config["trace_start"] != 30001):
            raise ValueError("stage executor/state output contract changed")
        if start == 0:
            if config["resume"] is not None:
                raise ValueError("first native stage unexpectedly resumed")
        elif config["resume"] != str(previous[0]):
            raise ValueError("native stage checkpoint chain changed")
        launch = read(directory / "launch.json")
        require_values(launch, {"stage": stage, "build_sha256": build_hash,
                                "runner_timeout_seconds": 3400}, "stage launch")
        entry = str(Path(__file__).resolve().with_name("std_entry.py"))
        if launch["command"] != [spec["new_job"]["command"][0], entry]:
            raise ValueError("stage launched a different executor")
        before, after = launch["nofile_before"], launch["nofile_after"]
        if after[1] != before[1] or (after[0] != -1 and after[0] < 4096):
            raise ValueError("native NOFILE guarantee changed")
        terminal = read(directory / "terminal.json")
        job_ids.append(check_terminal(terminal, case, semantic, stage,
                                      directory / "config.json", directory / "launch.json"))
        native = read(directory / "native_receipt.json")
        done = read(directory / "stage_complete.json")
        common = {"status": "complete", "method": "STDLoc", "phase": "train",
                  "mapping_semantics": semantic, "query_semantics": False, "seed": 2026}
        require_values(native, common, "native stage receipt")
        require_values(done, dict(common, exact_stage=stage, native_total_iterations=30000,
                                 gaussian_start=start, gaussian_stop=stop, probe_only=False,
                                 protocol_sha256=identity["protocol_sha256"], build_sha256=build_hash,
                                 scope="single_stage_excludes_other_stages"), "complete stage")
        for key, value in native.items():
            if done.get(key) != value:
                raise ValueError("stage_complete differs from native receipt: " + key)
        if not math.isfinite(done["wall_seconds"]) or done["wall_seconds"] <= 0:
            raise ValueError("invalid stage timing")
        if start:
            check_restore(read(directory / "restore_verified.json"), previous[1], previous[2], start)
        names = ["config.json", "launch.json", "terminal.json", "native_receipt.json",
                 "stage_complete.json", "native.log"]
        if start:
            names.append("restore_verified.json")
        if stage != "detector":
            require_values(read(directory / "gaussian_complete.json"),
                           {"complete": True, "start": start, "stop": stop,
                            "native_total_iterations": 30000, "all_observed_native_losses_finite": True,
                            "continuously_floating_point_equivalent_trajectory_claimed": False,
                            "native_terminal_skips_optimizer_step": stop == 30000, "probe_only": False},
                           "Gaussian completion")
            check_trace(read(directory / "trace.json"), start, stop)
            checkpoint_path = directory / ("checkpoint_" + str(stop))
            checkpoint, rng = check_checkpoint(checkpoint_path, stop, identity)
            previous = (checkpoint_path, checkpoint, rng)
            names.extend(["gaussian_complete.json", "trace.json",
                          "checkpoint_" + str(stop) + "/complete.json",
                          "checkpoint_" + str(stop) + "/roundtrip_verified.json",
                          "checkpoint_" + str(stop) + "/state.pt"])
        else:
            require_values(read(directory / "detector_entry.json"),
                           {"complete": True, "iteration": 30000,
                            "rng_digest": previous[2],
                            "natural_function": "train_detector.training_detector",
                            "train_iteration": 30000, "landmark_num": 16384, "landmark_k": 32},
                           "natural detector handoff")
            require_values(read(directory / "detector_complete.json"),
                           {"complete": True, "stage": "detector", "train_iteration": 30000,
                            "terminal_detector_tensors_finite": True,
                            "terminal_sha256": sha(model / "detector/30000_detector.pth")},
                           "native detector completion")
            if [x.get("stage") for x in done["build_stages"]] != ["training_detector"]:
                raise ValueError("natural detector execution missing")
            names.extend(["detector_entry.json", "detector_complete.json"])
        log = (directory / "native.log").read_text(errors="replace")
        progress = "30000/30000" if stage == "detector" else str(stop - start) + "/" + str(30000 - start)
        if progress not in log or "Training complete." not in log:
            raise ValueError("missing native terminal progress/log")
        evidence[stage] = {name: ref(directory / name) for name in names}
        stages.append(done)
    if len(set(job_ids)) != 4:
        raise ValueError("Runner terminal job reused across stages")
    return stages, evidence


def check_acceptance_review(path, build_hash):
    value = read(path)
    require_values(value, {"complete": True, "verdict": "PASS", "scope": "std_final_acceptance",
                           "method": "STDLoc", "training_build_sha256": build_hash,
                           "training_protocol_sha256": PROTOCOL_SHA}, "acceptance code review")
    expected = [Path(__file__).resolve(), Path(__file__).resolve().parents[2] / "tests/test_native_baseline_std_acceptance.py"]
    if value.get("code_sha256") != {str(p): sha(p) for p in expected} or not value.get("reviewer"):
        raise ValueError("acceptance code review source mismatch")
    return ref(path)


def check_frozen(root, semantic):
    root = Path(root)
    protocol_path = root / "training_protocol_v4.json"
    build_path = root / "std_execution/build.json"
    if sha(protocol_path) != PROTOCOL_SHA or sha(build_path) != BUILD_SHA:
        raise ValueError("unapproved training protocol/build")
    protocol, build = read(protocol_path), read(build_path)
    if build.get("protocol_sha256") != PROTOCOL_SHA:
        raise ValueError("training build detached from protocol")
    review_path = root / "std_execution/independent_review.json"
    review = read(review_path)
    require_values(review, {"complete": True, "verdict": "PASS", "protocol_sha256": PROTOCOL_SHA,
                            "build_sha256": BUILD_SHA, "approved_method": "STDLoc",
                            "approved_scene": "StMarysChurch",
                            "approved_training_contract": "state_exact_native_30000_plus_detector"},
                   "training review")
    if semantic not in review["approved_mapping_semantics"] or any(s not in review["approved_stages"] for s, _, _ in STAGES):
        raise ValueError("training review scope mismatch")
    for path, expected in {**protocol["native_code_sha256"], **build["file_sha256"]}.items():
        if sha(path) != expected:
            raise ValueError("frozen code drift: " + path)
    for reference in list(protocol["inputs"].values()) + protocol["evidence"] + review["evidence"]:
        verify(reference)
    if not os.path.samefile(protocol["source"], protocol["canonical_source"]):
        raise ValueError("Cambridge source alias changed")
    return protocol, build, {"training_build": ref(build_path), "training_review": ref(review_path)}


def atomic_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".acceptance-", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def accept(root, semantic):
    import torch
    if torch.cuda.is_initialized():
        raise ValueError("acceptance must run CPU-only in a fresh process")
    root = Path(root).resolve()
    case = "STDLoc_StMarysChurch_mapping_semantics_" + semantic + "_train"
    destination = root / "acceptance" / (case + ".json")
    if destination.exists():
        raise FileExistsError(destination)
    protocol, build, frozen_evidence = check_frozen(root, semantic)
    code_review = check_acceptance_review(root / "std_execution/acceptance_review.json", BUILD_SHA)
    spec = protocol["jobs"][case]
    model = Path(spec["fresh_model_path"])
    stages, stage_evidence = check_stage_chain(root, semantic, protocol, BUILD_SHA, build)
    final_path = Path(spec["fresh_receipt_path"])
    final = read(final_path)
    require_values(final, {"status": "complete", "method": "STDLoc", "phase": "train",
                           "mapping_semantics": semantic, "query_semantics": False, "seed": 2026,
                           "scope": "all_four_native_stages", "exact_stage": "all_complete",
                           "native_total_iterations": 30000, "native_optimizer_steps": 29999,
                           "native_terminal_skips_optimizer_step": True,
                           "detector_train_iterations": 30000, "landmark_num": 16384, "landmark_k": 32,
                           "native_output": str(model), "protocol_sha256": PROTOCOL_SHA,
                           "build_sha256": BUILD_SHA, "probe_only": False}, "final native receipt")
    if final["stage_receipts"] != [stage_evidence[s]["stage_complete.json"] for s, _, _ in STAGES]:
        raise ValueError("final receipt detached from verified stages")
    if not math.isclose(final["wall_seconds"], sum(s["wall_seconds"] for s in stages), rel_tol=1e-12):
        raise ValueError("final stage wall time sum mismatch")
    artifacts = {name: ref(model / relative) for name, relative in {
        "cfg_args": "cfg_args", "cameras": "cameras.json", "ply": "point_cloud/iteration_30000/point_cloud.ply",
        "indices": "detector/sampled_idx.pkl", "detector": "detector/30000_detector.pth"}.items()}
    if final["artifacts"] != [artifacts[name] for name in ("ply", "indices", "detector")]:
        raise ValueError("final model differs from native terminal artifact hashes")
    check_cfg(artifacts["cfg_args"]["path"], model, protocol["source"])
    cameras = check_cameras(artifacts["cameras"]["path"], protocol["source"])
    ply = check_ply(artifacts["ply"]["path"])
    indices = check_indices(load_indices(artifacts["indices"]["path"]), ply["point_count"])
    detector = check_detector(artifacts["detector"]["path"])
    # Recheck final products after parsing, before publishing acceptance.
    for item in artifacts.values():
        verify(item)
    if torch.cuda.is_initialized():
        raise ValueError("unexpected CUDA initialization during acceptance")
    result = {"schema": "native_std_training_acceptance_v1", "complete": True, "case": case,
              "method": "STDLoc", "semantics": semantic, "seed": 2026, "iterations": 30000,
              "native_optimizer_steps": 29999, "native_terminal_skip_step_preserved": True,
              "detector_iterations": 30000, "landmark_k": 32, **cameras, **ply, **indices, **detector,
              "state_continuation_verified": True, "floating_point_trajectory_equivalence_claimed": False,
              "initialization": "original native SfM, 419837 initial points including mixed and test-supported tracks; not strict mapping-only upstream",
              "training_protocol": ref(root / "training_protocol_v4.json"), **frozen_evidence,
              "acceptance_code_review": code_review, "artifacts": artifacts,
              "native_final_receipt": ref(final_path), "stages": stage_evidence,
              "wall_seconds": final["wall_seconds"], "validation_device": "cpu", "gpu_validation": False,
              "checkpoint_validation": "Hash-bound original blob, immutable full-state manifest, actual save/load roundtrip proof and subsequent restore proof verified; checkpoint pickle code is never executed by this acceptance.",
              "camera_numeric_check": "Source pose/intrinsics serialization consistency only; not a training trajectory tolerance."}
    atomic_new(destination, result)
    return {"path": str(destination), "sha256": sha(destination), "case": case,
            "point_count": ply["point_count"], "landmarks": indices["landmarks"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--semantics", choices=("off", "on"), required=True)
    args = parser.parse_args()
    print(json.dumps(accept(args.root, args.semantics), indent=2))
