"""Exact-boundary continuation for the frozen MAtCha recipe.

State-exact checkpoint transport only. Native floating trajectory equality is NOT claimed.
No tensor arithmetic, RNG draws, scheduler changes, or optimizer steps are added.
Only trusted locally produced checkpoint directories may be loaded.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
import resource
import math

import numpy as np
import torch
import diagnostic_cpu
import project_a_reference
from diagnostic_cpu import diagnostic_only, drain_records

SCHEMA = "matcha_exact_resume_v2"
PHASE = "after_optimizer_and_zero_grad"
PARAMS = {"xyz": "_xyz", "f_dc": "_features_dc", "f_rest": "_features_rest",
          "opacity": "_opacity", "scaling": "_scaling", "rotation": "_rotation"}
MODEL_FIELDS = ("active_sh_degree", "max_sh_degree", "max_radii2D",
                "xyz_gradient_accum", "denom", "percent_dense", "spatial_lr_scale",
                "use_mip_filter", "mip_filter", "non_position_base_lrs",
                "non_position_lr_decay_from", "non_position_lr_decay_until",
                "non_position_lr_final_mult")
LOOP_FIELDS = ("viewpoint_idx_stack", "dense_viewpoint_idx_stack",
               "chart_sampling_phase", "use_chart_view_every_n_iter",
               "ema_loss_for_log", "ema_dist_for_log", "ema_normal_for_log",
               "ema_prior_depth_for_log", "ema_prior_normal_for_log",
               "ema_prior_curvature_for_log", "ema_prior_anisotropy_for_log")
STATIC_FIELDS = ("background", "charts_scale_factor", "charts_prior_depths",
                 "charts_depths", "charts_confs", "charts_normals", "charts_curvs",
                 "chart_prior_weights", "chart_late_prior_weights",
                 "chart_regularization_indices", "chart_late_regularization_indices",
                 "dense_depth_priors", "use_dense_supervision",
                 "use_depth_order_regularization")
CAMERA_FIELDS = ("uid", "colmap_id", "R", "T", "FoVx", "FoVy", "image_name",
                 "data_device", "original_image", "image_width", "image_height",
                 "gt_alpha_mask", "zfar", "znear", "trans", "scale",
                 "world_view_transform", "projection_matrix", "full_proj_transform",
                 "camera_center", "focal_x", "focal_y")
CONTROL_ARGS = {"exact_manifest", "exact_dir", "exact_resume", "exact_stop",
                "exact_save", "exact_windows", "exact_reference_trace", "exact_continue_gate",
                "exact_reference_terminal", "exact_prefix_spec", "model_path", "quiet", "port"}
# Output path/quiet/GUI-port are recorded separately; all science args compare.
REQUIRED_TOP = {"schema", "phase", "completed_iteration", "next_iteration",
                "binding", "static_ref", "model", "optimizer", "loop", "rng",
                "events", "pipeline", "scheduler_next", "static_digest"}

COMPARISON_FIELDS = ("model", "optimizer", "loop", "rng", "events", "pipeline",
                     "scheduler_next", "static_digest")
@diagnostic_only
def snapshot_static(local):
    static = {k: pack(local[k]) for k in STATIC_FIELDS}
    static["chart_cameras"] = [{k: pack(getattr(c, k)) for k in CAMERA_FIELDS}
                              for c in local["viewpoint_cams"]]
    static["dense_cameras"] = [{k: pack(getattr(c, k)) for k in CAMERA_FIELDS}
                              for c in local["dense_viewpoint_cams"]]
    static["scene_extent"] = local["scene"].cameras_extent
    static["rgb_mask_lookup"] = pack(vars(local["rgb_mask_lookup"])) if local["rgb_mask_lookup"] else None
    static["geometry_mask_lookup"] = pack(vars(local["geometry_mask_lookup"])) if local["geometry_mask_lookup"] else None
    return static

def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v) for v in value)
    return 0

def scalar_record(value):
    if isinstance(value, torch.Tensor):
        require(value.numel() == 1, "non-scalar loss component")
        data = value.detach().cpu().contiguous().reshape(-1)
        require(bool(torch.isfinite(data).all()), "nonfinite loss component")
        return {"dtype": str(data.dtype), "bits": data.view(torch.uint8).numpy().tobytes().hex()}
    require(isinstance(value, (int, float)), "loss component missing/non-numeric")
    require(math.isfinite(float(value)), "nonfinite Python loss component")
    return {"dtype": type(value).__name__, "hex": float(value).hex()}

def loss_components(local, selected_pool):
    common = ("Ll1", "loss", "normal_loss", "dist_loss", "depth_prior_loss",
              "normal_prior_loss", "curv_prior_loss", "total_regularization_loss", "total_loss")
    require(all(k in local for k in common), "missing required loss component")
    result = {k: scalar_record(local[k]) for k in common}
    require("rgb_losses" in local and "dssim" in local["rgb_losses"], "missing native DSSIM component")
    result["dssim"] = scalar_record(local["rgb_losses"]["dssim"]) if local["rgb_losses"]["dssim"] is not None else None
    result["lambda_dssim"] = scalar_record(local["opt"].lambda_dssim)
    result["depth_order_prior_loss"] = (
        scalar_record(local["depth_order_prior_loss"])
        if selected_pool == "chart" and local["use_depth_order_regularization"] else None)
    result["anisotropy_loss"] = scalar_record(local["anisotropy_loss"]) if local["lambda_anisotropy"] > 0 else None
    require(local["large_splat_radius_reg"] == 0., "unsupported added loss")
    result["large_splat_loss"] = None
    coefficients = ["lambda_normal", "lambda_dist", "lambda_anisotropy"]
    if selected_pool == "chart":
        coefficients += ["lambda_prior_depth", "lambda_prior_depth_derivative",
                         "lambda_prior_normal", "lambda_prior_curvature", "regularization_factor"]
        if local["use_depth_order_regularization"]:
            coefficients.append("lambda_depth_order")
    else:
        coefficients.append("lambda_dense_depth")
    result["coefficients"] = {k: scalar_record(local[k]) for k in coefficients}
    return result

def require(ok, message):
    if not ok:
        raise RuntimeError("exact-resume: " + message)

def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def pack(value):
    """CPU snapshot with exact original tensor device, dtype, stride and values."""
    if isinstance(value, torch.Tensor):
        return {"@": "tensor", "device": str(value.device),
                "dtype": str(value.dtype), "shape": tuple(value.shape),
                "stride": tuple(value.stride()), "requires_grad": value.requires_grad,
                "data": value.detach().cpu().clone(memory_format=torch.preserve_format)}
    if isinstance(value, np.ndarray):
        return {"@": "numpy", "data": value.copy()}
    if isinstance(value, torch.device):
        return {"@": "device", "value": str(value)}
    if isinstance(value, Path):
        return {"@": "path", "value": str(value)}
    if isinstance(value, dict):
        return {k: pack(v) for k, v in value.items()}
    if isinstance(value, list):
        return [pack(v) for v in value]
    if isinstance(value, tuple):
        return tuple(pack(v) for v in value)
    require(value is None or isinstance(value, (str, bool, int, float, np.generic)),
            "unsupported state type " + str(type(value)))
    return copy.deepcopy(value)

def unpack(value):
    if isinstance(value, dict):
        tag = value.get("@")
        if tag == "tensor":
            t = value["data"]
            require(str(t.dtype) == value["dtype"] and tuple(t.shape) == value["shape"],
                    "tensor descriptor mismatch")
            out = torch.empty_strided(value["shape"], value["stride"], dtype=t.dtype, device=value["device"])
            out.copy_(t.to(device=value["device"]))
            require(tuple(out.stride()) == value["stride"], "tensor stride mismatch")
            return out.requires_grad_(value["requires_grad"])
        if tag == "numpy":
            return value["data"].copy()
        if tag == "device":
            return torch.device(value["value"])
        if tag == "path":
            return Path(value["value"])
        return {k: unpack(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unpack(v) for v in value]
    if isinstance(value, tuple):
        return tuple(unpack(v) for v in value)
    return copy.deepcopy(value)

def equal(a, b):
    if isinstance(a, torch.Tensor):
        return (isinstance(b, torch.Tensor) and a.dtype == b.dtype and
                a.shape == b.shape and
                a.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes() ==
                b.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    if isinstance(a, np.ndarray):
        return (isinstance(b, np.ndarray) and a.dtype == b.dtype and
                a.shape == b.shape and a.tobytes() == b.tobytes())
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b

def semantic_digest(value):
    """Exact-value digest for bounded comparison windows; never used every step."""
    h = hashlib.sha256()
    def visit(v):
        h.update(type(v).__name__.encode() + b":")
        if isinstance(v, torch.Tensor):
            h.update(str(v.dtype).encode() + str(tuple(v.shape)).encode())
            a = v.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
            h.update(memoryview(a).cast("B"))
        elif isinstance(v, np.ndarray):
            h.update(v.dtype.str.encode() + str(v.shape).encode())
            h.update(memoryview(np.ascontiguousarray(v)).cast("B"))
        elif isinstance(v, dict):
            for k in sorted(v, key=repr):
                visit(k)
                visit(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(str(len(v)).encode())
            for x in v:
                visit(x)
        else:
            h.update(repr(v).encode())
        h.update(b";")
    visit(value)
    return h.hexdigest()

@diagnostic_only
def diagnostic_digest(value):
    return semantic_digest(value)

@diagnostic_only
def diagnostic_equal(a, b):
    return equal(a, b)

def next_lrs(model, iteration):
    result = {g["name"]: g["lr"] for g in model.optimizer.param_groups}
    if model.non_position_lr_decay_from >= 0:
        span = max(model.non_position_lr_decay_until - model.non_position_lr_decay_from, 1)
        progress = min(max((iteration - model.non_position_lr_decay_from) / span, 0.0), 1.0)
        mult = max(model.non_position_lr_final_mult, 1e-8) ** progress
        for name, base in model.non_position_base_lrs.items():
            result[name] = base * mult
    result["xyz"] = model.xyz_scheduler_args(iteration)
    return result

def synchronize():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)

def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "cuda_all": [s.clone() for s in torch.cuda.get_rng_state_all()]
                        if torch.cuda.is_available() else []}

def restore_rng(state):
    require(set(state) == {"python", "numpy", "torch_cpu", "cuda_all"}, "RNG schema")
    require(len(state["cuda_all"]) == torch.cuda.device_count(), "visible CUDA count")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["cuda_all"]:
        torch.cuda.set_rng_state_all(state["cuda_all"])

@diagnostic_only
def atomic_save(path, payload):
    """Atomic directory publication; state and JSON checksums verified before rename."""
    path = Path(path)
    require(not path.exists(), "refusing checkpoint overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="." + path.name + ".", dir=str(path.parent)))
    try:
        state_path = tmp / "state.pt"
        with state_path.open("wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        digest = file_sha(state_path)
        loaded = torch.load(state_path, map_location="cpu", weights_only=False)
        require(equal(payload, loaded), "checkpoint readback differs")
        del loaded
        info = {"schema": SCHEMA, "sha256": digest, "bytes": state_path.stat().st_size}
        with (tmp / "manifest.json").open("w") as f:
            json.dump(info, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        require(file_sha(state_path) == digest, "checkpoint changed on readback")
        fd = os.open(str(tmp), os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
        os.rename(str(tmp), str(path))
        fd = os.open(str(path.parent), os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
        return {"path": str(path.resolve()), **info}
    except Exception:
        # Unpublished partial directories are deliberately retained for diagnosis.
        raise

@diagnostic_only
def read_atomic(path, expected_ref=None):
    path = Path(path)
    require(path.is_dir() and not path.is_symlink(), "checkpoint directory missing")
    info = json.loads((path / "manifest.json").read_text())
    require(set(info) == {"schema", "sha256", "bytes"} and info["schema"] == SCHEMA,
            "file manifest schema")
    if expected_ref is not None:
        require(expected_ref == {"path": str(path.resolve()), **info}, "static reference changed")
    p = path / "state.pt"
    require(p.stat().st_size == info["bytes"] and file_sha(p) == info["sha256"],
            "truncated or modified state")
    return torch.load(p, map_location="cpu", weights_only=False)

def bindings(model):
    result = []
    for g in model.optimizer.param_groups:
        name = g.get("name")
        require(name in PARAMS and len(g["params"]) == 1, "unknown optimizer group")
        p = g["params"][0]
        require(p is getattr(model, PARAMS[name]), "optimizer parameter binding")
        require(p.grad is None, "boundary has gradient")
        result.append((name, PARAMS[name]))
    require(result == list(PARAMS.items()), "optimizer group order")
    return result

@diagnostic_only
def capture_model(model):
    binding = bindings(model)
    require(model.use_mip_filter and hasattr(model, "mip_filter"), "missing actual mip filter")
    return {"parameters": {name: pack(getattr(model, attr)) for name, attr in PARAMS.items()},
            "fields": {k: pack(getattr(model, k)) for k in MODEL_FIELDS},
            "bindings": binding}, pack(model.optimizer.state_dict())

def validate_model(state, optimizer):
    require(set(state) == {"parameters", "fields", "bindings"}, "model schema")
    require(state["bindings"] == list(PARAMS.items()), "saved optimizer bindings")
    require(set(state["parameters"]) == set(PARAMS), "missing parameter")
    require(set(state["fields"]) == set(MODEL_FIELDS), "missing model/mip/scheduler field")
    fields = state["fields"]
    require(fields["use_mip_filter"] is True, "mip disabled")
    n = state["parameters"]["xyz"]["shape"][0]
    require(n > 0, "empty Gaussian model")
    for name, p in state["parameters"].items():
        require(p.get("@") == "tensor" and p["shape"][0] == n and p["requires_grad"],
                "parameter descriptor")
    for name in ("max_radii2D", "xyz_gradient_accum", "denom", "mip_filter"):
        require(fields[name].get("@") == "tensor" and fields[name]["shape"][0] == n,
                "accumulator/mip topology")
    require(0 <= fields["active_sh_degree"] <= fields["max_sh_degree"] == 3, "SH degree")
    require(set(optimizer) == {"state", "param_groups"}, "optimizer schema")
    groups = optimizer["param_groups"]
    require([g.get("name") for g in groups] == list(PARAMS), "saved optimizer order")
    ids = [g["params"][0] for g in groups if len(g["params"]) == 1]
    require(len(ids) == 6 and len(set(ids)) == 6, "saved optimizer IDs")
    require(set(optimizer["state"]) == set(ids), "missing or unbound Adam state")
    for group, idx in zip(groups, ids):
        state_for_parameter = optimizer["state"][idx]
        required = {"step", "exp_avg", "exp_avg_sq"}
        if group.get("amsgrad"):
            required.add("max_exp_avg_sq")
        require(set(state_for_parameter) == required, "missing Adam moment or step")
        parameter = state["parameters"][group["name"]]
        for key in required - {"step"}:
            moment = state_for_parameter[key]
            require(moment.get("@") == "tensor" and moment["shape"] == parameter["shape"] and
                    moment["dtype"] == parameter["dtype"] and moment["device"] == parameter["device"],
                    "Adam moment descriptor/binding")
        step = state_for_parameter["step"]
        require(isinstance(step, dict) and step.get("@") == "tensor" and
                step["shape"] == (), "Adam step descriptor")

def restore_model(model, state, optimizer, opt):
    validate_model(state, optimizer)
    for name, attr in PARAMS.items():
        setattr(model, attr, torch.nn.Parameter(unpack(state["parameters"][name])))
    model.spatial_lr_scale = state["fields"]["spatial_lr_scale"]
    model.training_setup(opt)  # Rebuild original scheduler from unchanged 30000-step opts.
    for name in ("non_position_base_lrs", "non_position_lr_decay_from",
                 "non_position_lr_decay_until", "non_position_lr_final_mult"):
        require(equal(pack(getattr(model, name)), state["fields"][name]),
                "rebuilt scheduler differs: " + name)
    for k, v in state["fields"].items():
        setattr(model, k, unpack(v))
    model.optimizer.load_state_dict(unpack(optimizer))
    require(bindings(model) == state["bindings"], "restored optimizer binding")
    actual, actual_optimizer = capture_model(model)
    require(diagnostic_equal(actual, state) and diagnostic_equal(actual_optimizer, optimizer), "model/Adam restore differs")

def runtime_binding():
    cuda = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            cuda.append({"name": p.name, "major": p.major, "minor": p.minor,
                         "total_memory": p.total_memory, "uuid": str(getattr(p, "uuid", None))})
    return {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpus": cuda, "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "threads": torch.get_num_threads(), "interop": torch.get_num_interop_threads(),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "env": {k: os.environ.get(k) for k in
                    ("CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "PYTORCH_CUDA_ALLOC_CONF", "PYTHONHASHSEED", "LD_LIBRARY_PATH")}}

def validate_loop(loop, static):
    require(set(loop) == set(LOOP_FIELDS), "loop schema")
    for name, count in (("viewpoint_idx_stack", len(static["chart_cameras"])),
                        ("dense_viewpoint_idx_stack", len(static["dense_cameras"]))):
        value = loop[name]
        require(value is None or (type(value) is list and len(value) == len(set(value)) and
                all(type(i) is int and 0 <= i < count for i in value)), "sampling stack")
    require(loop["chart_sampling_phase"] in (None, "early", "late"), "chart phase")
    require(type(loop["use_chart_view_every_n_iter"]) is int and
            loop["use_chart_view_every_n_iter"] > 0, "chart cadence")

def validate_checkpoint(state, binding, static):
    require(set(state) == REQUIRED_TOP and state["schema"] == SCHEMA, "checkpoint schema/version")
    require(state["phase"] == PHASE, "unsafe boundary")
    k = state["completed_iteration"]
    require(type(k) is int and 0 < k < 30000 and state["next_iteration"] == k + 1,
            "invalid continuation iteration or terminal state")
    require(state["binding"] == binding, "source/input/config/runtime binding differs")
    require(set(state["rng"]) == {"python", "numpy", "torch_cpu", "cuda_all"}, "RNG schema")
    require(len(state["rng"]["cuda_all"]) == torch.cuda.device_count(), "CUDA RNG count")
    validate_model(state["model"], state["optimizer"])
    validate_loop(state["loop"], static)
    require(set(state["events"]) == {"evaluation", "save", "native_checkpoint"}, "event schema")
    for values in state["events"].values():
        require(values == sorted(set(values)) and all(0 < i <= k for i in values), "event state")

def add_arguments(parser):
    parser.add_argument("--exact_manifest", required=True)
    parser.add_argument("--exact_dir", required=True)
    parser.add_argument("--exact_resume")
    parser.add_argument("--exact_stop", type=int)
    parser.add_argument("--exact_save", type=int, nargs="*", default=[])
    parser.add_argument("--exact_windows", type=str, nargs="*", default=[])
    parser.add_argument("--exact_reference_trace")
    parser.add_argument("--exact_continue_gate")
    parser.add_argument("--exact_reference_terminal")
    parser.add_argument("--exact_prefix_spec")


@diagnostic_only
def finite_state(value):
    """Finite screen only; no normalization, mutation or floating tolerance."""
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            require(bool(torch.isfinite(value).all()), "nonfinite saved tensor")
    elif isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.inexact):
            require(bool(np.isfinite(value).all()), "nonfinite saved array")
    elif isinstance(value, dict):
        for item in value.values(): finite_state(item)
    elif isinstance(value, (list, tuple)):
        for item in value: finite_state(item)
    elif isinstance(value, (float, np.floating)):
        require(math.isfinite(float(value)), "nonfinite saved scalar")

@diagnostic_only
def finite_terminal_model(model):
    # Original step30000 leaves gradients; do not call capture_model/zero_grad/step here.
    finite_state([getattr(model, name) for name in PARAMS.values()])
    finite_state(model.optimizer.state_dict())
    finite_state([getattr(model, name) for name in MODEL_FIELDS])
    finite_state([p.grad for g in model.optimizer.param_groups for p in g["params"]])

class Controller:
    def __init__(self, args, local):
        self.started = time.monotonic()
        self.args = args
        self.gate = None
        self.gate_contract = None
        self.prefix_gate = None
        require(args.iterations == 30000, "optimization horizon must remain 30000")
        for key in ("use_color_correction", "use_appearance_embedding", "use_transient_model"):
            require(not getattr(args, key), "unsupported auxiliary model " + key)
        require(args.start_checkpoint is None and args.early_stop_iteration is None,
                "native warm resume or pre-optimizer early stop unsupported")
        require(args.use_mip_filter and args.dense_data_path and args.dense_regul == "default",
                "not the frozen dense+mip recipe")
        require(args.exact_stop is None or 0 < args.exact_stop <= 30000, "invalid stop")
        require(all(0 < k < 30000 for k in args.exact_save), "invalid save boundary")
        self.root = Path(args.exact_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = json.loads(Path(args.exact_manifest).read_text())
        require(set(manifest) == {"schema", "files", "input_receipt_sha256", "science_args"} and
                manifest["schema"] == "matcha_exact_binding_v1", "binding manifest schema")
        require(manifest["files"] and len(manifest["input_receipt_sha256"]) == 64,
                "missing independently reviewed input receipt")
        for path, digest in manifest["files"].items():
            require(file_sha(path) == digest, "bound file changed: " + path)
        require(str(Path(__file__).resolve()) in manifest["files"], "unbound continuation module")
        require(str(Path(diagnostic_cpu.__file__).resolve()) in manifest["files"], "unbound diagnostic helper")
        require(str(Path(project_a_reference.__file__).resolve()) in manifest["files"], "unbound A projection helper")
        require(str(Path(sys.argv[0]).resolve()) in manifest["files"], "unbound instrumented trainer")
        require(manifest["science_args"] == {k: v for k, v in vars(args).items() if k not in CONTROL_ARGS},
                "parsed science arguments differ from independently frozen manifest")
        self.binding = {"manifest": manifest,
                        "science_args": {k: v for k, v in vars(args).items() if k not in CONTROL_ARGS},
                        "runtime": runtime_binding()}
        self.has_resume = args.exact_resume is not None
        self.state = read_atomic(args.exact_resume) if self.has_resume else None
        self.static = read_atomic(self.state["static_ref"]["path"], self.state["static_ref"]) if self.has_resume else None
        self.events = {"evaluation": [], "save": [], "native_checkpoint": []}
        require(not args.exact_windows and not args.exact_reference_trace and
                not args.exact_continue_gate and not args.exact_reference_terminal and
                not args.exact_prefix_spec, "v3 has no trajectory comparison or promotion gate")
        if self.has_resume:
            require(self.state["completed_iteration"] == 14999 and self.state["next_iteration"] == 15000 and
                    args.exact_stop == 30000 and args.exact_save == [], "only14999->30000 resume")
        else:
            require(args.exact_stop == 14999 and args.exact_save == [14999], "only fresh1->14999")
        self.windows, self.reference = [], {}
        self.next_expected = 15000 if self.has_resume else 1
        self.restore_verified = not self.has_resume
        self.in_step = False
        self.trace = (self.root / ("trace_from_%d.jsonl" %
                      (self.state["next_iteration"] if self.has_resume else 1))).open("x")
        self.performance = (self.root / "performance.jsonl").open("x")
        if self.has_resume:
            validate_checkpoint(self.state, self.binding, self.static)

    def check_reference(self, record, key):
        require(not self.reference, "trajectory reference forbidden in v3")

    def metric(self, value):
        value["diagnostic_phases"] = drain_records()
        value.update(wall_seconds=time.monotonic() - self.started,
                     peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                     peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(0) if torch.cuda.is_available() else 0)
        self.performance.write(json.dumps(value, sort_keys=True) + "\n")
        self.performance.flush()

    def prepare(self, local):
        require(local["color_correction"] is None and local["color_correction_optimizer"] is None and
                local["transient_model"] is None and local["transient_optimizer"] is None,
                "unexpected auxiliary training state")
        static = snapshot_static(local)
        if self.has_resume:
            # Camera reconstruction is verified bitwise; caches below restore saved values.
            for key in ("chart_cameras", "dense_cameras", "scene_extent",
                        "rgb_mask_lookup", "geometry_mask_lookup"):
                require(diagnostic_equal(static[key], self.static[key]), "rebuilt static input differs: " + key)
            self.static_ref = self.state["static_ref"]
            self.static_digest = diagnostic_digest(self.static)
            require(self.static_digest == self.state["static_digest"], "static checkpoint digest differs")
            self.restored_expected_digest = diagnostic_digest({k: self.state[k] for k in COMPARISON_FIELDS})
            restore_model(local["gaussians"], self.state["model"], self.state["optimizer"], local["opt"])
            require(vars(local["pipe"]) == self.state["pipeline"], "pipeline/debug changed")
            next_lr = next_lrs(local["gaussians"], self.state["next_iteration"])
            require(next_lr == self.state["scheduler_next"], "next-step LRs differ")
            self.events = copy.deepcopy(self.state["events"])
            restore = {k: unpack(self.static[k]) for k in STATIC_FIELDS}
            restore.update(copy.deepcopy(self.state["loop"]))
            restore["first_iter"] = self.state["completed_iteration"]
            self.metric({"event": "restore_setup", "checkpoint_iteration": self.state["completed_iteration"],
                         "static_tensor_bytes": tensor_bytes(self.static),
                         "dynamic_tensor_bytes": tensor_bytes(self.state["model"]) + tensor_bytes(self.state["optimizer"])})
            # RNG restoration is explicitly separate and LAST in the trainer.
            return restore
        before = rng_state()
        self.static = static
        self.static_digest = diagnostic_digest(static)
        static_started = time.monotonic()
        self.static_ref = atomic_save(self.root / "static", static)
        self.metric({"event": "fresh_setup", "static_tensor_bytes": tensor_bytes(static),
                     "static_file_bytes": self.static_ref["bytes"],
                     "static_publication_seconds": time.monotonic() - static_started})
        require(equal(before, rng_state()), "static capture consumed RNG")
        return None

    def restore_rng_last(self):
        if self.has_resume:
            restore_rng(self.state["rng"])
            self.state = None
        # Keep only immutable pool sizes after reconstruction; do not retain multi-GB CPU snapshots.
        self.static = {key: [None] * len(self.static[key]) for key in ("chart_cameras", "dense_cameras")}


    def verify_restored_boundary(self, local):
        if not self.has_resume:
            return
        before = rng_state()
        model, optimizer = capture_model(local["gaussians"])
        cache_digest = diagnostic_digest(snapshot_static(local))
        require(cache_digest == self.static_digest, "actual restored cache differs")
        comparison = {"model": model, "optimizer": optimizer,
            "loop": {k: copy.deepcopy(local[k]) for k in LOOP_FIELDS}, "rng": before,
            "events": copy.deepcopy(self.events), "pipeline": copy.deepcopy(vars(local["pipe"])),
            "scheduler_next": next_lrs(local["gaussians"], local["first_iter"]),
            "static_digest": cache_digest}
        digest = diagnostic_digest(comparison)
        require(digest == self.restored_expected_digest, "restored boundary full state differs")
        if self.gate is not None:
            self.check_reference({"iteration": 14999, "exact_digest": digest}, (14999, True))
            self.gate.restored_boundary(digest)
        require(equal(before, rng_state()), "restored verification consumed RNG")
        self.restore_verified = True
        proof = {"schema": "matcha_state_restore_proof_v3", "verdict": "PASS",
                 "completed_iteration": 14999, "next_iteration": 15000,
                 "saved_full_digest": self.restored_expected_digest, "restored_full_digest": digest,
                 "static_digest": cache_digest, "all_state_bitwise": True,
                 "floating_trajectory_equivalence_claimed": False}
        with (self.root / "restore_bitwise_PASS.json").open("x") as f:
            json.dump(proof, f, sort_keys=True); f.flush(); os.fsync(f.fileno())

    def before_step(self, iteration):
        require(self.restore_verified and not self.in_step and iteration == self.next_expected,
                "unverified restore, skipped or repeated step")
        require(torch.get_num_threads() == self.binding["runtime"]["threads"] and
                torch.get_num_interop_threads() == self.binding["runtime"]["interop"],
                "diagnostic thread settings leaked into training")
        self.in_step = True

    def observe_view(self, local):
        self.selection = {"pool": "chart" if local["iteration"] %
                           local["use_chart_view_every_n_iter"] == 0 else "dense",
                          "index": int(local["viewpoint_idx"]),
                          "name": local["viewpoint_cam"].image_name,
                          "cadence_at_selection": local["use_chart_view_every_n_iter"]}

    def boundary(self, local):
        k = local["iteration"]
        require(self.in_step and k == self.next_expected, "boundary order")
        for name, key in (("evaluation", "testing_iterations"), ("save", "saving_iterations"),
                          ("native_checkpoint", "checkpoint_iterations")):
            if k in local[key]:
                self.events[name].append(k)
        record = {"iteration": k, **self.selection,
                  "loss_components": loss_components(local, self.selection["pool"]),
                  "stacks": {key: copy.deepcopy(local[key]) for key in
                             ("viewpoint_idx_stack", "dense_viewpoint_idx_stack")},
                  "rng_digest": semantic_digest(rng_state()), "events": copy.deepcopy(self.events),
                  "phase": local["chart_sampling_phase"],
                  "cadence_after_step": local["use_chart_view_every_n_iter"],
                  "lr": {g["name"]: g["lr"] for g in local["gaussians"].optimizer.param_groups},
                  "loss": float(local["total_loss"].item()),
                  "points": len(local["gaussians"]._xyz),
                  "active_sh_degree": local["gaussians"].active_sh_degree}
        self.check_reference(record, (k, False))
        self.trace.write(json.dumps(record, sort_keys=True) + "\n")
        self.trace.flush()
        if k % 100 == 0:
            self.metric({"event": "progress", "iteration": k, "points": len(local["gaussians"]._xyz)})
        full = (k in self.args.exact_save or k == self.args.exact_stop or
                (k, True) in self.reference or any(a <= k <= b for a, b in self.windows))
        digest_record = None
        if full and k < 30000:
            capture_started = time.monotonic()
            synchronize()
            before = rng_state()
            model, optimizer = capture_model(local["gaussians"])
            cache_digest = diagnostic_digest(snapshot_static(local))
            require(cache_digest == self.static_digest, "actual static cache changed")
            state = {"schema": SCHEMA, "phase": PHASE, "completed_iteration": k,
                     "next_iteration": k + 1, "binding": self.binding,
                     "static_ref": self.static_ref, "static_digest": cache_digest, "model": model, "optimizer": optimizer,
                     "loop": {key: copy.deepcopy(local[key]) for key in LOOP_FIELDS},
                     "rng": before, "events": copy.deepcopy(self.events),
                     "pipeline": copy.deepcopy(vars(local["pipe"])),
                     "scheduler_next": next_lrs(local["gaussians"], k + 1)}
            validate_checkpoint(state, self.binding, self.static)
            finite_state((model, optimizer, state["loop"]))
            comparison = {key: state[key] for key in COMPARISON_FIELDS}
            digest_record = {"iteration": k, "exact_digest": diagnostic_digest(comparison)}
            self.check_reference(digest_record, (k, True))
            self.trace.write(json.dumps(digest_record) + "\n")
            self.trace.flush()
            capture_seconds = time.monotonic() - capture_started
            save_seconds = 0.
            disk_bytes = None
            if k in self.args.exact_save or k == self.args.exact_stop:
                save_started = time.monotonic()
                saved = atomic_save(self.root / ("step_%05d" % k), state)
                save_seconds = time.monotonic() - save_started
                disk_bytes = saved["bytes"]
            self.metric({"event": "full_capture", "iteration": k, "points": len(local["gaussians"]._xyz),
                         "dynamic_tensor_bytes": tensor_bytes(model) + tensor_bytes(optimizer),
                         "capture_hash_seconds": capture_seconds, "publication_seconds": save_seconds,
                         "checkpoint_file_bytes": disk_bytes})
            require(equal(before, rng_state()), "checkpoint capture consumed RNG")
        self.in_step = False
        self.next_expected = k + 1
        if k == self.args.exact_stop:
            if k == 30000:
                finite_terminal_model(local["gaussians"])
            self.metric({"event": "stop", "iteration": k})
            self.trace.close()
            self.performance.close()
            return True
        return False

