"""Exact native STDLoc stage boundaries. No replacement numerical kernels.

Checkpoint blobs are trusted local artifacts: verified SHA and a typed, complete
manifest are required before torch.load. CUDA is never initialized on import.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile

SCHEMA = "std_native_exact_state_v1"
PARAMETERS = ("_xyz", "_features_dc", "_features_rest", "_scaling",
              "_rotation", "_opacity", "_loc_feature")
CAMERA_FIELDS = ("uid", "colmap_id", "image_name", "R", "T", "FoVx", "FoVy",
                 "image_width", "image_height", "zfar", "znear", "trans", "scale",
                 "world_view_transform", "projection_matrix", "full_proj_transform",
                 "camera_center")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tree_manifest(value, device_locations=None):
    """Typed structure plus every tensor's exact bytes, shape, stride and device."""
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        return {"tensor": hashlib.sha256(raw.tobytes()).hexdigest(),
                "dtype": str(value.dtype), "shape": list(value.shape),
                "stride": list(value.stride()),
                "device": (device_locations or {}).get(value.untyped_storage()._cdata, str(value.device)),
                "requires_grad": value.requires_grad,
                "parameter": isinstance(value, torch.nn.Parameter)}
    if isinstance(value, np.ndarray):
        return {"numpy": hashlib.sha256(value.tobytes()).hexdigest(),
                "dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return tree_manifest(value.item(), device_locations)
    if isinstance(value, dict):
        return {"dict": [[tree_manifest(k, device_locations), tree_manifest(v, device_locations)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {type(value).__name__: [tree_manifest(v, device_locations) for v in value]}
    if value is None or isinstance(value, (bool, int, str)):
        return {type(value).__name__: value}
    if isinstance(value, float):
        return {"float": value.hex()}
    raise TypeError("unsupported state leaf: " + type(value).__name__)


def digest(value, device_locations=None):
    data = json.dumps(tree_manifest(value, device_locations), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def rng_state():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def set_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("visible CUDA device count changed")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def module_extra(module):
    return {"grads": {name: p.grad for name, p in module.named_parameters()},
            "requires_grad": {name: p.requires_grad for name, p in module.named_parameters()},
            "modes": {name: part.training for name, part in module.named_modules()}}


def restore_extra(module, extra):
    params = dict(module.named_parameters())
    if set(params) != set(extra["grads"]) or set(params) != set(extra["requires_grad"]):
        raise ValueError("parameter names changed")
    for name, p in params.items():
        p.requires_grad_(extra["requires_grad"][name])
        grad = extra["grads"][name]
        p.grad = None if grad is None else grad.detach()
    modules = dict(module.named_modules())
    if set(modules) != set(extra["modes"]):
        raise ValueError("module names changed")
    for name, part in modules.items():
        part.training = extra["modes"][name]


def optimizer_binding(gaussians):
    names = {id(p): name for name, p in gaussians.named_parameters()}
    return [[names[id(p)] for p in group["params"]]
            for group in gaussians.optimizer.param_groups]


def runtime_flags():
    import torch
    return {"torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "cuda_device_types": [(torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
                                  for i in range(torch.cuda.device_count())] if torch.cuda.is_initialized() else [],
            "cudnn_version": torch.backends.cudnn.version(),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "num_threads": torch.get_num_threads(),
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def scene_contract(scene):
    def cameras(items):
        names = [x.image_name for x in items]
        if len(set(names)) != len(names):
            raise ValueError("camera names must be unique")
        return [{"name": cam.image_name,
                 "fingerprint": digest({k: getattr(cam, k) for k in CAMERA_FIELDS})}
                for cam in items]
    return {"train": cameras(scene.getTrainCameras()),
            "test": cameras(scene.getTestCameras()), "extent": scene.cameras_extent,
            "resolution_scales": scene.resolution_scales,
            "feature_type": scene.feature_type, "longest_edge": scene.longest_edge}


def stack_indices(stack, scene):
    if stack is None:
        return None
    index = {id(c): i for i, c in enumerate(scene.getTrainCameras())}
    result = [index[id(c)] for c in stack]
    if len(set(result)) != len(result):
        raise ValueError("duplicate camera in remaining stack")
    return result


def stack_from_indices(indices, scene):
    if indices is None:
        return None
    cams = scene.getTrainCameras()
    if len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < len(cams) for i in indices):
        raise ValueError("invalid remaining camera stack")
    return [cams[i] for i in indices]


def mask_contract(masks, scene):
    """Byte manifests of original wrapper masks, rebuilt from frozen inputs."""
    if masks is None:
        return None
    result = {}
    for camera in scene.getTrainCameras() + scene.getTestCameras():
        name = camera.image_name
        if name not in result:
            layers = masks[name]
            if len(layers) != 3:
                raise ValueError("native object/sky/geometry mask triplet required")
            result[name] = tree_manifest(layers)
    return result


def require_finite_tensors(value):
    import torch
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite checkpoint tensor")
    elif isinstance(value, dict):
        for v in value.values():
            require_finite_tensors(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            require_finite_tensors(v)


def capture_state(ns, contract, identity):
    g, extractor = ns["gaussians"], ns["feature_extractor"]
    if tuple(dict(g.named_parameters())) != PARAMETERS:
        # Native registration order differs from capture tuple order.
        if set(dict(g.named_parameters())) != set(PARAMETERS):
            raise ValueError("unrecognized Gaussian parameters")
    return {"schema": SCHEMA, "identity": identity, "iteration": ns["iteration"],
            "scene": contract, "options": vars(ns["opt"]).copy(),
            "masks": mask_contract(ns.get("masks"), ns["scene"]),
            "dataset": vars(ns["dataset"]).copy(),
            "gaussian_capture": g.capture(), "gaussian_extra": module_extra(g),
            "gaussian_constants": {"max_sh_degree": g.max_sh_degree,
                                   "percent_dense": g.percent_dense},
            "optimizer_binding": optimizer_binding(g),
            "feature_state": extractor.state_dict(),
            "feature_extra": module_extra(extractor),
            "background": ns["background"],
            "remaining_stack": stack_indices(ns["viewpoint_stack"], ns["scene"]),
            "ema_loss_for_log": ns["ema_loss_for_log"],
            "runtime_flags": runtime_flags(), "rng": rng_state()}


def restore_state(state, ns, contract, identity):
    if state["schema"] != SCHEMA or state["identity"] != identity:
        raise ValueError("checkpoint identity mismatch")
    for key, current in (("scene", contract), ("masks", mask_contract(ns.get("masks"), ns["scene"])),
                         ("options", vars(ns["opt"])),
                         ("dataset", vars(ns["dataset"])), ("runtime_flags", runtime_flags())):
        if digest(state[key]) != digest(current):
            raise ValueError("checkpoint contract drift: " + key)
    g, extractor = ns["gaussians"], ns["feature_extractor"]
    g.restore(state["gaussian_capture"], ns["opt"])
    for name, value in state["gaussian_constants"].items():
        setattr(g, name, value)
    restore_extra(g, state["gaussian_extra"])
    if optimizer_binding(g) != state["optimizer_binding"]:
        raise ValueError("optimizer parameter binding drift")
    extractor.load_state_dict(state["feature_state"], strict=True)
    restore_extra(extractor, state["feature_extra"])
    return {"first_iter": state["iteration"], "background": state["background"],
            "viewpoint_stack": stack_from_indices(state["remaining_stack"], ns["scene"]),
            "ema_loss_for_log": state["ema_loss_for_log"]}


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix="." + path.name,
                                     delete=False) as f:
        tmp = Path(f.name)
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)  # exclusive publication; never overwrite a prior receipt
        fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        tmp.unlink(missing_ok=True)


def publish_checkpoint(directory, state):
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    blob = directory / "state.pt"
    temp = directory / ".state.pt.partial"
    require_finite_tensors(state)
    manifest = tree_manifest(state)
    with temp.open("xb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.link(temp, blob)
    temp.unlink()
    record = {"schema": SCHEMA, "complete": True, "iteration": state["iteration"],
              "identity": state["identity"], "blob": "state.pt",
              "blob_bytes": blob.stat().st_size, "blob_sha256": sha(blob),
              "state_digest": digest(state), "state_manifest": manifest}
    atomic_json(directory / "complete.json", record)
    roundtrip, _ = load_checkpoint(directory, require_roundtrip=False, cpu_verify=True)
    del roundtrip
    atomic_json(directory / "roundtrip_verified.json",
                {"complete": True, "iteration": state["iteration"],
                 "state_digest": record["state_digest"], "blob_sha256": record["blob_sha256"],
                 "roundtrip_location": "cpu", "original_storage_device_metadata_verified": True})
    return record


def load_checkpoint(directory, require_roundtrip=True, cpu_verify=False):
    import torch
    directory = Path(directory)
    record = json.loads((directory / "complete.json").read_text())
    if record.get("complete") is not True or record.get("schema") != SCHEMA or record.get("blob") != "state.pt":
        raise ValueError("incomplete checkpoint")
    blob = directory / "state.pt"
    if blob.is_symlink() or blob.stat().st_size != record["blob_bytes"] or sha(blob) != record["blob_sha256"]:
        raise ValueError("checkpoint blob drift")
    # Boundary verification stays on CPU to avoid doubling live CUDA memory.
    # PyTorch supplies the original device for every serialized storage.
    # Actual resume leaves map_location=None and restores original devices.
    device_locations = {}
    def cpu_storage(storage, location):
        device_locations[storage._cdata] = location
        return storage
    with blob.open("rb") as stream:
        state = torch.load(stream, weights_only=False, map_location=cpu_storage if cpu_verify else None)
    if (tree_manifest(state, device_locations) != record["state_manifest"]
            or digest(state, device_locations) != record["state_digest"]):
        raise ValueError("checkpoint state bytes mismatch")
    if state["iteration"] != record["iteration"] or state["identity"] != record["identity"]:
        raise ValueError("checkpoint header mismatch")
    if require_roundtrip:
        verified = json.loads((directory / "roundtrip_verified.json").read_text())
        if (verified.get("complete") is not True or verified.get("iteration") != record["iteration"]
                or verified.get("state_digest") != record["state_digest"]
                or verified.get("blob_sha256") != record["blob_sha256"]):
            raise ValueError("checkpoint roundtrip proof missing or changed")
    return state, record


class Exact:
    def __init__(self):
        self.config = json.loads(Path(os.environ["STD_EXACT_CONFIG"]).read_text())
        self.state = None
        self.contract = None
        self.expected_state_manifest = None
        self.expected_state_digest = None
        self.trace = []
        self.start_rng_digest = None
        self.last_step = self.config["start"]
        self.identity = self.config["identity"]

    def prepare(self, ns):
        if ns["checkpoint"] is not None or ns["opt"].iterations != 30000:
            raise ValueError("native checkpoint/reduced iteration overrides forbidden")
        if not ns["train_detector"] or ns["dataset"].gaussian_type != "3dgs":
            raise ValueError("only original 3DGS plus natural detector supported")
        self.contract = scene_contract(ns["scene"])
        if self.config["start"] == 0:
            return None
        self.state, record = load_checkpoint(self.config["resume"])
        # Keep the JSON expectation separate from live restored tensor storage.
        self.expected_state_manifest = record["state_manifest"]
        self.expected_state_digest = record["state_digest"]
        if self.state["iteration"] != self.config["start"]:
            raise ValueError("wrong stage start")
        return restore_state(self.state, ns, self.contract, self.identity)

    def restore_rng_last(self, ns):
        if self.state is None:
            return
        set_rng(self.state["rng"])
        live = dict(ns)
        live["iteration"] = self.state["iteration"]
        current = capture_state(live, self.contract, self.identity)
        if (tree_manifest(current) != self.expected_state_manifest
                or digest(current) != self.expected_state_digest):
            raise ValueError("restored state not byte-exact before next native step")
        atomic_json(Path(self.config["run_dir"]) / "restore_verified.json",
                    {"complete": True, "iteration": self.state["iteration"],
                     "state_digest": digest(current), "rng_digest": digest(current["rng"])})
        self.state = None

    def after_step(self, ns):
        iteration = ns["iteration"]
        if iteration != self.last_step + 1:
            raise ValueError("native iteration sequence changed")
        self.last_step = iteration
        if not math.isfinite(ns["loss"].item()):
            raise ValueError("nonfinite native loss at iteration " + str(iteration))
        observe = (iteration <= self.config["start"] + 8 or iteration > self.config["stop"] - 8
                   or (iteration < getattr(ns["opt"], "densify_until_iter", 0)
                       and iteration % getattr(ns["opt"], "densification_interval", 100) == 0)
                   or iteration % 1000 == 0 or self.config.get("trace_start", 30001) <= iteration)
        if observe:
            self.trace.append({"iteration": iteration, "camera": ns["viewpoint_cam"].image_name,
                               "loss": ns["loss"].item(), "l1": ns["Ll1"].item(),
                               "feature_l1": ns["Ll1_feature"].item(),
                               "rng_digest": digest(rng_state()),
                               "gaussians": ns["gaussians"].get_xyz.shape[0] if hasattr(ns["gaussians"], "get_xyz") else ns["gaussians"]._xyz.shape[0],
                               "active_sh_degree": ns["gaussians"].active_sh_degree,
                               "optimizer_lr": [g["lr"] for g in ns["gaussians"].optimizer.param_groups]})
        checkpoint_at = self.config.get("checkpoint_at", self.config["stop"])
        if iteration == checkpoint_at:
            state = capture_state(ns, self.contract, self.identity)
            publish_checkpoint(self.config["checkpoint_out"], state)
            if digest(rng_state()) != digest(state["rng"]):
                raise ValueError("checkpoint instrumentation consumed RNG")
        if iteration == self.config["stop"]:
            if self.trace:
                atomic_json(Path(self.config["run_dir"]) / "trace.json", self.trace)
            atomic_json(Path(self.config["run_dir"]) / "gaussian_complete.json",
                        {"complete": True, "start": self.config["start"], "stop": iteration,
                         "native_total_iterations": 30000,
                         "all_observed_native_losses_finite": True,
                         "continuously_floating_point_equivalent_trajectory_claimed": False,
                         "native_terminal_skips_optimizer_step": iteration == 30000,
                         "probe_only": self.config.get("probe_only", False)})
            return True
        return False

    def before_detector(self, ns):
        if self.config["stage"] != "detector" or self.config["start"] != 30000:
            raise ValueError("Gaussian stage may not fall through to detector")
        if self.last_step != 30000:
            raise ValueError("detector handoff iteration mismatch")
        atomic_json(Path(self.config["run_dir"]) / "detector_entry.json",
                    {"complete": True, "iteration": 30000, "rng_digest": digest(rng_state()),
                     "natural_function": "train_detector.training_detector",
                     "train_iteration": 30000, "landmark_num": ns["landmark_num"],
                     "landmark_k": ns["landmark_k"]})

    def detector_complete(self):
        import torch
        terminal = Path(self.identity["model_path"]) / "detector/30000_detector.pth"
        with terminal.open("rb") as stream:
            terminal_state = torch.load(stream, map_location="cpu", weights_only=True)
        require_finite_tensors(terminal_state)
        atomic_json(Path(self.config["run_dir"]) / "detector_complete.json",
                    {"complete": True, "stage": "detector", "train_iteration": 30000,
                     "terminal_detector_tensors_finite": True, "terminal_sha256": sha(terminal)})
