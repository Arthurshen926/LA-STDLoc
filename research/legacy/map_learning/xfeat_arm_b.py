"""CPU-only, mapping-only XFeat Arm-B probe materialization.

The candidate detector is deliberately never called.  This module recreates
the frozen SuperPoint mapping image input, applies the locked XFeat resize,
and samples the independent 64D dense field at every exact SuperPoint row.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
from types import ModuleType

import torch
import torch.nn.functional as F

from data.datasets import ColmapDataset
from data.images import resolution_from_longest_edge
from features.multiview_fusion import sample_mask_at_grid_uv
from map_learning.frontend_upper_bound import (
    PROBE_SCHEMA,
    PROBE_VERSION,
    file_sha256,
    tensor_sha256,
    validate_probe,
)


DESCRIPTOR_DIM = 64
COORDINATE_CONVENTION = (
    "reference_grid_index_then_cached_pixel_center_offset"
)
PRODUCER_SCHEMA = "lafgs_xfeat_arm_b_producer"
PRODUCER_VERSION = 1
TEACHER_SCHEMA = "lafgs_v9_active_map_complete_positive_teacher"
NATIVE_COORDINATE_CONVENTION = (
    "superpoint_grid_index_then_pnp_plus_half_v1"
)
QUERY_CONTRACT = {
    "query_feature_contract": "native_resized_input",
    "feature_resize_mode": "resize_image_then_native_stride8",
    "descriptor_source": "superpoint_native_dense_resized_input",
    "coordinate_convention": "feature_grid_index_plus_half_physical_v1",
    "valid_mask_policy": "object_and_sky_and_distortion_v1",
    "native_sparse_enabled": True,
    "native_sparse_coordinate_convention": NATIVE_COORDINATE_CONVENTION,
}
XFEAT_MODEL_RELATIVE = Path("encoders/XFeat/modules/model.py")
XFEAT_INTERPOLATOR_RELATIVE = Path(
    "encoders/XFeat/modules/interpolator.py"
)
XFEAT_WRAPPER_RELATIVE = Path("encoders/XFeat/modules/xfeat.py")
XFEAT_WEIGHTS_RELATIVE = Path("encoders/XFeat/weights/xfeat.pt")
XFEAT_LICENSE_RELATIVE = Path("encoders/XFeat/LICENSE")


@dataclass(frozen=True)
class XFeatArtifactSpec:
    """Expected immutable identity of one local XFeat checkout."""

    worktree: Path
    weights: Path
    expected_weights_sha256: str
    expected_parent_commit: str
    expected_xfeat_tree: str
    expected_model_sha256: str
    expected_interpolator_sha256: str
    expected_wrapper_sha256: str


@dataclass(frozen=True)
class ReferenceContext:
    """Validated mapping-only reference registry."""

    dataset: ColmapDataset
    cameras: tuple
    names: tuple[str, ...]
    queries: Mapping[str, Mapping]
    signature_payload: Mapping
    requested_keypoint_count: int


def _require_digest(value: str, *, length: int, label: str) -> str:
    digest = str(value).lower()
    if re.fullmatch(f"[0-9a-f]{{{length}}}", digest) is None:
        raise ValueError(f"{label} must be a {length}-character hex digest")
    return digest


def _git(worktree: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(worktree), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        raise ValueError(
            f"cannot verify local XFeat Git lineage: {' '.join(arguments)}"
        ) from error
    return completed.stdout.strip()


def _require_within(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the locked worktree: {resolved}") from error
    return resolved


def validate_xfeat_artifact(spec: XFeatArtifactSpec) -> dict:
    """Verify a clean local Git tree, code hashes, and the exact checkpoint."""
    worktree = Path(spec.worktree).expanduser().resolve()
    if not worktree.is_dir():
        raise FileNotFoundError(f"XFeat worktree not found: {worktree}")
    top_level = Path(_git(worktree, "rev-parse", "--show-toplevel")).resolve()
    if top_level != worktree:
        raise ValueError(
            f"XFeat worktree must be the Git top level: {worktree} != {top_level}"
        )
    expected_commit = _require_digest(
        spec.expected_parent_commit, length=40, label="expected parent commit"
    )
    actual_commit = _git(worktree, "rev-parse", "HEAD").lower()
    if actual_commit != expected_commit:
        raise ValueError(
            f"XFeat parent commit mismatch: {actual_commit} != {expected_commit}"
        )
    if _git(worktree, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("XFeat external worktree is not clean")

    expected_tree = _require_digest(
        spec.expected_xfeat_tree, length=40, label="expected XFeat tree"
    )
    actual_tree = _git(worktree, "rev-parse", "HEAD:encoders/XFeat").lower()
    if actual_tree != expected_tree:
        raise ValueError(f"XFeat tree mismatch: {actual_tree} != {expected_tree}")

    expected_paths = {
        "model": worktree / XFEAT_MODEL_RELATIVE,
        "interpolator": worktree / XFEAT_INTERPOLATOR_RELATIVE,
        "wrapper": worktree / XFEAT_WRAPPER_RELATIVE,
        "weights": worktree / XFEAT_WEIGHTS_RELATIVE,
        "license": worktree / XFEAT_LICENSE_RELATIVE,
    }
    weights = _require_within(
        Path(spec.weights), worktree, label="XFeat checkpoint"
    )
    if weights != expected_paths["weights"].resolve():
        raise ValueError(
            "Arm B accepts only encoders/XFeat/weights/xfeat.pt; "
            f"got {weights}"
        )
    for label, path in expected_paths.items():
        resolved = _require_within(path, worktree, label=f"XFeat {label}")
        if not resolved.is_file():
            raise FileNotFoundError(f"XFeat {label} file not found: {resolved}")
        relative = resolved.relative_to(worktree).as_posix()
        _git(worktree, "ls-files", "--error-unmatch", relative)

    expected_hashes = {
        "model": _require_digest(
            spec.expected_model_sha256,
            length=64,
            label="expected model SHA256",
        ),
        "interpolator": _require_digest(
            spec.expected_interpolator_sha256,
            length=64,
            label="expected interpolator SHA256",
        ),
        "wrapper": _require_digest(
            spec.expected_wrapper_sha256,
            length=64,
            label="expected wrapper SHA256",
        ),
        "weights": _require_digest(
            spec.expected_weights_sha256,
            length=64,
            label="expected weights SHA256",
        ),
    }
    actual_hashes = {
        label: file_sha256(path) for label, path in expected_paths.items()
    }
    for label in expected_hashes:
        if actual_hashes[label] != expected_hashes[label]:
            raise ValueError(
                f"XFeat {label} SHA256 mismatch: "
                f"{actual_hashes[label]} != {expected_hashes[label]}"
            )
    return {
        "worktree": str(worktree),
        "parent_commit": actual_commit,
        "xfeat_tree": actual_tree,
        "git_clean": True,
        "files": {
            label: {
                "path": str(path.resolve()),
                "sha256": actual_hashes[label],
                "size_bytes": int(path.stat().st_size),
                "git_blob": _git(
                    worktree,
                    "rev-parse",
                    f"HEAD:{path.resolve().relative_to(worktree).as_posix()}",
                ).lower(),
            }
            for label, path in expected_paths.items()
        },
    }


def _load_module(path: Path, *, identity: str) -> ModuleType:
    name = f"_lafgs_locked_xfeat_{identity}_{file_sha256(path)[:16]}"
    # Compile verified source directly so Python cannot create __pycache__ in
    # the external clean worktree.
    source = path.read_bytes()
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def load_xfeat_cpu(artifact: Mapping) -> tuple[torch.nn.Module, torch.nn.Module, dict]:
    """Instantiate the exact audited model and bicubic sampler on CPU."""
    files = artifact["files"]
    model_module = _load_module(Path(files["model"]["path"]), identity="model")
    interpolator_module = _load_module(
        Path(files["interpolator"]["path"]), identity="interpolator"
    )
    model_class = getattr(model_module, "XFeatModel", None)
    interpolator_class = getattr(interpolator_module, "InterpolateSparse2d", None)
    if model_class is None or interpolator_class is None:
        raise ValueError("locked XFeat implementation lacks required classes")
    model = model_class().cpu().eval()
    try:
        state = torch.load(
            files["weights"]["path"], map_location="cpu", weights_only=True
        )
    except Exception as error:
        raise ValueError("cannot safely load locked XFeat state dict") from error
    if not isinstance(state, Mapping) or not state:
        raise ValueError("XFeat checkpoint must be a non-empty state dict")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise ValueError("XFeat checkpoint contains non-tensor state entries")
    for key, value in state.items():
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError(f"XFeat checkpoint contains non-finite tensor: {key}")
    try:
        incompatible = model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError("XFeat state dict does not strictly match model.py") from error
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("XFeat strict state load returned incompatible keys")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()) or any(
        buffer.device.type != "cpu" for buffer in model.buffers()
    ):
        raise RuntimeError("XFeat model escaped the CPU-only contract")
    interpolator = interpolator_class("bicubic").cpu().eval()
    if str(getattr(interpolator, "mode", "")) != "bicubic":
        raise ValueError("XFeat descriptor sampler is not bicubic")
    state_summary = {
        "state_dict_entries": len(state),
        "state_tensor_element_count": sum(
            int(value.numel()) for value in state.values()
        ),
        "strict_load": True,
        "device": "cpu",
        "dtype": "float32",
    }
    return model, interpolator, state_summary


def _canonical_signature(payload: Mapping) -> str:
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_query_name(name: str) -> str:
    normalized = str(name).replace("\\", "/")
    path = Path(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe query image name: {name!r}")
    return normalized


def _query_names_sha256(names: Sequence[str]) -> str:
    payload = "".join(f"{len(name)}:{name}\n" for name in names)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _producer_code_identity() -> dict:
    root = Path(__file__).resolve().parents[1]
    relatives = (
        "map_learning/xfeat_arm_b.py",
        "map_learning/frontend_upper_bound.py",
        "data/datasets.py",
        "data/images.py",
        "features/multiview_fusion.py",
    )
    return {
        relative: {
            "path": str((root / relative).resolve()),
            "sha256": file_sha256(root / relative),
        }
        for relative in relatives
    }


def validate_reference_context(
    query_cache: Mapping,
    teacher: Mapping,
    *,
    dataset_root: str | Path,
) -> ReferenceContext:
    """Fail closed unless cache and teacher are the exact mapping split."""
    if not isinstance(query_cache, Mapping) or not isinstance(teacher, Mapping):
        raise TypeError("query cache and teacher must be mappings")
    if int(query_cache.get("version", -1)) < 3:
        raise ValueError("query cache must use version 3 or newer")
    signature = str(query_cache.get("signature", "")).lower()
    _require_digest(signature, length=64, label="query-cache signature")
    signature_payload = query_cache.get("signature_payload")
    if not isinstance(signature_payload, Mapping):
        raise ValueError("query cache lacks a signature_payload mapping")
    if _canonical_signature(signature_payload) != signature:
        raise ValueError("query-cache signature does not match signature_payload")
    if int(signature_payload.get("version", -1)) < 11:
        raise ValueError("query-cache signature contract must be version 11+")
    for key, expected in QUERY_CONTRACT.items():
        if signature_payload.get(key) != expected:
            raise ValueError(
                f"query-cache signature field {key!r} is not locked: "
                f"{signature_payload.get(key)!r} != {expected!r}"
            )
    if float(signature_payload.get("pixel_center_offset", -1.0)) != 0.5:
        raise ValueError("query-cache pixel center offset must be exactly 0.5")
    resolution = int(signature_payload.get("resolution", 0))
    longest_edge = int(signature_payload.get("longest_edge", -1))
    if resolution <= 0 or longest_edge < 0:
        raise ValueError("query-cache image resolution contract is invalid")

    root = Path(dataset_root).expanduser().resolve()
    source_root = Path(str(signature_payload.get("source_path", ""))).expanduser().resolve()
    if root != source_root:
        raise ValueError(f"dataset root differs from query-cache lineage: {root} != {source_root}")
    images = str(signature_payload.get("images", ""))
    if not images or Path(images).is_absolute() or ".." in Path(images).parts:
        raise ValueError("query-cache images directory is unsafe")
    dataset = ColmapDataset(root, images=images)
    cameras = tuple(dataset.split("mapping"))
    mapping_names = tuple(_safe_query_name(camera.image_name) for camera in cameras)
    if len(set(mapping_names)) != len(mapping_names):
        raise ValueError("mapping query names are not unique")

    raw_names = teacher.get("query_names")
    if not isinstance(raw_names, Sequence) or isinstance(raw_names, (str, bytes)):
        raise ValueError("teacher query_names must be an ordered sequence")
    teacher_names = tuple(_safe_query_name(name) for name in raw_names)
    if teacher.get("schema") != TEACHER_SCHEMA or int(teacher.get("version", -1)) != 1:
        raise ValueError("unsupported complete-positive teacher schema")
    if teacher_names != mapping_names:
        raise ValueError("teacher names/order are not the exact mapping-only split")

    queries = query_cache.get("queries")
    if not isinstance(queries, Mapping):
        raise ValueError("query cache must contain an explicit queries mapping")
    if set(queries) != set(mapping_names):
        raise ValueError("query cache is not the exact mapping-only query set")

    records = teacher.get("records")
    if not isinstance(records, Sequence) or len(records) != len(mapping_names):
        raise ValueError("teacher must contain one record per mapping query")
    by_index = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("teacher records must be mappings")
        index = int(record.get("query_index", -1))
        if index in by_index:
            raise ValueError("teacher query indices must be unique")
        by_index[index] = record
    if set(by_index) != set(range(len(mapping_names))):
        raise ValueError("teacher query indices do not cover the mapping split")

    requested_counts = set()
    signature_k = int(signature_payload.get("native_sparse_keypoint_count", -1))
    signature_nms = int(signature_payload.get("native_sparse_nms_radius", -1))
    if signature_k <= 0 or signature_nms < 0:
        raise ValueError("query-cache sparse K/NMS lineage is invalid")
    for index, name in enumerate(mapping_names):
        record = by_index[index]
        if _safe_query_name(record.get("query_name", "")) != name:
            raise ValueError(f"teacher record/name mismatch at index {index}")
        cached = queries[name]
        if not isinstance(cached, Mapping):
            raise ValueError(f"query cache record is not a mapping: {name}")
        keypoints = torch.as_tensor(cached.get("native_keypoints")).detach().cpu().float()
        descriptors = torch.as_tensor(cached.get("native_descriptors")).detach().cpu()
        if keypoints.ndim != 2 or keypoints.shape[1] != 2:
            raise ValueError(f"native keypoints must be [N,2]: {name}")
        if descriptors.ndim != 2 or descriptors.shape[0] != keypoints.shape[0]:
            raise ValueError(f"native descriptor rows do not match keypoints: {name}")
        if not bool(torch.isfinite(keypoints).all()):
            raise ValueError(f"native keypoints are non-finite: {name}")
        height, width = (int(value) for value in cached.get("native_input_hw", ()))
        if height <= 0 or width <= 0:
            raise ValueError(f"native input dimensions are invalid: {name}")
        if keypoints.numel():
            inside = (
                (keypoints[:, 0] >= 0)
                & (keypoints[:, 0] <= width - 1)
                & (keypoints[:, 1] >= 0)
                & (keypoints[:, 1] <= height - 1)
            )
            if not bool(inside.all()):
                raise ValueError(f"native keypoints escape cached input: {name}")
        valid_mask = torch.as_tensor(cached.get("native_valid_mask")).bool().squeeze()
        if tuple(valid_mask.shape) != (height, width):
            raise ValueError(f"native valid-mask shape mismatch: {name}")
        if keypoints.numel() and not bool(
            sample_mask_at_grid_uv(valid_mask, keypoints).all()
        ):
            raise ValueError(f"native keypoints violate frozen valid mask: {name}")
        if float(cached.get("pixel_center_offset", 0.5)) != 0.5:
            raise ValueError(f"native pixel center offset is not locked: {name}")
        sparse = cached.get("native_sparse_metadata")
        if not isinstance(sparse, Mapping):
            raise ValueError(f"native sparse metadata is missing: {name}")
        detected_k = int(sparse.get("detect_num", -1))
        requested_k = int(sparse.get("requested_keypoint_count", detected_k))
        if detected_k != requested_k or detected_k != signature_k:
            raise ValueError(f"native requested K lineage mismatch: {name}")
        if int(sparse.get("nms_radius", -1)) != signature_nms:
            raise ValueError(f"native NMS lineage mismatch: {name}")
        if sparse.get("coordinate_convention") != NATIVE_COORDINATE_CONVENTION:
            raise ValueError(f"native coordinate convention mismatch: {name}")
        if int(sparse.get("keypoint_count_after_mask", keypoints.shape[0])) != int(
            keypoints.shape[0]
        ):
            raise ValueError(f"native post-mask row count mismatch: {name}")
        rows = torch.as_tensor(record.get("query_rows")).long().reshape(-1)
        if rows.numel():
            if int(rows.min()) < 0 or int(rows.max()) >= keypoints.shape[0]:
                raise ValueError(f"teacher rows exceed reference rows: {name}")
            if int(torch.unique(rows).numel()) != int(rows.numel()):
                raise ValueError(f"teacher rows are not unique: {name}")
        requested_counts.add(detected_k)
    if requested_counts != {signature_k}:
        raise ValueError("reference detector K is not globally fixed")
    return ReferenceContext(
        dataset=dataset,
        cameras=cameras,
        names=mapping_names,
        queries=queries,
        signature_payload=signature_payload,
        requested_keypoint_count=signature_k,
    )


def _mask_at_hw(dataset: ColmapDataset, name: str, target_hw: tuple[int, int]) -> torch.Tensor:
    masks = getattr(dataset, "_masks", None)
    if masks is None or name not in masks:
        return torch.ones(target_hw, dtype=torch.bool)
    channels = masks[name]
    if len(channels) < 3:
        raise ValueError(f"valid mask requires three channels: {name}")
    resized = []
    for channel in channels[:3]:
        value = torch.as_tensor(channel, dtype=torch.float32, device="cpu")
        while value.ndim > 2:
            value = value.squeeze(0)
        if value.ndim != 2:
            raise ValueError(f"valid-mask channel must be two-dimensional: {name}")
        resized.append(
            F.interpolate(value[None, None], size=target_hw, mode="nearest")[0, 0].bool()
        )
    return resized[0] & resized[1] & resized[2]


def recreate_native_input(
    context: ReferenceContext,
    camera,
    cached: Mapping,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Recreate bootstrap._native_feature_input exactly on CPU."""
    image_sha256 = file_sha256(camera.image_path)
    image = context.dataset.load_image(camera).cpu().float()
    if file_sha256(camera.image_path) != image_sha256:
        raise RuntimeError(f"mapping image changed while loading: {camera.image_name}")
    if image.ndim != 3 or image.shape[0] not in (1, 3):
        raise ValueError(f"mapping image must have one or three channels: {camera.image_name}")
    if not bool(torch.isfinite(image).all()) or bool((image < 0).any()) or bool((image > 1).any()):
        raise ValueError(f"mapping image is not finite RGB in [0,1]: {camera.image_name}")
    original_hw = (int(image.shape[-2]), int(image.shape[-1]))
    resolution = int(context.signature_payload["resolution"])
    scene_hw = original_hw
    if resolution != 1:
        scene_hw = (
            round(original_hw[0] / float(resolution)),
            round(original_hw[1] / float(resolution)),
        )
        if min(scene_hw) <= 0:
            raise ValueError(f"scene resolution collapses image: {camera.image_name}")
        image = F.interpolate(
            image[None], size=scene_hw, mode="bilinear", align_corners=False
        )[0]
    valid_mask = _mask_at_hw(context.dataset, camera.image_name, scene_hw)
    native_hw = resolution_from_longest_edge(
        scene_hw[0],
        scene_hw[1],
        int(context.signature_payload["longest_edge"]),
    )
    if tuple(image.shape[-2:]) != tuple(native_hw):
        image = F.interpolate(
            image[None], size=native_hw, mode="bilinear", align_corners=False
        )[0]
        valid_mask = F.interpolate(
            valid_mask[None, None].float(), size=native_hw, mode="nearest"
        )[0, 0].bool()
    expected_hw = tuple(int(value) for value in cached["native_input_hw"])
    if tuple(native_hw) != expected_hw:
        raise ValueError(
            f"recreated native HW differs from query cache for {camera.image_name}: "
            f"{tuple(native_hw)} != {expected_hw}"
        )
    cached_mask = torch.as_tensor(cached["native_valid_mask"]).bool().squeeze()
    if not torch.equal(valid_mask, cached_mask):
        raise ValueError(f"recreated valid mask differs from query cache: {camera.image_name}")
    masked = image * valid_mask[None].to(dtype=image.dtype)
    return masked.contiguous(), valid_mask.contiguous(), {
        "source_image_logical_path": str(Path(camera.image_path)),
        "source_image_path": str(Path(camera.image_path).resolve()),
        "source_image_sha256": image_sha256,
        "source_camera_hw": list(original_hw),
        "scene_resolution_hw": list(scene_hw),
        "native_input_hw": list(native_hw),
        "native_valid_mask_sha256": tensor_sha256(valid_mask),
        "native_masked_rgb_sha256": tensor_sha256(masked),
        "rgb_resize_mode": "bilinear_align_corners_false",
        "mask_resize_mode": "nearest",
        "mask_policy": QUERY_CONTRACT["valid_mask_policy"],
    }


def xfeat_resize_contract(native_hw: Sequence[int]) -> dict:
    height, width = (int(value) for value in native_hw)
    xfeat_height, xfeat_width = (height // 32) * 32, (width // 32) * 32
    if xfeat_height < 32 or xfeat_width < 32:
        raise ValueError("XFeat requires native height and width of at least 32")
    return {
        "native_input_hw": [height, width],
        "xfeat_input_hw": [xfeat_height, xfeat_width],
        "rh": float(height / xfeat_height),
        "rw": float(width / xfeat_width),
        "resize_mode": "bilinear_align_corners_false",
        "native_to_xfeat_xy": "(x/rw,y/rh)",
        "xfeat_to_native_xy": "(rw*x,rh*y)",
    }


def native_to_xfeat_coordinates(
    native_keypoints: torch.Tensor,
    contract: Mapping,
) -> torch.Tensor:
    keypoints = torch.as_tensor(native_keypoints).detach().cpu().float()
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError("native keypoints must have shape [N,2]")
    scale = keypoints.new_tensor([float(contract["rw"]), float(contract["rh"])])
    return keypoints / scale


def sample_xfeat_descriptor_field(
    dense_field: torch.Tensor,
    native_keypoints: torch.Tensor,
    *,
    contract: Mapping,
    interpolator: torch.nn.Module,
) -> torch.Tensor:
    """Apply locked XFeat bicubic sampling and final row normalization."""
    dense = torch.as_tensor(dense_field).cpu().float()
    xfeat_height, xfeat_width = (
        int(value) for value in contract["xfeat_input_hw"]
    )
    expected_shape = (1, DESCRIPTOR_DIM, xfeat_height // 8, xfeat_width // 8)
    if tuple(dense.shape) != expected_shape:
        raise ValueError(
            f"XFeat dense descriptor shape mismatch: {tuple(dense.shape)} != {expected_shape}"
        )
    keypoints = native_to_xfeat_coordinates(native_keypoints, contract)
    if keypoints.shape[0] == 0:
        return torch.empty((0, DESCRIPTOR_DIM), dtype=torch.float32)
    sampled = interpolator(
        dense,
        keypoints[None],
        H=xfeat_height,
        W=xfeat_width,
    )[0]
    if tuple(sampled.shape) != (keypoints.shape[0], DESCRIPTOR_DIM):
        raise ValueError("XFeat interpolator returned an invalid descriptor shape")
    sampled = F.normalize(sampled.float(), dim=1)
    if not bool(torch.isfinite(sampled).all()):
        raise ValueError("XFeat sampled descriptors are non-finite")
    if bool((torch.linalg.norm(sampled, dim=1) <= 0).any()):
        raise ValueError("XFeat sampled descriptors contain a zero row")
    return sampled.contiguous()


def _extract_descriptors(
    model: torch.nn.Module,
    interpolator: torch.nn.Module,
    native_image: torch.Tensor,
    native_keypoints: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    contract = xfeat_resize_contract(native_image.shape[-2:])
    xfeat_hw = tuple(int(value) for value in contract["xfeat_input_hw"])
    model_input = F.interpolate(
        native_image[None].cpu().float(),
        size=xfeat_hw,
        mode="bilinear",
        align_corners=False,
    )
    with torch.inference_mode():
        outputs = model(model_input)
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
            raise ValueError("XFeat model must return descriptor/logit/reliability outputs")
        dense = F.normalize(torch.as_tensor(outputs[0]).cpu().float(), dim=1)
        descriptors = sample_xfeat_descriptor_field(
            dense,
            native_keypoints,
            contract=contract,
            interpolator=interpolator,
        )
    return descriptors, {
        **contract,
        "dense_stride_px": 8,
        "dense_descriptor_dim": DESCRIPTOR_DIM,
        "dense_l2_normalized_before_sampling": True,
        "sparse_interpolation": {
            "implementation": "locked_XFeat_InterpolateSparse2d",
            "mode": "bicubic",
            "align_corners": False,
            "normgrid": (
                "(2*x/(xfeat_w-1)-1,2*y/(xfeat_h-1)-1)"
            ),
        },
        "sampled_rows_l2_normalized": True,
    }


def materialize_xfeat_arm_b(
    *,
    query_cache: Mapping,
    teacher: Mapping,
    query_cache_path: str | Path,
    teacher_path: str | Path,
    dataset_root: str | Path,
    artifact_spec: XFeatArtifactSpec,
) -> dict:
    """Materialize one complete consumer-valid Arm-B probe in memory."""
    query_cache_path = Path(query_cache_path).expanduser().resolve()
    teacher_path = Path(teacher_path).expanduser().resolve()
    if not query_cache_path.is_file() or not teacher_path.is_file():
        raise FileNotFoundError("query cache and teacher must be local files")
    artifact = validate_xfeat_artifact(artifact_spec)
    context = validate_reference_context(
        query_cache, teacher, dataset_root=dataset_root
    )
    model, interpolator, state_summary = load_xfeat_cpu(artifact)
    queries = {}
    total_rows = 0
    for query_index, (name, camera) in enumerate(zip(context.names, context.cameras)):
        cached = context.queries[name]
        native_keypoints = torch.as_tensor(cached["native_keypoints"]).cpu().float()
        native_image, _, image_lineage = recreate_native_input(
            context, camera, cached
        )
        descriptors, coordinate_lineage = _extract_descriptors(
            model, interpolator, native_image, native_keypoints
        )
        row_indices = torch.arange(native_keypoints.shape[0], dtype=torch.long)
        queries[name] = {
            "query_index": int(query_index),
            "query_name": name,
            "reference_row_count": int(native_keypoints.shape[0]),
            "reference_row_indices": row_indices,
            "reference_row_indices_sha256": tensor_sha256(row_indices),
            "reference_keypoints_sha256": tensor_sha256(native_keypoints),
            "descriptor_at_reference_keypoints": descriptors,
            "descriptor_sha256": tensor_sha256(descriptors),
            "image_lineage": image_lineage,
            "coordinate_lineage": coordinate_lineage,
        }
        total_rows += int(native_keypoints.shape[0])

    final_artifact = validate_xfeat_artifact(artifact_spec)
    if final_artifact != artifact:
        raise RuntimeError("XFeat artifact changed while materializing Arm B")

    weights = artifact["files"]["weights"]
    implementation_id = (
        f"xfeat_tree_{artifact['xfeat_tree'][:12]}"
        f"__model_{artifact['files']['model']['sha256'][:12]}"
        "__arm_b_v1"
    )
    probe = {
        "schema": PROBE_SCHEMA,
        "version": PROBE_VERSION,
        "mapping_only": True,
        "uses_test_queries": False,
        "reference": {
            "query_cache_path": str(query_cache_path),
            "query_cache_sha256": file_sha256(query_cache_path),
            "query_cache_signature": query_cache["signature"],
            "teacher_path": str(teacher_path),
            "teacher_sha256": file_sha256(teacher_path),
            "teacher_schema": teacher["schema"],
            "query_names": list(context.names),
            "query_names_sha256": _query_names_sha256(context.names),
            "reference_rows": "all_exact_cached_superpoint_native_rows",
        },
        "frontend": {
            "name": "xfeat_sparse_64d",
            "family": "independent_local_frontend",
            "implementation_id": implementation_id,
            "coordinate_convention": COORDINATE_CONVENTION,
            "descriptor_dim": DESCRIPTOR_DIM,
            "requested_keypoint_count": context.requested_keypoint_count,
            "weights": {
                "path": weights["path"],
                "sha256": weights["sha256"],
                "size_bytes": weights["size_bytes"],
            },
            "code": {
                "external_worktree": artifact["worktree"],
                "external_parent_commit": artifact["parent_commit"],
                "xfeat_tree": artifact["xfeat_tree"],
                "git_clean": artifact["git_clean"],
                "model": artifact["files"]["model"],
                "interpolator": artifact["files"]["interpolator"],
                "wrapper": artifact["files"]["wrapper"],
            },
        },
        "capabilities": {
            "detector_repeatability": False,
            "descriptor_identity": True,
        },
        "producer": {
            "schema": PRODUCER_SCHEMA,
            "version": PRODUCER_VERSION,
            "arm": "B_descriptor_identity",
            "device": "cpu",
            "dtype": "float32",
            "gpu_used": False,
            "network_access_used": False,
            "candidate_detector_used": False,
            "pair_matcher_used": False,
            "implementation_files": _producer_code_identity(),
            "state_dict": state_summary,
            "query_count": len(context.names),
            "reference_row_count": total_rows,
        },
        "queries": queries,
    }
    validation = validate_probe(
        probe,
        query_cache,
        teacher,
        require_descriptor=True,
        verify_weight_artifact=True,
        query_cache_path=query_cache_path,
        teacher_path=teacher_path,
    )
    if validation["candidate_descriptor_dim"] != DESCRIPTOR_DIM:
        raise RuntimeError("consumer validation changed the candidate dimension")
    probe["producer"]["consumer_validation"] = validation
    return probe
