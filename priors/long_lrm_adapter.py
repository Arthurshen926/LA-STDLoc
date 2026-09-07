"""Strict Cambridge adapters for mapping-only Long-LRM reconstruction.

Long-LRM normalizes the supplied camera rig before predicting Gaussians.  This
module records that normalization and converts the predicted standard 3DGS PLY
back to the Cambridge/AnyGSLoc mapping frame, including covariance rotations
and view-dependent spherical harmonics.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil

import cv2
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

from data.colmap import qvec2rotmat
from priors.i3dgs_adapter import (
    _camera_matrix,
    _ordered_fields,
    _read_colmap_model,
    _read_mapping_names,
    _sh_world_rotation,
)


PREPARED_SCHEMA = "anygsloc_long_lrm_prepared_mapping"
PREPARED_VERSION = 1
EXPORT_SCHEMA = "anygsloc_long_lrm_world_prior"
EXPORT_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _flat_name(image_name: str) -> str:
    path = Path(image_name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe image name: {image_name!r}")
    return "__".join(path.parts)


def _w2c(image) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = qvec2rotmat(image.qvec)
    matrix[:3, 3] = image.tvec
    return matrix


def _pose_diverse_indices(images: list, count: int) -> list[int]:
    """Deterministic farthest-point sampling over position and view direction."""

    if count <= 0 or len(images) < count:
        raise ValueError("Long-LRM selection needs at least the requested view count")
    c2ws = np.stack([np.linalg.inv(_w2c(image)) for image in images])
    centers = c2ws[:, :3, 3]
    forwards = c2ws[:, :3, 2]
    forwards /= np.linalg.norm(forwards, axis=1, keepdims=True)
    center = np.median(centers, axis=0)
    radii = np.linalg.norm(centers - center, axis=1)
    spatial_scale = float(np.quantile(radii, 0.9))
    if not np.isfinite(spatial_scale) or spatial_scale <= 1e-9:
        raise ValueError("mapping cameras do not span a valid spatial extent")
    spatial = centers / spatial_scale
    # A 180-degree direction change contributes one scene radius.  This keeps
    # position dominant while avoiding 32 near-identical views on one pass.
    direction = forwards * 0.5
    features = np.concatenate((spatial, direction), axis=1)
    start_scores = np.linalg.norm(centers - center, axis=1)
    first = int(np.flatnonzero(start_scores == start_scores.max())[0])
    selected = [first]
    minimum = np.linalg.norm(features - features[first], axis=1)
    minimum[first] = -np.inf
    while len(selected) < count:
        candidate = int(np.flatnonzero(minimum == minimum.max())[0])
        selected.append(candidate)
        distance = np.linalg.norm(features - features[candidate], axis=1)
        minimum = np.minimum(minimum, distance)
        minimum[selected] = -np.inf
    # The released inference helper samples frames in list order.  Preserve the
    # original trajectory ordering after choosing the diverse subset; feeding
    # the greedy FPS visitation order would create artificial camera jumps.
    return sorted(selected)


def long_lrm_pose_normalization(w2cs: np.ndarray) -> dict:
    """Reproduce the public Long-LRM inference camera normalization."""

    w2cs = np.asarray(w2cs, dtype=np.float32)
    if w2cs.shape != (32, 4, 4) or not np.isfinite(w2cs).all():
        raise ValueError("expected 32 finite world-to-camera matrices")
    c2ws = np.linalg.inv(w2cs).astype(np.float32)
    position_avg = c2ws[:, :3, 3].mean(axis=0)
    forward_avg = c2ws[:, :3, 2].sum(axis=0)
    down_avg = c2ws[:, :3, 1].sum(axis=0)

    def normalize(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm <= 1e-9:
            raise ValueError("Long-LRM pose normalization is degenerate")
        return vector / norm

    forward_avg = normalize(forward_avg)
    right_avg = normalize(np.cross(down_avg, forward_avg))
    down_avg = normalize(np.cross(forward_avg, right_avg))
    pose_average = np.eye(4, dtype=np.float32)
    pose_average[:3, :4] = np.stack(
        (right_avg, down_avg, forward_avg, position_avg), axis=1
    )
    rigid_world_to_normalized = np.linalg.inv(pose_average).astype(np.float32)
    normalized_c2ws = rigid_world_to_normalized[None] @ c2ws
    maximum = float(np.abs(normalized_c2ws[:, :3, 3]).max())
    if not np.isfinite(maximum) or maximum <= 1e-9:
        raise ValueError("Long-LRM mapping rig has zero normalized extent")
    scene_scale = float(1.0 / (1.35 * maximum))
    normalized_c2ws[:, :3, 3] *= scene_scale
    world_to_normalized = np.eye(4, dtype=np.float64)
    world_to_normalized[:3, :3] = (
        scene_scale * rigid_world_to_normalized[:3, :3]
    )
    world_to_normalized[:3, 3] = (
        scene_scale * rigid_world_to_normalized[:3, 3]
    )
    return {
        "scene_scale": scene_scale,
        "rigid_world_to_normalized": rigid_world_to_normalized.tolist(),
        "world_to_normalized_sim3": world_to_normalized.tolist(),
        "normalized_to_world_sim3": np.linalg.inv(world_to_normalized).tolist(),
        "normalized_c2ws": normalized_c2ws.tolist(),
    }


def prepare_cambridge_mapping(
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    images: str = "processed",
    view_count: int = 32,
) -> dict:
    """Prepare exactly 32 pose-diverse, mapping-only Cambridge views."""

    if view_count != 32:
        raise ValueError("the released Long-LRM checkpoint requires exactly 32 views")
    dataset_root = Path(dataset_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite prepared dataset: {output_root}")
    mapping_names = _read_mapping_names(dataset_root)
    extrinsics, cameras, extrinsic_path, intrinsic_path = _read_colmap_model(
        dataset_root
    )
    by_name = {image.name: image for image in extrinsics.values()}
    missing = sorted(set(mapping_names) - set(by_name))
    if missing:
        raise ValueError(f"mapping images missing from COLMAP model: {missing[:3]}")
    ordered_names = sorted(mapping_names)
    ordered_images = [by_name[name] for name in ordered_names]
    selected_indices = _pose_diverse_indices(ordered_images, view_count)
    selected = [(ordered_names[index], ordered_images[index]) for index in selected_indices]
    w2cs = np.stack([_w2c(image) for _, image in selected]).astype(np.float32)
    normalization = long_lrm_pose_normalization(w2cs)

    temporary = output_root.with_name(f".{output_root.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    try:
        (temporary / "images").mkdir(parents=True)
        frames = []
        records = []
        aggregate = hashlib.sha256()
        for ordinal, (name, image) in enumerate(selected):
            camera = cameras[image.camera_id]
            source = dataset_root / images / name
            pixels = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if pixels is None:
                raise FileNotFoundError(f"failed to load mapping image: {source}")
            if pixels.shape[:2] != (int(camera.height), int(camera.width)):
                raise ValueError(f"image/camera size mismatch for {source}")
            matrix, distortion = _camera_matrix(camera)
            undistorted = cv2.undistort(pixels, matrix, distortion, None, matrix)
            flat = _flat_name(name)
            destination = temporary / "images" / flat
            if not cv2.imwrite(str(destination), undistorted):
                raise OSError(f"failed to write Long-LRM input: {destination}")
            image_sha = _sha256(destination)
            aggregate.update(name.encode("utf8"))
            aggregate.update(bytes.fromhex(image_sha))
            frames.append(
                {
                    "file_path": f"images/{flat}",
                    "w2c": _w2c(image).tolist(),
                    "h": int(camera.height),
                    "w": int(camera.width),
                    "fx": float(matrix[0, 0]),
                    "fy": float(matrix[1, 1]),
                    "cx": float(matrix[0, 2]),
                    "cy": float(matrix[1, 2]),
                }
            )
            records.append(
                {
                    "ordinal": ordinal,
                    "source_name": name,
                    "sequence_id": Path(name).parts[0],
                    "prepared_name": flat,
                    "prepared_image_sha256": image_sha,
                }
            )
        camera_payload = {
            "scene_name": dataset_root.name,
            "frames": frames,
        }
        camera_json = temporary / "opencv_cameras.json"
        camera_json.write_text(json.dumps(camera_payload, indent=2))
        payload = {
            "schema": PREPARED_SCHEMA,
            "version": PREPARED_VERSION,
            "role": "mapping_only",
            "uses_test_queries": False,
            "dataset_root": str(dataset_root),
            "images_source": images,
            "mapping_split": str(dataset_root / "dataset_train.txt"),
            "mapping_split_sha256": _sha256(dataset_root / "dataset_train.txt"),
            "source_colmap_images": str(extrinsic_path),
            "source_colmap_images_sha256": _sha256(extrinsic_path),
            "source_colmap_cameras": str(intrinsic_path),
            "source_colmap_cameras_sha256": _sha256(intrinsic_path),
            "selection": "deterministic_pose_farthest_point_position_plus_direction",
            "view_count": view_count,
            "prepared_images_sha256": aggregate.hexdigest(),
            "camera_json": str(output_root / camera_json.name),
            "camera_json_sha256": _sha256(camera_json),
            "long_lrm_internal_pose_normalization": normalization,
            "records": records,
        }
        (temporary / "anygsloc_long_lrm_input.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True)
        )
        os.rename(temporary, output_root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return payload


def export_long_lrm_world_prior(
    source_ply: str | Path,
    prepared_manifest: str | Path,
    output_ply: str | Path,
    *,
    output_manifest: str | Path | None = None,
) -> dict:
    """Convert a Long-LRM standard PLY from its normalized frame to world."""

    source_ply = Path(source_ply).expanduser().resolve()
    prepared_manifest = Path(prepared_manifest).expanduser().resolve()
    output_ply = Path(output_ply).expanduser().resolve()
    if output_ply.exists():
        raise FileExistsError(f"refusing to overwrite prior: {output_ply}")
    prepared = json.loads(prepared_manifest.read_text())
    if prepared.get("schema") != PREPARED_SCHEMA or prepared.get("version") != 1:
        raise ValueError("invalid Long-LRM prepared-input manifest")
    if prepared.get("role") != "mapping_only" or prepared.get("uses_test_queries"):
        raise ValueError("Long-LRM prior must be mapping-only")
    camera_json = Path(prepared["camera_json"])
    if _sha256(camera_json) != prepared["camera_json_sha256"]:
        raise ValueError("Long-LRM camera JSON changed after preparation")
    camera_payload = json.loads(camera_json.read_text())
    w2cs = np.asarray([frame["w2c"] for frame in camera_payload["frames"]])
    recomputed = long_lrm_pose_normalization(w2cs)
    stored = prepared["long_lrm_internal_pose_normalization"]
    if not np.allclose(
        recomputed["world_to_normalized_sim3"],
        stored["world_to_normalized_sim3"],
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError("stored Long-LRM pose normalization is inconsistent")

    source = PlyData.read(source_ply)
    if len(source.elements) != 1 or source.elements[0].name != "vertex":
        raise ValueError("expected one vertex element in Long-LRM PLY")
    vertex = source.elements[0].data
    names = tuple(vertex.dtype.names or ())
    required = {
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    }
    missing = sorted(required - set(names))
    if missing:
        raise ValueError(f"Long-LRM PLY is missing fields: {missing}")
    rest_fields = _ordered_fields(names, "f_rest_")
    if len(rest_fields) % 3:
        raise ValueError("malformed Long-LRM SH fields")
    coefficient_count = len(rest_fields) // 3 + 1
    degree = math.isqrt(coefficient_count) - 1
    if (degree + 1) ** 2 != coefficient_count or degree > 3:
        raise ValueError("unsupported Long-LRM SH degree")
    rows = int(vertex.shape[0])
    xyz_normalized = np.stack(
        [vertex[name] for name in ("x", "y", "z")], axis=1
    ).astype(np.float64)
    coefficients = np.empty((rows, coefficient_count, 3), dtype=np.float64)
    coefficients[:, 0] = np.stack(
        [vertex[f"f_dc_{index}"] for index in range(3)], axis=1
    )
    if rest_fields:
        raw_rest = np.stack([vertex[name] for name in rest_fields], axis=1)
        coefficients[:, 1:] = raw_rest.reshape(
            rows, 3, coefficient_count - 1
        ).transpose(0, 2, 1)

    rigid = np.asarray(stored["rigid_world_to_normalized"], dtype=np.float64)
    scene_scale = float(stored["scene_scale"])
    if rigid.shape != (4, 4) or not np.isfinite(scene_scale) or scene_scale <= 0:
        raise ValueError("invalid Long-LRM normalization")
    rotation_world_to_normalized = rigid[:3, :3]
    rotation_normalized_to_world = rotation_world_to_normalized.T
    translation = rigid[:3, 3]
    xyz_world = (
        xyz_normalized / scene_scale - translation[None]
    ) @ rotation_world_to_normalized

    sh_rotation = _sh_world_rotation(rotation_world_to_normalized, degree)
    coefficients_world = np.einsum(
        "ij,njc->nic", sh_rotation, coefficients, optimize=True
    )
    raw_quaternions = np.stack(
        [vertex[f"rot_{index}"] for index in range(4)], axis=1
    ).astype(np.float64)
    norms = np.linalg.norm(raw_quaternions, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-9):
        raise ValueError("Long-LRM PLY contains invalid rotations")
    normalized_xyzw = (raw_quaternions / norms[:, None])[:, [1, 2, 3, 0]]
    normalized_rotations = Rotation.from_quat(normalized_xyzw).as_matrix()
    world_rotations = rotation_normalized_to_world[None] @ normalized_rotations
    world_xyzw = Rotation.from_matrix(world_rotations).as_quat()
    world_wxyz = world_xyzw[:, [3, 0, 1, 2]]
    scales = np.stack(
        [vertex[f"scale_{index}"] for index in range(3)], axis=1
    ).astype(np.float64)
    scales -= math.log(scene_scale)
    opacity = np.asarray(vertex["opacity"], dtype=np.float64)
    finite = (
        np.isfinite(xyz_world).all(axis=1)
        & np.isfinite(coefficients_world).all(axis=(1, 2))
        & np.isfinite(world_wxyz).all(axis=1)
        & np.isfinite(scales).all(axis=1)
        & np.isfinite(opacity)
    )
    if not finite.any():
        raise ValueError("all exported Long-LRM Gaussians are non-finite")
    dropped = int((~finite).sum())
    xyz_world = xyz_world[finite]
    coefficients_world = coefficients_world[finite]
    world_wxyz = world_wxyz[finite]
    scales = scales[finite]
    opacity = opacity[finite]
    rows = int(finite.sum())

    fields = ["x", "y", "z", "nx", "ny", "nz"]
    fields += [f"f_dc_{index}" for index in range(3)]
    fields += [f"f_rest_{index}" for index in range(3 * (coefficient_count - 1))]
    fields += ["opacity", "scale_0", "scale_1", "scale_2"]
    fields += [f"rot_{index}" for index in range(4)]
    output = np.empty(rows, dtype=[(name, "<f4") for name in fields])
    for index, name in enumerate(("x", "y", "z")):
        output[name] = xyz_world[:, index]
    for name in ("nx", "ny", "nz"):
        output[name] = 0.0
    for index in range(3):
        output[f"f_dc_{index}"] = coefficients_world[:, 0, index]
    standard_rest = coefficients_world[:, 1:].transpose(0, 2, 1).reshape(rows, -1)
    for index in range(standard_rest.shape[1]):
        output[f"f_rest_{index}"] = standard_rest[:, index]
    output["opacity"] = opacity
    for index in range(3):
        output[f"scale_{index}"] = scales[:, index]
    for index in range(4):
        output[f"rot_{index}"] = world_wxyz[:, index]

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_ply.with_name(f".{output_ply.name}.tmp-{os.getpid()}")
    PlyData([PlyElement.describe(output, "vertex")], text=False).write(temporary)
    os.link(temporary, output_ply)
    temporary.unlink()
    payload = {
        "schema": EXPORT_SCHEMA,
        "version": EXPORT_VERSION,
        "source_method": "long_lrm_32view",
        "coordinate_frame": "mapping_world",
        "mapping_only": True,
        "source_ply": str(source_ply),
        "source_ply_sha256": _sha256(source_ply),
        "prepared_manifest": str(prepared_manifest),
        "prepared_manifest_sha256": _sha256(prepared_manifest),
        "source_primitive_count": int(vertex.shape[0]),
        "dropped_nonfinite_primitive_count": dropped,
        "primitive_count": rows,
        "sh_degree": degree,
        "sh_rotated_to_mapping_world": True,
        "output_ply": str(output_ply),
        "output_ply_sha256": _sha256(output_ply),
    }
    if output_manifest is not None:
        output_manifest = Path(output_manifest).expanduser().resolve()
        if output_manifest.exists():
            raise FileExistsError(f"refusing to overwrite export manifest: {output_manifest}")
        temporary_manifest = output_manifest.with_name(
            f".{output_manifest.name}.tmp-{os.getpid()}"
        )
        temporary_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.link(temporary_manifest, output_manifest)
        temporary_manifest.unlink()
    return payload
