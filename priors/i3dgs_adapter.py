"""Strict adapters between Cambridge mapping data, i3DGS, and AnyGSLoc.

i3DGS consumes a flat image directory and internally normalizes COLMAP poses.
Cambridge instead stores repeated basenames below sequence directories.  This
module prepares a mapping-only, collision-free input set and exports the leaf
Gaussians back into the original Cambridge world frame.
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

from data.colmap import (
    Image,
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)


PREPARED_SCHEMA = "anygsloc_i3dgs_prepared_mapping"
PREPARED_VERSION = 1
EXPORT_SCHEMA = "anygsloc_i3dgs_world_prior"
EXPORT_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_mapping_names(dataset_root: Path) -> list[str]:
    split = dataset_root / "dataset_train.txt"
    if not split.is_file():
        raise FileNotFoundError(f"missing Cambridge mapping split: {split}")
    names = []
    for line in split.read_text().splitlines():
        fields = line.split()
        if len(fields) < 8 or not fields[0].lower().endswith(
            (".png", ".jpg", ".jpeg", ".webp")
        ):
            continue
        try:
            tuple(float(value) for value in fields[1:8])
        except ValueError:
            continue
        names.append(fields[0])
    if not names or len(names) != len(set(names)):
        raise ValueError("mapping split must contain unique pose-bearing image rows")
    return names


def _read_colmap_model(dataset_root: Path):
    sparse = dataset_root / "sparse" / "0"
    if (sparse / "images.bin").is_file() and (sparse / "cameras.bin").is_file():
        return (
            read_extrinsics_binary(str(sparse / "images.bin")),
            read_intrinsics_binary(str(sparse / "cameras.bin")),
            sparse / "images.bin",
            sparse / "cameras.bin",
        )
    return (
        read_extrinsics_text(str(sparse / "images.txt")),
        read_intrinsics_text(str(sparse / "cameras.txt")),
        sparse / "images.txt",
        sparse / "cameras.txt",
    )


def _flat_name(image_name: str) -> str:
    path = Path(image_name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe image name: {image_name!r}")
    return "__".join(path.parts)


def _camera_matrix(camera) -> tuple[np.ndarray, np.ndarray]:
    params = np.asarray(camera.params, dtype=np.float64)
    if camera.model == "SIMPLE_PINHOLE":
        fx = fy = params[0]
        cx, cy = params[1:3]
        distortion = np.zeros(5, dtype=np.float64)
    elif camera.model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
        distortion = np.zeros(5, dtype=np.float64)
    elif camera.model == "SIMPLE_RADIAL":
        fx = fy = params[0]
        cx, cy, k1 = params[1:4]
        distortion = np.asarray([k1, 0.0, 0.0, 0.0, 0.0])
    elif camera.model == "RADIAL":
        fx = fy = params[0]
        cx, cy, k1, k2 = params[1:5]
        distortion = np.asarray([k1, k2, 0.0, 0.0, 0.0])
    elif camera.model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
        distortion = np.asarray([k1, k2, p1, p2, 0.0])
    else:
        raise ValueError(f"unsupported camera model for i3DGS: {camera.model}")
    matrix = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return matrix, distortion


def _write_prepared_image(
    source: Path,
    destination: Path,
    camera,
    downscale: float,
) -> tuple[int, int, np.ndarray]:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to load mapping image: {source}")
    if image.shape[1] != int(camera.width) or image.shape[0] != int(camera.height):
        raise ValueError(f"image/camera size mismatch for {source}")
    matrix, distortion = _camera_matrix(camera)
    width = int(round(camera.width / downscale))
    height = int(round(camera.height / downscale))
    if width < 16 or height < 16:
        raise ValueError("prepared image dimensions are unreasonably small")
    target_matrix = matrix.copy()
    target_matrix[0] /= downscale
    target_matrix[1] /= downscale
    map_x, map_y = cv2.initUndistortRectifyMap(
        matrix,
        distortion,
        None,
        target_matrix,
        (width, height),
        cv2.CV_32FC1,
    )
    prepared = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), prepared):
        raise OSError(f"failed to write prepared image: {destination}")
    return width, height, target_matrix


def _normalization_from_ordered_images(images: list[Image]) -> dict:
    if len(images) < 6:
        raise ValueError("i3DGS pose normalization requires at least six images")
    extrinsics = []
    centers = []
    for image in images[:6]:
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = qvec2rotmat(image.qvec)
        w2c[:3, 3] = image.tvec
        extrinsics.append(w2c)
        centers.append(np.linalg.inv(w2c)[:3, 3])
    centers = np.stack(centers)
    mean_step = np.linalg.norm(centers[:-1] - centers[1:], axis=1).mean()
    if not np.isfinite(mean_step) or mean_step <= 1e-9:
        raise ValueError("first six mapping images do not define a valid scale")
    scale = 0.1 / mean_step
    first_w2c = extrinsics[0]
    world_to_i3dgs = np.eye(4, dtype=np.float64)
    world_to_i3dgs[:3, :3] = scale * first_w2c[:3, :3]
    world_to_i3dgs[:3, 3] = scale * first_w2c[:3, 3]
    i3dgs_to_world = np.linalg.inv(world_to_i3dgs)
    return {
        "scale": float(scale),
        "first_mapping_w2c": first_w2c.tolist(),
        "world_to_i3dgs": world_to_i3dgs.tolist(),
        "i3dgs_to_world": i3dgs_to_world.tolist(),
    }


def prepare_cambridge_mapping(
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    images: str = "processed",
    downscale: float = 2.0,
) -> dict:
    """Materialize an undistorted, flat, mapping-only i3DGS dataset."""

    dataset_root = Path(dataset_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite prepared dataset: {output_root}")
    if not np.isfinite(downscale) or downscale < 1.0:
        raise ValueError("downscale must be finite and at least one")
    mapping_names = _read_mapping_names(dataset_root)
    extrinsics, cameras, extrinsic_path, intrinsic_path = _read_colmap_model(
        dataset_root
    )
    by_name = {image.name: image for image in extrinsics.values()}
    missing = sorted(set(mapping_names) - set(by_name))
    if missing:
        raise ValueError(f"mapping images missing from COLMAP model: {missing[:3]}")

    records = []
    flat_names = [_flat_name(name) for name in mapping_names]
    if len(flat_names) != len(set(flat_names)):
        raise ValueError("flattened i3DGS image names collide")
    ordered_names = sorted(mapping_names, key=lambda name: _flat_name(name))
    ordered_images = [by_name[name] for name in ordered_names]
    normalization = _normalization_from_ordered_images(ordered_images)

    temporary = output_root.with_name(f".{output_root.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    try:
        (temporary / "images").mkdir(parents=True)
        (temporary / "sparse" / "0").mkdir(parents=True)
        camera_lines = [
            "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
            f"# Number of cameras: {len(mapping_names)}",
        ]
        image_lines = [
            "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
            f"# Number of images: {len(mapping_names)}",
        ]
        aggregate = hashlib.sha256()
        for ordinal, name in enumerate(ordered_names):
            image = by_name[name]
            camera = cameras[image.camera_id]
            flat = _flat_name(name)
            source = dataset_root / images / name
            destination = temporary / "images" / flat
            width, height, target_matrix = _write_prepared_image(
                source, destination, camera, downscale
            )
            camera_id = ordinal + 1
            image_id = ordinal + 1
            fx = float(target_matrix[0, 0])
            fy = float(target_matrix[1, 1])
            cx = float(target_matrix[0, 2])
            cy = float(target_matrix[1, 2])
            camera_lines.append(
                f"{camera_id} PINHOLE {width} {height} "
                f"{fx:.17g} {fy:.17g} {cx:.17g} {cy:.17g}"
            )
            pose_values = [*image.qvec.tolist(), *image.tvec.tolist()]
            image_lines.append(
                " ".join(
                    [str(image_id)]
                    + [f"{float(value):.17g}" for value in pose_values]
                    + [str(camera_id), flat]
                )
            )
            image_lines.append("")
            image_sha = _sha256(destination)
            aggregate.update(flat.encode("utf8"))
            aggregate.update(bytes.fromhex(image_sha))
            records.append(
                {
                    "ordinal": ordinal,
                    "source_name": name,
                    "prepared_name": flat,
                    "source_camera_id": int(image.camera_id),
                    "prepared_camera_id": camera_id,
                    "prepared_image_sha256": image_sha,
                }
            )

        sparse = temporary / "sparse" / "0"
        (sparse / "cameras.txt").write_text("\n".join(camera_lines) + "\n")
        (sparse / "images.txt").write_text("\n".join(image_lines) + "\n")
        (sparse / "points3D.txt").write_text("# Number of points: 0\n")
        payload = {
            "schema": PREPARED_SCHEMA,
            "version": PREPARED_VERSION,
            "role": "mapping_only",
            "dataset_root": str(dataset_root),
            "images_source": images,
            "mapping_split": str(dataset_root / "dataset_train.txt"),
            "mapping_split_sha256": _sha256(dataset_root / "dataset_train.txt"),
            "source_colmap_images": str(extrinsic_path),
            "source_colmap_images_sha256": _sha256(extrinsic_path),
            "source_colmap_cameras": str(intrinsic_path),
            "source_colmap_cameras_sha256": _sha256(intrinsic_path),
            "downscale": float(downscale),
            "undistortion": "opencv_same_intrinsics_then_downscale",
            "image_count": len(records),
            "prepared_images_sha256": aggregate.hexdigest(),
            "i3dgs_internal_pose_normalization": normalization,
            "records": records,
        }
        manifest = temporary / "anygsloc_i3dgs_input.json"
        manifest.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.rename(temporary, output_root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return payload


def _real_sh_basis(directions: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate the real SH basis used by GraphDeco/gsplat (degree <= 3)."""

    if degree < 0 or degree > 3:
        raise ValueError("only SH degrees zero through three are supported")
    directions = np.asarray(directions, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("directions must have shape [N,3]")
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    x, y, z = directions.T
    basis = [np.full_like(x, 0.28209479177387814)]
    if degree >= 1:
        c1 = 0.4886025119029199
        basis.extend((-c1 * y, c1 * z, -c1 * x))
    if degree >= 2:
        c2 = (
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396,
        )
        basis.extend(
            (
                c2[0] * x * y,
                c2[1] * y * z,
                c2[2] * (2.0 * z * z - x * x - y * y),
                c2[3] * x * z,
                c2[4] * (x * x - y * y),
            )
        )
    if degree >= 3:
        c3 = (
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435,
        )
        basis.extend(
            (
                c3[0] * y * (3.0 * x * x - y * y),
                c3[1] * x * y * z,
                c3[2] * y * (4.0 * z * z - x * x - y * y),
                c3[3] * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y),
                c3[4] * x * (4.0 * z * z - x * x - y * y),
                c3[5] * z * (x * x - y * y),
                c3[6] * x * (x * x - 3.0 * y * y),
            )
        )
    return np.stack(basis, axis=1)


def _sh_world_rotation(first_w2c_rotation: np.ndarray, degree: int) -> np.ndarray:
    count = max(64, 8 * (degree + 1) ** 2)
    indices = np.arange(count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * indices / count
    phi = math.pi * (1.0 + math.sqrt(5.0)) * indices
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    world = np.stack((radius * np.cos(phi), radius * np.sin(phi), z), axis=1)
    local = world @ np.asarray(first_w2c_rotation, dtype=np.float64).T
    world_basis = _real_sh_basis(world, degree)
    local_basis = _real_sh_basis(local, degree)
    transform, *_ = np.linalg.lstsq(world_basis, local_basis, rcond=None)
    return transform


def _ordered_fields(names: tuple[str, ...], prefix: str) -> list[str]:
    return sorted(
        (name for name in names if name.startswith(prefix)),
        key=lambda name: int(name[len(prefix) :]),
    )


def export_i3dgs_world_prior(
    hierarchy_ply: str | Path,
    prepared_manifest: str | Path,
    output_ply: str | Path,
    *,
    output_manifest: str | Path | None = None,
) -> dict:
    """Export i3DGS leaves as a standard world-frame 3DGS PLY."""

    hierarchy_ply = Path(hierarchy_ply).expanduser().resolve()
    prepared_manifest = Path(prepared_manifest).expanduser().resolve()
    output_ply = Path(output_ply).expanduser().resolve()
    if output_ply.exists():
        raise FileExistsError(f"refusing to overwrite prior: {output_ply}")
    prepared = json.loads(prepared_manifest.read_text())
    if prepared.get("schema") != PREPARED_SCHEMA or prepared.get("version") != 1:
        raise ValueError("invalid i3DGS prepared-input manifest")
    source = PlyData.read(hierarchy_ply)
    if len(source.elements) != 1 or source.elements[0].name != "vertex":
        raise ValueError("expected one vertex element in i3DGS hierarchy PLY")
    vertex = source.elements[0].data
    names = tuple(vertex.dtype.names or ())
    required = {
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }
    missing = sorted(required - set(names))
    if missing:
        raise ValueError(f"i3DGS PLY is missing fields: {missing}")
    child_fields = _ordered_fields(names, "child_")
    if child_fields:
        children = np.stack([vertex[name] for name in child_fields], axis=1)
        leaf_mask = np.all(children == -1, axis=1)
        source_layout = "coefficient_major"
    else:
        leaf_mask = np.ones(vertex.shape[0], dtype=bool)
        source_layout = "channel_major"
    if not leaf_mask.any():
        raise ValueError("i3DGS hierarchy contains no leaf Gaussians")

    rest_fields = _ordered_fields(names, "f_rest_")
    coefficient_count = len(rest_fields) // 3 + 1
    degree = math.isqrt(coefficient_count) - 1
    if (degree + 1) ** 2 != coefficient_count or degree > 3:
        raise ValueError("unsupported or malformed i3DGS SH fields")
    rows = int(leaf_mask.sum())
    xyz_local = np.stack([vertex[name][leaf_mask] for name in ("x", "y", "z")], axis=1).astype(np.float64)
    dc = np.stack([vertex[f"f_dc_{index}"][leaf_mask] for index in range(3)], axis=1).astype(np.float64)
    raw_rest = np.stack([vertex[name][leaf_mask] for name in rest_fields], axis=1).astype(np.float64)
    if source_layout == "coefficient_major":
        rest = raw_rest.reshape(rows, coefficient_count - 1, 3)
    else:
        rest = raw_rest.reshape(rows, 3, coefficient_count - 1).transpose(0, 2, 1)
    coefficients = np.concatenate((dc[:, None, :], rest), axis=1)

    normalization = prepared["i3dgs_internal_pose_normalization"]
    scale = float(normalization["scale"])
    first_w2c = np.asarray(normalization["first_mapping_w2c"], dtype=np.float64)
    if not np.isfinite(scale) or scale <= 0.0 or first_w2c.shape != (4, 4):
        raise ValueError("invalid i3DGS pose normalization")
    rotation_local_to_world = first_w2c[:3, :3].T
    camera_center = -rotation_local_to_world @ first_w2c[:3, 3]
    xyz_world = xyz_local @ rotation_local_to_world.T / scale + camera_center

    sh_rotation = _sh_world_rotation(first_w2c[:3, :3], degree)
    coefficients_world = np.einsum(
        "ij,njc->nic", sh_rotation, coefficients, optimize=True
    )

    raw_quaternions = np.stack(
        [vertex[f"rot_{index}"][leaf_mask] for index in range(4)], axis=1
    ).astype(np.float64)
    norms = np.linalg.norm(raw_quaternions, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-9):
        raise ValueError("i3DGS PLY contains invalid rotations")
    local_xyzw = (raw_quaternions / norms[:, None])[:, [1, 2, 3, 0]]
    local_rotations = Rotation.from_quat(local_xyzw).as_matrix()
    world_rotations = rotation_local_to_world[None] @ local_rotations
    world_xyzw = Rotation.from_matrix(world_rotations).as_quat()
    world_wxyz = world_xyzw[:, [3, 0, 1, 2]]

    scales = np.stack(
        [vertex[f"scale_{index}"][leaf_mask] for index in range(3)], axis=1
    ).astype(np.float64)
    scales -= math.log(scale)
    opacity = np.asarray(vertex["opacity"][leaf_mask], dtype=np.float64)
    finite = (
        np.isfinite(xyz_world).all(axis=1)
        & np.isfinite(coefficients_world).all(axis=(1, 2))
        & np.isfinite(world_wxyz).all(axis=1)
        & np.isfinite(scales).all(axis=1)
        & np.isfinite(opacity)
    )
    if not finite.any():
        raise ValueError("all exported leaf Gaussians are non-finite")
    dropped_nonfinite = int((~finite).sum())
    xyz_world = xyz_world[finite]
    coefficients_world = coefficients_world[finite]
    world_wxyz = world_wxyz[finite]
    scales = scales[finite]
    opacity = opacity[finite]
    rows = int(finite.sum())

    output_fields = ["x", "y", "z", "nx", "ny", "nz"]
    output_fields += [f"f_dc_{index}" for index in range(3)]
    output_fields += [f"f_rest_{index}" for index in range(3 * (coefficient_count - 1))]
    output_fields += ["opacity"]
    output_fields += [f"scale_{index}" for index in range(3)]
    output_fields += [f"rot_{index}" for index in range(4)]
    output = np.empty(rows, dtype=[(name, "<f4") for name in output_fields])
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
        "source_method": "i3dgs",
        "coordinate_frame": "mapping_world",
        "source_hierarchy_ply": str(hierarchy_ply),
        "source_hierarchy_ply_sha256": _sha256(hierarchy_ply),
        "prepared_manifest": str(prepared_manifest),
        "prepared_manifest_sha256": _sha256(prepared_manifest),
        "source_vertex_count": int(vertex.shape[0]),
        "source_leaf_count": int(leaf_mask.sum()),
        "dropped_nonfinite_leaf_count": dropped_nonfinite,
        "primitive_count": rows,
        "sh_degree": degree,
        "source_sh_layout": source_layout,
        "sh_rotated_to_mapping_world": True,
        "output_ply": str(output_ply),
        "output_ply_sha256": _sha256(output_ply),
    }
    if output_manifest is not None:
        output_manifest = Path(output_manifest).expanduser().resolve()
        if output_manifest.exists():
            raise FileExistsError(f"refusing to overwrite export manifest: {output_manifest}")
        output_manifest.parent.mkdir(parents=True, exist_ok=True)
        temporary_manifest = output_manifest.with_name(
            f".{output_manifest.name}.tmp-{os.getpid()}"
        )
        temporary_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.link(temporary_manifest, output_manifest)
        temporary_manifest.unlink()
    return payload
