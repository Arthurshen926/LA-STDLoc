#!/usr/bin/env python3
"""Stage a mapping-only Cambridge COLMAP model for off-the-shelf RGB GS."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import struct
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData, PlyElement


Camera = collections.namedtuple("Camera", "id model width height params")
Image = collections.namedtuple(
    "Image", "id qvec tvec camera_id name xys point3D_ids"
)
Point3D = collections.namedtuple(
    "Point3D", "id xyz rgb error image_ids point2D_idxs"
)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
CAMERA_MODEL_IDS = {name: model_id for model_id, (name, _) in CAMERA_MODELS.items()}


def _unpack(handle, size: int, fmt: str):
    return struct.unpack("<" + fmt, handle.read(size))


def _pack(handle, values, fmt: str) -> None:
    if not isinstance(values, (tuple, list)):
        values = (values,)
    handle.write(struct.pack("<" + fmt, *values))


def read_cameras_binary(path: Path) -> dict[int, Camera]:
    cameras = {}
    with path.open("rb") as handle:
        count = _unpack(handle, 8, "Q")[0]
        for _ in range(count):
            camera_id, model_id, width, height = _unpack(handle, 24, "iiQQ")
            model, parameter_count = CAMERA_MODELS[model_id]
            params = np.asarray(
                _unpack(handle, 8 * parameter_count, "d" * parameter_count)
            )
            cameras[camera_id] = Camera(
                camera_id, model, width, height, params
            )
    return cameras


def read_images_binary(path: Path) -> dict[int, Image]:
    images = {}
    with path.open("rb") as handle:
        count = _unpack(handle, 8, "Q")[0]
        for _ in range(count):
            values = _unpack(handle, 64, "idddddddi")
            image_id = int(values[0])
            name_bytes = bytearray()
            while True:
                value = handle.read(1)
                if value == b"\x00":
                    break
                name_bytes.extend(value)
            point_count = _unpack(handle, 8, "Q")[0]
            triples = _unpack(handle, 24 * point_count, "ddq" * point_count)
            xys = np.column_stack((triples[0::3], triples[1::3]))
            point_ids = np.asarray(triples[2::3], dtype=np.int64)
            images[image_id] = Image(
                image_id,
                np.asarray(values[1:5]),
                np.asarray(values[5:8]),
                int(values[8]),
                name_bytes.decode("utf-8"),
                xys,
                point_ids,
            )
    return images


def read_points3d_binary(path: Path) -> dict[int, Point3D]:
    points = {}
    with path.open("rb") as handle:
        count = _unpack(handle, 8, "Q")[0]
        for _ in range(count):
            values = _unpack(handle, 43, "QdddBBBd")
            track_count = _unpack(handle, 8, "Q")[0]
            track = _unpack(handle, 8 * track_count, "ii" * track_count)
            point_id = int(values[0])
            points[point_id] = Point3D(
                point_id,
                np.asarray(values[1:4]),
                np.asarray(values[4:7], dtype=np.uint8),
                float(values[7]),
                np.asarray(track[0::2], dtype=np.int32),
                np.asarray(track[1::2], dtype=np.int32),
            )
    return points


def write_cameras_binary(cameras: dict[int, Camera], path: Path) -> None:
    with path.open("wb") as handle:
        _pack(handle, len(cameras), "Q")
        for camera_id in sorted(cameras):
            camera = cameras[camera_id]
            model_id = CAMERA_MODEL_IDS[camera.model]
            _pack(
                handle,
                (camera.id, model_id, camera.width, camera.height),
                "iiQQ",
            )
            _pack(handle, tuple(camera.params), "d" * len(camera.params))


def write_images_binary(images: dict[int, Image], path: Path) -> None:
    with path.open("wb") as handle:
        _pack(handle, len(images), "Q")
        for image_id in sorted(images):
            image = images[image_id]
            _pack(
                handle,
                (
                    image.id,
                    *image.qvec.tolist(),
                    *image.tvec.tolist(),
                    image.camera_id,
                ),
                "idddddddi",
            )
            handle.write(image.name.encode("utf-8") + b"\x00")
            _pack(handle, len(image.xys), "Q")
            for xy, point_id in zip(image.xys, image.point3D_ids):
                _pack(handle, (float(xy[0]), float(xy[1]), int(point_id)), "ddq")


def write_points3d_binary(points: dict[int, Point3D], path: Path) -> None:
    with path.open("wb") as handle:
        _pack(handle, len(points), "Q")
        for point_id in sorted(points):
            point = points[point_id]
            _pack(
                handle,
                (
                    point.id,
                    *point.xyz.tolist(),
                    *point.rgb.tolist(),
                    point.error,
                ),
                "QdddBBBd",
            )
            _pack(handle, len(point.image_ids), "Q")
            for image_id, point2d_idx in zip(
                point.image_ids, point.point2D_idxs
            ):
                _pack(handle, (int(image_id), int(point2d_idx)), "ii")


def write_points3d_ply(points: dict[int, Point3D], path: Path) -> None:
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    values = np.empty(len(points), dtype=dtype)
    for row, point_id in enumerate(sorted(points)):
        point = points[point_id]
        values[row] = (*point.xyz, 0.0, 0.0, 0.0, *point.rgb)
    PlyData([PlyElement.describe(values, "vertex")]).write(path)


def read_cambridge_names(path: Path) -> list[str]:
    names = []
    for line in path.read_text().splitlines():
        value = line.strip()
        if not value or value.startswith("Visual Landmark") or value.startswith(
            "ImageFile"
        ):
            continue
        name = value.split()[0]
        if "/" in name:
            names.append(name)
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate Cambridge names in {path}")
    return names


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _opencv_camera(camera: Camera) -> tuple[np.ndarray, np.ndarray, Camera]:
    """Convert a supported COLMAP camera to a fixed-size PINHOLE target."""
    params = np.asarray(camera.params, dtype=np.float64)
    if camera.model == "SIMPLE_PINHOLE":
        focal, cx, cy = params
        fx = fy = focal
        distortion = np.zeros(5, dtype=np.float64)
    elif camera.model == "PINHOLE":
        fx, fy, cx, cy = params
        distortion = np.zeros(5, dtype=np.float64)
    elif camera.model == "SIMPLE_RADIAL":
        focal, cx, cy, k1 = params
        fx = fy = focal
        distortion = np.asarray([k1, 0.0, 0.0, 0.0, 0.0])
    elif camera.model == "RADIAL":
        focal, cx, cy, k1, k2 = params
        fx = fy = focal
        distortion = np.asarray([k1, k2, 0.0, 0.0, 0.0])
    elif camera.model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params
        distortion = np.asarray([k1, k2, p1, p2, 0.0])
    else:
        raise ValueError(
            "OpenCV fixed-intrinsics undistortion does not support COLMAP "
            f"camera model {camera.model}"
        )
    intrinsic = np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    target = Camera(
        camera.id,
        "PINHOLE",
        camera.width,
        camera.height,
        np.asarray([fx, fy, cx, cy], dtype=np.float64),
    )
    return intrinsic, distortion, target


def _undistort_image_and_observations(
    *, source: Path, destination: Path, image: Image, camera: Camera
) -> Image:
    intrinsic, distortion, _ = _opencv_camera(camera)
    pixels = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if pixels is None:
        raise ValueError(f"failed to read mapping RGB image: {source}")
    if pixels.shape[1] != camera.width or pixels.shape[0] != camera.height:
        raise ValueError(
            f"image/camera size mismatch for {source}: "
            f"{pixels.shape[1]}x{pixels.shape[0]} versus "
            f"{camera.width}x{camera.height}"
        )
    undistorted = cv2.undistort(
        pixels, intrinsic, distortion, None, intrinsic
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), undistorted):
        raise OSError(f"failed to write undistorted image: {destination}")
    xys = np.asarray(image.xys, dtype=np.float64)
    if len(xys):
        xys = cv2.undistortPoints(
            xys.reshape(-1, 1, 2),
            intrinsic,
            distortion,
            P=intrinsic,
        ).reshape(-1, 2)
    return image._replace(xys=xys)


def stage_mapping_only_colmap(
    *,
    source: Path,
    output: Path,
    images_dir: str,
    minimum_track_length: int,
    undistort_images: bool = False,
) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite mapping-only dataset: {output}")
    sparse = source / "sparse" / "0"
    cameras = read_cameras_binary(sparse / "cameras.bin")
    images = read_images_binary(sparse / "images.bin")
    points = read_points3d_binary(sparse / "points3D.bin")
    mapping_names = read_cambridge_names(source / "dataset_train.txt")
    test_names = set(read_cambridge_names(source / "dataset_test.txt"))
    image_by_name = {image.name: image for image in images.values()}
    missing = sorted(set(mapping_names).difference(image_by_name))
    if missing:
        raise ValueError(f"mapping image absent from COLMAP model: {missing[0]}")
    overlap = set(mapping_names).intersection(test_names)
    if overlap:
        raise ValueError(f"mapping/test split overlap: {sorted(overlap)[0]}")

    mapping_ids = {image_by_name[name].id for name in mapping_names}
    retained_points = {}
    dropped_test_observed = 0
    dropped_short = 0
    for point_id, point in points.items():
        track_ids = set(map(int, point.image_ids.tolist()))
        if not track_ids.issubset(mapping_ids):
            dropped_test_observed += 1
            continue
        if len(track_ids) < int(minimum_track_length):
            dropped_short += 1
            continue
        retained_points[point_id] = point
    retained_ids = set(retained_points)
    retained_images = {}
    for name in mapping_names:
        image = image_by_name[name]
        point_ids = np.asarray(
            [value if int(value) in retained_ids else -1 for value in image.point3D_ids],
            dtype=np.int64,
        )
        retained_images[image.id] = image._replace(point3D_ids=point_ids)
    camera_ids = {image.camera_id for image in retained_images.values()}
    retained_cameras = {camera_id: cameras[camera_id] for camera_id in camera_ids}
    source_camera_models = sorted(
        {camera.model for camera in retained_cameras.values()}
    )
    if undistort_images:
        target_cameras = {
            camera_id: _opencv_camera(camera)[2]
            for camera_id, camera in retained_cameras.items()
        }
    else:
        target_cameras = retained_cameras

    staged_sparse = output / "sparse" / "0"
    staged_images = output / "images"
    staged_sparse.mkdir(parents=True)
    staged_images.mkdir(parents=True)
    image_root = source / images_dir
    for name in mapping_names:
        source_image = (image_root / name).resolve()
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        destination = staged_images / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        image = retained_images[image_by_name[name].id]
        if undistort_images:
            retained_images[image.id] = _undistort_image_and_observations(
                source=source_image,
                destination=destination,
                image=image,
                camera=retained_cameras[image.camera_id],
            )
        else:
            os.symlink(source_image, destination)
    write_cameras_binary(target_cameras, staged_sparse / "cameras.bin")
    write_images_binary(retained_images, staged_sparse / "images.bin")
    write_points3d_binary(retained_points, staged_sparse / "points3D.bin")
    write_points3d_ply(retained_points, staged_sparse / "points3D.ply")
    (output / "mapping_names.txt").write_text("\n".join(mapping_names) + "\n")

    manifest = {
        "schema": "lafgs_off_the_shelf_mapping_only_colmap",
        "version": 1,
        "source": str(source.resolve()),
        "images_source": str(image_root.resolve()),
        "image_policy": "Cambridge mapping split only; no test RGB",
        "camera_policy": (
            "OpenCV fixed-intrinsics undistortion to PINHOLE"
            if undistort_images
            else "retain source COLMAP camera model"
        ),
        "source_camera_models": source_camera_models,
        "target_camera_models": sorted(
            {camera.model for camera in target_cameras.values()}
        ),
        "undistortion_used": bool(undistort_images),
        "point_policy": (
            "retain only SfM points whose complete observation track is in the "
            "mapping split"
        ),
        "semantic_mask_used": False,
        "mapping_image_count": len(retained_images),
        "excluded_test_image_count": len(test_names),
        "retained_camera_count": len(target_cameras),
        "input_point_count": len(points),
        "retained_point_count": len(retained_points),
        "dropped_test_observed_point_count": dropped_test_observed,
        "dropped_short_track_point_count": dropped_short,
        "minimum_track_length": int(minimum_track_length),
        "source_hashes": {
            name: sha256(sparse / name)
            for name in ("cameras.bin", "images.bin", "points3D.bin")
        },
    }
    (output / "mapping_only_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images-dir", default="processed")
    parser.add_argument("--minimum-track-length", type=int, default=2)
    parser.add_argument(
        "--undistort-images",
        action="store_true",
        help=(
            "Undistort RGB and sparse observations with fixed intrinsics and "
            "emit PINHOLE cameras for official Gaussian implementations."
        ),
    )
    args = parser.parse_args()
    manifest = stage_mapping_only_colmap(
        source=args.source,
        output=args.output,
        images_dir=args.images_dir,
        minimum_track_length=args.minimum_track_length,
        undistort_images=args.undistort_images,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
