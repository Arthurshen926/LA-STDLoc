"""Prepare official indoor relocalization datasets for LaFGS.

The output is a COLMAP text model with ground-truth camera poses and an explicit
mapping/test split.  Published reference models retain their full SfM point
cloud by default, matching the Gaussian-initialization contract used by STDLoc
and ULF-Loc, while Gaussian RGB supervision remains mapping-only.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import pickle
import struct
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
import re

import cv2
import numpy as np
from PIL import Image

from data.colmap import (
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    rotmat2qvec,
)


_SEVEN_SCENES = (
    "chess",
    "fire",
    "heads",
    "office",
    "pumpkin",
    "redkitchen",
    "stairs",
)

_DEPTH_TO_RGB = np.array(
    [
        [
            9.9996518012567637e-01,
            2.6765126468950343e-03,
            -7.9041012313000904e-03,
            -2.5558943178152542e-02,
        ],
        [
            -2.7409311281316700e-03,
            9.9996302803027592e-01,
            -8.1504520778013286e-03,
            1.0109636268061706e-04,
        ],
        [
            7.8819942130445332e-03,
            8.1718328771890631e-03,
            9.9993554558014031e-01,
            2.0318321729487039e-03,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class PreparedFrame:
    image_name: str
    source_image: Path
    pose_c2w: np.ndarray
    is_test: bool


@dataclass(frozen=True)
class CameraRectification:
    """A COLMAP-calibrated image remap into the canonical pinhole domain."""

    pinhole: tuple[float, float, float, float]
    remap_x: np.ndarray | None
    remap_y: np.ndarray | None
    valid_mask: np.ndarray
    source_model: str
    source_parameters: tuple[float, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_pose(path: Path) -> np.ndarray:
    pose = np.loadtxt(path, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"Invalid camera-to-world pose: {path}")
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"Pose has an invalid homogeneous row: {path}")
    return pose


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.resolve() != source.resolve():
            raise FileExistsError(f"Refusing to replace {target}")
        return
    target.symlink_to(source.resolve())


def _read_reference_model(reference_model: Path):
    try:
        images = read_extrinsics_binary(str(reference_model / "images.bin"))
        cameras = read_intrinsics_binary(str(reference_model / "cameras.bin"))
        image_path = reference_model / "images.bin"
        camera_path = reference_model / "cameras.bin"
    except (FileNotFoundError, OSError, ValueError):
        images = read_extrinsics_text(str(reference_model / "images.txt"))
        cameras = read_intrinsics_text(str(reference_model / "cameras.txt"))
        image_path = reference_model / "images.txt"
        camera_path = reference_model / "cameras.txt"
    if not images or not cameras:
        raise ValueError(f"Empty COLMAP reference model: {reference_model}")
    return images, cameras, image_path, camera_path


def _reference_points_metadata(reference_model: Path) -> dict:
    binary_path = reference_model / "points3D.bin"
    text_path = reference_model / "points3D.txt"
    if binary_path.is_file():
        if binary_path.stat().st_size < 8:
            raise ValueError(f"Truncated COLMAP point model: {binary_path}")
        with binary_path.open("rb") as handle:
            point_count = struct.unpack("<Q", handle.read(8))[0]
        point_path = binary_path
        point_format = "binary"
    elif text_path.is_file():
        point_count = 0
        for line_number, line in enumerate(text_path.read_text().splitlines(), 1):
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            fields = value.split()
            if len(fields) < 8:
                raise ValueError(f"Malformed COLMAP point at {text_path}:{line_number}")
            numeric = np.asarray([float(item) for item in fields[1:8]])
            if not np.isfinite(numeric).all():
                raise ValueError(
                    f"Non-finite COLMAP point at {text_path}:{line_number}"
                )
            if any(int(value) < 0 or int(value) > 255 for value in fields[4:7]):
                raise ValueError(
                    f"Invalid COLMAP point color at {text_path}:{line_number}"
                )
            point_count += 1
        point_path = text_path
        point_format = "text"
    else:
        raise FileNotFoundError(
            f"Reference model has no points3D.bin or points3D.txt: {reference_model}"
        )
    if point_count <= 0:
        raise ValueError(f"Reference point model is empty: {point_path}")
    return {
        "path": point_path,
        "format": point_format,
        "count": int(point_count),
        "sha256": _sha256(point_path),
    }


def _canonical_image_name(name: str) -> str:
    value = PurePosixPath(str(name).replace("\\", "/"))
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise ValueError(f"Unsafe reference image name: {name!r}")
    return value.as_posix()


def _resolve_reference_image(source: Path, name: str) -> Path:
    candidates = (source / name, source / "data" / name)
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"Expected one source image for {name!r}, found {existing}"
        )
    return existing[0]


def _pinhole_parameters(camera) -> tuple[float, float, float, float]:
    params = np.asarray(camera.params, dtype=np.float64)
    if camera.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL"}:
        if params.size < 3:
            raise ValueError(f"Invalid {camera.model} camera {camera.id}")
        return float(params[0]), float(params[0]), float(params[1]), float(params[2])
    if camera.model in {"PINHOLE", "OPENCV"}:
        if params.size < 4:
            raise ValueError(f"Invalid {camera.model} camera {camera.id}")
        return tuple(float(value) for value in params[:4])
    raise ValueError(
        f"Unsupported reference camera model {camera.model!r}; "
        "LaFGS requires a pinhole-compatible RGB model"
    )


def _camera_rectification(camera) -> CameraRectification:
    """Build a same-resolution COLMAP-to-OpenCV rectification map.

    COLMAP coordinates place the top-left pixel center at ``(0.5, 0.5)``;
    OpenCV's remap grid uses ``(0, 0)``.  Shifting the principal point by half
    a pixel while constructing the remap preserves the coordinate contract
    used by sparse keypoints and PnP throughout LaFGS.
    """
    params = np.asarray(camera.params, dtype=np.float64)
    fx, fy, cx, cy = _pinhole_parameters(camera)
    width, height = int(camera.width), int(camera.height)
    if camera.model in {"SIMPLE_PINHOLE", "PINHOLE"}:
        return CameraRectification(
            pinhole=(fx, fy, cx, cy),
            remap_x=None,
            remap_y=None,
            valid_mask=np.ones((height, width), dtype=np.bool_),
            source_model=str(camera.model),
            source_parameters=tuple(float(value) for value in params),
        )
    if camera.model == "SIMPLE_RADIAL":
        if params.size != 4:
            raise ValueError(f"Invalid SIMPLE_RADIAL camera {camera.id}")
        distortion = np.array([params[3], 0.0, 0.0, 0.0, 0.0])
    elif camera.model == "OPENCV":
        if params.size != 8:
            raise ValueError(f"Invalid OPENCV camera {camera.id}")
        distortion = np.array(
            [params[4], params[5], params[6], params[7], 0.0],
            dtype=np.float64,
        )
    else:
        raise ValueError(
            f"Unsupported reference camera model {camera.model!r}; "
            "LaFGS requires a pinhole-compatible RGB model"
        )

    camera_matrix = np.array(
        [[fx, 0.0, cx - 0.5], [0.0, fy, cy - 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    remap_x, remap_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        np.eye(3, dtype=np.float64),
        camera_matrix,
        (width, height),
        cv2.CV_32FC1,
    )
    source_support = np.ones((height, width), dtype=np.uint8)
    valid_mask = cv2.remap(
        source_support,
        remap_x,
        remap_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.bool_)
    return CameraRectification(
        pinhole=(fx, fy, cx, cy),
        remap_x=remap_x,
        remap_y=remap_y,
        valid_mask=valid_mask,
        source_model=str(camera.model),
        source_parameters=tuple(float(value) for value in params),
    )


def _write_rectified_image(
    source: Path,
    target: Path,
    rectification: CameraRectification,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to replace rectified image {target}")
    expected_size = (
        int(rectification.valid_mask.shape[1]),
        int(rectification.valid_mask.shape[0]),
    )
    with Image.open(source) as image:
        if image.size != expected_size:
            raise ValueError(
                f"Image/calibration size mismatch for {source}: "
                f"{image.size} != {expected_size}"
            )
        value = (
            np.asarray(image.convert("RGB"))
            if rectification.remap_x is not None
            else None
        )
    if rectification.remap_x is None:
        _link(source, target)
        return
    rectified = cv2.remap(
        value,
        rectification.remap_x,
        rectification.remap_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    save_kwargs = {}
    if target.suffix.lower() in {".jpg", ".jpeg"}:
        save_kwargs = {"quality": 95, "subsampling": 0}
    elif target.suffix.lower() == ".png":
        save_kwargs = {"compress_level": 1}
    Image.fromarray(rectified).save(target, **save_kwargs)


def _write_rectified_images(tasks) -> None:
    workers = min(8, max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_write_rectified_image, *task) for task in tasks]
        for future in futures:
            future.result()


def _rectification_manifest(camera, value: CameraRectification) -> dict:
    height, width = value.valid_mask.shape
    if value.remap_x is None:
        max_displacement = 0.0
    else:
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        displacement = np.hypot(value.remap_x - grid_x, value.remap_y - grid_y)
        max_displacement = float(displacement.max())
    fx, fy, cx, cy = value.pinhole
    return {
        "source_model": value.source_model,
        "source_parameters": list(value.source_parameters),
        "output_model": "PINHOLE",
        "output_parameters": [fx, fy, cx, cy],
        "output_width": int(camera.width),
        "output_height": int(camera.height),
        "maximum_remap_displacement_px": max_displacement,
        "valid_pixel_fraction": float(value.valid_mask.mean()),
        "interpolation": "opencv_linear",
        "pixel_center_conversion": "colmap_plus_half_to_opencv_zero",
    }


def _sequence_names(path: Path) -> tuple[str, ...]:
    values = []
    for line in path.read_text().splitlines():
        match = re.search(r"sequence(\d+)", line, flags=re.IGNORECASE)
        if match:
            values.append(f"seq-{int(match.group(1)):02d}")
    if not values:
        raise ValueError(f"No sequence IDs found in {path}")
    return tuple(values)


def _seven_scene_frames(source: Path) -> list[PreparedFrame]:
    training = set(_sequence_names(source / "TrainSplit.txt"))
    testing = set(_sequence_names(source / "TestSplit.txt"))
    if training & testing:
        raise ValueError("7Scenes train/test sequence sets overlap")
    available = {path.name for path in source.glob("seq-*") if path.is_dir()}
    missing = (training | testing) - available
    if missing:
        raise FileNotFoundError(f"Missing 7Scenes sequences: {sorted(missing)}")
    frames = []
    for sequence in sorted(training | testing):
        directory = source / sequence
        images = sorted(directory.glob("frame-*.color.png"))
        if not images:
            # The compact local smoke fixture uses color_*.png names.
            images = sorted(directory.glob("color_*.png"))
        for image in images:
            if image.name.startswith("frame-"):
                pose_name = image.name.replace(".color.png", ".pose.txt")
            else:
                pose_name = image.name.replace("color_", "pose_").replace(
                    ".png", ".txt"
                )
            pose = _read_pose(directory / pose_name) @ np.linalg.inv(_DEPTH_TO_RGB)
            frames.append(
                PreparedFrame(
                    image_name=f"{sequence}/{image.name}",
                    source_image=image,
                    pose_c2w=pose,
                    is_test=sequence in testing,
                )
            )
    return frames


def _parse_12scenes_info(path: Path) -> tuple[int, int, float, float, float, float]:
    lines = path.read_text().splitlines()
    values: dict[str, list[float]] = {}
    for line in lines:
        key, _, remainder = line.partition("=")
        numbers = [
            float(value)
            for value in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", remainder)
        ]
        if numbers:
            values[key.strip().lower()] = numbers

    def first_matching(*tokens: str) -> list[float] | None:
        for key, number_list in values.items():
            if all(token in key for token in tokens):
                return number_list
        return None

    width_values = first_matching("color", "width") or first_matching("image", "width")
    height_values = first_matching("color", "height") or first_matching(
        "image", "height"
    )
    intrinsics = first_matching("color", "intrinsic") or first_matching("intrinsic")
    if width_values and height_values and intrinsics and len(intrinsics) >= 9:
        # The official files flatten a 4x4 matrix; compact fixtures and some
        # converted releases use a 3x3 matrix.
        fy_index, cy_index = (5, 6) if len(intrinsics) >= 16 else (4, 5)
        return (
            int(width_values[0]),
            int(height_values[0]),
            float(intrinsics[0]),
            float(intrinsics[fy_index]),
            float(intrinsics[2]),
            float(intrinsics[cy_index]),
        )

    # Official 12Scenes info.txt stores dimensions on line 4 and a 4x4 color
    # calibration matrix on line 8. Keep this fallback explicit and validated.
    height_numbers = [float(value) for value in re.findall(r"[-+]?\d*\.?\d+", lines[3])]
    calibration = [
        float(value)
        for value in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", lines[7])
    ]
    if not height_numbers or len(calibration) < 8:
        raise ValueError(f"Unsupported 12Scenes calibration schema: {path}")
    height = int(height_numbers[-1])
    width = int(round(4 * height / 3))
    return width, height, calibration[0], calibration[5], calibration[2], calibration[6]


def _twelve_scene_test_end(path: Path) -> int:
    ranges = [
        tuple(map(int, match))
        for match in re.findall(
            r"start\s*=\s*(\d+)\s*;\s*end\s*=\s*(\d+)", path.read_text()
        )
    ]
    if not ranges:
        raise ValueError(f"Invalid 12Scenes split file: {path}")
    expected_start = 0
    for start, end in ranges:
        if start != expected_start or end < start:
            raise ValueError(f"Non-contiguous 12Scenes sequence ranges: {path}")
        expected_start = end + 1
    # The standard relocalization protocol uses sequence0 for testing and all
    # subsequent captures of the room for mapping/training.
    return ranges[0][1] + 1


def _twelve_scene_frames(source: Path) -> list[PreparedFrame]:
    test_end = _twelve_scene_test_end(source / "split.txt")
    images = sorted((source / "data").glob("frame-*.color.jpg"))
    poses = sorted((source / "data").glob("frame-*.pose.txt"))
    if len(images) != len(poses) or not images:
        raise ValueError(f"12Scenes RGB/pose count mismatch in {source}")
    frames = []
    for image, pose_path in zip(images, poses):
        image_id = int(re.search(r"frame-(\d+)", image.name).group(1))
        if image_id != int(re.search(r"frame-(\d+)", pose_path.name).group(1)):
            raise ValueError(f"12Scenes RGB/pose ordering mismatch in {source}")
        try:
            pose = _read_pose(pose_path)
        except ValueError:
            continue
        frames.append(
            PreparedFrame(
                image_name=image.name,
                source_image=image,
                pose_c2w=pose,
                is_test=image_id < test_end,
            )
        )
    return frames


def _write_colmap_scene(
    output: Path,
    frames: list[PreparedFrame],
    *,
    intrinsics: tuple[int, int, float, float, float, float],
    dataset: str,
    source: Path,
) -> dict:
    if not frames or not any(frame.is_test for frame in frames):
        raise ValueError("Prepared scene must contain mapping and test frames")
    if not any(not frame.is_test for frame in frames):
        raise ValueError("Prepared scene has no mapping frames")
    width, height, fx, fy, cx, cy = intrinsics
    processed = output / "processed"
    sparse = output / "sparse/0"
    prior_images = output / "prior_input/images"
    prior_sparse = output / "prior_input/sparse/0"
    sparse.mkdir(parents=True, exist_ok=True)
    prior_sparse.mkdir(parents=True, exist_ok=True)
    image_lines = ["# Ground-truth camera poses; second lines have no SfM points."]
    mapping_image_lines = [
        "# Mapping-only ground-truth camera poses; second lines have no SfM points."
    ]
    test_names = []
    for image_id, frame in enumerate(
        sorted(frames, key=lambda item: item.image_name), 1
    ):
        with Image.open(frame.source_image) as image:
            if image.size != (width, height):
                raise ValueError(
                    f"Image/calibration size mismatch for {frame.source_image}: "
                    f"{image.size} != {(width, height)}"
                )
        _link(frame.source_image, processed / frame.image_name)
        pose_w2c = np.linalg.inv(frame.pose_c2w)
        qvec = rotmat2qvec(pose_w2c[:3, :3])
        tvec = pose_w2c[:3, 3]
        values = [image_id, *qvec.tolist(), *tvec.tolist(), 1, frame.image_name]
        image_lines.extend([" ".join(map(str, values)), ""])
        if frame.is_test:
            test_names.append(frame.image_name)
        else:
            _link(frame.source_image, prior_images / frame.image_name)
            mapping_image_lines.extend([" ".join(map(str, values)), ""])
    camera_text = (
        "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS\n"
        f"1 PINHOLE {width} {height} {fx:.12g} {fy:.12g} {cx:.12g} {cy:.12g}\n"
    )
    (sparse / "cameras.txt").write_text(camera_text)
    (prior_sparse / "cameras.txt").write_text(camera_text)
    (sparse / "images.txt").write_text("\n".join(image_lines) + "\n")
    (prior_sparse / "images.txt").write_text("\n".join(mapping_image_lines) + "\n")
    (sparse / "points3D.txt").write_text(
        "# Empty before external RGB-only SfM triangulation.\n"
    )
    (prior_sparse / "points3D.txt").write_text(
        "# Empty before external RGB-only SfM triangulation.\n"
    )
    (sparse / "list_test.txt").write_text("\n".join(test_names) + "\n")
    manifest = {
        "schema_version": 1,
        "dataset": dataset,
        "source": str(source.resolve()),
        "camera_convention": "COLMAP_world_to_camera",
        "input_pose_convention": "camera_to_world",
        "pixel_center_convention": "grid_index_plus_half_at_pnp",
        "width": width,
        "height": height,
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "mapping_frames": len(frames) - len(test_names),
        "test_frames": len(test_names),
        "total_frames": len(frames),
        "test_list_sha256": _sha256(sparse / "list_test.txt"),
        "prior_input": {
            "mapping_only": True,
            "images": "prior_input/images",
            "sparse_model": "prior_input/sparse/0",
        },
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def prepare_reference_model_scene(
    source: str | Path,
    reference_model: str | Path,
    output: str | Path,
    *,
    dataset: str,
    use_reference_points: bool = True,
) -> dict:
    """Prepare a scene from a published COLMAP camera registry.

    The full reference point cloud is retained by default as Gaussian
    initialization.  Per-image feature observations are deliberately discarded
    and the output prior tree contains mapping RGB only.  Thus test RGB is never
    Gaussian supervision, although the published SfM point cloud can contain
    geometry reconstructed from the full reference registry.
    """
    source = Path(source).resolve()
    reference_model = Path(reference_model).resolve()
    output = Path(output).resolve()
    images, cameras, image_model_path, camera_model_path = _read_reference_model(
        reference_model
    )
    reference_points = (
        _reference_points_metadata(reference_model) if use_reference_points else None
    )
    test_list = reference_model / "list_test.txt"
    if not test_list.is_file():
        raise FileNotFoundError(test_list)
    test_names = {
        _canonical_image_name(line.strip())
        for line in test_list.read_text().splitlines()
        if line.strip()
    }
    registered = {_canonical_image_name(image.name): image for image in images.values()}
    if len(registered) != len(images):
        raise ValueError("Reference model image names must be unique")
    missing_test = test_names - set(registered)
    if missing_test:
        raise ValueError(
            f"Reference test list contains {len(missing_test)} unregistered images"
        )
    if not test_names or len(test_names) == len(registered):
        raise ValueError("Reference model must contain mapping and test images")

    processed = output / "processed"
    sparse = output / "sparse/0"
    prior_images = output / "prior_input/images"
    prior_sparse = output / "prior_input/sparse/0"
    sparse.mkdir(parents=True, exist_ok=True)
    prior_sparse.mkdir(parents=True, exist_ok=True)

    rectifications = {
        int(camera_id): _camera_rectification(camera)
        for camera_id, camera in sorted(cameras.items())
    }
    camera_lines = ["# Rectified pinhole cameras from the published reference model."]
    for camera_id, camera in sorted(cameras.items()):
        fx, fy, cx, cy = rectifications[int(camera_id)].pinhole
        camera_lines.append(
            f"{camera_id} PINHOLE {int(camera.width)} {int(camera.height)} "
            f"{fx:.17g} {fy:.17g} {cx:.17g} {cy:.17g}"
        )
    camera_text = "\n".join(camera_lines) + "\n"
    (sparse / "cameras.txt").write_text(camera_text)
    (prior_sparse / "cameras.txt").write_text(camera_text)

    all_lines = ["# Published pseudo-GT cameras; feature observations discarded."]
    mapping_lines = [
        "# Mapping-only published pseudo-GT cameras; feature observations discarded."
    ]
    mapping_count = 0
    rectification_tasks = []
    mask_channels = {}
    camera_masks = {
        camera_id: (
            np.ones_like(value.valid_mask),
            np.ones_like(value.valid_mask),
            value.valid_mask,
        )
        for camera_id, value in rectifications.items()
    }
    for name, image in sorted(registered.items()):
        camera = cameras.get(int(image.camera_id))
        if camera is None:
            raise ValueError(
                f"Image {name!r} refers to missing camera {image.camera_id}"
            )
        source_image = _resolve_reference_image(source, name)
        processed_image = processed / name
        rectification_tasks.append(
            (
                source_image,
                processed_image,
                rectifications[int(image.camera_id)],
            )
        )
        mask_channels[name] = camera_masks[int(image.camera_id)]
        values = [
            int(image.id),
            *np.asarray(image.qvec, dtype=np.float64).tolist(),
            *np.asarray(image.tvec, dtype=np.float64).tolist(),
            int(image.camera_id),
            name,
        ]
        row = " ".join(map(str, values))
        all_lines.extend((row, ""))
        if name not in test_names:
            mapping_count += 1
            _link(processed_image, prior_images / name)
            # A mapping-only subset can inherit sparse, non-contiguous image
            # IDs from the full reference model.  Reindex it so COLMAP's
            # single-camera frame IDs remain a closed 1..N registry during
            # known-pose triangulation.
            mapping_values = [
                mapping_count,
                *np.asarray(image.qvec, dtype=np.float64).tolist(),
                *np.asarray(image.tvec, dtype=np.float64).tolist(),
                int(image.camera_id),
                name,
            ]
            mapping_lines.extend((" ".join(map(str, mapping_values)), ""))

    _write_rectified_images(rectification_tasks)
    with (output / "masks.pkl").open("wb") as handle:
        pickle.dump(mask_channels, handle, protocol=pickle.HIGHEST_PROTOCOL)

    (sparse / "images.txt").write_text("\n".join(all_lines) + "\n")
    (prior_sparse / "images.txt").write_text("\n".join(mapping_lines) + "\n")
    if reference_points is not None:
        point_name = (
            "points3D.bin"
            if reference_points["format"] == "binary"
            else "points3D.txt"
        )
        _link(reference_points["path"], sparse / point_name)
        _link(reference_points["path"], prior_sparse / point_name)
    else:
        empty_points = "# Reference points explicitly excluded for legacy replay.\n"
        (sparse / "points3D.txt").write_text(empty_points)
        (prior_sparse / "points3D.txt").write_text(empty_points)
    ordered_test = sorted(test_names)
    (sparse / "list_test.txt").write_text("\n".join(ordered_test) + "\n")

    manifest = {
        "schema_version": 1,
        "dataset": dataset,
        "source": str(source),
        "pose_source": "published_sfm_pseudo_ground_truth",
        "reference_model": str(reference_model),
        "reference_image_model_sha256": _sha256(image_model_path),
        "reference_camera_model_sha256": _sha256(camera_model_path),
        "reference_camera_models": {
            str(camera_id): camera.model
            for camera_id, camera in sorted(cameras.items())
        },
        "reference_camera_parameters": {
            str(camera_id): np.asarray(camera.params, dtype=np.float64).tolist()
            for camera_id, camera in sorted(cameras.items())
        },
        "camera_model_normalization": "calibrated_undistortion_to_pinhole",
        "prior_input_image_ids": "contiguous_mapping_only_1_based",
        "undistortion": {
            "enabled": any(
                value.remap_x is not None for value in rectifications.values()
            ),
            "mapping_and_test_share_domain": True,
            "camera_models": {
                str(camera_id): _rectification_manifest(
                    cameras[camera_id], rectifications[camera_id]
                )
                for camera_id in sorted(rectifications)
            },
            "valid_masks": "masks.pkl",
        },
        "reference_test_list_sha256": _sha256(test_list),
        "camera_convention": "COLMAP_world_to_camera",
        "pixel_center_convention": "grid_index_plus_half_at_pnp",
        "mapping_frames": mapping_count,
        "test_frames": len(test_names),
        "total_frames": len(registered),
        "reference_points_used": reference_points is not None,
        "reference_points": (
            {
                "path": str(reference_points["path"]),
                "format": reference_points["format"],
                "count": reference_points["count"],
                "sha256": reference_points["sha256"],
                "may_include_test_view_reconstruction_evidence": True,
                "role": "gaussian_initialization_only",
            }
            if reference_points is not None
            else None
        ),
        "reference_feature_observations_used": False,
        "test_images_in_prior_input": False,
        "test_list_sha256": _sha256(sparse / "list_test.txt"),
        "prior_input": {
            "mapping_only": True,
            "images": "prior_input/images",
            "sparse_model": "prior_input/sparse/0",
        },
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def prepare_7scenes(source: str | Path, output: str | Path) -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    if source.name not in _SEVEN_SCENES:
        raise ValueError(f"Unknown 7Scenes scene: {source.name}")
    return _write_colmap_scene(
        output,
        _seven_scene_frames(source),
        intrinsics=(640, 480, 525.0, 525.0, 320.0, 240.0),
        dataset=f"7Scenes/{source.name}",
        source=source,
    )


def prepare_12scenes(source: str | Path, output: str | Path) -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    return _write_colmap_scene(
        output,
        _twelve_scene_frames(source),
        intrinsics=_parse_12scenes_info(source / "info.txt"),
        dataset=f"12Scenes/{source.parent.name}/{source.name}",
        source=source,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("7scenes", "12scenes"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference-model",
        type=Path,
        help=(
            "Optional published COLMAP pseudo-GT model. Its full 3D point cloud "
            "initializes the Gaussian prior, while registered poses and the test "
            "split keep RGB supervision mapping-only."
        ),
    )
    parser.add_argument(
        "--discard-reference-points",
        action="store_true",
        help="Explicit legacy replay: discard reference points and rebuild RGB-only geometry.",
    )
    args = parser.parse_args()
    if args.discard_reference_points and not args.reference_model:
        parser.error("--discard-reference-points requires --reference-model")
    if args.reference_model:
        dataset_name = (
            f"7Scenes/{args.source.name}"
            if args.dataset == "7scenes"
            else f"12Scenes/{args.source.parent.name}/{args.source.name}"
        )
        manifest = prepare_reference_model_scene(
            args.source,
            args.reference_model,
            args.output,
            dataset=dataset_name,
            use_reference_points=not args.discard_reference_points,
        )
    else:
        prepare = prepare_7scenes if args.dataset == "7scenes" else prepare_12scenes
        manifest = prepare(args.source, args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
