"""Prepare official indoor relocalization datasets for LaFGS.

The output is a COLMAP text model with ground-truth camera poses and an explicit
mapping/test split. Point triangulation and Gaussian reconstruction remain
external steps so this module does not introduce a hidden prior implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
import re

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
        numbers = [float(value) for value in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", remainder)]
        if numbers:
            values[key.strip().lower()] = numbers

    def first_matching(*tokens: str) -> list[float] | None:
        for key, number_list in values.items():
            if all(token in key for token in tokens):
                return number_list
        return None

    width_values = first_matching("color", "width") or first_matching("image", "width")
    height_values = first_matching("color", "height") or first_matching("image", "height")
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
    calibration = [float(value) for value in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", lines[7])]
    if not height_numbers or len(calibration) < 8:
        raise ValueError(f"Unsupported 12Scenes calibration schema: {path}")
    height = int(height_numbers[-1])
    width = int(round(4 * height / 3))
    return width, height, calibration[0], calibration[5], calibration[2], calibration[6]


def _twelve_scene_test_end(path: Path) -> int:
    ranges = [
        tuple(map(int, match))
        for match in re.findall(r"start\s*=\s*(\d+)\s*;\s*end\s*=\s*(\d+)", path.read_text())
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
    for image_id, frame in enumerate(sorted(frames, key=lambda item: item.image_name), 1):
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
    (prior_sparse / "images.txt").write_text(
        "\n".join(mapping_image_lines) + "\n"
    )
    (sparse / "points3D.txt").write_text("# Empty before external RGB-only SfM triangulation.\n")
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
) -> dict:
    """Prepare a scene from a published COLMAP camera registry.

    Reference 3D points and feature observations are deliberately discarded.
    The output prior tree contains mapping images and camera poses only, so an
    external RGB reconstruction cannot inherit geometry built from test views.
    """
    source = Path(source).resolve()
    reference_model = Path(reference_model).resolve()
    output = Path(output).resolve()
    images, cameras, image_model_path, camera_model_path = _read_reference_model(
        reference_model
    )
    test_list = reference_model / "list_test.txt"
    if not test_list.is_file():
        raise FileNotFoundError(test_list)
    test_names = {
        _canonical_image_name(line.strip())
        for line in test_list.read_text().splitlines()
        if line.strip()
    }
    registered = {
        _canonical_image_name(image.name): image for image in images.values()
    }
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

    camera_lines = ["# Canonical pinhole cameras from the published reference model."]
    for camera_id, camera in sorted(cameras.items()):
        fx, fy, cx, cy = _pinhole_parameters(camera)
        camera_lines.append(
            f"{camera_id} PINHOLE {int(camera.width)} {int(camera.height)} "
            f"{fx:.17g} {fy:.17g} {cx:.17g} {cy:.17g}"
        )
    camera_text = "\n".join(camera_lines) + "\n"
    (sparse / "cameras.txt").write_text(camera_text)
    (prior_sparse / "cameras.txt").write_text(camera_text)

    all_lines = ["# Published pseudo-GT cameras; reference points discarded."]
    mapping_lines = [
        "# Mapping-only published pseudo-GT cameras; reference points discarded."
    ]
    mapping_count = 0
    for name, image in sorted(registered.items()):
        camera = cameras.get(int(image.camera_id))
        if camera is None:
            raise ValueError(f"Image {name!r} refers to missing camera {image.camera_id}")
        source_image = _resolve_reference_image(source, name)
        with Image.open(source_image) as value:
            if value.size != (int(camera.width), int(camera.height)):
                raise ValueError(
                    f"Image/calibration size mismatch for {source_image}: "
                    f"{value.size} != {(int(camera.width), int(camera.height))}"
                )
        _link(source_image, processed / name)
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
            _link(source_image, prior_images / name)
            mapping_lines.extend((row, ""))

    (sparse / "images.txt").write_text("\n".join(all_lines) + "\n")
    (prior_sparse / "images.txt").write_text("\n".join(mapping_lines) + "\n")
    empty_points = "# Reference points deliberately excluded.\n"
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
        "camera_model_normalization": (
            "pinhole_focal_principal_point; distortion coefficients ignored"
        ),
        "reference_test_list_sha256": _sha256(test_list),
        "camera_convention": "COLMAP_world_to_camera",
        "pixel_center_convention": "grid_index_plus_half_at_pnp",
        "mapping_frames": mapping_count,
        "test_frames": len(test_names),
        "total_frames": len(registered),
        "reference_points_used": False,
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
            "Optional published COLMAP pseudo-GT camera model. Its 3D points "
            "and observations are discarded; only registered poses and the "
            "test split are imported."
        ),
    )
    args = parser.parse_args()
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
        )
    else:
        prepare = prepare_7scenes if args.dataset == "7scenes" else prepare_12scenes
        manifest = prepare(args.source, args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
