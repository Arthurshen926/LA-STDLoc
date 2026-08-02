#!/usr/bin/env python3
"""Stage an evaluation-only undistorted Cambridge COLMAP scene."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

try:
    from scripts.prepare_cambridge_mapping_only_colmap import (
        _opencv_camera,
        _undistort_image_and_observations,
        read_cambridge_names,
        read_cameras_binary,
        read_images_binary,
        read_points3d_binary,
        sha256,
        write_cameras_binary,
        write_images_binary,
        write_points3d_binary,
        write_points3d_ply,
    )
except ModuleNotFoundError:
    from prepare_cambridge_mapping_only_colmap import (
        _opencv_camera,
        _undistort_image_and_observations,
        read_cambridge_names,
        read_cameras_binary,
        read_images_binary,
        read_points3d_binary,
        sha256,
        write_cameras_binary,
        write_images_binary,
        write_points3d_binary,
        write_points3d_ply,
    )


def _undistort_observations(image, camera):
    intrinsic, distortion, _ = _opencv_camera(camera)
    xys = np.asarray(image.xys, dtype=np.float64)
    if len(xys):
        xys = cv2.undistortPoints(
            xys.reshape(-1, 1, 2),
            intrinsic,
            distortion,
            P=intrinsic,
        ).reshape(-1, 2)
    return image._replace(xys=xys)


def stage_official_eval_scene(
    *, source: Path, mapping_dataset: Path, output: Path, images_dir: str
) -> dict:
    source = source.expanduser().resolve()
    mapping_dataset = mapping_dataset.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation scene: {output}")
    mapping_manifest_path = mapping_dataset / "mapping_only_manifest.json"
    mapping_manifest = json.loads(mapping_manifest_path.read_text())
    if mapping_manifest.get("undistortion_used") is not True:
        raise ValueError("mapping parent must be undistorted")
    if mapping_manifest.get("semantic_mask_used") is not False:
        raise ValueError("mapping parent must be mask-free")

    sparse = source / "sparse" / "0"
    cameras = read_cameras_binary(sparse / "cameras.bin")
    images = read_images_binary(sparse / "images.bin")
    points = read_points3d_binary(sparse / "points3D.bin")
    train_names = read_cambridge_names(source / "dataset_train.txt")
    test_names = read_cambridge_names(source / "dataset_test.txt")
    expected_names = set(train_names) | set(test_names)
    image_by_name = {image.name: image for image in images.values()}
    missing = sorted(expected_names.difference(image_by_name))
    if missing:
        raise ValueError(f"evaluation image absent from COLMAP model: {missing[0]}")

    selected_images = {
        image_by_name[name].id: image_by_name[name] for name in expected_names
    }
    selected_camera_ids = {image.camera_id for image in selected_images.values()}
    selected_cameras = {
        camera_id: cameras[camera_id] for camera_id in selected_camera_ids
    }
    target_cameras = {
        camera_id: _opencv_camera(camera)[2]
        for camera_id, camera in selected_cameras.items()
    }

    output_sparse = output / "sparse" / "0"
    output_images = output / "images"
    output_sparse.mkdir(parents=True)
    output_images.mkdir(parents=True)
    transformed_images = {}
    image_root = source / images_dir
    train_set = set(train_names)
    for image_id, image in selected_images.items():
        destination = output_images / image.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if image.name in train_set:
            parent_image = mapping_dataset / "images" / image.name
            if not parent_image.is_file():
                raise FileNotFoundError(parent_image)
            os.symlink(parent_image, destination)
            transformed = _undistort_observations(
                image, selected_cameras[image.camera_id]
            )
        else:
            transformed = _undistort_image_and_observations(
                source=image_root / image.name,
                destination=destination,
                image=image,
                camera=selected_cameras[image.camera_id],
            )
        transformed_images[image_id] = transformed

    write_cameras_binary(target_cameras, output_sparse / "cameras.bin")
    write_images_binary(transformed_images, output_sparse / "images.bin")
    write_points3d_binary(points, output_sparse / "points3D.bin")
    write_points3d_ply(points, output_sparse / "points3D.ply")
    shutil.copy2(source / "dataset_train.txt", output / "dataset_train.txt")
    shutil.copy2(source / "dataset_test.txt", output / "dataset_test.txt")

    manifest = {
        "schema": "lafgs_off_the_shelf_prior_evaluation_scene",
        "version": 1,
        "evaluation_only": True,
        "used_for_prior_training": False,
        "used_for_lafgs_training": False,
        "source": str(source),
        "mapping_parent": str(mapping_dataset),
        "mapping_parent_manifest_sha256": sha256(mapping_manifest_path),
        "semantic_mask_used": False,
        "undistortion_used": True,
        "mapping_image_count": len(train_names),
        "test_image_count": len(test_names),
        "camera_count": len(target_cameras),
        "target_camera_models": sorted(
            {camera.model for camera in target_cameras.values()}
        ),
    }
    (output / "evaluation_scene_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mapping-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images-dir", default="processed")
    args = parser.parse_args()
    report = stage_official_eval_scene(
        source=args.source,
        mapping_dataset=args.mapping_dataset,
        output=args.output,
        images_dir=args.images_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
