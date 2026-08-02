import cv2
import numpy as np

from scripts.prepare_cambridge_mapping_only_colmap import (
    Camera,
    Image,
    Point3D,
    read_cameras_binary,
    stage_mapping_only_colmap,
    write_cameras_binary,
    write_images_binary,
    write_points3d_binary,
)
from scripts.prepare_cambridge_official_eval_colmap import (
    stage_official_eval_scene,
)


def _camera_image(image_id, name, point_index):
    return Image(
        image_id,
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 0.0]),
        image_id,
        name,
        np.asarray([[30.0 + point_index, 20.0]]),
        np.asarray([10], dtype=np.int64),
    )


def test_official_eval_scene_is_separate_from_mapping_training(tmp_path):
    source = tmp_path / "source"
    sparse = source / "sparse" / "0"
    sparse.mkdir(parents=True)
    names = ["seq1/train.png", "seq2/test.png"]
    for index, name in enumerate(names):
        path = source / "processed" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = np.full((48, 64, 3), index * 100, dtype=np.uint8)
        assert cv2.imwrite(str(path), pixels)
    (source / "dataset_train.txt").write_text(
        "Visual Landmark Dataset V1\nImageFile, pose\n\nseq1/train.png 0\n"
    )
    (source / "dataset_test.txt").write_text(
        "Visual Landmark Dataset V1\nImageFile, pose\n\nseq2/test.png 0\n"
    )
    cameras = {
        index: Camera(
            index,
            "SIMPLE_RADIAL",
            64,
            48,
            np.asarray([50.0, 32.0, 24.0, 0.05]),
        )
        for index in (1, 2)
    }
    images = {
        1: _camera_image(1, names[0], 0),
        2: _camera_image(2, names[1], 1),
    }
    point = Point3D(
        10,
        np.asarray([0.0, 0.0, 1.0]),
        np.asarray([255, 0, 0], dtype=np.uint8),
        0.1,
        np.asarray([1, 2], dtype=np.int32),
        np.asarray([0, 0], dtype=np.int32),
    )
    write_cameras_binary(cameras, sparse / "cameras.bin")
    write_images_binary(images, sparse / "images.bin")
    write_points3d_binary({10: point}, sparse / "points3D.bin")

    mapping = tmp_path / "mapping"
    stage_mapping_only_colmap(
        source=source,
        output=mapping,
        images_dir="processed",
        minimum_track_length=1,
        undistort_images=True,
    )
    evaluation = tmp_path / "evaluation"
    report = stage_official_eval_scene(
        source=source,
        mapping_dataset=mapping,
        output=evaluation,
        images_dir="processed",
    )

    assert (evaluation / "images" / names[0]).is_symlink()
    assert (evaluation / "images" / names[1]).is_file()
    assert not (evaluation / "images" / names[1]).is_symlink()
    assert {
        camera.model
        for camera in read_cameras_binary(
            evaluation / "sparse" / "0" / "cameras.bin"
        ).values()
    } == {"PINHOLE"}
    assert report["evaluation_only"] is True
    assert report["used_for_prior_training"] is False
    assert report["mapping_image_count"] == 1
    assert report["test_image_count"] == 1
