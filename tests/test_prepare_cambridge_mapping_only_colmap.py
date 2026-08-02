from pathlib import Path

import numpy as np

from scripts.prepare_cambridge_mapping_only_colmap import (
    Camera,
    Image,
    Point3D,
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
    stage_mapping_only_colmap,
    write_cameras_binary,
    write_images_binary,
    write_points3d_binary,
)


def _image(image_id, name, point_ids):
    return Image(
        image_id,
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 0.0]),
        image_id,
        name,
        np.asarray([[10.0 + row, 20.0] for row in range(len(point_ids))]),
        np.asarray(point_ids, dtype=np.int64),
    )


def test_mapping_only_colmap_excludes_test_cameras_and_shared_points(tmp_path):
    source = tmp_path / "source"
    sparse = source / "sparse" / "0"
    sparse.mkdir(parents=True)
    names = ["seq1/a.png", "seq1/b.png", "seq2/test.png"]
    for name in names:
        path = source / "processed" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"rgb")
    (source / "dataset_train.txt").write_text(
        "Visual Landmark Dataset V1\nImageFile, pose\n\n"
        "seq1/a.png 0\nseq1/b.png 0\n"
    )
    (source / "dataset_test.txt").write_text(
        "Visual Landmark Dataset V1\nImageFile, pose\n\nseq2/test.png 0\n"
    )
    cameras = {
        index: Camera(index, "PINHOLE", 64, 48, np.ones(4))
        for index in (1, 2, 3)
    }
    images = {
        1: _image(1, names[0], [10, 11, 12]),
        2: _image(2, names[1], [10]),
        3: _image(3, names[2], [11]),
    }
    points = {
        10: Point3D(
            10,
            np.asarray([0.0, 0.0, 1.0]),
            np.asarray([255, 0, 0], dtype=np.uint8),
            0.1,
            np.asarray([1, 2], dtype=np.int32),
            np.asarray([0, 0], dtype=np.int32),
        ),
        11: Point3D(
            11,
            np.asarray([1.0, 0.0, 1.0]),
            np.asarray([0, 255, 0], dtype=np.uint8),
            0.2,
            np.asarray([1, 3], dtype=np.int32),
            np.asarray([1, 0], dtype=np.int32),
        ),
        12: Point3D(
            12,
            np.asarray([2.0, 0.0, 1.0]),
            np.asarray([0, 0, 255], dtype=np.uint8),
            0.3,
            np.asarray([1], dtype=np.int32),
            np.asarray([2], dtype=np.int32),
        ),
    }
    write_cameras_binary(cameras, sparse / "cameras.bin")
    write_images_binary(images, sparse / "images.bin")
    write_points3d_binary(points, sparse / "points3D.bin")

    output = tmp_path / "mapping"
    report = stage_mapping_only_colmap(
        source=source,
        output=output,
        images_dir="processed",
        minimum_track_length=2,
    )

    staged_images = read_images_binary(output / "sparse" / "0" / "images.bin")
    staged_points = read_points3d_binary(output / "sparse" / "0" / "points3D.bin")
    assert {image.name for image in staged_images.values()} == set(names[:2])
    assert set(staged_points) == {10}
    assert staged_images[1].point3D_ids.tolist() == [10, -1, -1]
    assert report["dropped_test_observed_point_count"] == 1
    assert report["dropped_short_track_point_count"] == 1
    assert report["semantic_mask_used"] is False
    assert (output / "images" / names[0]).is_symlink()


def test_mapping_only_colmap_can_emit_official_pinhole_inputs(tmp_path):
    source = tmp_path / "source"
    sparse = source / "sparse" / "0"
    sparse.mkdir(parents=True)
    name = "seq1/a.png"
    image_path = source / "processed" / name
    image_path.parent.mkdir(parents=True)
    pixels = np.zeros((48, 64, 3), dtype=np.uint8)
    pixels[20:28, 28:36] = 255
    import cv2

    assert cv2.imwrite(str(image_path), pixels)
    (source / "dataset_train.txt").write_text(
        "Visual Landmark Dataset V1\nImageFile, pose\n\nseq1/a.png 0\n"
    )
    (source / "dataset_test.txt").write_text(
        "Visual Landmark Dataset V1\nImageFile, pose\n\nseq2/test.png 0\n"
    )
    camera = Camera(
        1,
        "SIMPLE_RADIAL",
        64,
        48,
        np.asarray([50.0, 32.0, 24.0, 0.05]),
    )
    image = _image(1, name, [10, 10])
    point = Point3D(
        10,
        np.asarray([0.0, 0.0, 1.0]),
        np.asarray([255, 0, 0], dtype=np.uint8),
        0.1,
        np.asarray([1, 1], dtype=np.int32),
        np.asarray([0, 1], dtype=np.int32),
    )
    write_cameras_binary({1: camera}, sparse / "cameras.bin")
    write_images_binary({1: image}, sparse / "images.bin")
    write_points3d_binary({10: point}, sparse / "points3D.bin")

    output = tmp_path / "mapping"
    report = stage_mapping_only_colmap(
        source=source,
        output=output,
        images_dir="processed",
        minimum_track_length=1,
        undistort_images=True,
    )

    staged_camera = read_cameras_binary(
        output / "sparse" / "0" / "cameras.bin"
    )[1]
    staged_image = read_images_binary(
        output / "sparse" / "0" / "images.bin"
    )[1]
    assert staged_camera.model == "PINHOLE"
    assert staged_camera.params.tolist() == [50.0, 50.0, 32.0, 24.0]
    assert (output / "images" / name).is_file()
    assert not (output / "images" / name).is_symlink()
    assert not np.allclose(staged_image.xys, image.xys)
    assert report["undistortion_used"] is True
    assert report["source_camera_models"] == ["SIMPLE_RADIAL"]
    assert report["target_camera_models"] == ["PINHOLE"]
