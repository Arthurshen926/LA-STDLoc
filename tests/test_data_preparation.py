from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from data.datasets import ColmapDataset
from data.preparation import (
    _camera_rectification,
    prepare_7scenes,
    prepare_12scenes,
    prepare_reference_model_scene,
)


def _write_image(path: Path, size=(640, 480)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(20, 40, 80)).save(path)


def _write_pose(path: Path, translation: float) -> None:
    pose = np.eye(4)
    pose[0, 3] = translation
    np.savetxt(path, pose)


def test_prepare_7scenes_preserves_official_sequence_split(tmp_path: Path):
    source = tmp_path / "chess"
    source.mkdir()
    (source / "TrainSplit.txt").write_text("sequence1\n")
    (source / "TestSplit.txt").write_text("sequence2\n")
    for sequence, translation in (("seq-01", 1.0), ("seq-02", 2.0)):
        image = source / sequence / "frame-000000.color.png"
        _write_image(image)
        _write_pose(
            source / sequence / "frame-000000.pose.txt", translation
        )
    output = tmp_path / "prepared"
    manifest = prepare_7scenes(source, output)
    dataset = ColmapDataset(output)
    assert manifest["mapping_frames"] == 1
    assert manifest["test_frames"] == 1
    assert [item.image_name for item in dataset.split("mapping")] == [
        "seq-01/frame-000000.color.png"
    ]
    assert [item.image_name for item in dataset.split("test")] == [
        "seq-02/frame-000000.color.png"
    ]
    assert (output / "prior_input/images/seq-01/frame-000000.color.png").is_symlink()
    assert not (output / "prior_input/images/seq-02/frame-000000.color.png").exists()


def test_prepare_12scenes_skips_nonfinite_poses(tmp_path: Path):
    source = tmp_path / "apt1" / "kitchen"
    data = source / "data"
    data.mkdir(parents=True)
    (source / "split.txt").write_text(
        "sequence0 [frames=1] [start=0 ; end=0]\n"
        "sequence1 [frames=2] [start=1 ; end=2]\n"
    )
    (source / "info.txt").write_text(
        "colorWidth = 640\n"
        "colorHeight = 480\n"
        "unused = 0\n"
        "imageHeight = 480\n"
        "unused = 0\n"
        "unused = 0\n"
        "unused = 0\n"
        "colorIntrinsic = 525 0 320 0 525 240 0 0 1\n"
    )
    for index in range(3):
        _write_image(data / f"frame-{index:06d}.color.jpg")
        _write_pose(data / f"frame-{index:06d}.pose.txt", float(index))
    pose = data / "frame-000002.pose.txt"
    pose.write_text("INF 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")
    output = tmp_path / "prepared"
    manifest = prepare_12scenes(source, output)
    dataset = ColmapDataset(output)
    assert manifest["mapping_frames"] == 1
    assert manifest["test_frames"] == 1
    assert len(dataset.cameras) == 2
    assert manifest["intrinsics"] == {
        "fx": 525.0,
        "fy": 525.0,
        "cx": 320.0,
        "cy": 240.0,
    }
    assert (output / "prior_input/images/frame-000001.color.jpg").is_symlink()
    assert not (output / "prior_input/images/frame-000000.color.jpg").exists()


def test_prepare_12scenes_reads_official_4x4_intrinsics(tmp_path: Path):
    source = tmp_path / "office2" / "5a"
    data = source / "data"
    data.mkdir(parents=True)
    (source / "split.txt").write_text(
        "sequence0 [frames=1] [start=0 ; end=0]\n"
        "sequence1 [frames=1] [start=1 ; end=1]\n"
    )
    (source / "info.txt").write_text(
        "m_colorWidth = 1296\n"
        "m_colorHeight = 968\n"
        "unused = 0\n"
        "imageHeight = 968\n"
        "unused = 0\n"
        "unused = 0\n"
        "unused = 0\n"
        "m_calibrationColorIntrinsic = "
        "1158.3 0 649 0 0 1153.53 483.5 0 0 0 1 0 0 0 0 1\n"
    )
    for index in range(2):
        _write_image(data / f"frame-{index:06d}.color.jpg", size=(1296, 968))
        _write_pose(data / f"frame-{index:06d}.pose.txt", float(index))

    manifest = prepare_12scenes(source, tmp_path / "prepared")
    assert manifest["intrinsics"] == {
        "fx": 1158.3,
        "fy": 1153.53,
        "cx": 649.0,
        "cy": 483.5,
    }


def test_prepare_reference_model_discards_points_and_test_prior_images(
    tmp_path: Path,
):
    source = tmp_path / "source"
    mapping_name = "seq-01/frame-000000.color.png"
    test_name = "seq-02/frame-000000.color.png"
    _write_image(source / mapping_name)
    _write_image(source / test_name)

    reference = tmp_path / "reference" / "sfm_gt"
    reference.mkdir(parents=True)
    (reference / "cameras.txt").write_text(
        "1 PINHOLE 640 480 527.745 527.745 320 240\n"
    )
    (reference / "images.txt").write_text(
        "1 1 0 0 0 1 0 0 1 seq-01/frame-000000.color.png\n\n"
        "2 1 0 0 0 2 0 0 1 seq-02/frame-000000.color.png\n\n"
    )
    (reference / "points3D.txt").write_text(
        "1 0 0 1 255 255 255 0.1 1 0\n"
    )
    (reference / "list_test.txt").write_text(test_name + "\n")

    output = tmp_path / "prepared"
    manifest = prepare_reference_model_scene(
        source, reference, output, dataset="7Scenes/chess"
    )
    dataset = ColmapDataset(output)
    prior_text = (output / "prior_input/sparse/0/images.txt").read_text()
    assert manifest["pose_source"] == "published_sfm_pseudo_ground_truth"
    assert manifest["mapping_frames"] == 1
    assert manifest["test_frames"] == 1
    assert manifest["reference_points_used"] is False
    assert manifest["reference_feature_observations_used"] is False
    assert [camera.image_name for camera in dataset.split("test")] == [test_name]
    assert mapping_name in prior_text
    assert test_name not in prior_text
    assert (output / "prior_input/images" / mapping_name).is_symlink()
    assert not (output / "prior_input/images" / test_name).exists()
    assert "Reference points deliberately excluded" in (
        output / "prior_input/sparse/0/points3D.txt"
    ).read_text()


def test_prepare_reference_model_reindexes_sparse_mapping_ids(tmp_path: Path):
    source = tmp_path / "source"
    test_name = "seq-01/frame-000000.color.png"
    mapping_names = [
        "seq-02/frame-000000.color.png",
        "seq-03/frame-000000.color.png",
    ]
    for name in [test_name, *mapping_names]:
        _write_image(source / name)

    reference = tmp_path / "reference" / "sfm_gt"
    reference.mkdir(parents=True)
    (reference / "cameras.txt").write_text(
        "1 SIMPLE_RADIAL 640 480 525 320 240 -0.025\n"
    )
    (reference / "images.txt").write_text(
        "10 1 0 0 0 1 0 0 1 seq-01/frame-000000.color.png\n\n"
        "501 1 0 0 0 2 0 0 1 seq-02/frame-000000.color.png\n\n"
        "900 1 0 0 0 3 0 0 1 seq-03/frame-000000.color.png\n\n"
    )
    (reference / "points3D.txt").write_text("")
    (reference / "list_test.txt").write_text(test_name + "\n")

    output = tmp_path / "prepared"
    manifest = prepare_reference_model_scene(
        source, reference, output, dataset="7Scenes/stairs"
    )
    prior_rows = [
        line.split()
        for line in (output / "prior_input/sparse/0/images.txt")
        .read_text()
        .splitlines()
        if line and not line.startswith("#")
    ]
    assert [int(row[0]) for row in prior_rows] == [1, 2]
    assert [row[-1] for row in prior_rows] == mapping_names
    assert manifest["prior_input_image_ids"] == (
        "contiguous_mapping_only_1_based"
    )


def test_simple_radial_rectification_respects_colmap_pixel_centers():
    camera = SimpleNamespace(
        id=1,
        model="SIMPLE_RADIAL",
        width=640,
        height=480,
        params=np.array([525.0, 320.0, 240.0, -0.025]),
    )
    rectification = _camera_rectification(camera)
    output_x_cv, output_y_cv = 520, 340
    output_x_colmap = output_x_cv + 0.5
    output_y_colmap = output_y_cv + 0.5
    x = (output_x_colmap - 320.0) / 525.0
    y = (output_y_colmap - 240.0) / 525.0
    radial = 1.0 - 0.025 * (x * x + y * y)
    expected_source_x_cv = 525.0 * x * radial + 320.0 - 0.5
    expected_source_y_cv = 525.0 * y * radial + 240.0 - 0.5
    assert np.isclose(
        rectification.remap_x[output_y_cv, output_x_cv],
        expected_source_x_cv,
        atol=1e-4,
    )
    assert np.isclose(
        rectification.remap_y[output_y_cv, output_x_cv],
        expected_source_y_cv,
        atol=1e-4,
    )


def test_prepare_reference_model_rectifies_simple_radial_images(tmp_path: Path):
    source = tmp_path / "source"
    mapping_name = "seq-01/frame-000000.color.png"
    test_name = "seq-02/frame-000000.color.png"
    for name in (mapping_name, test_name):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        x = np.arange(64, dtype=np.uint8)[None].repeat(48, axis=0)
        Image.fromarray(np.stack((x, x, x), axis=-1)).save(path)

    reference = tmp_path / "reference" / "sfm_gt"
    reference.mkdir(parents=True)
    (reference / "cameras.txt").write_text(
        "1 SIMPLE_RADIAL 64 48 50 32 24 -0.1\n"
    )
    (reference / "images.txt").write_text(
        "1 1 0 0 0 1 0 0 1 seq-01/frame-000000.color.png\n\n"
        "2 1 0 0 0 2 0 0 1 seq-02/frame-000000.color.png\n\n"
    )
    (reference / "points3D.txt").write_text("")
    (reference / "list_test.txt").write_text(test_name + "\n")

    output = tmp_path / "prepared"
    manifest = prepare_reference_model_scene(
        source, reference, output, dataset="7Scenes/chess"
    )
    processed_mapping = output / "processed" / mapping_name
    prior_mapping = output / "prior_input/images" / mapping_name
    assert processed_mapping.is_file() and not processed_mapping.is_symlink()
    assert prior_mapping.is_symlink()
    assert prior_mapping.resolve() == processed_mapping.resolve()
    assert manifest["camera_model_normalization"] == (
        "calibrated_undistortion_to_pinhole"
    )
    assert manifest["undistortion"]["enabled"] is True
    assert manifest["undistortion"]["camera_models"]["1"][
        "maximum_remap_displacement_px"
    ] > 1.0
    assert (output / "masks.pkl").is_file()
    assert "PINHOLE 64 48 50 50 32 24" in (
        output / "sparse/0/cameras.txt"
    ).read_text()
