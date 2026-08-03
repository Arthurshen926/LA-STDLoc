from pathlib import Path

import numpy as np
from PIL import Image

from data.datasets import ColmapDataset
from data.preparation import prepare_7scenes, prepare_12scenes


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
