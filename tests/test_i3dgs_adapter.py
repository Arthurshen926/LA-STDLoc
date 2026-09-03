from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData, PlyElement

from priors.i3dgs_adapter import (
    PREPARED_SCHEMA,
    _real_sh_basis,
    _sh_world_rotation,
    export_i3dgs_world_prior,
    prepare_cambridge_mapping,
)


def _write_cambridge_fixture(root: Path) -> None:
    (root / "processed" / "seq1").mkdir(parents=True)
    (root / "sparse" / "0").mkdir(parents=True)
    lines = ["Visual Landmark Dataset V1", "ImageFile, pose"]
    image_lines = ["# images"]
    camera_lines = ["# cameras"]
    for index in range(6):
        name = f"seq1/frame{index:05d}.png"
        image = np.full((48, 64, 3), 20 + index, dtype=np.uint8)
        assert cv2.imwrite(str(root / "processed" / name), image)
        lines.append(f"{name} {index} 0 0 1 0 0 0")
        camera_lines.append(f"{index + 1} SIMPLE_RADIAL 64 48 50 32 24 0.01")
        image_lines.append(
            f"{index + 1} 1 0 0 0 {-index} 0 0 {index + 1} {name}"
        )
        image_lines.append("")
    (root / "dataset_train.txt").write_text("\n".join(lines) + "\n")
    sparse = root / "sparse" / "0"
    (sparse / "cameras.txt").write_text("\n".join(camera_lines) + "\n")
    (sparse / "images.txt").write_text("\n".join(image_lines) + "\n")
    (sparse / "points3D.txt").write_text("# Number of points: 0\n")


def test_prepare_cambridge_mapping_is_flat_mapping_only(tmp_path: Path) -> None:
    source = tmp_path / "scene"
    output = tmp_path / "prepared"
    _write_cambridge_fixture(source)
    payload = prepare_cambridge_mapping(source, output, downscale=2.0)

    assert payload["schema"] == PREPARED_SCHEMA
    assert payload["image_count"] == 6
    assert len(list((output / "images").glob("*.png"))) == 6
    assert (output / "images" / "seq1__frame00000.png").is_file()
    assert " PINHOLE 32 24 " in (output / "sparse/0/cameras.txt").read_text()
    assert payload["i3dgs_internal_pose_normalization"]["scale"] == 0.1


def test_sh_rotation_preserves_directional_colour() -> None:
    angle = np.deg2rad(37.0)
    world_to_local = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]],
        dtype=np.float64,
    )
    transform = _sh_world_rotation(world_to_local, 3)
    generator = np.random.default_rng(3)
    directions = generator.normal(size=(200, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    local_coefficients = generator.normal(size=(16, 3))
    world_coefficients = transform @ local_coefficients
    local_directions = directions @ world_to_local.T
    np.testing.assert_allclose(
        _real_sh_basis(directions, 3) @ world_coefficients,
        _real_sh_basis(local_directions, 3) @ local_coefficients,
        atol=1e-10,
    )


def test_export_i3dgs_leaves_to_mapping_world(tmp_path: Path) -> None:
    angle = np.deg2rad(90.0)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]],
        dtype=np.float64,
    )
    first_w2c = np.eye(4)
    first_w2c[:3, :3] = rotation
    first_w2c[:3, 3] = [1.0, 2.0, 3.0]
    prepared = {
        "schema": PREPARED_SCHEMA,
        "version": 1,
        "i3dgs_internal_pose_normalization": {
            "scale": 0.5,
            "first_mapping_w2c": first_w2c.tolist(),
        },
    }
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text(json.dumps(prepared))

    names = ["x", "y", "z"]
    names += [f"f_dc_{index}" for index in range(3)]
    names += [f"f_rest_{index}" for index in range(9)]
    names += ["opacity"]
    names += [f"scale_{index}" for index in range(3)]
    names += [f"rot_{index}" for index in range(4)]
    names += ["child_0", "parent"]
    dtype = [
        (name, "<i4" if name in {"child_0", "parent"} else "<f4")
        for name in names
    ]
    vertices = np.zeros(2, dtype=dtype)
    vertices["x"] = [1.0, 100.0]
    vertices["y"] = [2.0, 100.0]
    vertices["z"] = [3.0, 100.0]
    vertices["f_dc_0"] = 0.1
    vertices["f_dc_1"] = 0.2
    vertices["f_dc_2"] = 0.3
    vertices["opacity"] = 1.0
    vertices["rot_0"] = 1.0
    vertices["child_0"] = [-1, 0]
    vertices["parent"] = [1, -1]
    source_path = tmp_path / "hierarchy.ply"
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(source_path)

    output_path = tmp_path / "world.ply"
    report = export_i3dgs_world_prior(
        source_path,
        prepared_path,
        output_path,
        output_manifest=tmp_path / "export.json",
    )
    output = PlyData.read(output_path)["vertex"].data
    assert report["source_vertex_count"] == 2
    assert report["primitive_count"] == 1
    expected_center = -rotation.T @ first_w2c[:3, 3]
    expected_xyz = np.asarray([1.0, 2.0, 3.0]) @ rotation / 0.5 + expected_center
    np.testing.assert_allclose(
        np.asarray([output["x"][0], output["y"][0], output["z"][0]]),
        expected_xyz,
        atol=1e-5,
    )
    np.testing.assert_allclose(output["scale_0"][0], np.log(2.0), atol=1e-6)
    assert not any(name.startswith("child_") for name in output.dtype.names)
