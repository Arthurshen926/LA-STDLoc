from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData, PlyElement
import torch

from priors.long_lrm_adapter import (
    PREPARED_SCHEMA,
    export_long_lrm_world_prior,
    prepare_cambridge_mapping,
)


def _write_cambridge_fixture(root: Path) -> None:
    (root / "sparse" / "0").mkdir(parents=True)
    split = ["Visual Landmark Dataset V1", "ImageFile, pose"]
    camera_lines = ["# cameras"]
    image_lines = ["# images"]
    for index in range(40):
        sequence = f"seq{index % 4}"
        name = f"{sequence}/frame{index:05d}.png"
        (root / "processed" / sequence).mkdir(parents=True, exist_ok=True)
        pixels = np.full((48, 64, 3), 20 + index, dtype=np.uint8)
        assert cv2.imwrite(str(root / "processed" / name), pixels)
        split.append(f"{name} {index} 0 0 1 0 0 0")
        camera_lines.append(f"{index + 1} SIMPLE_RADIAL 64 48 50 32 24 0.01")
        # Camera centres span x and y; identity camera rotation keeps the
        # Long-LRM average forward/down basis well conditioned.
        x = float(index % 10)
        y = float(index // 10)
        image_lines.append(
            f"{index + 1} 1 0 0 0 {-x} {-y} 0 {index + 1} {name}"
        )
        image_lines.append("")
    (root / "dataset_train.txt").write_text("\n".join(split) + "\n")
    sparse = root / "sparse" / "0"
    (sparse / "cameras.txt").write_text("\n".join(camera_lines) + "\n")
    (sparse / "images.txt").write_text("\n".join(image_lines) + "\n")
    (sparse / "points3D.txt").write_text("# Number of points: 0\n")


def _write_standard_ply(path: Path) -> None:
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{index}" for index in range(3)]
    names += [f"f_rest_{index}" for index in range(45)]
    names += ["opacity"]
    names += [f"scale_{index}" for index in range(3)]
    names += [f"rot_{index}" for index in range(4)]
    vertices = np.zeros(1, dtype=[(name, "<f4") for name in names])
    vertices["x"] = 0.25
    vertices["y"] = -0.5
    vertices["z"] = 0.75
    vertices["f_dc_0"] = 0.1
    vertices["f_dc_1"] = 0.2
    vertices["f_dc_2"] = 0.3
    vertices["opacity"] = 1.0
    vertices["rot_0"] = 1.0
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def test_prepare_and_export_long_lrm_mapping_only(tmp_path: Path) -> None:
    dataset = tmp_path / "ShopFacade"
    prepared_root = tmp_path / "prepared"
    _write_cambridge_fixture(dataset)
    prepared = prepare_cambridge_mapping(dataset, prepared_root, input_image_domain='rectified')
    assert prepared["schema"] == PREPARED_SCHEMA
    assert prepared["role"] == "mapping_only"
    assert prepared["uses_test_queries"] is False
    assert prepared["view_count"] == 32
    assert len(prepared["records"]) == 32
    assert len({row["source_name"] for row in prepared["records"]}) == 32
    assert len(list((prepared_root / "images").glob("*.png"))) == 32

    source = tmp_path / "normalized.ply"
    output = tmp_path / "world.ply"
    _write_standard_ply(source)
    report = export_long_lrm_world_prior(
        source,
        prepared_root / "anygsloc_long_lrm_input.json",
        output,
        output_manifest=tmp_path / "export.json",
    )
    assert report["source_method"] == "long_lrm_32view"
    assert report["primitive_count"] == 1
    vertex = PlyData.read(output)["vertex"].data
    transform = np.asarray(
        prepared["long_lrm_internal_pose_normalization"][
            "normalized_to_world_sim3"
        ]
    )
    expected = transform @ np.asarray([0.25, -0.5, 0.75, 1.0])
    np.testing.assert_allclose(
        [vertex["x"][0], vertex["y"][0], vertex["z"][0]],
        expected[:3],
        atol=1e-5,
    )
    scale = prepared["long_lrm_internal_pose_normalization"]["scene_scale"]
    np.testing.assert_allclose(vertex["scale_0"][0], -np.log(scale), atol=1e-6)


def test_vectorized_writer_preserves_official_layout_and_threshold(tmp_path: Path) -> None:
    from scripts.run_long_lrm_cambridge import _save_gaussian_ply_vectorized

    generator = torch.Generator().manual_seed(12)
    values = {
        "xyz": torch.randn(7, 3, generator=generator),
        "feature": torch.randn(7, 16, 3, generator=generator),
        "scale": torch.randn(7, 3, generator=generator),
        "rotation": torch.randn(7, 4, generator=generator),
        "opacity": torch.tensor([[-20.], [-6.], [-5.], [0.], [2.], [5.], [10.]]),
    }
    # Independent reference follows the upstream concatenation, field order,
    # channel-major SH layout, and CPU float32 opacity predicate.
    attributes = np.concatenate([
        values["xyz"].numpy(), np.zeros((7, 3), dtype=np.uint8),
        values["feature"][:, 0].numpy(),
        values["feature"][:, 1:].transpose(1, 2).reshape(7, 45).numpy(),
        values["opacity"].numpy(), values["scale"].numpy(),
        values["rotation"].numpy(),
    ], axis=1)
    attributes = attributes[values["opacity"].squeeze(-1).sigmoid().numpy() > 0.004]
    path = tmp_path / "vectorized.ply"
    _save_gaussian_ply_vectorized(None, values, path, opacity_threshold=0.004)
    output = PlyData.read(path)["vertex"].data
    assert len(output) == 5
    for column, name in enumerate(output.dtype.names):
        np.testing.assert_array_equal(output[name], attributes[:, column])
