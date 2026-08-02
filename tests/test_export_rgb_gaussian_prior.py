import json

import numpy as np
from plyfile import PlyData, PlyElement

from scripts.export_rgb_gaussian_prior import export_rgb_prior


def test_export_rgb_prior_removes_localization_state(tmp_path):
    names = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
        "loc_feature_0",
        "loc_anchor_offset_0",
    ]
    source = np.zeros(3, dtype=[(name, "<f4") for name in names])
    source["rot_0"] = 1.0
    source["x"][2] = np.nan
    input_ply = tmp_path / "input.ply"
    PlyData([PlyElement.describe(source, "vertex")], text=False).write(
        input_ply
    )
    output_model = tmp_path / "rgb_prior"

    manifest = export_rgb_prior(
        input_ply,
        output_model,
        gaussian_type="3dgs",
        sh_degree=0,
        source_path=tmp_path / "dataset",
        images="images",
        longest_edge=0,
        iteration=30000,
        prior_kind="feature_stripped",
        prior_training_used_feature_loss=True,
    )

    output_ply = (
        output_model
        / "point_cloud"
        / "iteration_30000"
        / "point_cloud.ply"
    )
    output_names = set(
        PlyData.read(output_ply).elements[0].data.dtype.names or ()
    )
    assert "loc_feature_0" not in output_names
    assert "loc_anchor_offset_0" not in output_names
    assert manifest["localization_state_present"] is False
    assert manifest["source_localization_state_present"] is True
    assert manifest["removed_localization_property_count"] == 2
    assert manifest["source_primitive_count"] == 3
    assert manifest["primitive_count"] == 2
    assert manifest["dropped_nonfinite_primitive_count"] == 1
    assert manifest["exported_ply_sha256"]
    assert json.loads(
        (output_model / "rgb_prior_manifest.json").read_text()
    ) == manifest


def test_export_rgb_prior_accepts_native_2dgs_two_scale_schema(tmp_path):
    names = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
        "loc_feature_0",
    ]
    source = np.zeros(2, dtype=[(name, "<f4") for name in names])
    source["rot_0"] = 1.0
    input_ply = tmp_path / "input_2dgs.ply"
    PlyData([PlyElement.describe(source, "vertex")], text=False).write(
        input_ply
    )

    output_model = tmp_path / "rgb_2dgs_prior"
    manifest = export_rgb_prior(
        input_ply,
        output_model,
        gaussian_type="2dgs",
        sh_degree=0,
        source_path=tmp_path / "dataset",
        images="processed",
        longest_edge=0,
        iteration=30000,
        prior_kind="rgb_only",
        prior_training_used_feature_loss=False,
        white_background=False,
    )

    output_names = set(
        PlyData.read(
            output_model
            / "point_cloud"
            / "iteration_30000"
            / "point_cloud.ply"
        )
        .elements[0]
        .data.dtype.names
        or ()
    )
    assert {"scale_0", "scale_1"} <= output_names
    assert "scale_2" not in output_names
    assert "loc_feature_0" not in output_names
    assert manifest["gaussian_type"] == "2dgs"
    assert manifest["prior_training_used_feature_loss"] is False
    assert manifest["white_background"] is False
    assert "white_background=False" in (output_model / "cfg_args").read_text()
