import numpy as np
from plyfile import PlyData, PlyElement
import torch

from topology.anchor_covariance import (
    COVARIANCE_GAUSSIAN_SURFACE_PRIOR,
    COVARIANCE_TRIANGULATION,
    attach_gaussian_prior_covariance,
)
from topology.anchor_registry import SCHEMA as REGISTRY_SCHEMA


def test_gaussian_covariance_enrichment_is_auxiliary_and_calibrated(tmp_path):
    values = np.zeros(
        2,
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("scale_0", "f4"), ("scale_1", "f4"),
            ("rot_0", "f4"), ("rot_1", "f4"),
            ("rot_2", "f4"), ("rot_3", "f4"),
        ],
    )
    values["x"] = [0.0, 2.0]
    values["scale_0"] = np.log([0.3, 0.3])
    values["scale_1"] = np.log([0.4, 0.4])
    values["rot_0"] = 1.0
    ply = tmp_path / "prior.ply"
    PlyData([PlyElement.describe(values, "vertex")], text=False).write(ply)
    registry = {
        "schema": REGISTRY_SCHEMA,
        "anchor_ids": torch.arange(2),
        "anchor_type": torch.tensor([1, 0]),
        "source_primitive_ids": torch.tensor([0, 1]),
        "anchor_xyz": torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        "anchor_position_covariance": torch.stack(
            (torch.eye(3) * 0.01, torch.full((3, 3), float("nan")))
        ),
        "compatibility": {"localization_tensors_preserved_exactly": True},
    }
    state = {
        "track_centric_reconstruction": {
            "calibration": {
                "parameters": {
                    "surface_max_distance_m": 0.3,
                    "surface_point_plane_m": 0.03,
                }
            }
        }
    }
    output = attach_gaussian_prior_covariance(registry, state, ply)
    torch.testing.assert_close(output["anchor_xyz"], registry["anchor_xyz"])
    torch.testing.assert_close(
        output["anchor_position_covariance"][0],
        registry["anchor_position_covariance"][0],
    )
    torch.testing.assert_close(
        output["anchor_position_covariance"],
        registry["anchor_position_covariance"],
        equal_nan=True,
    )
    expected = torch.diag(torch.tensor([0.01, 0.01, 0.0001]))
    torch.testing.assert_close(
        output["anchor_position_covariance_enriched"][1], expected
    )
    assert output["covariance_source"].tolist() == [
        COVARIANCE_TRIANGULATION,
        COVARIANCE_GAUSSIAN_SURFACE_PRIOR,
    ]
    assert output["gaussian_prior_center_distance_m"][1] == 0.0
    assert not output["covariance_enrichment"]["changes_localization_tensors"]
