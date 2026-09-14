import torch

from topology.geometry_materializer import (
    GEOMETRY_IMAGE_TRIANGULATED,
    GEOMETRY_SURFACE_INITIALIZED,
    GEOMETRY_SURFACE_REGULARIZED,
    materialize_geometry,
    materialize_legacy_map_geometry,
    materialize_track_geometry_compatibility,
)


def _legacy_deployment_reference(geometry, image_only_mask):
    if "triangulation_image_only_xyz" not in geometry:
        return geometry
    revised = dict(geometry)
    core = torch.as_tensor(image_only_mask).bool()
    replacements = {
        "triangulated_xyz": "triangulation_image_only_xyz",
        "triangulation_covariance_trace": (
            "triangulation_image_only_covariance_trace"
        ),
        "triangulation_covariance_matrix": (
            "triangulation_image_only_covariance_matrix"
        ),
        "triangulation_reprojection_median_px": (
            "triangulation_image_only_reprojection_median_px"
        ),
        "triangulation_reprojection_p90_px": (
            "triangulation_image_only_reprojection_p90_px"
        ),
    }
    for target, source in replacements.items():
        if source not in geometry:
            continue
        value = torch.as_tensor(geometry[target]).clone()
        value[core] = torch.as_tensor(geometry[source])[core]
        revised[target] = value
    return revised


def test_materialize_geometry_separates_three_evidence_modes():
    fallback_xyz = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ]
    )
    output = materialize_geometry(
        {
            "fallback_xyz": fallback_xyz,
            "fallback_covariance": torch.eye(3).repeat(3, 1, 1) * 3.0,
            "track_rows": torch.tensor([0, 1]),
            "track_image_only_xyz": torch.tensor(
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
            ),
            "track_surface_xyz": torch.tensor(
                [[1.1, 0.0, 0.0], [2.1, 0.0, 0.0]]
            ),
            "track_prefer_image_only": torch.tensor([True, False]),
            "track_surface_supported": torch.tensor([True, True]),
            "track_image_only_covariance": torch.eye(3).repeat(2, 1, 1),
            "track_surface_covariance": torch.eye(3).repeat(2, 1, 1) * 2.0,
        }
    )

    torch.testing.assert_close(
        output["xyz"],
        torch.tensor(
            [[1.0, 0.0, 0.0], [2.1, 0.0, 0.0], [30.0, 0.0, 0.0]]
        ),
    )
    assert output["geometry_mode"].tolist() == [
        GEOMETRY_IMAGE_TRIANGULATED,
        GEOMETRY_SURFACE_REGULARIZED,
        GEOMETRY_SURFACE_INITIALIZED,
    ]
    assert output["surface_evidence"].tolist() == [True, True, True]
    assert output["surface_dependence"].tolist() == [False, True, True]
    torch.testing.assert_close(output["covariance"][0], torch.eye(3))
    torch.testing.assert_close(output["covariance"][1], torch.eye(3) * 2.0)
    torch.testing.assert_close(output["covariance"][2], torch.eye(3) * 3.0)


def test_track_compatibility_adapter_is_bitwise_equal_to_v3_p5_policy():
    geometry = {
        "triangulated_xyz": torch.tensor(
            [[1.1, 0.0, 0.0], [2.1, 0.0, 0.0], [3.1, 0.0, 0.0]]
        ),
        "triangulation_image_only_xyz": torch.tensor(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        ),
        "triangulation_covariance_trace": torch.tensor([0.01, 0.02, 0.03]),
        "triangulation_image_only_covariance_trace": torch.tensor(
            [0.10, 0.20, 0.30]
        ),
        "triangulation_covariance_matrix": torch.stack(
            [torch.eye(3) * value for value in (0.01, 0.02, 0.03)]
        ),
        "triangulation_image_only_covariance_matrix": torch.stack(
            [torch.eye(3) * value for value in (0.10, 0.20, 0.30)]
        ),
        "triangulation_reprojection_median_px": torch.tensor([1.0, 2.0, 3.0]),
        "triangulation_image_only_reprojection_median_px": torch.tensor(
            [4.0, 5.0, 6.0]
        ),
        "triangulation_reprojection_p90_px": torch.tensor([7.0, 8.0, 9.0]),
        "triangulation_image_only_reprojection_p90_px": torch.tensor(
            [10.0, 11.0, 12.0]
        ),
        "triangulation_surface_supported": torch.tensor([True, True, False]),
    }
    mask = torch.tensor([True, False, True])
    expected = _legacy_deployment_reference(geometry, mask)
    actual = materialize_track_geometry_compatibility(geometry, mask)

    assert actual.keys() == expected.keys()
    for key in expected:
        if isinstance(expected[key], torch.Tensor):
            assert torch.equal(actual[key], expected[key]), key
        else:
            assert actual[key] == expected[key]
    assert torch.equal(
        geometry["triangulated_xyz"],
        torch.tensor(
            [[1.1, 0.0, 0.0], [2.1, 0.0, 0.0], [3.1, 0.0, 0.0]]
        ),
    )


def test_legacy_map_annotation_preserves_xyz_and_registry_covariance():
    state = {
        "anchor_type": torch.tensor([1, 1, 0]),
        "track_cluster_ids": torch.tensor([0, 1, -1]),
        "anchor_xyz": torch.tensor(
            [[1.0, 0.0, 0.0], [2.1, 0.0, 0.0], [3.0, 0.0, 0.0]]
        ),
        "anchor_position_covariance": torch.eye(3).repeat(3, 1, 1) * 9.0,
    }
    final_covariance = torch.stack(
        (torch.eye(3) * 0.01, torch.eye(3) * 0.02)
    )
    payload = {
        "track_geometry": {
            "triangulated_xyz": torch.tensor(
                [[1.1, 0.0, 0.0], [2.1, 0.0, 0.0]]
            ),
            "triangulation_image_only_xyz": torch.tensor(
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
            ),
            "triangulation_covariance_matrix": final_covariance,
            "triangulation_image_only_covariance_matrix": torch.eye(3).repeat(
                2, 1, 1
            ),
            "triangulation_surface_supported": torch.tensor([True, True]),
        }
    }
    output = materialize_legacy_map_geometry(state, payload)

    assert torch.equal(output["xyz"], state["anchor_xyz"])
    assert output["geometry_mode"].tolist() == [
        GEOMETRY_IMAGE_TRIANGULATED,
        GEOMETRY_SURFACE_REGULARIZED,
        GEOMETRY_SURFACE_INITIALIZED,
    ]
    assert output["surface_dependence"].tolist() == [False, True, True]
    assert torch.equal(output["covariance"][:2], final_covariance)
    assert torch.equal(
        output["covariance"][2], state["anchor_position_covariance"][2]
    )
