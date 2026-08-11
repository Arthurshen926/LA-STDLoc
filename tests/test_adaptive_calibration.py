import math

import torch
import pytest

from common.calibration import (
    REFERENCE_EFFECTIVE_BASELINE_M,
    calibrate_scene,
    derive_adaptive_parameters,
    derive_mapping_statistics,
    write_query_calibration_sidecar,
)


def _synthetic_scene(scale=1.0, image_scale=1.0):
    queries = {}
    names = []
    for index, center_x in enumerate((0.0, 1.0, 2.0)):
        name = f"frame{index}.png"
        names.append(name)
        pose = torch.eye(4)
        pose[0, 3] = -center_x * scale
        queries[name] = {
            "pose_w2c": pose,
            "native_input_hw": [
                round(1080 * image_scale),
                round(1920 * image_scale),
            ],
            "native_K": torch.tensor(
                [
                    [1672 * image_scale, 0, 960 * image_scale],
                    [0, 1672 * image_scale, 540 * image_scale],
                    [0, 0, 1],
                ]
            ),
            "native_keypoints": torch.zeros((2000, 2)),
        }
    payload = {
        "query_names": names,
        "tracks": {
            "track_index": torch.tensor([0, 0, 0, 1, 1]),
            "query_index": torch.tensor([0, 1, 2, 0, 2]),
            "keypoint_index": torch.tensor([0, 1, 2, 3, 4]),
            "confidence": torch.ones(5),
        },
        "track_geometry": {
            "triangulated": torch.tensor([True, True]),
            "triangulated_xyz": torch.tensor(
                [[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]]
            )
            * scale,
        },
    }
    return {"queries": queries}, payload


def test_metric_thresholds_follow_mapping_sim3_scale():
    query, payload = _synthetic_scene()
    scaled_query, scaled_payload = _synthetic_scene(scale=0.1)
    base = derive_adaptive_parameters(
        derive_mapping_statistics(query, payload)
    )
    scaled = derive_adaptive_parameters(
        derive_mapping_statistics(scaled_query, scaled_payload)
    )
    assert math.isclose(
        scaled.dependency_voxel_m / base.dependency_voxel_m,
        0.1,
        rel_tol=1e-5,
    )
    assert math.isclose(
        scaled.track_covariance_trace_m2
        / base.track_covariance_trace_m2,
        0.01,
        rel_tol=1e-5,
    )
    assert scaled.task_translation_m == base.task_translation_m == 0.05
    assert scaled.task_rotation_deg == base.task_rotation_deg == 5.0


def test_reference_track_span_resolves_unit_metric_scale():
    query, payload = _synthetic_scene(
        scale=REFERENCE_EFFECTIVE_BASELINE_M / 2.0
    )
    parameters = derive_adaptive_parameters(
        derive_mapping_statistics(query, payload)
    )
    assert math.isclose(parameters.metric_scale, 1.0, rel_tol=1e-6)
    assert math.isclose(parameters.dependency_voxel_m, 0.5, rel_tol=1e-6)
    assert math.isclose(
        parameters.evidence_depth_abs_tolerance_m, 0.05, rel_tol=1e-6
    )


def test_pixel_thresholds_follow_processed_resolution():
    query, payload = _synthetic_scene()
    small_query, small_payload = _synthetic_scene(image_scale=0.5)
    base = derive_adaptive_parameters(
        derive_mapping_statistics(query, payload)
    )
    small = derive_adaptive_parameters(
        derive_mapping_statistics(small_query, small_payload)
    )
    assert math.isclose(
        small.positive_radius_px / base.positive_radius_px,
        0.5,
        rel_tol=1e-3,
    )
    assert math.isclose(
        small.ransac_reprojection_px / base.ransac_reprojection_px,
        0.5,
        rel_tol=1e-3,
    )


def test_geometric_pixel_thresholds_follow_focal_not_only_diagonal():
    query, payload = _synthetic_scene()
    wide_query, wide_payload = _synthetic_scene()
    for record in wide_query["queries"].values():
        record["native_K"] = record["native_K"].clone()
        record["native_K"][0, 0] *= 0.5
        record["native_K"][1, 1] *= 0.5
    base = derive_adaptive_parameters(derive_mapping_statistics(query, payload))
    wide = derive_adaptive_parameters(
        derive_mapping_statistics(wide_query, wide_payload)
    )
    assert base.image_pixel_scale == wide.image_pixel_scale
    assert math.isclose(wide.angular_pixel_scale / base.angular_pixel_scale, 0.5)
    assert math.isclose(wide.ransac_reprojection_px / base.ransac_reprojection_px, 0.5)


def test_training_steps_are_query_exposure_epochs():
    query, payload = _synthetic_scene()
    parameters = derive_adaptive_parameters(
        derive_mapping_statistics(query, payload),
        {"stage_a_query_epochs": 2.0, "metric_query_epochs": 0.5},
    )
    assert parameters.stage_a_steps == 6
    assert parameters.metric_steps == 2


def test_scene_calibration_uses_lightweight_query_sidecar(tmp_path):
    query, payload = _synthetic_scene()
    query_path = tmp_path / "query_cache.pt"
    track_path = tmp_path / "tracks.pt"
    # The full cache intentionally lacks query records. Calibration can only
    # succeed if the validated sidecar is used.
    torch.save({"not_queries": True}, query_path)
    torch.save(payload, track_path)
    write_query_calibration_sidecar(query_path, query)
    calibration = calibrate_scene(query_path, track_path)
    assert calibration["statistics"]["query_count"] == 3
    assert calibration["statistics"]["track_count"] == 2


def test_pose_bins_grow_with_mapping_sequence_but_remain_bounded():
    query, payload = _synthetic_scene()
    small = derive_adaptive_parameters(
        derive_mapping_statistics(query, payload)
    )
    statistics = derive_mapping_statistics(query, payload)
    large_statistics = type(statistics)(
        **{**statistics.__dict__, "query_count": 4096}
    )
    large = derive_adaptive_parameters(large_statistics)
    assert small.view_bin_count == 2
    assert large.view_bin_count == 8


def test_ransac_threshold_uses_mapping_track_residual_floor():
    query, payload = _synthetic_scene(image_scale=0.5)
    payload["track_geometry"].update(
        {
            "triangulation_reprojection_p90_px": torch.tensor([4.0, 8.0]),
            "track_confidence_level": torch.tensor([2, 2]),
        }
    )
    statistics = derive_mapping_statistics(
        query, payload, track_residual_quantile=0.95
    )
    parameters = derive_adaptive_parameters(
        statistics, {"ransac_reprojection_maximum_px": 12.0}
    )
    expected = float(torch.quantile(torch.tensor([4.0, 8.0]), 0.95))
    assert parameters.ransac_reprojection_px == pytest.approx(expected)
    assert parameters.harm_radius_px == pytest.approx(expected)


def test_ransac_track_floor_is_capped_without_clipping_angular_scale():
    query, payload = _synthetic_scene(image_scale=2.0)
    payload["track_geometry"].update(
        {
            "triangulation_reprojection_p90_px": torch.tensor([40.0, 80.0]),
            "track_confidence_level": torch.tensor([2, 2]),
        }
    )
    parameters = derive_adaptive_parameters(
        derive_mapping_statistics(query, payload),
        {"ransac_reprojection_maximum_px": 12.0},
    )
    assert parameters.ransac_reprojection_px == pytest.approx(24.0, rel=2e-5)
