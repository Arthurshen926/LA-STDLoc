import math

import torch

from common.calibration import (
    derive_adaptive_parameters,
    derive_mapping_statistics,
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


def test_training_steps_are_query_exposure_epochs():
    query, payload = _synthetic_scene()
    parameters = derive_adaptive_parameters(
        derive_mapping_statistics(query, payload),
        {"stage_a_query_epochs": 2.0, "metric_query_epochs": 0.5},
    )
    assert parameters.stage_a_steps == 6
    assert parameters.metric_steps == 2


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
