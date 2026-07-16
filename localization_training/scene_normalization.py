from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SceneNormalization:
    camera_count: int
    point_count: int
    scene_radius_m: float
    camera_spacing_m: float
    surfel_radius_p50_m: float
    surfel_radius_p90_m: float
    pixel_scale: float
    field_steps: int
    bootstrap_steps: int
    joint_steps: int
    pnp_start_steps: int
    geometry_start_steps: int
    mvinit_views: int
    mvinit_min_observations: int
    full_bank_landmark_count: int
    translation_scale_m: float
    pnp_voxel_size_m: float
    surfel_tangent_bound_m: float
    surfel_normal_bound_m: float
    geometry_xyz_lr: float
    loc_anchor_lr: float
    landmark_count: int
    train_detect_num: int
    eval_detect_num: int
    min_loc_observations: int
    detector_steps: int
    candidate_steps: int
    nms_radius_px: int
    positive_radius_px: float
    negative_radius_px: float
    reprojection_sigma_px: float
    inlier_sigma_px: float
    residual_clip_px: float

    def to_dict(self):
        return asdict(self)


def _round_to_multiple(value, multiple):
    return int(math.ceil(float(value) / int(multiple)) * int(multiple))


def _nearest_power_of_two(value, minimum, maximum):
    value = max(float(value), 1.0)
    exponent = int(round(math.log2(value)))
    return int(min(maximum, max(minimum, 2**exponent)))


def _camera_spacing(positions):
    if positions.shape[0] < 2:
        return 1.0
    delta = positions[:, None, :] - positions[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(distance, np.inf)
    nearest = np.min(distance, axis=1)
    finite = nearest[np.isfinite(nearest) & (nearest > 0)]
    return float(np.median(finite)) if finite.size else 1.0


def compute_scene_normalization(
    camera_positions,
    point_count,
    surfel_radii,
    image_size,
    *,
    target_longest_edge=640,
    field_steps=30000,
    detector_epochs=32,
    candidate_epochs=24,
    max_similarity_pairs=4096 * 16384,
    full_bank_max_similarity_pairs=512 * 16384,
):
    positions = np.asarray(camera_positions, dtype=np.float64).reshape(-1, 3)
    if positions.shape[0] == 0:
        raise ValueError("camera_positions must not be empty")
    point_count = int(point_count)
    if point_count <= 0:
        raise ValueError("point_count must be positive")

    center = np.median(positions, axis=0)
    radii = np.linalg.norm(positions - center, axis=1)
    positive_radii = radii[np.isfinite(radii) & (radii > 0)]
    scene_radius = (
        float(np.quantile(positive_radii, 0.9)) if positive_radii.size else 1.0
    )
    camera_spacing = _camera_spacing(positions)

    surfel_radii = np.asarray(surfel_radii, dtype=np.float64).reshape(-1)
    surfel_radii = surfel_radii[
        np.isfinite(surfel_radii) & (surfel_radii > 0)
    ]
    if surfel_radii.size:
        surfel_p50 = float(np.quantile(surfel_radii, 0.5))
        surfel_p90 = float(np.quantile(surfel_radii, 0.9))
    else:
        surfel_p50 = scene_radius / 1000.0
        surfel_p90 = scene_radius / 500.0

    metric_eps = max(scene_radius, 1.0) * 1e-9
    metric_reference = math.sqrt(
        max(scene_radius, metric_eps) * max(camera_spacing, metric_eps)
    )
    translation_scale = 0.02 * metric_reference
    translation_scale = min(
        max(translation_scale, 0.001 * scene_radius), 0.02 * scene_radius
    )

    tangent_bound = max(2.0 * surfel_p50, 0.0025 * scene_radius)
    tangent_bound = min(tangent_bound, 0.02 * scene_radius)
    normal_bound = 0.2 * tangent_bound
    pnp_voxel = max(8.0 * surfel_p90, scene_radius / 64.0)
    pnp_voxel = min(pnp_voxel, scene_radius / 8.0)

    width, height = (int(image_size[0]), int(image_size[1]))
    longest_edge = max(width, height, 1)
    effective_edge = min(longest_edge, int(target_longest_edge))
    pixel_scale = max(float(effective_edge) / 640.0, 0.25)
    inlier_sigma = 4.0 * pixel_scale

    camera_count = int(positions.shape[0])
    landmark_count = _nearest_power_of_two(
        math.sqrt(point_count * camera_count), 8192, 32768
    )
    eval_detect_num = 4096
    # Candidate-set training must observe the same query cardinality as the
    # final frontend; otherwise quota, dustbin, and Fisher objectives optimize
    # a different order-statistics problem.
    train_detect_num = eval_detect_num

    detector_steps = min(
        int(field_steps), max(10000, _round_to_multiple(camera_count * detector_epochs, 500))
    )
    candidate_steps = min(
        int(field_steps), max(5000, _round_to_multiple(camera_count * candidate_epochs, 500))
    )
    mvinit_views = min(
        camera_count, max(64, int(math.ceil(8.0 * math.sqrt(camera_count))))
    )
    min_observations = max(2, min(8, int(round(math.log2(camera_count) / 2.0))))

    field_steps = int(field_steps)
    return SceneNormalization(
        camera_count=camera_count,
        point_count=point_count,
        scene_radius_m=scene_radius,
        camera_spacing_m=camera_spacing,
        surfel_radius_p50_m=surfel_p50,
        surfel_radius_p90_m=surfel_p90,
        pixel_scale=pixel_scale,
        field_steps=field_steps,
        bootstrap_steps=max(1, int(round(0.10 * field_steps))),
        joint_steps=max(2, int(round(0.50 * field_steps))),
        pnp_start_steps=max(1, int(round(0.10 * field_steps))),
        geometry_start_steps=max(2, int(round(0.50 * field_steps))),
        mvinit_views=mvinit_views,
        mvinit_min_observations=max(1, min(3, min_observations - 1)),
        full_bank_landmark_count=min(
            point_count,
            max(8192, int(full_bank_max_similarity_pairs // 512)),
        ),
        translation_scale_m=translation_scale,
        pnp_voxel_size_m=pnp_voxel,
        surfel_tangent_bound_m=tangent_bound,
        surfel_normal_bound_m=normal_bound,
        # Adam applies raw-XYZ steps throughout the geometry stage. At this
        # scale, a same-sign update over the full second half of training is
        # still only about one centimetre. The bounded localization anchor can
        # adapt faster because it cannot leave the surfel support region.
        geometry_xyz_lr=tangent_bound * 1e-5,
        loc_anchor_lr=tangent_bound * 1e-4,
        landmark_count=landmark_count,
        train_detect_num=train_detect_num,
        eval_detect_num=eval_detect_num,
        min_loc_observations=min_observations,
        detector_steps=detector_steps,
        candidate_steps=candidate_steps,
        nms_radius_px=max(1, int(round(2.0 * pixel_scale))),
        positive_radius_px=2.0 * pixel_scale,
        negative_radius_px=6.0 * pixel_scale,
        reprojection_sigma_px=1.0 * pixel_scale,
        inlier_sigma_px=inlier_sigma,
        residual_clip_px=3.0 * inlier_sigma,
    )
