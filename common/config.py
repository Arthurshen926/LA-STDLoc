"""Strict configuration contracts for frozen V1 and adaptive V2 mainlines."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from common.hashing import canonical_json, sha256_file


SCHEMA = "lafgs_paper_mainline"
VERSIONS = frozenset({1, 2})
OFFLINE_CHAIN = (
    "frozen_rgb_gaussian_prior",
    "native_superpoint_mapping_observations",
    "kcs_gwff_initialization",
    "wide_scaffold_self_localization_reconstruction",
    "track_first_cross_view_tracks",
    "robust_triangulation",
    "gaussian_raster_provenance",
    "track_centric_localization_anchor_pool",
    "localization_topology_distillation",
    "compact_map_self_localization_metric_refresh",
    "compact_single_descriptor_map",
)
TOP_LEVEL_KEYS = frozenset(
    {
        "schema",
        "version",
        "method",
        "prior",
        "initialization",
        "reconstruction",
        "deployment",
        "seeds",
        "adaptive",
        "geometry",
    }
)


@dataclass(frozen=True)
class MainlineConfig:
    path: Path
    values: Mapping[str, Any]
    file_sha256: str
    resolved_sha256: str

    def validate(self) -> None:
        values = self.values
        unknown = set(values) - TOP_LEVEL_KEYS
        if unknown:
            raise ValueError(f"unknown paper-mainline config sections: {sorted(unknown)}")
        version = values.get("version")
        if values.get("schema") != SCHEMA or version not in VERSIONS:
            raise ValueError("unsupported paper-mainline config schema")
        method = values.get("method", {})
        expected_status = "frozen" if version == 1 else "adaptive"
        if method.get("status") != expected_status:
            raise ValueError(
                f"V{version} paper-mainline method must be {expected_status}"
            )
        if tuple(method.get("offline_chain", ())) != OFFLINE_CHAIN:
            raise ValueError("offline method chain differs from the frozen contract")
        if values.get("prior", {}).get("frozen") is not True:
            raise ValueError("RGB Gaussian prior must remain frozen")
        if version == 1 and "adaptive" in values:
            raise ValueError("frozen V1 config cannot contain adaptive policy")
        if version == 2:
            adaptive = values.get("adaptive")
            if not isinstance(adaptive, dict):
                raise ValueError("adaptive V2 config requires an adaptive policy")
            if adaptive.get("calibration_split") != "all_mapping_train":
                raise ValueError("adaptive calibration may only use mapping images")
            if float(adaptive.get("task_translation_m", 0)) <= 0:
                raise ValueError("task translation tolerance must be positive")
            if float(adaptive.get("task_rotation_deg", 0)) <= 0:
                raise ValueError("task rotation tolerance must be positive")
            residual_quantile = float(
                adaptive.get("ransac_track_residual_quantile", 0.975)
            )
            if not 0.5 <= residual_quantile < 1.0:
                raise ValueError(
                    "RANSAC track-residual quantile must lie in [0.5, 1)"
                )
            if float(adaptive.get("ransac_reprojection_maximum_px", 12.0)) <= 0:
                raise ValueError("RANSAC residual safety cap must be positive")
        deployment = values.get("deployment", {})
        required = {
            "sparse_frontend": "ulfloc_native_metric",
            "nms": 4,
            "global_topk": 1,
            "max_matches_per_keypoint": 0,
            "max_matches_per_landmark": 0,
            "reprojection_error_px": 12,
            "confidence": 0.99999,
            "maximum_iterations": 100000,
            "minimum_iterations": 1000,
            "solver": "poselib",
            "pose_solves": 1,
        }
        mismatched = {
            key: (deployment.get(key), expected)
            for key, expected in required.items()
            if deployment.get(key) != expected
        }
        if mismatched:
            raise ValueError(f"deployment contract changed: {mismatched}")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "lafgs_resolved_mainline_config",
            "version": int(self.values["version"]),
            "config_path": str(self.path),
            "config_sha256": self.file_sha256,
            "resolved_config_sha256": self.resolved_sha256,
            "offline_chain": list(OFFLINE_CHAIN),
            "deployment": dict(self.values["deployment"]),
        }


def load_mainline_config(path: str | Path) -> MainlineConfig:
    path = Path(path).expanduser().resolve()
    values = yaml.safe_load(path.read_text())
    if not isinstance(values, dict):
        raise ValueError("mainline config must resolve to a mapping")
    config = MainlineConfig(
        path=path,
        values=values,
        file_sha256=sha256_file(path),
        resolved_sha256=__import__("hashlib").sha256(
            canonical_json(values).encode("ascii")
        ).hexdigest(),
    )
    config.validate()
    return config


def materialize_keypoint_factor_config(
    source: str | Path,
    output: str | Path,
    keypoints: int,
) -> Path:
    """Freeze one detector-density factor as a complete mainline config."""
    keypoints = int(keypoints)
    if keypoints <= 0:
        raise ValueError("keypoint factor must be positive")
    values = __import__("copy").deepcopy(load_mainline_config(source).values)
    deployment = values["deployment"]
    deployment["keypoints"] = keypoints
    deployment["keypoint_minimum"] = keypoints
    deployment["keypoint_maximum"] = keypoints
    values["initialization"]["kcs_keypoints"] = keypoints
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(values, sort_keys=False))
    load_mainline_config(output)
    return output


def materialize_deployment_keypoint_config(
    source: str | Path,
    output: str | Path,
    keypoints: int,
) -> Path:
    """Freeze K_deploy while preserving the mapping-evidence configuration."""
    keypoints = int(keypoints)
    if keypoints <= 0:
        raise ValueError("deployment keypoint count must be positive")
    values = __import__("copy").deepcopy(load_mainline_config(source).values)
    deployment = values["deployment"]
    deployment["keypoints"] = keypoints
    deployment["keypoint_minimum"] = keypoints
    deployment["keypoint_maximum"] = keypoints
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(values, sort_keys=False))
    load_mainline_config(output)
    return output


def materialize_surface_track_config(
    source: str | Path,
    output: str | Path,
    *,
    enabled: bool = True,
) -> Path:
    """Materialize the surface-supported track-geometry factor."""
    values = __import__("copy").deepcopy(load_mainline_config(source).values)
    geometry = values.setdefault("geometry", {})
    geometry.update(
        {
            "surface_supported_tracks": bool(enabled),
            "surface_max_correction_depth_sigmas": 4.0,
            "surface_max_weak_information_ratio": 0.25,
            "surface_min_depth_improvement_fraction": 0.10,
            "surface_max_reprojection_increase_px": 0.05,
        }
    )
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(values, sort_keys=False))
    load_mainline_config(output)
    return output


def resolve_reprojection_error_px(
    deployment: Mapping[str, Any], cameras, scene_calibration: Mapping[str, Any] | None = None
) -> float:
    """Resolve the PnP threshold at the evaluated processed resolution."""
    cameras = list(cameras)
    if scene_calibration is not None:
        if scene_calibration.get("schema") != "lafgs_mapping_only_scene_calibration":
            raise ValueError("unsupported scene-calibration schema")
        sources = scene_calibration.get("sources", {})
        uses_test = scene_calibration.get(
            "uses_test_queries", sources.get("uses_test_queries")
        )
        if uses_test is not False:
            raise ValueError("deployment calibration must be mapping-only")
        statistics = scene_calibration.get("statistics", {})
        calibrated_count = statistics.get("query_count")
        if calibrated_count is not None and int(calibrated_count) != len(cameras):
            raise ValueError(
                "scene calibration mapping-query count differs from the dataset"
            )
        calibrated_focal = statistics.get("focal_px")
        if calibrated_focal is not None and cameras:
            focals = sorted(
                math.sqrt(
                    (
                        float(camera.width)
                        / (2.0 * math.tan(float(camera.fov_x) / 2.0))
                    )
                    * (
                        float(camera.height)
                        / (2.0 * math.tan(float(camera.fov_y) / 2.0))
                    )
                )
                for camera in cameras
            )
            dataset_focal = focals[len(focals) // 2]
            if not math.isclose(
                dataset_focal,
                float(calibrated_focal),
                rel_tol=0.01,
                abs_tol=1e-3,
            ):
                raise ValueError(
                    "scene calibration focal scale differs from the dataset"
                )
        parameters = scene_calibration.get("parameters", {})
        value = float(parameters.get("ransac_reprojection_px", 0.0))
        if not math.isfinite(value) or value <= 0:
            raise ValueError("scene calibration has no valid RANSAC threshold")
        return value
    angular_tangent = deployment.get("reprojection_error_angular_tangent")
    fraction = deployment.get("reprojection_error_diagonal_fraction")
    if not cameras:
        return float(deployment["reprojection_error_px"])
    if angular_tangent is not None:
        focals = sorted(
            math.sqrt(
                (float(camera.width) / (2.0 * math.tan(float(camera.fov_x) / 2.0)))
                * (float(camera.height) / (2.0 * math.tan(float(camera.fov_y) / 2.0)))
            )
            for camera in cameras
        )
        return max(
            2.0,
            float(angular_tangent) * focals[len(focals) // 2],
        )
    if fraction is None:
        return float(deployment["reprojection_error_px"])
    diagonals = sorted(
        math.hypot(float(camera.width), float(camera.height))
        for camera in cameras
    )
    return max(2.0, float(fraction) * diagonals[len(diagonals) // 2])


def load_scene_calibration(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("scene calibration must resolve to a mapping")
    return payload


def resolve_keypoint_count(deployment: Mapping[str, Any], cameras) -> int:
    """Keep detector density comparable while retaining a safety floor."""
    reference = deployment.get("keypoint_reference_area_px")
    cameras = list(cameras)
    if reference is None or not cameras:
        return int(deployment["keypoints"])
    areas = sorted(int(camera.width) * int(camera.height) for camera in cameras)
    count = round(
        int(deployment["keypoints"])
        * areas[len(areas) // 2]
        / float(reference)
    )
    return max(
        int(deployment.get("keypoint_minimum", 1)),
        min(int(deployment.get("keypoint_maximum", count)), count),
    )
