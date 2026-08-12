from pathlib import Path

import pytest
import yaml

from common.config import (
    OFFLINE_CHAIN,
    load_mainline_config,
    mapping_keypoint_policy_source,
    materialize_keypoint_factor_config,
    materialize_deployment_keypoint_config,
    materialize_mapping_keypoint_config,
    resolve_keypoint_count,
    resolve_mapping_keypoint_count,
    resolve_mapping_nms_radius,
    resolve_reprojection_error_px,
)
from features.superpoint import resolve_superpoint_weights


def test_frozen_config_contract():
    config = load_mainline_config("configs/paper_mainline.yaml")
    assert tuple(config.values["method"]["offline_chain"]) == OFFLINE_CHAIN
    assert config.values["prior"]["frozen"] is True
    assert config.values["deployment"]["global_topk"] == 1
    assert config.values["deployment"]["nms"] == 4
    assert config.values["deployment"]["pose_solves"] == 1
    assert config.values["version"] == 2
    assert config.values["adaptive"]["calibration_split"] == "all_mapping_train"
    assert config.values["adaptive"]["ransac_track_residual_quantile"] == 0.975


def test_legacy_frozen_config_remains_available():
    config = load_mainline_config("configs/paper_mainline_frozen_v1.yaml")
    assert config.values["version"] == 1
    assert config.values["reconstruction"]["track_core"] == 8000
    assert config.values["deployment"]["reprojection_error_px"] == 12


def test_adaptive_keypoint_density_has_a_safety_floor():
    class Camera:
        width = 640
        height = 480

    deployment = load_mainline_config(
        "configs/paper_mainline.yaml"
    ).values["deployment"]
    assert resolve_keypoint_count(deployment, [Camera()]) == 1024


def test_keypoint_factor_config_locks_mapping_and_deployment_density(tmp_path):
    path = materialize_keypoint_factor_config(
        "configs/paper_mainline.yaml", tmp_path / "k2048.yaml", 2048
    )
    values = load_mainline_config(path).values
    assert values["initialization"]["kcs_keypoints"] == 2048
    assert values["deployment"]["keypoints"] == 2048
    assert values["deployment"]["keypoint_minimum"] == 2048
    assert values["deployment"]["keypoint_maximum"] == 2048
    assert values["mapping"]["keypoint_minimum"] == 2048
    assert values["mapping"]["keypoint_maximum"] == 2048


def test_deployment_keypoint_config_preserves_evidence_density(tmp_path):
    source = load_mainline_config("configs/paper_mainline.yaml").values
    path = materialize_deployment_keypoint_config(
        "configs/paper_mainline.yaml", tmp_path / "deploy2048.yaml", 2048
    )
    values = load_mainline_config(path).values
    assert values["initialization"]["kcs_keypoints"] == source["initialization"][
        "kcs_keypoints"
    ]
    assert values["deployment"]["keypoints"] == 2048
    assert values["deployment"]["keypoint_minimum"] == 2048
    assert values["deployment"]["keypoint_maximum"] == 2048
    assert values["mapping"] == source["mapping"]


def test_mapping_keypoint_config_preserves_deployment_density(tmp_path):
    source = load_mainline_config("configs/paper_mainline.yaml").values
    path = materialize_mapping_keypoint_config(
        "configs/paper_mainline.yaml", tmp_path / "mapping2048.yaml", 2048
    )
    values = load_mainline_config(path).values
    assert values["mapping"]["keypoints"] == 2048
    assert values["mapping"]["keypoint_minimum"] == 2048
    assert values["mapping"]["keypoint_maximum"] == 2048
    assert values["mapping"]["nms"] == 4
    assert values["deployment"] == source["deployment"]
    assert resolve_mapping_keypoint_count(values, []) == 2048
    assert resolve_mapping_nms_radius(values) == 4


def test_adaptive_mapping_density_preserves_legacy_resolution():
    class Camera:
        width = 640
        height = 480

    values = load_mainline_config("configs/paper_mainline.yaml").values
    assert values["mapping"]["keypoints"] == 2048
    assert resolve_mapping_keypoint_count(values, [Camera()]) == 1024
    assert (
        mapping_keypoint_policy_source(values)
        == "independent_area_adaptive_mapping_config"
    )


def test_legacy_mapping_policy_source_does_not_report_fixed():
    values = load_mainline_config("configs/paper_mainline.yaml").values
    values.pop("mapping")
    assert (
        mapping_keypoint_policy_source(values)
        == "independent_legacy_deployment_policy"
    )


def test_deployment_pnp_threshold_follows_focal_scale():
    class ReferenceCamera:
        width = 1920
        height = 1080
        fov_x = 2.0 * __import__("math").atan(width / (2.0 * 1672.028076171875))
        fov_y = 2.0 * __import__("math").atan(height / (2.0 * 1672.028076171875))

    class WideCamera:
        width = 1920
        height = 1080
        fov_x = 2.0 * __import__("math").atan(width / (2.0 * 836.0140380859375))
        fov_y = 2.0 * __import__("math").atan(height / (2.0 * 836.0140380859375))

    deployment = load_mainline_config(
        "configs/paper_mainline.yaml"
    ).values["deployment"]
    reference = resolve_reprojection_error_px(deployment, [ReferenceCamera()])
    wide = resolve_reprojection_error_px(deployment, [WideCamera()])
    assert reference == pytest.approx(12.0)
    assert wide == pytest.approx(6.0)


def test_deployment_prefers_mapping_only_scene_calibration():
    class Camera:
        width = 1920
        height = 1080
        fov_x = 2.0 * __import__("math").atan(width / (2.0 * 1672.028076171875))
        fov_y = 2.0 * __import__("math").atan(height / (2.0 * 1672.028076171875))

    deployment = load_mainline_config(
        "configs/paper_mainline.yaml"
    ).values["deployment"]
    calibration = {
        "schema": "lafgs_mapping_only_scene_calibration",
        "sources": {"uses_test_queries": False},
        "parameters": {"ransac_reprojection_px": 7.25},
        "statistics": {"query_count": 1, "focal_px": 1672.028076171875},
    }
    assert resolve_reprojection_error_px(deployment, [Camera()], calibration) == 7.25
    calibration["sources"]["uses_test_queries"] = True
    with pytest.raises(ValueError, match="mapping-only"):
        resolve_reprojection_error_px(deployment, [Camera()], calibration)


def test_deployment_rejects_calibration_from_another_scene():
    class Camera:
        width = 640
        height = 480
        fov_x = fov_y = 1.0

    deployment = load_mainline_config(
        "configs/paper_mainline.yaml"
    ).values["deployment"]
    calibration = {
        "schema": "lafgs_mapping_only_scene_calibration",
        "sources": {"uses_test_queries": False},
        "parameters": {"ransac_reprojection_px": 7.25},
        "statistics": {"query_count": 2, "focal_px": 500.0},
    }
    with pytest.raises(ValueError, match="query count"):
        resolve_reprojection_error_px(deployment, [Camera()], calibration)


def test_unknown_config_section_fails_closed(tmp_path: Path):
    values = yaml.safe_load(Path("configs/paper_mainline.yaml").read_text())
    values["historical_ablation"] = {}
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="unknown"):
        load_mainline_config(path)


def test_all_declared_packages_are_present():
    root = Path(__file__).resolve().parents[1]
    declared = {
        "common",
        "data",
        "evaluation",
        "evidence",
        "features",
        "localization",
        "map_learning",
        "priors",
        "topology",
        "visualization",
    }
    for package in declared:
        assert (root / package / "__init__.py").is_file()


def test_missing_superpoint_weight_fails_with_download_guidance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("LAFGS_SUPERPOINT_WEIGHTS", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    packaged = Path(__file__).resolve().parents[1] / "features/weights/superpoint_v1.pth"
    if packaged.is_file():
        pytest.skip("source checkout still has a private local weight")
    with pytest.raises(FileNotFoundError, match="not redistributed"):
        resolve_superpoint_weights()
