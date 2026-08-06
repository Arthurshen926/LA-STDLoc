from pathlib import Path

import pytest
import yaml

from common.config import (
    OFFLINE_CHAIN,
    load_mainline_config,
    resolve_keypoint_count,
)
from features.superpoint import resolve_superpoint_weights


def test_frozen_config_contract():
    config = load_mainline_config("configs/paper_mainline.yaml")
    assert tuple(config.values["method"]["offline_chain"]) == OFFLINE_CHAIN
    assert config.values["prior"]["frozen"] is True
    assert config.values["deployment"]["global_topk"] == 1
    assert config.values["deployment"]["pose_solves"] == 1
    assert config.values["version"] == 2
    assert config.values["adaptive"]["calibration_split"] == "all_mapping_train"


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
