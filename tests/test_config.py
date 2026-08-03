from pathlib import Path

import pytest
import yaml

from common.config import OFFLINE_CHAIN, load_mainline_config


def test_frozen_config_contract():
    config = load_mainline_config("configs/paper_mainline.yaml")
    assert tuple(config.values["method"]["offline_chain"]) == OFFLINE_CHAIN
    assert config.values["prior"]["frozen"] is True
    assert config.values["deployment"]["global_topk"] == 1
    assert config.values["deployment"]["pose_solves"] == 1


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
