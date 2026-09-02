from pathlib import Path

import pytest
import yaml

from common.config import ANYGSLOC_SCHEMA, load_mainline_config


CONFIG = Path("configs/anygsloc_mainline.yaml")


def test_anygsloc_mainline_is_mapping_only_and_frozen():
    config = load_mainline_config(CONFIG)
    assert config.values["schema"] == ANYGSLOC_SCHEMA
    assert config.values["method"]["offline_self_localization_feedback"] is False
    assert config.values["method"]["test_query_map_adaptation"] is False
    assert config.values["online_refinement"]["optional"] is True
    assert config.manifest()["offline_self_localization_feedback"] is False


def test_anygsloc_mainline_rejects_feedback(tmp_path):
    values = yaml.safe_load(CONFIG.read_text())
    values["method"]["offline_self_localization_feedback"] = True
    path = tmp_path / "feedback.yaml"
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="forbidden"):
        load_mainline_config(path)


def test_anygsloc_mainline_rejects_non_optional_refinement(tmp_path):
    values = yaml.safe_load(CONFIG.read_text())
    values["online_refinement"]["optional"] = False
    path = tmp_path / "required_refinement.yaml"
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="optional"):
        load_mainline_config(path)
