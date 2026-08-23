from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pytest
import torch

from common.hashing import sha256_file
from common.v6_contracts import ordered_query_registry_sha256
from common.v6_pipeline_contract import FEEDBACK_CALIBRATION_BINDING_SCHEMA
from scripts import bind_v6_feedback_calibration as binder


_OBSERVATION_SHA256 = "a" * 64
_QUERY_NAMES = ["seq1/frame00001.png", "seq2/frame00002.png"]


def _write_map(path: Path, *, observation_sha256: str = _OBSERVATION_SHA256) -> str:
    torch.save(
        {
            "schema": "lafgs_materialized_anchor_map",
            "v6_mapping_query_names": _QUERY_NAMES,
            "provenance": {
                "uses_source_mapping_rgb": False,
                "uses_test_queries": False,
                "mapping_source": "gaussian_render_valid_projective_v6",
                "v6_observation_cache_sha256": observation_sha256,
            },
        },
        path,
    )
    return sha256_file(path)


def _write_calibration(path: Path, *, query_count: int = 2) -> str:
    path.write_text(
        json.dumps(
            {
                "schema": "lafgs_mapping_only_scene_calibration",
                "version": 2,
                "sources": {
                    "uses_source_mapping_rgb": False,
                    "uses_test_queries": False,
                    "mapping_source": "gaussian_render",
                },
                "parameters": {"ransac_reprojection_px": 11.954343111400277},
                "statistics": {"query_count": query_count},
            }
        )
    )
    return sha256_file(path)


def _arguments(tmp_path: Path) -> Namespace:
    map_path = tmp_path / "map.pt"
    calibration_path = tmp_path / "scene_calibration.json"
    return Namespace(
        map=map_path,
        expected_map_sha256=_write_map(map_path),
        observation_cache_sha256=_OBSERVATION_SHA256,
        scene_calibration=calibration_path,
        expected_scene_calibration_sha256=_write_calibration(calibration_path),
        output=tmp_path / "binding.json",
    )


def _cli(arguments: Namespace) -> list[str]:
    return [
        "--map",
        str(arguments.map),
        "--expected-map-sha256",
        arguments.expected_map_sha256,
        "--observation-cache-sha256",
        arguments.observation_cache_sha256,
        "--scene-calibration",
        str(arguments.scene_calibration),
        "--expected-scene-calibration-sha256",
        arguments.expected_scene_calibration_sha256,
        "--output",
        str(arguments.output),
    ]


def test_binding_cli_binds_registry_without_loading_observation_cache(
    tmp_path: Path,
    monkeypatch,
):
    arguments = _arguments(tmp_path)
    real_load = torch.load
    loaded_paths = []

    def tracked_load(path, *args, **kwargs):
        loaded_paths.append(Path(path).resolve())
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(binder.torch, "load", tracked_load)

    binder.main(_cli(arguments))

    payload = json.loads(arguments.output.read_text())
    assert loaded_paths == [arguments.map.resolve()]
    assert payload == {
        "schema": FEEDBACK_CALIBRATION_BINDING_SCHEMA,
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "map_sha256": arguments.expected_map_sha256,
        "observation_cache_sha256": _OBSERVATION_SHA256,
        "calibration_sha256": arguments.expected_scene_calibration_sha256,
        "ordered_query_registry_sha256": ordered_query_registry_sha256(_QUERY_NAMES),
        "query_count": len(_QUERY_NAMES),
    }


def test_binding_rejects_map_with_different_observation_lineage(tmp_path: Path):
    arguments = _arguments(tmp_path)
    arguments.expected_map_sha256 = _write_map(
        arguments.map,
        observation_sha256="b" * 64,
    )

    with pytest.raises(ValueError, match="observation cache SHA lineage differ"):
        binder.build_binding(arguments)


def test_binding_rejects_calibration_query_count_mismatch(tmp_path: Path):
    arguments = _arguments(tmp_path)
    arguments.expected_scene_calibration_sha256 = _write_calibration(
        arguments.scene_calibration,
        query_count=3,
    )

    with pytest.raises(ValueError, match="calibration and map query counts differ"):
        binder.build_binding(arguments)


def test_binding_output_is_immutable(tmp_path: Path):
    arguments = _arguments(tmp_path)
    arguments.output.write_text("existing\n")

    with pytest.raises(FileExistsError):
        binder.main(_cli(arguments))
