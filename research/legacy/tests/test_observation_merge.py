from __future__ import annotations

from pathlib import Path

import pytest
import torch

from map_learning.merge_observations import (
    DIAGNOSTIC_KEYS,
    merge_observation_shards,
)


def _shard(indices: list[int]) -> dict:
    diagnostics = {key: len(indices) for key in DIAGNOSTIC_KEYS}
    diagnostics["query_count"] = len(indices)
    return {
        "schema": "lafgs_v9_active_map_complete_positive_teacher",
        "version": 1,
        "anchor_count": 2,
        "query_names": ["q0", "q1", "q2", "q3"],
        "records": [
            {"query_index": index, "value": index} for index in indices
        ],
        "diagnostics": diagnostics,
        "config": {
            "strong_radius_px": 2.0,
            "num_shards": 2,
            "shard_index": 0,
        },
        "anchor_map": "/map.pt",
        "query_cache": "/queries.pt",
        "raster_provenance": "/provenance.pt",
        "track_payload": "/tracks.pt",
    }


def test_merge_observation_shards_restores_query_order(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / "even.pt", tmp_path / "odd.pt"]
    torch.save(_shard([0, 2]), paths[0])
    torch.save(_shard([1, 3]), paths[1])

    merged = merge_observation_shards(paths)

    assert [record["query_index"] for record in merged["records"]] == [
        0,
        1,
        2,
        3,
    ]
    assert merged["diagnostics"]["query_count"] == 4
    assert merged["config"]["num_shards"] == 2
    assert merged["config"]["shard_index"] == -1


def test_merge_observation_shards_rejects_missing_query(
    tmp_path: Path,
) -> None:
    path = tmp_path / "partial.pt"
    torch.save(_shard([0, 2]), path)

    with pytest.raises(ValueError, match="cover every query"):
        merge_observation_shards([path])
