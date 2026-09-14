from __future__ import annotations

from pathlib import Path

import pytest
import torch

from evidence.merge_function_graph import (
    COUNTER_KEYS,
    merge_function_graph_shards,
)


def _shard(indices: list[int]) -> dict:
    counters = {
        key: torch.tensor([len(indices), sum(indices)], dtype=torch.int64)
        for key in COUNTER_KEYS
    }
    return {
        "schema": "lafgs_keypoint_function_graph_shard",
        "version": 2,
        "anchor_map": "/map.pt",
        "query_cache": "/queries.pt",
        "anchor_count": 2,
        "query_count_total": 4,
        "query_names": ["q0", "q1", "q2", "q3"],
        "query_indices": torch.tensor(indices, dtype=torch.int32),
        "source_primitive_ids": torch.tensor([1, 2]),
        "track_cluster_ids": torch.tensor([3, 4]),
        "anchor_type": torch.tensor([0, 1], dtype=torch.int8),
        "records": [
            {"query_index": index, "value": index} for index in indices
        ],
        "query_diagnostics": [
            {"query_index": index, "value": index} for index in indices
        ],
        "config": {"num_shards": 2, "shard_index": 0},
        "raster_visibility_enabled": True,
        **counters,
    }


def test_merge_function_graph_shards_restores_query_order(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / "even.pt", tmp_path / "odd.pt"]
    torch.save(_shard([0, 2]), paths[0])
    torch.save(_shard([1, 3]), paths[1])

    merged = merge_function_graph_shards(paths)

    assert merged["query_indices"].tolist() == [0, 1, 2, 3]
    assert [record["query_index"] for record in merged["records"]] == [
        0,
        1,
        2,
        3,
    ]
    assert merged["config"]["num_shards"] == 2
    assert merged["config"]["shard_index"] == -1
    for key in COUNTER_KEYS:
        assert merged[key].tolist() == [4, 6]


def test_merge_function_graph_shards_rejects_missing_query(
    tmp_path: Path,
) -> None:
    path = tmp_path / "partial.pt"
    torch.save(_shard([0, 2]), path)

    with pytest.raises(ValueError, match="cover every query"):
        merge_function_graph_shards([path])
