from __future__ import annotations

from pathlib import Path

import pytest
import torch

from priors.merge_provenance import merge_provenance_shards


def _shard(indices: list[int]) -> dict:
    return {
        "schema": "lafgs_native_keypoint_raster_provenance",
        "version": 1,
        "anchor_map": "/map.pt",
        "query_cache": "/queries.pt",
        "gaussian_ply": "/prior.ply",
        "query_names": ["q0", "q1", "q2", "q3"],
        "primitive_count": 5,
        "source_universe": torch.tensor([0, 2, 4]),
        "anchor_source_offsets": torch.tensor([0, 1, 3]),
        "anchor_source_primitive_ids": torch.tensor([0, 2, 4]),
        "anchor_source_weights": torch.tensor([1.0, 0.4, 0.6]),
        "records": [
            {"query_index": index, "value": index} for index in indices
        ],
        "config": {"num_shards": 2, "shard_index": 0},
    }


def test_merge_provenance_shards_restores_query_order(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / "even.pt", tmp_path / "odd.pt"]
    torch.save(_shard([0, 2]), paths[0])
    torch.save(_shard([1, 3]), paths[1])

    merged = merge_provenance_shards(paths)

    assert [record["query_index"] for record in merged["records"]] == [
        0,
        1,
        2,
        3,
    ]
    assert merged["config"]["num_shards"] == 2
    assert merged["config"]["shard_index"] == -1


def test_merge_provenance_shards_rejects_duplicate_query(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / "left.pt", tmp_path / "right.pt"]
    torch.save(_shard([0, 1]), paths[0])
    torch.save(_shard([1, 2, 3]), paths[1])

    with pytest.raises(ValueError, match="Duplicate query"):
        merge_provenance_shards(paths)
