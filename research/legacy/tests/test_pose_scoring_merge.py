from __future__ import annotations

from pathlib import Path

import pytest
import torch

from topology.merge_pose_scoring import merge_pose_scoring_shards


def _shard(indices: list[int]) -> dict:
    return {
        "identity": {"core": "/core.pt"},
        "query_indices": indices,
        "query_candidates": [[(index, 0.5)] for index in indices],
        "query_diagnostics": [
            {"query": f"q{index}"} for index in indices
        ],
    }


def test_merge_pose_scoring_shards_restores_query_order(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / "even.pt", tmp_path / "odd.pt"]
    torch.save(_shard([0, 2]), paths[0])
    torch.save(_shard([1, 3]), paths[1])

    merged = merge_pose_scoring_shards(paths)

    assert merged["query_indices"] == [0, 1, 2, 3]
    assert merged["query_candidates"] == [
        [(0, 0.5)],
        [(1, 0.5)],
        [(2, 0.5)],
        [(3, 0.5)],
    ]


def test_merge_pose_scoring_shards_rejects_duplicate_query(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / "left.pt", tmp_path / "right.pt"]
    torch.save(_shard([0, 1]), paths[0])
    torch.save(_shard([1, 2]), paths[1])

    with pytest.raises(ValueError, match="Duplicate query"):
        merge_pose_scoring_shards(paths)
