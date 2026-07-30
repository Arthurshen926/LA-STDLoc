import pytest

from scripts.merge_lafgs_dynamic_outcomes import merge_dynamic_outcomes


def _shard(names):
    return {
        "schema": "lafgs_dynamic_self_localization_outcomes",
        "version": 1,
        "anchor_count": 3,
        "map": "map.pt",
        "metric_state": "metric.pt",
        "query_names": names,
        "records": [{"query_name": name} for name in names],
    }


def test_dynamic_merge_uses_reference_order():
    merged = merge_dynamic_outcomes(
        [_shard(["b"]), _shard(["a", "c"])],
        reference_query_names=["a", "b", "c"],
    )
    assert merged["query_names"] == ["a", "b", "c"]
    assert [row["query_name"] for row in merged["records"]] == [
        "a",
        "b",
        "c",
    ]


def test_dynamic_merge_rejects_duplicate_queries():
    with pytest.raises(ValueError, match="duplicate"):
        merge_dynamic_outcomes(
            [_shard(["a"]), _shard(["a"])],
            reference_query_names=["a"],
        )
