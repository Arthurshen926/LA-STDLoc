from pathlib import Path

import pytest

from common.v6_contracts import (
    ASSOCIATION_GRAPH_SCHEMA,
    round_directory,
    require_schema,
    validate_ordered_query_registry,
)


def test_v6_contract_rejects_non_mapping_scope() -> None:
    require_schema(
        {
            "schema": ASSOCIATION_GRAPH_SCHEMA,
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
        },
        ASSOCIATION_GRAPH_SCHEMA,
        label="association",
    )
    with pytest.raises(ValueError):
        require_schema(
            {
                "schema": ASSOCIATION_GRAPH_SCHEMA,
                "uses_source_mapping_rgb": False,
            },
            ASSOCIATION_GRAPH_SCHEMA,
            label="association",
        )


def test_v6_round_and_registry_contracts() -> None:
    assert round_directory(Path("run"), 2) == Path("run/round_2")
    assert validate_ordered_query_registry(["a", "b"]) == ("a", "b")
    with pytest.raises(ValueError):
        validate_ordered_query_registry(["a", "a"])
