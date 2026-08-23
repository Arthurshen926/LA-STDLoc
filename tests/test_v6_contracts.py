from pathlib import Path

import pytest

from common.v6_contracts import (
    ASSOCIATION_GRAPH_SCHEMA,
    FEEDBACK_SCHEMA,
    FEEDBACK_VERSION,
    exact_identity_positive_contract,
    round_directory,
    require_exact_identity_positive_contract,
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


def test_v6_feedback_contract_rejects_radius_only_legacy_artifacts() -> None:
    contract = exact_identity_positive_contract()
    require_exact_identity_positive_contract(contract)
    with pytest.raises(ValueError, match="geometry_compatible_nonidentity"):
        require_exact_identity_positive_contract(
            {**contract, "geometry_compatible_nonidentity": "positive"}
        )
    with pytest.raises(ValueError, match="schema differs"):
        require_schema(
            {
                "schema": "self_localization_feedback_v1",
                "uses_source_mapping_rgb": False,
                "uses_test_queries": False,
            },
            FEEDBACK_SCHEMA,
            label="legacy feedback",
        )
    with pytest.raises(ValueError, match="positive identity contract is missing"):
        require_schema(
            {
                "schema": FEEDBACK_SCHEMA,
                "version": FEEDBACK_VERSION,
                "uses_source_mapping_rgb": False,
                "uses_test_queries": False,
            },
            FEEDBACK_SCHEMA,
            label="incomplete feedback",
        )
