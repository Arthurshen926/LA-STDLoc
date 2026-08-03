from pathlib import Path

import pytest

from common.registry import AnchorRegistry, ArtifactReference, QueryRegistry
from evaluation.golden import verify_fixture


def test_registries_are_order_strict():
    QueryRegistry.from_names(["a", "b"]).require_exact(
        QueryRegistry.from_names(["a", "b"])
    )
    with pytest.raises(ValueError):
        QueryRegistry.from_names(["a", "b"]).require_exact(
            QueryRegistry.from_names(["b", "a"])
        )
    with pytest.raises(ValueError):
        AnchorRegistry.from_ids([1, 1])


def test_artifact_reference_detects_mutation(tmp_path: Path):
    path = tmp_path / "artifact"
    path.write_bytes(b"one")
    reference = ArtifactReference.capture(path)
    reference.verify()
    path.write_bytes(b"two")
    with pytest.raises(ValueError, match="stale"):
        reference.verify()


def test_committed_golden_fixture_hashes():
    manifest = verify_fixture("paper_baseline/golden_fixture")
    assert manifest["query_count"] == 16
