import json

import pytest

from common.hashing import sha256_file
from common.calibration import (
    validate_equivalent_query_cache_calibration_parent,
)
from scripts.rebind_equivalent_query_cache_calibration import (
    rebind_equivalent_query_cache_calibration,
)
from scripts.rebind_equivalent_query_cache_manifest import (
    rebind_equivalent_query_cache_manifest,
)
from scripts.run_track_pair_factor import _validate_factor_input_lineage


def _entry(path):
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def test_equivalent_cache_rebind_is_exact_and_factor_consumable(tmp_path):
    source = tmp_path / "legacy.pt"
    refreshed = tmp_path / "fresh.pt"
    track = tmp_path / "tracks.pt"
    source.write_bytes(b"legacy-cache")
    refreshed.write_bytes(b"fresh-cache")
    track.write_bytes(b"frozen-tracks")
    parent_path = tmp_path / "parent.json"
    parent = {
        "arguments": {
            "query_cache_path": str(source.resolve()),
            "native_keypoint_count": 2048,
        },
        "inputs": {"query_cache_path": {"path": str(source.resolve()), "sha256": None}},
        "sentinel": {"unchanged": [1, 2, 3]},
    }
    parent_path.write_text(json.dumps(parent))
    equivalence_path = tmp_path / "equivalence.json"
    equivalence = {
        "schema": "lafgs_mapping_sparse_refresh_equivalence",
        "version": 2,
        "uses_test_queries": False,
        "valid": True,
        "checks": {"all_exact": True},
        "audit": {"content_equivalent_track_payload_reuse_authorized": True},
        "sources": {
            "source_cache": _entry(source),
            "refreshed_cache": _entry(refreshed),
            "source_track_payload": _entry(track),
        },
    }
    equivalence_path.write_text(json.dumps(equivalence))
    rebound = rebind_equivalent_query_cache_manifest(
        parent=parent,
        equivalence=equivalence,
        parent_manifest_path=parent_path,
        parent_manifest_sha256=sha256_file(parent_path),
        equivalence_path=equivalence_path,
        equivalence_sha256=sha256_file(equivalence_path),
        source_cache_path=source,
        source_cache_sha256=sha256_file(source),
        refreshed_cache_path=refreshed,
        refreshed_cache_sha256=sha256_file(refreshed),
        source_track_payload_path=track,
        source_track_payload_sha256=sha256_file(track),
    )
    assert rebound["sentinel"] == parent["sentinel"]
    assert rebound["arguments"]["query_cache_path"] == str(refreshed.resolve())
    assert rebound["inputs"]["query_cache_path"] == _entry(refreshed)
    assert rebound["equivalent_query_cache_rebind"]["parent_manifest"] == _entry(
        parent_path
    )

    rebound_path = tmp_path / "rebound.json"
    rebound_path.write_text(json.dumps(rebound))
    lineage = _validate_factor_input_lineage(
        manifest_payload=rebound,
        manifest_path=rebound_path,
        query_cache_path=refreshed,
        frozen_track_payload_path=track,
        expected_manifest_sha256=sha256_file(rebound_path),
        expected_query_cache_sha256=sha256_file(refreshed),
        expected_frozen_track_payload_sha256=sha256_file(track),
    )
    assert lineage["query_cache"] == _entry(refreshed)
    assert lineage["frozen_track_payload"] == _entry(track)


def test_equivalent_cache_rebind_rejects_failed_audit(tmp_path):
    source = tmp_path / "legacy.pt"
    refreshed = tmp_path / "fresh.pt"
    track = tmp_path / "tracks.pt"
    for path in (source, refreshed, track):
        path.write_bytes(path.name.encode())
    parent_path = tmp_path / "parent.json"
    parent = {
        "arguments": {"query_cache_path": str(source.resolve())},
        "inputs": {"query_cache_path": {"path": str(source.resolve()), "sha256": None}},
    }
    parent_path.write_text(json.dumps(parent))
    equivalence_path = tmp_path / "equivalence.json"
    equivalence = {
        "schema": "lafgs_mapping_sparse_refresh_equivalence",
        "version": 2,
        "uses_test_queries": False,
        "valid": False,
        "checks": {"all_exact": False},
        "audit": {"content_equivalent_track_payload_reuse_authorized": False},
        "sources": {
            "source_cache": _entry(source),
            "refreshed_cache": _entry(refreshed),
            "source_track_payload": _entry(track),
        },
    }
    equivalence_path.write_text(json.dumps(equivalence))
    with pytest.raises(ValueError, match="not valid"):
        rebind_equivalent_query_cache_manifest(
            parent=parent,
            equivalence=equivalence,
            parent_manifest_path=parent_path,
            parent_manifest_sha256=sha256_file(parent_path),
            equivalence_path=equivalence_path,
            equivalence_sha256=sha256_file(equivalence_path),
            source_cache_path=source,
            source_cache_sha256=sha256_file(source),
            refreshed_cache_path=refreshed,
            refreshed_cache_sha256=sha256_file(refreshed),
            source_track_payload_path=track,
            source_track_payload_sha256=sha256_file(track),
        )


def test_equivalent_calibration_rebind_preserves_numbers_and_validates(tmp_path):
    source = tmp_path / "legacy.pt"
    refreshed = tmp_path / "fresh.pt"
    track = tmp_path / "tracks.pt"
    source.write_bytes(b"legacy-cache")
    refreshed.write_bytes(b"fresh-cache")
    track.write_bytes(b"tracks")
    parent_path = tmp_path / "parent_calibration.json"
    parent = {
        "schema": "lafgs_mapping_only_scene_calibration",
        "version": 2,
        "sources": {
            "query_cache": str(source.resolve()),
            "track_payload": str(track.resolve()),
            "uses_test_queries": False,
        },
        "statistics": {"query_count": 3},
        "parameters": {"metric_steps": 8},
        "policy": {"matching_rows_fraction": 0.06},
    }
    parent_path.write_text(json.dumps(parent))
    equivalence_path = tmp_path / "equivalence.json"
    equivalence = {
        "schema": "lafgs_mapping_sparse_refresh_equivalence",
        "version": 2,
        "uses_test_queries": False,
        "valid": True,
        "checks": {"all_exact": True},
        "audit": {"content_equivalent_track_payload_reuse_authorized": True},
        "sources": {
            "source_cache": _entry(source),
            "refreshed_cache": _entry(refreshed),
            "source_track_payload": _entry(track),
        },
    }
    equivalence_path.write_text(json.dumps(equivalence))
    rebound = rebind_equivalent_query_cache_calibration(
        parent=parent,
        parent_path=parent_path,
        parent_sha256=sha256_file(parent_path),
        equivalence_path=equivalence_path,
        equivalence_sha256=sha256_file(equivalence_path),
        source_cache_path=source,
        source_cache_sha256=sha256_file(source),
        refreshed_cache_path=refreshed,
        refreshed_cache_sha256=sha256_file(refreshed),
        source_track_payload_path=track,
        source_track_payload_sha256=sha256_file(track),
    )
    rebound_path = tmp_path / "rebound_calibration.json"
    rebound_path.write_text(json.dumps(rebound))
    validate_equivalent_query_cache_calibration_parent(
        rebound,
        parent_path=rebound_path,
        query_cache_path=refreshed,
    )
    assert rebound["statistics"] == parent["statistics"]
    assert rebound["parameters"] == parent["parameters"]
    assert rebound["policy"] == parent["policy"]
