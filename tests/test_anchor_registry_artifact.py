import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from common.anchor_registry_artifact import (
    materialize_anchor_registry,
    verify_anchor_registry_contract,
)
from common.artifact_contract import anchor_registry
from common.hashing import sha256_file
from topology.anchor_registry import build_anchor_registry


def _state() -> dict:
    return {
        "schema": "lafgs_materialized_anchor_map",
        "anchor_ids": torch.arange(3),
        "anchor_xyz": torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]]
        ),
        "anchor_features": torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        ),
        "source_primitive_ids": torch.tensor([7, 8, 9]),
        "track_cluster_ids": torch.tensor([1, -1, -1]),
        "anchor_type": torch.tensor([1, 0, 0]),
        "dependency_group_ids": torch.tensor([0, 1, 2]),
        "coarse_dependency_group_ids": torch.tensor([0, 1, 2]),
        "fine_identity_ids": torch.tensor([10, 11, 12]),
        "source_dependency_group_ids": torch.tensor([20, 21, 22]),
        "track_centric_reconstruction": {
            "base_canonical_rows": torch.tensor([2, 4]),
            "calibration": {
                "parameters": {
                    "surface_max_distance_m": 0.3,
                    "surface_point_plane_m": 0.1,
                }
            },
        },
    }


def _save(path: Path, payload: dict) -> Path:
    torch.save(payload, path)
    return path


def _write_ply(path: Path) -> Path:
    rows = []
    for index in range(10):
        rows.append(f"{index} 0 1 -2 -2 1 0 0 0")
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 10",
                "property float x",
                "property float y",
                "property float z",
                "property float scale_0",
                "property float scale_1",
                "property float rot_0",
                "property float rot_1",
                "property float rot_2",
                "property float rot_3",
                "end_header",
                *rows,
                "",
            ]
        )
    )
    return path


def _pipeline_parents(tmp_path: Path) -> dict[str, tuple[Path, str]]:
    compact = _save(tmp_path / "compact.pt", _state())
    trained_state = _state()
    trained_state["anchor_features"] = trained_state["anchor_features"] + 0.01
    trained = _save(tmp_path / "trained.pt", trained_state)
    query = _save(
        tmp_path / "query.pt",
        {
            "queries": {
                name: {"native_descriptors": torch.zeros(1, 2)}
                for name in ("a", "b")
            }
        },
    )
    tracks = _save(
        tmp_path / "tracks.pt",
        {
            "schema": "lafgs_track_first_payload",
            "query_names": ["a", "b"],
            "tracks": {
                "track_index": torch.tensor([1, 1]),
                "query_index": torch.tensor([0, 1]),
                "keypoint_index": torch.tensor([3, 5]),
            },
            "track_geometry": {
                "triangulated_xyz": torch.zeros(5, 3),
                "triangulation_covariance_matrix": torch.eye(3).repeat(5, 1, 1),
            },
        },
    )
    teacher = _save(
        tmp_path / "teacher.pt",
        {
            "schema": "lafgs_v9_active_map_complete_positive_teacher",
            "anchor_count": 3,
            "anchor_map": str(compact),
            "query_names": ["a", "b"],
            "records": [
                {
                    "query_index": index,
                    "query_rows": torch.tensor([10 + index]),
                    "positive_offsets": torch.tensor([0, 1]),
                    "positive_indices": torch.tensor([1]),
                }
                for index in range(2)
            ],
        },
    )
    raster = _save(
        tmp_path / "raster.pt",
        {
            "schema": "lafgs_native_keypoint_raster_provenance",
            "anchor_map": str(compact),
            "query_cache": str(query),
            "query_names": ["a", "b"],
        },
    )
    selection = _save(
        tmp_path / "selection.pt",
        {
            "schema": "lafgs_adaptive_selection_provenance",
            "version": 1,
            "track_universe_count": 5,
            "track_core_universe_ids": torch.tensor([1]),
            "coverage_track_universe_ids": torch.empty(0, dtype=torch.long),
            "coverage_gaussian_universe_ids": torch.tensor([7]),
            "pose_track_universe_ids": torch.empty(0, dtype=torch.long),
            "pose_gaussian_universe_ids": torch.tensor([9]),
        },
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema": "lafgs_mapping_only_scene_calibration",
                "version": 2,
                "sources": {
                    "query_cache": str(query),
                    "track_payload": str(tracks),
                    "uses_test_queries": False,
                },
            }
        )
    )
    metric = _save(
        tmp_path / "metric.pt",
        {
            "schema": "lafgs_shared_metric_state",
            "landmark_indices": torch.arange(3),
            "map_path": str(trained),
        },
    )
    config = Path("configs/paper_mainline.yaml").resolve()
    gaussian = _write_ply(tmp_path / "gaussians.ply")
    paths = {
        "trained_map": trained,
        "compact_map": compact,
        "positive_teacher": teacher,
        "track_payload": tracks,
        "query_cache": query,
        "raster_provenance": raster,
        "selection_provenance": selection,
        "scene_calibration": calibration,
        "metric_state": metric,
        "config": config,
        "gaussian_ply": gaussian,
    }
    return {name: (path, sha256_file(path)) for name, path in paths.items()}


def test_complete_registry_digest_covers_geometry_observations_and_evidence() -> None:
    payload = _state()
    legacy = anchor_registry(payload)
    changed = copy.deepcopy(payload)
    changed["anchor_xyz"][0, 0] = 4.0
    changed_identity = anchor_registry(changed)
    assert changed_identity["registry_sha256"] == legacy["registry_sha256"]
    assert (
        changed_identity["full_registry_sha256"]
        != legacy["full_registry_sha256"]
    )
    assert (
        changed_identity["field_sha256"]["anchor_xyz"]
        != legacy["field_sha256"]["anchor_xyz"]
    )

    registry = build_anchor_registry(payload)
    registry_changed = copy.deepcopy(registry)
    registry_changed["evidence_mask"][0] ^= 1
    registry_changed["observation_offsets"][1] += 1
    before = anchor_registry(registry)
    after = anchor_registry(registry_changed)
    assert before["full_registry_sha256"] != after["full_registry_sha256"]
    assert (
        before["field_sha256"]["observation_offsets"]
        != after["field_sha256"]["observation_offsets"]
    )
    assert (
        before["field_sha256"]["evidence_mask"]
        != after["field_sha256"]["evidence_mask"]
    )


def test_pipeline_registry_is_sibling_and_preserves_localization_tensors(
    tmp_path: Path,
) -> None:
    parents = _pipeline_parents(tmp_path)
    output = tmp_path / "registry.pt"
    contract = tmp_path / "registry.contract.json"
    result = materialize_anchor_registry(
        parents=parents,
        output=output,
        contract_output=contract,
        require_pipeline_parents=True,
    )
    persisted = torch.load(output, map_location="cpu", weights_only=False)
    source = torch.load(
        parents["trained_map"][0], map_location="cpu", weights_only=False
    )
    for key in (
        "anchor_ids",
        "anchor_xyz",
        "anchor_features",
        "source_primitive_ids",
        "track_cluster_ids",
        "anchor_type",
    ):
        assert torch.equal(persisted[key], source[key])
    assert persisted["localization_input"] is False
    verified = verify_anchor_registry_contract(
        result["contract"],
        expected_sha256=result["contract_sha256"],
        require_pipeline_eligible=True,
    )
    assert verified["selection"]["exact"] is True
    assert verified["selection"]["legacy_unresolved_count"] == 0


def test_registry_contract_rejects_parent_tamper(tmp_path: Path) -> None:
    parents = _pipeline_parents(tmp_path)
    result = materialize_anchor_registry(
        parents=parents,
        output=tmp_path / "registry.pt",
        require_pipeline_parents=True,
    )
    parents["positive_teacher"][0].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="parent changed"):
        verify_anchor_registry_contract(result["contract"])


def test_registry_rejects_zero_byte_and_partial_outputs(tmp_path: Path) -> None:
    state = _save(tmp_path / "state.pt", _state())
    empty = tmp_path / "empty.pt"
    empty.touch()
    with pytest.raises(ValueError, match="parent is empty"):
        materialize_anchor_registry(
            parents={
                "trained_map": (state, sha256_file(state)),
                "positive_teacher": (empty, hashlib.sha256(b"").hexdigest()),
            },
            output=tmp_path / "registry.pt",
            allow_legacy_unresolved_audit=True,
        )
    partial = tmp_path / "partial.pt"
    partial.write_bytes(b"partial")
    with pytest.raises(FileExistsError, match="both be absent"):
        materialize_anchor_registry(
            parents={"trained_map": (state, sha256_file(state))},
            output=partial,
            allow_legacy_unresolved_audit=True,
        )


def test_legacy_unresolved_requires_explicit_audit_and_never_pipeline_eligible(
    tmp_path: Path,
) -> None:
    state = _save(tmp_path / "state.pt", _state())
    parents = {"trained_map": (state, sha256_file(state))}
    with pytest.raises(ValueError, match="explicit audit flag"):
        materialize_anchor_registry(parents=parents, output=tmp_path / "rejected.pt")
    result = materialize_anchor_registry(
        parents=parents,
        output=tmp_path / "audit.pt",
        allow_legacy_unresolved_audit=True,
    )
    contract = verify_anchor_registry_contract(result["contract"])
    assert contract["pipeline_eligible"] is False
    assert contract["selection"]["legacy_unresolved_count"] == 3
    with pytest.raises(ValueError, match="not pipeline eligible"):
        verify_anchor_registry_contract(
            result["contract"], require_pipeline_eligible=True
        )


def test_explicit_malformed_selection_fails_instead_of_becoming_legacy(
    tmp_path: Path,
) -> None:
    state = _save(tmp_path / "state.pt", _state())
    selection = _save(
        tmp_path / "selection.pt",
        {
            "track_universe_count": 5,
            "track_core_universe_ids": torch.tensor([1]),
            "coverage_track_universe_ids": torch.empty(0, dtype=torch.long),
            "coverage_gaussian_universe_ids": torch.tensor([7]),
            "pose_track_universe_ids": torch.empty(0, dtype=torch.long),
            "pose_gaussian_universe_ids": torch.empty(0, dtype=torch.long),
        },
    )
    with pytest.raises(ValueError, match="does not exactly cover"):
        materialize_anchor_registry(
            parents={
                "trained_map": (state, sha256_file(state)),
                "selection_provenance": (selection, sha256_file(selection)),
            },
            output=tmp_path / "registry.pt",
            allow_legacy_unresolved_audit=True,
        )
