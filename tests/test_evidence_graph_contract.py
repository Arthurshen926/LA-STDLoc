from pathlib import Path

import pytest
import torch

from localization_training.evidence_graph_contract import (
    build_evidence_graph_contract,
    verify_evidence_graph_contract,
)


def _save(path: Path, payload) -> Path:
    torch.save(payload, path)
    return path


def _artifacts(tmp_path: Path) -> dict[str, Path]:
    names = ["q0"]
    query = {
        "q0": {
            "native_descriptors": torch.ones(2, 4),
            "native_keypoints": torch.zeros(2, 2),
            "native_input_hw": (8, 8),
        }
    }
    tracks = {
        "query_names": names,
        "tracks": {
            "track_index": torch.tensor([0]),
            "query_index": torch.tensor([0]),
            "keypoint_index": torch.tensor([1]),
        },
        "track_geometry": {"triangulated_xyz": torch.zeros(1, 3)},
    }
    anchors = {
        "anchor_xyz": torch.zeros(2, 3),
        "anchor_ids": torch.arange(2),
        "source_primitive_ids": torch.tensor([3, 4]),
        "track_cluster_ids": torch.tensor([-1, 0]),
        "anchor_type": torch.tensor([0, 1]),
        "dependency_group_ids": torch.tensor([0, 1]),
    }
    graph = {
        "query_names": names,
        "anchor_count": 2,
        "records": [{"query_rows": torch.tensor([0, 1])}],
    }
    provenance = {
        "query_names": names,
        "anchor_source_offsets": torch.tensor([0, 1, 2]),
    }
    teacher = {
        "query_names": names,
        "anchor_count": 2,
        "records": [{"query_rows": torch.tensor([0, 1])}],
        "diagnostics": {"strong_pair_count": 2},
    }
    prior = tmp_path / "prior.ply"
    prior.write_bytes(b"ply\n")
    return {
        "query_cache_path": _save(tmp_path / "query.pt", query),
        "track_payload_path": _save(tmp_path / "tracks.pt", tracks),
        "primitive_prior_path": prior,
        "anchor_map_path": _save(tmp_path / "anchors.pt", anchors),
        "function_graph_path": _save(tmp_path / "graph.pt", graph),
        "raster_provenance_path": _save(tmp_path / "provenance.pt", provenance),
        "positive_teacher_path": _save(tmp_path / "teacher.pt", teacher),
    }


def test_evidence_graph_contract_build_and_verify(tmp_path):
    contract = build_evidence_graph_contract(**_artifacts(tmp_path))
    assert contract["schema"] == "lafgs_localization_evidence_graph"
    assert contract["registries"]["query"]["query_count"] == 1
    assert contract["registries"]["anchor"]["anchor_count"] == 2
    assert contract["edge_sets"]["keypoint_track_observation"] == 1
    verify_evidence_graph_contract(contract)


def test_evidence_graph_contract_rejects_query_mismatch(tmp_path):
    artifacts = _artifacts(tmp_path)
    graph = torch.load(artifacts["function_graph_path"], weights_only=False)
    graph["query_names"] = ["wrong"]
    torch.save(graph, artifacts["function_graph_path"])
    with pytest.raises(ValueError, match="query registry"):
        build_evidence_graph_contract(**artifacts)


def test_evidence_graph_contract_detects_artifact_mutation(tmp_path):
    contract = build_evidence_graph_contract(**_artifacts(tmp_path))
    Path(contract["artifacts"]["primitive_prior"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_evidence_graph_contract(contract)
