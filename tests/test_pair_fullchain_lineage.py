import json

import torch

from common.artifacts import build_evidence_graph_contract
from scripts.audit_pair_fullchain_lineage import audit_canonical, audit_compact


def _artifact(path, payload):
    torch.save(payload, path)
    return path


def test_compact_pair_lineage_requires_variant_payload_everywhere(tmp_path):
    tracks = _artifact(tmp_path / "variant.pt", {"schema": "tracks"})
    compact = _artifact(
        tmp_path / "compact.pt",
        {"provenance": {"track_payload": str(tracks)}},
    )
    provenance = _artifact(
        tmp_path / "provenance.pt",
        {"anchor_map": str(compact), "config": {"track_payload": str(tracks)}},
    )
    teacher = _artifact(
        tmp_path / "teacher.pt",
        {
            "anchor_map": str(compact),
            "raster_provenance": str(provenance),
            "track_payload": str(tracks),
        },
    )
    assert audit_compact(
        compact_map=compact,
        provenance=provenance,
        teacher=teacher,
        expected_track_payload=tracks,
    )["valid"]
    wrong = _artifact(tmp_path / "wrong.pt", {"schema": "tracks"})
    assert not audit_compact(
        compact_map=compact,
        provenance=provenance,
        teacher=teacher,
        expected_track_payload=wrong,
    )["valid"]


def test_canonical_pair_lineage_verifies_contract_track_hash(tmp_path):
    query = _artifact(
        tmp_path / "query.pt",
        {
            "frame": {
                "native_descriptors": torch.zeros(1, 2),
                "native_input_hw": [1, 1],
                "native_keypoints": torch.zeros(1, 2),
            }
        },
    )
    tracks = _artifact(
        tmp_path / "variant.pt",
        {
            "query_names": ["frame"],
            "tracks": {
                "track_index": torch.tensor([0]),
                "query_index": torch.tensor([0]),
                "keypoint_index": torch.tensor([0]),
            },
            "track_geometry": {"triangulated_xyz": torch.zeros(1, 3)},
        },
    )
    prior = tmp_path / "prior.ply"
    prior.write_bytes(b"ply")
    anchors = _artifact(
        tmp_path / "anchors.pt",
        {
            "anchor_xyz": torch.zeros(1, 3),
            "anchor_features": torch.zeros(1, 2),
            "anchor_ids": torch.tensor([0]),
            "source_primitive_ids": torch.tensor([0]),
            "track_cluster_ids": torch.tensor([-1]),
            "anchor_type": torch.tensor([0]),
        },
    )
    graph = _artifact(
        tmp_path / "graph.pt",
        {
            "query_names": ["frame"],
            "anchor_count": 1,
            "records": [{"query_rows": torch.tensor([0])}],
        },
    )
    provenance = _artifact(
        tmp_path / "provenance.pt",
        {"query_names": ["frame"], "anchor_count": 1},
    )
    teacher = _artifact(
        tmp_path / "teacher.pt",
        {
            "query_names": ["frame"],
            "anchor_count": 1,
            "records": [{"query_rows": torch.tensor([0])}],
        },
    )
    contract = build_evidence_graph_contract(
        query_cache_path=query,
        track_payload_path=tracks,
        primitive_prior_path=prior,
        anchor_map_path=anchors,
        function_graph_path=graph,
        raster_provenance_path=provenance,
        positive_teacher_path=teacher,
    )
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    assert audit_canonical(
        evidence_contract=path, expected_track_payload=tracks
    )["valid"]
