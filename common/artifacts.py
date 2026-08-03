"""Logical contract for the distributed LaFGS localization evidence graph."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from common.artifact_contract import (
    anchor_registry,
    query_registry,
    sha256_file,
    sha256_json,
)


SCHEMA = "lafgs_localization_evidence_graph"
SCHEMA_VERSION = 1


def _query_names(payload: dict) -> list[str]:
    if "query_names" in payload:
        return list(payload["query_names"])
    cache = payload.get("queries", payload)
    return [
        name
        for name, value in cache.items()
        if isinstance(value, dict) and "native_descriptors" in value
    ]


def _tensor_digest(values: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = torch.as_tensor(value).detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _anchor_count(payload: dict) -> int:
    if "anchor_count" in payload:
        return int(payload["anchor_count"])
    if "anchor_source_offsets" in payload:
        return int(torch.as_tensor(payload["anchor_source_offsets"]).numel() - 1)
    raise ValueError("artifact does not expose an anchor registry size")


def build_evidence_graph_contract(
    *,
    query_cache_path: str | Path,
    track_payload_path: str | Path,
    primitive_prior_path: str | Path,
    anchor_map_path: str | Path,
    function_graph_path: str | Path,
    raster_provenance_path: str | Path,
    positive_teacher_path: str | Path,
) -> dict:
    paths = {
        "query_cache": Path(query_cache_path).resolve(),
        "track_payload": Path(track_payload_path).resolve(),
        "primitive_prior": Path(primitive_prior_path).resolve(),
        "anchor_map": Path(anchor_map_path).resolve(),
        "function_graph": Path(function_graph_path).resolve(),
        "raster_provenance": Path(raster_provenance_path).resolve(),
        "positive_teacher": Path(positive_teacher_path).resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    query_cache = torch.load(paths["query_cache"], map_location="cpu", weights_only=False)
    tracks = torch.load(paths["track_payload"], map_location="cpu", weights_only=False)
    anchors = torch.load(paths["anchor_map"], map_location="cpu", weights_only=False)
    graph = torch.load(paths["function_graph"], map_location="cpu", weights_only=False)
    provenance = torch.load(
        paths["raster_provenance"], map_location="cpu", weights_only=False
    )
    teacher = torch.load(
        paths["positive_teacher"], map_location="cpu", weights_only=False
    )

    names = _query_names(query_cache)
    for label, payload in (
        ("track payload", tracks),
        ("function graph", graph),
        ("raster provenance", provenance),
        ("positive teacher", teacher),
    ):
        if _query_names(payload) != names:
            raise ValueError(f"{label} query registry does not match query cache")

    anchor_count = int(torch.as_tensor(anchors["anchor_xyz"]).shape[0])
    for label, payload in (
        ("function graph", graph),
        ("raster provenance", provenance),
        ("positive teacher", teacher),
    ):
        if _anchor_count(payload) != anchor_count:
            raise ValueError(f"{label} anchor count does not match anchor map")

    observations = tracks["tracks"]
    track_indices = torch.as_tensor(observations["track_index"]).long()
    query_indices = torch.as_tensor(observations["query_index"]).long()
    keypoint_indices = torch.as_tensor(observations["keypoint_index"]).long()
    if not (track_indices.numel() == query_indices.numel() == keypoint_indices.numel()):
        raise ValueError("track observation edge arrays do not align")
    track_count = int(
        torch.as_tensor(tracks["track_geometry"]["triangulated_xyz"]).shape[0]
    )
    if track_indices.numel() and (
        int(track_indices.min()) < 0 or int(track_indices.max()) >= track_count
    ):
        raise ValueError("track observation references an invalid track")
    if query_indices.numel() and (
        int(query_indices.min()) < 0 or int(query_indices.max()) >= len(names)
    ):
        raise ValueError("track observation references an invalid query")

    graph_rows = sum(
        int(torch.as_tensor(record["query_rows"]).numel())
        for record in graph["records"]
    )
    teacher_rows = sum(
        int(torch.as_tensor(record["query_rows"]).numel())
        for record in teacher["records"]
    )
    if graph_rows != teacher_rows:
        raise ValueError("positive teacher rows do not align with function graph")

    registries = {
        "query": query_registry(query_cache),
        "track": {
            "track_count": track_count,
            "observation_count": int(track_indices.numel()),
            "registry_sha256": _tensor_digest(
                [track_indices, query_indices, keypoint_indices]
            ),
        },
        "primitive": {
            "referenced_primitive_count": int(
                torch.unique(
                    torch.as_tensor(anchors["source_primitive_ids"]).long()
                ).numel()
            ),
            "registry_sha256": _tensor_digest(
                [torch.as_tensor(anchors["source_primitive_ids"]).long()]
            ),
        },
        "anchor": anchor_registry(anchors),
    }
    artifacts = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    edge_sets = {
        "keypoint_track_observation": int(track_indices.numel()),
        "keypoint_candidate_rows": graph_rows,
        "keypoint_positive_rows": teacher_rows,
        "strong_positive_pairs": int(
            teacher.get("diagnostics", {}).get("strong_pair_count", 0)
        ),
    }
    identity = {
        "registries": registries,
        "artifact_hashes": {name: value["sha256"] for name, value in artifacts.items()},
        "edge_sets": edge_sets,
    }
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "identity_sha256": sha256_json(identity),
        "registries": registries,
        "edge_sets": edge_sets,
        "artifacts": artifacts,
        "dynamic_outcome_edges": None,
    }


def verify_evidence_graph_contract(contract: dict) -> None:
    if contract.get("schema") != SCHEMA or contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported localization evidence graph contract")
    for name, record in contract["artifacts"].items():
        path = Path(record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"{name} hash mismatch")
    identity = {
        "registries": contract["registries"],
        "artifact_hashes": {
            name: value["sha256"] for name, value in contract["artifacts"].items()
        },
        "edge_sets": contract["edge_sets"],
    }
    if sha256_json(identity) != contract["identity_sha256"]:
        raise ValueError("evidence graph identity hash mismatch")
