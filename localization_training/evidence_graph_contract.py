"""Logical contract for the distributed LaFGS localization evidence graph."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from localization_training.artifact_contract import (
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


def build_dynamic_round_contract(
    *,
    static_contract: dict,
    round_id: int,
    active_map_path: str | Path,
    dynamic_outcomes_path: str | Path,
    metric_state_path: str | Path | None = None,
    pose_critical_teacher_path: str | Path | None = None,
    sampler_state_path: str | Path | None = None,
) -> dict:
    verify_evidence_graph_contract(static_contract)
    if int(round_id) < 0:
        raise ValueError("round_id must be non-negative")
    active_map_path = Path(active_map_path).resolve()
    dynamic_outcomes_path = Path(dynamic_outcomes_path).resolve()
    metric_state_path = (
        Path(metric_state_path).resolve() if metric_state_path else None
    )
    pose_critical_teacher_path = (
        Path(pose_critical_teacher_path).resolve()
        if pose_critical_teacher_path
        else None
    )
    sampler_state_path = (
        Path(sampler_state_path).resolve() if sampler_state_path else None
    )
    active = torch.load(active_map_path, map_location="cpu", weights_only=False)
    outcomes = torch.load(
        dynamic_outcomes_path, map_location="cpu", weights_only=False
    )
    active_count = int(torch.as_tensor(active["anchor_xyz"]).shape[0])
    if int(outcomes["anchor_count"]) != active_count:
        raise ValueError("dynamic outcomes do not align with active map")
    static_query_count = int(
        static_contract["registries"]["query"]["query_count"]
    )
    if len(outcomes["query_names"]) != static_query_count:
        raise ValueError("dynamic outcomes do not cover the static query registry")
    if sha256_json(list(outcomes["query_names"])) != static_contract["registries"][
        "query"
    ]["ordered_query_sha256"]:
        raise ValueError("dynamic outcomes do not match the static query order")
    for record in outcomes["records"]:
        indices = torch.as_tensor(record["top1_anchor_indices"]).long()
        if indices.numel() and (
            int(indices.min()) < 0 or int(indices.max()) >= active_count
        ):
            raise ValueError("dynamic outcome references an invalid active anchor")
    artifacts = {
        "active_map": {
            "path": str(active_map_path),
            "sha256": sha256_file(active_map_path),
        },
        "dynamic_outcomes": {
            "path": str(dynamic_outcomes_path),
            "sha256": sha256_file(dynamic_outcomes_path),
        },
    }
    if metric_state_path:
        artifacts["metric_state"] = {
            "path": str(metric_state_path),
            "sha256": sha256_file(metric_state_path),
        }
    pose_critical_edge_count = None
    if pose_critical_teacher_path:
        critical = torch.load(
            pose_critical_teacher_path, map_location="cpu", weights_only=False
        )
        if int(critical["anchor_count"]) != active_count:
            raise ValueError("pose-critical teacher does not align with active map")
        if list(critical["query_names"]) != list(outcomes["query_names"]):
            raise ValueError("pose-critical teacher query order mismatch")
        pose_critical_edge_count = int(
            sum(
                torch.as_tensor(record["positive_weights"]).numel()
                for record in critical["records"]
            )
        )
        artifacts["pose_critical_teacher"] = {
            "path": str(pose_critical_teacher_path),
            "sha256": sha256_file(pose_critical_teacher_path),
        }
    if sampler_state_path:
        artifacts["sampler_state"] = {
            "path": str(sampler_state_path),
            "sha256": sha256_file(sampler_state_path),
        }
    summary = dict(outcomes["summary"])
    minimal_set_outcome_count = int(
        sum(
            len(record.get("minimal_set_records", []))
            for record in outcomes["records"]
        )
    )
    identity = {
        "static_identity_sha256": static_contract["identity_sha256"],
        "round_id": int(round_id),
        "artifacts": {key: value["sha256"] for key, value in artifacts.items()},
        "summary": summary,
    }
    return {
        "schema": "lafgs_localization_evidence_graph_round",
        "schema_version": 2,
        "round_id": int(round_id),
        "static_identity_sha256": static_contract["identity_sha256"],
        "identity_sha256": sha256_json(identity),
        "artifacts": artifacts,
        "active_anchor_registry": anchor_registry(active),
        "dynamic_outcome_edges": {
            "query_count": len(outcomes["records"]),
            "candidate_edge_count": int(
                sum(
                    torch.as_tensor(record["top1_anchor_indices"]).numel()
                    for record in outcomes["records"]
                )
            ),
            "clean_survivor_count": int(
                sum(
                    torch.as_tensor(record["clean_inlier_mask"]).sum()
                    for record in outcomes["records"]
                )
            ),
            "harmful_survivor_count": int(
                sum(
                    torch.as_tensor(record["harmful_inlier_mask"]).sum()
                    for record in outcomes["records"]
                )
            ),
            "pose_critical_edge_count": pose_critical_edge_count,
            "minimal_set_outcome_count": minimal_set_outcome_count,
        },
        "pose_risk_summary": summary,
    }
