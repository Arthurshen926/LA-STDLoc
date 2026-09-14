"""Strict lineage checks for compact-map localization artifacts."""

from __future__ import annotations

from pathlib import Path

import torch

from common.hashing import sha256_file


def _load(path: str | Path) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def audit_compact_artifact_lineage(
    *,
    anchor_map: str | Path,
    function_graph: str | Path,
    complete_positive_teacher: str | Path,
    metric_state: str | Path,
) -> dict:
    """Reject a graph, teacher, or metric that does not align with one Map.

    Counts alone are insufficient: two maps may have the same number of rows but
    different identities or query-row registries.  The metric carries the Map's
    exact anchor IDs, while the graph and teacher must share every query name and
    deployment row in the same order.
    """
    paths = {
        "anchor_map": Path(anchor_map).resolve(),
        "function_graph": Path(function_graph).resolve(),
        "complete_positive_teacher": Path(complete_positive_teacher).resolve(),
        "metric_state": Path(metric_state).resolve(),
    }
    payloads = {name: _load(path) for name, path in paths.items()}
    state = payloads["anchor_map"]
    graph = payloads["function_graph"]
    teacher = payloads["complete_positive_teacher"]
    metric = payloads["metric_state"]

    anchor_ids = torch.as_tensor(state["anchor_ids"]).long().reshape(-1)
    anchor_count = int(anchor_ids.numel())
    counts = {
        "anchor_map": anchor_count,
        "function_graph": int(graph["anchor_count"]),
        "complete_positive_teacher": int(teacher["anchor_count"]),
        "metric_state": int(
            torch.as_tensor(metric["landmark_indices"]).numel()
        ),
    }
    if any(count != anchor_count for count in counts.values()):
        raise ValueError(f"compact artifact anchor counts differ: {counts}")

    metric_ids = torch.as_tensor(metric["landmark_indices"]).long().reshape(-1)
    if not torch.equal(metric_ids, anchor_ids):
        raise ValueError("metric landmark IDs do not align with compact Map IDs")

    graph_names = list(graph["query_names"])
    teacher_names = list(teacher["query_names"])
    if graph_names != teacher_names:
        raise ValueError("function graph and teacher query registries differ")
    graph_records = graph["records"]
    teacher_records = teacher["records"]
    if len(graph_records) != len(graph_names) or len(teacher_records) != len(
        graph_names
    ):
        raise ValueError("function graph or teacher query records are incomplete")
    for query_index, (graph_record, teacher_record) in enumerate(
        zip(graph_records, teacher_records)
    ):
        graph_rows = torch.as_tensor(graph_record["query_rows"]).long()
        teacher_rows = torch.as_tensor(teacher_record["query_rows"]).long()
        if not torch.equal(graph_rows, teacher_rows):
            raise ValueError(
                "function graph and teacher deployment rows differ at "
                f"query {query_index}"
            )

    return {
        "schema": "lafgs_compact_artifact_lineage_audit",
        "version": 1,
        "valid": True,
        "anchor_count": anchor_count,
        "query_count": len(graph_names),
        "metric_anchor_ids_bitwise_equal_map": True,
        "teacher_rows_bitwise_equal_function_graph": True,
        "metric_initial_state": metric.get("initial_metric_state"),
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
    }
