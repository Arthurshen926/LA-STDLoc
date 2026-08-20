"""Contracts and exact merging for mapping-only pose-evaluation shards."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import torch

from common.hashing import sha256_file
from topology.deployment_revision import _summary


STATISTICS_SCHEMA = "lafgs_mapping_cache_evaluation_statistics"
REPORT_SCHEMA = "lafgs_mapping_cache_evaluation"


def json_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_query_range(
    count: int,
    *,
    query_start: int | None,
    query_stop: int | None,
    shard_index: int | None,
    shard_count: int | None,
) -> tuple[int, int, str]:
    count = int(count)
    uses_range = query_start is not None or query_stop is not None
    uses_shard = shard_index is not None or shard_count is not None
    if uses_range and uses_shard:
        raise ValueError("query range and shard index/count are mutually exclusive")
    if uses_shard:
        if shard_index is None or shard_count is None:
            raise ValueError("shard index and shard count must be supplied together")
        if int(shard_count) <= 0 or not 0 <= int(shard_index) < int(shard_count):
            raise ValueError("invalid query shard index/count")
        if int(shard_count) > count:
            raise ValueError("query shard count exceeds selected query count")
        begin = count * int(shard_index) // int(shard_count)
        end = count * (int(shard_index) + 1) // int(shard_count)
        return begin, end, "indexed_shard"
    begin = 0 if query_start is None else int(query_start)
    end = count if query_stop is None else int(query_stop)
    if begin < 0 or end > count or begin >= end:
        raise ValueError("query range must be a non-empty subset of the registry")
    return begin, end, "unsharded" if (begin, end) == (0, count) else "range_shard"


def write_statistics(
    *,
    output: Path,
    statistics: Mapping,
    evaluation_contract: Mapping,
    query_range: tuple[int, int],
    selected_query_indices: Sequence[int],
) -> dict:
    required = {"counters", "queries", "summary"}
    if not required.issubset(statistics):
        raise ValueError("deployment statistics lack mergeable query/counter state")
    begin, end = query_range
    rows = list(statistics["queries"])
    expected = [int(value) for value in selected_query_indices]
    actual = [int(row["query_index"]) for row in rows]
    if actual != expected or len(rows) != end - begin:
        raise ValueError("deployment statistics escape the fixed query shard order")
    payload = {
        "schema": STATISTICS_SCHEMA,
        "version": 1,
        "evaluation_contract": dict(evaluation_contract),
        "evaluation_contract_sha256": json_sha256(evaluation_contract),
        "query_range": {"start": begin, "stop": end},
        "selected_query_indices": expected,
        "counters": {
            name: torch.as_tensor(value).cpu()
            for name, value in statistics["counters"].items()
        },
        "queries": rows,
        "summary": dict(statistics["summary"]),
    }
    path = output / "mapping_cache_statistics.pt"
    atomic_torch_save(payload, path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _load_verified_statistics(report: Mapping, summary_path: Path) -> dict:
    record = report.get("statistics")
    if not isinstance(record, Mapping):
        raise ValueError(f"mapping shard lacks statistics record: {summary_path}")
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"mapping shard statistics file is missing: {path}")
    if path.stat().st_size != int(record.get("size_bytes", -1)):
        raise ValueError(f"mapping shard statistics size changed: {path}")
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"mapping shard statistics SHA-256 changed: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != STATISTICS_SCHEMA or payload.get("version") != 1:
        raise ValueError("unexpected mapping shard statistics schema")
    return payload


def merge_reports(summary_paths: Sequence[Path], output: Path) -> dict:
    if not summary_paths:
        raise ValueError("at least one mapping shard summary is required")
    loaded = []
    for supplied in summary_paths:
        path = Path(supplied).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"mapping shard summary is missing: {path}")
        report = json.loads(path.read_text())
        if report.get("schema") != REPORT_SCHEMA or int(report.get("version", 0)) < 3:
            raise ValueError("mapping shard report does not support exact merging")
        statistics = _load_verified_statistics(report, path)
        contract = report.get("evaluation_contract", {})
        protocol = report.get("evaluation_protocol", {})
        shard = protocol.get("query_shard", {})
        statistics_range = statistics.get("query_range", {})
        if (
            report.get("evaluation_code") != contract.get("evaluation_code")
            or report.get("artifacts") != contract.get("artifacts")
            or report.get("seed") != contract.get("seed")
            or report.get("deployment_row_limit")
            != contract.get("deployment_row_limit")
        ):
            raise ValueError("mapping shard report differs from its evaluation contract")
        if (
            int(shard.get("start", -1)) != int(statistics_range.get("start", -2))
            or int(shard.get("stop", -1)) != int(statistics_range.get("stop", -2))
            or protocol.get("selected_query_indices")
            != statistics.get("selected_query_indices")
            or report.get("query_count") != len(statistics.get("queries", []))
            or report.get("summary") != statistics.get("summary")
        ):
            raise ValueError("mapping shard report and atomic statistics differ")
        loaded.append((path, report, statistics))
    expected_contract = loaded[0][1].get("evaluation_contract")
    expected_identity = loaded[0][1].get("evaluation_contract_sha256")
    if json_sha256(expected_contract) != expected_identity:
        raise ValueError("mapping shard evaluation contract identity is invalid")
    for _, report, statistics in loaded:
        if (
            report.get("evaluation_contract") != expected_contract
            or report.get("evaluation_contract_sha256") != expected_identity
            or statistics.get("evaluation_contract") != expected_contract
            or statistics.get("evaluation_contract_sha256") != expected_identity
        ):
            raise ValueError("mapping shards differ in input/code/config/seed")
    loaded.sort(key=lambda item: int(item[2]["query_range"]["start"]))
    full_indices = [
        int(value) for value in expected_contract["selected_query_indices"]
    ]
    cursor = 0
    query_rows = []
    counters = None
    shard_records = []
    for path, report, statistics in loaded:
        begin = int(statistics["query_range"]["start"])
        end = int(statistics["query_range"]["stop"])
        if begin != cursor or end <= begin:
            raise ValueError("mapping shard ranges overlap or leave a gap")
        expected_indices = full_indices[begin:end]
        if statistics.get("selected_query_indices") != expected_indices:
            raise ValueError("mapping shard indices differ from the fixed registry")
        rows = list(statistics["queries"])
        if [int(row["query_index"]) for row in rows] != expected_indices:
            raise ValueError("mapping shard query rows are out of registry order")
        shard_counters = {
            name: torch.as_tensor(value).cpu()
            for name, value in statistics["counters"].items()
        }
        if counters is None:
            counters = {name: value.clone() for name, value in shard_counters.items()}
        else:
            if set(counters) != set(shard_counters):
                raise ValueError("mapping shard counter fields differ")
            for name, value in shard_counters.items():
                if counters[name].shape != value.shape or counters[name].dtype != value.dtype:
                    raise ValueError("mapping shard counter tensor contract differs")
                counters[name] += value
        query_rows.extend(rows)
        shard_records.append(
            {
                "summary": str(path),
                "statistics_sha256": report["statistics"]["sha256"],
                "query_start": begin,
                "query_stop": end,
            }
        )
        cursor = end
    if cursor != len(full_indices) or counters is None:
        raise ValueError("mapping shard set is incomplete")
    summary = _summary(query_rows, counters)
    first = loaded[0][1]
    merged = {
        **first,
        "query_count": len(full_indices),
        "summary": summary,
        "merge": {
            "kind": "exact_query_shard_merge",
            "shard_count": len(loaded),
            "shards": shard_records,
        },
    }
    protocol = dict(first["evaluation_protocol"])
    protocol.update(
        {
            "evaluated_query_count": len(full_indices),
            "selected_query_indices": full_indices,
            "selected_query_indices_sha256": json_sha256(full_indices),
            "selected_query_names_sha256": expected_contract[
                "selected_query_names_sha256"
            ],
            "query_shard": {
                "kind": "merged",
                "start": 0,
                "stop": len(full_indices),
                "registry_count": len(full_indices),
            },
        }
    )
    merged["evaluation_protocol"] = protocol
    statistics_record = write_statistics(
        output=output,
        statistics={"counters": counters, "queries": query_rows, "summary": summary},
        evaluation_contract=expected_contract,
        query_range=(0, len(full_indices)),
        selected_query_indices=full_indices,
    )
    merged["statistics"] = statistics_record
    atomic_json_save(merged, output / "mapping_cache_summary.json")
    return merged
