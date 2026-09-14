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
FULLMAP_REPORT_SCHEMA = "lafgs_rendered_track_full_mapping_loo_report"
FULLMAP_STATISTICS_SCHEMA = "lafgs_rendered_track_full_mapping_loo_statistics"


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


def _merge_mapping_cache_reports(
    summary_paths: Sequence[Path], output: Path
) -> dict:
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
        if _summary(rows, shard_counters) != statistics.get("summary"):
            raise ValueError("mapping shard summary is stale or inconsistent")
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
    if (
        json_sha256([row["image_name"] for row in query_rows])
        != expected_contract["selected_query_names_sha256"]
    ):
        raise ValueError("mapping shard image names differ from the fixed registry")
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


def _load_fullmap_shard(path: Path) -> tuple[dict, dict]:
    report = json.loads(path.read_text())
    if report.get("schema") != FULLMAP_REPORT_SCHEMA:
        raise ValueError("fullmap shard report schema differs")
    statistics_path = Path(str(report.get("statistics", ""))).expanduser().resolve()
    if not statistics_path.is_file():
        raise ValueError(f"fullmap shard statistics file is missing: {statistics_path}")
    if statistics_path.stat().st_size != int(report.get("statistics_size_bytes", -1)):
        raise ValueError("fullmap shard statistics size changed")
    if sha256_file(statistics_path) != report.get("statistics_sha256"):
        raise ValueError("fullmap shard statistics SHA-256 changed")
    statistics = torch.load(
        statistics_path, map_location="cpu", weights_only=False
    )
    if statistics.get("schema") != FULLMAP_STATISTICS_SCHEMA:
        raise ValueError("fullmap shard statistics schema differs")
    contract = report.get("evaluation_contract", {})
    shard = report.get("query_shard", {})
    statistics_range = statistics.get("query_range", {})
    if (
        report.get("producer_identity") != contract.get("producer_identity")
        or report.get("inputs") != contract.get("inputs")
        or report.get("input_sha256") != contract.get("input_sha256")
        or report.get("seed") != contract.get("seed")
        or any(
            report.get("configuration", {}).get(name) != value
            for name, value in contract.get("configuration", {}).items()
        )
    ):
        raise ValueError("fullmap shard report differs from its evaluation contract")
    if (
        int(shard.get("start", -1)) != int(statistics_range.get("start", -2))
        or int(shard.get("stop", -1)) != int(statistics_range.get("stop", -2))
        or report.get("summary") != statistics.get("summary")
        or report.get("loo") != statistics.get("loo")
    ):
        raise ValueError("fullmap shard report and atomic statistics differ")
    return report, statistics


def _merge_fullmap_reports(
    summary_paths: Sequence[Path], output: Path
) -> dict:
    loaded = []
    for supplied in summary_paths:
        path = Path(supplied).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"fullmap shard report is missing: {path}")
        report, statistics = _load_fullmap_shard(path)
        loaded.append((path, report, statistics))
    expected_contract = loaded[0][1].get("evaluation_contract")
    expected_identity = loaded[0][1].get("evaluation_contract_sha256")
    if json_sha256(expected_contract) != expected_identity:
        raise ValueError("fullmap evaluation contract identity is invalid")
    for _, report, statistics in loaded:
        if (
            report.get("evaluation_contract") != expected_contract
            or report.get("evaluation_contract_sha256") != expected_identity
            or statistics.get("evaluation_contract") != expected_contract
            or statistics.get("evaluation_contract_sha256") != expected_identity
        ):
            raise ValueError("fullmap shards differ in input/code/config/seed")
    loaded.sort(key=lambda item: int(item[2]["query_range"]["start"]))
    full_indices = [
        int(value) for value in expected_contract["selected_query_indices"]
    ]
    cursor = 0
    query_rows = []
    counters = None
    affected_parts = []
    shard_records = []
    loo_static = None
    maximum_query_local_eligible = 0
    affected_updates = 0
    aggregate_loo_fields = {
        "affected_anchor_updates",
        "maximum_query_local_eligible_k2_count",
        "minimum_affected_anchors_per_query",
        "maximum_affected_anchors_per_query",
        "mean_affected_anchors_per_query",
    }
    for path, report, statistics in loaded:
        begin = int(statistics["query_range"]["start"])
        end = int(statistics["query_range"]["stop"])
        if begin != cursor or end <= begin:
            raise ValueError("fullmap shard ranges overlap or leave a gap")
        expected_indices = full_indices[begin:end]
        rows = list(statistics["queries"])
        if (
            statistics.get("selected_query_indices") != expected_indices
            or [int(row["query_index"]) for row in rows] != expected_indices
        ):
            raise ValueError("fullmap shard rows differ from the fixed registry")
        shard_counters = {
            name: torch.as_tensor(value).cpu()
            for name, value in statistics["counters"].items()
        }
        if _summary(rows, shard_counters) != statistics.get("summary"):
            raise ValueError("fullmap shard summary is stale or inconsistent")
        if counters is None:
            counters = {name: value.clone() for name, value in shard_counters.items()}
        else:
            if set(counters) != set(shard_counters):
                raise ValueError("fullmap shard counter fields differ")
            for name, value in shard_counters.items():
                if counters[name].shape != value.shape or counters[name].dtype != value.dtype:
                    raise ValueError("fullmap shard counter tensor contract differs")
                counters[name] += value
        affected = torch.as_tensor(
            statistics.get("affected_anchor_count_by_query")
        ).long()
        if affected.numel() != end - begin:
            raise ValueError("fullmap shard affected-Anchor rows differ")
        static = {
            name: value
            for name, value in statistics["loo"].items()
            if name not in aggregate_loo_fields
        }
        if loo_static is None:
            loo_static = static
        elif static != loo_static:
            raise ValueError("fullmap shard LOO contracts differ")
        maximum_query_local_eligible = max(
            maximum_query_local_eligible,
            int(statistics["loo"]["maximum_query_local_eligible_k2_count"]),
        )
        affected_updates += int(statistics["loo"]["affected_anchor_updates"])
        affected_parts.append(affected)
        query_rows.extend(rows)
        shard_records.append(
            {
                "report": str(path),
                "statistics_sha256": report["statistics_sha256"],
                "query_start": begin,
                "query_stop": end,
            }
        )
        cursor = end
    if cursor != len(full_indices) or counters is None or loo_static is None:
        raise ValueError("fullmap shard set is incomplete")
    if (
        json_sha256([row["image_name"] for row in query_rows])
        != expected_contract["selected_query_names_sha256"]
    ):
        raise ValueError("fullmap shard image names differ from the fixed registry")
    affected = torch.cat(affected_parts)
    summary = _summary(query_rows, counters)
    loo = {
        **loo_static,
        "affected_anchor_updates": affected_updates,
        "maximum_query_local_eligible_k2_count": maximum_query_local_eligible,
        "minimum_affected_anchors_per_query": int(affected.min()),
        "maximum_affected_anchors_per_query": int(affected.max()),
        "mean_affected_anchors_per_query": float(affected.float().mean()),
    }
    statistics = {
        "schema": FULLMAP_STATISTICS_SCHEMA,
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "queries": query_rows,
        "counters": counters,
        "summary": summary,
        "loo": loo,
        "evaluation_contract": expected_contract,
        "evaluation_contract_sha256": expected_identity,
        "query_range": {"start": 0, "stop": len(full_indices)},
        "selected_query_indices": full_indices,
        "affected_anchor_count_by_query": affected,
    }
    output.mkdir(parents=True, exist_ok=True)
    statistics_path = output / "full_mapping_loo_statistics.pt"
    atomic_torch_save(statistics, statistics_path)
    first = loaded[0][1]
    configuration = dict(first["configuration"])
    mixture = configuration.get("view_mixture_contract")
    if mixture is not None:
        mixture = dict(mixture)
        mixture["maximum_query_local_eligible_k2_count"] = (
            maximum_query_local_eligible
        )
        anchor_count = int(expected_contract["anchor_count"])
        mixture["maximum_query_local_prototype_ratio"] = (
            anchor_count + maximum_query_local_eligible
        ) / anchor_count
        configuration["view_mixture_contract"] = mixture
    merged = {
        **first,
        "configuration": configuration,
        "statistics": str(statistics_path.resolve()),
        "statistics_sha256": sha256_file(statistics_path),
        "statistics_size_bytes": statistics_path.stat().st_size,
        "loo": loo,
        "summary": summary,
        "query_shard": {
            "kind": "merged",
            "start": 0,
            "stop": len(full_indices),
            "registry_count": len(full_indices),
        },
        "merge": {
            "kind": "exact_fullmap_query_shard_merge",
            "shard_count": len(loaded),
            "shards": shard_records,
        },
    }
    atomic_json_save(merged, output / "full_mapping_loo_report.json")
    return merged


def merge_reports(summary_paths: Sequence[Path], output: Path) -> dict:
    if not summary_paths:
        raise ValueError("at least one evaluator shard report is required")
    first_path = Path(summary_paths[0]).expanduser().resolve()
    if not first_path.is_file():
        raise ValueError(f"evaluator shard report is missing: {first_path}")
    schema = json.loads(first_path.read_text()).get("schema")
    if schema == REPORT_SCHEMA:
        return _merge_mapping_cache_reports(summary_paths, output)
    if schema == FULLMAP_REPORT_SCHEMA:
        return _merge_fullmap_reports(summary_paths, output)
    raise ValueError(f"unsupported evaluator shard report schema: {schema}")
