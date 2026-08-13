"""Fail-closed materialization contract for the neutral Anchor Registry.

The Registry is a sibling audit artifact.  It never replaces the trained map
consumed by localization and cannot authorize a scientific behavior change.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import uuid

import torch

from common.artifact_contract import anchor_registry
from common.config import load_mainline_config
from common.hashing import sha256_file
from common.producer_identity import (
    capture_producer_identity,
    verify_producer_identity,
)
from topology.anchor_covariance import attach_gaussian_prior_covariance
from topology.anchor_registry import (
    SCHEMA as REGISTRY_SCHEMA,
    SELECTION_LEGACY_UNRESOLVED,
    build_anchor_registry,
    validate_registry_compatibility,
)


CONTRACT_SCHEMA = "lafgs_neutral_anchor_registry_materialization"
CONTRACT_VERSION = 1
PIPELINE_PARENT_NAMES = frozenset(
    {
        "trained_map",
        "compact_map",
        "positive_teacher",
        "track_payload",
        "query_cache",
        "raster_provenance",
        "selection_provenance",
        "scene_calibration",
        "metric_state",
        "config",
        "gaussian_ply",
    }
)
SHA256 = re.compile(r"[0-9a-f]{64}")
PRODUCER_SCHEMA = "lafgs_neutral_anchor_registry_producer"
PRODUCER_SOURCE_PATHS = (
    "common/anchor_registry_artifact.py",
    "common/artifact_contract.py",
    "common/config.py",
    "common/hashing.py",
    "common/producer_identity.py",
    "scripts/materialize_anchor_registry.py",
    "topology/anchor_covariance.py",
    "topology/anchor_registry.py",
    "topology/geometry_materializer.py",
)


def _torch_load(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact is not a mapping: {path}")
    return payload


def _normalized_sha256(value: str, *, label: str) -> str:
    value = str(value).strip().lower()
    if SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    return value


def lock_parent_artifacts(
    parents: Mapping[str, tuple[str | Path, str]],
) -> dict[str, dict]:
    """Resolve and attest every explicitly supplied path/SHA pair."""
    if "trained_map" not in parents:
        raise ValueError("Anchor Registry requires an explicit trained_map parent")
    records = {}
    for name, (raw_path, raw_sha256) in sorted(parents.items()):
        if not name:
            raise ValueError("Anchor Registry parent name must be non-empty")
        path = Path(raw_path).expanduser().resolve()
        expected = _normalized_sha256(raw_sha256, label=f"expected {name} SHA-256")
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size <= 0:
            raise ValueError(f"Anchor Registry parent is empty: {name}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Anchor Registry parent hash mismatch for {name}: "
                f"expected {expected}, found {actual}"
            )
        records[name] = {
            "path": str(path),
            "expected_sha256": expected,
            "sha256": actual,
            "size_bytes": size,
        }
    return records


def _query_names(payload: Mapping) -> list[str]:
    if "query_names" in payload:
        return [str(value) for value in payload["query_names"]]
    cache = payload.get("queries", payload)
    return [
        str(name)
        for name, value in cache.items()
        if isinstance(value, Mapping) and "native_descriptors" in value
    ]


def _same_locked_path(declared: object, expected: Path) -> bool:
    if not isinstance(declared, (str, Path)) or not str(declared):
        return False
    path = Path(declared).expanduser().resolve()
    return path == expected


def _require_declared_parent(
    payload: Mapping,
    key: str,
    expected: Path,
    *,
    label: str,
    required: bool = True,
) -> None:
    if key not in payload:
        if required:
            raise ValueError(f"{label} does not declare its {key} parent")
        return
    if not _same_locked_path(payload[key], expected):
        raise ValueError(f"{label} declares a different {key} parent")


def _validate_parent_lineage(records: Mapping[str, Mapping]) -> dict[str, dict]:
    """Load supplied parents and reject row-, query-, or path-mixed chains."""
    paths = {name: Path(record["path"]) for name, record in records.items()}
    payloads = {
        name: _torch_load(path)
        for name, path in paths.items()
        if name
        in {
            "trained_map",
            "compact_map",
            "positive_teacher",
            "track_payload",
            "query_cache",
            "raster_provenance",
            "selection_provenance",
            "metric_state",
        }
    }
    state = payloads["trained_map"]
    if state.get("schema") != "lafgs_materialized_anchor_map":
        raise ValueError("trained_map is not a materialized Anchor map")
    count = int(torch.as_tensor(state.get("anchor_ids", ())).numel())
    if count <= 0:
        raise ValueError("trained_map contains no Anchors")

    compact = payloads.get("compact_map")
    if compact is not None:
        if compact.get("schema") != "lafgs_materialized_anchor_map":
            raise ValueError("compact_map is not a materialized Anchor map")
        for key in (
            "anchor_ids",
            "anchor_xyz",
            "source_primitive_ids",
            "track_cluster_ids",
            "anchor_type",
            "dependency_group_ids",
            "coarse_dependency_group_ids",
            "fine_identity_ids",
            "source_dependency_group_ids",
        ):
            if key in compact or key in state:
                if key not in compact or key not in state or not torch.equal(
                    torch.as_tensor(compact[key]).cpu(),
                    torch.as_tensor(state[key]).cpu(),
                ):
                    raise ValueError(
                        f"trained_map and compact_map differ in topology field {key}"
                    )

    teacher = payloads.get("positive_teacher")
    tracks = payloads.get("track_payload")
    query_cache = payloads.get("query_cache")
    raster = payloads.get("raster_provenance")
    selection = payloads.get("selection_provenance")
    metric = payloads.get("metric_state")

    if teacher is not None:
        if teacher.get("schema") != "lafgs_v9_active_map_complete_positive_teacher":
            raise ValueError("positive_teacher has an unsupported schema")
        if int(teacher.get("anchor_count", -1)) != count:
            raise ValueError("positive_teacher anchor count differs from trained_map")
        if compact is not None:
            _require_declared_parent(
                teacher,
                "anchor_map",
                paths["compact_map"],
                label="positive_teacher",
            )
    if tracks is not None and tracks.get("schema") != "lafgs_track_first_payload":
        raise ValueError("track_payload has an unsupported schema")
    if raster is not None:
        if raster.get("schema") != "lafgs_native_keypoint_raster_provenance":
            raise ValueError("raster_provenance has an unsupported schema")
        if compact is not None:
            _require_declared_parent(
                raster,
                "anchor_map",
                paths["compact_map"],
                label="raster_provenance",
            )
    if selection is not None and (
        selection.get("schema") != "lafgs_adaptive_selection_provenance"
        or selection.get("version") != 1
    ):
        raise ValueError("selection_provenance has an unsupported schema")
        if tracks is not None:
            _require_declared_parent(
                raster,
                "track_payload",
                paths["track_payload"],
                label="raster_provenance",
                required=False,
            )
        if query_cache is not None:
            _require_declared_parent(
                raster,
                "query_cache",
                paths["query_cache"],
                label="raster_provenance",
            )

    named = {
        name: _query_names(payload)
        for name, payload in (
            ("positive_teacher", teacher),
            ("track_payload", tracks),
            ("query_cache", query_cache),
            ("raster_provenance", raster),
        )
        if payload is not None
    }
    if named:
        reference_name, reference = next(iter(named.items()))
        for name, names in named.items():
            if names != reference:
                raise ValueError(
                    f"query registry differs between {reference_name} and {name}"
                )
        if len(reference) != len(set(reference)):
            raise ValueError("mapping query registry contains duplicate names")

    if metric is not None:
        if metric.get("schema") != "lafgs_shared_metric_state":
            raise ValueError("metric_state has an unsupported schema")
        landmark_indices = torch.as_tensor(metric.get("landmark_indices", ())).long()
        if not torch.equal(landmark_indices.cpu(), torch.arange(count)):
            raise ValueError("metric_state Anchor registry differs from trained_map")
        _require_declared_parent(
            metric,
            "map_path",
            paths["trained_map"],
            label="metric_state",
        )

    if "scene_calibration" in paths:
        calibration = json.loads(paths["scene_calibration"].read_text())
        sources = calibration.get("sources", {})
        if (
            calibration.get("schema") != "lafgs_mapping_only_scene_calibration"
            or sources.get("uses_test_queries") is not False
        ):
            raise ValueError("scene_calibration is not mapping-only/test-free")
        if query_cache is not None and not _same_locked_path(
            sources.get("query_cache"), paths["query_cache"]
        ):
            raise ValueError("scene_calibration declares a different query_cache")
        if tracks is not None and not _same_locked_path(
            sources.get("track_payload"), paths["track_payload"]
        ):
            raise ValueError("scene_calibration declares a different track_payload")
    if "config" in paths:
        load_mainline_config(paths["config"])
    return payloads


def _assert_parents_unchanged(records: Mapping[str, Mapping]) -> None:
    for name, record in records.items():
        path = Path(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"Anchor Registry parent changed during build: {name}")


def _temporary_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")


def _install_without_overwrite(temporary: Path, target: Path) -> None:
    """Atomically install a same-filesystem temporary while preserving target."""
    try:
        os.link(temporary, target)
    except FileExistsError as error:
        raise FileExistsError(f"Output already exists: {target}") from error
    temporary.unlink()


def materialize_anchor_registry(
    *,
    parents: Mapping[str, tuple[str | Path, str]],
    output: str | Path,
    contract_output: str | Path | None = None,
    require_pipeline_parents: bool = False,
    allow_legacy_unresolved_audit: bool = False,
) -> dict:
    """Build a neutral sibling Registry and install its contract atomically last."""
    output = Path(output).expanduser().resolve()
    contract_output = (
        Path(contract_output).expanduser().resolve()
        if contract_output is not None
        else output.with_suffix(output.suffix + ".contract.json")
    )
    if output == contract_output:
        raise ValueError("Registry artifact and contract outputs must differ")
    if output.exists() or contract_output.exists():
        raise FileExistsError(
            "Registry output and contract must both be absent; quarantine partial roots"
        )
    records = lock_parent_artifacts(parents)
    if require_pipeline_parents and set(records) != PIPELINE_PARENT_NAMES:
        missing = sorted(PIPELINE_PARENT_NAMES - set(records))
        extra = sorted(set(records) - PIPELINE_PARENT_NAMES)
        raise ValueError(
            f"Pipeline Registry parent set is incomplete or mixed; "
            f"missing={missing}, extra={extra}"
        )
    payloads = _validate_parent_lineage(records)
    producer = capture_producer_identity(
        schema=PRODUCER_SCHEMA, source_paths=PRODUCER_SOURCE_PATHS
    )
    state = payloads["trained_map"]
    registry = build_anchor_registry(
        state,
        teacher=payloads.get("positive_teacher"),
        track_payload=payloads.get("track_payload"),
        selection_provenance=payloads.get("selection_provenance"),
    )
    unresolved = int(
        (torch.as_tensor(registry["selection_reason"]) == SELECTION_LEGACY_UNRESOLVED)
        .sum()
        .item()
    )
    exact_selection = bool(
        registry["compatibility"]["selection_provenance_exact"]
    )
    if not exact_selection and not allow_legacy_unresolved_audit:
        raise ValueError(
            "legacy_unresolved selection is audit-only; pass the explicit audit flag"
        )
    if require_pipeline_parents and (not exact_selection or unresolved):
        raise ValueError("new pipeline Registry requires exact selection provenance")
    if "gaussian_ply" in records:
        registry = attach_gaussian_prior_covariance(
            registry, state, Path(records["gaussian_ply"]["path"])
        )
    registry["materialization"] = {
        "schema": CONTRACT_SCHEMA,
        "version": CONTRACT_VERSION,
        "parent_artifacts": deepcopy(records),
        "producer_identity": deepcopy(producer),
        "pipeline_parent_set_complete": set(records) == PIPELINE_PARENT_NAMES,
        "legacy_unresolved_audit_explicit": bool(allow_legacy_unresolved_audit),
        "changes_localization_tensors": False,
    }
    validate_registry_compatibility(registry, state)
    identity = anchor_registry(registry)
    output.parent.mkdir(parents=True, exist_ok=True)
    contract_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_artifact = _temporary_path(output)
    temporary_contract = _temporary_path(contract_output)
    installed_artifact = False
    try:
        torch.save(registry, temporary_artifact)
        if temporary_artifact.stat().st_size <= 0:
            raise RuntimeError("temporary Anchor Registry is empty")
        reloaded = _torch_load(temporary_artifact)
        if reloaded.get("schema") != REGISTRY_SCHEMA:
            raise RuntimeError("reloaded Anchor Registry has the wrong schema")
        if anchor_registry(reloaded) != identity:
            raise RuntimeError("reloaded Anchor Registry differs from construction")
        validate_registry_compatibility(reloaded, state)
        _assert_parents_unchanged(records)
        artifact_sha256 = sha256_file(temporary_artifact)
        contract = {
            "schema": CONTRACT_SCHEMA,
            "version": CONTRACT_VERSION,
            "uses_test_queries": False,
            "mapping_only": True,
            "audit_only": True,
            "localization_input": False,
            "complete": True,
            "partial": False,
            "pipeline_eligible": bool(
                set(records) == PIPELINE_PARENT_NAMES and exact_selection
            ),
            "artifact": {
                "path": str(output),
                "sha256": artifact_sha256,
                "size_bytes": temporary_artifact.stat().st_size,
            },
            "parent_artifacts": deepcopy(records),
            "producer_identity": deepcopy(producer),
            "anchor_registry": identity,
            "selection": {
                "exact": exact_selection,
                "legacy_unresolved_count": unresolved,
                "legacy_unresolved_is_epistemic_unknown": True,
            },
            "atomic_last": True,
            "failure_recovery": "quarantine_partial_outputs_and_restart",
        }
        temporary_contract.write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n"
        )
        json.loads(temporary_contract.read_text())
        _assert_parents_unchanged(records)
        verify_producer_identity(
            producer,
            schema=PRODUCER_SCHEMA,
            source_paths=PRODUCER_SOURCE_PATHS,
        )
        _install_without_overwrite(temporary_artifact, output)
        installed_artifact = True
        if sha256_file(output) != artifact_sha256:
            raise RuntimeError("installed Anchor Registry changed before completion")
        _install_without_overwrite(temporary_contract, contract_output)
    finally:
        for temporary in (temporary_artifact, temporary_contract):
            if temporary.exists():
                temporary.unlink()
    if not installed_artifact or not contract_output.is_file():
        raise RuntimeError("Anchor Registry completion contract was not installed")
    verify_anchor_registry_contract(contract_output)
    return {
        "registry": output,
        "contract": contract_output,
        "registry_sha256": sha256_file(output),
        "contract_sha256": sha256_file(contract_output),
        "report": deepcopy(registry["report"]),
    }


def verify_anchor_registry_contract(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    require_pipeline_eligible: bool = False,
) -> dict:
    """Recursively reload the contract, parents, and complete Registry schema."""
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    if expected_sha256 is not None:
        expected = _normalized_sha256(
            expected_sha256, label="expected Anchor Registry contract SHA-256"
        )
        if sha256_file(path) != expected:
            raise ValueError("Anchor Registry contract hash mismatch")
    contract = json.loads(path.read_text())
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("version") != CONTRACT_VERSION
        or contract.get("uses_test_queries") is not False
        or contract.get("mapping_only") is not True
        or contract.get("audit_only") is not True
        or contract.get("localization_input") is not False
        or contract.get("complete") is not True
        or contract.get("partial") is not False
        or contract.get("atomic_last") is not True
    ):
        raise ValueError("unsupported or incomplete Anchor Registry contract")
    if require_pipeline_eligible and contract.get("pipeline_eligible") is not True:
        raise ValueError("Anchor Registry contract is not pipeline eligible")
    producer = contract.get("producer_identity")
    if not isinstance(producer, dict):
        raise ValueError("Anchor Registry contract lacks producer identity")
    verify_producer_identity(
        producer,
        schema=PRODUCER_SCHEMA,
        source_paths=PRODUCER_SOURCE_PATHS,
    )
    records = contract.get("parent_artifacts")
    if not isinstance(records, dict) or "trained_map" not in records:
        raise ValueError("Anchor Registry contract lacks explicit parents")
    _assert_parents_unchanged(records)
    payloads = _validate_parent_lineage(records)
    artifact = contract.get("artifact", {})
    registry_path = Path(str(artifact.get("path", ""))).expanduser().resolve()
    if (
        not registry_path.is_file()
        or registry_path.stat().st_size != int(artifact.get("size_bytes", -1))
        or sha256_file(registry_path) != artifact.get("sha256")
    ):
        raise ValueError("Anchor Registry artifact differs from its contract")
    registry = _torch_load(registry_path)
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise ValueError("Anchor Registry artifact has an unsupported schema")
    if anchor_registry(registry) != contract.get("anchor_registry"):
        raise ValueError("Anchor Registry full schema digest differs")
    if registry.get("materialization", {}).get("parent_artifacts") != records:
        raise ValueError("Anchor Registry embedded parent registry differs")
    if registry.get("materialization", {}).get("producer_identity") != producer:
        raise ValueError("Anchor Registry embedded producer identity differs")
    validate_registry_compatibility(registry, payloads["trained_map"])
    selection = contract.get("selection", {})
    if require_pipeline_eligible and (
        selection.get("exact") is not True
        or int(selection.get("legacy_unresolved_count", -1)) != 0
        or set(records) != PIPELINE_PARENT_NAMES
    ):
        raise ValueError("pipeline Registry contains unresolved selection semantics")
    return contract
