"""Atomic-last, recursively verifiable completion for the public pipeline."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

from common.anchor_registry_artifact import verify_anchor_registry_contract
from common.hashing import canonical_json, sha256_file
from common.producer_identity import (
    capture_producer_identity,
    verify_producer_identity,
)
from common.tensor_identity import recursive_bitwise_equal


SCHEMA = "lafgs_fail_closed_pipeline_completion"
VERSION = 1
SHA256 = re.compile(r"[0-9a-f]{64}")
REQUIRED_ARTIFACTS = frozenset(
    {
        "anchor_registry",
        "anchor_registry_contract",
        "trained_map",
        "metric_state",
        "compact_map",
        "compact_positive_teacher",
        "compact_provenance",
        "track_payload",
        "query_cache",
        "selection_provenance",
        "scene_calibration",
        "prior_ply",
        "config",
    }
)
REGISTRY_PARENT_TO_PIPELINE_ARTIFACT = {
    "trained_map": "trained_map",
    "compact_map": "compact_map",
    "positive_teacher": "compact_positive_teacher",
    "track_payload": "track_payload",
    "query_cache": "query_cache",
    "raster_provenance": "compact_provenance",
    "selection_provenance": "selection_provenance",
    "scene_calibration": "scene_calibration",
    "metric_state": "metric_state",
    "config": "config",
    "gaussian_ply": "prior_ply",
}
PRODUCER_SCHEMA = "lafgs_fail_closed_pipeline_completion_producer"
PRODUCER_SOURCE_PATHS = (
    "common/anchor_registry_artifact.py",
    "common/artifact_contract.py",
    "common/config.py",
    "common/hashing.py",
    "common/pipeline_completion.py",
    "common/producer_identity.py",
    "common/tensor_identity.py",
    "map_learning/pipeline.py",
    "scripts/run_pipeline.py",
    "topology/anchor_covariance.py",
    "topology/anchor_registry.py",
    "topology/geometry_materializer.py",
)
EXPERIMENTAL_FACTOR_KEYS = frozenset(
    {"joint_keypoints", "mapping_keypoints", "surface_supported_tracks"}
)


def _normalized_sha256(value: str, *, label: str) -> str:
    value = str(value).strip().lower()
    if SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    return value


def _file_record(path: Path) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"pipeline artifact is empty: {path}")
    return {
        "kind": "file",
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": size,
    }


def _directory_files(path: Path) -> list[dict]:
    files = []
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            raise ValueError(f"pipeline artifact directory contains a symlink: {child}")
        if child.is_file():
            record = _file_record(child)
            record["relative_path"] = str(child.relative_to(path))
            record.pop("path")
            files.append(record)
    if not files:
        raise ValueError(f"pipeline artifact directory is empty: {path}")
    return files


def _path_record(raw_path: str | Path) -> dict:
    path = Path(raw_path).expanduser().resolve()
    if path.is_file():
        return _file_record(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = _directory_files(path)
    return {
        "kind": "directory",
        "path": str(path),
        "file_count": len(files),
        "files": files,
        "tree_sha256": hashlib.sha256(canonical_json({"files": files}).encode()).hexdigest(),
    }


def _verify_path_record(record: Mapping) -> None:
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    if record.get("kind") == "file":
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("size_bytes", -1))
            or sha256_file(path) != record.get("sha256")
        ):
            raise ValueError(f"pipeline artifact differs from completion: {path}")
        return
    if record.get("kind") != "directory" or not path.is_dir():
        raise ValueError(f"unsupported pipeline artifact record: {path}")
    files = _directory_files(path)
    if (
        files != record.get("files")
        or len(files) != int(record.get("file_count", -1))
        or hashlib.sha256(canonical_json({"files": files}).encode()).hexdigest()
        != record.get("tree_sha256")
    ):
        raise ValueError(f"pipeline artifact directory differs: {path}")


def atomic_json_install(payload: Mapping, target: str | Path) -> Path:
    """Install JSON atomically without ever replacing an existing target."""
    target = Path(target).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        json.loads(temporary.read_text())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise FileExistsError(f"Output already exists: {target}") from error
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _read_flat_manifest(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or any(
        not isinstance(name, str) or not isinstance(value, str)
        for name, value in payload.items()
    ):
        raise ValueError("legacy pipeline manifest must remain a flat path mapping")
    return payload


def _validated_factor_contract(
    factors: object,
    active: object | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    if not isinstance(factors, dict) or set(factors) != EXPERIMENTAL_FACTOR_KEYS:
        raise ValueError("pipeline completion has an invalid experimental factor set")
    joint = factors["joint_keypoints"]
    mapping = factors["mapping_keypoints"]
    surface = factors["surface_supported_tracks"]
    if joint is not None and (type(joint) is not int or joint not in {1024, 2048}):
        raise ValueError("joint keypoint factor is invalid")
    if mapping is not None and (
        type(mapping) is not int or mapping not in {1024, 2048}
    ):
        raise ValueError("mapping keypoint factor is invalid")
    if not isinstance(surface, bool):
        raise ValueError("surface Track factor must be boolean")
    if joint is not None and mapping is not None:
        raise ValueError("joint and mapping keypoint factors are mutually exclusive")
    canonical_active = {
        name: value
        for name, value in factors.items()
        if (value is not None if name != "surface_supported_tracks" else value is True)
    }
    if active is not None:
        if not isinstance(active, dict) or not recursive_bitwise_equal(
            active, canonical_active
        ):
            raise ValueError("active experimental factors differ from canonical replay")
    return dict(factors), canonical_active


def _require_same_registry_parents(
    registry_contract: Mapping, records: Mapping[str, Mapping]
) -> None:
    parents = registry_contract.get("parent_artifacts", {})
    if set(parents) != set(REGISTRY_PARENT_TO_PIPELINE_ARTIFACT):
        raise ValueError("pipeline Registry contract has a mixed parent set")
    for parent_name, artifact_name in REGISTRY_PARENT_TO_PIPELINE_ARTIFACT.items():
        parent = parents[parent_name]
        artifact = records.get(artifact_name, {})
        if (
            artifact.get("kind") != "file"
            or Path(str(artifact.get("path", ""))).expanduser().resolve()
            != Path(parent["path"]).expanduser().resolve()
            or artifact.get("sha256") != parent.get("sha256")
            or int(artifact.get("size_bytes", -1))
            != int(parent.get("size_bytes", -2))
        ):
            raise ValueError(
                f"pipeline artifact {artifact_name} differs from Registry parent "
                f"{parent_name}"
            )


def _reject_artifact_aliases_and_completion_ancestors(
    records: Mapping[str, Mapping], completion_path: Path
) -> None:
    resolved: dict[str, Path] = {}
    for name, record in records.items():
        path = Path(str(record.get("path", ""))).expanduser().resolve()
        if path in resolved.values():
            other = next(key for key, value in resolved.items() if value == path)
            raise ValueError(f"pipeline artifacts {other} and {name} alias one path")
        resolved[name] = path
        if record.get("kind") == "directory" and path in completion_path.parents:
            raise ValueError(
                f"pipeline directory artifact {name} contains pipeline completion"
            )
    names = list(resolved)
    for index, left_name in enumerate(names):
        left = resolved[left_name]
        for right_name in names[index + 1 :]:
            right = resolved[right_name]
            left_record = records[left_name]
            right_record = records[right_name]
            if (
                left_record.get("kind") == "directory"
                and left in right.parents
            ) or (
                right_record.get("kind") == "directory"
                and right in left.parents
            ):
                raise ValueError(
                    f"pipeline artifacts {left_name} and {right_name} overlap"
                )


def write_pipeline_completion(
    *,
    output: str | Path,
    artifacts: Mapping[str, str | Path],
    pipeline_manifest: str | Path,
    anchor_registry_contract: str | Path,
    config: str | Path,
    evaluation_requested: bool,
    experimental_factors: Mapping[str, object],
) -> dict:
    """Verify the whole output snapshot and install completion atomically last."""
    output = Path(output).expanduser().resolve()
    completion_path = output / "pipeline_completion.json"
    if completion_path.exists():
        raise FileExistsError(f"Pipeline completion already exists: {completion_path}")
    if "pipeline_manifest" in artifacts:
        raise ValueError("pipeline_manifest is a reserved completion artifact name")
    if not isinstance(evaluation_requested, bool):
        raise ValueError("evaluation_requested must be boolean")
    factors, active_factors = _validated_factor_contract(experimental_factors)
    if evaluation_requested and active_factors:
        raise ValueError("experimental factors and test evaluation are mutually exclusive")
    if evaluation_requested is not ("evaluation" in artifacts):
        raise ValueError("pipeline evaluation artifact does not match test opt-in")
    producer = capture_producer_identity(
        schema=PRODUCER_SCHEMA, source_paths=PRODUCER_SOURCE_PATHS
    )
    registry_contract_path = Path(anchor_registry_contract).expanduser().resolve()
    registry_contract = verify_anchor_registry_contract(
        registry_contract_path, require_pipeline_eligible=True
    )
    manifest_path = Path(pipeline_manifest).expanduser().resolve()
    if manifest_path != output / "pipeline_manifest.json":
        raise ValueError("pipeline manifest must be the canonical output-root sibling")
    flat_manifest = _read_flat_manifest(manifest_path)
    missing = sorted(REQUIRED_ARTIFACTS - set(artifacts))
    if missing:
        raise ValueError(f"pipeline completion artifact registry is incomplete: {missing}")
    if set(flat_manifest) != set(artifacts):
        raise ValueError("pipeline manifest and completion artifact names differ")
    required_flat = {name: str(Path(path)) for name, path in artifacts.items()}
    for name, expected in required_flat.items():
        if Path(flat_manifest.get(name, "")).expanduser().resolve() != Path(
            expected
        ).expanduser().resolve():
            raise ValueError(f"pipeline manifest lacks the exact {name} sibling")
    records = {
        name: _path_record(path)
        for name, path in sorted(
            {**artifacts, "pipeline_manifest": manifest_path}.items()
        )
    }
    _reject_artifact_aliases_and_completion_ancestors(records, completion_path)
    _require_same_registry_parents(registry_contract, records)
    if any(Path(record["path"]) == completion_path for record in records.values()):
        raise ValueError("pipeline completion cannot recursively include itself")
    contract_record = _file_record(registry_contract_path)
    if records.get("anchor_registry_contract") != contract_record:
        raise ValueError("Anchor Registry contract artifact record is inconsistent")
    registry_record = _file_record(Path(registry_contract["artifact"]["path"]))
    if records.get("anchor_registry") != registry_record:
        raise ValueError("Anchor Registry artifact differs from its pipeline sibling")
    config_record = _file_record(Path(config))
    if records.get("config") != config_record:
        raise ValueError("pipeline config artifact record is inconsistent")
    payload = {
        "schema": SCHEMA,
        "version": VERSION,
        "complete": True,
        "partial": False,
        "atomic_last": True,
        "uses_test_queries": bool(evaluation_requested),
        "mapping_only": not bool(evaluation_requested),
        "evaluation": {
            "requested": bool(evaluation_requested),
            "split": "test" if evaluation_requested else None,
            "explicit_opt_in_required": True,
        },
        "experimental_factors": factors,
        "active_experimental_factors": active_factors,
        "artifacts": records,
        "anchor_registry_contract": contract_record,
        "config": config_record,
        "producer_identity": producer,
        "failure_recovery": "quarantine_entire_output_root_and_restart_fresh",
    }
    for record in records.values():
        _verify_path_record(record)
    verify_anchor_registry_contract(
        registry_contract_path,
        expected_sha256=contract_record["sha256"],
        require_pipeline_eligible=True,
    )
    verify_producer_identity(
        producer,
        schema=PRODUCER_SCHEMA,
        source_paths=PRODUCER_SOURCE_PATHS,
    )
    atomic_json_install(payload, completion_path)
    verified = verify_pipeline_completion(completion_path)
    return {
        **verified,
        "path": str(completion_path),
        "sha256": sha256_file(completion_path),
    }


def verify_pipeline_completion(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict:
    """Recursively verify completion, Registry lineage, and every artifact."""
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    if expected_sha256 is not None:
        expected = _normalized_sha256(
            expected_sha256, label="expected pipeline completion SHA-256"
        )
        if sha256_file(path) != expected:
            raise ValueError("pipeline completion hash mismatch")
    payload = json.loads(path.read_text())
    evaluation = payload.get("evaluation", {})
    stored_active = payload.get("active_experimental_factors")
    if not isinstance(stored_active, dict):
        raise ValueError("pipeline completion lacks active experimental factors")
    _, active = _validated_factor_contract(
        payload.get("experimental_factors"),
        stored_active,
    )
    requested = evaluation.get("requested")
    if not isinstance(requested, bool):
        raise ValueError("pipeline evaluation.requested must be boolean")
    if (
        payload.get("schema") != SCHEMA
        or payload.get("version") != VERSION
        or payload.get("complete") is not True
        or payload.get("partial") is not False
        or payload.get("atomic_last") is not True
        or evaluation.get("explicit_opt_in_required") is not True
        or not isinstance(payload.get("uses_test_queries"), bool)
        or payload.get("uses_test_queries") is not requested
        or not isinstance(payload.get("mapping_only"), bool)
        or payload.get("mapping_only") is requested
        or evaluation.get("split") != ("test" if requested else None)
        or (requested and bool(active))
    ):
        raise ValueError("unsupported, partial, or mixed pipeline completion")
    producer = payload.get("producer_identity")
    if not isinstance(producer, dict):
        raise ValueError("pipeline completion lacks producer identity")
    verify_producer_identity(
        producer,
        schema=PRODUCER_SCHEMA,
        source_paths=PRODUCER_SOURCE_PATHS,
    )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("pipeline completion has no artifact registry")
    missing = sorted(REQUIRED_ARTIFACTS - set(artifacts))
    if missing:
        raise ValueError(f"pipeline completion artifact registry is incomplete: {missing}")
    if requested != ("evaluation" in artifacts):
        raise ValueError("pipeline evaluation artifact does not match test opt-in")
    _reject_artifact_aliases_and_completion_ancestors(artifacts, path)
    for record in artifacts.values():
        _verify_path_record(record)
    contract_record = payload.get("anchor_registry_contract", {})
    _verify_path_record(contract_record)
    contract_path = Path(contract_record["path"])
    registry_contract = verify_anchor_registry_contract(
        contract_path,
        expected_sha256=contract_record["sha256"],
        require_pipeline_eligible=True,
    )
    _require_same_registry_parents(registry_contract, artifacts)
    if artifacts.get("anchor_registry_contract") != contract_record:
        raise ValueError("pipeline and Registry completion records differ")
    registry_record = _file_record(Path(registry_contract["artifact"]["path"]))
    if artifacts.get("anchor_registry") != registry_record:
        raise ValueError("pipeline and Registry artifact records differ")
    config_record = payload.get("config", {})
    _verify_path_record(config_record)
    if artifacts.get("config") != config_record:
        raise ValueError("pipeline and config completion records differ")
    manifest_record = artifacts.get("pipeline_manifest")
    if not isinstance(manifest_record, dict) or manifest_record.get("kind") != "file":
        raise ValueError("pipeline completion lacks the flat pipeline manifest")
    expected_manifest_path = path.parent / "pipeline_manifest.json"
    if Path(str(manifest_record.get("path", ""))).resolve() != expected_manifest_path:
        raise ValueError("pipeline completion names a non-canonical manifest sibling")
    flat_manifest = _read_flat_manifest(Path(manifest_record["path"]))
    expected_flat_names = set(artifacts) - {"pipeline_manifest"}
    if set(flat_manifest) != expected_flat_names:
        raise ValueError("flat pipeline manifest artifact names differ")
    for name in expected_flat_names:
        record = artifacts.get(name, {})
        if Path(flat_manifest.get(name, "")).expanduser().resolve() != Path(
            str(record.get("path", ""))
        ).expanduser().resolve():
            raise ValueError(f"flat pipeline manifest differs for {name}")
    return payload
