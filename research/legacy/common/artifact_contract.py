"""Content-addressed contracts for LaFGS cross-stage artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import subprocess
from pathlib import Path
from typing import Any

import torch

from common.tensor_identity import tensor_bytes


CONTRACT_SCHEMA = "lafgs_artifact_contract"
CONTRACT_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def query_registry(payload: dict) -> dict:
    cache = payload.get("queries", payload)
    names = [
        name
        for name, value in cache.items()
        if isinstance(value, dict) and "native_descriptors" in value
    ]
    if len(names) != len(set(names)):
        raise ValueError("query names must be unique")
    rows = []
    for name in names:
        value = cache[name]
        rows.append(
            {
                "name": name,
                "native_input_hw": list(value["native_input_hw"]),
                "native_keypoint_count": int(
                    torch.as_tensor(value["native_keypoints"]).shape[0]
                ),
                "pixel_center_offset": float(
                    value.get("pixel_center_offset", 0.5)
                ),
            }
        )
    return {
        "query_count": len(rows),
        "ordered_query_sha256": sha256_json(names),
        "registry_sha256": sha256_json(rows),
    }


_LOCALIZATION_ANCHOR_FIELDS = (
    "anchor_ids",
    "anchor_xyz",
    "anchor_features",
    "source_primitive_ids",
    "track_cluster_ids",
    "anchor_type",
    "dependency_group_ids",
    "coarse_dependency_group_ids",
    "fine_identity_ids",
    "source_dependency_group_ids",
    "anchor_reliability",
    "anchor_matchability",
    "anchor_alias_risk",
    "anchor_position_covariance",
)


def _digest_frame(digest: "hashlib._Hash", label: str, value: bytes) -> None:
    """Hash one typed, length-delimited value without concatenation ambiguity."""
    encoded_label = label.encode("utf-8")
    digest.update(struct.pack(">Q", len(encoded_label)))
    digest.update(encoded_label)
    digest.update(struct.pack(">Q", len(value)))
    digest.update(value)


def _semantic_digest(digest: "hashlib._Hash", path: str, value: Any) -> None:
    """Recursively hash schema data with field, dtype, shape, and type framing."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        _digest_frame(digest, f"{path}:kind", b"tensor")
        _digest_frame(digest, f"{path}:dtype", str(tensor.dtype).encode("ascii"))
        _digest_frame(
            digest,
            f"{path}:shape",
            json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"),
        )
        _digest_frame(digest, f"{path}:bytes", tensor_bytes(tensor))
        return
    if isinstance(value, dict):
        _digest_frame(digest, f"{path}:kind", b"mapping")
        keys = list(value)
        if any(not isinstance(key, str) for key in keys):
            raise TypeError(f"{path} contains a non-string schema key")
        keys.sort()
        _digest_frame(
            digest,
            f"{path}:keys",
            json.dumps(keys, separators=(",", ":")).encode("utf-8"),
        )
        for key in keys:
            _semantic_digest(digest, f"{path}.{key}", value[key])
        return
    if isinstance(value, (list, tuple)):
        _digest_frame(
            digest,
            f"{path}:kind",
            b"tuple" if isinstance(value, tuple) else b"list",
        )
        _digest_frame(digest, f"{path}:length", str(len(value)).encode("ascii"))
        for index, item in enumerate(value):
            _semantic_digest(digest, f"{path}[{index}]", item)
        return
    if value is None:
        _digest_frame(digest, f"{path}:kind", b"null")
        return
    if isinstance(value, bool):
        _digest_frame(digest, f"{path}:bool", b"true" if value else b"false")
        return
    if isinstance(value, int):
        _digest_frame(digest, f"{path}:int", str(value).encode("ascii"))
        return
    if isinstance(value, float):
        if math.isnan(value):
            encoded = b"nan"
        elif math.isinf(value):
            encoded = b"+inf" if value > 0 else b"-inf"
        else:
            encoded = struct.pack(">d", value)
        _digest_frame(digest, f"{path}:float64", encoded)
        return
    if isinstance(value, str):
        _digest_frame(digest, f"{path}:string", value.encode("utf-8"))
        return
    raise TypeError(f"{path} has unsupported schema value {type(value).__name__}")


def _semantic_sha256(path: str, value: Any) -> str:
    digest = hashlib.sha256()
    _semantic_digest(digest, path, value)
    return digest.hexdigest()


def anchor_registry(payload: dict) -> dict:
    """Return legacy and complete identities for an Anchor map or Registry.

    ``registry_sha256`` intentionally retains the V1 four-field digest for
    readers that persist it.  ``full_registry_sha256`` is the fail-closed V2
    identity and covers field names, value kinds, tensor dtype/shape/bytes,
    localization tensors, and (for an Evidence-Grounded Registry) the complete
    schema including observation CSR, identity, geometry, selection, and
    evidence.
    """
    legacy_required = (
        "anchor_ids",
        "source_primitive_ids",
        "track_cluster_ids",
        "anchor_type",
    )
    missing = [key for key in legacy_required if key not in payload]
    if missing:
        raise ValueError(f"anchor registry missing fields: {missing}")
    count = int(torch.as_tensor(payload["anchor_ids"]).numel())
    required = (*legacy_required, "anchor_xyz", "anchor_features")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"anchor registry missing fields: {missing}")
    for key in required:
        value = torch.as_tensor(payload[key]).detach().cpu()
        if value.ndim == 0 or value.shape[0] != count:
            raise ValueError(f"{key} does not align with anchor IDs")
    legacy_digest = hashlib.sha256()
    for key in legacy_required:
        value = torch.as_tensor(payload[key]).detach().cpu().contiguous()
        if value.numel() != count:
            raise ValueError(f"{key} does not align with anchor IDs")
        legacy_digest.update(value.numpy().tobytes())
    if "dependency_group_ids" in payload:
        dependency = (
            torch.as_tensor(payload["dependency_group_ids"])
            .detach()
            .cpu()
            .contiguous()
        )
        if dependency.numel() != count:
            raise ValueError("dependency groups do not align with anchors")
        legacy_digest.update(dependency.numpy().tobytes())

    schema = str(payload.get("schema", ""))
    if schema == "lafgs_evidence_grounded_anchor_registry":
        semantic = dict(payload)
    else:
        semantic = {
            "schema": schema,
            "version": payload.get("version"),
            **{
                key: payload[key]
                for key in _LOCALIZATION_ANCHOR_FIELDS
                if key in payload
            },
        }
    for key in _LOCALIZATION_ANCHOR_FIELDS:
        if key not in payload:
            continue
        value = torch.as_tensor(payload[key])
        if value.ndim == 0 or value.shape[0] != count:
            raise ValueError(f"{key} does not align with anchor IDs")
    field_sha256 = {
        key: _semantic_sha256(f"anchor_registry.{key}", value)
        for key, value in sorted(semantic.items())
    }
    return {
        "anchor_count": count,
        "registry_hash_version": 2,
        "covered_fields": sorted(semantic),
        "field_sha256": field_sha256,
        "registry_sha256": legacy_digest.hexdigest(),
        "full_registry_sha256": _semantic_sha256("anchor_registry", semantic),
    }


def git_commit(repo_root: str | Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(repo_root),
        text=True,
    ).strip()


def git_worktree_state(repo_root: str | Path) -> dict:
    root = Path(repo_root)
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        text=True,
    )
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD"],
        cwd=root,
    )
    return {
        "commit": git_commit(root),
        "status_porcelain": status,
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def build_contract(
    *,
    artifact: str | Path,
    kind: str,
    run_type: str,
    repo_root: str | Path,
    parents: dict[str, str | Path],
    resolved_config: dict,
    query_registry_payload: dict | None = None,
    anchor_registry_payload: dict | None = None,
) -> dict:
    artifact = Path(artifact).resolve()
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    parent_records = {}
    for name, path in sorted(parents.items()):
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        parent_records[name] = {
            "path": str(resolved),
            "sha256": sha256_file(resolved),
        }
    return {
        "schema": CONTRACT_SCHEMA,
        "schema_version": CONTRACT_VERSION,
        "kind": kind,
        "run_type": run_type,
        "producer_git": git_worktree_state(repo_root),
        "artifact": {
            "path": str(artifact),
            "sha256": sha256_file(artifact),
        },
        "resolved_config": resolved_config,
        "resolved_config_hash": sha256_json(resolved_config),
        "parent_artifacts": parent_records,
        "query_registry": (
            query_registry(query_registry_payload)
            if query_registry_payload is not None
            else None
        ),
        "anchor_registry": (
            anchor_registry(anchor_registry_payload)
            if anchor_registry_payload is not None
            else None
        ),
    }


def verify_contract(contract: dict) -> None:
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("schema_version") != CONTRACT_VERSION
    ):
        raise ValueError("unsupported artifact contract")
    records = {
        "artifact": contract["artifact"],
        **contract.get("parent_artifacts", {}),
    }
    for name, record in records.items():
        path = Path(record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise ValueError(
                f"{name} hash mismatch: expected {record['sha256']}, "
                f"found {actual}"
            )
    config_hash = sha256_json(contract["resolved_config"])
    if config_hash != contract["resolved_config_hash"]:
        raise ValueError("resolved config hash mismatch")
