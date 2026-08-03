"""Content-addressed contracts for LaFGS cross-stage artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch


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


def anchor_registry(payload: dict) -> dict:
    required = (
        "anchor_ids",
        "source_primitive_ids",
        "track_cluster_ids",
        "anchor_type",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"anchor registry missing fields: {missing}")
    count = int(torch.as_tensor(payload["anchor_ids"]).numel())
    rows = []
    for key in required:
        value = torch.as_tensor(payload[key]).detach().cpu().contiguous()
        if value.numel() != count:
            raise ValueError(f"{key} does not align with anchor IDs")
        rows.append(value.numpy().tobytes())
    digest = hashlib.sha256()
    for value in rows:
        digest.update(value)
    if "dependency_group_ids" in payload:
        dependency = (
            torch.as_tensor(payload["dependency_group_ids"])
            .detach()
            .cpu()
            .contiguous()
        )
        if dependency.numel() != count:
            raise ValueError("dependency groups do not align with anchors")
        digest.update(dependency.numpy().tobytes())
    return {"anchor_count": count, "registry_sha256": digest.hexdigest()}


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
