"""Bounded, fail-closed content-addressed storage for expensive pipeline nodes."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import tempfile
from typing import Mapping
import uuid

import torch

from common.hashing import canonical_json, sha256_file


SCHEMA = "lafgs_content_addressed_dag_node"
VERSION = 1
FICLONE = 0x40049409


def _safe_name(value: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"invalid DAG artifact name: {value!r}")
    return value


def path_content_record(path: str | Path) -> dict:
    """Hash a file or directory tree without following symbolic links."""
    path = Path(path).expanduser()
    if path.is_symlink():
        raise ValueError(f"DAG input is a symbolic link: {path}")
    path = path.resolve()
    if path.is_file():
        return {
            "kind": "file",
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = []
    total = 0
    for child in sorted(path.rglob("*")):
        relative = str(child.relative_to(path))
        if child.is_symlink():
            raise ValueError(f"DAG input tree contains a symlink: {child}")
        if child.is_file():
            size = child.stat().st_size
            total += size
            files.append(
                {
                    "relative_path": relative,
                    "sha256": sha256_file(child),
                    "size_bytes": size,
                }
            )
    digest = hashlib.sha256(canonical_json({"files": files}).encode()).hexdigest()
    return {
        "kind": "directory",
        "tree_sha256": digest,
        "size_bytes": total,
        "file_count": len(files),
    }


def source_identity(root: str | Path, relative_paths: tuple[str, ...]) -> dict:
    """Return a commit-independent identity for exactly the node-producing code."""
    root = Path(root).expanduser().resolve()
    if not relative_paths or len(relative_paths) != len(set(relative_paths)):
        raise ValueError("DAG producer paths must be nonempty and unique")
    sources = {}
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        sources[relative] = sha256_file(path)
    return {"schema": "lafgs_dag_source_identity", "version": 1, "sources": sources}


def runtime_identity() -> dict:
    """Bind numerical runtime and rasterizer ABI to cached tensor artifacts."""
    try:
        distribution = importlib.metadata.distribution("gsplat")
        gsplat_version = distribution.version
        binaries = {}
        for relative in distribution.files or ():
            path = Path(distribution.locate_file(relative))
            if path.is_file() and path.suffix in {".so", ".pyd", ".dll", ".dylib"}:
                binaries[str(relative)] = sha256_file(path)
    except importlib.metadata.PackageNotFoundError:
        gsplat_version = None
        binaries = {}
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "name": properties.name,
                    "compute_capability": [properties.major, properties.minor],
                    "total_memory": properties.total_memory,
                }
            )
    return {
        "schema": "lafgs_dag_runtime_identity",
        "version": 1,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cudnn": int(torch.backends.cudnn.version() or 0),
        "cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST", ""),
        "visible_cuda_devices": cuda_devices,
        "gsplat": gsplat_version,
        "gsplat_binary_sha256": binaries,
    }


def node_spec(
    *,
    node: str,
    config: Mapping,
    upstream: Mapping[str, Mapping],
    producer: Mapping,
) -> dict:
    """Build the canonical node key payload from config, inputs, and producer."""
    if not node or not isinstance(config, Mapping) or not isinstance(upstream, Mapping):
        raise ValueError("invalid DAG node specification")
    payload = {
        "schema": "lafgs_content_addressed_dag_key",
        "version": 1,
        "node": node,
        "config": dict(config),
        "upstream": dict(upstream),
        "producer": dict(producer),
    }
    payload["key_sha256"] = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    return payload


def _clone_or_copy(source: Path, target: Path) -> str:
    """Prefer a copy-on-write reflink and fall back to a byte copy."""
    with source.open("rb") as source_handle, target.open("xb") as target_handle:
        try:
            fcntl.ioctl(target_handle.fileno(), FICLONE, source_handle.fileno())
            mode = "reflink"
        except OSError:
            source_handle.seek(0)
            shutil.copyfileobj(source_handle, target_handle, length=8 << 20)
            mode = "byte_copy"
        target_handle.flush()
        os.fsync(target_handle.fileno())
    shutil.copystat(source, target, follow_symlinks=False)
    return mode


class ContentAddressedStore:
    """An immutable DAG store with explicit capacity limits and atomic publish."""

    def __init__(
        self,
        root: str | Path,
        *,
        maximum_node_bytes: int,
        maximum_store_bytes: int,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.maximum_node_bytes = int(maximum_node_bytes)
        self.maximum_store_bytes = int(maximum_store_bytes)
        if self.maximum_node_bytes <= 0 or self.maximum_store_bytes <= 0:
            raise ValueError("DAG cache byte limits must be positive")
        if self.maximum_node_bytes > self.maximum_store_bytes:
            raise ValueError("DAG node limit cannot exceed the store limit")
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(spec: Mapping) -> str:
        key = str(spec.get("key_sha256", ""))
        unsigned = dict(spec)
        unsigned.pop("key_sha256", None)
        expected = hashlib.sha256(canonical_json(unsigned).encode()).hexdigest()
        if key != expected:
            raise ValueError("DAG node key does not match its canonical specification")
        return key

    def node_path(self, spec: Mapping) -> Path:
        return self.root / _safe_name(str(spec["node"])) / self._key(spec)

    def artifact_path(self, spec: Mapping, name: str) -> Path:
        return self.node_path(spec) / "artifacts" / _safe_name(name)

    def _store_size(self) -> int:
        return sum(
            path.stat().st_size
            for path in self.root.rglob("*")
            if path.is_file() and not path.is_symlink()
        )

    def load(self, spec: Mapping) -> dict[str, Path] | None:
        node = self.node_path(spec)
        manifest_path = node / "manifest.json"
        if not manifest_path.is_file():
            return None
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("schema") != SCHEMA
            or manifest.get("version") != VERSION
            or manifest.get("complete") is not True
            or manifest.get("spec") != dict(spec)
        ):
            raise ValueError(f"stale or incompatible DAG manifest: {manifest_path}")
        records = manifest.get("artifacts")
        if not isinstance(records, dict) or not records:
            raise ValueError(f"DAG manifest has no artifacts: {manifest_path}")
        expected_files = {"manifest.json"}
        resolved = {}
        for name, record in records.items():
            _safe_name(name)
            file_name = _safe_name(str(record.get("file", "")))
            if file_name != name:
                raise ValueError(f"DAG artifact name/file mismatch: {name}")
            path = node / "artifacts" / file_name
            expected_files.add(str(path.relative_to(node)))
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != int(record["size_bytes"])
                or sha256_file(path) != record["sha256"]
            ):
                raise ValueError(f"DAG artifact failed SHA/size verification: {path}")
            resolved[name] = path.resolve()
        actual_files = {
            str(path.relative_to(node)) for path in node.rglob("*") if path.is_file()
        }
        if actual_files != expected_files:
            raise ValueError(f"DAG node contains unregistered files: {node}")
        return resolved

    def materialize(
        self,
        spec: Mapping,
        destination: str | Path,
    ) -> tuple[dict[str, Path], dict[str, str]]:
        """Create a run-owned, cache-prune-safe snapshot of one verified node."""
        cached = self.load(spec)
        if cached is None:
            raise FileNotFoundError(self.node_path(spec))
        destination = Path(destination).expanduser().resolve()
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        )
        temporary.mkdir()
        modes = {}
        try:
            for name, source in cached.items():
                target = temporary / name
                modes[name] = _clone_or_copy(source, target)
                if (
                    target.stat().st_size != source.stat().st_size
                    or sha256_file(target) != sha256_file(source)
                ):
                    raise ValueError(f"run-local DAG materialization differs: {target}")
            temporary.rename(destination)
            return (
                {name: (destination / name).resolve() for name in cached},
                modes,
            )
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def publish(self, spec: Mapping, artifacts: Mapping[str, str | Path]) -> dict[str, Path]:
        """Atomically install one immutable node, refusing unbounded growth."""
        if not artifacts:
            raise ValueError("cannot publish an empty DAG node")
        final = self.node_path(spec)
        existing = self.load(spec)
        if existing is not None:
            return existing
        sources = {}
        for name, raw_path in artifacts.items():
            raw = Path(raw_path).expanduser()
            if raw.is_symlink():
                raise ValueError(f"DAG artifact source is a symlink: {raw}")
            sources[name] = raw.resolve()
        if len(set(sources.values())) != len(sources):
            raise ValueError("DAG artifacts must have unique source paths")
        source_inodes = []
        for name, source in sources.items():
            _safe_name(name)
            if source.is_symlink() or not source.is_file():
                raise FileNotFoundError(source)
            identity = (source.stat().st_dev, source.stat().st_ino)
            if identity in source_inodes:
                raise ValueError("DAG artifacts must not alias one source inode")
            source_inodes.append(identity)
        lock_path = self.root / ".publish.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            existing = self.load(spec)
            if existing is not None:
                return existing
            if final.exists():
                # A node without its atomic-last manifest is incomplete derived
                # state. Remove only this exact content-addressed directory.
                if (final / "manifest.json").exists():
                    raise ValueError(f"invalid completed DAG node: {final}")
                shutil.rmtree(final)
            node_bytes = sum(path.stat().st_size for path in sources.values())
            planned_records = {
                name: {
                    "file": name,
                    "sha256": sha256_file(source),
                    "size_bytes": source.stat().st_size,
                    "copy_mode": "byte_copy",
                }
                for name, source in sorted(sources.items())
            }
            planned_manifest = {
                "schema": SCHEMA,
                "version": VERSION,
                "complete": True,
                "immutable": True,
                "spec": dict(spec),
                "artifacts": planned_records,
                "artifact_bytes": node_bytes,
            }
            manifest_reserve = len(
                (json.dumps(planned_manifest, indent=2, sort_keys=True) + "\n").encode()
            )
            planned_total = node_bytes + manifest_reserve
            if planned_total > self.maximum_node_bytes:
                raise ValueError(
                    f"DAG node including manifest is at least {planned_total} bytes, "
                    f"above limit {self.maximum_node_bytes}"
                )
            store_bytes_before = self._store_size()
            if store_bytes_before + planned_total > self.maximum_store_bytes:
                raise ValueError(
                    "DAG cache capacity including manifest would be exceeded; "
                    "prune it explicitly"
                )
            final.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(
                tempfile.mkdtemp(prefix=f".{final.name}.", dir=final.parent)
            )
            try:
                artifact_dir = temporary / "artifacts"
                artifact_dir.mkdir()
                records = {}
                for name, source in sorted(sources.items()):
                    target = artifact_dir / name
                    copy_mode = _clone_or_copy(source, target)
                    target.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
                    records[name] = {
                        "file": name,
                        "sha256": sha256_file(target),
                        "size_bytes": target.stat().st_size,
                        "copy_mode": copy_mode,
                    }
                    if (
                        records[name]["sha256"] != planned_records[name]["sha256"]
                        or records[name]["size_bytes"]
                        != planned_records[name]["size_bytes"]
                    ):
                        raise ValueError(f"DAG publish source changed during copy: {source}")
                manifest = {
                    "schema": SCHEMA,
                    "version": VERSION,
                    "complete": True,
                    "immutable": True,
                    "spec": dict(spec),
                    "artifacts": records,
                    "artifact_bytes": node_bytes,
                }
                manifest_path = temporary / "manifest.json"
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                )
                with manifest_path.open("rb") as handle:
                    os.fsync(handle.fileno())
                actual_node_bytes = sum(
                    path.stat().st_size
                    for path in temporary.rglob("*")
                    if path.is_file()
                )
                if actual_node_bytes > self.maximum_node_bytes:
                    raise ValueError(
                        f"DAG node including manifest is {actual_node_bytes} bytes, "
                        f"above limit {self.maximum_node_bytes}"
                    )
                if store_bytes_before + actual_node_bytes > self.maximum_store_bytes:
                    raise ValueError(
                        "DAG cache capacity including manifest would be exceeded; "
                        "prune it explicitly"
                    )
                temporary.rename(final)
                result = self.load(spec)
                if result is None:
                    raise RuntimeError("DAG node publication did not become visible")
                return result
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
