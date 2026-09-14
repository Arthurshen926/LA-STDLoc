"""Small, source-addressed producer identities for completion contracts."""

from __future__ import annotations

import platform
from pathlib import Path
import re
import subprocess
import sys

import torch

from common.hashing import sha256_file


SHA1 = re.compile(r"[0-9a-f]{40}")


def capture_producer_identity(*, schema: str, source_paths: tuple[str, ...]) -> dict:
    root = Path(__file__).resolve().parents[1]
    if not schema or not source_paths or len(source_paths) != len(set(source_paths)):
        raise ValueError("producer identity requires a schema and unique source paths")
    paths = [root / relative for relative in source_paths]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if SHA1.fullmatch(commit) is None:
        raise RuntimeError("producer Git commit is invalid")
    return {
        "schema": schema,
        "version": 1,
        "git_commit": commit,
        "source_paths": list(source_paths),
        "source_file_sha256": {
            relative: sha256_file(root / relative) for relative in source_paths
        },
        "runtime": {
            "python": platform.python_version(),
            "python_executable": str(Path(sys.executable).resolve()),
            "torch": str(torch.__version__),
        },
    }


def verify_producer_identity(
    identity: dict,
    *,
    schema: str,
    source_paths: tuple[str, ...],
) -> None:
    expected = capture_producer_identity(schema=schema, source_paths=source_paths)
    if identity != expected:
        raise ValueError("producer identity is stale or differs from current source")
