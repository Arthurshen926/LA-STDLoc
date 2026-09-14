"""Strict query, anchor, and parent-artifact registries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from common.hashing import sha256_file


def _unique(values: Iterable[str], *, kind: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{kind} registry contains duplicate entries")
    return result


@dataclass(frozen=True)
class QueryRegistry:
    names: tuple[str, ...]

    @classmethod
    def from_names(cls, names: Iterable[str]) -> "QueryRegistry":
        return cls(_unique(names, kind="query"))

    def require_exact(self, other: "QueryRegistry") -> None:
        if self.names != other.names:
            raise ValueError("query registry order or membership differs")


@dataclass(frozen=True)
class AnchorRegistry:
    ids: torch.Tensor

    @classmethod
    def from_ids(cls, values) -> "AnchorRegistry":
        ids = torch.as_tensor(values).detach().cpu().long().reshape(-1)
        if torch.unique(ids).numel() != ids.numel():
            raise ValueError("anchor registry contains duplicate IDs")
        return cls(ids)

    def require_exact(self, other: "AnchorRegistry") -> None:
        if not torch.equal(self.ids, other.ids):
            raise ValueError("anchor registry order or membership differs")


@dataclass(frozen=True)
class ArtifactReference:
    path: Path
    sha256: str

    @classmethod
    def capture(cls, path: str | Path) -> "ArtifactReference":
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return cls(resolved, sha256_file(resolved))

    def verify(self) -> None:
        actual = sha256_file(self.path)
        if actual != self.sha256:
            raise ValueError(f"stale or replaced artifact: {self.path}")
