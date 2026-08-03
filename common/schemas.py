"""Versioned artifact schemas shared by the release pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PRIOR_SCHEMA = "lafgs_gaussian_prior"
PRIOR_VERSION = 1


@dataclass(frozen=True)
class ParentArtifact:
    role: str
    path: Path
    sha256: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class PriorManifest:
    prior_type: str
    source_method: str
    source_commit: str
    coordinate_frame: str
    camera_convention: str
    pixel_center_convention: str
    primitive_count: int
    ply_sha256: str
    has_semantic_mask: bool
    sim3_alignment: Mapping[str, Any] | None = None

    def validate(self) -> None:
        if self.prior_type not in {"2dgs", "3dgs"}:
            raise ValueError(f"unsupported Gaussian prior type: {self.prior_type}")
        if self.primitive_count <= 0:
            raise ValueError("Gaussian prior must contain at least one primitive")
        if len(self.ply_sha256) != 64:
            raise ValueError("prior PLY SHA-256 is malformed")
        if self.camera_convention != "w2c":
            raise ValueError("release pipeline requires w2c cameras")
        if self.pixel_center_convention != "grid_index_plus_half":
            raise ValueError("release pipeline requires the +0.5 pixel convention")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": PRIOR_SCHEMA,
            "version": PRIOR_VERSION,
            "type": self.prior_type,
            "source_method": self.source_method,
            "source_commit": self.source_commit,
            "coordinate_frame": self.coordinate_frame,
            "camera_convention": self.camera_convention,
            "pixel_center_convention": self.pixel_center_convention,
            "primitive_count": self.primitive_count,
            "ply_sha256": self.ply_sha256,
            "has_semantic_mask": self.has_semantic_mask,
            "sim3_alignment": self.sim3_alignment,
        }
