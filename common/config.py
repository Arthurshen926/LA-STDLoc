"""Strict configuration contract for the frozen paper mainline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from common.hashing import canonical_json, sha256_file


SCHEMA = "lafgs_paper_mainline"
VERSION = 1
OFFLINE_CHAIN = (
    "frozen_rgb_gaussian_prior",
    "native_superpoint_mapping_observations",
    "kcs_gwff_initialization",
    "wide_scaffold_self_localization_reconstruction",
    "track_first_cross_view_tracks",
    "robust_triangulation",
    "gaussian_raster_provenance",
    "track_centric_localization_anchor_pool",
    "localization_topology_distillation",
    "compact_map_self_localization_metric_refresh",
    "compact_single_descriptor_map",
)
TOP_LEVEL_KEYS = frozenset(
    {"schema", "version", "method", "prior", "initialization", "reconstruction", "deployment", "seeds"}
)


@dataclass(frozen=True)
class MainlineConfig:
    path: Path
    values: Mapping[str, Any]
    file_sha256: str
    resolved_sha256: str

    def validate(self) -> None:
        values = self.values
        unknown = set(values) - TOP_LEVEL_KEYS
        if unknown:
            raise ValueError(f"unknown paper-mainline config sections: {sorted(unknown)}")
        if values.get("schema") != SCHEMA or values.get("version") != VERSION:
            raise ValueError("unsupported paper-mainline config schema")
        method = values.get("method", {})
        if method.get("status") != "frozen":
            raise ValueError("paper-mainline method must be frozen")
        if tuple(method.get("offline_chain", ())) != OFFLINE_CHAIN:
            raise ValueError("offline method chain differs from the frozen contract")
        if values.get("prior", {}).get("frozen") is not True:
            raise ValueError("RGB Gaussian prior must remain frozen")
        deployment = values.get("deployment", {})
        required = {
            "sparse_frontend": "ulfloc_native_metric",
            "global_topk": 1,
            "max_matches_per_keypoint": 0,
            "max_matches_per_landmark": 0,
            "reprojection_error_px": 12,
            "confidence": 0.99999,
            "maximum_iterations": 100000,
            "minimum_iterations": 1000,
            "solver": "poselib",
            "pose_solves": 1,
        }
        mismatched = {
            key: (deployment.get(key), expected)
            for key, expected in required.items()
            if deployment.get(key) != expected
        }
        if mismatched:
            raise ValueError(f"deployment contract changed: {mismatched}")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "lafgs_resolved_mainline_config",
            "version": 1,
            "config_path": str(self.path),
            "config_sha256": self.file_sha256,
            "resolved_config_sha256": self.resolved_sha256,
            "offline_chain": list(OFFLINE_CHAIN),
            "deployment": dict(self.values["deployment"]),
        }


def load_mainline_config(path: str | Path) -> MainlineConfig:
    path = Path(path).expanduser().resolve()
    values = yaml.safe_load(path.read_text())
    if not isinstance(values, dict):
        raise ValueError("mainline config must resolve to a mapping")
    config = MainlineConfig(
        path=path,
        values=values,
        file_sha256=sha256_file(path),
        resolved_sha256=__import__("hashlib").sha256(
            canonical_json(values).encode("ascii")
        ).hexdigest(),
    )
    config.validate()
    return config
