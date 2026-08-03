"""Typed validation for the frozen Rendering-to-Localization mainline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


SCHEMA = "lafgs_paper_mainline"
VERSION = 1

REQUIRED_OFFLINE_CHAIN = (
    "frozen_rgb_gaussian_prior",
    "native_superpoint_mapping_observations",
    "kcs_gwff_initialization",
    "track_first_cross_view_tracks",
    "robust_triangulation",
    "gaussian_raster_provenance",
    "track_centric_localization_anchor_pool",
    "localization_topology_distillation",
    "self_localization_guided_descriptor_reconstruction",
    "compact_single_descriptor_map",
)

RESEARCH_ONLY_COMPONENTS = frozenset(
    {
        "viewpoint_completion",
        "trajectory_stable_candidate_teacher",
        "family_prototypes",
        "selector",
        "pair_lgcv",
        "dense_refinement",
        "differentiable_pnp",
        "bounded_bundle_adjustment",
        "learned_sampler",
        "custom_consensus_solver",
        "test_time_rendering",
    }
)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


@dataclass(frozen=True)
class MainlineProtocol:
    """Resolved, validated method contract used by training and figures."""

    path: Path
    resolved: Mapping[str, Any]
    sha256: str
    resolved_sha256: str

    @property
    def offline_chain(self) -> tuple[str, ...]:
        return tuple(self.resolved["method"]["offline_chain"])

    @property
    def excluded_components(self) -> frozenset[str]:
        return frozenset(
            self.resolved["method"]["excluded_by_default"]
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "lafgs_resolved_mainline_protocol",
            "version": 1,
            "config_path": str(self.path),
            "config_sha256": self.sha256,
            "resolved_config_sha256": self.resolved_sha256,
            "method_name": self.resolved["method"]["name"],
            "offline_chain": list(self.offline_chain),
            "excluded_by_default": sorted(self.excluded_components),
            "deployment": dict(self.resolved["deployment"]),
        }

    def validate(self) -> None:
        config = self.resolved
        if config.get("schema") != SCHEMA or int(config.get("version", -1)) != VERSION:
            raise ValueError("unsupported LaFGS paper-mainline schema")
        method = config.get("method", {})
        if method.get("status") != "frozen":
            raise ValueError("paper mainline must be frozen")
        if self.offline_chain != REQUIRED_OFFLINE_CHAIN:
            raise ValueError(
                "offline chain differs from the frozen paper method: "
                f"{self.offline_chain!r}"
            )
        missing = RESEARCH_ONLY_COMPONENTS - self.excluded_components
        if missing:
            raise ValueError(
                "research-only components are not excluded by default: "
                f"{sorted(missing)}"
            )
        if config.get("prior", {}).get("frozen") is not True:
            raise ValueError("RGB Gaussian prior must be frozen")
        deployment = config.get("deployment", {})
        expected = {
            "sparse_frontend": "ulfloc_native_metric",
            "global_topk": 1,
            "max_matches_per_keypoint": 0,
            "max_matches_per_landmark": 0,
            "solver": "poselib",
            "pose_solves": 1,
            "dense_refinement": False,
            "rendering": False,
        }
        mismatched = {
            key: (deployment.get(key), value)
            for key, value in expected.items()
            if deployment.get(key) != value
        }
        if mismatched:
            raise ValueError(
                f"deployment no longer matches the one-shot sparse contract: {mismatched}"
            )
        for section in ("family", "selector"):
            if config.get(section, {}).get("enabled_by_default") is not False:
                raise ValueError(f"{section} must remain disabled by default")


def load_mainline_protocol(path: str | Path) -> MainlineProtocol:
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    resolved = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(resolved, dict):
        raise ValueError("mainline config must resolve to a mapping")
    protocol = MainlineProtocol(
        path=path,
        resolved=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
        resolved_sha256=hashlib.sha256(
            _canonical_json(resolved).encode("ascii")
        ).hexdigest(),
    )
    protocol.validate()
    return protocol
