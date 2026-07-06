"""Artifact-aware pseudo-query utilities for LA-STDLoc."""

from .no_reference_valid_mask import (
    NoReferenceValidMask,
    NoReferenceValidMaskBuilder,
    NoReferenceValidMaskConfig,
)
from .repair import ArtifactRepair, ArtifactRepairConfig
from .valid_mask import ArtifactValidMask, ArtifactValidMaskBuilder, ArtifactValidMaskConfig
from .pseudo_query import (
    PseudoQueryCamera,
    PseudoQueryManifest,
    PseudoQueryRecord,
    PseudoTeacherCache,
)
from .rgb_teacher import RgbTeacherManifest, RgbTeacherSpec

__all__ = [
    "ArtifactDetector",
    "ArtifactDetectorConfig",
    "ArtifactEvidence",
    "NoReferenceValidMask",
    "NoReferenceValidMaskBuilder",
    "NoReferenceValidMaskConfig",
    "ArtifactRepair",
    "ArtifactRepairConfig",
    "ArtifactValidMask",
    "ArtifactValidMaskBuilder",
    "ArtifactValidMaskConfig",
    "PseudoQueryCamera",
    "PseudoQueryManifest",
    "PseudoQueryRecord",
    "PseudoTeacherCache",
    "RgbTeacherManifest",
    "RgbTeacherSpec",
]

_LAZY_EXPORTS = {
    "ArtifactDetector": (".detector", "ArtifactDetector"),
    "ArtifactDetectorConfig": (".detector", "ArtifactDetectorConfig"),
    "ArtifactEvidence": (".detector", "ArtifactEvidence"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(importlib.import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
