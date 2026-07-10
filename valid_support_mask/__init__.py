"""Portable no-reference valid/support mask utilities.

This package intentionally has no STDLoc, teacher-cache, or 3DGS dependency.
It can be copied into another project and used on plain RGB renders.
"""

from .no_reference import (
    NoReferenceValidMask,
    NoReferenceValidMaskBuilder,
    NoReferenceValidMaskConfig,
    NoReferenceValidSupportMask,
    NoReferenceValidSupportMaskBuilder,
    NoReferenceValidSupportMaskConfig,
    save_mask_bundle_pngs,
    save_no_reference_valid_mask_pngs,
)
from .score_mask import (
    ScoreValidMask,
    ScoreValidMaskBuilder,
    ScoreValidMaskConfig,
    save_score_valid_mask_png,
)

ArtifactValidMask = ScoreValidMask
ArtifactValidMaskBuilder = ScoreValidMaskBuilder
ArtifactValidMaskConfig = ScoreValidMaskConfig
save_valid_mask_png = save_score_valid_mask_png

__all__ = [
    "NoReferenceValidSupportMask",
    "NoReferenceValidSupportMaskBuilder",
    "NoReferenceValidSupportMaskConfig",
    "NoReferenceValidMask",
    "NoReferenceValidMaskBuilder",
    "NoReferenceValidMaskConfig",
    "save_mask_bundle_pngs",
    "save_no_reference_valid_mask_pngs",
    "ScoreValidMask",
    "ScoreValidMaskBuilder",
    "ScoreValidMaskConfig",
    "save_score_valid_mask_png",
    "ArtifactValidMask",
    "ArtifactValidMaskBuilder",
    "ArtifactValidMaskConfig",
    "save_valid_mask_png",
]
