"""Compatibility wrapper for the portable valid_support_mask package."""

from valid_support_mask.no_reference import (
    NoReferenceValidMask,
    NoReferenceValidMaskBuilder,
    NoReferenceValidMaskConfig,
    NoReferenceValidSupportMask,
    NoReferenceValidSupportMaskBuilder,
    NoReferenceValidSupportMaskConfig,
    save_mask_bundle_pngs,
    save_no_reference_valid_mask_pngs,
)

__all__ = [
    "NoReferenceValidMask",
    "NoReferenceValidMaskBuilder",
    "NoReferenceValidMaskConfig",
    "NoReferenceValidSupportMask",
    "NoReferenceValidSupportMaskBuilder",
    "NoReferenceValidSupportMaskConfig",
    "save_mask_bundle_pngs",
    "save_no_reference_valid_mask_pngs",
]
