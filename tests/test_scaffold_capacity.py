from __future__ import annotations

import pytest
from types import SimpleNamespace

from map_learning.bootstrap import (
    _assert_cached_consensus_scaffold,
    _resolve_consensus_capacity,
)


def test_adaptive_scaffold_treats_budget_as_safety_cap() -> None:
    resolved, fallback, policy = _resolve_consensus_capacity(
        44_412,
        48_000,
        allow_nonconsensus_fallback=False,
        allow_underfill=True,
    )
    assert resolved == 44_412
    assert fallback is False
    assert policy == "consensus_saturation_cap"


def test_legacy_fixed_scaffold_requires_explicit_fallback() -> None:
    resolved, fallback, policy = _resolve_consensus_capacity(
        44_412,
        48_000,
        allow_nonconsensus_fallback=True,
        allow_underfill=False,
    )
    assert resolved == 48_000
    assert fallback is True
    assert policy == "fixed_with_nonconsensus_fallback"


def test_strict_scaffold_rejects_unexplained_underfill() -> None:
    with pytest.raises(RuntimeError, match="too few consensus"):
        _resolve_consensus_capacity(
            44_412,
            48_000,
            allow_nonconsensus_fallback=False,
            allow_underfill=False,
        )


def _resume_args() -> SimpleNamespace:
    return SimpleNamespace(
        scaffold_budget=48_000,
        ulf_consensus_keypoints=1024,
        ulf_consensus_radius_px=0.5,
        ulf_consensus_min_votes=2,
        ulf_consensus_min_visible_views=4,
        ulf_consensus_min_rate=0.01,
        ulf_consensus_view_bins=8,
        ulf_consensus_min_distinct_view_bins=2,
        ulf_consensus_trajectory_bins=8,
        ulf_consensus_min_distinct_trajectory_bins=2,
        ulf_consensus_independent_bin_scoring=True,
        ulf_consensus_max_candidates_per_view=0,
        ulf_consensus_max_per_voxel=8,
        ulf_consensus_extent_quantile=0.01,
        ulf_support_view_sampling="uniform",
        ulf_support_mask_policy="support_rgb_only",
        ulf_consensus_allow_underfill=True,
    )


def _resume_metadata() -> dict:
    args = _resume_args()
    return {
        "requested_budget": args.scaffold_budget,
        "consensus_sparse_keypoints": args.ulf_consensus_keypoints,
        "consensus_radius_px": args.ulf_consensus_radius_px,
        "minimum_votes": args.ulf_consensus_min_votes,
        "minimum_visible_views": args.ulf_consensus_min_visible_views,
        "minimum_consensus_rate": args.ulf_consensus_min_rate,
        "distinct_view_bins": args.ulf_consensus_view_bins,
        "minimum_distinct_view_bins": args.ulf_consensus_min_distinct_view_bins,
        "distinct_trajectory_bins": args.ulf_consensus_trajectory_bins,
        "minimum_distinct_trajectory_bins": (
            args.ulf_consensus_min_distinct_trajectory_bins
        ),
        "independent_bin_scoring": True,
        "candidate_cap_per_view": 0,
        "max_per_voxel": 8,
        "voxel_extent_quantile": 0.01,
        "support_view_sampling": "uniform",
        "support_mask_policy": "support_rgb_only",
        "capacity_policy": "consensus_saturation_cap",
    }


def test_cached_adaptive_scaffold_requires_matching_policy() -> None:
    _assert_cached_consensus_scaffold(_resume_metadata(), _resume_args())
    stale = _resume_metadata()
    stale["consensus_sparse_keypoints"] = 2048
    with pytest.raises(ValueError, match="policy mismatch"):
        _assert_cached_consensus_scaffold(stale, _resume_args())
