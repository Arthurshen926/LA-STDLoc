import torch

from evidence.virtual_track_experiment import (
    DRY_RUN_THRESHOLDS,
    dry_run_passes,
    enforce_one_observation_per_family,
)


def test_duplicate_pose_family_cannot_supply_two_track_observations():
    tracks = {
        "track_index": torch.tensor([0, 0, 0, 1, 1, 1]),
        "query_index": torch.tensor([0, 1, 2, 0, 2, 3]),
        "keypoint_index": torch.tensor([4, 5, 6, 7, 8, 9]),
        "confidence": torch.tensor([0.3, 0.9, 0.7, 0.8, 0.2, 0.5]),
        "track_level": torch.tensor([2, 1], dtype=torch.int8),
    }
    filtered, audit = enforce_one_observation_per_family(
        tracks, torch.tensor([10, 10, 11, 12])
    )
    assert filtered["query_index"].tolist() == [1, 2, 0, 2, 3]
    assert audit["duplicate_family_observation_count"] == 1
    assert audit["maximum_observations_per_track_family"] == 1


def test_dry_run_gate_is_frozen_and_test_independent():
    metrics = dict(DRY_RUN_THRESHOLDS)
    metrics.update(family_contract_passed=True, gt_visible_diagnostic=None)
    passed, failures = dry_run_passes(metrics)
    assert passed and failures == []
    metrics["test_median_translation_cm"] = 10_000  # irrelevant field
    assert dry_run_passes(metrics)[0]
    metrics["new_anchor_count"] = 0
    assert not dry_run_passes(metrics)[0]
