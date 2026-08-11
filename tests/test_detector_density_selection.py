import torch

from scripts.select_detector_density import select_smallest_stable
from topology.deployment_revision import _csr_contains_per_row


def _candidate(keypoints, median, mean, p90, raw, inlier):
    return {
        "keypoints": keypoints,
        "anchor_count": keypoints * 8,
        "worst_median_te_cm": median,
        "worst_mean_te_cm": mean,
        "worst_p90_te_cm": p90,
        "catastrophic_100cm_count": 0,
        "minimum_raw_gt_precision_percent": raw,
        "minimum_inlier_gt_precision_percent": inlier,
        "worst_mean_hypotheses": 1000,
        "normalized_coverage_p10": 1.0,
    }


def test_density_selector_prefers_smallest_non_degraded_frontend():
    selected, report = select_smallest_stable(
        [
            _candidate(1024, 1.00, 1.00, 2.00, 12.0, 30.0),
            _candidate(2048, 0.99, 0.99, 1.99, 12.2, 30.5),
        ]
    )
    assert selected["keypoints"] == 1024
    assert report["policy"].startswith("minimum_density")


def test_density_selector_keeps_larger_frontend_for_tail_gain():
    selected, _ = select_smallest_stable(
        [
            _candidate(1024, 1.00, 1.20, 2.40, 12.0, 30.0),
            _candidate(2048, 0.98, 1.00, 2.00, 11.8, 29.5),
        ]
    )
    assert selected["keypoints"] == 2048


def test_density_prefix_membership_uses_matching_csr_rows():
    record = {
        "positive_offsets": torch.tensor([0, 2, 2, 3, 5]),
        "positive_indices": torch.tensor([3, 7, 5, 1, 9]),
    }
    matched, nonempty = _csr_contains_per_row(
        record, "positive", torch.tensor([7, 2, 4])
    )
    assert matched.tolist() == [True, False, False]
    assert nonempty.tolist() == [True, False, True]
