import torch

from localization_training.verified_structure_update import (
    VerifiedStructureConfig,
    collect_coverage_evidence,
    robust_structure_descriptor,
    safe_retirement_candidates,
)


def test_coverage_evidence_requires_distinct_trajectories():
    records = []
    for index, name in enumerate(
        ("seq1/a", "seq2/a", "seq3/a", "seq1/b")
    ):
        records.append(
            {
                "query_index": index,
                "query_name": name,
                "query_rows": torch.tensor([2]),
                "top1_anchor_indices": torch.tensor([5]),
                "category": torch.tensor([2]),
                "canonical_positive_offsets": torch.tensor([0, 1]),
                "canonical_positive_indices": torch.tensor([7]),
                "canonical_positive_reprojection_errors_px": torch.tensor(
                    [1.0]
                ),
                "canonical_positive_contribution_mass": torch.tensor([0.4]),
            }
        )
    candidates, diagnostics = collect_coverage_evidence(
        {"records": records},
        config=VerifiedStructureConfig(
            minimum_trajectories=3, minimum_events=3
        ),
    )
    assert diagnostics["canonical_candidate_count"] == 1
    assert candidates[0]["canonical_index"] == 7
    assert candidates[0]["trajectory_count"] == 3


def test_robust_descriptor_trims_one_opposite_outlier():
    descriptor = robust_structure_descriptor(
        torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [-1.0, 0.0]]
        ),
        torch.ones(4),
        torch.tensor([1.0, 0.0]),
        trim_fraction=0.25,
    )
    assert float(descriptor[0]) > 0.95


def test_safe_retirement_never_removes_base_or_family_parent():
    dynamic = {
        "records": [
            {
                "top1_anchor_indices": torch.tensor([0, 3, 4]),
                "gt_reprojection_errors_px": torch.tensor(
                    [100.0, 100.0, 1.0]
                ),
            }
        ]
    }
    triage_records = []
    for index, seq in enumerate(("seq1", "seq2", "seq3")):
        triage_records.append(
            {
                "query_name": f"{seq}/a",
                "category": torch.tensor([2, 2]),
                "top1_anchor_indices": torch.tensor([0, 3]),
            }
        )
    retired = safe_retirement_candidates(
        active_count=5,
        base_anchor_count=2,
        triage={"records": triage_records},
        dynamic_outcomes=dynamic,
        family_parent_indices=torch.tensor([3]),
        config=VerifiedStructureConfig(),
    )
    assert retired == []
