import torch

from evidence.v7_anchor_contamination import (
    aggregate_anchor_reliability,
    bounded_descriptor_reconstruction,
    enrichment_table,
    gather_observation_rows,
)


def test_observation_rows_and_anchor_reliability_are_lineage_exact() -> None:
    rows = [torch.tensor([True, False]), torch.tensor([False, True, True])]
    query = torch.tensor([0, 1, 0, 1, 1, 0])
    keypoint = torch.tensor([0, 0, 1, 1, 2, 1])
    gathered = gather_observation_rows(rows, query, keypoint)
    assert gathered.tolist() == [True, False, False, True, True, False]
    aggregate = aggregate_anchor_reliability(
        observation_valid=gathered,
        observation_structure_supported=torch.tensor(
            [True, False, False, True, True, False]
        ),
        observation_offsets=torch.tensor([0, 3, 6]),
        observation_query_indices=query,
        query_family_ids=torch.tensor([0, 1]),
    )
    assert aggregate["valid_observation_count"].tolist() == [1, 2]
    assert aggregate["valid_view_family_count"].tolist() == [1, 1]
    assert not bool(aggregate["pure_contamination"].any())


def test_pure_contamination_requires_two_families_and_zero_valid_rows() -> None:
    result = aggregate_anchor_reliability(
        observation_valid=torch.tensor([False, False, False, True, True, True]),
        observation_structure_supported=torch.tensor(
            [False, False, False, True, True, True]
        ),
        observation_offsets=torch.tensor([0, 3, 6]),
        observation_query_indices=torch.tensor([0, 1, 0, 0, 1, 1]),
        query_family_ids=torch.tensor([2, 7]),
    )
    assert result["pure_contamination"].tolist() == [True, False]
    assert result["descriptor_reconstructable"].tolist() == [False, True]


def test_descriptor_reconstruction_obeys_trust_region() -> None:
    current = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    proposed = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    output, angles = bounded_descriptor_reconstruction(
        current, proposed, torch.tensor([True, False]), maximum_angle_deg=5.0
    )
    assert 4.9 <= float(angles[0]) <= 5.01
    assert float(angles[1]) == 0.0
    assert torch.allclose(torch.linalg.norm(output, dim=1), torch.ones(2))


def test_enrichment_uses_anchor_universe_reference() -> None:
    result = enrichment_table(
        anchor_rows=torch.tensor([0, 2]),
        anchor_positive=torch.tensor([True, False, True, False]),
    )
    assert result["event_positive_fraction"] == 1.0
    assert result["reference_positive_fraction"] == 0.5
    assert result["enrichment_ratio"] == 2.0
