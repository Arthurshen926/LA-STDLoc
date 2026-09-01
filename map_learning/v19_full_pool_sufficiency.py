"""Truth/competition decomposition for the V19 full-pool audit."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from map_learning.v18_provenance_truth import TRUTH_EQUIVALENT, TRUTH_UNIQUE


def audit_full_pool_sufficiency_rows(
    *,
    truth: Mapping,
    projection_candidate_graph: Mapping,
    retrieved_anchor_rows: torch.Tensor,
    active_anchor_mask: torch.Tensor,
    equivalence_class_ids: torch.Tensor,
    candidate_pool_deficit_authorized: bool,
) -> dict:
    """Separate selection, retrieval, competition, and unresolved deficits."""

    retrieved = torch.as_tensor(retrieved_anchor_rows).long().cpu()
    row_count = int(truth["row_count"])
    if retrieved.ndim != 2 or retrieved.shape[0] != row_count:
        raise ValueError("V19 full-pool audit competition rows do not align")
    offsets = torch.as_tensor(truth["truth_offsets"]).long().cpu()
    truth_anchors = torch.as_tensor(truth["truth_anchor_rows"]).long().cpu()
    status = torch.as_tensor(truth["truth_status"]).long().cpu()
    graph_offsets = torch.as_tensor(
        projection_candidate_graph["candidate_offsets"]
    ).long().cpu()
    active = torch.as_tensor(active_anchor_mask).bool().cpu().reshape(-1)
    equivalence = torch.as_tensor(equivalence_class_ids).long().cpu().reshape(-1)
    if (
        offsets.shape != (row_count + 1,)
        or graph_offsets.shape != (row_count + 1,)
        or retrieved.numel()
        and int(retrieved.max()) >= active.numel()
        or truth_anchors.numel()
        and int(truth_anchors.max()) >= active.numel()
        or equivalence.numel() != active.numel()
    ):
        raise ValueError("V19 full-pool audit Anchor registry differs")

    decisive = (status == TRUTH_UNIQUE) | (status == TRUTH_EQUIVALENT)
    full_truth_count = offsets[1:] - offsets[:-1]
    truth_rows = torch.repeat_interleave(
        torch.arange(row_count), full_truth_count
    )
    active_truth_count = torch.bincount(
        truth_rows[active[truth_anchors]], minlength=row_count
    )
    truth_class = torch.full((row_count,), -1, dtype=torch.long)
    if truth_anchors.numel():
        truth_class[truth_rows] = equivalence[truth_anchors]
    retrieved_truth = equivalence[retrieved] == truth_class[:, None]
    truth_in_retrieval = decisive & retrieved_truth.any(1)
    truth_wins = decisive & retrieved_truth[:, 0]
    active_truth_exists = decisive & (active_truth_count > 0)
    selection_deficit = decisive & ~active_truth_exists
    retrieval_miss = active_truth_exists & ~truth_in_retrieval
    competition_miss = active_truth_exists & truth_in_retrieval & ~truth_wins
    correct_winner = active_truth_exists & truth_wins

    projection_count = graph_offsets[1:] - graph_offsets[:-1]
    unresolved_identity = ~decisive & (projection_count > 0)
    empty_projection = ~decisive & (projection_count == 0)
    certified_candidate_pool_deficit = (
        empty_projection
        if bool(candidate_pool_deficit_authorized)
        else torch.zeros(row_count, dtype=torch.bool)
    )
    unresolved_empty_projection = empty_projection & ~certified_candidate_pool_deficit
    return {
        "schema": "lafgs_v19_full_pool_sufficiency_rows",
        "version": 1,
        "row_count": row_count,
        "candidate_pool_deficit_authorized": bool(
            candidate_pool_deficit_authorized
        ),
        "full_pool_true_anchor_count": full_truth_count,
        "active_map_true_anchor_count": active_truth_count,
        "decisive": decisive,
        "selection_deficit": selection_deficit,
        "retrieval_miss": retrieval_miss,
        "competition_miss": competition_miss,
        "correct_winner": correct_winner,
        "unresolved_identity": unresolved_identity,
        "unresolved_empty_projection": unresolved_empty_projection,
        "certified_candidate_pool_deficit": certified_candidate_pool_deficit,
        "counts": {
            "decisive": int(decisive.sum()),
            "selection_deficit": int(selection_deficit.sum()),
            "retrieval_miss": int(retrieval_miss.sum()),
            "competition_miss": int(competition_miss.sum()),
            "correct_winner": int(correct_winner.sum()),
            "unresolved_identity": int(unresolved_identity.sum()),
            "unresolved_empty_projection": int(unresolved_empty_projection.sum()),
            "certified_candidate_pool_deficit": int(
                certified_candidate_pool_deficit.sum()
            ),
        },
    }


__all__ = ["audit_full_pool_sufficiency_rows"]
