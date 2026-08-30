"""Current-competition proposals for reversible active-set map control."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch

from map_learning.v18_provenance_truth import TRUTH_EQUIVALENT, TRUTH_UNIQUE


def propose_current_competition_actions(
    *,
    records: Sequence[Mapping],
    anchor_count: int,
    minimum_pose_families: int = 2,
    minimum_queries: int = 2,
    dominated_margin: float = 0.005,
    maximum_inactive_redundancy_candidates: int = 4096,
    active_anchor_mask: torch.Tensor | None = None,
) -> dict:
    """Generate harmful/dominated/inactive proposals from the current graph.

    Historical removal authorization is intentionally not an input.  Every
    proposal is recomputed from current winners and provenance truth.  Proposal
    status never authorizes deployment; exact Top-1/PoseLib and reserve floors
    remain mandatory controller gates.
    """

    count = int(anchor_count)
    if count < 1:
        raise ValueError("active-set controller requires a non-empty map")
    active_registry = (
        torch.ones(count, dtype=torch.bool)
        if active_anchor_mask is None
        else torch.as_tensor(active_anchor_mask).bool().cpu().reshape(-1)
    )
    if active_registry.numel() != count:
        raise ValueError("current active registry does not align with the map")
    stats: dict[int, dict] = defaultdict(
        lambda: {
            "topl_queries": set(),
            "winner_queries": set(),
            "winner_families": set(),
            "wrong_winner_queries": set(),
            "wrong_winner_families": set(),
            "truth_queries": set(),
            "truth_families": set(),
            "correct_winner_queries": set(),
            "dominated_queries": set(),
            "dominated_families": set(),
        }
    )
    for record in records:
        query = int(record["query_index"])
        family = int(record["pose_family_id"])
        candidates = torch.as_tensor(record["candidate_anchor_rows"]).long()
        scores = torch.as_tensor(record["candidate_scores"]).float()
        truth = record["truth"]
        status = torch.as_tensor(truth["truth_status"]).long()
        offsets = torch.as_tensor(truth["truth_offsets"]).long()
        truth_anchors = torch.as_tensor(truth["truth_anchor_rows"]).long()
        if candidates.shape != scores.shape or candidates.shape[0] != status.numel():
            raise ValueError("current competition record fields do not align")
        if candidates.numel() and (
            int(candidates.min()) < 0 or int(candidates.max()) >= count
        ):
            raise ValueError("current competition references an Anchor outside the map")
        current_winners = torch.as_tensor(
            record.get("current_winner_anchor_rows", candidates[:, 0])
        ).long()
        if current_winners.shape != candidates.shape[:1]:
            raise ValueError("current winner rows do not align with competition rows")
        active_columns = active_registry[candidates]
        for anchor in torch.unique(candidates[active_columns]).tolist():
            stats[int(anchor)]["topl_queries"].add(query)
        decisive_rows = torch.nonzero(
            (status == TRUTH_UNIQUE) | (status == TRUTH_EQUIVALENT),
            as_tuple=False,
        ).reshape(-1)
        for row in decisive_rows.tolist():
            winner = int(current_winners[row])
            item = stats[winner]
            item["winner_queries"].add(query)
            item["winner_families"].add(family)
            start, stop = int(offsets[row]), int(offsets[row + 1])
            local_truth = truth_anchors[start:stop]
            truth_set = set(local_truth.tolist())
            for anchor in truth_set:
                stats[int(anchor)]["truth_queries"].add(query)
                stats[int(anchor)]["truth_families"].add(family)
            if winner in truth_set:
                item["correct_winner_queries"].add(query)
            else:
                item["wrong_winner_queries"].add(query)
                item["wrong_winner_families"].add(family)
            truth_columns = torch.nonzero(
                active_columns[row] & torch.isin(candidates[row], local_truth),
                as_tuple=False,
            ).reshape(-1)
            if truth_columns.numel() < 2:
                continue
            best_truth_column = truth_columns[scores[row, truth_columns].argmax()]
            best_truth_anchor = int(candidates[row, best_truth_column])
            best_truth_score = float(scores[row, best_truth_column])
            for column in truth_columns.tolist():
                anchor = int(candidates[row, column])
                if anchor == best_truth_anchor:
                    continue
                if best_truth_score >= float(scores[row, column]) + float(dominated_margin):
                    stats[anchor]["dominated_queries"].add(query)
                    stats[anchor]["dominated_families"].add(family)

    proposals = []
    observed_anchors = sorted(stats)
    for anchor in observed_anchors:
        if not bool(active_registry[anchor]):
            continue
        item = stats[anchor]
        truth_protected = bool(item["truth_queries"])
        if (
            not truth_protected
            and len(item["wrong_winner_queries"]) >= int(minimum_queries)
            and len(item["wrong_winner_families"]) >= int(minimum_pose_families)
        ):
            kind = "harmful_removal"
            evidence_queries = item["wrong_winner_queries"]
            evidence_families = item["wrong_winner_families"]
        elif (
            truth_protected
            and not item["correct_winner_queries"]
            and len(item["dominated_queries"]) >= int(minimum_queries)
            and len(item["dominated_families"]) >= int(minimum_pose_families)
        ):
            kind = "dominated_removal"
            evidence_queries = item["dominated_queries"]
            evidence_families = item["dominated_families"]
        elif not truth_protected and not item["winner_queries"]:
            kind = "inactive_redundancy_probe"
            evidence_queries = item["topl_queries"]
            evidence_families = set()
        else:
            continue
        proposals.append(
            {
                "anchor_row": anchor,
                "kind": kind,
                "evidence_query_count": len(evidence_queries),
                "evidence_pose_family_count": len(evidence_families),
                "truth_query_count": len(item["truth_queries"]),
                "correct_winner_query_count": len(item["correct_winner_queries"]),
                "requires_exact_response_gate": True,
                "requires_necessity_views": kind != "inactive_redundancy_probe",
            }
        )
    kind_priority = {
        "harmful_removal": 0,
        "dominated_removal": 1,
        "inactive_redundancy_probe": 2,
    }
    proposals.sort(
        key=lambda item: (
            kind_priority[item["kind"]],
            -item["evidence_pose_family_count"],
            -item["evidence_query_count"],
            item["anchor_row"],
        )
    )
    inactive_seen = 0
    bounded = []
    for proposal in proposals:
        if proposal["kind"] == "inactive_redundancy_probe":
            if inactive_seen >= int(maximum_inactive_redundancy_candidates):
                continue
            inactive_seen += 1
        bounded.append(proposal)
    return {
        "schema": "lafgs_v18_current_competition_action_proposals",
        "version": 1,
        "uses_test_queries": False,
        "loo_used": False,
        "proposal_count": len(bounded),
        "proposals": bounded,
        "proposal_anchor_rows": torch.tensor(
            [proposal["anchor_row"] for proposal in bounded], dtype=torch.long
        ),
        "kind_counts": {
            kind: sum(proposal["kind"] == kind for proposal in bounded)
            for kind in kind_priority
        },
        "historical_removal_audit_used": False,
    }


def apply_reversible_active_action(
    active_mask: torch.Tensor,
    *,
    deactivate_rows: torch.Tensor = torch.empty(0, dtype=torch.long),
    reactivate_rows: torch.Tensor = torch.empty(0, dtype=torch.long),
) -> tuple[torch.Tensor, dict]:
    """Apply a reversible active-mask update with explicit swap provenance."""

    before = torch.as_tensor(active_mask).bool().cpu().reshape(-1)
    deactivate = torch.unique(torch.as_tensor(deactivate_rows).long().cpu(), sorted=True)
    reactivate = torch.unique(torch.as_tensor(reactivate_rows).long().cpu(), sorted=True)
    for rows in (deactivate, reactivate):
        if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= before.numel()):
            raise ValueError("active-set action row is outside the map")
    if bool(torch.isin(deactivate, reactivate).any()):
        raise ValueError("an Anchor cannot be deactivated and reactivated together")
    after = before.clone()
    after[deactivate] = False
    after[reactivate] = True
    if not bool(after.any()):
        raise ValueError("active-set action cannot empty the map")
    return after, {
        "schema": "lafgs_v18_reversible_active_action",
        "version": 1,
        "deactivate_rows": deactivate,
        "reactivate_rows": reactivate,
        "before_active_count": int(before.sum()),
        "after_active_count": int(after.sum()),
        "reversible": True,
    }


def propose_truth_reactivation_actions(
    *,
    records: Sequence[Mapping],
    anchor_count: int,
    active_anchor_mask: torch.Tensor,
    minimum_pose_families: int = 2,
    minimum_queries: int = 2,
) -> dict:
    """Propose reactivation when current winners suppress inactive truth."""

    count = int(anchor_count)
    active = torch.as_tensor(active_anchor_mask).bool().cpu().reshape(-1)
    if active.numel() != count:
        raise ValueError("reactivation registry does not align with the map")
    evidence: dict[int, dict[str, set[int]]] = defaultdict(
        lambda: {"queries": set(), "families": set()}
    )
    for record in records:
        query = int(record["query_index"])
        family = int(record["pose_family_id"])
        candidates = torch.as_tensor(record["candidate_anchor_rows"]).long()
        truth = record["truth"]
        status = torch.as_tensor(truth["truth_status"]).long()
        offsets = torch.as_tensor(truth["truth_offsets"]).long()
        truth_rows = torch.as_tensor(truth["truth_anchor_rows"]).long()
        winners = torch.as_tensor(
            record.get("current_winner_anchor_rows", candidates[:, 0])
        ).long()
        winner_available = torch.as_tensor(
            record.get(
                "current_winner_available",
                torch.ones(candidates.shape[0], dtype=torch.bool),
            )
        ).bool()
        decisive = (status == TRUTH_UNIQUE) | (status == TRUTH_EQUIVALENT)
        for row in torch.nonzero(decisive, as_tuple=False).reshape(-1).tolist():
            start, stop = int(offsets[row]), int(offsets[row + 1])
            local_truth = truth_rows[start:stop]
            if bool(winner_available[row]) and bool(
                torch.isin(winners[row : row + 1], local_truth).any()
            ):
                continue
            for anchor in local_truth.tolist():
                if not bool(active[anchor]) and bool((candidates[row] == anchor).any()):
                    evidence[int(anchor)]["queries"].add(query)
                    evidence[int(anchor)]["families"].add(family)
    proposals = []
    for anchor, item in evidence.items():
        if (
            len(item["queries"]) >= int(minimum_queries)
            and len(item["families"]) >= int(minimum_pose_families)
        ):
            proposals.append(
                {
                    "anchor_row": anchor,
                    "kind": "truth_reactivation",
                    "evidence_query_count": len(item["queries"]),
                    "evidence_pose_family_count": len(item["families"]),
                    "truth_query_count": len(item["queries"]),
                    "correct_winner_query_count": 0,
                    "requires_exact_response_gate": True,
                    "requires_necessity_views": False,
                }
            )
    proposals.sort(
        key=lambda item: (
            -item["evidence_pose_family_count"],
            -item["evidence_query_count"],
            item["anchor_row"],
        )
    )
    return {
        "schema": "lafgs_v18_truth_reactivation_proposals",
        "version": 1,
        "uses_test_queries": False,
        "loo_used": False,
        "proposal_count": len(proposals),
        "proposals": proposals,
    }


__all__ = [
    "apply_reversible_active_action",
    "propose_current_competition_actions",
    "propose_truth_reactivation_actions",
]
