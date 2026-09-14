"""Leakage-safe row policy and Top-K evidence for sparse feedback repair.

Every detected row remains part of the deployed plant replay.  Only rows with
decisive, render-valid identity evidence may train an Anchor descriptor action;
definite rendering nuisances are retained as diagnostics and uncertain rows
abstain.  The competition builder is listwise and equivalence-aware.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch

from map_learning.v18_provenance_truth import TRUTH_EQUIVALENT, TRUTH_UNIQUE


ROW_REPAIR = 0
ROW_NUISANCE = 1
ROW_UNKNOWN = 2
ROW_ROLE_NAMES = ("REPAIR", "NUISANCE", "UNKNOWN")


def partition_feedback_rows(
    *,
    row_valid: torch.Tensor,
    truth_status: torch.Tensor,
    definite_nuisance: torch.Tensor | None = None,
) -> dict:
    """Partition supervision roles while keeping every row in plant replay."""

    valid = torch.as_tensor(row_valid).bool().reshape(-1)
    status = torch.as_tensor(truth_status).long().reshape(-1)
    if status.shape != valid.shape:
        raise ValueError("feedback row validity and truth status do not align")
    nuisance = (
        torch.zeros_like(valid)
        if definite_nuisance is None
        else torch.as_tensor(definite_nuisance).bool().reshape(-1)
    )
    if nuisance.shape != valid.shape:
        raise ValueError("definite nuisance mask does not align with feedback rows")
    if bool((valid & nuisance).any()):
        raise ValueError("a render-valid row cannot be a definite nuisance")

    decisive = (status == TRUTH_UNIQUE) | (status == TRUTH_EQUIVALENT)
    repair = valid & decisive
    # A caller must explicitly certify nuisance rows.  Invalid or unsupported
    # rows are otherwise UNKNOWN; absence of teacher evidence is not a negative.
    nuisance = nuisance & ~repair
    unknown = ~(repair | nuisance)
    role = torch.full(valid.shape, ROW_UNKNOWN, dtype=torch.uint8)
    role[repair] = ROW_REPAIR
    role[nuisance] = ROW_NUISANCE
    if not bool((repair | nuisance | unknown).all()):
        raise RuntimeError("feedback row partition is incomplete")
    return {
        "schema": "lafgs_v20_feedback_row_policy",
        "version": 1,
        "plant_eligible": torch.ones_like(valid),
        "repair_eligible": repair,
        "definite_nuisance": nuisance,
        "unknown": unknown,
        "row_role": role,
        "counts": {
            "plant": int(valid.numel()),
            "repair": int(repair.sum()),
            "nuisance": int(nuisance.sum()),
            "unknown": int(unknown.sum()),
        },
    }


def _truth_rows(truth: Mapping, row: int) -> torch.Tensor:
    offsets = torch.as_tensor(truth["truth_offsets"]).long()
    anchors = torch.as_tensor(truth["truth_anchor_rows"]).long()
    return anchors[int(offsets[row]) : int(offsets[row + 1])]


def _ranked_unique(values: torch.Tensor) -> torch.Tensor:
    output: list[int] = []
    seen: set[int] = set()
    for value in torch.as_tensor(values).long().tolist():
        if int(value) not in seen:
            seen.add(int(value))
            output.append(int(value))
    return torch.tensor(output, dtype=torch.long)


def _csr(parts: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    counts = torch.tensor([part.numel() for part in parts], dtype=torch.long)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    values = torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)
    return offsets, values


def build_topk_competition_evidence(
    *,
    records: Sequence[Mapping],
    anchor_count: int,
    equivalence_class_ids: torch.Tensor,
    active_anchor_mask: torch.Tensor | None = None,
    minimum_wrong_winner_pose_families: int = 2,
    minimum_negative_action_clean_pose_families: int = 2,
    maximum_repair_rows_per_query: int = 256,
    maximum_protection_rows_per_query: int = 256,
) -> dict:
    """Build multi-positive repair and clean-margin protection evidence.

    A repair row must be valid and decisive, have an active truth identity in
    Top-K, currently lose Top-1, improve the query-level exact-pose
    counterfactual, and share its wrong winner with another pose family.
    """

    if anchor_count <= 0:
        raise ValueError("V20 evidence requires a non-empty Anchor bank")
    if int(minimum_wrong_winner_pose_families) < 2:
        raise ValueError("V20 repair requires at least two pose families")
    if int(minimum_negative_action_clean_pose_families) < 2:
        raise ValueError("V20 negative action requires cross-family clean support")
    if min(maximum_repair_rows_per_query, maximum_protection_rows_per_query) < 1:
        raise ValueError("V20 per-query evidence limits must be positive")
    equivalence = torch.as_tensor(equivalence_class_ids).long().reshape(-1)
    if equivalence.numel() != int(anchor_count):
        raise ValueError("V20 equivalence classes do not align with Anchors")
    active = (
        torch.ones(anchor_count, dtype=torch.bool)
        if active_anchor_mask is None
        else torch.as_tensor(active_anchor_mask).bool().reshape(-1)
    )
    if active.numel() != int(anchor_count):
        raise ValueError("V20 active mask does not align with Anchors")

    prepared: list[dict] = []
    wrong_families: dict[int, set[int]] = defaultdict(set)
    # A recurrent wrong winner is not automatically a safe negative action.
    # Moving it is authorized only when that exact Anchor is also a clean
    # winner for at least one decisive row, so its native role is observable
    # and protected by a query-specific margin constraint.
    clean_winner_families: dict[int, set[int]] = defaultdict(set)
    totals = defaultdict(int)
    nuisance_max_scores: list[torch.Tensor] = []
    for record in records:
        descriptors = torch.as_tensor(record["query_descriptors"]).float()
        candidates = torch.as_tensor(record["candidate_anchor_rows"]).long()
        scores = torch.as_tensor(record["candidate_scores"]).float()
        policy = record["row_policy"]
        repair_mask = torch.as_tensor(policy["repair_eligible"]).bool().reshape(-1)
        nuisance_mask = torch.as_tensor(policy["definite_nuisance"]).bool().reshape(-1)
        unknown_mask = torch.as_tensor(policy["unknown"]).bool().reshape(-1)
        plant_mask = torch.as_tensor(policy["plant_eligible"]).bool().reshape(-1)
        row_count = descriptors.shape[0]
        source_rows = torch.as_tensor(
            record.get("source_query_rows", torch.arange(row_count))
        ).long().reshape(-1)
        if (
            descriptors.ndim != 2
            or not bool(torch.isfinite(descriptors).all())
            or bool((torch.linalg.norm(descriptors, dim=1) <= 1e-8).any())
            or candidates.ndim != 2
            or scores.shape != candidates.shape
            or not bool(torch.isfinite(scores).all())
            or candidates.shape[0] != row_count
            or repair_mask.numel() != row_count
            or nuisance_mask.numel() != row_count
            or unknown_mask.numel() != row_count
            or plant_mask.numel() != row_count
            or source_rows.numel() != row_count
            or bool((source_rows < 0).any())
            or torch.unique(source_rows).numel() != source_rows.numel()
        ):
            raise ValueError(
                "V20 competition record rows do not align for query "
                f"{record.get('query_index')}: descriptors={tuple(descriptors.shape)}, "
                f"candidates={tuple(candidates.shape)}, scores={tuple(scores.shape)}, "
                f"policy={(repair_mask.numel(), nuisance_mask.numel(), unknown_mask.numel())}"
            )
        if candidates.numel() and (
            int(candidates.min()) < 0 or int(candidates.max()) >= anchor_count
        ):
            raise ValueError("V20 Top-K candidate is outside the Anchor bank")
        if (
            bool((repair_mask & nuisance_mask).any())
            or bool((repair_mask & unknown_mask).any())
            or bool((nuisance_mask & unknown_mask).any())
            or not bool((repair_mask | nuisance_mask | unknown_mask).all())
            or not bool(plant_mask.all())
        ):
            raise ValueError("V20 row roles are not disjoint and exhaustive")
        truth = record["truth"]
        if int(truth["row_count"]) != row_count:
            raise ValueError("V20 truth rows do not align with descriptors")
        truth_offsets = torch.as_tensor(truth["truth_offsets"]).long().reshape(-1)
        truth_anchors = torch.as_tensor(truth["truth_anchor_rows"]).long().reshape(-1)
        truth_status = torch.as_tensor(truth["truth_status"]).long().reshape(-1)
        if (
            truth_offsets.shape != (row_count + 1,)
            or truth_status.shape != (row_count,)
            or int(truth_offsets[0]) != 0
            or int(truth_offsets[-1]) != truth_anchors.numel()
            or bool((truth_offsets[1:] < truth_offsets[:-1]).any())
            or (
                truth_anchors.numel() > 0
                and (
                    int(truth_anchors.min()) < 0
                    or int(truth_anchors.max()) >= anchor_count
                )
            )
        ):
            raise ValueError("V20 truth CSR is invalid")
        family = int(record["pose_family_id"])
        can_train = bool(record.get("can_train_descriptor", False)) and float(
            record.get("actual_query_task_gain", 0.0)
        ) > 0.0
        local_rows = []
        for row in range(row_count):
            candidate = candidates[row]
            available = active[candidate]
            if not bool(available.any()):
                raise ValueError("V20 Top-K row has no active candidate")
            winner_column = int(scores[row].masked_fill(~available, -torch.inf).argmax())
            winner = int(candidate[winner_column])
            truth_rows = _truth_rows(truth, row)
            active_truth = _ranked_unique(truth_rows[active[truth_rows]])
            truth_classes = set(equivalence[active_truth].tolist())
            candidate_positive = torch.tensor(
                [int(equivalence[value]) in truth_classes for value in candidate.tolist()],
                dtype=torch.bool,
            ) & available
            positive_rows = _ranked_unique(
                torch.cat((active_truth, candidate[candidate_positive]))
            )
            truth_in_topk = bool(candidate_positive.any())
            winner_positive = bool(candidate_positive[winner_column])
            has_nonpositive = bool((available & ~candidate_positive).any())
            if repair_mask[row]:
                totals["decisive"] += 1
                if active_truth.numel() == 0:
                    totals["selection_deficit"] += 1
                elif not truth_in_topk:
                    totals["retrieval_miss"] += 1
                elif winner_positive:
                    totals["correct_winner"] += 1
                    if has_nonpositive:
                        clean_winner_families[winner].add(family)
                else:
                    totals["competition_miss"] += 1
                    if can_train:
                        wrong_families[winner].add(family)
            elif nuisance_mask[row]:
                nuisance_max_scores.append(scores[row, available].max().reshape(1))
                totals["nuisance"] += 1
            else:
                totals["unknown"] += 1
            local_rows.append(
                {
                    "row": row,
                    "winner": winner,
                    "winner_score": float(scores[row, winner_column]),
                    "winner_positive": winner_positive,
                    "truth_in_topk": truth_in_topk,
                    "active_truth": active_truth,
                    "positive_rows": positive_rows,
                    "candidate_positive": candidate_positive,
                    "can_train": can_train,
                }
            )
        prepared.append(
            {
                "record": record,
                "descriptors": descriptors,
                "candidates": candidates,
                "scores": scores,
                "repair_mask": repair_mask,
                "family": family,
                "source_rows": source_rows,
                "rows": local_rows,
            }
        )

    repairs: list[dict] = []
    protections: list[dict] = []
    per_query: list[dict] = []
    for item in prepared:
        record = item["record"]
        repair_local = []
        protect_local = []
        for state in item["rows"]:
            row = state["row"]
            if not bool(item["repair_mask"][row]):
                continue
            candidate = item["candidates"][row]
            available = active[candidate]
            negative = _ranked_unique(
                candidate[available & ~state["candidate_positive"]]
            )
            if state["winner_positive"] and negative.numel():
                protect_local.append((state["winner_score"] - float(
                    item["scores"][row][available & ~state["candidate_positive"]].max()
                ), row, negative))
            elif (
                state["truth_in_topk"]
                and state["can_train"]
                and len(wrong_families[state["winner"]])
                >= int(minimum_wrong_winner_pose_families)
                and state["positive_rows"].numel()
                and negative.numel()
            ):
                best_positive_score = float(
                    item["scores"][row][state["candidate_positive"]].max()
                )
                deficit = state["winner_score"] - best_positive_score
                repair_local.append((deficit, row, negative))
        repair_local.sort(key=lambda value: (-value[0], value[1]))
        protect_local.sort(key=lambda value: (value[0], value[1]))
        repair_local = repair_local[: int(maximum_repair_rows_per_query)]
        protect_local = protect_local[: int(maximum_protection_rows_per_query)]
        gain = min(max(float(record.get("actual_query_task_gain", 0.0)), 0.01), 4.0)
        for _, row, negative in repair_local:
            repairs.append(
                {
                    "query": int(record["query_index"]),
                    "family": item["family"],
                    "descriptor": item["descriptors"][row],
                    "source_row": int(item["source_rows"][row]),
                    "positive": item["rows"][row]["positive_rows"],
                    "negative": negative,
                    "winner": item["rows"][row]["winner"],
                    "winner_clean_support_family_count": len(
                        clean_winner_families[item["rows"][row]["winner"]]
                    ),
                    "gain": gain,
                }
            )
        for margin, row, negative in protect_local:
            protections.append(
                {
                    "query": int(record["query_index"]),
                    "family": item["family"],
                    "descriptor": item["descriptors"][row],
                    # Preserve the exact Anchor that currently wins this clean
                    # row.  A max over all equivalent positives could conceal
                    # damage to the wrong-winner Anchor being moved elsewhere.
                    "positive": torch.tensor(
                        [item["rows"][row]["winner"]], dtype=torch.long
                    ),
                    "negative": negative,
                    "margin": margin,
                }
            )
        per_query.append(
            {
                "query_index": int(record["query_index"]),
                "pose_family_id": item["family"],
                "repair_row_count": len(repair_local),
                "protection_row_count": len(protect_local),
            }
        )
    if not repairs:
        raise RuntimeError("no V20 Top-K repair row passed causal cross-family gates")

    family_queries: dict[int, set[int]] = defaultdict(set)
    for sample in repairs:
        family_queries[sample["family"]].add(sample["query"])
    family_count = len(family_queries)
    per_query_count = defaultdict(int)
    for sample in repairs:
        per_query_count[sample["query"]] += 1
    repair_weights = torch.tensor(
        [
            sample["gain"]
            / per_query_count[sample["query"]]
            / len(family_queries[sample["family"]])
            / family_count
            for sample in repairs
        ],
        dtype=torch.float32,
    )
    repair_positive_offsets, repair_positive_rows = _csr(
        [sample["positive"] for sample in repairs]
    )
    repair_negative_offsets, repair_negative_rows = _csr(
        [sample["negative"] for sample in repairs]
    )
    protection_positive_offsets, protection_positive_rows = _csr(
        [sample["positive"] for sample in protections]
    )
    protection_negative_offsets, protection_negative_rows = _csr(
        [sample["negative"] for sample in protections]
    )
    descriptor_dim = int(repairs[0]["descriptor"].numel())
    return {
        "schema": "lafgs_v20_topk_competition_evidence",
        "version": 2,
        "uses_test_queries": False,
        "loo_used": False,
        "feedback_descriptors_copied_into_map": False,
        "repair_query_descriptors": torch.stack(
            [sample["descriptor"] for sample in repairs]
        ).reshape(-1, descriptor_dim),
        "repair_positive_offsets": repair_positive_offsets,
        "repair_positive_anchor_rows": repair_positive_rows,
        "repair_negative_offsets": repair_negative_offsets,
        "repair_negative_anchor_rows": repair_negative_rows,
        "repair_wrong_winner_anchor_rows": torch.tensor(
            [sample["winner"] for sample in repairs], dtype=torch.long
        ),
        "repair_wrong_winner_clean_support_family_counts": torch.tensor(
            [
                sample["winner_clean_support_family_count"]
                for sample in repairs
            ],
            dtype=torch.long,
        ),
        "negative_action_anchor_rows": torch.tensor(
            sorted(
                {
                    sample["winner"]
                    for sample in repairs
                    if sample["winner_clean_support_family_count"]
                    >= int(minimum_negative_action_clean_pose_families)
                }
            ),
            dtype=torch.long,
        ),
        "repair_pose_family_ids": torch.tensor(
            [sample["family"] for sample in repairs], dtype=torch.long
        ),
        "repair_query_indices": torch.tensor(
            [sample["query"] for sample in repairs], dtype=torch.long
        ),
        "repair_source_query_rows": torch.tensor(
            [sample["source_row"] for sample in repairs], dtype=torch.long
        ),
        "repair_sample_weights": repair_weights,
        "protection_query_descriptors": (
            torch.stack([sample["descriptor"] for sample in protections])
            if protections
            else torch.empty((0, descriptor_dim))
        ),
        "protection_positive_offsets": protection_positive_offsets,
        "protection_positive_anchor_rows": protection_positive_rows,
        "protection_negative_offsets": protection_negative_offsets,
        "protection_negative_anchor_rows": protection_negative_rows,
        "protection_initial_margin": torch.tensor(
            [sample["margin"] for sample in protections], dtype=torch.float32
        ),
        "minimum_wrong_winner_pose_families": int(
            minimum_wrong_winner_pose_families
        ),
        "minimum_negative_action_clean_pose_families": int(
            minimum_negative_action_clean_pose_families
        ),
        "repair_pose_family_count": int(torch.unique(
            torch.tensor([sample["family"] for sample in repairs])
        ).numel()),
        "counts": dict(totals),
        "nuisance_max_scores": (
            torch.cat(nuisance_max_scores)
            if nuisance_max_scores
            else torch.empty(0)
        ),
        "per_query": per_query,
    }


__all__ = [
    "ROW_NUISANCE",
    "ROW_REPAIR",
    "ROW_ROLE_NAMES",
    "ROW_UNKNOWN",
    "build_topk_competition_evidence",
    "partition_feedback_rows",
]
