"""One hierarchical state machine for precision, matching, and pose sufficiency."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from topology.dynamic_reserve import greedy_dynamic_pose_reserve
from topology.matching_coverage import (
    CandidateEdges,
    IncrementalBipartiteCoverage,
    greedy_matching_reserve,
)


def select_precision_compatibility(
    edges: CandidateEdges,
    quality_order: torch.Tensor,
    query_count: int,
    target_rows: int,
    *,
    minimum_count: int,
    maximum_count: int,
    check_interval: int,
) -> tuple[torch.Tensor, dict]:
    """Reproduce the V3 Track-Core stopping rule without type semantics."""
    state = IncrementalBipartiteCoverage(query_count, edges)
    selected = []
    stop_reason = "eligible_track_exhaustion"
    p10_history = []
    for rank, candidate in enumerate(
        torch.as_tensor(quality_order).long()[:maximum_count].tolist(), start=1
    ):
        state.add(candidate)
        selected.append(candidate)
        should_check = rank % int(check_interval) == 0 or rank == len(quality_order)
        if not should_check:
            continue
        counts = state.counts
        p10 = float(np.percentile(counts, 10))
        # Preserve the V3 report key in compatibility mode.  The unified trace
        # carries candidate-neutral terminology separately.
        p10_history.append({"track_count": rank, "matching_rank_p10": p10})
        if rank >= int(minimum_count) and p10 >= int(target_rows):
            stop_reason = "p10_matching_rank_target"
            break
    counts = state.counts
    report = {
        "selection": "quality_order_until_matching_feasible_p10_saturation",
        "target_rows": int(target_rows),
        "minimum_count": int(minimum_count),
        "maximum_count": int(maximum_count),
        "realized_count": len(selected),
        "matching_rank_p10": float(np.percentile(counts, 10)),
        "matching_rank_median": float(np.median(counts)),
        "stop_reason": stop_reason,
        "history": p10_history,
    }
    return torch.as_tensor(selected, dtype=torch.long), report


class HierarchicalSufficiencySelector:
    """Share one candidate registry, selected set, and causal selection trace.

    The stages are intentionally hierarchical: precision establishes a stable
    core, then matching and observability may add candidates only when they
    contribute missing sufficiency.  All stages mutate one selected state.
    """

    def __init__(
        self,
        edges: CandidateEdges,
        query_count: int,
        *,
        track_candidate_count: int | None = None,
    ) -> None:
        self.edges = edges
        self.query_count = int(query_count)
        self.track_candidate_count = (
            None if track_candidate_count is None else int(track_candidate_count)
        )
        if self.track_candidate_count is not None and not (
            0 <= self.track_candidate_count <= len(edges)
        ):
            raise ValueError("track candidate partition is outside the registry")
        self._selected: list[int] = []
        self._selected_set: set[int] = set()
        self._compatibility_selected = torch.empty(0, dtype=torch.long)
        self._trace: list[dict] = []
        self._reports: dict[str, dict] = {}
        self.matching: IncrementalBipartiteCoverage | None = None

    def _append(self, values: torch.Tensor, reason: str) -> None:
        for stage_ordinal, candidate in enumerate(
            torch.as_tensor(values).long().tolist()
        ):
            if candidate in self._selected_set:
                raise ValueError(
                    f"candidate {candidate} was selected by more than one reason"
                )
            self._selected.append(candidate)
            self._selected_set.add(candidate)
            self._trace.append(
                {
                    "candidate_universe_id": candidate,
                    "primary_reason": reason,
                    "global_ordinal": len(self._trace),
                    "reason_ordinal": stage_ordinal,
                }
            )

    def select_precision(
        self,
        quality_order: torch.Tensor,
        target_rows: int,
        *,
        minimum_count: int,
        maximum_count: int,
        check_interval: int,
    ) -> tuple[torch.Tensor, dict]:
        if self._selected:
            raise ValueError("precision selection must be the first selector stage")
        selected, report = select_precision_compatibility(
            self.edges,
            quality_order,
            self.query_count,
            target_rows,
            minimum_count=minimum_count,
            maximum_count=maximum_count,
            check_interval=check_interval,
        )
        self._append(selected, "precision")
        self._compatibility_selected = selected.clone()
        self._reports["precision"] = report
        return selected, report

    def complete_matching(
        self,
        candidates: Sequence[int] | torch.Tensor,
        utility: torch.Tensor,
        query_groups: torch.Tensor,
        *,
        requested_rows_per_query: int | Sequence[int],
        maximum_reserve: int,
        alias_risk: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, IncrementalBipartiteCoverage, dict]:
        if "precision" not in self._reports:
            raise ValueError("matching completion requires precision anchors")
        if "matching_completion" in self._reports:
            raise ValueError("matching completion has already run")
        selected, matching, report = greedy_matching_reserve(
            self.edges,
            self._selected,
            torch.as_tensor(candidates).long().tolist(),
            utility,
            query_groups,
            requested_rows_per_query=requested_rows_per_query,
            maximum_reserve=maximum_reserve,
            alias_risk=alias_risk,
        )
        self._append(selected, "matching_completion")
        self._compatibility_selected = torch.unique(
            torch.cat((self._compatibility_selected, selected)), sorted=False
        )
        self._reports["matching_completion"] = report
        self.matching = matching
        return selected, matching, report

    def complete_observability(
        self,
        evidence_by_candidate,
        initial_information: torch.Tensor,
        initial_used_rows,
        initial_image_cells,
        initial_depth_bins,
        initial_spatial_voxels,
        candidates: Sequence[int] | torch.Tensor,
        source_ids: torch.Tensor,
        voxel_ids: torch.Tensor,
        **policy,
    ) -> tuple[torch.Tensor, dict]:
        if "matching_completion" not in self._reports:
            raise ValueError("observability completion requires matching completion")
        if "observability_completion" in self._reports:
            raise ValueError("observability completion has already run")
        selected, report = greedy_dynamic_pose_reserve(
            evidence_by_candidate,
            initial_information,
            initial_used_rows,
            initial_image_cells,
            initial_depth_bins,
            initial_spatial_voxels,
            torch.as_tensor(candidates).long().tolist(),
            source_ids,
            voxel_ids,
            **policy,
        )
        self._append(selected, "observability_completion")
        self._compatibility_selected = torch.unique(
            torch.cat((self._compatibility_selected, selected)), sorted=False
        )
        self._reports["observability_completion"] = report
        return selected, report

    @property
    def selected_ids(self) -> torch.Tensor:
        return torch.as_tensor(self._selected, dtype=torch.long)

    @property
    def compatibility_materialization_ids(self) -> torch.Tensor:
        """Return the exact V3 unique-order contract used before materialization.

        The semantic trace intentionally preserves stage order.  V3 passed the
        concatenated IDs through ``torch.unique(sorted=False)``; retaining that
        separate row-order contract is required for bitwise map compatibility
        when downstream stable sorts encounter equal quality values.
        """
        return self._compatibility_selected.clone()

    def artifact(self) -> dict:
        reasons = [row["primary_reason"] for row in self._trace]
        return {
            "schema": "lafgs_unified_sufficiency_selection",
            "version": 1,
            "policy": "hierarchical_sufficiency_v4",
            "numerical_policy": "declared_by_materialization_report",
            "candidate_count": len(self.edges),
            "candidate_partitions": {
                "track_evidence_count": self.track_candidate_count,
                "surface_evidence_count": (
                    None
                    if self.track_candidate_count is None
                    else len(self.edges) - self.track_candidate_count
                ),
            },
            "selected_count": len(self._selected),
            "selected_universe_ids": self.selected_ids.clone(),
            "compatibility_materialization_ids": (
                self.compatibility_materialization_ids.clone()
            ),
            "primary_selection_reasons": reasons,
            "trace": list(self._trace),
            "reports": dict(self._reports),
            "single_candidate_registry": True,
            "single_selected_state": True,
            "completion_candidate_provider": "always_available_when_materialized",
        }


# Historical import compatibility only.  The formal V4 name describes the
# actual method; keeping the alias avoids breaking archived experiment code.
CompatibilitySufficiencySelector = HierarchicalSufficiencySelector
