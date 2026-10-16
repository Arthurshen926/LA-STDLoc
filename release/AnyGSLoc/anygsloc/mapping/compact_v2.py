"""Opt-in v2 compact-map selector.

v1 methods remain untouched. v2 uses the identical v1 quality / coverage /
conditional pose-information selection, then performs one support-preserving
redundancy cleanup as the last substep of the pose-aware stage. Archived v1
traces can be replayed without rerunning mapping renders for controlled audits.
"""

from __future__ import annotations

from typing import Any

import torch

from anygsloc.mapping.compact import select_compact_rows as select_compact_rows_v1
from anygsloc.mapping.post_pose_cleanup import prune_redundant_coverage


def replay_v2_cleanup(
    full_state: dict[str, Any],
    archived_selector_trace: dict[str, Any],
    *,
    reference_support_target: int = 256,
) -> tuple[torch.Tensor, dict[str, Any]]:
    originally_selected = (
        torch.as_tensor(
            archived_selector_trace["selected_universe_ids"], dtype=torch.long
        )
        .sort()
        .values
    )
    final_rows, diagnostic = prune_redundant_coverage(
        full_state,
        originally_selected,
        archived_selector_trace,
        support_target=reference_support_target,
    )
    return final_rows, diagnostic


def select_compact_rows_v2(
    full_state: dict[str, Any],
    query_cache: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    # Compute exactly the original conditional pose-optimized selection first.
    original_rows, v1_trace = select_compact_rows_v1(full_state, query_cache)
    final_rows, cleanup = prune_redundant_coverage(
        full_state, original_rows, v1_trace, support_target=256
    )
    report = dict(v1_trace)
    report.update(
        schema_version="compact-fast-v2",
        v1_selected_count=int(original_rows.numel()),
        final_selected_count=int(final_rows.numel()),
        final_source_ids=final_rows.clone(),
        v2_pose_completion_finalization=cleanup,
    )
    return final_rows, report
