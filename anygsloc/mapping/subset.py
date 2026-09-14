"""Frozen mainline kernel extracted from anygsloc.mapping.subset."""

from __future__ import annotations


import torch


def subset(state, rows):
    n = len(state["anchor_ids"])
    result = dict(state)
    for key, value in state.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim
            and len(value) == n
            and key != "v6_mapping_query_bins"
        ):
            result[key] = value[rows].clone()
    result["anchor_candidate_kind"] = [
        state["anchor_candidate_kind"][i] for i in rows.tolist()
    ]
    result["anchor_ids"] = torch.arange(len(rows))
    csr = state["projective_anchor_observations"]
    offsets = csr["observation_offsets"]
    selected = torch.cat(
        [torch.arange(int(offsets[i]), int(offsets[i + 1])) for i in rows.tolist()]
    )
    counts = (offsets[1:] - offsets[:-1])[rows]
    result["projective_anchor_observations"] = dict(
        csr,
        observation_offsets=torch.cat(
            [torch.zeros(1, dtype=torch.long), counts.cumsum(0)]
        ),
        query_indices=csr["query_indices"][selected],
        keypoint_indices=csr["keypoint_indices"][selected],
    )
    result["anchor_view_support"] = {
        k: v[rows].clone()
        if isinstance(v, torch.Tensor) and v.ndim and len(v) == n
        else v
        for k, v in state["anchor_view_support"].items()
    }
    result["canonical_anchor_count"] = len(rows)
    result["micro_anchor_count"] = int((result["anchor_type"] == 1).sum())
    result["base_anchor_count"] = len(rows) - result["micro_anchor_count"]
    assert int(
        result["projective_anchor_observations"]["observation_offsets"][-1]
    ) == len(selected)
    assert len(result["anchor_view_support"]["mode_count"]) == len(rows)
    return result
