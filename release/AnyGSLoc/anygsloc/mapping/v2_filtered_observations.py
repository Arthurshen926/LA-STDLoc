"""Frozen mainline kernel extracted from evidence.v2_filtered_observations."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from anygsloc.mapping.observation_provider import GaussianRenderObservationProvider

SPARSE_ROW_FIELDS = (
    "native_keypoints",
    "native_descriptors",
    "native_scores",
    "native_surface_support",
    "native_valid_keypoint_mask",
    "native_keypoint_alpha",
    "native_keypoint_depth",
)


def build_v2_filtered_provider(
    cache: dict,
    *,
    rows_by_query: Sequence[torch.Tensor],
    allow_empty_queries: bool = False,
    maximum_rows_per_query: int | None = None,
) -> tuple[GaussianRenderObservationProvider, list[torch.Tensor], dict]:
    """Filter sparse rows while retaining dense render evidence by reference.

    SuperPoint has already seen the complete unmasked RGB.  The returned
    provider is consumed by pair association, Track construction,
    triangulation, completion and descriptor fusion, so invalid rows cannot
    re-enter through a later stage.
    """

    records = cache.get("queries")
    if not isinstance(records, dict) or not records:
        raise ValueError("V2 filtering requires a render observation cache")
    names = list(records)
    if len(rows_by_query) != len(names):
        raise ValueError("V2 row registry must contain one mask per query")
    if maximum_rows_per_query is not None and int(maximum_rows_per_query) <= 0:
        raise ValueError("maximum_rows_per_query must be positive when provided")
    row_budget = None if maximum_rows_per_query is None else int(maximum_rows_per_query)
    filtered_records = {}
    source_rows = []
    raw_count = 0
    v2_valid_count = 0
    retained_count = 0
    budget_removed_count = 0
    overflow_recovered_count = 0
    empty_query_names = []
    for query_index, name in enumerate(names):
        record = records[name]
        keypoints = torch.as_tensor(record["native_keypoints"])
        mask = torch.as_tensor(rows_by_query[query_index]).bool().reshape(-1)
        if mask.numel() != keypoints.shape[0]:
            raise ValueError(f"V2 rows do not align for {name}")
        selected = torch.nonzero(mask, as_tuple=False).reshape(-1)
        valid_before_budget = int(selected.numel())
        raw_top_budget = None
        if row_budget is not None:
            scores = torch.as_tensor(record["native_scores"]).float().reshape(-1)
            if scores.numel() != mask.numel():
                raise ValueError(f"detector scores do not align for {name}")
            raw_top_budget = torch.zeros(mask.numel(), dtype=torch.bool)
            raw_top_budget[
                torch.argsort(scores, descending=True, stable=True)[:row_budget]
            ] = True
            if selected.numel() > row_budget:
                order = torch.argsort(scores[selected], descending=True, stable=True)[
                    :row_budget
                ]
                selected = selected[order]
        # Rows beyond the retained association budget would have been absent
        # from a conventional top-K detector cache.  Count how many valid rows
        # the over-detect -> V2 -> refill policy actually recovers.
        overflow_recovered_count += int(
            (~raw_top_budget[selected]).sum() if raw_top_budget is not None else 0
        )
        if selected.numel() == 0:
            if not bool(allow_empty_queries):
                raise ValueError(f"V2 removed every detector row for {name}")
            empty_query_names.append(name)
        output = dict(record)
        for field in SPARSE_ROW_FIELDS:
            value = record.get(field)
            if value is None:
                continue
            tensor = torch.as_tensor(value)
            if tensor.ndim >= 1 and tensor.shape[0] == mask.numel():
                output[field] = tensor[selected].clone()
        # The filtered rows already satisfy the V2 certificate.  This field is
        # the legacy render-domain assertion consumed by projective association.
        output["native_valid_keypoint_mask"] = torch.ones(
            selected.numel(), dtype=torch.bool
        )
        output["v2_source_keypoint_indices"] = selected
        filtered_records[name] = output
        source_rows.append(selected)
        raw_count += int(mask.numel())
        v2_valid_count += valid_before_budget
        retained_count += int(selected.numel())
        budget_removed_count += valid_before_budget - int(selected.numel())
    if retained_count == 0:
        raise ValueError("V2 removed every detector row from every mapping query")
    payload = {
        **dict(cache),
        "queries": filtered_records,
        "v2_preassociation_filter": {
            "schema": "lafgs_v2_preassociation_filter",
            "version": 1,
            "raw_row_count": raw_count,
            "v2_valid_row_count_before_budget": v2_valid_count,
            "retained_row_count": retained_count,
            "v2_removed_row_count": raw_count - v2_valid_count,
            "association_budget_removed_row_count": budget_removed_count,
            "removed_row_count": raw_count - retained_count,
            "retained_fraction": retained_count / raw_count,
            "maximum_rows_per_query": row_budget,
            "row_budget_selection": (
                "all_v2_valid_rows"
                if row_budget is None
                else "stable_detector_score_topk_after_v2"
            ),
            "overflow_valid_rows_recovered": overflow_recovered_count,
            "input_query_count": len(names),
            "retained_query_count": len(names) - len(empty_query_names),
            "empty_query_count": len(empty_query_names),
            "empty_query_names": empty_query_names,
            "empty_query_policy": (
                "retain_zero_row_mapping_view"
                if bool(allow_empty_queries)
                else "fail_closed"
            ),
            "superpoint_input": "complete_unmasked_rgb",
            "filter_stage": "after_detection_before_pair_association",
        },
    }
    provider = GaussianRenderObservationProvider(payload)
    return provider, source_rows, dict(payload["v2_preassociation_filter"])


def remap_candidate_rows_to_source(
    candidate: dict, source_rows_by_query: Sequence[torch.Tensor]
) -> dict:
    """Restore filtered local keypoint rows to immutable source-cache rows."""

    output = dict(candidate)
    csr = dict(candidate["projective_anchor_observations"])
    query = torch.as_tensor(csr["query_indices"]).long()
    local = torch.as_tensor(csr["keypoint_indices"]).long()
    if query.shape != local.shape:
        raise ValueError("candidate observation columns are not aligned")
    remapped = torch.empty_like(local)
    for query_index in torch.unique(query).tolist():
        rows = torch.nonzero(query == int(query_index), as_tuple=False).reshape(-1)
        source = torch.as_tensor(source_rows_by_query[int(query_index)]).long()
        if rows.numel() and int(local[rows].max()) >= source.numel():
            raise ValueError("filtered candidate row exceeds source registry")
        remapped[rows] = source[local[rows]]
    csr["keypoint_indices"] = remapped
    csr["keypoint_index_semantics"] = "original_unfiltered_observation_cache_row"
    output["projective_anchor_observations"] = csr
    output["contract"] = {
        **dict(candidate.get("contract", {})),
        "v2_filter_stage": "after_detection_before_pair_association",
        "observation_rows_remapped_to_source_cache": True,
    }
    return output
