"""V2 row filtering before any projective association or Track construction."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from evidence.observation_provider import GaussianRenderObservationProvider


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
    filtered_records = {}
    source_rows = []
    raw_count = 0
    retained_count = 0
    for query_index, name in enumerate(names):
        record = records[name]
        keypoints = torch.as_tensor(record["native_keypoints"])
        mask = torch.as_tensor(rows_by_query[query_index]).bool().reshape(-1)
        if mask.numel() != keypoints.shape[0]:
            raise ValueError(f"V2 rows do not align for {name}")
        selected = torch.nonzero(mask, as_tuple=False).reshape(-1)
        if selected.numel() == 0:
            raise ValueError(f"V2 removed every detector row for {name}")
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
        retained_count += int(selected.numel())
    payload = {
        **dict(cache),
        "queries": filtered_records,
        "v2_preassociation_filter": {
            "schema": "lafgs_v2_preassociation_filter",
            "version": 1,
            "raw_row_count": raw_count,
            "retained_row_count": retained_count,
            "removed_row_count": raw_count - retained_count,
            "retained_fraction": retained_count / raw_count,
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
