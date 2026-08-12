"""Audit-only materialization of observation-grounded Anchor descriptors.

This module intentionally does not replace ``anchor_features``.  It builds a
parallel descriptor bank from the real mapping-observation CSR in the unified
Anchor Registry, making Stage-5 representability measurable before any
deployment-changing experiment is attempted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath

import torch
import torch.nn.functional as F


SCHEMA = "lafgs_observation_descriptor_audit"
VERSION = 1


def _trajectory_ids(query_names: Sequence[str]) -> torch.Tensor:
    """Return stable sequence IDs without pretending flat names have groups."""
    labels: list[str | None] = []
    for name in query_names:
        parts = PurePosixPath(str(name).replace("\\", "/")).parts
        labels.append(parts[0] if len(parts) > 1 else None)
    vocabulary = {
        label: index
        for index, label in enumerate(
            sorted({label for label in labels if label is not None})
        )
    }
    return torch.tensor(
        [-1 if label is None else vocabulary[label] for label in labels],
        dtype=torch.long,
    )


def _quantiles(values: torch.Tensor) -> dict[str, float | None]:
    values = torch.as_tensor(values).float().reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {key: None for key in ("mean", "p10", "median", "p90", "maximum")}
    return {
        "mean": float(values.mean()),
        "p10": float(torch.quantile(values, 0.1)),
        "median": float(torch.quantile(values, 0.5)),
        "p90": float(torch.quantile(values, 0.9)),
        "maximum": float(values.max()),
    }


def _weighted_unit_mean(
    descriptors: torch.Tensor, confidence: torch.Tensor
) -> torch.Tensor:
    weight = torch.as_tensor(confidence).float().reshape(-1).clamp_min(1e-6)
    return F.normalize((descriptors * weight[:, None]).sum(dim=0), dim=0)


def robust_observation_fusion(
    descriptors: torch.Tensor,
    query_indices: torch.Tensor,
    view_group_ids: torch.Tensor,
    trajectory_ids: torch.Tensor,
    confidence: torch.Tensor,
    *,
    trim_fraction: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Fuse observations with per-image balance and trimmed stratum medoids.

    Observations are first collapsed within an image.  Image prototypes are
    then balanced over joint ``(trajectory, view-bin)`` strata when both labels
    exist (and the available label otherwise).  The final descriptor is the
    mean of stratum prototypes closest to their cosine medoid after trimming.
    """
    descriptors = torch.as_tensor(descriptors).float()
    if descriptors.ndim != 2 or descriptors.shape[0] == 0:
        raise ValueError("descriptor observations must be a non-empty matrix")
    descriptors = F.normalize(descriptors, dim=1)
    query_indices = torch.as_tensor(query_indices).long().reshape(-1)
    view_group_ids = torch.as_tensor(view_group_ids).long().reshape(-1)
    trajectory_ids = torch.as_tensor(trajectory_ids).long().reshape(-1)
    confidence = torch.as_tensor(confidence).float().reshape(-1)
    count = descriptors.shape[0]
    if any(
        value.numel() != count
        for value in (
            query_indices,
            view_group_ids,
            trajectory_ids,
            confidence,
        )
    ):
        raise ValueError("observation fusion metadata does not align")
    if not 0.0 <= float(trim_fraction) < 1.0:
        raise ValueError("trim_fraction must be in [0, 1)")

    image_descriptors = []
    image_confidence = []
    image_view_groups = []
    image_trajectories = []
    for query in torch.unique(query_indices, sorted=True).tolist():
        selected = query_indices == int(query)
        image_descriptors.append(
            _weighted_unit_mean(descriptors[selected], confidence[selected])
        )
        image_confidence.append(confidence[selected].clamp_min(1e-6).mean())
        image_view_groups.append(view_group_ids[selected][0])
        image_trajectories.append(trajectory_ids[selected][0])
    image_descriptors_tensor = torch.stack(image_descriptors)
    image_confidence_tensor = torch.stack(image_confidence)
    image_view_groups_tensor = torch.stack(image_view_groups)
    image_trajectories_tensor = torch.stack(image_trajectories)

    # Negative labels mean unavailable.  A per-image fallback keeps each real
    # mapping view equally weighted without inventing a semantic group.
    labels: list[tuple[int, int] | tuple[str, int]] = []
    for row in range(image_descriptors_tensor.shape[0]):
        view = int(image_view_groups_tensor[row])
        trajectory = int(image_trajectories_tensor[row])
        if view >= 0 and trajectory >= 0:
            labels.append((trajectory, view))
        elif view >= 0:
            labels.append((-1, view))
        elif trajectory >= 0:
            labels.append((trajectory, -1))
        else:
            labels.append(("query", int(row)))
    unique_labels = sorted(set(labels), key=str)
    stratum_descriptors = []
    for label in unique_labels:
        selected = torch.tensor([value == label for value in labels])
        stratum_descriptors.append(
            _weighted_unit_mean(
                image_descriptors_tensor[selected],
                image_confidence_tensor[selected],
            )
        )
    strata = torch.stack(stratum_descriptors)
    similarity = strata @ strata.T
    medoid_index = int(similarity.mean(dim=1).argmax())
    keep_count = max(
        1,
        int(round(strata.shape[0] * (1.0 - float(trim_fraction)))),
    )
    keep = torch.topk(similarity[medoid_index], k=keep_count).indices
    fused = F.normalize(strata[keep].mean(dim=0), dim=0)
    observation_distance = 1.0 - descriptors @ fused
    balanced_distance = 1.0 - strata @ fused
    return fused, {
        "image_count": int(image_descriptors_tensor.shape[0]),
        "stratum_count": int(strata.shape[0]),
        "retained_stratum_count": int(keep_count),
        "observation_cosine_distance_mean": observation_distance.mean(),
        "observation_cosine_distance_p90": torch.quantile(
            observation_distance, 0.9
        ),
        "observation_cosine_distance_maximum": observation_distance.max(),
        "balanced_cosine_dispersion": balanced_distance.mean(),
    }


def _validate_registry(registry: Mapping) -> tuple[int, torch.Tensor]:
    if registry.get("schema") != "lafgs_evidence_grounded_anchor_registry":
        raise ValueError("unsupported Anchor Registry schema")
    required = (
        "anchor_ids",
        "anchor_features",
        "anchor_type",
        "observation_offsets",
        "observation_query_indices",
        "observation_keypoint_indices",
        "query_names",
    )
    missing = [key for key in required if key not in registry]
    if missing:
        raise ValueError(f"Anchor Registry missing fields: {missing}")
    count = int(torch.as_tensor(registry["anchor_ids"]).numel())
    offsets = torch.as_tensor(registry["observation_offsets"]).long().reshape(-1)
    if offsets.numel() != count + 1 or int(offsets[0]) != 0:
        raise ValueError("Anchor observation offsets are invalid")
    edge_count = int(offsets[-1])
    if bool((offsets[1:] < offsets[:-1]).any()):
        raise ValueError("Anchor observation offsets are not monotonic")
    for key in ("observation_query_indices", "observation_keypoint_indices"):
        if torch.as_tensor(registry[key]).numel() != edge_count:
            raise ValueError(f"{key} does not align with observation offsets")
    return count, offsets


def _category_report(
    selected: torch.Tensor,
    *,
    valid: torch.Tensor,
    observation_count: torch.Tensor,
    query_count: torch.Tensor,
    view_group_count: torch.Tensor,
    trajectory_count: torch.Tensor,
    representability: torch.Tensor,
    dispersion: torch.Tensor,
    deployment_cosine: torch.Tensor,
) -> dict:
    selected = torch.as_tensor(selected).bool()
    supported = selected & valid
    return {
        "anchor_count": int(selected.sum()),
        "valid_descriptor_count": int(supported.sum()),
        "zero_observation_count": int((selected & (observation_count == 0)).sum()),
        "single_observation_count": int((selected & (observation_count == 1)).sum()),
        "observation_count": _quantiles(observation_count[selected].float()),
        "distinct_query_count": _quantiles(query_count[selected].float()),
        "view_group_count": _quantiles(view_group_count[selected].float()),
        "trajectory_count": _quantiles(trajectory_count[selected].float()),
        "representability": _quantiles(representability[supported]),
        "balanced_cosine_dispersion": _quantiles(dispersion[supported]),
        "fused_vs_deployment_cosine": _quantiles(deployment_cosine[supported]),
    }


def materialize_observation_descriptor_audit(
    registry: Mapping,
    query_cache: Mapping,
    *,
    trim_fraction: float = 0.2,
) -> dict:
    """Build a parallel observation descriptor bank without mutating the map."""
    anchor_count, offsets = _validate_registry(registry)
    cache = query_cache.get("queries", query_cache)
    if not isinstance(cache, Mapping):
        raise ValueError("query cache must be a mapping")
    query_names = list(registry["query_names"])
    queries = torch.as_tensor(registry["observation_query_indices"]).long()
    keypoints = torch.as_tensor(registry["observation_keypoint_indices"]).long()
    if queries.numel() and (
        int(queries.min()) < 0 or int(queries.max()) >= len(query_names)
    ):
        raise ValueError("Anchor observation references an invalid query")
    missing_names = sorted(
        {query_names[int(row)] for row in torch.unique(queries)} - set(cache)
    )
    if missing_names:
        raise ValueError(f"query cache is missing mapping query: {missing_names[0]}")
    feature_dim = int(torch.as_tensor(registry["anchor_features"]).shape[1])
    edge_count = int(queries.numel())
    observation_descriptors = torch.empty(
        edge_count, feature_dim, dtype=torch.float16
    )
    observation_confidence = torch.ones(edge_count, dtype=torch.float32)
    observation_valid = torch.zeros(edge_count, dtype=torch.bool)
    for query in torch.unique(queries, sorted=True).tolist():
        positions = torch.nonzero(queries == int(query), as_tuple=False).reshape(-1)
        payload = cache[query_names[int(query)]]
        descriptors = torch.as_tensor(payload["native_descriptors"])
        if descriptors.ndim != 2 or descriptors.shape[1] != feature_dim:
            raise ValueError("query descriptor dimension does not match Anchor map")
        rows = keypoints[positions]
        if rows.numel() and (
            int(rows.min()) < 0 or int(rows.max()) >= descriptors.shape[0]
        ):
            raise ValueError("Anchor observation references an invalid keypoint row")
        selected = descriptors[rows].float()
        finite = torch.isfinite(selected).all(dim=1) & (selected.norm(dim=1) > 0)
        observation_descriptors[positions] = selected.to(torch.float16)
        observation_valid[positions] = finite
        if "native_scores" in payload:
            scores = torch.as_tensor(payload["native_scores"])[rows].float()
            observation_confidence[positions] = torch.where(
                torch.isfinite(scores) & (scores > 0),
                scores,
                torch.ones_like(scores),
            )

    query_groups = torch.as_tensor(
        registry.get("query_group_ids", torch.full((len(query_names),), -1))
    ).long().reshape(-1)
    if query_groups.numel() != len(query_names):
        raise ValueError("query group IDs do not align with query names")
    trajectories = _trajectory_ids(query_names)
    descriptors = torch.zeros(anchor_count, feature_dim, dtype=torch.float32)
    valid = torch.zeros(anchor_count, dtype=torch.bool)
    observation_count = offsets[1:] - offsets[:-1]
    valid_observation_count = torch.zeros(anchor_count, dtype=torch.long)
    query_count = torch.zeros(anchor_count, dtype=torch.long)
    view_group_count = torch.zeros(anchor_count, dtype=torch.long)
    trajectory_count = torch.zeros(anchor_count, dtype=torch.long)
    stratum_count = torch.zeros(anchor_count, dtype=torch.long)
    retained_stratum_count = torch.zeros(anchor_count, dtype=torch.long)
    representability = torch.full((anchor_count,), float("nan"))
    observation_distance_mean = torch.full((anchor_count,), float("nan"))
    observation_distance_p90 = torch.full((anchor_count,), float("nan"))
    observation_distance_maximum = torch.full((anchor_count,), float("nan"))
    balanced_dispersion = torch.full((anchor_count,), float("nan"))
    deployment_cosine = torch.full((anchor_count,), float("nan"))
    deployment = F.normalize(
        torch.as_tensor(registry["anchor_features"]).float(), dim=1
    )

    for anchor in range(anchor_count):
        begin, end = int(offsets[anchor]), int(offsets[anchor + 1])
        if end == begin:
            continue
        keep = observation_valid[begin:end]
        valid_observation_count[anchor] = int(keep.sum())
        if not bool(keep.any()):
            continue
        anchor_queries = queries[begin:end][keep]
        raw = observation_descriptors[begin:end][keep].float()
        raw = F.normalize(raw, dim=1)
        groups = query_groups[anchor_queries]
        anchor_trajectories = trajectories[anchor_queries]
        fused, diagnostics = robust_observation_fusion(
            raw,
            anchor_queries,
            groups,
            anchor_trajectories,
            observation_confidence[begin:end][keep],
            trim_fraction=float(trim_fraction),
        )
        descriptors[anchor] = fused
        valid[anchor] = True
        query_count[anchor] = torch.unique(anchor_queries).numel()
        view_group_count[anchor] = torch.unique(groups[groups >= 0]).numel()
        trajectory_count[anchor] = torch.unique(
            anchor_trajectories[anchor_trajectories >= 0]
        ).numel()
        stratum_count[anchor] = int(diagnostics["stratum_count"])
        retained_stratum_count[anchor] = int(diagnostics["retained_stratum_count"])
        # This is the exact single-descriptor representability definition from
        # the V4 proposal, evaluated on normalized real observations.
        representability[anchor] = 1.0 - raw.mean(dim=0).norm()
        observation_distance_mean[anchor] = diagnostics[
            "observation_cosine_distance_mean"
        ]
        observation_distance_p90[anchor] = diagnostics[
            "observation_cosine_distance_p90"
        ]
        observation_distance_maximum[anchor] = diagnostics[
            "observation_cosine_distance_maximum"
        ]
        balanced_dispersion[anchor] = diagnostics["balanced_cosine_dispersion"]
        deployment_cosine[anchor] = torch.dot(fused, deployment[anchor])

    anchor_type = torch.as_tensor(registry["anchor_type"]).long().reshape(-1)
    report_arguments = {
        "valid": valid,
        "observation_count": observation_count,
        "query_count": query_count,
        "view_group_count": view_group_count,
        "trajectory_count": trajectory_count,
        "representability": representability,
        "dispersion": balanced_dispersion,
        "deployment_cosine": deployment_cosine,
    }
    report = {
        "anchor_count": anchor_count,
        "observation_edge_count": edge_count,
        "valid_observation_edge_count": int(observation_valid.sum()),
        "valid_descriptor_count": int(valid.sum()),
        "zero_observation_count": int((observation_count == 0).sum()),
        "single_observation_count": int((observation_count == 1).sum()),
        "invalid_only_observation_anchor_count": int(
            ((observation_count > 0) & ~valid).sum()
        ),
        "all": _category_report(
            torch.ones(anchor_count, dtype=torch.bool), **report_arguments
        ),
        "track": _category_report(anchor_type != 0, **report_arguments),
        "surface": _category_report(anchor_type == 0, **report_arguments),
    }
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "uses_test_queries": False,
        "audit_only": True,
        "deployment_descriptor_mutated": False,
        "source_registry_schema": str(registry["schema"]),
        "source_registry_version": int(registry.get("version", 0)),
        "query_cache_signature": query_cache.get("signature"),
        "fusion_policy": {
            "descriptor_source": "native_sparse_mapping_observation",
            "per_image_balance": True,
            "view_group_balance": True,
            "trajectory_group_balance": True,
            "prototype": "confidence_weighted_unit_mean",
            "robust_reducer": "cosine_medoid_trimmed_mean",
            "trim_fraction": float(trim_fraction),
        },
        "observation_descriptor": descriptors,
        "descriptor_valid_mask": valid,
        "valid_observation_count": valid_observation_count,
        "distinct_query_count": query_count,
        "distinct_view_group_count": view_group_count,
        "distinct_trajectory_count": trajectory_count,
        "stratum_count": stratum_count,
        "retained_stratum_count": retained_stratum_count,
        "descriptor_representability": representability,
        "observation_cosine_distance_mean": observation_distance_mean,
        "observation_cosine_distance_p90": observation_distance_p90,
        "observation_cosine_distance_maximum": observation_distance_maximum,
        "balanced_cosine_dispersion": balanced_dispersion,
        "fused_vs_deployment_cosine": deployment_cosine,
        "report": report,
    }
