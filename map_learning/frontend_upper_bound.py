"""Paired, mapping-only upper bounds for a stronger local frontend.

This module deliberately does not instantiate a feature network.  An extractor
must first materialize a provenance-locked probe cache.  The audit then keeps
the frozen map, mapping queries, keypoint budget, teacher, support folds, and
global cosine ranking fixed while measuring two independent failure domains:

* detector reachability of GT-projected, depth-legal anchors; and
* descriptor identity recall at the *reference SuperPoint keypoints*.

Pair matchers (for example LoFTR) do not satisfy this contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from map_learning.context_booster_crossfit import (
    _empty_retrieval,
    _update_descriptor_counts,
    accumulate_view_descriptors,
    combine_additive_counts,
    summarize_retrieval,
)
from map_learning.repeated_assignment_audit import _selected_csr_edges
from topology.crossfit_swap_revision import temporal_crossfit_split


PROBE_SCHEMA = "lafgs_frontend_upper_bound_probe_cache"
PROBE_VERSION = 1
DEFAULT_TOPKS = (1, 2, 4, 8, 16, 32)
DEFAULT_REACHABILITY_RADII_PX = (2.0, 4.0, 8.0)


def file_sha256(path: str | Path) -> str:
    """Hash an artifact without loading it into accelerator memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor shape, dtype, and canonical contiguous CPU bytes."""
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def query_cache_queries(query_cache: Mapping) -> Mapping[str, Mapping]:
    queries = query_cache.get("queries", query_cache)
    if not isinstance(queries, Mapping):
        raise TypeError("query cache must contain a query-name mapping")
    return queries


def teacher_records(teacher: Mapping) -> dict[int, Mapping]:
    records = {
        int(record["query_index"]): record for record in teacher["records"]
    }
    if len(records) != len(teacher["records"]):
        raise ValueError("teacher query indices must be unique")
    return records


def reference_registry(query_cache: Mapping, teacher: Mapping) -> dict:
    """Return the exact row registry an independent extractor must replay."""
    queries = query_cache_queries(query_cache)
    names = list(teacher["query_names"])
    records = teacher_records(teacher)
    registry = {}
    for query_index, record in sorted(records.items()):
        name = names[query_index]
        cached = queries[name]
        keypoints = torch.as_tensor(cached["native_keypoints"]).float()
        rows = torch.as_tensor(record["query_rows"]).long()
        if rows.numel() and int(rows.max()) >= keypoints.shape[0]:
            raise ValueError(f"teacher rows exceed query cache for {name}")
        registry[name] = {
            "query_index": int(query_index),
            "native_row_count": int(keypoints.shape[0]),
            "teacher_row_count": int(rows.numel()),
            "reference_keypoints_sha256": tensor_sha256(keypoints),
            "native_input_hw": [
                int(value) for value in cached["native_input_hw"]
            ],
            "pixel_center_offset": float(
                cached.get("pixel_center_offset", 0.5)
            ),
        }
    return registry


def probe_contract(query_cache: Mapping, teacher: Mapping) -> dict:
    """Create a serializable, unfilled producer contract for a probe cache."""
    queries = query_cache_queries(query_cache)
    names = list(teacher["query_names"])
    requested_k = {
        int(
            queries[name]["native_sparse_metadata"]["detect_num"]
        )
        for name in names
    }
    if len(requested_k) != 1:
        raise ValueError("reference cache does not use one frozen detector K")
    descriptor_dims = {
        int(torch.as_tensor(queries[name]["native_descriptors"]).shape[1])
        for name in names
    }
    if len(descriptor_dims) != 1:
        raise ValueError("reference descriptor dimension is inconsistent")
    return {
        "schema": PROBE_SCHEMA,
        "version": PROBE_VERSION,
        "mapping_only": True,
        "uses_test_queries": False,
        "reference_frontend": "frozen_superpoint",
        "reference_descriptor_dim": descriptor_dims.pop(),
        "requested_keypoint_count": requested_k.pop(),
        "required_coordinate_convention": (
            "reference_grid_index_then_cached_pixel_center_offset"
        ),
        "reference_query_cache_signature": query_cache.get("signature"),
        "reference_teacher_schema": teacher.get("schema"),
        "required_capabilities": {
            "detector_repeatability": {
                "payload": [
                    "detector_keypoints",
                    "detector_scores",
                    "detected_count_before_mask",
                ],
                "same_requested_k": True,
                "descriptors_used": False,
            },
            "descriptor_identity": {
                "payload": [
                    "reference_keypoints_sha256",
                    "descriptor_at_reference_keypoints",
                ],
                "same_rows": True,
                "same_descriptor_dim": True,
                "candidate_detector_used": False,
            },
        },
        "reference_registry": reference_registry(query_cache, teacher),
    }


def _validate_local_artifact(frontend: Mapping) -> dict:
    artifact = frontend.get("weights", {})
    path_text = str(artifact.get("path", ""))
    expected = str(artifact.get("sha256", "")).lower()
    if not path_text or not expected:
        raise ValueError("frontend weights require local path and SHA256")
    if "://" in path_text:
        raise ValueError("network weight references are forbidden")
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"locked frontend weights not found: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"frontend weight SHA256 mismatch: expected {expected}, got {actual}"
        )
    return {"path": str(path), "sha256": actual, "verified": True}


def validate_probe(
    probe: Mapping,
    query_cache: Mapping,
    teacher: Mapping,
    *,
    require_detector: bool = False,
    require_descriptor: bool = False,
    verify_weight_artifact: bool = True,
) -> dict:
    """Fail closed when a candidate is not exactly paired to the reference."""
    if probe.get("schema") != PROBE_SCHEMA or int(probe.get("version", -1)) != 1:
        raise ValueError("unsupported frontend upper-bound probe schema")
    if probe.get("mapping_only") is not True:
        raise ValueError("probe must attest mapping_only=true")
    if probe.get("uses_test_queries") is not False:
        raise ValueError("probe must attest uses_test_queries=false")
    reference = probe.get("reference", {})
    cache_signature = query_cache.get("signature")
    if cache_signature is not None and reference.get(
        "query_cache_signature"
    ) != cache_signature:
        raise ValueError("probe query-cache signature mismatch")
    if reference.get("teacher_schema") != teacher.get("schema"):
        raise ValueError("probe teacher schema mismatch")
    frontend = probe.get("frontend", {})
    family = str(frontend.get("family", ""))
    if family == "pair_matcher":
        raise ValueError("pair matchers are outside the global-descriptor contract")
    if family != "independent_local_frontend":
        raise ValueError("probe is not an independent local frontend")
    if not str(frontend.get("implementation_id", "")):
        raise ValueError("frontend implementation/version is not locked")
    if frontend.get("coordinate_convention") != (
        "reference_grid_index_then_cached_pixel_center_offset"
    ):
        raise ValueError("candidate coordinate convention is not reference-locked")
    artifact = (
        _validate_local_artifact(frontend)
        if verify_weight_artifact
        else {"verified": False}
    )
    capabilities = probe.get("capabilities", {})
    if require_detector and capabilities.get("detector_repeatability") is not True:
        raise ValueError("probe lacks detector_repeatability capability")
    if require_descriptor and capabilities.get("descriptor_identity") is not True:
        raise ValueError("probe lacks descriptor_identity capability")

    queries = query_cache_queries(query_cache)
    names = list(teacher["query_names"])
    probe_queries = probe.get("queries", {})
    if set(probe_queries) != set(names):
        raise ValueError("probe and teacher query-name sets differ")
    requested_k_values = {
        int(queries[name]["native_sparse_metadata"]["detect_num"])
        for name in names
    }
    if len(requested_k_values) != 1:
        raise ValueError("reference cache has inconsistent keypoint budgets")
    requested_k = requested_k_values.pop()
    if int(frontend.get("requested_keypoint_count", -1)) != requested_k:
        raise ValueError("candidate and reference detector K differ")

    reference_dims = {
        int(torch.as_tensor(queries[name]["native_descriptors"]).shape[1])
        for name in names
    }
    if len(reference_dims) != 1:
        raise ValueError("reference descriptor dimension is inconsistent")
    reference_dim = reference_dims.pop()
    if require_descriptor and int(frontend.get("descriptor_dim", -1)) != reference_dim:
        raise ValueError("descriptor identity arm requires the same dimension")

    validated_rows = 0
    validated_keypoints = 0
    for name in names:
        cached = queries[name]
        candidate = probe_queries[name]
        reference_keypoints = torch.as_tensor(
            cached["native_keypoints"]
        ).float()
        expected_hash = tensor_sha256(reference_keypoints)
        if str(candidate.get("reference_keypoints_sha256", "")) != expected_hash:
            raise ValueError(f"reference keypoint registry mismatch for {name}")
        if require_descriptor:
            descriptor = torch.as_tensor(
                candidate["descriptor_at_reference_keypoints"]
            ).float()
            expected_shape = (reference_keypoints.shape[0], reference_dim)
            if tuple(descriptor.shape) != expected_shape:
                raise ValueError(
                    f"descriptor replay shape mismatch for {name}: "
                    f"{tuple(descriptor.shape)} != {expected_shape}"
                )
            if not bool(torch.isfinite(descriptor).all()):
                raise ValueError(f"non-finite candidate descriptor for {name}")
            if bool((torch.linalg.norm(descriptor, dim=1) <= 0).any()):
                raise ValueError(f"zero candidate descriptor for {name}")
            validated_rows += int(descriptor.shape[0])
        if require_detector:
            keypoints = torch.as_tensor(candidate["detector_keypoints"]).float()
            scores = torch.as_tensor(candidate["detector_scores"]).float().reshape(-1)
            if keypoints.ndim != 2 or keypoints.shape[1] != 2:
                raise ValueError(f"candidate keypoints must be [N,2] for {name}")
            if scores.numel() != keypoints.shape[0]:
                raise ValueError(f"candidate keypoint/score rows differ for {name}")
            if not bool(torch.isfinite(scores).all()):
                raise ValueError(f"non-finite candidate detector score for {name}")
            if scores.numel() > 1 and not bool((scores[:-1] >= scores[1:]).all()):
                raise ValueError(
                    f"candidate detector scores are not top-K ordered for {name}"
                )
            before = int(candidate["detected_count_before_mask"])
            if not keypoints.shape[0] <= before <= requested_k:
                raise ValueError(f"candidate detector violates K cap for {name}")
            _validate_legal_keypoints(keypoints, cached, name)
            validated_keypoints += int(keypoints.shape[0])
    return {
        "artifact": artifact,
        "query_count": len(names),
        "requested_keypoint_count": requested_k,
        "reference_descriptor_dim": reference_dim,
        "validated_descriptor_rows": validated_rows,
        "validated_detector_keypoints": validated_keypoints,
    }


def _mask_2d(value: torch.Tensor) -> torch.Tensor:
    mask = torch.as_tensor(value).bool().squeeze()
    if mask.ndim != 2:
        raise ValueError(f"expected 2D mask, got {tuple(mask.shape)}")
    return mask


def _validate_legal_keypoints(
    keypoints: torch.Tensor, cached: Mapping, name: str
) -> None:
    height, width = (int(value) for value in cached["native_input_hw"])
    if keypoints.numel() == 0:
        return
    if not bool(torch.isfinite(keypoints).all()):
        raise ValueError(f"non-finite candidate keypoint for {name}")
    pixels = torch.floor(keypoints).long()
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    if not bool(inside.all()):
        raise ValueError(f"candidate keypoint outside reference image for {name}")
    mask = _mask_2d(cached["native_valid_mask"])
    if not bool(mask[pixels[:, 1], pixels[:, 0]].all()):
        raise ValueError(f"candidate keypoint violates frozen valid mask for {name}")


def _legal_anchor_projections(
    state: Mapping,
    cached: Mapping,
    *,
    depth_abs_tolerance_m: float,
    depth_rel_tolerance: float,
    alpha_minimum: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    xyz = torch.as_tensor(state["anchor_xyz"]).float().cpu()
    pose = torch.as_tensor(cached["pose_w2c"]).float().cpu()
    intrinsic = torch.as_tensor(cached["native_K"]).float().cpu()
    camera = xyz @ pose[:3, :3].T + pose[:3, 3]
    z = camera[:, 2]
    projected = camera @ intrinsic.T
    uv = projected[:, :2] / z[:, None].clamp_min(1e-8)
    pixels = torch.floor(uv).long()
    height, width = (int(value) for value in cached["native_input_hw"])
    inside = (
        torch.isfinite(uv).all(dim=1)
        & torch.isfinite(z)
        & (z > 0)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    safe_x = pixels[:, 0].clamp(0, width - 1)
    safe_y = pixels[:, 1].clamp(0, height - 1)
    valid_mask = _mask_2d(cached["native_valid_mask"])
    depth = torch.as_tensor(cached["native_depth"]).float().squeeze()
    alpha = torch.as_tensor(cached["native_alpha"]).float().squeeze()
    rendered_depth = depth[safe_y, safe_x]
    rendered_alpha = alpha[safe_y, safe_x]
    tolerance = float(depth_abs_tolerance_m) + (
        float(depth_rel_tolerance) * rendered_depth.abs()
    )
    legal = (
        inside
        & valid_mask[safe_y, safe_x]
        & torch.isfinite(rendered_depth)
        & (rendered_depth > 0)
        & torch.isfinite(rendered_alpha)
        & (rendered_alpha >= float(alpha_minimum))
        & ((z - rendered_depth).abs() <= tolerance)
    )
    return uv[legal], torch.nonzero(legal, as_tuple=False).reshape(-1)


def _nearest_distances(
    targets: torch.Tensor, keypoints: torch.Tensor, chunk_size: int = 4096
) -> torch.Tensor:
    if targets.numel() == 0:
        return torch.empty(0)
    if keypoints.numel() == 0:
        return torch.full((targets.shape[0],), torch.inf)
    chunks = []
    for start in range(0, targets.shape[0], int(chunk_size)):
        distances = torch.cdist(
            targets[start : start + int(chunk_size)].float(),
            keypoints.float(),
        )
        chunks.append(distances.min(dim=1).values)
    return torch.cat(chunks)


def _reachability_counts(
    distances: torch.Tensor,
    anchor_indices: torch.Tensor,
    anchor_type: torch.Tensor,
    radii_px: Sequence[float],
) -> dict:
    output = {"target_count": int(distances.numel()), "hit_count": {}}
    for name, mask in (
        ("all", torch.ones_like(anchor_indices, dtype=torch.bool)),
        ("track_core", anchor_type[anchor_indices] != 0),
        ("gaussian_reserve", anchor_type[anchor_indices] == 0),
    ):
        total = int(mask.sum())
        output.setdefault("by_anchor_kind", {})[name] = {
            "target_count": total,
            "hit_count": {
                str(float(radius)): int((distances[mask] <= float(radius)).sum())
                for radius in radii_px
            },
        }
    output["hit_count"] = output["by_anchor_kind"]["all"]["hit_count"]
    return output


def _combine_reachability(rows: Sequence[Mapping], radii_px: Sequence[float]) -> dict:
    combined = {"query_count": len(rows), "by_anchor_kind": {}}
    for kind in ("all", "track_core", "gaussian_reserve"):
        total = sum(int(row["by_anchor_kind"][kind]["target_count"]) for row in rows)
        hits = {
            str(float(radius)): sum(
                int(row["by_anchor_kind"][kind]["hit_count"][str(float(radius))])
                for row in rows
            )
            for radius in radii_px
        }
        combined["by_anchor_kind"][kind] = {
            "target_count": total,
            "hit_count": hits,
            "reachable_fraction": {
                key: float(value / max(total, 1)) for key, value in hits.items()
            },
        }
    return combined


@torch.inference_mode()
def audit_detector_repeatability(
    *,
    state: Mapping,
    query_cache: Mapping,
    teacher: Mapping,
    probe: Mapping,
    radii_px: Sequence[float] = DEFAULT_REACHABILITY_RADII_PX,
    depth_abs_tolerance_m: float | None = None,
    depth_rel_tolerance: float | None = None,
    alpha_minimum: float | None = None,
    verify_weight_artifact: bool = True,
) -> dict:
    """Measure same-K detector reachability of GT/depth-legal map anchors."""
    attestation = validate_probe(
        probe,
        query_cache,
        teacher,
        require_detector=True,
        verify_weight_artifact=verify_weight_artifact,
    )
    config = teacher.get("config", {})
    depth_abs = float(
        config.get("depth_abs_tolerance_m", 0.05)
        if depth_abs_tolerance_m is None
        else depth_abs_tolerance_m
    )
    depth_rel = float(
        config.get("depth_rel_tolerance", 0.02)
        if depth_rel_tolerance is None
        else depth_rel_tolerance
    )
    alpha = float(
        config.get("alpha_minimum", 0.01)
        if alpha_minimum is None
        else alpha_minimum
    )
    radii = tuple(sorted(set(float(value) for value in radii_px)))
    if not radii or radii[0] <= 0:
        raise ValueError("reachability radii must be positive")
    queries = query_cache_queries(query_cache)
    anchor_type = torch.as_tensor(state["anchor_type"]).long().cpu()
    if anchor_type.numel() != torch.as_tensor(state["anchor_xyz"]).shape[0]:
        raise ValueError("anchor type and xyz registries differ")
    baseline_rows = []
    candidate_rows = []
    per_query = []
    for name in teacher["query_names"]:
        cached = queries[name]
        target_uv, anchor_indices = _legal_anchor_projections(
            state,
            cached,
            depth_abs_tolerance_m=depth_abs,
            depth_rel_tolerance=depth_rel,
            alpha_minimum=alpha,
        )
        offset = float(cached.get("pixel_center_offset", 0.5))
        baseline_keypoints = (
            torch.as_tensor(cached["native_keypoints"]).float() + offset
        )
        candidate_keypoints = (
            torch.as_tensor(probe["queries"][name]["detector_keypoints"]).float()
            + offset
        )
        baseline = _reachability_counts(
            _nearest_distances(target_uv, baseline_keypoints),
            anchor_indices,
            anchor_type,
            radii,
        )
        candidate = _reachability_counts(
            _nearest_distances(target_uv, candidate_keypoints),
            anchor_indices,
            anchor_type,
            radii,
        )
        baseline_rows.append(baseline)
        candidate_rows.append(candidate)
        per_query.append(
            {
                "query_name": name,
                "legal_anchor_count": int(anchor_indices.numel()),
                "reference_keypoint_count": int(baseline_keypoints.shape[0]),
                "candidate_keypoint_count": int(candidate_keypoints.shape[0]),
                "frozen_superpoint": baseline,
                "candidate": candidate,
            }
        )
    baseline_summary = _combine_reachability(baseline_rows, radii)
    candidate_summary = _combine_reachability(candidate_rows, radii)
    delta = {}
    for kind in baseline_summary["by_anchor_kind"]:
        delta[kind] = {
            key: float(
                candidate_summary["by_anchor_kind"][kind]["reachable_fraction"][key]
                - baseline_summary["by_anchor_kind"][kind]["reachable_fraction"][key]
            )
            for key in baseline_summary["by_anchor_kind"][kind][
                "reachable_fraction"
            ]
        }
    return {
        "schema": "lafgs_mapping_detector_repeatability_upper_bound",
        "version": 1,
        "mapping_only": True,
        "uses_test_queries": False,
        "attestation": attestation,
        "config": {
            "radii_px": list(radii),
            "depth_abs_tolerance_m": depth_abs,
            "depth_rel_tolerance": depth_rel,
            "alpha_minimum": alpha,
            "target_universe": "frozen_map_gt_projection_depth_alpha_mask_legal",
            "same_requested_k": True,
        },
        "frozen_superpoint": baseline_summary,
        "candidate": candidate_summary,
        "delta_candidate_minus_superpoint": delta,
        "per_query": per_query,
    }


def _build_descriptor_banks(
    *,
    query_cache: Mapping,
    teacher: Mapping,
    probe: Mapping,
    support_query_indices: Sequence[int],
    minimum_support_views: int,
) -> tuple[dict[str, torch.Tensor], dict]:
    names = list(teacher["query_names"])
    records = teacher_records(teacher)
    queries = query_cache_queries(query_cache)
    anchor_count = int(teacher["anchor_count"])
    descriptor_dim = int(probe["frontend"]["descriptor_dim"])
    raw_sum = torch.zeros((anchor_count, descriptor_dim))
    candidate_sum = torch.zeros_like(raw_sum)
    view_counts = torch.zeros(anchor_count, dtype=torch.long)
    positive_edges = 0
    for query_index in support_query_indices:
        query_index = int(query_index)
        record = records[query_index]
        name = names[query_index]
        rows = torch.as_tensor(record["query_rows"]).long()
        selected = torch.arange(rows.numel())
        _, edge_rows, edge_anchors = _selected_csr_edges(
            record, "positive", selected
        )
        if not edge_rows.numel():
            continue
        raw = F.normalize(
            torch.as_tensor(queries[name]["native_descriptors"]).float(), dim=1
        )
        candidate = F.normalize(
            torch.as_tensor(
                probe["queries"][name]["descriptor_at_reference_keypoints"]
            ).float(),
            dim=1,
        )
        native_rows = rows[edge_rows]
        observed = accumulate_view_descriptors(
            raw_sum, view_counts, edge_anchors, raw[native_rows]
        )
        candidate_counts = torch.zeros_like(view_counts)
        candidate_observed = accumulate_view_descriptors(
            candidate_sum,
            candidate_counts,
            edge_anchors,
            candidate[native_rows],
        )
        if candidate_observed != observed or int(candidate_counts.sum()) != observed:
            raise AssertionError("paired descriptor support observations diverged")
        positive_edges += int(edge_rows.numel())
    supported = view_counts >= int(minimum_support_views)
    if not bool(supported.any()):
        raise ValueError("support fold has no anchor meeting the view threshold")
    anchor_indices = torch.nonzero(supported, as_tuple=False).reshape(-1)
    return {
        "anchor_indices": anchor_indices,
        "frozen_superpoint": F.normalize(raw_sum[anchor_indices], dim=1),
        "candidate": F.normalize(candidate_sum[anchor_indices], dim=1),
    }, {
        "support_query_count": len(support_query_indices),
        "positive_edge_count": positive_edges,
        "minimum_support_views": int(minimum_support_views),
        "supported_anchor_count": int(anchor_indices.numel()),
    }


def _evaluate_descriptor_banks(
    *,
    state: Mapping,
    query_cache: Mapping,
    teacher: Mapping,
    probe: Mapping,
    gate_query_indices: Sequence[int],
    banks: Mapping[str, torch.Tensor],
    topks: Sequence[int],
) -> dict:
    names = list(teacher["query_names"])
    records = teacher_records(teacher)
    queries = query_cache_queries(query_cache)
    anchor_indices = torch.as_tensor(banks["anchor_indices"]).long()
    supported = torch.zeros(int(teacher["anchor_count"]), dtype=torch.bool)
    supported[anchor_indices] = True
    anchor_type = torch.as_tensor(state["anchor_type"]).long().cpu()
    maximum_k = min(max(topks), int(anchor_indices.numel()))
    counts = {
        name: _empty_retrieval(topks)
        for name in ("frozen_superpoint", "candidate")
    }
    for query_index in gate_query_indices:
        query_index = int(query_index)
        name = names[query_index]
        record = records[query_index]
        rows = torch.as_tensor(record["query_rows"]).long()
        selected = torch.arange(rows.numel())
        if not rows.numel():
            continue
        _, positive_rows, positive_anchors = _selected_csr_edges(
            record, "positive", selected
        )
        positive_keep = supported[positive_anchors]
        positive_rows = positive_rows[positive_keep]
        positive_anchors = positive_anchors[positive_keep]
        _, ambiguous_rows, ambiguous_anchors = _selected_csr_edges(
            record, "ambiguous", selected
        )
        ambiguous_keep = supported[ambiguous_anchors]
        ambiguous_rows = ambiguous_rows[ambiguous_keep]
        ambiguous_anchors = ambiguous_anchors[ambiguous_keep]
        descriptors = {
            "frozen_superpoint": F.normalize(
                torch.as_tensor(queries[name]["native_descriptors"]).float()[rows],
                dim=1,
            ),
            "candidate": F.normalize(
                torch.as_tensor(
                    probe["queries"][name]["descriptor_at_reference_keypoints"]
                ).float()[rows],
                dim=1,
            ),
        }
        for descriptor_name, query_descriptor in descriptors.items():
            scores = query_descriptor @ banks[descriptor_name].T
            local_ranked = torch.topk(scores, k=maximum_k, dim=1).indices
            ranked = anchor_indices[local_ranked]
            _update_descriptor_counts(
                counts[descriptor_name],
                ranked=ranked,
                positive_edge_rows=positive_rows,
                positive_edge_anchors=positive_anchors,
                ambiguous_edge_rows=ambiguous_rows,
                ambiguous_edge_anchors=ambiguous_anchors,
                anchor_type=anchor_type,
                topks=topks,
            )
    return counts


def _recall_delta(candidate: Mapping, baseline: Mapping) -> dict:
    output = {
        key: float(candidate["positive_recall_at_k"][key])
        - float(baseline["positive_recall_at_k"][key])
        for key in baseline["positive_recall_at_k"]
    }
    output["by_anchor_kind"] = {
        kind: {
            key: float(
                candidate["positive_recall_at_k_by_anchor_kind"][kind][key]
            )
            - float(baseline["positive_recall_at_k_by_anchor_kind"][kind][key])
            for key in baseline["positive_recall_at_k_by_anchor_kind"][kind]
        }
        for kind in baseline["positive_recall_at_k_by_anchor_kind"]
    }
    return output


@torch.inference_mode()
def audit_descriptor_identity_crossfit(
    *,
    state: Mapping,
    query_cache: Mapping,
    teacher: Mapping,
    probe: Mapping,
    crossfit_blocks: int = 8,
    minimum_support_views: int = 2,
    topks: Sequence[int] = DEFAULT_TOPKS,
    verify_weight_artifact: bool = True,
) -> dict:
    """Run bidirectional support/gate identity recall at fixed SP keypoints."""
    attestation = validate_probe(
        probe,
        query_cache,
        teacher,
        require_descriptor=True,
        verify_weight_artifact=verify_weight_artifact,
    )
    topks = tuple(sorted(set(int(value) for value in topks)))
    if not topks or topks[0] < 1:
        raise ValueError("top-K values must be positive")
    selection, gate, split = temporal_crossfit_split(
        list(teacher["query_names"]), int(crossfit_blocks)
    )
    directions = []
    pooled_counts = {
        name: [] for name in ("frozen_superpoint", "candidate")
    }
    for direction, support, heldout in (
        ("selection_to_gate", selection, gate),
        ("gate_to_selection", gate, selection),
    ):
        banks, bank_report = _build_descriptor_banks(
            query_cache=query_cache,
            teacher=teacher,
            probe=probe,
            support_query_indices=support,
            minimum_support_views=minimum_support_views,
        )
        counts = _evaluate_descriptor_banks(
            state=state,
            query_cache=query_cache,
            teacher=teacher,
            probe=probe,
            gate_query_indices=heldout,
            banks=banks,
            topks=topks,
        )
        summaries = {
            name: summarize_retrieval(value, topks)
            for name, value in counts.items()
        }
        for name in pooled_counts:
            pooled_counts[name].append(counts[name])
        directions.append(
            {
                "direction": direction,
                "support": bank_report,
                "heldout_query_count": len(heldout),
                **summaries,
                "delta_candidate_minus_superpoint": _recall_delta(
                    summaries["candidate"], summaries["frozen_superpoint"]
                ),
            }
        )
    pooled = {
        name: summarize_retrieval(
            combine_additive_counts(values, topks), topks
        )
        for name, values in pooled_counts.items()
    }
    return {
        "schema": "lafgs_mapping_descriptor_identity_upper_bound",
        "version": 1,
        "mapping_only": True,
        "uses_test_queries": False,
        "attestation": attestation,
        "protocol": {
            "query_coordinates": "exact_frozen_superpoint_keypoint_rows",
            "positive_labels": str(teacher.get("schema", "unknown")),
            "map_bank": "same_positive_edges_view_balanced_support_only",
            "ranking": "global_cosine",
            "crossfit": "bidirectional_temporal_block",
            "crossfit_blocks": int(crossfit_blocks),
            "minimum_support_views": int(minimum_support_views),
            "topks": list(topks),
            "candidate_detector_used": False,
        },
        "split": split,
        "directions": directions,
        "pooled": pooled,
        "delta_candidate_minus_superpoint": _recall_delta(
            pooled["candidate"], pooled["frozen_superpoint"]
        ),
    }
