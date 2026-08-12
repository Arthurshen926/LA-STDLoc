"""Summaries and a preregistered mechanism gate for K_mapping factors."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from topology.adaptive_distillation import _adaptive_track_eligibility


def _depth_at_sparse_rows(record: Mapping[str, Any]) -> torch.Tensor:
    keypoints = torch.as_tensor(record["native_keypoints"]).float()
    source = record.get("native_depth_at_keypoints", record.get("native_depth"))
    if source is None:
        raise ValueError("mapping cache lacks native sparse depth")
    depth = torch.as_tensor(source)
    if depth.ndim == 1:
        if depth.shape != (keypoints.shape[0],):
            raise ValueError("native sparse depth row count differs from keypoints")
        return depth
    x = keypoints[:, 0].round().long().clamp(0, int(depth.shape[1]) - 1)
    y = keypoints[:, 1].round().long().clamp(0, int(depth.shape[0]) - 1)
    return depth[y, x]


def audit_sparse_refresh_equivalence(
    source_payload: Mapping[str, Any], refreshed_payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove a fresh attested cache is Track-input identical to legacy data."""
    source = source_payload.get("queries", source_payload)
    refreshed = refreshed_payload.get("queries", refreshed_payload)
    if set(source) != set(refreshed):
        raise ValueError("source and refreshed caches have different query sets")
    query_order_exact = list(source) == list(refreshed)
    track_fields = (
        "native_keypoints",
        "native_descriptors",
        "native_scores",
        "native_K",
        "pose_w2c",
        "native_depth",
        "native_valid_mask",
        "native_input_hw",
    )
    track_input_exact = 0
    effective_sparse_depth_exact = 0
    raster_exact = 0
    metadata_pass = 0
    payload = refreshed_payload["signature_payload"]
    target_k = int(payload["native_sparse_keypoint_count"])
    target_nms = int(payload["native_sparse_nms_radius"])
    for name in sorted(source):
        left = source[name]
        right = refreshed[name]
        track_input_exact += int(
            all(
                torch.equal(torch.as_tensor(left[field]), torch.as_tensor(right[field]))
                for field in track_fields
            )
        )
        effective_sparse_depth_exact += int(
            torch.equal(
                _depth_at_sparse_rows(left),
                _depth_at_sparse_rows(right),
            )
        )
        raster_exact += int(
            torch.equal(
                torch.as_tensor(left["native_alpha"]),
                torch.as_tensor(right["native_alpha"]),
            )
        )
        metadata = right.get("native_sparse_metadata", {})
        metadata_pass += int(
            int(metadata.get("requested_keypoint_count", -1)) == target_k
            and int(metadata.get("nms_radius", -1)) == target_nms
        )
    count = len(source)
    exact = (
        query_order_exact
        and track_input_exact == count
        and effective_sparse_depth_exact == count
        and raster_exact == count
        and metadata_pass == count
        and target_nms == 4
    )
    return {
        "query_count": count,
        "query_order_exact": query_order_exact,
        "target_k_mapping": target_k,
        "target_nms_radius": target_nms,
        "track_input_exact_query_count": track_input_exact,
        "effective_sparse_depth_exact_query_count": (effective_sparse_depth_exact),
        "native_alpha_exact_query_count": raster_exact,
        "refreshed_metadata_pass_count": metadata_pass,
        "content_equivalent_track_payload_reuse_authorized": exact,
    }


def audit_density_cache_pair(
    control_payload: Mapping[str, Any], high_payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove that K is the only sparse mapping-cache factor."""
    control_signature = dict(control_payload["signature_payload"])
    high_signature = dict(high_payload["signature_payload"])
    control_k = int(control_signature.pop("native_sparse_keypoint_count"))
    high_k = int(high_signature.pop("native_sparse_keypoint_count"))
    control_nms = int(control_signature.pop("native_sparse_nms_radius"))
    high_nms = int(high_signature.pop("native_sparse_nms_radius"))
    signatures_match_except_k = control_signature == high_signature
    control = control_payload.get("queries", control_payload)
    high = high_payload.get("queries", high_payload)
    if set(control) != set(high):
        raise ValueError("density cache arms have different mapping query sets")
    prefix_exact = 0
    prefix_close = 0
    geometry_exact = 0
    metadata_pass = 0
    fields = ("native_K", "pose_w2c", "native_depth", "native_alpha")
    for name in sorted(control):
        left = control[name]
        right = high[name]
        rows = int(torch.as_tensor(left["native_keypoints"]).shape[0])
        left_prefix = (
            torch.as_tensor(left["native_keypoints"]),
            torch.as_tensor(left["native_descriptors"]),
            torch.as_tensor(left["native_scores"]),
        )
        right_prefix = (
            torch.as_tensor(right["native_keypoints"])[:rows],
            torch.as_tensor(right["native_descriptors"])[:rows],
            torch.as_tensor(right["native_scores"])[:rows],
        )
        prefix_exact += int(
            all(torch.equal(a, b) for a, b in zip(left_prefix, right_prefix))
        )
        prefix_close += int(
            all(
                torch.allclose(a.float(), b.float(), atol=1e-5, rtol=1e-5)
                for a, b in zip(left_prefix, right_prefix)
            )
        )
        geometry_exact += int(
            all(
                torch.equal(torch.as_tensor(left[field]), torch.as_tensor(right[field]))
                for field in fields
            )
        )
        left_metadata = left.get("native_sparse_metadata", {})
        right_metadata = right.get("native_sparse_metadata", {})
        metadata_pass += int(
            int(
                left_metadata.get(
                    "requested_keypoint_count", left_metadata.get("detect_num", -1)
                )
            )
            == control_k
            and int(
                right_metadata.get(
                    "requested_keypoint_count", right_metadata.get("detect_num", -1)
                )
            )
            == high_k
            and int(left_metadata.get("nms_radius", -1)) == control_nms
            and int(right_metadata.get("nms_radius", -1)) == high_nms
        )
    count = len(control)
    strict = (
        control_k < high_k
        and control_nms == high_nms == 4
        and signatures_match_except_k
        and metadata_pass == count
        and prefix_close == count
        and geometry_exact == count
    )
    return {
        "query_count": count,
        "control_k_mapping": control_k,
        "high_k_mapping": high_k,
        "control_nms_radius": control_nms,
        "high_nms_radius": high_nms,
        "signature_payload_equal_except_k": signatures_match_except_k,
        "sparse_prefix_exact_query_count": prefix_exact,
        "sparse_prefix_allclose_query_count": prefix_close,
        "immutable_geometry_exact_query_count": geometry_exact,
        "per_query_metadata_pass_count": metadata_pass,
        "strict_single_factor_contract_passed": strict,
    }


def distribution(values: torch.Tensor) -> dict[str, float | int | None]:
    value = torch.as_tensor(values).float().reshape(-1)
    value = value[torch.isfinite(value)]
    if not value.numel():
        return {"count": 0, "median": None, "p90": None, "mean": None}
    return {
        "count": int(value.numel()),
        "median": float(torch.quantile(value, 0.5)),
        "p90": float(torch.quantile(value, 0.9)),
        "mean": float(value.mean()),
    }


def summarize_track_payload(
    payload: Mapping[str, Any], calibration: Mapping[str, Any]
) -> dict[str, Any]:
    if payload.get("schema") != "lafgs_track_first_payload":
        raise ValueError("unsupported Track payload")
    if calibration.get("schema") != "lafgs_mapping_only_scene_calibration":
        raise ValueError("density factor requires mapping-only calibration")
    uses_test = calibration.get(
        "uses_test_queries", calibration.get("sources", {}).get("uses_test_queries")
    )
    if uses_test is not False:
        raise ValueError("density factor cannot use test queries")
    parameters = calibration["parameters"]
    geometry = payload["track_geometry"]
    diagnostics = payload["diagnostics"]
    triangulated = torch.as_tensor(geometry["triangulated"]).bool()
    strict = _adaptive_track_eligibility(
        geometry,
        median_px=float(parameters["track_reprojection_median_px"]),
        p90_px=float(parameters["track_reprojection_p90_px"]),
        covariance_m2=float(parameters["track_covariance_trace_m2"]),
        broad=False,
    )
    broad = _adaptive_track_eligibility(
        geometry,
        median_px=float(parameters["track_reprojection_median_px"]),
        p90_px=float(parameters["track_reprojection_p90_px"]),
        covariance_m2=float(parameters["track_covariance_trace_m2"]),
        broad=True,
    )
    covariance = torch.as_tensor(geometry["triangulation_covariance_trace"]).float()
    parallax = torch.as_tensor(geometry["triangulation_parallax_deg"]).float()
    accepted_cycle = int(diagnostics["track_graded_cycle_edge_count"])
    accepted_chain = int(diagnostics["track_graded_chain_edge_count"])
    return {
        "query_count": len(payload["query_names"]),
        "camera_pair_candidate_count": int(
            diagnostics["track_camera_pair_candidate_count"]
        ),
        "camera_pair_matched_count": int(
            diagnostics["track_camera_pair_matched_count"]
        ),
        "raw_reciprocal_epipolar_edge_count": int(
            diagnostics["track_raw_reciprocal_epipolar_edge_count"]
        ),
        "accepted_cycle_edge_count": accepted_cycle,
        "accepted_chain_edge_count": accepted_chain,
        "accepted_edge_count": accepted_cycle + accepted_chain,
        "conflict_rejected_edge_count": int(
            diagnostics["track_graded_conflict_rejected_edge_count"]
        ),
        "track_count": int(diagnostics["track_count"]),
        "track_observation_count": int(diagnostics["track_observation_count"]),
        "level_a_track_count": int(diagnostics["track_level_a_count"]),
        "level_b_track_count": int(diagnostics["track_level_b_count"]),
        "triangulated_track_count": int(triangulated.sum()),
        "strict_track_count": int(strict.sum()),
        "broad_track_count": int(broad.sum()),
        "high_confidence_track_count": int(
            torch.as_tensor(geometry["triangulation_high_confidence"]).sum()
        ),
        "triangulated_covariance_trace_m2": distribution(covariance[triangulated]),
        "strict_covariance_trace_m2": distribution(covariance[strict]),
        "broad_covariance_trace_m2": distribution(covariance[broad]),
        "triangulated_parallax_deg": distribution(parallax[triangulated]),
        "strict_parallax_deg": distribution(parallax[strict]),
        "broad_parallax_deg": distribution(parallax[broad]),
    }


def compare_density_arms(control: Mapping[str, Any], high: Mapping[str, Any]) -> dict:
    """Apply the mechanism gate fixed before inspecting the high-K payload."""

    def ratio(key: str) -> float:
        return float(high[key]) / max(float(control[key]), 1.0)

    def stat_ratio(group: str, statistic: str) -> float:
        numerator = high[group][statistic]
        denominator = control[group][statistic]
        if numerator is None or denominator is None:
            return float("nan")
        return float(numerator) / max(float(denominator), 1e-12)

    ratios = {
        "raw_edges": ratio("raw_reciprocal_epipolar_edge_count"),
        "accepted_edges": ratio("accepted_edge_count"),
        "tracks": ratio("track_count"),
        "triangulated_tracks": ratio("triangulated_track_count"),
        "strict_tracks": ratio("strict_track_count"),
        "broad_tracks": ratio("broad_track_count"),
        "broad_covariance_median": stat_ratio("broad_covariance_trace_m2", "median"),
        "broad_covariance_p90": stat_ratio("broad_covariance_trace_m2", "p90"),
        "broad_parallax_median": stat_ratio("broad_parallax_deg", "median"),
    }
    checks = {
        "accepted_edges_at_least_1p25x": ratios["accepted_edges"] >= 1.25,
        "triangulated_tracks_at_least_1p10x": ratios["triangulated_tracks"] >= 1.10,
        "broad_tracks_at_least_1p10x": ratios["broad_tracks"] >= 1.10,
        "strict_tracks_at_least_1p05x": ratios["strict_tracks"] >= 1.05,
        "broad_covariance_median_at_most_1p10x": (
            ratios["broad_covariance_median"] <= 1.10
        ),
        "broad_covariance_p90_at_most_1p15x": (ratios["broad_covariance_p90"] <= 1.15),
        "broad_parallax_median_at_least_0p90x": (
            ratios["broad_parallax_median"] >= 0.90
        ),
    }
    passed = all(bool(value) for value in checks.values()) and all(
        math.isfinite(value) for value in ratios.values()
    )
    return {
        "gate_defined_before_high_payload_inspection": True,
        "ratios_high_over_control": ratios,
        "checks": checks,
        "mechanism_gate_passed": passed,
        "decision": (
            "go_to_complete_single_factor_pipeline"
            if passed
            else "stop_before_complete_pose_pipeline"
        ),
    }
