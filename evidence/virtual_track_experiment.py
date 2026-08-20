"""Contracts for the bounded virtual-render Track closed-loop experiment."""

from __future__ import annotations

import torch


DRY_RUN_THRESHOLDS = {
    "selected_view_count": 8,
    "median_detector_rows": 512,
    "median_supported_detector_rows": 256,
    "raw_track_count": 50,
    "stable_broad_track_count": 20,
    "new_anchor_count": 20,
    "distinct_view_bins": 2,
}


def enforce_one_observation_per_family(
    tracks: dict, pose_family: torch.Tensor
) -> tuple[dict, dict]:
    """Keep at most one deterministic highest-confidence row per Track/family.

    This is a hard evidence-independence contract, not a selector heuristic.
    Track IDs are compacted after filtering so triangulation receives a closed
    registry and cannot accidentally count siblings as independent views.
    """

    family = torch.as_tensor(pose_family, dtype=torch.long).reshape(-1)
    track = torch.as_tensor(tracks["track_index"], dtype=torch.long).reshape(-1)
    query = torch.as_tensor(tracks["query_index"], dtype=torch.long).reshape(-1)
    keypoint = torch.as_tensor(tracks["keypoint_index"], dtype=torch.long).reshape(-1)
    confidence = torch.as_tensor(tracks["confidence"], dtype=torch.float32).reshape(-1)
    if not (track.numel() == query.numel() == keypoint.numel() == confidence.numel()):
        raise ValueError("Track observation columns must align")
    if query.numel() and (int(query.min()) < 0 or int(query.max()) >= family.numel()):
        raise ValueError("Track query row exceeds pose-family registry")
    keep = []
    for track_id in torch.unique(track, sorted=True).tolist():
        rows = torch.nonzero(track == int(track_id), as_tuple=False).reshape(-1)
        for family_id in torch.unique(family[query[rows]], sorted=True).tolist():
            candidates = rows[family[query[rows]] == int(family_id)]
            rank = torch.argsort(confidence[candidates], descending=True, stable=True)
            keep.append(int(candidates[rank[0]]))
    keep = torch.tensor(sorted(keep), dtype=torch.long)
    retained_track = track[keep]
    unique_track = torch.unique(retained_track, sorted=True)
    remap = torch.full(
        (int(track.max()) + 1 if track.numel() else 0,), -1, dtype=torch.long
    )
    if unique_track.numel():
        remap[unique_track] = torch.arange(unique_track.numel())
    level = torch.as_tensor(tracks.get("track_level", torch.ones(
        int(track.max()) + 1 if track.numel() else 0, dtype=torch.int8
    )), dtype=torch.int8)
    result = {
        "track_index": remap[retained_track],
        "query_index": query[keep],
        "keypoint_index": keypoint[keep],
        "confidence": confidence[keep],
        "track_level": level[unique_track],
    }
    _, compact_family = torch.unique(family, sorted=True, return_inverse=True)
    pair = (
        result["track_index"] * max(int(torch.unique(family).numel()), 1)
        + compact_family[result["query_index"]]
    )
    if pair.unique().numel() != pair.numel():
        raise AssertionError("family observation contract was not enforced")
    return result, {
        "input_track_count": int(torch.unique(track).numel()),
        "retained_track_count": int(unique_track.numel()),
        "input_observation_count": int(track.numel()),
        "retained_observation_count": int(keep.numel()),
        "duplicate_family_observation_count": int(track.numel() - keep.numel()),
        "maximum_observations_per_track_family": 1,
    }


def dry_run_passes(metrics: dict) -> tuple[bool, list[str]]:
    failures = []
    exact = {
        "selected_view_count": DRY_RUN_THRESHOLDS["selected_view_count"],
    }
    minima = {
        key: value for key, value in DRY_RUN_THRESHOLDS.items() if key not in exact
    }
    for key, expected in exact.items():
        if int(metrics.get(key, -1)) != int(expected):
            failures.append(f"{key}!={expected}")
    for key, minimum in minima.items():
        if float(metrics.get(key, float("-inf"))) < float(minimum):
            failures.append(f"{key}<{minimum}")
    if metrics.get("family_contract_passed") is not True:
        failures.append("family_contract_passed!=true")
    if metrics.get("gt_visible_diagnostic", "missing") is not None:
        failures.append("gt_visible_diagnostic!=null")
    return not failures, failures
