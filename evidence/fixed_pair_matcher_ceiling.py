"""P9 fixed-pair MNN/LighterGlue probe and mapping-only Pair-Gate metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
import re
from typing import Callable

import torch
import torch.nn.functional as F

from common.hashing import canonical_json
from map_learning.fixed_pair_matcher_ceiling import (
    ALPHA_THRESHOLD,
    FEATURE_CACHE_SCHEMA,
    LIGHTGLUE_CONFIG,
    LIGHTGLUE_CONFIG_SHA256,
    pair_table_sha256,
    preregistration,
    tensor_sha256,
    validate_feature_cache,
)


PROBE_SCHEMA = "lafgs_p9_fixed_pair_match_probe"
PROBE_VERSION = 1
COMPLETION_SCHEMA = "lafgs_p9_fixed_pair_match_probe_completion"
COMPLETION_VERSION = 1
PAIR_GATE_SCHEMA = "lafgs_p9_fixed_pair_matcher_ceiling_pair_gate"
PAIR_GATE_VERSION = 1
ARM_NAMES = ("mnn_control", "lighterglue_variant")
EPIPOLAR_THRESHOLD_PX = 2.0
TEACHER_THRESHOLD_PX = 2.0
PIXEL_CENTER_OFFSET = 0.5
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

PAIR_GATE_REQUIRED_KEYS = {
    "schema",
    "version",
    "scene",
    "mapping_only",
    "uses_test_queries",
    "valid",
    "policy",
    "inputs",
    "parent_stairs_gate",
    "compiled_identity",
    "producer_identity",
    "control",
    "variant",
    "comparisons",
    "gates",
    "scene_pair_gate_passed",
    "requires_other_scene",
    "advance_to_track_implementation_review",
    "authorizes_real_track_run",
    "advance_to_pose",
    "authorizes_test",
    "changes_method_default",
    "decision",
}


def canonical_pairs(
    pairs: Sequence[Sequence[int]], *, query_count: int
) -> list[tuple[int, int]]:
    result = [(int(pair[0]), int(pair[1])) for pair in pairs]
    if (
        not result
        or result != sorted(set(result))
        or any(
            left < 0 or left >= right or right >= query_count for left, right in result
        )
    ):
        raise ValueError("P9 fixed pair table must be sorted, unique and in range")
    return result


def mutual_nearest_neighbor(
    descriptor0: torch.Tensor,
    descriptor1: torch.Tensor,
    *,
    minimum_cosine: float = -1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact MNN control without any hidden positive cosine threshold."""
    if float(minimum_cosine) != -1.0:
        raise ValueError("P9 MNN minimum cosine is frozen at -1")
    left = F.normalize(torch.as_tensor(descriptor0).cpu().float(), dim=1)
    right = F.normalize(torch.as_tensor(descriptor1).cpu().float(), dim=1)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1:] != right.shape[1:]:
        raise ValueError("P9 MNN descriptor tables are incompatible")
    if left.shape[0] == 0 or right.shape[0] == 0:
        empty_index = torch.empty(0, dtype=torch.long)
        return empty_index, empty_index.clone(), torch.empty(0)
    similarity = left @ right.T
    target = similarity.argmax(dim=1)
    reverse = similarity.argmax(dim=0)
    source = torch.arange(left.shape[0], dtype=torch.long)
    keep = reverse[target] == source
    source = source[keep]
    target = target[keep]
    confidence = similarity[source, target]
    return source.contiguous(), target.contiguous(), confidence.contiguous()


def direct_lighterglue_match(
    matcher: torch.nn.Module,
    left: Mapping,
    right: Mapping,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Run one direct matcher forward over already-materialized E1 rows."""
    descriptor0 = torch.as_tensor(left["descriptor"]).cpu().float()
    descriptor1 = torch.as_tensor(right["descriptor"]).cpu().float()
    keypoints0 = torch.as_tensor(left["native_xy"]).cpu().float()
    keypoints1 = torch.as_tensor(right["native_xy"]).cpu().float()
    if descriptor0.shape[0] == 0 or descriptor1.shape[0] == 0:
        raise ValueError("P9 LighterGlue requires nonempty fixed feature rows")
    height0, width0 = (int(value) for value in left["native_input_hw"])
    height1, width1 = (int(value) for value in right["native_input_hw"])
    data = {
        "image0": {
            "keypoints": keypoints0[None],
            "descriptors": descriptor0[None],
            "image_size": torch.tensor([[width0, height0]], dtype=torch.float32),
        },
        "image1": {
            "keypoints": keypoints1[None],
            "descriptors": descriptor1[None],
            "image_size": torch.tensor([[width1, height1]], dtype=torch.float32),
        },
    }
    with torch.inference_mode():
        output = matcher(data)
    if not isinstance(output, Mapping) or set(("matches", "scores")) - set(output):
        raise ValueError("P9 direct LighterGlue output is incomplete")
    matches = torch.as_tensor(output["matches"][0]).detach().cpu().long()
    score = torch.as_tensor(output["scores"][0]).detach().cpu().float().reshape(-1)
    if matches.ndim != 2 or matches.shape[1] != 2 or score.numel() != matches.shape[0]:
        raise ValueError("P9 direct LighterGlue compact matches are misaligned")
    source = matches[:, 0].contiguous()
    target = matches[:, 1].contiguous()
    if (
        source.unique().numel() != source.numel()
        or target.unique().numel() != target.numel()
    ):
        raise ValueError("P9 LighterGlue output is not reciprocal one-to-one")
    return (
        source,
        target,
        score.contiguous(),
        {
            "direct_forward_count": 1,
            "network_layers_executed": int(output.get("stop", -1)),
        },
    )


def _pose_3x4(record: Mapping) -> torch.Tensor:
    pose = torch.as_tensor(record["pose_w2c"]).double()
    if pose.shape == (4, 4):
        pose = pose[:3]
    if pose.shape != (3, 4):
        raise ValueError("P9 pose must be 3x4 or 4x4")
    return pose


def _fundamental(left: Mapping, right: Mapping) -> torch.Tensor:
    pose0, pose1 = _pose_3x4(left), _pose_3x4(right)
    rotation = pose1[:, :3] @ pose0[:, :3].T
    translation = pose1[:, 3] - rotation @ pose0[:, 3]
    tx, ty, tz = translation.unbind()
    zero = translation.new_zeros(())
    skew = torch.stack(
        (
            torch.stack((zero, -tz, ty)),
            torch.stack((tz, zero, -tx)),
            torch.stack((-ty, tx, zero)),
        )
    )
    essential = skew @ rotation
    k0 = torch.as_tensor(left["native_K"]).double()
    k1 = torch.as_tensor(right["native_K"]).double()
    return torch.linalg.inv(k1).T @ essential @ torch.linalg.inv(k0)


def symmetric_epipolar_error(
    left: Mapping,
    right: Mapping,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
) -> torch.Tensor:
    """Maximum of the two point-to-epipolar-line distances in pixels."""
    source = torch.as_tensor(source_index).long().reshape(-1)
    target = torch.as_tensor(target_index).long().reshape(-1)
    if source.numel() != target.numel():
        raise ValueError("P9 epipolar match columns are misaligned")
    if source.numel() == 0:
        return torch.empty(0, dtype=torch.float64)
    xy0 = torch.as_tensor(left["native_xy"]).double()[source] + PIXEL_CENTER_OFFSET
    xy1 = torch.as_tensor(right["native_xy"]).double()[target] + PIXEL_CENTER_OFFSET
    ones = torch.ones((source.numel(), 1), dtype=torch.float64)
    point0, point1 = torch.cat([xy0, ones], 1), torch.cat([xy1, ones], 1)
    fundamental = _fundamental(left, right)
    line1 = point0 @ fundamental.T
    line0 = point1 @ fundamental
    numerator = (point1 * line1).sum(1).abs()
    distance1 = numerator / torch.linalg.norm(line1[:, :2], dim=1).clamp_min(1e-12)
    distance0 = numerator / torch.linalg.norm(line0[:, :2], dim=1).clamp_min(1e-12)
    error = torch.maximum(distance0, distance1)
    if not bool(torch.isfinite(error).all()):
        raise ValueError("P9 epipolar error is non-finite")
    return error.contiguous()


def _sample_teacher(record: Mapping, indices: torch.Tensor) -> tuple[torch.Tensor, ...]:
    xy = torch.as_tensor(record["native_xy"]).double()[indices]
    depth = torch.as_tensor(record["native_depth_resampled"]).double().squeeze()
    alpha = torch.as_tensor(record["native_alpha_resampled"]).double().squeeze()
    mask = torch.as_tensor(record["native_valid_mask"]).bool().squeeze()
    rounded = xy.round().long()
    height, width = depth.shape
    inside = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    safe_x = rounded[:, 0].clamp(0, width - 1)
    safe_y = rounded[:, 1].clamp(0, height - 1)
    sampled_depth = depth[safe_y, safe_x]
    sampled_alpha = alpha[safe_y, safe_x]
    sampled_mask = mask[safe_y, safe_x]
    valid = (
        inside
        & sampled_mask
        & torch.isfinite(sampled_depth)
        & (sampled_depth > 0)
        & torch.isfinite(sampled_alpha)
        & (sampled_alpha >= ALPHA_THRESHOLD)
    )
    return xy + PIXEL_CENTER_OFFSET, sampled_depth, valid


def _depth_reproject(
    source: Mapping,
    target: Mapping,
    source_xy: torch.Tensor,
    source_depth: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    k_source = torch.as_tensor(source["native_K"]).double()
    k_target = torch.as_tensor(target["native_K"]).double()
    pose_source = _pose_3x4(source)
    pose_target = _pose_3x4(target)
    homogeneous = torch.cat(
        [source_xy, torch.ones((source_xy.shape[0], 1), dtype=torch.float64)], 1
    )
    camera = (homogeneous @ torch.linalg.inv(k_source).T) * source_depth[:, None]
    world = (camera - pose_source[:, 3]) @ pose_source[:, :3]
    target_camera = world @ pose_target[:, :3].T + pose_target[:, 3]
    projected = target_camera @ k_target.T
    target_depth = target_camera[:, 2]
    uv = projected[:, :2] / target_depth[:, None].clamp_min(1e-12)
    return uv, target_depth


def dense_teacher_correctness(
    left: Mapping,
    right: Mapping,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return evaluable, correct, and maximum bidirectional errors."""
    source = torch.as_tensor(source_index).long().reshape(-1)
    target = torch.as_tensor(target_index).long().reshape(-1)
    if source.numel() == 0:
        return (
            torch.empty(0, dtype=torch.bool),
            torch.empty(0, dtype=torch.bool),
            torch.empty(0, dtype=torch.float64),
        )
    xy0, depth0, valid0 = _sample_teacher(left, source)
    xy1, depth1, valid1 = _sample_teacher(right, target)
    projected1, positive1 = _depth_reproject(left, right, xy0, depth0)
    projected0, positive0 = _depth_reproject(right, left, xy1, depth1)
    error01 = torch.linalg.norm(projected1 - xy1, dim=1)
    error10 = torch.linalg.norm(projected0 - xy0, dim=1)
    maximum = torch.maximum(error01, error10)
    evaluable = (
        valid0 & valid1 & torch.isfinite(maximum) & (positive0 > 0) & (positive1 > 0)
    )
    correct = evaluable & (maximum <= TEACHER_THRESHOLD_PX)
    return evaluable.contiguous(), correct.contiguous(), maximum.contiguous()


def _common_confidence(
    left: Mapping,
    right: Mapping,
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    score0 = torch.as_tensor(left["detector_score"]).double()[source].clamp_min(0)
    score1 = torch.as_tensor(right["detector_score"]).double()[target].clamp_min(0)
    return torch.sqrt(score0 * score1).contiguous()


def _pack_arm(
    *,
    name: str,
    pairs: list[tuple[int, int]],
    feature_queries: Sequence[Mapping],
    matcher: Callable[
        [Mapping, Mapping], tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]
    ],
    run_uuid: str,
    producer_identity: Mapping,
) -> dict:
    offsets = [0]
    source_columns, target_columns = [], []
    matcher_confidence, common_confidence, epipolar_error = [], [], []
    teacher_evaluable, teacher_correct, teacher_error = [], [], []
    diagnostics = {
        "raw_match_count": [],
        "epipolar_accepted_count": [],
        "teacher_evaluable_count": [],
        "teacher_correct_count": [],
        "direct_matcher_forward_count": [],
    }
    for left_index, right_index in pairs:
        left, right = feature_queries[left_index], feature_queries[right_index]
        source, target, raw_confidence, matcher_diagnostic = matcher(left, right)
        source = torch.as_tensor(source).long().cpu().reshape(-1)
        target = torch.as_tensor(target).long().cpu().reshape(-1)
        raw_confidence = torch.as_tensor(raw_confidence).float().cpu().reshape(-1)
        if (
            source.numel() != target.numel()
            or source.numel() != raw_confidence.numel()
            or source.unique().numel() != source.numel()
            or target.unique().numel() != target.numel()
            or not bool(torch.isfinite(raw_confidence).all())
        ):
            raise ValueError(f"P9 {name} raw match table is invalid")
        if source.numel() and (
            int(source.min()) < 0
            or int(source.max()) >= int(left["row_count"])
            or int(target.min()) < 0
            or int(target.max()) >= int(right["row_count"])
        ):
            raise ValueError(f"P9 {name} raw match index is out of range")
        order = torch.argsort(source, stable=True)
        source = source[order]
        target = target[order]
        raw_confidence = raw_confidence[order]
        error = symmetric_epipolar_error(left, right, source, target)
        keep = torch.isfinite(error) & (error <= EPIPOLAR_THRESHOLD_PX)
        source, target = source[keep], target[keep]
        raw_confidence, error = raw_confidence[keep], error[keep]
        common = _common_confidence(left, right, source, target)
        evaluable, correct, depth_error = dense_teacher_correctness(
            left, right, source, target
        )
        source_columns.append(source)
        target_columns.append(target)
        matcher_confidence.append(raw_confidence)
        common_confidence.append(common)
        epipolar_error.append(error)
        teacher_evaluable.append(evaluable)
        teacher_correct.append(correct)
        teacher_error.append(depth_error)
        offsets.append(offsets[-1] + int(source.numel()))
        diagnostics["raw_match_count"].append(int(keep.numel()))
        diagnostics["epipolar_accepted_count"].append(int(keep.sum()))
        diagnostics["teacher_evaluable_count"].append(int(evaluable.sum()))
        diagnostics["teacher_correct_count"].append(int(correct.sum()))
        diagnostics["direct_matcher_forward_count"].append(
            int(matcher_diagnostic.get("direct_forward_count", 0))
        )

    def concatenate(values: list[torch.Tensor], *, dtype: torch.dtype) -> torch.Tensor:
        return torch.cat(values).to(dtype) if values else torch.empty(0, dtype=dtype)

    return {
        "name": name,
        "run_uuid": run_uuid,
        "producer_identity": dict(producer_identity),
        "pair_count": len(pairs),
        "matches": {
            "offsets": torch.tensor(offsets, dtype=torch.long),
            "source_row": concatenate(source_columns, dtype=torch.long),
            "target_row": concatenate(target_columns, dtype=torch.long),
            "matcher_confidence": concatenate(matcher_confidence, dtype=torch.float32),
            "common_confidence": concatenate(common_confidence, dtype=torch.float64),
            "symmetric_epipolar_error_px": concatenate(
                epipolar_error, dtype=torch.float64
            ),
            "teacher_evaluable": concatenate(teacher_evaluable, dtype=torch.bool),
            "teacher_correct": concatenate(teacher_correct, dtype=torch.bool),
            "teacher_bidirectional_error_px": concatenate(
                teacher_error, dtype=torch.float64
            ),
        },
        "pair_diagnostics": {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in diagnostics.items()
        },
    }


def _arm_pair_matches(arm: Mapping, pairs: Sequence[tuple[int, int]]) -> dict:
    matches = arm["matches"]
    offsets = torch.as_tensor(matches["offsets"]).long()
    source = torch.as_tensor(matches["source_row"]).long()
    target = torch.as_tensor(matches["target_row"]).long()
    confidence = torch.as_tensor(matches["common_confidence"]).double()
    return {
        pair: (
            source[int(offsets[index]) : int(offsets[index + 1])],
            target[int(offsets[index]) : int(offsets[index + 1])],
            confidence[int(offsets[index]) : int(offsets[index + 1])],
        )
        for index, pair in enumerate(pairs)
    }


def _camera_triangles(pairs: Sequence[tuple[int, int]]) -> list[tuple[int, int, int]]:
    edge = set(pairs)
    neighbors: dict[int, set[int]] = defaultdict(set)
    for left, right in pairs:
        neighbors[left].add(right)
        neighbors[right].add(left)
    triangles = []
    for left in sorted(neighbors):
        right_neighbors = sorted(value for value in neighbors[left] if value > left)
        for offset, middle in enumerate(right_neighbors):
            for right in right_neighbors[offset + 1 :]:
                if (middle, right) in edge:
                    triangles.append((left, middle, right))
    return triangles


def _triangulate_and_verify(
    observations: Sequence[torch.Tensor],
    records: Sequence[Mapping],
) -> torch.Tensor:
    count = observations[0].shape[0]
    if count == 0:
        return torch.empty(0, dtype=torch.bool)
    matrices = [
        torch.as_tensor(record["native_K"]).double() @ _pose_3x4(record)
        for record in records
    ]
    rows = []
    for uv, matrix in zip(observations, matrices):
        physical = torch.as_tensor(uv).double() + PIXEL_CENTER_OFFSET
        rows.extend(
            [
                physical[:, 0, None] * matrix[2] - matrix[0],
                physical[:, 1, None] * matrix[2] - matrix[1],
            ]
        )
    design = torch.stack(rows, dim=1)
    _, _, vh = torch.linalg.svd(design)
    homogeneous = vh[:, -1]
    finite_w = homogeneous[:, 3].abs() > 1e-12
    safe_w = torch.where(
        finite_w,
        homogeneous[:, 3],
        torch.ones_like(homogeneous[:, 3]),
    )
    xyz = homogeneous[:, :3] / safe_w[:, None]
    valid = finite_w & torch.isfinite(xyz).all(1)
    maximum_error = torch.zeros(count, dtype=torch.float64)
    for uv, matrix in zip(observations, matrices):
        xyz_h = torch.cat([xyz, torch.ones((count, 1), dtype=torch.float64)], 1)
        projected = xyz_h @ matrix.T
        depth = projected[:, 2]
        projected_uv = projected[:, :2] / depth[:, None].clamp_min(1e-12)
        error = torch.linalg.norm(
            projected_uv - (torch.as_tensor(uv).double() + PIXEL_CENTER_OFFSET),
            dim=1,
        )
        maximum_error = torch.maximum(maximum_error, error)
        valid &= depth > 0
    return (
        valid & torch.isfinite(maximum_error) & (maximum_error <= EPIPOLAR_THRESHOLD_PX)
    )


def cycle_metrics(
    *,
    arm: Mapping,
    pairs: list[tuple[int, int]],
    feature_queries: Sequence[Mapping],
) -> dict:
    """Count exact three-edge row closures and graph identity conflicts."""
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    pair_matches = _arm_pair_matches(arm, pairs)
    verified_count = 0
    verified_cameras: set[int] = set()
    cycle_edges: set[tuple[int, int, int]] = set()
    for left, middle, right in _camera_triangles(pairs):
        lm_source, lm_target, _ = pair_matches[(left, middle)]
        lr_source, lr_target, _ = pair_matches[(left, right)]
        mr_source, mr_target, _ = pair_matches[(middle, right)]
        if not lm_source.numel() or not lr_source.numel() or not mr_source.numel():
            continue
        lr_by_left = {int(a): int(b) for a, b in zip(lr_source, lr_target)}
        mr_by_middle = {int(a): int(b) for a, b in zip(mr_source, mr_target)}
        candidates = []
        for lm_row, (left_row, middle_row) in enumerate(zip(lm_source, lm_target)):
            right_from_left = lr_by_left.get(int(left_row))
            right_from_middle = mr_by_middle.get(int(middle_row))
            if right_from_left is not None and right_from_left == right_from_middle:
                candidates.append(
                    (lm_row, int(left_row), int(middle_row), int(right_from_left))
                )
        if not candidates:
            continue
        left_rows = torch.tensor([value[1] for value in candidates], dtype=torch.long)
        middle_rows = torch.tensor([value[2] for value in candidates], dtype=torch.long)
        right_rows = torch.tensor([value[3] for value in candidates], dtype=torch.long)
        valid = _triangulate_and_verify(
            [
                torch.as_tensor(feature_queries[left]["native_xy"])[left_rows],
                torch.as_tensor(feature_queries[middle]["native_xy"])[middle_rows],
                torch.as_tensor(feature_queries[right]["native_xy"])[right_rows],
            ],
            [feature_queries[left], feature_queries[middle], feature_queries[right]],
        )
        for local in torch.nonzero(valid, as_tuple=False).reshape(-1).tolist():
            left_row = int(left_rows[local])
            middle_row = int(middle_rows[local])
            right_row = int(right_rows[local])
            verified_count += 1
            verified_cameras.update((left, middle, right))
            cycle_edges.add((pair_index[(left, middle)], left_row, middle_row))
            cycle_edges.add((pair_index[(left, right)], left_row, right_row))
            cycle_edges.add((pair_index[(middle, right)], middle_row, right_row))

    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(node: tuple[int, int]) -> tuple[int, int]:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left_node: tuple[int, int], right_node: tuple[int, int]) -> None:
        root_left, root_right = find(left_node), find(right_node)
        if root_left != root_right:
            parent[root_right] = root_left

    nonempty_pairs = 0
    for pair, (source, target, _) in pair_matches.items():
        if source.numel():
            nonempty_pairs += 1
        for source_row, target_row in zip(source.tolist(), target.tolist()):
            union((pair[0], int(source_row)), (pair[1], int(target_row)))
    components: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for node in parent:
        components[find(node)].append(node)
    conflict_count = 0
    conflict_components = 0
    for nodes in components.values():
        camera_counts: dict[int, int] = defaultdict(int)
        for camera, _ in nodes:
            camera_counts[camera] += 1
        conflicts = sum(max(count - 1, 0) for count in camera_counts.values())
        conflict_count += conflicts
        conflict_components += int(conflicts > 0)
    return {
        "verified_keypoint_triangle_count": verified_count,
        "cycle_supported_edge_count": len(cycle_edges),
        "verified_triangle_camera_indices": sorted(verified_cameras),
        "verified_triangle_camera_count": len(verified_cameras),
        "nonempty_pair_coverage_count": nonempty_pairs,
        "identity_conflict_count": conflict_count,
        "identity_conflict_component_count": conflict_components,
    }


def summarize_arm(
    *,
    arm: Mapping,
    pairs: list[tuple[int, int]],
    feature_queries: Sequence[Mapping],
) -> dict:
    matches = arm["matches"]
    diagnostics = arm["pair_diagnostics"]
    raw_count = int(torch.as_tensor(diagnostics["raw_match_count"]).sum())
    epipolar_count = int(torch.as_tensor(matches["source_row"]).numel())
    evaluable = torch.as_tensor(matches["teacher_evaluable"]).bool()
    correct = torch.as_tensor(matches["teacher_correct"]).bool()
    evaluable_count = int(evaluable.sum())
    correct_count = int(correct.sum())
    return {
        "raw_match_count": raw_count,
        "epipolar_accepted_count": epipolar_count,
        "epipolar_acceptance_rate": (
            float(epipolar_count / raw_count) if raw_count else 0.0
        ),
        "teacher_evaluable_count": evaluable_count,
        "correct_correspondence_count": correct_count,
        "correct_correspondence_precision": (
            float(correct_count / evaluable_count) if evaluable_count else 0.0
        ),
        **cycle_metrics(arm=arm, pairs=pairs, feature_queries=feature_queries),
    }


def _arm_tensor_hashes(arm: Mapping) -> dict:
    return {
        "matches": {
            name: tensor_sha256(torch.as_tensor(value))
            for name, value in arm["matches"].items()
        },
        "pair_diagnostics": {
            name: tensor_sha256(torch.as_tensor(value))
            for name, value in arm["pair_diagnostics"].items()
        },
    }


def probe_content_sha256(payload: Mapping) -> str:
    pair_table = payload.get("pair_table", {})
    summary = {
        key: value
        for key, value in payload.items()
        if key not in {"pair_table", "arms", "content_sha256"}
    }
    summary.update(
        {
            "pair_table": {
                key: value
                for key, value in pair_table.items()
                if key not in {"left_query_index", "right_query_index"}
            },
            "pair_table_tensor_sha256": {
                "left_query_index": tensor_sha256(
                    torch.as_tensor(pair_table.get("left_query_index"))
                ),
                "right_query_index": tensor_sha256(
                    torch.as_tensor(pair_table.get("right_query_index"))
                ),
            },
            "arms": {
                name: {
                    "name": payload["arms"][name].get("name"),
                    "run_uuid": payload["arms"][name].get("run_uuid"),
                    "producer_identity": payload["arms"][name].get("producer_identity"),
                    "pair_count": payload["arms"][name].get("pair_count"),
                    "matcher": payload["arms"][name].get("matcher"),
                    "tensor_sha256": payload["arms"][name].get("tensor_sha256"),
                    "metrics": payload["arms"][name].get("metrics"),
                }
                for name in ARM_NAMES
            },
        }
    )
    return hashlib.sha256(canonical_json(summary).encode("utf-8")).hexdigest()


def materialize_paired_probe(
    *,
    scene: str,
    feature_cache: Mapping,
    feature_cache_path: str,
    feature_cache_sha256: str,
    pairs: Sequence[Sequence[int]],
    proposal_lineage: Mapping,
    matcher: torch.nn.Module,
    matcher_identity: Mapping,
    run_uuid: str,
    producer_identity: Mapping,
    detector_sentinel: Callable[[], None] | None = None,
) -> dict:
    """Build both arms from the same E1 rows; detector code is unreachable."""
    validate_feature_cache(feature_cache, expected_scene=scene)
    if (
        run_uuid != feature_cache.get("run_uuid")
        or feature_cache.get("producer_identity", {}).get("compiled_identity")
        != producer_identity.get("compiled_identity")
        or SHA256_PATTERN.fullmatch(str(feature_cache_sha256)) is None
        or Path(str(feature_cache_path)).name != "p9_fixed_pair_feature_cache.pt"
    ):
        raise ValueError("P9 feature/probe UUID, producer, or lineage differs")
    pair_list = canonical_pairs(pairs, query_count=feature_cache["query_count"])
    if (
        pair_table_sha256(pair_list) != proposal_lineage.get("pair_table_sha256")
        or proposal_lineage.get("arm") != "nearest"
        or proposal_lineage.get("match_rows_reused") is not False
        or any(
            SHA256_PATTERN.fullmatch(str(proposal_lineage.get(field, ""))) is None
            for field in ("sha256", "content_sha256", "pair_table_sha256")
        )
    ):
        raise ValueError("P9 fixed pair table differs from proposal lineage")
    if detector_sentinel is not None:
        # The sentinel is an assertion token supplied by orchestration/tests,
        # never a detector callback. Calling it would prove re-entry.
        if not getattr(detector_sentinel, "p9_forbidden_detector_sentinel", False):
            raise ValueError("P9 detector sentinel is not armed")
    feature_queries = [
        feature_cache["queries"][name] for name in feature_cache["query_names"]
    ]

    def mnn(left: Mapping, right: Mapping):
        source, target, score = mutual_nearest_neighbor(
            left["descriptor"], right["descriptor"], minimum_cosine=-1.0
        )
        return source, target, score, {"direct_forward_count": 0}

    def lg(left: Mapping, right: Mapping):
        return direct_lighterglue_match(matcher, left, right)

    control = _pack_arm(
        name="mnn_control",
        pairs=pair_list,
        feature_queries=feature_queries,
        matcher=mnn,
        run_uuid=run_uuid,
        producer_identity=producer_identity,
    )
    variant = _pack_arm(
        name="lighterglue_variant",
        pairs=pair_list,
        feature_queries=feature_queries,
        matcher=lg,
        run_uuid=run_uuid,
        producer_identity=producer_identity,
    )
    control["matcher"] = {
        "name": "mutual_nearest_neighbor_cosine",
        "minimum_cosine": -1.0,
        "same_e1_rows": True,
    }
    variant["matcher"] = {
        "name": "bundled_lighterglue",
        "config": dict(LIGHTGLUE_CONFIG),
        "config_sha256": LIGHTGLUE_CONFIG_SHA256,
        "identity": dict(matcher_identity),
        "direct_forward_only": True,
    }
    for arm in (control, variant):
        arm["metrics"] = summarize_arm(
            arm=arm, pairs=pair_list, feature_queries=feature_queries
        )
        arm["tensor_sha256"] = _arm_tensor_hashes(arm)
    payload = {
        "schema": PROBE_SCHEMA,
        "version": PROBE_VERSION,
        "scene": scene,
        "mapping_only": True,
        "uses_test_queries": False,
        "run_uuid": run_uuid,
        "query_count": feature_cache["query_count"],
        "query_names_sha256": feature_cache["query_names_sha256"],
        "feature_cache": {
            "schema": FEATURE_CACHE_SCHEMA,
            "path": str(feature_cache_path),
            "sha256": str(feature_cache_sha256),
            "content_sha256": feature_cache["content_sha256"],
            "same_exact_rows_for_both_arms": True,
        },
        "pair_table": {
            **dict(proposal_lineage),
            "pair_count": len(pair_list),
            "left_query_index": torch.tensor([pair[0] for pair in pair_list]),
            "right_query_index": torch.tensor([pair[1] for pair in pair_list]),
        },
        "shared_contract": {
            "maximum_symmetric_epipolar_error_px": EPIPOLAR_THRESHOLD_PX,
            "common_confidence": "sqrt(left_detector_score*right_detector_score)",
            "minimum_common_confidence": 0.0,
            "dense_teacher": {
                "mapping_only": True,
                "pixel_center_offset": PIXEL_CENTER_OFFSET,
                "minimum_alpha": ALPHA_THRESHOLD,
                "correct_if_bidirectional_depth_reprojection_max_px": (
                    TEACHER_THRESHOLD_PX
                ),
            },
            "detector_reentry_forbidden": True,
            "detector_forward_count_during_pair_stage": 0,
            "p8_probe_rows_reused": False,
            "old_xfeat_arm_probe_rows_reused": False,
        },
        "producer_identity": dict(producer_identity),
        "arms": {
            "mnn_control": control,
            "lighterglue_variant": variant,
        },
    }
    payload["content_sha256"] = probe_content_sha256(payload)
    validate_paired_probe(
        payload,
        feature_cache=feature_cache,
        expected_scene=scene,
        expected_pairs=pair_list,
    )
    return payload


def validate_paired_probe(
    payload: Mapping,
    *,
    feature_cache: Mapping,
    expected_scene: str | None = None,
    expected_pairs: Sequence[Sequence[int]] | None = None,
    expected_content_sha256: str | None = None,
) -> dict:
    """Fail closed on row mutation, pair mutation, or arm splicing."""
    validate_feature_cache(feature_cache, expected_scene=expected_scene)
    schema_contract = preregistration()["artifact_schemas"]["paired_probe"]
    if (
        not set(schema_contract["required_top_level_keys"]).issubset(payload)
        or payload.get("schema") != PROBE_SCHEMA
        or payload.get("version") != PROBE_VERSION
        or payload.get("mapping_only") is not True
        or payload.get("uses_test_queries") is not False
        or payload.get("scene") != feature_cache.get("scene")
        or payload.get("query_count") != feature_cache.get("query_count")
        or payload.get("query_names_sha256") != feature_cache.get("query_names_sha256")
        or set(payload.get("arms", {})) != set(ARM_NAMES)
    ):
        raise ValueError("unexpected P9 paired-probe schema/scope")
    run_uuid = str(payload.get("run_uuid", ""))
    producer = payload.get("producer_identity")
    feature_lineage = payload.get("feature_cache", {})
    if (
        re.fullmatch(r"[0-9a-f]{32}", run_uuid) is None
        or run_uuid != feature_cache.get("run_uuid")
        or not isinstance(producer, Mapping)
        or SHA256_PATTERN.fullmatch(str(producer.get("compiled_identity", ""))) is None
        or producer.get("compiled_identity")
        != feature_cache.get("producer_identity", {}).get("compiled_identity")
        or feature_lineage.get("schema") != FEATURE_CACHE_SCHEMA
        or Path(str(feature_lineage.get("path", ""))).name
        != "p9_fixed_pair_feature_cache.pt"
        or SHA256_PATTERN.fullmatch(str(feature_lineage.get("sha256", ""))) is None
        or feature_lineage.get("content_sha256") != feature_cache.get("content_sha256")
        or feature_lineage.get("same_exact_rows_for_both_arms") is not True
    ):
        raise ValueError("P9 paired probe lacks run/producer identity")
    pair_table = payload.get("pair_table", {})
    left_value = torch.as_tensor(pair_table.get("left_query_index"))
    right_value = torch.as_tensor(pair_table.get("right_query_index"))
    if left_value.dtype != torch.int64 or right_value.dtype != torch.int64:
        raise ValueError("P9 paired-probe pair table dtype differs")
    left = left_value.reshape(-1)
    right = right_value.reshape(-1)
    pairs = canonical_pairs(
        list(zip(left.tolist(), right.tolist())),
        query_count=int(payload["query_count"]),
    )
    if (
        pair_table_sha256(pairs) != pair_table.get("pair_table_sha256")
        or int(pair_table.get("pair_count", -1)) != len(pairs)
        or pair_table.get("arm") != "nearest"
        or pair_table.get("match_rows_reused") is not False
        or any(
            SHA256_PATTERN.fullmatch(str(pair_table.get(field, ""))) is None
            for field in ("sha256", "content_sha256", "pair_table_sha256")
        )
        or (
            expected_pairs is not None
            and pairs
            != canonical_pairs(expected_pairs, query_count=int(payload["query_count"]))
        )
    ):
        raise ValueError("P9 paired-probe fixed pair table is stale")
    shared = payload.get("shared_contract", {})
    if (
        shared.get("maximum_symmetric_epipolar_error_px") != EPIPOLAR_THRESHOLD_PX
        or shared.get("common_confidence")
        != "sqrt(left_detector_score*right_detector_score)"
        or shared.get("minimum_common_confidence") != 0.0
        or shared.get("dense_teacher", {}).get("mapping_only") is not True
        or shared.get("dense_teacher", {}).get("pixel_center_offset")
        != PIXEL_CENTER_OFFSET
        or shared.get("dense_teacher", {}).get("minimum_alpha") != ALPHA_THRESHOLD
        or shared.get("dense_teacher", {}).get(
            "correct_if_bidirectional_depth_reprojection_max_px"
        )
        != TEACHER_THRESHOLD_PX
        or shared.get("detector_reentry_forbidden") is not True
        or shared.get("detector_forward_count_during_pair_stage") != 0
        or shared.get("p8_probe_rows_reused") is not False
        or shared.get("old_xfeat_arm_probe_rows_reused") is not False
    ):
        raise ValueError("P9 paired-probe shared matcher contract differs")
    feature_queries = [
        feature_cache["queries"][name] for name in feature_cache["query_names"]
    ]
    for name in ARM_NAMES:
        arm = payload["arms"][name]
        if (
            not set(schema_contract["arm_required_keys"]).issubset(arm)
            or arm.get("name") != name
            or arm.get("run_uuid") != run_uuid
            or arm.get("producer_identity") != producer
            or arm.get("pair_count") != len(pairs)
        ):
            raise ValueError("P9 paired-probe arms were spliced or are partial")
        matches = arm.get("matches", {})
        expected_dtypes = {
            "offsets": torch.int64,
            "source_row": torch.int64,
            "target_row": torch.int64,
            "matcher_confidence": torch.float32,
            "common_confidence": torch.float64,
            "symmetric_epipolar_error_px": torch.float64,
            "teacher_evaluable": torch.bool,
            "teacher_correct": torch.bool,
            "teacher_bidirectional_error_px": torch.float64,
        }
        tensor_values = {
            key: torch.as_tensor(matches.get(key)) for key in expected_dtypes
        }
        if any(
            tensor_values[key].dtype != dtype for key, dtype in expected_dtypes.items()
        ):
            raise ValueError(f"P9 {name} match tensor dtype differs")
        offsets = tensor_values["offsets"].reshape(-1)
        columns = {
            key: tensor_values[key].reshape(-1)
            for key in expected_dtypes
            if key != "offsets"
        }
        match_count = columns["source_row"].numel()
        if (
            offsets.numel() != len(pairs) + 1
            or int(offsets[0]) != 0
            or int(offsets[-1]) != match_count
            or bool((offsets[1:] < offsets[:-1]).any())
            or any(value.numel() != match_count for value in columns.values())
            or not all(
                bool(torch.isfinite(columns[key]).all())
                for key in (
                    "matcher_confidence",
                    "common_confidence",
                    "symmetric_epipolar_error_px",
                    "teacher_bidirectional_error_px",
                )
            )
            or bool((columns["common_confidence"] < 0).any())
            or bool(
                (columns["symmetric_epipolar_error_px"] > EPIPOLAR_THRESHOLD_PX).any()
            )
            or bool((columns["teacher_correct"] & ~columns["teacher_evaluable"]).any())
        ):
            raise ValueError(f"P9 {name} flattened match columns are invalid")
        for pair_index, (pair_left, pair_right) in enumerate(pairs):
            begin, end = int(offsets[pair_index]), int(offsets[pair_index + 1])
            source = columns["source_row"][begin:end].long()
            target = columns["target_row"][begin:end].long()
            if source.numel() and (
                int(source.min()) < 0
                or int(source.max()) >= int(feature_queries[pair_left]["row_count"])
                or int(target.min()) < 0
                or int(target.max()) >= int(feature_queries[pair_right]["row_count"])
                or source.unique().numel() != source.numel()
                or target.unique().numel() != target.numel()
                or bool((source[1:] < source[:-1]).any())
            ):
                raise ValueError(f"P9 {name} match row is out of range/non-reciprocal")
        diagnostics = arm.get("pair_diagnostics", {})
        if set(diagnostics) != {
            "raw_match_count",
            "epipolar_accepted_count",
            "teacher_evaluable_count",
            "teacher_correct_count",
            "direct_matcher_forward_count",
        } or any(
            torch.as_tensor(value).numel() != len(pairs)
            for value in diagnostics.values()
        ):
            raise ValueError(f"P9 {name} diagnostics are partial")
        diagnostic_values = {
            key: torch.as_tensor(value) for key, value in diagnostics.items()
        }
        if any(value.dtype != torch.int64 for value in diagnostic_values.values()):
            raise ValueError(f"P9 {name} diagnostic tensor dtype differs")
        accepted_per_pair = offsets[1:] - offsets[:-1]
        if (
            not torch.equal(
                diagnostic_values["epipolar_accepted_count"], accepted_per_pair
            )
            or bool(
                (
                    diagnostic_values["raw_match_count"]
                    < diagnostic_values["epipolar_accepted_count"]
                ).any()
            )
            or not torch.equal(
                diagnostic_values["teacher_evaluable_count"],
                torch.stack(
                    [
                        columns["teacher_evaluable"][
                            int(offsets[index]) : int(offsets[index + 1])
                        ].sum()
                        for index in range(len(pairs))
                    ]
                ).long(),
            )
            or not torch.equal(
                diagnostic_values["teacher_correct_count"],
                torch.stack(
                    [
                        columns["teacher_correct"][
                            int(offsets[index]) : int(offsets[index + 1])
                        ].sum()
                        for index in range(len(pairs))
                    ]
                ).long(),
            )
            or (
                name == "mnn_control"
                and bool((diagnostic_values["direct_matcher_forward_count"] != 0).any())
            )
            or (
                name == "lighterglue_variant"
                and bool((diagnostic_values["direct_matcher_forward_count"] != 1).any())
            )
        ):
            raise ValueError(f"P9 {name} pair diagnostics are stale")
        matcher_contract = arm.get("matcher", {})
        if name == "mnn_control" and matcher_contract != {
            "name": "mutual_nearest_neighbor_cosine",
            "minimum_cosine": -1.0,
            "same_e1_rows": True,
        }:
            raise ValueError("P9 MNN control matcher contract differs")
        if name == "lighterglue_variant" and (
            matcher_contract.get("name") != "bundled_lighterglue"
            or matcher_contract.get("config") != LIGHTGLUE_CONFIG
            or matcher_contract.get("config_sha256") != LIGHTGLUE_CONFIG_SHA256
            or matcher_contract.get("direct_forward_only") is not True
        ):
            raise ValueError("P9 LighterGlue matcher contract differs")
        observed_hashes = _arm_tensor_hashes(arm)
        if arm.get("tensor_sha256") != observed_hashes:
            raise ValueError(f"P9 {name} scientific tensor hash is stale")
        metrics = summarize_arm(arm=arm, pairs=pairs, feature_queries=feature_queries)
        if arm.get("metrics") != metrics:
            raise ValueError(f"P9 {name} metrics are stale")
    content = probe_content_sha256(payload)
    if payload.get("content_sha256") != content or (
        expected_content_sha256 is not None and content != str(expected_content_sha256)
    ):
        raise ValueError("P9 paired-probe content SHA-256 is stale")
    return {
        "scene": payload["scene"],
        "pair_count": len(pairs),
        "content_sha256": content,
    }


def _pair_gate_axes(control: Mapping, variant: Mapping) -> dict[str, bool]:
    control_cameras = set(control["verified_triangle_camera_indices"])
    variant_cameras = set(variant["verified_triangle_camera_indices"])
    return {
        "correct_correspondence_count_not_lower": variant[
            "correct_correspondence_count"
        ]
        >= control["correct_correspondence_count"],
        "correct_correspondence_precision_not_lower": variant[
            "correct_correspondence_precision"
        ]
        >= control["correct_correspondence_precision"],
        "epipolar_accepted_count_not_lower": variant["epipolar_accepted_count"]
        >= control["epipolar_accepted_count"],
        "epipolar_acceptance_rate_not_lower": variant["epipolar_acceptance_rate"]
        >= control["epipolar_acceptance_rate"],
        "verified_keypoint_triangle_count_not_lower": variant[
            "verified_keypoint_triangle_count"
        ]
        >= control["verified_keypoint_triangle_count"],
        "cycle_supported_edge_count_not_lower": variant["cycle_supported_edge_count"]
        >= control["cycle_supported_edge_count"],
        "verified_triangle_camera_set_not_lower": control_cameras.issubset(
            variant_cameras
        ),
        "nonempty_pair_coverage_not_lower": variant["nonempty_pair_coverage_count"]
        >= control["nonempty_pair_coverage_count"],
        "identity_conflict_count_not_higher": variant["identity_conflict_count"]
        <= control["identity_conflict_count"],
        "at_least_one_primary_strict_gain": any(
            variant[name] > control[name]
            for name in (
                "correct_correspondence_count",
                "verified_keypoint_triangle_count",
                "cycle_supported_edge_count",
            )
        ),
    }


def _pair_gate_comparisons(control: Mapping, variant: Mapping) -> dict:
    def comparison(name: str) -> dict:
        before, after = control[name], variant[name]
        return {
            "control": before,
            "variant": after,
            "delta": after - before,
            "ratio": None if before == 0 else after / before,
        }

    return {
        name: comparison(name)
        for name in (
            "correct_correspondence_count",
            "correct_correspondence_precision",
            "epipolar_accepted_count",
            "epipolar_acceptance_rate",
            "verified_keypoint_triangle_count",
            "cycle_supported_edge_count",
            "verified_triangle_camera_count",
            "nonempty_pair_coverage_count",
            "identity_conflict_count",
        )
    }


def validate_pair_gate_report(
    payload: Mapping, *, expected_scene: str | None = None
) -> dict:
    """Reject edited metrics, decisions, parent scope, or downstream authority."""
    if (
        not PAIR_GATE_REQUIRED_KEYS.issubset(payload)
        or payload.get("schema") != PAIR_GATE_SCHEMA
        or payload.get("version") != PAIR_GATE_VERSION
        or payload.get("scene") not in {"stairs", "greatcourt"}
        or payload.get("mapping_only") is not True
        or payload.get("uses_test_queries") is not False
        or payload.get("valid") is not True
        or payload.get("advance_to_track_implementation_review") is not False
        or payload.get("authorizes_real_track_run") is not False
        or payload.get("advance_to_pose") is not False
        or payload.get("authorizes_test") is not False
        or payload.get("changes_method_default") is not False
        or not isinstance(payload.get("producer_identity"), Mapping)
        or SHA256_PATTERN.fullmatch(str(payload.get("compiled_identity", ""))) is None
        or payload.get("compiled_identity")
        != payload["producer_identity"].get("compiled_identity")
    ):
        raise ValueError("P9 scene Pair Gate is structurally invalid")
    if expected_scene is not None and payload["scene"] != expected_scene:
        raise ValueError("P9 scene Pair Gate names the wrong scene")
    control, variant = payload.get("control"), payload.get("variant")
    policy = payload.get("policy")
    inputs = payload.get("inputs")
    if (
        not isinstance(control, Mapping)
        or not isinstance(variant, Mapping)
        or not isinstance(policy, Mapping)
        or policy.get("control") != "mnn_control"
        or policy.get("variant") != "lighterglue_variant"
        or policy.get("same_extractor_rows") is not True
        or SHA256_PATTERN.fullmatch(str(policy.get("fixed_pair_table_sha256", "")))
        is None
        or not isinstance(inputs, Mapping)
        or set(inputs) != {"paired_probe", "completion"}
        or Path(str(inputs.get("paired_probe", {}).get("path", ""))).name
        != "fixed_pair_match_probe.pt"
        or Path(str(inputs.get("completion", {}).get("path", ""))).name
        != "paired_match_completion.json"
        or any(
            SHA256_PATTERN.fullmatch(str(value)) is None
            for value in (
                inputs.get("paired_probe", {}).get("sha256"),
                inputs.get("paired_probe", {}).get("content_sha256"),
                inputs.get("completion", {}).get("sha256"),
            )
        )
    ):
        raise ValueError("P9 scene Pair Gate lacks arm metrics")
    gates = _pair_gate_axes(control, variant)
    comparisons = _pair_gate_comparisons(control, variant)
    passed = all(gates.values())
    if (
        payload.get("gates") != gates
        or payload.get("comparisons") != comparisons
        or payload.get("scene_pair_gate_passed") is not passed
        or payload.get("requires_other_scene") is not passed
        or payload.get("decision")
        != (
            "SCENE_PAIR_PASS_REQUIRES_OTHER_SCENE"
            if passed
            else "STOP_FIXED_PAIR_MATCHER_CEILING"
        )
    ):
        raise ValueError("P9 scene Pair Gate metrics/decision are stale")
    parent = payload.get("parent_stairs_gate")
    if payload["scene"] == "stairs" and parent is not None:
        raise ValueError("P9 Stairs scene Pair Gate cannot have a parent")
    if payload["scene"] == "greatcourt" and (
        not isinstance(parent, Mapping)
        or parent.get("scientific_projection", {}).get("scene") != "stairs"
        or parent.get("scientific_projection", {}).get("scene_pair_gate_passed")
        is not True
        or parent.get("scientific_projection", {}).get("decision")
        != "SCENE_PAIR_PASS_REQUIRES_OTHER_SCENE"
    ):
        raise ValueError("P9 GreatCourt scene Pair Gate lacks its passing parent")
    return {"scene": payload["scene"], "scene_pair_gate_passed": passed}


def pair_gate_report(
    *,
    probe: Mapping,
    probe_path: str,
    probe_sha256: str,
    completion_path: str,
    completion_sha256: str,
    producer_identity: Mapping,
    compiled_identity: str,
    parent_stairs_gate: Mapping | None,
) -> dict:
    """Apply the exact preregistered scientific Pair Gate."""
    control = probe["arms"]["mnn_control"]["metrics"]
    variant = probe["arms"]["lighterglue_variant"]["metrics"]
    gates = _pair_gate_axes(control, variant)
    passed = all(gates.values())
    report = {
        "schema": PAIR_GATE_SCHEMA,
        "version": PAIR_GATE_VERSION,
        "scene": probe["scene"],
        "mapping_only": True,
        "uses_test_queries": False,
        "valid": True,
        "policy": {
            "control": "mnn_control",
            "variant": "lighterglue_variant",
            "same_extractor_rows": True,
            "fixed_pair_table_sha256": probe["pair_table"]["pair_table_sha256"],
        },
        "inputs": {
            "paired_probe": {
                "path": probe_path,
                "sha256": probe_sha256,
                "content_sha256": probe["content_sha256"],
            },
            "completion": {
                "path": completion_path,
                "sha256": completion_sha256,
            },
        },
        "parent_stairs_gate": (
            dict(parent_stairs_gate) if parent_stairs_gate is not None else None
        ),
        "compiled_identity": str(compiled_identity),
        "producer_identity": dict(producer_identity),
        "control": control,
        "variant": variant,
        "comparisons": _pair_gate_comparisons(control, variant),
        "gates": gates,
        "scene_pair_gate_passed": passed,
        "requires_other_scene": passed,
        "advance_to_track_implementation_review": False,
        "authorizes_real_track_run": False,
        "advance_to_pose": False,
        "authorizes_test": False,
        "changes_method_default": False,
        "decision": (
            "SCENE_PAIR_PASS_REQUIRES_OTHER_SCENE"
            if passed
            else "STOP_FIXED_PAIR_MATCHER_CEILING"
        ),
    }
    validate_pair_gate_report(report, expected_scene=probe["scene"])
    return report
