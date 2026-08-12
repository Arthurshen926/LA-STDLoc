#!/usr/bin/env python3
"""CPU-only paired postmortem for a frozen descriptor-only mapping replay.

The ordinary mapping-pose gate intentionally emits only aggregate summaries.
This audit reconstructs the missing row/query sidecar without changing the
map, solver, query subset, or test split.  It is deliberately specialized to a
pair whose anchor registry and mapping geometry are bitwise identical.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from localization.localizer import load_shared_metric
from localization.pose_solver import pose_error, solve_absolute_pose
from map_learning.trainer import _pose_error_cm, _project_errors
from topology.pose_information import (
    fisher_contributions,
    pose_jacobian_analytic,
    task_scaled_pose_jacobian,
    translation_schur_complement,
)


LOCKED_BATCH_QUERIES = 8
LOCKED_CPU_THREADS = 32
BASELINE_CROSS_DEVICE_MEAN_TOLERANCE = 5e-5
SOLVER_PROTOCOL = {
    "solver": "poselib_absolute_pose",
    "confidence": 0.99999,
    "max_iterations": 100000,
    "min_iterations": 1000,
    "progressive_sampling": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _membership(
    record: dict, prefix: str, winners: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.as_tensor(record[f"{prefix}_offsets"]).long()
    indices = torch.as_tensor(record[f"{prefix}_indices"]).long()
    winners = torch.as_tensor(winners).long().reshape(-1)
    counts = offsets[1:] - offsets[:-1]
    row_ids = torch.repeat_interleave(torch.arange(winners.numel()), counts)
    matched = torch.zeros(winners.numel(), dtype=torch.bool)
    if indices.numel():
        hits = indices == winners[row_ids]
        if bool(hits.any()):
            matched[row_ids[hits]] = True
    return matched, counts > 0


def _positive_type_eligibility(
    record: dict, anchor_type: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.as_tensor(record["positive_offsets"]).long()
    indices = torch.as_tensor(record["positive_indices"]).long()
    counts = offsets[1:] - offsets[:-1]
    row_ids = torch.repeat_interleave(torch.arange(counts.numel()), counts)
    reserve = torch.zeros(counts.numel(), dtype=torch.bool)
    track = torch.zeros(counts.numel(), dtype=torch.bool)
    if indices.numel():
        is_track = anchor_type[indices] != 0
        if bool(is_track.any()):
            track[row_ids[is_track]] = True
        if bool((~is_track).any()):
            reserve[row_ids[~is_track]] = True
    return track, reserve


def _row_state(correct: np.ndarray, ambiguous: np.ndarray, positive: np.ndarray) -> np.ndarray:
    state = np.full(correct.shape, "false", dtype="<U12")
    state[~positive] = "no_positive"
    state[ambiguous & ~correct] = "ambiguous"
    state[correct] = "correct"
    return state


def _convex_hull_fraction(points: np.ndarray, width: float, height: float) -> float:
    if points.shape[0] < 3 or width <= 0 or height <= 0:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    return float(cv2.contourArea(hull) / (width * height))


def _geometry_metrics(
    *,
    selected_rows: np.ndarray,
    winners: np.ndarray,
    keypoints: torch.Tensor,
    xyz: torch.Tensor,
    intrinsic: torch.Tensor,
    gt_pose: torch.Tensor,
    task_translation_m: float,
    task_rotation_deg: float,
    variance_px2: float,
) -> dict[str, Any]:
    rows = np.asarray(selected_rows, dtype=np.int64).reshape(-1)
    output: dict[str, Any] = {"count": int(rows.size)}
    if not rows.size:
        return output
    selected_winners = torch.as_tensor(winners[rows]).long()
    selected_xyz = xyz[selected_winners].double()
    selected_keypoints = keypoints[torch.as_tensor(rows).long()].double()
    unique, duplicate_counts = torch.unique(selected_winners, return_counts=True)
    width = float(intrinsic[0, 2]) * 2.0
    height = float(intrinsic[1, 2]) * 2.0
    xy = selected_keypoints.numpy()
    grid_x = np.clip((xy[:, 0] / max(width, 1.0) * 8).astype(np.int64), 0, 7)
    grid_y = np.clip((xy[:, 1] / max(height, 1.0) * 6).astype(np.int64), 0, 5)
    output.update(
        {
            "unique_anchor_count": int(unique.numel()),
            "unique_anchor_fraction": float(unique.numel() / rows.size),
            "max_anchor_multiplicity": int(duplicate_counts.max()),
            "max_anchor_fraction": float(duplicate_counts.max() / rows.size),
            "image_hull_fraction": _convex_hull_fraction(xy, width, height),
            "image_grid_8x6_cells": int(np.unique(grid_y * 8 + grid_x).size),
        }
    )
    centered = selected_xyz - selected_xyz.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(rows.size - 1, 1)
    spatial_eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    output["xyz_covariance_eigenvalues_m2"] = [
        float(value) for value in spatial_eigenvalues
    ]
    output["xyz_covariance_smallest_m2"] = float(spatial_eigenvalues[0])
    output["xyz_covariance_condition"] = float(
        spatial_eigenvalues[-1] / spatial_eigenvalues[0].clamp_min(1e-12)
    )
    jacobian = task_scaled_pose_jacobian(
        pose_jacobian_analytic(selected_xyz, intrinsic.double(), gt_pose.double()),
        translation_scale=float(task_translation_m),
        rotation_scale=math.radians(float(task_rotation_deg)),
    )
    information = fisher_contributions(
        jacobian,
        measurement_covariance=torch.full(
            (rows.size,), float(variance_px2), dtype=torch.float64
        ),
    ).sum(dim=0)
    information = information + torch.eye(6, dtype=torch.float64) * 1e-4
    eigenvalues = torch.linalg.eigvalsh(information).clamp_min(1e-12)
    translation = translation_schur_complement(information).double()
    translation_eigenvalues = torch.linalg.eigvalsh(translation).clamp_min(1e-12)
    output.update(
        {
            "fisher_logdet": float(eigenvalues.log().sum()),
            "fisher_smallest_eigenvalue": float(eigenvalues[0]),
            "fisher_condition": float(eigenvalues[-1] / eigenvalues[0]),
            "translation_fisher_logdet": float(translation_eigenvalues.log().sum()),
            "translation_fisher_smallest_eigenvalue": float(
                translation_eigenvalues[0]
            ),
            "translation_fisher_condition": float(
                translation_eigenvalues[-1] / translation_eigenvalues[0]
            ),
        }
    )
    return output


def _arm_paths(args: argparse.Namespace, prefix: str) -> dict[str, Path]:
    return {
        role: Path(getattr(args, f"{prefix}_{role}")).resolve()
        for role in ("map", "metric", "teacher", "cache")
    }


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()


def _require_artifact_record(
    record: dict,
    *,
    path: Path,
    label: str,
) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"formal gate misses {label} artifact record")
    if not _same_path(record.get("path", ""), path):
        raise ValueError(f"formal gate {label} path does not match replay input")
    digest = _sha256(path)
    if record.get("sha256") != digest:
        raise ValueError(f"formal gate {label} SHA-256 does not match replay input")
    if record.get("expected_sha256_matches") is not True:
        raise ValueError(f"formal gate {label} was not checked against its expected SHA-256")
    if record.get("expected_sha256") != digest:
        raise ValueError(f"formal gate {label} expected SHA-256 is not the replay input")


def _load_locked_protocol(
    *,
    gate_path: Path,
    expected_gate_sha256: str,
    arm_paths: dict[str, dict[str, Path]],
    summary_roots: dict[str, Path],
    requested_query_count: int,
    requested_seeds: list[int],
) -> tuple[dict, dict[str, Path], dict[str, dict[str, dict]], list[int]]:
    if _sha256(gate_path) != str(expected_gate_sha256):
        raise ValueError("formal pose-gate SHA-256 does not match the explicit expectation")
    gate = json.loads(gate_path.read_text())
    if (
        gate.get("schema") != "lafgs_mapping_pose_pair_gate"
        or int(gate.get("version", 0)) != 1
        or gate.get("uses_test_queries") is not False
        or gate.get("valid") is not True
        or gate.get("decision", {}).get("verdict") != "STOP"
    ):
        raise ValueError("postmortem requires the valid mapping-only STOP pose gate")
    checks = gate.get("lineage", {}).get("checks", {})
    if not checks or any(value is not True for value in checks.values()):
        raise ValueError("formal pose gate has an incomplete lineage check")

    protocol = gate.get("preregistered_protocol", {})
    locked_query_count = int(protocol.get("query_count", -1))
    locked_seeds = [int(value) for value in protocol.get("seeds", [])]
    if int(protocol.get("deployment_row_limit", -1)) != 0:
        raise ValueError("postmortem requires the full mapping rows used by the pose gate")
    if int(requested_query_count) != locked_query_count:
        raise ValueError("query count differs from the formal pose gate")
    if [int(value) for value in requested_seeds] != locked_seeds:
        raise ValueError("seed list/order differs from the formal pose gate")

    gate_arms = gate.get("lineage", {}).get("arms", {})
    baseline_indices = [
        int(value)
        for value in gate_arms.get("baseline", {}).get("uniform_q256_indices", [])
    ]
    variant_indices = [
        int(value)
        for value in gate_arms.get("variant", {}).get("uniform_q256_indices", [])
    ]
    if (
        len(baseline_indices) != locked_query_count
        or baseline_indices != variant_indices
        or _json_sha256(baseline_indices)
        != gate_arms.get("baseline", {}).get("uniform_q256_indices_sha256")
        or _json_sha256(variant_indices)
        != gate_arms.get("variant", {}).get("uniform_q256_indices_sha256")
    ):
        raise ValueError("formal pose gate does not bind one exact paired query subset")

    inputs = gate.get("lineage", {}).get("inputs", {})
    calibration_paths: dict[str, Path] = {}
    source_summaries: dict[str, dict[str, dict]] = {
        "baseline": {},
        "candidate": {},
    }
    for local_arm, gate_arm in (("baseline", "baseline"), ("candidate", "variant")):
        for local_role, gate_role in (
            ("map", "map"),
            ("metric", "metric"),
            ("teacher", "teacher"),
            ("cache", "query_cache"),
        ):
            _require_artifact_record(
                inputs.get(f"{gate_arm}.{gate_role}"),
                path=arm_paths[local_arm][local_role],
                label=f"{gate_arm}.{gate_role}",
            )
        calibration_record = inputs.get(f"{gate_arm}.calibration")
        if not isinstance(calibration_record, dict):
            raise ValueError(f"formal gate misses {gate_arm}.calibration artifact record")
        calibration_path = Path(str(calibration_record.get("path", ""))).resolve()
        _require_artifact_record(
            calibration_record,
            path=calibration_path,
            label=f"{gate_arm}.calibration",
        )
        calibration_paths[local_arm] = calibration_path
        for seed in locked_seeds:
            summary_path = (
                summary_roots[local_arm]
                / f"seed{seed}"
                / "mapping_cache_summary.json"
            ).resolve()
            record = inputs.get(f"{gate_arm}.seed{seed}_summary")
            _require_artifact_record(
                record,
                path=summary_path,
                label=f"{gate_arm}.seed{seed}_summary",
            )
            source_summaries[local_arm][str(seed)] = json.loads(
                summary_path.read_text()
            )
    return gate, calibration_paths, source_summaries, baseline_indices


def _validate_calibration(
    calibration: dict,
    *,
    calibration_path: Path,
    cache_path: Path,
) -> None:
    sources = calibration.get("sources", {})
    if (
        calibration.get("schema") != "lafgs_mapping_only_scene_calibration"
        or int(calibration.get("version", 0)) < 2
        or sources.get("uses_test_queries") is not False
        or not _same_path(sources.get("query_cache", ""), cache_path)
    ):
        raise ValueError(f"calibration is not bound to the mapping-only cache: {calibration_path}")
    expected_cache_sha256 = sources.get("query_cache_sha256")
    if expected_cache_sha256 is not None and expected_cache_sha256 != _sha256(cache_path):
        raise ValueError(f"calibration query-cache SHA-256 mismatch: {calibration_path}")


def _validate_source_summary(
    report: dict,
    *,
    arm: str,
    seed: int,
    selected: list[int],
    paths: dict[str, Path],
    calibration_path: Path,
    expected_sha256: dict[str, str],
) -> None:
    protocol = report.get("evaluation_protocol", {})
    if (
        report.get("schema") != "lafgs_mapping_cache_evaluation"
        or int(report.get("version", 0)) != 2
        or report.get("uses_test_queries") is not False
        or int(report.get("seed", -1)) != int(seed)
        or int(report.get("query_count", -1)) != len(selected)
        or report.get("query_selection") != "uniform_mapping_gate"
        or protocol.get("split") != "mapping_only"
        or protocol.get("query_selection") != "uniform_mapping_gate"
        or int(protocol.get("requested_query_count", -1)) != len(selected)
        or int(protocol.get("evaluated_query_count", -1)) != len(selected)
        or int(protocol.get("deployment_row_limit", -1)) != 0
        or [int(value) for value in protocol.get("selected_query_indices", [])]
        != selected
        or protocol.get("selected_query_indices_sha256") != _json_sha256(selected)
    ):
        raise ValueError(f"{arm} seed {seed} summary is not the locked mapping-only replay")
    direct_paths = {
        "map": "map",
        "metric_state": "metric",
        "complete_positive_teacher": "teacher",
        "query_cache": "cache",
    }
    for field, role in direct_paths.items():
        if not _same_path(report.get(field, ""), paths[role]):
            raise ValueError(f"{arm} seed {seed} summary names a different {role}")
    if not _same_path(report.get("scene_calibration", ""), calibration_path):
        raise ValueError(f"{arm} seed {seed} summary names a different calibration")
    artifacts = report.get("artifacts", {})
    for summary_role, local_role in (
        ("map", "map"),
        ("metric", "metric"),
        ("teacher", "teacher"),
        ("query_cache", "cache"),
    ):
        record = artifacts.get(summary_role, {})
        if (
            not _same_path(record.get("path", ""), paths[local_role])
            or record.get("sha256") != expected_sha256[local_role]
        ):
            raise ValueError(f"{arm} seed {seed} summary artifact {summary_role} mismatch")
    calibration_record = artifacts.get("calibration", {})
    if (
        not _same_path(calibration_record.get("path", ""), calibration_path)
        or calibration_record.get("sha256") != expected_sha256["calibration"]
    ):
        raise ValueError(f"{arm} seed {seed} summary calibration artifact mismatch")


@torch.inference_mode()
def _match_arm(
    *,
    label: str,
    paths: dict[str, Path],
    selected_queries: list[int],
    batch_queries: int,
    common_geometry: dict[int, dict[str, torch.Tensor | str | float]] | None,
) -> tuple[dict, dict[int, dict[str, torch.Tensor | str | float]], dict]:
    state = torch.load(paths["map"], map_location="cpu", weights_only=False)
    teacher = torch.load(paths["teacher"], map_location="cpu", weights_only=False)
    cache_root = torch.load(paths["cache"], map_location="cpu", weights_only=False)
    cache = cache_root.get("queries", cache_root)
    names = list(teacher["query_names"])
    anchor_type = torch.as_tensor(state["anchor_type"]).long().cpu()
    bank = F.normalize(torch.as_tensor(state["anchor_features"]).float(), dim=1)
    metric = load_shared_metric(
        paths["metric"],
        anchor_ids=torch.as_tensor(state["anchor_ids"]).long(),
        device=torch.device("cpu"),
    )
    output: dict[int, dict[str, np.ndarray | str | int | float]] = {}
    geometry = {} if common_geometry is None else common_geometry
    for start in range(0, len(selected_queries), int(batch_queries)):
        batch = selected_queries[start : start + int(batch_queries)]
        descriptors = []
        lengths = []
        batch_records = []
        for query_index in batch:
            name = names[query_index]
            record = teacher["records"][query_index]
            cached = cache[name]
            rows = torch.as_tensor(record["query_rows"]).long()
            descriptor = F.normalize(
                torch.as_tensor(cached["native_descriptors"]).float()[rows], dim=1
            )
            descriptors.append(descriptor)
            lengths.append(int(rows.numel()))
            batch_records.append((query_index, name, record, cached, rows))
        adapted, _ = metric(torch.cat(descriptors, dim=0))
        scores = adapted @ bank.T
        top_scores, top_indices = torch.topk(scores, k=2, dim=1)
        del scores, adapted, descriptors
        cursor = 0
        for length, (query_index, name, record, cached, rows) in zip(
            lengths, batch_records
        ):
            winners = top_indices[cursor : cursor + length, 0].cpu()
            correct, has_positive = _membership(record, "positive", winners)
            ambiguous, _ = _membership(record, "ambiguous", winners)
            track_eligible, reserve_eligible = _positive_type_eligibility(
                record, anchor_type
            )
            arm_row = {
                "query_index": int(query_index),
                "image_name": name,
                "winners": winners.numpy().astype(np.int32, copy=False),
                "correct": correct.numpy(),
                "ambiguous": ambiguous.numpy(),
                "has_positive": has_positive.numpy(),
                "winner_type": anchor_type[winners].numpy().astype(np.int8),
                "track_eligible": track_eligible.numpy(),
                "reserve_eligible": reserve_eligible.numpy(),
                "top1_score": top_scores[cursor : cursor + length, 0].cpu().numpy(),
                "top1_margin": (
                    top_scores[cursor : cursor + length, 0]
                    - top_scores[cursor : cursor + length, 1]
                ).cpu().numpy(),
            }
            output[query_index] = arm_row
            candidate_geometry = {
                "image_name": name,
                "rows": rows.cpu(),
                "keypoints": (
                    torch.as_tensor(cached["native_keypoints"]).float()[rows]
                    + float(cached.get("pixel_center_offset", 0.5))
                ).cpu(),
                "intrinsic": torch.as_tensor(cached["native_K"]).float().cpu(),
                "gt_pose": torch.as_tensor(cached["pose_w2c"]).float().cpu(),
                "pixel_center_offset": float(cached.get("pixel_center_offset", 0.5)),
            }
            if common_geometry is None:
                geometry[query_index] = candidate_geometry
            else:
                reference = geometry[query_index]
                for key in ("rows", "keypoints", "intrinsic", "gt_pose"):
                    if not torch.equal(reference[key], candidate_geometry[key]):
                        raise ValueError(
                            f"{label} query geometry differs at {query_index}:{key}"
                        )
                if reference["image_name"] != name:
                    raise ValueError(f"{label} query name differs at {query_index}")
            cursor += length
        print(
            json.dumps(
                {
                    "event": "postmortem_matching",
                    "arm": label,
                    "queries_complete": min(start + int(batch_queries), len(selected_queries)),
                    "query_count": len(selected_queries),
                }
            ),
            flush=True,
        )
        del top_scores, top_indices
    metadata = {
        "paths": {
            role: {"path": str(path), "sha256": _sha256(path)}
            for role, path in paths.items()
        },
        "anchor_count": int(torch.as_tensor(state["anchor_ids"]).numel()),
        "anchor_ids_sha256": _json_sha256(
            torch.as_tensor(state["anchor_ids"]).long().tolist()
        ),
        "xyz": torch.as_tensor(state["anchor_xyz"]).float().cpu(),
        "anchor_type": anchor_type,
        "teacher": teacher,
    }
    del cache, cache_root, metric, bank, state
    gc.collect()
    return output, geometry, metadata


def _counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {str(key): int(value) for key, value in zip(unique, counts)}


def _matching_report(
    baseline: dict[int, dict], candidate: dict[int, dict], selected: list[int]
) -> tuple[dict, dict[int, dict]]:
    states = ("correct", "false", "ambiguous", "no_positive")
    transition = {left: {right: 0 for right in states} for left in states}
    winner_type_transition = {
        "reserve": {"reserve": 0, "track": 0},
        "track": {"reserve": 0, "track": 0},
    }
    per_query = {}
    all_baseline_correct = 0
    all_candidate_correct = 0
    all_rows = 0
    all_changed = 0
    arm_type = {
        "baseline": {"reserve": {"wins": 0, "correct": 0}, "track": {"wins": 0, "correct": 0}},
        "candidate": {"reserve": {"wins": 0, "correct": 0}, "track": {"wins": 0, "correct": 0}},
    }
    changed_by_candidate_type = {
        "reserve": {"count": 0, "correct_gain": 0, "correct_loss": 0},
        "track": {"count": 0, "correct_gain": 0, "correct_loss": 0},
    }
    for query_index in selected:
        left = baseline[query_index]
        right = candidate[query_index]
        b_correct = np.asarray(left["correct"], dtype=bool)
        c_correct = np.asarray(right["correct"], dtype=bool)
        b_winner = np.asarray(left["winners"])
        c_winner = np.asarray(right["winners"])
        changed = b_winner != c_winner
        b_state = _row_state(
            b_correct, np.asarray(left["ambiguous"]), np.asarray(left["has_positive"])
        )
        c_state = _row_state(
            c_correct, np.asarray(right["ambiguous"]), np.asarray(right["has_positive"])
        )
        for source in states:
            for target in states:
                transition[source][target] += int(np.count_nonzero((b_state == source) & (c_state == target)))
        b_kind = np.where(np.asarray(left["winner_type"]) == 0, "reserve", "track")
        c_kind = np.where(np.asarray(right["winner_type"]) == 0, "reserve", "track")
        for source in ("reserve", "track"):
            for target in ("reserve", "track"):
                winner_type_transition[source][target] += int(
                    np.count_nonzero((b_kind == source) & (c_kind == target))
                )
        for arm, row, kind in (
            ("baseline", left, b_kind), ("candidate", right, c_kind)
        ):
            correct = np.asarray(row["correct"], dtype=bool)
            for label in ("reserve", "track"):
                mask = kind == label
                arm_type[arm][label]["wins"] += int(mask.sum())
                arm_type[arm][label]["correct"] += int((mask & correct).sum())
        for label in ("reserve", "track"):
            mask = changed & (c_kind == label)
            changed_by_candidate_type[label]["count"] += int(mask.sum())
            changed_by_candidate_type[label]["correct_gain"] += int(
                (mask & ~b_correct & c_correct).sum()
            )
            changed_by_candidate_type[label]["correct_loss"] += int(
                (mask & b_correct & ~c_correct).sum()
            )
        row_count = int(b_correct.size)
        per_query[query_index] = {
            "query_index": int(query_index),
            "image_name": left["image_name"],
            "row_count": row_count,
            "changed_top1_count": int(changed.sum()),
            "changed_top1_fraction": float(changed.mean()),
            "baseline_correct_count": int(b_correct.sum()),
            "candidate_correct_count": int(c_correct.sum()),
            "correct_delta": int(c_correct.sum() - b_correct.sum()),
            "wrong_to_correct": int((~b_correct & c_correct).sum()),
            "correct_to_wrong": int((b_correct & ~c_correct).sum()),
            "baseline_unique_winners": int(np.unique(b_winner).size),
            "candidate_unique_winners": int(np.unique(c_winner).size),
            "baseline_margin_mean": float(np.asarray(left["top1_margin"]).mean()),
            "candidate_margin_mean": float(np.asarray(right["top1_margin"]).mean()),
        }
        all_rows += row_count
        all_changed += int(changed.sum())
        all_baseline_correct += int(b_correct.sum())
        all_candidate_correct += int(c_correct.sum())
    for arm in arm_type.values():
        for kind in arm.values():
            kind["precision_percent"] = float(
                100.0 * kind["correct"] / max(kind["wins"], 1)
            )
    return {
        "row_count": all_rows,
        "changed_top1_count": all_changed,
        "changed_top1_fraction": float(all_changed / max(all_rows, 1)),
        "baseline_correct_count": all_baseline_correct,
        "candidate_correct_count": all_candidate_correct,
        "net_correct_gain": all_candidate_correct - all_baseline_correct,
        "baseline_raw_precision_percent": 100.0 * all_baseline_correct / all_rows,
        "candidate_raw_precision_percent": 100.0 * all_candidate_correct / all_rows,
        "raw_precision_delta_pp": 100.0 * (all_candidate_correct - all_baseline_correct) / all_rows,
        "correctness_transition": transition,
        "winner_type_transition": winner_type_transition,
        "winner_type_breakdown": arm_type,
        "changed_rows_by_candidate_winner_type": changed_by_candidate_type,
    }, per_query


def _pose_one(
    *,
    winners: np.ndarray,
    correct: np.ndarray,
    winner_type: np.ndarray,
    changed: np.ndarray,
    geometry: dict,
    xyz: torch.Tensor,
    seed: int,
    ransac_px: float,
    clean_px: float,
    task_translation_m: float,
    task_rotation_deg: float,
) -> tuple[dict, np.ndarray, np.ndarray]:
    keypoints = geometry["keypoints"]
    intrinsic = geometry["intrinsic"]
    gt_pose = geometry["gt_pose"]
    estimate = solve_absolute_pose(
        keypoints.numpy(),
        xyz[torch.as_tensor(winners).long()].numpy(),
        intrinsic.numpy(),
        reprojection_error_px=float(ransac_px),
        confidence=float(SOLVER_PROTOCOL["confidence"]),
        max_iterations=int(SOLVER_PROTOCOL["max_iterations"]),
        min_iterations=int(SOLVER_PROTOCOL["min_iterations"]),
        seed=int(seed),
        progressive_sampling=bool(SOLVER_PROTOCOL["progressive_sampling"]),
    )
    inliers = np.asarray(estimate.inliers, dtype=np.int64).reshape(-1)
    clean = np.zeros(inliers.size, dtype=bool)
    if inliers.size:
        errors = _project_errors(
            xyz[torch.as_tensor(winners[inliers]).long()],
            keypoints[torch.as_tensor(inliers).long()],
            intrinsic,
            gt_pose,
        )
        clean = (errors <= float(clean_px)).numpy()
    ae_deg, _ = pose_error(estimate.pose_w2c, gt_pose.numpy())
    te_cm = _pose_error_cm(estimate.pose_w2c, gt_pose)
    correct_rows = np.flatnonzero(correct)
    clean_rows = inliers[clean]
    harmful_rows = inliers[~clean]
    result = {
        "te_cm": float(te_cm),
        "ae_deg": float(ae_deg),
        "inlier_count": int(inliers.size),
        "clean_inlier_count": int(clean.sum()),
        "harmful_inlier_count": int((~clean).sum()),
        "inlier_gt_precision_percent": float(100.0 * clean.sum() / max(inliers.size, 1)),
        "correct_top1_inlier_count": int(correct[inliers].sum()) if inliers.size else 0,
        "changed_top1_inlier_count": int(changed[inliers].sum()) if inliers.size else 0,
        "changed_top1_clean_inlier_count": int(changed[clean_rows].sum()) if clean_rows.size else 0,
        "changed_top1_harmful_inlier_count": int(changed[harmful_rows].sum()) if harmful_rows.size else 0,
        "reserve_clean_inlier_count": int((winner_type[clean_rows] == 0).sum()) if clean_rows.size else 0,
        "track_clean_inlier_count": int((winner_type[clean_rows] != 0).sum()) if clean_rows.size else 0,
        "reserve_harmful_inlier_count": int((winner_type[harmful_rows] == 0).sum()) if harmful_rows.size else 0,
        "track_harmful_inlier_count": int((winner_type[harmful_rows] != 0).sum()) if harmful_rows.size else 0,
        "hypotheses": int(estimate.diagnostics.get("iterations", 0)),
        "correct_top1_geometry": _geometry_metrics(
            selected_rows=correct_rows,
            winners=winners,
            keypoints=keypoints,
            xyz=xyz,
            intrinsic=intrinsic,
            gt_pose=gt_pose,
            task_translation_m=task_translation_m,
            task_rotation_deg=task_rotation_deg,
            variance_px2=max(float(clean_px), 0.5) ** 2,
        ),
        "clean_inlier_geometry": _geometry_metrics(
            selected_rows=clean_rows,
            winners=winners,
            keypoints=keypoints,
            xyz=xyz,
            intrinsic=intrinsic,
            gt_pose=gt_pose,
            task_translation_m=task_translation_m,
            task_rotation_deg=task_rotation_deg,
            variance_px2=max(float(clean_px), 0.5) ** 2,
        ),
        "solver_inlier_geometry": _geometry_metrics(
            selected_rows=inliers,
            winners=winners,
            keypoints=keypoints,
            xyz=xyz,
            intrinsic=intrinsic,
            gt_pose=gt_pose,
            task_translation_m=task_translation_m,
            task_rotation_deg=task_rotation_deg,
            variance_px2=max(float(ransac_px), 0.5) ** 2,
        ),
    }
    return result, inliers, clean


def _numeric_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
    }


def _pose_summary(
    rows: list[dict],
    *,
    raw_count: int,
    correct_count: int,
) -> dict[str, float | int]:
    te = np.asarray([row["te_cm"] for row in rows], dtype=np.float64)
    ae = np.asarray([row["ae_deg"] for row in rows], dtype=np.float64)
    tail_count = max(int(math.ceil(0.05 * te.size)), 1)
    inlier_count = int(sum(row["inlier_count"] for row in rows))
    clean_inlier_count = int(sum(row["clean_inlier_count"] for row in rows))
    harmful_inlier_count = int(sum(row["harmful_inlier_count"] for row in rows))
    return {
        "query_count": int(te.size),
        "median_te_cm": float(np.median(te)),
        "mean_te_cm": float(te.mean()),
        "p90_te_cm": float(np.percentile(te, 90)),
        "p95_te_cm": float(np.percentile(te, 95)),
        "p99_te_cm": float(np.percentile(te, 99)),
        "cvar95_te_cm": float(np.sort(te)[-tail_count:].mean()),
        "median_ae_deg": float(np.median(ae)),
        "mean_ae_deg": float(ae.mean()),
        "p90_ae_deg": float(np.percentile(ae, 90)),
        "p95_ae_deg": float(np.percentile(ae, 95)),
        "recall_5cm_5deg_percent": float(100.0 * np.mean((te < 5.0) & (ae < 5.0))),
        "catastrophic_100cm_count": int(np.count_nonzero(te >= 100.0)),
        "raw_gt_precision_percent": float(100.0 * correct_count / max(raw_count, 1)),
        "inlier_gt_precision_percent": float(
            100.0 * clean_inlier_count / max(inlier_count, 1)
        ),
        "solver_inlier_ratio_percent": float(
            100.0 * inlier_count / max(raw_count, 1)
        ),
        "retained_matches_mean": float(raw_count / max(te.size, 1)),
        "inlier_count": inlier_count,
        "clean_inlier_count": clean_inlier_count,
        "harmful_inlier_count": harmful_inlier_count,
        "mean_hypotheses": float(np.mean([row["hypotheses"] for row in rows])),
    }


def _validate_reproduction(
    *,
    arm: str,
    seed: int,
    computed: dict,
    original: dict,
) -> dict[str, float]:
    missing = sorted(set(original) - set(computed))
    if missing:
        raise ValueError(
            f"{arm} seed {seed} replay does not recompute summary fields: {missing}"
        )
    differences = {
        key: float(computed[key] - original[key])
        for key in sorted(original)
        if isinstance(computed[key], (int, float))
        and isinstance(original[key], (int, float))
    }
    nonzero = {key: abs(value) for key, value in differences.items() if value != 0.0}
    if arm == "candidate":
        if nonzero:
            raise ValueError(
                f"candidate seed {seed} does not exactly reproduce its source summary: {nonzero}"
            )
    else:
        permitted = {"mean_te_cm", "mean_ae_deg"}
        if set(nonzero) - permitted or any(
            value > BASELINE_CROSS_DEVICE_MEAN_TOLERANCE
            for value in nonzero.values()
        ):
            raise ValueError(
                f"baseline seed {seed} exceeds the bounded CPU/GPU replay difference: {nonzero}"
            )
    return differences


def _correlation(left: list[float], right: list[float]) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 2 or float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix in ("baseline", "candidate"):
        parser.add_argument(f"--{prefix}-map", required=True)
        parser.add_argument(f"--{prefix}-metric", required=True)
        parser.add_argument(f"--{prefix}-teacher", required=True)
        parser.add_argument(f"--{prefix}-cache", required=True)
        parser.add_argument(f"--{prefix}-summary-root", required=True)
    parser.add_argument("--formal-pose-gate", type=Path, required=True)
    parser.add_argument("--expected-formal-pose-gate-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-count", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    args = parser.parse_args()
    if torch.cuda.is_initialized():
        raise RuntimeError("this audit must run before any CUDA initialization")
    torch.set_num_threads(LOCKED_CPU_THREADS)
    arm_paths = {
        arm: _arm_paths(args, arm) for arm in ("baseline", "candidate")
    }
    summary_roots = {
        arm: Path(getattr(args, f"{arm}_summary_root")).resolve()
        for arm in ("baseline", "candidate")
    }
    gate_path = args.formal_pose_gate.resolve()
    gate, calibration_paths, source_summary_reports, selected = _load_locked_protocol(
        gate_path=gate_path,
        expected_gate_sha256=args.expected_formal_pose_gate_sha256,
        arm_paths=arm_paths,
        summary_roots=summary_roots,
        requested_query_count=args.query_count,
        requested_seeds=args.seeds,
    )
    calibrations = {
        arm: json.loads(path.read_text())
        for arm, path in calibration_paths.items()
    }
    for arm in ("baseline", "candidate"):
        _validate_calibration(
            calibrations[arm],
            calibration_path=calibration_paths[arm],
            cache_path=arm_paths[arm]["cache"],
        )
    for numeric_scope in ("parameters", "policy", "statistics"):
        if calibrations["baseline"].get(numeric_scope) != calibrations["candidate"].get(
            numeric_scope
        ):
            raise ValueError(f"paired calibrations differ in {numeric_scope}")
    parameters = calibrations["baseline"]["parameters"]
    gate_inputs = gate["lineage"]["inputs"]
    for arm, gate_arm in (("baseline", "baseline"), ("candidate", "variant")):
        expected_sha256 = {
            "map": gate_inputs[f"{gate_arm}.map"]["sha256"],
            "metric": gate_inputs[f"{gate_arm}.metric"]["sha256"],
            "teacher": gate_inputs[f"{gate_arm}.teacher"]["sha256"],
            "cache": gate_inputs[f"{gate_arm}.query_cache"]["sha256"],
            "calibration": gate_inputs[f"{gate_arm}.calibration"]["sha256"],
        }
        for seed in args.seeds:
            _validate_source_summary(
                source_summary_reports[arm][str(seed)],
                arm=arm,
                seed=int(seed),
                selected=selected,
                paths=arm_paths[arm],
                calibration_path=calibration_paths[arm],
                expected_sha256=expected_sha256,
            )

    baseline_teacher_header = torch.load(
        arm_paths["baseline"]["teacher"], map_location="cpu", weights_only=False
    )
    total_queries = len(baseline_teacher_header["records"])
    computed_selected = (
        torch.linspace(0, total_queries - 1, steps=int(args.query_count))
        .round()
        .long()
        .unique(sorted=True)
        .tolist()
    )
    if computed_selected != selected:
        raise ValueError("teacher does not reproduce the gate-bound uniform query subset")
    del baseline_teacher_header
    baseline, geometry, baseline_meta = _match_arm(
        label="baseline",
        paths=arm_paths["baseline"],
        selected_queries=selected,
        batch_queries=LOCKED_BATCH_QUERIES,
        common_geometry=None,
    )
    candidate, geometry, candidate_meta = _match_arm(
        label="candidate",
        paths=arm_paths["candidate"],
        selected_queries=selected,
        batch_queries=LOCKED_BATCH_QUERIES,
        common_geometry=geometry,
    )
    if not torch.equal(baseline_meta["xyz"], candidate_meta["xyz"]):
        raise ValueError("paired maps do not have bitwise-equal anchor xyz")
    if not torch.equal(baseline_meta["anchor_type"], candidate_meta["anchor_type"]):
        raise ValueError("paired maps do not have bitwise-equal anchor types")
    for query_index in selected:
        left = baseline_meta["teacher"]["records"][query_index]
        right = candidate_meta["teacher"]["records"][query_index]
        for key in (
            "query_rows", "positive_offsets", "positive_indices",
            "ambiguous_offsets", "ambiguous_indices",
        ):
            if not torch.equal(torch.as_tensor(left[key]), torch.as_tensor(right[key])):
                raise ValueError(f"teacher pair differs at {query_index}:{key}")

    matching, per_query = _matching_report(baseline, candidate, selected)
    sidecar: dict[str, np.ndarray] = {
        "selected_query_indices": np.asarray(selected, dtype=np.int32),
        "query_row_offsets": np.asarray(
            [0] + list(np.cumsum([baseline[index]["winners"].size for index in selected])),
            dtype=np.int64,
        ),
    }
    for label, arm in (("baseline", baseline), ("candidate", candidate)):
        for key in ("winners", "correct", "ambiguous", "has_positive", "winner_type", "top1_score", "top1_margin"):
            sidecar[f"{label}_{key}"] = np.concatenate(
                [np.asarray(arm[index][key]) for index in selected]
            )

    xyz = baseline_meta["xyz"]
    pose_results: dict[str, dict[str, list[dict]]] = {}
    tail_sets: dict[str, dict[str, list[int]]] = {"baseline": {}, "candidate": {}}
    original_summaries: dict[str, dict[str, dict]] = {"baseline": {}, "candidate": {}}
    reproduction: dict[str, dict[str, dict]] = {"baseline": {}, "candidate": {}}
    seed_correlations = {}
    for seed in args.seeds:
        seed_key = str(seed)
        pose_results[seed_key] = {"baseline": [], "candidate": []}
        for arm_label, arm in (("baseline", baseline), ("candidate", candidate)):
            inlier_offsets = [0]
            clean_offsets = [0]
            inlier_values = []
            clean_values = []
            for completed, query_index in enumerate(selected, start=1):
                row = arm[query_index]
                changed = baseline[query_index]["winners"] != candidate[query_index]["winners"]
                result, inliers, clean = _pose_one(
                    winners=row["winners"],
                    correct=row["correct"],
                    winner_type=row["winner_type"],
                    changed=changed,
                    geometry=geometry[query_index],
                    xyz=xyz,
                    seed=int(seed),
                    ransac_px=float(parameters["ransac_reprojection_px"]),
                    clean_px=float(parameters["clean_radius_px"]),
                    task_translation_m=float(parameters["task_translation_m"]),
                    task_rotation_deg=float(parameters["task_rotation_deg"]),
                )
                result.update(
                    {"query_index": int(query_index), "image_name": row["image_name"]}
                )
                pose_results[seed_key][arm_label].append(result)
                inlier_values.append(inliers.astype(np.int32, copy=False))
                clean_values.append(clean.astype(np.uint8, copy=False))
                inlier_offsets.append(inlier_offsets[-1] + inliers.size)
                clean_offsets.append(clean_offsets[-1] + clean.size)
                if completed % 25 == 0 or completed == len(selected):
                    print(
                        json.dumps(
                            {
                                "event": "postmortem_pose",
                                "arm": arm_label,
                                "seed": int(seed),
                                "queries_complete": completed,
                                "query_count": len(selected),
                            }
                        ),
                        flush=True,
                    )
            sidecar[f"{arm_label}_seed{seed}_inlier_offsets"] = np.asarray(inlier_offsets, dtype=np.int64)
            sidecar[f"{arm_label}_seed{seed}_inlier_indices"] = np.concatenate(inlier_values).astype(np.int32, copy=False)
            sidecar[f"{arm_label}_seed{seed}_inlier_clean"] = np.concatenate(clean_values).astype(np.uint8, copy=False)
            computed = _pose_summary(
                pose_results[seed_key][arm_label],
                raw_count=int(matching["row_count"]),
                correct_count=int(matching[f"{arm_label}_correct_count"]),
            )
            original_path = (
                summary_roots[arm_label]
                / f"seed{seed}"
                / "mapping_cache_summary.json"
            ).resolve()
            original = source_summary_reports[arm_label][seed_key]["summary"]
            original_summaries[arm_label][seed_key] = {
                "path": str(original_path.resolve()), "sha256": _sha256(original_path), "summary": original
            }
            differences = _validate_reproduction(
                arm=arm_label,
                seed=int(seed),
                computed=computed,
                original=original,
            )
            reproduction[arm_label][seed_key] = {
                "computed": computed,
                "absolute_differences_from_gpu_gate_summary": {
                    key: abs(value) for key, value in differences.items()
                },
                "max_absolute_difference": max([abs(value) for value in differences.values()] or [0.0]),
            }
            errors = np.asarray([row["te_cm"] for row in pose_results[seed_key][arm_label]])
            tail_count = max(int(math.ceil(0.05 * errors.size)), 1)
            tail_local = np.argsort(errors, kind="stable")[-tail_count:][::-1]
            tail_sets[arm_label][seed_key] = [int(selected[index]) for index in tail_local]

        baseline_rows = pose_results[seed_key]["baseline"]
        candidate_rows = pose_results[seed_key]["candidate"]
        te_delta = [right["te_cm"] - left["te_cm"] for left, right in zip(baseline_rows, candidate_rows)]
        seed_correlations[seed_key] = {
            "te_delta_vs_correct_count_delta_pearson": _correlation(
                [per_query[index]["correct_delta"] for index in selected], te_delta
            ),
            "te_delta_vs_changed_top1_fraction_pearson": _correlation(
                [per_query[index]["changed_top1_fraction"] for index in selected], te_delta
            ),
            "te_delta_vs_clean_inlier_delta_pearson": _correlation(
                [right["clean_inlier_count"] - left["clean_inlier_count"] for left, right in zip(baseline_rows, candidate_rows)], te_delta
            ),
            "te_delta_vs_harmful_inlier_delta_pearson": _correlation(
                [right["harmful_inlier_count"] - left["harmful_inlier_count"] for left, right in zip(baseline_rows, candidate_rows)], te_delta
            ),
            "te_delta_vs_clean_translation_fisher_logdet_delta_pearson": _correlation(
                [
                    right["clean_inlier_geometry"].get("translation_fisher_logdet", 0.0)
                    - left["clean_inlier_geometry"].get("translation_fisher_logdet", 0.0)
                    for left, right in zip(baseline_rows, candidate_rows)
                ],
                te_delta,
            ),
        }

    tail_consistency = {}
    for arm_label in ("baseline", "candidate"):
        sets = [set(tail_sets[arm_label][str(seed)]) for seed in args.seeds]
        intersection = set.intersection(*sets)
        union = set.union(*sets)
        tail_consistency[arm_label] = {
            "tail_count_per_seed": len(sets[0]),
            "by_seed": tail_sets[arm_label],
            "intersection": sorted(intersection),
            "union": sorted(union),
            "intersection_fraction": float(len(intersection) / max(len(sets[0]), 1)),
            "union_count": len(union),
        }

    tail_query_indices = sorted(
        set(tail_consistency["baseline"]["union"])
        | set(tail_consistency["candidate"]["union"])
        | {
            selected[index]
            for seed in args.seeds
            for index, (left, right) in enumerate(
                zip(pose_results[str(seed)]["baseline"], pose_results[str(seed)]["candidate"])
            )
            if (left["te_cm"] < 5.0) != (right["te_cm"] < 5.0)
        }
    )
    pose_by_seed_index = {
        str(seed): {
            arm: {row["query_index"]: row for row in pose_results[str(seed)][arm]}
            for arm in ("baseline", "candidate")
        }
        for seed in args.seeds
    }
    tail_queries = []
    for query_index in tail_query_indices:
        row = dict(per_query[query_index])
        row["seeds"] = {}
        for seed in args.seeds:
            key = str(seed)
            left = pose_by_seed_index[key]["baseline"][query_index]
            right = pose_by_seed_index[key]["candidate"][query_index]
            row["seeds"][key] = {
                "baseline": left,
                "candidate": right,
                "delta_candidate_minus_baseline": {
                    "te_cm": right["te_cm"] - left["te_cm"],
                    "ae_deg": right["ae_deg"] - left["ae_deg"],
                    "inlier_count": right["inlier_count"] - left["inlier_count"],
                    "clean_inlier_count": right["clean_inlier_count"] - left["clean_inlier_count"],
                    "harmful_inlier_count": right["harmful_inlier_count"] - left["harmful_inlier_count"],
                },
            }
        tail_queries.append(row)

    aggregate_geometry = {}
    for arm in ("baseline", "candidate"):
        aggregate_geometry[arm] = {}
        for seed in args.seeds:
            rows = pose_results[str(seed)][arm]
            aggregate_geometry[arm][str(seed)] = {
                scope: {
                    metric: _numeric_summary(
                        [row[scope][metric] for row in rows if metric in row[scope]]
                    )
                    for metric in (
                        "count", "unique_anchor_fraction", "max_anchor_fraction",
                        "image_hull_fraction", "image_grid_8x6_cells",
                        "xyz_covariance_smallest_m2", "fisher_logdet",
                        "fisher_smallest_eigenvalue", "translation_fisher_logdet",
                        "translation_fisher_smallest_eigenvalue",
                    )
                }
                for scope in ("correct_top1_geometry", "clean_inlier_geometry", "solver_inlier_geometry")
            }

    args.output.mkdir(parents=True, exist_ok=True)
    sidecar_path = args.output / "paired_row_pose_sidecar.npz"
    np.savez_compressed(sidecar_path, **sidecar)
    report = {
        "schema": "lafgs_equal_energy_pose_postmortem",
        "version": 2,
        "uses_test_queries": False,
        "device": "cpu",
        "scope": "mapping_only_q256_frozen_pair",
        "protocol": {
            "query_count": len(selected),
            "selected_query_indices": selected,
            "selected_query_indices_sha256": _json_sha256(selected),
            "seeds": [int(seed) for seed in args.seeds],
            "batch_queries": LOCKED_BATCH_QUERIES,
            "cpu_threads": LOCKED_CPU_THREADS,
            "matching": "same shared-metric/global-top1 implementation as gate, CPU replay",
            "pose": "same PoseLib parameters and seeds as gate, CPU replay",
            "solver": {
                **SOLVER_PROTOCOL,
                "ransac_reprojection_px": float(
                    parameters["ransac_reprojection_px"]
                ),
                "clean_radius_px": float(parameters["clean_radius_px"]),
            },
            "baseline_cross_device_mean_tolerance": BASELINE_CROSS_DEVICE_MEAN_TOLERANCE,
            "primary_artifacts_modified": False,
        },
        "lineage": {
            "baseline": {key: value for key, value in baseline_meta.items() if key not in ("xyz", "anchor_type", "teacher")},
            "candidate": {key: value for key, value in candidate_meta.items() if key not in ("xyz", "anchor_type", "teacher")},
            "anchor_xyz_bitwise_equal": True,
            "anchor_type_bitwise_equal": True,
            "selected_teacher_rows_and_labels_bitwise_equal": True,
            "selected_query_geometry_bitwise_equal": True,
            "formal_pose_gate": {
                "path": str(gate_path),
                "sha256": _sha256(gate_path),
            },
            "calibrations": {
                arm: {"path": str(path), "sha256": _sha256(path)}
                for arm, path in calibration_paths.items()
            },
            "audit_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
        },
        "source_gate_summaries": original_summaries,
        "cpu_reproduction": reproduction,
        "matching": matching,
        "per_query_matching": [per_query[index] for index in selected],
        "pose": pose_results,
        "seed_correlations": seed_correlations,
        "tail_consistency": tail_consistency,
        "tail_queries": tail_queries,
        "aggregate_geometry": aggregate_geometry,
        "missing_primary_sidecar": {
            "original_gate_outputs_contained_only_aggregate_summaries": True,
            "minimum_needed": [
                "per-query top1 winner indices/correctness",
                "per-seed pose errors and solver inlier row indices",
            ],
            "reconstructed_sidecar": str(sidecar_path),
        },
    }
    report_path = args.output / "postmortem_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    sidecar_sha = _sha256(sidecar_path)
    report["missing_primary_sidecar"]["reconstructed_sidecar_sha256"] = sidecar_sha
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(report_path), "report_sha256": _sha256(report_path), "sidecar": str(sidecar_path), "sidecar_sha256": sidecar_sha}, indent=2))


if __name__ == "__main__":
    main()
