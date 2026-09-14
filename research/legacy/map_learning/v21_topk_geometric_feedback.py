"""Query-specific Top-K geometric feedback evaluated by exact PoseLib."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import re
import uuid

import torch
import torch.nn.functional as F

from localization.matcher import global_cosine_topk
from map_learning.v21_pose_feedback_transductive import (
    replay_pose_with_contract,
    validate_complete_cache_payloads,
)


SCHEMA = "lafgs_v21_topk_geometric_feedback_evaluation"
FINAL_SCHEMA = "lafgs_v21_topk_geometric_feedback_final_decision"
VERSION = 1
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def default_config() -> dict:
    """The single configuration selected on adaptation before control replay."""

    return {
        "topk": 64,
        "projection_gate_px": 8.0,
        "maximum_score_drop_from_top1": 0.1,
        "minimum_baseline_inlier_count_inclusive": 128,
        "maximum_baseline_inlier_count_exclusive": 256,
        "minimum_candidate_inlier_gain": 4,
        "preserve_all_baseline_inlier_correspondences": True,
        "iterations": 1,
        "candidate_selection": "minimum_reprojection_residual_then_descriptor_rank",
        "candidate_acceptance": "candidate_inlier_count_at_least_baseline_plus_gain",
    }


def validate_config(
    value: Mapping, *, allow_runtime_inlier_band: bool = False
) -> dict:
    config = dict(value)
    expected = default_config()
    if allow_runtime_inlier_band:
        for key in (
            "minimum_baseline_inlier_count_inclusive",
            "maximum_baseline_inlier_count_exclusive",
        ):
            expected.pop(key)
        comparable = {key: config.get(key) for key in expected}
        minimum = int(config.get("minimum_baseline_inlier_count_inclusive", -1))
        maximum = int(config.get("maximum_baseline_inlier_count_exclusive", -1))
        if (
            comparable != expected
            or minimum < 4
            or maximum < 0
            or (maximum and maximum <= minimum)
        ):
            raise ValueError("V21 Top-K runtime inlier band is invalid")
        return config
    if config != expected:
        raise ValueError("V21 Top-K geometric configuration is not the frozen arm")
    return config


def _source(value: object, *, label: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"V21 Top-K {label} source is missing")
    path = str(Path(str(value.get("path", ""))).expanduser().resolve())
    digest = str(value.get("sha256", ""))
    size = int(value.get("size_bytes", 0))
    if not path or SHA256_PATTERN.fullmatch(digest) is None or size <= 0:
        raise ValueError(f"V21 Top-K {label} source is invalid")
    return {"path": path, "sha256": digest, "size_bytes": size}


def _baseline_outcome(record: Mapping) -> dict:
    return {
        "pose_w2c": torch.as_tensor(record["baseline_pose_w2c"]).float().cpu(),
        "translation_error_cm": float(record["baseline_translation_error_cm"]),
        "rotation_error_deg": float(record["baseline_rotation_error_deg"]),
        "task_error": float(record["baseline_task_error"]),
        "r5_success": bool(record["baseline_r5"]),
        "inlier_count": int(record["baseline_inlier_count"]),
        "inlier_query_rows": torch.as_tensor(record["baseline_inliers"])
        .long()
        .cpu(),
    }


def select_topk_geometry_rows(
    *,
    keypoints: torch.Tensor,
    topk_anchor_rows: torch.Tensor,
    topk_scores: torch.Tensor,
    baseline_anchor_rows: torch.Tensor,
    baseline_scores: torch.Tensor,
    baseline_inlier_rows: torch.Tensor,
    anchor_xyz: torch.Tensor,
    intrinsic: torch.Tensor,
    baseline_pose_w2c: torch.Tensor,
    config: Mapping,
    allow_runtime_inlier_band: bool = False,
) -> dict:
    """Replace only baseline outliers by pose-consistent descriptor candidates."""

    cfg = validate_config(
        config, allow_runtime_inlier_band=allow_runtime_inlier_band
    )
    xy = torch.as_tensor(keypoints).float().cpu()
    candidates = torch.as_tensor(topk_anchor_rows).long().cpu()
    scores = torch.as_tensor(topk_scores).float().cpu()
    baseline = torch.as_tensor(baseline_anchor_rows).long().cpu().reshape(-1)
    base_scores = torch.as_tensor(baseline_scores).float().cpu().reshape(-1)
    inliers = torch.as_tensor(baseline_inlier_rows).long().cpu().reshape(-1)
    xyz = torch.as_tensor(anchor_xyz).float().cpu()
    calibration = torch.as_tensor(intrinsic).float().cpu()
    pose = torch.as_tensor(baseline_pose_w2c).float().cpu()
    count, topk = candidates.shape
    if not (
        xy.shape == (count, 2)
        and scores.shape == candidates.shape
        and baseline.shape == base_scores.shape == (count,)
        and topk == int(cfg["topk"])
        and xyz.ndim == 2
        and xyz.shape[1] == 3
        and calibration.shape == (3, 3)
        and pose.shape == (4, 4)
        and bool(torch.isfinite(xy).all())
        and bool(torch.isfinite(scores).all())
        and int(candidates.min()) >= 0
        and int(candidates.max()) < xyz.shape[0]
        and torch.equal(candidates[:, 0], baseline)
        and torch.allclose(scores[:, 0], base_scores, rtol=1e-5, atol=1e-5)
        and (not inliers.numel() or (int(inliers.min()) >= 0 and int(inliers.max()) < count))
    ):
        raise ValueError("V21 Top-K geometric assignment inputs differ")

    points = xyz[candidates.reshape(-1)].reshape(count, topk, 3)
    camera = torch.einsum("ij,nkj->nki", pose[:3, :3], points) + pose[:3, 3]
    projected = torch.einsum("ij,nkj->nki", calibration, camera)
    depth = projected[:, :, 2]
    uv = projected[:, :, :2] / depth.clamp_min(1e-12).unsqueeze(2)
    residual = (uv - xy[:, None, :]).norm(dim=2)
    residual[depth <= 1e-12] = torch.inf
    protected = torch.zeros(count, dtype=torch.bool)
    protected[inliers] = True
    eligible = (
        (scores >= base_scores[:, None] - float(cfg["maximum_score_drop_from_top1"]))
        & (residual <= float(cfg["projection_gate_px"]))
        & (~protected[:, None])
    )
    ranked_residual = residual.clone()
    ranked_residual[~eligible] = torch.inf
    best_residual, best_rank = ranked_residual.min(dim=1)
    selected = baseline.clone()
    has_candidate = torch.isfinite(best_residual)
    selected[has_candidate] = candidates.gather(1, best_rank[:, None]).squeeze(1)[
        has_candidate
    ]
    changed = selected != baseline
    if bool((changed & protected).any()):
        raise RuntimeError("V21 Top-K geometry changed a protected baseline inlier")
    return {
        "anchor_rows": selected,
        "changed_query_rows": torch.nonzero(changed, as_tuple=False).reshape(-1),
        "selected_candidate_ranks": best_rank[changed] + 1,
        "selected_reprojection_residual_px": best_residual[changed],
    }


def evaluate_record(
    *,
    record: Mapping,
    normalized_anchor_features: torch.Tensor,
    anchor_xyz: torch.Tensor,
    baseline_contract: Mapping,
    config: Mapping,
    device: str | torch.device,
    matcher_chunk_size: int,
) -> dict:
    cfg = validate_config(config)
    baseline = _baseline_outcome(record)
    inlier_count = int(baseline["inlier_count"])
    eligible_band = (
        int(cfg["minimum_baseline_inlier_count_inclusive"])
        <= inlier_count
        < int(cfg["maximum_baseline_inlier_count_exclusive"])
    )
    candidate = baseline
    accepted = False
    assignment = {
        "anchor_rows": torch.as_tensor(record["winner_anchor_rows"]).long().cpu(),
        "changed_query_rows": torch.empty(0, dtype=torch.long),
        "selected_candidate_ranks": torch.empty(0, dtype=torch.long),
        "selected_reprojection_residual_px": torch.empty(0),
    }
    if eligible_band:
        matches = global_cosine_topk(
            torch.as_tensor(record["descriptors"], device=device).float(),
            normalized_anchor_features,
            topk=int(cfg["topk"]),
            chunk_size=int(matcher_chunk_size),
            anchor_descriptors_normalized=True,
        )
        assignment = select_topk_geometry_rows(
            keypoints=torch.as_tensor(record["keypoints"]).float()
            + float(baseline_contract["pixel_center_offset"]),
            topk_anchor_rows=matches.anchor_indices.cpu(),
            topk_scores=matches.scores.cpu(),
            baseline_anchor_rows=record["winner_anchor_rows"],
            baseline_scores=record["winner_scores"],
            baseline_inlier_rows=record["baseline_inliers"],
            anchor_xyz=anchor_xyz,
            intrinsic=record["intrinsics"],
            baseline_pose_w2c=record["baseline_pose_w2c"],
            config=cfg,
        )
        if assignment["changed_query_rows"].numel():
            replay = replay_pose_with_contract(
                keypoints=torch.as_tensor(record["keypoints"]).float()
                + float(baseline_contract["pixel_center_offset"]),
                anchor_rows=assignment["anchor_rows"],
                anchor_xyz=anchor_xyz,
                intrinsic=record["intrinsics"],
                ground_truth_w2c=record["pose_w2c"],
                baseline_contract=baseline_contract,
            )
            accepted = replay["inlier_count"] >= (
                baseline["inlier_count"] + int(cfg["minimum_candidate_inlier_gain"])
            )
            if accepted:
                candidate = replay
    gain = bool(not baseline["r5_success"] and candidate["r5_success"])
    loss = bool(baseline["r5_success"] and not candidate["r5_success"])
    return {
        "query_index": int(record["query_index"]),
        "image_name": str(record["image_name"]),
        "sequence_id": str(record["sequence_id"]),
        "block_id": str(record["block_id"]),
        "source_record_sha256": str(record["source_record_sha256"]),
        "eligible_by_baseline_inlier_band": eligible_band,
        "candidate_accepted": accepted,
        "changed_query_row_count": int(assignment["changed_query_rows"].numel()),
        "changed_query_rows": assignment["changed_query_rows"],
        "selected_candidate_ranks": assignment["selected_candidate_ranks"],
        "selected_reprojection_residual_px": assignment[
            "selected_reprojection_residual_px"
        ],
        "baseline": baseline,
        "candidate": candidate,
        "r5_gain": gain,
        "r5_loss": loss,
        "catastrophe": loss,
        "paired_delta_task_error": float(candidate["task_error"] - baseline["task_error"]),
    }


def _summary(records: Sequence[Mapping]) -> dict:
    count = len(records)
    baseline = sum(bool(value["baseline"]["r5_success"]) for value in records)
    candidate = sum(bool(value["candidate"]["r5_success"]) for value in records)
    gains = sum(bool(value["r5_gain"]) for value in records)
    losses = sum(bool(value["r5_loss"]) for value in records)
    return {
        "query_count": count,
        "eligible_query_count": sum(
            bool(value["eligible_by_baseline_inlier_band"]) for value in records
        ),
        "accepted_query_count": sum(bool(value["candidate_accepted"]) for value in records),
        "baseline_r5_success_count": baseline,
        "candidate_r5_success_count": candidate,
        "paired_r5_gain_count": gains,
        "paired_r5_loss_count": losses,
        "paired_r5_net_count": gains - losses,
        "paired_r5_rate_delta": float((candidate - baseline) / count) if count else 0.0,
        "catastrophe_count": losses,
        "changed_query_row_count_total": sum(
            int(value["changed_query_row_count"]) for value in records
        ),
    }


def build_evaluation(
    *,
    stable_map: Mapping,
    cache_payloads: Sequence[Mapping],
    stable_map_source: Mapping,
    cache_sources: Sequence[Mapping],
    producer_sources: Sequence[Mapping],
    device: str | torch.device = "cpu",
    matcher_chunk_size: int = 8192,
) -> dict:
    ordered, records, baseline_contract = validate_complete_cache_payloads(cache_payloads)
    role = str(ordered[0]["role"])
    if role not in {"adaptation", "control", "confirmation"}:
        raise ValueError("V21 Top-K evaluation role is unsupported")
    stable_source = _source(stable_map_source, label="stable map")
    sources = [_source(value, label="frontend cache") for value in cache_sources]
    if len(sources) != len(cache_payloads):
        raise ValueError("V21 Top-K cache source registry differs")
    for payload in cache_payloads:
        declared = _source(payload["inputs"]["stable_map"], label="cache stable map")
        if (declared["path"], declared["sha256"]) != (
            stable_source["path"],
            stable_source["sha256"],
        ):
            raise ValueError("V21 Top-K cache/stable-map lineage differs")
    features = torch.as_tensor(stable_map.get("anchor_features")).float()
    xyz = torch.as_tensor(stable_map.get("anchor_xyz")).float().cpu()
    if features.ndim != 2 or xyz.shape != (features.shape[0], 3):
        raise ValueError("V21 Top-K stable map is invalid")
    bank = F.normalize(F.normalize(features.to(device), dim=1), dim=1)
    cfg = default_config()
    evaluated = [
        evaluate_record(
            record=record,
            normalized_anchor_features=bank,
            anchor_xyz=xyz,
            baseline_contract=baseline_contract,
            config=cfg,
            device=device,
            matcher_chunk_size=matcher_chunk_size,
        )
        for record in records
    ]
    output = {
        "schema": SCHEMA,
        "version": VERSION,
        "protocol": "test_adapted",
        "uses_test_queries": True,
        "test_adapted": True,
        "evaluation_role": role,
        "adaptation_selected_configuration": True,
        "control_outcomes_used_to_form_configuration": False,
        "confirmation_features_consumed": role == "confirmation",
        "confirmation_outcomes_consumed": role == "confirmation",
        "writes_map_or_metric": False,
        "controller_authorized": False,
        "deployment_authorized": False,
        "matching_semantics": "global_cosine_topk_then_baseline_pose_geometric_outlier_reassignment",
        "baseline_inliers_are_bit_exact_protected": True,
        "ground_truth_used_by_candidate_selection": False,
        "ground_truth_used_for_metrics_only": True,
        "configuration": cfg,
        "baseline_contract": baseline_contract,
        "matcher_chunk_size": int(matcher_chunk_size),
        "inputs": {
            "stable_map": stable_source,
            "frontend_caches": sources,
            "producer_sources": [_source(value, label="producer") for value in producer_sources],
            "split_manifest": dict(ordered[0]["inputs"]["split_manifest"]),
        },
        "records": evaluated,
        "summary": _summary(evaluated),
    }
    validate_evaluation(output)
    return output


def validate_evaluation(payload: Mapping) -> None:
    records = payload.get("records")
    if not (
        payload.get("schema") == SCHEMA
        and payload.get("version") == VERSION
        and payload.get("protocol") == "test_adapted"
        and payload.get("evaluation_role") in {"adaptation", "control", "confirmation"}
        and payload.get("writes_map_or_metric") is False
        and payload.get("controller_authorized") is False
        and payload.get("deployment_authorized") is False
        and payload.get("ground_truth_used_by_candidate_selection") is False
        and payload.get("baseline_inliers_are_bit_exact_protected") is True
        and isinstance(records, list)
        and records
    ):
        raise ValueError("unsupported V21 Top-K geometric evaluation")
    validate_config(payload.get("configuration", {}))
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("V21 Top-K input lineage is missing")
    _source(inputs.get("stable_map"), label="stable map")
    caches = inputs.get("frontend_caches")
    producers = inputs.get("producer_sources")
    if not isinstance(caches, list) or not caches or not isinstance(producers, list) or not producers:
        raise ValueError("V21 Top-K source registries are empty")
    [_source(value, label="frontend cache") for value in caches]
    [_source(value, label="producer") for value in producers]
    seen = set()
    cfg = validate_config(payload["configuration"])
    for record in records:
        query = int(record.get("query_index", -1))
        changed = torch.as_tensor(record.get("changed_query_rows")).long().reshape(-1)
        ranks = torch.as_tensor(record.get("selected_candidate_ranks")).long().reshape(-1)
        residual = torch.as_tensor(record.get("selected_reprojection_residual_px")).float().reshape(-1)
        baseline = record.get("baseline")
        candidate = record.get("candidate")
        if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
            raise ValueError("V21 Top-K pose outcomes are missing")
        baseline_inliers = torch.as_tensor(baseline.get("inlier_query_rows")).long().reshape(-1)
        eligible = (
            int(cfg["minimum_baseline_inlier_count_inclusive"])
            <= int(baseline.get("inlier_count", -1))
            < int(cfg["maximum_baseline_inlier_count_exclusive"])
        )
        accepted = bool(record.get("candidate_accepted"))
        gain = bool(not baseline.get("r5_success") and candidate.get("r5_success"))
        loss = bool(baseline.get("r5_success") and not candidate.get("r5_success"))
        if (
            query < 0
            or query in seen
            or changed.shape != ranks.shape
            or ranks.shape != residual.shape
            or int(record.get("changed_query_row_count", -1)) != changed.numel()
            or (changed.numel() and (int(changed.min()) < 0 or int(ranks.min()) < 1 or int(ranks.max()) > 64))
            or not bool(torch.isfinite(residual).all())
            or (residual.numel() and float(residual.max()) > float(cfg["projection_gate_px"]))
            or torch.unique(changed).numel() != changed.numel()
            or bool(torch.isin(changed, baseline_inliers).any())
            or bool(record.get("eligible_by_baseline_inlier_band")) != eligible
            or (not eligible and (accepted or changed.numel() != 0))
            or (accepted and changed.numel() == 0)
            or (
                accepted
                and int(candidate.get("inlier_count", -1))
                < int(baseline.get("inlier_count", -1))
                + int(cfg["minimum_candidate_inlier_gain"])
            )
            or bool(record.get("r5_gain")) != gain
            or bool(record.get("r5_loss")) != loss
            or bool(record.get("catastrophe")) != loss
        ):
            raise ValueError("V21 Top-K record is invalid")
        seen.add(query)
    if payload.get("summary") != _summary(records):
        raise ValueError("V21 Top-K summary differs")


def atomic_torch_save_fresh(payload: Mapping, output: str | Path) -> Path:
    path = Path(output).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"V21 Top-K output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        torch.save(dict(payload), temporary)
        validate_evaluation(torch.load(temporary, map_location="cpu", weights_only=False))
        os.link(temporary, path)
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return path


def finalize_evaluations(
    evaluations: Sequence[Mapping], evaluation_sources: Sequence[Mapping]
) -> dict:
    """Make the one-way deployment decision after confirmation is consumed."""

    if len(evaluations) != 3 or len(evaluation_sources) != 3:
        raise ValueError("V21 Top-K finalizer requires adaptation/control/confirmation")
    by_role = {}
    sources = {}
    for payload, source in zip(evaluations, evaluation_sources):
        validate_evaluation(payload)
        role = str(payload["evaluation_role"])
        if role in by_role:
            raise ValueError("V21 Top-K finalizer role is duplicated")
        by_role[role] = payload
        sources[role] = _source(source, label=f"{role} evaluation")
    if set(by_role) != {"adaptation", "control", "confirmation"}:
        raise ValueError("V21 Top-K finalizer role coverage differs")
    first = by_role["adaptation"]
    stable_identity = (
        first["inputs"]["stable_map"]["path"],
        first["inputs"]["stable_map"]["sha256"],
    )
    split_identity = (
        first["inputs"]["split_manifest"]["path"],
        first["inputs"]["split_manifest"]["sha256"],
    )
    seen_queries = set()
    for payload in by_role.values():
        if (
            payload["configuration"] != first["configuration"]
            or payload["baseline_contract"] != first["baseline_contract"]
            or (
                payload["inputs"]["stable_map"]["path"],
                payload["inputs"]["stable_map"]["sha256"],
            )
            != stable_identity
            or (
                payload["inputs"]["split_manifest"]["path"],
                payload["inputs"]["split_manifest"]["sha256"],
            )
            != split_identity
        ):
            raise ValueError("V21 Top-K finalizer protocol lineage differs")
        queries = {int(record["query_index"]) for record in payload["records"]}
        if seen_queries & queries:
            raise ValueError("V21 Top-K evaluation query registries overlap")
        seen_queries |= queries
    gates = {}
    for role in ("adaptation", "control", "confirmation"):
        summary = by_role[role]["summary"]
        gates[role] = {
            "minimum_paired_r5_gain": 1,
            "required_paired_r5_loss": 0,
            "observed_gain": int(summary["paired_r5_gain_count"]),
            "observed_loss": int(summary["paired_r5_loss_count"]),
            "passed": bool(
                int(summary["paired_r5_gain_count"]) >= 1
                and int(summary["paired_r5_loss_count"]) == 0
                and int(summary["catastrophe_count"]) == 0
            ),
        }
    deployment = all(value["passed"] for value in gates.values())
    baseline_total = sum(
        int(payload["summary"]["baseline_r5_success_count"])
        for payload in by_role.values()
    )
    candidate_total = sum(
        int(payload["summary"]["candidate_r5_success_count"])
        for payload in by_role.values()
    )
    return {
        "schema": FINAL_SCHEMA,
        "version": VERSION,
        "protocol": "test_adapted_forward_adaptation_control_confirmation",
        "uses_test_queries": True,
        "test_adapted": True,
        "configuration_frozen_before_control": True,
        "confirmation_consumed_once": True,
        "decision": (
            "GO_DEPLOYMENT_CANDIDATE"
            if deployment
            else "STOP_CONFIRMATION_GATE_FAILED"
        ),
        "deployment_authorized": deployment,
        "controller_authorized": deployment,
        "writes_map_or_metric": False,
        "inputs": {"evaluations": sources},
        "configuration": first["configuration"],
        "gates": gates,
        "combined_scored_query_count": sum(
            int(payload["summary"]["query_count"]) for payload in by_role.values()
        ),
        "combined_baseline_r5_success_count": baseline_total,
        "combined_candidate_r5_success_count": candidate_total,
        "combined_paired_r5_net_count": candidate_total - baseline_total,
        "excluded_embargo_queries_are_not_scored": True,
        "reason": (
            "all three forward gates passed"
            if deployment
            else "confirmation produced no net R5 recovery"
        ),
    }
