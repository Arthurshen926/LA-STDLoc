"""Exact tuning-control selection of atomic V21 prototype actions.

The parent candidate is formed from adaptation queries only.  This module may
use the complete tuning-control cache to choose an action subset and one
pre-registered prototype activation threshold.  It never accepts confirmation
data, never mutates the stable map or parent candidate, and always emits a
quarantined, non-deployable candidate whose control use is explicit.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import math

import numpy as np
import torch
import torch.nn.functional as F

from localization.pose_solver import solve_absolute_pose
from map_learning.v21_pose_feedback_transductive import (
    METADATA_FIELD,
    PROTOTYPE_FEATURE_FIELD,
    PROTOTYPE_OWNER_FIELD,
    _assert_cached_baseline,
    _as_cpu,
    _clone_frozen_value,
    _same_source,
    _source_identity,
    _validate_stable_map,
    assert_base_fields_bit_exact,
    evaluate_cached_record,
    replay_pose_with_contract,
    summarize_cached_evaluation,
    tensor_sha256,
    validate_baseline_contract,
    validate_candidate_map,
    validate_complete_cache_payloads,
)


SELECTED_CANDIDATE_SCHEMA = "lafgs_v21_pose_feedback_control_selected_candidate"
SELECTED_CANDIDATE_VERSION = 1
SEARCH_AUDIT_SCHEMA = "lafgs_v21_pose_feedback_control_subset_search_audit"
SEARCH_AUDIT_VERSION = 1
SEARCH_ALGORITHM = "single_scan_greedy_beam_backward_prune_exact_poselib"
MATCHING_SEMANTICS = (
    "global_owner_prototype_top1_with_optional_absolute_cosine_activation_threshold"
)


class ControlSubsetSearchStopped(ValueError):
    """A scientifically valid STOP with a complete immutable search audit."""

    def __init__(self, message: str, *, audit: Mapping) -> None:
        super().__init__(message)
        self.audit = dict(audit)


def _threshold_key(value: float | None) -> tuple[int, float]:
    return (0, -2.0) if value is None else (1, float(value))


def _validate_threshold_menu(values: Sequence[float | None]) -> tuple[float | None, ...]:
    if not values:
        raise ValueError("V21 control selection threshold menu is empty")
    output: list[float | None] = []
    for value in values:
        parsed = None if value is None else float(value)
        if parsed is not None and (not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0):
            raise ValueError("V21 control prototype threshold must be in [0, 1]")
        if parsed not in output:
            output.append(parsed)
    if tuple(output) != tuple(sorted(output, key=_threshold_key)):
        raise ValueError("V21 control threshold menu must be sorted and unique")
    return tuple(output)


def _action_registry(metadata: Mapping, prototype_count: int) -> list[dict]:
    actions = metadata.get("selected_actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("V21 parent candidate action registry is empty")
    output = []
    seen: list[int] = []
    for ordinal, action in enumerate(actions):
        indices = _as_cpu(action.get("prototype_indices"), dtype=torch.long).reshape(-1)
        if indices.numel() == 0 or torch.unique(indices).numel() != indices.numel():
            raise ValueError("V21 parent action prototype registry is invalid")
        output.append({"ordinal": ordinal, "indices": indices, "action": action})
        seen.extend(int(value) for value in indices.tolist())
    if seen != list(range(prototype_count)):
        raise ValueError("V21 parent action ordering is not atomic and contiguous")
    return output


def _prepare_control_records(
    *,
    cache_records: Sequence[Mapping],
    actions: Sequence[Mapping],
    prototypes: torch.Tensor,
    prototype_owners: torch.Tensor,
    anchor_xyz: torch.Tensor,
    baseline_contract: Mapping,
    device: str | torch.device,
    solver: Callable,
) -> list[dict]:
    device_prototypes = F.normalize(prototypes.to(device=device).float(), dim=1)
    prepared = []
    for record in cache_records:
        descriptors = F.normalize(
            torch.as_tensor(record["descriptors"], device=device).float(), dim=1
        )
        scores = descriptors @ device_prototypes.T
        action_scores = []
        action_prototype_indices = []
        for action in actions:
            indices = action["indices"].to(device)
            values, local = scores[:, indices].max(dim=1)
            action_scores.append(values.detach().cpu().float())
            action_prototype_indices.append(indices[local].detach().cpu().long())
        baseline_rows = _as_cpu(record["winner_anchor_rows"], dtype=torch.long).reshape(-1)
        baseline_scores = _as_cpu(record["winner_scores"], dtype=torch.float32).reshape(-1)
        physical_keypoints = _as_cpu(record["keypoints"], dtype=torch.float32) + float(
            baseline_contract["pixel_center_offset"]
        )
        baseline = replay_pose_with_contract(
            keypoints=physical_keypoints,
            anchor_rows=baseline_rows,
            anchor_xyz=anchor_xyz,
            intrinsic=record["intrinsics"],
            ground_truth_w2c=record["pose_w2c"],
            baseline_contract=baseline_contract,
            solver=solver,
        )
        _assert_cached_baseline(record, baseline)
        prepared.append(
            {
                "record": record,
                "physical_keypoints": physical_keypoints,
                "baseline_rows": baseline_rows,
                "baseline_scores": baseline_scores,
                "baseline": baseline,
                "action_scores": torch.stack(action_scores),
                "action_prototype_indices": torch.stack(action_prototype_indices),
                "pose_cache": {baseline_rows.numpy().tobytes(): baseline},
            }
        )
    return prepared


def _winner_rows(prepared: Mapping, action_indices: tuple[int, ...], threshold: float | None, owners: torch.Tensor) -> torch.Tensor:
    baseline_rows = prepared["baseline_rows"]
    if not action_indices:
        return baseline_rows
    selected = np.asarray(action_indices, dtype=np.int64)
    action_scores = prepared["action_scores"].numpy()[selected]
    local_action = np.argmax(action_scores, axis=0)
    rows = np.arange(action_scores.shape[1], dtype=np.int64)
    best_scores = action_scores[local_action, rows]
    source_actions = selected[local_action]
    prototype_indices = prepared["action_prototype_indices"].numpy()[
        source_actions, rows
    ]
    activate = best_scores > prepared["baseline_scores"].numpy()
    if threshold is not None:
        activate &= best_scores >= float(threshold)
    winners = baseline_rows.numpy().copy()
    winners[activate] = owners.numpy()[prototype_indices[activate]]
    return torch.from_numpy(winners)


def _pose_outcome(
    *, prepared: dict, winner_rows: torch.Tensor, anchor_xyz: torch.Tensor,
    baseline_contract: Mapping, solver: Callable,
) -> dict:
    key = winner_rows.numpy().tobytes()
    cached = prepared["pose_cache"].get(key)
    if cached is not None:
        return cached
    record = prepared["record"]
    outcome = replay_pose_with_contract(
        keypoints=prepared["physical_keypoints"],
        anchor_rows=winner_rows,
        anchor_xyz=anchor_xyz,
        intrinsic=record["intrinsics"],
        ground_truth_w2c=record["pose_w2c"],
        baseline_contract=baseline_contract,
        solver=solver,
    )
    prepared["pose_cache"][key] = outcome
    return outcome


def _evaluate_subset(
    *, action_indices: tuple[int, ...], threshold: float | None,
    prepared_records: Sequence[dict], owners: torch.Tensor, anchor_xyz: torch.Tensor,
    baseline_contract: Mapping, solver: Callable,
) -> dict:
    gains = losses = winner_flips = changed_queries = 0
    positive_task_risk = 0.0
    maximum_positive_task_delta = 0.0
    signed_task_delta = 0.0
    outcome_digest_parts = []
    for prepared in prepared_records:
        winners = _winner_rows(prepared, action_indices, threshold, owners)
        outcome = _pose_outcome(
            prepared=prepared,
            winner_rows=winners,
            anchor_xyz=anchor_xyz,
            baseline_contract=baseline_contract,
            solver=solver,
        )
        baseline = prepared["baseline"]
        gain = not bool(baseline["r5_success"]) and bool(outcome["r5_success"])
        loss = bool(baseline["r5_success"]) and not bool(outcome["r5_success"])
        delta = float(outcome["task_error"] - baseline["task_error"])
        gains += int(gain)
        losses += int(loss)
        positive_task_risk += max(0.0, delta)
        maximum_positive_task_delta = max(maximum_positive_task_delta, max(0.0, delta))
        signed_task_delta += delta
        flips = int((winners != prepared["baseline_rows"]).sum())
        winner_flips += flips
        changed_queries += int(flips > 0)
        outcome_digest_parts.extend(
            [int(prepared["record"]["query_index"]), int(outcome["r5_success"]), flips]
        )
    return {
        "threshold": threshold,
        "action_indices": action_indices,
        "action_mask": sum(1 << value for value in action_indices),
        "action_count": len(action_indices),
        "paired_r5_gain_count": gains,
        "paired_r5_loss_count": losses,
        "catastrophe_count": losses,
        "catastrophe_definition": "baseline_r5_success_to_candidate_r5_failure",
        "positive_task_risk_sum": float(positive_task_risk),
        "maximum_positive_task_delta": float(maximum_positive_task_delta),
        "signed_task_delta_sum": float(signed_task_delta),
        "query_with_winner_flip_count": changed_queries,
        "winner_flip_count_total": winner_flips,
        "outcome_registry_sha256": tensor_sha256(
            torch.tensor(outcome_digest_parts, dtype=torch.long)
        ),
    }


def _final_key(value: Mapping, action_sizes: Sequence[int]) -> tuple:
    indices = tuple(int(index) for index in value["action_indices"])
    return (
        -int(value["paired_r5_gain_count"]),
        float(value["positive_task_risk_sum"]),
        float(value["maximum_positive_task_delta"]),
        len(indices),
        sum(int(action_sizes[index]) for index in indices),
        _threshold_key(value["threshold"]),
        indices,
    )


def _frontier_key(value: Mapping, action_sizes: Sequence[int]) -> tuple:
    return (
        int(value["paired_r5_loss_count"]),
        -int(value["paired_r5_gain_count"]),
        float(value["positive_task_risk_sum"]),
        float(value["maximum_positive_task_delta"]),
        sum(int(action_sizes[index]) for index in value["action_indices"]),
        tuple(value["action_indices"]),
    )


def _trace_row_key(value: Mapping) -> tuple:
    threshold = value.get("threshold")
    mask = int(value.get("action_mask", -1))
    return (_threshold_key(threshold), mask)


def _search_aggregate(
    trace: Sequence[Mapping], *, minimum_paired_r5_gain: int
) -> dict:
    if not trace:
        raise ValueError("V21 control search trace is empty")

    def compact(value: Mapping) -> dict:
        output = {
            key: deepcopy(value[key])
            for key in (
                "threshold",
                "action_indices",
                "action_mask",
                "action_count",
                "paired_r5_gain_count",
                "paired_r5_loss_count",
                "catastrophe_count",
                "positive_task_risk_sum",
                "maximum_positive_task_delta",
                "winner_flip_count_total",
                "outcome_registry_sha256",
            )
        }
        output["action_indices"] = _as_cpu(
            value["action_indices"], dtype=torch.long
        ).tolist()
        return output

    zero_loss = [
        value
        for value in trace
        if int(value["paired_r5_loss_count"]) == 0
        and int(value["catastrophe_count"]) == 0
    ]
    max_any = max(int(value["paired_r5_gain_count"]) for value in trace)
    max_zero_loss = max(
        (int(value["paired_r5_gain_count"]) for value in zero_loss), default=-1
    )
    any_best = min(
        (value for value in trace if int(value["paired_r5_gain_count"]) == max_any),
        key=lambda value: (
            int(value["paired_r5_loss_count"]),
            float(value["positive_task_risk_sum"]),
            int(value["action_count"]),
            _trace_row_key(value),
        ),
    )
    zero_loss_best = min(
        (value for value in zero_loss if int(value["paired_r5_gain_count"]) == max_zero_loss),
        key=lambda value: (
            float(value["positive_task_risk_sum"]),
            float(value["maximum_positive_task_delta"]),
            int(value["action_count"]),
            _trace_row_key(value),
        ),
    )
    thresholds = sorted({value.get("threshold") for value in trace}, key=_threshold_key)
    per_threshold = []
    for threshold in thresholds:
        candidates = [
            value
            for value in zero_loss
            if value.get("threshold") == threshold
        ]
        best_gain = max(int(value["paired_r5_gain_count"]) for value in candidates)
        best = min(
            (value for value in candidates if int(value["paired_r5_gain_count"]) == best_gain),
            key=lambda value: (
                float(value["positive_task_risk_sum"]),
                int(value["action_count"]),
                int(value["action_mask"]),
            ),
        )
        per_threshold.append(compact(best))
    return {
        "evaluated_subset_count": len(trace),
        "maximum_paired_r5_gain_any": max_any,
        "best_any": compact(any_best),
        "maximum_paired_r5_gain_zero_loss": max_zero_loss,
        "best_zero_loss": compact(zero_loss_best),
        "best_zero_loss_by_threshold": per_threshold,
        "accepted_subset_count": sum(
            int(value["paired_r5_gain_count"]) >= int(minimum_paired_r5_gain)
            for value in zero_loss
        ),
    }


def _build_search_audit(
    *, trace: Sequence[Mapping], thresholds: Sequence[float | None],
    stable_map_source: Mapping, parent_candidate_source: Mapping,
    control_cache_sources: Sequence[Mapping], parent_metadata: Mapping,
    parent_action_count: int, parent_prototype_count: int,
    control_query_count: int, beam_width: int, maximum_beam_depth: int,
    maximum_greedy_depth: int, maximum_backward_steps: int,
    minimum_paired_r5_gain: int, decision: str,
) -> dict:
    aggregate = _search_aggregate(
        trace, minimum_paired_r5_gain=minimum_paired_r5_gain
    )
    return {
        "schema": SEARCH_AUDIT_SCHEMA,
        "version": SEARCH_AUDIT_VERSION,
        "protocol": "test_adapted_tuning_control_selection_audit",
        "uses_test_queries": True,
        "test_adapted": True,
        "selection_role": "control",
        "control_features_consumed": True,
        "control_outcomes_consumed_for_selection": True,
        "confirmation_input_supported_by_search_tool": False,
        "confirmation_features_consumed": False,
        "confirmation_outcomes_consumed": False,
        "confirmation_unread_during_selection": True,
        "deployment_authorized": False,
        "controller_authorized": False,
        "artifact_is_candidate_map": False,
        "decision": decision,
        "candidate_map_materialization_authorized": decision == "GO_SELECTED_ACTION",
        "inputs": {
            "stable_map": dict(stable_map_source),
            "parent_adaptation_candidate": dict(parent_candidate_source),
            "split_manifest": dict(parent_metadata["inputs"]["split_manifest"]),
            "adaptation_caches": [
                dict(value) for value in parent_metadata["inputs"]["adaptation_caches"]
            ],
            "control_caches": [dict(value) for value in control_cache_sources],
            "confirmation_caches": [],
        },
        "parent_action_count": int(parent_action_count),
        "parent_prototype_count": int(parent_prototype_count),
        "control_query_count": int(control_query_count),
        "baseline_contract": validate_baseline_contract(
            parent_metadata["baseline_contract"]
        ),
        "matching_semantics": MATCHING_SEMANTICS,
        "search_contract": {
            "algorithm": SEARCH_ALGORITHM,
            "activation_threshold_menu": list(thresholds),
            "beam_width": int(beam_width),
            "maximum_beam_depth": int(maximum_beam_depth),
            "maximum_greedy_depth": int(maximum_greedy_depth),
            "maximum_backward_steps": int(maximum_backward_steps),
            "minimum_paired_r5_gain": int(minimum_paired_r5_gain),
            "required_paired_r5_loss_count": 0,
            "required_catastrophe_count": 0,
            "catastrophe_definition": "baseline_r5_success_to_candidate_r5_failure",
            "exact_poselib_for_every_evaluated_subset": True,
            "baseline_replayed_and_cache_exact": True,
            "prototype_scores_precomputed_once": True,
            "pose_outcomes_memoized_by_exact_winner_registry": True,
            "pre_registered_core_complete": True,
        },
        "aggregate": aggregate,
        "candidate_results": [deepcopy(dict(value)) for value in trace],
    }


def validate_control_subset_search_audit(payload: Mapping) -> None:
    """Validate a GO/STOP audit independently of any candidate map."""

    if not (
        payload.get("schema") == SEARCH_AUDIT_SCHEMA
        and payload.get("version") == SEARCH_AUDIT_VERSION
        and payload.get("protocol") == "test_adapted_tuning_control_selection_audit"
        and payload.get("uses_test_queries") is True
        and payload.get("test_adapted") is True
        and payload.get("selection_role") == "control"
        and payload.get("control_features_consumed") is True
        and payload.get("control_outcomes_consumed_for_selection") is True
        and payload.get("confirmation_input_supported_by_search_tool") is False
        and payload.get("confirmation_features_consumed") is False
        and payload.get("confirmation_outcomes_consumed") is False
        and payload.get("confirmation_unread_during_selection") is True
        and payload.get("deployment_authorized") is False
        and payload.get("controller_authorized") is False
        and payload.get("artifact_is_candidate_map") is False
        and payload.get("decision") in {"GO_SELECTED_ACTION", "STOP_NO_ACTION"}
        and payload.get("candidate_map_materialization_authorized")
        is (payload.get("decision") == "GO_SELECTED_ACTION")
    ):
        raise ValueError("V21 control subset search audit contract is invalid")
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping) or inputs.get("confirmation_caches") != []:
        raise ValueError("V21 control subset search confirmation lineage is invalid")
    for field in ("stable_map", "parent_adaptation_candidate", "split_manifest"):
        _source_identity(inputs.get(field), label=f"search audit {field}")
    for field in ("adaptation_caches", "control_caches"):
        values = inputs.get(field)
        if not isinstance(values, list) or not values:
            raise ValueError("V21 control subset search cache lineage is empty")
        identities = [_source_identity(value, label=f"search audit {field}") for value in values]
        if len(set(identities)) != len(identities):
            raise ValueError("V21 control subset search cache lineage is duplicated")
    if {
        _source_identity(value, label="audit adaptation cache")
        for value in inputs["adaptation_caches"]
    } & {
        _source_identity(value, label="audit control cache")
        for value in inputs["control_caches"]
    }:
        raise ValueError("V21 control subset search cache roles overlap")
    validate_baseline_contract(payload.get("baseline_contract"))
    contract = payload.get("search_contract")
    trace = payload.get("candidate_results")
    if not (
        isinstance(contract, Mapping)
        and contract.get("algorithm") == SEARCH_ALGORITHM
        and contract.get("exact_poselib_for_every_evaluated_subset") is True
        and contract.get("pre_registered_core_complete") is True
        and int(contract.get("minimum_paired_r5_gain", 0)) >= 1
        and isinstance(trace, list)
        and trace
    ):
        raise ValueError("V21 control subset search trace contract is invalid")
    thresholds = _validate_threshold_menu(contract.get("activation_threshold_menu", ()))
    action_count = int(payload.get("parent_action_count", 0))
    seen = set()
    for value in trace:
        threshold = value.get("threshold")
        indices = _as_cpu(value.get("action_indices"), dtype=torch.long).reshape(-1)
        expected_mask = sum(1 << int(index) for index in indices.tolist())
        key = (threshold, expected_mask)
        if not (
            threshold in thresholds
            and indices.tolist() == sorted(set(indices.tolist()))
            and (indices.numel() == 0 or (int(indices.min()) >= 0 and int(indices.max()) < action_count))
            and int(value.get("action_mask", -1)) == expected_mask
            and int(value.get("action_count", -1)) == indices.numel()
            and int(value.get("paired_r5_gain_count", -1)) >= 0
            and int(value.get("paired_r5_loss_count", -1)) >= 0
            and int(value.get("catastrophe_count", -1)) == int(value["paired_r5_loss_count"])
            and math.isfinite(float(value.get("positive_task_risk_sum", math.nan)))
            and float(value["positive_task_risk_sum"]) >= 0.0
            and key not in seen
        ):
            raise ValueError("V21 control subset search candidate result is invalid")
        seen.add(key)
    full = tuple(range(action_count))
    required = set()
    for threshold in thresholds:
        required.add((threshold, 0))
        required.add((threshold, sum(1 << value for value in full)))
        required.update((threshold, 1 << value) for value in full)
        required.update(
            (
                threshold,
                sum(1 << value for value in full if value != removed),
            )
            for removed in full
        )
    if not required <= seen:
        raise ValueError("V21 control subset pre-registered core is incomplete")
    expected_aggregate = _search_aggregate(
        trace, minimum_paired_r5_gain=int(contract["minimum_paired_r5_gain"])
    )
    if payload.get("aggregate") != expected_aggregate:
        raise ValueError("V21 control subset search aggregate differs")
    accepted = int(expected_aggregate["accepted_subset_count"])
    if (payload["decision"] == "STOP_NO_ACTION") != (accepted == 0):
        raise ValueError("V21 control subset search decision differs")


def _selected_record(
    *, record: Mapping, anchor_features: torch.Tensor, anchor_xyz: torch.Tensor,
    prototypes: torch.Tensor, owners: torch.Tensor, baseline_contract: Mapping,
    threshold: float | None, matcher_chunk_size: int, device: str | torch.device,
    solver: Callable,
) -> dict:
    return evaluate_cached_record(
        record=record,
        anchor_features=anchor_features,
        anchor_xyz=anchor_xyz,
        extra_prototypes=prototypes,
        prototype_owner_rows=owners,
        baseline_contract=baseline_contract,
        matcher_chunk_size=matcher_chunk_size,
        device=device,
        prototype_activation_threshold=threshold,
        solver=solver,
    )


def build_control_selected_candidate(
    *, stable_map: Mapping, parent_candidate: Mapping,
    control_cache_payloads: Sequence[Mapping], stable_map_source: Mapping,
    parent_candidate_source: Mapping, control_cache_sources: Sequence[Mapping],
    activation_threshold_menu: Sequence[float | None] = (None, 0.8, 0.85, 0.9, 0.95),
    beam_width: int = 8, maximum_beam_depth: int = 8,
    maximum_greedy_depth: int | None = None,
    maximum_backward_steps: int | None = None,
    minimum_paired_r5_gain: int = 1, matcher_chunk_size: int = 8192,
    device: str | torch.device = "cpu", solver: Callable = solve_absolute_pose,
) -> dict:
    """Select an atomic action subset using tuning-control and exact PoseLib."""

    stable_features, anchor_xyz = _validate_stable_map(stable_map)
    parent_metadata = validate_candidate_map(parent_candidate, stable_map=stable_map)
    _same_source(parent_metadata["inputs"]["stable_map"], stable_map_source, label="parent stable map")
    payloads, records, baseline_contract = validate_complete_cache_payloads(
        control_cache_payloads, required_role="control"
    )
    if any(
        payload.get("training_consumers_allowed") is not False
        or payload.get("training_consumer_allowed") is not False
        for payload in payloads
    ):
        raise ValueError("V21 control selection cache is not held-out tuning data")
    if len(control_cache_sources) != len(payloads) or len(
        {(_source_identity(value, label="control cache")) for value in control_cache_sources}
    ) != len(control_cache_sources):
        raise ValueError("V21 control selection cache source registry differs")
    for payload in payloads:
        _same_source(payload["inputs"]["stable_map"], stable_map_source, label="control stable map")
        _same_source(payload["inputs"]["split_manifest"], parent_metadata["inputs"]["split_manifest"], label="control split")
        if payload.get("preprocessing_config_sha256") != parent_metadata.get("preprocessing_config_sha256"):
            raise ValueError("V21 control/parent preprocessing differs")
    if baseline_contract != parent_metadata.get("baseline_contract"):
        raise ValueError("V21 control/parent PoseLib contract differs")
    parent_adaptation_sources = {
        _source_identity(value, label="parent adaptation cache")
        for value in parent_metadata["inputs"]["adaptation_caches"]
    }
    if parent_adaptation_sources & {
        _source_identity(value, label="control cache") for value in control_cache_sources
    }:
        raise ValueError("V21 control selection overlaps adaptation formation data")
    thresholds = _validate_threshold_menu(activation_threshold_menu)
    if not (1 <= int(beam_width) <= 256 and 1 <= int(maximum_beam_depth) <= 32):
        raise ValueError("V21 control search beam bounds are invalid")
    greedy_depth = (
        int(maximum_beam_depth)
        if maximum_greedy_depth is None
        else int(maximum_greedy_depth)
    )
    backward_steps = (
        len(parent_metadata["selected_actions"])
        if maximum_backward_steps is None
        else int(maximum_backward_steps)
    )
    if not (1 <= greedy_depth <= 32 and 1 <= backward_steps <= 32):
        raise ValueError("V21 control search greedy/backward bounds are invalid")
    if int(minimum_paired_r5_gain) < 1 or int(matcher_chunk_size) < 1:
        raise ValueError("V21 control selection acceptance bounds are invalid")
    parent_prototypes = torch.as_tensor(parent_candidate[PROTOTYPE_FEATURE_FIELD]).float().cpu()
    parent_owners = torch.as_tensor(parent_candidate[PROTOTYPE_OWNER_FIELD]).long().cpu()
    actions = _action_registry(parent_metadata, parent_owners.numel())
    if len(actions) > 24:
        raise ValueError("V21 exact subset search is bounded to at most 24 actions")
    prepared = _prepare_control_records(
        cache_records=records, actions=actions, prototypes=parent_prototypes,
        prototype_owners=parent_owners, anchor_xyz=anchor_xyz,
        baseline_contract=baseline_contract, device=device, solver=solver,
    )
    action_sizes = [int(value["indices"].numel()) for value in actions]
    evaluated: dict[tuple[float | None, int], dict] = {}
    stages: dict[tuple[float | None, int], set[str]] = {}

    def evaluate(indices: tuple[int, ...], threshold: float | None, stage: str) -> dict:
        indices = tuple(sorted(set(int(value) for value in indices)))
        if any(value < 0 or value >= len(actions) for value in indices):
            raise ValueError("V21 control search action is outside the parent registry")
        mask = sum(1 << value for value in indices)
        key = (threshold, mask)
        stages.setdefault(key, set()).add(stage)
        if key not in evaluated:
            evaluated[key] = _evaluate_subset(
                action_indices=indices, threshold=threshold,
                prepared_records=prepared, owners=parent_owners,
                anchor_xyz=anchor_xyz, baseline_contract=baseline_contract,
                solver=solver,
            )
        return evaluated[key]

    for threshold in thresholds:
        evaluate((), threshold, "empty_baseline")
        singles = [evaluate((index,), threshold, "single_arm_scan") for index in range(len(actions))]

        current: tuple[int, ...] = ()
        for _ in range(min(greedy_depth, len(actions))):
            choices = [
                evaluate((*current, index), threshold, "greedy_forward")
                for index in range(len(actions)) if index not in current
            ]
            if not choices:
                break
            current = tuple(min(choices, key=lambda value: _frontier_key(value, action_sizes))["action_indices"])

        frontier = sorted(singles, key=lambda value: _frontier_key(value, action_sizes))[: int(beam_width)]
        for depth in range(2, min(int(maximum_beam_depth), len(actions)) + 1):
            masks: set[tuple[int, ...]] = set()
            for state in frontier:
                existing = tuple(state["action_indices"])
                for index in range(existing[-1] + 1, len(actions)):
                    masks.add((*existing, index))
            if not masks:
                break
            candidates = [evaluate(mask, threshold, f"beam_depth_{depth}") for mask in sorted(masks)]
            frontier = sorted(candidates, key=lambda value: _frontier_key(value, action_sizes))[: int(beam_width)]

        current = tuple(range(len(actions)))
        evaluate(current, threshold, "full_set")
        for _ in range(min(backward_steps, len(actions))):
            if not current:
                break
            choices = [
                evaluate(tuple(value for value in current if value != removed), threshold, "backward_prune")
                for removed in current
            ]
            feasible = [value for value in choices if int(value["paired_r5_loss_count"]) == 0]
            chosen = min(
                feasible,
                key=lambda value: _final_key(value, action_sizes),
                default=min(choices, key=lambda value: _frontier_key(value, action_sizes)),
            )
            current = tuple(chosen["action_indices"])

    trace = []
    for key, value in sorted(
        evaluated.items(), key=lambda item: (_threshold_key(item[0][0]), item[0][1])
    ):
        row = dict(value)
        row["action_indices"] = torch.tensor(row["action_indices"], dtype=torch.long)
        row["discovery_stages"] = sorted(stages[key])
        trace.append(row)
    feasible = [
        value for value in evaluated.values()
        if int(value["paired_r5_loss_count"]) == 0
        and int(value["catastrophe_count"]) == 0
        and int(value["paired_r5_gain_count"]) >= int(minimum_paired_r5_gain)
    ]
    if not feasible:
        audit = _build_search_audit(
            trace=trace,
            thresholds=thresholds,
            stable_map_source=stable_map_source,
            parent_candidate_source=parent_candidate_source,
            control_cache_sources=control_cache_sources,
            parent_metadata=parent_metadata,
            parent_action_count=len(actions),
            parent_prototype_count=parent_owners.numel(),
            control_query_count=len(records),
            beam_width=beam_width,
            maximum_beam_depth=maximum_beam_depth,
            maximum_greedy_depth=greedy_depth,
            maximum_backward_steps=backward_steps,
            minimum_paired_r5_gain=minimum_paired_r5_gain,
            decision="STOP_NO_ACTION",
        )
        validate_control_subset_search_audit(audit)
        raise ControlSubsetSearchStopped(
            "V21 control search found no zero-loss positive-R5 action subset",
            audit=audit,
        )
    selected = min(feasible, key=lambda value: _final_key(value, action_sizes))
    selected_indices = tuple(selected["action_indices"])
    selected_parent_prototypes = torch.cat([actions[index]["indices"] for index in selected_indices])
    selected_prototypes = parent_prototypes[selected_parent_prototypes].contiguous()
    selected_owners = parent_owners[selected_parent_prototypes].contiguous()
    selected_records = [
        _selected_record(
            record=record, anchor_features=stable_features, anchor_xyz=anchor_xyz,
            prototypes=selected_prototypes, owners=selected_owners,
            baseline_contract=baseline_contract, threshold=selected["threshold"],
            matcher_chunk_size=matcher_chunk_size, device=device, solver=solver,
        )
        for record in records
    ]
    selected_summary = summarize_cached_evaluation(selected_records)
    if not (
        int(selected_summary["paired_r5_gain_count"]) == int(selected["paired_r5_gain_count"])
        and int(selected_summary["paired_r5_loss_count"]) == 0
        and int(selected_summary["catastrophe_count"]) == 0
    ):
        raise RuntimeError("V21 selected-bank exact matcher replay differs from subset search")
    selected_actions = []
    cursor = 0
    for index in selected_indices:
        source = actions[index]
        action = deepcopy(dict(source["action"]))
        size = int(source["indices"].numel())
        action["parent_action_index"] = index
        action["parent_prototype_indices"] = source["indices"].clone()
        action["prototype_indices"] = torch.arange(cursor, cursor + size, dtype=torch.long)
        action["prototype_features_sha256"] = tensor_sha256(selected_prototypes[cursor: cursor + size])
        action["prototype_owner_rows_sha256"] = tensor_sha256(selected_owners[cursor: cursor + size])
        selected_actions.append(action)
        cursor += size
    search_audit = _build_search_audit(
        trace=trace,
        thresholds=thresholds,
        stable_map_source=stable_map_source,
        parent_candidate_source=parent_candidate_source,
        control_cache_sources=control_cache_sources,
        parent_metadata=parent_metadata,
        parent_action_count=len(actions),
        parent_prototype_count=parent_owners.numel(),
        control_query_count=len(records),
        beam_width=beam_width,
        maximum_beam_depth=maximum_beam_depth,
        maximum_greedy_depth=greedy_depth,
        maximum_backward_steps=backward_steps,
        minimum_paired_r5_gain=minimum_paired_r5_gain,
        decision="GO_SELECTED_ACTION",
    )
    validate_control_subset_search_audit(search_audit)
    metadata = {
        "schema": SELECTED_CANDIDATE_SCHEMA,
        "version": SELECTED_CANDIDATE_VERSION,
        "protocol": "test_adapted_tuning_control_selected",
        "uses_test_queries": True,
        "test_adapted": True,
        "formation_role": "adaptation",
        "selection_role": "control",
        "adaptation_features_consumed": True,
        "control_features_consumed": True,
        "control_outcomes_consumed_for_selection": True,
        "control_used_for_selection": True,
        "confirmation_input_supported_by_search_tool": False,
        "confirmation_features_consumed": False,
        "confirmation_outcomes_consumed": False,
        "confirmation_unread_during_selection": True,
        "deployment_authorized": False,
        "controller_authorized": False,
        "quarantined": True,
        "base_anchor_fields_bit_exact": True,
        "base_anchor_features_moved_or_lowered": False,
        "geometry_changed": False,
        "pose_valid_edge_claimed": True,
        "identity_truth_claimed": False,
        "negative_anchor_labels_created": False,
        "action_atomicity": "complete_parent_source_query_bundle_or_absent",
        "matching_semantics": MATCHING_SEMANTICS,
        "matching_contract": {
            "prototype_activation_score": "absolute_cosine",
            "prototype_activation_threshold": selected["threshold"],
            "threshold_comparison": "greater_than_or_equal",
            "base_winner_comparison": "prototype_score_strictly_greater_than_base_score",
            "tie_break": "base_then_lower_parent_prototype_index",
        },
        "inputs": {
            "stable_map": dict(stable_map_source),
            "parent_adaptation_candidate": dict(parent_candidate_source),
            "split_manifest": dict(parent_metadata["inputs"]["split_manifest"]),
            "adaptation_caches": [dict(value) for value in parent_metadata["inputs"]["adaptation_caches"]],
            "control_caches": [dict(value) for value in control_cache_sources],
            "confirmation_caches": [],
        },
        "preprocessing_config_sha256": parent_metadata["preprocessing_config_sha256"],
        "baseline_contract": validate_baseline_contract(baseline_contract),
        "parent_action_count": len(actions),
        "parent_prototype_count": int(parent_owners.numel()),
        "selected_action_count": len(selected_actions),
        "added_prototype_count": int(selected_owners.numel()),
        "selected_actions": selected_actions,
        "prototype_features_sha256": tensor_sha256(selected_prototypes),
        "prototype_owner_rows_sha256": tensor_sha256(selected_owners),
        "selection_objective": (
            "lexicographic_max_paired_r5_gain_subject_to_zero_r5_loss_and_zero_"
            "catastrophe_then_min_positive_task_risk_then_min_actions"
        ),
        "search": {
            "algorithm": SEARCH_ALGORITHM,
            "activation_threshold_menu": list(thresholds),
            "beam_width": int(beam_width),
            "maximum_beam_depth": int(maximum_beam_depth),
            "maximum_greedy_depth": greedy_depth,
            "maximum_backward_steps": backward_steps,
            "minimum_paired_r5_gain": int(minimum_paired_r5_gain),
            "evaluated_subset_count": len(trace),
            "exact_poselib_for_every_evaluated_subset": True,
            "baseline_replayed_and_cache_exact": True,
            "prototype_scores_precomputed_once": True,
            "pose_outcomes_memoized_by_exact_winner_registry": True,
            "search_trace": trace,
        },
        "selected_control_summary": selected_summary,
        "selected_control_records": selected_records,
        "selected_search_outcome": dict(selected),
        "search_audit": search_audit,
    }
    output = _clone_frozen_value(stable_map)
    output[PROTOTYPE_FEATURE_FIELD] = selected_prototypes
    output[PROTOTYPE_OWNER_FIELD] = selected_owners
    output[METADATA_FIELD] = metadata
    validate_control_selected_candidate(output, stable_map=stable_map, parent_candidate=parent_candidate)
    return output


def validate_control_selected_candidate(
    candidate: Mapping, *, stable_map: Mapping, parent_candidate: Mapping | None = None
) -> dict:
    """Fail closed on selected action integrity, lineage, and control outcome."""

    stable_features, _ = _validate_stable_map(stable_map)
    assert_base_fields_bit_exact(stable_map, candidate)
    prototypes = torch.as_tensor(candidate.get(PROTOTYPE_FEATURE_FIELD)).float()
    owners = torch.as_tensor(candidate.get(PROTOTYPE_OWNER_FIELD)).long().reshape(-1)
    metadata = candidate.get(METADATA_FIELD)
    if not (
        isinstance(metadata, Mapping)
        and metadata.get("schema") == SELECTED_CANDIDATE_SCHEMA
        and metadata.get("version") == SELECTED_CANDIDATE_VERSION
        and metadata.get("protocol") == "test_adapted_tuning_control_selected"
        and metadata.get("uses_test_queries") is True
        and metadata.get("test_adapted") is True
        and metadata.get("selection_role") == "control"
        and metadata.get("control_features_consumed") is True
        and metadata.get("control_outcomes_consumed_for_selection") is True
        and metadata.get("control_used_for_selection") is True
        and metadata.get("confirmation_input_supported_by_search_tool") is False
        and metadata.get("confirmation_features_consumed") is False
        and metadata.get("confirmation_outcomes_consumed") is False
        and metadata.get("confirmation_unread_during_selection") is True
        and metadata.get("deployment_authorized") is False
        and metadata.get("controller_authorized") is False
        and metadata.get("quarantined") is True
        and metadata.get("base_anchor_fields_bit_exact") is True
        and prototypes.shape == (owners.numel(), stable_features.shape[1])
        and owners.numel() > 0
        and bool(torch.isfinite(prototypes).all())
        and int(owners.min()) >= 0
        and int(owners.max()) < stable_features.shape[0]
        and torch.allclose(prototypes.norm(dim=1), torch.ones(owners.numel()), rtol=1e-5, atol=1e-6)
        and int(metadata.get("added_prototype_count", -1)) == owners.numel()
        and metadata.get("prototype_features_sha256") == tensor_sha256(prototypes)
        and metadata.get("prototype_owner_rows_sha256") == tensor_sha256(owners)
    ):
        raise ValueError("V21 control-selected candidate contract is invalid")
    inputs = metadata.get("inputs")
    if not isinstance(inputs, Mapping) or inputs.get("confirmation_caches") != []:
        raise ValueError("V21 control-selected candidate confirmation lineage is invalid")
    for field in ("stable_map", "parent_adaptation_candidate", "split_manifest"):
        _source_identity(inputs.get(field), label=f"selected {field}")
    controls = inputs.get("control_caches")
    adaptations = inputs.get("adaptation_caches")
    if not isinstance(controls, list) or not controls or not isinstance(adaptations, list) or not adaptations:
        raise ValueError("V21 control-selected candidate cache lineage is incomplete")
    control_ids = {_source_identity(value, label="selected control cache") for value in controls}
    adaptation_ids = {_source_identity(value, label="selected adaptation cache") for value in adaptations}
    if len(control_ids) != len(controls) or len(adaptation_ids) != len(adaptations) or control_ids & adaptation_ids:
        raise ValueError("V21 control-selected candidate cache lineage overlaps")
    contract = metadata.get("matching_contract")
    threshold = contract.get("prototype_activation_threshold") if isinstance(contract, Mapping) else "missing"
    if not (
        metadata.get("matching_semantics") == MATCHING_SEMANTICS
        and isinstance(contract, Mapping)
        and contract.get("prototype_activation_score") == "absolute_cosine"
        and (threshold is None or 0.0 <= float(threshold) <= 1.0)
        and contract.get("threshold_comparison") == "greater_than_or_equal"
        and contract.get("base_winner_comparison") == "prototype_score_strictly_greater_than_base_score"
    ):
        raise ValueError("V21 control-selected candidate matching contract is invalid")
    validate_baseline_contract(metadata.get("baseline_contract"))
    actions = metadata.get("selected_actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("V21 control-selected candidate action registry is empty")
    seen = []
    parent_indices = []
    cursor = 0
    for action in actions:
        indices = _as_cpu(action.get("prototype_indices"), dtype=torch.long).reshape(-1)
        source_indices = _as_cpu(action.get("parent_prototype_indices"), dtype=torch.long).reshape(-1)
        action_owners = _as_cpu(action.get("owner_anchor_rows"), dtype=torch.long).reshape(-1)
        if not (
            indices.tolist() == list(range(cursor, cursor + indices.numel()))
            and indices.shape == source_indices.shape == action_owners.shape
            and indices.numel() > 0
            and torch.equal(owners[indices], action_owners)
            and action.get("prototype_features_sha256") == tensor_sha256(prototypes[indices])
            and action.get("prototype_owner_rows_sha256") == tensor_sha256(action_owners)
        ):
            raise ValueError("V21 control-selected atomic action is invalid")
        cursor += indices.numel()
        seen.extend(indices.tolist())
        parent_indices.extend(source_indices.tolist())
    if seen != list(range(owners.numel())) or len(set(parent_indices)) != len(parent_indices):
        raise ValueError("V21 control-selected prototype registry is incomplete")
    summary = metadata.get("selected_control_summary")
    records = metadata.get("selected_control_records")
    search_audit = metadata.get("search_audit")
    if not isinstance(search_audit, Mapping):
        raise ValueError("V21 control-selected search audit is missing")
    validate_control_subset_search_audit(search_audit)
    if not (
        isinstance(records, list)
        and isinstance(summary, Mapping)
        and dict(summary) == summarize_cached_evaluation(records)
        and int(summary.get("paired_r5_gain_count", 0)) >= int(metadata["search"]["minimum_paired_r5_gain"])
        and int(summary.get("paired_r5_loss_count", -1)) == 0
        and int(summary.get("catastrophe_count", -1)) == 0
        and metadata.get("selection_objective", "").startswith("lexicographic_max_paired_r5_gain")
        and metadata.get("search", {}).get("algorithm") == SEARCH_ALGORITHM
        and metadata.get("search", {}).get("exact_poselib_for_every_evaluated_subset") is True
        and search_audit.get("decision") == "GO_SELECTED_ACTION"
    ):
        raise ValueError("V21 control-selected exact outcome contract is invalid")
    if parent_candidate is not None:
        parent_metadata = validate_candidate_map(parent_candidate, stable_map=stable_map)
        parent_prototypes = torch.as_tensor(parent_candidate[PROTOTYPE_FEATURE_FIELD]).float()
        parent_owners = torch.as_tensor(parent_candidate[PROTOTYPE_OWNER_FIELD]).long()
        source_indices = torch.tensor(parent_indices, dtype=torch.long)
        if not (
            torch.equal(prototypes, parent_prototypes[source_indices])
            and torch.equal(owners, parent_owners[source_indices])
            and int(metadata["parent_action_count"]) == len(parent_metadata["selected_actions"])
            and int(metadata["parent_prototype_count"]) == parent_owners.numel()
        ):
            raise ValueError("V21 selected candidate is not an exact parent action subset")
    return dict(metadata)
