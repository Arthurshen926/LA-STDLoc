#!/usr/bin/env python3
"""Run greedy truth-aware delete/compression control with safety floors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.v16_competitive_sufficiency import (
    certify_topl_relations,
    competitive_reserve_state,
    reserve_transition_is_safe,
)
from map_learning.v18_active_set_controller import (
    propose_current_competition_actions,
    propose_truth_reactivation_actions,
)
from map_learning.v18_provenance_truth import truth_membership_mask
from map_learning.v9_causal_feedback import standard_pose_replay
from scripts.run_v7_closed_loop import _materialize_selected_map


def _save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _success(pose: dict) -> bool:
    return bool(
        float(pose["translation_error_cm"]) < 5.0
        and float(pose["rotation_error_deg"]) < 5.0
    )


def _pose(query: dict, state: dict, xyz: torch.Tensor) -> dict:
    retained = ~torch.as_tensor(state["topl_exhausted"]).bool()
    return standard_pose_replay(
        keypoints=query["keypoints"][retained],
        anchor_rows=state["winner_anchor_rows"][retained],
        anchor_xyz=xyz,
        intrinsic=query["intrinsic"],
        ground_truth_w2c=query["pose_w2c"],
    )


def _state(query: dict, active: torch.Tensor, xyz: torch.Tensor, margin: float) -> dict:
    return competitive_reserve_state(
        candidate_anchor_rows=query["candidate_anchor_rows"],
        candidate_scores=query["candidate_scores"],
        certified_positive=query["certified_positive"],
        active_anchor_mask=active,
        keypoints=query["keypoints"],
        anchor_xyz=xyz,
        intrinsic=query["intrinsic"],
        pose_w2c=query["pose_w2c"],
        image_hw=query["image_hw"],
        margin_delta=float(margin),
        certification_confidence=query["certification_confidence"],
        measurement_covariance_px2=query["measurement_covariance_px2"],
    )


def _proposal_records(queries: list[dict]) -> list[dict]:
    return [
        {
            "query_index": query["query_index"],
            "pose_family_id": query["pose_family_id"],
            "candidate_anchor_rows": query["candidate_anchor_rows"],
            "candidate_scores": query["candidate_scores"],
            "current_winner_anchor_rows": query["state"]["winner_anchor_rows"],
            "current_winner_available": ~query["state"]["topl_exhausted"],
            "truth": query["truth"],
        }
        for query in queries
    ]


def _family_fold(family: int, fold_count: int) -> int:
    digest = hashlib.sha256(f"v18-crossfit:{int(family)}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % int(fold_count)


def _action_proposals(
    records: list[dict],
    *,
    active: torch.Tensor,
    anchor_count: int,
    maximum_inactive: int,
) -> list[dict]:
    deletions = propose_current_competition_actions(
        records=records,
        anchor_count=anchor_count,
        active_anchor_mask=active,
        maximum_inactive_redundancy_candidates=maximum_inactive,
    )["proposals"]
    reactivations = propose_truth_reactivation_actions(
        records=records,
        anchor_count=anchor_count,
        active_anchor_mask=active,
    )["proposals"]
    proposals = deletions + reactivations
    kind_priority = {
        "truth_reactivation": 0,
        "harmful_removal": 1,
        "dominated_removal": 2,
        "inactive_redundancy_probe": 3,
    }
    proposals.sort(
        key=lambda item: (
            kind_priority[item["kind"]],
            -int(item["evidence_pose_family_count"]),
            -int(item["evidence_query_count"]),
            int(item["anchor_row"]),
        )
    )
    return proposals


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-batch", type=Path, required=True)
    parser.add_argument("--feedback-truth", type=Path, required=True)
    parser.add_argument(
        "--allow-certified-topl-fallback",
        action="store_true",
        help="explicitly use legacy V16 Top-L geometric truth after provenance gate failure",
    )
    parser.add_argument("--baseline-map", type=Path, required=True)
    parser.add_argument(
        "--initial-active-map",
        type=Path,
        help="optional prior-round delete-only subset; inactive Anchors remain reversible",
    )
    parser.add_argument("--mapping-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-actions", type=int, default=512)
    parser.add_argument("--maximum-proposals-per-step", type=int, default=512)
    parser.add_argument("--maximum-inactive-proposals", type=int, default=2048)
    parser.add_argument("--margin-delta", type=float, default=0.005)
    parser.add_argument("--minimum-safe-correspondences", type=int, default=12)
    parser.add_argument("--minimum-spatial-cells", type=int, default=6)
    parser.add_argument("--maximum-pose-logdet-drop", type=float, default=0.5)
    parser.add_argument("--minimum-pose-eigenvalue-retention", type=float, default=0.8)
    parser.add_argument("--minimum-effective-correspondences", type=float, default=12.0)
    parser.add_argument("--minimum-improving-families", type=int, default=2)
    parser.add_argument("--maximum-single-regression", type=float, default=0.25)
    parser.add_argument("--map-size-reward", type=float, default=1e-4)
    parser.add_argument("--crossfit-folds", type=int, default=3)
    parser.add_argument("--minimum-crossfit-support", type=int, default=2)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if min(args.maximum_actions, args.maximum_proposals_per_step) < 1:
        parser.error("V18 action/proposal bounds must be positive")
    if not 0.0 <= args.minimum_pose_eigenvalue_retention <= 1.0:
        parser.error("V18 eigenvalue retention must lie in [0, 1]")
    if not 2 <= int(args.minimum_crossfit_support) <= int(args.crossfit_folds):
        parser.error("V18 cross-fit support must lie in [2, crossfit-folds]")

    design = json.loads(args.design_batch.read_text())
    truth_batch = torch.load(args.feedback_truth, map_location="cpu", weights_only=False)
    baseline = torch.load(args.baseline_map, map_location="cpu", weights_only=False)
    evidence = torch.load(args.mapping_evidence, map_location="cpu", weights_only=False)
    common_contract = bool(
        design.get("schema") == "lafgs_v9_no_loo_causal_feedback_batch"
        and design.get("role") == "controller_design"
        and design.get("uses_test_queries") is False
        and design.get("loo_used") is False
        and truth_batch.get("uses_test_queries") is False
        and truth_batch.get("loo_used") is False
    )
    provenance_contract = bool(
        truth_batch.get("schema") == "lafgs_v18_feedback_provenance_truth_batch"
        and truth_batch.get("full_gaussian_prior_evaluated") is True
        and truth_batch.get("full_depth_ordered_compositing") is True
        and truth_batch.get("full_provenance_anchor_enumeration") is True
        and float(truth_batch.get("minimum_retained_composition_mass", 0.0))
        >= 0.95
        and truth_batch.get("controller_replacement_authorized_by_mapping_validation")
        is True
    )
    fallback_contract = bool(
        args.allow_certified_topl_fallback
        and truth_batch.get("schema")
        == "lafgs_v18_feedback_certified_projection_truth_batch"
        and truth_batch.get("explicit_fallback_required") is True
        and truth_batch.get("descriptor_independent_full_map_truth") is False
        and truth_batch.get("controller_replacement_authorized") is False
    )
    if not (common_contract and (provenance_contract or fallback_contract)):
        raise ValueError("V18 truth-aware controller contract/gate differs")
    count = int(torch.as_tensor(baseline["anchor_ids"]).numel())
    if not (
        evidence.get("schema") == "lafgs_v7_reconstructed_mapping_candidate_evidence"
        and int(evidence.get("candidate_count", -1)) == count
    ):
        raise ValueError("V18 mapping evidence differs from the baseline map")
    truth_by_query = {
        int(record["query_index"]): record for record in truth_batch["records"]
    }
    baseline_ids = torch.as_tensor(baseline["anchor_ids"]).long()
    xyz = torch.as_tensor(baseline["anchor_xyz"]).float()
    active = torch.ones(count, dtype=torch.bool)
    initial_active_path = None
    if args.initial_active_map is not None:
        initial_active_path = args.initial_active_map.resolve()
        initial_state = torch.load(
            initial_active_path, map_location="cpu", weights_only=False
        )
        initial_ids = torch.as_tensor(initial_state["anchor_ids"]).long()
        if not bool(torch.isin(initial_ids, baseline_ids).all()):
            raise ValueError("V18 initial active map is not a baseline subset")
        active = torch.isin(baseline_ids, initial_ids)
    initial_active = active.clone()
    queries = []
    for item in design["records"]:
        observed = torch.load(item["path"], map_location="cpu", weights_only=False)
        if observed["certificate_decision"] != "ACCEPT":
            continue
        truth_record = truth_by_query.get(int(observed["query_index"]))
        if truth_record is None:
            raise ValueError("V18 feedback truth does not cover every ACCEPT design query")
        source_path = Path(observed["source_record"]).resolve()
        if sha256_file(source_path) != observed["source_record_sha256"]:
            raise ValueError("V18 source render SHA256 differs")
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        rows = torch.as_tensor(observed["source_query_rows"]).long()
        if not torch.equal(rows, torch.as_tensor(truth_record["source_query_rows"]).long()):
            raise ValueError("V18 truth and competition query rows differ")
        keypoints = torch.as_tensor(source["keypoints"])[rows].float() + 0.5
        candidates = torch.as_tensor(observed["topk_anchor_rows"]).long()
        scores = torch.as_tensor(observed["topk_scores"]).float()
        truth = truth_record["truth"]
        positive = truth_membership_mask(truth, candidates)
        relations = certify_topl_relations(
            keypoints=keypoints,
            candidate_anchor_rows=candidates,
            anchor_xyz=xyz,
            anchor_covariance=baseline["anchor_position_covariance"],
            observation_count=evidence["observation_count"],
            view_family_count=evidence["view_family_count"],
            pose_w2c=source["pose_w2c"],
            intrinsic=source["intrinsics"],
            alpha=source["alpha_float16"],
            depth=source["depth_float16"],
            surface_median_depth=source.get("surface_median_depth_float16"),
            row_valid=torch.ones(rows.numel(), dtype=torch.bool),
        )
        confidence = torch.exp(
            -0.5
            * (
                relations["reprojection_mahalanobis_squared"] / 9.210340371976184
                + relations["depth_mahalanobis_squared"] / 6.6348966010212145
            )
        ).clamp(0.0, 1.0)
        confidence = confidence.masked_fill(~positive, 0.0)
        query = {
            "query_index": int(observed["query_index"]),
            "pose_family_id": int(observed["pose_family_id"]),
            "keypoints": keypoints,
            "candidate_anchor_rows": candidates,
            "candidate_scores": scores,
            "certified_positive": positive,
            "certification_confidence": confidence,
            "measurement_covariance_px2": relations["measurement_covariance_px2"],
            "intrinsic": torch.as_tensor(source["intrinsics"]).float(),
            "pose_w2c": torch.as_tensor(source["pose_w2c"]).float(),
            "image_hw": torch.as_tensor(source["image_hw"]).long(),
            "truth": truth,
        }
        query["state"] = _state(query, active, xyz, args.margin_delta)
        query["pose"] = _pose(query, query["state"], xyz)
        if bool(initial_active.all()) and abs(
            float(query["pose"]["task_error"])
            - float(observed["baseline"]["task_error"])
        ) > 1e-4:
            raise ValueError("V18 current competition does not reproduce the frozen plant")
        queries.append(query)
        if len(queries) % 10 == 0:
            print(
                json.dumps(
                    {
                        "stage": "initialize_competitive_state",
                        "accepted_queries": len(queries),
                    }
                ),
                flush=True,
            )

    actions = []
    proposal_snapshots = []
    for step in range(int(args.maximum_actions)):
        current_records = _proposal_records(queries)
        proposals = _action_proposals(
            current_records,
            active=active,
            anchor_count=count,
            maximum_inactive=int(args.maximum_inactive_proposals),
        )
        support: dict[tuple[int, str], list[int]] = defaultdict(list)
        for heldout_fold in range(int(args.crossfit_folds)):
            training_records = [
                record
                for record in current_records
                if _family_fold(record["pose_family_id"], args.crossfit_folds)
                != heldout_fold
            ]
            for proposal in _action_proposals(
                training_records,
                active=active,
                anchor_count=count,
                maximum_inactive=int(args.maximum_inactive_proposals),
            ):
                support[(int(proposal["anchor_row"]), str(proposal["kind"]))].append(
                    heldout_fold
                )
        for proposal in proposals:
            proposal["crossfit_proposal_folds"] = support.get(
                (int(proposal["anchor_row"]), str(proposal["kind"])), []
            )
        proposals = [
            proposal
            for proposal in proposals
            if len(proposal["crossfit_proposal_folds"])
            >= int(args.minimum_crossfit_support)
        ][: int(args.maximum_proposals_per_step)]
        kind_counts = {
            kind: sum(proposal["kind"] == kind for proposal in proposals)
            for kind in (
                "harmful_removal",
                "dominated_removal",
                "inactive_redundancy_probe",
                "truth_reactivation",
            )
        }
        proposal_snapshots.append(
            {
                "step": step,
                "proposal_count": len(proposals),
                "evaluated_proposal_count": len(proposals),
                "kind_counts": kind_counts,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "evaluate_action_proposals",
                    "step": step,
                    "proposal_count": len(proposals),
                    "kind_counts": kind_counts,
                }
            ),
            flush=True,
        )
        if not proposals:
            break
        evaluations = []
        for proposal_index, proposal in enumerate(proposals, start=1):
            anchor = int(proposal["anchor_row"])
            kind = str(proposal["kind"])
            trial_active_value = kind == "truth_reactivation"
            previous_active_value = bool(active[anchor])
            affected = [
                index
                for index, query in enumerate(queries)
                if bool((query["candidate_anchor_rows"] == anchor).any())
            ]
            active[anchor] = trial_active_value
            trial_states = {}
            unsafe = []
            for query_index in affected:
                query = queries[query_index]
                trial = _state(query, active, xyz, args.margin_delta)
                safe, reasons = reserve_transition_is_safe(
                    query["state"],
                    trial,
                    minimum_anchor_unique_safe_count=args.minimum_safe_correspondences,
                    minimum_spatial_cell_count=args.minimum_spatial_cells,
                    maximum_pose_logdet_drop=args.maximum_pose_logdet_drop,
                    minimum_pose_eigenvalue_retention=args.minimum_pose_eigenvalue_retention,
                    minimum_effective_correspondence_count=args.minimum_effective_correspondences,
                )
                trial_states[query_index] = trial
                if not safe:
                    unsafe.append(
                        {"query_index": query["query_index"], "reasons": reasons}
                    )
            active[anchor] = previous_active_value
            changed = [
                index
                for index in affected
                if not torch.equal(
                    queries[index]["state"]["winner_anchor_rows"],
                    trial_states[index]["winner_anchor_rows"],
                )
            ]
            trial_poses = {}
            gains = []
            fold_gains = defaultdict(float)
            families = set()
            lost = 0
            maximum_regression = 0.0
            if not unsafe:
                for query_index in changed:
                    query = queries[query_index]
                    trial_pose = _pose(query, trial_states[query_index], xyz)
                    trial_poses[query_index] = trial_pose
                    gain = max(
                        -4.0,
                        min(
                            4.0,
                            float(query["pose"]["task_error"])
                            - float(trial_pose["task_error"]),
                        ),
                    )
                    gains.append(gain)
                    fold = _family_fold(
                        query["pose_family_id"], int(args.crossfit_folds)
                    )
                    fold_gains[fold] += gain
                    if gain > 0.0:
                        families.add(query["pose_family_id"])
                    maximum_regression = max(maximum_regression, -gain)
                    lost += int(_success(query["pose"]) and not _success(trial_pose))
            cumulative = float(sum(gains))
            heldout_positive_folds = [
                fold
                for fold in proposal["crossfit_proposal_folds"]
                if fold_gains[fold] > 0.0
            ]
            exact_positive = bool(
                changed
                and cumulative > 0.0
                and len(families) >= int(args.minimum_improving_families)
                and len(heldout_positive_folds)
                >= int(args.minimum_crossfit_support)
                and lost == 0
                and maximum_regression <= float(args.maximum_single_regression)
            )
            zero_effect_compression = bool(
                not changed
                and kind in {"dominated_removal", "inactive_redundancy_probe"}
            )
            accepted = bool(not unsafe and (exact_positive or zero_effect_compression))
            evaluations.append(
                {
                    **proposal,
                    "accepted": accepted,
                    "unsafe_queries": unsafe,
                    "affected_query_count": len(affected),
                    "changed_top1_query_count": len(changed),
                    "improving_pose_family_count": len(families),
                    "heldout_positive_crossfit_folds": heldout_positive_folds,
                    "heldout_fold_task_gains": dict(fold_gains),
                    "bounded_cumulative_task_gain": cumulative,
                    "maximum_capped_regression": maximum_regression,
                    "lost_success_count": lost,
                    "objective": (
                        cumulative
                        + (
                            -float(args.map_size_reward)
                            if kind == "truth_reactivation"
                            else float(args.map_size_reward)
                        )
                        if accepted
                        else -float("inf")
                    ),
                    "trial_states": trial_states,
                    "trial_poses": trial_poses,
                }
            )
            if proposal_index % 16 == 0 or proposal_index == len(proposals):
                print(
                    json.dumps(
                        {
                            "stage": "evaluate_action_proposals",
                            "step": step,
                            "completed": proposal_index,
                            "total": len(proposals),
                        }
                    ),
                    flush=True,
                )
        accepted_evaluations = [item for item in evaluations if item["accepted"]]
        if not accepted_evaluations:
            actions.extend(
                {key: value for key, value in item.items() if key not in {"trial_states", "trial_poses"}}
                for item in evaluations
            )
            break
        selected = max(
            accepted_evaluations,
            key=lambda item: (
                item["objective"],
                item["improving_pose_family_count"],
                -item["anchor_row"],
            ),
        )
        anchor = int(selected["anchor_row"])
        active[anchor] = selected["kind"] == "truth_reactivation"
        for query_index, state in selected["trial_states"].items():
            queries[query_index]["state"] = state
        for query_index, pose in selected["trial_poses"].items():
            queries[query_index]["pose"] = pose
        actions.append(
            {
                key: value
                for key, value in selected.items()
                if key not in {"trial_states", "trial_poses"}
            }
        )
        print(
            json.dumps(
                {
                    "step": step,
                    "accepted_anchor": anchor,
                    "kind": selected["kind"],
                    "task_gain": selected["bounded_cumulative_task_gain"],
                    "active_count": int(active.sum()),
                }
            ),
            flush=True,
        )

    selected_rows = torch.nonzero(active, as_tuple=False).reshape(-1)
    selected_map = _materialize_selected_map(baseline, selected_rows)
    selected_map["provenance"] = {
        **dict(selected_map.get("provenance", {})),
        "v18_provenance_truth_controller": bool(provenance_contract),
        "v18_truth_sha256": sha256_file(args.feedback_truth),
        "v18_current_competition_proposals": True,
        "v18_safety_floor_reserve": True,
        "v18_greedy_largest_task_gain_first": True,
        "v18_pose_family_crossfit": True,
        "v18_reversible_reactivation": True,
        "uses_test_queries": False,
        "v18_truth_source": truth_batch.get("truth_source", "gaussian_provenance"),
        "v18_certified_topl_fallback": bool(fallback_contract),
    }
    args.output_dir.mkdir(parents=True)
    map_path = args.output_dir / "projective_anchor_map.pt"
    audit_path = args.output_dir / "controller_audit.pt"
    _save(selected_map, map_path)
    _save(
        {
            "schema": "lafgs_v18_truth_aware_controller_audit",
            "version": 1,
            "uses_test_queries": False,
            "loo_used": False,
            "initial_active_mask": initial_active,
            "active_mask": active,
            "selected_anchor_rows": selected_rows,
            "actions": actions,
            "proposal_snapshots": proposal_snapshots,
        },
        audit_path,
    )
    report = {
        "schema": "lafgs_v18_truth_aware_controller_report",
        "version": 1,
        "uses_test_queries": False,
        "loo_used": False,
        "truth_source": truth_batch.get("truth_source", "gaussian_provenance"),
        "certified_topl_fallback": bool(fallback_contract),
        "baseline_anchor_count": count,
        "initial_active_anchor_count": int(initial_active.sum()),
        "selected_anchor_count": int(selected_rows.numel()),
        "removed_anchor_count": int(count - selected_rows.numel()),
        "round_deactivated_anchor_count": int((initial_active & ~active).sum()),
        "round_reactivated_anchor_count": int((~initial_active & active).sum()),
        "compression_fraction": float((count - selected_rows.numel()) / count),
        "crossfit": {
            "fold_count": int(args.crossfit_folds),
            "minimum_proposal_and_positive_fold_support": int(
                args.minimum_crossfit_support
            ),
            "family_hash": "sha256(v18-crossfit:pose_family_id)",
        },
        "accepted_kind_counts": {
            kind: sum(item["kind"] == kind and item["accepted"] for item in actions)
            for kind in (
                "harmful_removal",
                "dominated_removal",
                "inactive_redundancy_probe",
                "truth_reactivation",
            )
        },
        "map": str(map_path.resolve()),
        "map_sha256": sha256_file(map_path),
        "controller_audit": str(audit_path.resolve()),
        "controller_audit_sha256": sha256_file(audit_path),
        "inputs": {
            "design_batch": str(args.design_batch.resolve()),
            "design_batch_sha256": sha256_file(args.design_batch),
            "feedback_truth": str(args.feedback_truth.resolve()),
            "feedback_truth_sha256": sha256_file(args.feedback_truth),
            "baseline_map": str(args.baseline_map.resolve()),
            "baseline_map_sha256": sha256_file(args.baseline_map),
            "mapping_evidence": str(args.mapping_evidence.resolve()),
            "mapping_evidence_sha256": sha256_file(args.mapping_evidence),
            "initial_active_map": (
                None if initial_active_path is None else str(initial_active_path)
            ),
            "initial_active_map_sha256": (
                None
                if initial_active_path is None
                else sha256_file(initial_active_path)
            ),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
