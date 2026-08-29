#!/usr/bin/env python3
"""Run bounded competition-aware reverse pruning on frozen design queries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.v16_competitive_sufficiency import (
    certify_topl_relations,
    competitive_reserve_state,
    reserve_transition_is_safe,
)
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


def _pose_for_state(query: dict, state: dict, anchor_xyz: torch.Tensor) -> dict:
    return standard_pose_replay(
        keypoints=query["keypoints"],
        anchor_rows=state["winner_anchor_rows"],
        anchor_xyz=anchor_xyz,
        intrinsic=query["intrinsic"],
        ground_truth_w2c=query["pose_w2c"],
    )


def _serializable_state(state: dict) -> dict:
    keep = (
        "topl_exhausted",
        "winner_anchor_rows",
        "winner_scores",
        "winner_certified_positive",
        "best_noncertified_scores",
        "safe_positive_count_per_row",
        "anchor_unique_safe_query_rows",
        "anchor_unique_safe_anchor_rows",
        "anchor_unique_safe_count",
        "spatial_cell_count",
        "pose_logdet",
        "pose_minimum_eigenvalue",
        "margin_delta",
    )
    return {name: state[name] for name in keep}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-batch", type=Path, required=True)
    parser.add_argument("--baseline-map", type=Path, required=True)
    parser.add_argument("--mapping-evidence", type=Path, required=True)
    parser.add_argument("--removal-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--margin-delta", type=float, default=0.005)
    parser.add_argument("--maximum-actions", type=int, default=128)
    parser.add_argument("--maximum-rounds", type=int, default=3)
    parser.add_argument("--maximum-single-regression", type=float, default=0.25)
    parser.add_argument("--minimum-improving-families", type=int, default=2)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.margin_delta < 0 or args.maximum_actions < 1 or args.maximum_rounds < 1:
        parser.error("invalid competition controller bounds")

    design_path = args.design_batch.resolve()
    map_path = args.baseline_map.resolve()
    evidence_path = args.mapping_evidence.resolve()
    audit_path = args.removal_audit.resolve()
    design = json.loads(design_path.read_text())
    if not (
        design.get("schema") == "lafgs_v9_no_loo_causal_feedback_batch"
        and design.get("role") == "controller_design"
        and design.get("uses_test_queries") is False
        and design.get("loo_used") is False
        and design.get("accepted_query_row_policy") == "v2_row_valid_only"
    ):
        raise ValueError("V16 requires frozen no-test V2-valid design records")
    baseline = torch.load(map_path, map_location="cpu", weights_only=False)
    evidence = torch.load(evidence_path, map_location="cpu", weights_only=False)
    audit = torch.load(audit_path, map_location="cpu", weights_only=False)
    count = int(torch.as_tensor(baseline["anchor_ids"]).numel())
    if not (
        evidence.get("schema") == "lafgs_v7_reconstructed_mapping_candidate_evidence"
        and int(evidence.get("candidate_count", -1)) == count
        and audit.get("schema") == "lafgs_v9_actual_removal_gain_audit"
        and audit.get("loo_used") is False
    ):
        raise ValueError("V16 frozen map evidence/audit contract differs")
    xyz = torch.as_tensor(baseline["anchor_xyz"]).float()
    active = torch.ones(count, dtype=torch.bool)
    candidate_order = torch.as_tensor(audit["authorized_anchor_rows"]).long().tolist()

    queries = []
    candidate_to_queries: dict[int, set[int]] = {
        int(anchor): set() for anchor in candidate_order
    }
    for item in design["records"]:
        record_path = Path(item["path"]).resolve()
        if sha256_file(record_path) != item["sha256"]:
            raise ValueError("V16 design record SHA256 differs")
        observed = torch.load(record_path, map_location="cpu", weights_only=False)
        if observed["certificate_decision"] != "ACCEPT":
            continue
        source_path = Path(observed["source_record"]).resolve()
        if sha256_file(source_path) != observed["source_record_sha256"]:
            raise ValueError("V16 certified source record SHA256 differs")
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        rows = torch.as_tensor(observed["source_query_rows"]).long()
        keypoints = torch.as_tensor(source["keypoints"])[rows].float() + 0.5
        candidates = torch.as_tensor(observed["topk_anchor_rows"]).long()
        scores = torch.as_tensor(observed["topk_scores"]).float()
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
        query = {
            "query_index": int(observed["query_index"]),
            "pose_family_id": int(observed["pose_family_id"]),
            "keypoints": keypoints,
            "candidate_anchor_rows": candidates,
            "candidate_scores": scores,
            "certified_positive": relations["positive"],
            "ambiguous": relations["ambiguous"],
            "intrinsic": torch.as_tensor(source["intrinsics"]).float(),
            "pose_w2c": torch.as_tensor(source["pose_w2c"]).float(),
            "image_hw": torch.as_tensor(source["image_hw"]).long(),
        }
        query["state"] = competitive_reserve_state(
            candidate_anchor_rows=candidates,
            candidate_scores=scores,
            certified_positive=relations["positive"],
            active_anchor_mask=active,
            keypoints=keypoints,
            anchor_xyz=xyz,
            intrinsic=query["intrinsic"],
            pose_w2c=query["pose_w2c"],
            image_hw=query["image_hw"],
            margin_delta=args.margin_delta,
        )
        query["pose"] = _pose_for_state(query, query["state"], xyz)
        observer_pose = observed["baseline"]
        if (
            abs(float(query["pose"]["task_error"]) - float(observer_pose["task_error"]))
            > 1e-4
        ):
            raise ValueError("V16 cached Top-L does not reproduce the frozen plant")
        query_row = len(queries)
        present = set(torch.unique(candidates).tolist()) & set(candidate_order)
        for anchor in present:
            candidate_to_queries[int(anchor)].add(query_row)
        queries.append(query)
        if len(queries) % 8 == 0:
            print(f"V16 competitive state: {len(queries)} ACCEPT queries", flush=True)

    competition_cache = {
        "schema": "lafgs_v16_topl_competitive_state",
        "version": 1,
        "uses_test_queries": False,
        "loo_used": False,
        "topl": 64,
        "margin_delta": float(args.margin_delta),
        "queries": [
            {
                **{key: value for key, value in query.items() if key not in {"state", "pose"}},
                "initial_state": _serializable_state(query["state"]),
                "initial_pose": query["pose"],
            }
            for query in queries
        ],
    }
    args.output_dir.mkdir(parents=True)
    cache_path = args.output_dir / "competitive_state.pt"
    _save(competition_cache, cache_path)

    actions = []
    accepted: list[int] = []
    remaining = list(candidate_order)
    for round_index in range(args.maximum_rounds):
        accepted_this_round = 0
        next_remaining = []
        for anchor in remaining:
            if len(accepted) >= args.maximum_actions:
                next_remaining.append(anchor)
                continue
            query_indices = sorted(candidate_to_queries.get(int(anchor), ()))
            influential = []
            for query_index in query_indices:
                query = queries[query_index]
                state = query["state"]
                candidates = query["candidate_anchor_rows"]
                occurrence = candidates == int(anchor)
                winner = state["winner_anchor_rows"] == int(anchor)
                safe = state["safe_edge_mask"] & occurrence
                best_wrong = occurrence & ~query["certified_positive"] & (
                    query["candidate_scores"]
                    >= state["best_noncertified_scores"][:, None] - 1e-6
                )
                if bool(winner.any() or safe.any() or best_wrong.any()):
                    influential.append(query_index)
            if not influential:
                actions.append(
                    {"anchor_row": anchor, "round": round_index, "accepted": False, "reason": "no_competitive_state_effect"}
                )
                continue

            active[anchor] = False
            trial_states = {}
            unsafe = []
            for query_index in influential:
                query = queries[query_index]
                trial = competitive_reserve_state(
                    candidate_anchor_rows=query["candidate_anchor_rows"],
                    candidate_scores=query["candidate_scores"],
                    certified_positive=query["certified_positive"],
                    active_anchor_mask=active,
                    keypoints=query["keypoints"],
                    anchor_xyz=xyz,
                    intrinsic=query["intrinsic"],
                    pose_w2c=query["pose_w2c"],
                    image_hw=query["image_hw"],
                    margin_delta=args.margin_delta,
                )
                safe, reasons = reserve_transition_is_safe(query["state"], trial)
                trial_states[query_index] = trial
                if not safe:
                    unsafe.append(
                        {"query_index": query["query_index"], "reasons": reasons}
                    )
            if unsafe:
                active[anchor] = True
                next_remaining.append(anchor)
                actions.append(
                    {
                        "anchor_row": anchor,
                        "round": round_index,
                        "accepted": False,
                        "reason": "novel_view_reserve_violation",
                        "unsafe_queries": unsafe,
                    }
                )
                continue

            changed = [
                query_index
                for query_index in influential
                if not torch.equal(
                    queries[query_index]["state"]["winner_anchor_rows"],
                    trial_states[query_index]["winner_anchor_rows"],
                )
            ]
            if not changed:
                active[anchor] = True
                actions.append(
                    {"anchor_row": anchor, "round": round_index, "accepted": False, "reason": "no_deployed_top1_effect"}
                )
                continue
            trial_poses = {}
            gains = []
            lost = 0
            improving_families = set()
            maximum_regression = 0.0
            for query_index in changed:
                query = queries[query_index]
                trial_pose = _pose_for_state(query, trial_states[query_index], xyz)
                trial_poses[query_index] = trial_pose
                gain = max(-4.0, min(4.0, float(query["pose"]["task_error"]) - float(trial_pose["task_error"])))
                gains.append(gain)
                if gain > 0:
                    improving_families.add(query["pose_family_id"])
                maximum_regression = max(maximum_regression, -gain)
                lost += int(_success(query["pose"]) and not _success(trial_pose))
            accepted_action = bool(
                sum(gains) > 0.0
                and len(improving_families) >= args.minimum_improving_families
                and lost == 0
                and maximum_regression <= args.maximum_single_regression
            )
            action = {
                "anchor_row": anchor,
                "round": round_index,
                "accepted": accepted_action,
                "reason": "accepted_exact_response" if accepted_action else "task_response_gate",
                "influential_query_count": len(influential),
                "changed_top1_query_count": len(changed),
                "improving_pose_family_count": len(improving_families),
                "bounded_cumulative_task_gain": float(sum(gains)),
                "maximum_capped_regression": float(maximum_regression),
                "lost_success_count": int(lost),
            }
            actions.append(action)
            if accepted_action:
                accepted.append(anchor)
                accepted_this_round += 1
                for query_index in influential:
                    queries[query_index]["state"] = trial_states[query_index]
                for query_index, pose in trial_poses.items():
                    queries[query_index]["pose"] = pose
            else:
                active[anchor] = True
                next_remaining.append(anchor)
        print(
            f"V16 controller round {round_index}: accepted {accepted_this_round}, total {len(accepted)}",
            flush=True,
        )
        if not accepted_this_round or len(accepted) >= args.maximum_actions:
            break
        remaining = next_remaining

    selected_rows = torch.nonzero(active, as_tuple=False).reshape(-1)
    selected_map = _materialize_selected_map(baseline, selected_rows)
    selected_map["provenance"] = {
        **dict(selected_map.get("provenance", {})),
        "v16_competition_aware_controller": True,
        "v16_design_only": True,
        "v16_topl": 64,
        "v16_margin_delta": float(args.margin_delta),
        "uses_test_queries": False,
    }
    map_output = args.output_dir / "projective_anchor_map.pt"
    _save(selected_map, map_output)
    audit_output = args.output_dir / "controller_audit.pt"
    _save(
        {
            "schema": "lafgs_v16_competition_aware_controller_audit",
            "version": 1,
            "uses_test_queries": False,
            "loo_used": False,
            "active_set_only": True,
            "accepted_anchor_rows": torch.tensor(accepted, dtype=torch.long),
            "selected_anchor_rows": selected_rows,
            "actions": actions,
        },
        audit_output,
    )
    report = {
        "schema": "lafgs_v16_competition_aware_controller_report",
        "version": 1,
        "uses_test_queries": False,
        "loo_used": False,
        "arm": "v16_competitive_pruning",
        "baseline_anchor_count": count,
        "selected_anchor_count": int(selected_rows.numel()),
        "removed_anchor_count": len(accepted),
        "compression_fraction": float(len(accepted) / count),
        "design_accept_query_count": len(queries),
        "candidate_action_count": len(candidate_order),
        "map": str(map_output.resolve()),
        "map_sha256": sha256_file(map_output),
        "controller_audit": str(audit_output.resolve()),
        "controller_audit_sha256": sha256_file(audit_output),
        "competitive_state": str(cache_path.resolve()),
        "competitive_state_sha256": sha256_file(cache_path),
        "inputs": {
            "design_batch": str(design_path),
            "design_batch_sha256": sha256_file(design_path),
            "baseline_map": str(map_path),
            "baseline_map_sha256": sha256_file(map_path),
            "mapping_evidence": str(evidence_path),
            "mapping_evidence_sha256": sha256_file(evidence_path),
            "removal_audit": str(audit_path),
            "removal_audit_sha256": sha256_file(audit_path),
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
