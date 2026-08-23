#!/usr/bin/env python3
"""Run fixed-stage V6 convergence without metric hard-gate acceptance."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from common.hashing import sha256_file


def _run(command: list[str], *, root: Path) -> None:
    subprocess.run(command, cwd=root, check=True)


def _evaluation_metadata(summary: dict, *, baseline: bool = False) -> dict:
    if baseline:
        policy = summary.get("contract", {}).get("affected_anchor_policy", "unknown")
        role = {
            "purge": "mapping_purged_holdout_baseline",
            "rebuild": "mapping_exact_rebuild_holdout_baseline",
        }.get(policy, f"mapping_{policy}_holdout_baseline")
    else:
        roles = []
        if summary.get("independent_mapping_validation_summary") is not None:
            roles.append("independent_mapping_validation")
        elif summary.get("descriptor_validation_summary") is not None:
            roles.append("descriptor_only_holdout_with_other_dependency_replay")
        if summary.get("descriptor_training_replay_summary") is not None:
            roles.append("descriptor_training_replay")
        if summary.get("reconstruction_target_replay_summary") is not None:
            roles.append("reconstruction_target_replay")
        if summary.get("selection_training_replay_summary") is not None:
            roles.append("selection_training_replay")
        role = "+".join(roles) if roles else "mapping_purged_holdout_candidate"
    return {
        "evaluation_role": role,
        "descriptor_validation_summary": summary.get(
            "descriptor_validation_summary"
        ),
        "independent_mapping_validation_summary": summary.get(
            "independent_mapping_validation_summary"
        ),
        "descriptor_training_replay_summary": summary.get(
            "descriptor_training_replay_summary"
        ),
        "descriptor_gradient_reuse_summary": summary.get(
            "descriptor_gradient_reuse_summary"
        ),
        "reconstruction_target_replay_summary": summary.get(
            "reconstruction_target_replay_summary"
        ),
        "selection_training_replay_summary": summary.get(
            "selection_training_replay_summary"
        ),
        "failure_layer_counts_are_overlapping": summary.get(
            "failure_layer_counts_are_overlapping"
        ),
        "failure_query_count": summary.get("failure_query_count"),
        "multi_layer_failure_query_count": summary.get(
            "multi_layer_failure_query_count"
        ),
        "contract": summary.get("contract"),
    }


def _proposal_metadata(proposal: dict) -> dict:
    output = proposal.get("output", {})
    return {
        "proposal_report": proposal.get("_report_path"),
        "proposal_report_sha256": proposal.get("_report_sha256"),
        "training_checkpoint_size_bytes": output.get(
            "training_checkpoint_size_bytes"
        ),
        "deployment_map_size_bytes": output.get("deployment_map_size_bytes"),
    }


def _evaluate(
    args: argparse.Namespace,
    *,
    root: Path,
    map_path: Path,
    map_sha: str,
    metric_path: Path,
    metric_sha: str,
    output: Path,
) -> tuple[Path, dict]:
    _run(
        [
            sys.executable,
            str(root / "scripts/evaluate_v6_self_localization.py"),
            "--map", str(map_path),
            "--expected-map-sha256", map_sha,
            "--metric", str(metric_path),
            "--expected-metric-sha256", metric_sha,
            "--observation-cache", str(args.observation_cache),
            "--expected-observation-cache-sha256",
            args.expected_observation_cache_sha256,
            "--output-dir", str(output),
            "--device", args.device,
            "--cpu-threads", str(args.cpu_threads),
            "--positive-radius-px", str(args.positive_radius_px),
            "--alpha-minimum", str(args.alpha_minimum),
            "--required-rank", str(args.required_rank),
            "--required-visibility-rank", str(args.required_visibility_rank),
            "--required-detectable-rank", str(args.required_detectable_rank),
            "--loo-pose-neighbors", str(args.loo_pose_neighbors),
            "--loo-affected-anchor-policy", args.loo_affected_anchor_policy,
            "--ransac-reprojection-px", str(args.ransac_reprojection_px),
            "--seed", str(args.seed),
        ],
        root=root,
    )
    summary_path = output / "summary.json"
    return summary_path, json.loads(summary_path.read_text())


def _propose(
    args: argparse.Namespace,
    *,
    root: Path,
    arm: str,
    map_path: Path,
    map_sha: str,
    feedback_summary: dict,
    output: Path,
) -> dict:
    command = [
        sys.executable,
        str(root / "scripts/propose_v6_round.py"),
        "--arm", arm,
        "--map", str(map_path),
        "--expected-map-sha256", map_sha,
        "--observation-cache", str(args.observation_cache),
        "--expected-observation-cache-sha256",
        args.expected_observation_cache_sha256,
        "--feedback", str(feedback_summary["feedback_path"]),
        "--expected-feedback-sha256", feedback_summary["feedback_sha256"],
        "--output-dir", str(output),
        "--device", args.device,
        "--descriptor-trust-region", str(args.descriptor_trust_region),
        "--descriptor-margin", str(args.descriptor_margin),
        "--descriptor-temperature", str(args.descriptor_temperature),
        "--descriptor-learning-rate", str(args.descriptor_learning_rate),
        "--descriptor-epochs", str(args.descriptor_epochs),
        "--descriptor-batch-size", str(args.descriptor_batch_size),
        "--descriptor-maximum-triplets-per-query",
        str(args.descriptor_maximum_triplets_per_query),
        "--descriptor-clean-fraction", str(args.descriptor_clean_fraction),
        "--descriptor-clean-weight", str(args.descriptor_clean_weight),
        "--descriptor-trust-weight", str(args.descriptor_trust_weight),
        "--maximum-anchors", str(args.selection_maximum_anchors),
        "--visibility-target", str(args.required_visibility_rank),
        "--detectability-target", str(args.required_detectable_rank),
        "--matching-target", str(args.required_rank),
        "--pose-logdet-target", str(args.pose_logdet_target),
        "--pose-min-eigenvalue-target", str(args.pose_min_eigenvalue_target),
    ]
    if (
        arm in {"descriptor_loss", "descriptor_selection", "selection"}
        and args.descriptor_training_query_indices is not None
    ):
        command.extend(
            [
                "--mapping-training-query-indices",
                str(args.descriptor_training_query_indices),
                "--expected-mapping-training-query-indices-sha256",
                args.expected_descriptor_training_query_indices_sha256,
            ]
        )
    if arm == "reconstruction":
        command.extend(
            [
                "--association-graph", str(args.association_graph),
                "--expected-association-graph-sha256",
                args.expected_association_graph_sha256,
            ]
        )
    _run(command, root=root)
    report_path = output / "proposal.json"
    report = json.loads(report_path.read_text())
    report["_report_path"] = str(report_path.resolve())
    report["_report_sha256"] = sha256_file(report_path)
    return report


def run(args: argparse.Namespace) -> dict:
    root = Path(__file__).resolve().parents[1]
    for field in (
        "map",
        "metric",
        "observation_cache",
        "association_graph",
        "descriptor_training_query_indices",
        "output_dir",
    ):
        value = getattr(args, field, None)
        if value is not None:
            setattr(args, field, Path(value).resolve())
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("V6 convergence runner requires a clean worktree")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if int(args.descriptor_rounds) < 1:
        raise ValueError("at least one descriptor round is required")
    if args.loo_affected_anchor_policy != "rebuild":
        raise ValueError(
            "V6 mainline requires exact query-local Anchor rebuild; "
            "purge is evaluator-only diagnostic mode"
        )
    if (args.descriptor_training_query_indices is None) != (
        args.expected_descriptor_training_query_indices_sha256 is None
    ):
        raise ValueError("descriptor training split path and SHA must be paired")
    if (
        args.descriptor_training_query_indices is not None
        and sha256_file(args.descriptor_training_query_indices)
        != args.expected_descriptor_training_query_indices_sha256
    ):
        raise ValueError("descriptor training split SHA differs")
    if bool(args.run_reconstruction) and (
        args.association_graph is None
        or args.expected_association_graph_sha256 is None
    ):
        raise ValueError("reconstruction requires the SHA-bound association graph")
    args.output_dir.mkdir(parents=True)

    map_path = args.map.resolve()
    map_sha = args.expected_map_sha256
    metric_path = args.metric.resolve()
    metric_sha = args.expected_metric_sha256
    baseline_map_path = map_path
    baseline_map_sha = map_sha
    baseline_metric_path = metric_path
    baseline_metric_sha = metric_sha
    deployment_map_path = None
    deployment_map_sha = None
    deployment_metric_path = None
    deployment_metric_sha = None
    candidate_available = False
    stages = []

    summary_path, summary = _evaluate(
        args,
        root=root,
        map_path=map_path,
        map_sha=map_sha,
        metric_path=metric_path,
        metric_sha=metric_sha,
        output=args.output_dir / "stage_00_baseline",
    )
    stages.append(
        {
            "stage": "baseline",
            "stage_kind": "evaluation",
            "map": str(map_path),
            "map_sha256": map_sha,
            "summary": str(summary_path),
            "metrics": summary["summary"],
            "failure_layer_counts": summary["failure_layer_counts"],
            "metric": str(metric_path),
            "metric_sha256": metric_sha,
            "deployment_map": str(map_path),
            "deployment_map_sha256": map_sha,
            "deployment_metric": str(metric_path),
            "deployment_metric_sha256": metric_sha,
            **_evaluation_metadata(summary, baseline=True),
        }
    )

    for round_index in range(int(args.descriptor_rounds)):
        if int(summary["failure_layer_counts"].get("L3", 0)) == 0:
            break
        proposal = _propose(
            args,
            root=root,
            arm="descriptor_loss",
            map_path=map_path,
            map_sha=map_sha,
            feedback_summary=summary,
            output=args.output_dir / f"stage_{len(stages):02d}_descriptor_proposal",
        )
        if proposal.get("proposal_available") is False:
            stages.append(
                {
                    "stage": f"descriptor_round_{round_index + 1}_unavailable",
                    "stage_kind": "proposal_attempt",
                    "arm": "descriptor_loss",
                    "proposal_available": False,
                    "unavailable_reason": proposal.get("unavailable_reason"),
                    **_proposal_metadata(proposal),
                }
            )
            break
        map_path = Path(proposal["output"]["map"])
        map_sha = proposal["output"]["map_sha256"]
        metric_path = Path(proposal["output"]["metric"])
        metric_sha = proposal["output"]["metric_sha256"]
        deployment_map_path = Path(proposal["output"]["deployment_map"])
        deployment_map_sha = proposal["output"]["deployment_map_sha256"]
        deployment_metric_path = Path(proposal["output"]["deployment_metric"])
        deployment_metric_sha = proposal["output"]["deployment_metric_sha256"]
        candidate_available = True
        summary_path, summary = _evaluate(
            args,
            root=root,
            map_path=map_path,
            map_sha=map_sha,
            metric_path=metric_path,
            metric_sha=metric_sha,
            output=args.output_dir / f"stage_{len(stages):02d}_descriptor_evaluation",
        )
        stages.append(
            {
                "stage": f"descriptor_round_{round_index + 1}",
                "stage_kind": "evaluation",
                "map": str(map_path),
                "map_sha256": map_sha,
                "metric": str(metric_path),
                "metric_sha256": metric_sha,
                "deployment_map": str(deployment_map_path),
                "deployment_map_sha256": deployment_map_sha,
                "deployment_metric": str(deployment_metric_path),
                "deployment_metric_sha256": deployment_metric_sha,
                "summary": str(summary_path),
                "metrics": summary["summary"],
                "failure_layer_counts": summary["failure_layer_counts"],
                **_proposal_metadata(proposal),
                **_evaluation_metadata(summary),
            }
        )

    if args.run_reconstruction and int(summary["failure_layer_counts"].get("L1", 0)):
        proposal = _propose(
            args,
            root=root,
            arm="reconstruction",
            map_path=map_path,
            map_sha=map_sha,
            feedback_summary=summary,
            output=args.output_dir / f"stage_{len(stages):02d}_reconstruction_proposal",
        )
        if proposal.get("proposal_available") is not False:
            map_path = Path(proposal["output"]["map"])
            map_sha = proposal["output"]["map_sha256"]
            metric_path = Path(proposal["output"]["metric"])
            metric_sha = proposal["output"]["metric_sha256"]
            deployment_map_path = Path(proposal["output"]["deployment_map"])
            deployment_map_sha = proposal["output"]["deployment_map_sha256"]
            deployment_metric_path = Path(proposal["output"]["deployment_metric"])
            deployment_metric_sha = proposal["output"]["deployment_metric_sha256"]
            candidate_available = True
            summary_path, summary = _evaluate(
                args,
                root=root,
                map_path=map_path,
                map_sha=map_sha,
                metric_path=metric_path,
                metric_sha=metric_sha,
                output=args.output_dir / f"stage_{len(stages):02d}_reconstruction_evaluation",
            )
            stages.append(
                {
                    "stage": "reconstruction",
                    "stage_kind": "evaluation",
                    "map": str(map_path),
                    "map_sha256": map_sha,
                    "metric": str(metric_path),
                    "metric_sha256": metric_sha,
                    "deployment_map": str(deployment_map_path),
                    "deployment_map_sha256": deployment_map_sha,
                    "deployment_metric": str(deployment_metric_path),
                    "deployment_metric_sha256": deployment_metric_sha,
                    "summary": str(summary_path),
                    "metrics": summary["summary"],
                    "failure_layer_counts": summary["failure_layer_counts"],
                    **_proposal_metadata(proposal),
                    **_evaluation_metadata(summary),
                }
            )
        else:
            stages.append(
                {
                    "stage": "reconstruction_unavailable",
                    "stage_kind": "proposal_attempt",
                    "arm": "reconstruction",
                    "proposal_available": False,
                    "unavailable_reason": proposal.get("unavailable_reason"),
                    **_proposal_metadata(proposal),
                }
            )

    if args.run_selection:
        proposal = _propose(
            args,
            root=root,
            arm="selection",
            map_path=map_path,
            map_sha=map_sha,
            feedback_summary=summary,
            output=args.output_dir / f"stage_{len(stages):02d}_selection_proposal",
        )
        map_path = Path(proposal["output"]["map"])
        map_sha = proposal["output"]["map_sha256"]
        metric_path = Path(proposal["output"]["metric"])
        metric_sha = proposal["output"]["metric_sha256"]
        deployment_map_path = Path(proposal["output"]["deployment_map"])
        deployment_map_sha = proposal["output"]["deployment_map_sha256"]
        deployment_metric_path = Path(proposal["output"]["deployment_metric"])
        deployment_metric_sha = proposal["output"]["deployment_metric_sha256"]
        candidate_available = True
        summary_path, summary = _evaluate(
            args,
            root=root,
            map_path=map_path,
            map_sha=map_sha,
            metric_path=metric_path,
            metric_sha=metric_sha,
            output=args.output_dir / f"stage_{len(stages):02d}_selection_evaluation",
        )
        stages.append(
            {
                "stage": "selection_after_fresh_feedback",
                "stage_kind": "evaluation",
                "map": str(map_path),
                "map_sha256": map_sha,
                "metric": str(metric_path),
                "metric_sha256": metric_sha,
                "deployment_map": str(deployment_map_path),
                "deployment_map_sha256": deployment_map_sha,
                "deployment_metric": str(deployment_metric_path),
                "deployment_metric_sha256": deployment_metric_sha,
                "summary": str(summary_path),
                "metrics": summary["summary"],
                "failure_layer_counts": summary["failure_layer_counts"],
                **_proposal_metadata(proposal),
                **_evaluation_metadata(summary),
            }
        )

    result = {
        "schema": "lafgs_v6_mainline_convergence_run",
        "version": 2,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "automatic_hard_gate_acceptance": False,
        "candidate_acceptance_status": "external_review_required",
        "candidate_available": candidate_available,
        "feedback_protocol": {
            "loo_pose_neighbors": int(args.loo_pose_neighbors),
            "affected_anchor_policy": args.loo_affected_anchor_policy,
            "descriptor_training_split": (
                None
                if args.descriptor_training_query_indices is None
                else {
                    "path": str(args.descriptor_training_query_indices.resolve()),
                    "sha256": (
                        args.expected_descriptor_training_query_indices_sha256
                    ),
                }
            ),
            "mapping_training_split": (
                None
                if args.descriptor_training_query_indices is None
                else {
                    "path": str(args.descriptor_training_query_indices.resolve()),
                    "sha256": (
                        args.expected_descriptor_training_query_indices_sha256
                    ),
                }
            ),
        },
        "configuration": {
            "device": args.device,
            "cpu_threads": int(args.cpu_threads),
            "seed": int(args.seed),
            "positive_radius_px": float(args.positive_radius_px),
            "alpha_minimum": float(args.alpha_minimum),
            "required_matching_rank": int(args.required_rank),
            "required_visibility_rank": int(args.required_visibility_rank),
            "required_detectable_rank": int(args.required_detectable_rank),
            "ransac_reprojection_px": float(args.ransac_reprojection_px),
            "descriptor_rounds": int(args.descriptor_rounds),
            "descriptor_trust_region": float(args.descriptor_trust_region),
            "descriptor_margin": float(args.descriptor_margin),
            "descriptor_temperature": float(args.descriptor_temperature),
            "descriptor_learning_rate": float(args.descriptor_learning_rate),
            "descriptor_epochs": int(args.descriptor_epochs),
            "descriptor_batch_size": int(args.descriptor_batch_size),
            "descriptor_maximum_triplets_per_query": int(
                args.descriptor_maximum_triplets_per_query
            ),
            "descriptor_clean_fraction": float(args.descriptor_clean_fraction),
            "descriptor_clean_weight": float(args.descriptor_clean_weight),
            "descriptor_trust_weight": float(args.descriptor_trust_weight),
            "run_reconstruction": bool(args.run_reconstruction),
            "run_selection": bool(args.run_selection),
            "selection_maximum_anchors": int(args.selection_maximum_anchors),
            "pose_logdet_target": float(args.pose_logdet_target),
            "pose_min_eigenvalue_target": float(
                args.pose_min_eigenvalue_target
            ),
        },
        "stages": stages,
        "baseline_map": str(baseline_map_path),
        "baseline_map_sha256": baseline_map_sha,
        "baseline_metric": str(baseline_metric_path),
        "baseline_metric_sha256": baseline_metric_sha,
        "last_candidate_map": str(map_path) if candidate_available else None,
        "last_candidate_map_sha256": map_sha if candidate_available else None,
        "last_candidate_metric": str(metric_path) if candidate_available else None,
        "last_candidate_metric_sha256": metric_sha if candidate_available else None,
        "last_candidate_deployment_map": (
            str(deployment_map_path) if candidate_available else None
        ),
        "last_candidate_deployment_map_sha256": (
            deployment_map_sha if candidate_available else None
        ),
        "last_candidate_deployment_metric": (
            str(deployment_metric_path) if candidate_available else None
        ),
        "last_candidate_deployment_metric_sha256": (
            deployment_metric_sha if candidate_available else None
        ),
        "online_protocol": "native_superpoint_global_top1_one_standard_poselib",
    }
    temporary = args.output_dir / f".run.{os.getpid()}.tmp"
    try:
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, args.output_dir / "run.json")
    finally:
        temporary.unlink(missing_ok=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--expected-map-sha256", required=True)
    parser.add_argument("--metric", type=Path, required=True)
    parser.add_argument("--expected-metric-sha256", required=True)
    parser.add_argument("--observation-cache", type=Path, required=True)
    parser.add_argument("--expected-observation-cache-sha256", required=True)
    parser.add_argument("--association-graph", type=Path)
    parser.add_argument("--expected-association-graph-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--positive-radius-px", type=float, default=2.0)
    parser.add_argument("--alpha-minimum", type=float, default=0.05)
    parser.add_argument("--required-rank", type=int, default=16)
    parser.add_argument("--required-visibility-rank", type=int, default=4)
    parser.add_argument("--required-detectable-rank", type=int, default=16)
    parser.add_argument("--loo-pose-neighbors", type=int, default=3)
    parser.add_argument(
        "--loo-affected-anchor-policy",
        choices=("purge", "rebuild"),
        default="rebuild",
    )
    parser.add_argument("--ransac-reprojection-px", type=float, default=4.0)
    parser.add_argument("--descriptor-rounds", type=int, default=1)
    parser.add_argument("--descriptor-trust-region", type=float, default=0.05)
    parser.add_argument("--descriptor-margin", type=float, default=0.05)
    parser.add_argument("--descriptor-temperature", type=float, default=0.04)
    parser.add_argument("--descriptor-learning-rate", type=float, default=0.02)
    parser.add_argument("--descriptor-epochs", type=int, default=5)
    parser.add_argument("--descriptor-batch-size", type=int, default=8192)
    parser.add_argument("--descriptor-maximum-triplets-per-query", type=int, default=128)
    parser.add_argument("--descriptor-clean-fraction", type=float, default=0.25)
    parser.add_argument("--descriptor-clean-weight", type=float, default=0.25)
    parser.add_argument("--descriptor-trust-weight", type=float, default=0.1)
    parser.add_argument("--descriptor-training-query-indices", type=Path)
    parser.add_argument("--expected-descriptor-training-query-indices-sha256")
    parser.add_argument("--run-reconstruction", action="store_true")
    parser.add_argument("--run-selection", action="store_true")
    parser.add_argument("--selection-maximum-anchors", type=int, default=20000)
    parser.add_argument("--pose-logdet-target", type=float, default=0.0)
    parser.add_argument("--pose-min-eigenvalue-target", type=float, default=0.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
