#!/usr/bin/env python3
"""Preflight or run paired mapping-only local-frontend ceiling probes.

No model is downloaded or instantiated by this runner.  ``evaluate`` consumes
an already materialized, provenance-locked probe cache and never consults test
queries or changes a deployment artifact.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
from typing import Sequence

import torch

from common.evaluation_code import (
    frontend_descriptor_evaluation_code_identity,
    frontend_detector_evaluation_code_identity,
)
from map_learning.frontend_upper_bound import (
    PROBE_SCHEMA,
    audit_descriptor_equal_energy_crossfit,
    audit_descriptor_identity_crossfit,
    audit_detector_repeatability,
    file_sha256,
    probe_contract,
)


SUPERPOINT_SHA256 = "52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e"
FEATUREBOOSTER_SHA256 = (
    "5334d9aa861e877a2b99baff0d682e1ac8a749cdd65eb1d4b8bd0a8bb8bf0359"
)
AUDITED_CODE_PATHS = (
    "features/superpoint.py",
    "features/extractor.py",
    "localization/frontend.py",
    "map_learning/bootstrap.py",
    "map_learning/observations.py",
    "map_learning/context_booster.py",
    "map_learning/context_booster_crossfit.py",
    "map_learning/context_metric.py",
    "map_learning/context_metric_crossfit.py",
    "map_learning/metric_context_uplift.py",
    "map_learning/frontend_upper_bound.py",
    "scripts/audit_frontend_upper_bound.py",
)


def _torch_load(path: str):
    return torch.load(path, map_location="cpu", weights_only=False)


def _write_json(path: str | Path, report: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _detector_geometry_overrides(args: argparse.Namespace) -> dict:
    """Require explicit geometry tolerances for every detector evaluation."""
    if args.arm not in ("detector", "both"):
        return {}
    raw = {
        "depth_abs_tolerance_m": args.depth_abs_tolerance_m,
        "depth_rel_tolerance": args.depth_rel_tolerance,
        "alpha_minimum": args.alpha_minimum,
    }
    missing = [name for name, value in raw.items() if value is None]
    if missing:
        raise ValueError(
            "detector evaluation requires explicit geometry tolerances: "
            + ", ".join(missing)
        )
    values = {name: float(value) for name, value in raw.items()}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("detector geometry tolerances must be finite")
    if values["depth_abs_tolerance_m"] < 0.0:
        raise ValueError("detector absolute depth tolerance must be nonnegative")
    if values["depth_rel_tolerance"] < 0.0:
        raise ValueError("detector relative depth tolerance must be nonnegative")
    if not 0.0 <= values["alpha_minimum"] <= 1.0:
        raise ValueError("detector alpha minimum must lie in [0, 1]")
    return values


def _inspect_artifact(path: str | Path, expected_sha256: str | None = None) -> dict:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return {
            "path": str(resolved),
            "status": "missing",
            "expected_sha256": expected_sha256,
        }
    actual = file_sha256(resolved)
    return {
        "path": str(resolved),
        "status": (
            "verified"
            if expected_sha256 is not None and actual == expected_sha256
            else "present_unverified"
            if expected_sha256 is None
            else "sha256_mismatch"
        ),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": actual,
        "expected_sha256": expected_sha256,
    }


def _inspect_kornia(python: str | Path) -> dict:
    executable = Path(python).expanduser().resolve()
    if not executable.is_file():
        return {"python": str(executable), "status": "python_missing"}
    code = """
import json
import kornia
names = ['DISK', 'DeDoDe', 'LoFTR', 'HardNet', 'SOSNet']
print(json.dumps({
    'version': kornia.__version__,
    'available_feature_symbols': [
        name for name in names if hasattr(kornia.feature, name)
    ],
}))
"""
    try:
        completed = subprocess.run(
            [str(executable), "-c", code],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        return {
            "python": str(executable),
            "status": "code_available",
            **payload,
            "weight_status": "no_locked_compatible_weight_attested",
            "network_access_used": False,
        }
    except (subprocess.SubprocessError, json.JSONDecodeError) as error:
        return {
            "python": str(executable),
            "status": "inspection_failed",
            "error": str(error),
        }


def _repository_attestation() -> dict:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except subprocess.SubprocessError:
        commit = "unavailable"
    return {
        "root": str(root),
        "git_commit_at_audit": commit,
        "code_sha256": {
            relative: file_sha256(root / relative)
            for relative in AUDITED_CODE_PATHS
            if (root / relative).is_file()
        },
    }


def _candidate_preflight(args: argparse.Namespace) -> dict:
    if not args.candidate_weights:
        return {
            "status": "BLOCKED_BY_ARTIFACT",
            "reason": "no candidate weight path was supplied",
        }
    artifact = _inspect_artifact(args.candidate_weights, args.candidate_weights_sha256)
    detector_reasons = []
    if not args.candidate_weights_sha256:
        detector_reasons.append("candidate weight SHA256 was not supplied")
    if artifact["status"] != "verified":
        detector_reasons.append(f"candidate weight status is {artifact['status']}")
    if args.candidate_family != "independent_local_frontend":
        detector_reasons.append("candidate is not an independent local frontend")
    if not args.candidate_code_id:
        detector_reasons.append("candidate implementation/version is not locked")
    detector_eligible = not detector_reasons
    descriptor_reasons = list(detector_reasons)
    if int(args.candidate_descriptor_dim) <= 0:
        descriptor_reasons.append(
            "descriptor identity arm requires a positive candidate dimension"
        )
    descriptor_eligible = not descriptor_reasons
    status = (
        "READY_FOR_BOTH_PROBES"
        if descriptor_eligible
        else "READY_FOR_DETECTOR_PROBE_ONLY"
        if detector_eligible
        else "BLOCKED"
    )
    return {
        "name": args.candidate_name,
        "family": args.candidate_family,
        "code_id": args.candidate_code_id,
        "descriptor_dim": int(args.candidate_descriptor_dim),
        "weights": artifact,
        "detector_arm_eligible": detector_eligible,
        "descriptor_arm_eligible": descriptor_eligible,
        "status": status,
        "detector_blockers": detector_reasons,
        "descriptor_blockers": descriptor_reasons,
    }


def preflight(args: argparse.Namespace) -> dict:
    superpoint = _inspect_artifact(args.superpoint_weights, SUPERPOINT_SHA256)
    featurebooster = _inspect_artifact(
        args.featurebooster_weights, FEATUREBOOSTER_SHA256
    )
    featurebooster.update(
        {
            "classification": "same_dim_superpoint_postprocessor_control",
            "independent_local_frontend": False,
            "historical_status": "already_crossfit_evaluated_not_deployed",
        }
    )
    loftr = _inspect_artifact(args.loftr_weights)
    loftr.update(
        {
            "classification": "pair_matcher",
            "admissible": False,
            "reason": (
                "pair-conditioned matches cannot replace the current global "
                "descriptor bank in a one-factor paired experiment"
            ),
        }
    )
    candidate = _candidate_preflight(args)
    descriptor_ready = bool(candidate.get("descriptor_arm_eligible", False))
    detector_ready = bool(candidate.get("detector_arm_eligible", False))
    baseline_verified = superpoint["status"] == "verified"
    return {
        "schema": "lafgs_frontend_ceiling_probe_environment_preflight",
        "version": 1,
        "mapping_only": True,
        "uses_test_queries": False,
        "network_access_used": False,
        "deployment_modified": False,
        "probe_schema": PROBE_SCHEMA,
        "audited_repository": _repository_attestation(),
        "baseline": {
            "name": "frozen_superpoint",
            "descriptor_dim": 256,
            "weights": superpoint,
            "status": "READY" if baseline_verified else "BLOCKED",
        },
        "available_but_not_admissible_as_stronger_frontend": {
            "featurebooster": featurebooster,
            "loftr": loftr,
        },
        "kornia": _inspect_kornia(args.kornia_python),
        "candidate": candidate,
        "ceiling_probe_arms": {
            "A_detector_repeatability": {
                "status": (
                    "READY_FOR_PROBE_MATERIALIZATION"
                    if baseline_verified and detector_ready
                    else "BLOCKED_BY_ARTIFACT"
                ),
                "fixed_variables": [
                    "mapping queries",
                    "frozen map xyz and anchor types",
                    "requested K",
                    "image preprocessing and valid mask",
                    "GT pose/intrinsics/depth/alpha legality",
                ],
                "candidate_descriptors_used": False,
            },
            "B_descriptor_identity": {
                "status": (
                    "READY_FOR_PROBE_MATERIALIZATION"
                    if baseline_verified and descriptor_ready
                    else "BLOCKED_BY_ARTIFACT"
                ),
                "fixed_variables": [
                    "exact SuperPoint keypoint rows",
                    "complete-positive teacher",
                    "support observations and view-balanced fusion",
                    "bidirectional temporal crossfit",
                    "global cosine R@K",
                ],
                "candidate_detector_used": False,
                "descriptor_dimension_policy": (
                    "candidate dimension may differ; map bytes, MACs, and "
                    "ranking wall time must be reported"
                ),
            },
        },
        "conclusion": (
            "No admissible locked independent stronger-local-frontend "
            "artifact is present, so the SuperPoint representation ceiling "
            "cannot yet be accepted or rejected. The audit must remain "
            "BLOCKED_BY_ARTIFACT; code availability alone is not a result."
            if not descriptor_ready
            else "An admissible candidate is present; materialize a probe cache."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser(
        "preflight", help="audit local code/weight availability without a model run"
    )
    preflight_parser.add_argument("--output", required=True)
    preflight_parser.add_argument(
        "--superpoint-weights",
        default="~/.cache/lafgs/superpoint_v1.pth",
    )
    preflight_parser.add_argument(
        "--featurebooster-weights",
        default="~/.cache/lafgs/SuperPoint+Boost-F.pth",
    )
    preflight_parser.add_argument(
        "--loftr-weights",
        default="~/.cache/torch/hub/checkpoints/loftr_outdoor.ckpt",
    )
    preflight_parser.add_argument(
        "--kornia-python",
        default="/root/miniconda3/envs/g4splat/bin/python",
    )
    preflight_parser.add_argument("--candidate-name", default="unprovisioned")
    preflight_parser.add_argument(
        "--candidate-family",
        choices=(
            "independent_local_frontend",
            "superpoint_postprocessor",
            "pair_matcher",
        ),
        default="independent_local_frontend",
    )
    preflight_parser.add_argument("--candidate-code-id", default="")
    preflight_parser.add_argument("--candidate-weights")
    preflight_parser.add_argument("--candidate-weights-sha256")
    preflight_parser.add_argument("--candidate-descriptor-dim", type=int, default=256)

    contract_parser = subparsers.add_parser(
        "contract", help="write the exact reference-row probe producer contract"
    )
    contract_parser.add_argument("--query-cache", required=True)
    contract_parser.add_argument("--teacher", required=True)
    contract_parser.add_argument("--output", required=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="consume a locked probe; no extractor or download occurs"
    )
    evaluate_parser.add_argument("--state", required=True)
    evaluate_parser.add_argument("--query-cache", required=True)
    evaluate_parser.add_argument("--teacher", required=True)
    evaluate_parser.add_argument("--probe-cache", required=True)
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.add_argument(
        "--arm",
        choices=("detector", "descriptor", "descriptor_equal_energy", "both"),
        required=True,
    )
    evaluate_parser.add_argument("--crossfit-blocks", type=int, default=8)
    evaluate_parser.add_argument("--minimum-support-views", type=int, default=2)
    evaluate_parser.add_argument(
        "--topks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32]
    )
    evaluate_parser.add_argument(
        "--reachability-radii-px", type=float, nargs="+", default=[2, 4, 8]
    )
    evaluate_parser.add_argument(
        "--depth-abs-tolerance-m",
        type=float,
        help="explicit detector target-universe absolute depth tolerance",
    )
    evaluate_parser.add_argument(
        "--depth-rel-tolerance",
        type=float,
        help="explicit detector target-universe relative depth tolerance",
    )
    evaluate_parser.add_argument(
        "--alpha-minimum",
        type=float,
        help="explicit detector target-universe minimum rendered alpha",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        report = preflight(args)
    elif args.command == "contract":
        report = probe_contract(
            _torch_load(args.query_cache), _torch_load(args.teacher)
        )
        report["source_artifacts"] = {
            "query_cache": _inspect_artifact(args.query_cache),
            "teacher": _inspect_artifact(args.teacher),
        }
    else:
        detector_geometry = _detector_geometry_overrides(args)
        evaluation_code = (
            frontend_detector_evaluation_code_identity(require_clean=True)
            if args.arm == "detector"
            else frontend_descriptor_evaluation_code_identity(require_clean=True)
        )
        state = _torch_load(args.state)
        query_cache = _torch_load(args.query_cache)
        teacher = _torch_load(args.teacher)
        probe = _torch_load(args.probe_cache)
        report = {
            "schema": "lafgs_frontend_ceiling_probe_audit_bundle",
            "version": 1,
            "mapping_only": True,
            "uses_test_queries": False,
            "deployment_modified": False,
            "evaluation_code": evaluation_code,
            "probe_cache": str(Path(args.probe_cache).resolve()),
            "source_artifacts": {
                "state": _inspect_artifact(args.state),
                "query_cache": _inspect_artifact(args.query_cache),
                "teacher": _inspect_artifact(args.teacher),
                "probe_cache": _inspect_artifact(args.probe_cache),
            },
        }
        if args.arm in ("detector", "both"):
            report["detector_repeatability"] = audit_detector_repeatability(
                state=state,
                query_cache=query_cache,
                teacher=teacher,
                probe=probe,
                radii_px=args.reachability_radii_px,
                **detector_geometry,
                query_cache_path=args.query_cache,
                teacher_path=args.teacher,
            )
        if args.arm in ("descriptor", "both"):
            report["descriptor_identity"] = audit_descriptor_identity_crossfit(
                state=state,
                query_cache=query_cache,
                teacher=teacher,
                probe=probe,
                crossfit_blocks=args.crossfit_blocks,
                minimum_support_views=args.minimum_support_views,
                topks=args.topks,
                query_cache_path=args.query_cache,
                teacher_path=args.teacher,
            )
        if args.arm == "descriptor_equal_energy":
            report["descriptor_identity"] = audit_descriptor_equal_energy_crossfit(
                state=state,
                query_cache=query_cache,
                teacher=teacher,
                probe=probe,
                crossfit_blocks=args.crossfit_blocks,
                minimum_support_views=args.minimum_support_views,
                topks=args.topks,
                query_cache_path=args.query_cache,
                teacher_path=args.teacher,
            )
    _write_json(args.output, report)
    print(json.dumps({"output": str(Path(args.output).resolve())}, indent=2))


if __name__ == "__main__":
    main()
