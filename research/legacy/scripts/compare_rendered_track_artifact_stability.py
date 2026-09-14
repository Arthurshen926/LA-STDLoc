#!/usr/bin/env python3
"""Compare fixed R0/R1 full-mapping LOO reports under the preregistered gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import torch

from common.calibration import validate_equivalent_query_cache_calibration_parent
from common.hashing import sha256_file
from common.tensor_identity import recursive_bitwise_equal


_PREREG = Path(
    "docs/evidence/rendered_rgb_track_artifact_stability_preregistration.json"
)
_SOURCE_PATHS = (
    "common/calibration.py",
    "common/tensor_identity.py",
    str(_PREREG),
    "scripts/compare_rendered_track_artifact_stability.py",
)
_SCENES = {
    "shopfacade": {
        "baseline": {
            "path": "/mnt/pool/sqy/lafgs_render_track_only_fullmap_v14_20260815/shopfacade/full_mapping_loo_selector_identity_seed2026/full_mapping_loo_report.json",
            "sha256": "cf7f6d5090119648a96136d7fd890482271945d8beac19a98c864f8aeb169723",
        },
        "materialization_report": {
            "path": "/mnt/pool/sqy/lafgs_render_track_only_artifact_r1_20260815/shopfacade/artifact_R1_v5/artifact_stability_report.json",
            "sha256": "0d473181c9996fd48ba6f6bef6ad20f21ba253bd41aa5e59eb0b848cf8442539",
        },
        "r1_inputs": {
            "map": "6735e4a74ba79645cd4357a4e2bd54980ebb4fcb328dbe31c94772cb266e5784",
            "metric": "69cf169076eeb46826166ca6c6afdb49a917a97b02876e91589ad5985be1f6fa",
            "query_cache": "c10dacd71745fbba59d207bef6ae415dca721f427a391a0044dcd33e28c711f8",
            "scene_calibration": "c3af4efc06a20a927070e6ac990a27eb9f6d2d13d5542ffe30ccdbbd8fc9b04f",
            "teacher": "05782bad63a618524d859e2032e5ca0734821eb87b92ace22751dde01ae143ba",
            "track_payload": "1c358882053bf97edbe94946ca4d1d0c9e8dffaddd3c527626fa8be95095ed14",
        },
    },
    "stairs": {
        "baseline": {
            "path": "/mnt/pool/sqy/lafgs_render_track_only_fullmap_v14_20260815/stairs/full_mapping_loo_selector_identity_seed2026/full_mapping_loo_report.json",
            "sha256": "349167671ac79c388d9b4ba3a41ffac3166ff515eecffbaeff5ead0a84fd25bf",
        },
        "materialization_report": {
            "path": "/mnt/pool/sqy/lafgs_render_track_only_artifact_r1_20260815/stairs/artifact_R1_v5/artifact_stability_report.json",
            "sha256": "1849650df3fa317a098bb74b50db34ce4d8f0aa8e62e11ee7005ff817f661368",
        },
        "r1_inputs": {
            "map": "21b3eaff5b8000560e689edcbc5cb355cb04ae9e9cd6ccff06c9b2cbc16b837c",
            "metric": "cbc6be941cca7a8327e890a3df936793301b908d33930f1a9917cded64a73b8a",
            "query_cache": "0e7d9302629136010b93c1e48b6b38117023fd9c4e9df012ea779535ca9050d8",
            "scene_calibration": "bb06173e992c9752102485e782f29a596898ca0b12f0605ad327a128a567ad01",
            "teacher": "d212b9136cfbf6c8ddef426e1be50950f2ab3e44356c616941cc15998fb0d73b",
            "track_payload": "a5d16927d3bdbebb6cfb9192c03ec433c598b4a2069e957d39e4d5ff2e8943ba",
        },
    },
}


def _producer_identity() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("R1 mapping gate producer worktree must be clean")
    return {
        "git_commit": commit,
        "worktree_clean": True,
        "torch_version": torch.__version__,
        "source_sha256": {
            relative: sha256_file(repository / relative) for relative in _SOURCE_PATHS
        },
    }


def evaluate_mapping_gate(
    scene: str, baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate one scene using only the preregistered numeric thresholds."""
    base = baseline.get("summary")
    r1 = candidate.get("summary")
    if not isinstance(base, Mapping) or not isinstance(r1, Mapping):
        raise ValueError("mapping report summary is missing")
    required = {
        "catastrophic_100cm_count",
        "cvar95_te_cm",
        "median_te_cm",
        "p90_te_cm",
        "query_count",
        "raw_gt_precision_percent",
        "recall_5cm_5deg_percent",
    }
    if not required <= set(base) or not required <= set(r1):
        raise ValueError("mapping report summary is incomplete")
    if int(base["query_count"]) != int(r1["query_count"]):
        raise ValueError("baseline and R1 query counts differ")
    if scene == "shopfacade":
        thresholds = {
            "median_te_cm_ceiling": 1.02 * float(base["median_te_cm"]),
            "p90_te_cm_ceiling": 1.02 * float(base["p90_te_cm"]),
            "recall_5cm_5deg_percent_floor": float(base["recall_5cm_5deg_percent"])
            - 0.25,
            "catastrophic_100cm_count_ceiling": int(base["catastrophic_100cm_count"]),
        }
        gates = {
            "median_te_within_1p02x": float(r1["median_te_cm"])
            <= thresholds["median_te_cm_ceiling"],
            "p90_te_within_1p02x": float(r1["p90_te_cm"])
            <= thresholds["p90_te_cm_ceiling"],
            "recall_not_lower_by_more_than_0p25pp": float(r1["recall_5cm_5deg_percent"])
            >= thresholds["recall_5cm_5deg_percent_floor"],
            "catastrophic_count_not_higher": int(r1["catastrophic_100cm_count"])
            <= thresholds["catastrophic_100cm_count_ceiling"],
        }
    elif scene == "stairs":
        thresholds = {
            "p90_te_cm_ceiling": float(base["p90_te_cm"]),
            "cvar95_te_cm_ceiling": float(base["cvar95_te_cm"]),
            "raw_gt_precision_percent_floor": float(base["raw_gt_precision_percent"])
            - 0.05,
            "catastrophic_100cm_count_ceiling": int(base["catastrophic_100cm_count"]),
        }
        gates = {
            "p90_te_not_higher": float(r1["p90_te_cm"])
            <= thresholds["p90_te_cm_ceiling"],
            "cvar95_te_not_higher": float(r1["cvar95_te_cm"])
            <= thresholds["cvar95_te_cm_ceiling"],
            "raw_gt_precision_not_lower_by_more_than_0p05pp": float(
                r1["raw_gt_precision_percent"]
            )
            >= thresholds["raw_gt_precision_percent_floor"],
            "catastrophic_count_not_higher": int(r1["catastrophic_100cm_count"])
            <= thresholds["catastrophic_100cm_count_ceiling"],
        }
    else:
        raise ValueError(f"unsupported R1 scene {scene}")
    return {
        "baseline": dict(base),
        "candidate": dict(r1),
        "delta_candidate_minus_baseline": {
            key: float(r1[key]) - float(base[key]) for key in sorted(required)
        },
        "thresholds": thresholds,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _load_report(path: Path, expected_sha256: str, *, label: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} report SHA differs")
    report = json.loads(path.read_text())
    if (
        report.get("schema") != "lafgs_rendered_track_full_mapping_loo_report"
        or report.get("version") != 1
        or report.get("uses_source_mapping_rgb") is not False
        or report.get("uses_test_queries") is not False
        or report.get("formal_method_uses_crossfit") is not False
        or report.get("seed") != 2026
        or report.get("configuration")
        != {
            "deployment_row_limit": 0,
            "descriptor_trim_fraction": 0.2,
            "one_global_top1_per_query_row": True,
            "one_poselib_call_per_mapping_query": True,
        }
    ):
        raise ValueError(f"{label} report is not the frozen mapping-LOO protocol")
    statistics_path = Path(str(report.get("statistics", ""))).resolve()
    if not statistics_path.is_file() or sha256_file(statistics_path) != report.get(
        "statistics_sha256"
    ):
        raise ValueError(f"{label} statistics are missing or changed")
    statistics = torch.load(statistics_path, map_location="cpu", weights_only=False)
    if (
        statistics.get("schema") != "lafgs_rendered_track_full_mapping_loo_statistics"
        or statistics.get("version") != 1
        or statistics.get("uses_source_mapping_rgb") is not False
        or statistics.get("uses_test_queries") is not False
        or not recursive_bitwise_equal(statistics.get("summary"), report.get("summary"))
        or not recursive_bitwise_equal(statistics.get("loo"), report.get("loo"))
    ):
        raise ValueError(f"{label} report does not replay its statistics")
    return report


def _validate_materialization(scene: str, candidate: Mapping[str, Any]) -> None:
    contract = _SCENES[scene]
    materialization_path = Path(contract["materialization_report"]["path"])
    if (
        sha256_file(materialization_path)
        != contract["materialization_report"]["sha256"]
    ):
        raise ValueError(f"{scene} materialization report SHA differs")
    materialization = json.loads(materialization_path.read_text())
    if (
        materialization.get("schema")
        != "lafgs_rendered_track_artifact_stability_r1_materialization"
        or materialization.get("version") != 1
        or materialization.get("uses_source_mapping_rgb") is not False
        or materialization.get("uses_test_queries") is not False
        or materialization.get("fixed_pair_graph") is not True
        or materialization.get("fixed_track_components") is not True
        or materialization.get("fixed_anchor_xyz") is not True
        or materialization.get("fixed_selector_membership") is not True
        or materialization.get("fixed_map_cardinality") is not True
        or materialization.get("only_observation_fusion_weight_changes") is not True
    ):
        raise ValueError(f"{scene} R1 materialization contract is invalid")
    if candidate.get("input_sha256") != contract["r1_inputs"]:
        raise ValueError(f"{scene} R1 report inputs differ from the compiled contract")
    inputs = candidate.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != set(contract["r1_inputs"]):
        raise ValueError(f"{scene} R1 report input registry is incomplete")
    for name, expected_sha256 in contract["r1_inputs"].items():
        path = Path(str(inputs[name])).resolve()
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"{scene} R1 {name} changed after evaluation")
    calibration_path = Path(str(inputs["scene_calibration"])).resolve()
    calibration = json.loads(calibration_path.read_text())
    validate_equivalent_query_cache_calibration_parent(
        calibration,
        parent_path=calibration_path,
        query_cache_path=inputs["query_cache"],
    )


def _atomic_json(payload: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if json.loads(temporary.read_text()) != payload:
            raise RuntimeError("temporary R1 gate did not reload exactly")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shopfacade-r1-report", type=Path, required=True)
    parser.add_argument("--expected-shopfacade-r1-report-sha256", required=True)
    parser.add_argument("--stairs-r1-report", type=Path, required=True)
    parser.add_argument("--expected-stairs-r1-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("refusing to overwrite an R1 mapping gate")
    identity = _producer_identity()
    scene_results: dict[str, Any] = {}
    inputs: dict[str, Any] = {}
    requested = {
        "shopfacade": (
            args.shopfacade_r1_report.resolve(),
            args.expected_shopfacade_r1_report_sha256.lower(),
        ),
        "stairs": (
            args.stairs_r1_report.resolve(),
            args.expected_stairs_r1_report_sha256.lower(),
        ),
    }
    for scene, (candidate_path, candidate_sha256) in requested.items():
        baseline_contract = _SCENES[scene]["baseline"]
        baseline_path = Path(baseline_contract["path"]).resolve()
        baseline = _load_report(
            baseline_path, baseline_contract["sha256"], label=f"{scene} baseline"
        )
        candidate = _load_report(candidate_path, candidate_sha256, label=f"{scene} R1")
        if not recursive_bitwise_equal(baseline.get("loo"), candidate.get("loo")):
            raise ValueError(f"{scene} R1 changed the LOO membership contract")
        _validate_materialization(scene, candidate)
        scene_results[scene] = evaluate_mapping_gate(scene, baseline, candidate)
        inputs[scene] = {
            "baseline_report": {
                "path": str(baseline_path),
                "sha256": baseline_contract["sha256"],
            },
            "r1_report": {
                "path": str(candidate_path),
                "sha256": candidate_sha256,
            },
        }
    both_passed = all(value["passed"] for value in scene_results.values())
    gate = {
        "schema": "lafgs_rendered_track_artifact_stability_r1_mapping_gate",
        "version": 1,
        "valid": True,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "producer_identity": identity,
        "inputs": inputs,
        "scenes": scene_results,
        "both_scenes_passed": both_passed,
        "advance_to_frozen_three_seed_test": both_passed,
        "advance_to_artifact_aware_track_identity_r2": False,
        "decision": (
            "GO_TO_FROZEN_THREE_SEED_TEST"
            if both_passed
            else "STOP_R1_BEFORE_TEST_AND_R2"
        ),
    }
    _atomic_json(gate, output)
    if _producer_identity() != identity:
        raise RuntimeError("R1 mapping gate producer identity changed")
    print(
        json.dumps(
            {"output": str(output), "sha256": sha256_file(output), **gate},
            indent=2,
            sort_keys=True,
        )
    )
    if not both_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
