from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from common.hashing import sha256_file
from common.v6_contracts import (
    FEEDBACK_SCHEMA,
    FEEDBACK_VERSION,
    exact_identity_positive_contract,
)
from scripts.compare_v6_feedback import compare_feedback_files, run


_CACHE_SHA = "c" * 64
_SCENE_CALIBRATION_SHA = "e" * 64
_FEEDBACK_CALIBRATION_BINDING_SHA = "f" * 64
_MASK_FIELDS = (
    "top1_exact_identity_correct_mask",
    "top1_geometry_compatible_ambiguous_mask",
    "top1_identity_projective_incompatible_mask",
    "top1_negative_mask",
)
_CLASS_INDEX = {"exact": 0, "ambiguous": 1, "incompatible": 2, "negative": 3}
_SOURCES = {
    path: character * 64
    for path, character in zip(
        (
            "scripts/evaluate_v6_self_localization.py",
            "common/v6_contracts.py",
            "common/v6_pipeline_contract.py",
            "evidence/observation_provider.py",
            "map_learning/v6_feedback_evaluator.py",
            "map_learning/self_localization_feedback.py",
            "evidence/projective_loo.py",
            "evidence/projective_reconstruction.py",
            "localization/matcher.py",
            "localization/pose_solver.py",
            "topology/layered_sufficiency.py",
            "topology/pose_information.py",
        ),
        "123456789abc",
    )
}


def _record(
    index: int,
    *,
    winners: list[int],
    classes: list[str],
    positives: list[tuple[int, int]],
    inliers: list[int],
    te_cm: float,
    ae_deg: float,
    layers: list[str],
    correct_rank: int,
    independent: bool,
    pose_offset: float,
) -> dict:
    masks = {field: torch.zeros(len(winners), dtype=torch.bool) for field in _MASK_FIELDS}
    for row, label in enumerate(classes):
        masks[_MASK_FIELDS[_CLASS_INDEX[label]]][row] = True
    pose = torch.eye(4)
    pose[0, 3] = pose_offset
    inlier_rows = torch.tensor(inliers, dtype=torch.long)
    return {
        "query_index": index,
        "image_name": f"q{index}",
        "query_rows": torch.arange(len(winners)),
        "winner_anchor_ids": torch.tensor(winners),
        **masks,
        "inlier_query_rows": inlier_rows,
        "inlier_clean_mask": masks[_MASK_FIELDS[0]][inlier_rows],
        "exact_identity_positive_pairs": torch.tensor(positives).reshape(-1, 2),
        "identity_positive_count": len(positives),
        "failure_layers": layers,
        "pose_success": te_cm < 5.0 and ae_deg < 5.0,
        "te_cm": te_cm,
        "ae_deg": ae_deg,
        "correct_anchor_rank": correct_rank,
        "independent_mapping_validation_query": independent,
        "estimated_pose_w2c": pose,
    }


def _protocol(independent_count: int) -> dict:
    return {
        "positive_identity": exact_identity_positive_contract(),
        "positive_radius_px": 4.0,
        "alpha_minimum": 0.1,
        "required_matching_rank": 4,
        "required_visibility_rank": 4,
        "required_detectable_rank": 4,
        "pose_logdet_target": 0.0,
        "pose_min_eigenvalue_target": 0.0,
        "ransac_reprojection_px": 4.0,
        "ransac_seed": 7,
        "loo_pose_neighbors": 3,
        "affected_anchor_policy": "rebuild",
        "global_top1": True,
        "pose_solves_per_query": 1,
        "retrieval": False,
        "refinement": False,
        "independent_mapping_validation_query_count": independent_count,
        "independent_mapping_validation_available": bool(independent_count),
    }


def _evaluation(
    records: list[dict],
    *,
    map_sha: str,
    anchor_count: int,
    calibration_source_map_sha: str | None = None,
    candidate_arm: str | None = None,
) -> dict:
    names = [record["image_name"] for record in records]
    queries = []
    for record in records:
        masks = [record[field] for field in _MASK_FIELDS]
        positive_rows = record["identity_positive_count"]
        rank = record["correct_anchor_rank"]
        inlier_clean = record["inlier_clean_mask"]
        queries.append(
            {
                "query_index": record["query_index"],
                "image_name": record["image_name"],
                "te_cm": record["te_cm"],
                "ae_deg": record["ae_deg"],
                "positive_rows": positive_rows,
                "correct_anchor_rank_le_1": int(rank == 1) * positive_rows,
                "correct_anchor_rank_le_16": int(0 < rank <= 16) * positive_rows,
                "correct_winners": int(masks[0].sum()),
                "top1_exact_identity_correct_rows": int(masks[0].sum()),
                "top1_geometry_ambiguous_rows": int(masks[1].sum()),
                "top1_identity_incompatible_rows": int(masks[2].sum()),
                "top1_negative_rows": int(masks[3].sum()),
                "correspondences": len(record["query_rows"]),
                "inliers": len(record["inlier_query_rows"]),
                "clean_inliers": int(inlier_clean.sum()),
                "pose_solves": 1,
                "independent_mapping_validation_query": record[
                    "independent_mapping_validation_query"
                ],
            }
        )
    independent_count = sum(
        int(record["independent_mapping_validation_query"]) for record in records
    )
    failure_counts = {
        layer: sum(layer in record["failure_layers"] for record in records)
        for layer in ("L1", "L2", "L3", "L4")
    }
    class_totals = [sum(int(record[field].sum()) for record in records) for field in _MASK_FIELDS]
    feedback = {
        "schema": FEEDBACK_SCHEMA,
        "version": FEEDBACK_VERSION,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "positive_identity_contract": exact_identity_positive_contract(),
        "query_names": names,
        "records": records,
        "required_matching_rank": 4,
        "required_visibility_rank": 4,
        "required_detectable_rank": 4,
        "pose_logdet_target": 0.0,
        "pose_min_eigenvalue_target": 0.0,
        "failure_layer_counts": failure_counts,
        "independent_mapping_validation_query_count": independent_count,
        "top1_exact_identity_correct_count": class_totals[0],
        "top1_geometry_compatible_ambiguous_count": class_totals[1],
        "top1_identity_projective_incompatible_count": class_totals[2],
        "top1_negative_count": class_totals[3],
        "input_sha256": {
            "map": map_sha,
            "query_cache": _CACHE_SHA,
            "scene_calibration": _SCENE_CALIBRATION_SHA,
            "feedback_calibration_binding": _FEEDBACK_CALIBRATION_BINDING_SHA,
        },
    }
    contract = _protocol(independent_count)
    contract.update(
        {
            "calibration_binding_map_role": (
                "current_map"
                if calibration_source_map_sha is None
                else "candidate_parent_map"
            ),
            "calibration_binding_source_map_sha256": (
                map_sha
                if calibration_source_map_sha is None
                else calibration_source_map_sha
            ),
            "calibration_binding_candidate_arm": candidate_arm,
        }
    )
    return {
        "schema": "lafgs_v6_query_local_feedback_evaluation",
        "version": 4,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "queries": queries,
        "summary": {"anchor_count": anchor_count},
        "feedback": feedback,
        "contract": contract,
        "producer": {
            "git_commit": "a" * 40,
            "worktree_clean": True,
            "source_sha256": dict(_SOURCES),
            "torch_version": torch.__version__,
        },
        "input_sha256": {
            "map": map_sha,
            "metric": "d" * 64,
            "observation_cache": _CACHE_SHA,
            "scene_calibration": _SCENE_CALIBRATION_SHA,
            "feedback_calibration_binding": _FEEDBACK_CALIBRATION_BINDING_SHA,
        },
    }


def _save(value: dict, path: Path) -> str:
    torch.save(value, path)
    return sha256_file(path)


def _write_map(
    path: Path,
    identities: list[list[tuple[int, int]]],
    *,
    parent_sha: str | None = None,
    candidate_arm: str = "descriptor_loss",
    updated_rows: list[int] | None = None,
) -> str:
    offsets = [0]
    query_indices = []
    keypoint_indices = []
    for identity in identities:
        for query_index, keypoint_index in identity:
            query_indices.append(query_index)
            keypoint_indices.append(keypoint_index)
        offsets.append(len(query_indices))
    provenance = {
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
    }
    if parent_sha is not None:
        provenance.update(
            {
                "v6_parent_map_sha256": parent_sha,
                "v6_latest_proposal_arm": candidate_arm,
                "v6_proposal_history": [
                    {"parent_map_sha256": parent_sha, "arm": candidate_arm}
                ],
            }
        )
    state = {
        "schema": "lafgs_materialized_anchor_map",
        "version": 1,
        "anchor_ids": torch.arange(len(identities)),
        "v6_mapping_query_names": ["q0", "q1"],
        "provenance": provenance,
        "projective_anchor_observations": {
            "schema": "lafgs_projective_anchor_observations",
            "version": 1,
            "observation_offsets": torch.tensor(offsets),
            "query_indices": torch.tensor(query_indices),
            "keypoint_indices": torch.tensor(keypoint_indices),
        },
    }
    if updated_rows is not None:
        state["v6_descriptor_distillation"] = {
            "schema": "lafgs_v6_counterfactual_descriptor_loss_distillation",
            "version": 3,
            "updated_anchor_rows": torch.tensor(updated_rows),
            "updated_anchor_count": len(updated_rows),
        }
    return _save(state, path)


def _artifacts(tmp_path: Path) -> dict:
    baseline_map = tmp_path / "baseline_map.pt"
    baseline_map_sha = _write_map(
        baseline_map,
        [[(0, 0), (1, 0)], [(0, 1), (1, 1)], [(0, 2), (1, 2)]],
    )
    candidate_map = tmp_path / "candidate_map.pt"
    candidate_map_sha = _write_map(
        candidate_map,
        [[(0, 1), (1, 1)], [(0, 2), (1, 2)], [(0, 0)], [(1, 0)]],
        parent_sha=baseline_map_sha,
        updated_rows=[1, 2],
    )
    baseline_records = [
        _record(
            0,
            winners=[1, 1],
            classes=["negative", "exact"],
            positives=[(0, 0), (1, 1)],
            inliers=[0],
            te_cm=10.0,
            ae_deg=1.0,
            layers=["L3", "L4"],
            correct_rank=2,
            independent=True,
            pose_offset=0.10,
        ),
        _record(
            1,
            winners=[0, 2],
            classes=["exact", "ambiguous"],
            positives=[(0, 0), (1, 1)],
            inliers=[0],
            te_cm=4.0,
            ae_deg=1.0,
            layers=[],
            correct_rank=1,
            independent=False,
            pose_offset=0.04,
        ),
    ]
    candidate_records = [
        _record(
            0,
            winners=[2, 0],
            classes=["exact", "exact"],
            positives=[(0, 2), (1, 0)],
            inliers=[0, 1],
            te_cm=2.0,
            ae_deg=0.5,
            layers=[],
            correct_rank=1,
            independent=True,
            pose_offset=0.02,
        ),
        _record(
            1,
            winners=[1, 1],
            classes=["negative", "ambiguous"],
            positives=[(0, 3), (1, 0)],
            inliers=[0],
            te_cm=120.0,
            ae_deg=6.0,
            layers=["L3", "L4"],
            correct_rank=3,
            independent=True,
            pose_offset=1.20,
        ),
    ]
    baseline = _evaluation(baseline_records, map_sha=baseline_map_sha, anchor_count=3)
    candidate = _evaluation(
        candidate_records,
        map_sha=candidate_map_sha,
        anchor_count=4,
        calibration_source_map_sha=baseline_map_sha,
        candidate_arm="descriptor_loss",
    )
    baseline_feedback = tmp_path / "baseline_feedback.pt"
    candidate_feedback = tmp_path / "candidate_feedback.pt"
    return {
        "baseline_map": baseline_map,
        "baseline_map_sha": baseline_map_sha,
        "candidate_map": candidate_map,
        "candidate_map_sha": candidate_map_sha,
        "baseline": baseline,
        "candidate": candidate,
        "baseline_feedback": baseline_feedback,
        "candidate_feedback": candidate_feedback,
        "baseline_feedback_sha": _save(baseline, baseline_feedback),
        "candidate_feedback_sha": _save(candidate, candidate_feedback),
    }


def _compare_kwargs(artifacts: dict) -> dict:
    return {
        "baseline_feedback": artifacts["baseline_feedback"],
        "expected_baseline_feedback_sha256": artifacts["baseline_feedback_sha"],
        "candidate_feedback": artifacts["candidate_feedback"],
        "expected_candidate_feedback_sha256": artifacts["candidate_feedback_sha"],
        "baseline_map": artifacts["baseline_map"],
        "expected_baseline_map_sha256": artifacts["baseline_map_sha"],
        "candidate_map": artifacts["candidate_map"],
        "expected_candidate_map_sha256": artifacts["candidate_map_sha"],
    }


def test_paired_diagnostics_use_stable_anchor_identity_and_cover_metrics(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(tmp_path)
    output = tmp_path / "diagnostics.json"
    result = run(SimpleNamespace(**_compare_kwargs(artifacts), output=output))

    assert output.is_file()
    assert result["comparison_contract"]["cross_map_anchor_identity_verified"] is True
    assert (
        result["comparison_contract"]["shared_scene_calibration_sha256"]
        == _SCENE_CALIBRATION_SHA
    )
    assert (
        result["comparison_contract"][
            "shared_feedback_calibration_binding_sha256"
        ]
        == _FEEDBACK_CALIBRATION_BINDING_SHA
    )
    assert result["inputs"]["candidate_map"]["matched_parent_anchor_count"] == 2
    assert result["inputs"]["candidate_map"]["new_anchor_count"] == 2
    full = result["scopes"]["full"]
    assert full["top1"]["changed_row_count"] == 2
    assert full["top1"]["exact_wrong_to_correct_count"] == 1
    assert full["top1"]["exact_correct_to_wrong_count"] == 1
    assert full["top1"]["winner_class_transition"]["negative"]["exact_identity_correct"] == 1
    assert full["poselib_inlier_pair_set"]["changed_query_count"] == 2
    assert full["pose_success_flips"]["failure_to_success_count"] == 1
    assert full["pose_success_flips"]["success_to_failure_count"] == 1
    assert full["failure_layer_transitions"]["L3"]["entered_count"] == 1
    assert full["failure_layer_transitions"]["L3"]["exited_count"] == 1
    assert full["metrics"]["candidate"]["catastrophic_100cm_count"] == 1
    assert full["correct_rank"]["candidate"]["query_minimum_correct_rank_median"] == 2.0
    assert full["pose_output_changed"]["changed_query_count"] == 2
    coverage = full["updated_anchor_coverage"]
    assert coverage["winner"]["updated_occurrence_count"] == 3
    assert coverage["poselib_inlier"]["updated_occurrence_count"] == 2
    assert coverage["catastrophic_queries"]["with_updated_inlier_count"] == 1
    independent = result["scopes"]["independent_validation_intersection"]
    assert independent["query_indices"] == [0]
    with pytest.raises(FileExistsError):
        run(SimpleNamespace(**_compare_kwargs(artifacts), output=output))


@pytest.mark.parametrize(
    "arm", ("descriptor_loss", "selection", "reconstruction")
)
def test_paired_calibration_lineage_accepts_each_formal_candidate_arm(
    tmp_path: Path,
    arm: str,
) -> None:
    artifacts = _artifacts(tmp_path)
    state = torch.load(
        artifacts["candidate_map"], map_location="cpu", weights_only=False
    )
    state["provenance"]["v6_latest_proposal_arm"] = arm
    state["provenance"]["v6_proposal_history"][-1]["arm"] = arm
    artifacts["candidate_map_sha"] = _save(state, artifacts["candidate_map"])
    candidate = _evaluation(
        artifacts["candidate"]["feedback"]["records"],
        map_sha=artifacts["candidate_map_sha"],
        anchor_count=4,
        calibration_source_map_sha=artifacts["baseline_map_sha"],
        candidate_arm=arm,
    )
    artifacts["candidate_feedback_sha"] = _save(
        candidate, artifacts["candidate_feedback"]
    )

    result = compare_feedback_files(**_compare_kwargs(artifacts))

    assert result["comparison_contract"]["candidate_calibration_binding_arm"] == arm
    assert result["comparison_contract"][
        "immutable_calibration_source_map_sha256"
    ] == artifacts["baseline_map_sha"]


def test_rank_zero_transitions_and_missing_pose_fail_closed(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    baseline = deepcopy(artifacts["baseline"])
    candidate = deepcopy(artifacts["candidate"])
    baseline_record = baseline["feedback"]["records"][0]
    baseline_record["exact_identity_positive_pairs"] = torch.empty((0, 2), dtype=torch.long)
    baseline_record["identity_positive_count"] = 0
    baseline_record["correct_anchor_rank"] = 0
    baseline_record["top1_exact_identity_correct_mask"][:] = False
    baseline_record["top1_negative_mask"][:] = True
    candidate_record = candidate["feedback"]["records"][1]
    candidate_record["exact_identity_positive_pairs"] = torch.empty((0, 2), dtype=torch.long)
    candidate_record["identity_positive_count"] = 0
    candidate_record["correct_anchor_rank"] = 0
    baseline = _evaluation(
        baseline["feedback"]["records"],
        map_sha=artifacts["baseline_map_sha"],
        anchor_count=3,
    )
    candidate = _evaluation(
        candidate["feedback"]["records"],
        map_sha=artifacts["candidate_map_sha"],
        anchor_count=4,
        calibration_source_map_sha=artifacts["baseline_map_sha"],
        candidate_arm="descriptor_loss",
    )
    artifacts["baseline_feedback_sha"] = _save(baseline, artifacts["baseline_feedback"])
    artifacts["candidate_feedback_sha"] = _save(candidate, artifacts["candidate_feedback"])
    result = compare_feedback_files(**_compare_kwargs(artifacts))
    rank = result["scopes"]["full"]["correct_rank"]
    assert rank["availability_gained_query_indices"] == [0]
    assert rank["availability_lost_query_indices"] == [1]

    candidate["feedback"]["records"][0].pop("estimated_pose_w2c")
    artifacts["candidate_feedback_sha"] = _save(candidate, artifacts["candidate_feedback"])
    with pytest.raises(ValueError, match="estimated pose is missing"):
        compare_feedback_files(**_compare_kwargs(artifacts))


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("cache", "observation caches differ"),
        ("scene_calibration", "scene calibrations differ"),
        ("calibration_binding", "feedback calibration bindings differ"),
        ("protocol", "evaluation protocols differ"),
        ("source", "evaluator source registries differ"),
        ("registry", "ordered query registry differs"),
    ),
)
def test_pairing_rejects_mismatched_inputs(
    tmp_path: Path, mutation: str, message: str
) -> None:
    artifacts = _artifacts(tmp_path)
    candidate = deepcopy(artifacts["candidate"])
    if mutation == "cache":
        candidate["input_sha256"]["observation_cache"] = "e" * 64
        candidate["feedback"]["input_sha256"]["query_cache"] = "e" * 64
    elif mutation == "scene_calibration":
        candidate["input_sha256"]["scene_calibration"] = "1" * 64
        candidate["feedback"]["input_sha256"]["scene_calibration"] = "1" * 64
    elif mutation == "calibration_binding":
        candidate["input_sha256"]["feedback_calibration_binding"] = "2" * 64
        candidate["feedback"]["input_sha256"][
            "feedback_calibration_binding"
        ] = "2" * 64
    elif mutation == "protocol":
        candidate["contract"]["positive_radius_px"] = 8.0
    elif mutation == "source":
        candidate["producer"]["source_sha256"][
            "map_learning/v6_feedback_evaluator.py"
        ] = "e" * 64
    else:
        candidate["feedback"]["query_names"][1] = "other"
    artifacts["candidate_feedback_sha"] = _save(candidate, artifacts["candidate_feedback"])
    with pytest.raises(ValueError, match=message):
        compare_feedback_files(**_compare_kwargs(artifacts))


@pytest.mark.parametrize(
    "field",
    ("scene_calibration", "feedback_calibration_binding"),
)
def test_evaluation_rejects_outer_embedded_calibration_sha_mismatch(
    tmp_path: Path, field: str
) -> None:
    artifacts = _artifacts(tmp_path)
    candidate = deepcopy(artifacts["candidate"])
    candidate["feedback"]["input_sha256"][field] = "0" * 64
    artifacts["candidate_feedback_sha"] = _save(
        candidate, artifacts["candidate_feedback"]
    )

    with pytest.raises(ValueError, match=rf"outer/embedded {field} SHA differs"):
        compare_feedback_files(**_compare_kwargs(artifacts))


@pytest.mark.parametrize(
    ("registry", "field"),
    (
        ("outer", "scene_calibration"),
        ("outer", "feedback_calibration_binding"),
        ("embedded", "scene_calibration"),
        ("embedded", "feedback_calibration_binding"),
    ),
)
def test_evaluation_requires_calibration_sha_registry(
    tmp_path: Path, registry: str, field: str
) -> None:
    artifacts = _artifacts(tmp_path)
    candidate = deepcopy(artifacts["candidate"])
    target = (
        candidate["input_sha256"]
        if registry == "outer"
        else candidate["feedback"]["input_sha256"]
    )
    target.pop(field)
    artifacts["candidate_feedback_sha"] = _save(
        candidate, artifacts["candidate_feedback"]
    )

    with pytest.raises(ValueError, match="calibration SHA|embedded .* SHA"):
        compare_feedback_files(**_compare_kwargs(artifacts))


def test_map_lineage_and_unique_fingerprints_fail_closed(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    state = torch.load(artifacts["candidate_map"], map_location="cpu", weights_only=False)
    state["provenance"]["v6_parent_map_sha256"] = "b" * 64
    artifacts["candidate_map_sha"] = _save(state, artifacts["candidate_map"])
    candidate = _evaluation(
        artifacts["candidate"]["feedback"]["records"],
        map_sha=artifacts["candidate_map_sha"],
        anchor_count=4,
        calibration_source_map_sha=artifacts["baseline_map_sha"],
        candidate_arm="descriptor_loss",
    )
    artifacts["candidate_feedback_sha"] = _save(candidate, artifacts["candidate_feedback"])
    with pytest.raises(ValueError, match="parent is not the baseline"):
        compare_feedback_files(**_compare_kwargs(artifacts))

    duplicate_map = tmp_path / "duplicate_map.pt"
    duplicate_sha = _write_map(
        duplicate_map,
        [[(0, 1), (1, 1)], [(0, 1), (1, 1)], [(0, 0)], [(1, 0)]],
        parent_sha=artifacts["baseline_map_sha"],
        updated_rows=[1, 2],
    )
    duplicate_candidate = _evaluation(
        artifacts["candidate"]["feedback"]["records"],
        map_sha=duplicate_sha,
        anchor_count=4,
        calibration_source_map_sha=artifacts["baseline_map_sha"],
        candidate_arm="descriptor_loss",
    )
    artifacts["candidate_map"] = duplicate_map
    artifacts["candidate_map_sha"] = duplicate_sha
    artifacts["candidate_feedback_sha"] = _save(
        duplicate_candidate, artifacts["candidate_feedback"]
    )
    with pytest.raises(ValueError, match="fingerprints are not unique"):
        compare_feedback_files(**_compare_kwargs(artifacts))


def test_candidate_calibration_lineage_rejects_forged_proposal_history(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(tmp_path)
    state = torch.load(
        artifacts["candidate_map"], map_location="cpu", weights_only=False
    )
    state["provenance"]["v6_proposal_history"][-1]["parent_map_sha256"] = (
        "0" * 64
    )
    artifacts["candidate_map_sha"] = _save(state, artifacts["candidate_map"])
    candidate = _evaluation(
        artifacts["candidate"]["feedback"]["records"],
        map_sha=artifacts["candidate_map_sha"],
        anchor_count=4,
        calibration_source_map_sha=artifacts["baseline_map_sha"],
        candidate_arm="descriptor_loss",
    )
    artifacts["candidate_feedback_sha"] = _save(
        candidate, artifacts["candidate_feedback"]
    )

    with pytest.raises(ValueError, match="proposal history differs"):
        compare_feedback_files(**_compare_kwargs(artifacts))
