from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from common.hashing import sha256_file
from localization.pose_solver import pose_error
from map_learning.v8_feedback_controller import task_error
from map_learning.v21_pose_feedback_transductive import (
    METADATA_FIELD,
    PROTOTYPE_FEATURE_FIELD,
    PROTOTYPE_OWNER_FIELD,
    atomic_torch_save_fresh,
    build_cached_evaluation_payload,
    build_pose_feedback_transductive_candidate,
    replay_pose_with_contract,
    source_record,
    validate_cached_evaluation_payload,
    validate_candidate_map,
    verify_source_record,
)
from map_learning.v21_test_cache import (
    build_shard_registry,
    validate_cache_payload,
    validate_split_manifest,
)
from tests.test_v21_test_cache import _manifest, _query_payload


BASELINE_CONTRACT = {
    "matching": "exact_global_cosine_top1_lower_anchor_row_tie_break",
    "pose_solver": "single_standard_poselib_absolute_pose",
    "cached_keypoints": "native_integer_grid_without_pixel_center_offset",
    "pose_solver_points_2d": "cached_keypoints_plus_0.5",
    "pixel_center_offset": 0.5,
    "reprojection_error_px": 12.0,
    "confidence": 0.99999,
    "maximum_iterations": 100000,
    "minimum_iterations": 1000,
    "seed": 2026,
    "r5": "translation_cm_strictly_below_5_and_rotation_deg_strictly_below_5",
}


def _source(path: str, digest: str) -> dict:
    return {"path": path, "sha256": digest, "size_bytes": 1}


class MarkerSolver:
    """A deterministic discontinuous plant with role-specific outcomes."""

    def __init__(self, truth_by_marker: dict[int, np.ndarray]) -> None:
        self.truth_by_marker = truth_by_marker

    def __call__(self, points_2d, points_3d, intrinsic, **kwargs):
        del intrinsic, kwargs
        marker = int(round(float(np.asarray(points_2d)[0, 0])))
        truth = self.truth_by_marker[marker]
        changed = bool(np.isclose(np.asarray(points_3d)[0, 0], 2.0))
        control = marker == 4
        failure = changed if control else not changed
        pose = truth.copy()
        pose[0, 3] += 0.10 if failure else (0.03 if changed else 0.0)
        return SimpleNamespace(
            pose_w2c=pose.astype(np.float32),
            inliers=np.arange(len(points_2d), dtype=np.int64),
        )


def _stable_map() -> dict:
    return {
        "schema": "lafgs_materialized_anchor_map",
        "version": 1,
        "anchor_ids": torch.tensor([10, 11, 12]),
        "anchor_xyz": torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]]
        ),
        "anchor_features": torch.tensor(
            [[0.8, 0.6, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        ),
        "nested_frozen_provenance": {
            "tensor": torch.tensor([7, 9], dtype=torch.int16),
            "names": ["mapping", "frozen"],
        },
    }


def _one_record_cache(tmp_path: Path, *, role: str, marker: float) -> dict:
    manifest, cameras, _, _ = _manifest(tmp_path / role)
    payload = _query_payload(manifest, cameras, role=role)
    selected = validate_split_manifest(manifest, role=role)
    registry = build_shard_registry(
        [selected[0]],
        role=role,
        shard_count=1,
        split_manifest_sha256="a" * 64,
    )
    payload["shard_registry"] = registry
    payload["role_query_count"] = 1
    payload["query_count"] = 1
    payload["baseline_contract"] = dict(BASELINE_CONTRACT)
    record = payload["records"][0]
    record["keypoints"] = torch.tensor([[marker, 1.0], [2.0, 2.0]])
    record["descriptors"] = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    record["winner_scores"] = torch.tensor([0.8, 1.0])
    validate_cache_payload(payload)
    return payload


def _set_cached_baseline(cache: dict, outcome: dict) -> None:
    record = cache["records"][0]
    record["baseline_pose_w2c"] = outcome["pose_w2c"]
    record["baseline_inliers"] = outcome["inlier_query_rows"]
    record["baseline_inlier_count"] = outcome["inlier_count"]
    record["baseline_rotation_error_deg"] = outcome["rotation_error_deg"]
    record["baseline_translation_error_cm"] = outcome["translation_error_cm"]
    record["baseline_task_error"] = outcome["task_error"]
    record["baseline_r5"] = outcome["r5_success"]


def _replay(cache: dict, stable: dict, solver: MarkerSolver, rows: torch.Tensor) -> dict:
    record = cache["records"][0]
    return replay_pose_with_contract(
        keypoints=record["keypoints"] + 0.5,
        anchor_rows=rows,
        anchor_xyz=stable["anchor_xyz"],
        intrinsic=record["intrinsics"],
        ground_truth_w2c=record["pose_w2c"],
        baseline_contract=BASELINE_CONTRACT,
        solver=solver,
    )


def _oracle(cache: dict, stable: dict, solver: MarkerSolver) -> dict:
    record = cache["records"][0]
    winners = record["winner_anchor_rows"]
    patched = winners.clone()
    patched[0] = 2
    baseline = _replay(cache, stable, solver, winners)
    recovered = _replay(cache, stable, solver, patched)
    query_index = int(record["query_index"])
    oracle_record = {
        "query_index": query_index,
        "image_name": record["image_name"],
        "sequence_id": record["sequence_id"],
        "block_id": record["block_id"],
        "controller_authorized": False,
        "legal_positive_csr": {
            "positive_evidence_mode": "gaussian_geometry_supported_upper_bound",
            "geometry_supported_candidate": True,
            "deployable_positive_authorized": False,
        },
        "protection": {
            "positive_evidence_mode": "gaussian_geometry_supported_upper_bound"
        },
        "baseline": baseline,
        "one_assignment_lower_bound": recovered,
        "correction_candidates": {
            "candidate_rows": torch.tensor([0]),
            "candidate_positive_anchor_rows": torch.tensor([2]),
        },
        "recovery_bundle": {
            "query_rows": torch.tensor([0]),
            "anchor_rows": torch.tensor([2]),
            "pose": recovered,
            "exact_delta_r5": 1,
            "inclusion_minimal": True,
        },
    }
    return {
        "schema": "lafgs_v21_pose_recovery_oracle_aggregate",
        "version": 1,
        "protocol": "test_adapted",
        "uses_test_queries": True,
        "test_adapted": True,
        "role": "adaptation",
        "geometry_recovery_is_upper_bound_only": True,
        "pose_recovery_is_diagnostic_upper_bound_only": True,
        "deployment_authorized": False,
        "correspondence_identity_authority_present": False,
        "source_query_count": 1,
        "input": {
            "frozen_map": "/stable_map.pt",
            "frozen_map_sha256": "b" * 64,
            "adaptation_caches": [
                {"path": "/adaptation.pt", "sha256": "c" * 64}
            ],
            "gaussian_support": [
                {"path": "/gaussian_support.pt", "sha256": "1" * 64}
            ],
            "correspondence_truth": None,
        },
        "oracle_shards": [
            {"path": "/oracle_shard.pt", "sha256": "2" * 64}
        ],
        "frontend_query_registry": [
            {
                "ordinal": 0,
                "query_index": query_index,
                "image_name": record["image_name"],
                "source_record_sha256": record["source_record_sha256"],
            }
        ],
        "records": [oracle_record],
    }


@pytest.fixture
def setup(tmp_path: Path):
    stable = _stable_map()
    adaptation = _one_record_cache(tmp_path, role="adaptation", marker=1.0)
    control = _one_record_cache(tmp_path, role="control", marker=3.0)
    truth_by_marker = {
        2: adaptation["records"][0]["pose_w2c"].numpy(),
        4: control["records"][0]["pose_w2c"].numpy(),
    }
    solver = MarkerSolver(truth_by_marker)
    adaptation_baseline = _replay(
        adaptation,
        stable,
        solver,
        adaptation["records"][0]["winner_anchor_rows"],
    )
    control_baseline = _replay(
        control, stable, solver, control["records"][0]["winner_anchor_rows"]
    )
    assert adaptation_baseline["r5_success"] is False
    assert control_baseline["r5_success"] is True
    _set_cached_baseline(adaptation, adaptation_baseline)
    _set_cached_baseline(control, control_baseline)
    validate_cache_payload(adaptation)
    validate_cache_payload(control)
    oracle = _oracle(adaptation, stable, solver)
    return stable, adaptation, control, oracle, solver


def _candidate(setup, **kwargs) -> dict:
    stable, adaptation, _, oracle, solver = setup
    return build_pose_feedback_transductive_candidate(
        stable_map=stable,
        adaptation_cache_payloads=[adaptation],
        gaussian_oracle_aggregate=oracle,
        stable_map_source=_source("/stable_map.pt", "b" * 64),
        adaptation_cache_sources=[_source("/adaptation.pt", "c" * 64)],
        oracle_source=_source("/oracle.pt", "d" * 64),
        solver=solver,
        **kwargs,
    )


def test_candidate_only_appends_exact_owner_prototypes(setup) -> None:
    stable, adaptation, _, _, _ = setup
    candidate = _candidate(setup)
    assert set(candidate) == set(stable) | {
        PROTOTYPE_FEATURE_FIELD,
        PROTOTYPE_OWNER_FIELD,
        METADATA_FIELD,
    }
    for key in stable:
        if isinstance(stable[key], torch.Tensor):
            assert torch.equal(candidate[key], stable[key])
        elif key != "nested_frozen_provenance":
            assert candidate[key] == stable[key]
    assert torch.equal(
        candidate["nested_frozen_provenance"]["tensor"],
        stable["nested_frozen_provenance"]["tensor"],
    )
    assert (
        candidate["nested_frozen_provenance"]["names"]
        == stable["nested_frozen_provenance"]["names"]
    )
    assert candidate[PROTOTYPE_OWNER_FIELD].tolist() == [2]
    assert torch.allclose(
        candidate[PROTOTYPE_FEATURE_FIELD],
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
    metadata = validate_candidate_map(candidate, stable_map=stable)
    assert metadata["test_adapted"] is True
    assert metadata["identity_truth_claimed"] is False
    assert metadata["deployment_authorized"] is False
    assert metadata["control_features_consumed"] is False
    assert metadata["selected_actions"][0]["query_index"] == int(
        adaptation["records"][0]["query_index"]
    )

    candidate["anchor_xyz"][0, 0] = 17.0
    candidate["nested_frozen_provenance"]["tensor"][0] = 42
    assert float(stable["anchor_xyz"][0, 0]) == 0.0
    assert stable["nested_frozen_provenance"]["tensor"].tolist() == [7, 9]
    candidate["anchor_xyz"][0, 0] = stable["anchor_xyz"][0, 0]
    candidate["nested_frozen_provenance"]["tensor"][0] = 7

    tampered = deepcopy(candidate)
    tampered["anchor_xyz"][0, 0] = 99.0
    with pytest.raises(ValueError, match="changed frozen base field"):
        validate_candidate_map(tampered, stable_map=stable)

    inconsistent = deepcopy(candidate)
    inconsistent[METADATA_FIELD]["selected_actions"][0][
        "one_assignment_translation_error_cm"
    ] = 999.0
    with pytest.raises(ValueError, match="action row is invalid"):
        validate_candidate_map(inconsistent, stable_map=stable)

    rebound = deepcopy(candidate)
    rebound[PROTOTYPE_FEATURE_FIELD][0, 0] = 0.5
    with pytest.raises(ValueError, match="candidate contract is invalid"):
        validate_candidate_map(rebound, stable_map=stable)


def test_candidate_enforces_te_margin_and_complete_provisional_bundle(
    setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="no eligible"):
        _candidate(setup, require_one_assignment_translation_below_cm=2.0)

    query_index = int(setup[1]["records"][0]["query_index"])
    monkeypatch.setattr(
        "map_learning.v21_pose_feedback_transductive._validate_calibration_join",
        lambda *args, **kwargs: {query_index: set()},
    )
    with pytest.raises(ValueError, match="no eligible"):
        build_pose_feedback_transductive_candidate(
            stable_map=setup[0],
            adaptation_cache_payloads=[setup[1]],
            gaussian_oracle_aggregate=setup[3],
            stable_map_source=_source("/stable_map.pt", "b" * 64),
            adaptation_cache_sources=[_source("/adaptation.pt", "c" * 64)],
            oracle_source=_source("/oracle.pt", "d" * 64),
            provisional_calibration={"synthetic": True},
            calibration_source=_source("/calibration.pt", "e" * 64),
            require_provisional_edge=True,
            solver=setup[4],
        )


def test_cached_evaluator_reports_adaptation_gain_and_heldout_catastrophe(setup) -> None:
    stable, adaptation, control, _, solver = setup
    candidate = _candidate(setup)
    stable_source = _source("/stable_map.pt", "b" * 64)
    candidate_source = _source("/candidate.pt", "f" * 64)
    adaptation_result = build_cached_evaluation_payload(
        stable_map=stable,
        candidate_map=candidate,
        cache_payloads=[adaptation],
        stable_map_source=stable_source,
        candidate_map_source=candidate_source,
        cache_sources=[_source("/adaptation.pt", "c" * 64)],
        device="cpu",
        matcher_chunk_size=2,
        solver=solver,
    )
    assert adaptation_result["summary"]["paired_r5_gain_count"] == 1
    assert adaptation_result["summary"]["paired_r5_loss_count"] == 0
    assert adaptation_result["records"][0]["winner_flip_count"] == 1

    control_result = build_cached_evaluation_payload(
        stable_map=stable,
        candidate_map=candidate,
        cache_payloads=[control],
        stable_map_source=stable_source,
        candidate_map_source=candidate_source,
        cache_sources=[_source("/control.pt", "9" * 64)],
        device="cpu",
        matcher_chunk_size=2,
        solver=solver,
    )
    validate_cached_evaluation_payload(control_result)
    assert control_result["evaluation_role"] == "control"
    assert control_result["heldout_outcomes_feed_candidate"] is False
    assert control_result["candidate_map_mutated"] is False
    assert control_result["summary"]["paired_r5_gain_count"] == 0
    assert control_result["summary"]["paired_r5_loss_count"] == 1
    assert control_result["summary"]["catastrophe_count"] == 1
    record = control_result["records"][0]
    assert record["paired_delta_translation_error_cm"] > 9.0
    assert record["catastrophe"] is True


def test_heldout_cache_cannot_overlap_candidate_formation_source(setup) -> None:
    stable, _, control, _, solver = setup
    candidate = _candidate(setup)
    with pytest.raises(ValueError, match="consumed by candidate formation"):
        build_cached_evaluation_payload(
            stable_map=stable,
            candidate_map=candidate,
            cache_payloads=[control],
            stable_map_source=_source("/stable_map.pt", "b" * 64),
            candidate_map_source=_source("/candidate.pt", "f" * 64),
            cache_sources=[_source("/adaptation.pt", "c" * 64)],
            device="cpu",
            solver=solver,
        )


def test_fresh_evaluation_output_is_reload_validated(
    setup, tmp_path: Path
) -> None:
    stable, adaptation, _, _, solver = setup
    candidate = _candidate(setup)
    result = build_cached_evaluation_payload(
        stable_map=stable,
        candidate_map=candidate,
        cache_payloads=[adaptation],
        stable_map_source=_source("/stable_map.pt", "b" * 64),
        candidate_map_source=_source("/candidate.pt", "f" * 64),
        cache_sources=[_source("/adaptation.pt", "c" * 64)],
        device="cpu",
        solver=solver,
    )
    output = tmp_path / "evaluation.pt"
    atomic_torch_save_fresh(
        result, output, validator=validate_cached_evaluation_payload
    )
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        atomic_torch_save_fresh(
            result, output, validator=validate_cached_evaluation_payload
        )
    assert output.read_bytes() == original


def test_task_and_pose_fixture_are_standard() -> None:
    truth = np.eye(4, dtype=np.float32)
    predicted = truth.copy()
    predicted[0, 3] = 0.03
    rotation, translation = pose_error(predicted, truth)
    assert rotation == pytest.approx(0.0)
    assert translation == pytest.approx(3.0)
    assert task_error(translation, rotation) == pytest.approx(0.6)


def test_source_recheck_binds_size_and_sha(tmp_path: Path) -> None:
    path = tmp_path / "source.pt"
    path.write_bytes(b"immutable")
    source = source_record(path, sha256_file_fn=sha256_file)
    verify_source_record(source, sha256_file_fn=sha256_file)
    path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="bound source changed"):
        verify_source_record(source, sha256_file_fn=sha256_file)
