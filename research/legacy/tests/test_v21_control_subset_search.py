from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from localization.matcher import global_owner_prototype_top1
from map_learning.v21_control_subset_search import (
    ControlSubsetSearchStopped,
    build_control_selected_candidate,
    validate_control_selected_candidate,
    validate_control_subset_search_audit,
)
from map_learning.v21_pose_feedback_transductive import (
    METADATA_FIELD,
    atomic_torch_save_fresh,
    build_pose_feedback_transductive_candidate,
)
from map_learning.v21_test_cache import validate_cache_payload
from tests.test_v21_pose_feedback_transductive import (
    MarkerSolver,
    _one_record_cache,
    _oracle,
    _replay,
    _set_cached_baseline,
    _source,
    _stable_map,
)


class GainSolver:
    """Make owner row two recover a deterministic R5 failure."""

    def __init__(self, truth_by_marker: dict[int, np.ndarray]) -> None:
        self.truth_by_marker = truth_by_marker

    def __call__(self, points_2d, points_3d, intrinsic, **kwargs):
        del intrinsic, kwargs
        marker = int(round(float(np.asarray(points_2d)[0, 0])))
        changed = bool(np.isclose(np.asarray(points_3d)[0, 0], 2.0))
        pose = self.truth_by_marker[marker].copy()
        pose[0, 3] += 0.03 if changed else 0.10
        return SimpleNamespace(
            pose_w2c=pose.astype(np.float32),
            inliers=np.arange(len(points_2d), dtype=np.int64),
        )


@pytest.fixture
def search_setup(tmp_path: Path):
    stable = _stable_map()
    adaptation = _one_record_cache(tmp_path, role="adaptation", marker=1.0)
    control = _one_record_cache(tmp_path, role="control", marker=3.0)
    formation_truth = {
        2: adaptation["records"][0]["pose_w2c"].numpy(),
        4: control["records"][0]["pose_w2c"].numpy(),
    }
    formation_solver = MarkerSolver(formation_truth)
    adaptation_baseline = _replay(
        adaptation,
        stable,
        formation_solver,
        adaptation["records"][0]["winner_anchor_rows"],
    )
    _set_cached_baseline(adaptation, adaptation_baseline)
    validate_cache_payload(adaptation)
    oracle = _oracle(adaptation, stable, formation_solver)
    parent = build_pose_feedback_transductive_candidate(
        stable_map=stable,
        adaptation_cache_payloads=[adaptation],
        gaussian_oracle_aggregate=oracle,
        stable_map_source=_source("/stable_map.pt", "b" * 64),
        adaptation_cache_sources=[_source("/adaptation.pt", "c" * 64)],
        oracle_source=_source("/oracle.pt", "d" * 64),
        solver=formation_solver,
    )
    gain_solver = GainSolver(formation_truth)
    control_baseline = _replay(
        control,
        stable,
        gain_solver,
        control["records"][0]["winner_anchor_rows"],
    )
    assert control_baseline["r5_success"] is False
    _set_cached_baseline(control, control_baseline)
    validate_cache_payload(control)
    return stable, parent, control, gain_solver


def _search(search_setup):
    stable, parent, control, solver = search_setup
    return build_control_selected_candidate(
        stable_map=stable,
        parent_candidate=parent,
        control_cache_payloads=[control],
        stable_map_source=_source("/stable_map.pt", "b" * 64),
        parent_candidate_source=_source("/parent.pt", "f" * 64),
        control_cache_sources=[_source("/control.pt", "9" * 64)],
        activation_threshold_menu=(None, 0.95),
        beam_width=2,
        maximum_beam_depth=2,
        device="cpu",
        matcher_chunk_size=2,
        solver=solver,
    )


def test_control_search_emits_explicit_quarantined_positive_subset(search_setup) -> None:
    stable, parent, _, _ = search_setup
    selected = _search(search_setup)
    metadata = validate_control_selected_candidate(
        selected, stable_map=stable, parent_candidate=parent
    )
    assert metadata["control_used_for_selection"] is True
    assert metadata["confirmation_unread_during_selection"] is True
    assert metadata["inputs"]["confirmation_caches"] == []
    assert metadata["deployment_authorized"] is False
    assert metadata["selected_control_summary"]["paired_r5_gain_count"] == 1
    assert metadata["selected_control_summary"]["paired_r5_loss_count"] == 0
    assert metadata["selected_action_count"] == 1
    assert metadata["search"]["exact_poselib_for_every_evaluated_subset"] is True

    tampered = deepcopy(selected)
    tampered[METADATA_FIELD]["confirmation_unread_during_selection"] = False
    with pytest.raises(ValueError, match="candidate contract is invalid"):
        validate_control_selected_candidate(
            tampered, stable_map=stable, parent_candidate=parent
        )


def test_control_search_fails_closed_without_positive_zero_loss_arm(search_setup) -> None:
    stable, parent, control, _ = search_setup
    record = control["records"][0]
    truth = {4: record["pose_w2c"].numpy()}
    harmful_solver = MarkerSolver(truth)
    baseline = _replay(
        control, stable, harmful_solver, record["winner_anchor_rows"]
    )
    assert baseline["r5_success"] is True
    _set_cached_baseline(control, baseline)
    validate_cache_payload(control)
    with pytest.raises(ControlSubsetSearchStopped, match="no zero-loss positive-R5") as stopped:
        build_control_selected_candidate(
            stable_map=stable,
            parent_candidate=parent,
            control_cache_payloads=[control],
            stable_map_source=_source("/stable_map.pt", "b" * 64),
            parent_candidate_source=_source("/parent.pt", "f" * 64),
            control_cache_sources=[_source("/control.pt", "9" * 64)],
            activation_threshold_menu=(None,),
            beam_width=1,
            maximum_beam_depth=1,
            device="cpu",
            solver=harmful_solver,
        )
    audit = stopped.value.audit
    validate_control_subset_search_audit(audit)
    assert audit["decision"] == "STOP_NO_ACTION"
    assert audit["artifact_is_candidate_map"] is False
    assert audit["confirmation_unread_during_selection"] is True
    assert audit["aggregate"]["accepted_subset_count"] == 0
    output = Path(control["records"][0]["image_path"]).parent / "stop_audit.pt"
    atomic_torch_save_fresh(
        audit, output, validator=validate_control_subset_search_audit
    )
    with pytest.raises(FileExistsError):
        atomic_torch_save_fresh(
            audit, output, validator=validate_control_subset_search_audit
        )


def test_owner_prototype_absolute_activation_threshold() -> None:
    query = torch.tensor([[0.8, 0.6]])
    anchors = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
    prototypes = torch.tensor([[1.0, 0.0]])
    owners = torch.tensor([1])
    active = global_owner_prototype_top1(
        query,
        anchors,
        prototypes,
        owners,
        prototype_activation_threshold=0.75,
    )
    blocked = global_owner_prototype_top1(
        query,
        anchors,
        prototypes,
        owners,
        prototype_activation_threshold=0.85,
    )
    assert active.anchor_indices.tolist() == [1]
    assert blocked.anchor_indices.tolist() == [0]
    with pytest.raises(ValueError, match="threshold"):
        global_owner_prototype_top1(
            query,
            anchors,
            prototypes,
            owners,
            prototype_activation_threshold=1.1,
        )
