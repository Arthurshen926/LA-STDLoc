from argparse import Namespace
from pathlib import Path

import pytest
import torch

from common.hashing import sha256_file
from scripts import materialize_rendered_track_completion_oracle as oracle


def _map(track_ids: list[int]) -> dict:
    count = len(track_ids)
    rows = torch.arange(count)
    return {
        "schema": "lafgs_materialized_anchor_map",
        "version": 1,
        "anchor_ids": rows.clone(),
        "anchor_xyz": torch.stack(
            (rows.float(), rows.float() + 1, rows.float() + 2), 1
        ),
        "anchor_features": torch.nn.functional.one_hot(rows % 3, 3).float(),
        "source_primitive_ids": torch.full((count,), -1, dtype=torch.long),
        "track_cluster_ids": torch.tensor(track_ids),
        "anchor_type": torch.ones(count, dtype=torch.long),
        "dependency_group_ids": rows + 10,
        "coarse_dependency_group_ids": rows + 20,
        "fine_identity_ids": rows + 30,
        "parent_source_track_ids": rows + 40,
        "repair_child_index": torch.zeros(count, dtype=torch.long),
        "repair_parent_child_count": torch.ones(count, dtype=torch.long),
        "provenance": {"fixture": True},
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    candidate = _map([7, 11, 19])
    selected_rows = torch.tensor([0, 2])
    selected = {
        key: (
            value[selected_rows]
            if torch.is_tensor(value) and value.ndim and value.shape[0] == 3
            else value
        )
        for key, value in candidate.items()
    }
    selected["anchor_ids"] = torch.arange(2)
    selected_dependency = torch.unique(
        torch.floor(selected["anchor_xyz"] / 1.0).long(),
        dim=0,
        return_inverse=True,
    )[1]
    selected["dependency_group_ids"] = selected_dependency
    selected["coarse_dependency_group_ids"] = selected_dependency.clone()
    selected["source_dependency_group_ids"] = torch.full((2,), -1, dtype=torch.long)
    statistics = {
        "schema": "lafgs_rendered_track_full_mapping_loo_statistics",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "queries": [
            {"te_cm": 1.0, "ae_deg": 1.0},
            {"te_cm": 7.0, "ae_deg": 1.0},
            {"te_cm": 130.0, "ae_deg": 8.0},
        ],
    }
    paths = (tmp_path / "candidate.pt", tmp_path / "selected.pt", tmp_path / "stats.pt")
    for path, payload in zip(paths, (candidate, selected, statistics)):
        torch.save(payload, path)
    return paths


def _args(paths: tuple[Path, Path, Path], output: Path) -> Namespace:
    candidate, selected, statistics = paths
    return Namespace(
        candidate_map=candidate,
        expected_candidate_map_sha256=sha256_file(candidate),
        selected_map=selected,
        expected_selected_map_sha256=sha256_file(selected),
        mapping_statistics=statistics,
        expected_mapping_statistics_sha256=sha256_file(statistics),
        output=output,
        task_translation_cm=5.0,
        task_rotation_deg=5.0,
        dependency_voxel_size=1.0,
    )


def test_completion_oracle_expands_exact_track_subset(tmp_path, monkeypatch):
    monkeypatch.setattr(
        oracle,
        "_producer_identity",
        lambda: {"git_commit": "f" * 40, "worktree_clean": True},
    )
    paths = _write_inputs(tmp_path)
    output = tmp_path / "oracle.pt"
    report = oracle.materialize(_args(paths, output))
    state = torch.load(output, map_location="cpu", weights_only=False)
    assert report["method_role"] == "mapping_only_oracle_not_a_gate_not_deployable"
    assert report["selected_anchor_count"] == 2
    assert report["oracle_anchor_count"] == 3
    assert report["control_task_failure_count"] == 2
    assert report["control_catastrophic_count"] == 1
    assert state["source_dependency_group_ids"].tolist() == [-1, -1, -1]
    expected_dependency = torch.unique(
        torch.floor(state["anchor_xyz"] / 1.0).long(),
        dim=0,
        return_inverse=True,
    )[1]
    assert torch.equal(state["dependency_group_ids"], expected_dependency)
    assert state["track_centric_reconstruction"]["track_anchor_count"] == 3
    provenance = state["provenance"]["rendered_track_completion_oracle"]
    assert provenance["uses_test_queries"] is False
    assert provenance["input_sha256"] == report["input_sha256"]


def test_completion_oracle_rejects_selected_topology_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(
        oracle,
        "_producer_identity",
        lambda: {"git_commit": "f" * 40, "worktree_clean": True},
    )
    paths = _write_inputs(tmp_path)
    selected = torch.load(paths[1], map_location="cpu", weights_only=False)
    selected["anchor_xyz"][1, 0] += 0.25
    torch.save(selected, paths[1])
    args = _args(paths, tmp_path / "oracle.pt")
    with pytest.raises(ValueError, match="topology differs"):
        oracle.materialize(args)


def test_completion_oracle_rejects_test_statistics(tmp_path, monkeypatch):
    monkeypatch.setattr(
        oracle,
        "_producer_identity",
        lambda: {"git_commit": "f" * 40, "worktree_clean": True},
    )
    paths = _write_inputs(tmp_path)
    statistics = torch.load(paths[2], map_location="cpu", weights_only=False)
    statistics["uses_test_queries"] = True
    torch.save(statistics, paths[2])
    args = _args(paths, tmp_path / "oracle.pt")
    with pytest.raises(ValueError, match="mapping-only"):
        oracle.materialize(args)
