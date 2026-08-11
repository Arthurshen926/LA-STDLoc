import json
from pathlib import Path

import pytest
import torch

from scripts.run_sparse_factor_matrix import (
    _assert_nested,
    _evaluate,
    _reuse_pipeline_evaluation,
)


def _state(tracks, bases):
    return {
        "track_centric_reconstruction": {
            "track_indices": torch.as_tensor(tracks),
            "base_canonical_rows": torch.as_tensor(bases),
        }
    }


def test_capacity_factor_requires_nested_track_and_base_sets(tmp_path: Path):
    small, large = tmp_path / "small.pt", tmp_path / "large.pt"
    torch.save(_state([1, 3], [4]), small)
    torch.save(_state([1, 2, 3], [4, 5]), large)
    report = _assert_nested(small, large)
    assert report["track_16k_subset_of_20k"]
    assert report["base_16k_subset_of_20k"]


def test_capacity_factor_rejects_non_nested_maps(tmp_path: Path):
    small, large = tmp_path / "small.pt", tmp_path / "large.pt"
    torch.save(_state([1, 3], [4]), small)
    torch.save(_state([1, 2], [4, 5]), large)
    with pytest.raises(RuntimeError, match="not nested"):
        _assert_nested(small, large)


def test_parallel_seed_evaluation_preserves_sparse_contract(tmp_path: Path):
    output = tmp_path / "evaluations"
    for seed in (2026, 2027, 2028):
        seed_output = output / f"evaluation_seed{seed}"
        seed_output.mkdir(parents=True)
        (seed_output / "summary.json").write_text(
            '{"translation_median_cm": 1.0}\n'
        )
        contract = {"pose_solves": 1}
        if seed != 2026:
            contract.update(
                duplicate_anchor_suppression=False,
                guided_sampling=False,
            )
        (seed_output / "deployment_contract.json").write_text(
            json.dumps(contract) + "\n"
        )

    summaries = _evaluate(
        dataset=tmp_path,
        map_path=tmp_path / "map.pt",
        metric_state=tmp_path / "metric.pt",
        calibration=tmp_path / "calibration.json",
        config=tmp_path / "config.yaml",
        output=output,
        device="cpu",
        seeds=[2026, 2027, 2028],
        workers=3,
    )
    assert list(summaries) == ["2026", "2027", "2028"]


def test_pipeline_seed_evaluation_is_reused_only_when_complete(tmp_path: Path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    for name in ("summary.json", "results.json", "deployment_contract.json"):
        (source / name).write_text(f"{name}\n")
    assert _reuse_pipeline_evaluation(source, destination)
    assert (destination / "summary.json").read_text() == "summary.json\n"

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "summary.json").write_text("{}\n")
    assert not _reuse_pipeline_evaluation(incomplete, tmp_path / "unused")
