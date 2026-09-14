from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch

from common.hashing import sha256_file
from evaluation.mapping_shards import (
    atomic_torch_save,
    json_sha256,
    merge_reports,
    resolve_query_range,
)
from scripts import evaluate_mapping_cache
from topology.deployment_revision import _summary


CODE_IDENTITY = {
    "schema": "lafgs_mapping_pose_evaluation_code",
    "version": 1,
    "repository": "/fixed/repository",
    "git_commit": "a" * 40,
    "git_worktree_clean": True,
    "entrypoints": {"scripts/evaluate_mapping_cache.py": "b" * 64},
}


def test_explicit_query_ranges_are_fixed_registry_positions():
    assert resolve_query_range(
        12,
        query_start=3,
        query_stop=8,
        shard_index=None,
        shard_count=None,
    ) == (3, 8, "range_shard")
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_query_range(
            12,
            query_start=0,
            query_stop=6,
            shard_index=0,
            shard_count=2,
        )


def _inputs(root: Path, query_count: int = 12) -> tuple[dict[str, Path], list[str]]:
    root.mkdir()
    paths = {
        "map": root / "map.pt",
        "metric": root / "metric.pt",
        "teacher": root / "teacher.pt",
        "cache": root / "cache.pt",
        "calibration": root / "calibration.json",
    }
    names = [f"q{index:03d}" for index in range(query_count)]
    torch.save({"anchor_ids": torch.tensor([0])}, paths["map"])
    torch.save({"metric": "fixed"}, paths["metric"])
    torch.save({"queries": {}}, paths["cache"])
    torch.save(
        {
            "anchor_count": 1,
            "query_names": names,
            "records": [{} for _ in names],
            "query_cache": str(paths["cache"].resolve()),
        },
        paths["teacher"],
    )
    paths["calibration"].write_text(
        json.dumps(
            {
                "schema": "lafgs_mapping_only_scene_calibration",
                "version": 2,
                "sources": {
                    "query_cache": str(paths["cache"].resolve()),
                    "uses_test_queries": False,
                },
                "parameters": {
                    "ransac_reprojection_px": 12.0,
                    "clean_radius_px": 1.0,
                    "task_translation_m": 0.05,
                    "task_rotation_deg": 5.0,
                },
            }
        )
    )
    return paths, names


def _statistics(indices: list[int], names: list[str]) -> dict:
    rows = [
        {
            "query_index": index,
            "image_name": names[index],
            "te_cm": float(index + 1),
            "ae_deg": float(index) / 10.0,
            "inliers": index + 3,
            "clean_inliers": index + 2,
            "hypotheses": 1000 + index,
            "group_diverse_selected": False,
            "correspondences": 20 + index,
            "assignment_topk": 0,
            "assignment_unmatched_queries": 0,
            "assignment_reassigned_queries": index % 2,
            "assignment_top1_collisions": index % 3,
        }
        for index in indices
    ]
    count = float(len(indices))
    counters = {
        "winner_count": torch.tensor([count * 20], dtype=torch.float64),
        "correct_winner_count": torch.tensor([count * 15], dtype=torch.float64),
        "clean_inlier_count": torch.tensor([count * 10], dtype=torch.float64),
        "harmful_inlier_count": torch.tensor([count * 2], dtype=torch.float64),
    }
    return {"queries": rows, "counters": counters, "summary": _summary(rows, counters)}


def _run(
    monkeypatch,
    paths: dict[str, Path],
    names: list[str],
    output: Path,
    *,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> Path:
    def collect_stub(**kwargs):
        indices = (
            list(range(len(names)))
            if kwargs["query_indices"] is None
            else [int(value) for value in kwargs["query_indices"].tolist()]
        )
        return _statistics(indices, names)

    monkeypatch.setattr(
        evaluate_mapping_cache, "collect_deployment_statistics", collect_stub
    )
    monkeypatch.setattr(
        evaluate_mapping_cache,
        "mapping_pose_evaluation_code_identity",
        lambda **_: CODE_IDENTITY,
    )
    arguments = [
        "scripts.evaluate_mapping_cache",
        "--map",
        str(paths["map"]),
        "--metric-state",
        str(paths["metric"]),
        "--complete-positive-teacher",
        str(paths["teacher"]),
        "--query-cache",
        str(paths["cache"]),
        "--scene-calibration",
        str(paths["calibration"]),
        "--device",
        "cpu",
        "--output",
        str(output),
    ]
    if shard_index is not None:
        arguments.extend(
            ["--shard-index", str(shard_index), "--shard-count", str(shard_count)]
        )
    monkeypatch.setattr(sys, "argv", arguments)
    evaluate_mapping_cache.main()
    assert not list(output.glob("*.tmp"))
    return output / "mapping_cache_summary.json"


@pytest.mark.parametrize("shard_count", [2, 4])
def test_two_and_four_query_shards_merge_exactly_like_unsharded(
    tmp_path, monkeypatch, shard_count
):
    paths, names = _inputs(tmp_path / "inputs")
    unsharded_path = _run(monkeypatch, paths, names, tmp_path / "unsharded")
    shard_paths = [
        _run(
            monkeypatch,
            paths,
            names,
            tmp_path / f"shard-{index}",
            shard_index=index,
            shard_count=shard_count,
        )
        for index in range(shard_count)
    ]
    merged = merge_reports(shard_paths, tmp_path / f"merged-{shard_count}")
    unsharded = json.loads(unsharded_path.read_text())
    assert merged["summary"] == unsharded["summary"]
    merged_statistics = torch.load(
        merged["statistics"]["path"], map_location="cpu", weights_only=False
    )
    unsharded_statistics = torch.load(
        unsharded["statistics"]["path"], map_location="cpu", weights_only=False
    )
    assert merged_statistics["queries"] == unsharded_statistics["queries"]
    for name in unsharded_statistics["counters"]:
        assert torch.equal(
            merged_statistics["counters"][name],
            unsharded_statistics["counters"][name],
        )


def test_query_shard_merge_rejects_partial_tampered_and_missing_statistics(
    tmp_path, monkeypatch
):
    paths, names = _inputs(tmp_path / "inputs")
    shard_paths = [
        _run(
            monkeypatch,
            paths,
            names,
            tmp_path / f"shard-{index}",
            shard_index=index,
            shard_count=2,
        )
        for index in range(2)
    ]
    with pytest.raises(ValueError, match="incomplete"):
        merge_reports(shard_paths[:1], tmp_path / "partial")

    second_report = json.loads(shard_paths[1].read_text())
    statistics_path = Path(second_report["statistics"]["path"])
    original = statistics_path.read_bytes()
    statistics_path.write_bytes(original + b"tamper")
    with pytest.raises(ValueError, match="size changed"):
        merge_reports(shard_paths, tmp_path / "tampered")

    statistics_path.write_bytes(original)
    statistics_path.unlink()
    with pytest.raises(ValueError, match="missing"):
        merge_reports(shard_paths, tmp_path / "missing")


def _write_fullmap_shard(
    root: Path,
    names: list[str],
    *,
    begin: int,
    end: int,
) -> Path:
    indices = list(range(begin, end))
    core = _statistics(indices, names)
    affected = torch.tensor([index % 5 + 1 for index in indices])
    maximum_eligible = max(index % 3 + 2 for index in indices)
    contract = {
        "schema": "lafgs_rendered_track_full_mapping_loo_evaluation_contract",
        "version": 1,
        "producer_identity": CODE_IDENTITY,
        "inputs": {"map": "/fixed/map.pt"},
        "input_sha256": {"map": "c" * 64},
        "seed": 2026,
        "device": "cuda",
        "anchor_count": 10,
        "configuration": {
            "cpu_threads": 1,
            "descriptor_trim_fraction": 0.2,
            "deployment_row_limit": 0,
            "view_mixture": True,
            "group_aware_pose": False,
            "group_field": None,
            "group_hypothesis_samples": 0,
            "assignment_topk": 0,
            "assignment_dustbin_score": -1.0,
        },
        "calibration_parameters": {"ransac_reprojection_px": 12.0},
        "selected_query_indices": list(range(len(names))),
        "selected_query_indices_sha256": json_sha256(list(range(len(names)))),
        "selected_query_names_sha256": json_sha256(names),
    }
    loo = {
        "construction_uses_all_mapping_observations": True,
        "track_identity_and_geometry_remain_full_mapping": True,
        "query_descriptor_excluded_from_affected_anchor_fusion": True,
        "affected_anchor_updates": int(affected.sum()),
        "query_local_view_mixture_eligibility_recomputed": True,
        "maximum_query_local_eligible_k2_count": maximum_eligible,
        "minimum_affected_anchors_per_query": int(affected.min()),
        "maximum_affected_anchors_per_query": int(affected.max()),
        "mean_affected_anchors_per_query": float(affected.float().mean()),
    }
    statistics = {
        "schema": "lafgs_rendered_track_full_mapping_loo_statistics",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        **core,
        "loo": loo,
        "evaluation_contract": contract,
        "evaluation_contract_sha256": json_sha256(contract),
        "query_range": {"start": begin, "stop": end},
        "selected_query_indices": indices,
        "affected_anchor_count_by_query": affected,
    }
    root.mkdir(parents=True)
    statistics_path = root / "full_mapping_loo_statistics.pt"
    atomic_torch_save(statistics, statistics_path)
    report = {
        "schema": "lafgs_rendered_track_full_mapping_loo_report",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "producer_identity": CODE_IDENTITY,
        "seed": 2026,
        "configuration": {
            **contract["configuration"],
            "view_mixture_contract": {
                "maximum_query_local_eligible_k2_count": maximum_eligible,
                "maximum_query_local_prototype_ratio": (
                    10 + maximum_eligible
                ) / 10,
            },
        },
        "inputs": contract["inputs"],
        "input_sha256": contract["input_sha256"],
        "statistics": str(statistics_path.resolve()),
        "statistics_sha256": sha256_file(statistics_path),
        "statistics_size_bytes": statistics_path.stat().st_size,
        "evaluation_contract": contract,
        "evaluation_contract_sha256": json_sha256(contract),
        "query_shard": {
            "kind": "unsharded" if (begin, end) == (0, len(names)) else "indexed_shard",
            "start": begin,
            "stop": end,
            "registry_count": len(names),
        },
        "loo": loo,
        "summary": core["summary"],
    }
    report_path = root / "full_mapping_loo_report.json"
    report_path.write_text(json.dumps(report))
    return report_path


@pytest.mark.parametrize("shard_count", [2, 4])
def test_view_mixture_fullmap_two_and_four_shards_are_exact(
    tmp_path, shard_count
):
    names = [f"q{index:03d}" for index in range(12)]
    unsharded_path = _write_fullmap_shard(
        tmp_path / "unsharded", names, begin=0, end=len(names)
    )
    shards = []
    for index in range(shard_count):
        begin = len(names) * index // shard_count
        end = len(names) * (index + 1) // shard_count
        shards.append(
            _write_fullmap_shard(
                tmp_path / f"shard-{index}", names, begin=begin, end=end
            )
        )
    merged = merge_reports(shards, tmp_path / "merged")
    unsharded = json.loads(unsharded_path.read_text())
    assert merged["summary"] == unsharded["summary"]
    assert merged["loo"] == unsharded["loo"]
    assert merged["configuration"] == unsharded["configuration"]
    merged_statistics = torch.load(
        merged["statistics"], map_location="cpu", weights_only=False
    )
    unsharded_statistics = torch.load(
        unsharded["statistics"], map_location="cpu", weights_only=False
    )
    assert merged_statistics["queries"] == unsharded_statistics["queries"]
    for name in unsharded_statistics["counters"]:
        assert torch.equal(
            merged_statistics["counters"][name],
            unsharded_statistics["counters"][name],
        )


def test_view_mixture_fullmap_merge_fails_closed_on_partial_and_tamper(
    tmp_path,
):
    names = [f"q{index:03d}" for index in range(8)]
    shards = [
        _write_fullmap_shard(
            tmp_path / f"shard-{index}", names, begin=index * 4, end=(index + 1) * 4
        )
        for index in range(2)
    ]
    with pytest.raises(ValueError, match="incomplete"):
        merge_reports(shards[:1], tmp_path / "partial")
    report = json.loads(shards[1].read_text())
    statistics = Path(report["statistics"])
    statistics.write_bytes(statistics.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="size changed"):
        merge_reports(shards, tmp_path / "tamper")
