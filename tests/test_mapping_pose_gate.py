import hashlib
import json
from pathlib import Path

import pytest
import torch

from common.hashing import sha256_file
from scripts import compare_mapping_pose_gate as mapping_pose_gate
from scripts.compare_mapping_pose_gate import compare_mapping_pose_gate


SEEDS = (2026, 2027, 2028)
EVALUATION_CODE = {
    "schema": "lafgs_mapping_pose_evaluation_code",
    "version": 1,
    "repository": "/clean/repository",
    "git_commit": "a" * 40,
    "git_worktree_clean": True,
    "entrypoints": {
        "scripts/evaluate_mapping_cache.py": "b" * 64,
        "scripts/compare_mapping_pose_gate.py": "c" * 64,
    },
}


@pytest.fixture(autouse=True)
def _frozen_evaluation_code(monkeypatch):
    monkeypatch.setattr(
        mapping_pose_gate,
        "mapping_pose_evaluation_code_identity",
        lambda **_: EVALUATION_CODE,
    )


def _json_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _arm(tmp_path: Path, name: str, query_cache: Path, *, query_names=None):
    root = tmp_path / name
    root.mkdir()
    paths = {
        role: root / filename
        for role, filename in {
            "map": "map.pt",
            "metric": "metric.pt",
            "teacher": "teacher.pt",
            "query_cache": "query_cache.pt",
            "calibration": "calibration.json",
        }.items()
    }
    paths["query_cache"] = query_cache
    anchor_ids = torch.tensor([11, 17, 23] if name == "baseline" else [101, 103])
    torch.save({"anchor_ids": anchor_ids}, paths["map"])
    torch.save(
        {
            "landmark_indices": anchor_ids.clone(),
            "map_path": str(paths["map"].resolve()),
        },
        paths["metric"],
    )
    names = list(query_names or [f"query-{index:04d}" for index in range(300)])
    torch.save(
        {
            "anchor_count": int(anchor_ids.numel()),
            "query_names": names,
            "records": [
                {"query_index": index, "query_name": query_name}
                for index, query_name in enumerate(names)
            ],
            "anchor_map": str(paths["map"].resolve()),
            "query_cache": str(query_cache.resolve()),
        },
        paths["teacher"],
    )
    paths["calibration"].write_text(
        json.dumps(
            {
                "schema": "lafgs_mapping_only_scene_calibration",
                "version": 2,
                "uses_test_queries": False,
                "sources": {
                    "query_cache": str(query_cache.resolve()),
                    "uses_test_queries": False,
                },
                "statistics": {"mapping_views": 300},
                "parameters": {"metric_steps": 1520},
                "policy": {"calibration_split": "mapping_only"},
            }
        )
    )
    return paths


def _metrics(**updates):
    values = {
        "query_count": 256,
        "raw_gt_precision_percent": 12.0,
        "median_te_cm": 1.0,
        "mean_te_cm": 1.2,
        "p90_te_cm": 2.0,
        "cvar95_te_cm": 3.0,
        "median_ae_deg": 0.50,
        "mean_ae_deg": 0.60,
        "p90_ae_deg": 0.80,
        "p95_ae_deg": 1.00,
        "recall_5cm_5deg_percent": 60.0,
        "catastrophic_100cm_count": 0,
    }
    values.update(updates)
    return values


def _summaries(tmp_path, arm, artifacts, metrics_by_seed=None):
    result = {}
    metrics_by_seed = metrics_by_seed or {}
    teacher = torch.load(artifacts["teacher"], map_location="cpu", weights_only=False)
    query_names = list(teacher["query_names"])
    indices = (
        torch.linspace(0, len(query_names) - 1, steps=256)
        .round()
        .long()
        .unique(sorted=True)
        .tolist()
    )
    selected_names = [query_names[index] for index in indices]
    artifact_records = {
        role: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
        for role, path in artifacts.items()
    }
    for seed in SEEDS:
        path = tmp_path / arm / f"seed{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "lafgs_mapping_cache_evaluation",
                    "version": 2,
                    "uses_test_queries": False,
                    "seed": seed,
                    "evaluation_code": EVALUATION_CODE,
                    "map": str(artifacts["map"].resolve()),
                    "metric_state": str(artifacts["metric"].resolve()),
                    "complete_positive_teacher": str(artifacts["teacher"].resolve()),
                    "query_cache": str(artifacts["query_cache"].resolve()),
                    "scene_calibration": str(artifacts["calibration"].resolve()),
                    "artifacts": artifact_records,
                    "deployment_row_limit": 0,
                    "pose_error_units": {
                        "translation": "cm",
                        "rotation": "deg",
                    },
                    "query_count": 256,
                    "query_selection": "uniform_mapping_gate",
                    "evaluation_protocol": {
                        "split": "mapping_only",
                        "query_selection": "uniform_mapping_gate",
                        "requested_query_count": 256,
                        "evaluated_query_count": 256,
                        "teacher_query_count": len(query_names),
                        "ordered_teacher_query_names_sha256": _json_sha256(query_names),
                        "selected_query_indices": indices,
                        "selected_query_indices_sha256": _json_sha256(indices),
                        "selected_query_names_sha256": _json_sha256(selected_names),
                        "deployment_row_limit": 0,
                    },
                    "summary": metrics_by_seed.get(seed, _metrics()),
                }
            )
        )
        result[seed] = path
    return result


def _inputs(
    tmp_path, *, variant_names=None, baseline_metrics=None, variant_metrics=None
):
    query_cache = tmp_path / "frozen_query_cache.pt"
    query_cache.write_bytes(b"frozen mapping cache")
    baseline = _arm(tmp_path, "baseline", query_cache)
    variant = _arm(tmp_path, "variant", query_cache, query_names=variant_names)
    baseline_summaries = _summaries(tmp_path, "baseline", baseline, baseline_metrics)
    variant_summaries = _summaries(tmp_path, "variant", variant, variant_metrics)
    return baseline, variant, baseline_summaries, variant_summaries


def _compare(inputs, **kwargs):
    baseline, variant, baseline_summaries, variant_summaries = inputs
    return compare_mapping_pose_gate(
        baseline_summaries=baseline_summaries,
        variant_summaries=variant_summaries,
        baseline_artifacts=baseline,
        variant_artifacts=variant,
        **kwargs,
    )


def _rewrite(path: Path, update) -> None:
    payload = json.loads(path.read_text())
    update(payload)
    path.write_text(json.dumps(payload))


def test_mapping_pose_gate_passes_only_safe_substantive_gain_and_reports_hashes(
    tmp_path,
):
    variant_metrics = {seed: _metrics(mean_te_cm=1.16) for seed in SEEDS}
    inputs = _inputs(tmp_path, variant_metrics=variant_metrics)
    expected_map_sha = sha256_file(inputs[0]["map"])
    report = _compare(inputs, expected_sha256={"baseline.map": expected_map_sha})

    assert report["decision"]["verdict"] == "PASS"
    assert report["decision"]["authorizes_next_stage"] is True
    assert report["uses_test_queries"] is False
    assert report["per_seed"]["2026"]["seed_binding"] == (
        "embedded_report_and_explicit_cli"
    )
    assert report["three_seed_mean"]["substantive_improvement_checks"]["mean_te"]
    assert (
        report["lineage"]["inputs"]["baseline.map"]["expected_sha256_matches"] is True
    )
    assert report["lineage"]["checks"]["uniform_q256_query_names_equal"]


def test_mapping_pose_gate_stops_on_one_seed_regression_even_when_mean_improves(
    tmp_path,
):
    variant_metrics = {
        2026: _metrics(median_te_cm=1.03),
        2027: _metrics(median_te_cm=0.90),
        2028: _metrics(median_te_cm=0.90),
    }
    report = _compare(_inputs(tmp_path, variant_metrics=variant_metrics))

    assert report["three_seed_mean"]["has_substantive_improvement"] is True
    assert report["per_seed"]["2026"]["passes_non_regression"] is False
    assert report["decision"]["verdict"] == "STOP"
    assert report["decision"]["reason"] == "PER_SEED_NON_REGRESSION_FAILED"


def test_mapping_pose_gate_stops_without_substantive_gain(tmp_path):
    report = _compare(_inputs(tmp_path))

    assert report["decision"]["verdict"] == "STOP"
    assert report["decision"]["all_per_seed_non_regression_checks_pass"] is True
    assert report["decision"]["reason"] == "NO_SUBSTANTIVE_THREE_SEED_MEAN_GAIN"


def test_mapping_pose_gate_rejects_different_teacher_query_order(tmp_path):
    names = [f"query-{index:04d}" for index in range(300)]
    names[0], names[1] = names[1], names[0]
    with pytest.raises(ValueError, match="paired mapping-pose lineage differs"):
        _compare(_inputs(tmp_path, variant_names=names))


def test_mapping_pose_gate_rejects_map_metric_id_mismatch(tmp_path):
    inputs = _inputs(tmp_path)
    variant = inputs[1]
    metric = torch.load(variant["metric"], map_location="cpu", weights_only=False)
    metric["landmark_indices"] = metric["landmark_indices"].flip(0)
    torch.save(metric, variant["metric"])

    with pytest.raises(ValueError, match="map and metric anchor IDs"):
        _compare(inputs)


def test_mapping_pose_gate_rejects_expected_hash_mismatch(tmp_path):
    with pytest.raises(ValueError, match="SHA-256 mismatch for baseline.map"):
        _compare(_inputs(tmp_path), expected_sha256={"baseline.map": "0" * 64})


def test_mapping_pose_gate_rejects_relabelled_seed(tmp_path):
    inputs = _inputs(tmp_path)
    summary = inputs[3][2026]
    _rewrite(summary, lambda payload: payload.__setitem__("seed", 2027))

    with pytest.raises(ValueError, match="embeds a different seed"):
        _compare(inputs)


def test_mapping_pose_gate_rejects_different_evaluation_code(tmp_path):
    inputs = _inputs(tmp_path)
    summary = inputs[3][2026]

    def replace_commit(payload):
        payload["evaluation_code"]["git_commit"] = "d" * 40

    _rewrite(summary, replace_commit)
    with pytest.raises(ValueError, match="evaluation-code identity differs"):
        _compare(inputs)


def test_mapping_pose_gate_rejects_summary_bound_to_another_teacher(tmp_path):
    inputs = _inputs(tmp_path)
    summary = inputs[3][2026]

    def replace_teacher(payload):
        payload["complete_positive_teacher"] = str(inputs[0]["teacher"].resolve())
        payload["artifacts"]["teacher"] = {
            "path": str(inputs[0]["teacher"].resolve()),
            "sha256": sha256_file(inputs[0]["teacher"]),
        }

    _rewrite(summary, replace_teacher)
    with pytest.raises(ValueError, match="complete_positive_teacher path differs"):
        _compare(inputs)


def test_mapping_pose_gate_rejects_forged_query_indices(tmp_path):
    inputs = _inputs(tmp_path)
    summary = inputs[3][2026]

    def forge_indices(payload):
        indices = payload["evaluation_protocol"]["selected_query_indices"]
        indices[0] = indices[0] + 1
        payload["evaluation_protocol"]["selected_query_indices_sha256"] = _json_sha256(
            indices
        )

    _rewrite(summary, forge_indices)
    with pytest.raises(ValueError, match="selected query indices differ"):
        _compare(inputs)


def test_mapping_pose_gate_stops_on_rotation_regression_despite_translation_gain(
    tmp_path,
):
    variant_metrics = {
        seed: _metrics(mean_te_cm=1.16, p95_ae_deg=1.03) for seed in SEEDS
    }
    report = _compare(_inputs(tmp_path, variant_metrics=variant_metrics))

    assert report["three_seed_mean"]["substantive_improvement_checks"]["mean_te"]
    assert report["per_seed"]["2026"]["checks"]["p95_ae_non_regression"] is False
    assert report["decision"]["verdict"] == "STOP"


def test_mapping_pose_gate_accepts_substantive_rotation_gain(tmp_path):
    variant_metrics = {seed: _metrics(mean_ae_deg=0.57) for seed in SEEDS}
    report = _compare(_inputs(tmp_path, variant_metrics=variant_metrics))

    assert report["three_seed_mean"]["substantive_improvement_checks"]["mean_ae"]
    assert report["decision"]["verdict"] == "PASS"
