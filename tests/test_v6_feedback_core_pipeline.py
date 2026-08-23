from __future__ import annotations

import json
from pathlib import Path

import torch

from common.hashing import sha256_file
from common.v6_contracts import DESCRIPTOR_SPLIT_SCHEMA, FEEDBACK_VERSION
from scripts import run_v6_feedback_core_pipeline as runner


_FEEDBACK_BYTES = b"mock-v6-feedback-v4\n"


def _write_torch(path: Path, value: dict) -> str:
    torch.save(value, path)
    return sha256_file(path)


def _write_json(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return sha256_file(path)


def _flag(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _write_feedback_summary(
    directory: Path,
    *,
    map_sha256: str,
    metric_sha256: str,
    cache_sha256: str,
) -> tuple[Path, str]:
    directory.mkdir(parents=True, exist_ok=True)
    feedback_path = directory / "feedback.pt"
    feedback_path.write_bytes(_FEEDBACK_BYTES)
    feedback_sha256 = sha256_file(feedback_path)
    summary_path = directory / "summary.json"
    _write_json(
        summary_path,
        {
            "schema": "lafgs_v6_query_local_feedback_summary",
            "version": FEEDBACK_VERSION,
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
            "summary": {"anchor_count": 1},
            "failure_layer_counts": {"L1": 0, "L2": 0, "L3": 0, "L4": 0},
            "contract": {
                "affected_anchor_policy": "rebuild",
                "affected_anchor_holdout_is_exact_rebuild": True,
                "descriptor_identity_supervision_available": True,
            },
            "feedback_path": str(feedback_path.resolve()),
            "feedback_sha256": feedback_sha256,
            "input_sha256": {
                "map": map_sha256,
                "metric": metric_sha256,
                "observation_cache": cache_sha256,
            },
        },
    )
    return summary_path, sha256_file(summary_path)


def _arguments(tmp_path: Path) -> tuple[object, dict[str, str]]:
    map_path = tmp_path / "baseline_map.pt"
    map_sha256 = _write_torch(
        map_path,
        {
            "anchor_ids": torch.tensor([0]),
            "anchor_features": torch.ones((1, 2)),
        },
    )
    metric_path = tmp_path / "baseline_metric.pt"
    metric_sha256 = _write_torch(metric_path, {})
    cache_path = tmp_path / "render_cache.pt"
    cache_sha256 = _write_torch(cache_path, {})
    association_path = tmp_path / "association.pt"
    association_sha256 = _write_torch(association_path, {})
    materialization_path = tmp_path / "materialization.json"
    materialization_sha256 = _write_json(materialization_path, {})
    template = tmp_path / "feedback-template.pt"
    template.write_bytes(_FEEDBACK_BYTES)
    feedback_sha256 = sha256_file(template)
    split_path = tmp_path / "mapping_split.json"
    split_sha256 = _write_json(
        split_path,
        {
            "schema": DESCRIPTOR_SPLIT_SCHEMA,
            "version": 1,
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
            "source_feedback_sha256": feedback_sha256,
            "training_query_indices": [0],
            "validation_query_indices": [1],
        },
    )
    args = runner.build_parser().parse_args(
        [
            "--map",
            str(map_path),
            "--expected-map-sha256",
            map_sha256,
            "--metric",
            str(metric_path),
            "--expected-metric-sha256",
            metric_sha256,
            "--observation-cache",
            str(cache_path),
            "--expected-observation-cache-sha256",
            cache_sha256,
            "--association-graph",
            str(association_path),
            "--expected-association-graph-sha256",
            association_sha256,
            "--materialization-report",
            str(materialization_path),
            "--expected-materialization-report-sha256",
            materialization_sha256,
            "--mapping-training-query-indices",
            str(split_path),
            "--expected-mapping-training-query-indices-sha256",
            split_sha256,
            "--output-dir",
            str(tmp_path / "run"),
            "--device",
            "cpu",
            "--descriptor-pose-critical-weight",
            "2.5",
            "--descriptor-tail-query-weight",
            "1.75",
        ]
    )
    return args, {
        "map": map_sha256,
        "metric": metric_sha256,
        "cache": cache_sha256,
        "association": association_sha256,
        "materialization": materialization_sha256,
        "split": split_sha256,
        "feedback": feedback_sha256,
    }


def _install_mocks(monkeypatch, calls: list[list[str]], contract_calls: list[dict]):
    def fake_pipeline_contract(**kwargs):
        contract_calls.append(kwargs)
        return {
            "schema": "lafgs_v6_full_pipeline_input_contract",
            "version": 1,
            "validated": True,
        }

    def fake_command(command: list[str], *, root: Path) -> None:
        del root
        calls.append(command)
        script = Path(command[1]).name
        if script == "evaluate_v6_self_localization.py":
            _write_feedback_summary(
                Path(_flag(command, "--output-dir")),
                map_sha256=_flag(command, "--expected-map-sha256"),
                metric_sha256=_flag(command, "--expected-metric-sha256"),
                cache_sha256=_flag(command, "--expected-observation-cache-sha256"),
            )
            return
        if script == "propose_v6_round.py":
            output_dir = Path(_flag(command, "--output-dir"))
            output_dir.mkdir(parents=True)
            arm = _flag(command, "--arm")
            map_path = output_dir / "proposal_map.pt"
            deployed_fields = {
                "anchor_ids": torch.tensor([0]),
                "anchor_xyz": torch.zeros((1, 3)),
                "anchor_features": torch.ones((1, 2)),
            }
            map_sha = _write_torch(map_path, {"arm": arm, **deployed_fields})
            metric_path = output_dir / "identity_metric.pt"
            metric_sha = _write_torch(metric_path, {})
            deployment_path = output_dir / "deployment_map.pt"
            deployment_sha = _write_torch(
                deployment_path,
                {
                    "arm": arm,
                    **deployed_fields,
                    "provenance": {"v6_compact_deployment_export": True},
                },
            )
            deployment_metric_path = output_dir / "deployment_metric.pt"
            deployment_metric_sha = _write_torch(deployment_metric_path, {})
            input_sha = {
                "map": _flag(command, "--expected-map-sha256"),
                "observation_cache": _flag(
                    command, "--expected-observation-cache-sha256"
                ),
                "feedback": _flag(command, "--expected-feedback-sha256"),
            }
            if "--mapping-training-query-indices" in command:
                split_sha = _flag(
                    command,
                    "--expected-mapping-training-query-indices-sha256",
                )
                input_sha.update(
                    {
                        "mapping_training_query_indices": split_sha,
                        "descriptor_training_query_indices": split_sha,
                    }
                )
            if "--association-graph" in command:
                input_sha["association_graph"] = _flag(
                    command, "--expected-association-graph-sha256"
                )
            _write_json(
                output_dir / "proposal.json",
                {
                    "schema": "lafgs_v6_round_proposal",
                    "version": 2,
                    "uses_source_mapping_rgb": False,
                    "uses_test_queries": False,
                    "arm": arm,
                    "proposal_available": True,
                    "configuration": {"arm": arm},
                    "input_sha256": input_sha,
                    "output": {
                        "map": str(map_path.resolve()),
                        "map_sha256": map_sha,
                        "metric": str(metric_path.resolve()),
                        "metric_sha256": metric_sha,
                        "deployment_map": str(deployment_path.resolve()),
                        "deployment_map_sha256": deployment_sha,
                        "deployment_metric": str(deployment_metric_path.resolve()),
                        "deployment_metric_sha256": deployment_metric_sha,
                    },
                },
            )
            return
        if script == "compare_v6_feedback.py":
            output = Path(_flag(command, "--output"))
            _write_json(
                output,
                {
                    "schema": "lafgs_v6_paired_feedback_diagnostics",
                    "version": 1,
                    "uses_source_mapping_rgb": False,
                    "uses_test_queries": False,
                    "valid": True,
                    "scopes": {"all_mapping_queries": {}},
                    "comparison_contract": {"paired": True},
                },
            )
            return
        raise AssertionError(f"unexpected subprocess: {command}")

    monkeypatch.setattr(runner, "validate_v6_pipeline_inputs", fake_pipeline_contract)
    monkeypatch.setattr(runner, "validate_v6_identity_metric", lambda *a, **k: None)
    monkeypatch.setattr(
        runner,
        "_producer",
        lambda root: {
            "git_commit": "a" * 40,
            "worktree_clean": True,
            "source_sha256": {"runner": "b" * 64},
        },
    )
    monkeypatch.setattr(runner, "_run_command", fake_command)


def test_runner_uses_one_fresh_baseline_and_independent_compact_arms(
    tmp_path: Path, monkeypatch
):
    args, hashes = _arguments(tmp_path)
    args.run_reconstruction = True
    calls: list[list[str]] = []
    contract_calls: list[dict] = []
    _install_mocks(monkeypatch, calls, contract_calls)

    result = runner.run(args, invocation_argv=["formal-v6-runner", "--mock"])

    assert len(contract_calls) == 1
    assert contract_calls[0]["map_sha256"] == hashes["map"]
    assert contract_calls[0]["observation_cache_sha256"] == hashes["cache"]
    assert contract_calls[0]["association_graph_sha256"] == hashes["association"]
    assert result["automatic_hard_gate_acceptance"] is False
    assert result["candidate_acceptance_status"] == "external_review_required"
    assert result["winner_selection_performed"] is False
    assert result["selected_winner"] is None
    assert result["baseline"]["reused_precomputed"] is False
    assert result["inputs"]["mapping_training_split"]["sha256"] == hashes["split"]
    assert result["independent_arm_contract"] == {
        "all_arms_parent_map_sha256": hashes["map"],
        "all_arms_feedback_sha256": hashes["feedback"],
        "candidate_chaining": False,
        "arms": ["descriptor_loss", "selection", "reconstruction"],
    }

    proposal_calls = [
        command for command in calls if Path(command[1]).name == "propose_v6_round.py"
    ]
    evaluation_calls = [
        command
        for command in calls
        if Path(command[1]).name == "evaluate_v6_self_localization.py"
    ]
    paired_calls = [
        command
        for command in calls
        if Path(command[1]).name == "compare_v6_feedback.py"
    ]
    assert len(proposal_calls) == 3
    assert len(evaluation_calls) == 4
    assert len(paired_calls) == 3
    for command in proposal_calls:
        assert _flag(command, "--expected-map-sha256") == hashes["map"]
        assert _flag(command, "--expected-feedback-sha256") == hashes["feedback"]
        assert _flag(command, "--descriptor-pose-critical-weight") == "2.5"
        assert _flag(command, "--descriptor-tail-query-weight") == "1.75"
        if _flag(command, "--arm") == "reconstruction":
            assert "--mapping-training-query-indices" not in command
            assert (
                _flag(command, "--expected-association-graph-sha256")
                == hashes["association"]
            )
        else:
            assert (
                command.count("--expected-mapping-training-query-indices-sha256")
                == 1
            )
            assert (
                _flag(
                    command,
                    "--expected-mapping-training-query-indices-sha256",
                )
                == hashes["split"]
            )
    for command in evaluation_calls:
        assert _flag(command, "--loo-affected-anchor-policy") == "rebuild"
    for command in evaluation_calls[1:]:
        assert Path(_flag(command, "--map")).name == "proposal_map.pt"
        assert Path(_flag(command, "--metric")).name == "identity_metric.pt"
    for command in paired_calls:
        assert _flag(command, "--expected-baseline-map-sha256") == hashes["map"]
        assert Path(_flag(command, "--candidate-map")).name == "proposal_map.pt"
    for arm in result["arms"]:
        assert arm["parent"] == {
            "map_sha256": hashes["map"],
            "feedback_sha256": hashes["feedback"],
        }
        assert arm["proposal"]["available"] is True
        assert arm["proposal"]["deployment_equivalence"] == {
            "anchor_ids_equal": True,
            "anchor_xyz_equal": True,
            "deployed_anchor_features_equal": True,
            "compact_map_exact_loo_rebuild_capable": False,
        }
        assert (
            arm["evaluation_target_contract"]["fresh_mapping_feedback_target"]
            == "full_proposal_training_checkpoint"
        )
        assert (
            arm["evaluation_target_contract"]["compact_deployment_target"]
            == "final_online_localization_and_size_loading"
        )
        assert arm["paired_diagnostics"]["artifact"]["sha256"] == sha256_file(
            arm["paired_diagnostics"]["artifact"]["path"]
        )
    assert json.loads((args.output_dir / "run.json").read_text()) == result


def test_runner_reuses_only_sha_bound_rebuild_baseline(tmp_path: Path, monkeypatch):
    args, hashes = _arguments(tmp_path)
    summary_path, summary_sha = _write_feedback_summary(
        tmp_path / "precomputed",
        map_sha256=hashes["map"],
        metric_sha256=hashes["metric"],
        cache_sha256=hashes["cache"],
    )
    args.baseline_feedback_summary = summary_path
    args.expected_baseline_feedback_summary_sha256 = summary_sha
    calls: list[list[str]] = []
    contract_calls: list[dict] = []
    _install_mocks(monkeypatch, calls, contract_calls)

    result = runner.run(args, invocation_argv=["formal-v6-runner", "--reuse"])

    assert result["baseline"]["reused_precomputed"] is True
    assert result["baseline"]["command"] is None
    evaluation_calls = [
        command
        for command in calls
        if Path(command[1]).name == "evaluate_v6_self_localization.py"
    ]
    assert len(evaluation_calls) == 2
    assert all(
        Path(_flag(command, "--map")).name == "proposal_map.pt"
        for command in evaluation_calls
    )
