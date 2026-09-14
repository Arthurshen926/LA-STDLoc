from pathlib import Path

import pytest
import torch

from map_learning import pipeline


def test_adaptive_distillation_forwards_explicit_frozen_calibration(
    tmp_path: Path, monkeypatch
) -> None:
    calibration = tmp_path / "variant_calibration.json"
    calibration.touch()
    calls = []
    output = tmp_path / "topology"

    def capture(*args):
        calls.append(args)
        produced = output / "compact.pt"
        produced.touch()
        (output / "adaptive_distillation_build.json").write_text(
            '{"map": "' + str(produced) + '"}'
        )

    monkeypatch.setattr(pipeline, "_run", capture)

    pipeline.distill_compact_map(
        canonical_map=tmp_path / "canonical.pt",
        function_graph=tmp_path / "function_graph.pt",
        positive_teacher=tmp_path / "teacher.pt",
        track_payload=tmp_path / "variant_tracks.pt",
        query_cache=tmp_path / "query.pt",
        output=output,
        config="configs/paper_mainline.yaml",
        scene_calibration=calibration,
    )

    command = calls[0]
    index = command.index("--frozen-scene-calibration")
    assert command[index + 1] == calibration.resolve()


def test_adaptive_distillation_default_does_not_freeze_calibration(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []
    output = tmp_path / "topology"

    def capture(*args):
        calls.append(args)
        produced = output / "compact.pt"
        produced.touch()
        (output / "adaptive_distillation_build.json").write_text(
            '{"map": "' + str(produced) + '"}'
        )

    monkeypatch.setattr(pipeline, "_run", capture)

    pipeline.distill_compact_map(
        canonical_map=tmp_path / "canonical.pt",
        function_graph=tmp_path / "function_graph.pt",
        positive_teacher=tmp_path / "teacher.pt",
        track_payload=tmp_path / "tracks.pt",
        query_cache=tmp_path / "query.pt",
        output=output,
        config="configs/paper_mainline.yaml",
    )

    assert "--frozen-scene-calibration" not in calls[0]


def test_frozen_distillation_rejects_stale_resumed_report(tmp_path: Path) -> None:
    calibration = tmp_path / "variant_calibration.json"
    calibration.write_text("{}")
    output = tmp_path / "topology"
    output.mkdir()
    stale_map = output / "old.pt"
    stale_map.touch()
    (output / "adaptive_distillation_build.json").write_text(
        '{"map": "'
        + str(stale_map)
        + '", "calibration_contract": {'
        + '"mode": "derived_from_current_track_payload", '
        + '"input": null, "input_sha256": null}}'
    )
    with pytest.raises(RuntimeError, match="start from an empty output"):
        pipeline.distill_compact_map(
            canonical_map=tmp_path / "canonical.pt",
            function_graph=tmp_path / "function_graph.pt",
            positive_teacher=tmp_path / "teacher.pt",
            track_payload=tmp_path / "variant_tracks.pt",
            query_cache=tmp_path / "query.pt",
            output=output,
            config="configs/paper_mainline.yaml",
            scene_calibration=calibration,
        )


def test_canonical_evidence_build_does_not_apply_compact_path_contract(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "evidence"
    output.mkdir()
    for name in (
        "canonical_map.pt",
        "function_graph_v2.pt",
        "raster_provenance.pt",
        "function_graph.pt",
        "complete_positive_teacher.pt",
        "evidence_contract.json",
    ):
        (output / name).touch()

    monkeypatch.setattr(
        pipeline,
        "_load_or_compute_scene_calibration",
        lambda **_: {"parameters": {}},
    )
    monkeypatch.setattr(
        pipeline, "_assert_adaptive_threshold_contract", lambda **_: None
    )

    def reject_compact_contract(**_):
        raise AssertionError("compact-map contract leaked into canonical build")

    monkeypatch.setattr(
        pipeline, "_assert_compact_evidence_path_contract", reject_compact_contract
    )
    result = pipeline.build_evidence(
        base_state=tmp_path / "base.pt",
        track_payload=tmp_path / "tracks.pt",
        query_cache=tmp_path / "queries.pt",
        prior_ply=tmp_path / "prior.ply",
        gaussian_type="2dgs",
        sh_degree=3,
        visibility_cache="",
        output=output,
    )

    assert result["canonical_map"] == output / "canonical_map.pt"


def test_compact_refresh_rebuilds_one_aligned_evidence_universe(
    tmp_path: Path, monkeypatch
) -> None:
    compact_map = tmp_path / "compact.pt"
    fallback_graph = tmp_path / "canonical_graph.pt"
    query_cache = tmp_path / "queries.pt"
    track_payload = tmp_path / "tracks.pt"
    prior = tmp_path / "prior.ply"
    for path in (compact_map, fallback_graph, query_cache, track_payload, prior):
        path.touch()

    parameters = {
        "positive_radius_px": 2.0,
        "clean_radius_px": 4.0,
        "negative_radius_px": 8.0,
        "ransac_reprojection_px": 12.0,
        "harm_radius_px": 12.0,
        "evidence_depth_abs_tolerance_m": 0.05,
        "metric_steps": 11,
        "task_translation_m": 0.05,
        "task_rotation_deg": 5.0,
    }
    monkeypatch.setattr(
        pipeline,
        "_load_or_compute_scene_calibration",
        lambda **_: {"parameters": parameters},
    )

    shard_calls = []

    def fake_shards(**kwargs):
        shard_calls.append(kwargs)
        kwargs["output"].touch()

    run_calls = []

    def fake_run(module, *arguments):
        run_calls.append((module, arguments))
        output_index = arguments.index("--output") + 1
        Path(arguments[output_index]).touch()

    trained = {}

    def fake_train(**kwargs):
        trained.update(kwargs)

    monkeypatch.setattr(pipeline, "_run_query_shards", fake_shards)
    monkeypatch.setattr(pipeline, "_run", fake_run)
    monkeypatch.setattr(pipeline, "train", fake_train)
    monkeypatch.setattr(
        pipeline, "_assert_adaptive_threshold_contract", lambda **_: None
    )
    compact_contract_calls = []
    monkeypatch.setattr(
        pipeline,
        "_assert_compact_evidence_path_contract",
        lambda **kwargs: compact_contract_calls.append(kwargs),
    )
    monkeypatch.setattr(
        pipeline, "_assert_compact_training_threshold_contract", lambda *_: None
    )

    output = tmp_path / "refresh"
    result = pipeline.train_compact_map(
        compact_map=compact_map,
        function_graph=fallback_graph,
        track_payload=track_payload,
        query_cache=query_cache,
        prior_ply=prior,
        gaussian_type="2dgs",
        sh_degree=3,
        output=output,
        config=Path("configs/paper_mainline.yaml"),
        rebuild_function_graph=True,
        function_graph_shards=3,
        provenance_shards=2,
        observation_shards=4,
    )

    assert [call["module"] for call in shard_calls] == [
        "evidence.function_graph",
        "priors.provenance",
        "map_learning.observations",
    ]
    graph_v2 = output / "compact_function_graph_v2.pt"
    provenance_arguments = shard_calls[1]["arguments"]
    graph_argument = provenance_arguments.index("--function-graph") + 1
    assert Path(provenance_arguments[graph_argument]) == graph_v2
    assert shard_calls[0]["shard_count"] == 3
    assert shard_calls[1]["shard_count"] == 2
    assert shard_calls[2]["shard_count"] == 4

    assert run_calls == [
        (
            "evidence.evidence_graph",
            (
                "--function-graph-v2",
                graph_v2,
                "--raster-provenance",
                output / "raster_provenance.pt",
                "--output",
                output / "compact_function_graph.pt",
            ),
        )
    ]
    assert trained["function_graph_path"] == output / "compact_function_graph.pt"
    assert compact_contract_calls == [
        {
            "compact_map": compact_map,
            "graph_v2": output / "compact_function_graph_v2.pt",
            "provenance": output / "raster_provenance.pt",
            "graph": output / "compact_function_graph.pt",
            "teacher": output / "complete_positive_teacher.pt",
        }
    ]
    assert result["compact_function_graph"] == output / "compact_function_graph.pt"
    assert result["compact_function_graph_v2"] == graph_v2


def test_compact_evidence_path_contract_rejects_stale_provenance(
    tmp_path: Path,
) -> None:
    compact_map = tmp_path / "compact.pt"
    graph_v2 = tmp_path / "compact_function_graph_v2.pt"
    provenance = tmp_path / "raster_provenance.pt"
    graph = tmp_path / "compact_function_graph.pt"
    teacher = tmp_path / "complete_positive_teacher.pt"
    compact_map.touch()
    torch.save({"anchor_map": str(compact_map)}, graph_v2)
    torch.save(
        {
            "anchor_map": str(compact_map),
            "config": {"function_graph": str(graph_v2)},
        },
        provenance,
    )
    torch.save(
        {
            "anchor_map": str(compact_map),
            "raster_provenance": str(provenance),
        },
        graph,
    )
    torch.save(
        {
            "anchor_map": str(compact_map),
            "raster_provenance": str(provenance),
        },
        teacher,
    )
    arguments = {
        "compact_map": compact_map,
        "graph_v2": graph_v2,
        "provenance": provenance,
        "graph": graph,
        "teacher": teacher,
    }
    pipeline._assert_compact_evidence_path_contract(**arguments)

    torch.save(
        {
            "anchor_map": str(compact_map),
            "config": {"function_graph": str(tmp_path / "stale_graph.pt")},
        },
        provenance,
    )
    with pytest.raises(RuntimeError, match="do not share one anchor universe"):
        pipeline._assert_compact_evidence_path_contract(**arguments)


def test_compact_evidence_path_contract_accepts_identical_map_copy(
    tmp_path: Path,
) -> None:
    compact_map = tmp_path / "compact.pt"
    original_map = tmp_path / "original_compact.pt"
    graph_v2 = tmp_path / "compact_function_graph_v2.pt"
    provenance = tmp_path / "raster_provenance.pt"
    graph = tmp_path / "compact_function_graph.pt"
    teacher = tmp_path / "complete_positive_teacher.pt"
    compact_map.write_bytes(b"identical compact map")
    original_map.write_bytes(compact_map.read_bytes())
    torch.save({"anchor_map": str(original_map)}, graph_v2)
    torch.save(
        {
            "anchor_map": str(original_map),
            "config": {"function_graph": str(graph_v2)},
        },
        provenance,
    )
    torch.save(
        {
            "anchor_map": str(original_map),
            "raster_provenance": str(provenance),
        },
        graph,
    )
    torch.save(
        {
            "anchor_map": str(original_map),
            "raster_provenance": str(provenance),
        },
        teacher,
    )
    pipeline._assert_compact_evidence_path_contract(
        compact_map=compact_map,
        graph_v2=graph_v2,
        provenance=provenance,
        graph=graph,
        teacher=teacher,
    )
