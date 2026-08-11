from __future__ import annotations

import json

from map_learning import pipeline


def test_exact_scene_calibration_sidecar_avoids_query_cache_reload(
    tmp_path, monkeypatch
) -> None:
    query_cache = tmp_path / "query_cache.pt"
    track_payload = tmp_path / "tracks.pt"
    query_cache.touch()
    track_payload.touch()
    policy = {"matching_rows_fraction": 0.04735}
    cached = {
        "schema": "lafgs_mapping_only_scene_calibration",
        "version": 2,
        "statistics": {"query_count": 10},
        "parameters": {"metric_steps": 8},
        "policy": policy,
        "sources": {
            "query_cache": str(query_cache.resolve()),
            "track_payload": str(track_payload.resolve()),
            "uses_test_queries": False,
        },
    }
    path = tmp_path / "scene_calibration.json"
    path.write_text(json.dumps(cached))

    def unexpected_calibration(*args, **kwargs):
        raise AssertionError("the exact calibration sidecar must be reused")

    monkeypatch.setattr(pipeline, "calibrate_scene", unexpected_calibration)
    assert pipeline._load_or_compute_scene_calibration(
        query_cache=query_cache,
        track_payload=track_payload,
        policy=policy,
        cached_path=path,
    ) == cached


def test_stale_scene_calibration_sidecar_is_recomputed(tmp_path, monkeypatch) -> None:
    query_cache = tmp_path / "query_cache.pt"
    track_payload = tmp_path / "tracks.pt"
    query_cache.touch()
    track_payload.touch()
    path = tmp_path / "scene_calibration.json"
    path.write_text(
        json.dumps(
            {
                "schema": "lafgs_mapping_only_scene_calibration",
                "version": 2,
                "policy": {"value": "stale"},
                "sources": {
                    "query_cache": str(query_cache.resolve()),
                    "track_payload": str(track_payload.resolve()),
                    "uses_test_queries": False,
                },
            }
        )
    )
    expected = {"parameters": {"metric_steps": 12}}
    monkeypatch.setattr(pipeline, "calibrate_scene", lambda *args, **kwargs: expected)
    assert pipeline._load_or_compute_scene_calibration(
        query_cache=query_cache,
        track_payload=track_payload,
        policy={"value": "current"},
        cached_path=path,
    ) == expected
