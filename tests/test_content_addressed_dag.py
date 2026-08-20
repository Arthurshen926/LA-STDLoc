import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from common.content_addressed_dag import (
    ContentAddressedStore,
    node_spec,
    path_content_record,
    source_identity,
)
from scripts import run_pipeline


def _spec(tmp_path: Path, *, value: int = 1) -> dict:
    source = tmp_path / "producer.py"
    source.write_text("VALUE = 1\n")
    upstream = tmp_path / "input.bin"
    upstream.write_bytes(b"upstream")
    return node_spec(
        node="observation_track_geometry",
        config={"value": value},
        upstream={"input": path_content_record(upstream)},
        producer=source_identity(tmp_path, ("producer.py",)),
    )


def _store(tmp_path: Path, *, node_limit: int = 1024) -> ContentAddressedStore:
    return ContentAddressedStore(
        tmp_path / "cache",
        maximum_node_bytes=node_limit,
        maximum_store_bytes=4096,
    )


def test_publish_and_exact_hit(tmp_path: Path):
    spec = _spec(tmp_path)
    payload = tmp_path / "payload.pt"
    payload.write_bytes(b"payload")
    store = _store(tmp_path)
    published = store.publish(spec, {"track_payload.pt": payload})
    hit = store.load(spec)
    assert hit == published
    assert hit["track_payload.pt"].read_bytes() == b"payload"
    assert spec["key_sha256"] != _spec(tmp_path, value=2)["key_sha256"]


def test_producer_source_change_invalidates_key(tmp_path: Path):
    first = _spec(tmp_path)
    (tmp_path / "producer.py").write_text("VALUE = 2\n")
    upstream = tmp_path / "input.bin"
    second = node_spec(
        node="observation_track_geometry",
        config={"value": 1},
        upstream={"input": path_content_record(upstream)},
        producer=source_identity(tmp_path, ("producer.py",)),
    )
    assert first["key_sha256"] != second["key_sha256"]


def test_sha_tamper_fails_closed(tmp_path: Path):
    spec = _spec(tmp_path)
    payload = tmp_path / "payload.pt"
    payload.write_bytes(b"payload")
    store = _store(tmp_path)
    published = store.publish(spec, {"track_payload.pt": payload})
    published["track_payload.pt"].chmod(0o644)
    published["track_payload.pt"].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA/size"):
        store.load(spec)


def test_manifest_schema_and_extra_file_fail_closed(tmp_path: Path):
    spec = _spec(tmp_path)
    payload = tmp_path / "payload.pt"
    payload.write_bytes(b"payload")
    store = _store(tmp_path)
    store.publish(spec, {"track_payload.pt": payload})
    node = store.node_path(spec)
    (node / "unregistered").write_text("bad")
    with pytest.raises(ValueError, match="unregistered"):
        store.load(spec)
    (node / "unregistered").unlink()
    manifest_path = node / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["version"] = 99
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incompatible"):
        store.load(spec)


def test_node_and_store_capacity_are_bounded(tmp_path: Path):
    spec = _spec(tmp_path)
    payload = tmp_path / "payload.pt"
    payload.write_bytes(b"0123456789")
    store = _store(tmp_path, node_limit=4)
    with pytest.raises(ValueError, match="above limit"):
        store.publish(spec, {"track_payload.pt": payload})


def test_directory_identity_rejects_symlink(tmp_path: Path):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "value").write_text("value")
    assert path_content_record(tree)["file_count"] == 1
    (tree / "link").symlink_to(tree / "value")
    with pytest.raises(ValueError, match="symlink"):
        path_content_record(tree)


def test_pipeline_bootstrap_node_is_reused_across_fresh_roots(
    tmp_path: Path, monkeypatch
):
    spec = _spec(tmp_path)
    builds = []

    def build(**kwargs):
        builds.append(kwargs["output"])
        root = Path(kwargs["output"]) / "bootstrap"
        root.mkdir(parents=True)
        outputs = {}
        for name, filename in run_pipeline._BOOTSTRAP_DAG_FILES.items():
            path = root / filename
            if name == "scene_calibration":
                path.write_text(
                    json.dumps(
                        {
                            "sources": {
                                "query_cache": "old-query",
                                "track_payload": "old-track",
                            }
                        }
                    )
                )
            elif name == "mapping_frontend_contract":
                path.write_text(json.dumps({"query_cache": "old-query"}))
            else:
                path.write_bytes(name.encode())
            outputs[name] = path
        return outputs

    monkeypatch.setattr(run_pipeline, "_bootstrap_dag_spec", lambda *_args, **_kw: spec)
    monkeypatch.setattr(run_pipeline, "build_bootstrap_and_tracks", build)
    common = {
        "artifact_cache": tmp_path / "cache",
        "artifact_cache_max_node_gib": 0.001,
        "artifact_cache_max_total_gib": 0.002,
        "dataset": tmp_path / "dataset",
        "prior": tmp_path / "prior",
        "gaussian_type": "2dgs",
        "sh_degree": 3,
    }
    first_root = tmp_path / "run-one"
    first_root.mkdir()
    first, first_report = run_pipeline._build_or_reuse_bootstrap(
        args=SimpleNamespace(output=first_root, **common),
        config=tmp_path / "config.yaml",
    )
    second_root = tmp_path / "run-two"
    second_root.mkdir()
    second, second_report = run_pipeline._build_or_reuse_bootstrap(
        args=SimpleNamespace(output=second_root, **common),
        config=tmp_path / "config.yaml",
    )
    assert len(builds) == 1
    assert first_report["cache_hit"] is False
    assert second_report["cache_hit"] is True
    assert first["track_payload"] == second["track_payload"]
    calibration = json.loads(second["scene_calibration"].read_text())
    assert calibration["sources"]["query_cache"] == str(second["query_cache"])
    assert calibration["sources"]["track_payload"] == str(second["track_payload"])
