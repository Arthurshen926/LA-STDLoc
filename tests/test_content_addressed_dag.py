import ast
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
import shutil
from types import SimpleNamespace

import pytest
import torch

from common import content_addressed_dag
from common.content_addressed_dag import (
    ContentAddressedStore,
    node_spec,
    path_content_record,
    runtime_identity,
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


def _store(tmp_path: Path, *, node_limit: int = 8192) -> ContentAddressedStore:
    return ContentAddressedStore(
        tmp_path / "cache",
        maximum_node_bytes=node_limit,
        maximum_store_bytes=65536,
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


def test_runtime_and_rasterizer_abi_are_keyed(tmp_path: Path):
    runtime = runtime_identity()
    assert {"torch", "torch_cuda", "cudnn", "gsplat", "gsplat_binary_sha256"} <= set(runtime)
    assert set(runtime["numerical_dependencies"]) == {
        "torchvision",
        "numpy",
        "scipy",
        "opencv",
        "pillow",
        "plyfile",
    }
    for name, identity in runtime["numerical_dependencies"].items():
        assert identity["module"]
        assert {"distribution", "version"} <= set(identity), name
    first = _spec(tmp_path)
    changed_producer = dict(first["producer"])
    changed_producer["runtime"] = runtime
    second = node_spec(
        node=first["node"],
        config=first["config"],
        upstream=first["upstream"],
        producer=changed_producer,
    )
    assert first["key_sha256"] != second["key_sha256"]


def test_each_runtime_dependency_version_field_invalidates_key(tmp_path: Path):
    runtime = runtime_identity()
    baseline = _spec(tmp_path)
    producer = dict(baseline["producer"])
    producer["runtime"] = runtime
    keyed = node_spec(
        node=baseline["node"],
        config=baseline["config"],
        upstream=baseline["upstream"],
        producer=producer,
    )
    for name in runtime["numerical_dependencies"]:
        changed_runtime = deepcopy(runtime)
        changed_runtime["numerical_dependencies"][name]["version"] = "EVIL"
        changed_producer = dict(baseline["producer"])
        changed_producer["runtime"] = changed_runtime
        changed = node_spec(
            node=baseline["node"],
            config=baseline["config"],
            upstream=baseline["upstream"],
            producer=changed_producer,
        )
        assert changed["key_sha256"] != keyed["key_sha256"], name


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


def test_manifest_rejects_artifact_name_file_mismatch(tmp_path: Path):
    spec = _spec(tmp_path)
    payload = tmp_path / "payload.pt"
    payload.write_bytes(b"payload")
    store = _store(tmp_path)
    store.publish(spec, {"track_payload.pt": payload})
    manifest_path = store.node_path(spec) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["track_payload.pt"]["file"] = "other.pt"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="name/file mismatch"):
        store.load(spec)


def test_node_and_store_capacity_are_bounded(tmp_path: Path):
    spec = _spec(tmp_path)
    payload = tmp_path / "payload.pt"
    payload.write_bytes(b"0123456789")
    store = _store(tmp_path, node_limit=4)
    with pytest.raises(ValueError, match="above limit"):
        store.publish(spec, {"track_payload.pt": payload})


def test_capacity_includes_manifest_overhead(tmp_path: Path):
    spec = _spec(tmp_path)
    payload = tmp_path / "payload.pt"
    payload.write_bytes(b"x" * 32)
    store = _store(tmp_path, node_limit=payload.stat().st_size + 16)
    with pytest.raises(ValueError, match="including manifest"):
        store.publish(spec, {"track_payload.pt": payload})


def test_cache_hit_obeys_current_node_limit(tmp_path: Path):
    spec = _spec(tmp_path)
    payload = tmp_path / "payload.pt"
    payload.write_bytes(b"payload")
    store = _store(tmp_path)
    store.publish(spec, {"track_payload.pt": payload})
    bounded = ContentAddressedStore(
        store.root, maximum_node_bytes=1, maximum_store_bytes=65536
    )
    with pytest.raises(ValueError, match="current node limit"):
        bounded.load(spec)


def test_cache_hit_obeys_current_store_limit(tmp_path: Path):
    spec = _spec(tmp_path)
    payload = tmp_path / "payload.pt"
    payload.write_bytes(b"payload")
    store = _store(tmp_path)
    store.publish(spec, {"track_payload.pt": payload})
    (store.root / "capacity-filler").write_bytes(b"x" * 9000)
    bounded = ContentAddressedStore(
        store.root, maximum_node_bytes=8192, maximum_store_bytes=8192
    )
    with pytest.raises(ValueError, match="current store limit"):
        bounded.load(spec)


def test_directory_identity_rejects_symlink(tmp_path: Path):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "value").write_text("value")
    assert path_content_record(tree)["file_count"] == 1
    (tree / "link").symlink_to(tree / "value")
    with pytest.raises(ValueError, match="symlink"):
        path_content_record(tree)
    with pytest.raises(ValueError, match="symlink"):
        path_content_record(tree / "link")


def test_publish_rejects_symlink_source(tmp_path: Path):
    spec = _spec(tmp_path)
    payload = tmp_path / "payload.pt"
    payload.write_bytes(b"payload")
    alias = tmp_path / "alias.pt"
    alias.symlink_to(payload)
    with pytest.raises(ValueError, match="symlink"):
        _store(tmp_path).publish(spec, {"track_payload.pt": alias})


def test_symlinked_parent_boundaries_fail_closed(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    (real / "payload.pt").write_bytes(b"payload")
    with pytest.raises(ValueError, match="symbolic-link boundary"):
        path_content_record(linked / "payload.pt")
    with pytest.raises(ValueError, match="symbolic-link boundary"):
        ContentAddressedStore(
            linked / "cache", maximum_node_bytes=8192, maximum_store_bytes=65536
        )


def test_partial_final_node_is_recovered_under_publish_lock(tmp_path: Path):
    spec = _spec(tmp_path)
    payload = tmp_path / "payload.pt"
    payload.write_bytes(b"payload")
    store = _store(tmp_path)
    partial = store.node_path(spec)
    (partial / "artifacts").mkdir(parents=True)
    (partial / "artifacts/junk").write_text("partial")
    result = store.publish(spec, {"track_payload.pt": payload})
    assert result["track_payload.pt"].read_bytes() == b"payload"
    assert not (partial / "artifacts/junk").exists()


def test_artifact_aliases_are_rejected(tmp_path: Path):
    spec = _spec(tmp_path)
    payload = tmp_path / "payload.pt"
    payload.write_bytes(b"payload")
    alias = tmp_path / "alias.pt"
    os.link(payload, alias)
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="alias"):
        store.publish(spec, {"left.pt": payload, "right.pt": alias})


def test_concurrent_same_key_publish_has_one_complete_node(tmp_path: Path):
    spec = _spec(tmp_path)
    payload = tmp_path / "payload.pt"
    payload.write_bytes(b"payload")
    store = _store(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda _: store.publish(spec, {"track_payload.pt": payload}),
                range(2),
            )
        )
    assert results[0] == results[1] == store.load(spec)


def test_run_local_materialization_survives_cache_prune(tmp_path: Path):
    spec = _spec(tmp_path)
    payload = tmp_path / "payload.pt"
    payload.write_bytes(b"payload")
    store = _store(tmp_path)
    store.publish(spec, {"track_payload.pt": payload})
    local, _ = store.materialize(spec, tmp_path / "run-local")
    shutil.rmtree(store.root)
    assert local["track_payload.pt"].read_bytes() == b"payload"


def test_materialize_rejects_load_copy_evil_toctou(tmp_path: Path, monkeypatch):
    spec = _spec(tmp_path)
    payload = tmp_path / "payload.pt"
    payload.write_bytes(b"GOOD")
    store = _store(tmp_path)
    store.publish(spec, {"track_payload.pt": payload})
    original = content_addressed_dag._clone_or_copy
    injected = False

    def inject_evil(source_path, target_path):
        nonlocal injected
        if not injected:
            source_path.write_bytes(b"EVIL")
            injected = True
        return original(source_path, target_path)

    monkeypatch.setattr(content_addressed_dag, "_clone_or_copy", inject_evil)
    destination = tmp_path / "run-local"
    with pytest.raises(ValueError, match="manifest SHA/size"):
        store.materialize(spec, destination)
    assert not destination.exists()


def test_materialize_rejects_symlinked_destination_parent(tmp_path: Path):
    spec = _spec(tmp_path)
    payload = tmp_path / "payload.pt"
    payload.write_bytes(b"payload")
    store = _store(tmp_path)
    store.publish(spec, {"track_payload.pt": payload})
    real = tmp_path / "real-destination"
    real.mkdir()
    linked = tmp_path / "linked-destination"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic-link boundary"):
        store.materialize(spec, linked / "snapshot")


def test_every_declared_bootstrap_source_invalidates_identity(tmp_path: Path):
    paths = run_pipeline._BOOTSTRAP_DAG_SOURCES
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    baseline = source_identity(tmp_path, paths)
    for relative in paths:
        path = tmp_path / relative
        original = path.read_text()
        path.write_text(original + " changed")
        assert source_identity(tmp_path, paths) != baseline
        path.write_text(original)


def test_declared_bootstrap_sources_cover_transitive_local_imports():
    root = Path(run_pipeline.__file__).resolve().parents[1]
    seen = set()
    pending = ["map_learning/bootstrap.py", "map_learning/pipeline.py"]
    while pending:
        relative = pending.pop()
        if relative in seen:
            continue
        path = root / relative
        if not path.is_file():
            continue
        seen.add(relative)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            else:
                modules = []
            for module in modules:
                candidate = str(Path(*module.split(".")).with_suffix(".py"))
                package = str(Path(*module.split(".")) / "__init__.py")
                if (root / candidate).is_file():
                    pending.append(candidate)
                elif (root / package).is_file():
                    pending.append(package)
    assert not (seen - set(run_pipeline._BOOTSTRAP_DAG_SOURCES))


def test_mapping_mask_key_ignores_test_and_unused_channels(tmp_path: Path):
    masks = {
        "mapping.png": [torch.ones(2, 2), torch.ones(2, 2), torch.ones(2, 2), torch.zeros(2, 2)],
        "test.png": [torch.ones(2, 2), torch.ones(2, 2), torch.ones(2, 2)],
    }
    with (tmp_path / "masks.pkl").open("wb") as handle:
        pickle.dump(masks, handle)
    baseline = run_pipeline._mapping_mask_record(tmp_path, ["mapping.png"])
    masks["test.png"][0].zero_()
    masks["mapping.png"][3].fill_(1)
    with (tmp_path / "masks.pkl").open("wb") as handle:
        pickle.dump(masks, handle)
    assert run_pipeline._mapping_mask_record(tmp_path, ["mapping.png"]) == baseline
    masks["mapping.png"][0].zero_()
    with (tmp_path / "masks.pkl").open("wb") as handle:
        pickle.dump(masks, handle)
    assert run_pipeline._mapping_mask_record(tmp_path, ["mapping.png"]) != baseline


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
            elif name in {"base_state", "track_payload", "query_cache", "visibility_cache"}:
                torch.save({"path": str(root / "old-parent"), "value": name}, path)
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
    assert first["track_payload"] != second["track_payload"]
    for name in run_pipeline._BOOTSTRAP_DAG_FILES:
        assert first[name].is_relative_to(first_root)
        assert second[name].is_relative_to(second_root)
    calibration = json.loads(second["scene_calibration"].read_text())
    assert calibration["sources"]["query_cache"] == str(second["query_cache"])
    assert calibration["sources"]["track_payload"] == str(second["track_payload"])
    second_state = torch.load(second["base_state"], map_location="cpu", weights_only=False)
    assert second_state["path"].startswith("@dag-origin-output/")
    assert str(common["artifact_cache"]) not in json.dumps(second_state)
    shutil.rmtree(common["artifact_cache"])
    assert first["track_payload"].is_file()
    assert second["track_payload"].is_file()


def test_pipeline_rejects_input_toctou_before_publish(tmp_path: Path, monkeypatch):
    first_spec = _spec(tmp_path)
    second_spec = _spec(tmp_path, value=2)
    specs = iter((first_spec, second_spec))
    monkeypatch.setattr(run_pipeline, "_bootstrap_dag_spec", lambda *_a, **_k: next(specs))

    def build(**kwargs):
        root = Path(kwargs["output"]) / "bootstrap"
        root.mkdir(parents=True)
        result = {}
        for name, filename in run_pipeline._BOOTSTRAP_DAG_FILES.items():
            path = root / filename
            if path.suffix == ".json":
                path.write_text(json.dumps({"sources": {}}))
            elif name in {"base_state", "track_payload", "query_cache", "visibility_cache"}:
                torch.save({"value": name}, path)
            else:
                path.write_bytes(name.encode())
            result[name] = path
        return result

    monkeypatch.setattr(run_pipeline, "build_bootstrap_and_tracks", build)
    output = tmp_path / "run"
    output.mkdir()
    args = SimpleNamespace(
        output=output,
        artifact_cache=tmp_path / "cache",
        artifact_cache_max_node_gib=0.001,
        artifact_cache_max_total_gib=0.002,
        dataset=tmp_path / "dataset",
        prior=tmp_path / "prior",
        gaussian_type="2dgs",
        sh_degree=3,
    )
    with pytest.raises(RuntimeError, match="changed during DAG build"):
        run_pipeline._build_or_reuse_bootstrap(args=args, config=tmp_path / "c.yaml")
    assert not (Path(args.artifact_cache) / first_spec["node"] / first_spec["key_sha256"]).exists()
