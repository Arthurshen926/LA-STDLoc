"""Fail-closed I/O, scene registry and producer identity for P9 CLIs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys
import uuid

import torch
import kornia
import numpy
import PIL

from common.hashing import canonical_json, sha256_file
from evidence.fixed_pair_matcher_ceiling import (
    COMPLETION_SCHEMA,
    COMPLETION_VERSION,
    validate_pair_gate_report,
    validate_paired_probe,
)
from map_learning.fixed_pair_matcher_ceiling import (
    IMPLEMENTATION_REGISTRY_PATH,
    PREREGISTRATION_BLOB_SHA256,
    PREREGISTRATION_COMMIT,
    PREREGISTRATION_AMENDMENT_COMMIT,
    PREREGISTRATION_PATH,
    PRODUCER_SOURCE_PATHS,
    implementation_registry,
    pair_table_sha256,
    preregistration,
    validate_feature_cache,
)


SHA256 = re.compile(r"[0-9a-f]{64}")


def _expected_runtime_identity() -> dict:
    contract = preregistration()["runtime"]
    return {
        "python": contract["python"],
        "python_executable": contract["python_executable"],
        "torch": contract["torch"],
        "kornia": contract["kornia"],
        "numpy": contract["numpy"],
        "pillow": contract["pillow"],
        "device": contract["device"],
        "cuda_visible_devices": contract["cuda_visible_devices"],
        "dtype": contract["extractor_dtype"],
        "eval": contract["extractor_eval"],
        "inference_mode": contract["torch_inference_mode"],
        "autocast_enabled": contract["autocast_enabled"],
        "torch_intraop_threads": contract["torch_intraop_threads"],
        "torch_interop_threads": contract["torch_interop_threads"],
        "torch_deterministic_algorithms": contract["torch_deterministic_algorithms"],
        "torch_mkldnn_enabled": contract["torch_mkldnn_enabled"],
        "torch_float32_matmul_precision": contract["torch_float32_matmul_precision"],
        "self_attention_backend": contract["self_attention_backend"],
        "cross_attention_backend": contract["cross_attention_backend"],
        "environment": contract["environment"],
    }


def validate_producer_identity_payload(identity: dict) -> dict:
    """Validate the recursively hashed producer/runtime identity in artifacts."""
    required = set(preregistration()["producer_identity_contract"]["required_keys"])
    source_hashes = identity.get("source_file_sha256")
    if (
        not required.issubset(identity)
        or identity.get("schema") != "lafgs_p9_fixed_pair_matcher_ceiling_producer"
        or identity.get("version") != 1
        or identity.get("algorithm") != "p9_fixed_pair_matcher_ceiling"
        or not isinstance(identity.get("entrypoint"), str)
        or re.fullmatch(r"[0-9a-f]{40}", str(identity.get("git_commit", ""))) is None
        or re.fullmatch(r"[0-9a-f]{40}", str(identity.get("implementation_commit", "")))
        is None
        or identity.get("source_paths") != list(PRODUCER_SOURCE_PATHS)
        or not isinstance(source_hashes, dict)
        or set(source_hashes) != set(PRODUCER_SOURCE_PATHS)
        or any(SHA256.fullmatch(str(value)) is None for value in source_hashes.values())
        or identity.get("required_source_paths_clean") is not True
        or identity.get("runtime") != _expected_runtime_identity()
        or identity.get("preregistration", {}).get("original_commit")
        != PREREGISTRATION_COMMIT
        or identity.get("preregistration", {}).get("commit")
        != PREREGISTRATION_AMENDMENT_COMMIT
        or identity.get("preregistration", {}).get("blob_sha256")
        != PREREGISTRATION_BLOB_SHA256
    ):
        raise ValueError("P9 producer identity is structurally invalid")
    compiled_domain = {
        "implementation_commit": identity["implementation_commit"],
        "preregistration_commit": PREREGISTRATION_AMENDMENT_COMMIT,
        "preregistration_blob_sha256": PREREGISTRATION_BLOB_SHA256,
        "source_paths": list(PRODUCER_SOURCE_PATHS),
        "source_file_sha256": source_hashes,
        "runtime": identity["runtime"],
    }
    observed = hashlib.sha256(
        canonical_json(compiled_domain).encode("utf-8")
    ).hexdigest()
    if identity.get("compiled_identity") != observed:
        raise ValueError("P9 producer compiled identity hash is stale")
    return {"compiled_identity": observed}


def configure_formal_cpu_runtime() -> None:
    """Apply and verify the preregistered deterministic CPU execution axes."""
    contract = preregistration()["runtime"]
    if str(Path(sys.executable).resolve()) != contract["python_executable"]:
        raise RuntimeError("P9 must use the preregistered Python executable")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != contract["cuda_visible_devices"]:
        raise RuntimeError("P9 requires CUDA_VISIBLE_DEVICES to be the empty string")
    if torch.cuda.is_available():
        raise RuntimeError("P9 formal CPU runtime unexpectedly exposes CUDA")
    if torch.get_default_dtype() != torch.float32 or torch.is_autocast_enabled():
        raise RuntimeError("P9 requires default float32 with autocast disabled")
    for name, expected in contract["environment"].items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"P9 formal environment requires {name}={expected}")
    torch.set_num_threads(int(contract["torch_intraop_threads"]))
    try:
        torch.set_num_interop_threads(int(contract["torch_interop_threads"]))
    except RuntimeError:
        if torch.get_num_interop_threads() != int(contract["torch_interop_threads"]):
            raise
    torch.use_deterministic_algorithms(bool(contract["torch_deterministic_algorithms"]))
    torch.backends.mkldnn.enabled = bool(contract["torch_mkldnn_enabled"])
    torch.set_float32_matmul_precision(contract["torch_float32_matmul_precision"])


def expected_sha256(value: str, *, label: str) -> str:
    digest = str(value).strip().lower()
    if SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    return digest


def local_file(path: str | Path, *, label: str) -> Path:
    if "://" in str(path):
        raise ValueError(f"{label} must be a local file")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def attest_file(path: str | Path, digest: str, *, label: str) -> Path:
    resolved = local_file(path, label=label)
    if sha256_file(resolved) != expected_sha256(
        digest, label=f"expected {label} SHA-256"
    ):
        raise ValueError(f"{label} SHA-256 differs from explicit P9 contract")
    return resolved


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, TypeError):
        return torch.load(path, map_location="cpu", weights_only=False)


def scene_contract(scene: str) -> dict:
    normalized = str(scene).lower()
    fixed = preregistration()["fixed_scene_registry"]
    if normalized not in fixed:
        raise ValueError("P9 scene must be stairs or greatcourt")
    return {"scene": normalized, **fixed[normalized]}


def require_fixed_path(path: str | Path, expected: str, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    frozen = Path(expected).expanduser().resolve()
    if resolved != frozen:
        raise ValueError(f"{label} path differs from the P9 fixed scene registry")
    return resolved


def load_mapping_scope(
    *,
    path: str | Path,
    expected_file_sha256: str,
    query_cache_path: Path,
    query_cache_sha256: str,
    query_count: int,
    mapping_keypoints: int,
) -> dict:
    """Validate the existing V2 proof without promoting any P8 match rows."""
    proof_path = attest_file(path, expected_file_sha256, label="mapping-scope proof")
    payload = json.loads(proof_path.read_text())
    checks = payload.get("checks")
    audit = payload.get("audit")
    sources = payload.get("sources")
    refreshed = sources.get("refreshed_cache") if isinstance(sources, dict) else None
    expected = payload.get("expected")
    if (
        payload.get("schema") != "lafgs_mapping_sparse_refresh_equivalence"
        or payload.get("version") != 2
        or payload.get("uses_test_queries") is not False
        or payload.get("valid") is not True
        or not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
        or not isinstance(audit, dict)
        or audit.get("query_order_exact") is not True
        or not isinstance(refreshed, dict)
        or Path(str(refreshed.get("path", ""))).resolve() != query_cache_path
        or refreshed.get("sha256") != query_cache_sha256
        or not isinstance(expected, dict)
        or int(expected.get("mapping_keypoints", -1)) != int(mapping_keypoints)
        or any(
            int(audit.get(field, -1)) != int(query_count)
            for field in (
                "query_count",
                "effective_sparse_depth_exact_query_count",
                "native_alpha_exact_query_count",
                "refreshed_metadata_pass_count",
                "track_input_exact_query_count",
            )
        )
    ):
        raise ValueError("P9 mapping-scope proof is invalid")
    return {
        "mode": "mapping_sparse_refresh_equivalence_v2",
        "uses_test_queries": False,
        "equivalence_report": {
            "path": str(proof_path),
            "sha256": sha256_file(proof_path),
        },
    }


def _canonical_proposal_pairs(
    pairs: Sequence[Sequence[int]], *, query_count: int
) -> list[tuple[int, int]]:
    result = [(int(value[0]), int(value[1])) for value in pairs]
    if result != sorted(set(result)) or any(
        left < 0 or left >= right or right >= query_count for left, right in result
    ):
        raise ValueError("P9 proposal pairs are not canonical, unique, and in range")
    return result


def _proposal_graph(pairs: Sequence[tuple[int, int]], query_count: int) -> dict:
    parent = list(range(query_count))
    degree = [0] * query_count

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for left, right in pairs:
        degree[left] += 1
        degree[right] += 1
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left
    return {
        "component_count": len({find(index) for index in range(query_count)}),
        "isolated_camera_count": sum(value == 0 for value in degree),
        "minimum_degree": min(degree, default=0),
        "maximum_degree": max(degree, default=0),
    }


def _candidate_union_sha256(pairs: Sequence[tuple[int, int]]) -> str:
    encoded = json.dumps(
        {
            "keypoint_counts": [],
            "pairs": [[left, right] for left, right in pairs],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _hash_proposal_tensor(hasher, value: torch.Tensor) -> None:
    tensor = torch.as_tensor(value).detach().cpu().reshape(-1)
    hasher.update(struct.pack("<Q", int(tensor.numel())))
    for item in tensor.long().tolist():
        hasher.update(struct.pack("<q", int(item)))


def _proposal_content_sha256(payload: dict) -> str:
    hasher = hashlib.sha256()
    arms = payload.get("arms", {})
    header = {
        "schema": payload.get("schema"),
        "version": payload.get("version"),
        "uses_test_queries": payload.get("uses_test_queries"),
        "query_count": payload.get("query_count"),
        "query_names_sha256": payload.get("query_names_sha256"),
        "query_cache_sha256": payload.get("query_cache_sha256"),
        "mapping_keypoint_count": payload.get("mapping_keypoint_count"),
        "mapping_nms_radius": payload.get("mapping_nms_radius"),
        "exact_pair_budget": payload.get("exact_pair_budget"),
        "source_contract": payload.get("source_contract"),
        "arm_sources": {
            name: {
                "source_policy": value.get("source_policy"),
                "source_artifact_sha256": value.get("source_artifact", {}).get(
                    "sha256"
                ),
                "unavailable_source_lineage": value.get("unavailable_source_lineage"),
            }
            for name, value in sorted(arms.items())
        },
        "candidate_union": payload.get("candidate_union"),
    }
    hasher.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    for name, value in sorted(arms.items()):
        hasher.update(name.encode())
        pair = value.get("pair", {})
        _hash_proposal_tensor(hasher, pair.get("left_query_index"))
        _hash_proposal_tensor(hasher, pair.get("right_query_index"))
    return hasher.hexdigest()


def _validate_fixed_proposal_payload(
    payload: dict,
    *,
    query_names_sha256: str,
    query_cache_path: str,
    query_cache_sha256: str,
    mapping_keypoints: int,
    mapping_scope: dict,
    pair_budget: int,
    content_sha256: str,
) -> list[tuple[int, int]]:
    """Locally validate the exact P8 pair-only table without P8 code imports."""
    source_contract = payload.get("source_contract")
    if (
        payload.get("schema") != "lafgs_cycle_verified_pair_proposal_table"
        or payload.get("version") != 1
        or payload.get("uses_test_queries") is not False
        or payload.get("query_names_sha256") != query_names_sha256
        or payload.get("query_cache_path") != query_cache_path
        or payload.get("query_cache_sha256") != query_cache_sha256
        or payload.get("mapping_keypoint_count") != mapping_keypoints
        or payload.get("mapping_nms_radius") != 4
        or payload.get("mapping_scope") != mapping_scope
        or payload.get("exact_pair_budget") != pair_budget
        or not isinstance(source_contract, dict)
        or source_contract.get("scope") != "archived_pair_tables_only"
        or source_contract.get("track_factor_lineage_reused") is not False
        or source_contract.get("track_or_geometry_measurements_reused") is not False
        or source_contract.get("fresh_cache_is_authoritative_for_query_order_k_nms")
        is not True
    ):
        raise ValueError("P9 fixed proposal contract/lineage differs")
    query_count = int(payload.get("query_count", -1))
    arms = payload.get("arms")
    if (
        query_count <= 1
        or not isinstance(arms, dict)
        or set(arms)
        != {
            "nearest",
            "mapping_geometry",
        }
    ):
        raise ValueError("P9 fixed proposal arms/query count are invalid")
    pair_sets = []
    for name, policy in (
        ("nearest", "nearest"),
        ("mapping_geometry", "parallax_diverse"),
    ):
        arm = arms[name]
        source = arm.get("source_artifact") if isinstance(arm, dict) else None
        pair = arm.get("pair", {}) if isinstance(arm, dict) else {}
        left = torch.as_tensor(pair.get("left_query_index"))
        right = torch.as_tensor(pair.get("right_query_index"))
        if (
            not isinstance(source, dict)
            or arm.get("source_policy") != policy
            or not source.get("path")
            or SHA256.fullmatch(str(source.get("sha256", ""))) is None
            or not isinstance(arm.get("unavailable_source_lineage"), list)
            or left.dtype != torch.int64
            or right.dtype != torch.int64
            or left.numel() != pair_budget
            or right.numel() != pair_budget
        ):
            raise ValueError(f"P9 fixed proposal arm {name} differs")
        pair_sets.append(
            _canonical_proposal_pairs(
                list(zip(left.tolist(), right.tolist())), query_count=query_count
            )
        )
    union = sorted(set(pair_sets[0]) | set(pair_sets[1]))
    graph = _proposal_graph(union, query_count)
    if not union or len(union) > 2 * pair_budget or graph["isolated_camera_count"]:
        raise ValueError("P9 proposal candidate union is invalid")
    expected_union = {
        "arm_count": 2,
        "pair_count": len(union),
        **graph,
        "sha256": _candidate_union_sha256(union),
    }
    if payload.get("candidate_union") != expected_union:
        raise ValueError("P9 proposal candidate-union diagnostics are stale")
    observed_content = _proposal_content_sha256(payload)
    if (
        payload.get("content_sha256") != observed_content
        or observed_content != content_sha256
    ):
        raise ValueError("P9 proposal content SHA-256 is stale")
    return pair_sets[0]


def load_fixed_proposals(
    *,
    scene: str,
    path: str | Path,
    expected_file_sha256: str,
    expected_content_sha256: str,
    feature_cache: dict,
) -> dict:
    contract = scene_contract(scene)
    proposal_path = require_fixed_path(
        path, contract["pair_proposals"]["path"], label="pair proposals"
    )
    proposal_path = attest_file(
        proposal_path, expected_file_sha256, label="pair proposals"
    )
    if (
        expected_sha256(expected_file_sha256, label="expected proposals SHA-256")
        != contract["pair_proposals"]["sha256"]
        or expected_sha256(
            expected_content_sha256,
            label="expected proposals content SHA-256",
        )
        != contract["pair_proposals"]["content_sha256"]
    ):
        raise ValueError("P9 proposal digests differ from fixed scene registry")
    payload = torch_load(proposal_path)
    cache_lineage = feature_cache["inputs"]["query_cache"]
    pairs = _validate_fixed_proposal_payload(
        payload,
        query_names_sha256=feature_cache["query_names_sha256"],
        query_cache_path=cache_lineage["path"],
        query_cache_sha256=cache_lineage["sha256"],
        mapping_keypoints=contract["requested_keypoint_count"],
        mapping_scope=cache_lineage["mapping_scope"],
        pair_budget=contract["exact_nearest_pair_count"],
        content_sha256=contract["pair_proposals"]["content_sha256"],
    )
    if (
        len(pairs) != contract["exact_nearest_pair_count"]
        or pair_table_sha256(pairs) != contract["nearest_pair_table_sha256"]
    ):
        raise ValueError("P9 nearest pair table differs from fixed registry")
    return {
        "path": proposal_path,
        "sha256": sha256_file(proposal_path),
        "content_sha256": payload["content_sha256"],
        "payload": payload,
        "pairs": pairs,
        "pair_table_sha256": contract["nearest_pair_table_sha256"],
    }


def load_feature_cache(
    *,
    scene: str,
    path: str | Path,
    expected_file_sha256: str,
    expected_content_sha256: str,
) -> dict:
    cache_path = attest_file(path, expected_file_sha256, label="P9 feature cache")
    if cache_path.name != "p9_fixed_pair_feature_cache.pt":
        raise ValueError("P9 feature cache must use its preregistered fixed filename")
    payload = torch_load(cache_path)
    validate_feature_cache(
        payload,
        expected_scene=scene,
        expected_content_sha256=expected_content_sha256,
    )
    contract = scene_contract(scene)
    cache_input = payload.get("inputs", {}).get("query_cache", {})
    if (
        payload.get("query_count") != contract["query_count"]
        or payload.get("query_names_sha256") != contract["query_names_sha256"]
        or payload.get("requested_keypoint_count")
        != contract["requested_keypoint_count"]
        or Path(str(cache_input.get("path", ""))).resolve()
        != Path(contract["query_cache"]["path"]).resolve()
        or cache_input.get("sha256") != contract["query_cache"]["sha256"]
    ):
        raise ValueError("P9 feature cache differs from the fixed scene registry")
    validate_producer_identity_payload(payload["producer_identity"])
    return {
        "path": cache_path,
        "sha256": sha256_file(cache_path),
        "content_sha256": payload["content_sha256"],
        "payload": payload,
    }


def producer_identity(*, entrypoint: str) -> dict:
    """Bind real output to reviewed code, exact runtime, and a clean source set."""
    registry = implementation_registry()
    root = Path(__file__).resolve().parents[1]
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", *PRODUCER_SOURCE_PATHS],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("P9 producer source paths are dirty")
    runtime = {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "torch": torch.__version__,
        "kornia": kornia.__version__,
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
        "device": "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dtype": "float32",
        "eval": True,
        "inference_mode": True,
        "autocast_enabled": False,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_mkldnn_enabled": torch.backends.mkldnn.enabled,
        "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
        "self_attention_backend": "torch_scaled_dot_product_attention_cpu_no_flash",
        "cross_attention_backend": "kornia_manual_einsum_softmax_cpu",
        "environment": {
            name: os.environ.get(name)
            for name in preregistration()["runtime"]["environment"]
        },
    }
    source_hashes = {name: sha256_file(root / name) for name in PRODUCER_SOURCE_PATHS}
    compiled_domain = {
        "implementation_commit": registry["implementation_commit"],
        "preregistration_commit": PREREGISTRATION_AMENDMENT_COMMIT,
        "preregistration_blob_sha256": PREREGISTRATION_BLOB_SHA256,
        "source_paths": list(PRODUCER_SOURCE_PATHS),
        "source_file_sha256": source_hashes,
        "runtime": runtime,
    }
    identity = {
        "schema": "lafgs_p9_fixed_pair_matcher_ceiling_producer",
        "version": 1,
        "algorithm": "p9_fixed_pair_matcher_ceiling",
        "entrypoint": str(entrypoint),
        "git_commit": current,
        "implementation_commit": registry["implementation_commit"],
        "implementation_registry": {
            "path": str(IMPLEMENTATION_REGISTRY_PATH),
            "sha256": sha256_file(IMPLEMENTATION_REGISTRY_PATH),
        },
        "preregistration": {
            "path": str(PREREGISTRATION_PATH),
            "original_commit": PREREGISTRATION_COMMIT,
            "commit": PREREGISTRATION_AMENDMENT_COMMIT,
            "blob_sha256": PREREGISTRATION_BLOB_SHA256,
        },
        "source_paths": list(PRODUCER_SOURCE_PATHS),
        "source_file_sha256": source_hashes,
        "required_source_paths_clean": True,
        "runtime": runtime,
        "compiled_identity": hashlib.sha256(
            canonical_json(compiled_domain).encode("utf-8")
        ).hexdigest(),
    }
    if runtime != _expected_runtime_identity():
        raise RuntimeError("P9 formal runtime/backend differs from preregistration")
    validate_producer_identity_payload(identity)
    return identity


def validate_fresh_file_output(path: str | Path, *, protected: Sequence[Path]) -> Path:
    output = Path(path).expanduser().resolve()
    if output in {Path(value).resolve() for value in protected}:
        raise ValueError("P9 output must not overwrite a frozen input")
    if output.exists():
        raise FileExistsError(f"P9 output must be fresh: {output}")
    return output


def atomic_torch_save_fresh(
    payload: dict,
    output: Path,
    *,
    validator: Callable[[dict], object] | None = None,
) -> Path:
    """Save, reload, validate, then atomically expose a fresh tensor file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        torch.save(payload, temporary)
        reloaded = torch_load(temporary)
        if validator is not None:
            validator(reloaded)
        if output.exists():
            raise FileExistsError(
                f"P9 output appeared during materialization: {output}"
            )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def atomic_json_save_fresh(
    payload: dict,
    output: Path,
    *,
    validator: Callable[[dict], object] | None = None,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        reloaded = json.loads(temporary.read_text())
        if validator is not None:
            validator(reloaded)
        if output.exists():
            raise FileExistsError(
                f"P9 output appeared during materialization: {output}"
            )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def completion_payload(
    *,
    scene: str,
    run_uuid: str,
    producer: dict,
    compiled_identity: str,
    feature_cache: dict,
    proposals: dict,
    probe_path: Path,
    probe_logical_path: Path,
    probe: dict,
    summary_path: Path,
    summary_logical_path: Path,
) -> dict:
    return {
        "schema": COMPLETION_SCHEMA,
        "version": COMPLETION_VERSION,
        "scene": scene,
        "mapping_only": True,
        "uses_test_queries": False,
        "complete": True,
        "partial": False,
        "resume_allowed": False,
        "run_uuid": run_uuid,
        "producer_identity": producer,
        "compiled_identity": str(compiled_identity),
        "build_order": ["mnn_control", "lighterglue_variant"],
        "inputs": {
            "feature_cache": {
                "path": str(feature_cache["path"]),
                "sha256": feature_cache["sha256"],
                "content_sha256": feature_cache["content_sha256"],
            },
            "pair_proposals": {
                "path": str(proposals["path"]),
                "sha256": proposals["sha256"],
                "content_sha256": proposals["content_sha256"],
                "nearest_pair_table_sha256": proposals["pair_table_sha256"],
            },
        },
        "artifacts": {
            "paired_probe": {
                "path": str(probe_logical_path),
                "sha256": sha256_file(probe_path),
                "content_sha256": probe["content_sha256"],
            },
            "summary": {
                "path": str(summary_logical_path),
                "sha256": sha256_file(summary_path),
            },
        },
        "failure_recovery": "isolate_root_and_rebuild_both_arms_from_fresh_cache",
    }


def load_completion(*, path: str | Path, expected_file_sha256: str) -> dict:
    completion_path = attest_file(
        path, expected_file_sha256, label="P9 paired completion"
    )
    if completion_path.name != "paired_match_completion.json":
        raise ValueError("P9 completion must use its preregistered fixed filename")
    completion = json.loads(completion_path.read_text())
    completion_contract = preregistration()["artifact_schemas"]["paired_completion"]
    required_keys = set(completion_contract["required_keys"])
    if (
        not required_keys.issubset(completion)
        or completion.get("schema") != COMPLETION_SCHEMA
        or completion.get("version") != COMPLETION_VERSION
        or completion.get("mapping_only") is not True
        or completion.get("uses_test_queries") is not False
        or completion.get("complete") is not True
        or completion.get("partial") is not False
        or completion.get("resume_allowed") is not False
        or completion.get("build_order") != list(("mnn_control", "lighterglue_variant"))
        or completion.get("failure_recovery")
        != completion_contract["failure_recovery_exact_value"]
    ):
        raise ValueError("P9 paired completion is partial or invalid")
    feature_lineage = completion.get("inputs", {}).get("feature_cache", {})
    feature = load_feature_cache(
        scene=completion["scene"],
        path=feature_lineage.get("path", ""),
        expected_file_sha256=feature_lineage.get("sha256", ""),
        expected_content_sha256=feature_lineage.get("content_sha256", ""),
    )
    proposal_lineage = completion.get("inputs", {}).get("pair_proposals", {})
    proposals = load_fixed_proposals(
        scene=completion["scene"],
        path=proposal_lineage.get("path", ""),
        expected_file_sha256=proposal_lineage.get("sha256", ""),
        expected_content_sha256=proposal_lineage.get("content_sha256", ""),
        feature_cache=feature["payload"],
    )
    probe_lineage = completion.get("artifacts", {}).get("paired_probe", {})
    probe_path = attest_file(
        probe_lineage.get("path", ""),
        probe_lineage.get("sha256", ""),
        label="P9 paired probe",
    )
    if probe_path.name != "fixed_pair_match_probe.pt":
        raise ValueError("P9 paired probe must use its fixed filename")
    probe = torch_load(probe_path)
    validate_paired_probe(
        probe,
        feature_cache=feature["payload"],
        expected_scene=completion["scene"],
        expected_pairs=proposals["pairs"],
        expected_content_sha256=probe_lineage.get("content_sha256"),
    )
    if (
        completion.get("run_uuid") != probe.get("run_uuid")
        or completion.get("producer_identity") != probe.get("producer_identity")
        or completion.get("compiled_identity")
        != probe.get("producer_identity", {}).get("compiled_identity")
        or completion.get("compiled_identity")
        != feature["payload"].get("producer_identity", {}).get("compiled_identity")
        or completion.get("run_uuid") != feature["payload"].get("run_uuid")
        or proposal_lineage.get("nearest_pair_table_sha256")
        != proposals["pair_table_sha256"]
        or probe.get("feature_cache")
        != {
            "schema": "lafgs_p9_fixed_pair_feature_cache",
            "path": str(feature["path"]),
            "sha256": feature["sha256"],
            "content_sha256": feature["content_sha256"],
            "same_exact_rows_for_both_arms": True,
        }
        or {
            key: probe.get("pair_table", {}).get(key)
            for key in (
                "path",
                "sha256",
                "content_sha256",
                "arm",
                "pair_table_sha256",
                "match_rows_reused",
            )
        }
        != {
            "path": str(proposals["path"]),
            "sha256": proposals["sha256"],
            "content_sha256": proposals["content_sha256"],
            "arm": "nearest",
            "pair_table_sha256": proposals["pair_table_sha256"],
            "match_rows_reused": False,
        }
    ):
        raise ValueError("P9 completion/probe arms were spliced")
    validate_producer_identity_payload(completion["producer_identity"])
    summary = completion.get("artifacts", {}).get("summary", {})
    summary_path = attest_file(
        summary.get("path", ""),
        summary.get("sha256", ""),
        label="P9 probe summary",
    )
    if summary_path.name != "fixed_pair_match_probe.json":
        raise ValueError("P9 probe summary must use its fixed filename")
    if not (probe_path.parent == summary_path.parent == completion_path.parent):
        raise ValueError("P9 paired root artifacts were spliced across directories")
    summary_payload = json.loads(summary_path.read_text())
    expected_summary = {
        "schema": "lafgs_p9_fixed_pair_match_probe_summary",
        "version": 1,
        "scene": probe["scene"],
        "mapping_only": True,
        "uses_test_queries": False,
        "run_uuid": probe["run_uuid"],
        "query_count": probe["query_count"],
        "pair_count": probe["pair_table"]["pair_count"],
        "pair_table_sha256": probe["pair_table"]["pair_table_sha256"],
        "content_sha256": probe["content_sha256"],
        "arms": {
            name: probe["arms"][name]["metrics"]
            for name in ("mnn_control", "lighterglue_variant")
        },
    }
    if summary_payload != expected_summary:
        raise ValueError("P9 probe summary is not an exact validated projection")
    return {
        "path": completion_path,
        "sha256": sha256_file(completion_path),
        "payload": completion,
        "feature_cache": feature,
        "proposals": proposals,
        "probe_path": probe_path,
        "probe_sha256": sha256_file(probe_path),
        "probe": probe,
    }


def load_scene_gate(
    *,
    path: str | Path,
    expected_file_sha256: str,
    expected_scene: str,
    require_pass: bool | None,
) -> dict:
    """Reload a P9 scene gate with its exact parent/authority boundary."""
    gate_path = attest_file(
        path, expected_file_sha256, label=f"P9 {expected_scene} scene Pair Gate"
    )
    if gate_path.name != "p9_fixed_pair_matcher_ceiling_pair_gate.json":
        raise ValueError("P9 scene Pair Gate must use its fixed filename")
    gate = json.loads(gate_path.read_text())
    validate_pair_gate_report(gate, expected_scene=expected_scene)
    validate_producer_identity_payload(gate["producer_identity"])
    required = set(
        preregistration()["artifact_schemas"]["scene_pair_gate"]["required_keys"]
    )
    if (
        not required.issubset(gate)
        or gate.get("schema") != "lafgs_p9_fixed_pair_matcher_ceiling_pair_gate"
        or gate.get("version") != 1
        or gate.get("scene") != expected_scene
        or gate.get("mapping_only") is not True
        or gate.get("uses_test_queries") is not False
        or gate.get("valid") is not True
        or gate.get("advance_to_track_implementation_review") is not False
        or gate.get("authorizes_real_track_run") is not False
        or gate.get("advance_to_pose") is not False
        or gate.get("authorizes_test") is not False
        or gate.get("changes_method_default") is not False
        or not isinstance(gate.get("producer_identity"), dict)
        or gate.get("compiled_identity")
        != gate["producer_identity"].get("compiled_identity")
        or SHA256.fullmatch(str(gate.get("compiled_identity", ""))) is None
        or gate.get("policy", {}).get("control") != "mnn_control"
        or gate.get("policy", {}).get("variant") != "lighterglue_variant"
        or gate.get("policy", {}).get("same_extractor_rows") is not True
        or gate.get("policy", {}).get("fixed_pair_table_sha256")
        != scene_contract(expected_scene)["nearest_pair_table_sha256"]
        or not isinstance(gate.get("scene_pair_gate_passed"), bool)
        or gate.get("requires_other_scene") is not gate.get("scene_pair_gate_passed")
    ):
        raise ValueError("P9 scene Pair Gate is structurally invalid")
    passed = gate["scene_pair_gate_passed"]
    expected_decision = (
        "SCENE_PAIR_PASS_REQUIRES_OTHER_SCENE"
        if passed
        else "STOP_FIXED_PAIR_MATCHER_CEILING"
    )
    gates = gate.get("gates")
    if (
        gate.get("decision") != expected_decision
        or not isinstance(gates, dict)
        or not gates
        or any(not isinstance(value, bool) for value in gates.values())
        or passed is not all(gates.values())
    ):
        raise ValueError("P9 scene Pair Gate decision/gates are inconsistent")
    parent = gate.get("parent_stairs_gate")
    if expected_scene == "stairs" and parent is not None:
        raise ValueError("P9 Stairs scene Pair Gate cannot name a parent")
    if expected_scene == "greatcourt" and (
        not isinstance(parent, dict)
        or not isinstance(parent.get("path"), str)
        or SHA256.fullmatch(str(parent.get("sha256", ""))) is None
        or parent.get("scientific_projection", {}).get("scene") != "stairs"
        or parent.get("scientific_projection", {}).get("scene_pair_gate_passed")
        is not True
        or parent.get("scientific_projection", {}).get("decision")
        != "SCENE_PAIR_PASS_REQUIRES_OTHER_SCENE"
    ):
        raise ValueError("P9 GreatCourt scene Pair Gate lacks its passing parent")
    if require_pass and (
        gate.get("scene_pair_gate_passed") is not True
        or gate.get("decision") != "SCENE_PAIR_PASS_REQUIRES_OTHER_SCENE"
    ):
        raise ValueError("P9 parent scene Pair Gate did not pass")
    if require_pass is False and gate.get("scene_pair_gate_passed") is not False:
        raise ValueError("P9 expected a scientific STOP scene gate")
    return {
        "path": gate_path,
        "sha256": sha256_file(gate_path),
        "payload": gate,
        "scientific_projection": {
            "scene": gate["scene"],
            "scene_pair_gate_passed": gate["scene_pair_gate_passed"],
            "decision": gate["decision"],
            "compiled_identity": gate["compiled_identity"],
            "fixed_pair_table_sha256": gate["policy"]["fixed_pair_table_sha256"],
        },
    }
