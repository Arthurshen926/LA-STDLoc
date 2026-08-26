"""Fail-closed contracts for the V7 safe closed-loop mainline."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
import fnmatch
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml


V7_CONFIG_SCHEMA = "lafgs_v7_safe_closed_loop"
V7_P0_REPORT_SCHEMA = "lafgs_v7_p0_noop_report"

METHOD_INVARIANTS = (
    "no_source_mapping_rgb",
    "no_detector_training",
    "no_learned_match_scorer",
    "no_query_adapter_or_context_network",
    "no_multi_prototype",
    "one_anchor_one_id_xyz_descriptor",
    "no_gaussian_or_rendered_depth_pnp_coordinates",
    "feedback_queries_never_enter_tracks_or_observation_csr",
    "feedback_descriptors_never_copied_into_map",
    "same_selector_at_initialization_and_after_updates",
    "frozen_superpoint_exact_global_top1_one_poselib",
    "test_queries_never_update_or_select_the_map",
)

FORBIDDEN_FORMAL_FEATURES = (
    "detector_training",
    "learned_match_scorer",
    "query_adapter",
    "multi_prototype",
    "query_geometry_loo",
    "query_descriptor_loo",
    "fixed_probe_train_validation_split",
    "multiple_sensor_variants_per_pose",
    "retrieval",
    "refinement",
)

FORBIDDEN_IMPORT_PATTERNS = (
    "context_*",
    "feature_booster*",
    "*prototype*",
    "*_loo*",
    "*detector*trainer*",
    "*scorer*",
    "group_ransac*",
    "*refinement*",
    "*legacy*closed_loop*",
    "v6_proposals*",
)

FORBIDDEN_ARTIFACT_KEYS = (
    "anchor_extra_prototype_*",
    "detector_checkpoint",
    "scorer_checkpoint",
    "query_adapter_checkpoint",
    "loo_rebuilt_xyz",
)

DEPLOYMENT_CONTRACT_FIELDS = (
    "keypoint_count",
    "nms_radius",
    "ransac_reprojection_px",
    "calibration_split",
    "evaluated_split",
    "pose_solves",
    "duplicate_anchor_suppression",
    "guided_sampling",
    "group_aware_pose",
    "group_field",
    "group_hypothesis_samples",
    "capacity_assignment",
    "assignment_topk",
    "assignment_dustbin_score",
    "descriptor_protocol",
    "photometric_canonicalization_contract",
    "context_state",
)



def sha256_file(path: str | Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_v7_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("V7 config must be a mapping")
    if payload.get("schema") != V7_CONFIG_SCHEMA or payload.get("version") != 1:
        raise ValueError("unsupported V7 config schema or version")
    invariants = payload.get("method", {}).get("invariants")
    if tuple(invariants or ()) != METHOD_INVARIANTS:
        raise ValueError("V7 method invariants differ from the immutable contract")
    forbidden = payload.get("forbidden_formal_features")
    if not isinstance(forbidden, Mapping):
        raise ValueError("V7 forbidden feature registry is missing")
    for name in FORBIDDEN_FORMAL_FEATURES:
        if forbidden.get(name) is not True:
            raise ValueError(f"V7 forbidden feature is not fail-closed: {name}")
    online = payload.get("method", {}).get("online_protocol")
    if online != "frozen_superpoint_exact_global_top1_one_standard_poselib":
        raise ValueError("V7 online protocol differs from the frozen plant")
    if payload.get("method", {}).get("maximum_rounds") != 2:
        raise ValueError("V7 first version permits at most two rounds")
    return payload


def _walk_keys(value: Any, prefix: str = "") -> list[str]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            keys.append(path)
            keys.extend(_walk_keys(child, path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            keys.extend(_walk_keys(child, f"{prefix}[{index}]"))
    return keys


def validate_compact_map(state: Mapping[str, Any]) -> dict[str, int]:
    if state.get("schema") != "lafgs_materialized_anchor_map":
        raise ValueError("unsupported V7 compact map schema")
    ids = torch.as_tensor(state.get("anchor_ids", ())).long().reshape(-1)
    xyz = torch.as_tensor(state.get("anchor_xyz", ()))
    descriptors = torch.as_tensor(state.get("anchor_features", ()))
    if ids.numel() == 0:
        raise ValueError("V7 compact map cannot be empty")
    if xyz.shape != (ids.numel(), 3):
        raise ValueError("V7 compact map xyz rows do not align")
    if descriptors.ndim != 2 or descriptors.shape[0] != ids.numel():
        raise ValueError("V7 compact map descriptor rows do not align")
    if torch.unique(ids).numel() != ids.numel():
        raise ValueError("V7 Anchor IDs must be unique")
    if not bool(torch.isfinite(xyz).all()) or not bool(torch.isfinite(descriptors).all()):
        raise ValueError("V7 compact map contains non-finite tensors")
    for path in _walk_keys(state):
        leaf = path.rsplit(".", 1)[-1]
        if any(fnmatch.fnmatch(leaf, pattern) for pattern in FORBIDDEN_ARTIFACT_KEYS):
            raise ValueError(f"forbidden V7 formal artifact field: {path}")
    construction = state.get("projective_anchor_construction")
    if not isinstance(construction, Mapping):
        raise ValueError("V7 map lacks projective construction lineage")
    if construction.get("direct_gaussian_surface_anchor") is not False:
        raise ValueError("Gaussian centers cannot be V7 PnP coordinates")
    if construction.get("gaussian_depth_role") != "proposal_and_visibility_only":
        raise ValueError("rendered depth has an invalid V7 geometry role")
    final_xyz = str(construction.get("final_xyz_source", ""))
    if "ray_triangulation" not in final_xyz:
        raise ValueError("V7 final xyz must come from pure-ray triangulation")
    provenance = state.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("V7 map provenance is missing")
    if provenance.get("uses_source_mapping_rgb") is not False:
        raise ValueError("V7 map must not use source mapping RGB")
    if provenance.get("uses_test_queries") is not False:
        raise ValueError("V7 map must not use test queries")
    if provenance.get("uses_gaussian_geometry_for_triangulation") is not False:
        raise ValueError("V7 map xyz cannot use Gaussian geometry")
    return {"anchor_count": int(ids.numel()), "descriptor_dim": int(descriptors.shape[1])}


def tensor_tree_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(tensor_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            tensor_tree_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _is_timing_field(key: str) -> bool:
    return key == "runtime_ms" or key.endswith("_ms") or "_ms_" in key


def strip_timing_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: strip_timing_fields(child)
            for key, child in value.items()
            if not _is_timing_field(str(key))
        }
    if isinstance(value, list):
        return [strip_timing_fields(child) for child in value]
    return value


def compare_query_results(reference: Sequence[Any], candidate: Sequence[Any]) -> dict[str, int]:
    if len(reference) != len(candidate):
        raise ValueError("P0 reference and candidate query counts differ")
    for index, (left, right) in enumerate(zip(reference, candidate)):
        left_clean = strip_timing_fields(left)
        right_clean = strip_timing_fields(right)
        if left_clean != right_clean:
            keys = sorted(
                key for key in set(left_clean) | set(right_clean)
                if left_clean.get(key) != right_clean.get(key)
            ) if isinstance(left_clean, Mapping) and isinstance(right_clean, Mapping) else []
            raise ValueError(
                f"P0 query {index} non-timing fields differ"
                + (f" at: {', '.join(keys)}" if keys else "")
            )
    return {"query_count": len(reference), "non_timing_mismatch_count": 0}


def compare_deployment_contracts(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    differing = [
        field
        for field in DEPLOYMENT_CONTRACT_FIELDS
        if reference.get(field) != candidate.get(field)
    ]
    if differing:
        raise ValueError("P0 online deployment contract differs at: " + ", ".join(differing))
    return {field: candidate.get(field) for field in DEPLOYMENT_CONTRACT_FIELDS}


def _module_path(root: Path, module: str) -> Path | None:
    relative = Path(*module.split("."))
    file_path = root / relative.with_suffix(".py")
    if file_path.is_file():
        return file_path
    init_path = root / relative / "__init__.py"
    return init_path if init_path.is_file() else None


def audit_formal_import_graph(
    *, root: str | Path, entrypoint: str | Path, allowlist_path: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    entrypoint = Path(entrypoint).resolve()
    allowlist_payload = json.loads(Path(allowlist_path).read_text())
    allowed = set(allowlist_payload.get("allowed_source_files", ()))
    if not allowed:
        raise ValueError("V7 formal source allowlist is empty")
    pending = [entrypoint]
    visited: set[Path] = set()
    imported_modules: set[str] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        relative = path.relative_to(root).as_posix()
        if relative not in allowed:
            raise ValueError(f"V7 formal source is not allowlisted: {relative}")
        tree = ast.parse(path.read_text(), filename=str(path))
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.append(node.module)
        for module in modules:
            imported_modules.add(module)
            pieces = module.split(".")
            candidates = {module, pieces[-1], module.replace(".", "/")}
            if any(
                fnmatch.fnmatch(candidate, pattern)
                for pattern in FORBIDDEN_IMPORT_PATTERNS
                for candidate in candidates
            ):
                raise ValueError(f"forbidden V7 formal import: {module}")
            dependency = _module_path(root, module)
            if dependency is not None:
                pending.append(dependency)
    return {
        "entrypoint": entrypoint.relative_to(root).as_posix(),
        "source_files": sorted(path.relative_to(root).as_posix() for path in visited),
        "imported_modules": sorted(imported_modules),
        "forbidden_import_count": 0,
    }
