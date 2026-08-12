"""Fail-closed I/O helpers for the P8 cycle-verified Fisher CLIs."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid

import torch

from common.hashing import sha256_file
from evidence.cycle_verified_fisher import (
    _verified_cycle_table,
    proposal_arm_pairs,
    validate_cycle_verified_fisher_selection,
    validate_pair_match_probe,
    validate_pair_proposal_table,
)
from features.multiview_fusion import PIXEL_CENTER_OFFSET


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SCENE_CONTRACTS = {
    "stairs": {
        "mapping_keypoints": 1024,
        "nms_radius": 4,
        "pair_budget": 7450,
        "candidate_pair_count": 14835,
        "candidate_component_count": 2,
    },
    "greatcourt": {
        "mapping_keypoints": 2048,
        "nms_radius": 4,
        "pair_budget": 5254,
        "candidate_pair_count": 9875,
        "candidate_component_count": 1,
    },
}
MATCHER_CONTRACT = {
    "minimum_similarity": 0.65,
    "minimum_margin": 0.01,
    "maximum_epipolar_error_px": 2.0,
    "epipolar_candidate_topk": 1,
    "epipolar_recovered_minimum_similarity": -1.0,
    "epipolar_recovered_minimum_margin": -1.0,
}


def validate_scene_contract(
    *,
    scene: str,
    mapping_keypoints: int,
    nms_radius: int,
    pair_budget: int,
    candidate_pair_count: int,
    candidate_component_count: int,
) -> dict:
    normalized = str(scene).lower()
    if normalized not in SCENE_CONTRACTS:
        raise ValueError("P8 V1 scene must be Stairs or GreatCourt")
    observed = {
        "mapping_keypoints": int(mapping_keypoints),
        "nms_radius": int(nms_radius),
        "pair_budget": int(pair_budget),
        "candidate_pair_count": int(candidate_pair_count),
        "candidate_component_count": int(candidate_component_count),
    }
    if observed != SCENE_CONTRACTS[normalized]:
        raise ValueError(f"{normalized} axes differ from the P8 V1 preregistration")
    return {"scene": normalized, **observed}


def validate_matcher_contract(parameters: dict) -> dict:
    normalized = {
        "minimum_similarity": float(parameters["minimum_similarity"]),
        "minimum_margin": float(parameters["minimum_margin"]),
        "maximum_epipolar_error_px": float(
            parameters["maximum_epipolar_error_px"]
        ),
        "epipolar_candidate_topk": int(parameters["epipolar_candidate_topk"]),
        "epipolar_recovered_minimum_similarity": float(
            parameters["epipolar_recovered_minimum_similarity"]
        ),
        "epipolar_recovered_minimum_margin": float(
            parameters["epipolar_recovered_minimum_margin"]
        ),
    }
    if normalized != MATCHER_CONTRACT:
        raise ValueError("Matcher thresholds differ from the P8 V1 preregistration")
    return normalized


def expected_sha256(value: str, *, label: str) -> str:
    digest = str(value).strip().lower()
    if SHA256_PATTERN.fullmatch(digest) is None:
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
    expected = expected_sha256(digest, label=f"expected {label} SHA-256")
    if sha256_file(resolved) != expected:
        raise ValueError(f"{label} SHA-256 differs from the explicit contract")
    return resolved


def validate_output_target(
    output: str | Path, *, protected_paths: list[Path]
) -> Path:
    resolved = Path(output).expanduser().resolve()
    if resolved in {Path(path).resolve() for path in protected_paths}:
        raise ValueError("Output must not overwrite a frozen input artifact")
    if resolved.exists() and not resolved.is_file():
        raise ValueError("Output target exists and is not a regular file")
    return resolved


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, TypeError):
        return torch.load(path, map_location="cpu", weights_only=False)


def ordered_names_sha256(names: list[str]) -> str:
    return hashlib.sha256(
        ("\n".join(str(name) for name in names) + "\n").encode("utf-8")
    ).hexdigest()


def load_mapping_cache(
    *,
    path: Path,
    expected_file_sha256: str,
    expected_query_names_sha256: str,
    expected_mapping_keypoints: int,
    expected_nms_radius: int,
) -> dict:
    path = attest_file(
        path, expected_file_sha256, label="mapping query cache"
    )
    payload = torch_load(path)
    if not isinstance(payload, dict) or payload.get("uses_test_queries") is True:
        raise ValueError("Query cache is invalid or attests test-query use")
    records = payload.get("queries", payload)
    if not isinstance(records, dict) or len(records) < 2:
        raise ValueError("Query cache lacks an ordered mapping registry")
    names = [str(name) for name in records]
    expected_names = expected_sha256(
        expected_query_names_sha256, label="expected query-name SHA-256"
    )
    if ordered_names_sha256(names) != expected_names:
        raise ValueError("Query-cache order differs from the expected mapping registry")
    mapping_k = int(expected_mapping_keypoints)
    nms_radius = int(expected_nms_radius)
    if mapping_k <= 0 or nms_radius <= 0:
        raise ValueError("Expected mapping K/NMS must be positive")
    signature = payload.get("signature_payload")
    if not isinstance(signature, dict):
        raise ValueError("Query cache lacks a signed sparse-frontend contract")
    if (
        int(signature.get("native_sparse_keypoint_count", -1)) != mapping_k
        or int(signature.get("native_sparse_nms_radius", -1)) != nms_radius
    ):
        raise ValueError("Query-cache signature differs from the expected K/NMS")

    descriptors: list[torch.Tensor] = []
    keypoints: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []
    intrinsics: list[torch.Tensor] = []
    poses: list[torch.Tensor] = []
    for name in names:
        record = records[name]
        if not isinstance(record, dict):
            raise ValueError(f"Mapping query {name} is not a cache record")
        metadata = record.get("native_sparse_metadata")
        if not isinstance(metadata, dict) or (
            int(metadata.get("nms_radius", -1)) != nms_radius
            or int(
                metadata.get(
                    "requested_keypoint_count", metadata.get("detect_num", -1)
                )
            )
            != mapping_k
        ):
            raise ValueError(f"Mapping query {name} does not attest K/NMS")
        descriptor = torch.as_tensor(record.get("native_descriptors")).float()
        keypoint = torch.as_tensor(record.get("native_keypoints")).float()
        score = torch.as_tensor(record.get("native_scores")).float().reshape(-1)
        camera_K = torch.as_tensor(record.get("native_K")).float()
        pose_w2c = torch.as_tensor(record.get("pose_w2c")).float()
        if (
            descriptor.ndim != 2
            or keypoint.ndim != 2
            or keypoint.shape[1] != 2
            or descriptor.shape[0] != keypoint.shape[0]
            or score.numel() != keypoint.shape[0]
            or keypoint.shape[0] > mapping_k
            or camera_K.shape != (3, 3)
            or pose_w2c.shape not in {(3, 4), (4, 4)}
        ):
            raise ValueError(f"Mapping query {name} has misaligned camera tables")
        if not all(
            bool(torch.isfinite(value).all())
            for value in (descriptor, keypoint, score, camera_K, pose_w2c)
        ):
            raise ValueError(f"Mapping query {name} contains non-finite values")
        descriptors.append(descriptor)
        keypoints.append(keypoint + float(PIXEL_CENTER_OFFSET))
        scores.append(score)
        intrinsics.append(camera_K)
        poses.append(pose_w2c)
    return {
        "path": path,
        "sha256": sha256_file(path),
        "payload": payload,
        "names": names,
        "query_names_sha256": expected_names,
        "descriptors": descriptors,
        "keypoints": keypoints,
        "scores": scores,
        "camera_K": torch.stack(intrinsics),
        "pose_w2c": torch.stack(poses),
    }


def _pair_table(factor: dict) -> list[tuple[int, int]]:
    sidecar = factor.get("pair_sidecar")
    if not isinstance(sidecar, dict):
        raise ValueError("Track factor lacks its pair sidecar")
    pair = sidecar.get("pair")
    if not isinstance(pair, dict):
        raise ValueError("Track factor lacks its exact pair table")
    left = torch.as_tensor(pair.get("left_query_index"), dtype=torch.long).reshape(-1)
    right = torch.as_tensor(pair.get("right_query_index"), dtype=torch.long).reshape(-1)
    if left.numel() == 0 or left.numel() != right.numel():
        raise ValueError("Track-factor pair table is empty or misaligned")
    pairs = list(zip(left.tolist(), right.tolist()))
    if pairs != sorted(set(pairs)):
        raise ValueError("Track-factor pairs must be unique and sorted")
    return pairs


def load_track_factor(
    *,
    path: Path,
    expected_file_sha256: str,
    expected_policy: str,
    expected_query_names: list[str],
    expected_query_names_sha256: str,
    expected_query_cache_path: Path,
    expected_query_cache_sha256: str,
    expected_mapping_keypoints: int,
    expected_nms_radius: int,
    expected_pair_budget: int,
) -> dict:
    path = attest_file(path, expected_file_sha256, label=f"{expected_policy} factor")
    factor = torch_load(path)
    if (
        not isinstance(factor, dict)
        or factor.get("schema") != "lafgs_pair_policy_track_factor"
        or int(factor.get("version", -1)) != 1
        or factor.get("uses_test_queries") is not False
        or factor.get("pair_policy") != expected_policy
        or int(factor.get("mapping_keypoint_factor", -1))
        != int(expected_mapping_keypoints)
        or int(factor.get("mapping_nms_radius", -1)) != int(expected_nms_radius)
        or [str(value) for value in factor.get("query_names", [])]
        != expected_query_names
        or factor.get("query_names_sha256") != expected_query_names_sha256
    ):
        raise ValueError(f"{expected_policy} factor differs from the frozen contract")
    pairs = _pair_table(factor)
    if len(pairs) != int(expected_pair_budget):
        raise ValueError(f"{expected_policy} factor violates the exact pair budget")
    query_count = len(expected_query_names)
    if any(left < 0 or left >= right or right >= query_count for left, right in pairs):
        raise ValueError(f"{expected_policy} factor pair index is out of range")
    sidecar_policy = factor["pair_sidecar"].get("policy", {})
    if sidecar_policy.get("uses_test_queries") is not False:
        raise ValueError(f"{expected_policy} pair sidecar is not mapping-only")
    lineage = factor.get("input_lineage")
    if not isinstance(lineage, dict):
        raise ValueError(f"{expected_policy} factor lacks input lineage")
    cache = lineage.get("query_cache")
    if (
        not isinstance(cache, dict)
        or Path(str(cache.get("path", ""))).resolve()
        != expected_query_cache_path.resolve()
        or cache.get("sha256") != expected_query_cache_sha256
    ):
        raise ValueError(f"{expected_policy} factor names a different query cache")
    return {
        "path": path,
        "sha256": sha256_file(path),
        "payload": factor,
        "pairs": pairs,
    }


def load_probe(
    *,
    path: Path,
    expected_file_sha256: str,
    expected_content_sha256: str,
    cache: dict,
    expected_mapping_keypoints: int,
    expected_nms_radius: int,
    expected_candidate_pair_count: int,
) -> dict:
    path = attest_file(path, expected_file_sha256, label="pair-match probe")
    payload = torch_load(path)
    content_sha = expected_sha256(
        expected_content_sha256, label="expected probe content SHA-256"
    )
    validate_pair_match_probe(
        payload,
        expected_query_names_sha256=cache["query_names_sha256"],
        expected_query_cache_sha256=cache["sha256"],
        expected_mapping_keypoint_count=int(expected_mapping_keypoints),
        expected_mapping_nms_radius=int(expected_nms_radius),
        expected_content_sha256=content_sha,
    )
    if int(payload["candidate_pool"]["left_query_index"].numel()) != int(
        expected_candidate_pair_count
    ):
        raise ValueError("Probe candidate count differs from the explicit contract")
    observed_counts = torch.as_tensor(
        [int(value.shape[0]) for value in cache["keypoints"]], dtype=torch.long
    )
    if not torch.equal(observed_counts, payload["keypoint_counts"].long()):
        raise ValueError("Probe keypoint rows differ from the exact query cache")
    return {
        "path": path,
        "sha256": sha256_file(path),
        "content_sha256": content_sha,
        "payload": payload,
    }


def load_proposals(
    *,
    path: Path,
    expected_file_sha256: str,
    expected_content_sha256: str,
    cache: dict,
    expected_mapping_keypoints: int,
    expected_nms_radius: int,
    expected_pair_budget: int,
    expected_candidate_pair_count: int,
    expected_candidate_components: int,
) -> dict:
    path = attest_file(path, expected_file_sha256, label="pair-proposal table")
    payload = torch_load(path)
    content_sha = expected_sha256(
        expected_content_sha256, label="expected proposal content SHA-256"
    )
    validate_pair_proposal_table(
        payload,
        expected_query_names_sha256=cache["query_names_sha256"],
        expected_query_cache_path=str(cache["path"]),
        expected_query_cache_sha256=cache["sha256"],
        expected_mapping_keypoint_count=int(expected_mapping_keypoints),
        expected_mapping_nms_radius=int(expected_nms_radius),
        expected_pair_budget=int(expected_pair_budget),
        expected_candidate_pair_count=int(expected_candidate_pair_count),
        expected_candidate_component_count=int(expected_candidate_components),
        expected_content_sha256=content_sha,
    )
    return {
        "path": path,
        "sha256": sha256_file(path),
        "content_sha256": content_sha,
        "payload": payload,
        "nearest_pairs": proposal_arm_pairs(payload, "nearest"),
        "geometry_pairs": proposal_arm_pairs(payload, "mapping_geometry"),
    }


def validate_probe_proposal_lineage(*, probe: dict, proposals: dict) -> None:
    candidate = probe["payload"]["candidate_pool"]
    parameters = candidate.get("parameters")
    if (
        candidate.get("construction")
        != "attested_nearest_union_mapping_geometry_v1"
        or not isinstance(parameters, dict)
        or parameters.get("proposal_table_sha256") != proposals["sha256"]
        or parameters.get("proposal_table_content_sha256")
        != proposals["content_sha256"]
    ):
        raise ValueError("Pair-match probe does not bind the proposal table")
    expected_pairs = sorted(
        set(proposals["nearest_pairs"]) | set(proposals["geometry_pairs"])
    )
    observed_pairs = list(
        zip(
            candidate["left_query_index"].long().tolist(),
            candidate["right_query_index"].long().tolist(),
        )
    )
    if observed_pairs != expected_pairs:
        raise ValueError("Pair-match probe candidate table differs from proposals")


def load_selection(
    *,
    path: Path,
    expected_file_sha256: str,
    expected_content_sha256: str,
    probe: dict,
    expected_pair_budget: int,
) -> dict:
    path = attest_file(path, expected_file_sha256, label="pair selection")
    payload = torch_load(path)
    content_sha = expected_sha256(
        expected_content_sha256, label="expected selection content SHA-256"
    )
    validate_cycle_verified_fisher_selection(
        payload,
        pair_match_probe=probe["payload"],
        expected_content_sha256=content_sha,
    )
    if int(payload.get("exact_pair_budget", -1)) != int(expected_pair_budget):
        raise ValueError("Selection differs from the exact pair budget")
    return {
        "path": path,
        "sha256": sha256_file(path),
        "content_sha256": content_sha,
        "payload": payload,
    }


def load_stage_a_gate(
    *,
    path: Path,
    expected_file_sha256: str,
    cache: dict,
    proposals: dict,
    probe: dict,
    selection: dict,
    require_go: bool,
) -> dict:
    path = attest_file(path, expected_file_sha256, label="P8 Stage-A gate")
    payload = json.loads(path.read_text())
    if (
        payload.get("schema") != "lafgs_cycle_verified_fisher_stage_a_gate"
        or int(payload.get("version", -1)) != 1
        or payload.get("uses_test_queries") is not False
        or payload.get("mapping_only") is not True
        or payload.get("valid") is not True
        or not isinstance(payload.get("gates"), dict)
        or not all(isinstance(value, bool) for value in payload["gates"].values())
        or bool(payload.get("stage_a_passed")) != all(payload["gates"].values())
        or bool(payload.get("advance_to_reuse_only_track_build"))
        != bool(payload.get("stage_a_passed"))
    ):
        raise ValueError("P8 Stage-A gate is invalid or internally inconsistent")
    expected = {
        "query_cache": cache,
        "pair_proposals": proposals,
        "pair_match_probe": probe,
        "pair_selection": selection,
    }
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(expected):
        raise ValueError("P8 Stage-A gate has an unexpected input registry")
    for name, artifact in expected.items():
        entry = inputs[name]
        if (
            not isinstance(entry, dict)
            or Path(str(entry.get("path", ""))).resolve() != artifact["path"]
            or entry.get("sha256") != artifact["sha256"]
            or (
                "content_sha256" in artifact
                and entry.get("content_sha256") != artifact["content_sha256"]
            )
        ):
            raise ValueError(f"P8 Stage-A gate {name} lineage differs")
    passed = payload.get("stage_a_passed") is True
    if require_go and not passed:
        raise ValueError("P8 Stage-A STOP does not authorize Track construction")
    if passed and payload.get("decision") != "GO_TO_TRACK_REUSE":
        raise ValueError("P8 Stage-A GO decision is inconsistent")
    if not passed and payload.get("decision") != "STOP_BEFORE_TRACK_REUSE":
        raise ValueError("P8 Stage-A STOP decision is inconsistent")
    return {"path": path, "sha256": sha256_file(path), "payload": payload}


def selection_pairs(selection: dict) -> list[tuple[int, int]]:
    selected = selection["selected_pair"]
    return list(
        zip(
            torch.as_tensor(selected["left_query_index"]).long().tolist(),
            torch.as_tensor(selected["right_query_index"]).long().tolist(),
        )
    )


def evaluate_pair_subsets(
    *,
    probe: dict,
    cache: dict,
    subsets: dict[str, list[tuple[int, int]]],
    maximum_reprojection_error_px: float,
) -> dict[str, dict]:
    candidate = probe["candidate_pool"]
    candidate_pairs = list(
        zip(
            candidate["left_query_index"].long().tolist(),
            candidate["right_query_index"].long().tolist(),
        )
    )
    candidate_index = {pair: index for index, pair in enumerate(candidate_pairs)}
    from evidence.cycle_verified_fisher import pair_matches_from_probe

    matches, _ = pair_matches_from_probe(probe)
    triangle = _verified_cycle_table(
        pairs=candidate_pairs,
        pair_matches=matches,
        keypoints=cache["keypoints"],
        camera_K=cache["camera_K"],
        pose_w2c=cache["pose_w2c"],
        maximum_reprojection_error_px=float(maximum_reprojection_error_px),
    )
    result = {}
    for name, pairs in subsets.items():
        if pairs != sorted(set(pairs)) or any(
            pair not in candidate_index for pair in pairs
        ):
            raise ValueError("Evaluated pair subset is not an exact probe subset")
        selected_index = {candidate_index[pair] for pair in pairs}
        completed = torch.as_tensor(
            [
                all(int(edge) in selected_index for edge in edges.tolist())
                for edges in triangle["pair_index"]
            ],
            dtype=torch.bool,
        )
        camera_mask = torch.zeros(int(probe["query_count"]), dtype=torch.bool)
        if bool(completed.any()):
            camera_mask[triangle["camera_index"][completed].reshape(-1)] = True
        result[name] = {
            "completed_verified_keypoint_triangle_count": int(completed.sum()),
            "completed_verified_triangle_camera_count": int(camera_mask.sum()),
            "completed_verified_triangle_camera_fraction": float(
                camera_mask.double().mean()
            ),
            "confidence_weighted_fisher_utility_sum": float(
                triangle["utility"][completed].sum()
            ),
            "fisher_logdet_gain_sum": float(
                triangle["fisher_logdet_gain"][completed].sum()
            ),
            "candidate_verified_keypoint_triangle_count": int(
                triangle["pair_index"].shape[0]
            ),
        }
    return result


def evaluate_pair_subset(
    *,
    probe: dict,
    cache: dict,
    pairs: list[tuple[int, int]],
    maximum_reprojection_error_px: float,
) -> dict:
    return evaluate_pair_subsets(
        probe=probe,
        cache=cache,
        subsets={"selection": pairs},
        maximum_reprojection_error_px=maximum_reprojection_error_px,
    )["selection"]


def assert_selection_metrics(selection: dict, evaluation: dict) -> None:
    triangle = selection["verified_triangle"]
    exact = {
        "candidate_count": evaluation["candidate_verified_keypoint_triangle_count"],
        "selected_completed_count": evaluation[
            "completed_verified_keypoint_triangle_count"
        ],
        "selected_camera_count": evaluation[
            "completed_verified_triangle_camera_count"
        ],
    }
    for name, expected in exact.items():
        if int(triangle.get(name, -1)) != int(expected):
            raise ValueError(f"Selection {name} differs from same-probe replay")
    floating = {
        "selected_camera_fraction": evaluation[
            "completed_verified_triangle_camera_fraction"
        ],
        "selected_fisher_logdet_gain_sum": evaluation["fisher_logdet_gain_sum"],
        "selected_confidence_weighted_utility_sum": evaluation[
            "confidence_weighted_fisher_utility_sum"
        ],
    }
    for name, expected in floating.items():
        actual = float(triangle.get(name, math.nan))
        if not math.isclose(actual, float(expected), rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"Selection {name} differs from same-probe replay")


def atomic_torch_save(payload: dict, output: Path, *, overwrite: bool) -> Path:
    output = Path(output).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f".{output.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def atomic_json_save(payload: dict, output: Path, *, overwrite: bool) -> Path:
    output = Path(output).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f".{output.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def validate_variant_reuse_lineage(
    *,
    factor: dict,
    probe: dict,
    selection: dict,
) -> None:
    diagnostics = factor.get("diagnostics")
    if not isinstance(diagnostics, dict) or int(
        diagnostics.get("track_pair_matches_reused", -1)
    ) != 1:
        raise ValueError("Variant Track factor does not attest probe-match reuse")
    sidecar_policy = factor.get("pair_sidecar", {}).get("policy", {})
    if sidecar_policy.get("uses_precomputed_pair_matches") is not True:
        raise ValueError("Variant Track sidecar does not attest precomputed matches")
    lineage = factor.get("input_lineage", {})
    expected = {
        "pair_match_probe": probe,
        "pair_selection": selection,
    }
    for name, artifact in expected.items():
        entry = lineage.get(name)
        if (
            not isinstance(entry, dict)
            or Path(str(entry.get("path", ""))).resolve() != artifact["path"]
            or entry.get("sha256") != artifact["sha256"]
            or entry.get("content_sha256") != artifact["content_sha256"]
        ):
            raise ValueError(f"Variant Track factor {name} lineage differs")
    if _pair_table(factor) != selection_pairs(selection["payload"]):
        raise ValueError("Variant Track pairs differ from the attested selection")
