"""Fail-closed contracts shared by P8 coverage-V2 Track and Stage-B CLIs."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import torch

from common.hashing import sha256_file
from evidence.cycle_verified_fisher import COVERAGE_POLICY_NAME
from scripts.cycle_verified_fisher_cli_common import (
    MATCHER_CONTRACT,
    SCENE_CONTRACTS,
    attest_file,
    load_coverage_selection,
    load_mapping_cache,
    load_probe,
    load_proposals,
    load_track_factor,
    load_verified_cycle_table,
    selection_pairs,
    validate_probe_proposal_lineage,
    validate_v2_frozen_source_contract,
)
from scripts.run_track_pair_factor import _track_report


PREREGISTRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/evidence/"
    "p8_cycle_verified_fisher_coverage_v2_stage_b_preregistration.json"
)
PREREGISTRATION_COMMIT = "9384986df4c1d22ea5aeac71b5caa460f9785589"
PREREGISTRATION_BLOB_SHA256 = (
    "eab20cb3fac8440eb2b22b4c4c88a2e1efb911aebeefe06fbd1e41cf26f8a54e"
)
IMPLEMENTATION_REGISTRY_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/evidence/"
    "p8_cycle_verified_fisher_coverage_v2_stage_b_implementation.json"
)
CONTROL_POLICY_NAME = "cycle_verified_fisher_coverage_nearest_control"
VARIANT_POLICY_NAME = COVERAGE_POLICY_NAME
CONTROL_SUBSET_ROLE = "attested_nearest_same_probe_control"
VARIANT_SUBSET_ROLE = "cycle_verified_fisher_coverage_selection"
COMPLETION_SCHEMA = "lafgs_cycle_verified_fisher_coverage_paired_track_completion"
TRACK_PRODUCER_SCHEMA = "lafgs_cycle_verified_fisher_coverage_track_producer"
STAGE_B_PRODUCER_SCHEMA = "lafgs_cycle_verified_fisher_coverage_stage_b_producer"
CROSS_B_PRODUCER_SCHEMA = (
    "lafgs_cycle_verified_fisher_coverage_cross_scene_stage_b_producer"
)
SHA256 = re.compile(r"[0-9a-f]{64}")

TRACK_PRODUCER_SOURCE_PATHS = (
    "common/hashing.py",
    "evidence/camera_pair_policy.py",
    "evidence/cycle_verified_fisher.py",
    "evidence/tracks.py",
    "evidence/triangulation.py",
    "features/multiview_fusion.py",
    "scripts/cycle_verified_fisher_cli_common.py",
    "scripts/cycle_verified_fisher_coverage_track_common.py",
    "scripts/materialize_cycle_verified_fisher_coverage_track_factor.py",
    "scripts/run_track_pair_factor.py",
    "topology/track_core.py",
    "docs/evidence/p8_cycle_verified_fisher_coverage_v2_stage_b_preregistration.json",
)
STAGE_B_PRODUCER_SOURCE_PATHS = (
    "common/hashing.py",
    "evidence/camera_pair_policy.py",
    "evidence/cycle_verified_fisher.py",
    "evidence/tracks.py",
    "evidence/triangulation.py",
    "features/multiview_fusion.py",
    "scripts/cycle_verified_fisher_cli_common.py",
    "scripts/cycle_verified_fisher_coverage_track_common.py",
    "scripts/compare_cycle_verified_fisher_coverage_mechanism.py",
    "scripts/run_track_pair_factor.py",
    "topology/track_core.py",
    "docs/evidence/p8_cycle_verified_fisher_coverage_v2_stage_b_preregistration.json",
)
CROSS_B_PRODUCER_SOURCE_PATHS = (
    *STAGE_B_PRODUCER_SOURCE_PATHS,
    "scripts/aggregate_cycle_verified_fisher_coverage_cross_scene.py",
)


def preregistration() -> dict:
    payload = json.loads(PREREGISTRATION_PATH.read_text())
    prereg_source = payload.get("producer_identity", {}).get("required_source_paths")
    expected_pass_contract = {
        "greatcourt": "all(base_gates_8)",
        "stairs": (
            "all(base_gates_8) and "
            "v1_nearest_control_scientific_projection_exact and "
            "all(stairs_v1_retention_gates_5)"
        ),
    }
    exit_contract = payload.get("pass_and_exit_contract", {})
    if (
        payload.get("schema")
        != "lafgs_cycle_verified_fisher_coverage_stage_b_preregistration"
        or payload.get("version") != 1
        or payload.get("valid") is not True
        or payload.get("uses_test_queries") is not False
        or payload.get("mapping_only") is not True
        or payload.get("policy")
        != {"control": CONTROL_POLICY_NAME, "variant": VARIANT_POLICY_NAME}
        or prereg_source != list(TRACK_PRODUCER_SOURCE_PATHS)
        or not set(
            payload.get("stage_b_producer_identity", {}).get(
                "required_source_paths_include", []
            )
        ).issubset(STAGE_B_PRODUCER_SOURCE_PATHS)
        or not set(
            payload.get("cross_scene_stage_b_producer_identity", {}).get(
                "required_source_paths_include", []
            )
        ).issubset(CROSS_B_PRODUCER_SOURCE_PATHS)
        or exit_contract.get("scene_pass_formulas") != expected_pass_contract
        or exit_contract.get("cross_pass_formula")
        != (
            "stairs_scene_specific_mechanism_pass and "
            "greatcourt_scene_specific_mechanism_pass and "
            "same_track_producer_identity and same_stage_b_producer_identity "
            "and same_compiled_identity"
        )
        or exit_contract.get("lineage_or_input_invalid", {}).get("exit_code") != 1
        or exit_contract.get("lineage_or_input_invalid", {}).get("writes_gate")
        is not False
        or exit_contract.get("scientific_gate_failure", {}).get("exit_code") != 2
        or exit_contract.get("scientific_gate_failure", {}).get(
            "writes_valid_stop_gate"
        )
        is not True
    ):
        raise RuntimeError("P8 coverage-V2 Stage-B preregistration is invalid")
    return payload


def implementation_registry() -> dict:
    """Load the post-review implementation boundary required for real Track."""
    if not IMPLEMENTATION_REGISTRY_PATH.is_file():
        raise RuntimeError(
            "Reviewed P8 coverage-V2 implementation registry is not committed"
        )
    payload = json.loads(IMPLEMENTATION_REGISTRY_PATH.read_text())
    required_source_paths = sorted(
        set(TRACK_PRODUCER_SOURCE_PATHS)
        | set(STAGE_B_PRODUCER_SOURCE_PATHS)
        | set(CROSS_B_PRODUCER_SOURCE_PATHS)
    )
    if (
        payload.get("schema")
        != "lafgs_cycle_verified_fisher_coverage_stage_b_implementation_registry"
        or payload.get("version") != 1
        or payload.get("valid") is not True
        or payload.get("uses_test_queries") is not False
        or payload.get("mapping_only") is not True
        or payload.get("preregistration", {}).get("path")
        != str(PREREGISTRATION_PATH.relative_to(PREREGISTRATION_PATH.parents[2]))
        or payload.get("preregistration", {}).get("commit")
        != PREREGISTRATION_COMMIT
        or payload.get("preregistration", {}).get("blob_sha256")
        != PREREGISTRATION_BLOB_SHA256
        or sha256_file(PREREGISTRATION_PATH) != PREREGISTRATION_BLOB_SHA256
        or re.fullmatch(
            r"[0-9a-f]{40}", str(payload.get("implementation_commit", ""))
        )
        is None
        or payload.get("required_source_paths") != required_source_paths
        or payload.get("source_file_sha256")
        != {
            name: _file_sha256(Path(__file__).resolve().parents[1] / name)
            for name in required_source_paths
        }
        or payload.get("full_cpu_tests", {}).get("passed") is not True
        or payload.get("independent_review", {}).get("passed") is not True
        or payload.get("authorizes_real_track_execution") is not True
        or payload.get("authorizes_test") is not False
        or payload.get("authorizes_method_default_change") is not False
    ):
        raise RuntimeError("P8 reviewed implementation registry is invalid or stale")
    root = Path(__file__).resolve().parents[1]
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            payload["implementation_commit"],
            current_commit,
        ],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("Reviewed P8 implementation commit is not in current history")
    for name, digest in payload["source_file_sha256"].items():
        committed = hashlib.sha256(
            subprocess.run(
                ["git", "show", f"{payload['implementation_commit']}:{name}"],
                cwd=root,
                check=True,
                capture_output=True,
            ).stdout
        ).hexdigest()
        if committed != digest:
            raise RuntimeError("P8 implementation commit/source registry differs")
    return payload


def scene_preregistration(scene: str) -> dict:
    scene = str(scene).lower()
    payload = preregistration().get("fixed_scene_contracts", {}).get(scene)
    if not isinstance(payload, dict) or scene not in SCENE_CONTRACTS:
        raise ValueError("P8 coverage-V2 scene must be stairs or greatcourt")
    expected = SCENE_CONTRACTS[scene]
    observed = {
        "mapping_keypoints": payload.get("mapping_keypoints"),
        "nms_radius": payload.get("mapping_nms_radius"),
        "pair_budget": payload.get("exact_pair_budget"),
        "candidate_pair_count": payload.get("candidate_pair_count"),
        "candidate_component_count": payload.get("candidate_component_count"),
    }
    if observed != expected:
        raise RuntimeError(f"Compiled {scene} P8 scene contract is inconsistent")
    return deepcopy(payload)


def _sha256(value: object, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if SHA256.fullmatch(normalized) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    return normalized


def pair_table_sha256(pairs: list[tuple[int, int]]) -> str:
    encoded = json.dumps(
        [[int(left), int(right)] for left, right in pairs], separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _compiled_reference(scene: str, name: str) -> dict:
    value = scene_preregistration(scene).get(name)
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("path"), str)
        or SHA256.fullmatch(str(value.get("sha256", ""))) is None
    ):
        raise RuntimeError(f"Compiled {scene}/{name} reference is invalid")
    return deepcopy(value)


def require_compiled_reference(
    *,
    scene: str,
    name: str,
    path: Path,
    expected_sha256: str,
    expected_content_sha256: str | None = None,
) -> dict:
    compiled = _compiled_reference(scene, name)
    if (
        Path(path).expanduser().resolve()
        != Path(compiled["path"]).expanduser().resolve()
        or _sha256(expected_sha256, label=f"expected {name} SHA-256")
        != compiled["sha256"]
    ):
        raise ValueError(f"{scene}/{name} differs from the compiled preregistration")
    if expected_content_sha256 is not None:
        expected_content = _sha256(
            expected_content_sha256, label=f"expected {name} content SHA-256"
        )
        if expected_content != compiled.get("content_sha256"):
            raise ValueError(
                f"{scene}/{name} content differs from the compiled preregistration"
            )
    return compiled


def _same_reference(observed: object, expected: dict) -> bool:
    if not isinstance(observed, dict):
        return False
    return (
        Path(str(observed.get("path", ""))).expanduser().resolve()
        == Path(str(expected["path"])).expanduser().resolve()
        and observed.get("sha256") == expected.get("sha256")
        and (
            "content_sha256" not in expected
            or observed.get("content_sha256") == expected["content_sha256"]
        )
    )


def load_cross_scene_authority(
    *, path: Path, expected_sha256: str, scene: str
) -> dict:
    compiled = preregistration()["authorization"]["cross_scene_stage_a_gate"]
    resolved = Path(path).expanduser().resolve()
    expected = _sha256(expected_sha256, label="expected cross-scene Stage-A SHA-256")
    if (
        resolved != Path(compiled["path"]).resolve()
        or expected != compiled["sha256"]
    ):
        raise ValueError("Cross-scene Stage-A authority is not the compiled GO")
    resolved = attest_file(resolved, expected, label="cross-scene V2 Stage-A gate")
    payload = json.loads(resolved.read_text())
    inputs = payload.get("inputs")
    if (
        payload.get("schema") != compiled["schema"]
        or payload.get("version") != compiled["version"]
        or payload.get("uses_test_queries") is not False
        or payload.get("mapping_only") is not True
        or payload.get("valid") is not True
        or payload.get("policy") != VARIANT_POLICY_NAME
        or payload.get("both_scene_stage_a_passed") is not True
        or payload.get("advance_to_v2_aware_reuse_only_track_build") is not True
        or payload.get("authorizes_existing_v1_track_runner") is not False
        or payload.get("decision") != compiled["decision"]
        or not isinstance(inputs, dict)
        or set(inputs) != {"stairs", "greatcourt"}
    ):
        raise ValueError("Cross-scene Stage-A gate does not authorize V2-aware Track")
    for registered_scene in ("stairs", "greatcourt"):
        expected_scene = _compiled_reference(
            registered_scene, "scene_stage_a_gate"
        )
        if not _same_reference(inputs[registered_scene], expected_scene):
            raise ValueError("Cross-scene Stage-A input registry changed")
        attest_file(
            Path(expected_scene["path"]),
            expected_scene["sha256"],
            label=f"{registered_scene} V2 Stage-A gate",
        )
    if scene not in inputs:
        raise ValueError("Cross-scene authority does not cover this scene")
    return {"path": resolved, "sha256": sha256_file(resolved), "payload": payload}


def load_scene_stage_a_gate(
    *, scene: str, path: Path, expected_sha256: str, authority: dict
) -> dict:
    compiled = require_compiled_reference(
        scene=scene,
        name="scene_stage_a_gate",
        path=path,
        expected_sha256=expected_sha256,
    )
    resolved = attest_file(path, compiled["sha256"], label=f"{scene} V2 Stage-A gate")
    if not _same_reference(authority["payload"]["inputs"].get(scene), compiled):
        raise ValueError("Scene Stage-A gate is not the cross-scene authorized input")
    payload = json.loads(resolved.read_text())
    gates = payload.get("gates")
    contract = {"scene": scene, **SCENE_CONTRACTS[scene]}
    if (
        payload.get("schema")
        != "lafgs_cycle_verified_fisher_coverage_stage_a_gate"
        or payload.get("version") != 1
        or payload.get("uses_test_queries") is not False
        or payload.get("mapping_only") is not True
        or payload.get("valid") is not True
        or payload.get("policy") != VARIANT_POLICY_NAME
        or payload.get("scene_contract") != contract
        or not isinstance(gates, dict)
        or not gates
        or not all(value is True for value in gates.values())
        or payload.get("stage_a_passed") is not True
        or payload.get("requires_other_scene") is not True
        or payload.get("requires_v2_aware_track_lineage_implementation") is not True
        or payload.get("advance_to_reuse_only_track_build") is not False
        or payload.get("decision") != "SCENE_STAGE_A_PASS_REQUIRES_OTHER_SCENE"
    ):
        raise ValueError(f"{scene} V2 Stage-A gate is not a valid scene Pass")
    inputs = payload.get("inputs")
    required = {
        "query_cache",
        "pair_proposals",
        "pair_match_probe",
        "verified_cycle_table",
        "pair_selection",
    }
    if scene == "stairs":
        required.add("stairs_v1_pair_selection")
    if not isinstance(inputs, dict) or set(inputs) != required:
        raise ValueError(f"{scene} V2 Stage-A gate input registry is incomplete")
    compiled_names = {
        "query_cache": "query_cache",
        "pair_proposals": "pair_proposals",
        "pair_match_probe": "pair_match_probe",
        "verified_cycle_table": "verified_cycle_table",
        "pair_selection": "pair_selection",
    }
    for input_name, prereg_name in compiled_names.items():
        expected_reference = _compiled_reference(scene, prereg_name)
        if not _same_reference(inputs[input_name], expected_reference):
            raise ValueError(f"{scene} Stage-A {input_name} differs from preregistration")
    for name, reference in inputs.items():
        if not isinstance(reference, dict):
            raise ValueError(f"{scene} Stage-A {name} reference is invalid")
        attest_file(
            Path(str(reference.get("path", ""))),
            str(reference.get("sha256", "")),
            label=f"{scene} Stage-A {name}",
        )
    return {"path": resolved, "sha256": sha256_file(resolved), "payload": payload}


def load_scene_inputs(
    *,
    scene: str,
    cross_scene_stage_a_gate: Path,
    expected_cross_scene_stage_a_gate_sha256: str,
    scene_stage_a_gate: Path,
    expected_scene_stage_a_gate_sha256: str,
    manifest: Path,
    expected_manifest_sha256: str,
    frozen_track_payload: Path,
    expected_frozen_track_payload_sha256: str,
    query_cache: Path,
    expected_query_cache_sha256: str,
    mapping_scope_equivalence: Path,
    expected_mapping_scope_equivalence_sha256: str,
    proposals: Path,
    expected_proposals_sha256: str,
    expected_proposals_content_sha256: str,
    probe: Path,
    expected_probe_sha256: str,
    expected_probe_content_sha256: str,
    verified_cycle_table: Path,
    expected_verified_cycle_table_sha256: str,
    expected_verified_cycle_table_content_sha256: str,
    selection: Path,
    expected_selection_sha256: str,
    expected_selection_content_sha256: str,
    expected_query_names_sha256: str,
    expected_mapping_keypoints: int,
    expected_nms_radius: int,
    expected_pair_budget: int,
    expected_candidate_pair_count: int,
    expected_candidate_components: int,
) -> dict:
    scene = str(scene).lower()
    compiled = scene_preregistration(scene)
    observed_axes = {
        "query_names_sha256": str(expected_query_names_sha256),
        "mapping_keypoints": int(expected_mapping_keypoints),
        "mapping_nms_radius": int(expected_nms_radius),
        "exact_pair_budget": int(expected_pair_budget),
        "candidate_pair_count": int(expected_candidate_pair_count),
        "candidate_component_count": int(expected_candidate_components),
    }
    expected_axes = {name: compiled[name] for name in observed_axes}
    if observed_axes != expected_axes:
        raise ValueError(f"{scene} CLI axes differ from the compiled preregistration")
    references = {
        "bootstrap_manifest": (
            manifest,
            expected_manifest_sha256,
            None,
        ),
        "frozen_track_payload": (
            frozen_track_payload,
            expected_frozen_track_payload_sha256,
            None,
        ),
        "query_cache": (query_cache, expected_query_cache_sha256, None),
        "mapping_scope_equivalence": (
            mapping_scope_equivalence,
            expected_mapping_scope_equivalence_sha256,
            None,
        ),
        "pair_proposals": (
            proposals,
            expected_proposals_sha256,
            expected_proposals_content_sha256,
        ),
        "pair_match_probe": (
            probe,
            expected_probe_sha256,
            expected_probe_content_sha256,
        ),
        "verified_cycle_table": (
            verified_cycle_table,
            expected_verified_cycle_table_sha256,
            expected_verified_cycle_table_content_sha256,
        ),
        "pair_selection": (
            selection,
            expected_selection_sha256,
            expected_selection_content_sha256,
        ),
    }
    for name, (path, file_sha, content_sha) in references.items():
        require_compiled_reference(
            scene=scene,
            name=name,
            path=path,
            expected_sha256=file_sha,
            expected_content_sha256=content_sha,
        )
    authority = load_cross_scene_authority(
        path=cross_scene_stage_a_gate,
        expected_sha256=expected_cross_scene_stage_a_gate_sha256,
        scene=scene,
    )
    stage_a = load_scene_stage_a_gate(
        scene=scene,
        path=scene_stage_a_gate,
        expected_sha256=expected_scene_stage_a_gate_sha256,
        authority=authority,
    )
    cache = load_mapping_cache(
        path=query_cache,
        expected_file_sha256=compiled["query_cache"]["sha256"],
        expected_query_names_sha256=compiled["query_names_sha256"],
        expected_mapping_keypoints=compiled["mapping_keypoints"],
        expected_nms_radius=compiled["mapping_nms_radius"],
        mapping_scope_equivalence=mapping_scope_equivalence,
        expected_mapping_scope_equivalence_sha256=(
            compiled["mapping_scope_equivalence"]["sha256"]
        ),
    )
    proposal_record = load_proposals(
        path=proposals,
        expected_file_sha256=compiled["pair_proposals"]["sha256"],
        expected_content_sha256=compiled["pair_proposals"]["content_sha256"],
        cache=cache,
        expected_mapping_keypoints=compiled["mapping_keypoints"],
        expected_nms_radius=compiled["mapping_nms_radius"],
        expected_pair_budget=compiled["exact_pair_budget"],
        expected_candidate_pair_count=compiled["candidate_pair_count"],
        expected_candidate_components=compiled["candidate_component_count"],
    )
    probe_record = load_probe(
        path=probe,
        expected_file_sha256=compiled["pair_match_probe"]["sha256"],
        expected_content_sha256=compiled["pair_match_probe"]["content_sha256"],
        cache=cache,
        expected_mapping_keypoints=compiled["mapping_keypoints"],
        expected_nms_radius=compiled["mapping_nms_radius"],
        expected_candidate_pair_count=compiled["candidate_pair_count"],
    )
    validate_probe_proposal_lineage(probe=probe_record, proposals=proposal_record)
    validate_v2_frozen_source_contract(
        scene=scene, cache=cache, probe=probe_record, proposals=proposal_record
    )
    table_record = load_verified_cycle_table(
        path=verified_cycle_table,
        expected_file_sha256=compiled["verified_cycle_table"]["sha256"],
        expected_content_sha256=compiled["verified_cycle_table"]["content_sha256"],
        probe=probe_record,
        expected_maximum_reprojection_error_px=2.0,
    )
    selection_record = load_coverage_selection(
        path=selection,
        expected_file_sha256=compiled["pair_selection"]["sha256"],
        expected_content_sha256=compiled["pair_selection"]["content_sha256"],
        probe=probe_record,
        coverage_reference_pairs=proposal_record["nearest_pairs"],
        verified_cycle_table=table_record,
        expected_pair_budget=compiled["exact_pair_budget"],
    )
    selected_pairs = selection_pairs(selection_record["payload"])
    if (
        pair_table_sha256(proposal_record["nearest_pairs"])
        != compiled["control_pair_table_sha256"]
        or pair_table_sha256(selected_pairs)
        != compiled["variant_pair_table_sha256"]
    ):
        raise ValueError(f"{scene} pair table differs from the compiled registry")
    records = {
        "cross_scene_stage_a_gate": authority,
        "scene_stage_a_gate": stage_a,
        "query_cache": cache,
        "pair_proposals": proposal_record,
        "pair_match_probe": probe_record,
        "verified_cycle_table": table_record,
        "pair_selection": selection_record,
    }
    for name, record in records.items():
        if name in {"cross_scene_stage_a_gate", "scene_stage_a_gate"}:
            continue
        stage_name = name
        if stage_name not in stage_a["payload"]["inputs"]:
            raise ValueError(f"{scene} Stage-A gate lacks {stage_name}")
        if not _same_reference(stage_a["payload"]["inputs"][stage_name], record):
            raise ValueError(f"{scene} Stage-A/{stage_name} does not bind loaded input")
    return {
        "scene": scene,
        "compiled": compiled,
        **records,
        "manifest_path": Path(manifest).resolve(),
        "manifest_sha256": compiled["bootstrap_manifest"]["sha256"],
        "frozen_track_payload_path": Path(frozen_track_payload).resolve(),
        "frozen_track_payload_sha256": compiled["frozen_track_payload"]["sha256"],
    }


def load_compiled_scene_inputs(scene: str) -> dict:
    """Load a scene exclusively from the machine preregistration trust root."""
    scene = str(scene).lower()
    compiled = scene_preregistration(scene)
    authority = preregistration()["authorization"]["cross_scene_stage_a_gate"]
    return load_scene_inputs(
        scene=scene,
        cross_scene_stage_a_gate=Path(authority["path"]),
        expected_cross_scene_stage_a_gate_sha256=authority["sha256"],
        scene_stage_a_gate=Path(compiled["scene_stage_a_gate"]["path"]),
        expected_scene_stage_a_gate_sha256=compiled["scene_stage_a_gate"]["sha256"],
        manifest=Path(compiled["bootstrap_manifest"]["path"]),
        expected_manifest_sha256=compiled["bootstrap_manifest"]["sha256"],
        frozen_track_payload=Path(compiled["frozen_track_payload"]["path"]),
        expected_frozen_track_payload_sha256=compiled["frozen_track_payload"]["sha256"],
        query_cache=Path(compiled["query_cache"]["path"]),
        expected_query_cache_sha256=compiled["query_cache"]["sha256"],
        mapping_scope_equivalence=Path(compiled["mapping_scope_equivalence"]["path"]),
        expected_mapping_scope_equivalence_sha256=compiled[
            "mapping_scope_equivalence"
        ]["sha256"],
        proposals=Path(compiled["pair_proposals"]["path"]),
        expected_proposals_sha256=compiled["pair_proposals"]["sha256"],
        expected_proposals_content_sha256=compiled["pair_proposals"][
            "content_sha256"
        ],
        probe=Path(compiled["pair_match_probe"]["path"]),
        expected_probe_sha256=compiled["pair_match_probe"]["sha256"],
        expected_probe_content_sha256=compiled["pair_match_probe"][
            "content_sha256"
        ],
        verified_cycle_table=Path(compiled["verified_cycle_table"]["path"]),
        expected_verified_cycle_table_sha256=compiled["verified_cycle_table"][
            "sha256"
        ],
        expected_verified_cycle_table_content_sha256=compiled[
            "verified_cycle_table"
        ]["content_sha256"],
        selection=Path(compiled["pair_selection"]["path"]),
        expected_selection_sha256=compiled["pair_selection"]["sha256"],
        expected_selection_content_sha256=compiled["pair_selection"][
            "content_sha256"
        ],
        expected_query_names_sha256=compiled["query_names_sha256"],
        expected_mapping_keypoints=compiled["mapping_keypoints"],
        expected_nms_radius=compiled["mapping_nms_radius"],
        expected_pair_budget=compiled["exact_pair_budget"],
        expected_candidate_pair_count=compiled["candidate_pair_count"],
        expected_candidate_components=compiled["candidate_component_count"],
    )


def artifact_reference(artifact: dict) -> dict:
    result = {"path": str(artifact["path"]), "sha256": artifact["sha256"]}
    if "content_sha256" in artifact:
        result["content_sha256"] = artifact["content_sha256"]
    if "mapping_scope" in artifact:
        result["mapping_scope"] = deepcopy(artifact["mapping_scope"])
    return result


def frozen_track_lineage(registry: dict, base_lineage: dict) -> dict:
    lineage = deepcopy(base_lineage)
    lineage.update(
        {
            "cross_scene_stage_a_gate": artifact_reference(
                registry["cross_scene_stage_a_gate"]
            ),
            "scene_stage_a_gate": artifact_reference(
                registry["scene_stage_a_gate"]
            ),
            "pair_proposals": artifact_reference(registry["pair_proposals"]),
            "pair_match_probe": artifact_reference(registry["pair_match_probe"]),
            "verified_cycle_table": artifact_reference(
                registry["verified_cycle_table"]
            ),
            "pair_selection": artifact_reference(registry["pair_selection"]),
            "probe_matcher": deepcopy(registry["pair_match_probe"]["payload"]["matcher"]),
        }
    )
    lineage["query_cache"]["mapping_scope"] = deepcopy(
        registry["query_cache"]["mapping_scope"]
    )
    if "greatcourt_stage_b_parent" in registry:
        lineage["greatcourt_stage_b_parent"] = artifact_reference(
            registry["greatcourt_stage_b_parent"]
        )
    return lineage


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def code_identity(
    *,
    schema: str,
    algorithm: str,
    entrypoint: str,
    source_paths: tuple[str, ...],
    device: str | None,
) -> dict:
    root = Path(__file__).resolve().parents[1]
    actual = {name: _file_sha256(root / name) for name in source_paths}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        committed = {
            name: hashlib.sha256(
                subprocess.run(
                    ["git", "show", f"{commit}:{name}"],
                    cwd=root,
                    check=True,
                    capture_output=True,
                ).stdout
            ).hexdigest()
            for name in source_paths
        }
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("P8 producer identity requires a Git worktree") from error
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError("P8 producer Git commit is invalid")
    runtime = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch": str(torch.__version__),
    }
    if device is not None:
        runtime["device"] = str(torch.device(device))
    return {
        "schema": schema,
        "version": 1,
        "algorithm": algorithm,
        "entrypoint": entrypoint,
        "git_commit": commit,
        "required_source_paths_clean": actual == committed,
        "source_paths": list(source_paths),
        "source_file_sha256": actual,
        "runtime": runtime,
    }


def track_producer_identity(device: str) -> dict:
    return code_identity(
        schema=TRACK_PRODUCER_SCHEMA,
        algorithm="p8_cycle_verified_fisher_coverage_v2_reuse_track",
        entrypoint=(
            "python -m "
            "scripts.materialize_cycle_verified_fisher_coverage_track_factor"
        ),
        source_paths=TRACK_PRODUCER_SOURCE_PATHS,
        device=device,
    )


def stage_b_producer_identity() -> dict:
    return code_identity(
        schema=STAGE_B_PRODUCER_SCHEMA,
        algorithm="p8_cycle_verified_fisher_coverage_v2_stage_b",
        entrypoint=(
            "python -m scripts.compare_cycle_verified_fisher_coverage_mechanism"
        ),
        source_paths=STAGE_B_PRODUCER_SOURCE_PATHS,
        device=None,
    )


def cross_b_producer_identity() -> dict:
    return code_identity(
        schema=CROSS_B_PRODUCER_SCHEMA,
        algorithm="p8_cycle_verified_fisher_coverage_v2_cross_scene_stage_b",
        entrypoint=(
            "python -m scripts.aggregate_cycle_verified_fisher_coverage_cross_scene"
        ),
        source_paths=CROSS_B_PRODUCER_SOURCE_PATHS,
        device=None,
    )


def require_clean_identity(identity: dict, *, label: str) -> None:
    if identity.get("required_source_paths_clean") is not True:
        raise RuntimeError(f"{label} requires clean committed source bytes")


def validate_code_identity(
    identity: object,
    *,
    schema: str,
    algorithm: str,
    entrypoint: str,
    source_paths: tuple[str, ...],
    device: str | None,
    label: str,
) -> dict:
    if not isinstance(identity, dict):
        raise ValueError(f"{label} lacks producer identity")
    current = code_identity(
        schema=schema,
        algorithm=algorithm,
        entrypoint=entrypoint,
        source_paths=source_paths,
        device=device,
    )
    immutable = (
        "schema",
        "version",
        "algorithm",
        "entrypoint",
        "git_commit",
        "source_paths",
        "source_file_sha256",
        "runtime",
    )
    if any(identity.get(name) != current[name] for name in immutable):
        raise ValueError(f"{label} producer source/commit/runtime identity differs")
    if identity.get("required_source_paths_clean") is not True:
        raise ValueError(f"{label} producer did not attest clean source bytes")
    require_clean_identity(current, label=label)
    return deepcopy(identity)


def validate_track_producer_identity(identity: object, *, label: str) -> dict:
    device = (
        identity.get("runtime", {}).get("device")
        if isinstance(identity, dict)
        else None
    )
    if not isinstance(device, str):
        raise ValueError(f"{label} producer lacks a device identity")
    return validate_code_identity(
        identity,
        schema=TRACK_PRODUCER_SCHEMA,
        algorithm="p8_cycle_verified_fisher_coverage_v2_reuse_track",
        entrypoint=(
            "python -m "
            "scripts.materialize_cycle_verified_fisher_coverage_track_factor"
        ),
        source_paths=TRACK_PRODUCER_SOURCE_PATHS,
        device=device,
        label=label,
    )


def recursive_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        try:
            left_tensor = torch.as_tensor(left).detach().cpu()
            right_tensor = torch.as_tensor(right).detach().cpu()
            if (
                left_tensor.dtype != right_tensor.dtype
                or left_tensor.shape != right_tensor.shape
            ):
                return False
            if torch.equal(left_tensor, right_tensor):
                return True
            if left_tensor.is_floating_point() or left_tensor.is_complex():
                return bool(
                    torch.all(
                        (left_tensor == right_tensor)
                        | (torch.isnan(left_tensor) & torch.isnan(right_tensor))
                    )
                )
            return False
        except (TypeError, ValueError):
            return False
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and set(left) == set(right)
            and all(recursive_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(recursive_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def track_metrics(factor: dict, *, query_count: int) -> dict:
    report = _track_report(
        factor["tracks"], factor["track_geometry"], query_count=query_count
    )
    covariance = report["triangulated_covariance_trace_m2"]["p90"]
    return {
        "triangulated_tracks": int(report["triangulated_track_count"]),
        "broad_eligible_tracks": int(report["broad_eligible_track_count"]),
        "high_confidence_tracks": int(report["high_confidence_track_count"]),
        "triangulated_covariance_p90_m2": (
            None if covariance is None else float(covariance)
        ),
        "mapping_query_with_broad_track_fraction": float(
            report["mapping_query_with_broad_track_fraction"]
        ),
    }


def completion_artifact_names() -> dict[str, tuple[str, str, str]]:
    return {
        "control": (
            CONTROL_POLICY_NAME,
            CONTROL_SUBSET_ROLE,
            "cycle_verified_fisher_coverage_nearest_control_track_factor",
        ),
        "variant": (
            VARIANT_POLICY_NAME,
            VARIANT_SUBSET_ROLE,
            "cycle_verified_fisher_coverage_track_factor",
        ),
    }


def validate_completion_manifest(
    *, path: Path, expected_sha256: str, expected_scene: str
) -> dict:
    path = attest_file(path, expected_sha256, label="paired Track completion manifest")
    payload = json.loads(path.read_text())
    root = path.parent.resolve()
    artifacts = payload.get("artifacts")
    inputs = payload.get("inputs")
    summaries = payload.get("summaries")
    implementation = payload.get("implementation_registry")
    compiled_implementation = implementation_registry()
    compiled_implementation_reference = {
        "path": str(IMPLEMENTATION_REGISTRY_PATH),
        "sha256": sha256_file(IMPLEMENTATION_REGISTRY_PATH),
        "implementation_commit": compiled_implementation["implementation_commit"],
    }
    if (
        payload.get("schema") != COMPLETION_SCHEMA
        or payload.get("version") != 1
        or payload.get("uses_test_queries") is not False
        or payload.get("mapping_only") is not True
        or payload.get("complete") is not True
        or payload.get("partial") is not False
        or payload.get("resume_allowed") is not False
        or payload.get("scene") != expected_scene
        or payload.get("build_order") != ["control", "variant"]
        or not isinstance(payload.get("run_uuid"), str)
        or re.fullmatch(r"[0-9a-f]{32}", payload["run_uuid"]) is None
        or not isinstance(artifacts, dict)
        or set(artifacts) != {"control_factor", "control_report", "variant_factor", "variant_report"}
        or not isinstance(inputs, dict)
        or not isinstance(summaries, dict)
        or set(summaries) != {"control", "variant"}
        or payload.get("failure_recovery")
        != "isolate_entire_output_root_and_rebuild_both_arms_from_scratch"
        or implementation != compiled_implementation_reference
    ):
        raise ValueError("Paired Track completion manifest is invalid or partial")
    producer = validate_track_producer_identity(
        payload.get("track_producer_identity"), label="paired Track completion"
    )
    loaded_artifacts = {}
    expected_stems = completion_artifact_names()
    for role in ("control", "variant"):
        stem = expected_stems[role][2]
        for kind, suffix in (("factor", ".pt"), ("report", ".json")):
            name = f"{role}_{kind}"
            reference = artifacts[name]
            expected_path = root / f"{stem}{suffix}"
            if (
                not isinstance(reference, dict)
                or Path(str(reference.get("path", ""))).resolve() != expected_path
                or reference.get("sha256") != sha256_file(expected_path)
            ):
                raise ValueError(f"Completion manifest {name} is missing or changed")
            loaded_artifacts[name] = {
                "path": expected_path,
                "sha256": reference["sha256"],
            }
    return {
        "path": path,
        "sha256": sha256_file(path),
        "payload": payload,
        "producer": producer,
        "artifacts": loaded_artifacts,
    }


def load_completed_arms(*, completion: dict, registry: dict) -> dict:
    scene = registry["scene"]
    compiled = registry["compiled"]
    payload = completion["payload"]
    expected_input_refs = {
        "cross_scene_stage_a_gate": artifact_reference(
            registry["cross_scene_stage_a_gate"]
        ),
        "scene_stage_a_gate": artifact_reference(registry["scene_stage_a_gate"]),
        "query_cache": artifact_reference(registry["query_cache"]),
        "pair_proposals": artifact_reference(registry["pair_proposals"]),
        "pair_match_probe": artifact_reference(registry["pair_match_probe"]),
        "verified_cycle_table": artifact_reference(registry["verified_cycle_table"]),
        "pair_selection": artifact_reference(registry["pair_selection"]),
        "manifest": {
            "path": str(registry["manifest_path"]),
            "sha256": registry["manifest_sha256"],
        },
        "frozen_track_payload": {
            "path": str(registry["frozen_track_payload_path"]),
            "sha256": registry["frozen_track_payload_sha256"],
        },
    }
    if scene == "stairs":
        parent = payload.get("inputs", {}).get("greatcourt_stage_b_parent")
        if not isinstance(parent, dict):
            raise ValueError("Stairs completion lacks its GreatCourt Stage-B parent")
        expected_input_refs["greatcourt_stage_b_parent"] = deepcopy(parent)
        from scripts.compare_cycle_verified_fisher_coverage_mechanism import (
            validate_stage_b_gate,
        )

        validated_parent = validate_stage_b_gate(
            scene="greatcourt",
            path=Path(str(parent.get("path", ""))),
            expected_sha256=str(parent.get("sha256", "")),
        )
        if validated_parent["payload"]["scene_specific_mechanism_pass"] is not True:
            raise ValueError("Stairs completion binds a GreatCourt Stage-B STOP")
    if payload.get("inputs") != expected_input_refs:
        raise ValueError("Completion manifest input registry differs from preregistration")
    expected_names = registry["query_cache"]["names"]
    common = {
        "expected_query_names": expected_names,
        "expected_query_names_sha256": registry["query_cache"]["query_names_sha256"],
        "expected_query_cache_path": registry["query_cache"]["path"],
        "expected_query_cache_sha256": registry["query_cache"]["sha256"],
        "expected_mapping_keypoints": compiled["mapping_keypoints"],
        "expected_nms_radius": compiled["mapping_nms_radius"],
        "expected_pair_budget": compiled["exact_pair_budget"],
    }
    result = {}
    expected_scene_contract = {
        "scene": scene,
        "mapping_keypoints": compiled["mapping_keypoints"],
        "nms_radius": compiled["mapping_nms_radius"],
        "pair_budget": compiled["exact_pair_budget"],
        "candidate_pair_count": compiled["candidate_pair_count"],
        "candidate_component_count": compiled["candidate_component_count"],
    }
    for role, (policy, subset_role, _) in completion_artifact_names().items():
        factor_ref = completion["artifacts"][f"{role}_factor"]
        report_ref = completion["artifacts"][f"{role}_report"]
        factor = load_track_factor(
            path=factor_ref["path"],
            expected_file_sha256=factor_ref["sha256"],
            expected_policy=policy,
            **common,
        )
        report = json.loads(report_ref["path"].read_text())
        factor_payload = factor["payload"]
        lineage = factor_payload.get("input_lineage")
        parameters = factor_payload.get("pair_policy_parameters")
        diagnostics = factor_payload.get("diagnostics")
        sidecar_policy = factor_payload.get("pair_sidecar", {}).get("policy", {})
        if (
            report.get("schema") != "lafgs_pair_policy_track_factor"
            or report.get("version") != 1
            or report.get("uses_test_queries") is not False
            or report.get("reuse_only") is not True
            or report.get("pair_policy") != policy
            or report.get("scene_contract") != expected_scene_contract
            or report.get("mapping_keypoint_factor") != compiled["mapping_keypoints"]
            or report.get("mapping_nms_radius") != compiled["mapping_nms_radius"]
            or report.get("exact_pair_budget") != compiled["exact_pair_budget"]
            or report.get("mapping_query_count") != len(expected_names)
            or report.get("query_names_sha256")
            != registry["query_cache"]["query_names_sha256"]
            or report.get("probe_matcher") != MATCHER_CONTRACT
            or Path(str(report.get("artifact", ""))).resolve() != factor["path"]
            or report.get("artifact_sha256") != factor["sha256"]
            or report.get("inputs") != lineage
            or report.get("pair_policy_parameters") != parameters
            or report.get("paired_run_uuid") != payload["run_uuid"]
            or report.get("track_producer_identity") != completion["producer"]
            or factor_payload.get("paired_run_uuid") != payload["run_uuid"]
            or factor_payload.get("track_producer_identity") != completion["producer"]
            or not isinstance(lineage, dict)
            or lineage.get("paired_run_uuid") != payload["run_uuid"]
            or lineage.get("track_producer_identity") != completion["producer"]
            or lineage.get("pair_subset_role") != subset_role
            or lineage.get("probe_matcher") != MATCHER_CONTRACT
            or not isinstance(parameters, dict)
            or parameters.get("reuse_only") is not True
            or parameters.get("pair_subset_role") != subset_role
            or parameters.get("probe_matcher") != MATCHER_CONTRACT
            or not isinstance(parameters.get("track_science_contract"), dict)
            or not isinstance(diagnostics, dict)
            or diagnostics.get("track_pair_matches_reused") != 1
            or diagnostics.get("track_camera_pair_policy") != policy
            or diagnostics.get("track_camera_pair_budget")
            != compiled["exact_pair_budget"]
            or sidecar_policy.get("name") != policy
            or sidecar_policy.get("exact_pair_budget")
            != compiled["exact_pair_budget"]
            or sidecar_policy.get("uses_precomputed_pair_matches") is not True
            or sidecar_policy.get("uses_test_queries") is not False
        ):
            raise ValueError(f"Completed {scene}/{role} Track contract is invalid")
        expected_lineage = deepcopy(expected_input_refs)
        expected_lineage["query_cache"]["mapping_scope"] = deepcopy(
            registry["query_cache"]["mapping_scope"]
        )
        expected_lineage["equivalent_query_cache_rebind"] = lineage.get(
            "equivalent_query_cache_rebind"
        )
        expected_lineage.update(
            {
                "probe_matcher": deepcopy(MATCHER_CONTRACT),
                "paired_run_uuid": payload["run_uuid"],
                "track_producer_identity": deepcopy(completion["producer"]),
                "pair_subset_role": subset_role,
            }
        )
        if lineage != expected_lineage:
            raise ValueError(f"Completed {scene}/{role} lineage differs")
        expected_pairs = (
            registry["pair_proposals"]["nearest_pairs"]
            if role == "control"
            else selection_pairs(registry["pair_selection"]["payload"])
        )
        if factor["pairs"] != expected_pairs:
            raise ValueError(f"Completed {scene}/{role} pair subset differs")
        expected_pair_sha = compiled[
            "control_pair_table_sha256"
            if role == "control"
            else "variant_pair_table_sha256"
        ]
        if pair_table_sha256(factor["pairs"]) != expected_pair_sha:
            raise ValueError(f"Completed {scene}/{role} pair-table hash differs")
        expected_track = _track_report(
            factor_payload["tracks"],
            factor_payload["track_geometry"],
            query_count=len(expected_names),
        )
        if report.get("track") != expected_track:
            raise ValueError(f"Completed {scene}/{role} report metrics are stale")
        summary = payload["summaries"].get(role)
        if summary != {
            "pair_policy": policy,
            "pair_subset_role": subset_role,
            "track": expected_track,
        }:
            raise ValueError(f"Completed {scene}/{role} summary differs from report")
        result[role] = {
            "factor": factor,
            "report": {
                "path": report_ref["path"],
                "sha256": report_ref["sha256"],
                "payload": report,
            },
            "metrics": track_metrics(factor_payload, query_count=len(expected_names)),
            "science": deepcopy(parameters["track_science_contract"]),
        }
    control_lineage = deepcopy(result["control"]["factor"]["payload"]["input_lineage"])
    variant_lineage = deepcopy(result["variant"]["factor"]["payload"]["input_lineage"])
    control_lineage.pop("pair_subset_role")
    variant_lineage.pop("pair_subset_role")
    control_parameters = deepcopy(
        result["control"]["factor"]["payload"]["pair_policy_parameters"]
    )
    variant_parameters = deepcopy(
        result["variant"]["factor"]["payload"]["pair_policy_parameters"]
    )
    control_parameters.pop("pair_subset_role")
    variant_parameters.pop("pair_subset_role")
    if control_lineage != variant_lineage or control_parameters != variant_parameters:
        raise ValueError("Paired Track arms differ outside policy-derived outputs")
    return result


def reference_registry_unchanged(registry: dict) -> None:
    artifacts = {
        "cross_scene_stage_a_gate": registry["cross_scene_stage_a_gate"],
        "scene_stage_a_gate": registry["scene_stage_a_gate"],
        "query_cache": registry["query_cache"],
        "pair_proposals": registry["pair_proposals"],
        "pair_match_probe": registry["pair_match_probe"],
        "verified_cycle_table": registry["verified_cycle_table"],
        "pair_selection": registry["pair_selection"],
    }
    for name, artifact in artifacts.items():
        if sha256_file(artifact["path"]) != artifact["sha256"]:
            raise RuntimeError(f"Frozen P8 input changed during execution: {name}")
    extra = {
        "manifest": (
            registry["manifest_path"],
            registry["manifest_sha256"],
        ),
        "frozen_track_payload": (
            registry["frozen_track_payload_path"],
            registry["frozen_track_payload_sha256"],
        ),
    }
    for name, (path, digest) in extra.items():
        if sha256_file(path) != digest:
            raise RuntimeError(f"Frozen P8 input changed during execution: {name}")
