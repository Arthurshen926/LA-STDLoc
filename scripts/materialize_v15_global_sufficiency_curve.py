#!/usr/bin/env python3
"""Materialize mapping-only and feedback-conditioned global budget curves."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import yaml

from common.hashing import sha256_file
from map_learning.v15_global_sufficiency import (
    feedback_conditioned_reliability,
    feedback_utility_components,
)
from scripts.run_v7_closed_loop import (
    _materialize_selected_map,
    _v7_candidate_eligibility,
)
from topology.v7_sufficiency_selector import (
    CompactEdgeRegistry,
    CompactPoseInformation,
    SufficiencyTargets,
    select_v7_sufficiency,
)


def _save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_design_records(path: Path, expected_map_sha: str) -> list[dict]:
    payload = json.loads(path.read_text())
    if not (
        payload.get("schema") == "lafgs_v9_no_loo_causal_feedback_batch"
        and payload.get("role") == "controller_design"
        and payload.get("loo_used") is False
        and payload.get("uses_test_queries") is False
        and payload.get("accepted_query_row_policy") == "v2_row_valid_only"
        and payload.get("input", {}).get("map_sha256") == expected_map_sha
    ):
        raise ValueError("feedback design batch violates the V15 split contract")
    records = []
    for item in payload["records"]:
        record_path = Path(item["path"])
        if sha256_file(record_path) != item["sha256"]:
            raise ValueError("feedback design record SHA256 differs")
        records.append(torch.load(record_path, map_location="cpu", weights_only=False))
    return records


def _feasible_unique(
    identity: torch.Tensor,
    *,
    edge_eligible: torch.Tensor,
    query_indices: torch.Tensor,
    query_count: int,
) -> torch.Tensor:
    identity = torch.as_tensor(identity).long()
    valid = edge_eligible & (identity >= 0)
    if not bool(valid.any()):
        return torch.zeros(query_count, dtype=torch.long)
    base = int(identity[valid].max()) + 1
    unique = torch.unique(query_indices[valid] * base + identity[valid])
    return torch.bincount(
        torch.div(unique, base, rounding_mode="floor"), minlength=query_count
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--candidate-evidence", type=Path, required=True)
    parser.add_argument("--baseline-map", type=Path, required=True)
    parser.add_argument("--design-batch", type=Path, required=True)
    parser.add_argument("--actual-removal-audit", type=Path, required=True)
    parser.add_argument(
        "--profile",
        action="append",
        help="Materialize only this fixed profile (repeatable; default: all)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    config = yaml.safe_load(args.config.read_text())
    if config.get("schema") != "lafgs_v15_global_sufficiency_closed_loop":
        raise ValueError("invalid V15 config")
    selected_profiles = args.profile or list(config["profiles"])
    if unknown := set(selected_profiles) - set(config["profiles"]):
        raise ValueError(f"unknown V15 profiles: {sorted(unknown)}")
    candidate_path = args.candidate_pool.resolve()
    evidence_path = args.candidate_evidence.resolve()
    map_path = args.baseline_map.resolve()
    map_sha = sha256_file(map_path)
    candidates = torch.load(candidate_path, map_location="cpu", weights_only=False)
    evidence = torch.load(evidence_path, map_location="cpu", weights_only=False)
    baseline = torch.load(map_path, map_location="cpu", weights_only=False)
    count = int(torch.as_tensor(candidates["anchor_ids"]).numel())
    if not (
        candidates.get("schema") == "projective_anchor_candidates_v2"
        and evidence.get("schema") == "lafgs_v7_reconstructed_mapping_candidate_evidence"
        and int(evidence.get("candidate_count", -1)) == count
        and torch.equal(candidates["anchor_ids"], baseline["anchor_ids"])
    ):
        raise ValueError("V15 mapping inputs do not describe one frozen V2 map")

    audit = torch.load(
        args.actual_removal_audit, map_location="cpu", weights_only=False
    )
    if not (
        audit.get("schema") == "lafgs_v9_actual_removal_gain_audit"
        and audit.get("loo_used") is False
    ):
        raise ValueError("V15 requires the corrected exact-removal audit")
    design_records = _load_design_records(args.design_batch.resolve(), map_sha)
    components = feedback_utility_components(
        design_records,
        anchor_count=count,
        harmful_anchor_rows=audit["authorized_anchor_rows"],
    )
    mapping_reliability = (
        torch.as_tensor(candidates["geometry_reliability"]).float()
        * torch.as_tensor(candidates["identity_reliability"]).float()
    )
    priorities = {
        "mapping_only": mapping_reliability,
        "feedback_conditioned": feedback_conditioned_reliability(
            mapping_reliability, components
        ),
    }

    selector_config = {
        "evidence": config["evidence"],
    }
    eligible, exclusions, thresholds = _v7_candidate_eligibility(
        candidates, evidence, selector_config
    )
    observations = candidates["projective_anchor_observations"]
    offsets = torch.as_tensor(observations["observation_offsets"]).long()
    query_indices = torch.as_tensor(observations["query_indices"]).long()
    keypoint_indices = torch.as_tensor(observations["keypoint_indices"]).long()
    query_count = len(candidates["query_names"])
    layers = {
        "matching": CompactEdgeRegistry(offsets, query_indices, keypoint_indices),
        "image_cell": CompactEdgeRegistry(
            offsets, query_indices, evidence["image_cell_identities"]
        ),
        "view_family": CompactEdgeRegistry(
            offsets, query_indices, evidence["view_family_identities"]
        ),
        "depth_range": CompactEdgeRegistry(
            offsets, query_indices, evidence["depth_range_identities"]
        ),
    }
    observation_count = torch.as_tensor(evidence["observation_count"]).long()
    anchor_for_edge = torch.repeat_interleave(torch.arange(count), observation_count)
    edge_eligible = eligible[anchor_for_edge]
    feasible_matching = torch.bincount(
        query_indices[edge_eligible], minlength=query_count
    )
    feasible = {
        "image_cell": _feasible_unique(
            evidence["image_cell_identities"],
            edge_eligible=edge_eligible,
            query_indices=query_indices,
            query_count=query_count,
        ),
        "view_family": _feasible_unique(
            evidence["view_family_identities"],
            edge_eligible=edge_eligible,
            query_indices=query_indices,
            query_count=query_count,
        ),
        "depth_range": _feasible_unique(
            evidence["depth_range_identities"],
            edge_eligible=edge_eligible,
            query_indices=query_indices,
            query_count=query_count,
        ),
    }
    pose_damping = float(config["pose_damping"])
    full_pose = torch.eye(6, dtype=torch.float64).repeat(query_count, 1, 1) * pose_damping
    full_pose.reshape(query_count, 36).index_add_(
        0,
        query_indices[edge_eligible],
        torch.as_tensor(evidence["pose_information_contributions"])[edge_eligible]
        .double()
        .reshape(-1, 36),
    )
    eigenvalues = torch.linalg.eigvalsh(full_pose)
    feasible_logdet = eigenvalues.clamp_min(1e-12).log().sum(1)
    feasible_minimum = eigenvalues[:, 0]

    def integer_target(requested: int, maximum: torch.Tensor) -> list[int]:
        return torch.minimum(maximum, torch.full_like(maximum, requested)).tolist()

    args.output_dir.mkdir(parents=True)
    arms = []
    for profile_name in selected_profiles:
        profile = config["profiles"][profile_name]
        targets = SufficiencyTargets(
            precision_matching_rank=integer_target(
                int(profile["precision_rank"]), feasible_matching
            ),
            completion_matching_rank=integer_target(
                int(profile["matching_rank"]), feasible_matching
            ),
            image_cells=integer_target(
                int(profile["image_cells"]), feasible["image_cell"]
            ),
            view_families=integer_target(
                int(profile["view_families"]), feasible["view_family"]
            ),
            depth_ranges=integer_target(
                int(profile["depth_ranges"]), feasible["depth_range"]
            ),
            pose_logdet=torch.minimum(
                feasible_logdet,
                torch.full_like(feasible_logdet, float(profile["pose_logdet"])),
            ).tolist(),
            pose_minimum_eigenvalue=torch.minimum(
                feasible_minimum,
                torch.full_like(
                    feasible_minimum, float(profile["pose_minimum_eigenvalue"])
                ),
            ).tolist(),
            maximum_anchors=int(profile["maximum_anchors"]),
            pose_damping=pose_damping,
        )
        for policy, reliability in priorities.items():
            print(f"V15 selecting {profile_name}/{policy}", flush=True)
            selection = select_v7_sufficiency(
                anchor_ids=candidates["anchor_ids"],
                reliability=reliability,
                eligible=eligible,
                layer_edges=layers,
                pose_information=CompactPoseInformation(
                    offsets,
                    query_indices,
                    evidence["pose_information_contributions"],
                ),
                query_count=query_count,
                targets=targets,
            )
            selection["schema"] = "lafgs_v15_global_sufficiency_selection"
            selection["profile"] = profile_name
            selection["selection_policy"] = policy
            selection["feedback_utility_contract"] = {
                "design_pose_families_only": policy == "feedback_conditioned",
                "mapping_constraints_changed": False,
                "exact_removal_harm_demoted_not_forcibly_excluded": True,
            }
            unmet_total = sum(int(value) for value in selection["unmet"].values())
            selection["budget_feasible"] = unmet_total == 0
            core_rows = torch.as_tensor(selection["selected_anchor_rows"]).long()
            selection["minimum_sufficient_core_anchor_count"] = int(core_rows.numel())
            if selection["budget_feasible"] and bool(config.get("fill_to_budget", False)):
                budget = int(profile["maximum_anchors"])
                selected_set = set(core_rows.tolist())
                order = sorted(
                    range(count),
                    key=lambda row: (
                        -float(torch.nan_to_num(reliability[row], nan=-torch.inf)),
                        int(candidates["anchor_ids"][row]),
                    ),
                )
                for row in order:
                    if len(selected_set) >= budget:
                        break
                    if row not in selected_set and bool(eligible[row]):
                        selected_set.add(row)
                        selection["primary_selection_reason"][
                            int(candidates["anchor_ids"][row])
                        ] = "budgeted_redundancy_fill"
                filled_rows = sorted(
                    selected_set, key=lambda row: int(candidates["anchor_ids"][row])
                )
                selection["selected_anchor_rows"] = torch.tensor(
                    filled_rows, dtype=torch.long
                )
                selection["selected_anchor_ids"] = candidates["anchor_ids"][filled_rows]
            selection["budget_fill_anchor_count"] = int(
                selection["selected_anchor_rows"].numel() - core_rows.numel()
            )
            selection["achieved_values_are_core_lower_bounds"] = bool(
                selection["budget_fill_anchor_count"]
            )
            arm_name = f"{profile_name}_{policy}"
            arm_dir = args.output_dir / arm_name
            selected_rows = selection["selected_anchor_rows"]
            selected_map = _materialize_selected_map(baseline, selected_rows)
            selected_map["provenance"] = {
                **dict(selected_map.get("provenance", {})),
                "v15_global_sufficiency": True,
                "v15_profile": profile_name,
                "v15_selection_policy": policy,
                "v15_feedback_design_only": policy == "feedback_conditioned",
                "uses_test_queries": False,
            }
            selection_path = arm_dir / "selection.pt"
            output_map = arm_dir / "projective_anchor_map.pt"
            _save(selection, selection_path)
            _save(selected_map, output_map)
            report = {
                "arm": arm_name,
                "profile": profile_name,
                "selection_policy": policy,
                "maximum_anchors": int(profile["maximum_anchors"]),
                "selected_anchor_count": int(selected_rows.numel()),
                "minimum_sufficient_core_anchor_count": int(core_rows.numel()),
                "budget_fill_anchor_count": int(
                    selection["budget_fill_anchor_count"]
                ),
                "compression_fraction": float(1.0 - selected_rows.numel() / count),
                "budget_feasible": selection["budget_feasible"],
                "unmet": selection["unmet"],
                "selection": str(selection_path.resolve()),
                "selection_sha256": sha256_file(selection_path),
                "map": str(output_map.resolve()),
                "map_sha256": sha256_file(output_map),
            }
            (arm_dir / "report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n"
            )
            arms.append(report)

    component_path = args.output_dir / "feedback_utility_components.pt"
    _save(components, component_path)
    manifest = {
        "schema": "lafgs_v15_global_sufficiency_curve",
        "version": 1,
        "uses_test_queries": False,
        "loo_used": False,
        "baseline_anchor_count": count,
        "inputs": {
            "config": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config.resolve()),
            "candidate_pool": str(candidate_path),
            "candidate_pool_sha256": sha256_file(candidate_path),
            "candidate_evidence": str(evidence_path),
            "candidate_evidence_sha256": sha256_file(evidence_path),
            "baseline_map": str(map_path),
            "baseline_map_sha256": map_sha,
            "design_batch": str(args.design_batch.resolve()),
            "design_batch_sha256": sha256_file(args.design_batch.resolve()),
            "actual_removal_audit": str(args.actual_removal_audit.resolve()),
            "actual_removal_audit_sha256": sha256_file(
                args.actual_removal_audit.resolve()
            ),
        },
        "eligibility": {
            "eligible_count": int(eligible.sum()),
            "exclusions": exclusions,
            "thresholds": thresholds.__dict__,
        },
        "feedback": {
            "accepted_pose_family_count": int(
                components["accepted_pose_family_count"]
            ),
            "clean_supported_anchor_count": int(
                (components["clean_pose_family_count"] > 0).sum()
            ),
            "task_supported_anchor_count": int(
                (components["task_pose_family_count"] > 0).sum()
            ),
            "causally_harmful_anchor_count": int(
                components["causally_harmful"].sum()
            ),
            "components": str(component_path.resolve()),
            "components_sha256": sha256_file(component_path),
        },
        "arms": arms,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
