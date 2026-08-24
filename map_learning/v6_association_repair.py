"""Training-split-only local Track repair for V6 Projective Anchors."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

from common.v6_contracts import ASSOCIATION_GRAPH_SCHEMA, FEEDBACK_SCHEMA, require_schema
from evidence.observation_provider import ObservationProvider
from evidence.projective_reconstruction import reconstruct_projective_anchors
from topology.v6_anchor_map import (
    materialize_projective_anchor_map,
    merge_projective_candidates,
    projective_candidates_from_map,
    subset_projective_anchor_map,
)


def select_association_repair_pairs(
    state: dict,
    feedback: dict,
    *,
    training_query_indices: Sequence[int] | torch.Tensor,
    minimum_descriptor_similarity: float,
    maximum_xyz_distance_m: float,
    minimum_query_evidence: int,
) -> tuple[torch.Tensor, dict]:
    """Select disjoint fragmented-Track pairs using training feedback only."""

    require_schema(feedback, FEEDBACK_SCHEMA, label="association-repair feedback")
    training = torch.unique(
        torch.as_tensor(training_query_indices, dtype=torch.long).reshape(-1),
        sorted=True,
    )
    records = list(feedback.get("records", ()))
    if training.numel() == 0 or int(training.min()) < 0 or int(training.max()) >= len(records):
        raise ValueError("association-repair training query registry is invalid")
    if not 0.0 <= float(minimum_descriptor_similarity) <= 1.0:
        raise ValueError("association-repair descriptor similarity is invalid")
    if not float(maximum_xyz_distance_m) > 0.0 or int(minimum_query_evidence) < 1:
        raise ValueError("association-repair geometry/evidence threshold is invalid")
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    features = F.normalize(torch.as_tensor(state["anchor_features"]).float(), dim=1)
    count = int(xyz.shape[0])
    csr = state.get("projective_anchor_observations", {})
    offsets = torch.as_tensor(csr.get("observation_offsets")).long()
    query = torch.as_tensor(csr.get("query_indices")).long()
    if offsets.shape != (count + 1,) or int(offsets[-1]) != query.numel():
        raise ValueError("association-repair source observation CSR is invalid")

    evidence: dict[tuple[int, int], set[int]] = defaultdict(set)
    raw_pair_occurrences = 0
    threshold_pair_occurrences = 0
    for query_index in training.tolist():
        record = records[query_index]
        ambiguous_by_row: dict[int, list[int]] = defaultdict(list)
        ambiguous = torch.as_tensor(
            record.get("certified_pose_valid_alternative_pairs", ())
        ).long().reshape(-1, 2)
        inactive = torch.as_tensor(
            record.get("identity_inactive_pairs", ())
        ).long().reshape(-1, 2)
        for row, anchor in ambiguous.tolist():
            ambiguous_by_row[int(row)].append(int(anchor))
        for row, source in inactive.tolist():
            candidates = ambiguous_by_row.get(int(row), ())
            raw_pair_occurrences += len(candidates)
            if not candidates:
                continue
            candidate_rows = torch.as_tensor(candidates, dtype=torch.long)
            similarity = features[candidate_rows] @ features[int(source)]
            distance = torch.linalg.norm(xyz[candidate_rows] - xyz[int(source)], dim=1)
            eligible = candidate_rows[
                (similarity >= float(minimum_descriptor_similarity))
                & (distance <= float(maximum_xyz_distance_m))
            ]
            threshold_pair_occurrences += int(eligible.numel())
            for target in eligible.tolist():
                if int(source) == int(target):
                    continue
                evidence[tuple(sorted((int(source), int(target))))].add(query_index)

    candidates = []
    observation_conflict_count = 0
    for pair, queries in evidence.items():
        if len(queries) < int(minimum_query_evidence):
            continue
        first, second = pair
        first_queries = query[offsets[first] : offsets[first + 1]]
        second_queries = query[offsets[second] : offsets[second + 1]]
        if bool(torch.isin(first_queries, second_queries).any()):
            observation_conflict_count += 1
            continue
        similarity = float(features[first] @ features[second])
        distance = float(torch.linalg.norm(xyz[first] - xyz[second]))
        candidates.append((pair, len(queries), similarity, distance))
    candidates.sort(key=lambda value: (-value[1], -value[2], value[3], value[0]))
    selected = []
    used: set[int] = set()
    for pair, evidence_count, similarity, distance in candidates:
        if pair[0] in used or pair[1] in used:
            continue
        used.update(pair)
        selected.append((pair, evidence_count, similarity, distance))
    pairs = torch.tensor(
        [pair for pair, _, _, _ in selected], dtype=torch.long
    ).reshape(-1, 2)
    return pairs, {
        "schema": "lafgs_v6_feedback_association_repair_selection",
        "version": 1,
        "training_query_indices": training,
        "minimum_descriptor_similarity": float(minimum_descriptor_similarity),
        "maximum_xyz_distance_m": float(maximum_xyz_distance_m),
        "minimum_query_evidence": int(minimum_query_evidence),
        "raw_pair_occurrence_count": raw_pair_occurrences,
        "threshold_pair_occurrence_count": threshold_pair_occurrences,
        "evidence_pair_count": len(evidence),
        "eligible_pair_count": len(candidates),
        "observation_conflict_pair_count": observation_conflict_count,
        "selected_pair_count": len(selected),
        "selected_pairs": pairs,
        "selected_pair_evidence_counts": torch.tensor(
            [value[1] for value in selected], dtype=torch.long
        ),
        "selected_pair_descriptor_similarities": torch.tensor(
            [value[2] for value in selected], dtype=torch.float32
        ),
        "selected_pair_xyz_distances_m": torch.tensor(
            [value[3] for value in selected], dtype=torch.float32
        ),
        "pairwise_disjoint_source_anchors": True,
        "one_observation_per_camera_before_reconstruction": True,
    }


def deploy_association_repair_rule_globally(
    state: dict,
    certified_pairs: torch.Tensor,
    certified_report: dict,
    *,
    minimum_descriptor_similarity: float,
    maximum_xyz_distance_m: float,
) -> tuple[torch.Tensor, dict]:
    """Apply a training-feedback-certified repair rule to the whole source map.

    The action holdout never contributes an edge label.  It may nevertheless be
    transformed by the rule learned/certified on the training split, just as a
    deployed model is applied to unseen inputs.  Ray reconstruction remains the
    final geometric acceptance test.
    """

    certified_pairs = torch.as_tensor(certified_pairs, dtype=torch.long).reshape(-1, 2)
    if certified_pairs.numel() == 0:
        raise ValueError("global association repair requires training-certified pairs")
    xyz = torch.as_tensor(state["anchor_xyz"]).float().cpu()
    features = F.normalize(torch.as_tensor(state["anchor_features"]).float().cpu(), dim=1)
    spatial = cKDTree(xyz.numpy()).query_pairs(
        r=float(maximum_xyz_distance_m), output_type="ndarray"
    )
    spatial = torch.from_numpy(np.asarray(spatial, dtype=np.int64)).long().reshape(-1, 2)
    if spatial.numel() == 0:
        raise ValueError("global association repair found no spatial neighbor pair")
    similarity = (features[spatial[:, 0]] * features[spatial[:, 1]]).sum(dim=1)
    keep = similarity >= float(minimum_descriptor_similarity)
    spatial = spatial[keep]
    similarity = similarity[keep]

    csr = state.get("projective_anchor_observations", {})
    offsets = torch.as_tensor(csr.get("observation_offsets")).long()
    query = torch.as_tensor(csr.get("query_indices")).long()
    certified_counts = torch.as_tensor(
        certified_report["selected_pair_evidence_counts"]
    ).long()
    certified = {
        tuple(sorted(map(int, pair))): int(count)
        for pair, count in zip(certified_pairs.tolist(), certified_counts.tolist())
    }
    candidates = []
    observation_conflict_count = 0
    for index, pair in enumerate(spatial.tolist()):
        first, second = map(int, pair)
        first_queries = query[offsets[first] : offsets[first + 1]]
        second_queries = query[offsets[second] : offsets[second + 1]]
        if bool(torch.isin(first_queries, second_queries).any()):
            observation_conflict_count += 1
            continue
        key = (first, second)
        candidates.append(
            (
                key,
                certified.get(key, 0),
                float(similarity[index]),
                float(torch.linalg.norm(xyz[first] - xyz[second])),
            )
        )
    # Training-certified pairs receive priority.  The frozen local rule is then
    # deployed by descriptor agreement and finally by spatial proximity.
    candidates.sort(key=lambda value: (-value[1], -value[2], value[3], value[0]))
    selected = []
    used: set[int] = set()
    for candidate in candidates:
        pair = candidate[0]
        if pair[0] in used or pair[1] in used:
            continue
        used.update(pair)
        selected.append(candidate)
    pairs = torch.tensor(
        [pair for pair, _, _, _ in selected], dtype=torch.long
    ).reshape(-1, 2)
    return pairs, {
        **certified_report,
        "schema": "lafgs_v6_feedback_association_repair_selection",
        "version": 2,
        "feedback_certified_pair_count": int(certified_pairs.shape[0]),
        "global_spatial_pair_count": int(keep.numel()),
        "global_threshold_pair_count": int(spatial.shape[0]),
        "global_observation_conflict_pair_count": observation_conflict_count,
        "global_eligible_pair_count": len(candidates),
        "selected_pair_count": len(selected),
        "selected_pairs": pairs,
        "selected_pair_evidence_counts": torch.tensor(
            [value[1] for value in selected], dtype=torch.long
        ),
        "selected_pair_descriptor_similarities": torch.tensor(
            [value[2] for value in selected], dtype=torch.float32
        ),
        "selected_pair_xyz_distances_m": torch.tensor(
            [value[3] for value in selected], dtype=torch.float32
        ),
        "selection_uses_validation_feedback": False,
        "feedback_certified_rule_deployed_globally": True,
    }


@torch.no_grad()
def association_repair_proposal(
    state: dict,
    observations: ObservationProvider,
    feedback: dict,
    *,
    training_query_indices: Sequence[int] | torch.Tensor,
    training_split_sha256: str,
    lineage: dict,
    minimum_descriptor_similarity: float = 0.9,
    maximum_xyz_distance_m: float = 0.02,
    minimum_query_evidence: int = 5,
    minimum_views: int = 3,
    minimum_view_bins: int = 2,
    maximum_reprojection_px: float = 2.0,
    deploy_rule_globally: bool = False,
) -> tuple[dict, dict]:
    """Merge certified fragmented pairs and retriangulate them from rays."""

    pairs, selection = select_association_repair_pairs(
        state,
        feedback,
        training_query_indices=training_query_indices,
        minimum_descriptor_similarity=minimum_descriptor_similarity,
        maximum_xyz_distance_m=maximum_xyz_distance_m,
        minimum_query_evidence=minimum_query_evidence,
    )
    if deploy_rule_globally:
        pairs, selection = deploy_association_repair_rule_globally(
            state,
            pairs,
            selection,
            minimum_descriptor_similarity=minimum_descriptor_similarity,
            maximum_xyz_distance_m=maximum_xyz_distance_m,
        )
    if pairs.numel() == 0:
        raise ValueError("association repair selected no fragmented Track pair")
    csr = state["projective_anchor_observations"]
    offsets = torch.as_tensor(csr["observation_offsets"]).long()
    source_query = torch.as_tensor(csr["query_indices"]).long()
    source_keypoint = torch.as_tensor(csr["keypoint_indices"]).long()
    track = []
    query = []
    keypoint = []
    confidence = []
    matchability = torch.as_tensor(state["anchor_matchability"]).float()
    evidence_counts = torch.as_tensor(selection["selected_pair_evidence_counts"]).float()
    for component, pair in enumerate(pairs.tolist()):
        positions = torch.cat(
            [
                torch.arange(offsets[anchor], offsets[anchor + 1], dtype=torch.long)
                for anchor in pair
            ]
        )
        track.append(torch.full((positions.numel(),), component, dtype=torch.long))
        query.append(source_query[positions])
        keypoint.append(source_keypoint[positions])
        confidence.append(
            torch.full(
                (positions.numel(),),
                float(min(matchability[pair[0]], matchability[pair[1]])),
            )
        )
    association = {
        "schema": ASSOCIATION_GRAPH_SCHEMA,
        "version": 2,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "query_names": list(observations.names),
        "query_bins": torch.as_tensor(state["v6_mapping_query_bins"]).long(),
        "tracks": {
            "track_index": torch.cat(track),
            "query_index": torch.cat(query),
            "keypoint_index": torch.cat(keypoint),
            "confidence": torch.cat(confidence),
        },
        "diagnostics": {"track_count": int(pairs.shape[0])},
        "component_statistics": {
            "identity_reliability": (
                evidence_counts / evidence_counts.max().clamp_min(1.0)
            ).clamp(0.05, 1.0)
        },
    }
    repaired = reconstruct_projective_anchors(
        observations,
        association,
        minimum_views=int(minimum_views),
        minimum_view_bins=int(minimum_view_bins),
        maximum_reprojection_px=float(maximum_reprojection_px),
        parallel_workers=1,
        parallel_minimum_tracks=10**9,
    )
    repaired["candidate_kind"] = "feedback_association_repair"
    successful_components = torch.as_tensor(repaired["source_component_ids"]).long()
    removed = torch.unique(pairs[successful_components].reshape(-1), sorted=True)
    source_count = int(torch.as_tensor(state["anchor_ids"]).numel())
    retained = torch.arange(source_count, dtype=torch.long)
    retained = retained[~torch.isin(retained, removed)]
    retained_state = subset_projective_anchor_map(state, retained)
    merged = merge_projective_candidates(
        [projective_candidates_from_map(retained_state), repaired]
    )
    proposal = materialize_projective_anchor_map(merged, lineage=lineage)
    training = torch.unique(
        torch.as_tensor(training_query_indices, dtype=torch.long).reshape(-1),
        sorted=True,
    )
    all_queries = torch.arange(len(observations), dtype=torch.long)
    validation = all_queries[~torch.isin(all_queries, training)]
    report = {
        **selection,
        "schema": "lafgs_v6_feedback_association_repair",
        "version": 1,
        "successful_pair_count": int(successful_components.numel()),
        "rejected_pair_count": int(pairs.shape[0] - successful_components.numel()),
        "removed_source_anchor_count": int(removed.numel()),
        "output_repaired_anchor_count": int(successful_components.numel()),
        "output_anchor_count": int(torch.as_tensor(proposal["anchor_ids"]).numel()),
        "training_query_indices": training,
        "round_training_query_indices": training,
        "validation_query_indices": validation,
        "round_validation_query_indices": validation,
        "target_query_indices": torch.empty(0, dtype=torch.long),
        "round_target_query_indices": torch.empty(0, dtype=torch.long),
        "training_query_registry_explicit": True,
        "training_split_artifact_sha256s": [str(training_split_sha256)],
        "round_training_split_artifact_sha256": str(training_split_sha256),
        "validation_queries_used_as_edge_evidence": False,
        "validation_observations_may_be_retriangulated_only_after_training_edge_selection": True,
        "final_xyz_source": "fixed_camera_robust_ray_triangulation",
        "descriptor_source": "view_balanced_robust_fusion_after_pair_merge",
        "global_support_repair": False,
        "local_pairwise_feedback_repair": True,
        "feedback_certified_rule_deployed_globally": bool(deploy_rule_globally),
    }
    proposal["v6_reconstruction_distillation"] = report
    return proposal, report
