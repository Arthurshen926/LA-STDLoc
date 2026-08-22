"""Independent descriptor, selection, and reconstruction proposal arms for V6."""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F

from common.v6_contracts import FEEDBACK_SCHEMA, require_schema
from evidence.observation_provider import ObservationProvider
from topology.layered_sufficiency import select_layered_sufficiency
from topology.v6_anchor_map import subset_projective_anchor_map


@torch.no_grad()
def descriptor_only_proposal(
    state: dict,
    observations: ObservationProvider,
    feedback: dict,
    *,
    trust_region: float = 0.05,
) -> dict:
    """Counterfactual map-vector update; geometry and topology stay exact."""

    require_schema(feedback, FEEDBACK_SCHEMA, label="self-localization feedback")
    if list(feedback["query_names"]) != list(observations.names):
        raise ValueError("feedback and observation registries differ")
    if not 0.0 < float(trust_region) <= 0.1:
        raise ValueError("descriptor trust region must lie in (0,0.1]")
    features = F.normalize(torch.as_tensor(state["anchor_features"]).float(), dim=1)
    positive: dict[int, list[torch.Tensor]] = defaultdict(list)
    negative: dict[int, list[torch.Tensor]] = defaultdict(list)
    for query_index, record in enumerate(feedback["records"]):
        view = observations.build_view(query_index)
        rows = torch.as_tensor(record["query_rows"]).long()
        winners = torch.as_tensor(record["winner_anchor_ids"]).long()
        if rows.shape != winners.shape:
            raise ValueError("feedback query rows and winners differ")
        descriptor = F.normalize(view.descriptors[rows].float(), dim=1)
        correct_rows = {
            int(row): int(anchor)
            for row, anchor in torch.as_tensor(record["matching_pairs"]).long().tolist()
        }
        for row, anchor in correct_rows.items():
            if int(winners[row]) == anchor:
                positive[anchor].append(descriptor[row])
        inlier_rows = torch.as_tensor(record["inlier_query_rows"]).long()
        inlier_clean = torch.as_tensor(record["inlier_clean_mask"]).bool()
        for row, clean in zip(inlier_rows.tolist(), inlier_clean.tolist()):
            if not clean:
                negative[int(winners[row])].append(descriptor[row])
    output = features.clone()
    updated = []
    for anchor in sorted(set(positive) | set(negative)):
        base = features[anchor]
        direction = torch.zeros_like(base)
        if positive[anchor]:
            direction += F.normalize(torch.stack(positive[anchor]).mean(0), dim=0)
        if negative[anchor]:
            wrong = F.normalize(torch.stack(negative[anchor]).mean(0), dim=0)
            direction -= wrong - (wrong @ base) * base
        tangent = direction - (direction @ base) * base
        norm = torch.linalg.norm(tangent)
        if float(norm) == 0.0:
            continue
        residual = tangent * min(float(trust_region) / float(norm), 1.0)
        output[anchor] = F.normalize(base + residual, dim=0)
        updated.append(anchor)
    proposal = dict(state)
    proposal["anchor_features"] = output
    proposal["v6_descriptor_distillation"] = {
        "schema": "lafgs_v6_counterfactual_descriptor_distillation",
        "version": 1,
        "updated_anchor_rows": torch.tensor(updated, dtype=torch.long),
        "trust_region": float(trust_region),
        "query_local_feedback": True,
        "geometry_changed": False,
        "selection_changed": False,
        "online_model_added": False,
    }
    return proposal


def selection_only_proposal(
    state: dict,
    feedback: dict,
    *,
    maximum_anchors: int,
    matching_target: int,
    pose_logdet_target: float,
) -> tuple[dict, dict]:
    """Hierarchical visibility→detectability→matching→pose selection arm."""

    require_schema(feedback, FEEDBACK_SCHEMA, label="self-localization feedback")
    count = int(torch.as_tensor(state["anchor_ids"]).numel())
    query_count = len(feedback["records"])
    layers = {
        name: [defaultdict(set) for _ in range(count)]
        for name in ("visibility", "detectability", "matching")
    }
    information = torch.zeros((query_count, count, 6, 6), dtype=torch.float64)
    for query_index, record in enumerate(feedback["records"]):
        for anchor in torch.as_tensor(record["visible_anchor_ids"]).long().tolist():
            layers["visibility"][anchor][query_index].add(anchor)
        for row, anchor in torch.as_tensor(record["detectable_pairs"]).long().tolist():
            layers["detectability"][anchor][query_index].add(row)
        for row, anchor in torch.as_tensor(record["matching_pairs"]).long().tolist():
            layers["matching"][anchor][query_index].add(row)
        for anchor in torch.as_tensor(record["clean_inlier_anchor_ids"]).long().tolist():
            information[query_index, anchor] += torch.eye(6, dtype=torch.float64)
    candidate_edges = {
        name: [
            {query: tuple(sorted(rows)) for query, rows in candidate.items()}
            for candidate in layers[name]
        ]
        for name in ("visibility", "detectability", "matching")
    }
    result = select_layered_sufficiency(
        layer_edges=candidate_edges,
        reliability=torch.as_tensor(state["anchor_matchability"]).float(),
        pose_information=information,
        matching_target=int(matching_target),
        pose_logdet_target=float(pose_logdet_target),
        maximum_anchors=int(maximum_anchors),
    )
    selected = torch.sort(result["selected_anchor_rows"]).values
    proposal = subset_projective_anchor_map(state, selected)
    proposal["v6_selection_distillation"] = {
        "schema": "lafgs_v6_layered_sufficiency_selection",
        "version": 1,
        "selected_source_rows": selected,
        "hierarchy": ["visibility", "detectability", "matching", "pose"],
        "weighted_heuristic_sum": False,
        "report": result,
    }
    return proposal, result
