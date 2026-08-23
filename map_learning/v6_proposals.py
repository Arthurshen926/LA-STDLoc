"""Independent descriptor, selection, and reconstruction proposal arms for V6."""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F

from common.v6_contracts import FEEDBACK_SCHEMA, require_schema
from evidence.observation_provider import ObservationProvider
from topology.layered_sufficiency import select_layered_sufficiency
from topology.v6_anchor_map import subset_projective_anchor_map


def _bounded_descriptor_bank(
    base: torch.Tensor,
    residual: torch.Tensor,
    trust_region: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    base = F.normalize(base, dim=1)
    tangent = residual - (residual * base).sum(1, keepdim=True) * base
    norm = torch.linalg.norm(tangent, dim=1, keepdim=True)
    tangent = tangent * torch.clamp(
        float(trust_region) / norm.clamp_min(1e-8), max=1.0
    )
    return F.normalize(base + tangent, dim=1), tangent


def descriptor_loss_proposal(
    state: dict,
    observations: ObservationProvider,
    feedback: dict,
    *,
    trust_region: float = 0.05,
    margin: float = 0.05,
    temperature: float = 0.04,
    learning_rate: float = 0.02,
    epochs: int = 5,
    batch_size: int = 8192,
    maximum_triplets_per_query: int = 128,
    clean_fraction: float = 0.25,
    clean_weight: float = 0.25,
    trust_weight: float = 0.1,
    device: str = "cuda",
) -> dict:
    """Train bounded map-side residuals from actual LOO ranking triplets.

    Query descriptors and the online frontend remain frozen.  Incorrect
    winners provide swap/miss supervision, while a deterministic clean subset
    preserves already-correct margins.  The residual is stored separately so
    query-local LOO can reapply it after rebuilding an Anchor descriptor.
    """

    require_schema(feedback, FEEDBACK_SCHEMA, label="self-localization feedback")
    if list(feedback["query_names"]) != list(observations.names):
        raise ValueError("feedback and observation registries differ")
    if not 0.0 < float(trust_region) <= 0.2:
        raise ValueError("descriptor trust region must lie in (0,0.2]")
    if int(epochs) < 1 or int(batch_size) < 1 or int(maximum_triplets_per_query) < 1:
        raise ValueError("descriptor training schedule must be positive")
    if not 0.0 <= float(clean_fraction) <= 1.0:
        raise ValueError("clean triplet fraction must lie in [0,1]")

    features = F.normalize(torch.as_tensor(state["anchor_features"]).float(), dim=1)
    observation_features = F.normalize(
        torch.as_tensor(state.get("anchor_observation_features", features)).float(),
        dim=1,
    )
    initial_residual = torch.as_tensor(
        state.get("anchor_descriptor_residual", torch.zeros_like(features))
    ).float()
    if observation_features.shape != features.shape or initial_residual.shape != features.shape:
        raise ValueError("descriptor base/residual rows do not align with the map")

    query_parts = []
    positive_parts = []
    negative_parts = []
    clean_parts = []
    selected_per_query = []
    clean_budget = int(round(int(maximum_triplets_per_query) * float(clean_fraction)))
    error_budget = int(maximum_triplets_per_query) - clean_budget
    for query_index, record in enumerate(feedback["records"]):
        triplets = torch.as_tensor(record.get("descriptor_triplets", ())).long().reshape(-1, 4)
        if triplets.numel() == 0:
            continue
        view = observations.build_view(query_index)
        rows, positive, negative, clean = triplets.T
        valid = (
            (rows >= 0)
            & (rows < view.descriptors.shape[0])
            & (positive >= 0)
            & (positive < features.shape[0])
            & (negative >= 0)
            & (negative < features.shape[0])
            & (positive != negative)
        )
        rows, positive, negative, clean = (
            value[valid] for value in (rows, positive, negative, clean)
        )
        if rows.numel() == 0:
            continue
        descriptors = F.normalize(view.descriptors[rows].float(), dim=1)
        current_margin = (
            (descriptors * features[positive]).sum(1)
            - (descriptors * features[negative]).sum(1)
        )
        error_rows = torch.nonzero(clean == 0, as_tuple=False).reshape(-1)
        clean_rows = torch.nonzero(clean != 0, as_tuple=False).reshape(-1)
        error_rows = error_rows[
            torch.argsort(current_margin[error_rows], stable=True)
        ][:error_budget]
        clean_rows = clean_rows[
            torch.argsort(current_margin[clean_rows], stable=True)
        ][:clean_budget]
        chosen = torch.cat((error_rows, clean_rows))
        if chosen.numel() == 0:
            continue
        query_parts.append(descriptors[chosen])
        positive_parts.append(positive[chosen])
        negative_parts.append(negative[chosen])
        clean_parts.append(clean[chosen].bool())
        selected_per_query.append(int(chosen.numel()))
    if not query_parts:
        raise ValueError("feedback contains no trainable descriptor triplets")

    query = torch.cat(query_parts)
    positive = torch.cat(positive_parts)
    negative = torch.cat(negative_parts)
    clean = torch.cat(clean_parts)
    active = torch.unique(torch.cat((positive, negative)), sorted=True)
    lookup = torch.full((features.shape[0],), -1, dtype=torch.long)
    lookup[active] = torch.arange(active.numel())
    positive_local = lookup[positive]
    negative_local = lookup[negative]
    train_device = torch.device(device)
    base_active = observation_features[active].to(train_device)
    residual = torch.nn.Parameter(initial_residual[active].to(train_device))
    optimizer = torch.optim.Adam([residual], lr=float(learning_rate))
    generator = torch.Generator().manual_seed(2026)

    def full_loss(bank: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        q = query[rows].to(train_device)
        positive_score = (q * bank[positive_local[rows].to(train_device)]).sum(1)
        negative_score = (q * bank[negative_local[rows].to(train_device)]).sum(1)
        weight = torch.where(
            clean[rows].to(train_device),
            torch.full_like(positive_score, float(clean_weight)),
            torch.ones_like(positive_score),
        )
        ranking = F.softplus(
            (float(margin) + negative_score - positive_score)
            / max(float(temperature), 1e-6)
        ) * float(temperature)
        return (ranking * weight).sum() / weight.sum().clamp_min(1e-8)

    with torch.no_grad():
        initial_bank, _ = _bounded_descriptor_bank(
            base_active, residual, trust_region
        )
        initial_loss = float(
            full_loss(initial_bank, torch.arange(query.shape[0])).cpu()
        )
    for _ in range(int(epochs)):
        order = torch.randperm(query.shape[0], generator=generator)
        for start in range(0, query.shape[0], int(batch_size)):
            rows = order[start : start + int(batch_size)]
            bank, tangent = _bounded_descriptor_bank(
                base_active, residual, trust_region
            )
            ranking_loss = full_loss(bank, rows)
            regularizer = tangent.square().sum(1).mean()
            loss = ranking_loss + float(trust_weight) * regularizer
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        trained_bank, trained_residual = _bounded_descriptor_bank(
            base_active, residual, trust_region
        )
        final_loss = float(
            full_loss(trained_bank, torch.arange(query.shape[0])).cpu()
        )
    output_features = features.clone()
    output_features[active] = trained_bank.cpu()
    output_residual = initial_residual.clone()
    output_residual[active] = trained_residual.cpu()
    proposal = dict(state)
    proposal["anchor_observation_features"] = observation_features
    proposal["anchor_descriptor_residual"] = output_residual
    proposal["anchor_features"] = output_features
    proposal["v6_descriptor_distillation"] = {
        "schema": "lafgs_v6_counterfactual_descriptor_loss_distillation",
        "version": 2,
        "updated_anchor_rows": active,
        "triplet_count": int(query.shape[0]),
        "error_triplet_count": int((~clean).sum()),
        "clean_triplet_count": int(clean.sum()),
        "queries_with_triplets": len(selected_per_query),
        "initial_ranking_loss": initial_loss,
        "final_ranking_loss": final_loss,
        "margin": float(margin),
        "temperature": float(temperature),
        "trust_region": float(trust_region),
        "epochs": int(epochs),
        "online_model_added": False,
        "query_encoder_changed": False,
    }
    return proposal


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
    layers = {
        name: [defaultdict(set) for _ in range(count)]
        for name in ("visibility", "detectability", "matching")
    }
    information: list[dict[int, torch.Tensor]] = [dict() for _ in range(count)]
    for query_index, record in enumerate(feedback["records"]):
        for anchor in torch.as_tensor(record["visible_anchor_ids"]).long().tolist():
            layers["visibility"][anchor][query_index].add(anchor)
        for row, anchor in torch.as_tensor(record["detectable_pairs"]).long().tolist():
            layers["detectability"][anchor][query_index].add(row)
        for row, anchor in torch.as_tensor(record["matching_pairs"]).long().tolist():
            layers["matching"][anchor][query_index].add(row)
        pose_ids = torch.as_tensor(
            record.get("clean_inlier_pose_anchor_ids", ())
        ).long()
        pose_information = torch.as_tensor(
            record.get("clean_inlier_pose_information", ()), dtype=torch.float64
        ).reshape(-1, 6, 6)
        if pose_ids.numel() != pose_information.shape[0]:
            raise ValueError("pose information and Anchor IDs do not align")
        for anchor, contribution in zip(pose_ids.tolist(), pose_information):
            previous = information[anchor].get(
                query_index, torch.zeros((6, 6), dtype=torch.float64)
            )
            information[anchor][query_index] = previous + contribution
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
