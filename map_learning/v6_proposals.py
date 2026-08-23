"""Independent descriptor, selection, and reconstruction proposal arms for V6."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import math

import torch
import torch.nn.functional as F

from common.v6_contracts import (
    DESCRIPTOR_CLEAN_LABEL_SEMANTICS,
    DESCRIPTOR_POSE_WEIGHT_SEMANTICS,
    FEEDBACK_SCHEMA,
    require_schema,
)
from evidence.observation_provider import ObservationProvider
from evidence.projective_loo import LeaveOneQueryOutProjectiveMap
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
    tangent = tangent * torch.clamp(float(trust_region) / norm.clamp_min(1e-8), max=1.0)
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
    pose_critical_weight: float = 0.0,
    tail_query_weight: float = 0.0,
    training_query_indices: torch.Tensor | Sequence[int] | None = None,
    eligible_failure_layers: Sequence[str] = ("L3",),
    device: str = "cuda",
) -> dict:
    """Train bounded map-side residuals from actual LOO ranking triplets.

    Query descriptors and the online frontend remain frozen.  Incorrect
    winners provide swap/miss supervision, while a deterministic clean subset
    preserves already-correct margins.  Every score is evaluated against the
    same query-local LOO Anchor base that generated its feedback triplet.  A
    single map-side residual remains shared across queries and is stored
    separately so deployment and later LOO replay use the same update.
    """

    require_schema(feedback, FEEDBACK_SCHEMA, label="self-localization feedback")
    if list(feedback["query_names"]) != list(observations.names):
        raise ValueError("feedback and observation registries differ")
    if not 0.0 < float(trust_region) <= 0.2:
        raise ValueError("descriptor trust region must lie in (0,0.2]")
    if float(margin) < 0.0 or float(temperature) <= 0.0:
        raise ValueError("descriptor margin/temperature must be non-negative/positive")
    if float(learning_rate) <= 0.0:
        raise ValueError("descriptor learning rate must be positive")
    if int(epochs) < 1 or int(batch_size) < 1 or int(maximum_triplets_per_query) < 1:
        raise ValueError("descriptor training schedule must be positive")
    if not 0.0 <= float(clean_fraction) <= 1.0:
        raise ValueError("clean triplet fraction must lie in [0,1]")
    if (
        float(clean_weight) < 0.0
        or float(trust_weight) < 0.0
        or float(pose_critical_weight) < 0.0
        or float(tail_query_weight) < 0.0
    ):
        raise ValueError("descriptor loss weights must be non-negative")
    if (
        feedback.get("descriptor_triplet_pose_weight_semantics")
        != DESCRIPTOR_POSE_WEIGHT_SEMANTICS
    ):
        raise ValueError("descriptor feedback pose-weight semantics differ")
    if (
        feedback.get("descriptor_triplet_clean_semantics")
        != DESCRIPTOR_CLEAN_LABEL_SEMANTICS
    ):
        raise ValueError("descriptor feedback clean-label semantics differ")

    query_count = len(feedback["records"])
    training_registry_explicit = training_query_indices is not None
    round_training_queries = (
        torch.arange(query_count, dtype=torch.long)
        if training_query_indices is None
        else torch.as_tensor(training_query_indices, dtype=torch.long).reshape(-1)
    )
    round_training_queries = torch.unique(round_training_queries, sorted=True)
    if round_training_queries.numel() == 0 or (
        int(round_training_queries.min()) < 0
        or int(round_training_queries.max()) >= query_count
    ):
        raise ValueError("descriptor training query registry is empty or invalid")
    training_query_set = set(round_training_queries.tolist())
    eligible_layers = {str(layer) for layer in eligible_failure_layers}
    if not eligible_layers:
        raise ValueError("descriptor eligible failure layers must be non-empty")

    features = F.normalize(torch.as_tensor(state["anchor_features"]).float(), dim=1)
    observation_features = F.normalize(
        torch.as_tensor(state.get("anchor_observation_features", features)).float(),
        dim=1,
    )
    initial_residual = torch.as_tensor(
        state.get("anchor_descriptor_residual", torch.zeros_like(features))
    ).float()
    if (
        observation_features.shape != features.shape
        or initial_residual.shape != features.shape
    ):
        raise ValueError("descriptor base/residual rows do not align with the map")
    if "anchor_observation_features" not in state and bool(
        initial_residual.abs().max() > 0
    ):
        raise ValueError(
            "learned descriptor maps require the pre-residual observation bank"
        )
    feedback_policies = {
        str(record.get("affected_anchor_policy", "rebuild"))
        for record in feedback["records"]
    }
    if len(feedback_policies) != 1:
        raise ValueError("descriptor feedback mixes affected-Anchor LOO policies")
    loo_policy = next(iter(feedback_policies))
    if loo_policy != "rebuild":
        raise ValueError(
            "descriptor training requires exact query-local Anchor rebuild; "
            "purge feedback is diagnostic-only"
        )
    loo_replay = LeaveOneQueryOutProjectiveMap(
        state,
        observations,
        affected_anchor_policy=loo_policy,
    )

    def query_local_observation_bank(
        query_index: int,
        record: dict,
        anchor_rows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return sparse LOO bases and the rows rebuilt for one query."""

        active_rows = torch.unique(anchor_rows.long(), sorted=True)
        excluded_queries = torch.as_tensor(
            record.get("excluded_query_indices", (query_index,)),
            dtype=torch.long,
        ).reshape(-1)
        update = loo_replay.query_update(
            query_index,
            excluded_queries=excluded_queries,
            requested_anchor_rows=active_rows,
        )
        local_base = observation_features[active_rows].clone()
        affected_rows = torch.as_tensor(update["anchor_rows"]).long()
        if affected_rows.numel():
            affected_local = torch.searchsorted(active_rows, affected_rows)
            if not torch.equal(active_rows[affected_local], affected_rows):
                raise RuntimeError("sparse LOO replay returned an unrequested Anchor")
            valid = torch.as_tensor(update["valid"]).bool()
            if not bool(valid.all()):
                invalid = affected_rows[~valid]
                raise ValueError(
                    "descriptor triplet references a query-local LOO-invalid "
                    f"Anchor: {invalid[:8].tolist()}"
                )
            local_base[affected_local] = F.normalize(
                torch.as_tensor(update["anchor_observation_features"]).float(),
                dim=1,
            )
        return active_rows, local_base, affected_rows

    prior_report = state.get("v6_descriptor_distillation")
    prior_training_queries = torch.empty(0, dtype=torch.long)
    prior_selected_queries = torch.empty(0, dtype=torch.long)
    prior_updated_anchors = torch.empty(0, dtype=torch.long)
    prior_dependency_present = False
    if isinstance(prior_report, dict):
        prior_dependency_present = True
        prior_training_queries = torch.as_tensor(
            prior_report.get(
                "training_query_indices",
                prior_report.get("selected_query_indices", ()),
            ),
            dtype=torch.long,
        ).reshape(-1)
        prior_selected_queries = torch.as_tensor(
            prior_report.get("selected_query_indices", prior_training_queries),
            dtype=torch.long,
        ).reshape(-1)
        prior_updated_anchors = torch.as_tensor(
            prior_report.get("updated_anchor_rows", ()), dtype=torch.long
        ).reshape(-1)
    elif bool(initial_residual.abs().max() > 0):
        # Legacy learned maps did not persist query dependencies.  Treat all
        # mapping queries as dependencies instead of manufacturing a holdout.
        prior_dependency_present = True
        prior_training_queries = torch.arange(query_count, dtype=torch.long)
        prior_selected_queries = prior_training_queries
        prior_updated_anchors = torch.nonzero(
            initial_residual.abs().sum(1) > 0, as_tuple=False
        ).reshape(-1)
    for label, rows in (
        ("prior training", prior_training_queries),
        ("prior selected", prior_selected_queries),
    ):
        if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= query_count):
            raise ValueError(f"{label} query registry is invalid")
    if prior_updated_anchors.numel() and (
        int(prior_updated_anchors.min()) < 0
        or int(prior_updated_anchors.max()) >= features.shape[0]
    ):
        raise ValueError("prior updated Anchor registry is invalid")

    query_parts = []
    positive_parts = []
    negative_parts = []
    clean_parts = []
    pose_weight_parts = []
    harmful_parts = []
    selected_query_parts = []
    positive_loo_base_parts = []
    negative_loo_base_parts = []
    affected_pair_key_parts = []
    selected_per_query = []
    clean_budget = int(round(int(maximum_triplets_per_query) * float(clean_fraction)))
    error_budget = int(maximum_triplets_per_query) - clean_budget
    selected_query_indices = []
    finite_training_te = torch.tensor(
        [
            float(record["te_cm"])
            for query_index, record in enumerate(feedback["records"])
            if query_index in training_query_set
            and record.get("te_cm") is not None
            and math.isfinite(float(record["te_cm"]))
        ],
        dtype=torch.float64,
    )
    tail_te_threshold_cm = (
        None
        if finite_training_te.numel() == 0
        else float(torch.quantile(finite_training_te, 0.95))
    )
    tail_query_set = {
        query_index
        for query_index, record in enumerate(feedback["records"])
        if query_index in training_query_set
        and tail_te_threshold_cm is not None
        and record.get("te_cm") is not None
        and float(record["te_cm"]) >= tail_te_threshold_cm
    }
    for query_index, record in enumerate(feedback["records"]):
        if query_index not in training_query_set:
            continue
        triplets = (
            torch.as_tensor(record.get("descriptor_triplets", ())).long().reshape(-1, 4)
        )
        if triplets.numel() == 0:
            continue
        if record.get("descriptor_identity_supervision_available") is not True:
            raise ValueError("descriptor triplets lack rebuild identity supervision")
        if bool(((triplets[:, 3] != 0) & (triplets[:, 3] != 1)).any()):
            raise ValueError("descriptor triplet clean labels must be binary")
        pose_weights = torch.as_tensor(
            record.get("descriptor_triplet_pose_weights", ()), dtype=torch.float32
        ).reshape(-1)
        harmful = torch.as_tensor(
            record.get("descriptor_triplet_harmful_inlier_mask", ()), dtype=torch.bool
        ).reshape(-1)
        if pose_weights.numel() != triplets.shape[0] or harmful.numel() != triplets.shape[0]:
            raise ValueError("descriptor triplet supervision rows are not aligned")
        if not bool(torch.isfinite(pose_weights).all()) or bool(
            ((pose_weights < 0.0) | (pose_weights > 1.0)).any()
        ):
            raise ValueError("descriptor triplet pose weights must be finite in [0,1]")

        def pair_registry(key: str) -> set[tuple[int, int]]:
            pairs = torch.as_tensor(record.get(key, ()), dtype=torch.long).reshape(-1, 2)
            return {(int(row), int(anchor)) for row, anchor in pairs.tolist()}

        exact_pairs = pair_registry("exact_identity_positive_pairs")
        ignored_pairs = (
            pair_registry("identity_inactive_pairs")
            | pair_registry("identity_projective_incompatible_pairs")
            | pair_registry("projective_compatible_ambiguous_pairs")
        )
        for row, positive_anchor, negative_anchor, _ in triplets.tolist():
            if (row, positive_anchor) not in exact_pairs:
                raise ValueError(
                    "descriptor triplet positive lacks exact active identity"
                )
            if (row, negative_anchor) in exact_pairs or (
                row,
                negative_anchor,
            ) in ignored_pairs:
                raise ValueError("descriptor triplet negative is not legally negative")

        # Formal V6 feedback is multi-label.  Normal descriptor ranking is an
        # L3 operation.  A pose-critical L4 row may additionally enter only
        # when its fixed-hypothesis counterfactual weight is positive.
        record_layers = record.get("failure_layers")
        layer_set = (
            None if record_layers is None else {str(layer) for layer in record_layers}
        )
        regular_eligible = layer_set is None or bool(eligible_layers & layer_set)
        pose_eligible = bool(
            float(pose_critical_weight) > 0.0
            and layer_set is not None
            and "L4" in layer_set
            and (pose_weights > 0).any()
        )
        if not regular_eligible and not pose_eligible:
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
        if not bool(valid.all()):
            raise ValueError("descriptor triplet contains an invalid row or Anchor")
        descriptors = F.normalize(view.descriptors[rows].float(), dim=1)
        candidate_active, candidate_loo_base, affected_rows = (
            query_local_observation_bank(
                query_index,
                record,
                torch.cat((positive, negative)),
            )
        )
        positive_candidate_local = torch.searchsorted(candidate_active, positive)
        negative_candidate_local = torch.searchsorted(candidate_active, negative)
        positive_loo_base = candidate_loo_base[positive_candidate_local]
        negative_loo_base = candidate_loo_base[negative_candidate_local]
        _, positive_map_residual = _bounded_descriptor_bank(
            observation_features[positive],
            initial_residual[positive],
            trust_region,
        )
        _, negative_map_residual = _bounded_descriptor_bank(
            observation_features[negative],
            initial_residual[negative],
            trust_region,
        )
        positive_current, _ = _bounded_descriptor_bank(
            positive_loo_base,
            positive_map_residual,
            trust_region,
        )
        negative_current, _ = _bounded_descriptor_bank(
            negative_loo_base,
            negative_map_residual,
            trust_region,
        )
        current_margin = (descriptors * positive_current).sum(1) - (
            descriptors * negative_current
        ).sum(1)
        clean = current_margin >= float(margin)
        error_rows = torch.nonzero(clean == 0, as_tuple=False).reshape(-1)
        clean_rows = torch.nonzero(clean != 0, as_tuple=False).reshape(-1)
        error_rows = error_rows[torch.argsort(current_margin[error_rows], stable=True)]
        if float(pose_critical_weight) > 0.0 and error_rows.numel():
            error_rows = error_rows[
                torch.argsort(
                    pose_weights[error_rows], descending=True, stable=True
                )
            ]
        error_rows = error_rows[:error_budget]
        clean_rows = clean_rows[torch.argsort(current_margin[clean_rows], stable=True)][
            :clean_budget
        ]
        chosen = torch.cat((error_rows, clean_rows))
        if chosen.numel() == 0:
            continue
        query_parts.append(descriptors[chosen])
        positive_parts.append(positive[chosen])
        negative_parts.append(negative[chosen])
        clean_parts.append(clean[chosen].bool())
        pose_weight_parts.append(pose_weights[chosen])
        harmful_parts.append(harmful[chosen])
        selected_query_parts.append(torch.full_like(positive[chosen], query_index))
        positive_loo_base_parts.append(positive_loo_base[chosen])
        negative_loo_base_parts.append(negative_loo_base[chosen])
        selected_anchors = torch.unique(
            torch.cat((positive[chosen], negative[chosen])), sorted=True
        )
        selected_affected = affected_rows[torch.isin(affected_rows, selected_anchors)]
        if selected_affected.numel():
            affected_pair_key_parts.append(
                query_index * features.shape[0] + selected_affected
            )
        selected_per_query.append(int(chosen.numel()))
        selected_query_indices.append(query_index)
    if not query_parts:
        raise ValueError("feedback contains no trainable descriptor triplets")

    round_selected_queries = torch.tensor(selected_query_indices, dtype=torch.long)
    cumulative_training_queries = torch.unique(
        torch.cat((prior_training_queries, round_training_queries)), sorted=True
    )
    cumulative_selected_queries = torch.unique(
        torch.cat((prior_selected_queries, round_selected_queries)), sorted=True
    )
    cumulative_registry_explicit = training_registry_explicit
    if prior_dependency_present:
        cumulative_registry_explicit = bool(
            isinstance(prior_report, dict)
            and prior_report.get("training_query_registry_explicit", False)
            and training_registry_explicit
        )

    query = torch.cat(query_parts)
    positive = torch.cat(positive_parts)
    negative = torch.cat(negative_parts)
    clean = torch.cat(clean_parts)
    pose_weight = torch.cat(pose_weight_parts)
    harmful = torch.cat(harmful_parts)
    selected_query = torch.cat(selected_query_parts)
    positive_loo_base = torch.cat(positive_loo_base_parts)
    negative_loo_base = torch.cat(negative_loo_base_parts)
    active = torch.unique(torch.cat((positive, negative)), sorted=True)
    base_triplet_weight = torch.where(
        clean,
        torch.full_like(pose_weight, float(clean_weight)),
        torch.ones_like(pose_weight),
    )
    raw_triplet_weight = base_triplet_weight * (
        1.0 + float(pose_critical_weight) * pose_weight
    )
    query_weight_sum = torch.zeros(query_count, dtype=raw_triplet_weight.dtype)
    query_weight_sum.scatter_add_(0, selected_query, raw_triplet_weight)
    normalized_triplet_weight = raw_triplet_weight / query_weight_sum[
        selected_query
    ].clamp_min(1e-8)
    selected_tail_query = torch.tensor(
        [int(query_index) in tail_query_set for query_index in selected_query.tolist()],
        dtype=torch.bool,
    )
    query_tail_factor = torch.where(
        selected_tail_query,
        torch.full_like(
            normalized_triplet_weight, 1.0 + float(tail_query_weight)
        ),
        torch.ones_like(normalized_triplet_weight),
    )
    training_triplet_weight = normalized_triplet_weight * query_tail_factor
    if not bool(torch.isfinite(training_triplet_weight).all()) or not bool(
        training_triplet_weight.sum() > 0
    ):
        raise ValueError("descriptor triplet weights have no finite positive mass")
    cumulative_updated_anchors = torch.unique(
        torch.cat((prior_updated_anchors, active.cpu())), sorted=True
    )
    lookup = torch.full((features.shape[0],), -1, dtype=torch.long)
    lookup[active] = torch.arange(active.numel())
    pair_keys = torch.cat(
        (
            selected_query * features.shape[0] + positive,
            selected_query * features.shape[0] + negative,
        )
    )
    pair_bases = torch.cat((positive_loo_base, negative_loo_base))
    pair_order = torch.argsort(pair_keys, stable=True)
    ordered_pair_keys = pair_keys[pair_order]
    first_pair = torch.ones_like(ordered_pair_keys, dtype=torch.bool)
    first_pair[1:] = ordered_pair_keys[1:] != ordered_pair_keys[:-1]
    unique_pair_keys = ordered_pair_keys[first_pair]
    loo_pair_base = pair_bases[pair_order[first_pair]]
    positive_pair = torch.searchsorted(
        unique_pair_keys,
        selected_query * features.shape[0] + positive,
    )
    negative_pair = torch.searchsorted(
        unique_pair_keys,
        selected_query * features.shape[0] + negative,
    )
    pair_anchor = torch.remainder(unique_pair_keys, features.shape[0])
    pair_anchor_local = lookup[pair_anchor]
    if bool((pair_anchor_local < 0).any()):
        raise RuntimeError("query-local LOO pair references an inactive Anchor")
    affected_pair_count = int(
        torch.unique(torch.cat(affected_pair_key_parts), sorted=True).numel()
        if affected_pair_key_parts
        else 0
    )
    train_device = torch.device(device)
    base_active = observation_features[active].to(train_device)
    loo_pair_base = loo_pair_base.to(train_device)
    pair_anchor_local = pair_anchor_local.to(train_device)
    positive_pair = positive_pair.to(train_device)
    negative_pair = negative_pair.to(train_device)
    residual = torch.nn.Parameter(initial_residual[active].to(train_device))
    # Adam's nominal learning rate is applied per coordinate.  Without this
    # normalization, a 256-D descriptor receives a first vector step about
    # sqrt(256) times larger than the requested trust-scale step.
    effective_coordinate_learning_rate = (
        float(learning_rate) / float(features.shape[1]) ** 0.5
    )
    optimizer = torch.optim.Adam([residual], lr=effective_coordinate_learning_rate)
    generator = torch.Generator().manual_seed(2026)

    def raw_tangent(value: torch.Tensor) -> torch.Tensor:
        return value - (value * base_active).sum(1, keepdim=True) * base_active

    def trust_penalty(tangent: torch.Tensor) -> torch.Tensor:
        # Normalize by the squared radius so ``trust_weight`` has a stable
        # meaning when the radius changes.  This must act on the *unclipped*
        # parameter tangent: the norm of a hard-clipped vector is constant
        # outside the ball and therefore supplies no radial gradient.
        return tangent.square().sum(1).mean() / float(trust_region) ** 2

    def full_loss(
        residual_value: torch.Tensor,
        rows: torch.Tensor,
    ) -> torch.Tensor:
        q = query[rows].to(train_device)
        device_rows = rows.to(train_device)
        _, bounded_map_residual = _bounded_descriptor_bank(
            base_active,
            residual_value,
            trust_region,
        )
        positive_pair_rows = positive_pair[device_rows]
        negative_pair_rows = negative_pair[device_rows]
        positive_bank, _ = _bounded_descriptor_bank(
            loo_pair_base[positive_pair_rows],
            bounded_map_residual[pair_anchor_local[positive_pair_rows]],
            trust_region,
        )
        negative_bank, _ = _bounded_descriptor_bank(
            loo_pair_base[negative_pair_rows],
            bounded_map_residual[pair_anchor_local[negative_pair_rows]],
            trust_region,
        )
        positive_score = (q * positive_bank).sum(1)
        negative_score = (q * negative_bank).sum(1)
        weight = training_triplet_weight[rows].to(train_device)
        ranking = F.softplus(
            (float(margin) + negative_score - positive_score)
            / max(float(temperature), 1e-6)
        ) * float(temperature)
        return (ranking * weight).sum() / weight.sum().clamp_min(1e-8)

    all_rows = torch.arange(query.shape[0])
    with torch.no_grad():
        _, initial_bounded_tangent = _bounded_descriptor_bank(
            base_active, residual, trust_region
        )
        initial_active_residual = residual.detach().clone()
        initial_raw_tangent = raw_tangent(residual)
        initial_loss = float(full_loss(residual, all_rows).cpu())
        initial_regularizer = float(trust_penalty(initial_raw_tangent).cpu())
        initial_objective = initial_loss + float(trust_weight) * initial_regularizer
        best_objective = initial_objective
        best_residual = residual.detach().clone()
        best_epoch = 0
        optimizer_last_objective = initial_objective
    for epoch in range(1, int(epochs) + 1):
        order = torch.randperm(query.shape[0], generator=generator)
        for start in range(0, query.shape[0], int(batch_size)):
            rows = order[start : start + int(batch_size)]
            ranking_loss = full_loss(residual, rows)
            regularizer = trust_penalty(raw_tangent(residual))
            loss = ranking_loss + float(trust_weight) * regularizer
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            epoch_ranking = float(full_loss(residual, all_rows).cpu())
            epoch_regularizer = float(trust_penalty(raw_tangent(residual)).cpu())
            optimizer_last_objective = (
                epoch_ranking + float(trust_weight) * epoch_regularizer
            )
            if (
                torch.isfinite(torch.tensor(optimizer_last_objective))
                and optimizer_last_objective < best_objective
            ):
                best_objective = optimizer_last_objective
                best_residual = residual.detach().clone()
                best_epoch = epoch
    with torch.no_grad():
        trained_bank, trained_residual = _bounded_descriptor_bank(
            base_active, best_residual, trust_region
        )
        final_raw_tangent = raw_tangent(best_residual)
        final_loss = float(full_loss(best_residual, all_rows).cpu())
        error_rows = torch.nonzero(~clean, as_tuple=False).reshape(-1)
        clean_rows = torch.nonzero(clean, as_tuple=False).reshape(-1)

        def split_loss(value: torch.Tensor, rows: torch.Tensor) -> float | None:
            return None if rows.numel() == 0 else float(full_loss(value, rows).cpu())

        initial_error_loss = split_loss(initial_active_residual, error_rows)
        final_error_loss = split_loss(best_residual, error_rows)
        initial_clean_loss = split_loss(initial_active_residual, clean_rows)
        final_clean_loss = split_loss(best_residual, clean_rows)
        final_regularizer = float(trust_penalty(final_raw_tangent).cpu())
        initial_mean_norm = float(
            torch.linalg.norm(initial_bounded_tangent, dim=1).mean().cpu()
        )
        final_mean_norm = float(torch.linalg.norm(trained_residual, dim=1).mean().cpu())
        trained_norm = torch.linalg.norm(trained_residual, dim=1)
        cap_tolerance = max(float(trust_region) * 1e-4, 1e-7)
        cap_hit_count = int(
            (trained_norm >= float(trust_region) - cap_tolerance).sum().cpu()
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
        "version": 3,
        "updated_anchor_rows": cumulative_updated_anchors,
        "round_updated_anchor_rows": active,
        "triplet_count": int(query.shape[0]),
        "error_triplet_count": int((~clean).sum()),
        "clean_triplet_count": int(clean.sum()),
        "harmful_inlier_triplet_count": int(harmful.sum()),
        "positive_pose_weight_triplet_count": int((pose_weight > 0).sum()),
        "pose_weight_sum": float(pose_weight.sum()),
        "pose_weight_mean": float(pose_weight.mean()),
        "pose_critical_error_triplet_count": int(
            ((~clean) & (pose_weight > 0)).sum()
        ),
        "queries_with_triplets": len(selected_per_query),
        "training_query_indices": cumulative_training_queries,
        "training_query_registry_explicit": cumulative_registry_explicit,
        "selected_query_indices": cumulative_selected_queries,
        "round_training_query_indices": round_training_queries,
        "round_selected_query_indices": round_selected_queries,
        "descriptor_training_round": int(
            prior_report.get("descriptor_training_round", 0)
            if isinstance(prior_report, dict)
            else 0
        )
        + 1,
        "eligible_failure_layers": sorted(eligible_layers),
        "initial_ranking_loss": initial_loss,
        "final_ranking_loss": final_loss,
        "initial_error_ranking_loss": initial_error_loss,
        "final_error_ranking_loss": final_error_loss,
        "initial_clean_ranking_loss": initial_clean_loss,
        "final_clean_ranking_loss": final_clean_loss,
        "initial_regularizer": initial_regularizer,
        "final_regularizer": final_regularizer,
        "trust_regularizer_normalized_by_radius": True,
        "initial_residual_mean_norm": initial_mean_norm,
        "final_residual_mean_norm": final_mean_norm,
        "initial_objective": initial_loss + float(trust_weight) * initial_regularizer,
        "final_objective": final_loss + float(trust_weight) * final_regularizer,
        "optimizer_last_objective": optimizer_last_objective,
        "best_epoch": best_epoch,
        "rolled_back_to_best_objective": best_epoch < int(epochs),
        "residual_cap_hit_count": cap_hit_count,
        "residual_cap_hit_fraction": cap_hit_count / max(int(active.numel()), 1),
        "updated_anchor_count": int(cumulative_updated_anchors.numel()),
        "round_updated_anchor_count": int(active.numel()),
        "query_local_loo_descriptor_training": True,
        "query_local_loo_base_source": "sparse_active_anchor_replay",
        "query_local_loo_affected_anchor_policy": loo_policy,
        "query_local_loo_pair_count": int(unique_pair_keys.numel()),
        "query_local_loo_affected_pair_count": affected_pair_count,
        "query_local_loo_dense_query_anchor_bank_materialized": False,
        "query_observations_excluded_from_training_anchor_bases": True,
        "margin": float(margin),
        "temperature": float(temperature),
        "trust_region": float(trust_region),
        "clean_fraction": float(clean_fraction),
        "clean_weight": float(clean_weight),
        "clean_labels_recomputed_from_query_local_current_margin": True,
        "trust_weight": float(trust_weight),
        "pose_critical_weight": float(pose_critical_weight),
        "tail_query_weight": float(tail_query_weight),
        "tail_te_quantile": 0.95,
        "tail_te_threshold_cm": tail_te_threshold_cm,
        "selected_tail_query_count": len(
            set(selected_query.tolist()) & tail_query_set
        ),
        "triplet_weights_normalized_within_query": True,
        "triplet_weight_sum_after_query_and_tail_weighting": float(
            training_triplet_weight.sum()
        ),
        "descriptor_triplet_pose_weight_semantics": (
            DESCRIPTOR_POSE_WEIGHT_SEMANTICS
        ),
        "pose_critical_l4_rows_require_positive_counterfactual_weight": True,
        "learning_rate": float(learning_rate),
        "effective_coordinate_learning_rate": effective_coordinate_learning_rate,
        "learning_rate_is_descriptor_vector_scale": True,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "maximum_triplets_per_query": int(maximum_triplets_per_query),
        "sampling_seed": 2026,
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
        "version": 2,
        "updated_anchor_rows": torch.tensor(updated, dtype=torch.long),
        "updated_anchor_count": len(updated),
        "training_query_indices": torch.arange(
            len(feedback["records"]), dtype=torch.long
        ),
        "selected_query_indices": torch.arange(
            len(feedback["records"]), dtype=torch.long
        ),
        "training_query_registry_explicit": False,
        "trust_region": float(trust_region),
        "query_local_feedback": True,
        "mapping_evaluation_role": "training_replay_not_query_descriptor_loo",
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
    visibility_target: int,
    detectability_target: int,
    matching_target: int,
    pose_logdet_target: float,
    pose_min_eigenvalue_target: float | None = None,
    training_query_indices: torch.Tensor | Sequence[int] | None = None,
) -> tuple[dict, dict]:
    """Hierarchical visibility→detectability→matching→pose selection arm."""

    require_schema(feedback, FEEDBACK_SCHEMA, label="self-localization feedback")
    count = int(torch.as_tensor(state["anchor_ids"]).numel())
    layers = {
        name: [defaultdict(set) for _ in range(count)]
        for name in ("visibility", "detectability", "matching")
    }
    information: list[dict[int, torch.Tensor]] = [dict() for _ in range(count)]
    query_count = len(feedback["records"])
    training_registry_explicit = training_query_indices is not None
    round_training_queries = (
        torch.arange(query_count, dtype=torch.long)
        if training_query_indices is None
        else torch.as_tensor(training_query_indices, dtype=torch.long).reshape(-1)
    )
    round_training_queries = torch.unique(round_training_queries, sorted=True)
    if round_training_queries.numel() == 0 or (
        int(round_training_queries.min()) < 0
        or int(round_training_queries.max()) >= query_count
    ):
        raise ValueError("selection training query registry is empty or invalid")
    prior_report = state.get("v6_selection_distillation")
    prior_training_queries = torch.empty(0, dtype=torch.long)
    prior_dependency_present = isinstance(prior_report, dict)
    if prior_dependency_present:
        prior_training_queries = torch.as_tensor(
            prior_report.get("training_query_indices", ()), dtype=torch.long
        ).reshape(-1)
        if prior_training_queries.numel() == 0:
            # Legacy selection maps did not serialize the dependency registry.
            prior_training_queries = torch.arange(query_count, dtype=torch.long)
        if (
            int(prior_training_queries.min()) < 0
            or int(prior_training_queries.max()) >= query_count
        ):
            raise ValueError("prior selection training query registry is invalid")
    cumulative_training_queries = torch.unique(
        torch.cat((prior_training_queries, round_training_queries)), sorted=True
    )
    cumulative_registry_explicit = training_registry_explicit
    if prior_dependency_present:
        cumulative_registry_explicit = bool(
            prior_report.get("training_query_registry_explicit", False)
            and training_registry_explicit
        )

    for local_query_index, source_query_index in enumerate(
        round_training_queries.tolist()
    ):
        record = feedback["records"][source_query_index]
        visible_ids = torch.as_tensor(record["visible_anchor_ids"]).long().reshape(-1)
        visible_cells = torch.as_tensor(
            record["visible_anchor_image_cells"]
        ).long().reshape(-1)
        if visible_ids.shape != visible_cells.shape:
            raise ValueError("visible Anchor IDs and image cells do not align")
        if visible_ids.numel() != torch.unique(visible_ids).numel():
            raise ValueError("visible Anchor IDs must be unique per query")
        if bool((visible_cells < 0).any()):
            raise ValueError("visibility image-cell IDs must be non-negative")
        if bool(((visible_ids < 0) | (visible_ids >= count)).any()):
            raise ValueError("visible Anchor IDs are outside the map registry")
        for anchor, image_cell in zip(visible_ids.tolist(), visible_cells.tolist()):
            layers["visibility"][anchor][local_query_index].add(image_cell)
        for row, anchor in torch.as_tensor(record["detectable_pairs"]).long().tolist():
            layers["detectability"][anchor][local_query_index].add(row)
        for row, anchor in torch.as_tensor(record["matching_pairs"]).long().tolist():
            layers["matching"][anchor][local_query_index].add(row)
        pose_ids = torch.as_tensor(
            record.get("clean_inlier_pose_anchor_ids", ())
        ).long()
        pose_information = torch.as_tensor(
            record.get("clean_inlier_pose_information", ()), dtype=torch.float64
        ).reshape(-1, 6, 6)
        if pose_ids.numel() != pose_information.shape[0]:
            raise ValueError("pose information and Anchor IDs do not align")
        if pose_ids.numel() != torch.unique(pose_ids).numel():
            raise ValueError("pose information must have one row per unique Anchor")
        if bool(((pose_ids < 0) | (pose_ids >= count)).any()):
            raise ValueError("pose-information Anchor IDs are outside the map registry")
        if not bool(torch.isfinite(pose_information).all()):
            raise ValueError("pose information must be finite")
        for anchor, contribution in zip(pose_ids.tolist(), pose_information):
            information[anchor][local_query_index] = contribution
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
        visibility_target=int(visibility_target),
        detectability_target=int(detectability_target),
        matching_target=int(matching_target),
        pose_logdet_target=float(pose_logdet_target),
        pose_min_eigenvalue_target=pose_min_eigenvalue_target,
        maximum_anchors=int(maximum_anchors),
        query_count=int(round_training_queries.numel()),
    )
    selected = torch.sort(result["selected_anchor_rows"]).values
    proposal = subset_projective_anchor_map(state, selected)
    proposal["v6_selection_distillation"] = {
        "schema": "lafgs_v6_layered_sufficiency_selection",
        "version": 3,
        "selected_source_rows": selected,
        "training_query_indices": cumulative_training_queries,
        "selected_query_indices": cumulative_training_queries,
        "round_training_query_indices": round_training_queries,
        "round_selected_query_indices": round_training_queries,
        "training_query_registry_explicit": cumulative_registry_explicit,
        "selection_round": int(
            prior_report.get("selection_round", 0)
            if isinstance(prior_report, dict)
            else 0
        )
        + 1,
        "hierarchy": ["visibility", "detectability", "matching", "pose"],
        "weighted_heuristic_sum": False,
        "visibility_evidence_unit": "query_image_grid_cell",
        "detectability_evidence_unit": "query_keypoint_row",
        "matching_evidence_unit": "query_keypoint_row",
        "pose_evidence_unit": "unique_anchor_per_query",
        "report": result,
    }
    return proposal, result
