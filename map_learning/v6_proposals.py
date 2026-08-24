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
from topology.layered_sufficiency import (
    DEFAULT_VISIBILITY_GRID,
    select_layered_sufficiency,
    visibility_image_cells,
)
from topology.pose_information import (
    fisher_contributions,
    pose_jacobian_analytic,
    task_scaled_pose_jacobian,
)
from topology.v6_anchor_map import subset_projective_anchor_map


_SELECTION_TRANSLATION_SCALE_M = 0.05
_SELECTION_ROTATION_SCALE_DEG = 5.0


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


def geometry_consensus_descriptor_feedback(feedback: dict) -> tuple[dict, int]:
    """Add pose-valid alternative correspondences without relabeling identity.

    Exact identities remain the only identity positives.  A geometry-compatible
    non-identity Anchor can nevertheless be a valid PnP correspondence, so for
    rows whose deployed winner is a true negative we add one deterministic
    alternative as weak set-formation supervision.
    """

    require_schema(feedback, FEEDBACK_SCHEMA, label="geometry-consensus feedback")
    output = dict(feedback)
    records = []
    added = 0
    for record in feedback["records"]:
        revised = dict(record)
        rows = torch.as_tensor(record.get("query_rows", ())).long().reshape(-1)
        winners = torch.as_tensor(record.get("winner_anchor_ids", ())).long().reshape(-1)
        negative = torch.as_tensor(record.get("top1_negative_mask", ())).bool().reshape(-1)
        if rows.shape != winners.shape or rows.shape != negative.shape:
            raise ValueError("geometry-consensus winner rows are not aligned")
        winner_by_row = {
            int(row): (int(anchor), bool(is_negative))
            for row, anchor, is_negative in zip(
                rows.tolist(), winners.tolist(), negative.tolist()
            )
        }
        alternatives = defaultdict(list)
        ambiguous = torch.as_tensor(
            record.get("projective_compatible_ambiguous_pairs", ())
        ).long().reshape(-1, 2)
        for row, anchor in ambiguous.tolist():
            alternatives[int(row)].append(int(anchor))
        existing = torch.as_tensor(
            record.get("descriptor_triplets", ())
        ).long().reshape(-1, 4)
        existing_rows = set(existing[:, 0].tolist()) if existing.numel() else set()
        extra = []
        inlier_rows = set(
            torch.as_tensor(record.get("inlier_query_rows", ())).long().tolist()
        )
        for row in sorted(alternatives):
            deployed = winner_by_row.get(row)
            if deployed is None or not deployed[1] or row in existing_rows:
                continue
            positive = min(alternatives[row])
            if positive == deployed[0]:
                continue
            extra.append((row, positive, deployed[0], 0))
        extra_tensor = torch.tensor(extra, dtype=torch.long).reshape(-1, 4)
        revised["descriptor_triplets"] = torch.cat((existing, extra_tensor), dim=0)
        old_harmful = torch.as_tensor(
            record.get("descriptor_triplet_harmful_inlier_mask", ())
        ).bool().reshape(-1)
        old_weight = torch.as_tensor(
            record.get("descriptor_triplet_pose_weights", ())
        ).float().reshape(-1)
        old_clean = torch.as_tensor(
            record.get("descriptor_triplet_legal_pair_clean_mask", ())
        ).bool().reshape(-1)
        if not (
            old_harmful.numel() == old_weight.numel() == old_clean.numel() == existing.shape[0]
        ):
            raise ValueError("geometry-consensus source triplet fields are not aligned")
        extra_harmful = torch.tensor(
            [row in inlier_rows for row, _, _, _ in extra], dtype=torch.bool
        )
        revised["descriptor_triplet_harmful_inlier_mask"] = torch.cat(
            (old_harmful, extra_harmful)
        )
        revised["descriptor_triplet_pose_weights"] = torch.cat(
            # The alternative is GT-geometry compatible, but we have not run
            # the fixed-hypothesis counterfactual required by the v5 pose-weight
            # contract.  Keep that distinct diagnostic at zero.
            (old_weight, torch.zeros(len(extra), dtype=torch.float32))
        )
        revised["descriptor_triplet_legal_pair_clean_mask"] = torch.cat(
            (old_clean, torch.zeros(len(extra), dtype=torch.bool))
        )
        revised["geometry_consensus_weak_positive_triplet_count"] = len(extra)
        added += len(extra)
        records.append(revised)
    output["records"] = records
    output["geometry_consensus_weak_positive_triplet_count"] = added
    output["geometry_consensus_uses_test_queries"] = False
    return output, added


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
    allow_geometry_compatible_positives: bool = False,
    loss_mode: str = "pairwise",
    consensus_count_target: float = 16.0,
    consensus_cell_target: float = 4.0,
    consensus_count_weight: float = 1.0,
    consensus_cell_weight: float = 1.0,
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
    if loss_mode not in {"pairwise", "set_consensus"}:
        raise ValueError("descriptor loss mode is invalid")
    if (
        float(consensus_count_target) <= 0.0
        or float(consensus_cell_target) <= 0.0
        or float(consensus_count_weight) < 0.0
        or float(consensus_cell_weight) < 0.0
    ):
        raise ValueError("descriptor consensus targets/weights are invalid")
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
    if any("affected_anchor_policy" not in record for record in feedback["records"]):
        raise ValueError("descriptor feedback requires an explicit LOO policy")
    feedback_policies = {
        str(record["affected_anchor_policy"]) for record in feedback["records"]
    }
    if len(feedback_policies) != 1:
        raise ValueError("descriptor feedback mixes affected-Anchor LOO policies")
    loo_policy = next(iter(feedback_policies))
    if loo_policy not in {"fixed_map", "descriptor_only", "rebuild"}:
        raise ValueError(
            "descriptor training requires a deployable fixed-map, descriptor-only, "
            "or exact-rebuild observer; purge feedback is diagnostic-only"
        )
    loo_replay = (
        None
        if loo_policy == "fixed_map"
        else LeaveOneQueryOutProjectiveMap(
            state,
            observations,
            affected_anchor_policy="rebuild",
        )
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
        local_base = observation_features[active_rows].clone()
        if loo_policy == "fixed_map":
            return active_rows, local_base, torch.empty(0, dtype=torch.long)
        if loo_policy == "descriptor_only":
            update = loo_replay.descriptor_only_update(query_index)
            selected = torch.isin(update["anchor_rows"], active_rows)
            update = {
                **update,
                "anchor_rows": update["anchor_rows"][selected],
                "valid": update["valid"][selected],
                "anchor_observation_features": update[
                    "anchor_observation_features"
                ][selected],
            }
        else:
            update = loo_replay.query_update(
                query_index,
                excluded_queries=excluded_queries,
                requested_anchor_rows=active_rows,
            )
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
    cell_parts = []
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
            registry = [
                (int(row), int(anchor)) for row, anchor in pairs.tolist()
            ]
            if len(registry) != len(set(registry)):
                raise ValueError(f"descriptor feedback {key} contains duplicates")
            return set(registry)

        lineage_pairs = pair_registry("exact_identity_pairs")
        active_pairs = pair_registry("active_identity_pairs")
        positive_pairs = pair_registry("exact_identity_positive_pairs")
        inactive_pairs = pair_registry("identity_inactive_pairs")
        incompatible_pairs = pair_registry(
            "identity_projective_incompatible_pairs"
        )
        ambiguous_pairs = pair_registry("projective_compatible_ambiguous_pairs")
        if active_pairs != positive_pairs | incompatible_pairs:
            raise ValueError("descriptor feedback active identity partition differs")
        if lineage_pairs != active_pairs | inactive_pairs:
            raise ValueError("descriptor feedback identity lineage partition differs")
        if lineage_pairs & ambiguous_pairs:
            raise ValueError("descriptor feedback identity and ambiguity overlap")
        winner_rows = torch.as_tensor(record.get("query_rows", ())).long().reshape(-1)
        winner_anchors = torch.as_tensor(
            record.get("winner_anchor_ids", ())
        ).long().reshape(-1)
        if winner_rows.shape != winner_anchors.shape:
            raise ValueError("descriptor feedback winner rows are not aligned")
        if winner_rows.numel() != torch.unique(winner_rows).numel():
            raise ValueError("descriptor feedback winner query rows contain duplicates")
        winner_by_row = {
            int(row): int(anchor)
            for row, anchor in zip(winner_rows.tolist(), winner_anchors.tolist())
        }
        ignored_pairs = lineage_pairs | ambiguous_pairs
        for triplet_index, values in enumerate(triplets.tolist()):
            row, positive_anchor, negative_anchor, _ = map(int, values)
            allowed_positive_pairs = (
                positive_pairs | ambiguous_pairs
                if allow_geometry_compatible_positives
                else positive_pairs
            )
            if (row, positive_anchor) not in allowed_positive_pairs:
                raise ValueError(
                    "descriptor triplet positive lacks an allowed active correspondence"
                )
            if (row, negative_anchor) in ignored_pairs:
                raise ValueError("descriptor triplet negative is not legally negative")
            if (
                float(pose_weights[triplet_index]) > 0.0
                and winner_by_row.get(row) != negative_anchor
            ):
                raise ValueError(
                    "positive pose weight must replace the deployed winner"
                )

        # Formal V6 feedback is multi-label.  Normal descriptor ranking is an
        # L3 operation.  A pose-critical L4 row may additionally enter only
        # when its fixed-hypothesis counterfactual weight is positive.
        record_layers = record.get("failure_layers")
        layer_set = (
            None if record_layers is None else {str(layer) for layer in record_layers}
        )
        regular_eligible = layer_set is None or bool(
            (eligible_layers - {"L4"}) & layer_set
        )
        if (
            allow_geometry_compatible_positives
            and int(record.get("geometry_consensus_weak_positive_triplet_count", 0)) > 0
        ):
            # Set formation is learned from every mapping-training query, not
            # only from queries already labelled as L3 failures.  Otherwise it
            # cannot generalize the pose-valid alternative concept to a held-out
            # sequence whose exact identities disappear under neighborhood LOO.
            regular_eligible = True
        pose_eligible = bool(
            float(pose_critical_weight) > 0.0
            and layer_set is not None
            and "L4" in layer_set
            and (pose_weights > 0).any()
        )
        if not regular_eligible and not pose_eligible:
            continue
        if not regular_eligible:
            pose_critical_rows = pose_weights > 0
            triplets = triplets[pose_critical_rows]
            pose_weights = pose_weights[pose_critical_rows]
            harmful = harmful[pose_critical_rows]
        view = observations.build_view(query_index)
        rows, positive, negative, clean = (
            value.contiguous() for value in triplets.T
        )
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
        cell_parts.append(
            visibility_image_cells(
                view.physical_keypoints[rows[chosen]].float(),
                image_hw=view.image_hw,
            ).long()
        )
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
    selected_cell = torch.cat(cell_parts)
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
    selected_query_device = selected_query.to(train_device)
    selected_cell_device = selected_cell.to(train_device)
    visibility_cell_count = int(
        DEFAULT_VISIBILITY_GRID[0] * DEFAULT_VISIBILITY_GRID[1]
    )
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
        *,
        normalization_weight: float | None = None,
        normalization_query_count: int | None = None,
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
        denominator = (
            weight.sum().clamp_min(1e-8)
            if normalization_weight is None
            else weight.new_tensor(float(normalization_weight)).clamp_min(1e-8)
        )
        pairwise = (ranking * weight).sum() / denominator
        if loss_mode == "pairwise":
            return pairwise
        probability = torch.sigmoid(
            (positive_score - negative_score) / max(float(temperature), 1e-6)
        )
        selected_queries = selected_query_device[device_rows]
        selected_cells = selected_cell_device[device_rows]
        query_probability = probability.new_zeros(query_count)
        query_probability.scatter_add_(0, selected_queries, probability)
        query_rows_count = probability.new_zeros(query_count)
        query_rows_count.scatter_add_(
            0, selected_queries, torch.ones_like(probability)
        )
        query_present = query_rows_count > 0
        query_denominator = max(
            int(query_present.sum())
            if normalization_query_count is None
            else int(normalization_query_count),
            1,
        )
        count_target = torch.minimum(
            query_rows_count,
            query_rows_count.new_full(
                query_rows_count.shape, float(consensus_count_target)
            ),
        )
        count_loss = (
            F.softplus(count_target - query_probability)[query_present].sum()
            / query_denominator
        )

        cell_key = selected_queries * visibility_cell_count + selected_cells
        cell_probability_sum = probability.new_zeros(
            query_count * visibility_cell_count
        )
        cell_probability_sum.scatter_add_(0, cell_key, probability)
        cell_present = cell_probability_sum > 0
        cell_coverage = 1.0 - torch.exp(-cell_probability_sum)
        query_cell_coverage = probability.new_zeros(query_count)
        query_cell_count = probability.new_zeros(query_count)
        populated_cells = torch.nonzero(cell_present, as_tuple=False).reshape(-1)
        populated_queries = torch.div(
            populated_cells, visibility_cell_count, rounding_mode="floor"
        )
        query_cell_coverage.scatter_add_(
            0, populated_queries, cell_coverage[populated_cells]
        )
        query_cell_count.scatter_add_(
            0, populated_queries, torch.ones_like(cell_coverage[populated_cells])
        )
        cell_target = torch.minimum(
            query_cell_count,
            query_cell_count.new_full(
                query_cell_count.shape, float(consensus_cell_target)
            ),
        )
        cell_loss = (
            F.softplus(cell_target - query_cell_coverage)[query_present].sum()
            / query_denominator
        )
        return (
            pairwise
            + float(consensus_count_weight) * count_loss
            + float(consensus_cell_weight) * cell_loss
        )

    all_rows = torch.arange(query.shape[0])
    global_training_weight = float(training_triplet_weight.sum())
    selected_query_count = int(torch.unique(selected_query).numel())
    training_chunks = [all_rows]
    if loss_mode == "set_consensus":
        _, query_row_counts = torch.unique_consecutive(
            selected_query, return_counts=True
        )
        training_chunks = []
        chunk_start = 0
        cursor = 0
        chunk_rows = 0
        for count in query_row_counts.tolist():
            count = int(count)
            if chunk_rows and chunk_rows + count > int(batch_size):
                training_chunks.append(torch.arange(chunk_start, cursor))
                chunk_start = cursor
                chunk_rows = 0
            cursor += count
            chunk_rows += count
        if cursor > chunk_start:
            training_chunks.append(torch.arange(chunk_start, cursor))
        # The incremental construction above is deliberately checked rather
        # than trusted: every query must occur in exactly one complete chunk.
        if not torch.equal(torch.cat(training_chunks), all_rows):
            raise RuntimeError("set-consensus query chunks do not cover training rows")
        for rows in training_chunks:
            values = selected_query[rows]
            for query_index in torch.unique(values).tolist():
                if int((selected_query == query_index).sum()) != int(
                    (values == query_index).sum()
                ):
                    raise RuntimeError("set-consensus chunk splits a training query")

    def evaluation_loss(value: torch.Tensor) -> torch.Tensor:
        if loss_mode == "pairwise":
            return full_loss(value, all_rows)
        return torch.stack(
            [
                full_loss(
                    value,
                    rows,
                    normalization_weight=global_training_weight,
                    normalization_query_count=selected_query_count,
                )
                for rows in training_chunks
            ]
        ).sum()
    with torch.no_grad():
        _, initial_bounded_tangent = _bounded_descriptor_bank(
            base_active, residual, trust_region
        )
        initial_active_residual = residual.detach().clone()
        initial_raw_tangent = raw_tangent(residual)
        initial_loss = float(evaluation_loss(residual).cpu())
        initial_regularizer = float(trust_penalty(initial_raw_tangent).cpu())
        initial_objective = initial_loss + float(trust_weight) * initial_regularizer
        best_objective = initial_objective
        best_residual = residual.detach().clone()
        best_epoch = 0
        optimizer_last_objective = initial_objective
    for epoch in range(1, int(epochs) + 1):
        optimizer.zero_grad(set_to_none=True)
        if loss_mode == "set_consensus":
            for rows in training_chunks:
                full_loss(
                    residual,
                    rows,
                    normalization_weight=global_training_weight,
                    normalization_query_count=selected_query_count,
                ).backward()
        else:
            order = torch.randperm(query.shape[0], generator=generator)
            for start in range(0, query.shape[0], int(batch_size)):
                rows = order[start : start + int(batch_size)]
                # ``batch_size`` is a memory chunk only.  Accumulating every chunk
                # against one frozen global denominator gives the exact weighted
                # full-objective gradient, even for batch_size=1.
                ranking_partial = full_loss(
                    residual,
                    rows,
                    normalization_weight=global_training_weight,
                )
                ranking_partial.backward()
        if float(trust_weight) > 0.0:
            (float(trust_weight) * trust_penalty(raw_tangent(residual))).backward()
        optimizer.step()
        with torch.no_grad():
            epoch_ranking = float(evaluation_loss(residual).cpu())
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
        final_loss = float(evaluation_loss(best_residual).cpu())
        error_rows = torch.nonzero(~clean, as_tuple=False).reshape(-1)
        clean_rows = torch.nonzero(clean, as_tuple=False).reshape(-1)

        def split_loss(value: torch.Tensor, rows: torch.Tensor) -> float | None:
            if rows.numel() == 0 or loss_mode == "set_consensus":
                return None
            return float(full_loss(value, rows).cpu())

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
        "version": 4,
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
        "query_local_loo_descriptor_training": loo_policy != "fixed_map",
        "feedback_observer_policy": loo_policy,
        "query_local_loo_base_source": (
            "fixed_deployment_observation_bank"
            if loo_policy == "fixed_map"
            else "descriptor_only_sparse_replay"
            if loo_policy == "descriptor_only"
            else "sparse_active_anchor_replay"
        ),
        "query_local_loo_affected_anchor_policy": loo_policy,
        "query_local_loo_pair_count": int(unique_pair_keys.numel()),
        "query_local_loo_affected_pair_count": affected_pair_count,
        "query_local_loo_dense_query_anchor_bank_materialized": False,
        "query_observations_excluded_from_training_anchor_bases": (
            loo_policy != "fixed_map"
        ),
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
        "tail_query_definition": "training_split_translation_error_q95_all_layers",
        "round_tail_query_indices": torch.tensor(
            sorted(tail_query_set), dtype=torch.long
        ),
        "selected_tail_query_indices": torch.tensor(
            sorted(set(selected_query.tolist()) & tail_query_set),
            dtype=torch.long,
        ),
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
        "batch_size_role": "memory_chunk_for_exact_full_objective_gradient",
        "optimizer_step_count": int(epochs),
        "weighted_gradient_uses_fixed_global_denominator": True,
        "maximum_triplets_per_query": int(maximum_triplets_per_query),
        "sampling_seed": 2026,
        "online_model_added": False,
        "query_encoder_changed": False,
        "descriptor_positive_mode": (
            "exact_identity_or_geometry_compatible_pose_alternative"
            if allow_geometry_compatible_positives
            else "exact_identity_only"
        ),
        "geometry_compatible_positive_is_identity_positive": False,
        "loss_mode": loss_mode,
        "set_consensus_joint_query_objective": loss_mode == "set_consensus",
        "consensus_count_target": float(consensus_count_target),
        "consensus_cell_target": float(consensus_cell_target),
        "consensus_count_weight": float(consensus_count_weight),
        "consensus_cell_weight": float(consensus_cell_weight),
        "set_consensus_complete_query_chunk_count": (
            len(training_chunks) if loss_mode == "set_consensus" else None
        ),
        "set_consensus_chunking_is_exact_gradient_accumulation": (
            loss_mode == "set_consensus"
        ),
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


def _selection_pairs(
    record: dict,
    field: str,
    *,
    anchor_count: int,
    query_row_count: int,
) -> torch.Tensor:
    pairs = torch.as_tensor(record.get(field, ()), dtype=torch.long)
    if pairs.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"{field} must have shape [N,2]")
    rows = pairs[:, 0]
    anchors = pairs[:, 1]
    if bool(((rows < 0) | (rows >= int(query_row_count))).any()):
        raise ValueError(f"{field} query rows are outside the observation registry")
    if bool(((anchors < 0) | (anchors >= int(anchor_count))).any()):
        raise ValueError(f"{field} Anchor IDs are outside the map registry")
    return pairs


@torch.no_grad()
def _potential_pose_information(
    xyz: torch.Tensor,
    anchor_ids: torch.Tensor,
    *,
    intrinsics: torch.Tensor,
    pose_w2c: torch.Tensor,
    chunk_size: int,
) -> dict[int, torch.Tensor]:
    """Return one GT-geometry Fisher row per unique candidate Anchor."""

    anchor_ids = torch.unique(torch.as_tensor(anchor_ids).long(), sorted=True)
    if anchor_ids.numel() == 0:
        return {}
    result: dict[int, torch.Tensor] = {}
    K = torch.as_tensor(intrinsics).double()
    pose = torch.as_tensor(pose_w2c).double()
    if not bool(torch.isfinite(K).all()) or not bool(torch.isfinite(pose).all()):
        raise ValueError("selection mapping camera calibration must be finite")
    for start in range(0, int(anchor_ids.numel()), int(chunk_size)):
        rows = anchor_ids[start : start + int(chunk_size)]
        points = torch.as_tensor(xyz)[rows].double()
        homogeneous = torch.cat(
            (points, torch.ones((points.shape[0], 1), dtype=points.dtype)), dim=1
        )
        camera = (pose @ homogeneous.T).T[:, :3]
        if not bool(torch.isfinite(camera).all()) or bool((camera[:, 2] <= 0).any()):
            raise ValueError(
                "potential-pose candidate is non-finite or not in front of its mapping camera"
            )
        jacobian = pose_jacobian_analytic(points, K, pose)
        jacobian = task_scaled_pose_jacobian(
            jacobian,
            translation_scale=_SELECTION_TRANSLATION_SCALE_M,
            rotation_scale=math.radians(_SELECTION_ROTATION_SCALE_DEG),
        )
        contributions = fisher_contributions(jacobian)
        if not bool(torch.isfinite(contributions).all()):
            raise ValueError("potential pose information must be finite")
        for anchor, contribution in zip(rows.tolist(), contributions):
            result[int(anchor)] = contribution
    return result


def selection_only_proposal(
    state: dict,
    observations: ObservationProvider,
    feedback: dict,
    *,
    maximum_anchors: int,
    visibility_target: int,
    detectability_target: int,
    matching_target: int,
    pose_logdet_target: float,
    pose_min_eigenvalue_target: float | None = None,
    pose_information_chunk_size: int = 4096,
    training_query_indices: torch.Tensor | Sequence[int] | None = None,
) -> tuple[dict, dict]:
    """Select with fixed-layer evidence and GT-geometry potential Fisher."""

    require_schema(feedback, FEEDBACK_SCHEMA, label="self-localization feedback")
    if list(feedback.get("query_names", ())) != list(observations.names):
        raise ValueError("feedback and observation registries differ")
    if int(pose_information_chunk_size) < 1:
        raise ValueError("selection pose-information chunk size must be positive")
    count = int(torch.as_tensor(state["anchor_ids"]).numel())
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    if xyz.shape != (count, 3) or not bool(torch.isfinite(xyz).all()):
        raise ValueError("selection Anchor xyz must be finite and map-aligned")
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

    potential_pose_edge_count = 0
    potential_pose_query_count = 0
    for local_query_index, source_query_index in enumerate(
        round_training_queries.tolist()
    ):
        record = feedback["records"][source_query_index]
        view = observations.build_view(source_query_index)
        visible_ids = torch.as_tensor(record["visible_anchor_ids"]).long().reshape(-1)
        visible_cells = (
            torch.as_tensor(record["visible_anchor_image_cells"]).long().reshape(-1)
        )
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
        detectable_pairs = _selection_pairs(
            record,
            "detectable_pairs",
            anchor_count=count,
            query_row_count=int(view.descriptors.shape[0]),
        )
        matching_pairs = _selection_pairs(
            record,
            "matching_pairs",
            anchor_count=count,
            query_row_count=int(view.descriptors.shape[0]),
        )
        exact_geometry_pairs = _selection_pairs(
            record,
            "exact_identity_positive_pairs",
            anchor_count=count,
            query_row_count=int(view.descriptors.shape[0]),
        )
        ambiguous_geometry_pairs = _selection_pairs(
            record,
            "projective_compatible_ambiguous_pairs",
            anchor_count=count,
            query_row_count=int(view.descriptors.shape[0]),
        )
        for row, anchor in detectable_pairs.tolist():
            layers["detectability"][anchor][local_query_index].add(row)
        for row, anchor in matching_pairs.tolist():
            layers["matching"][anchor][local_query_index].add(row)
        candidate_ids = torch.unique(
            torch.cat(
                (exact_geometry_pairs[:, 1], ambiguous_geometry_pairs[:, 1])
            ),
            sorted=True,
        )
        layered_candidate_ids = torch.unique(
            torch.cat((detectable_pairs[:, 1], matching_pairs[:, 1])), sorted=True
        )
        if layered_candidate_ids.numel() and not bool(
            torch.isin(layered_candidate_ids, candidate_ids).all()
        ):
            raise ValueError(
                "layered detectable/matching evidence is absent from geometry pairs"
            )
        if candidate_ids.numel() and not bool(
            torch.isin(candidate_ids, visible_ids).all()
        ):
            raise ValueError("pose-information candidates must be visible Anchors")
        potential = _potential_pose_information(
            xyz,
            candidate_ids,
            intrinsics=view.intrinsics,
            pose_w2c=view.pose_w2c,
            chunk_size=int(pose_information_chunk_size),
        )
        potential_pose_edge_count += len(potential)
        potential_pose_query_count += int(bool(potential))
        for anchor, contribution in potential.items():
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
    result["potential_pose_information"] = {
        "source": "mapping_gt_pose_intrinsics_and_source_map_xyz",
        "candidate_pool": (
            "all_unique_feedback_exact_or_ambiguous_geometry_candidate_anchors"
        ),
        "realized_clean_inlier_conditioned": False,
        "unique_anchor_per_query": True,
        "sparse_query_anchor_storage": True,
        "dense_query_anchor_tensor_materialized": False,
        "edge_count": int(potential_pose_edge_count),
        "query_count_with_edges": int(potential_pose_query_count),
        "chunk_size": int(pose_information_chunk_size),
        "translation_scale_m": _SELECTION_TRANSLATION_SCALE_M,
        "rotation_scale_deg": _SELECTION_ROTATION_SCALE_DEG,
        "measurement_weight": 1.0,
    }
    selected = torch.sort(result["selected_anchor_rows"]).values
    proposal = subset_projective_anchor_map(state, selected)
    proposal["v6_selection_distillation"] = {
        "schema": "lafgs_v6_layered_sufficiency_selection",
        "version": 4,
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
        "pose_evidence_unit": (
            "unique_detectable_or_matching_anchor_per_query_gt_geometry_potential"
        ),
        "pose_information_source": ("mapping_gt_pose_intrinsics_and_source_map_xyz"),
        "pose_information_realized_clean_inlier_conditioned": False,
        "pose_information_chunk_size": int(pose_information_chunk_size),
        "potential_pose_information_edge_count": int(potential_pose_edge_count),
        "report": result,
    }
    return proposal, result
