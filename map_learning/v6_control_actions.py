"""Control-oriented descriptor actions for the fixed V6 localization plant."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F

from common.v6_contracts import FEEDBACK_SCHEMA, require_schema
from evidence.observation_provider import ObservationProvider
from localization.pose_solver import solve_absolute_pose


def _pose_error_cm_deg(
    estimated_w2c: torch.Tensor,
    ground_truth_w2c: torch.Tensor,
) -> tuple[float, float]:
    estimated = torch.as_tensor(estimated_w2c).float()
    truth = torch.as_tensor(ground_truth_w2c).float()
    rotation = estimated[:3, :3] @ truth[:3, :3].T
    cosine = ((torch.trace(rotation) - 1.0) * 0.5).clamp(-1.0, 1.0)
    angular = float(torch.rad2deg(torch.acos(cosine)))
    estimated_center = -(estimated[:3, :3].T @ estimated[:3, 3])
    truth_center = -(truth[:3, :3].T @ truth[:3, 3])
    translation = float(torch.linalg.norm(estimated_center - truth_center) * 100.0)
    return translation, angular


def _pose_risk(te_cm: float, ae_deg: float) -> float:
    return float(te_cm) / 5.0 + float(ae_deg) / 5.0


def minimal_pose_correction_set(
    *,
    keypoints: torch.Tensor,
    xyz: torch.Tensor,
    winners: torch.Tensor,
    candidate_rows: torch.Tensor,
    candidate_positive_anchors: torch.Tensor,
    intrinsics: torch.Tensor,
    ground_truth_pose_w2c: torch.Tensor,
    reprojection_error_px: float,
    candidate_priority: torch.Tensor | None = None,
    maximum_candidates: int = 24,
    maximum_set_size: int = 8,
    beam_width: int = 4,
    seed: int = 2026,
    solver: Callable = solve_absolute_pose,
) -> dict:
    """Find a small winner-replacement set that changes the real pose output.

    The search is deliberately over a finite, auditable action set.  Every
    node replays the unchanged one-shot PoseLib plant; no soft-pose surrogate
    is used to declare a correction successful.
    """

    keypoints = torch.as_tensor(keypoints).float()
    xyz = torch.as_tensor(xyz).float()
    winners = torch.as_tensor(winners).long().reshape(-1)
    rows = torch.as_tensor(candidate_rows).long().reshape(-1)
    positives = torch.as_tensor(candidate_positive_anchors).long().reshape(-1)
    if keypoints.shape != (winners.numel(), 2) or rows.shape != positives.shape:
        raise ValueError("pose-correction rows, winners, or keypoints are not aligned")
    if rows.numel() and (
        int(rows.min()) < 0
        or int(rows.max()) >= winners.numel()
        or int(positives.min()) < 0
        or int(positives.max()) >= xyz.shape[0]
    ):
        raise ValueError("pose-correction action references an invalid row or Anchor")
    if int(maximum_candidates) < 1 or int(maximum_set_size) < 1 or int(beam_width) < 1:
        raise ValueError("pose-correction search budgets must be positive")
    priority = (
        torch.zeros(rows.numel(), dtype=torch.float32)
        if candidate_priority is None
        else torch.as_tensor(candidate_priority).float().reshape(-1)
    )
    if priority.shape != rows.shape or not bool(torch.isfinite(priority).all()):
        raise ValueError("pose-correction priorities are not finite and aligned")

    # One deterministic action per sparse detector row.  A higher-priority
    # certified positive wins; Anchor row breaks exact ties.
    order = sorted(
        range(rows.numel()),
        key=lambda index: (-float(priority[index]), int(rows[index]), int(positives[index])),
    )
    kept = []
    seen_rows = set()
    for index in order:
        row = int(rows[index])
        if row in seen_rows or int(positives[index]) == int(winners[row]):
            continue
        seen_rows.add(row)
        kept.append(index)
        if len(kept) == int(maximum_candidates):
            break
    rows = rows[kept]
    positives = positives[kept]
    priority = priority[kept]

    evaluations: dict[tuple[int, ...], dict] = {}

    def evaluate(action_indices: tuple[int, ...]) -> dict:
        if action_indices in evaluations:
            return evaluations[action_indices]
        patched = winners.clone()
        if action_indices:
            indices = torch.tensor(action_indices, dtype=torch.long)
            patched[rows[indices]] = positives[indices]
        estimate = solver(
            keypoints.numpy(),
            xyz[patched].numpy(),
            torch.as_tensor(intrinsics).float().numpy(),
            reprojection_error_px=float(reprojection_error_px),
            confidence=0.99999,
            max_iterations=100000,
            min_iterations=1000,
            seed=int(seed),
        )
        te_cm, ae_deg = _pose_error_cm_deg(
            torch.as_tensor(estimate.pose_w2c), ground_truth_pose_w2c
        )
        result = {
            "action_indices": action_indices,
            "te_cm": te_cm,
            "ae_deg": ae_deg,
            "risk": _pose_risk(te_cm, ae_deg),
            "success": bool(te_cm < 5.0 and ae_deg < 5.0),
            "inlier_count": len(estimate.inliers),
        }
        evaluations[action_indices] = result
        return result

    baseline = evaluate(())
    best = baseline
    if baseline["success"] or rows.numel() == 0:
        return {
            "baseline": baseline,
            "best": best,
            "correction_found": bool(baseline["success"]),
            "selected_rows": torch.empty(0, dtype=torch.long),
            "selected_positive_anchors": torch.empty(0, dtype=torch.long),
            "candidate_count": int(rows.numel()),
            "evaluated_action_set_count": len(evaluations),
        }
    beam = [()]
    successful = None
    for _depth in range(1, min(int(maximum_set_size), int(rows.numel())) + 1):
        expanded = set()
        for parent in beam:
            for action in range(rows.numel()):
                if action not in parent:
                    expanded.add(tuple(sorted((*parent, action))))
        ranked = []
        for action_set in sorted(expanded):
            outcome = evaluate(action_set)
            if outcome["risk"] < best["risk"]:
                best = outcome
            ranked.append(outcome)
        successes = [outcome for outcome in ranked if outcome["success"]]
        if successes:
            successful = min(
                successes,
                key=lambda value: (value["risk"], value["action_indices"]),
            )
            break
        beam = [
            value["action_indices"]
            for value in sorted(
                ranked,
                key=lambda value: (value["risk"], value["action_indices"]),
            )[: int(beam_width)]
        ]
        if not beam:
            break
    chosen = successful if successful is not None else best
    chosen_indices = torch.tensor(chosen["action_indices"], dtype=torch.long)
    return {
        "baseline": baseline,
        "best": chosen,
        "correction_found": successful is not None,
        "selected_rows": rows[chosen_indices]
        if chosen_indices.numel()
        else torch.empty(0, dtype=torch.long),
        "selected_positive_anchors": positives[chosen_indices]
        if chosen_indices.numel()
        else torch.empty(0, dtype=torch.long),
        "candidate_count": int(rows.numel()),
        "evaluated_action_set_count": len(evaluations),
        "candidate_rows": rows,
        "candidate_positive_anchors": positives,
        "candidate_priority": priority,
    }


def minimum_norm_score_boundary_action(
    *,
    anchor_features: torch.Tensor,
    query_descriptors: torch.Tensor,
    positive_anchors: torch.Tensor,
    negative_anchors: torch.Tensor,
    target_margins: torch.Tensor,
    maximum_iterations: int = 2000,
    tolerance: float = 1e-6,
) -> dict:
    """Solve the linearized minimum-norm cosine boundary correction.

    Hildreth coordinate ascent solves the dual of ``min 1/2 ||x||^2`` subject
    to the requested score half-spaces.  Variables are Anchor-tangent vectors,
    matching the deployed normalized descriptor parameterization.
    """

    features = F.normalize(torch.as_tensor(anchor_features).float(), dim=1)
    query = F.normalize(torch.as_tensor(query_descriptors).float(), dim=1)
    positive = torch.as_tensor(positive_anchors).long().reshape(-1)
    negative = torch.as_tensor(negative_anchors).long().reshape(-1)
    target = torch.as_tensor(target_margins).float().reshape(-1)
    if not (query.shape[0] == positive.numel() == negative.numel() == target.numel()):
        raise ValueError("score-boundary constraints are not aligned")
    if positive.numel() == 0:
        raise ValueError("score-boundary action requires at least one constraint")
    if bool((positive == negative).any()) or bool(
        (positive < 0).any()
        | (negative < 0).any()
        | (positive >= features.shape[0]).any()
        | (negative >= features.shape[0]).any()
    ):
        raise ValueError("score-boundary constraint references an invalid Anchor")
    active = torch.unique(torch.cat((positive, negative)), sorted=True)
    lookup = torch.full((features.shape[0],), -1, dtype=torch.long)
    lookup[active] = torch.arange(active.numel())
    width = int(features.shape[1])
    matrix = torch.zeros((positive.numel(), active.numel(), width))
    current_margin = (query * features[positive]).sum(1) - (
        query * features[negative]
    ).sum(1)
    for constraint in range(positive.numel()):
        for sign, anchor in ((1.0, int(positive[constraint])), (-1.0, int(negative[constraint]))):
            descriptor = features[anchor]
            gradient = query[constraint] - torch.dot(query[constraint], descriptor) * descriptor
            matrix[constraint, lookup[anchor]] += float(sign) * gradient
    flat = matrix.flatten(1)
    deficit = (target - current_margin).clamp_min(0.0)
    row_norm = flat.square().sum(1)
    impossible = (deficit > float(tolerance)) & (row_norm <= 1e-12)
    if bool(impossible.any()):
        return {
            "feasible": False,
            "active_anchor_rows": active,
            "action": torch.zeros((active.numel(), width)),
            "required_anchor_norms": torch.zeros(active.numel()),
            "achieved_linearized_margins": current_margin,
            "maximum_violation": float(deficit[impossible].max()),
            "iterations": 0,
        }
    dual = torch.zeros(positive.numel())
    action_flat = torch.zeros(flat.shape[1])
    iterations = 0
    for iterations in range(1, int(maximum_iterations) + 1):
        maximum_change = 0.0
        for row in range(flat.shape[0]):
            if float(row_norm[row]) <= 1e-12:
                continue
            old = float(dual[row])
            updated = max(
                0.0,
                old + float(deficit[row] - torch.dot(flat[row], action_flat))
                / float(row_norm[row]),
            )
            if updated != old:
                action_flat += (updated - old) * flat[row]
                dual[row] = updated
                maximum_change = max(maximum_change, abs(updated - old))
        if maximum_change <= float(tolerance):
            break
    achieved = current_margin + flat @ action_flat
    violation = (target - achieved).clamp_min(0.0)
    action = action_flat.reshape(active.numel(), width)
    norms = torch.linalg.norm(action, dim=1)
    return {
        "feasible": bool(float(violation.max()) <= max(float(tolerance) * 10.0, 1e-5)),
        "active_anchor_rows": active,
        "action": action,
        "required_anchor_norms": norms,
        "achieved_linearized_margins": achieved,
        "maximum_violation": float(violation.max()),
        "iterations": iterations,
        "initial_margins": current_margin,
        "target_margins": target,
        "action_l2_norm": float(torch.linalg.norm(action)),
        "maximum_anchor_action_norm": float(norms.max()),
    }


def control_oriented_descriptor_proposal(
    state: dict,
    observations: ObservationProvider,
    feedback: dict,
    *,
    training_query_indices: torch.Tensor | Sequence[int],
    trust_region: float = 0.05,
    margin: float = 0.05,
    reprojection_error_px: float,
    maximum_candidates_per_query: int = 24,
    maximum_correction_set_size: int = 8,
    beam_width: int = 4,
    seed: int = 2026,
    solver: Callable = solve_absolute_pose,
) -> dict:
    """Create a bounded map action from PoseLib-changing correction sets."""

    require_schema(feedback, FEEDBACK_SCHEMA, label="control-action feedback")
    if list(feedback["query_names"]) != list(observations.names):
        raise ValueError("control-action feedback and observation registries differ")
    if not 0.0 < float(trust_region) <= 0.2 or float(margin) < 0.0:
        raise ValueError("control-action trust region or margin is invalid")
    training = torch.unique(
        torch.as_tensor(training_query_indices).long().reshape(-1), sorted=True
    )
    if training.numel() == 0 or int(training.min()) < 0 or int(training.max()) >= len(observations):
        raise ValueError("control-action training query registry is invalid")
    features = F.normalize(torch.as_tensor(state["anchor_features"]).float(), dim=1)
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    audits = []
    bundles = []
    for query_index in training.tolist():
        record = feedback["records"][query_index]
        if bool(record.get("pose_success", False)):
            continue
        view = observations.build_view(query_index)
        winners = torch.as_tensor(record.get("winner_anchor_ids", ())).long().reshape(-1)
        triplets = torch.as_tensor(record.get("descriptor_triplets", ())).long().reshape(-1, 4)
        pose_weight = torch.as_tensor(
            record.get("descriptor_triplet_pose_weights", ())
        ).float().reshape(-1)
        if triplets.shape[0] != pose_weight.numel() or winners.numel() != view.descriptors.shape[0]:
            raise ValueError("control-action feedback rows are not aligned")
        deployed = triplets[:, 2] == winners[triplets[:, 0]] if triplets.numel() else torch.empty(0, dtype=torch.bool)
        triplets = triplets[deployed]
        pose_weight = pose_weight[deployed]
        certified = torch.as_tensor(
            record.get("certified_pose_valid_alternative_pairs", ())
        ).long().reshape(-1, 2)
        negative_winner = torch.as_tensor(
            record.get("top1_negative_mask", torch.zeros(winners.numel()))
        ).bool().reshape(-1)
        if negative_winner.numel() != winners.numel():
            raise ValueError("control-action negative-winner mask is not aligned")
        if certified.numel():
            certified = certified[negative_winner[certified[:, 0]]]
            certified_triplets = torch.column_stack(
                (
                    certified[:, 0],
                    certified[:, 1],
                    winners[certified[:, 0]],
                    torch.zeros(certified.shape[0], dtype=torch.long),
                )
            )
            existing_keys = {
                (int(row), int(positive)) for row, positive in triplets[:, :2].tolist()
            }
            keep = torch.tensor(
                [
                    (int(row), int(positive)) not in existing_keys
                    for row, positive in certified[:, :2].tolist()
                ],
                dtype=torch.bool,
            )
            certified_triplets = certified_triplets[keep]
            triplets = torch.cat((triplets, certified_triplets), dim=0)
            pose_weight = torch.cat(
                (pose_weight, torch.zeros(certified_triplets.shape[0]))
            )
        certified_candidate_count = int(
            certified_triplets.shape[0] if certified.numel() else 0
        )
        priority = pose_weight + 0.01 * (~triplets[:, 3].bool()).float()
        search = minimal_pose_correction_set(
            keypoints=view.physical_keypoints,
            xyz=xyz,
            winners=winners,
            candidate_rows=triplets[:, 0],
            candidate_positive_anchors=triplets[:, 1],
            candidate_priority=priority,
            intrinsics=view.intrinsics,
            ground_truth_pose_w2c=view.pose_w2c,
            reprojection_error_px=float(reprojection_error_px),
            maximum_candidates=int(maximum_candidates_per_query),
            maximum_set_size=int(maximum_correction_set_size),
            beam_width=int(beam_width),
            seed=int(seed),
            solver=solver,
        )
        audit = {
            "query_index": query_index,
            "image_name": view.image_name,
            "baseline_te_cm": search["baseline"]["te_cm"],
            "baseline_ae_deg": search["baseline"]["ae_deg"],
            "best_te_cm": search["best"]["te_cm"],
            "best_ae_deg": search["best"]["ae_deg"],
            "candidate_action_count": search["candidate_count"],
            "certified_pose_valid_candidate_action_count": (
                certified_candidate_count
            ),
            "evaluated_action_set_count": search["evaluated_action_set_count"],
            "minimal_correction_set_size": int(search["selected_rows"].numel()),
            "pose_correction_found": bool(search["correction_found"]),
            "selected_query_rows": search["selected_rows"],
            "selected_positive_anchors": search["selected_positive_anchors"],
        }
        if search["correction_found"] and search["selected_rows"].numel():
            rows = search["selected_rows"]
            positives = search["selected_positive_anchors"]
            negatives = winners[rows]
            local = minimum_norm_score_boundary_action(
                anchor_features=features,
                query_descriptors=view.descriptors[rows],
                positive_anchors=positives,
                negative_anchors=negatives,
                target_margins=torch.full((rows.numel(),), float(margin)),
            )
            controllable = bool(
                local["feasible"]
                and local["maximum_anchor_action_norm"] <= float(trust_region)
            )
            audit.update(
                {
                    "minimum_descriptor_action_l2_norm": local["action_l2_norm"],
                    "minimum_descriptor_anchor_norm": local[
                        "maximum_anchor_action_norm"
                    ],
                    "representation_controllable": controllable,
                }
            )
            if controllable:
                bundles.append(
                    {
                        "query_index": query_index,
                        "rows": rows,
                        "positive": positives,
                        "negative": negatives,
                        "query": view.descriptors[rows].float(),
                        "risk_gain": search["baseline"]["risk"] - search["best"]["risk"],
                    }
                )
        else:
            audit.update(
                {
                    "minimum_descriptor_action_l2_norm": None,
                    "minimum_descriptor_anchor_norm": None,
                    "representation_controllable": False,
                }
            )
        audit["controller_route"] = (
            "representation"
            if audit["representation_controllable"]
            else "structure_or_prior_limited"
        )
        audits.append(audit)
    if not bundles:
        raise ValueError("feedback contains no controllable pose correction sets")

    # Greedy MPC: admit the highest predicted pose-risk reduction only while
    # the joint minimum-norm action remains inside every Anchor trust ball.
    accepted = []
    joint = None
    for bundle in sorted(
        bundles, key=lambda value: (-float(value["risk_gain"]), value["query_index"])
    ):
        trial = [*accepted, bundle]
        joint = minimum_norm_score_boundary_action(
            anchor_features=features,
            query_descriptors=torch.cat([value["query"] for value in trial]),
            positive_anchors=torch.cat([value["positive"] for value in trial]),
            negative_anchors=torch.cat([value["negative"] for value in trial]),
            target_margins=torch.full(
                (sum(int(value["rows"].numel()) for value in trial),), float(margin)
            ),
        )
        if joint["feasible"] and joint["maximum_anchor_action_norm"] <= float(trust_region):
            accepted = trial
    if not accepted or joint is None:
        raise ValueError("controllable correction sets have no joint trust-region action")
    def solve_with_clean_protection(values: list[dict]) -> tuple[dict, int]:
        correction_query = torch.cat([value["query"] for value in values])
        correction_positive = torch.cat([value["positive"] for value in values])
        correction_negative = torch.cat([value["negative"] for value in values])
        correction_count = sum(int(value["rows"].numel()) for value in values)
        provisional = minimum_norm_score_boundary_action(
            anchor_features=features,
            query_descriptors=correction_query,
            positive_anchors=correction_positive,
            negative_anchors=correction_negative,
            target_margins=torch.full((correction_count,), float(margin)),
        )
        protected_query = []
        protected_positive = []
        protected_negative = []
        active_set = set(provisional["active_anchor_rows"].tolist())
        for query_index in training.tolist():
            record = feedback["records"][query_index]
            triplets = torch.as_tensor(
                record.get("descriptor_triplets", ())
            ).long().reshape(-1, 4)
            if triplets.numel() == 0:
                continue
            clean = triplets[:, 3].bool()
            adjacent = torch.tensor(
                [
                    int(positive) in active_set or int(negative) in active_set
                    for positive, negative in triplets[:, 1:3].tolist()
                ],
                dtype=torch.bool,
            )
            selected = torch.nonzero(clean & adjacent, as_tuple=False).reshape(-1)[:16]
            if selected.numel() == 0:
                continue
            view = observations.build_view(query_index)
            protected_query.append(view.descriptors[triplets[selected, 0]].float())
            protected_positive.append(triplets[selected, 1])
            protected_negative.append(triplets[selected, 2])
        protection_count = sum(int(value.shape[0]) for value in protected_query)
        if protection_count == 0:
            return provisional, 0
        return (
            minimum_norm_score_boundary_action(
                anchor_features=features,
                query_descriptors=torch.cat((correction_query, *protected_query)),
                positive_anchors=torch.cat(
                    (correction_positive, *protected_positive)
                ),
                negative_anchors=torch.cat(
                    (correction_negative, *protected_negative)
                ),
                target_margins=torch.cat(
                    (
                        torch.full((correction_count,), float(margin)),
                        torch.zeros(protection_count),
                    )
                ),
            ),
            protection_count,
        )

    clean_protection_count = 0
    while accepted:
        joint, clean_protection_count = solve_with_clean_protection(accepted)
        if joint["feasible"] and joint["maximum_anchor_action_norm"] <= float(
            trust_region
        ):
            break
        accepted.pop()
    if not accepted:
        raise ValueError(
            "controllable correction sets violate clean protection or trust region"
        )
    active = joint["active_anchor_rows"]
    output_features = features.clone()
    output_features[active] = F.normalize(features[active] + joint["action"], dim=1)
    observation_base = F.normalize(
        torch.as_tensor(state.get("anchor_observation_features", features)).float(), dim=1
    )
    dot = (observation_base[active] * output_features[active]).sum(1, keepdim=True)
    if bool((dot <= 0.0).any()):
        raise ValueError("control action cannot be represented in the deployment tangent")
    deployed_tangent = output_features[active] / dot - observation_base[active]
    deployed_norm = torch.linalg.norm(deployed_tangent, dim=1)
    if bool((deployed_norm > float(trust_region) + 1e-6).any()):
        raise ValueError("control action exceeds the cumulative deployment trust region")
    output_residual = torch.as_tensor(
        state.get("anchor_descriptor_residual", torch.zeros_like(features))
    ).float().clone()
    output_residual[active] = deployed_tangent
    selected_queries = torch.tensor(
        [value["query_index"] for value in accepted], dtype=torch.long
    )
    proposal = dict(state)
    proposal["anchor_observation_features"] = observation_base
    proposal["anchor_descriptor_residual"] = output_residual
    proposal["anchor_features"] = output_features
    proposal["v6_descriptor_distillation"] = {
        "schema": "lafgs_v6_control_oriented_descriptor_distillation",
        "version": 1,
        "training_query_indices": training,
        "selected_query_indices": selected_queries,
        "training_query_registry_explicit": True,
        "updated_anchor_rows": active,
        "round_updated_anchor_rows": active,
        "updated_anchor_count": int(active.numel()),
        "round_updated_anchor_count": int(active.numel()),
        "trust_region": float(trust_region),
        "margin": float(margin),
        "reprojection_error_px": float(reprojection_error_px),
        "minimal_correction_search": "bounded_deterministic_poselib_beam_search",
        "score_action_solver": "hildreth_linearized_minimum_norm_tangent_qp",
        "controller": "greedy_finite_action_model_predictive_control",
        "failed_query_audits": audits,
        "failed_query_count": len(audits),
        "representation_controllable_query_count": sum(
            bool(value["representation_controllable"]) for value in audits
        ),
        "accepted_query_indices": selected_queries,
        "accepted_query_count": len(accepted),
        "accepted_constraint_count": sum(
            int(value["rows"].numel()) for value in accepted
        ),
        "clean_winner_protection_constraint_count": clean_protection_count,
        "clean_winner_protection_margin": 0.0,
        "joint_action_l2_norm": joint["action_l2_norm"],
        "joint_maximum_anchor_action_norm": joint["maximum_anchor_action_norm"],
        "joint_maximum_linearized_violation": joint["maximum_violation"],
        "maximum_candidates_per_query": int(maximum_candidates_per_query),
        "maximum_correction_set_size": int(maximum_correction_set_size),
        "beam_width": int(beam_width),
        "sampling_seed": int(seed),
        "online_model_added": False,
        "query_encoder_changed": False,
        "geometry_changed": False,
        "selection_changed": False,
    }
    return proposal
