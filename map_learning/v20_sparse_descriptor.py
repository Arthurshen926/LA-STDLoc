"""Sparse, map-side descriptor repair for the V20 closed loop.

The online Query descriptor remains native.  Only a frozen set of evidence-
selected Anchors receives a trainable tangent residual, projected exactly into
an angular trust region.  Multi-positive Top-K ranking is combined with a
clean-margin preservation loss; deployment still requires exact pose control
and independent confirmation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F


def _padded_csr(
    offsets: torch.Tensor,
    values: torch.Tensor,
    *,
    row_count: int,
    value_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.as_tensor(offsets).long().reshape(-1)
    values = torch.as_tensor(values).long().reshape(-1)
    if offsets.shape != (row_count + 1,) or int(offsets[0]) != 0 or int(offsets[-1]) != values.numel():
        raise ValueError("V20 descriptor evidence CSR is invalid")
    counts = offsets[1:] - offsets[:-1]
    if bool((counts <= 0).any()):
        raise ValueError("V20 descriptor evidence rows require non-empty sets")
    if values.numel() and (
        int(values.min()) < 0 or int(values.max()) >= int(value_count)
    ):
        raise ValueError("V20 descriptor evidence references an invalid Anchor row")
    width = int(counts.max())
    padded = torch.zeros((row_count, width), dtype=torch.long)
    mask = torch.zeros((row_count, width), dtype=torch.bool)
    for row in range(row_count):
        count = int(counts[row])
        padded[row, :count] = values[int(offsets[row]) : int(offsets[row + 1])]
        mask[row, :count] = True
    return padded, mask


def _bounded_descriptors(
    native: torch.Tensor,
    raw_tangent: torch.Tensor,
    maximum_angle_deg: float,
) -> torch.Tensor:
    """Map unconstrained parameters into the native-centered spherical cap."""

    tangent = raw_tangent - (raw_tangent * native).sum(1, keepdim=True) * native
    norm = torch.linalg.norm(tangent, dim=1, keepdim=True)
    maximum_tangent = math.tan(math.radians(float(maximum_angle_deg)))
    bounded = maximum_tangent * tangent / torch.sqrt(1.0 + norm.square())
    return F.normalize(native + bounded, dim=1)


def _scores(
    query: torch.Tensor,
    rows: torch.Tensor,
    mask: torch.Tensor,
    *,
    native: torch.Tensor,
    selected_lookup: torch.Tensor,
    selected_descriptors: torch.Tensor,
) -> torch.Tensor:
    flat = rows.reshape(-1)
    local = selected_lookup[flat]
    selected = local >= 0
    selected_rows = selected_descriptors[local.clamp_min(0)]
    descriptors = torch.where(
        selected[:, None], selected_rows, native[flat]
    ).reshape(rows.shape[0], rows.shape[1], native.shape[1])
    score = torch.einsum("bd,bkd->bk", query, descriptors)
    return score.masked_fill(~mask, -torch.inf)


def _per_positive_listwise_ranking(
    positive_scores: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_scores: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Return one loss per row while giving every true Anchor a gradient.

    A single log-sum-exp over the positive set mostly rewards the already
    strongest positive.  Here each certified positive competes against the
    whole negative list, then the valid positive losses are averaged per row.
    """

    negative_lse = torch.logsumexp(
        negative_scores / float(temperature), dim=1
    )
    pair_loss = F.softplus(
        negative_lse[:, None] - positive_scores / float(temperature)
    )
    pair_loss = torch.where(
        positive_mask, pair_loss, torch.zeros_like(pair_loss)
    )
    return pair_loss.sum(1) / positive_mask.sum(1).clamp_min(1)


@torch.inference_mode()
def audit_materialized_sparse_action(
    *,
    baseline_anchor_features: torch.Tensor,
    candidate_anchor_features: torch.Tensor,
    selected_anchor_rows: torch.Tensor,
    evidence: Mapping,
    clean_margin_slack: float,
    maximum_angle_deg: float,
    device: str | torch.device = "cpu",
) -> dict:
    """Recompute safety on the exact descriptor dtype that will be deployed."""

    baseline_raw = torch.as_tensor(baseline_anchor_features)
    candidate_raw = torch.as_tensor(candidate_anchor_features)
    selected = torch.as_tensor(selected_anchor_rows).long().reshape(-1)
    if (
        baseline_raw.ndim != 2
        or candidate_raw.shape != baseline_raw.shape
        or candidate_raw.dtype != baseline_raw.dtype
        or not bool(torch.isfinite(baseline_raw.float()).all())
        or not bool(torch.isfinite(candidate_raw.float()).all())
        or bool((torch.linalg.norm(baseline_raw.float(), dim=1) <= 1e-8).any())
        or bool((torch.linalg.norm(candidate_raw.float(), dim=1) <= 1e-8).any())
        or selected.numel() == 0
        or torch.unique(selected).numel() != selected.numel()
        or int(selected.min()) < 0
        or int(selected.max()) >= baseline_raw.shape[0]
        or not math.isfinite(float(clean_margin_slack))
        or float(clean_margin_slack) < 0.0
        or not math.isfinite(float(maximum_angle_deg))
        or not 0.0 < float(maximum_angle_deg) < 90.0
    ):
        raise ValueError("V20 materialized sparse action contract differs")
    if not (
        evidence.get("schema") == "lafgs_v20_topk_competition_evidence"
        and evidence.get("uses_test_queries") is False
        and evidence.get("loo_used") is False
    ):
        raise ValueError("V20 materialized audit evidence contract differs")

    selected_mask = torch.zeros(baseline_raw.shape[0], dtype=torch.bool)
    selected_mask[selected] = True
    changed = torch.any(baseline_raw != candidate_raw, dim=1)
    changed_outside = int((changed & ~selected_mask).sum())
    selected_unchanged = int((~changed & selected_mask).sum())

    target = torch.device(device)
    native = F.normalize(baseline_raw.float().to(target), dim=1)
    selected_candidate = F.normalize(
        candidate_raw[selected].float().to(target), dim=1
    )
    lookup = torch.full(
        (native.shape[0],), -1, dtype=torch.long, device=target
    )
    selected_device = selected.to(target)
    lookup[selected_device] = torch.arange(selected.numel(), device=target)
    identity_lookup = torch.full_like(lookup, -1)

    protect_query_raw = torch.as_tensor(
        evidence["protection_query_descriptors"]
    ).float()
    if (
        protect_query_raw.ndim != 2
        or protect_query_raw.shape[0] == 0
        or protect_query_raw.shape[1] != native.shape[1]
        or not bool(torch.isfinite(protect_query_raw).all())
        or bool((torch.linalg.norm(protect_query_raw, dim=1) <= 1e-8).any())
    ):
        raise ValueError("V20 materialized audit protection descriptors differ")
    protect_count = int(protect_query_raw.shape[0])
    positive, positive_mask = _padded_csr(
        evidence["protection_positive_offsets"],
        evidence["protection_positive_anchor_rows"],
        row_count=protect_count,
        value_count=native.shape[0],
    )
    negative, negative_mask = _padded_csr(
        evidence["protection_negative_offsets"],
        evidence["protection_negative_anchor_rows"],
        row_count=protect_count,
        value_count=native.shape[0],
    )
    query = F.normalize(protect_query_raw.to(target), dim=1)
    positive = positive.to(target)
    positive_mask = positive_mask.to(target)
    negative = negative.to(target)
    negative_mask = negative_mask.to(target)
    native_selected = native[selected_device]
    native_margin = _scores(
        query,
        positive,
        positive_mask,
        native=native,
        selected_lookup=identity_lookup,
        selected_descriptors=native_selected,
    ).max(1).values - _scores(
        query,
        negative,
        negative_mask,
        native=native,
        selected_lookup=identity_lookup,
        selected_descriptors=native_selected,
    ).max(1).values
    candidate_margin = _scores(
        query,
        positive,
        positive_mask,
        native=native,
        selected_lookup=lookup,
        selected_descriptors=selected_candidate,
    ).max(1).values - _scores(
        query,
        negative,
        negative_mask,
        native=native,
        selected_lookup=lookup,
        selected_descriptors=selected_candidate,
    ).max(1).values
    native_wrong = int((native_margin < -1e-5).sum())
    broken = int((candidate_margin < -1e-6).sum())
    floor_violations = int(
        (
            candidate_margin
            < native_margin - float(clean_margin_slack) - 1e-6
        ).sum()
    )
    cosine = (native_selected * selected_candidate).sum(1).clamp(-1.0, 1.0)
    angles = torch.rad2deg(torch.acos(cosine))
    angular_violations = int(
        (angles > float(maximum_angle_deg) + 0.05).sum()
    )
    passed = bool(
        changed_outside == 0
        and native_wrong == 0
        and broken == 0
        and floor_violations == 0
        and angular_violations == 0
    )
    return {
        "schema": "lafgs_v20_materialized_sparse_action_audit",
        "version": 1,
        "passed": passed,
        "protection_row_count": protect_count,
        "native_wrong_winner_count": native_wrong,
        "broken_protection_row_count": broken,
        "protection_margin_floor_violation_count": floor_violations,
        "changed_outside_selected_count": changed_outside,
        "selected_unchanged_count": selected_unchanged,
        "materialized_angular_cap_violation_count": angular_violations,
        "maximum_observed_angle_deg": float(angles.max().cpu()),
        "minimum_candidate_protection_margin": float(candidate_margin.min().cpu()),
        "minimum_protection_margin_gain": float(
            (candidate_margin - native_margin).min().cpu()
        ),
    }


def train_sparse_anchor_descriptors(
    *,
    anchor_features: torch.Tensor,
    evidence: Mapping,
    mode: str = "positive_only",
    maximum_angle_deg: float = 5.0,
    steps: int = 400,
    batch_size: int = 512,
    learning_rate: float = 0.05,
    temperature: float = 0.05,
    clean_margin_slack: float = 0.002,
    clean_protection_weight: float = 4.0,
    angular_regularization_weight: float = 0.01,
    minimum_repair_margin_gain: float = 0.001,
    minimum_coordinate_ranking_gain: float = 1e-5,
    maximum_selected_anchor_count: int = 4096,
    seed: int = 20260831,
    device: str | torch.device = "cuda",
    strong_feedback_authorized: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Train a sparse map descriptor proposal from sealed V20 evidence."""

    if mode not in {"positive_only", "positive_and_repeated_negative"}:
        raise ValueError("unsupported V20 sparse descriptor mode")
    if not 0.0 < float(maximum_angle_deg) < 90.0:
        raise ValueError("V20 descriptor angle must lie in (0, 90) degrees")
    finite_parameters = (
        learning_rate,
        temperature,
        clean_margin_slack,
        clean_protection_weight,
        angular_regularization_weight,
        minimum_repair_margin_gain,
        minimum_coordinate_ranking_gain,
    )
    if not all(math.isfinite(float(value)) for value in finite_parameters):
        raise ValueError("V20 optimization parameters must be finite")
    if (
        min(int(steps), int(batch_size), int(maximum_selected_anchor_count)) < 1
        or float(learning_rate) <= 0.0
        or float(temperature) <= 0.0
        or float(clean_margin_slack) < 0.0
        or float(clean_protection_weight) < 0.0
        or float(angular_regularization_weight) < 0.0
        or float(minimum_repair_margin_gain) < 0.0
        or float(minimum_coordinate_ranking_gain) < 0.0
    ):
        raise ValueError("V20 optimization parameters must be positive")
    if not (
        evidence.get("schema") == "lafgs_v20_topk_competition_evidence"
        and evidence.get("uses_test_queries") is False
        and evidence.get("loo_used") is False
    ):
        raise ValueError("V20 sparse repair evidence contract differs")
    sealed_authorization = bool(evidence.get("strong_feedback_authorized", False))
    if bool(strong_feedback_authorized) != sealed_authorization:
        raise ValueError("V20 strong-feedback authorization is not evidence-bound")

    original_cpu = torch.as_tensor(anchor_features).clone()
    original_float = original_cpu.float()
    if (
        original_float.ndim != 2
        or not bool(torch.isfinite(original_float).all())
        or bool((torch.linalg.norm(original_float, dim=1) <= 1e-8).any())
    ):
        raise ValueError("V20 Anchor descriptors must be finite rows")
    native_cpu = F.normalize(original_float, dim=1)
    repair_query_raw = torch.as_tensor(
        evidence["repair_query_descriptors"]
    ).float()
    if (
        repair_query_raw.ndim != 2
        or repair_query_raw.shape[1] != native_cpu.shape[1]
    ):
        raise ValueError("V20 repair Query descriptor dimension differs")
    repair_query_cpu = F.normalize(repair_query_raw, dim=1)
    repair_count = repair_query_cpu.shape[0]
    if repair_count == 0:
        raise ValueError("V20 sparse repair requires competition rows")
    if not bool(torch.isfinite(repair_query_cpu).all()) or bool(
        (torch.linalg.norm(repair_query_cpu, dim=1) <= 1e-8).any()
    ):
        raise ValueError("V20 repair Query descriptors must be finite nonzero rows")
    repair_positive, repair_positive_mask = _padded_csr(
        evidence["repair_positive_offsets"],
        evidence["repair_positive_anchor_rows"],
        row_count=repair_count,
        value_count=native_cpu.shape[0],
    )
    repair_negative, repair_negative_mask = _padded_csr(
        evidence["repair_negative_offsets"],
        evidence["repair_negative_anchor_rows"],
        row_count=repair_count,
        value_count=native_cpu.shape[0],
    )
    weights_cpu = torch.as_tensor(evidence["repair_sample_weights"]).float().reshape(-1)
    if (
        weights_cpu.numel() != repair_count
        or not bool(torch.isfinite(weights_cpu).all())
        or bool((weights_cpu <= 0).any())
    ):
        raise ValueError("V20 repair weights do not align or are non-positive")

    wrong_winners = torch.as_tensor(
        evidence["repair_wrong_winner_anchor_rows"]
    ).long().reshape(-1)
    if (
        wrong_winners.numel() != repair_count
        or int(wrong_winners.min()) < 0
        or int(wrong_winners.max()) >= native_cpu.shape[0]
    ):
        raise ValueError("V20 repair wrong winners do not align with rows")
    clean_support_counts = torch.as_tensor(
        evidence.get(
            "repair_wrong_winner_clean_support_family_counts",
            torch.zeros(repair_count, dtype=torch.long),
        )
    ).long().reshape(-1)
    if clean_support_counts.numel() != repair_count or bool(
        (clean_support_counts < 0).any()
    ):
        raise ValueError("V20 wrong-winner clean support does not align")
    minimum_negative_support = int(
        evidence.get("minimum_negative_action_clean_pose_families", 2)
    )
    if minimum_negative_support < 2:
        raise ValueError("V20 negative action support gate is not cross-family")
    expected_negative_action_rows = torch.unique(
        wrong_winners[clean_support_counts >= minimum_negative_support],
        sorted=True,
    )
    negative_action_rows = torch.as_tensor(
        evidence.get("negative_action_anchor_rows", torch.empty(0))
    ).long().reshape(-1)
    if not torch.equal(negative_action_rows, expected_negative_action_rows):
        raise ValueError("V20 negative actions lack exact clean-positive support")

    selected_parts = [
        torch.as_tensor(evidence["repair_positive_anchor_rows"]).long()
    ]
    if mode == "positive_and_repeated_negative":
        if int(evidence.get("version", 0)) < 2:
            raise ValueError("V20 negative action requires version-2 evidence")
        if negative_action_rows.numel() == 0:
            raise ValueError(
                "V20 negative action requires a clean-positive-supported winner"
            )
        selected_parts.append(negative_action_rows)
    selected_rows = torch.unique(torch.cat(selected_parts), sorted=True)
    if selected_rows.numel() == 0 or int(selected_rows.min()) < 0 or int(selected_rows.max()) >= native_cpu.shape[0]:
        raise ValueError("V20 selected descriptor rows are outside the map")
    if selected_rows.numel() > int(maximum_selected_anchor_count):
        raise ValueError("V20 sparse action exceeds the selected-Anchor cap")

    protect_query_raw = torch.as_tensor(
        evidence["protection_query_descriptors"]
    ).float()
    if (
        protect_query_raw.ndim != 2
        or protect_query_raw.shape[1] != native_cpu.shape[1]
    ):
        raise ValueError("V20 protection Query descriptor dimension differs")
    protect_query_cpu = F.normalize(protect_query_raw, dim=1)
    protect_count = protect_query_cpu.shape[0]
    if protect_count == 0:
        raise ValueError("V20 sparse repair requires clean protection rows")
    if protect_count and (
        not bool(torch.isfinite(protect_query_cpu).all())
        or bool((torch.linalg.norm(protect_query_cpu, dim=1) <= 1e-8).any())
    ):
        raise ValueError("V20 protection Query descriptors must be finite nonzero rows")
    if protect_count:
        protect_positive, protect_positive_mask = _padded_csr(
            evidence["protection_positive_offsets"],
            evidence["protection_positive_anchor_rows"],
            row_count=protect_count,
            value_count=native_cpu.shape[0],
        )
        protect_negative, protect_negative_mask = _padded_csr(
            evidence["protection_negative_offsets"],
            evidence["protection_negative_anchor_rows"],
            row_count=protect_count,
            value_count=native_cpu.shape[0],
        )
        initial_margin_cpu = torch.as_tensor(
            evidence["protection_initial_margin"]
        ).float().reshape(-1)
        if initial_margin_cpu.numel() != protect_count:
            raise ValueError("V20 protection margins do not align")
        if not bool(torch.isfinite(initial_margin_cpu).all()):
            raise ValueError("V20 protection margins must be finite")
    else:
        protect_positive = protect_negative = torch.empty((0, 1), dtype=torch.long)
        protect_positive_mask = protect_negative_mask = torch.empty(
            (0, 1), dtype=torch.bool
        )
        initial_margin_cpu = torch.empty(0)

    target = torch.device(device)
    native = native_cpu.to(target)
    selected_rows_device = selected_rows.to(target)
    selected_native = native[selected_rows_device]
    lookup = torch.full((native.shape[0],), -1, dtype=torch.long, device=target)
    lookup[selected_rows_device] = torch.arange(selected_rows.numel(), device=target)
    raw_tangent = torch.zeros_like(selected_native, requires_grad=True)
    optimizer = torch.optim.Adam([raw_tangent], lr=float(learning_rate))
    generator = torch.Generator().manual_seed(int(seed))
    torch.manual_seed(int(seed))

    repair_query = repair_query_cpu.to(target)
    repair_positive = repair_positive.to(target)
    repair_positive_mask = repair_positive_mask.to(target)
    repair_negative = repair_negative.to(target)
    repair_negative_mask = repair_negative_mask.to(target)
    weights = weights_cpu.to(target)
    protect_query = protect_query_cpu.to(target)
    protect_positive = protect_positive.to(target)
    protect_positive_mask = protect_positive_mask.to(target)
    protect_negative = protect_negative.to(target)
    protect_negative_mask = protect_negative_mask.to(target)
    identity_lookup = torch.full_like(lookup, -1)
    native_protection_margin = native.new_empty((0,))
    if protect_count:
        with torch.inference_mode():
            native_protection_margin = _scores(
                protect_query,
                protect_positive,
                protect_positive_mask,
                native=native,
                selected_lookup=identity_lookup,
                selected_descriptors=selected_native,
            ).max(1).values - _scores(
                protect_query,
                protect_negative,
                protect_negative_mask,
                native=native,
                selected_lookup=identity_lookup,
                selected_descriptors=selected_native,
            ).max(1).values
        if bool((native_protection_margin < -1e-5).any()):
            raise ValueError("V20 clean protection contains a native wrong winner")
    history = []
    for step in range(int(steps)):
        selected_descriptors = _bounded_descriptors(
            selected_native, raw_tangent, maximum_angle_deg
        )
        rows = torch.randint(
            repair_count,
            (min(int(batch_size), repair_count),),
            generator=generator,
        ).to(target)
        positive_scores = _scores(
            repair_query[rows],
            repair_positive[rows],
            repair_positive_mask[rows],
            native=native,
            selected_lookup=lookup,
            selected_descriptors=selected_descriptors,
        )
        negative_scores = _scores(
            repair_query[rows],
            repair_negative[rows],
            repair_negative_mask[rows],
            native=native,
            selected_lookup=lookup,
            selected_descriptors=selected_descriptors,
        )
        ranking = _per_positive_listwise_ranking(
            positive_scores,
            repair_positive_mask[rows],
            negative_scores,
            temperature=float(temperature),
        )
        ranking_loss = (ranking * weights[rows]).sum() / weights[rows].sum()
        angular_regularization = (
            1.0 - (selected_native * selected_descriptors).sum(1)
        ).mean()

        protection_loss = ranking_loss.new_zeros(())
        if protect_count:
            protect_rows = torch.randint(
                protect_count,
                (min(int(batch_size), protect_count),),
                generator=generator,
            ).to(target)
            protect_positive_scores = _scores(
                protect_query[protect_rows],
                protect_positive[protect_rows],
                protect_positive_mask[protect_rows],
                native=native,
                selected_lookup=lookup,
                selected_descriptors=selected_descriptors,
            )
            protect_negative_scores = _scores(
                protect_query[protect_rows],
                protect_negative[protect_rows],
                protect_negative_mask[protect_rows],
                native=native,
                selected_lookup=lookup,
                selected_descriptors=selected_descriptors,
            )
            new_margin = protect_positive_scores.max(1).values - protect_negative_scores.max(1).values
            floor = (
                native_protection_margin[protect_rows]
                - float(clean_margin_slack)
            )
            protection_loss = F.relu(floor - new_margin).mean()
        loss = (
            ranking_loss
            + float(clean_protection_weight) * protection_loss
            + float(angular_regularization_weight) * angular_regularization
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in {0, int(steps) - 1} or (step + 1) % 50 == 0:
            history.append(
                {
                    "step": step + 1,
                    "ranking_loss": float(ranking_loss.detach()),
                    "clean_protection_loss": float(protection_loss.detach()),
                    "angular_regularization": float(
                        angular_regularization.detach()
                    ),
                    "total_loss": float(loss.detach()),
                }
            )

    with torch.inference_mode():
        before_positive_scores = _scores(
            repair_query,
            repair_positive,
            repair_positive_mask,
            native=native,
            selected_lookup=identity_lookup,
            selected_descriptors=selected_native,
        )
        before_negative = _scores(
            repair_query,
            repair_negative,
            repair_negative_mask,
            native=native,
            selected_lookup=identity_lookup,
            selected_descriptors=selected_native,
        ).max(1).values
        before_best_positive = before_positive_scores.max(1).values
        before_worst_positive = before_positive_scores.masked_fill(
            ~repair_positive_mask, torch.inf
        ).min(1).values
        before_margin = before_best_positive - before_negative
        before_worst_margin = before_worst_positive - before_negative
        before_positive_wins = (
            before_positive_scores > before_negative[:, None]
        ) & repair_positive_mask

        applied_scale = 0.0
        selected_descriptors = selected_native
        final_margin = native_protection_margin
        # Search from the smallest action upward.  A proposal is retained only
        # when it has a material repair-margin gain without losing positive
        # pair wins, while clean protection remains a hard postcondition.
        for scale in (
            0.0009765625,
            0.001953125,
            0.00390625,
            0.0078125,
            0.015625,
            0.03125,
            0.0625,
            0.125,
            0.25,
            0.5,
            1.0,
        ):
            proposal = _bounded_descriptors(
                selected_native,
                raw_tangent * float(scale),
                maximum_angle_deg,
            )
            if protect_count:
                candidate_margin = _scores(
                    protect_query,
                    protect_positive,
                    protect_positive_mask,
                    native=native,
                    selected_lookup=lookup,
                    selected_descriptors=proposal,
                ).max(1).values - _scores(
                    protect_query,
                    protect_negative,
                    protect_negative_mask,
                    native=native,
                    selected_lookup=lookup,
                    selected_descriptors=proposal,
                ).max(1).values
                safe = bool(
                    (
                        candidate_margin
                        >= native_protection_margin - float(clean_margin_slack) - 1e-6
                    ).all()
                ) and bool((candidate_margin >= -1e-6).all())
            else:
                candidate_margin = native_protection_margin
                safe = True
            if not safe:
                continue
            candidate_positive_scores = _scores(
                repair_query,
                repair_positive,
                repair_positive_mask,
                native=native,
                selected_lookup=lookup,
                selected_descriptors=proposal,
            )
            candidate_negative = _scores(
                repair_query,
                repair_negative,
                repair_negative_mask,
                native=native,
                selected_lookup=lookup,
                selected_descriptors=proposal,
            ).max(1).values
            candidate_best_margin = (
                candidate_positive_scores.max(1).values - candidate_negative
            )
            candidate_worst_margin = (
                candidate_positive_scores.masked_fill(
                    ~repair_positive_mask, torch.inf
                ).min(1).values
                - candidate_negative
            )
            candidate_positive_wins = (
                candidate_positive_scores > candidate_negative[:, None]
            ) & repair_positive_mask
            useful = (
                float(
                    (
                        candidate_worst_margin.mean()
                        - before_worst_margin.mean()
                    ).cpu()
                )
                >= float(minimum_repair_margin_gain) - 1e-8
                and bool(
                    (
                        candidate_positive_wins
                        | ~before_positive_wins
                    ).all()
                )
                and float(candidate_best_margin.mean().cpu())
                >= float(before_margin.mean().cpu()) - 1e-6
            )
            if useful:
                applied_scale = float(scale)
                selected_descriptors = proposal
                final_margin = candidate_margin
                break
        anchor_scales = torch.full(
            (selected_rows.numel(),),
            applied_scale,
            dtype=selected_native.dtype,
            device=target,
        )
        expanded_anchor_rows: set[int] = set()
        if applied_scale > 0.0:
            current_positive_scores = _scores(
                repair_query,
                repair_positive,
                repair_positive_mask,
                native=native,
                selected_lookup=lookup,
                selected_descriptors=selected_descriptors,
            )
            current_negative_scores = _scores(
                repair_query,
                repair_negative,
                repair_negative_mask,
                native=native,
                selected_lookup=lookup,
                selected_descriptors=selected_descriptors,
            )
            current_negative = current_negative_scores.max(1).values
            current_best_margin = (
                current_positive_scores.max(1).values - current_negative
            )
            current_worst_margin = (
                current_positive_scores.masked_fill(
                    ~repair_positive_mask, torch.inf
                ).min(1).values
                - current_negative
            )
            current_positive_wins = (
                current_positive_scores > current_negative[:, None]
            ) & repair_positive_mask
            current_ranking = _per_positive_listwise_ranking(
                current_positive_scores,
                repair_positive_mask,
                current_negative_scores,
                temperature=float(temperature),
            )
            current_ranking_loss = (
                current_ranking * weights
            ).sum() / weights.sum()
            scale_levels = (
                0.0009765625,
                0.001953125,
                0.00390625,
                0.0078125,
                0.015625,
                0.03125,
                0.0625,
                0.125,
                0.25,
                0.5,
                1.0,
            )
            # Grow one Anchor at a time from the minimum globally useful seed.
            # A coordinate step survives only if it is exactly clean-safe and
            # improves the sealed per-positive objective without losing wins.
            for local_row in range(selected_rows.numel()):
                for coordinate_scale in scale_levels:
                    if coordinate_scale <= float(anchor_scales[local_row]) + 1e-12:
                        continue
                    candidate_scales = anchor_scales.clone()
                    candidate_scales[local_row] = float(coordinate_scale)
                    coordinate_proposal = _bounded_descriptors(
                        selected_native,
                        raw_tangent * candidate_scales[:, None],
                        maximum_angle_deg,
                    )
                    coordinate_protection_margin = _scores(
                        protect_query,
                        protect_positive,
                        protect_positive_mask,
                        native=native,
                        selected_lookup=lookup,
                        selected_descriptors=coordinate_proposal,
                    ).max(1).values - _scores(
                        protect_query,
                        protect_negative,
                        protect_negative_mask,
                        native=native,
                        selected_lookup=lookup,
                        selected_descriptors=coordinate_proposal,
                    ).max(1).values
                    coordinate_safe = bool(
                        (
                            coordinate_protection_margin
                            >= native_protection_margin
                            - float(clean_margin_slack)
                            - 1e-6
                        ).all()
                    ) and bool((coordinate_protection_margin >= -1e-6).all())
                    if not coordinate_safe:
                        break
                    coordinate_positive_scores = _scores(
                        repair_query,
                        repair_positive,
                        repair_positive_mask,
                        native=native,
                        selected_lookup=lookup,
                        selected_descriptors=coordinate_proposal,
                    )
                    coordinate_negative_scores = _scores(
                        repair_query,
                        repair_negative,
                        repair_negative_mask,
                        native=native,
                        selected_lookup=lookup,
                        selected_descriptors=coordinate_proposal,
                    )
                    coordinate_negative = coordinate_negative_scores.max(1).values
                    coordinate_best_margin = (
                        coordinate_positive_scores.max(1).values
                        - coordinate_negative
                    )
                    coordinate_worst_margin = (
                        coordinate_positive_scores.masked_fill(
                            ~repair_positive_mask, torch.inf
                        ).min(1).values
                        - coordinate_negative
                    )
                    coordinate_positive_wins = (
                        coordinate_positive_scores
                        > coordinate_negative[:, None]
                    ) & repair_positive_mask
                    coordinate_ranking = _per_positive_listwise_ranking(
                        coordinate_positive_scores,
                        repair_positive_mask,
                        coordinate_negative_scores,
                        temperature=float(temperature),
                    )
                    coordinate_ranking_loss = (
                        coordinate_ranking * weights
                    ).sum() / weights.sum()
                    coordinate_useful = (
                        float(
                            (current_ranking_loss - coordinate_ranking_loss).cpu()
                        )
                        >= float(minimum_coordinate_ranking_gain) - 1e-12
                        and bool(
                            (
                                coordinate_positive_wins
                                | ~current_positive_wins
                            ).all()
                        )
                        and float(coordinate_best_margin.mean().cpu())
                        >= float(current_best_margin.mean().cpu()) - 1e-6
                        and float(coordinate_worst_margin.mean().cpu())
                        >= float(current_worst_margin.mean().cpu()) - 1e-6
                    )
                    if not coordinate_useful:
                        continue
                    anchor_scales = candidate_scales
                    selected_descriptors = coordinate_proposal
                    final_margin = coordinate_protection_margin
                    current_positive_scores = coordinate_positive_scores
                    current_negative_scores = coordinate_negative_scores
                    current_negative = coordinate_negative
                    current_best_margin = coordinate_best_margin
                    current_worst_margin = coordinate_worst_margin
                    current_positive_wins = coordinate_positive_wins
                    current_ranking_loss = coordinate_ranking_loss
                    expanded_anchor_rows.add(int(selected_rows[local_row]))
        updated = original_cpu.clone()
        if applied_scale > 0.0:
            updated[selected_rows] = selected_descriptors.cpu().to(updated.dtype)
            # The serialized map keeps the baseline dtype.  All final safety
            # and ranking statistics must therefore use the materialized
            # descriptors, not the pre-cast float32 optimizer proposal.
            selected_descriptors = F.normalize(
                updated[selected_rows].float().to(target), dim=1
            )
            final_margin = _scores(
                protect_query,
                protect_positive,
                protect_positive_mask,
                native=native,
                selected_lookup=lookup,
                selected_descriptors=selected_descriptors,
            ).max(1).values - _scores(
                protect_query,
                protect_negative,
                protect_negative_mask,
                native=native,
                selected_lookup=lookup,
                selected_descriptors=selected_descriptors,
            ).max(1).values
        after_positive_scores = _scores(
            repair_query,
            repair_positive,
            repair_positive_mask,
            native=native,
            selected_lookup=lookup,
            selected_descriptors=selected_descriptors,
        )
        after_negative = _scores(
            repair_query,
            repair_negative,
            repair_negative_mask,
            native=native,
            selected_lookup=lookup,
            selected_descriptors=selected_descriptors,
        ).max(1).values
        after_best_positive = after_positive_scores.max(1).values
        after_worst_positive = after_positive_scores.masked_fill(
            ~repair_positive_mask, torch.inf
        ).min(1).values
        after_margin = after_best_positive - after_negative
        after_worst_margin = after_worst_positive - after_negative
        after_positive_wins = (
            after_positive_scores > after_negative[:, None]
        ) & repair_positive_mask
        cosine = (selected_native * selected_descriptors).sum(1).clamp(-1.0, 1.0)
        angles = (
            torch.zeros_like(cosine)
            if applied_scale == 0.0
            else torch.rad2deg(torch.acos(cosine))
        )
        angular_cap_violations = int(
            (angles > float(maximum_angle_deg) + 0.05).sum()
        )
        broken_protection = int((final_margin < -1e-6).sum())
        margin_floor_violations = int(
            (
                final_margin
                < native_protection_margin - float(clean_margin_slack) - 1e-6
            ).sum()
        )
        clean_protection_passed = bool(
            broken_protection == 0
            and margin_floor_violations == 0
            and angular_cap_violations == 0
        )
    report = {
        "schema": "lafgs_v20_sparse_anchor_descriptor_training",
        "version": 2,
        "uses_test_queries": False,
        "loo_used": False,
        "mode": mode,
        "positive_objective": "per_positive_listwise_mean",
        "proposal_only": True,
        "strong_feedback_authorized": bool(strong_feedback_authorized),
        "deployment_status": (
            "REJECTED_CLEAN_PROTECTION"
            if not clean_protection_passed
            else (
                "NO_EFFECT_AFTER_CLEAN_BACKOFF"
                if applied_scale == 0.0
                else (
                    "REQUIRES_EXACT_POSE_CONTROL"
                    if strong_feedback_authorized
                    else "ANALYSIS_ONLY_TEACHER_NOT_AUTHORIZED"
                )
            )
        ),
        "query_descriptor_action": "native_unchanged",
        "feedback_descriptors_copied_into_map": False,
        "selected_anchor_rows": selected_rows.cpu(),
        "selected_anchor_count": int(selected_rows.numel()),
        "negative_action_anchor_count": int(
            negative_action_rows.numel()
            if mode == "positive_and_repeated_negative"
            else 0
        ),
        "minimum_negative_action_clean_pose_families": (
            minimum_negative_support
        ),
        "unsupported_wrong_winner_anchor_count": int(
            torch.unique(
                wrong_winners[clean_support_counts < minimum_negative_support]
            ).numel()
        ),
        "maximum_angle_deg": float(maximum_angle_deg),
        "maximum_observed_angle_deg": float(angles.max().cpu()),
        "per_anchor_observed_angle_deg": angles.cpu(),
        "global_seed_action_scale": applied_scale,
        "post_training_action_scale": (
            float(anchor_scales.min().cpu()) if applied_scale > 0.0 else 0.0
        ),
        "minimum_repair_margin_gain": float(minimum_repair_margin_gain),
        "minimum_coordinate_ranking_gain": float(
            minimum_coordinate_ranking_gain
        ),
        "per_anchor_action_scales": anchor_scales.cpu(),
        "maximum_applied_anchor_scale": float(anchor_scales.max().cpu()),
        "mean_applied_anchor_scale": float(anchor_scales.mean().cpu()),
        "coordinate_expanded_anchor_count": len(expanded_anchor_rows),
        "angular_regularization_weight": float(angular_regularization_weight),
        "maximum_selected_anchor_count": int(maximum_selected_anchor_count),
        "repair_row_count": repair_count,
        "repair_margin_before_mean": float(before_margin.mean().cpu()),
        "repair_margin_after_mean": float(after_margin.mean().cpu()),
        "repair_worst_positive_margin_before_mean": float(
            before_worst_margin.mean().cpu()
        ),
        "repair_worst_positive_margin_after_mean": float(
            after_worst_margin.mean().cpu()
        ),
        "repair_positive_pair_win_rate_before": float(
            before_positive_wins.sum().cpu()
            / repair_positive_mask.sum().cpu()
        ),
        "repair_positive_pair_win_rate_after": float(
            after_positive_wins.sum().cpu()
            / repair_positive_mask.sum().cpu()
        ),
        "all_positive_winner_row_count_before": int(
            (before_worst_margin > 0.0).sum().cpu()
        ),
        "all_positive_winner_row_count_after": int(
            (after_worst_margin > 0.0).sum().cpu()
        ),
        "recovered_top1_row_count": int(((before_margin <= 0.0) & (after_margin > 0.0)).sum().cpu()),
        "regressed_repair_row_count": int(
            (after_margin < before_margin - 1e-6).sum().cpu()
        ),
        "positive_win_nonregression_passed": bool(
            (after_positive_wins | ~before_positive_wins).all()
        ),
        "protection_row_count": protect_count,
        "protection_scope": (
            "sealed_design_clean_rows; independent_exact_confirmation_required"
        ),
        "broken_protection_row_count": broken_protection,
        "protection_margin_floor_violation_count": margin_floor_violations,
        "materialized_angular_cap_violation_count": angular_cap_violations,
        "clean_margin_slack": float(clean_margin_slack),
        "clean_protection_passed": clean_protection_passed,
        "requires_exact_pose_control": True,
        "requires_independent_confirmation": True,
        "history": history,
    }
    return updated.cpu(), report


__all__ = [
    "audit_materialized_sparse_action",
    "train_sparse_anchor_descriptors",
]
