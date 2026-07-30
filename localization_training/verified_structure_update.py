"""Evidence-gated structure proposals from localization coverage failures."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from localization_training.harmful_outcome_triage import COVERAGE_FAILURE


@dataclass(frozen=True)
class VerifiedStructureConfig:
    minimum_trajectories: int = 3
    minimum_events: int = 3
    maximum_additions: int = 128
    descriptor_trim_fraction: float = 0.2
    maximum_retirements: int = 128
    minimum_retire_harmful_trajectories: int = 3
    minimum_retire_harmful_events: int = 3


def collect_coverage_evidence(
    triage: dict,
    *,
    config: VerifiedStructureConfig,
) -> tuple[list[dict], dict]:
    """Aggregate explicit canonical positives across independent trajectories."""

    evidence: dict[int, list[dict]] = defaultdict(list)
    for record in triage["records"]:
        trajectory = str(record["query_name"]).split("/", 1)[0]
        categories = torch.as_tensor(record["category"]).long()
        rows = torch.as_tensor(record["query_rows"]).long()
        offsets = torch.as_tensor(
            record["canonical_positive_offsets"]
        ).long()
        candidates = torch.as_tensor(
            record["canonical_positive_indices"]
        ).long()
        errors = torch.as_tensor(
            record["canonical_positive_reprojection_errors_px"]
        ).float()
        mass = torch.as_tensor(
            record["canonical_positive_contribution_mass"]
        ).float()
        wrong = torch.as_tensor(
            record["top1_anchor_indices"]
        ).long()
        for local in torch.where(categories == COVERAGE_FAILURE)[0].tolist():
            start = int(offsets[local])
            stop = int(offsets[local + 1])
            for packed in range(start, stop):
                evidence[int(candidates[packed])].append(
                    {
                        "query_index": int(record["query_index"]),
                        "query_name": str(record["query_name"]),
                        "query_row": int(rows[local]),
                        "trajectory": trajectory,
                        "wrong_anchor": int(wrong[local]),
                        "reprojection_error_px": float(errors[packed]),
                        "contribution_mass": float(mass[packed]),
                    }
                )
    accepted = []
    rejected = Counter()
    for canonical_index, events in evidence.items():
        trajectories = {event["trajectory"] for event in events}
        if len(trajectories) < int(config.minimum_trajectories):
            rejected["insufficient_trajectories"] += 1
            continue
        if len(events) < int(config.minimum_events):
            rejected["insufficient_events"] += 1
            continue
        accepted.append(
            {
                "canonical_index": int(canonical_index),
                "events": events,
                "trajectory_count": len(trajectories),
                "event_count": len(events),
                "mean_contribution_mass": float(
                    sum(event["contribution_mass"] for event in events)
                    / len(events)
                ),
                "mean_reprojection_error_px": float(
                    sum(
                        event["reprojection_error_px"]
                        for event in events
                    )
                    / len(events)
                ),
            }
        )
    accepted.sort(
        key=lambda value: (
            -int(value["trajectory_count"]),
            -int(value["event_count"]),
            -float(value["mean_contribution_mass"]),
            float(value["mean_reprojection_error_px"]),
            int(value["canonical_index"]),
        )
    )
    accepted = accepted[: max(int(config.maximum_additions), 0)]
    return accepted, {
        "canonical_candidate_count": len(evidence),
        "verified_candidate_count": len(accepted),
        "rejected": dict(rejected),
    }


def robust_structure_descriptor(
    observation_features: torch.Tensor,
    observation_weights: torch.Tensor,
    canonical_feature: torch.Tensor,
    *,
    trim_fraction: float,
) -> torch.Tensor:
    """Fuse cross-view evidence around a weighted descriptor medoid."""

    observations = F.normalize(
        torch.as_tensor(observation_features).float(), dim=1
    )
    weights = torch.as_tensor(observation_weights).float().reshape(-1)
    canonical = F.normalize(
        torch.as_tensor(canonical_feature).float().reshape(1, -1), dim=1
    )[0]
    if len(observations) != len(weights) or not len(observations):
        raise ValueError("structure descriptor observations must align")
    weights = weights.clamp_min(1e-8)
    similarity = observations @ observations.T
    medoid = int(torch.argmax(similarity @ (weights / weights.sum())))
    keep_count = max(
        1,
        int(round(len(observations) * (1.0 - float(trim_fraction)))),
    )
    keep = torch.argsort(
        similarity[medoid], descending=True, stable=True
    )[:keep_count]
    kept_weights = weights[keep]
    prior_weight = kept_weights.median()
    fused = (
        (observations[keep] * kept_weights[:, None]).sum(dim=0)
        + canonical * prior_weight
    ) / (kept_weights.sum() + prior_weight).clamp_min(1e-8)
    return F.normalize(fused.reshape(1, -1), dim=1)[0]


def safe_retirement_candidates(
    *,
    active_count: int,
    base_anchor_count: int,
    triage: dict,
    dynamic_outcomes: dict,
    family_parent_indices: torch.Tensor,
    config: VerifiedStructureConfig,
) -> list[int]:
    """Find non-base anchors with repeated harm and no observed clean use."""

    clean_count = torch.zeros(active_count, dtype=torch.long)
    for record in dynamic_outcomes["records"]:
        anchors = torch.as_tensor(
            record["top1_anchor_indices"]
        ).long()
        clean = (
            torch.as_tensor(
                record["gt_reprojection_errors_px"]
            ).float()
            <= 2.0
        )
        clean_count.scatter_add_(
            0,
            anchors[clean],
            torch.ones(int(clean.sum()), dtype=torch.long),
        )
    harmful_events = Counter()
    harmful_trajectories: dict[int, set[str]] = defaultdict(set)
    for record in triage["records"]:
        trajectory = str(record["query_name"]).split("/", 1)[0]
        categories = torch.as_tensor(record["category"]).long()
        wrong = torch.as_tensor(
            record["top1_anchor_indices"]
        ).long()
        for anchor in wrong[categories == COVERAGE_FAILURE].tolist():
            harmful_events[int(anchor)] += 1
            harmful_trajectories[int(anchor)].add(trajectory)
    family_parents = set(
        torch.as_tensor(family_parent_indices).long().tolist()
    )
    candidates = []
    for anchor, event_count in harmful_events.items():
        if (
            anchor < int(base_anchor_count)
            or anchor >= int(active_count)
            or anchor in family_parents
            or int(clean_count[anchor]) > 0
            or event_count
            < int(config.minimum_retire_harmful_events)
            or len(harmful_trajectories[anchor])
            < int(config.minimum_retire_harmful_trajectories)
        ):
            continue
        candidates.append(
            (
                -len(harmful_trajectories[anchor]),
                -event_count,
                anchor,
            )
        )
    candidates.sort()
    return [
        int(value[2])
        for value in candidates[: max(int(config.maximum_retirements), 0)]
    ]


def serialize_config(config: VerifiedStructureConfig) -> dict:
    return asdict(config)
