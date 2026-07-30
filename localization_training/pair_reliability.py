"""Cross-trajectory reliability estimates for directed candidate switches."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
from typing import Iterable


@dataclass(frozen=True)
class PairOutcome:
    trajectory: str
    confusing_anchor: int
    correct_anchor: int
    success: bool


@dataclass(frozen=True)
class PairReliability:
    attempts: int
    successes: int
    positive_trajectories: int
    empirical_precision: float
    wilson_lower_bound: float


def wilson_lower_bound(successes: int, attempts: int, *, z: float = 1.0) -> float:
    """Return the one-sided Wilson lower bound for a Bernoulli rate."""

    attempts = int(attempts)
    successes = int(successes)
    if attempts <= 0:
        return 0.0
    if successes < 0 or successes > attempts:
        raise ValueError("successes must be between zero and attempts")
    z = max(float(z), 0.0)
    probability = successes / attempts
    denominator = 1.0 + z * z / attempts
    center = probability + z * z / (2.0 * attempts)
    radius = z * sqrt(
        probability * (1.0 - probability) / attempts
        + z * z / (4.0 * attempts * attempts)
    )
    return max(0.0, (center - radius) / denominator)


def aggregate_pair_reliability(
    outcomes: Iterable[PairOutcome],
    *,
    excluded_trajectory: str | None = None,
    z: float = 1.0,
) -> dict[tuple[int, int], PairReliability]:
    """Aggregate pair evidence, optionally excluding one deployment trajectory."""

    counts: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
    positive_trajectories: dict[tuple[int, int], set[str]] = defaultdict(set)
    for outcome in outcomes:
        if (
            excluded_trajectory is not None
            and outcome.trajectory == excluded_trajectory
        ):
            continue
        key = (int(outcome.confusing_anchor), int(outcome.correct_anchor))
        counts[key][0] += 1
        if bool(outcome.success):
            counts[key][1] += 1
            positive_trajectories[key].add(str(outcome.trajectory))
    result = {}
    for key, (attempts, successes) in counts.items():
        result[key] = PairReliability(
            attempts=attempts,
            successes=successes,
            positive_trajectories=len(positive_trajectories[key]),
            empirical_precision=successes / attempts,
            wilson_lower_bound=wilson_lower_bound(
                successes, attempts, z=float(z)
            ),
        )
    return result


def reliable_pair(
    reliability: PairReliability | None,
    *,
    minimum_attempts: int,
    minimum_successes: int,
    minimum_positive_trajectories: int,
    minimum_precision: float,
    minimum_wilson_lower_bound: float,
) -> bool:
    """Apply a conservative, interpretable deployment gate."""

    if reliability is None:
        return False
    return (
        reliability.attempts >= int(minimum_attempts)
        and reliability.successes >= int(minimum_successes)
        and reliability.positive_trajectories
        >= int(minimum_positive_trajectories)
        and reliability.empirical_precision >= float(minimum_precision)
        and reliability.wilson_lower_bound
        >= float(minimum_wilson_lower_bound)
    )
