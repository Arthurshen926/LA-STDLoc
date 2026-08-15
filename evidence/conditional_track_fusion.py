"""Conditional artifact-aware observation trimming for rendered Tracks.

Artifact evidence is deliberately not a descriptor weight.  It can only make
an observation eligible for removal when the descriptor is also an outlier
and the observation is neither identity-certified nor the final support of a
pose-view or mapping-sequence bin.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

import torch
import torch.nn.functional as F


MAXIMUM_TRIM_FRACTION = 0.20


def _sequence_name(name: str) -> str:
    return str(name).split("/", 1)[0]


def _aligned_vector(value: Any, *, count: int, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim != 1 or tensor.shape[0] != count:
        raise ValueError(f"{label} must be an exact [{count}] vector")
    return tensor


def conditional_artifact_keep_masks(
    *,
    payload: Mapping[str, Any],
    appearance_cache: Mapping[str, Any],
    artifact_cache: Mapping[str, Any],
    selected_tracks: torch.Tensor,
    maximum_trim_fraction: float = MAXIMUM_TRIM_FRACTION,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    """Build deterministic observation masks for conditional GWFF fusion."""
    if not 0.0 <= float(maximum_trim_fraction) <= 0.5:
        raise ValueError("maximum trim fraction must lie in [0, 0.5]")
    names = list(payload.get("query_names", ()))
    appearance = appearance_cache.get("queries", appearance_cache)
    artifact = artifact_cache.get("queries", artifact_cache)
    if not names or names != list(appearance) or names != list(artifact):
        raise ValueError("Track and cache query registries differ")
    tracks = payload.get("tracks")
    if not isinstance(tracks, Mapping):
        raise ValueError("Track payload has no observation table")
    required = {
        "track_index",
        "query_index",
        "keypoint_index",
        "confidence",
        "identity_positive_certified",
    }
    if not required <= set(tracks):
        raise ValueError("Track payload lacks conditional-fusion evidence")
    observation_track = torch.as_tensor(tracks["track_index"], dtype=torch.long)
    observation_query = torch.as_tensor(tracks["query_index"], dtype=torch.long)
    observation_keypoint = torch.as_tensor(tracks["keypoint_index"], dtype=torch.long)
    observation_confidence = torch.as_tensor(tracks["confidence"]).float()
    identity_certified = torch.as_tensor(
        tracks["identity_positive_certified"], dtype=torch.bool
    )
    observation_count = int(observation_track.numel())
    for label, value in (
        ("query", observation_query),
        ("keypoint", observation_keypoint),
        ("confidence", observation_confidence),
        ("identity certificate", identity_certified),
    ):
        if value.ndim != 1 or value.numel() != observation_count:
            raise ValueError(f"Track {label} rows do not align")
    query_bins = torch.as_tensor(payload.get("query_bins"), dtype=torch.long)
    if query_bins.ndim != 1 or query_bins.numel() != len(names):
        raise ValueError("Track query bins do not align with the query registry")
    selected_tracks = torch.as_tensor(selected_tracks, dtype=torch.long).reshape(-1)
    if selected_tracks.numel() == 0 or (
        selected_tracks.unique().numel() != selected_tracks.numel()
    ):
        raise ValueError("selected Track identities must be nonempty and unique")
    if bool((selected_tracks < 0).any()) or int(selected_tracks.max()) > int(
        observation_track.max()
    ):
        raise ValueError("selected Track identity is out of range")

    annotations: dict[str, dict[str, torch.Tensor]] = {}
    for name in names:
        row_count = int(torch.as_tensor(appearance[name]["native_keypoints"]).shape[0])
        artifact_reliability = _aligned_vector(
            artifact[name].get("native_artifact_reliability"),
            count=row_count,
            label=f"{name} artifact reliability",
        ).float()
        if not bool(torch.isfinite(artifact_reliability).all()) or not bool(
            ((artifact_reliability >= 0.0) & (artifact_reliability <= 1.0)).all()
        ):
            raise ValueError(f"{name} artifact reliability is invalid")
        annotations[name] = {
            "native_descriptor_fusion_keep_mask": torch.ones(
                row_count, dtype=torch.bool
            ),
            "native_conditional_artifact_trim_eligible": torch.zeros(
                row_count, dtype=torch.bool
            ),
            "native_conditional_descriptor_medoid_cosine": torch.ones(
                row_count, dtype=torch.float32
            ),
        }

    selected_lookup = torch.zeros(int(observation_track.max()) + 1, dtype=torch.bool)
    selected_lookup[selected_tracks] = True
    selected_observations = torch.nonzero(
        selected_lookup[observation_track], as_tuple=False
    ).reshape(-1)
    observation_keys = torch.stack(
        (
            observation_query[selected_observations],
            observation_keypoint[selected_observations],
        ),
        dim=1,
    )
    if torch.unique(observation_keys, dim=0).shape[0] != selected_observations.numel():
        raise ValueError("selected Tracks reuse a query/keypoint observation")

    sorted_observations = selected_observations[
        torch.argsort(observation_track[selected_observations], stable=True)
    ]
    sorted_tracks = observation_track[sorted_observations]
    unique_tracks, counts = torch.unique_consecutive(sorted_tracks, return_counts=True)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    selected_set = set(selected_tracks.tolist())
    trimmed_observation_count = 0
    initially_eligible_count = 0
    protected_identity_count = 0
    protected_unique_view_count = 0
    protected_unique_sequence_count = 0
    changed_track_count = 0
    maximum_observed_trim_fraction = 0.0

    for row, track in enumerate(unique_tracks.tolist()):
        if int(track) not in selected_set:
            continue
        observations = sorted_observations[offsets[row] : offsets[row + 1]]
        queries = observation_query[observations]
        keypoints = observation_keypoint[observations]
        validity = torch.as_tensor(
            [
                bool(
                    torch.as_tensor(
                        appearance[names[int(query)]].get(
                            "native_valid_keypoint_mask",
                            torch.ones(
                                torch.as_tensor(
                                    appearance[names[int(query)]]["native_keypoints"]
                                ).shape[0],
                                dtype=torch.bool,
                            ),
                        )
                    )[int(keypoint)]
                )
                for query, keypoint in zip(queries.tolist(), keypoints.tolist())
            ],
            dtype=torch.bool,
        )
        active = torch.nonzero(validity, as_tuple=False).reshape(-1)
        if active.numel() == 0:
            active = torch.arange(observations.numel())
        active_observations = observations[active]
        active_queries = queries[active]
        active_keypoints = keypoints[active]
        if active_observations.numel() < 3:
            continue
        descriptors = F.normalize(
            torch.stack(
                [
                    torch.as_tensor(
                        appearance[names[int(query)]]["native_descriptors"]
                    )[int(keypoint)]
                    for query, keypoint in zip(
                        active_queries.tolist(), active_keypoints.tolist()
                    )
                ]
            ).float(),
            dim=1,
        )
        base_reliability = torch.as_tensor(
            [
                float(
                    torch.as_tensor(
                        appearance[names[int(query)]].get(
                            "native_appearance_reliability",
                            torch.ones(
                                torch.as_tensor(
                                    appearance[names[int(query)]]["native_keypoints"]
                                ).shape[0]
                            ),
                        )
                    )[int(keypoint)]
                )
                for query, keypoint in zip(
                    active_queries.tolist(), active_keypoints.tolist()
                )
            ]
        ).clamp(0.0, 1.0)
        weights = observation_confidence[active_observations] * base_reliability
        sequences = [_sequence_name(names[int(query)]) for query in active_queries]
        pose_bins = query_bins[active_queries].tolist()
        joint_groups = list(zip(sequences, pose_bins))
        ordered_groups = sorted(set(joint_groups))
        prototypes = []
        for group in ordered_groups:
            group_mask = torch.as_tensor(
                [value == group for value in joint_groups], dtype=torch.bool
            )
            group_weight = weights[group_mask].clamp_min(1e-4)
            prototypes.append(
                F.normalize(
                    (descriptors[group_mask] * group_weight[:, None]).sum(dim=0),
                    dim=0,
                )
            )
        prototypes = torch.stack(prototypes)
        medoid = prototypes[(prototypes @ prototypes.T).mean(dim=1).argmax()]
        cosine = descriptors @ medoid
        artifact_reliability = torch.as_tensor(
            [
                float(
                    torch.as_tensor(
                        artifact[names[int(query)]]["native_artifact_reliability"]
                    )[int(keypoint)]
                )
                for query, keypoint in zip(
                    active_queries.tolist(), active_keypoints.tolist()
                )
            ]
        )
        view_counts = Counter(int(value) for value in pose_bins)
        sequence_counts = Counter(sequences)
        below_artifact_median = artifact_reliability < torch.median(
            artifact_reliability
        )
        below_descriptor_median = cosine < torch.median(cosine)
        strong = identity_certified[active_observations]
        non_unique_view = torch.as_tensor(
            [view_counts[int(value)] > 1 for value in pose_bins], dtype=torch.bool
        )
        non_unique_sequence = torch.as_tensor(
            [sequence_counts[value] > 1 for value in sequences], dtype=torch.bool
        )
        eligible = (
            below_artifact_median
            & below_descriptor_median
            & ~strong
            & non_unique_view
            & non_unique_sequence
        )
        initially_eligible_count += int(eligible.sum())
        protected_identity_count += int(
            (below_artifact_median & below_descriptor_median & strong).sum()
        )
        protected_unique_view_count += int(
            (below_artifact_median & below_descriptor_median & ~non_unique_view).sum()
        )
        protected_unique_sequence_count += int(
            (
                below_artifact_median & below_descriptor_median & ~non_unique_sequence
            ).sum()
        )
        for local, (query, keypoint) in enumerate(
            zip(active_queries.tolist(), active_keypoints.tolist())
        ):
            annotations[names[int(query)]][
                "native_conditional_descriptor_medoid_cosine"
            ][int(keypoint)] = cosine[local]
            annotations[names[int(query)]]["native_conditional_artifact_trim_eligible"][
                int(keypoint)
            ] = eligible[local]

        trim_limit = int(active_observations.numel() * float(maximum_trim_fraction))
        if trim_limit <= 0 or not bool(eligible.any()):
            continue
        severity = (1.0 - artifact_reliability) * ((1.0 - cosine) * 0.5)
        candidates = torch.nonzero(eligible, as_tuple=False).reshape(-1).tolist()
        candidates.sort(
            key=lambda local: (
                -float(severity[local]),
                int(active_observations[local]),
            )
        )
        removed = 0
        mutable_view_counts = Counter(int(value) for value in pose_bins)
        mutable_sequence_counts = Counter(sequences)
        for local in candidates:
            if removed >= trim_limit:
                break
            view = int(pose_bins[local])
            sequence = sequences[local]
            if mutable_view_counts[view] <= 1 or mutable_sequence_counts[sequence] <= 1:
                continue
            query = int(active_queries[local])
            keypoint = int(active_keypoints[local])
            annotations[names[query]]["native_descriptor_fusion_keep_mask"][
                keypoint
            ] = False
            mutable_view_counts[view] -= 1
            mutable_sequence_counts[sequence] -= 1
            removed += 1
        if removed:
            changed_track_count += 1
            trimmed_observation_count += removed
            maximum_observed_trim_fraction = max(
                maximum_observed_trim_fraction,
                removed / int(active_observations.numel()),
            )

    return annotations, {
        "selected_track_count": int(selected_tracks.numel()),
        "selected_observation_count": int(selected_observations.numel()),
        "changed_track_count": changed_track_count,
        "initially_eligible_observation_count": initially_eligible_count,
        "trimmed_observation_count": trimmed_observation_count,
        "trimmed_observation_fraction": (
            trimmed_observation_count / int(selected_observations.numel())
        ),
        "protected_identity_certified_count": protected_identity_count,
        "protected_unique_view_count": protected_unique_view_count,
        "protected_unique_sequence_count": protected_unique_sequence_count,
        "maximum_observed_track_trim_fraction": maximum_observed_trim_fraction,
        "maximum_allowed_track_trim_fraction": float(maximum_trim_fraction),
        "artifact_evidence_used_as_weight": False,
        "strong_identity_observations_never_trimmed": True,
        "each_pose_view_bin_retained": True,
        "each_mapping_sequence_retained": True,
    }
