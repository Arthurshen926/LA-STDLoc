"""Mapping-only descriptor-space appearance modes for Projective Anchors.

F0 already fuses mapping observations robustly into one descriptor.  This
module retains that descriptor as the immutable fallback and only adds a
small number of modes when the mapping observations contain a repeatable,
cross-sequence descriptor split.  Each mode is paired with the viewing
directions of the observations that support it, so an estimated first pose
selects exactly one descriptor per Anchor at refinement time.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from map_learning.v21_test_cache import tensor_sha256
from map_learning.v27_view_conditioned_anchor_descriptor import (
    _camera_center,
    _map_inputs,
    _mapping_family_ids,
    authorize_mapping_view_modes,
)


SCHEMA = "anygsloc_v32_mapping_descriptor_mode_anchor"
VERSION = 1


def _family_balanced_centroid(
    descriptors: torch.Tensor, families: torch.Tensor
) -> torch.Tensor:
    family_centroids = []
    for family in torch.unique(families, sorted=True):
        selected = families == family
        family_centroids.append(F.normalize(descriptors[selected].mean(0), dim=0))
    return F.normalize(torch.stack(family_centroids).mean(0), dim=0)


def _family_balanced_direction(
    directions: torch.Tensor, families: torch.Tensor
) -> torch.Tensor:
    family_directions = []
    for family in torch.unique(families, sorted=True):
        selected = families == family
        family_directions.append(F.normalize(directions[selected].mean(0), dim=0))
    return F.normalize(torch.stack(family_directions).mean(0), dim=0)


def _spherical_partition(
    descriptors: torch.Tensor,
    families: torch.Tensor,
    *,
    cluster_count: int,
    iterations: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic farthest-first spherical clustering."""

    observations = F.normalize(torch.as_tensor(descriptors).float(), dim=1)
    family_rows = torch.as_tensor(families).long().reshape(-1)
    count = int(observations.shape[0])
    requested = int(cluster_count)
    if not (
        observations.ndim == 2
        and count == family_rows.numel()
        and 1 <= requested <= count
        and int(iterations) >= 1
        and bool(torch.isfinite(observations).all())
    ):
        raise ValueError("V32 spherical partition inputs are invalid")

    global_center = _family_balanced_centroid(observations, family_rows)
    first = int((observations @ global_center).argmax())
    seed_rows = [first]
    while len(seed_rows) < requested:
        seed = observations[torch.as_tensor(seed_rows)]
        nearest = (observations @ seed.T).max(dim=1).values
        nearest[torch.as_tensor(seed_rows)] = torch.inf
        seed_rows.append(int(nearest.argmin()))
    centers = observations[torch.as_tensor(seed_rows)].clone()
    labels = torch.zeros(count, dtype=torch.long)
    for _ in range(int(iterations)):
        labels = (observations @ centers.T).argmax(dim=1)
        if torch.unique(labels).numel() != requested:
            break
        revised = []
        for cluster in range(requested):
            selected = labels == cluster
            revised.append(
                _family_balanced_centroid(
                    observations[selected], family_rows[selected]
                )
            )
        revised_tensor = torch.stack(revised)
        if torch.equal((observations @ revised_tensor.T).argmax(dim=1), labels):
            centers = revised_tensor
            break
        centers = revised_tensor
    labels = (observations @ centers.T).argmax(dim=1)
    return labels, centers


def _cluster_candidate(
    descriptors: torch.Tensor,
    directions: torch.Tensor,
    families: torch.Tensor,
    *,
    cluster_count: int,
    minimum_mode_observations: int,
    minimum_mapping_families: int,
    maximum_mode_cosine: float,
) -> dict | None:
    labels, centers = _spherical_partition(
        descriptors, families, cluster_count=cluster_count
    )
    count = int(cluster_count)
    cluster_rows = []
    for cluster in range(count):
        selected = labels == cluster
        observations = int(selected.sum())
        family_count = int(torch.unique(families[selected]).numel())
        if (
            observations < int(minimum_mode_observations)
            or family_count < int(minimum_mapping_families)
        ):
            return None
        cluster_rows.append(selected)
    pairwise = centers @ centers.T
    off_diagonal = ~torch.eye(count, dtype=torch.bool)
    if bool((pairwise[off_diagonal] > float(maximum_mode_cosine)).any()):
        return None

    prototypes = []
    view_directions = []
    radii = []
    concentrations = []
    observation_counts = []
    family_counts = []
    medoid_rows = []
    for selected, center in zip(cluster_rows, centers):
        local_descriptors = descriptors[selected]
        local_families = families[selected]
        local_directions = directions[selected]
        family_centroids = torch.stack(
            [
                F.normalize(local_descriptors[local_families == family].mean(0), dim=0)
                for family in torch.unique(local_families, sorted=True)
            ]
        )
        medoid_local = int((local_descriptors @ family_centroids.T).mean(1).argmax())
        global_rows = torch.nonzero(selected, as_tuple=False).reshape(-1)
        medoid_rows.append(int(global_rows[medoid_local]))
        prototypes.append(local_descriptors[medoid_local])
        view_direction = _family_balanced_direction(local_directions, local_families)
        view_directions.append(view_direction)
        angular = torch.rad2deg(
            torch.acos((local_directions @ view_direction).clamp(-1.0, 1.0))
        )
        radii.append(float(angular.max().item()) + 10.0)
        concentrations.append(float(family_centroids.mean(0).norm().item()))
        observation_counts.append(int(selected.sum()))
        family_counts.append(int(torch.unique(local_families).numel()))

    # Stable ordering makes the artifact bit-reproducible and keeps the most
    # broadly supported modes first.
    ordering = sorted(
        range(count), key=lambda index: (-observation_counts[index], medoid_rows[index])
    )
    return {
        "labels": labels,
        "centers": centers,
        "mode_features": torch.stack([prototypes[index] for index in ordering]),
        "mode_directions": torch.stack(
            [view_directions[index] for index in ordering]
        ),
        "mode_radius_deg": torch.tensor(
            [min(max(radii[index], 10.0), 120.0) for index in ordering]
        ),
        "mode_concentration": torch.tensor(
            [concentrations[index] for index in ordering]
        ),
        "mode_observation_count": torch.tensor(
            [observation_counts[index] for index in ordering], dtype=torch.long
        ),
        "mode_mapping_family_count": torch.tensor(
            [family_counts[index] for index in ordering], dtype=torch.long
        ),
    }


@torch.inference_mode()
def build_mapping_descriptor_modes(
    *,
    map_state: Mapping,
    observation_cache: Mapping,
    maximum_modes_per_anchor: int = 3,
    minimum_mode_observations: int = 3,
    minimum_mapping_families: int = 2,
    minimum_distortion_improvement: float = 0.02,
    maximum_mode_cosine: float = 0.95,
) -> dict:
    """Build cross-family appearance modes without consuming test queries."""

    (
        ids,
        xyz,
        base_features,
        offsets,
        observation_query_rows,
        observation_keypoint_rows,
        _,
        _,
    ) = _map_inputs(map_state)
    names = list(map_state["v6_mapping_query_names"])
    queries = observation_cache.get("queries")
    maximum_modes = int(maximum_modes_per_anchor)
    if not (
        observation_cache.get("schema") == "render_observation_cache_v2"
        and observation_cache.get("uses_source_mapping_rgb") is False
        and observation_cache.get("uses_test_queries") is False
        and isinstance(queries, Mapping)
        and list(queries) == names
        and 2 <= maximum_modes <= 4
        and int(minimum_mode_observations) >= 2
        and int(minimum_mapping_families) >= 2
        and 0.0 < float(minimum_distortion_improvement) < 0.5
        and 0.0 <= float(maximum_mode_cosine) < 1.0
    ):
        raise ValueError("V32 requires the exact mapping-only observation cache")

    anchor_count, dimension = base_features.shape
    observation_count = int(observation_query_rows.numel())
    anchor_rows = torch.repeat_interleave(
        torch.arange(anchor_count), offsets[1:] - offsets[:-1]
    )
    order = torch.argsort(observation_query_rows, stable=True)
    sorted_query = observation_query_rows[order]
    sorted_anchor = anchor_rows[order]
    sorted_keypoint = observation_keypoint_rows[order]
    descriptors = torch.empty((observation_count, dimension), dtype=torch.float32)
    directions = torch.empty((observation_count, 3), dtype=torch.float32)
    family_ids = _mapping_family_ids(names)
    observation_families = torch.empty(observation_count, dtype=torch.long)

    cursor = 0
    for query_index, name in enumerate(names):
        end = int(
            torch.searchsorted(
                sorted_query, torch.tensor(query_index), right=True
            ).item()
        )
        if end == cursor:
            continue
        record = queries[name]
        source = torch.as_tensor(record.get("native_descriptors")).float().cpu()
        local_anchor = sorted_anchor[cursor:end]
        local_keypoint = sorted_keypoint[cursor:end]
        if not (
            source.ndim == 2
            and source.shape[1] == dimension
            and int(local_keypoint.max()) < source.shape[0]
            and bool(torch.isfinite(source).all())
        ):
            raise ValueError(f"V32 mapping descriptor rows are invalid for {name}")
        positions = order[cursor:end]
        descriptors[positions] = F.normalize(source[local_keypoint], dim=1)
        center = _camera_center(torch.as_tensor(record.get("pose_w2c")))
        directions[positions] = F.normalize(
            center[None] - xyz[local_anchor], dim=1
        )
        observation_families[positions] = family_ids[query_index]
        cursor = end
    if cursor != observation_count:
        raise RuntimeError("V32 mapping observation traversal was incomplete")

    normalized_base = F.normalize(base_features, dim=1)
    mode_features = normalized_base[:, None, :].expand(-1, maximum_modes, -1).clone()
    mode_directions = torch.zeros((anchor_count, maximum_modes, 3))
    mode_directions[..., 2] = 1.0
    mode_radius = torch.full((anchor_count, maximum_modes), 120.0)
    mode_concentration = torch.zeros((anchor_count, maximum_modes))
    mode_observations = torch.zeros((anchor_count, maximum_modes), dtype=torch.long)
    mode_families = torch.zeros_like(mode_observations)
    mode_valid = torch.zeros((anchor_count, maximum_modes), dtype=torch.bool)
    selected_mode_count = torch.zeros(anchor_count, dtype=torch.long)
    single_distortion = torch.zeros(anchor_count)
    selected_distortion = torch.zeros(anchor_count)

    for anchor in range(anchor_count):
        start, stop = int(offsets[anchor]), int(offsets[anchor + 1])
        local_descriptors = descriptors[start:stop]
        local_directions = directions[start:stop]
        local_families = observation_families[start:stop]
        if (
            local_descriptors.shape[0] < 2 * int(minimum_mode_observations)
            or torch.unique(local_families).numel()
            < 2 * int(minimum_mapping_families)
        ):
            continue
        one_center = _family_balanced_centroid(local_descriptors, local_families)
        baseline_distortion = float(
            (1.0 - local_descriptors @ one_center).mean().item()
        )
        single_distortion[anchor] = baseline_distortion
        accepted = None
        accepted_distortion = baseline_distortion
        previous_distortion = baseline_distortion
        for candidate_count in range(2, maximum_modes + 1):
            candidate = _cluster_candidate(
                local_descriptors,
                local_directions,
                local_families,
                cluster_count=candidate_count,
                minimum_mode_observations=minimum_mode_observations,
                minimum_mapping_families=minimum_mapping_families,
                maximum_mode_cosine=maximum_mode_cosine,
            )
            if candidate is None:
                break
            distortion = float(
                (
                    1.0
                    - (local_descriptors @ candidate["centers"].T)
                    .max(dim=1)
                    .values
                )
                .mean()
                .item()
            )
            if previous_distortion - distortion < float(
                minimum_distortion_improvement
            ):
                break
            accepted = candidate
            accepted_distortion = distortion
            previous_distortion = distortion
        if accepted is None:
            continue
        count = int(accepted["mode_features"].shape[0])
        mode_features[anchor, :count] = accepted["mode_features"]
        mode_directions[anchor, :count] = accepted["mode_directions"]
        mode_radius[anchor, :count] = accepted["mode_radius_deg"]
        mode_concentration[anchor, :count] = accepted["mode_concentration"]
        mode_observations[anchor, :count] = accepted["mode_observation_count"]
        mode_families[anchor, :count] = accepted["mode_mapping_family_count"]
        mode_valid[anchor, :count] = True
        selected_mode_count[anchor] = count
        selected_distortion[anchor] = accepted_distortion

    return {
        "anchor_ids": ids,
        "mode_features": mode_features.contiguous(),
        "mode_direction_vectors": F.normalize(mode_directions, dim=2).contiguous(),
        "mode_direction_radius_deg": mode_radius.contiguous(),
        "mode_concentration": mode_concentration.contiguous(),
        "mode_observation_count": mode_observations.contiguous(),
        "mode_mapping_family_count": mode_families.contiguous(),
        "mode_valid": mode_valid.contiguous(),
        "selected_mode_count": selected_mode_count.contiguous(),
        "single_mode_distortion": single_distortion.contiguous(),
        "selected_mode_distortion": selected_distortion.contiguous(),
        "base_anchor_features_sha256": tensor_sha256(base_features),
        "mapping_observation_registry_sha256": tensor_sha256(
            torch.stack((observation_query_rows, observation_keypoint_rows), dim=1)
        ),
    }


def make_artifact(
    *,
    map_path: str | Path,
    observation_cache_path: str | Path,
    map_state: Mapping,
    observation_cache: Mapping,
    maximum_modes_per_anchor: int = 3,
    minimum_mode_observations: int = 3,
    minimum_mapping_families: int = 2,
    minimum_distortion_improvement: float = 0.02,
    maximum_mode_cosine: float = 0.95,
    minimum_owner_margin: float = 0.0,
    authorization_device: str | torch.device = "cpu",
    authorization_chunk_size: int = 256,
) -> dict:
    built = build_mapping_descriptor_modes(
        map_state=map_state,
        observation_cache=observation_cache,
        maximum_modes_per_anchor=maximum_modes_per_anchor,
        minimum_mode_observations=minimum_mode_observations,
        minimum_mapping_families=minimum_mapping_families,
        minimum_distortion_improvement=minimum_distortion_improvement,
        maximum_mode_cosine=maximum_mode_cosine,
    )
    authorization = authorize_mapping_view_modes(
        mode_features=built["mode_features"],
        mode_valid=built["mode_valid"],
        base_anchor_features=torch.as_tensor(map_state["anchor_features"]),
        minimum_owner_margin=minimum_owner_margin,
        device=authorization_device,
        chunk_size=authorization_chunk_size,
    )
    map_resolved = Path(map_path).resolve()
    cache_resolved = Path(observation_cache_path).resolve()
    payload = {
        "schema": SCHEMA,
        "version": VERSION,
        "protocol": "mapping_only_descriptor_space_modes_with_pose_selection",
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "uses_test_poses": False,
        "map_mutated": False,
        "adds_anchor_owners": False,
        "maximum_modes_per_anchor": int(maximum_modes_per_anchor),
        "minimum_mode_observations": int(minimum_mode_observations),
        "minimum_mapping_families": int(minimum_mapping_families),
        "minimum_distortion_improvement": float(minimum_distortion_improvement),
        "maximum_mode_cosine": float(maximum_mode_cosine),
        "minimum_owner_margin": float(minimum_owner_margin),
        "aggregation": "descriptor_space_spherical_cluster_family_balanced_medoid",
        "authorization": "exact_global_base_owner_margin",
        "selection_authority": "first_pass_estimated_pose_only",
        "inputs": {
            "stable_map": {
                "path": str(map_resolved),
                "sha256": sha256_file(map_resolved),
            },
            "mapping_observation_cache": {
                "path": str(cache_resolved),
                "sha256": sha256_file(cache_resolved),
            },
        },
        **built,
        **authorization,
    }
    validate_artifact(payload, map_state=map_state)
    return payload


def validate_artifact(payload: Mapping, *, map_state: Mapping | None = None) -> None:
    features = torch.as_tensor(payload.get("mode_features"))
    directions = torch.as_tensor(payload.get("mode_direction_vectors"))
    radii = torch.as_tensor(payload.get("mode_direction_radius_deg"))
    concentration = torch.as_tensor(payload.get("mode_concentration"))
    observations = torch.as_tensor(payload.get("mode_observation_count"))
    families = torch.as_tensor(payload.get("mode_mapping_family_count"))
    valid = torch.as_tensor(payload.get("mode_valid"))
    authorized = torch.as_tensor(payload.get("mode_authorized"))
    mode_count = torch.as_tensor(payload.get("selected_mode_count"))
    ids = torch.as_tensor(payload.get("anchor_ids")).long()
    margin = torch.as_tensor(payload.get("mode_owner_margin"))
    inputs = payload.get("inputs")
    maximum_modes = int(payload.get("maximum_modes_per_anchor", 0))
    okay = bool(
        payload.get("schema") == SCHEMA
        and int(payload.get("version", 0)) == VERSION
        and payload.get("protocol")
        == "mapping_only_descriptor_space_modes_with_pose_selection"
        and payload.get("uses_source_mapping_rgb") is False
        and payload.get("uses_test_queries") is False
        and payload.get("uses_test_poses") is False
        and payload.get("map_mutated") is False
        and payload.get("adds_anchor_owners") is False
        and 2 <= maximum_modes <= 4
        and int(payload.get("minimum_mode_observations", 0)) >= 2
        and int(payload.get("minimum_mapping_families", 0)) >= 2
        and 0.0 < float(payload.get("minimum_distortion_improvement", 0.0)) < 0.5
        and 0.0 <= float(payload.get("maximum_mode_cosine", -1.0)) < 1.0
        and float(payload.get("minimum_owner_margin", -1.0)) >= 0.0
        and payload.get("aggregation")
        == "descriptor_space_spherical_cluster_family_balanced_medoid"
        and payload.get("authorization") == "exact_global_base_owner_margin"
        and payload.get("selection_authority") == "first_pass_estimated_pose_only"
        and isinstance(inputs, Mapping)
        and isinstance(inputs.get("stable_map"), Mapping)
        and isinstance(inputs.get("mapping_observation_cache"), Mapping)
        and features.ndim == 3
        and features.shape[1] == maximum_modes
        and directions.shape == (*features.shape[:2], 3)
        and radii.shape
        == concentration.shape
        == observations.shape
        == families.shape
        == valid.shape
        == authorized.shape
        == margin.shape
        == features.shape[:2]
        and mode_count.shape == ids.shape == (features.shape[0],)
        and ids.unique().numel() == ids.numel()
        and valid.dtype == authorized.dtype == torch.bool
        and bool((mode_count == valid.sum(1)).all())
        and bool(((mode_count == 0) | (mode_count >= 2)).all())
        and bool((mode_count <= maximum_modes).all())
        and bool((~authorized | valid).all())
        and bool(
            (~authorized | (margin >= float(payload["minimum_owner_margin"]))).all()
        )
        and bool(
            (~valid | (observations >= int(payload["minimum_mode_observations"]))).all()
        )
        and bool(
            (~valid | (families >= int(payload["minimum_mapping_families"]))).all()
        )
        and bool(torch.isfinite(features).all())
        and bool(torch.isfinite(directions).all())
        and bool(torch.isfinite(radii).all())
        and bool(torch.isfinite(concentration).all())
        and bool(torch.isfinite(margin[valid]).all())
        and bool((radii > 0).all())
        and bool((concentration >= 0).all())
        and bool((concentration <= 1.00001).all())
        and bool(
            torch.allclose(
                features.norm(dim=2), torch.ones_like(radii), atol=2e-5
            )
        )
        and bool(
            torch.allclose(
                directions.norm(dim=2), torch.ones_like(radii), atol=2e-5
            )
        )
    )
    if not okay:
        raise ValueError("V32 descriptor-mode artifact is invalid")
    if map_state is not None:
        map_ids, _, map_features, _, query_rows, keypoint_rows, _, _ = _map_inputs(
            map_state
        )
        registry = tensor_sha256(torch.stack((query_rows, keypoint_rows), dim=1))
        if not (
            torch.equal(ids.cpu(), map_ids)
            and features.shape[2] == map_features.shape[1]
            and payload.get("base_anchor_features_sha256")
            == tensor_sha256(map_features)
            and payload.get("mapping_observation_registry_sha256") == registry
        ):
            raise ValueError("V32 artifact does not bind to this F0 map")
