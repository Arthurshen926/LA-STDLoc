"""Mapping-only view-conditioned descriptors for Projective Anchors.

Each Anchor retains its native descriptor.  Mapping observations are assigned
to the (at most two) viewing-direction modes already stored in the F0 map and
are summarized by a spherical descriptor mean.  At localization time a first
pose selects at most one mode per Anchor; it never adds another 3D owner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from map_learning.v21_test_cache import tensor_sha256


SCHEMA = "lafgs_v27_mapping_view_conditioned_anchor_descriptors"
VERSION = 1


def _camera_center(pose_w2c: torch.Tensor) -> torch.Tensor:
    pose = torch.as_tensor(pose_w2c).float()
    if pose.shape != (4, 4) or not bool(torch.isfinite(pose).all()):
        raise ValueError("mapping pose must be a finite [4,4] tensor")
    return -(pose[:3, :3].T @ pose[:3, 3])


def _map_inputs(state: Mapping) -> tuple[torch.Tensor, ...]:
    ids = torch.as_tensor(state.get("anchor_ids")).long().cpu()
    xyz = torch.as_tensor(state.get("anchor_xyz")).float().cpu()
    features = torch.as_tensor(state.get("anchor_features")).float().cpu()
    observations = state.get("projective_anchor_observations")
    support = state.get("anchor_view_support")
    names = state.get("v6_mapping_query_names")
    if not isinstance(observations, Mapping) or not isinstance(support, Mapping):
        raise ValueError("F0 map lacks observation or viewing-mode provenance")
    offsets = torch.as_tensor(observations.get("observation_offsets")).long().cpu()
    query_rows = torch.as_tensor(observations.get("query_indices")).long().cpu()
    keypoint_rows = torch.as_tensor(observations.get("keypoint_indices")).long().cpu()
    modes = torch.as_tensor(support.get("direction_modes")).float().cpu()
    mode_count = torch.as_tensor(support.get("mode_count")).long().cpu()
    count = ids.numel()
    if not (
        state.get("schema") == "lafgs_materialized_anchor_map"
        and state.get("provenance", {}).get("uses_test_queries") is False
        and ids.shape == (count,)
        and ids.unique().numel() == count
        and xyz.shape == (count, 3)
        and features.ndim == 2
        and features.shape[0] == count
        and offsets.shape == (count + 1,)
        and int(offsets[0]) == 0
        and int(offsets[-1]) == query_rows.numel() == keypoint_rows.numel()
        and bool((offsets[1:] >= offsets[:-1]).all())
        and support.get("schema") == "lafgs_v24_anchor_view_support"
        and support.get("uses_test_queries") is False
        and modes.shape == (count, 2, 3)
        and mode_count.shape == (count,)
        and bool(((mode_count == 1) | (mode_count == 2)).all())
        and isinstance(names, Sequence)
        and not isinstance(names, (str, bytes))
        and len(names) > 0
        and (not query_rows.numel() or int(query_rows.min()) >= 0)
        and (not query_rows.numel() or int(query_rows.max()) < len(names))
        and (not keypoint_rows.numel() or int(keypoint_rows.min()) >= 0)
        and bool(torch.isfinite(xyz).all())
        and bool(torch.isfinite(features).all())
        and bool(torch.isfinite(modes).all())
    ):
        raise ValueError("F0 map is invalid for V27 mapping-only materialization")
    return ids, xyz, features, offsets, query_rows, keypoint_rows, modes, mode_count


@torch.inference_mode()
def build_mapping_view_conditioned_descriptors(
    *,
    map_state: Mapping,
    observation_cache: Mapping,
    minimum_mode_observations: int = 2,
) -> dict:
    """Aggregate mapping descriptors into the F0 viewing-direction modes."""

    (
        ids,
        xyz,
        base_features,
        offsets,
        observation_query_rows,
        observation_keypoint_rows,
        direction_modes,
        map_mode_count,
    ) = _map_inputs(map_state)
    queries = observation_cache.get("queries")
    names = list(map_state["v6_mapping_query_names"])
    if not (
        observation_cache.get("schema") == "render_observation_cache_v2"
        and observation_cache.get("uses_source_mapping_rgb") is False
        and observation_cache.get("uses_test_queries") is False
        and isinstance(queries, Mapping)
        and list(queries) == names
        and int(minimum_mode_observations) >= 2
    ):
        raise ValueError("V27 requires the exact mapping-only observation cache")

    count, dimension = base_features.shape
    anchor_rows = torch.repeat_interleave(
        torch.arange(count), offsets[1:] - offsets[:-1]
    )
    order = torch.argsort(observation_query_rows, stable=True)
    sorted_query = observation_query_rows[order]
    sorted_anchor = anchor_rows[order]
    sorted_keypoint = observation_keypoint_rows[order]
    sums = torch.zeros((count * 2, dimension), dtype=torch.float32)
    mode_observations = torch.zeros(count * 2, dtype=torch.long)

    cursor = 0
    for query_index, name in enumerate(names):
        end = int(
            torch.searchsorted(
                sorted_query, torch.tensor(query_index), right=True
            ).item()
        )
        record = queries[name]
        if end == cursor:
            # A registered mapping view may contribute no Anchor surviving the
            # final F0 filters.  It remains part of the ordered provenance but
            # has no descriptor rows to aggregate.
            continue
        descriptors = torch.as_tensor(record.get("native_descriptors")).float().cpu()
        pose = torch.as_tensor(record.get("pose_w2c")).float().cpu()
        local_anchor = sorted_anchor[cursor:end]
        local_keypoint = sorted_keypoint[cursor:end]
        if not (
            descriptors.ndim == 2
            and descriptors.shape[1] == dimension
            and (not local_keypoint.numel() or int(local_keypoint.max()) < descriptors.shape[0])
            and bool(torch.isfinite(descriptors).all())
        ):
            raise ValueError(f"mapping descriptor rows are invalid for {name}")
        camera_center = _camera_center(pose)
        rays = F.normalize(camera_center[None] - xyz[local_anchor], dim=1)
        similarity = torch.einsum(
            "nd,nmd->nm", rays, direction_modes[local_anchor]
        )
        labels = similarity.argmax(dim=1)
        labels = torch.where(map_mode_count[local_anchor] == 1, 0, labels)
        flat_rows = local_anchor * 2 + labels
        normalized = F.normalize(descriptors[local_keypoint], dim=1)
        sums.index_add_(0, flat_rows, normalized)
        mode_observations.index_add_(
            0, flat_rows, torch.ones_like(flat_rows, dtype=torch.long)
        )
        cursor = end
    if cursor != order.numel():
        raise RuntimeError("V27 mapping observation traversal was incomplete")

    resultant_norm = sums.norm(dim=1)
    centroid = F.normalize(sums, dim=1).reshape(count, 2, dimension)
    counts = mode_observations.reshape(count, 2)
    concentration = (
        resultant_norm / mode_observations.clamp_min(1)
    ).reshape(count, 2)
    valid = counts >= int(minimum_mode_observations)
    valid &= torch.arange(2)[None] < map_mode_count[:, None]
    centroid = torch.where(
        valid[..., None], centroid, F.normalize(base_features, dim=1)[:, None, :]
    )
    return {
        "mode_features": centroid.contiguous(),
        "mode_observation_count": counts.contiguous(),
        "mode_concentration": concentration.contiguous(),
        "mode_valid": valid.contiguous(),
        "anchor_ids": ids,
        "direction_modes_sha256": tensor_sha256(direction_modes),
        "base_anchor_features_sha256": tensor_sha256(base_features),
    }


def make_artifact(
    *,
    map_path: str | Path,
    observation_cache_path: str | Path,
    map_state: Mapping,
    observation_cache: Mapping,
    minimum_mode_observations: int = 2,
) -> dict:
    map_resolved = Path(map_path).resolve()
    cache_resolved = Path(observation_cache_path).resolve()
    built = build_mapping_view_conditioned_descriptors(
        map_state=map_state,
        observation_cache=observation_cache,
        minimum_mode_observations=minimum_mode_observations,
    )
    payload = {
        "schema": SCHEMA,
        "version": VERSION,
        "protocol": "mapping_only_view_conditioned_anchor_descriptor",
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "uses_test_poses": False,
        "map_mutated": False,
        "adds_anchor_owners": False,
        "maximum_modes_per_anchor": 2,
        "minimum_mode_observations": int(minimum_mode_observations),
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
    }
    validate_artifact(payload, map_state=map_state)
    return payload


def validate_artifact(payload: Mapping, *, map_state: Mapping | None = None) -> None:
    features = torch.as_tensor(payload.get("mode_features"))
    counts = torch.as_tensor(payload.get("mode_observation_count"))
    concentration = torch.as_tensor(payload.get("mode_concentration"))
    valid = torch.as_tensor(payload.get("mode_valid"))
    ids = torch.as_tensor(payload.get("anchor_ids")).long()
    inputs = payload.get("inputs")
    structurally_valid = bool(
        payload.get("schema") == SCHEMA
        and payload.get("version") == VERSION
        and payload.get("protocol") == "mapping_only_view_conditioned_anchor_descriptor"
        and payload.get("uses_source_mapping_rgb") is False
        and payload.get("uses_test_queries") is False
        and payload.get("uses_test_poses") is False
        and payload.get("map_mutated") is False
        and payload.get("adds_anchor_owners") is False
        and payload.get("maximum_modes_per_anchor") == 2
        and int(payload.get("minimum_mode_observations", 0)) >= 2
        and payload.get("selection_authority") == "first_pass_estimated_pose_only"
        and isinstance(inputs, Mapping)
        and isinstance(inputs.get("stable_map"), Mapping)
        and isinstance(inputs.get("mapping_observation_cache"), Mapping)
        and features.ndim == 3
        and features.shape[1] == 2
        and counts.shape == concentration.shape == valid.shape == features.shape[:2]
        and ids.shape == (features.shape[0],)
        and ids.unique().numel() == ids.numel()
        and valid.dtype == torch.bool
        and bool(torch.isfinite(features).all())
        and bool(torch.isfinite(concentration).all())
        and bool((counts >= 0).all())
        and bool((concentration >= 0).all())
        and bool((concentration <= 1.00001).all())
        and bool((~valid | (counts >= int(payload["minimum_mode_observations"]))).all())
        and bool(torch.allclose(features.norm(dim=2), torch.ones_like(concentration), atol=2e-5))
    )
    if not structurally_valid:
        raise ValueError("V27 view-conditioned descriptor artifact is invalid")
    if map_state is not None:
        map_ids, _, map_features, _, _, _, modes, _ = _map_inputs(map_state)
        if not (
            torch.equal(ids.cpu(), map_ids)
            and features.shape[2] == map_features.shape[1]
            and payload.get("base_anchor_features_sha256") == tensor_sha256(map_features)
            and payload.get("direction_modes_sha256") == tensor_sha256(modes)
        ):
            raise ValueError("V27 artifact does not bind to this F0 map")


@torch.inference_mode()
def select_view_conditioned_anchor_features(
    *,
    base_anchor_features: torch.Tensor,
    anchor_xyz: torch.Tensor,
    direction_modes: torch.Tensor,
    baseline_pose_w2c: torch.Tensor,
    mode_features: torch.Tensor,
    mode_valid: torch.Tensor,
    minimum_concentration: float = 0.0,
    mode_concentration: torch.Tensor | None = None,
    anchor_rows: torch.Tensor | None = None,
    residual_scale: float = 1.0,
    require_two_valid_modes: bool = False,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Select one descriptor per Anchor using only the estimated first pose."""

    base = F.normalize(torch.as_tensor(base_anchor_features).float(), dim=1)
    xyz = torch.as_tensor(anchor_xyz, device=base.device).float()
    directions = torch.as_tensor(direction_modes, device=base.device).float()
    modes = torch.as_tensor(mode_features, device=base.device).float()
    valid = torch.as_tensor(mode_valid, device=base.device).bool()
    pose = torch.as_tensor(baseline_pose_w2c, device=base.device).float()
    concentration = (
        torch.ones_like(valid, dtype=torch.float32)
        if mode_concentration is None
        else torch.as_tensor(mode_concentration, device=base.device).float()
    )
    if not (
        xyz.shape == (base.shape[0], 3)
        and directions.shape == (base.shape[0], 2, 3)
        and modes.shape == (base.shape[0], 2, base.shape[1])
        and valid.shape == concentration.shape == (base.shape[0], 2)
        and pose.shape == (4, 4)
        and 0.0 <= float(minimum_concentration) <= 1.0
        and 0.0 < float(residual_scale) <= 1.0
        and bool(torch.isfinite(pose).all())
    ):
        raise ValueError("V27 runtime selection inputs are invalid")
    rows = (
        torch.arange(base.shape[0], device=base.device)
        if anchor_rows is None
        else torch.as_tensor(anchor_rows, device=base.device).long().reshape(-1)
    )
    if rows.unique().numel() != rows.numel() or (
        rows.numel()
        and (int(rows.min()) < 0 or int(rows.max()) >= base.shape[0])
    ):
        raise ValueError("V27 selected Anchor rows are invalid")
    center = -(pose[:3, :3].T @ pose[:3, 3])
    rays = F.normalize(center[None] - xyz[rows], dim=1)
    labels = torch.einsum("nd,nmd->nm", rays, directions[rows]).argmax(dim=1)
    use = valid[rows, labels] & (concentration[rows, labels] >= float(minimum_concentration))
    if bool(require_two_valid_modes):
        use &= valid[rows].all(dim=1)
    selected = base[rows].clone()
    if bool(use.any()):
        chosen = F.normalize(modes[rows[use], labels[use]], dim=1)
        selected[use] = F.normalize(
            selected[use] + float(residual_scale) * (chosen - selected[use]),
            dim=1,
        )
    return selected, {
        "selected_mode_anchor_count": int(use.sum().item()),
        "base_fallback_anchor_count": int((~use).sum().item()),
    }
