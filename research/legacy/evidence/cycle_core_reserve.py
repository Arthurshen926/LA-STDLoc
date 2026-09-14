"""Coverage-aware deployment selection for weak-prior projective Tracks."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from common.v6_contracts import ANCHOR_CANDIDATE_SCHEMA, ASSOCIATION_GRAPH_SCHEMA


def _subset_candidate(candidate: dict, selected: torch.Tensor) -> dict:
    count = int(torch.as_tensor(candidate["anchor_xyz"]).shape[0])
    selected = torch.as_tensor(selected, dtype=torch.long).reshape(-1)
    if selected.numel() == 0 or bool((selected < 0).any()) or int(selected.max()) >= count:
        raise ValueError("cycle-core selection must retain valid Anchor rows")
    if selected.numel() > 1 and bool((selected[1:] <= selected[:-1]).any()):
        raise ValueError("cycle-core selected rows must be strictly increasing")

    output = dict(candidate)
    output["anchor_ids"] = torch.arange(selected.numel(), dtype=torch.long)
    for key, value in candidate.items():
        if key in {"anchor_ids", "query_bins", "projective_anchor_observations", "geometry_diagnostics"}:
            continue
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == count:
            output[key] = value[selected].clone()

    diagnostics = candidate.get("geometry_diagnostics")
    if isinstance(diagnostics, dict):
        output["geometry_diagnostics"] = {
            key: (
                value[selected].clone()
                if isinstance(value, torch.Tensor)
                and value.ndim >= 1
                and value.shape[0] == count
                else value
            )
            for key, value in diagnostics.items()
        }

    csr = candidate["projective_anchor_observations"]
    offsets = torch.as_tensor(csr["observation_offsets"]).long()
    query = torch.as_tensor(csr["query_indices"]).long()
    keypoint = torch.as_tensor(csr["keypoint_indices"]).long()
    if offsets.shape != (count + 1,) or int(offsets[0]) != 0:
        raise ValueError("projective candidate CSR offsets are invalid")
    if int(offsets[-1]) != query.numel() or keypoint.shape != query.shape:
        raise ValueError("projective candidate CSR columns do not align")
    output_offsets = [0]
    output_query = []
    output_keypoint = []
    for anchor in selected.tolist():
        start, stop = int(offsets[anchor]), int(offsets[anchor + 1])
        output_query.append(query[start:stop])
        output_keypoint.append(keypoint[start:stop])
        output_offsets.append(output_offsets[-1] + stop - start)
    output["projective_anchor_observations"] = {
        "observation_offsets": torch.tensor(output_offsets, dtype=torch.long),
        "query_indices": torch.cat(output_query),
        "keypoint_indices": torch.cat(output_keypoint),
    }
    return output


def select_cycle_core_coverage_reserve(
    candidate: dict,
    association: dict,
    *,
    keypoints_by_query: Sequence[torch.Tensor],
    image_hw_by_query: Sequence[tuple[int, int]],
    spatial_voxel_size_m: float = 0.50,
    image_grid_shape: tuple[int, int] = (6, 8),
) -> tuple[dict, dict]:
    """Keep all cycle Tracks and only coverage-contributing chain Tracks.

    The selector is entirely mapping-only.  A chain Track is admitted to the
    reserve when, in descending mapping reliability order, it contributes a
    spatial voxel or a mapping-image cell not already covered by the cycle
    core or an earlier, stronger reserve Track.
    """

    if candidate.get("schema") != ANCHOR_CANDIDATE_SCHEMA:
        raise ValueError("cycle-core selection requires projective candidates")
    if association.get("schema") != ASSOCIATION_GRAPH_SCHEMA:
        raise ValueError("cycle-core selection requires the bound association graph")
    if candidate.get("uses_source_mapping_rgb") is not False or candidate.get(
        "uses_test_queries"
    ) is not False:
        raise ValueError("cycle-core selection must remain mapping-only")
    if association.get("uses_source_mapping_rgb") is not False or association.get(
        "uses_test_queries"
    ) is not False:
        raise ValueError("association graph is outside mapping-only scope")
    voxel_size = float(spatial_voxel_size_m)
    grid_y, grid_x = int(image_grid_shape[0]), int(image_grid_shape[1])
    if voxel_size <= 0 or grid_y < 1 or grid_x < 1:
        raise ValueError("cycle-core coverage parameters are invalid")

    xyz = torch.as_tensor(candidate["anchor_xyz"]).float().cpu()
    source_component = torch.as_tensor(candidate.get("source_component_ids")).long()
    count = int(xyz.shape[0])
    cycle = torch.as_tensor(
        association.get("component_statistics", {}).get("cycle_supported")
    ).bool()
    if (
        xyz.shape != (count, 3)
        or source_component.shape != (count,)
        or cycle.ndim != 1
        or bool((source_component < 0).any())
        or (source_component.numel() and int(source_component.max()) >= cycle.numel())
    ):
        raise ValueError("candidate/component cycle registry does not align")
    core = cycle[source_component]
    if not bool(core.any()):
        raise ValueError("cycle-core selection found no cycle-supported Anchor")

    names = list(candidate.get("query_names", ()))
    if len(keypoints_by_query) != len(names) or len(image_hw_by_query) != len(names):
        raise ValueError("mapping observation registry does not align")
    csr = candidate["projective_anchor_observations"]
    offsets = torch.as_tensor(csr["observation_offsets"]).long()
    query = torch.as_tensor(csr["query_indices"]).long()
    keypoint = torch.as_tensor(csr["keypoint_indices"]).long()
    if offsets.shape != (count + 1,) or int(offsets[-1]) != query.numel():
        raise ValueError("projective candidate CSR does not align")
    if query.numel() and (int(query.min()) < 0 or int(query.max()) >= len(names)):
        raise ValueError("candidate query index is outside the mapping registry")

    cell_by_query: list[torch.Tensor] = []
    for query_index, (points_value, hw_value) in enumerate(
        zip(keypoints_by_query, image_hw_by_query)
    ):
        points = torch.as_tensor(points_value).float().reshape(-1, 2).cpu()
        height, width = int(hw_value[0]), int(hw_value[1])
        if height < 1 or width < 1 or not bool(torch.isfinite(points).all()):
            raise ValueError(f"mapping view {query_index} has invalid image geometry")
        cell_x = (points[:, 0] * grid_x / width).long().clamp(0, grid_x - 1)
        cell_y = (points[:, 1] * grid_y / height).long().clamp(0, grid_y - 1)
        cell_by_query.append(cell_y * grid_x + cell_x)

    keypoint_count = torch.tensor(
        [value.numel() for value in cell_by_query], dtype=torch.long
    )
    if keypoint.numel() and (
        int(keypoint.min()) < 0
        or bool((keypoint >= keypoint_count[query]).any())
    ):
        raise ValueError("candidate keypoint index is outside its mapping view")
    keypoint_offsets = torch.cat(
        (torch.zeros(1, dtype=torch.long), keypoint_count.cumsum(0))
    )
    packed_cells = torch.cat(cell_by_query)
    observation_cell = (
        query * (grid_y * grid_x)
        + packed_cells[keypoint_offsets[query] + keypoint]
    )

    voxels = torch.floor(xyz / voxel_size).long()
    core_rows = torch.nonzero(core, as_tuple=False).reshape(-1)
    covered_voxels = {tuple(value) for value in voxels[core_rows].tolist()}
    covered_cells: set[int] = set()
    for anchor in core_rows.tolist():
        covered_cells.update(
            observation_cell[int(offsets[anchor]) : int(offsets[anchor + 1])].tolist()
        )

    identity = torch.as_tensor(candidate["identity_reliability"]).float().cpu()
    geometry = torch.as_tensor(candidate["geometry_reliability"]).float().cpu()
    if identity.shape != geometry.shape or identity.shape != (count,):
        raise ValueError("candidate mapping reliability does not align")
    quality = identity.clamp(0, 1) * geometry.clamp(0, 1)
    reserve_rows = torch.nonzero(~core, as_tuple=False).reshape(-1)
    reserve_rows = reserve_rows[
        torch.argsort(quality[reserve_rows], descending=True, stable=True)
    ]
    keep = core.clone()
    spatial_added = 0
    image_cell_added = 0
    for anchor in reserve_rows.tolist():
        voxel = tuple(voxels[anchor].tolist())
        cells = set(
            observation_cell[
                int(offsets[anchor]) : int(offsets[anchor + 1])
            ].tolist()
        )
        novel_spatial = voxel not in covered_voxels
        novel_cells = cells.difference(covered_cells)
        if not novel_spatial and not novel_cells:
            continue
        keep[anchor] = True
        if novel_spatial:
            spatial_added += 1
        if novel_cells:
            image_cell_added += 1
        covered_voxels.add(voxel)
        covered_cells.update(cells)

    selected = torch.nonzero(keep, as_tuple=False).reshape(-1)
    output = _subset_candidate(candidate, selected)
    report = {
        "schema": "anygsloc_cycle_core_coverage_reserve",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "input_anchor_count": count,
        "cycle_core_anchor_count": int(core.sum()),
        "chain_candidate_anchor_count": int((~core).sum()),
        "coverage_reserve_anchor_count": int((keep & ~core).sum()),
        "dropped_redundant_chain_anchor_count": int((~keep & ~core).sum()),
        "output_anchor_count": int(keep.sum()),
        "reserve_with_novel_spatial_voxel_count": spatial_added,
        "reserve_with_novel_mapping_image_cell_count": image_cell_added,
        "spatial_voxel_size_m": voxel_size,
        "image_grid_shape": [grid_y, grid_x],
        "reserve_order": "descending_identity_times_geometry_reliability",
        "deployment_form": "single_standard_anchor_map",
    }
    output["cycle_core_coverage_reserve"] = report
    return output, report
