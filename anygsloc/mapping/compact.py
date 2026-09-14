"""Offline-only compatibility boundary around the frozen mapping kernels.

Run from the research checkout with mapping dependencies installed. This module
is never imported by the deployed localizer. No evaluator or job queue is used.
"""

from pathlib import Path
import torch


def select_compact_rows(state, cache):
    """Preserve the historical full selector's ordering and numeric parameters."""
    from anygsloc.mapping.selector import HierarchicalSufficiencySelector
    from anygsloc.mapping.pose_evidence import _build_pose_evidence, _initial_pose_state
    from anygsloc.mapping.observability import spatial_voxel_ids

    if state["provenance"].get("uses_test_queries") is not False:
        raise ValueError("selection requires mapping-only observations")
    names = state["v6_mapping_query_names"]
    if any(name not in cache for name in names):
        raise ValueError("missing mapping observation cache")
    n, q = len(state["anchor_ids"]), len(names)
    csr = state["projective_anchor_observations"]
    offsets = csr["observation_offsets"].tolist()
    queries = csr["query_indices"].tolist()
    points = csr["keypoint_indices"].tolist()
    edges = []
    for i in range(n):
        edge = {}
        for j in range(offsets[i], offsets[i + 1]):
            edge.setdefault(queries[j], []).append(points[j])
        edges.append(edge)
    quality = state["anchor_matchability"].float()
    order = torch.tensor(
        sorted(range(n), key=lambda i: (-float(quality[i]), -len(edges[i]), i))
    )
    selector = HierarchicalSufficiencySelector(edges, q)
    core, _ = selector.select_precision(
        order, 256, minimum_count=1024, maximum_count=n, check_interval=512
    )
    mask = torch.ones(n, dtype=torch.bool)
    mask[core] = False
    _, matching, _ = selector.complete_matching(
        order[mask[order]],
        quality,
        state["v6_mapping_query_bins"],
        requested_rows_per_query=256,
        maximum_reserve=n - len(core),
    )
    selected = selector.selected_ids
    voxels = spatial_voxel_ids(state["anchor_xyz"], 0.5)
    evidence = _build_pose_evidence(
        edges,
        state["anchor_xyz"],
        quality,
        state["anchor_position_covariance"],
        names,
        cache,
        voxels,
        pixel_variance=4.0,
        task_translation_m=0.05,
        task_rotation_deg=5.0,
    )
    info, rows, cells, depths, spatial, assignments = _initial_pose_state(
        selected, matching, evidence, q
    )
    mask[selected] = False
    selector.complete_observability(
        evidence,
        info,
        rows,
        cells,
        depths,
        spatial,
        order[mask[order]],
        torch.arange(n),
        voxels,
        maximum_additions=min(4096, n - len(selected)),
        maximum_per_source=1,
        maximum_per_voxel=3,
        minimum_relative_gain=0.02,
        initial_assignments=assignments,
    )
    return selector.selected_ids.sort().values, selector.artifact()


def compact_map(map_dir, cache_path, output):
    """Select one release map without generating or launching evaluations."""
    from anygsloc.mapping.subset import subset
    from anygsloc.artifacts import sha256, export_map, validate_identity
    from anygsloc.config import FROZEN

    torch.set_num_threads(FROZEN.cpu_threads)

    map_dir, output = Path(map_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    source = map_dir / "projective_anchor_map.pt"
    state = torch.load(source, map_location="cpu", weights_only=False)
    metric = torch.load(
        map_dir / "identity_metric.pt", map_location="cpu", weights_only=False
    )
    validate_identity(metric, state, source)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)["queries"]
    rows, trace = select_compact_rows(state, cache)
    revised = subset(state, rows)
    revised["provenance"] = dict(
        state["provenance"],
        release_selector=dict(
            version="compact-fast-v1",
            source_map_sha256=sha256(source),
            cache_sha256=sha256(cache_path),
            retained_source_rows=rows,
            uses_test_queries=False,
        ),
    )
    output.mkdir(parents=True)
    target = output / "projective_anchor_map.pt"
    torch.save(revised, target)
    identity = output / "identity_metric.pt"
    torch.save(
        dict(
            metric,
            landmark_indices=revised["anchor_ids"],
            map_path=str(target),
            map_sha256=sha256(target),
        ),
        identity,
    )
    torch.save(trace, output / "selector_trace.pt")
    return export_map(target, identity, output / "deployment_map.pt")
