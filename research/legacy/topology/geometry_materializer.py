"""Compatibility geometry materialization for localization Anchors.

This module is the boundary between heterogeneous geometry evidence and the
single geometry consumed by localization.  It deliberately implements the
frozen V3/P5 policy only:

* image-stable Tracks keep their image-only estimate;
* weak Tracks may use an accepted surface-regularized estimate;
* non-Track rows keep their existing surface-initialized fallback.

No optimization, threshold, selection, descriptor, or coordinate-system
change belongs here.  A future geometry policy must be introduced as a
separate experiment rather than silently changing this compatibility API.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch


GEOMETRY_IMAGE_TRIANGULATED = 0
GEOMETRY_SURFACE_REGULARIZED = 1
GEOMETRY_SURFACE_INITIALIZED = 2


def _rows(value, count: int) -> torch.Tensor:
    rows = torch.as_tensor(value, dtype=torch.long).reshape(-1)
    if rows.numel() and (
        int(rows.min()) < 0 or int(rows.max()) >= int(count)
    ):
        raise ValueError("track_rows are outside the Anchor geometry table")
    if torch.unique(rows).numel() != rows.numel():
        raise ValueError("track_rows must be unique")
    return rows


def _aligned(
    evidence: Mapping,
    key: str,
    shape: tuple[int, ...],
    *,
    like: torch.Tensor,
    default: torch.Tensor | None = None,
) -> torch.Tensor:
    if key not in evidence:
        if default is None:
            raise ValueError(f"geometry evidence is missing {key}")
        value = default
    else:
        value = evidence[key]
    tensor = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    if tensor.shape != shape:
        raise ValueError(
            f"{key} must have shape {shape}, got {tuple(tensor.shape)}"
        )
    return tensor


def _bool_aligned(
    evidence: Mapping,
    key: str,
    count: int,
    *,
    default: bool,
    device: torch.device,
) -> torch.Tensor:
    if key not in evidence:
        return torch.full((count,), default, dtype=torch.bool, device=device)
    value = torch.as_tensor(evidence[key], device=device).bool().reshape(-1)
    if value.numel() != count:
        raise ValueError(f"{key} must align with track_rows")
    return value


def _covariance_from_trace(trace: torch.Tensor) -> torch.Tensor:
    trace = torch.as_tensor(trace)
    return torch.diag_embed((trace / 3.0)[:, None].expand(-1, 3))


def _trace_from_covariance(covariance: torch.Tensor) -> torch.Tensor:
    return torch.diagonal(covariance, dim1=-2, dim2=-1).sum(dim=-1)


def materialize_geometry(anchor_evidence: Mapping) -> dict[str, torch.Tensor]:
    """Resolve heterogeneous evidence into one row-aligned Anchor geometry.

    ``anchor_evidence`` is intentionally a small, tensor-only contract:

    ``fallback_xyz``
        Existing row-aligned ``[N, 3]`` Anchor positions.  Non-Track rows use
        these positions unchanged.
    ``track_rows``
        Rows backed by Track evidence.
    ``track_image_only_xyz`` / ``track_surface_xyz``
        The two Track hypotheses, aligned with ``track_rows``.
    ``track_prefer_image_only``
        Frozen V3/P5 decision mask.  This API does not derive or tune it.
    ``track_surface_supported``
        Whether an accepted surface constraint exists for each Track.
    ``track_surface_dependence``
        Optional explicit compatibility annotation for an already materialized
        Map.  New materialization derives it from the chosen hypothesis.

    Covariance, trace, and reprojection tensors are optional.  If supplied,
    they use the same ``fallback_*``, ``track_image_only_*``, and
    ``track_surface_*`` naming.  ``track_covariance_override`` exists only to
    reproduce the legacy Registry's covariance annotation; localization never
    consumes that override.

    Set ``preserve_fallback_xyz=True`` when annotating an already materialized
    legacy map.  The geometry mode is still inferred from the evidence, while
    the localization coordinate is copied bit-for-bit from ``fallback_xyz``.
    """

    fallback_xyz = torch.as_tensor(anchor_evidence["fallback_xyz"])
    if fallback_xyz.ndim != 2 or fallback_xyz.shape[1] != 3:
        raise ValueError("fallback_xyz must have shape [N, 3]")
    count = int(fallback_xyz.shape[0])
    rows = _rows(anchor_evidence.get("track_rows", ()), count).to(
        fallback_xyz.device
    )
    track_count = int(rows.numel())

    image_xyz = _aligned(
        anchor_evidence,
        "track_image_only_xyz",
        (track_count, 3),
        like=fallback_xyz,
        default=fallback_xyz[rows],
    )
    surface_xyz = _aligned(
        anchor_evidence,
        "track_surface_xyz",
        (track_count, 3),
        like=fallback_xyz,
        default=fallback_xyz[rows],
    )
    prefer_image = _bool_aligned(
        anchor_evidence,
        "track_prefer_image_only",
        track_count,
        default=True,
        device=fallback_xyz.device,
    )
    surface_supported = _bool_aligned(
        anchor_evidence,
        "track_surface_supported",
        track_count,
        default=False,
        device=fallback_xyz.device,
    )

    xyz = fallback_xyz.clone()
    chosen_track_xyz = torch.where(
        prefer_image[:, None], image_xyz, surface_xyz
    )
    preserve_xyz = bool(anchor_evidence.get("preserve_fallback_xyz", False))
    if not preserve_xyz:
        xyz[rows] = chosen_track_xyz

    mode = torch.full(
        (count,),
        GEOMETRY_SURFACE_INITIALIZED,
        dtype=torch.int8,
        device=fallback_xyz.device,
    )
    mode[rows] = GEOMETRY_IMAGE_TRIANGULATED
    changed_from_image = ~torch.isclose(
        surface_xyz, image_xyz, rtol=0.0, atol=1e-7
    ).all(dim=1)
    if "track_surface_dependence" in anchor_evidence:
        regularized = _bool_aligned(
            anchor_evidence,
            "track_surface_dependence",
            track_count,
            default=False,
            device=fallback_xyz.device,
        )
        if bool((regularized & ~surface_supported).any()):
            raise ValueError(
                "surface-dependent Track geometry requires surface evidence"
            )
    else:
        regularized = surface_supported & ~prefer_image & changed_from_image
    mode[rows[regularized]] = GEOMETRY_SURFACE_REGULARIZED

    surface_evidence = torch.ones(
        count, dtype=torch.bool, device=fallback_xyz.device
    )
    surface_evidence[rows] = surface_supported
    surface_dependence = torch.ones(
        count, dtype=torch.bool, device=fallback_xyz.device
    )
    surface_dependence[rows] = regularized

    fallback_covariance = anchor_evidence.get("fallback_covariance")
    if fallback_covariance is None:
        covariance = torch.full(
            (count, 3, 3),
            float("nan"),
            dtype=fallback_xyz.dtype,
            device=fallback_xyz.device,
        )
    else:
        covariance = torch.as_tensor(
            fallback_covariance, device=fallback_xyz.device
        )
        if covariance.shape != (count, 3, 3):
            raise ValueError(
                "fallback_covariance must have shape "
                f"{(count, 3, 3)}, got {tuple(covariance.shape)}"
            )
        covariance = covariance.clone()
    surface_covariance = _aligned(
        anchor_evidence,
        "track_surface_covariance",
        (track_count, 3, 3),
        like=covariance,
        default=covariance[rows],
    )
    image_covariance = _aligned(
        anchor_evidence,
        "track_image_only_covariance",
        (track_count, 3, 3),
        like=covariance,
        default=surface_covariance,
    )
    chosen_covariance = torch.where(
        prefer_image[:, None, None], image_covariance, surface_covariance
    )
    if "track_covariance_override" in anchor_evidence:
        chosen_covariance = _aligned(
            anchor_evidence,
            "track_covariance_override",
            (track_count, 3, 3),
            like=covariance,
        )
    covariance[rows] = chosen_covariance

    if "fallback_covariance_trace" in anchor_evidence:
        trace_like = torch.as_tensor(
            anchor_evidence["fallback_covariance_trace"],
            device=fallback_xyz.device,
        )
        covariance_trace = _aligned(
            anchor_evidence,
            "fallback_covariance_trace",
            (count,),
            like=trace_like,
        ).clone()
    else:
        covariance_trace = _trace_from_covariance(covariance)
    surface_trace = _aligned(
        anchor_evidence,
        "track_surface_covariance_trace",
        (track_count,),
        like=covariance_trace,
        default=_trace_from_covariance(surface_covariance),
    )
    image_trace = _aligned(
        anchor_evidence,
        "track_image_only_covariance_trace",
        (track_count,),
        like=covariance_trace,
        default=surface_trace,
    )
    chosen_trace = torch.where(prefer_image, image_trace, surface_trace)
    if "track_covariance_trace_override" in anchor_evidence:
        chosen_trace = _aligned(
            anchor_evidence,
            "track_covariance_trace_override",
            (track_count,),
            like=covariance_trace,
        )
    covariance_trace[rows] = chosen_trace

    output = {
        "xyz": xyz,
        "covariance": covariance,
        "covariance_trace": covariance_trace,
        "geometry_mode": mode,
        "surface_evidence": surface_evidence,
        "surface_dependence": surface_dependence,
    }
    for suffix in ("reprojection_median_px", "reprojection_p90_px"):
        fallback_key = f"fallback_{suffix}"
        surface_key = f"track_surface_{suffix}"
        image_key = f"track_image_only_{suffix}"
        if not any(
            key in anchor_evidence
            for key in (fallback_key, surface_key, image_key)
        ):
            continue
        fallback_like = torch.as_tensor(
            anchor_evidence.get(
                fallback_key,
                torch.full(
                    (count,),
                    float("nan"),
                    dtype=fallback_xyz.dtype,
                    device=fallback_xyz.device,
                ),
            ),
            device=fallback_xyz.device,
        )
        fallback = _aligned(
            anchor_evidence,
            fallback_key,
            (count,),
            like=fallback_like,
            default=fallback_like,
        ).clone()
        surface = _aligned(
            anchor_evidence,
            surface_key,
            (track_count,),
            like=fallback,
            default=fallback[rows],
        )
        image = _aligned(
            anchor_evidence,
            image_key,
            (track_count,),
            like=fallback,
            default=surface,
        )
        fallback[rows] = torch.where(prefer_image, image, surface)
        output[suffix] = fallback
    return output


def materialize_track_geometry_compatibility(
    geometry: Mapping,
    image_only_mask: torch.Tensor,
) -> dict:
    """Apply the V3/P5 Track policy while preserving the legacy payload API."""

    if "triangulation_image_only_xyz" not in geometry:
        return geometry
    surface_xyz = torch.as_tensor(geometry["triangulated_xyz"])
    count = int(surface_xyz.shape[0])
    mask = torch.as_tensor(image_only_mask).bool().reshape(-1)
    if mask.numel() != count:
        raise ValueError("image_only_mask does not align with Track geometry")
    rows = torch.arange(count, device=surface_xyz.device)
    evidence: dict[str, object] = {
        "fallback_xyz": surface_xyz,
        "track_rows": rows,
        "track_surface_xyz": surface_xyz,
        "track_image_only_xyz": geometry["triangulation_image_only_xyz"],
        "track_prefer_image_only": mask,
        "track_surface_supported": geometry.get(
            "triangulation_surface_supported",
            torch.zeros(count, dtype=torch.bool, device=surface_xyz.device),
        ),
    }
    if "triangulation_covariance_matrix" in geometry:
        evidence["fallback_covariance"] = geometry[
            "triangulation_covariance_matrix"
        ]
        evidence["track_surface_covariance"] = geometry[
            "triangulation_covariance_matrix"
        ]
    if "triangulation_image_only_covariance_matrix" in geometry:
        if "triangulation_covariance_matrix" not in geometry:
            raise ValueError(
                "Track geometry is missing triangulation_covariance_matrix"
            )
        evidence["track_image_only_covariance"] = geometry[
            "triangulation_image_only_covariance_matrix"
        ]
    if "triangulation_covariance_trace" in geometry:
        evidence["fallback_covariance_trace"] = geometry[
            "triangulation_covariance_trace"
        ]
        evidence["track_surface_covariance_trace"] = geometry[
            "triangulation_covariance_trace"
        ]
    if "triangulation_image_only_covariance_trace" in geometry:
        if "triangulation_covariance_trace" not in geometry:
            raise ValueError(
                "Track geometry is missing triangulation_covariance_trace"
            )
        evidence["track_image_only_covariance_trace"] = geometry[
            "triangulation_image_only_covariance_trace"
        ]
    for suffix in ("reprojection_median_px", "reprojection_p90_px"):
        target = f"triangulation_{suffix}"
        source = f"triangulation_image_only_{suffix}"
        if source in geometry:
            if target not in geometry:
                raise ValueError(f"Track geometry is missing {target}")
            evidence[f"fallback_{suffix}"] = geometry[target]
            evidence[f"track_surface_{suffix}"] = geometry[target]
            evidence[f"track_image_only_{suffix}"] = geometry[source]

    materialized = materialize_geometry(evidence)
    revised = dict(geometry)
    revised["triangulated_xyz"] = materialized["xyz"]
    replacements = {
        "triangulation_covariance_matrix": "covariance",
        "triangulation_covariance_trace": "covariance_trace",
        "triangulation_reprojection_median_px": "reprojection_median_px",
        "triangulation_reprojection_p90_px": "reprojection_p90_px",
    }
    for target, source in replacements.items():
        image_key = target.replace(
            "triangulation_", "triangulation_image_only_", 1
        )
        if image_key in geometry:
            revised[target] = materialized[source]
    return revised


def materialize_legacy_map_geometry(
    state: Mapping,
    track_payload: Mapping | None,
) -> dict[str, torch.Tensor]:
    """Annotate an existing V3/P5 Map without changing its coordinates.

    Track covariance intentionally follows the historical Registry behavior:
    it is copied from the final Track payload even when a core Track's deployed
    coordinate is image-only.  The explicit override makes that legacy
    annotation visible without changing it during this compatibility step.
    """

    anchor_type = torch.as_tensor(state["anchor_type"]).detach().cpu().long()
    track_ids = torch.as_tensor(state["track_cluster_ids"]).detach().cpu().long()
    xyz = torch.as_tensor(state["anchor_xyz"]).detach().cpu().float()
    count = int(anchor_type.numel())
    if xyz.shape != (count, 3) or track_ids.numel() != count:
        raise ValueError("legacy Anchor geometry tensors do not align")
    covariance = torch.full((count, 3, 3), float("nan"), dtype=torch.float32)
    if "anchor_position_covariance" in state:
        value = torch.as_tensor(state["anchor_position_covariance"]).detach().cpu().float()
        if value.shape != covariance.shape:
            raise ValueError("anchor position covariance does not align with map")
        covariance.copy_(value)
    rows = torch.nonzero(anchor_type == 1, as_tuple=False).reshape(-1)
    evidence: dict[str, object] = {
        "fallback_xyz": xyz,
        "fallback_covariance": covariance,
        "track_rows": rows,
        "preserve_fallback_xyz": True,
    }
    if track_payload is None or rows.numel() == 0:
        return materialize_geometry(evidence)

    geometry = track_payload["track_geometry"]
    selected_tracks = track_ids[rows]
    track_count = int(
        torch.as_tensor(geometry["triangulated_xyz"]).shape[0]
    )
    if selected_tracks.numel() and (
        int(selected_tracks.min()) < 0 or int(selected_tracks.max()) >= track_count
    ):
        raise ValueError("Anchor map references an invalid Track geometry row")
    surface_xyz = torch.as_tensor(
        geometry["triangulated_xyz"]
    ).detach().cpu().float()[selected_tracks]
    if "triangulation_image_only_xyz" in geometry:
        image_xyz = torch.as_tensor(
            geometry["triangulation_image_only_xyz"]
        ).detach().cpu().float()[selected_tracks]
    else:
        image_xyz = surface_xyz
    supported_all = torch.as_tensor(
        geometry.get(
            "triangulation_surface_supported",
            torch.zeros(track_count, dtype=torch.bool),
        )
    ).detach().cpu().bool()
    supported = supported_all[selected_tracks]
    regularized = supported & ~torch.isclose(
        xyz[rows], image_xyz, rtol=0.0, atol=1e-7
    ).all(dim=1)
    evidence.update(
        {
            "track_surface_xyz": surface_xyz,
            "track_image_only_xyz": image_xyz,
            "track_prefer_image_only": ~regularized,
            "track_surface_supported": supported,
            "track_surface_dependence": regularized,
        }
    )
    if "triangulation_covariance_matrix" in geometry:
        selected_covariance = torch.as_tensor(
            geometry["triangulation_covariance_matrix"]
        ).detach().cpu().float()[selected_tracks]
    elif "triangulation_covariance_trace" in geometry:
        selected_trace = torch.as_tensor(
            geometry["triangulation_covariance_trace"]
        ).detach().cpu().float()[selected_tracks]
        selected_covariance = _covariance_from_trace(selected_trace)
    else:
        selected_covariance = covariance[rows]
    evidence["track_surface_covariance"] = selected_covariance
    evidence["track_image_only_covariance"] = selected_covariance
    evidence["track_covariance_override"] = selected_covariance
    return materialize_geometry(evidence)
