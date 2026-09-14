#!/usr/bin/env python3
"""Audit V3/P5 geometry-materializer parity on a frozen mapping artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common.hashing import sha256_file
from topology.anchor_registry import (
    build_anchor_registry,
    validate_registry_compatibility,
)
from topology.geometry_materializer import (
    GEOMETRY_IMAGE_TRIANGULATED,
    GEOMETRY_SURFACE_INITIALIZED,
    GEOMETRY_SURFACE_REGULARIZED,
    materialize_legacy_map_geometry,
    materialize_track_geometry_compatibility,
)


def _load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _bitwise_equal(left, right) -> bool:
    left = torch.as_tensor(left).detach().cpu().contiguous()
    right = torch.as_tensor(right).detach().cpu().contiguous()
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return bool(torch.equal(left.view(torch.uint8), right.view(torch.uint8)))


def _legacy_reference(geometry: dict, image_only_mask: torch.Tensor) -> dict:
    if "triangulation_image_only_xyz" not in geometry:
        return geometry
    revised = dict(geometry)
    core = torch.as_tensor(image_only_mask).bool()
    replacements = {
        "triangulated_xyz": "triangulation_image_only_xyz",
        "triangulation_covariance_trace": (
            "triangulation_image_only_covariance_trace"
        ),
        "triangulation_covariance_matrix": (
            "triangulation_image_only_covariance_matrix"
        ),
        "triangulation_reprojection_median_px": (
            "triangulation_image_only_reprojection_median_px"
        ),
        "triangulation_reprojection_p90_px": (
            "triangulation_image_only_reprojection_p90_px"
        ),
    }
    for target, source in replacements.items():
        if source not in geometry:
            continue
        value = torch.as_tensor(geometry[target]).clone()
        value[core] = torch.as_tensor(geometry[source])[core]
        revised[target] = value
    return revised


def audit(map_path: Path, payload_path: Path) -> dict:
    state = _load(map_path)
    payload = _load(payload_path)
    geometry = payload["track_geometry"]
    track_count = int(torch.as_tensor(geometry["triangulated_xyz"]).shape[0])
    reconstruction = state.get("track_centric_reconstruction", {})
    provenance = reconstruction.get("selection_provenance", {})
    core_ids = torch.as_tensor(
        provenance.get("track_core_universe_ids", ())
    ).long().reshape(-1)
    if "triangulation_image_only_xyz" in geometry and core_ids.numel() == 0:
        raise ValueError(
            "map lacks exact Track-core provenance required for parity audit"
        )
    image_only_mask = torch.zeros(track_count, dtype=torch.bool)
    if core_ids.numel():
        if int(core_ids.min()) < 0 or int(core_ids.max()) >= track_count:
            raise ValueError("Track-core provenance is outside payload")
        image_only_mask[core_ids] = True
    reported_core_count = int(
        reconstruction.get("track_core_count", core_ids.numel())
    )

    reference = _legacy_reference(geometry, image_only_mask)
    materialized = materialize_track_geometry_compatibility(
        geometry, image_only_mask
    )
    compared_fields = [
        key
        for key in (
            "triangulated_xyz",
            "triangulation_covariance_trace",
            "triangulation_covariance_matrix",
            "triangulation_reprojection_median_px",
            "triangulation_reprojection_p90_px",
        )
        if key in reference
    ]
    field_parity = {
        key: _bitwise_equal(reference[key], materialized[key])
        for key in compared_fields
    }

    map_track_rows = torch.nonzero(
        torch.as_tensor(state["anchor_type"]) == 1, as_tuple=False
    ).reshape(-1)
    selected_tracks = torch.as_tensor(state["track_cluster_ids"])[
        map_track_rows
    ].long()
    selected_xyz = torch.as_tensor(materialized["triangulated_xyz"])[
        selected_tracks
    ]
    map_xyz_parity = _bitwise_equal(
        selected_xyz, torch.as_tensor(state["anchor_xyz"])[map_track_rows]
    )

    annotation = materialize_legacy_map_geometry(state, payload)
    annotation_xyz_parity = _bitwise_equal(
        annotation["xyz"], state["anchor_xyz"]
    )
    registry = build_anchor_registry(state, track_payload=payload)
    validate_registry_compatibility(registry, state)
    mode = torch.as_tensor(annotation["geometry_mode"])
    result = {
        "schema": "lafgs_geometry_materializer_parity_audit",
        "version": 1,
        "map": str(map_path.resolve()),
        "map_sha256": sha256_file(map_path),
        "track_payload": str(payload_path.resolve()),
        "track_payload_sha256": sha256_file(payload_path),
        "anchor_count": int(torch.as_tensor(state["anchor_ids"]).numel()),
        "selected_track_count": int(map_track_rows.numel()),
        "track_core_count": reported_core_count,
        "track_core_row_provenance_exact": bool(core_ids.numel()),
        "field_bitwise_parity": field_parity,
        "all_track_geometry_fields_bitwise_equal": bool(
            all(field_parity.values())
        ),
        "selected_track_xyz_matches_map_bitwise": map_xyz_parity,
        "legacy_annotation_xyz_matches_map_bitwise": annotation_xyz_parity,
        "registry_localization_tensors_bitwise_equal": True,
        "geometry_mode": {
            "image_triangulated": int(
                (mode == GEOMETRY_IMAGE_TRIANGULATED).sum()
            ),
            "surface_regularized": int(
                (mode == GEOMETRY_SURFACE_REGULARIZED).sum()
            ),
            "surface_initialized": int(
                (mode == GEOMETRY_SURFACE_INITIALIZED).sum()
            ),
        },
        "surface_evidence_count": int(
            torch.as_tensor(annotation["surface_evidence"]).sum()
        ),
        "surface_dependent_count": int(
            torch.as_tensor(annotation["surface_dependence"]).sum()
        ),
        "legacy_registry_track_covariance_policy": (
            "final_track_payload_compatibility_override"
        ),
        "surface_dependence_semantics": (
            "deployed_geometry_uses_surface_not_merely_surface_available"
        ),
        "changes_selector": False,
        "changes_descriptor": False,
        "changes_localization": False,
        "uses_test_queries": False,
    }
    result["pass"] = bool(
        result["all_track_geometry_fields_bitwise_equal"]
        and map_xyz_parity
        and annotation_xyz_parity
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--track-payload", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.map, args.track_payload)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
