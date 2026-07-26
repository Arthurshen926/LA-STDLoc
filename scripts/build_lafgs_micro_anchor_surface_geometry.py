#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

from localization_training.ulf_initializer import surface_normals_from_rotation


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-map", required=True)
    parser.add_argument("--gaussian-ply", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--covariance-reference-m2", type=float, default=1e-4)
    parser.add_argument("--maximum-normal-correction-m", type=float, default=0.03)
    args = parser.parse_args()
    anchor_path = Path(args.anchor_map).resolve()
    ply_path = Path(args.gaussian_ply).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state = torch.load(anchor_path, map_location="cpu", weights_only=False)
    if state.get("schema") != "lafgs_materialized_anchor_map":
        raise ValueError("unsupported materialized anchor map")
    base_count = int(state["base_anchor_count"])
    source_ids = state["source_primitive_ids"][base_count:].long()
    anchor_xyz = state["anchor_xyz"][base_count:].float()
    covariance = torch.as_tensor(
        state["micro_anchor_quality"]["covariance_trace_m2"]
    ).float()
    if covariance.numel() != source_ids.numel():
        raise ValueError("micro-anchor covariance does not align with rows")

    vertex = PlyData.read(ply_path)["vertex"].data
    centers_all = torch.from_numpy(
        np.stack((vertex["x"], vertex["y"], vertex["z"]), axis=1).copy()
    ).float()
    rotations_all = torch.from_numpy(
        np.stack(
            tuple(vertex[f"rot_{index}"] for index in range(4)), axis=1
        ).copy()
    ).float()
    if int(source_ids.max()) >= centers_all.shape[0]:
        raise ValueError("source primitive ID exceeds Gaussian PLY rows")
    centers = centers_all[source_ids]
    normals = surface_normals_from_rotation(rotations_all[source_ids])
    signed_normal_residual = ((anchor_xyz - centers) * normals).sum(dim=1)
    full_correction = signed_normal_residual[:, None] * normals

    provenance = {
        "anchor_map_path": str(anchor_path),
        "anchor_map_sha256": _sha256(anchor_path),
        "gaussian_ply_path": str(ply_path),
        "gaussian_ply_sha256": _sha256(ply_path),
    }
    outputs = {}
    for label, correction in (
        ("g1_full_tangent", full_correction),
        (
            "g2_covariance_bounded",
            full_correction
            * (
                covariance
                / (
                    covariance
                    + float(args.covariance_reference_m2)
                )
            )[:, None],
        ),
    ):
        correction_norm = torch.linalg.norm(correction, dim=1)
        correction = correction * (
            float(args.maximum_normal_correction_m)
            / correction_norm.clamp_min(1e-8)
        ).clamp(max=1.0)[:, None]
        output = dict(state)
        xyz = state["anchor_xyz"].clone()
        xyz[base_count:] = anchor_xyz - correction
        output["anchor_xyz"] = xyz
        output["surface_geometry"] = {
            "mode": label,
            "base_geometry_frozen": True,
            "descriptor_frozen": True,
            "tangent_coordinates_preserved": True,
            "covariance_reference_m2": float(
                args.covariance_reference_m2
            ),
            "maximum_normal_correction_m": float(
                args.maximum_normal_correction_m
            ),
            "raw_normal_residual_abs_mean_m": float(
                signed_normal_residual.abs().mean()
            ),
            "raw_normal_residual_abs_p95_m": float(
                torch.quantile(signed_normal_residual.abs(), 0.95)
            ),
            "raw_normal_residual_abs_max_m": float(
                signed_normal_residual.abs().max()
            ),
            "applied_correction_mean_m": float(
                torch.linalg.norm(correction, dim=1).mean()
            ),
            "applied_correction_p95_m": float(
                torch.quantile(torch.linalg.norm(correction, dim=1), 0.95)
            ),
            "applied_correction_max_m": float(
                torch.linalg.norm(correction, dim=1).max()
            ),
            "provenance": provenance,
        }
        output_path = output_dir / f"{label}.pt"
        torch.save(output, output_path)
        outputs[label] = {
            "path": str(output_path),
            **output["surface_geometry"],
        }
    (output_dir / "surface_geometry_summary.json").write_text(
        json.dumps(outputs, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
