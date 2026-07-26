#!/usr/bin/env python3

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

from localization_training.gaussian_prior import GaussianPriorGeometry
from localization_training.map_sanitization import build_sanitization_scores


def _properties(vertex, prefix):
    names = [name for name in vertex.data.dtype.names if name.startswith(prefix)]
    return sorted(names, key=lambda name: int(name.rsplit("_", 1)[1]))


def _selected_matrix(vertex, names, indices):
    values = np.stack(
        [np.asarray(vertex[name])[indices] for name in names], axis=1
    )
    return torch.from_numpy(values.copy()).float()


def _subset_state(source, selected):
    count = int(torch.as_tensor(source["landmark_indices"]).numel())
    output = {}
    for key, value in source.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == count:
            output[key] = value[selected].clone()
        else:
            output[key] = value
    return output


def _load_prior(rgb_ply_path, landmark_indices):
    ply = PlyData.read(str(rgb_ply_path))
    vertex = ply.elements[0]
    maximum_index = (
        int(landmark_indices.max().item())
        if landmark_indices.numel()
        else -1
    )
    if maximum_index >= len(vertex):
        raise ValueError("Localization landmark ID exceeds RGB PLY size")
    numpy_indices = landmark_indices.numpy()
    xyz = _selected_matrix(vertex, ["x", "y", "z"], numpy_indices)
    rotation_names = _properties(vertex, "rot_")
    scaling_names = _properties(vertex, "scale_")
    if len(rotation_names) != 4 or len(scaling_names) not in {2, 3}:
        raise ValueError("RGB PLY has unsupported rotation/scale properties")
    rotation = _selected_matrix(vertex, rotation_names, numpy_indices)
    scaling = _selected_matrix(vertex, scaling_names, numpy_indices).exp()
    gaussian_type = "2dgs" if scaling.shape[1] == 2 else "3dgs"
    return GaussianPriorGeometry(
        gaussian_type=gaussian_type,
        xyz=xyz,
        rotation=rotation,
        scaling=scaling,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply high-confidence localization-only surface-anchor repair "
            "while leaving the RGB Gaussian prior and descriptors frozen"
        )
    )
    parser.add_argument("--source_state", required=True)
    parser.add_argument("--statistics", required=True)
    parser.add_argument("--rgb_ply", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--operation",
        choices=("repair_only", "reject_and_repair"),
        default="repair_only",
    )
    parser.add_argument("--tangent_bound_m", type=float, default=0.005)
    parser.add_argument("--normal_bound_m", type=float, default=0.002)
    parser.add_argument("--covariance_scale", type=float, default=0.5)
    parser.add_argument("--absolute_bound_m", type=float, default=0.03)
    parser.add_argument("--min_distinct_view_bins", type=int, default=3)
    parser.add_argument("--min_provenance_support_views", type=int, default=2)
    parser.add_argument("--min_provenance_consensus_rate", type=float, default=0.1)
    parser.add_argument("--outlier_labels", default="")
    args = parser.parse_args()

    source_path = Path(args.source_state).expanduser().resolve()
    statistics_path = Path(args.statistics).expanduser().resolve()
    rgb_ply_path = Path(args.rgb_ply).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source = torch.load(source_path, map_location="cpu", weights_only=False)
    payload = torch.load(
        statistics_path, map_location="cpu", weights_only=False
    )
    source_indices = torch.as_tensor(
        source["landmark_indices"], dtype=torch.long
    ).reshape(-1)
    statistics_indices = torch.as_tensor(
        payload["landmark_indices"], dtype=torch.long
    ).reshape(-1)
    if not torch.equal(source_indices, statistics_indices):
        raise ValueError("Statistics and source state landmark IDs differ")
    source_xyz = torch.as_tensor(
        source["landmark_xyz"], dtype=torch.float32
    )
    if source_xyz.shape != (source_indices.numel(), 3):
        raise ValueError("Malformed source landmark_xyz")

    scores = build_sanitization_scores(
        payload["statistics"], payload["geometry_evidence"]
    )
    geometry = payload["geometry_evidence"]
    triangulated_xyz = torch.as_tensor(
        geometry["triangulated_xyz"], dtype=torch.float32
    )
    high_confidence = torch.as_tensor(
        geometry["triangulation_high_confidence"], dtype=torch.bool
    ).reshape(-1)
    distinct_bins = torch.as_tensor(
        geometry["triangulation_distinct_view_bin_count"], dtype=torch.long
    ).reshape(-1)
    provenance_views = torch.as_tensor(
        geometry.get(
            "track_provenance_support_views",
            torch.zeros_like(distinct_bins),
        ),
        dtype=torch.long,
    ).reshape(-1)
    provenance_rate = torch.as_tensor(
        geometry.get(
            "track_provenance_consensus_rate",
            torch.zeros_like(distinct_bins, dtype=torch.float32),
        ),
        dtype=torch.float32,
    ).reshape(-1)
    repair_mask = (
        (scores.state == 2)
        & high_confidence
        & (distinct_bins >= int(args.min_distinct_view_bins))
        & (
            provenance_views
            >= int(args.min_provenance_support_views)
        )
        & (
            provenance_rate
            >= float(args.min_provenance_consensus_rate)
        )
        & torch.isfinite(triangulated_xyz).all(dim=1)
    )
    reject_mask = (
        (scores.state == 3)
        if args.operation == "reject_and_repair"
        else torch.zeros_like(repair_mask)
    )

    prior = _load_prior(rgb_ply_path, source_indices)
    projected_target = prior.project_anchor_target(
        torch.where(
            repair_mask[:, None], triangulated_xyz, prior.xyz
        ),
        tangent_bound_m=args.tangent_bound_m,
        normal_bound_m=args.normal_bound_m,
        covariance_scale=args.covariance_scale,
        absolute_bound_m=args.absolute_bound_m,
    )
    repaired_xyz = source_xyz.clone()
    repaired_xyz[repair_mask] = projected_target[repair_mask]
    encoded_offset = torch.as_tensor(
        source.get("raw_anchor_offset", torch.zeros_like(source_xyz)),
        dtype=torch.float32,
    ).clone()
    if encoded_offset.shape != source_xyz.shape:
        raise ValueError("Malformed source raw_anchor_offset")
    encoded_offset[repair_mask] = prior.encode_anchor(
        projected_target,
        tangent_bound_m=args.tangent_bound_m,
        normal_bound_m=args.normal_bound_m,
        covariance_scale=args.covariance_scale,
        absolute_bound_m=args.absolute_bound_m,
    )[repair_mask]

    selected = torch.nonzero(~reject_mask, as_tuple=False).reshape(-1)
    output = _subset_state(source, selected)
    output["landmark_xyz"] = repaired_xyz[selected]
    output["raw_anchor_offset"] = encoded_offset[selected]
    output["version"] = max(int(output.get("version", 0)), 8)
    config = dict(output.get("config", {}))
    config.update(
        {
            "selective_geometry_operation": str(args.operation),
            "selective_geometry_source_state": str(source_path),
            "selective_geometry_statistics": str(statistics_path),
            "selective_geometry_rgb_ply": str(rgb_ply_path),
            "surface_anchor_parameterization": (
                "radial_tanh_tangent_plane_v1"
                if prior.gaussian_type == "2dgs"
                else "covariance_bounded_tanh_v1"
            ),
            "tangent_bound_m": float(args.tangent_bound_m),
            "normal_bound_m": float(args.normal_bound_m),
            "covariance_anchor_scale": float(args.covariance_scale),
            "covariance_anchor_absolute_bound_m": float(
                args.absolute_bound_m
            ),
        }
    )
    output["config"] = config

    displacement = torch.linalg.norm(repaired_xyz - source_xyz, dim=1)
    target_residual = torch.linalg.norm(
        repaired_xyz - triangulated_xyz.nan_to_num(), dim=1
    )
    diagnostics = dict(output.get("diagnostics", {}))
    diagnostics.update(
        {
            "selective_geometry_operation": str(args.operation),
            "selective_geometry_repaired_count": int(repair_mask.sum()),
            "selective_geometry_rejected_count": int(reject_mask.sum()),
            "selective_geometry_repair_displacement_mean_m": float(
                displacement[repair_mask].mean().item()
                if bool(repair_mask.any())
                else 0.0
            ),
            "selective_geometry_repair_displacement_max_m": float(
                displacement[repair_mask].max().item()
                if bool(repair_mask.any())
                else 0.0
            ),
            "selective_geometry_target_residual_mean_m": float(
                target_residual[repair_mask].mean().item()
                if bool(repair_mask.any())
                else 0.0
            ),
        }
    )
    output["diagnostics"] = diagnostics

    state_path = output_dir / "repaired_lafgs_map_state.pt"
    torch.save(output, state_path)
    with (output_dir / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(output["landmark_indices"], handle)
    torch.save(
        {
            "version": 1,
            "landmark_indices": output["landmark_indices"],
            "fixed_bank": True,
            "one_time_landmark_distillation": False,
            "feature_dim": int(output["landmark_features"].shape[1]),
            "state_path": str(state_path),
            "selective_geometry_operation": str(args.operation),
        },
        output_dir / "landmark_meta.pt",
    )
    evidence = {
        "version": 1,
        "source_landmark_indices": source_indices,
        "repair_mask": repair_mask,
        "reject_mask": reject_mask,
        "projected_target_xyz": projected_target,
        "state": scores.state,
    }
    torch.save(evidence, output_dir / "selective_geometry_evidence.pt")

    report = {
        "schema_version": 1,
        "source_state": str(source_path),
        "statistics": str(statistics_path),
        "rgb_ply": str(rgb_ply_path),
        "output_state": str(state_path),
        "operation": str(args.operation),
        "gaussian_type": str(prior.gaussian_type),
        "source_count": int(source_indices.numel()),
        "selected_count": int(selected.numel()),
        "repaired_count": int(repair_mask.sum()),
        "rejected_count": int(reject_mask.sum()),
        "repair_displacement_mean_m": diagnostics[
            "selective_geometry_repair_displacement_mean_m"
        ],
        "repair_displacement_max_m": diagnostics[
            "selective_geometry_repair_displacement_max_m"
        ],
        "target_residual_mean_m": diagnostics[
            "selective_geometry_target_residual_mean_m"
        ],
    }
    if args.outlier_labels:
        labels = torch.load(
            Path(args.outlier_labels).expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        corrupted = torch.as_tensor(
            labels["corrupted_mask"], dtype=torch.bool
        ).reshape(-1)
        if corrupted.numel() != source_indices.numel():
            raise ValueError("Outlier labels do not align with source state")
        report.update(
            {
                "repaired_corrupted_count": int(
                    (repair_mask & corrupted).sum()
                ),
                "repaired_clean_count": int(
                    (repair_mask & ~corrupted).sum()
                ),
                "rejected_corrupted_count": int(
                    (reject_mask & corrupted).sum()
                ),
                "rejected_clean_count": int(
                    (reject_mask & ~corrupted).sum()
                ),
            }
        )
    (output_dir / "selective_geometry_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
