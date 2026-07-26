#!/usr/bin/env python3

import argparse
import json
import pickle
from pathlib import Path

import torch

from localization_training.map_sanitization import (
    METRIC_ANCHOR_MIN_CONSISTENCY,
    binary_ranking_metrics,
    build_sanitization_scores,
    select_sanitized_landmarks,
)


def _subset_state(source, selected):
    count = int(torch.as_tensor(source["landmark_indices"]).numel())
    output = {}
    for key, value in source.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == count:
            output[key] = value[selected].clone()
        else:
            output[key] = value
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Build a compact localization map from independent reliability axes"
    )
    parser.add_argument("--source_state", required=True)
    parser.add_argument("--statistics", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--mode",
        choices=["loc", "loc_geo", "loc_geo_coverage"],
        required=True,
    )
    parser.add_argument("--budget", type=int, default=24000)
    parser.add_argument(
        "--outlier_labels",
        default="",
        help="Optional controlled-corruption labels for AUROC/AUPRC reporting.",
    )
    args = parser.parse_args()

    source_path = Path(args.source_state).expanduser().resolve()
    statistics_path = Path(args.statistics).expanduser().resolve()
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
        raise ValueError(
            "Statistics and source state do not describe the same ordered bank"
        )
    scores = build_sanitization_scores(
        payload["statistics"], payload["geometry_evidence"]
    )
    selected = select_sanitized_landmarks(
        scores,
        payload["statistics"],
        mode=args.mode,
        budget=args.budget,
    )
    if selected.numel() != min(int(args.budget), source_indices.numel()):
        raise RuntimeError("Sanitizer did not produce the requested exact budget")

    output = _subset_state(source, selected)
    output["version"] = max(int(output.get("version", 0)), 7)
    output["iteration"] = int(source.get("iteration", 0))
    config = dict(output.get("config", {}))
    config.update(
        {
            "dynamic_landmark_selection": False,
            "one_time_landmark_distillation": True,
            "sanitization_mode": str(args.mode),
            "sanitization_budget": int(selected.numel()),
            "sanitization_source_state": str(source_path),
            "sanitization_statistics": str(statistics_path),
            "sanitization_axes": [
                "localization_reliability",
                "geometry_reliability",
            ],
            "metric_anchor_min_consistency": float(
                METRIC_ANCHOR_MIN_CONSISTENCY
            ),
        }
    )
    output["config"] = config
    state_counts = {
        name: int((scores.state == value).sum().item())
        for name, value in (
            ("localization_excluded", 0),
            ("keep", 1),
            ("repairable", 2),
            ("reject", 3),
        )
    }
    selected_mask = torch.zeros(source_indices.numel(), dtype=torch.bool)
    selected_mask[selected] = True
    diagnostics = dict(output.get("diagnostics", {}))
    diagnostics.update(
        {
            "sanitization_mode": str(args.mode),
            "sanitization_source_count": int(source_indices.numel()),
            "sanitization_selected_count": int(selected.numel()),
            "sanitization_selected_loc_reliability_mean": float(
                scores.localization_reliability[selected].mean().item()
            ),
            "sanitization_selected_geo_reliability_mean": float(
                scores.geometry_reliability[selected].mean().item()
            ),
            "sanitization_rejected_loc_reliability_mean": float(
                scores.localization_reliability[~selected_mask].mean().item()
            ),
            "sanitization_rejected_geo_reliability_mean": float(
                scores.geometry_reliability[~selected_mask].mean().item()
            ),
            **{
                f"sanitization_state_{key}_count": value
                for key, value in state_counts.items()
            },
        }
    )
    output["diagnostics"] = diagnostics
    state_path = output_dir / "sanitized_lafgs_map_state.pt"
    torch.save(output, state_path)
    with (output_dir / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(output["landmark_indices"], handle)
    torch.save(
        {
            "version": 1,
            "landmark_indices": output["landmark_indices"],
            "fixed_bank": True,
            "one_time_landmark_distillation": True,
            "feature_dim": int(output["landmark_features"].shape[1]),
            "state_path": str(state_path),
            "sanitization_mode": str(args.mode),
        },
        output_dir / "landmark_meta.pt",
    )
    torch.save(
        {
            "version": 1,
            "source_landmark_indices": source_indices,
            "selected_relative_indices": selected,
            "localization_reliability": scores.localization_reliability,
            "geometry_reliability": scores.geometry_reliability,
            "components": scores.components,
            "state": scores.state,
        },
        output_dir / "sanitization_evidence.pt",
    )
    report = {
        "schema_version": 1,
        "source_state": str(source_path),
        "statistics": str(statistics_path),
        "mode": str(args.mode),
        "source_count": int(source_indices.numel()),
        "selected_count": int(selected.numel()),
        "state_counts": state_counts,
        "selected_localization_reliability_mean": diagnostics[
            "sanitization_selected_loc_reliability_mean"
        ],
        "selected_geometry_reliability_mean": diagnostics[
            "sanitization_selected_geo_reliability_mean"
        ],
        "excluded_localization_reliability_mean": diagnostics[
            "sanitization_rejected_loc_reliability_mean"
        ],
        "excluded_geometry_reliability_mean": diagnostics[
            "sanitization_rejected_geo_reliability_mean"
        ],
        "metric_anchor_min_consistency": float(
            METRIC_ANCHOR_MIN_CONSISTENCY
        ),
    }
    if args.outlier_labels:
        labels_path = Path(args.outlier_labels).expanduser().resolve()
        labels_payload = torch.load(
            labels_path, map_location="cpu", weights_only=False
        )
        outlier_label = torch.as_tensor(
            labels_payload["corrupted_mask"], dtype=torch.bool
        ).reshape(-1)
        if outlier_label.numel() != source_indices.numel():
            raise ValueError("Outlier labels do not align with the source bank")
        geometry_metrics = binary_ranking_metrics(
            1.0 - scores.geometry_reliability,
            outlier_label,
        )
        joint_metrics = binary_ranking_metrics(
            1.0
            - torch.sqrt(
                scores.localization_reliability
                * scores.geometry_reliability
            ),
            outlier_label,
        )
        report["controlled_outlier_evaluation"] = {
            "labels_path": str(labels_path),
            "geometry_outlier": geometry_metrics,
            "joint_outlier": joint_metrics,
            "outlier_rejection_rate": float(
                (~selected_mask[outlier_label]).float().mean().item()
            ),
            "clean_retention_rate": float(
                selected_mask[~outlier_label].float().mean().item()
            ),
        }
    (output_dir / "sanitization_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
