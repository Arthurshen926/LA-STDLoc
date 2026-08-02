#!/usr/bin/env python3
"""Summarize the frozen LaFGS off-the-shelf prior matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _first_map(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not report or not report.get("maps"):
        return None
    return next(iter(report["maps"].values()))


def _aggregate_metrics(results: dict[str, Any], stage: str) -> dict[str, Any] | None:
    stage_results = results.get("results", {}).get(stage, {})
    aggregate = stage_results.get("seed_aggregate")
    if not aggregate:
        return None
    keys = (
        "median_te_cm",
        "mean_te_cm",
        "p90_te_cm",
        "raw_gt_precision_2px_percent",
        "inlier_gt_precision_2px_percent",
        "solver_inlier_ratio_percent",
        "mean_hypotheses",
        "total_ms",
        "recall_5cm_5deg_percent",
    )
    return {
        key: aggregate[key]
        for key in keys
        if key in aggregate
    }


def _tail_diagnostics(
    results: dict[str, Any],
    stage: str,
    *,
    threshold_cm: float = 100.0,
) -> dict[str, Any] | None:
    stage_results = results.get("results", {}).get(stage, {})
    seed_rows = {
        seed: row
        for seed, row in stage_results.items()
        if seed != "seed_aggregate" and isinstance(row, dict)
    }
    failure_sets: list[set[str]] = []
    per_seed: dict[str, Any] = {}
    maximum_by_query: dict[str, float] = {}
    for seed, row in sorted(seed_rows.items()):
        result_path = Path(row.get("result_path", "")) / "results.json"
        records = _read_json(result_path)
        if not isinstance(records, list) or not records:
            continue
        errors = [float(record["sparse_TE"]) for record in records]
        names = [
            str(
                record.get("image_name")
                or record.get("name")
                or record.get("image")
                or record.get("query")
                or index
            )
            for index, record in enumerate(records)
        ]
        failures = {
            name for name, error in zip(names, errors) if error > threshold_cm
        }
        failure_sets.append(failures)
        for name, error in zip(names, errors):
            maximum_by_query[name] = max(maximum_by_query.get(name, 0.0), error)
        per_seed[seed] = {
            "query_count": len(errors),
            "mean_te_cm": sum(errors) / len(errors),
            "failure_count": len(failures),
            "maximum_te_cm": max(errors),
        }
    if not failure_sets:
        return None
    union = set().union(*failure_sets)
    persistent = set.intersection(*failure_sets)
    unstable = union - persistent
    top_failures = sorted(
        ((name, maximum_by_query[name]) for name in union),
        key=lambda item: item[1],
        reverse=True,
    )[:10]
    return {
        "threshold_cm": threshold_cm,
        "seed_count": len(failure_sets),
        "persistent_failure_count": len(persistent),
        "seed_unstable_failure_count": len(unstable),
        "failure_union_count": len(union),
        "persistent_queries": sorted(persistent),
        "seed_unstable_queries": sorted(unstable),
        "top_failure_maximum_te_cm": [
            {"image_name": name, "maximum_te_cm": error}
            for name, error in top_failures
        ],
        "per_seed": per_seed,
    }


def _raster_provenance(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    total_rows = 0
    valid_rows = 0
    for record in payload.get("records", []):
        primitive_ids = torch.as_tensor(record["primitive_ids"])
        total_rows += int(primitive_ids.shape[0])
        if primitive_ids.ndim == 1:
            valid_rows += int((primitive_ids >= 0).sum())
        else:
            valid_rows += int((primitive_ids >= 0).any(dim=1).sum())
    offsets = payload.get("anchor_source_offsets")
    anchor_count = int(torch.as_tensor(offsets).numel() - 1) if offsets is not None else 0
    source_universe = payload.get("source_universe")
    source_count = int(torch.as_tensor(source_universe).numel()) if source_universe is not None else 0
    return {
        "query_row_count": total_rows,
        "valid_query_row_count": valid_rows,
        "valid_query_row_fraction": valid_rows / total_rows if total_rows else 0.0,
        "anchor_count": anchor_count,
        "unique_source_primitive_count": source_count,
        "anchor_to_unique_source_ratio": source_count / anchor_count if anchor_count else 0.0,
    }


def summarize_profile(
    root: Path,
    scene: str,
    profile: str,
    *,
    lafgs_namespace: str = "lafgs",
) -> dict[str, Any]:
    prior_root = root / "priors" / scene / profile
    lafgs_root = root / lafgs_namespace / profile / scene
    protocol = _read_json(prior_root / "offtheshelf_prior_protocol.json")
    quality = _read_json(prior_root / "prior_quality.json")
    frozen = _read_json(lafgs_root / "frozen_results.json")
    bootstrap = _read_json(
        lafgs_root / "runs" / "frozen_v1" / "bootstrap" / "training_summary.json"
    )
    geometry_teacher_tag = (
        "g3_track_provenance_v1"
        if profile == "vanilla_2dgs"
        else "g2_track_first_v1"
    )
    statistics_root = (
        lafgs_root
        / "runs"
        / "frozen_v1"
        / f"statistics_combined_1000_frozen_{geometry_teacher_tag}"
    )
    statistics = _read_json(statistics_root / "training_summary.json")
    compact = _first_map(
        _read_json(lafgs_root / "compact_map" / "minimum_sufficient_build.json")
    )
    pose_reserve = _first_map(
        _read_json(lafgs_root / "pose_reserve" / "pose_sufficient_build.json")
    )
    result: dict[str, Any] = {
        "scene": scene,
        "profile": profile,
        "complete": bool(protocol and quality and frozen),
    }
    if protocol:
        prior = protocol.get("rgb_prior", {})
        source_ply = Path(prior.get("source_ply", ""))
        result["prior"] = {
            "official_repository": protocol.get("official_repository"),
            "official_commit": protocol.get("official_commit"),
            "mapping_image_count": protocol.get("mapping_input", {}).get(
                "mapping_image_count"
            ),
            "test_rgb_used": protocol.get("controls", {}).get("test_rgb_used"),
            "semantic_mask_used": protocol.get("controls", {}).get(
                "rgb_prior_semantic_object_sky_mask_used"
            ),
            "primitive_count": prior.get("primitive_count"),
            "ply_bytes": source_ply.stat().st_size if source_ply.is_file() else None,
            "training_seconds": protocol.get("training_seconds"),
        }
    if quality:
        quality_summary = quality.get("summary", quality)
        result["heldout_rgb_quality"] = {
            "query_count": quality_summary.get("query_count"),
            "psnr_db_mean": quality_summary.get("psnr_db", {}).get("mean"),
            "ssim_mean": quality_summary.get("ssim", {}).get("mean"),
            "lpips_mean": quality_summary.get("lpips", {}).get("mean"),
        }
    if bootstrap:
        mvinit = bootstrap.get("mvinit", {})
        result["bootstrap"] = {
            "landmark_count": mvinit.get("observed_landmarks"),
            "observed_landmarks": mvinit.get("observed_landmarks"),
            "observation_count_mean": mvinit.get("observation_count_mean"),
            "retained_observation_fraction": mvinit.get(
                "retained_observation_fraction"
            ),
        }
    if statistics:
        landmark = statistics.get("landmark_statistics", {})
        result["track_first"] = {
            "candidate_track_count": landmark.get("track_count"),
            "track_observation_count": landmark.get("track_observation_count"),
            "triangulated_track_count": landmark.get(
                "geometry_teacher_triangulated_track_count"
            ),
            "high_confidence_track_count": landmark.get(
                "geometry_teacher_high_confidence_track_count"
            ),
        }
    if compact:
        result["topology_distillation"] = {
            "track_core_count": compact.get("track_count"),
            "coverage_reserve_count": compact.get("base_reserve_count"),
            "compact_anchor_count": compact.get("total_count"),
            "unmet_mapping_query_count": compact.get("coverage", {}).get(
                "unmet_query_count"
            ),
        }
    if pose_reserve:
        result.setdefault("topology_distillation", {})[
            "final_anchor_count"
        ] = pose_reserve.get("anchor_count")
        result["topology_distillation"]["pose_reserve_count"] = pose_reserve.get(
            "pose_reserve_count"
        )
    provenance_path = (
        lafgs_root / "self_localization_reconstruction" / "raster_provenance.pt"
    )
    if provenance_path.is_file():
        result["raster_provenance"] = _raster_provenance(provenance_path)
    if frozen:
        result["deployment_bytes"] = frozen.get("deployment_total_bytes")
        result["A0_bootstrap"] = _aggregate_metrics(frozen, "A0_bootstrap")
        result["A1_reconstructed"] = _aggregate_metrics(
            frozen, "A1_reconstructed"
        )
        result["tail_diagnostics"] = {
            stage: diagnostics
            for stage in ("A0_bootstrap", "A1_reconstructed")
            if (diagnostics := _tail_diagnostics(frozen, stage)) is not None
        }
    return result


def summarize_enhanced_matcha_profile(
    enhanced_root: Path,
    quality_root: Path,
    scene: str,
) -> dict[str, Any]:
    """Summarize the historical mask-assisted MAtCha reference honestly.

    This reference does not use the mapping-only official-prior protocol: the
    audit records a selected support-view subset.  Keeping it separate avoids
    presenting it as a same-input official 2DGS control.
    """
    scene_root = enhanced_root / scene
    manifest = _read_json(
        scene_root / "prior" / "rgb_matcha_2dgs" / "rgb_prior_manifest.json"
    )
    audit = _read_json(scene_root / "audit" / "matcha_protocol.json")
    quality = _read_json(quality_root / f"{scene}.json")
    frozen = _read_json(scene_root / "frozen_results.json")
    scene_audit = (audit or {}).get("scenes", {}).get(scene, {})
    statistics_root = (
        scene_root
        / "runs"
        / "frozen_v1"
        / "statistics_combined_1000_frozen_g3_track_provenance_v1"
    )
    statistics = _read_json(statistics_root / "training_summary.json")
    compact = _first_map(
        _read_json(scene_root / "compact_map" / "minimum_sufficient_build.json")
    )
    pose_reserve = _first_map(
        _read_json(scene_root / "pose_reserve" / "pose_sufficient_build.json")
    )
    result: dict[str, Any] = {
        "scene": scene,
        "profile": "enhanced_matcha_2dgs",
        "complete": bool(manifest and audit and quality and frozen),
        "protocol_role": "mask_assisted_selected_view_reference",
    }
    if manifest:
        ply = Path(manifest.get("source_ply", ""))
        result["prior"] = {
            "official_repository": "MAtCha",
            "official_commit": None,
            "mapping_image_count": scene_audit.get("selected_camera_count"),
            "available_mapping_image_count": scene_audit.get(
                "dataset_training_camera_count"
            ),
            "test_rgb_used": False,
            "semantic_mask_used": True,
            "uses_full_mapping_split": scene_audit.get(
                "uses_full_cambridge_training_split"
            ),
            "primitive_count": manifest.get("primitive_count"),
            "ply_bytes": ply.stat().st_size if ply.is_file() else None,
            "training_seconds": None,
        }
    if quality:
        quality_summary = quality.get("summary", quality)
        result["heldout_rgb_quality"] = {
            "query_count": quality_summary.get("query_count"),
            "psnr_db_mean": quality_summary.get("psnr_db", {}).get("mean"),
            "ssim_mean": quality_summary.get("ssim", {}).get("mean"),
            "lpips_mean": quality_summary.get("lpips", {}).get("mean"),
        }
    if statistics:
        landmark = statistics.get("landmark_statistics", {})
        result["track_first"] = {
            "candidate_track_count": landmark.get("track_count"),
            "track_observation_count": landmark.get("track_observation_count"),
            "triangulated_track_count": landmark.get(
                "geometry_teacher_triangulated_track_count"
            ),
            "high_confidence_track_count": landmark.get(
                "geometry_teacher_high_confidence_track_count"
            ),
        }
    if compact:
        result["topology_distillation"] = {
            "track_core_count": compact.get("track_count"),
            "coverage_reserve_count": compact.get("base_reserve_count"),
            "compact_anchor_count": compact.get("total_count"),
            "unmet_mapping_query_count": compact.get("coverage", {}).get(
                "unmet_query_count"
            ),
        }
    if pose_reserve:
        result.setdefault("topology_distillation", {})[
            "final_anchor_count"
        ] = pose_reserve.get("anchor_count")
        result["topology_distillation"]["pose_reserve_count"] = pose_reserve.get(
            "pose_reserve_count"
        )
    provenance = scene_root / "self_localization_reconstruction" / "raster_provenance.pt"
    if provenance.is_file():
        result["raster_provenance"] = _raster_provenance(provenance)
    if frozen:
        result["deployment_bytes"] = frozen.get("deployment_total_bytes")
        result["A0_bootstrap"] = _aggregate_metrics(frozen, "A0_bootstrap")
        result["A1_reconstructed"] = _aggregate_metrics(
            frozen, "A1_reconstructed"
        )
        result["tail_diagnostics"] = {
            stage: diagnostics
            for stage in ("A0_bootstrap", "A1_reconstructed")
            if (diagnostics := _tail_diagnostics(frozen, stage)) is not None
        }
    return result


def _mean(record: dict[str, Any], stage: str, key: str) -> float | None:
    stage_record = record.get(stage) or {}
    value = stage_record.get(key)
    return value.get("mean") if isinstance(value, dict) else None


def _format(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def render_markdown(records: list[dict[str, Any]]) -> str:
    lines = [
        "# LaFGS Off-the-Shelf Prior Robustness",
        "",
        "| Scene | Prior | Prior views | Mask | Primitives | PSNR | Tracks (tri.) | Anchors | A0 Med/Mean/P90 cm | A1 Med/Mean/P90 cm | Raw P@2 A0->A1 | Inlier P@2 A0->A1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        prior = record.get("prior", {})
        quality = record.get("heldout_rgb_quality", {})
        tracks = record.get("track_first", {})
        topology = record.get("topology_distillation", {})
        a0_pose = "/".join(
            _format(_mean(record, "A0_bootstrap", key))
            for key in ("median_te_cm", "mean_te_cm", "p90_te_cm")
        )
        a1_pose = "/".join(
            _format(_mean(record, "A1_reconstructed", key))
            for key in ("median_te_cm", "mean_te_cm", "p90_te_cm")
        )
        raw = (
            f"{_format(_mean(record, 'A0_bootstrap', 'raw_gt_precision_2px_percent'))}"
            f"->{_format(_mean(record, 'A1_reconstructed', 'raw_gt_precision_2px_percent'))}"
        )
        inlier = (
            f"{_format(_mean(record, 'A0_bootstrap', 'inlier_gt_precision_2px_percent'))}"
            f"->{_format(_mean(record, 'A1_reconstructed', 'inlier_gt_precision_2px_percent'))}"
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    record["scene"],
                    record["profile"],
                    _format(prior.get("mapping_image_count")),
                    "yes" if prior.get("semantic_mask_used") else "no",
                    _format(prior.get("primitive_count")),
                    _format(quality.get("psnr_db_mean")),
                    _format(tracks.get("triangulated_track_count")),
                    _format(topology.get("final_anchor_count")),
                    a0_pose,
                    a1_pose,
                    raw,
                    inlier,
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Tail Diagnostics",
            "",
            "A failure is persistent when TE exceeds 1 m for every recorded "
            "PoseLib seed; unstable means it exceeds 1 m for only some seeds.",
            "",
            "| Scene | Prior | A0 persistent/unstable | A1 persistent/unstable |",
            "|---|---:|---:|---:|",
        ]
    )
    for record in records:
        tail = record.get("tail_diagnostics", {})
        values = []
        for stage in ("A0_bootstrap", "A1_reconstructed"):
            diagnostics = tail.get(stage, {})
            values.append(
                f"{diagnostics.get('persistent_failure_count', 0)}/"
                f"{diagnostics.get('seed_unstable_failure_count', 0)}"
            )
        lines.append(
            f"| {record['scene']} | {record['profile']} | "
            f"{values[0]} | {values[1]} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--lafgs-namespace", default="lafgs_strict_v2")
    parser.add_argument("--enhanced-root", type=Path)
    parser.add_argument("--enhanced-quality-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = [
        summarize_profile(
            args.root,
            scene,
            profile,
            lafgs_namespace=args.lafgs_namespace,
        )
        for scene in args.scenes
        for profile in args.profiles
    ]
    if bool(args.enhanced_root) != bool(args.enhanced_quality_root):
        parser.error(
            "--enhanced-root and --enhanced-quality-root must be supplied together"
        )
    if args.enhanced_root:
        records.extend(
            summarize_enhanced_matcha_profile(
                args.enhanced_root,
                args.enhanced_quality_root,
                scene,
            )
            for scene in args.scenes
        )
    output = args.output or args.root / "prior_robustness_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"records": records}, indent=2) + "\n")
    markdown = output.with_suffix(".md")
    markdown.write_text(render_markdown(records), encoding="utf-8")
    print(json.dumps({"json": str(output), "markdown": str(markdown)}, indent=2))


if __name__ == "__main__":
    main()
