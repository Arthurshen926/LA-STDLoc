#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema") != "lafgs_materialized_anchor_map":
        raise ValueError(f"Unsupported anchor map: {path}")
    return state


def _project(xyz, K, pose):
    camera = xyz @ pose[:3, :3].T + pose[:3, 3]
    homogeneous = camera @ K.T
    return (
        homogeneous[:, :2] / camera[:, 2:].clamp_min(1e-8),
        camera[:, 2],
    )


def _align_visibility_to_map(visibility, state):
    base_count = int(state["base_anchor_count"])
    sources = torch.as_tensor(
        state["source_primitive_ids"], dtype=torch.long
    )
    base_sources = sources[:base_count]
    source_to_row = {
        int(source): row for row, source in enumerate(base_sources.tolist())
    }
    lookup = torch.as_tensor(
        [source_to_row[int(source)] for source in sources.tolist()],
        dtype=torch.long,
    )
    aligned = {}
    for name, value in visibility.items():
        mask = torch.as_tensor(value, dtype=torch.bool).reshape(-1)
        if mask.numel() == sources.numel():
            aligned[name] = mask
        elif mask.numel() == base_count:
            aligned[name] = mask[lookup]
        else:
            raise ValueError(
                f"visibility rows for {name} do not align with canonical map"
            )
    return aligned


def _append(base, extension, extension_rows, *, profile, provenance, audit):
    rows = torch.as_tensor(extension_rows, dtype=torch.long)
    base_rows = int(base["anchor_ids"].numel())
    output = {
        "version": 2,
        "schema": "lafgs_materialized_anchor_map",
        "anchor_ids": torch.arange(base_rows + rows.numel()),
        "source_primitive_ids": torch.cat(
            (base["source_primitive_ids"], extension["source_primitive_ids"][rows])
        ),
        "track_cluster_ids": torch.cat(
            (base["track_cluster_ids"], extension["track_cluster_ids"][rows])
        ),
        "anchor_xyz": torch.cat(
            (base["anchor_xyz"], extension["anchor_xyz"][rows])
        ),
        "anchor_features": torch.cat(
            (base["anchor_features"], extension["anchor_features"][rows])
        ),
        "anchor_type": torch.cat(
            (base["anchor_type"], extension["anchor_type"][rows])
        ),
        "base_anchor_count": int(base["base_anchor_count"]),
        "canonical_anchor_count": int(base_rows),
        "requested_micro_anchor_budget": int(
            base["micro_anchor_count"] + rows.numel()
        ),
        "micro_anchor_count": int(base["micro_anchor_count"] + rows.numel()),
        "config": {
            "method": "query_counterfactual_extension_acceptance_v1",
            "profile": profile,
            "canonical_rows_frozen": True,
            "accepted_extension_count": int(rows.numel()),
        },
        "counterfactual_audit": audit,
        "provenance": provenance,
    }
    return output


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-map", required=True)
    parser.add_argument("--extension-map", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--visibility-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query-rows", type=int, default=256)
    parser.add_argument("--depth-abs-tolerance-m", type=float, default=0.05)
    parser.add_argument("--depth-rel-tolerance", type=float, default=0.02)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Counterfactual extension audit requires CUDA")
    device = torch.device("cuda")
    canonical_path = Path(args.canonical_map).resolve()
    extension_path = Path(args.extension_map).resolve()
    query_path = Path(args.query_cache).resolve()
    payload_path = Path(args.track_payload).resolve()
    visibility_path = Path(args.visibility_cache).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = _load(canonical_path)
    extension = _load(extension_path)
    canonical_rows = int(canonical["anchor_ids"].numel())
    if extension["anchor_ids"].numel() <= canonical_rows:
        raise ValueError("extension map has no appended rows")
    extension_rows = torch.arange(
        canonical_rows, extension["anchor_ids"].numel(), dtype=torch.long
    )
    candidate_xyz = extension["anchor_xyz"][extension_rows].float().to(device)
    candidate_features = F.normalize(
        extension["anchor_features"][extension_rows].float(), dim=1
    ).to(device)
    base_xyz = canonical["anchor_xyz"].float().to(device)
    base_features = F.normalize(
        canonical["anchor_features"].float(), dim=1
    ).to(device)
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    visibility_payload = torch.load(
        visibility_path, map_location="cpu", weights_only=False
    )
    visibility = _align_visibility_to_map(
        visibility_payload.get("visibility", visibility_payload), canonical
    )
    query_bins = torch.as_tensor(payload["query_bins"], dtype=torch.long)
    query_name_to_bin = {
        name: int(query_bins[index])
        for index, name in enumerate(payload["query_names"])
    }
    print(f"Loading query cache: {query_path}", flush=True)
    query_payload = torch.load(
        query_path, map_location="cpu", weights_only=False
    )
    query_cache = query_payload.get("queries", query_payload)
    candidate_count = int(candidate_features.shape[0])
    statistics = {
        name: torch.zeros(candidate_count, dtype=torch.long)
        for name in (
            "switch",
            "rescue2",
            "rescue4",
            "harmful2",
            "harmful4",
            "both_clean2",
            "dirty_to_dirty",
        )
    }
    rescue_queries = [set() for _ in range(candidate_count)]
    harmful_queries = [set() for _ in range(candidate_count)]
    rescue_bins = [set() for _ in range(candidate_count)]
    harmful_bins = [set() for _ in range(candidate_count)]

    for query_index, name in enumerate(payload["query_names"]):
        cached = query_cache[name]
        descriptors = F.normalize(
            torch.as_tensor(cached["native_descriptors"]).float(), dim=1
        )
        scores = torch.as_tensor(cached["native_scores"]).float()
        keep = min(int(args.query_rows), descriptors.shape[0])
        selected = torch.topk(scores, k=keep, sorted=False).indices
        descriptors = descriptors[selected].to(device)
        base_score, base_index = (descriptors @ base_features.T).max(dim=1)
        candidate_score, candidate_index = (
            descriptors @ candidate_features.T
        ).max(dim=1)
        switched = candidate_score > base_score
        if not bool(switched.any()):
            continue
        selected = selected[switched.cpu()]
        candidate_index = candidate_index[switched]
        base_index = base_index[switched]
        keypoints = (
            torch.as_tensor(cached["native_keypoints"]).float()[selected]
            + float(cached.get("pixel_center_offset", 0.5))
        ).to(device)
        native_keypoints = torch.as_tensor(
            cached["native_keypoints"]
        ).float()[selected]
        native_depth = torch.as_tensor(cached["native_depth"]).float()
        x = native_keypoints[:, 0].round().long().clamp(
            0, native_depth.shape[1] - 1
        )
        y = native_keypoints[:, 1].round().long().clamp(
            0, native_depth.shape[0] - 1
        )
        reference_depth = native_depth[y, x].to(device)
        depth_tolerance = float(args.depth_abs_tolerance_m) + (
            float(args.depth_rel_tolerance) * reference_depth.abs()
        )
        valid_reference_depth = (
            torch.isfinite(reference_depth) & (reference_depth > 0)
        )
        K = torch.as_tensor(cached["native_K"]).float().to(device)
        pose = torch.as_tensor(cached["pose_w2c"]).float().to(device)
        candidate_uv, candidate_depth = _project(
            candidate_xyz[candidate_index], K, pose
        )
        base_uv, base_depth = _project(base_xyz[base_index], K, pose)
        candidate_error = torch.linalg.norm(candidate_uv - keypoints, dim=1)
        base_error = torch.linalg.norm(base_uv - keypoints, dim=1)
        candidate_depth_clean = (
            valid_reference_depth
            & (candidate_depth > 0)
            & (
                (candidate_depth - reference_depth).abs()
                <= depth_tolerance
            )
        )
        base_visible = torch.as_tensor(
            visibility[name], dtype=torch.bool
        )[base_index.cpu()].to(device)
        base_depth_clean = (
            valid_reference_depth
            & base_visible
            & (base_depth > 0)
            & ((base_depth - reference_depth).abs() <= depth_tolerance)
        )
        candidate_clean2 = candidate_depth_clean & (candidate_error <= 2.0)
        candidate_clean4 = candidate_depth_clean & (candidate_error <= 4.0)
        base_clean2 = base_depth_clean & (base_error <= 2.0)
        base_clean4 = base_depth_clean & (base_error <= 4.0)
        for local, candidate in enumerate(candidate_index.tolist()):
            statistics["switch"][candidate] += 1
            rescue2 = bool(candidate_clean2[local] and not base_clean2[local])
            rescue4 = bool(candidate_clean4[local] and not base_clean4[local])
            harmful2 = bool(base_clean2[local] and not candidate_clean2[local])
            harmful4 = bool(base_clean4[local] and not candidate_clean4[local])
            statistics["rescue2"][candidate] += int(rescue2)
            statistics["rescue4"][candidate] += int(rescue4)
            statistics["harmful2"][candidate] += int(harmful2)
            statistics["harmful4"][candidate] += int(harmful4)
            statistics["both_clean2"][candidate] += int(
                bool(candidate_clean2[local] and base_clean2[local])
            )
            statistics["dirty_to_dirty"][candidate] += int(
                bool(not candidate_clean4[local] and not base_clean4[local])
            )
            if rescue2:
                rescue_queries[candidate].add(name)
                rescue_bins[candidate].add(query_name_to_bin[name])
            if harmful2:
                harmful_queries[candidate].add(name)
                harmful_bins[candidate].add(query_name_to_bin[name])

    profile_masks = {
        "strict": (
            (statistics["rescue2"] >= 2)
            & (statistics["harmful2"] == 0)
            & torch.as_tensor([len(value) for value in rescue_queries]).ge(2)
        ),
        "protected_ratio": (
            (statistics["rescue2"] >= 3)
            & (
                statistics["rescue2"]
                >= 3 * statistics["harmful2"] + 2
            )
            & (
                torch.as_tensor([len(value) for value in rescue_queries])
                >= 2
            )
            & (
                torch.as_tensor([len(value) for value in harmful_queries])
                <= 1
            )
        ),
    }
    track_ids = extension["track_cluster_ids"][extension_rows]
    rows_report = []
    for candidate in range(candidate_count):
        rows_report.append(
            {
                "candidate": candidate,
                "extension_row": int(extension_rows[candidate]),
                "track_cluster_id": int(track_ids[candidate]),
                **{
                    name: int(value[candidate])
                    for name, value in statistics.items()
                },
                "rescue_query_count": len(rescue_queries[candidate]),
                "harmful_query_count": len(harmful_queries[candidate]),
                "rescue_view_bin_count": len(rescue_bins[candidate]),
                "harmful_view_bin_count": len(harmful_bins[candidate]),
            }
        )
    provenance = {
        "canonical_map_path": str(canonical_path),
        "canonical_map_sha256": _sha256(canonical_path),
        "extension_map_path": str(extension_path),
        "extension_map_sha256": _sha256(extension_path),
        "query_cache_path": str(query_path),
        "query_cache_signature": query_payload.get("signature"),
        "track_payload_path": str(payload_path),
        "track_payload_sha256": _sha256(payload_path),
        "visibility_cache_path": str(visibility_path),
        "visibility_cache_sha256": _sha256(visibility_path),
        "query_rows": int(args.query_rows),
        "depth_abs_tolerance_m": float(args.depth_abs_tolerance_m),
        "depth_rel_tolerance": float(args.depth_rel_tolerance),
        "statistics_split": "all_895_mapping_train",
    }
    summary = {"profiles": {}, "candidate_rows": rows_report}
    for profile, mask in profile_masks.items():
        selected_candidates = torch.nonzero(
            mask, as_tuple=False
        ).reshape(-1)
        selected_rows = extension_rows[selected_candidates]
        audit = {
            "profile": profile,
            "candidate_count": candidate_count,
            "accepted_count": int(selected_rows.numel()),
            "accepted_track_cluster_ids": track_ids[
                selected_candidates
            ].tolist(),
            "accepted_rescue2": int(
                statistics["rescue2"][selected_candidates].sum()
            ),
            "accepted_harmful2": int(
                statistics["harmful2"][selected_candidates].sum()
            ),
            "accepted_switches": int(
                statistics["switch"][selected_candidates].sum()
            ),
        }
        state = _append(
            canonical,
            extension,
            selected_rows,
            profile=profile,
            provenance=provenance,
            audit=audit,
        )
        output_path = output_dir / f"identity_{profile}.pt"
        torch.save(state, output_path)
        summary["profiles"][profile] = {
            **audit,
            "path": str(output_path),
        }
    (output_dir / "counterfactual_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["profiles"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
