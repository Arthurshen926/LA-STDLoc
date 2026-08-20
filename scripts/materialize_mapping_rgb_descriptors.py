#!/usr/bin/env python3
"""Re-describe a frozen projective Anchor map from source mapping RGB.

Only the descriptor resource changes.  Anchor rows, identities, xyz, selection,
and the projective observation CSR remain bitwise identical to the input map.
Test images are never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from data.datasets import ColmapDataset
from evidence.tracks import (
    fuse_projective_anchor_observations,
    fuse_track_descriptors,
)
from features.extractor import FeatureExtractor
from features.superpoint import (
    SUPERPOINT_WEIGHT_SHA256,
    resolve_superpoint_weights,
    sample_descriptors,
)
from map_learning.metric import SharedLowRankMetric


TOPOLOGY_FIELDS = (
    "anchor_ids",
    "source_primitive_ids",
    "track_cluster_ids",
    "anchor_xyz",
    "anchor_type",
    "dependency_group_ids",
    "coarse_dependency_group_ids",
    "fine_identity_ids",
    "source_dependency_group_ids",
    "parent_source_track_ids",
    "repair_child_index",
    "repair_parent_child_count",
    "anchor_parent_identity_ids",
    "anchor_correlation_group_ids",
    "anchor_candidate_kind",
)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_frozen_inputs(state: dict, payload: dict, cache: dict) -> tuple[list[str], torch.Tensor]:
    if cache.get("uses_source_mapping_rgb") is not False or cache.get("uses_test_queries") is not False:
        raise ValueError("source cache must be mapping-only Gaussian-render evidence")
    if payload.get("rendered_rgb_only") is not True:
        raise ValueError("Track payload must be rendered-RGB-only")
    names = list(payload["query_names"])
    if names != list(cache["queries"]):
        raise ValueError("source cache and Track payload query order differ")
    observations = state.get("projective_anchor_observations", {})
    if observations.get("schema") != "lafgs_projective_anchor_observations":
        raise ValueError("map lacks the frozen projective observation CSR")
    count = int(torch.as_tensor(state["anchor_ids"]).numel())
    offsets = torch.as_tensor(observations["observation_offsets"])
    queries = torch.as_tensor(observations["query_indices"])
    keypoints = torch.as_tensor(observations["keypoint_indices"])
    if offsets.dtype != torch.long or offsets.shape != (count + 1,):
        raise ValueError("observation offsets must be int64 [N+1]")
    if int(offsets[0]) != 0 or bool((offsets[1:] <= offsets[:-1]).any()):
        raise ValueError("every frozen Anchor row must have observations")
    edge_count = int(offsets[-1])
    if queries.dtype != torch.long or keypoints.dtype != torch.long or queries.shape != (edge_count,) or keypoints.shape != (edge_count,):
        raise ValueError("observation CSR edge arrays differ")
    if queries.numel() and (int(queries.min()) < 0 or int(queries.max()) >= len(names)):
        raise ValueError("observation query is outside mapping schedule")
    for query in torch.unique(queries).tolist():
        name = names[int(query)]
        maximum = int(keypoints[queries == int(query)].max())
        if maximum >= int(torch.as_tensor(cache["queries"][name]["native_keypoints"]).shape[0]):
            raise ValueError(f"observation keypoint is outside frozen rows for {name}")
    return names, torch.as_tensor(cache["source_mapping_indices"]).long()


def fuse_frozen_rows(state: dict, payload: dict, cache: dict, *, trim_fraction: float) -> torch.Tensor:
    """Fuse all frozen rows, retaining historical Track and surface weights."""
    track_ids = torch.as_tensor(state["track_cluster_ids"]).long()
    track_rows = torch.nonzero(track_ids >= 0, as_tuple=False).reshape(-1)
    output = torch.empty_like(torch.as_tensor(state["anchor_features"]).float())
    if track_rows.numel():
        output[track_rows] = fuse_track_descriptors(
            payload=payload,
            query_cache=cache,
            track_indices=track_ids[track_rows],
            trim_fraction=trim_fraction,
        )
    observations = state["projective_anchor_observations"]
    offsets = torch.as_tensor(observations["observation_offsets"]).long()
    queries = torch.as_tensor(observations["query_indices"]).long()
    keypoints = torch.as_tensor(observations["keypoint_indices"]).long()
    names = list(payload["query_names"])
    query_bins = torch.as_tensor(payload["query_bins"]).long()
    surface_rows = torch.nonzero(track_ids < 0, as_tuple=False).reshape(-1)
    for row in surface_rows.tolist():
        start, end = int(offsets[row]), int(offsets[row + 1])
        q = queries[start:end]
        k = keypoints[start:end]
        records = [cache["queries"][names[int(index)]] for index in q.tolist()]
        descriptors = torch.stack([
            torch.as_tensor(record["native_descriptors"])[int(keypoint)]
            for record, keypoint in zip(records, k.tolist())
        ])
        detector = torch.stack([
            torch.as_tensor(record["native_scores"])[int(keypoint)]
            for record, keypoint in zip(records, k.tolist())
        ]).float()
        alpha = torch.stack([
            torch.as_tensor(record["native_alpha_at_keypoints"])[int(keypoint)]
            for record, keypoint in zip(records, k.tolist())
        ]).float()
        output[row] = fuse_projective_anchor_observations(
            F.normalize(descriptors.float(), dim=1),
            query_bins[q],
            detector_weight=detector.clamp_min(0),
            visibility_weight=alpha.clamp(0, 1),
            trim_fraction=trim_fraction,
        )
    return output


def _manifest_digest(rows: list[dict]) -> str:
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@torch.inference_mode()
def materialize(args) -> dict:
    started = time.perf_counter()
    torch.set_num_threads(int(args.cpu_threads))
    source_map = torch.load(args.selected_map, map_location="cpu", weights_only=False)
    payload = torch.load(args.track_payload, map_location="cpu", weights_only=False)
    source_cache = torch.load(args.source_cache, map_location="cpu", weights_only=False)
    names, mapping_indices = validate_frozen_inputs(source_map, payload, source_cache)

    # This closes the fusion-contract loop before any resource is changed.
    replay = fuse_frozen_rows(source_map, payload, source_cache, trim_fraction=args.descriptor_trim_fraction)
    if not torch.equal(replay, torch.as_tensor(source_map["anchor_features"]).float()):
        maximum = float((replay - torch.as_tensor(source_map["anchor_features"]).float()).abs().max())
        raise ValueError(f"frozen source fusion is not bitwise exact (maximum {maximum})")

    dataset = ColmapDataset(args.dataset, images=args.images)
    mapping = dataset.split("mapping")
    cameras = [mapping[int(index)] for index in mapping_indices]
    if names != [camera.image_name for camera in cameras]:
        raise ValueError("mapping RGB camera schedule differs from frozen evidence")
    extractor = FeatureExtractor("sp", nms_radius=args.nms_radius).cuda().eval()
    extractor.requires_grad_(False)
    records = {}
    image_manifest = []
    for index, camera in enumerate(cameras):
        source = source_cache["queries"][camera.image_name]
        if [camera.height, camera.width] != torch.as_tensor(source["native_input_hw"]).long().tolist():
            raise ValueError(f"mapping image geometry differs for {camera.image_name}")
        image = dataset.load_image(camera).cuda()
        dense, _ = extractor.detectAndComputeDense(image[None])
        keypoints = torch.as_tensor(source["native_keypoints"]).float().cuda()
        descriptors = sample_descriptors(keypoints[None], dense)[0].transpose(0, 1).cpu()
        records[camera.image_name] = {**source, "native_descriptors": descriptors, "source": "source_mapping_rgb_fixed_render_keypoints"}
        image_manifest.append({
            "image_name": camera.image_name,
            "relative_path": camera.image_path.relative_to(args.dataset).as_posix(),
            "sha256": sha256_file(camera.image_path),
            "height": camera.height,
            "width": camera.width,
        })
        if (index + 1) % max(args.progress_interval, 1) == 0 or index + 1 == len(cameras):
            print(json.dumps({"completed_views": index + 1, "mapping_views": len(cameras)}), flush=True)
    rgb_cache = {**source_cache, "schema": "lafgs_mapping_rgb_fixed_row_descriptor_cache", "version": 1, "uses_source_mapping_rgb": True, "uses_test_queries": False, "queries": records}
    features = fuse_frozen_rows(source_map, payload, rgb_cache, trim_fraction=args.descriptor_trim_fraction)
    output_map = dict(source_map)
    output_map["anchor_features"] = features
    output_map["v7_metric_raw_features"] = features.clone()
    lineage = {
        "schema": "lafgs_mapping_rgb_descriptor_rematerialization",
        "version": 1,
        "uses_source_mapping_rgb": True,
        "uses_test_queries": False,
        "descriptor_only": True,
        "fixed_anchor_rows_identity_xyz_selection": True,
        "fixed_render_keypoints": True,
        "image_manifest_sha256": _manifest_digest(image_manifest),
        "pixel_contract": "PIL RGB/gray float32 /255; bilinear resize to COLMAP HxW align_corners=false; frozen full-resolution keypoint index; sample_descriptors pixel_center=index+0.5 stride=8 align_corners=false",
        "nms_radius": int(args.nms_radius),
        "superpoint_weight_sha256": SUPERPOINT_WEIGHT_SHA256,
    }
    output_map["provenance"] = {**source_map.get("provenance", {}), "mapping_rgb_descriptor_rematerialization": lineage}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    map_path = args.output_dir / "mapping_rgb_descriptor_anchor_map.pt"
    metric_path = args.output_dir / "mapping_rgb_descriptor_identity_metric.pt"
    _atomic_torch_save(output_map, map_path)
    metric = SharedLowRankMetric(descriptor_dim=features.shape[1], rank=1, max_residual_norm=0.0)
    with torch.no_grad():
        for parameter in metric.parameters():
            parameter.zero_()
    metric_state = {
        "schema": "lafgs_shared_metric_state", "version": 1,
        "landmark_indices": torch.arange(features.shape[0]).long(),
        "metric_config": metric.export_config(),
        "metric_state_dict": {name: value.detach().cpu().clone() for name, value in metric.state_dict().items()},
        "map_path": str(map_path.resolve()), "step": 0,
        "protocol": "mapping_rgb_descriptor_only_fixed_projective_anchor_map",
    }
    _atomic_torch_save(metric_state, metric_path)
    for field in TOPOLOGY_FIELDS:
        if field in source_map and not torch.equal(torch.as_tensor(source_map[field]), torch.as_tensor(output_map[field])):
            raise AssertionError(f"topology field changed: {field}")
    source_csr = source_map["projective_anchor_observations"]
    output_csr = output_map["projective_anchor_observations"]
    for field in ("observation_offsets", "query_indices", "keypoint_indices"):
        if not torch.equal(torch.as_tensor(source_csr[field]), torch.as_tensor(output_csr[field])):
            raise AssertionError(f"observation CSR changed: {field}")
    report = {
        **lineage,
        "mapping_query_count": len(cameras),
        "anchor_count": int(features.shape[0]),
        "track_anchor_count": int((torch.as_tensor(source_map["track_cluster_ids"]) >= 0).sum()),
        "surface_completion_anchor_count": int((torch.as_tensor(source_map["track_cluster_ids"]) < 0).sum()),
        "image_manifest": image_manifest,
        "inputs": {"dataset": str(args.dataset), "selected_map": str(args.selected_map), "source_cache": str(args.source_cache), "track_payload": str(args.track_payload)},
        "input_sha256": {"selected_map": sha256_file(args.selected_map), "source_cache": sha256_file(args.source_cache), "track_payload": sha256_file(args.track_payload)},
        "outputs": {"anchor_map": str(map_path), "identity_metric": str(metric_path)},
        "output_sha256": {"anchor_map": sha256_file(map_path), "identity_metric": sha256_file(metric_path)},
        "runtime_identity": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name()},
        "code_sha256": {"materializer": sha256_file(Path(__file__).resolve()), "dataset_loader": sha256_file(Path(__file__).resolve().parents[1] / "data/datasets.py"), "superpoint": sha256_file(Path(__file__).resolve().parents[1] / "features/superpoint.py"), "track_fusion": sha256_file(Path(__file__).resolve().parents[1] / "evidence/tracks.py")},
        "superpoint_weights_path": str(resolve_superpoint_weights()),
        "timing_seconds": {"total": time.perf_counter() - started},
    }
    _atomic_json(report, args.output_dir / "mapping_rgb_descriptor_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--track-payload", type=Path, required=True)
    parser.add_argument("--selected-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nms-radius", type=int, default=4)
    parser.add_argument("--descriptor-trim-fraction", type=float, default=0.2)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("mapping RGB descriptor materialization requires CUDA")
    if int(args.cpu_threads) < 1:
        raise ValueError("cpu-threads must be positive")
    for field in ("dataset", "source_cache", "track_payload", "selected_map", "output_dir"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    print(json.dumps(materialize(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
