#!/usr/bin/env python3
"""Rebuild only native sparse mapping observations at an explicit K/NMS.

Dense SuperPoint maps and frozen-prior raster fields are copied from a source
cache.  The detector is rerun on every mapping image, so a legacy cache with
unattested NMS cannot silently become an attested factor arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from data.datasets import ColmapDataset
from features.extractor import FeatureExtractor
from features.multiview_fusion import sample_mask_at_grid_uv


def refreshed_signature_payload(
    source: dict, *, mapping_keypoints: int, nms_radius: int
) -> tuple[str, dict]:
    payload = dict(source)
    payload["native_sparse_enabled"] = True
    payload["native_sparse_keypoint_count"] = int(mapping_keypoints)
    payload["native_sparse_nms_radius"] = int(nms_radius)
    payload["version"] = max(int(payload.get("version", 0)), 11)
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def update_sparse_record(
    record: dict,
    *,
    sparse: dict,
    valid_mask: torch.Tensor,
    mapping_keypoints: int,
    nms_radius: int,
) -> dict:
    output = dict(record)
    keypoints = torch.as_tensor(sparse["keypoints"]).detach()
    descriptors = F.normalize(
        torch.as_tensor(sparse["descriptors"]).detach().float(), dim=1
    )
    scores = torch.as_tensor(sparse["keypoint_scores"]).detach()
    keep = sample_mask_at_grid_uv(valid_mask, keypoints)
    output.update(
        {
            "native_keypoints": keypoints[keep].cpu(),
            "native_descriptors": descriptors[keep].to(
                device="cpu", dtype=torch.float16
            ),
            "native_scores": scores[keep].to(device="cpu", dtype=torch.float16),
            "native_valid_mask": valid_mask.cpu(),
            "native_input_hw": [int(valid_mask.shape[0]), int(valid_mask.shape[1])],
            "native_sparse_metadata": {
                "detect_and_compute": True,
                "detect_num": int(mapping_keypoints),
                "requested_keypoint_count": int(mapping_keypoints),
                "nms_radius": int(nms_radius),
                "keypoint_count_before_mask": int(keypoints.shape[0]),
                "keypoint_count_after_mask": int(keep.sum()),
                "coordinate_convention": (
                    "superpoint_grid_index_then_pnp_plus_half_v1"
                ),
            },
        }
    )
    # Any sampled-at-keypoint raster sidecar would refer to old keypoint rows.
    output.pop("native_depth_at_keypoints", None)
    output.pop("native_alpha_at_keypoints", None)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mapping-keypoints", type=int, choices=(1024, 2048), required=True
    )
    parser.add_argument("--nms-radius", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    source = torch.load(args.source_cache, map_location="cpu", weights_only=False)
    records = source.get("queries", source)
    signature_payload = source.get("signature_payload")
    if not isinstance(records, dict) or not isinstance(signature_payload, dict):
        raise ValueError("source cache must contain queries and signature_payload")
    dataset = ColmapDataset(args.dataset, images=str(signature_payload["images"]))
    mapping = dataset.split("mapping")
    mapping_names = [camera.image_name for camera in mapping]
    if set(records) != set(mapping_names):
        raise ValueError("source cache is not the exact mapping-only query set")
    if str(args.dataset.resolve()) != str(signature_payload["source_path"]):
        raise ValueError("dataset path differs from source cache signature")
    if int(signature_payload.get("longest_edge", 0)) != 0:
        raise ValueError("sparse-only refresh currently requires native image scale")
    extractor = (
        FeatureExtractor(
            str(signature_payload["feature_type"]), nms_radius=args.nms_radius
        )
        .to(args.device)
        .eval()
    )
    refreshed = {}
    with torch.no_grad():
        for index, camera in enumerate(mapping):
            image = dataset.load_image(camera).to(args.device)
            valid = dataset.valid_mask(camera)
            if valid is None:
                valid = torch.ones(
                    image.shape[-2:], dtype=torch.bool, device=args.device
                )
            else:
                valid = valid.to(args.device)
            masked = image * valid[None].to(dtype=image.dtype)
            sparse = extractor.detectAndCompute(
                masked[None], top_k=int(args.mapping_keypoints)
            )[0]
            refreshed[camera.image_name] = update_sparse_record(
                records[camera.image_name],
                sparse=sparse,
                valid_mask=valid,
                mapping_keypoints=args.mapping_keypoints,
                nms_radius=args.nms_radius,
            )
            if (index + 1) % 100 == 0 or index + 1 == len(mapping):
                print(f"mapping sparse refresh {index + 1}/{len(mapping)}", flush=True)
    signature, new_signature_payload = refreshed_signature_payload(
        signature_payload,
        mapping_keypoints=args.mapping_keypoints,
        nms_radius=args.nms_radius,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    torch.save(
        {
            "version": max(int(source.get("version", 0)), 3),
            "signature": signature,
            "signature_payload": new_signature_payload,
            "factor_provenance": {
                "schema": "lafgs_mapping_sparse_refresh",
                "version": 1,
                "uses_test_queries": False,
                "source_cache": str(args.source_cache.resolve()),
                "detector_rerun_for_every_mapping_query": True,
                "preserved_dense_and_raster_fields": True,
                "mapping_keypoints": int(args.mapping_keypoints),
                "nms_radius": int(args.nms_radius),
            },
            "queries": refreshed,
        },
        temporary,
    )
    os.replace(temporary, output)
    print(
        json.dumps(
            {"output": str(output), "signature": signature, "queries": len(refreshed)}
        )
    )


if __name__ == "__main__":
    main()
