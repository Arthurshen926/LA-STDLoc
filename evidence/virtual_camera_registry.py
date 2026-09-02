"""Canonical cross-stage registry for source-image-free virtual cameras."""

from __future__ import annotations

import hashlib
import json
from typing import Sequence

import torch


POLICY = "geometry_pose_intrinsics_v1"
DEDUPLICATED_POLICY = "geometry_pose_intrinsics_deduplicated_v1"
SCHEMA = "lafgs_virtual_camera_registry"


def camera_intrinsic(camera) -> torch.Tensor:
    fx = camera.width / (2.0 * torch.tan(torch.tensor(camera.fov_x / 2.0)))
    fy = camera.height / (2.0 * torch.tan(torch.tensor(camera.fov_y / 2.0)))
    return torch.tensor(
        [[float(fx), 0.0, camera.width / 2.0],
         [0.0, float(fy), camera.height / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def _geometry_key(camera) -> tuple:
    return (
        int(camera.height), int(camera.width), float(camera.fov_y),
        float(camera.fov_x),
        *torch.as_tensor(camera.pose_w2c, dtype=torch.float64).reshape(-1).tolist(),
    )


def _digest(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_virtual_camera_registry(
    cameras: Sequence,
    maximum_views: int = 0,
    *,
    deduplicate_geometry: bool = False,
):
    indexed = [(source_index, camera, _geometry_key(camera))
               for source_index, camera in enumerate(cameras)]
    keys = [row[2] for row in indexed]
    if len(keys) != len(set(keys)) and not deduplicate_geometry:
        raise ValueError("virtual-camera registry contains duplicate geometry")
    full_ordered = sorted(indexed, key=lambda row: (row[2], row[1].image_name))
    if deduplicate_geometry:
        ordered = []
        seen = set()
        for row in full_ordered:
            if row[2] not in seen:
                ordered.append(row)
                seen.add(row[2])
    else:
        ordered = full_ordered
    limit = int(maximum_views)
    if 0 < limit < len(ordered):
        selected_rows = torch.div(torch.arange(limit) * len(ordered), limit,
                                  rounding_mode="floor").tolist()
    else:
        selected_rows = list(range(len(ordered)))
    selected = [ordered[row] for row in selected_rows]
    body = {
        "schema": SCHEMA,
        "version": 1,
        "policy": DEDUPLICATED_POLICY if deduplicate_geometry else POLICY,
        "full_camera_names": [row[1].image_name for row in full_ordered],
        "unique_geometry_count": len(ordered),
        "duplicate_geometry_count": len(full_ordered) - len(ordered),
        "duplicate_geometry_keeper": (
            "lexicographically_first_image_name" if deduplicate_geometry else None
        ),
        "selected_camera_names": [row[1].image_name for row in selected],
        "selected_canonical_indices": selected_rows,
        "selected_legacy_dataset_indices": [row[0] for row in selected],
        "entries": [
            {
                "name": row[1].image_name,
                "image_hw": [int(row[1].height), int(row[1].width)],
                "pose_w2c": torch.as_tensor(row[1].pose_w2c).double().tolist(),
                "intrinsic": camera_intrinsic(row[1]).double().tolist(),
            }
            for row in selected
        ],
    }
    body["registry_sha256"] = _digest(body)
    return [row[1] for row in selected], body


def resolve_virtual_camera_registry(cameras: Sequence, registry: dict):
    if not isinstance(registry, dict):
        raise ValueError("virtual-camera registry is required")
    body = dict(registry)
    digest = body.pop("registry_sha256", None)
    if (
        body.get("schema") != SCHEMA
        or body.get("version") != 1
        or body.get("policy") not in (POLICY, DEDUPLICATED_POLICY)
        or digest != _digest(body)
    ):
        raise ValueError("virtual-camera registry identity is invalid")
    by_name = {camera.image_name: camera for camera in cameras}
    if len(by_name) != len(cameras):
        raise ValueError("dataset mapping camera names are not unique")
    if set(by_name) != set(body["full_camera_names"]):
        raise ValueError("dataset mapping camera registry differs")
    if body["policy"] == DEDUPLICATED_POLICY:
        rebuilt_cameras, _ = build_virtual_camera_registry(
            cameras, deduplicate_geometry=True
        )
        expected_names = [
            rebuilt_cameras[int(index)].image_name
            for index in body["selected_canonical_indices"]
        ]
        if expected_names != body["selected_camera_names"]:
            raise ValueError("deduplicated virtual-camera selection differs")
    selected = []
    for entry in body["entries"]:
        camera = by_name.get(entry["name"])
        if camera is None:
            raise ValueError("selected virtual camera is absent from dataset")
        if (
            entry["image_hw"] != [int(camera.height), int(camera.width)]
            or not torch.equal(torch.tensor(entry["pose_w2c"], dtype=torch.float64),
                               torch.as_tensor(camera.pose_w2c, dtype=torch.float64))
            or not torch.equal(torch.tensor(entry["intrinsic"], dtype=torch.float64),
                               camera_intrinsic(camera).double())
        ):
            raise ValueError(f"virtual-camera calibration differs for {entry['name']}")
        selected.append(camera)
    if [camera.image_name for camera in selected] != body["selected_camera_names"]:
        raise ValueError("virtual-camera selected-name registry differs")
    return selected
