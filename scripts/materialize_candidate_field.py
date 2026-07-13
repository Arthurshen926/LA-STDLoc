#!/usr/bin/env python

import argparse
import ast
import hashlib
import json
import os
import shutil
from argparse import Namespace

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_fields(data, names):
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(data[name]).tobytes())
    return digest.hexdigest()


def _link_or_copy(source, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _read_namespace(path):
    with open(path) as handle:
        expression = ast.parse(handle.read().strip(), mode="eval").body
    if not (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Name)
        and expression.func.id == "Namespace"
        and not expression.args
    ):
        raise ValueError(f"Unsupported cfg_args format: {path}")
    values = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            raise ValueError(f"Unsupported cfg_args expansion: {path}")
        values[keyword.arg] = ast.literal_eval(keyword.value)
    return Namespace(**values)


def _write_model_config(source_model, output_model):
    config = _read_namespace(os.path.join(source_model, "cfg_args"))
    config.model_path = os.path.abspath(output_model)
    with open(os.path.join(output_model, "cfg_args"), "w") as handle:
        handle.write(repr(config))
        handle.write("\n")


def _resolve_path(model_root, path):
    if os.path.isabs(path):
        return path
    return os.path.join(model_root, path)


def materialize_candidate_field(
    source_model,
    state_path,
    output_model,
    *,
    iteration=30000,
    detector_folder="",
):
    source_model = os.path.abspath(source_model)
    output_model = os.path.abspath(output_model)
    state_path = _resolve_path(source_model, state_path)
    source_ply = os.path.join(
        source_model,
        "point_cloud",
        f"iteration_{int(iteration)}",
        "point_cloud.ply",
    )
    source_loc_state = os.path.join(
        source_model,
        "point_cloud",
        f"iteration_{int(iteration)}",
        "loc_state.pt",
    )
    for path in (source_ply, source_loc_state, state_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    if os.path.exists(output_model):
        raise FileExistsError(
            f"Refusing to overwrite materialized model directory: {output_model}"
        )

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    indices = torch.as_tensor(state.get("landmark_indices"), dtype=torch.long).reshape(-1)
    features = torch.as_tensor(state.get("landmark_features"), dtype=torch.float32)
    features = features.reshape(features.shape[0], -1)
    if indices.numel() != features.shape[0]:
        raise ValueError("landmark index and feature counts differ")
    if indices.numel() != torch.unique(indices).numel():
        raise ValueError("landmark indices must be duplicate-free")
    if not bool(torch.isfinite(features).all().item()):
        raise ValueError("landmark features contain non-finite values")

    ply = PlyData.read(source_ply)
    vertex_position = next(
        (position for position, element in enumerate(ply.elements) if element.name == "vertex"),
        None,
    )
    if vertex_position is None:
        raise ValueError(f"PLY has no vertex element: {source_ply}")
    vertex = ply.elements[vertex_position]
    source_data = np.array(vertex.data, copy=True)
    field_names = list(source_data.dtype.names or ())
    loc_names = sorted(
        (name for name in field_names if name.startswith("loc_")),
        key=lambda name: int(name.split("_", 1)[1]),
    )
    geometry_names = [name for name in field_names if name not in loc_names]
    if len(loc_names) != features.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: PLY={len(loc_names)} state={features.shape[1]}"
        )
    if indices.numel() and (
        int(indices.min().item()) < 0 or int(indices.max().item()) >= source_data.shape[0]
    ):
        raise IndexError("landmark indices fall outside the PLY vertex array")

    output_data = np.array(source_data, copy=True)
    index_array = indices.numpy()
    feature_array = features.numpy()
    previous = np.stack(
        [source_data[name][index_array] for name in loc_names],
        axis=1,
    )
    for column, name in enumerate(loc_names):
        output_data[name][index_array] = feature_array[:, column]

    point_cloud_dir = os.path.join(
        output_model,
        "point_cloud",
        f"iteration_{int(iteration)}",
    )
    os.makedirs(point_cloud_dir, exist_ok=False)
    output_ply = os.path.join(point_cloud_dir, "point_cloud.ply")
    elements = list(ply.elements)
    elements[vertex_position] = PlyElement.describe(
        output_data,
        vertex.name,
        comments=list(vertex.comments),
    )
    PlyData(
        elements,
        text=ply.text,
        byte_order=ply.byte_order,
        comments=list(ply.comments),
        obj_info=list(ply.obj_info),
    ).write(output_ply)
    _link_or_copy(source_loc_state, os.path.join(point_cloud_dir, "loc_state.pt"))

    for name in ("input.ply", "cameras.json"):
        source = os.path.join(source_model, name)
        if os.path.isfile(source):
            _link_or_copy(source, os.path.join(output_model, name))
    _write_model_config(source_model, output_model)

    if not detector_folder:
        detector_folder = os.path.relpath(os.path.dirname(state_path), source_model)
    source_detector = _resolve_path(source_model, detector_folder)
    output_detector = os.path.join(output_model, detector_folder)
    copied_artifacts = {}
    if os.path.isdir(source_detector):
        os.makedirs(output_detector, exist_ok=True)
        state_iteration = int(state.get("iteration", 0))
        artifact_names = (
            "sampled_idx.pkl",
            "landmark_meta.pt",
            f"{state_iteration}_detector.pth",
            f"{state_iteration}_candidate_teacher_state.pt",
        )
        for name in artifact_names:
            source = os.path.join(source_detector, name)
            if os.path.isfile(source):
                destination = os.path.join(output_detector, name)
                _link_or_copy(source, destination)
                copied_artifacts[name] = _sha256_file(destination)

    readback = PlyData.read(output_ply)["vertex"].data
    geometry_hash_before = _hash_fields(source_data, geometry_names)
    geometry_hash_after = _hash_fields(readback, geometry_names)
    if geometry_hash_before != geometry_hash_after:
        raise RuntimeError("Non-localization PLY fields changed during materialization")
    materialized = np.stack(
        [readback[name][index_array] for name in loc_names],
        axis=1,
    )
    if not np.array_equal(materialized, feature_array):
        raise RuntimeError("Materialized localization features failed readback verification")
    unselected = np.ones(source_data.shape[0], dtype=bool)
    unselected[index_array] = False
    for name in loc_names:
        if not np.array_equal(readback[name][unselected], source_data[name][unselected]):
            raise RuntimeError(f"Unselected localization field changed: {name}")

    delta = np.linalg.norm(feature_array - previous, axis=1)
    manifest = {
        "version": 1,
        "source_model": source_model,
        "output_model": output_model,
        "iteration": int(iteration),
        "source_ply": source_ply,
        "source_ply_sha256": _sha256_file(source_ply),
        "output_ply": output_ply,
        "output_ply_sha256": _sha256_file(output_ply),
        "state_path": state_path,
        "state_sha256": _sha256_file(state_path),
        "landmark_count": int(indices.numel()),
        "feature_dimension": int(features.shape[1]),
        "geometry_field_sha256_before": geometry_hash_before,
        "geometry_field_sha256_after": geometry_hash_after,
        "geometry_fields_exact": True,
        "unselected_localization_fields_exact": True,
        "selected_feature_delta_l2_mean": float(delta.mean()) if delta.size else 0.0,
        "selected_feature_delta_l2_p95": float(np.percentile(delta, 95)) if delta.size else 0.0,
        "selected_feature_delta_l2_max": float(delta.max()) if delta.size else 0.0,
        "detector_folder": detector_folder,
        "copied_artifact_sha256": copied_artifacts,
    }
    manifest_path = os.path.join(output_model, "materialization_manifest.json")
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Write candidate-trained localization embeddings into a derived map PLY."
    )
    parser.add_argument("--source_model", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--detector_folder", default="")
    args = parser.parse_args()
    manifest = materialize_candidate_field(
        args.source_model,
        args.state,
        args.output_model,
        iteration=args.iteration,
        detector_folder=args.detector_folder,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
