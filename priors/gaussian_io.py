#!/usr/bin/env python3

import argparse
import hashlib
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from common.schemas import PriorManifest


def _selected_property_names(names):
    exact = {
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "opacity",
    }
    prefixes = ("f_dc_", "f_rest_", "scale_", "rot_")
    return [
        name
        for name in names
        if name in exact or any(name.startswith(prefix) for prefix in prefixes)
    ]


def _array_sha256(array, names):
    digest = hashlib.sha256()
    for name in names:
        value = np.asarray(array[name]).astype("<f4", copy=False)
        digest.update(name.encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def export_rgb_prior(
    input_ply,
    output_model,
    *,
    gaussian_type,
    sh_degree,
    source_path,
    images,
    longest_edge,
    iteration,
    prior_kind,
    prior_training_used_feature_loss,
    white_background=True,
    source_method="unknown",
    source_commit="unknown",
    coordinate_frame="mapping_world",
    has_semantic_mask=False,
    sim3_alignment=None,
):
    input_ply = Path(input_ply).expanduser().resolve()
    output_model = Path(output_model).expanduser().resolve()
    ply = PlyData.read(input_ply)
    if len(ply.elements) != 1 or ply.elements[0].name != "vertex":
        raise ValueError("Expected one vertex element in Gaussian PLY")
    source = ply.elements[0].data
    source_names = list(source.dtype.names or ())
    selected = _selected_property_names(source_names)
    gaussian_type = str(gaussian_type).lower()
    expected_scale_names = (
        {"scale_0", "scale_1"}
        if gaussian_type == "2dgs"
        else {"scale_0", "scale_1", "scale_2"}
    )
    source_scale_names = {
        name for name in source_names if name.startswith("scale_")
    }
    if source_scale_names != expected_scale_names:
        raise ValueError(
            f"{gaussian_type} requires scales {sorted(expected_scale_names)}, "
            f"found {sorted(source_scale_names)}"
        )
    required = {
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    } | expected_scale_names
    missing = sorted(required - set(selected))
    if missing:
        raise ValueError(f"Input PLY is missing Gaussian fields: {missing}")
    rest_count = sum(name.startswith("f_rest_") for name in selected)
    expected_rest = 3 * (int(sh_degree) + 1) ** 2 - 3
    if rest_count != expected_rest:
        raise ValueError(
            f"SH degree {sh_degree} expects {expected_rest} f_rest fields, "
            f"found {rest_count}"
        )
    finite_fields = [
        name
        for name in selected
        if name in required or name.startswith("f_rest_")
    ]
    finite = np.ones(source.shape[0], dtype=bool)
    for name in finite_fields:
        finite &= np.isfinite(np.asarray(source[name]))
    dropped_nonfinite = int((~finite).sum())
    source = source[finite]
    output_dtype = [(name, "<f4") for name in selected]
    output = np.empty(source.shape[0], dtype=output_dtype)
    for name in selected:
        output[name] = np.asarray(source[name], dtype=np.float32)

    point_cloud = (
        output_model
        / "point_cloud"
        / f"iteration_{int(iteration)}"
        / "point_cloud.ply"
    )
    point_cloud.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(output, "vertex")], text=False).write(
        point_cloud
    )
    geometry_names = [
        name
        for name in selected
        if name in {"x", "y", "z", "opacity"}
        or name.startswith(("scale_", "rot_"))
    ]
    appearance_names = [
        name for name in selected if name.startswith(("f_dc_", "f_rest_"))
    ]
    removed = sorted(set(source_names) - set(selected))
    localization_removed = [
        name
        for name in removed
        if name.startswith("loc_")
        or name
        in {
            "localization_opacity",
            "loc_anchor_offset",
            "source_index",
        }
    ]
    manifest = {
        "schema_version": 1,
        "prior_kind": str(prior_kind),
        "gaussian_type": str(gaussian_type),
        "sh_degree": int(sh_degree),
        "iteration": int(iteration),
        "primitive_count": int(output.shape[0]),
        "source_primitive_count": int(finite.shape[0]),
        "dropped_nonfinite_primitive_count": dropped_nonfinite,
        "source_ply": str(input_ply),
        "exported_ply": str(point_cloud),
        "exported_ply_sha256": hashlib.sha256(
            point_cloud.read_bytes()
        ).hexdigest(),
        "geometry_sha256": _array_sha256(output, geometry_names),
        "appearance_sha256": _array_sha256(output, appearance_names),
        "prior_training_used_feature_loss": bool(
            prior_training_used_feature_loss
        ),
        "white_background": bool(white_background),
        "source_localization_state_present": bool(localization_removed),
        "localization_state_present": False,
        "detector_state_present": False,
        "removed_property_count": len(removed),
        "removed_localization_property_count": len(localization_removed),
        "removed_localization_properties": localization_removed,
        "retained_properties": selected,
    }
    with (output_model / "rgb_prior_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    normalized_manifest = PriorManifest(
        prior_type=str(gaussian_type),
        source_method=str(source_method),
        source_commit=str(source_commit),
        coordinate_frame=str(coordinate_frame),
        camera_convention="w2c",
        pixel_center_convention="grid_index_plus_half",
        primitive_count=int(output.shape[0]),
        ply_sha256=manifest["exported_ply_sha256"],
        has_semantic_mask=bool(has_semantic_mask),
        sim3_alignment=sim3_alignment,
    ).as_dict()
    normalized_manifest.update(
        {
            "gaussians": str(point_cloud.relative_to(output_model)),
            "sh_degree": int(sh_degree),
            "geometry_sha256": manifest["geometry_sha256"],
            "appearance_sha256": manifest["appearance_sha256"],
        }
    )
    with (output_model / "prior_manifest.json").open("w") as handle:
        json.dump(normalized_manifest, handle, indent=2, sort_keys=True)
    cfg = Namespace(
        data_device="cpu",
        eval=False,
        feature_type="sp",
        gaussian_type=str(gaussian_type),
        images=str(images),
        longest_edge=int(longest_edge),
        model_path=str(output_model),
        norm_before_render=True,
        render_items=["RGB", "Depth"],
        resolution=1,
        sh_degree=int(sh_degree),
        source_path=str(Path(source_path).expanduser().resolve()),
        speedup=False,
        white_background=bool(white_background),
    )
    (output_model / "cfg_args").write_text(str(cfg))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_ply", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--gaussian_type", choices=["2dgs", "3dgs"], required=True)
    parser.add_argument("--sh_degree", type=int, required=True)
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--longest_edge", type=int, default=0)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument(
        "--prior_kind",
        choices=["rgb_only", "feature_stripped"],
        required=True,
    )
    parser.add_argument(
        "--prior_training_used_feature_loss", action="store_true"
    )
    parser.add_argument(
        "--black_background",
        action="store_true",
        help="Record the official black-background RGB training convention.",
    )
    parser.add_argument(
        "--source_method",
        choices=["vanilla_3dgs", "vanilla_2dgs", "anysplat", "matcha", "unknown"],
        default="unknown",
    )
    parser.add_argument("--source_commit", default="unknown")
    parser.add_argument("--coordinate_frame", default="mapping_world")
    parser.add_argument("--has_semantic_mask", action="store_true")
    args = parser.parse_args()
    manifest = export_rgb_prior(
        args.input_ply,
        args.output_model,
        gaussian_type=args.gaussian_type,
        sh_degree=args.sh_degree,
        source_path=args.source_path,
        images=args.images,
        longest_edge=args.longest_edge,
        iteration=args.iteration,
        prior_kind=args.prior_kind,
        prior_training_used_feature_loss=(
            args.prior_training_used_feature_loss
        ),
        white_background=not args.black_background,
        source_method=args.source_method,
        source_commit=args.source_commit,
        coordinate_frame=args.coordinate_frame,
        has_semantic_mask=args.has_semantic_mask,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
