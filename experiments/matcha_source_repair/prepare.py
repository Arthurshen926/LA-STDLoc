"""CPU-only, isolated MAtCha camera/chart transport. Never launches training.

The output retains fitted per-view chart fields; it is NOT a new upstream fit.
All original files are read-only. Local chart coordinates are transported using
the original cameras.json affine C2W, not its COLMAP rotation projection.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import numpy as np
from plyfile import PlyData, PlyElement

R = Path("/mnt/pool/sqy/anygsloc_prior_full_compact_20261011_v1")
DEFAULT_OUT = R / "matcha_source_repair_v1"
REPAIRED = Path("/mnt/pool/sqy/anygsloc_stmary_pose_repair_20261011_v2/dataset/StMarysChurch")
DENSE = Path("/root/MAtCha/output_cambridge/datasets_opt/StMarysChurch/train_opt")
CHART = Path("/root/MAtCha/output_cambridge/runs/StMarysChurch_n20_long_masked/mast3r_sfm")
PRODUCER = Path("/root/MAtCha")
OLD_TRAIN = PRODUCER / "output_cambridge_full_retained_v2/StMarysChurch_n20_long_masked_retrain"
FROZEN = Path("/mnt/pool/sqy/stdloc_lafgs_v1_frozen_multiscene_20260731")
CODE_READER = Path("/mnt/pool/sqy/anygsloc_three_gpu_20260910/code/data/colmap.py")
FIXED_HASHES = {
    str(REPAIRED / "sparse/0/images.bin"): "f1a9a4de74acb658c070dd3925dea53aec3a6d5e67dded66e7a2f460b25df874",
    str(REPAIRED / "sparse/0/cameras.bin"): "0540831776ffd05a80afa033776ba023159b1c989c743ed3eee426433e6bf18a",
    str(DENSE / "sparse/0/images.bin"): "eae386d58979d809c08171623c223f8f320312e6a260c277146d23f8f2070820",
    str(DENSE / "source_split.txt"): "dac4d5452f02b62448a60325cec57a7ecc498d40563fbc984546e5a611890115",
    str(CHART / "sparse/0/images.bin"): "7d45c1801d15a603b77daca9c9c942e8874dc04ca2be6ab250e655e438c4692a",
    str(CHART / "cameras.json"): "36386f6caab1df38c313675bae1b9c0d0186533b23e31274c50ce8149992cb6c",
    str(CHART / "charts_data.npz"): "53be32588d69bf899a62e70eda05ebbf12176d8b0b79d21e88e4e351366de8a9",
    str(DENSE / "sparse/0/points3D.ply"): "0b31199c6123cdc770597921bd1f4538105d5844f91bc684e2f1c9ce52b1d5b2",
    str(OLD_TRAIN / "logs/train_30000_driver.log"): "d695c4472d21b730b8af937203d555e080b4cdd0ab43a557dfab16b79fd8104e",
}
POINT_DTYPE = np.dtype([
    ("id", "<u8"), ("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
    ("r", "u1"), ("g", "u1"), ("b", "u1"), ("error", "<f8"),
    ("track_length", "<u8"), ("image_id", "<i4"), ("point2D_idx", "<i4"),
], align=False)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def dump(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

def reader():
    spec = importlib.util.spec_from_file_location("matcha_repair_colmap_reader", CODE_READER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def w2c(image, colmap):
    q = np.asarray(image.qvec, dtype=np.float64)
    require(abs(np.linalg.norm(q) - 1.0) < 1e-10, "replacement q is not unit")
    matrix = np.eye(4)
    matrix[:3, :3] = colmap.qvec2rotmat(q)
    matrix[:3, 3] = image.tvec
    require(np.max(np.abs(matrix[:3, :3].T @ matrix[:3, :3] - np.eye(3))) < 1e-10,
            "replacement rotation not SO3")
    return matrix

def transport(points_world, old_c2w, new_w2c):
    """Column convention: P_new = inv(W_new) @ inv(C_old) @ P_old."""
    points = np.asarray(points_world, dtype=np.float64)
    old_c2w = np.asarray(old_c2w, dtype=np.float64)
    new_w2c = np.asarray(new_w2c, dtype=np.float64)
    require(points.shape[-1] == 3 and np.isfinite(points).all(), "nonfinite/invalid points")
    for x in (old_c2w, new_w2c):
        require(x.shape == (4, 4) and np.isfinite(x).all(), "invalid transform")
        require(np.array_equal(x[3], [0., 0., 0., 1.]), "invalid homogeneous row")
        require(abs(np.linalg.det(x[:3, :3])) > 1e-6, "singular transform")
    old_w2c = np.linalg.inv(old_c2w)
    local = points @ old_w2c[:3, :3].T + old_w2c[:3, 3]
    new_c2w = np.linalg.inv(new_w2c)
    converted = local @ new_c2w[:3, :3].T + new_c2w[:3, 3]
    return converted, local

def image_records(blob):
    count = struct.unpack_from("<Q", blob, 0)[0]
    offset = 8
    result = []
    for _ in range(count):
        start = offset
        fields = struct.unpack_from("<idddddddi", blob, offset)
        offset += 64
        end = blob.index(b"\0", offset)
        name = bytes(blob[offset:end]).decode("utf8")
        offset = end + 1
        npoints = struct.unpack_from("<Q", blob, offset)[0]
        offset += 8 + 24 * npoints
        require(offset <= len(blob), "truncated COLMAP image records")
        result.append((start, offset, fields[0], name))
    require(offset == len(blob), "trailing COLMAP image bytes")
    require(len({x[3] for x in result}) == count, "duplicate image names")
    return result

def patch_image_bytes(blob, replacements):
    records = image_records(blob)
    require(set(replacements) == {x[3] for x in records}, "replacement name-set mismatch")
    result = bytearray(blob)
    for start, end, image_id, name in records:
        q, t = replacements[name]
        require(abs(np.linalg.norm(q) - 1) < 1e-10, "nonunit replacement quaternion")
        require(np.isfinite(q).all() and np.isfinite(t).all(), "nonfinite pose")
        struct.pack_into("<ddddddd", result, start + 4, *q, *t)
        require(result[start:start+4] == blob[start:start+4] and
                result[start+60:end] == blob[start+60:end], "nonpose record changed")
    return bytes(result)

def link_frozen(source, destination, manifest):
    source, destination = Path(source), Path(destination)
    require(source.is_file(), f"missing input: {source}")
    require(not destination.exists() and not destination.is_symlink(), f"destination exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = sha(source)
    destination.symlink_to(source.resolve())
    manifest[str(destination)] = {"source": str(source), "resolved_source": str(source.resolve()), "sha256": digest}

def validate_dense_intrinsics(bindings, source_cameras, corrected, corrected_cameras):
    """Check each name's camera binding and original SIMPLE_RADIAL -> PINHOLE K convention."""
    report = []
    for flat, camera_id in sorted(bindings.items()):
        name = flat.replace("__", "/")
        require(name in corrected, f"dense intrinsic mapping name missing: {name}")
        reference_id = corrected[name].camera_id
        require(camera_id in source_cameras and reference_id in corrected_cameras,
                f"dense camera_id binding missing: {name}")
        old, ref = source_cameras[camera_id], corrected_cameras[reference_id]
        require(old.id == camera_id and ref.id == reference_id, f"dense camera_id object mismatch: {name}")
        require(old.model == "PINHOLE" and ref.model == "SIMPLE_RADIAL",
                f"dense camera model convention changed: {name}")
        require((old.width, old.height) == (ref.width, ref.height), f"dense image size differs: {name}")
        old_params, ref_params = np.asarray(old.params), np.asarray(ref.params)
        require(old_params.shape == (4,) and ref_params.shape == (4,) and
                np.isfinite(old_params).all() and np.isfinite(ref_params).all(),
                f"dense intrinsic parameters invalid: {name}")
        expected = np.array([ref_params[0], ref_params[0], ref_params[1], ref_params[2]])
        require(np.array_equal(old_params, expected), f"dense pinhole K differs: {name}")
        report.append({"name": name, "source_camera_id": int(camera_id),
                       "corrected_camera_id": int(reference_id), "source_model": old.model,
                       "corrected_model": ref.model, "width": int(old.width), "height": int(old.height),
                       "pinhole_fx_fy_cx_cy": old_params.tolist(),
                       "corrected_radial_k": float(ref_params[3])})
    return {"checked_count": len(report), "per_name": report,
            "contract": "Original PINHOLE K equals [f,f,cx,cy] from the corresponding corrected SIMPLE_RADIAL camera. Radial distortion is not asserted equal or newly applied; original pinhole conversion is retained."}

def validate_copied_intrinsics(original, copied):
    require(set(original) == set(copied), "copied camera_id set differs")
    for key in original:
        a, b = original[key], copied[key]
        require((a.id, a.model, a.width, a.height) == (b.id, b.model, b.width, b.height) and
                np.array_equal(a.params, b.params), "copied camera model/size/K differs")

def prepare_camera_input(source, target, expected_count, corrected, corrected_cameras, colmap, source_hashes, links, dense_pool=False):
    raw = (source / "sparse/0/images.bin").read_bytes()
    records = image_records(raw)
    require(len(records) == expected_count, "source camera count changed")
    original_cameras = colmap.read_intrinsics_binary(str(source / "sparse/0/cameras.bin"))
    bindings = {name: struct.unpack_from("<i", raw, start + 60)[0] for start, _, _, name in records}
    require(set(bindings.values()) <= set(original_cameras), "input camera_id binding missing")
    intrinsic_report = validate_dense_intrinsics(bindings, original_cameras, corrected, corrected_cameras) if dense_pool else {"policy": "Original chart camera IDs, models, dimensions and K retained; no source-K substitution", "count": len(bindings)}
    mapping_path, split_path = source / "name_mapping.json", source / "source_split.txt"
    flat_names = {r[3] for r in records}
    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text())
        require(set(mapping) == flat_names, "name mapping camera set differs")
        require(all(mapping[n] == n.replace("__", "/") for n in flat_names), "name mapping differs from canonical mapping")
    if split_path.exists():
        split = [x.strip() for x in split_path.read_text().splitlines() if x.strip()]
        require(len(split) == len(records) and set(split) == {n.replace("__", "/") for n in flat_names},
                "source split camera set differs")
    replacements, source_names = {}, []
    for _, _, _, flat in records:
        name = flat.replace("__", "/")
        require(name in corrected, f"missing corrected mapping pose: {name}")
        image = corrected[name]
        w2c(image, colmap)
        replacements[flat] = (image.qvec, image.tvec)
        source_names.append(name)
    sparse = target / "sparse/0"
    sparse.mkdir(parents=True)
    (sparse / "images.bin").write_bytes(patch_image_bytes(raw, replacements))
    for name in ("cameras.bin", "cameras.txt"):
        p = source / "sparse/0" / name
        if p.exists():
            shutil.copy2(p, sparse / name)
            require(sha(p) == sha(sparse / name), "intrinsics copy changed")
    copied_cameras = colmap.read_intrinsics_binary(str(sparse / "cameras.bin"))
    validate_copied_intrinsics(original_cameras, copied_cameras)
    for name in ("name_mapping.json", "source_split.txt"):
        p = source / name
        if p.exists():
            shutil.copy2(p, target / name)
    (target / "images").mkdir()
    for _, _, _, flat in records:
        link_frozen(source / "images" / flat, target / "images" / flat, links)
    written = colmap.read_extrinsics_binary(str(sparse / "images.bin"))
    for image in written.values():
        reference = corrected[image.name.replace("__", "/")]
        require(np.array_equal(image.qvec, reference.qvec) and
                np.array_equal(image.tvec, reference.tvec), "pose binary readback mismatch")
    return records, source_names, intrinsic_report

def transform_sparse_points(source, target, old_by_id, new_by_id):
    raw = (source / "sparse/0/points3D.bin").read_bytes()
    count = struct.unpack_from("<Q", raw, 0)[0]
    require(len(raw) == 8 + count * POINT_DTYPE.itemsize, "point tracks are not exactly one owner")
    points = np.frombuffer(raw, dtype=POINT_DTYPE, offset=8, count=count)
    require(np.all(points["track_length"] == 1), "ambiguous multi-chart point")
    require(set(np.unique(points["image_id"])) <= set(old_by_id), "unknown point chart owner")
    out = points.copy()
    xyz = np.column_stack([points[k] for k in ("x", "y", "z")])
    converted = np.empty_like(xyz)
    maximum = 0.0
    for image_id in sorted(old_by_id):
        rows = points["image_id"] == image_id
        new, local = transport(xyz[rows], old_by_id[image_id], new_by_id[image_id])
        converted[rows] = new
        check = new @ new_by_id[image_id][:3, :3].T + new_by_id[image_id][:3, 3]
        maximum = max(maximum, float(np.max(np.abs(check - local))))
    for index, key in enumerate(("x", "y", "z")):
        out[key] = converted[:, index]
    for key in set(POINT_DTYPE.names) - {"x", "y", "z"}:
        require(np.array_equal(points[key], out[key]), f"point attribute changed: {key}")
    (target / "sparse/0/points3D.bin").write_bytes(struct.pack("<Q", count) + out.tobytes())
    ply = PlyData.read(source / "sparse/0/points3D.ply")
    vertex = ply["vertex"].data.copy()
    require(len(vertex) == count, "native PLY and BIN counts differ")
    ply_xyz = np.column_stack([vertex[k] for k in ("x", "y", "z")])
    require(np.allclose(ply_xyz, xyz, rtol=1e-6, atol=1e-5), "native PLY and BIN row order mismatch")
    for key in ("nx", "ny", "nz"):
        if key in vertex.dtype.names:
            require(np.all(vertex[key] == 0), "nonzero native normals require an explicitly reviewed transport")
    for i, key in enumerate(("x", "y", "z")):
        vertex[key] = converted[:, i].astype(vertex.dtype[key])
    for key in set(vertex.dtype.names) - {"x", "y", "z"}:
        require(np.array_equal(vertex[key], ply["vertex"].data[key]), "PLY nonposition field changed")
    PlyData([PlyElement.describe(vertex, "vertex")], text=ply.text,
            byte_order=ply.byte_order, comments=ply.comments, obj_info=ply.obj_info).write(target / "sparse/0/points3D.ply")
    return {"count": int(count), "one_camera_track_per_point": True,
            "local_coordinate_float64_roundtrip_max_abs": maximum,
            "all_nonposition_attributes_equal": True}

def frozen_recipe(output, chart, dense):
    model = output / "native_training/free_gaussians"
    args = [
        str(PRODUCER / "2d-gaussian-splatting/train_with_charts.py"),
        "-s", str(chart), "-m", str(model), "--iterations", "30000",
        "--densify_until_iter", "15000", "--opacity_reset_interval", "3000",
        "--depth_ratio", "0.5", "--use_mip_filter", "--dense_data_path", str(dense),
        "--normal_consistency_from", "7000", "--distortion_from", "3000",
        "--depthanythingv2_checkpoint_dir", str(PRODUCER / "Depth-Anything-V2/checkpoints"),
        "--depthanything_encoder", "vitl", "--dense_regul", "default",
        "--cambridge_mask_pickle", "/mnt/pool/sqy/Cambridge_stdloc/StMarysChurch/processed/masks.pkl",
        "--cambridge_mask_dataset_path", str(dense), "--cambridge_mask_indices", "0", "2",
        "--rgb_loss_type", "l1", "--rgb_charbonnier_eps", "0.001",
    ]
    return {
        "executable": "/root/miniconda3/envs/cybersim_agent/bin/python", "argv": args,
        "cwd": str(PRODUCER), "seed": 0, "seed_source": "utils/general_utils.py:safe_state",
        "gpu_launched": False, "ready_for_gpu": False,
        "gpu_gate": "Independent input review, frozen-code review and parent-selected long-lived execution required",
        "gpu_binding": "Parent scheduler must assign CUDA_VISIBLE_DEVICES; no device claimed by this package",
        "env_candidate": {"PYTHONPATH": str(PRODUCER),
                          "PATH": "/root/miniconda3/envs/cybersim_agent/bin:/usr/local/cuda/bin:/usr/bin:/bin",
                          "LD_LIBRARY_PATH": "/root/miniconda3/envs/cybersim_agent/lib",
                          "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"},
        "historical_env_limits": "cybersim_agent is evidenced by original log; exact original GPU and inherited CPU thread counts were not archived",
        "historical_runtime": {"optimization_seconds": 4010, "logged_total_seconds": 4297},
        "long_runtime_risk": "Historical training exceeded 3600 seconds. Do not shorten 30k recipe to fit a Runner limit.",
        "checkpoint_limit": "Original recipe has no saved PTH. capture includes Gaussian/optimizer/iteration but excludes RNG and sampling stacks; native resume is not exact continuation.",
        "raw_output": str(model / "point_cloud/iteration_30000/point_cloud.ply"),
        "export_output_proposed": str(output / "export/rgb_matcha_2dgs"),
        "training_background": "black", "downstream_frozen_prior_background": "white",
        "type": "2dgs", "sh_degree": 3,
    }

def prepare(output):
    output = Path(output)
    require(output == DEFAULT_OUT, "output must be the authorized isolated directory")
    require(not output.exists() and not output.is_symlink(), "refusing existing output directory")
    source_hashes = {}
    for path, expected in FIXED_HASHES.items():
        actual = sha(path)
        require(actual == expected, f"source hash changed: {path}")
        source_hashes[path] = actual
    colmap = reader()
    corrected_cameras = colmap.read_intrinsics_binary(str(REPAIRED / "sparse/0/cameras.bin"))
    corrected_all = colmap.read_extrinsics_binary(str(REPAIRED / "sparse/0/images.bin"))
    train_names = {line.split()[0] for line in (REPAIRED / "dataset_train.txt").read_text().splitlines()
                   if line.split() and line.split()[0].endswith((".png", ".jpg"))}
    corrected = {v.name: v for v in corrected_all.values() if v.name in train_names}
    require(len(corrected) == 1487, "corrected mapping camera pool changed")
    for source in (DENSE, CHART):
        for name in ("cameras.bin", "cameras.txt", "points3D.bin", "points3D.ply"):
            path = source / "sparse/0" / name
            if path.exists():
                source_hashes[str(path)] = sha(path)
        for name in ("name_mapping.json", "source_split.txt"):
            path = source / name
            if path.exists():
                source_hashes[str(path)] = sha(path)
    source_hashes[str(REPAIRED / "dataset_train.txt")] = sha(REPAIRED / "dataset_train.txt")
    source_hashes[str(CODE_READER)] = sha(CODE_READER)
    output.mkdir()
    chart, dense = output / "inputs/mast3r_sfm", output / "inputs/train_opt"
    links = {}
    dense_records, dense_names, dense_intrinsics = prepare_camera_input(DENSE, dense, 512, corrected, corrected_cameras, colmap, source_hashes, links, dense_pool=True)
    chart_records, chart_names, chart_intrinsics = prepare_camera_input(CHART, chart, 20, corrected, corrected_cameras, colmap, source_hashes, links)
    for name in ("points3D.bin", "points3D.ply"):
        link_frozen(DENSE / "sparse/0" / name, dense / "sparse/0" / name, links)
    original_json = json.loads((CHART / "cameras.json").read_text())
    chart_order = [Path(x).name for x in original_json["filepaths"]]
    require(chart_order == sorted(chart_order) and set(chart_order) == {r[3] for r in chart_records},
            "chart array and camera ordering mismatch")
    old_by_id, new_by_id, old_by_name, new_by_name = {}, {}, {}, {}
    id_by_name = {r[3]: r[2] for r in chart_records}
    repaired_json = dict(original_json)
    repaired_json["filepaths"] = [str(chart / "images" / n) for n in chart_order]
    repaired_json["cams2world"] = []
    for i, name in enumerate(chart_order):
        old = np.asarray(original_json["cams2world"][i], dtype=np.float64)
        new = w2c(corrected[name.replace("__", "/")], colmap)
        old_by_id[id_by_name[name]], new_by_id[id_by_name[name]] = old, new
        old_by_name[name], new_by_name[name] = old, new
        repaired_json["cams2world"].append(np.linalg.inv(new).tolist())
    require(repaired_json["focals"] == original_json["focals"], "native focal changed")
    dump(chart / "cameras.json", repaired_json)
    with np.load(CHART / "charts_data.npz", allow_pickle=False) as arrays:
        original_arrays = {k: arrays[k].copy() for k in arrays.files}
    require(set(original_arrays) == {"pts", "depths", "prior_depths", "confs", "scale_factor"}, "chart schema changed")
    require(original_arrays["pts"].shape == (20, 288, 512, 3), "chart shape changed")
    scale = float(original_arrays["scale_factor"])
    require(np.isfinite(scale) and scale > 0, "invalid chart scale")
    repaired_arrays = dict(original_arrays)
    new_points = np.empty_like(original_arrays["pts"])
    chart_checks = []
    for i, name in enumerate(chart_order):
        old_points = original_arrays["pts"][i].astype(np.float64) / scale
        new, local = transport(old_points, old_by_name[name], new_by_name[name])
        new_points[i] = (new * scale).astype(new_points.dtype)
        cast_world = new_points[i].astype(np.float64) / scale
        w = new_by_name[name]
        cast_local = cast_world @ w[:3, :3].T + w[:3, 3]
        errors = np.abs(cast_local - local)
        tolerance = 5e-5 + 3e-7 * np.maximum(1, np.abs(local))
        require(np.all(errors <= tolerance), "chart float32 local-coordinate preservation failed")
        depth = original_arrays["depths"][i].astype(np.float64) / scale
        chart_checks.append({"name": name, "local_max_abs_error": float(errors.max()),
                             "old_depth_residual_max_abs": float(np.max(np.abs(local[..., 2] - depth))),
                             "new_depth_residual_max_abs": float(np.max(np.abs(cast_local[..., 2] - depth))),
                             "world_displacement_max": float(np.linalg.norm(new - old_points, axis=-1).max())})
    repaired_arrays["pts"] = new_points
    np.savez(chart / "charts_data.npz", **repaired_arrays)
    with np.load(chart / "charts_data.npz", allow_pickle=False) as loaded:
        for key in set(original_arrays) - {"pts"}:
            require(np.array_equal(loaded[key], original_arrays[key]), f"frozen chart field changed: {key}")
    sparse_check = transform_sparse_points(CHART, chart, old_by_id, new_by_id)
    code_files = [
        PRODUCER / "2d-gaussian-splatting/train_with_charts.py",
        PRODUCER / "2d-gaussian-splatting/scene/gaussian_model.py",
        PRODUCER / "2d-gaussian-splatting/scene/__init__.py",
        PRODUCER / "2d-gaussian-splatting/utils/general_utils.py",
        PRODUCER / "matcha/dm_scene/charts.py", PRODUCER / "matcha/dm_scene/cameras.py",
        PRODUCER / "matcha/pointmap/mast3r.py", PRODUCER / "matcha/dm_trainers/charts_alignment.py",
        PRODUCER / "configs/free_gaussians_refinement/long.yaml",
        Path(__file__).resolve(),
    ]
    for path in code_files:
        source_hashes[str(path)] = sha(path)
    # Verify all bound original files and linked RGB/point assets again.
    for path, expected in source_hashes.items():
        require(sha(path) == expected, f"original source changed during preparation: {path}")
    for link, info in links.items():
        require(sha(link) == info["sha256"], f"linked source changed during preparation: {link}")
    files = {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file() and not p.is_symlink()}
    plan = frozen_recipe(output, chart, dense)
    plan["bound_source_hashes"] = source_hashes
    plan["input_artifact_hashes"] = files
    plan["symlink_input_hashes"] = links
    dump(output / "training_plan.json", plan)
    report = {
        "schema": "matcha_source_pose_transport_v1", "complete": True, "cuda_initialized": False,
        "gpu_launched": False, "independent_review_pending": True,
        "scope": "Camera and coordinate repair of a frozen fitted chart representation; not equivalent to re-running upstream inference or alignment with corrected poses.",
        "formula_column_vectors": "X=pts/s; L=inv(C_old)@[X,1]; X_new=inv(W_new)@L; pts_new=s*X_new. Same s, depths, prior_depths and confs.",
        "old_basis": str(CHART / "cameras.json") + ":cams2world, full affine inverse, without SO3 projection",
        "new_basis": str(REPAIRED / "sparse/0/images.bin") + ":unit-q world-to-camera and original repaired t",
        "depth_precision": "Coordinate roundtrip is float64; stored pts retain float32. Frozen depth arrays are bitwise unchanged; historical point/depth floating discrepancy is preserved within transport tolerance.",
        "intrinsics_policy": "Per-name dense camera-ID bindings, model convention, image dimensions and PINHOLE [fx,fy,cx,cy] equal to corrected SIMPLE_RADIAL [f,f,cx,cy] are explicitly checked; this does not assert equal distortion models. Original chart K/focals retained, including fitted calibration; only q/t corrected.",
        "dense_intrinsics_validation": dense_intrinsics, "chart_intrinsics_validation": chart_intrinsics,
        "dense_count": 512, "chart_count": 20, "dense_original_names": dense_names,
        "chart_original_names": chart_names, "chart_array_order": chart_order,
        "chart_scale_factor": scale, "chart_checks": chart_checks, "sparse_points": sparse_check,
        "dense_initialization": {"point_count_ply": 419837, "path": str(dense / "sparse/0/points3D.ply"),
                                 "sha256": sha(dense / "sparse/0/points3D.ply"),
                                 "policy": "Preserve original world-coordinate point asset byte-for-byte; no vanilla replacement"},
        "source_hashes": source_hashes, "output_hashes": files, "linked_inputs": links,
        "original_assets_unchanged": True,
        "not_materialized": "Raw pointmap JSONs are not training inputs and are not copied. Do not use this package for fresh upstream alignment without separately reconstructing its inputs.",
    }
    dump(output / "cpu_validation.json", report)
    shutil.copy2(Path(__file__).resolve(), output / "prepare.py")
    print(json.dumps({"output": str(output), "complete": True, "chart_count": 20, "dense_count": 512,
                      "max_chart_local_error": max(x["local_max_abs_error"] for x in chart_checks),
                      "training_plan_sha256": sha(output / "training_plan.json"),
                      "cpu_validation_sha256": sha(output / "cpu_validation.json")}, indent=2), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    prepare(args.output)
