"""Read-only audit of the already-trained vanilla3DGS source chain (no training)."""
import ast
import json
from pathlib import Path
import struct
import numpy as np
from anygsloc.artifacts import sha256
from anygsloc.data.colmap import read_extrinsics_binary

def read(path):
    return json.loads(Path(path).read_text())

def namespace_values(path):
    expr = ast.parse(Path(path).read_text().strip(), mode="eval").body
    if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name) or expr.func.id != "Namespace" or expr.args:
        raise ValueError("unexpected cfg_args format")
    return {k.arg: ast.literal_eval(k.value) for k in expr.keywords}

def point_tracks(path):
    result = {}
    if Path(path).suffix == ".txt":
        for line in Path(path).read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            result[int(fields[0])] = (np.asarray(fields[1:4], dtype=float),
                np.asarray(fields[8::2], dtype=np.int64))
        return result
    with Path(path).open("rb") as stream:
        count = struct.unpack("<Q", stream.read(8))[0]
        for _ in range(count):
            h = struct.unpack("<QdddBBBd", stream.read(43))
            n = struct.unpack("<Q", stream.read(8))[0]
            track = np.frombuffer(stream.read(8*n), dtype="<i4").reshape(-1, 2)
            result[h[0]] = (np.asarray(h[1:4]), track[:, 0].copy())
        if stream.read(1):
            raise ValueError("unexpected trailing COLMAP points bytes")
    return result

def audit_mapping_only_initialization(original_points, prepared_points, mapping_image_ids):
    missing = changed = mixed = short = 0
    for point_id, (xyz, _) in prepared_points.items():
        if point_id not in original_points:
            missing += 1
            continue
        source_xyz, original_track = original_points[point_id]
        changed += int(not np.allclose(xyz, source_xyz, rtol=0, atol=1e-8))
        mixed += int(any(int(i) not in mapping_image_ids for i in original_track))
        short += int(len(original_track) < 2)
    result = {"prepared_points": len(prepared_points), "original_points": len(original_points),
              "missing_original_id": missing, "changed_xyz": changed,
              "original_track_contains_nonmapping": mixed, "short_track": short}
    if not prepared_points or any((missing, changed, mixed, short)):
        raise ValueError("Gaussian initialization is not complete-original-track mapping-only: " + str(result))
    return result

def audit_prepared_camera_registry(prepared, entries, mapping_cameras, intrinsics):
    from anygsloc.data.colmap import qvec2rotmat
    from anygsloc.mapping.virtual_camera_registry import camera_intrinsic
    reverse = {value[0]: key for key,value in entries.items()}
    names = [row.name for row in prepared.values()]
    if len(reverse) != len(entries) or len(names) != len(set(names)) or set(names) != set(reverse):
        raise ValueError("trainer sparse image registry differs from mapping whitelist")
    for row in prepared.values():
        expected = mapping_cameras[reverse[row.name]]
        pose = np.eye(4);pose[:3,:3] = qvec2rotmat(row.qvec);pose[:3,3] = row.tvec
        camera = intrinsics[row.camera_id]
        if camera.model == "PINHOLE":
            fx,fy,cx,cy = camera.params
        elif camera.model == "SIMPLE_PINHOLE":
            fx,cx,cy = camera.params;fy=fx
        else:
            raise ValueError("training images must use rectified pinhole cameras")
        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
        if (not np.array_equal(pose.astype(np.float32),expected.pose_w2c.astype(np.float32)) or
            not np.allclose(K,camera_intrinsic(expected).numpy(),rtol=0,atol=1e-3) or
            (camera.height,camera.width) != (expected.height,expected.width)):
            raise ValueError("trainer camera pose/intrinsics differ from mapping camera")
    return {"trainer_camera_count":len(names),"exact_mapping_names":True,
            "mapping_pose_comparison":"exact float32 dataset contract","mapping_K_atol":1e-3}

def revalidate_audit_record(scene, dataset, prior, audit_path):
    """Reuse this campaign's completed full audit, rebinding all referenced evidence."""
    audit_path = Path(audit_path)
    audit = read(audit_path)
    if audit["training_job_id"] != scene + "_vanilla_3dgs_reconstruct":
        raise ValueError("audit scene identity differs")
    expected_ply = Path(audit["native_path"]) / "point_cloud/iteration_30000/point_cloud.ply"
    if Path(prior["ply"]).resolve() != expected_ply.resolve():
        raise ValueError("audit prior identity differs")
    evidence = audit["evidence_sha256"]
    if not evidence or any(sha256(p) != expected for p, expected in evidence.items()):
        raise ValueError("audited provenance evidence changed")
    producer = read(audit["producer_manifest"])["prior"]
    if producer["ply_sha256"] != prior["ply_sha256"] or Path(producer["ply"]).resolve() != expected_ply.resolve():
        raise ValueError("audited producer does not bind selected PLY")
    counts = audit["complete_original_track_audit"]
    if counts["prepared_points"] <= 0 or any(counts[k] != 0 for k in (
            "missing_original_id", "changed_xyz", "original_track_contains_nonmapping",
            "short_track", "initial_ply_maximum_coordinate_difference")):
        raise ValueError("cached audit did not pass mapping-only initialization")
    info = read(audit["input_manifest"])
    if "test_used" in info:
        entries = {r["source_name"]: (r["prepared_name"], r["sha256"]) for r in info["images"]}
    else:
        prefix = dataset.root / "processed"
        entries = {str(Path(r["source"]).relative_to(prefix)): (r["name"], r["sha256"]) for r in info["images"]}
    expected_images = {name: value[1] for name, value in entries.items()}
    cameras = dataset.split("mapping")
    names = {c.image_name for c in cameras}
    if (set(expected_images) != names or names & {c.image_name for c in dataset.split("test")} or
            audit["mapping_image_count"] != len(cameras) or
            audit["prepared_camera_audit"]["trainer_camera_count"] != len(cameras) or
            audit["prepared_camera_audit"]["exact_mapping_names"] is not True):
        raise ValueError("cached audit mapping camera registry differs")
    # Rebind current dataset pose/K/size, not only the previously audited sparse bytes.
    from anygsloc.data.colmap import read_extrinsics_text, read_intrinsics_binary, read_intrinsics_text
    source = Path(audit["source_path"]) / "sparse/0"
    image_path = source / "images.bin"
    if not image_path.exists():
        image_path = image_path.with_suffix(".txt")
    camera_path = source / "cameras.bin"
    if not camera_path.exists():
        camera_path = camera_path.with_suffix(".txt")
    if str(image_path) not in evidence or str(camera_path) not in evidence:
        raise ValueError("prepared camera evidence is not bound")
    images = (read_extrinsics_binary if image_path.suffix == ".bin" else read_extrinsics_text)(str(image_path))
    intrinsics = (read_intrinsics_binary if camera_path.suffix == ".bin" else read_intrinsics_text)(str(camera_path))
    audit_prepared_camera_registry(images, entries, {c.image_name: c for c in cameras}, intrinsics)
    # Mapping RGB is the actual new experiment input, so rehash it once at prepare.
    for camera in cameras:
        if sha256(camera.image_path) != expected_images[camera.image_name]:
            raise ValueError("mapping RGB differs from completed prior audit")
    audit["reused_audit_record"] = {"path": str(audit_path.resolve()), "sha256": sha256(audit_path),
        "evidence_revalidated": True, "training_RGB_rehashed": False,
        "mapping_RGB_rehashed": True,
        "scope": "reuse completed full training-input audit; all referenced evidence and actual selected PLY rebound"}
    return audit, expected_images

def audit_prior(scene, dataset, prior):
    campaign = Path("/mnt/pool/sqy/anygsloc_three_gpu_20260910")
    native = campaign / "priors" / scene / "vanilla_3dgs/native"
    expected_ply = native / "point_cloud/iteration_30000/point_cloud.ply"
    if Path(prior["ply"]).resolve() != expected_ply.resolve():
        raise ValueError("prior is not the explicitly selected existing vanilla3DGS")
    producer_path = campaign / "runs" / (scene+"_vanilla_3dgs") / "anygsloc_scene_manifest.json"
    producer = read(producer_path)
    if producer["prior"]["ply_sha256"] != prior["ply_sha256"] or Path(producer["prior"]["ply"]).resolve() != expected_ply.resolve():
        raise ValueError("producer manifest does not bind selected PLY")
    cfg_path = native / "cfg_args"
    cfg = namespace_values(cfg_path)
    source = Path(cfg["source_path"])
    if cfg.get("images") != "images" or cfg.get("depths") != "" or cfg.get("white_background") is not False:
        raise ValueError("native Gaussian source policy differs")
    matches = []
    for jobs_path in sorted(campaign.glob("gpu*/jobs.json")):
        for job in read(jobs_path):
            if job["id"] == scene+"_vanilla_3dgs_reconstruct":
                matches.append((jobs_path, job))
    if len(matches) != 1:
        raise ValueError("training job identity is ambiguous or absent")
    jobs_path, job = matches[0]
    argv = job["command"]
    value = lambda flag: argv[argv.index(flag)+1]
    if (Path(value("--source_path")).resolve() != source.resolve() or
        Path(value("--model_path")).resolve() != native.resolve() or
        value("--iterations") != "30000" or
        value("--depth_l1_weight_init") != "0" or value("--depth_l1_weight_final") != "0" or
        "--start_checkpoint" in argv):
        raise ValueError("native training command differs from approved vanilla source")
    input_path = source / "input_manifest.json"
    if input_path.exists():
        info = read(input_path)
        if info.get("test_used") is not False or info.get("image_domain") != "rectified":
            raise ValueError("training input is not rectified mapping-only")
        entries = {row["source_name"]: (row["prepared_name"],row["sha256"]) for row in info["images"]}
        declaration = {k: info[k] for k in ("test_used","image_domain","point_policy","retained_points")}
    else:
        input_path = source / "rectification_replay_manifest.json"
        info = read(input_path)
        if info.get("mapping_rgb_only") is not True or info.get("uses_test_queries") is not False:
            raise ValueError("rectification replay is not mapping-only")
        prefix = dataset.root / "processed"
        entries = {str(Path(row["source"]).relative_to(prefix)): (row["name"],row["sha256"])
                   for row in info["images"]}
        declaration = {k: info[k] for k in ("mapping_rgb_only","uses_test_queries","image_policy","camera_policy")}
    allowed = {c.image_name for c in dataset.split("mapping")}
    if set(entries) != allowed or allowed & {c.image_name for c in dataset.split("test")}:
        raise ValueError("training images differ from exact mapping whitelist")
    images_path = dataset.root/"sparse/0/images.bin"
    original_images = read_extrinsics_binary(str(images_path))
    mapping_ids = {i for i,r in original_images.items() if r.name in allowed}
    original_points_path = dataset.root/"sparse/0/points3D.bin"
    def sparse_file(stem):
        binary = source/"sparse/0"/(stem+".bin")
        return binary if binary.exists() else binary.with_suffix(".txt")
    from anygsloc.data.colmap import read_extrinsics_text, read_intrinsics_binary, read_intrinsics_text
    prepared_images_path = sparse_file("images");prepared_cameras_path = sparse_file("cameras")
    prepared_images = (read_extrinsics_binary if prepared_images_path.suffix == ".bin" else read_extrinsics_text)(str(prepared_images_path))
    prepared_intrinsics = (read_intrinsics_binary if prepared_cameras_path.suffix == ".bin" else read_intrinsics_text)(str(prepared_cameras_path))
    camera_audit = audit_prepared_camera_registry(prepared_images, entries,
        {c.image_name:c for c in dataset.split("mapping")},prepared_intrinsics)
    prepared_points_path = sparse_file("points3D")
    prepared_points = point_tracks(prepared_points_path)
    point_audit = audit_mapping_only_initialization(point_tracks(original_points_path),
                  prepared_points, mapping_ids)
    from plyfile import PlyData
    initial_ply = source/"sparse/0/points3D.ply"
    vertices = PlyData.read(str(initial_ply))["vertex"]
    ply_xyz = np.stack([vertices[k] for k in ("x","y","z")], axis=1)
    expected_xyz = np.stack([row[0] for row in prepared_points.values()]).astype(np.float32)
    if ply_xyz.shape != expected_xyz.shape:
        raise ValueError("training PLY differs from mapping-only sparse point count")
    # Compare a coordinate multiset because PLY need not preserve original point IDs/order.
    a = ply_xyz[np.lexsort(ply_xyz.T[::-1])]
    b = expected_xyz[np.lexsort(expected_xyz.T[::-1])]
    if not np.allclose(a,b,rtol=0,atol=1e-5):
        raise ValueError("training PLY differs from mapping-only sparse coordinates")
    point_audit["initial_ply_maximum_coordinate_difference"] = float(np.max(np.abs(a-b)))
    # Mapping images only. Bind the actual training-image bytes, not just counts.
    for original_name, (prepared_name, expected) in entries.items():
        if sha256(source/"images"/prepared_name) != expected or sha256(dataset.root/"processed"/original_name) != expected:
            raise ValueError("prepared training image or real mapping RGB differs")
    train_script = Path(argv[2])
    reader_script = train_script.parent/"scene/dataset_readers.py"
    files = [producer_path,cfg_path,jobs_path,input_path,train_script,reader_script,
             images_path,original_points_path,prepared_points_path,
             sparse_file("images"),sparse_file("cameras"),initial_ply]
    for name, expected in info.get("sparse_sha256", {}).items():
        if sha256(source/"sparse/0"/name) != expected:
            raise ValueError("replayed sparse bytes differ from input manifest")
    return {"source_path": str(source), "native_path": str(native),
            "producer_manifest": str(producer_path), "training_job_id": job["id"],
            "training_argv": argv, "input_manifest": str(input_path),
            "input_declarations": declaration, "complete_original_track_audit": point_audit,
            "mapping_image_count": len(entries), "prepared_camera_audit": camera_audit,
            "evidence_sha256": {str(p):sha256(p) for p in files},
            "semantic_evidence": "native train/reader source and exact argv attached; no invented manifest boolean"}
