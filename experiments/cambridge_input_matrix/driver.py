"""Opt-in 3-input x 2-construction experiment. No launch occurs on import or prepare."""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from supplement_classic import assemble_tracks, fit_point
from supplement_experiments import spatial_rows, save_map
from anygsloc.artifacts import sha256
from anygsloc.data.datasets import ColmapDataset
from anygsloc.mapping.virtual_camera_registry import build_virtual_camera_registry, resolve_virtual_camera_registry
from .contracts import (MODES, SPARSE, GEOMETRY, AUXILIARY, MatrixObservationProvider,
                        classic_rgb_records, validate_payload)
from .projective_association import build_projective_association_graph
from .projective_reconstruction import reconstruct_projective_anchors
from .projective_completion import build_projective_completion

SCENES = ("ShopFacade", "OldHospital", "KingsCollege", "GreatCourt", "StMarysChurch")
METHODS = ("anygsloc", "classic")
DESIGN = {
    "schema": "cambridge_input_matrix_protocol_v1", "seed": 2026,
    "selected_prior_family": "vanilla_3dgs",
    "prior_choice_basis": "parent-approved candidate after exploratory prior comparison; not held-out model selection",
    "scene_order": list(SCENES), "input_modes": list(MODES), "methods": list(METHODS),
    "observation_budget": 2048, "hybrid_quota": {"real": 1024, "render": 1024},
    "quota_stage": "stable detector-score top-k BEFORE V2, no refill or transfer",
    "source_feature_pool": 2048, "full_available_hybrid_nominal_maximum": 4096,
    "same_pose_modalities_are_one_camera": True,
    "mapping_images": "processed", "image_domain": "rectified", "rectification_contract": "opencv_same_k",
    "query_semantics": True, "query_mask_policy": "explicit_valid_mask", "query_protocol": "frozen Localizer with dataset.valid_mask",
    "mapping_semantics": False, "uses_test_queries_for_mapping": False,
    "full_map": "all valid constructed landmarks at the frozen observation budget",
    "common_K": "min(reference scene K, N in all six cells); same spatial_rows in every cell",
    "selection": "0.5m spatial voxel round-robin, deterministic geometric-center/source-row ties",
    "detector": {"name": "SuperPoint", "keypoints": 2048, "nms_radius": 4, "threshold": 0.0,
                 "input": "complete unmasked RGB for both constructors"},
    "shared_pair_policy": {"neighbors": 6, "minimum_baseline_m": 0.03,
                          "maximum_baseline_m": 5.0, "maximum_axis_angle_deg": 75.0},
    "anygsloc": {"association": "existing cycle/chain tracks, top4 epipolar candidates",
                "geometry": "existing fixed-camera robust ray triangulation",
                "fusion": "existing detector/edge-weighted view-bin robust fusion",
                "raw_only": "AnyGSLoc without Gaussian; V2 and depth completion disabled",
                "joint_track_purification": False, "native_selector_in_primary": False},
    "classic": {"label": "self-implemented known-camera SfM-style reference, not official COLMAP",
                "association": "mutual top1, confidence-sorted conflict-free union",
                "geometry": "normalized DLT + fixed-camera point-only Huber",
                "fusion": "normalized arithmetic mean", "uses_depth": False,
                "uses_alpha": False, "uses_semantics": False},
    "changes_from_archived_pure_gs": [
        "Shared unmasked SuperPoint pool capped before V2; archived alpha-gated detector caches are not reused.",
        "Hybrid adds real RGB rows with one physical camera identity and fixed 1024+1024 quota.",
        "Gaussian V2 certificates are sampled at both modality rows for hybrid.",
        "Raw-only disables all Gaussian evidence and depth completion.",
        "Query semantics are on in every cell; archived semantic-off scores are never imported.",
        "Full and shared spatial equal-K are primary; native Any selector is not used to prune classic.",
    ],
    "reporting": "all five scenes, scene-equal means of per-scene metrics; failures stay in denominator",
    "no_test_based_scene_seed_budget_selection": True,
}

def read(path):
    return json.loads(Path(path).read_text())

def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")

def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        torch.save(value, f)

def fingerprints():
    paths = list((ROOT / "experiments/cambridge_input_matrix").glob("*.py"))
    paths.append(Path(__file__).with_name("kernel_adaptations.json"))
    # Include all imported production numerical kernels, frontend and evaluator.
    paths += list((ROOT / "anygsloc").rglob("*.py"))
    paths += [ROOT / "tools" / n for n in ("supplement_classic.py", "supplement_experiments.py")]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(paths))}

def check_kernel_adaptations():
    for patch in read(Path(__file__).with_name("kernel_adaptations.json"))["patches"]:
        text = (ROOT / patch["source"]).read_text()
        for e in patch["edits"]:
            if text.count(e["old_text"]) != 1:
                raise ValueError("upstream changed: " + patch["source"])
            text = text.replace(e["old_text"], e["new_text"])
        # Read APIs may omit the final newline; all source lines must still match exactly.
        if text.splitlines() != (ROOT / patch["target"]).read_text().splitlines():
            raise ValueError("unregistered numerical kernel drift: " + patch["target"])

def protocol(root):
    p = read(Path(root) / "protocol.json")
    if p["design"] != DESIGN or p["code_sha256"] != fingerprints():
        raise ValueError("frozen design or implementation changed; use a new experiment root")
    check_kernel_adaptations()
    return p

def prepare(root, reference, prior_manifest):
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    ref, priors = read(reference), read(prior_manifest)["scenes"]
    if set(priors) != set(SCENES):
        raise ValueError("prior manifest must cover all five predeclared scenes")
    scenes = {}
    for scene in SCENES:
        data = ColmapDataset(ref["scenes"][scene]["dataset"], load_masks=False)
        mapping, registry = build_virtual_camera_registry(data.split("mapping"))
        names = [c.image_name for c in mapping]
        tests = [c.image_name for c in data.split("test")]
        if not tests or set(names) & set(tests):
            raise ValueError("mapping/test split absent or overlaps")
        prior = dict(priors[scene])
        if set(prior) != {"ply", "gaussian_type", "sh_degree", "background"}:
            raise ValueError("prior fields must be exactly ply/type/sh_degree/background")
        if prior["gaussian_type"] != "3dgs" or prior["background"] != "black" or prior["sh_degree"] != 3:
            raise ValueError("this frozen matrix requires the selected black-background SH3 vanilla 3DGS priors")
        prior["ply"] = str(Path(prior["ply"]).resolve())
        prior["ply_sha256"] = sha256(prior["ply"])
        from .prior_provenance import audit_prior, revalidate_audit_record
        audit_path = Path(__file__).with_name("validation") / ("prior_" + scene + ".json")
        if audit_path.exists():
            prior["training_provenance"], image_hashes = revalidate_audit_record(scene, data, prior, audit_path)
        else:
            prior["training_provenance"] = audit_prior(scene, data, prior)
            image_hashes = {c.image_name: sha256(c.image_path) for c in mapping}
        mask = data.root / "processed/masks.pkl"
        if not mask.is_file():
            mask = data.root / "masks.pkl"
        if not mask.is_file():
            raise ValueError("semantic-on mask file missing: " + scene)
        # Mapping images were hashed above. No test RGB is opened during preparation.
        scenes[scene] = {"dataset": str(data.root), "registry": registry, "mapping_names": names,
                         "test_names": tests, "mapping_image_sha256": image_hashes,
                         "mask_path": str(mask), "mask_sha256": sha256(mask),
                         "reference_K": ref["scenes"][scene]["K"], "prior": prior}
    check_kernel_adaptations()
    p = {"design": DESIGN, "scenes": scenes, "code_sha256": fingerprints(),
         "reference_protocol_sha256": sha256(reference), "prior_manifest_sha256": sha256(prior_manifest)}
    write_new(root / "protocol.json", p)
    return p

def safe_mapping_image(dataset, camera, scene):
    if camera.image_name not in scene["mapping_names"] or camera.image_name in scene["test_names"]:
        raise ValueError("refusing test/unlisted RGB read: " + camera.image_name)
    if camera.image_path.resolve() != (dataset.root / "processed" / camera.image_name).resolve():
        raise ValueError("unexpected image root")
    if sha256(camera.image_path) != scene["mapping_image_sha256"][camera.image_name]:
        raise ValueError("mapping image changed")
    return dataset.load_image(camera)

def top_rows(record, count):
    return torch.argsort(record["native_scores"], descending=True, stable=True)[:count]

def subset_record(record, rows):
    n = len(record["native_keypoints"])
    row_fields = set(SPARSE) | set(AUXILIARY) | {"row_source", "source_row"}
    return {k: (v[rows].clone() if k in row_fields and isinstance(v, torch.Tensor) and
                v.ndim >= 1 and v.shape[0] == n else v) for k, v in record.items()}

def merge_camera(real, rendered):
    for k in GEOMETRY:
        if not torch.equal(torch.as_tensor(real[k]), torch.as_tensor(rendered[k])):
            raise ValueError("real/render camera geometry differs")
    r, g = subset_record(real, top_rows(real, 1024)), subset_record(rendered, top_rows(rendered, 1024))
    out = {k: r[k] for k in GEOMETRY}
    for k in SPARSE + ("row_source", "source_row", "v2_valid",
                       "native_depth_at_keypoints", "native_alpha_at_keypoints"):
        out[k] = torch.cat((r[k], g[k]))
    out["available_real"] = len(real["native_keypoints"])
    out["available_render"] = len(rendered["native_keypoints"])
    return out

def payload(mode, records, scene, protocol_sha):
    p = {"schema": "cambridge_input_matrix_observations_v1", "input_mode": mode,
         "uses_source_mapping_rgb": mode != "pure_gs", "uses_test_queries": False,
         "mapping_names": scene["mapping_names"], "test_names": scene["test_names"],
         "queries": records, "protocol_sha256": protocol_sha}
    validate_payload(p)
    return p

def extract(root, scene, device):
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Gaussian extraction requires an explicit CUDA device")
    # The existing renderer allocates K on current CUDA; keep model and renderer aligned.
    with torch.cuda.device(device):
        return _extract(root, scene, device)

@torch.inference_mode()
def _extract(root, scene, device):
    from anygsloc.features.extractor import FeatureExtractor
    from anygsloc.priors.models import GaussianModel2D, GaussianModel3D
    from anygsloc.priors.rendering import render_from_pose_gsplat
    from anygsloc.features.raster_sampling import sample_raster_at_grid_uv
    from anygsloc.mapping.render_certificate import CertificateThresholds, certify_v7_render, extreme_distortion_row_mask
    from anygsloc.mapping.virtual_camera_registry import camera_intrinsic
    root = Path(root); p = protocol(root); s = p["scenes"][scene]
    out = root / "observations" / scene
    out.mkdir(parents=True, exist_ok=False)
    dataset = ColmapDataset(s["dataset"], load_masks=False)
    cameras = resolve_virtual_camera_registry(dataset.split("mapping"), s["registry"])
    prior = s["prior"]
    if sha256(prior["ply"]) != prior["ply_sha256"]:
        raise ValueError("prior changed")
    cls = GaussianModel2D if prior["gaussian_type"] == "2dgs" else GaussianModel3D
    model = cls(3, device=device); model.load_ply(prior["ply"], loc_feature_dim=0); model = model.to(device).eval()
    detector = FeatureExtractor("sp", nms_radius=4).to(device).eval()
    all_records = {m: {} for m in MODES}
    thresholds = CertificateThresholds()
    start = time.perf_counter()
    for camera in cameras:
        rgb = safe_mapping_image(dataset, camera, s).to(device)
        pose = torch.from_numpy(camera.pose_w2c).float()
        pkg = render_from_pose_gsplat(model, pose.to(device), camera.fov_x, camera.fov_y,
              camera.width, camera.height, bg_color=torch.zeros(3, device=device),
              render_mode="RGB+ED", rgb_only=True, rasterize_mode="antialiased")
        rendered = pkg["render"][:3].float().clamp(0, 1)
        alpha = pkg.get("alphas", pkg.get("rend_alpha")); depth = pkg["depth"]
        if alpha is None:
            raise ValueError("renderer alpha missing")
        records = []
        for source, image in enumerate((rgb, rendered)):
            sparse = detector.detectAndCompute(image[None], top_k=2048, detection_threshold=0.)[0]
            xy = sparse["keypoints"].float()
            distortion = pkg.get("rend_dist")
            artifact = None if distortion is None else extreme_distortion_row_mask(
                distortion, xy, mad_multiplier=thresholds.distortion_mad_multiplier,
                tail_quantile=thresholds.distortion_tail_quantile)
            cert = certify_v7_render(rgb=rendered, alpha=alpha, depth=depth, keypoints=xy,
                   nearest_mapping_distance_m=0., median_adjacent_baseline_m=1.,
                   source_family_support=1, artifact_row_mask=artifact, thresholds=thresholds)
            record = {"native_keypoints": xy.cpu(),
                      "native_descriptors": F.normalize(sparse["descriptors"].cpu().float(), dim=1),
                      "native_scores": sparse["keypoint_scores"].cpu().float(),
                      "native_K": camera_intrinsic(camera), "pose_w2c": pose,
                      "native_input_hw": torch.tensor([camera.height, camera.width]),
                      "row_source": torch.full((len(xy),), source, dtype=torch.long),
                      "source_row": torch.arange(len(xy)),
                      "v2_valid": cert["row_valid"].cpu(),
                      "native_depth_at_keypoints": sample_raster_at_grid_uv(depth.squeeze(), xy).cpu(),
                      "native_alpha_at_keypoints": sample_raster_at_grid_uv(alpha.squeeze(), xy).cpu()}
            records.append(record)
        real, gs = records
        raw = {k: v for k, v in real.items() if k not in AUXILIARY}
        all_records["raw_only"][camera.image_name] = raw
        all_records["pure_gs"][camera.image_name] = gs
        all_records["hybrid"][camera.image_name] = merge_camera(real, gs)
        if len(all_records["raw_only"]) % 50 == 0:
            print("extract", scene, len(all_records["raw_only"]), "/", len(cameras), flush=True)
    if sha256(prior["ply"]) != prior["ply_sha256"]:
        raise ValueError("prior changed during extraction")
    for mode in MODES:
        save_new(out / (mode + ".pt"), payload(mode, all_records[mode], s, sha256(root / "protocol.json")))
    write_new(out / "summary.json", {"complete": True, "cameras": len(cameras),
              "wall_seconds": time.perf_counter() - start,
              "rows": {m: sum(len(r["native_keypoints"]) for r in records.values())
                       for m, records in all_records.items()},
              "hybrid_full_available_rows": sum(r["available_real"] + r["available_render"]
                                               for r in all_records["hybrid"].values()),
              "cache_sha256": {m: sha256(out / (m + ".pt")) for m in MODES}})

def assert_unique_track_cameras(csr):
    offsets = torch.as_tensor(csr["observation_offsets"]).tolist()
    queries = torch.as_tensor(csr["query_indices"])
    for a, b in zip(offsets[:-1], offsets[1:]):
        if len(torch.unique(queries[a:b])) != b-a:
            raise ValueError("track contains repeated physical camera")

def construct_any(data, device="cpu", workers=1):
    data = copy.deepcopy(data)
    if data["input_mode"] != "raw_only":
        for name, r in list(data["queries"].items()):
            rows = torch.nonzero(r["v2_valid"], as_tuple=False).reshape(-1)
            r = subset_record(r, rows)
            r["native_valid_keypoint_mask"] = torch.ones(len(rows), dtype=torch.bool)
            data["queries"][name] = r
    provider = MatrixObservationProvider(data)
    association = build_projective_association_graph(provider, device=device)
    base = reconstruct_projective_anchors(provider, association, parallel_workers=workers)
    parts = [base]; completion_reason = "disabled_without_Gaussian" if data["input_mode"] == "raw_only" else None
    if data["input_mode"] != "raw_only":
        try:
            parts.append(build_projective_completion(provider, association, voxel_size_m=.05,
                         alpha_minimum=.05, minimum_similarity=.7, device=device))
        except ValueError as e:
            if str(e) not in ("no unused render-valid observations for completion",
                              "depth proposals produced no descriptor-consistent component",
                              "association graph contains no ray-triangulated Anchor"):
                raise
            completion_reason = str(e)
    for part in parts:
        assert_unique_track_cameras(part["projective_anchor_observations"])
    return torch.cat([x["anchor_xyz"] for x in parts]).numpy(), torch.cat(
        [x["anchor_features"] for x in parts]).numpy(), {
        "association": association, "parts": parts, "completion_reason": completion_reason,
        "v2_enabled": data["input_mode"] != "raw_only", "filtered_observations": data}

def construct_classic(data, device="cpu", workers=1):
    from anygsloc.mapping.camera_pair_policy import candidate_camera_pairs
    from anygsloc.mapping.triangulation import reciprocal_epipolar_matches
    records = classic_rgb_records(data)
    poses = torch.stack([r["pose_w2c"] for r in records])
    K = torch.stack([r["native_K"] for r in records])
    pairs = candidate_camera_pairs(poses, neighbors=6, minimum_baseline_m=.03,
             maximum_baseline_m=5., maximum_axis_angle_deg=75., policy="nearest")
    descriptors = [r["native_descriptors"].to(device) for r in records]
    matches = []
    for a, b in pairs:
        x, y, confidence = reciprocal_epipolar_matches(descriptors[a], descriptors[b],
            records[a]["native_keypoints"]+.5, records[b]["native_keypoints"]+.5,
            K[a], poses[a], K[b], poses[b], minimum_similarity=.65,
            minimum_margin=.01, maximum_epipolar_error_px=2., epipolar_candidate_topk=1)
        matches.append((a, b, x.cpu().numpy(), y.cpu().numpy(), confidence.cpu().numpy()))
    tracks, stats = assemble_tracks(matches, [len(r["native_keypoints"]) for r in records])
    def fit(track):
        q, k = track
        uv = np.stack([records[a]["native_keypoints"][b].numpy()+.5 for a, b in zip(q, k)])
        try:
            fitted = fit_point(uv, K[q].numpy(), poses[q].numpy())
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return None
        if fitted is None:
            return None
        xyz, keep, error = fitted
        desc = np.stack([records[q[j]]["native_descriptors"][k[j]].numpy() for j in keep])
        desc /= np.maximum(np.linalg.norm(desc, axis=1, keepdims=True), 1e-12)
        feature = desc.mean(0); norm = np.linalg.norm(feature)
        if norm < 1e-12:
            return None
        return xyz, feature / norm, q[keep], k[keep], error
    with ThreadPoolExecutor(max_workers=workers) as pool:
        accepted = [x for x in pool.map(fit, tracks) if x is not None]
    if not accepted:
        raise ValueError("no valid classic landmarks")
    offsets = np.r_[0, np.cumsum([len(x[2]) for x in accepted])]
    csr = {"observation_offsets": offsets, "query_indices": np.concatenate([x[2] for x in accepted]),
           "keypoint_indices": np.concatenate([x[3] for x in accepted])}
    assert_unique_track_cameras(csr)
    return np.stack([x[0] for x in accepted]), np.stack([x[1] for x in accepted]), {
        "stats": stats, "pairs": len(pairs), "csr": csr,
        "uses_depth": False, "uses_alpha": False, "uses_semantics": False}

def build(root, scene, mode, method, device, workers):
    root = Path(root); p = protocol(root); scene_info = p["scenes"][scene]
    source = root / "observations" / scene / (mode + ".pt")
    data = torch.load(source, map_location="cpu", weights_only=False)
    if data["protocol_sha256"] != sha256(root / "protocol.json") or data["mapping_names"] != scene_info["mapping_names"]:
        raise ValueError("cache lineage differs from frozen protocol")
    if read(source.parent / "summary.json")["cache_sha256"][mode] != sha256(source):
        raise ValueError("observation cache changed")
    out = root / "maps" / scene / mode / method
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    fn = construct_any if method == "anygsloc" else construct_classic
    xyz, features, evidence = fn(data, device=device, workers=workers)
    provenance = {"experiment": DESIGN["schema"], "input_mode": mode, "method": method,
                  "uses_source_mapping_rgb": mode != "pure_gs", "uses_test_queries": False,
                  "protocol_sha256": sha256(root / "protocol.json"), "cache_sha256": sha256(source)}
    full = save_map(out / "full/deployment_map.pt", xyz, features, np.arange(len(xyz)), provenance)
    save_new(out / "construction.pt", evidence)
    write_new(out / "summary.json", {"complete": True, "full": full,
              "wall_seconds": time.perf_counter() - start, "provenance": provenance})

def common_rows(counts, reference_k):
    if len(counts) != 6 or min(counts) < 1 or reference_k < 1:
        raise ValueError("all six nonempty full maps are required")
    return min(reference_k, *counts)

def select(root, scene):
    root = Path(root); p = protocol(root)
    cells = [(mode, method) for mode in MODES for method in METHODS]
    loaded = []
    for mode, method in cells:
        folder = root / "maps" / scene / mode / method
        info = read(folder / "summary.json")
        path = folder / "full/deployment_map.pt"
        if sha256(path) != info["full"]["sha256"]:
            raise ValueError("full map changed")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["provenance"]["protocol_sha256"] != sha256(root / "protocol.json"):
            raise ValueError("full-map protocol mismatch")
        loaded.append((folder, state))
    k = common_rows([len(s["anchor_xyz"]) for _, s in loaded], p["scenes"][scene]["reference_K"])
    summaries = {}
    for (mode, method), (folder, state) in zip(cells, loaded):
        xyz, features = state["anchor_xyz"].numpy(), state["anchor_features"].numpy()
        rows = spatial_rows(xyz, k)
        info = save_map(folder / "common_spatial/deployment_map.pt", xyz, features, rows,
                        dict(state["provenance"], selection=DESIGN["selection"], common_K=k))
        summaries[mode + "/" + method] = info
    write_new(root / "selection" / (scene + ".json"), {
        "common_K": k, "reference_K": p["scenes"][scene]["reference_K"],
        "all_six_full_counts": {m+"/"+n: len(s["anchor_xyz"]) for (m,n),(_,s) in zip(cells,loaded)},
        "maps": summaries})

def effective_mask_path(dataset_root):
    for path in (Path(dataset_root)/"processed/masks.pkl", Path(dataset_root)/"masks.pkl"):
        if path.is_file():
            return path.resolve()
    raise ValueError("semantic mask file is missing")

def freeze_evaluation_snapshot(path, snapshot):
    path = Path(path)
    if path.exists():
        if read(path) != snapshot:
            raise ValueError("evaluation RGB/camera/mask snapshot changed between cells")
    else:
        write_new(path, snapshot)
    return sha256(path)

def evaluation_snapshot(root, scene, dataset, scene_info):
    """Only evaluation calls this function; mapping never reads test RGB."""
    from anygsloc.mapping.virtual_camera_registry import camera_intrinsic
    from anygsloc.data.calibration import resolve_scene_calibration
    mapping = dataset.split("mapping")
    resolve_virtual_camera_registry(mapping, scene_info["registry"])
    cameras = dataset.split("test")
    if [c.image_name for c in cameras] != scene_info["test_names"]:
        raise ValueError("evaluation query registry changed")
    if effective_mask_path(dataset.root) != Path(scene_info["mask_path"]).resolve():
        raise ValueError("effective semantic mask path changed")
    current_mask = sha256(scene_info["mask_path"])
    if current_mask != scene_info["mask_sha256"]:
        raise ValueError("evaluation masks changed")
    snapshot = {
        "protocol_sha256": sha256(Path(root)/"protocol.json"),
        "mapping_registry": scene_info["registry"],
        "calibration": resolve_scene_calibration(mapping),
        "mask_sha256": current_mask,
        "queries": [{"name": c.image_name, "pose_w2c": c.pose_w2c.tolist(),
                     "intrinsic": camera_intrinsic(c).tolist(), "image_hw": [c.height,c.width],
                     "image_sha256": sha256(c.image_path)} for c in cameras],
    }
    return freeze_evaluation_snapshot(Path(root)/"query_inputs"/(scene+".json"), snapshot)

def evaluate(root, scene, mode, method, selection, device):
    from anygsloc.evaluation.runner import main as run
    root = Path(root); p = protocol(root); s = p["scenes"][scene]
    if sha256(s["mask_path"]) != s["mask_sha256"]:
        raise ValueError("query semantic masks changed")
    dataset = ColmapDataset(s["dataset"], load_masks=True)
    cameras = dataset.split("test")
    if [c.image_name for c in cameras] != s["test_names"] or any(dataset.valid_mask(c) is None for c in cameras):
        raise ValueError("semantic-on query registry or masks incomplete")
    query_inputs_sha = evaluation_snapshot(root, scene, dataset, s)
    map_path = root / "maps" / scene / mode / method / selection / "deployment_map.pt"
    if selection == "common_spatial":
        expected = read(root / "selection" / (scene+".json"))["maps"][mode+"/"+method]["sha256"]
    else:
        expected = read(map_path.parents[1] / "summary.json")["full"]["sha256"]
    if sha256(map_path) != expected:
        raise ValueError("evaluation map changed")
    output = root / "evaluations" / scene / mode / method / selection
    run(["--dataset", s["dataset"], "--map", str(map_path), "--output", str(output),
         "--image-domain", "rectified", "--images", "processed", "--mask-policy", "explicit_valid_mask",
         "--rectification-contract", "opencv_same_k", "--seed", "2026", "--device", device])
    if evaluation_snapshot(root, scene, ColmapDataset(s["dataset"], load_masks=False), s) != query_inputs_sha:
        raise ValueError("evaluation inputs changed during localization")
    write_new(output / "input_matrix_attestation.json", {
        "protocol_sha256": sha256(root / "protocol.json"), "query_semantics": True,
        "mask_sha256": s["mask_sha256"], "map_sha256": expected,
        "query_inputs_sha256": query_inputs_sha,
        "output_sha256": {name: sha256(output/name) for name in ("results.json","manifest.json","summary.json")}})

def report(root):
    from anygsloc.evaluation.runner import summarize, json_safe
    root = Path(root); protocol(root)
    results = {}
    for mode in MODES:
        for method in METHODS:
            for selection in ("full", "common_spatial"):
                scenes = {}
                for scene in SCENES:
                    folder = root / "evaluations" / scene / mode / method / selection
                    a = read(folder / "input_matrix_attestation.json")
                    if a["query_semantics"] is not True or a["protocol_sha256"] != sha256(root / "protocol.json"):
                        raise ValueError("mixed query protocol")
                    if a["query_inputs_sha256"] != sha256(root/"query_inputs"/(scene+".json")):
                        raise ValueError("mixed query RGB/camera snapshot")
                    if any(sha256(folder/name) != expected for name,expected in a["output_sha256"].items()):
                        raise ValueError("evaluation output changed after attestation")
                    rows = read(folder / "results.json")
                    if [x["image_name"] for x in rows] != read(root/"protocol.json")["scenes"][scene]["test_names"]:
                        raise ValueError("query result registry differs")
                    scenes[scene] = summarize(rows, read(folder/"summary.json")["loop_wall_seconds"])
                    errors = np.asarray([r["translation_error_cm"] if r["success"] else float("inf") for r in rows])
                    scenes[scene]["p95_te_cm"] = float(np.percentile(errors,95))
                    construction = read(root/"maps"/scene/mode/method/"summary.json")
                    map_info = (construction["full"] if selection == "full" else
                        read(root/"selection"/(scene+".json"))["maps"][mode+"/"+method])
                    scenes[scene].update(map_landmarks=map_info["count"],map_bytes=map_info["bytes"],
                                         construction_seconds=construction["wall_seconds"])
                metrics = ["median_te_cm", "median_ae_deg", "recall_5cm_5deg_percent",
                           "recall_15cm_5deg_percent", "p90_te_cm", "p95_te_cm", "failed_query_count",
                           "catastrophic_100cm_count"]
                results[mode+"/"+method+"/"+selection] = {"per_scene": scenes, "scene_equal": {
                    k: float(np.mean([float("inf") if x[k] is None else x[k] for x in scenes.values()]))
                    for k in metrics}}
    write_new(root / "report.json", json_safe({"complete": True, "results": results}))

def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("command", choices=("design", "prepare", "extract", "build", "select", "evaluate", "report", "check-kernels"))
    a.add_argument("--root", type=Path)
    a.add_argument("--reference", type=Path)
    a.add_argument("--prior-manifest", type=Path)
    a.add_argument("--scene", choices=SCENES)
    a.add_argument("--input-mode", choices=MODES)
    a.add_argument("--method", choices=METHODS)
    a.add_argument("--selection", choices=("full", "common_spatial"))
    a.add_argument("--device", default="cuda:0")
    a.add_argument("--workers", type=int, default=4)
    args = a.parse_args(argv)
    if args.command == "design":
        print(json.dumps(DESIGN, indent=2)); return
    if args.command == "check-kernels":
        check_kernel_adaptations(); print("kernel adapters match exact registered changes"); return
    needed = {"prepare": ("root", "reference", "prior_manifest"), "extract": ("root", "scene"),
              "build": ("root", "scene", "input_mode", "method"), "select": ("root", "scene"),
              "evaluate": ("root", "scene", "input_mode", "method", "selection"), "report": ("root",)}
    for key in needed[args.command]:
        if getattr(args, key) is None:
            a.error("--"+key.replace("_", "-")+" is required")
    if args.workers < 1 or args.workers > 8:
        a.error("--workers must be 1..8")
    torch.set_num_threads(4)
    if args.command == "prepare": prepare(args.root, args.reference, args.prior_manifest)
    elif args.command == "extract": extract(args.root, args.scene, args.device)
    elif args.command == "build": build(args.root, args.scene, args.input_mode, args.method, args.device, args.workers)
    elif args.command == "select": select(args.root, args.scene)
    elif args.command == "evaluate": evaluate(args.root, args.scene, args.input_mode, args.method, args.selection, args.device)
    elif args.command == "report": report(args.root)

if __name__ == "__main__":
    main()
