"""Current Cambridge construction, with explicit observation and fusion controls."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from anygsloc.artifacts import SCHEMA, validate
from anygsloc.mapping.compact_v2 import select_compact_rows_v2
from anygsloc.mapping.projective_association import build_projective_association_graph
from anygsloc.mapping.projective_completion import build_projective_completion
from anygsloc.mapping.projective_reconstruction import reconstruct_projective_anchors
from anygsloc.pipeline.controls import quality_rows, single_descriptors, spatial_rows
from anygsloc.pipeline.io import complete, prepare_output, reference, require_complete
from anygsloc.pipeline.observations import AUXILIARY, SPARSE, MatrixObservationProvider
from anygsloc.pipeline.selector_state import (
    adapt_selector_state,
    assert_unique_track_cameras,
)

EMPTY_COMPLETION = (
    "no unused render-valid observations for completion",
    "depth proposals produced no descriptor-consistent component",
    "association graph contains no ray-triangulated Anchor",
)


def subset_record(record, rows):
    result = dict(record)
    for key in SPARSE + (
        "row_source",
        "source_row",
        "v2_valid",
        "native_depth_at_keypoints",
        "native_alpha_at_keypoints",
        "native_valid_keypoint_mask",
    ):
        if key in result:
            result[key] = result[key][rows]
    return result


def observation_payload(mode, records, names, tests):
    return {
        "input_mode": mode,
        "uses_source_mapping_rgb": mode != "pure_gs",
        "uses_test_queries": False,
        "mapping_names": names,
        "test_names": tests,
        "queries": records,
    }


def extract(config, scene, device):
    """Detect on complete RGB, then certify rows; pure-GS never reads mapping RGB."""
    from anygsloc.data.datasets import ColmapDataset
    from anygsloc.features.extractor import FeatureExtractor
    from anygsloc.features.raster_sampling import sample_raster_at_grid_uv
    from anygsloc.mapping.render_certificate import (
        CertificateThresholds,
        certify_v7_render,
        extreme_distortion_row_mask,
    )
    from anygsloc.mapping.virtual_camera_registry import (
        build_virtual_camera_registry,
        camera_intrinsic,
    )
    from anygsloc.priors.models import GaussianModel2D, GaussianModel3D
    from anygsloc.priors.rendering import render_from_pose_gsplat

    spec = config["scenes"][scene]
    modes = config.get("input_modes", ["pure_gs"])
    if not set(modes) <= {"pure_gs", "raw_only", "hybrid"}:
        raise ValueError("unsupported input mode")
    out = prepare_output(Path(config["output"]) / scene / "observations", spec)
    data = ColmapDataset(
        spec["dataset"], images=spec.get("images", "processed"), load_masks=False
    )
    cameras, registry = build_virtual_camera_registry(data.split("mapping"))
    names = [c.image_name for c in cameras]
    tests = [c.image_name for c in data.split("test")]
    if not cameras or set(names) & set(tests):
        raise ValueError("empty or overlapping mapping split")
    need_render = any(m != "raw_only" for m in modes)
    need_real = any(m != "pure_gs" for m in modes)
    records = {mode: {} for mode in modes}
    real_auxiliary = {}
    thresholds = CertificateThresholds()
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("render/feature extraction requires CUDA")
    with torch.cuda.device(device), torch.inference_mode():
        detector = FeatureExtractor("sp", nms_radius=4).to(device).eval()
        model = None
        if need_render:
            cls = (
                GaussianModel2D if spec["gaussian_type"] == "2dgs" else GaussianModel3D
            )
            model = cls(spec.get("sh_degree", 3), device=device)
            model.load_ply(spec["prior"], loc_feature_dim=0)
            model = model.to(device).eval()
        for index, camera in enumerate(cameras):
            pkg = None
            if need_render:
                pkg = render_from_pose_gsplat(
                    model,
                    torch.from_numpy(camera.pose_w2c).to(device),
                    camera.fov_x,
                    camera.fov_y,
                    camera.width,
                    camera.height,
                    bg_color=torch.full(
                        (3,), float(spec.get("white_background", False)), device=device
                    ),
                    render_mode="RGB+ED",
                    rgb_only=True,
                    rasterize_mode="antialiased",
                )
            images = {}
            if need_real:
                images[0] = data.load_image(camera).to(device)
            if need_render:
                images[1] = pkg["render"][:3].float().clamp(0, 1)
            current = {}
            for source, image in images.items():
                sparse = detector.detectAndCompute(
                    image[None], top_k=2048, detection_threshold=0.0
                )[0]
                xy = sparse["keypoints"].float()
                row = {
                    "native_keypoints": xy.cpu(),
                    "native_descriptors": F.normalize(
                        sparse["descriptors"].cpu().float(), dim=1
                    ),
                    "native_scores": sparse["keypoint_scores"].cpu().float(),
                    "native_K": camera_intrinsic(camera),
                    "pose_w2c": torch.from_numpy(camera.pose_w2c).float(),
                    "native_input_hw": torch.tensor([camera.height, camera.width]),
                    "row_source": torch.full((len(xy),), source, dtype=torch.long),
                    "source_row": torch.arange(len(xy)),
                }
                if need_render:
                    alpha = pkg.get("alphas", pkg.get("rend_alpha"))
                    depth = pkg["depth"]
                    distortion = pkg.get("rend_dist")
                    artifact = (
                        None
                        if distortion is None
                        else extreme_distortion_row_mask(
                            distortion,
                            xy,
                            mad_multiplier=thresholds.distortion_mad_multiplier,
                            tail_quantile=thresholds.distortion_tail_quantile,
                        )
                    )
                    cert = certify_v7_render(
                        rgb=images[1],
                        alpha=alpha,
                        depth=depth,
                        keypoints=xy,
                        nearest_mapping_distance_m=0.0,
                        median_adjacent_baseline_m=1.0,
                        source_family_support=1,
                        artifact_row_mask=artifact,
                        thresholds=thresholds,
                    )
                    row.update(
                        v2_valid=cert["row_valid"].cpu(),
                        native_depth_at_keypoints=sample_raster_at_grid_uv(
                            depth.squeeze(), xy
                        ).cpu(),
                        native_alpha_at_keypoints=sample_raster_at_grid_uv(
                            alpha.squeeze(), xy
                        ).cpu(),
                    )
                current[source] = row
            if need_real and need_render:
                real = current[0]
                real_auxiliary[camera.image_name] = {
                    **{
                        k: real[k]
                        for k in (
                            "native_keypoints",
                            "native_K",
                            "pose_w2c",
                            "native_input_hw",
                        )
                    },
                    "depth": real["native_depth_at_keypoints"],
                    "alpha": real["native_alpha_at_keypoints"],
                    "certificate_valid": real["v2_valid"],
                }
            if "pure_gs" in modes:
                records["pure_gs"][camera.image_name] = current[1]
            if "raw_only" in modes:
                records["raw_only"][camera.image_name] = {
                    k: v for k, v in current[0].items() if k not in AUXILIARY
                }
            if "hybrid" in modes:
                parts = [
                    subset_record(
                        current[k],
                        torch.argsort(
                            current[k]["native_scores"], descending=True, stable=True
                        )[:1024],
                    )
                    for k in (0, 1)
                ]
                merged = dict(parts[0])
                for key in SPARSE + (
                    "row_source",
                    "source_row",
                    "v2_valid",
                    "native_depth_at_keypoints",
                    "native_alpha_at_keypoints",
                ):
                    merged[key] = torch.cat([p[key] for p in parts])
                records["hybrid"][camera.image_name] = merged
            if (index + 1) % 50 == 0:
                print(
                    f"{scene}: extracted {index + 1}/{len(cameras)} reference views",
                    flush=True,
                )
    for mode in modes:
        torch.save(
            observation_payload(mode, records[mode], names, tests), out / f"{mode}.pt"
        )
    outputs = [f"{mode}.pt" for mode in modes]
    if real_auxiliary:
        torch.save(
            {
                "mapping_names": names,
                "uses_test_queries": False,
                "queries": real_auxiliary,
            },
            out / "real_auxiliary.pt",
        )
        outputs.append("real_auxiliary.pt")
    complete(
        out,
        outputs,
        registry=registry,
        prior=reference(spec["prior"]) if need_render else None,
    )


def construct(data, *, filtering=True, device="cpu", workers=1):
    filtering = filtering and data["input_mode"] != "raw_only"
    original = dict(data, filter_enabled=filtering)
    filtered = dict(original, queries={})
    for name, record in data["queries"].items():
        rows = (
            torch.nonzero(record["v2_valid"], as_tuple=False).flatten()
            if filtering
            else torch.arange(len(record["native_keypoints"]))
        )
        value = subset_record(record, rows)
        if filtering:
            value["native_valid_keypoint_mask"] = torch.ones(
                len(rows), dtype=torch.bool
            )
        filtered["queries"][name] = value
    provider = MatrixObservationProvider(filtered)
    association = build_projective_association_graph(
        provider, device=device, require_render_valid=filtering
    )
    parts = [
        reconstruct_projective_anchors(provider, association, parallel_workers=workers)
    ]
    reason = "disabled_without_Gaussian" if data["input_mode"] == "raw_only" else None
    if data["input_mode"] != "raw_only":
        try:
            parts.append(
                build_projective_completion(
                    provider,
                    association,
                    voxel_size_m=0.05,
                    alpha_minimum=0.05,
                    minimum_similarity=0.7,
                    device=device,
                )
            )
        except ValueError as error:
            if str(error) not in EMPTY_COMPLETION:
                raise
            reason = str(error)
    for part in parts:
        assert_unique_track_cameras(part["projective_anchor_observations"])
    evidence = {
        "association": association,
        "parts": parts,
        "completion_reason": reason,
        "v2_enabled": filtering,
        "filtered_observations": filtered,
    }
    deployed = {
        "anchor_xyz": torch.cat([p["anchor_xyz"] for p in parts]),
        "anchor_features": torch.cat([p["anchor_features"] for p in parts]),
    }
    state = adapt_selector_state(evidence, original, deployed)
    return state, evidence


def save_map(path, state, rows=None, features=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = (
        torch.arange(len(state["anchor_xyz"]))
        if rows is None
        else torch.as_tensor(rows).long()
    )
    feature = state["anchor_features"] if features is None else features
    output = {
        "schema": SCHEMA,
        "anchor_ids": torch.arange(len(rows)),
        "anchor_xyz": state["anchor_xyz"][rows].float().clone(),
        "anchor_features": feature[rows].float().clone(),
        "provenance": dict(state["provenance"], uses_test_queries=False),
    }
    validate(output)
    with path.open("xb") as stream:
        torch.save(output, stream)
    np.save(path.with_suffix(".source_rows.npy"), rows.numpy())
    return output


def build(config, scene, mode, filtering, device, workers):
    root = Path(config["output"]) / scene
    require_complete(root / "observations")
    cache = root / "observations" / f"{mode}.pt"
    out = prepare_output(
        root / mode / ("filtered" if filtering else "unfiltered"),
        {"cache": reference(cache), "filtering": filtering},
    )
    data = torch.load(cache, map_location="cpu", weights_only=False)
    state, evidence = construct(
        data, filtering=filtering, device=device, workers=workers
    )
    torch.save(state, out / "selector_state.pt")
    torch.save(evidence, out / "construction.pt")
    save_map(out / "full.pt", state)
    single, chosen = single_descriptors(
        state["projective_anchor_observations"], data["mapping_names"], data["queries"]
    )
    save_map(out / "single.pt", state, features=single)
    np.save(out / "single_observations.npy", chosen)
    complete(
        out,
        ["selector_state.pt", "construction.pt", "full.pt", "single.pt"],
        points=len(state["anchor_xyz"]),
    )


def select(root, cache):
    root = Path(root)
    require_complete(root)
    state = torch.load(
        root / "selector_state.pt", map_location="cpu", weights_only=False
    )
    data = torch.load(cache, map_location="cpu", weights_only=False)
    out = prepare_output(
        root / "selection",
        {"state": reference(root / "selector_state.pt"), "cache": reference(cache)},
    )
    rows, trace = select_compact_rows_v2(state, data["queries"])
    phases = trace["trace"]
    quality_rows_stage = sorted(
        r["candidate_universe_id"] for r in phases if r["primary_reason"] == "precision"
    )
    coverage_rows = sorted(
        r["candidate_universe_id"]
        for r in phases
        if r["primary_reason"] in ("precision", "matching_completion")
    )
    n = len(rows)
    controls = {
        "compact": rows,
        "quality_stage": quality_rows_stage,
        "coverage_stage": coverage_rows,
        "spatial": spatial_rows(state["anchor_xyz"].numpy(), n),
        "quality_topk": quality_rows(state["anchor_matchability"].numpy(), n),
        "tracklength_topk": quality_rows(
            np.diff(
                state["projective_anchor_observations"]["observation_offsets"].numpy()
            ),
            n,
        ),
    }
    for name, selected in controls.items():
        save_map(out / f"{name}.pt", state, selected)
    single = torch.load(root / "single.pt", map_location="cpu", weights_only=True)[
        "anchor_features"
    ]
    save_map(out / "compact_single.pt", state, rows, single)
    torch.save(trace, out / "trace.pt")
    complete(
        out,
        [f"{name}.pt" for name in controls] + ["compact_single.pt", "trace.pt"],
        points=n,
    )


def build_assisted(root, device="cuda:0"):
    from anygsloc.pipeline.assisted import construct as assisted_construct

    root = Path(root)
    base = root / "raw_only/filtered"
    require_complete(base)
    raw_path = root / "observations/raw_only.pt"
    auxiliary_path = root / "observations/real_auxiliary.pt"
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    auxiliary = torch.load(auxiliary_path, map_location="cpu", weights_only=False)
    state = torch.load(
        base / "selector_state.pt", map_location="cpu", weights_only=False
    )
    evidence = torch.load(
        base / "construction.pt", map_location="cpu", weights_only=False
    )
    full = torch.load(base / "full.pt", map_location="cpu", weights_only=True)
    out = prepare_output(
        root / "real_gs_assisted/filtered",
        {"base": reference(base / "full.pt"), "auxiliary": reference(auxiliary_path)},
    )
    result, details = assisted_construct(
        raw,
        auxiliary,
        evidence["association"],
        state,
        full,
        reference(raw_path)["sha256"],
        device,
    )
    torch.save(result, out / "selector_state.pt")
    torch.save(details, out / "construction.pt")
    save_map(out / "full.pt", result)
    rows, trace = select_compact_rows_v2(result, raw["queries"])
    save_map(out / "selection/compact.pt", result, rows)
    torch.save(trace, out / "selection/trace.pt")
    complete(
        out,
        [
            "selector_state.pt",
            "construction.pt",
            "full.pt",
            "selection/compact.pt",
            "selection/trace.pt",
        ],
    )
    complete(
        out,
        [
            "full.pt",
            "selector_state.pt",
            "construction.pt",
            "selection/compact.pt",
            "selection/trace.pt",
        ],
        base_points=len(state["anchor_xyz"]),
        added_points=len(result["anchor_xyz"]) - len(state["anchor_xyz"]),
    )
