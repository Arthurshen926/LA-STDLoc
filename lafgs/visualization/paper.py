"""Paper figures generated only from frozen configs and formal artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lafgs.protocol import MainlineProtocol


FIGURE_SCHEMA = "lafgs_paper_figure"
FIGURE_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _result_rows(result_path: str | Path) -> dict[str, dict]:
    path = Path(result_path)
    if path.is_dir():
        path = path / "results.json"
    rows = _load_json(path)
    if isinstance(rows, dict):
        rows = rows["results"]
    return {
        str(row["image_name"]).replace("\\", "/"): row for row in rows
    }


def _stage_result_file(
    frozen_results: Mapping[str, Any], stage: str, seed: int
) -> Path:
    stage_results = frozen_results["results"][stage]
    record = stage_results.get(str(seed), stage_results.get(seed))
    if record is None:
        raise KeyError(f"missing {stage} seed {seed}")
    path = Path(record["result_path"])
    return path / "results.json" if path.is_dir() else path


def _metric(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get("sparse", {}).get(key, default)
    return float(value)


def select_qualitative_query(
    a0_rows: Mapping[str, Mapping[str, Any]],
    a1_rows: Mapping[str, Mapping[str, Any]],
    prior_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Select a weak-render, dirty-A0, improved-A1 example deterministically."""

    names = sorted(set(a0_rows) & set(a1_rows) & set(prior_rows))
    if not names:
        raise ValueError("A0, A1 and prior-quality registries do not overlap")
    a1_precision = np.asarray(
        [
            100.0
            * _metric(a1_rows[name], "sparse_diag_all_gt_precision_2px")
            for name in names
        ]
    )
    a1_te = np.asarray([float(a1_rows[name]["sparse_TE"]) for name in names])
    prior_psnr = np.asarray(
        [float(prior_rows[name]["psnr_db"]) for name in names]
    )
    precision_floor = float(np.percentile(a1_precision, 50))
    te_ceiling = float(np.percentile(a1_te, 50))
    psnr_ceiling = float(np.percentile(prior_psnr, 60))
    candidates = []
    for name in names:
        a0 = a0_rows[name]
        a1 = a1_rows[name]
        quality = prior_rows[name]
        te0 = float(a0["sparse_TE"])
        te1 = float(a1["sparse_TE"])
        p0 = 100.0 * _metric(a0, "sparse_diag_all_gt_precision_2px")
        p1 = 100.0 * _metric(a1, "sparse_diag_all_gt_precision_2px")
        psnr = float(quality["psnr_db"])
        gain = te0 - te1
        eligible = (
            gain > 0
            and te1 <= te_ceiling
            and p1 >= precision_floor
            and psnr <= psnr_ceiling
        )
        score = (
            2.0 * math.log1p(max(gain, 0.0))
            + 0.08 * max(p1 - p0, 0.0)
            + 0.04 * max(psnr_ceiling - psnr, 0.0)
        )
        candidates.append(
            {
                "image_name": name,
                "eligible": eligible,
                "score": score,
                "a0_te_cm": te0,
                "a1_te_cm": te1,
                "te_gain_cm": gain,
                "a0_raw_p2_percent": p0,
                "a1_raw_p2_percent": p1,
                "prior_psnr_db": psnr,
            }
        )
    eligible = [row for row in candidates if row["eligible"]]
    pool = eligible if eligible else candidates
    return max(pool, key=lambda row: (row["score"], row["image_name"]))


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    paths = (
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
    )
    for path in paths:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    scale = min(size[0] / image.width, size[1] / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", size, "white")
    canvas.paste(
        resized,
        ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2),
    )
    return canvas


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    if hasattr(mask, "detach"):
        tensor = mask.detach().cpu().float()
    else:
        tensor = torch.as_tensor(np.asarray(mask), dtype=torch.float32)
    while tensor.ndim > 2:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2:
        raise ValueError(
            f"expected a two-dimensional mask, got {tuple(tensor.shape)}"
        )
    resized = F.interpolate(
        tensor[None, None], size=(size[1], size[0]), mode="nearest"
    )[0, 0]
    return resized.bool().numpy()


def _valid_mask(mask_path: Path, image_name: str, size: tuple[int, int]) -> np.ndarray:
    with mask_path.open("rb") as handle:
        masks = pickle.load(handle)
    channels = masks.get(image_name)
    if channels is None:
        return np.ones((size[1], size[0]), dtype=bool)
    if len(channels) < 3:
        raise ValueError(f"invalid deployment mask for {image_name}")
    valid = np.ones((size[1], size[0]), dtype=bool)
    for channel in channels[:3]:
        valid &= _resize_mask(channel, size)
    return valid


def _camera_intrinsic(source: Path, image_name: str, size: tuple[int, int]) -> np.ndarray:
    from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary

    root = source / "sparse" / "0"
    extrinsics = read_extrinsics_binary(str(root / "images.bin"))
    intrinsics = read_intrinsics_binary(str(root / "cameras.bin"))
    matches = [value for value in extrinsics.values() if value.name == image_name]
    if len(matches) != 1:
        raise ValueError(f"expected one COLMAP camera for {image_name}, found {len(matches)}")
    camera = intrinsics[matches[0].camera_id]
    if camera.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        fx = fy = float(camera.params[0])
    elif camera.model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE"}:
        fx, fy = map(float, camera.params[:2])
    else:
        raise ValueError(f"unsupported camera model for paper figure: {camera.model}")
    source_width = float(camera.width)
    source_height = float(camera.height)
    fx *= size[0] / source_width
    fy *= size[1] / source_height
    return np.asarray(
        [[fx, 0.0, size[0] / 2.0], [0.0, fy, size[1] / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _project(points: np.ndarray, K: np.ndarray, pose_w2c: np.ndarray) -> np.ndarray:
    camera = points @ pose_w2c[:3, :3].T + pose_w2c[:3, 3]
    projected = camera @ K.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return projected[:, :2] / projected[:, 2:3]


@dataclass
class CorrespondenceSet:
    keypoints: np.ndarray
    landmark_xyz: np.ndarray
    scores: np.ndarray
    gt_error_px: np.ndarray
    inliers: np.ndarray
    reproduced_pose_w2c: np.ndarray

    def diagnostics(self) -> dict[str, Any]:
        mask = np.zeros(len(self.keypoints), dtype=bool)
        mask[self.inliers] = True
        clean = np.isfinite(self.gt_error_px) & (self.gt_error_px <= 2.0)
        return {
            "match_count": int(len(self.keypoints)),
            "inlier_count": int(mask.sum()),
            "raw_gt_precision_2px_percent": float(100.0 * clean.mean()),
            "inlier_gt_precision_2px_percent": float(
                100.0 * clean[mask].mean() if mask.any() else 0.0
            ),
        }


def _extract_native_query(
    image: Image.Image,
    valid_mask: np.ndarray,
    *,
    keypoint_count: int,
    device: str,
):
    import torch
    from encoders.sp_encoder.export_image_embeddings import SuperPoint
    from localization_training.ulf_initializer import sample_mask_at_grid_uv

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).to(device)
    mask = torch.from_numpy(valid_mask).to(device)
    tensor = tensor * mask[None].to(dtype=tensor.dtype)
    model = SuperPoint().to(device).eval()
    with torch.inference_mode():
        sparse = model.detectAndCompute(tensor[None], top_k=keypoint_count)[0]
        keep = sample_mask_at_grid_uv(mask, sparse["keypoints"])
        keypoints = sparse["keypoints"][keep]
        descriptors = torch.nn.functional.normalize(
            sparse["descriptors"][keep], dim=1
        )
    return keypoints, descriptors


def _match_map(
    keypoints,
    descriptors,
    *,
    map_path: Path,
    stage: str,
    metric_path: Path | None,
    K: np.ndarray,
    gt_pose_w2c: np.ndarray,
    ransac_seed: int,
) -> CorrespondenceSet:
    import torch
    import torch.nn.functional as F
    from localization_training.full_primitive_retrieval import chunked_exact_topk
    from localization_training.shared_metric import SharedLowRankMetric
    from utils.pose_utils import solve_pose

    state = torch.load(map_path, map_location="cpu", weights_only=False)
    if stage == "A0_bootstrap":
        xyz = torch.as_tensor(state["landmark_xyz"])
        features = torch.as_tensor(state["landmark_features"])
    elif stage == "A1_reconstructed":
        xyz = torch.as_tensor(state["anchor_xyz"])
        features = torch.as_tensor(state["anchor_features"])
        if metric_path is None:
            raise ValueError("A1 correspondence extraction requires metric state")
        metric_state = torch.load(metric_path, map_location="cpu", weights_only=False)
        metric = SharedLowRankMetric(**metric_state["metric_config"]).to(
            descriptors.device
        )
        metric.load_state_dict(metric_state["metric_state_dict"])
        metric.eval()
        with torch.inference_mode():
            descriptors, _ = metric(descriptors)
    else:
        raise ValueError(f"unsupported paper-figure stage: {stage}")
    features = F.normalize(features.to(descriptors.device), dim=1)
    retrieval = chunked_exact_topk(
        descriptors, features, topk=1, chunk_size=8192
    )
    indices = retrieval.indices[:, 0].detach().cpu()
    scores = retrieval.scores[:, 0].detach().cpu().numpy()
    points2d = keypoints.detach().cpu().numpy()
    points3d = xyz[indices].numpy()
    reproduced_pose, inliers = solve_pose(
        points2d + 0.5,
        points3d,
        K,
        solver="poselib",
        reprojection_error=12.0,
        confidence=0.99999,
        max_iterations=100000,
        min_iterations=1000,
        ransac_seed=ransac_seed,
    )
    projected = _project(points3d, K, gt_pose_w2c)
    gt_error = np.linalg.norm(projected - (points2d + 0.5), axis=1)
    return CorrespondenceSet(
        keypoints=points2d,
        landmark_xyz=points3d,
        scores=scores,
        gt_error_px=gt_error,
        inliers=np.asarray(inliers, dtype=np.int64).reshape(-1),
        reproduced_pose_w2c=np.asarray(reproduced_pose),
    )


def _draw_matches(
    image: Image.Image,
    correspondences: CorrespondenceSet,
    *,
    max_points: int = 700,
) -> Image.Image:
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    clean = np.isfinite(correspondences.gt_error_px) & (
        correspondences.gt_error_px <= 2.0
    )
    inlier = np.zeros(len(clean), dtype=bool)
    inlier[correspondences.inliers] = True
    priority = np.lexsort(
        (
            -correspondences.scores,
            ~clean,
            ~inlier,
        )
    )
    selected = priority[: min(max_points, len(priority))]
    for index in selected[::-1]:
        x, y = correspondences.keypoints[index]
        color = (22, 163, 74, 210) if clean[index] else (220, 38, 38, 145)
        radius = 3.4 if clean[index] else 2.5
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        if inlier[index]:
            ring = radius + 2.0
            draw.ellipse(
                (x - ring, y - ring, x + ring, y + ring),
                outline=(6, 182, 212, 235),
                width=2,
            )
    return Image.alpha_composite(canvas, overlay).convert("RGB")


def _panel(
    image: Image.Image,
    *,
    title: str,
    subtitle: str,
    footer: list[str],
    size: tuple[int, int] = (820, 650),
) -> Image.Image:
    header = 92
    foot = 100
    content = _fit_image(image, (size[0], size[1] - header - foot))
    panel = Image.new("RGB", size, "white")
    panel.paste(content, (0, header))
    draw = ImageDraw.Draw(panel)
    draw.text((24, 16), title, fill="#111827", font=_font(28, bold=True))
    draw.text((24, 54), subtitle, fill="#4b5563", font=_font(18))
    y = size[1] - foot + 14
    for line in footer:
        draw.text((24, y), line, fill="#1f2937", font=_font(17))
        y += 27
    draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline="#d1d5db", width=1)
    return panel


def _official_metrics(row: Mapping[str, Any]) -> dict[str, float]:
    sparse = row["sparse"]
    return {
        "te_cm": float(row["sparse_TE"]),
        "re_deg": float(row["sparse_AE"]),
        "matches": int(sparse["matches"]),
        "inliers": int(sparse["inliers"]),
        "raw_p2_percent": 100.0
        * float(sparse["sparse_diag_all_gt_precision_2px"]),
        "inlier_p2_percent": 100.0
        * float(sparse["sparse_diag_inlier_gt_precision_2px"]),
    }


def build_anysplat_a0_a1_figure(
    *,
    protocol: MainlineProtocol,
    lafgs_root: str | Path,
    prior_quality_path: str | Path,
    render_path: str | Path,
    query_camera_source: str | Path,
    query_image_root: str | Path,
    mask_path: str | Path,
    output_dir: str | Path,
    seed: int = 2026,
    query_name: str | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Build the prior/A0/A1 qualitative panel with parity checks."""

    import torch

    root = Path(lafgs_root).expanduser().resolve()
    prior_quality_path = Path(prior_quality_path).expanduser().resolve()
    prior_quality = _load_json(prior_quality_path)
    frozen_results_path = root / "frozen_results.json"
    frozen = _load_json(frozen_results_path)
    a0_result_path = _stage_result_file(frozen, "A0_bootstrap", seed)
    a1_result_path = _stage_result_file(frozen, "A1_reconstructed", seed)
    a0_rows = _result_rows(a0_result_path)
    a1_rows = _result_rows(a1_result_path)
    prior_rows = {
        str(row["image_name"]).replace("\\", "/"): row
        for row in prior_quality["per_view"]
    }
    selection = select_qualitative_query(a0_rows, a1_rows, prior_rows)
    selected_name = query_name or selection["image_name"]
    if selected_name not in a0_rows or selected_name not in a1_rows:
        raise KeyError(f"query is absent from formal A0/A1 results: {selected_name}")
    prior_evaluation_source = Path(prior_quality["evaluation_source"])
    query_camera_source = Path(query_camera_source).expanduser().resolve()
    query_image_root = Path(query_image_root).expanduser().resolve()
    image_path = query_image_root / selected_name
    image = Image.open(image_path).convert("RGB")
    valid = _valid_mask(Path(mask_path), selected_name, image.size)
    K = _camera_intrinsic(query_camera_source, selected_name, image.size)
    gt_pose = np.asarray(a0_rows[selected_name]["gt_pose_w2c"], dtype=np.float64)
    keypoints, descriptors = _extract_native_query(
        image,
        valid,
        keypoint_count=int(protocol.resolved["deployment"]["keypoints"]),
        device=device,
    )
    a0_map = root / "runs" / "frozen_v1" / "bootstrap" / "0_lafgs_map_state.pt"
    metric_steps = int(protocol.resolved["reconstruction"]["metric_steps"])
    a1_map = (
        root
        / "self_localization_reconstruction"
        / f"anchor_map_step_{metric_steps:04d}.pt"
    )
    metric = (
        root
        / "self_localization_reconstruction"
        / f"metric_state_step_{metric_steps:04d}.pt"
    )
    a0_corr = _match_map(
        keypoints,
        descriptors,
        map_path=a0_map,
        stage="A0_bootstrap",
        metric_path=None,
        K=K,
        gt_pose_w2c=gt_pose,
        ransac_seed=seed,
    )
    a1_corr = _match_map(
        keypoints,
        descriptors,
        map_path=a1_map,
        stage="A1_reconstructed",
        metric_path=metric,
        K=K,
        gt_pose_w2c=gt_pose,
        ransac_seed=seed,
    )
    a0_official = _official_metrics(a0_rows[selected_name])
    a1_official = _official_metrics(a1_rows[selected_name])
    a0_reproduced = a0_corr.diagnostics()
    a1_reproduced = a1_corr.diagnostics()
    for name, official, reproduced in (
        ("A0", a0_official, a0_reproduced),
        ("A1", a1_official, a1_reproduced),
    ):
        if official["matches"] != reproduced["match_count"]:
            raise RuntimeError(
                f"{name} match-count parity failed: official={official['matches']} "
                f"reproduced={reproduced['match_count']}"
            )
        if abs(official["raw_p2_percent"] - reproduced["raw_gt_precision_2px_percent"]) > 1e-4:
            raise RuntimeError(
                f"{name} raw-P@2 parity failed: official={official['raw_p2_percent']:.9f} "
                f"reproduced={reproduced['raw_gt_precision_2px_percent']:.9f}"
            )

    prior = Image.open(render_path).convert("RGB")
    a0_visual = _draw_matches(image, a0_corr)
    a1_visual = _draw_matches(image, a1_corr)
    quality = prior_rows[selected_name]
    panels = [
        _panel(
            prior,
            title="Feed-forward Gaussian prior",
            subtitle="AnySplat RGB render at the held-out pose",
            footer=[
                f"PSNR {float(quality['psnr_db']):.2f} dB  |  SSIM {float(quality['ssim']):.3f}",
                "Frozen photometric scaffold; absent at deployment",
            ],
        ),
        _panel(
            a0_visual,
            title="A0  Bootstrap map",
            subtitle="Native SuperPoint + global cosine top-1",
            footer=[
                f"TE {a0_official['te_cm']:.2f} cm  |  RE {a0_official['re_deg']:.2f} deg",
                f"P@2 {a0_official['raw_p2_percent']:.2f}%  |  inliers {a0_official['inliers']}/{a0_official['matches']}",
            ],
        ),
        _panel(
            a1_visual,
            title="A1  Reconstructed map",
            subtitle="Same one-shot sparse localization protocol",
            footer=[
                f"TE {a1_official['te_cm']:.2f} cm  |  RE {a1_official['re_deg']:.2f} deg",
                f"P@2 {a1_official['raw_p2_percent']:.2f}%  |  inliers {a1_official['inliers']}/{a1_official['matches']}",
            ],
        ),
    ]
    gap = 18
    figure = Image.new(
        "RGB",
        (sum(panel.width for panel in panels) + gap * 2, panels[0].height + 76),
        "#f3f4f6",
    )
    draw = ImageDraw.Draw(figure)
    draw.text(
        (24, 20),
        f"{frozen['scene']}  |  {selected_name}  |  rendering quality is not localization utility",
        fill="#111827",
        font=_font(26, bold=True),
    )
    x = 0
    for panel in panels:
        figure.paste(panel, (x, 76))
        x += panel.width + gap
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "figure_anysplat_a0_a1.png"
    figure.save(figure_path)
    manifest = {
        "schema": FIGURE_SCHEMA,
        "version": FIGURE_VERSION,
        "figure": "anysplat_a0_a1",
        "scene": frozen["scene"],
        "query_name": selected_name,
        "seed": int(seed),
        "selection": selection,
        "protocol": protocol.manifest(),
        "inputs": {
            "frozen_results": {"path": str(frozen_results_path), "sha256": _sha256(frozen_results_path)},
            "prior_quality": {"path": str(prior_quality_path), "sha256": _sha256(prior_quality_path)},
            "a0_map": {"path": str(a0_map), "sha256": _sha256(a0_map)},
            "a1_map": {"path": str(a1_map), "sha256": _sha256(a1_map)},
            "metric": {"path": str(metric), "sha256": _sha256(metric)},
            "render": {"path": str(Path(render_path).resolve()), "sha256": _sha256(Path(render_path))},
            "query_image": {"path": str(image_path), "sha256": _sha256(image_path)},
            "query_camera_source": str(query_camera_source),
            "prior_evaluation_source": str(prior_evaluation_source),
        },
        "official": {"A0": a0_official, "A1": a1_official},
        "reproduced_correspondences": {"A0": a0_reproduced, "A1": a1_reproduced},
        "legend": {
            "green": "GT-clean match at <=2 px",
            "red": "false attractor at >2 px",
            "cyan_ring": "PoseLib RANSAC inlier",
            "displayed_match_limit_per_panel": 700,
        },
        "output": {"path": str(figure_path), "sha256": _sha256(figure_path)},
        "cuda_device": str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu",
    }
    manifest_path = output_dir / "figure_anysplat_a0_a1.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build_method_overview(
    protocol: MainlineProtocol, output_path: str | Path
) -> dict[str, Any]:
    labels = (
        "Photometric Gaussian scaffold",
        "Real-image feature tracks",
        "Robust triangulation + raster lineage",
        "Localization evidence universe",
        "Topology distillation",
        "Compact track-anchored map",
        "Self-localization reconstruction",
        "One-shot sparse localization",
    )
    width, height = 1700, 500
    image = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(image)
    draw.text((52, 34), protocol.resolved["method"]["name"], fill="#111827", font=_font(34, bold=True))
    draw.text((52, 82), "Offline reconstruction absorbs complexity; deployment stays conventional.", fill="#4b5563", font=_font(21))
    box_w, box_h, gap = 184, 126, 22
    x, y = 42, 175
    colors = ("#e5e7eb", "#dbeafe", "#dbeafe", "#cffafe", "#fef3c7", "#dcfce7", "#dcfce7", "#fee2e2")
    for index, (label, color) in enumerate(zip(labels, colors)):
        draw.rounded_rectangle((x, y, x + box_w, y + box_h), radius=7, fill=color, outline="#9ca3af", width=2)
        words = label.split()
        lines, line = [], ""
        for word in words:
            candidate = f"{line} {word}".strip()
            if draw.textlength(candidate, font=_font(18, bold=True)) > box_w - 24 and line:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
        ty = y + 22
        for text in lines:
            draw.text((x + 12, ty), text, fill="#111827", font=_font(18, bold=True))
            ty += 26
        if index < len(labels) - 1:
            draw.line((x + box_w + 4, y + box_h / 2, x + box_w + gap - 5, y + box_h / 2), fill="#374151", width=3)
            draw.polygon(((x + box_w + gap - 5, y + box_h / 2 - 6), (x + box_w + gap - 5, y + box_h / 2 + 6), (x + box_w + gap + 2, y + box_h / 2)), fill="#374151")
        x += box_w + gap
    draw.text((52, 438), "Gaussian renderer used offline", fill="#6b7280", font=_font(18))
    draw.text((1410, 438), "Renderer absent online", fill="#991b1b", font=_font(18, bold=True))
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return {"path": str(output_path), "sha256": _sha256(output_path), "labels": list(labels)}


def build_topology_distillation_figure(
    *,
    a0_map_path: str | Path,
    a1_map_path: str | Path,
    output_path: str | Path,
    seed: int = 2026,
) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    a0_path = Path(a0_map_path).expanduser().resolve()
    a1_path = Path(a1_map_path).expanduser().resolve()
    a0 = torch.load(a0_path, map_location="cpu", weights_only=False)
    a1 = torch.load(a1_path, map_location="cpu", weights_only=False)
    xyz0 = np.asarray(a0["landmark_xyz"], dtype=np.float64)
    xyz1 = np.asarray(a1["anchor_xyz"], dtype=np.float64)
    combined = np.concatenate((xyz0, xyz1), axis=0)
    centered = combined - np.median(combined, axis=0)
    _, _, vt = np.linalg.svd(centered[:: max(len(centered) // 20000, 1)], full_matrices=False)
    projection = vt[:2].T
    xy0 = centered[: len(xyz0)] @ projection
    xy1 = centered[len(xyz0) :] @ projection
    rng = np.random.default_rng(seed)
    subset = rng.choice(len(xy0), size=min(len(xy0), 18000), replace=False)
    anchor_type = np.asarray(a1.get("anchor_type", np.zeros(len(xy1))), dtype=np.int64)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    axes[0].scatter(xy0[subset, 0], xy0[subset, 1], s=1.0, c="#60a5fa", alpha=0.25, rasterized=True)
    axes[0].set_title(f"A0 evidence scaffold: {len(xyz0):,} landmarks")
    colors = np.where(anchor_type == 1, "#16a34a", "#eab308")
    axes[1].scatter(
        xy1[:, 0], xy1[:, 1], s=2.0, c=colors, alpha=0.72, rasterized=True
    )
    axes[1].set_title(f"A1 distilled map: {len(xyz1):,} anchors")
    limit_xy = xy0[subset]
    low = np.percentile(limit_xy, 0.5, axis=0)
    high = np.percentile(limit_xy, 99.5, axis=0)
    padding = 0.05 * np.maximum(high - low, 1e-6)
    for axis in axes:
        axis.set_xlim(low[0] - padding[0], high[0] + padding[0])
        axis.set_ylim(low[1] - padding[1], high[1] + padding[1])
        axis.set_aspect("equal", adjustable="box")
        axis.set_axis_off()
    from matplotlib.lines import Line2D

    axes[1].legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", label="Track core", markerfacecolor="#16a34a", markersize=7),
            Line2D([0], [0], marker="o", color="w", label="Gaussian-supported reserve", markerfacecolor="#eab308", markersize=7),
        ],
        loc="lower left",
        frameon=False,
    )
    fig.suptitle(
        "Rendering-to-Localization Topology Distillation  |  "
        f"{100.0 * (1.0 - len(xyz1) / len(xyz0)):.1f}% fewer landmarks",
        fontsize=17,
        fontweight="bold",
    )
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)
    return {
        "path": str(output_path),
        "sha256": _sha256(output_path),
        "a0_count": int(len(xyz0)),
        "a1_count": int(len(xyz1)),
        "compression_percent": float(100.0 * (1.0 - len(xyz1) / len(xyz0))),
        "a0_sha256": _sha256(a0_path),
        "a1_sha256": _sha256(a1_path),
    }
