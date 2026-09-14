"""Data-backed, publication-quality figures for the frozen LaFGS mainline."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Polygon
import numpy as np
from PIL import Image
from plyfile import PlyData
import torch

from common.config import load_mainline_config, resolve_keypoint_count
from data.cameras import load_camera
from data.datasets import ColmapDataset
from evaluation.bootstrap import materialize_a0
from evaluation.metrics import pose_error
from localization.frontend import NativeSuperPointFrontend
from localization.localizer import SparseLocalizer
from priors.models import GaussianModel2D, GaussianModel3D
from priors.rendering import render_gsplat
from visualization.publication import (
    COLORS,
    PCAProjection,
    clean_image_axis,
    publication_style,
    robust_limits,
    save_figure,
    source_record,
)


FIGURE_SCHEMA = "lafgs_publication_figure"
FIGURE_VERSION = 1
SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class FigureArtifacts:
    shop_root: Path
    shop_dataset: Path
    prior_experiment_root: Path
    old_hospital_dataset: Path
    old_hospital_eval_dataset: Path
    config: Path

    @classmethod
    def defaults(cls) -> "FigureArtifacts":
        return cls(
            shop_root=Path(
                "/mnt/pool/sqy/stdloc_lafgs_v1_frozen_multiscene_20260731/"
                "ShopFacade"
            ),
            shop_dataset=Path("/mnt/pool/sqy/Cambridge_stdloc/ShopFacade"),
            prior_experiment_root=Path(
                "/mnt/pool/sqy/stdloc_lafgs_offtheshelf_prior_20260802"
            ),
            old_hospital_dataset=Path(
                "/mnt/pool/sqy/Cambridge_stdloc/OldHospital"
            ),
            old_hospital_eval_dataset=Path(
                "/mnt/pool/sqy/stdloc_lafgs_offtheshelf_prior_20260802/"
                "datasets/OldHospital_official_eval_undistorted"
            ),
            config=Path("/root/STDLoc/configs/paper_mainline.yaml"),
        )

    def validate(self) -> None:
        required = {
            "ShopFacade root": self.shop_root,
            "ShopFacade dataset": self.shop_dataset,
            "prior experiment root": self.prior_experiment_root,
            "OldHospital dataset": self.old_hospital_dataset,
            "OldHospital evaluation dataset": self.old_hospital_eval_dataset,
            "mainline config": self.config,
        }
        missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError("missing figure inputs:\n" + "\n".join(missing))


@dataclass(frozen=True)
class MatchDiagnostics:
    image_name: str
    image: np.ndarray
    keypoints: np.ndarray
    projected_gt: np.ndarray
    errors_px: np.ndarray
    scores: np.ndarray
    inlier_mask: np.ndarray
    translation_error_cm: float
    rotation_error_deg: float

    @property
    def raw_p2(self) -> float:
        return float(100.0 * np.mean(self.errors_px <= 2.0))

    @property
    def inlier_p2(self) -> float:
        if not self.inlier_mask.any():
            return 0.0
        return float(100.0 * np.mean(self.errors_px[self.inlier_mask] <= 2.0))

    @property
    def inlier_count(self) -> int:
        return int(self.inlier_mask.sum())


@dataclass(frozen=True)
class TrackPrimitiveExample:
    source_id: int
    primitive_mean: np.ndarray
    primitive_scales: np.ndarray
    primitive_rotation: np.ndarray
    anchor_xyz: np.ndarray
    track_ids: np.ndarray
    observation_crops: tuple[np.ndarray, ...]
    observation_names: tuple[str, ...]
    observation_colors: tuple[str, ...]


def _json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _torch(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _result_rows(root: Path, stage: str, seed: int = 2026) -> dict[str, dict]:
    pointer = root / "evaluation" / stage / f"seed{seed}" / "result.path"
    result_dir = Path(pointer.read_text().strip())
    rows = _json(result_dir / "results.json")
    return {str(row["image_name"]).replace("\\", "/"): row for row in rows}


def _row_metric(row: dict, name: str) -> float:
    if name == "te":
        return float(row.get("translation_error_cm", row.get("sparse_TE")))
    if name == "ae":
        return float(row.get("rotation_error_deg", row.get("sparse_AE")))
    sparse = row.get("sparse", row)
    aliases = {
        "raw_p2": ("raw_gt_precision_2px_percent", "sparse_diag_all_gt_precision_2px"),
        "inlier_p2": (
            "inlier_gt_precision_2px_percent",
            "sparse_diag_inlier_gt_precision_2px",
        ),
    }
    for key in aliases[name]:
        if key in sparse:
            value = float(sparse[key])
            return value if "percent" in key else 100.0 * value
    return 0.0


def _select_query(a0: dict[str, dict], a1: dict[str, dict]) -> str:
    candidates = []
    for name in sorted(set(a0) & set(a1)):
        gain = _row_metric(a0[name], "te") - _row_metric(a1[name], "te")
        precision_gain = _row_metric(a1[name], "raw_p2") - _row_metric(a0[name], "raw_p2")
        a1_te = _row_metric(a1[name], "te")
        if gain > 0 and precision_gain > 0 and a1_te < 8.0:
            score = math.log1p(gain) + 0.12 * precision_gain - 0.025 * a1_te
            candidates.append((score, name))
    if not candidates:
        raise RuntimeError("no improved A0/A1 qualitative query satisfies the fixed gate")
    return max(candidates)[1]


def _ply_vertex(path: Path):
    return PlyData.read(str(path), mmap="r")["vertex"].data


def _ply_sample(
    path: Path,
    *,
    maximum_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    vertex = _ply_vertex(path)
    count = len(vertex)
    rng = np.random.default_rng(seed)
    rows = (
        rng.choice(count, size=maximum_points, replace=False)
        if count > maximum_points
        else np.arange(count)
    )
    xyz = np.column_stack([vertex[name][rows] for name in ("x", "y", "z")]).astype(
        np.float32
    )
    names = set(vertex.dtype.names or ())
    if {"f_dc_0", "f_dc_1", "f_dc_2"} <= names:
        dc = np.column_stack([vertex[f"f_dc_{index}"][rows] for index in range(3)])
        colors = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)
    else:
        colors = np.tile(np.asarray([0.55, 0.58, 0.63]), (len(rows), 1))
    return xyz, colors


def _project(points: np.ndarray, intrinsic: np.ndarray, pose_w2c: np.ndarray):
    camera = points @ pose_w2c[:3, :3].T + pose_w2c[:3, 3]
    projected = camera @ intrinsic.T
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = camera[:, 2] > 1e-8
    uv[valid] = projected[valid, :2] / projected[valid, 2:3]
    return uv


def _quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _crop(image: np.ndarray, xy: np.ndarray, radius: int = 58) -> np.ndarray:
    height, width = image.shape[:2]
    x, y = map(int, np.round(xy))
    x0, x1 = max(x - radius, 0), min(x + radius, width)
    y0, y1 = max(y - radius, 0), min(y + radius, height)
    patch = image[y0:y1, x0:x1]
    canvas = np.full((2 * radius, 2 * radius, 3), 245, dtype=np.uint8)
    oy = (canvas.shape[0] - patch.shape[0]) // 2
    ox = (canvas.shape[1] - patch.shape[1]) // 2
    canvas[oy : oy + patch.shape[0], ox : ox + patch.shape[1]] = patch
    return canvas


def _scatter_map(axis, xy, colors, *, size=2.0, alpha=0.72, rasterized=True):
    axis.scatter(
        xy[:, 0],
        xy[:, 1],
        s=size,
        c=colors,
        alpha=alpha,
        linewidths=0,
        rasterized=rasterized,
    )
    clean_image_axis(axis)
    axis.set_aspect("equal", adjustable="box")


class PaperFigurePipeline:
    """Build the complete Cambridge qualitative figure set from frozen artifacts."""

    def __init__(
        self,
        artifacts: FigureArtifacts,
        output_dir: str | Path,
        *,
        device: str = "cuda:0",
        seed: int = 2026,
    ) -> None:
        artifacts.validate()
        self.artifacts = artifacts
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.seed = int(seed)
        self._matches: dict[str, MatchDiagnostics] | None = None
        self._track_example: TrackPrimitiveExample | None = None
        self._prior_sample_cache: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def shop_prior_ply(self) -> Path:
        return self.artifacts.shop_root / (
            "prior/rgb_matcha_2dgs/point_cloud/iteration_30000/point_cloud.ply"
        )

    @property
    def shop_a0_state(self) -> Path:
        return self.artifacts.shop_root / (
            "runs/frozen_v1/bootstrap/0_lafgs_map_state.pt"
        )

    @property
    def shop_a1_map(self) -> Path:
        return self.artifacts.shop_root / (
            "self_localization_reconstruction/anchor_map_step_0175.pt"
        )

    @property
    def shop_a1_metric(self) -> Path:
        return self.artifacts.shop_root / (
            "self_localization_reconstruction/metric_state_step_0175.pt"
        )

    @property
    def shop_track_payload(self) -> Path:
        return self.artifacts.shop_root / (
            "runs/frozen_v1/statistics_combined_1000_frozen_g3_"
            "track_provenance_v1/track_micro_anchor_payload.pt"
        )

    def _shop_prior_sample(self) -> tuple[np.ndarray, np.ndarray]:
        if self._prior_sample_cache is None:
            self._prior_sample_cache = _ply_sample(
                self.shop_prior_ply, maximum_points=70000, seed=self.seed
            )
        return self._prior_sample_cache

    def _match_bundle(self) -> dict[str, MatchDiagnostics]:
        if self._matches is not None:
            return self._matches
        a0_rows = _result_rows(self.artifacts.shop_root, "A0_bootstrap", self.seed)
        a1_rows = _result_rows(self.artifacts.shop_root, "A1_reconstructed", self.seed)
        image_name = _select_query(a0_rows, a1_rows)
        dataset = ColmapDataset(self.artifacts.shop_dataset, images="processed")
        camera = dataset.camera(image_name)
        config = load_mainline_config(self.artifacts.config).values
        deployment = config["deployment"]
        keypoints = resolve_keypoint_count(deployment, dataset.split("test"))
        reprojection = float(deployment["reprojection_error_px"])
        a0_map, a0_metric = materialize_a0(
            self.shop_a0_state,
            self.output_dir / "diagnostics/materialized_a0",
            self.artifacts.config,
        )
        common = {
            "device": self.device,
            "keypoint_count": keypoints,
            "reprojection_error_px": reprojection,
            "confidence": float(deployment["confidence"]),
            "max_iterations": int(deployment["maximum_iterations"]),
            "min_iterations": int(deployment["minimum_iterations"]),
            "seed": self.seed,
        }
        image = dataset.load_image(camera)
        valid_mask = dataset.valid_mask(camera)
        output: dict[str, MatchDiagnostics] = {}
        official_rows = {"A0": a0_rows[image_name], "A1": a1_rows[image_name]}
        for name, map_path, metric_path in (
            ("A0", a0_map, a0_metric),
            ("A1", self.shop_a1_map, self.shop_a1_metric),
        ):
            localizer = SparseLocalizer(map_path, metric_path, **common)
            result = localizer.localize(
                image,
                fov_x=camera.fov_x,
                fov_y=camera.fov_y,
                valid_mask=valid_mask,
            )
            matched_keypoints = (
                result.sparse_features.keypoints[result.matches.keypoint_indices]
                .detach()
                .cpu()
                .numpy()
                + 0.5
            )
            xyz = (
                localizer.anchor_xyz[result.matches.anchor_indices]
                .detach()
                .cpu()
                .numpy()
            )
            projected = _project(xyz, result.intrinsic, camera.pose_w2c)
            errors = np.linalg.norm(projected - matched_keypoints, axis=1)
            inlier_mask = np.zeros(len(errors), dtype=bool)
            inliers = result.pose.inliers
            inliers = inliers[(inliers >= 0) & (inliers < len(errors))]
            inlier_mask[inliers] = True
            ae, te = pose_error(result.pose.pose_w2c, camera.pose_w2c)
            diagnostics = MatchDiagnostics(
                image_name=image_name,
                image=(image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8),
                keypoints=matched_keypoints,
                projected_gt=projected,
                errors_px=errors,
                scores=result.matches.scores.detach().cpu().numpy(),
                inlier_mask=inlier_mask,
                translation_error_cm=te,
                rotation_error_deg=ae,
            )
            official = official_rows[name]
            official_sparse = official.get("sparse", official)
            expected_matches = int(
                official_sparse.get("matches", official_sparse.get("raw_count", 0))
            )
            expected_inliers = int(
                official_sparse.get("inliers", official_sparse.get("inlier_count", 0))
            )
            parity = {
                "matches": (len(diagnostics.keypoints), expected_matches),
                "inliers": (diagnostics.inlier_count, expected_inliers),
            }
            mismatched = {
                key: values for key, values in parity.items() if values[0] != values[1]
            }
            if mismatched:
                raise RuntimeError(f"{name} frozen correspondence parity failed: {mismatched}")
            for metric, reproduced in (
                ("raw_p2", diagnostics.raw_p2),
                ("inlier_p2", diagnostics.inlier_p2),
            ):
                expected = _row_metric(official, metric)
                if not math.isclose(reproduced, expected, abs_tol=1e-5):
                    raise RuntimeError(
                        f"{name} {metric} parity failed: "
                        f"reproduced={reproduced:.8f}, expected={expected:.8f}"
                    )
            output[name] = diagnostics
            del localizer, result
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        self._matches = output
        return output

    def _track_primitive_example(self) -> TrackPrimitiveExample:
        if self._track_example is not None:
            return self._track_example
        anchor_map = _torch(self.shop_a1_map)
        source = torch.as_tensor(anchor_map["source_primitive_ids"]).long()
        anchor_type = torch.as_tensor(anchor_map["anchor_type"]).long()
        track_ids = torch.as_tensor(anchor_map["track_cluster_ids"]).long()
        eligible = (source >= 0) & (anchor_type == 1) & (track_ids >= 0)
        unique, counts = torch.unique(source[eligible], return_counts=True)
        repeated = unique[(counts >= 3) & (counts <= 12)]
        if repeated.numel() == 0:
            raise RuntimeError("no one-primitive-to-many-track example is available")

        vertex = _ply_vertex(self.shop_prior_ply)
        scale0 = np.exp(np.asarray(vertex["scale_0"][repeated.numpy()]))
        scale1 = np.exp(np.asarray(vertex["scale_1"][repeated.numpy()]))
        area = scale0 * scale1
        # Prefer a visibly large surfel, but avoid the pathological extreme tail.
        threshold = np.percentile(area, 75)
        candidates = repeated.numpy()[area >= threshold]
        source_id = int(candidates[np.argsort(candidates)[len(candidates) // 2]])
        rows = torch.nonzero(eligible & (source == source_id), as_tuple=False).flatten()
        rows = rows[: min(4, rows.numel())]
        xyz = torch.as_tensor(anchor_map["anchor_xyz"])[rows].numpy()
        selected_track_ids = track_ids[rows].numpy()

        primitive_mean = np.asarray(
            [vertex[name][source_id] for name in ("x", "y", "z")],
            dtype=np.float64,
        )
        primitive_scales = np.exp(
            np.asarray(
                [vertex[f"scale_{index}"][source_id] for index in range(2)],
                dtype=np.float64,
            )
        )
        primitive_rotation = np.asarray(
            [vertex[f"rot_{index}"][source_id] for index in range(4)],
            dtype=np.float64,
        )

        payload = _torch(self.shop_track_payload)
        observations = payload["tracks"]
        query_names = payload["query_names"]
        obs_track = torch.as_tensor(observations["track_index"]).long()
        obs_query = torch.as_tensor(observations["query_index"]).long()
        obs_keypoint = torch.as_tensor(observations["keypoint_index"]).long()
        dataset = ColmapDataset(self.artifacts.shop_dataset, images="processed")
        frontend = NativeSuperPointFrontend(
            device=self.device, keypoint_count=2048, metric=None
        )
        palette = (COLORS["track"], COLORS["anchor"], COLORS["candidate"], "#CC79A7")
        crops: list[np.ndarray] = []
        names: list[str] = []
        colors: list[str] = []
        used_queries: set[int] = set()
        for position, track_id in enumerate(selected_track_ids):
            indexes = torch.nonzero(obs_track == int(track_id), as_tuple=False).flatten()
            selected = None
            for index in indexes.tolist():
                query_index = int(obs_query[index])
                if query_index not in used_queries:
                    selected = index
                    break
            if selected is None:
                selected = int(indexes[0])
            query_index = int(obs_query[selected])
            keypoint_index = int(obs_keypoint[selected])
            camera = dataset.camera(query_names[query_index])
            image_tensor = dataset.load_image(camera)
            sparse = frontend(
                image_tensor, valid_mask=dataset.valid_mask(camera)
            )
            if keypoint_index >= sparse.keypoints.shape[0]:
                # Track payload indices can precede mask filtering in older caches.
                sparse = frontend(image_tensor, valid_mask=None)
            keypoint_index = min(keypoint_index, sparse.keypoints.shape[0] - 1)
            xy = sparse.keypoints[keypoint_index].detach().cpu().numpy()
            image = (
                image_tensor.permute(1, 2, 0).numpy().clip(0, 1) * 255
            ).round().astype(np.uint8)
            crops.append(_crop(image, xy))
            names.append(camera.image_name)
            colors.append(palette[position])
            used_queries.add(query_index)
        del frontend
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self._track_example = TrackPrimitiveExample(
            source_id=source_id,
            primitive_mean=primitive_mean,
            primitive_scales=primitive_scales,
            primitive_rotation=primitive_rotation,
            anchor_xyz=xyz,
            track_ids=selected_track_ids,
            observation_crops=tuple(crops),
            observation_names=tuple(names),
            observation_colors=tuple(colors),
        )
        return self._track_example

    @staticmethod
    def _draw_match_axis(axis, diagnostics: MatchDiagnostics, title: str) -> None:
        axis.imshow(diagnostics.image)
        clean = diagnostics.errors_px <= 2.0
        inlier = diagnostics.inlier_mask
        rng = np.random.default_rng(31)

        false_rows = np.flatnonzero(~clean)
        if len(false_rows) > 260:
            false_rows = rng.choice(false_rows, 260, replace=False)
        clean_rows = np.flatnonzero(clean)
        if len(clean_rows) > 240:
            clean_rows = clean_rows[np.argsort(diagnostics.scores[clean_rows])[-240:]]
        axis.scatter(
            diagnostics.keypoints[false_rows, 0],
            diagnostics.keypoints[false_rows, 1],
            s=5.0,
            c=COLORS["false"],
            alpha=0.32,
            linewidths=0,
            rasterized=True,
        )
        axis.scatter(
            diagnostics.keypoints[clean_rows, 0],
            diagnostics.keypoints[clean_rows, 1],
            s=8.0,
            c=COLORS["clean"],
            alpha=0.88,
            linewidths=0,
            rasterized=True,
        )

        # Directional residuals expose geometrically harmful RANSAC inliers.
        harmful = np.flatnonzero(inlier & ~clean & np.isfinite(diagnostics.projected_gt).all(1))
        harmful = harmful[np.argsort(diagnostics.scores[harmful])[-min(45, len(harmful)):]]
        segments = []
        for index in harmful:
            start = diagnostics.keypoints[index]
            delta = diagnostics.projected_gt[index] - start
            length = float(np.linalg.norm(delta))
            if length > 55.0:
                delta *= 55.0 / length
            segments.append([start, start + delta])
        if segments:
            axis.add_collection(
                LineCollection(
                    segments,
                    colors=COLORS["false"],
                    linewidths=0.65,
                    alpha=0.7,
                    rasterized=True,
                )
            )
        inlier_rows = np.flatnonzero(inlier)
        axis.scatter(
            diagnostics.keypoints[inlier_rows, 0],
            diagnostics.keypoints[inlier_rows, 1],
            s=16,
            facecolors="none",
            edgecolors=COLORS["inlier"],
            linewidths=0.55,
            alpha=0.7,
            rasterized=True,
        )
        axis.set_title(title, loc="left", pad=5)
        axis.text(
            0.018,
            0.025,
            f"TE {diagnostics.translation_error_cm:.2f} cm  |  "
            f"AE {diagnostics.rotation_error_deg:.2f}°\n"
            f"raw P@2 {diagnostics.raw_p2:.2f}%  |  "
            f"inlier P@2 {diagnostics.inlier_p2:.2f}%  |  "
            f"{diagnostics.inlier_count}/{len(diagnostics.keypoints)} inliers",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.1,
            color="white",
            bbox={
                "facecolor": "#111827",
                "edgecolor": "none",
                "alpha": 0.82,
                "boxstyle": "round,pad=0.32,rounding_size=0.12",
            },
        )
        clean_image_axis(axis)

    def build_match_comparison(self) -> dict[str, Any]:
        matches = self._match_bundle()
        with publication_style():
            figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.62))
            self._draw_match_axis(axes[0], matches["A0"], "(a)  A0 · wide scaffold")
            self._draw_match_axis(
                axes[1], matches["A1"], "(b)  A1 · reconstructed compact map"
            )
            figure.subplots_adjust(left=0.01, right=0.995, top=0.91, bottom=0.01, wspace=0.035)
            paths = save_figure(figure, self.output_dir, "figure_04_a0_vs_a1_matches")
        return {
            "schema": FIGURE_SCHEMA,
            "version": FIGURE_VERSION,
            "figure": "A0_vs_A1_matches",
            "query": matches["A0"].image_name,
            "seed": self.seed,
            "legend": {
                "green": "GT-clean correspondence (<=2 px)",
                "red": "false correspondence (>2 px)",
                "cyan_ring": "PoseLib RANSAC inlier",
                "red_segment": "clipped GT reprojection residual of a harmful inlier",
            },
            "metrics": {
                stage: {
                    "translation_error_cm": value.translation_error_cm,
                    "rotation_error_deg": value.rotation_error_deg,
                    "raw_gt_precision_2px_percent": value.raw_p2,
                    "inlier_gt_precision_2px_percent": value.inlier_p2,
                    "inliers": value.inlier_count,
                    "matches": len(value.keypoints),
                }
                for stage, value in matches.items()
            },
            "outputs": paths,
        }

    def build_primitive_identity(self) -> dict[str, Any]:
        example = self._track_primitive_example()
        with publication_style():
            figure = plt.figure(figsize=(7.2, 3.45))
            grid = figure.add_gridspec(
                2,
                4,
                height_ratios=(0.9, 1.45),
                left=0.035,
                right=0.985,
                bottom=0.08,
                top=0.94,
                hspace=0.24,
                wspace=0.09,
            )
            crop_axes = [figure.add_subplot(grid[0, index]) for index in range(4)]
            for index, axis in enumerate(crop_axes):
                if index < len(example.observation_crops):
                    axis.imshow(example.observation_crops[index])
                    axis.scatter(
                        [example.observation_crops[index].shape[1] / 2],
                        [example.observation_crops[index].shape[0] / 2],
                        s=48,
                        facecolors="none",
                        edgecolors=example.observation_colors[index],
                        linewidths=1.45,
                    )
                    axis.text(
                        0.025,
                        0.97,
                        f"track {int(example.track_ids[index])} · view {index + 1}",
                        transform=axis.transAxes,
                        ha="left",
                        va="top",
                        color=example.observation_colors[index],
                        fontsize=6.9,
                        fontweight="semibold",
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "none",
                            "alpha": 0.82,
                            "pad": 1.4,
                        },
                    )
                else:
                    axis.axis("off")
                clean_image_axis(axis)
            figure.text(
                0.035,
                0.985,
                "(a)  Real-image observations define localization identity",
                transform=figure.transFigure,
                ha="left",
                va="top",
                fontsize=8.6,
                fontweight="semibold",
                color=COLORS["ink"],
            )

            local_axis = figure.add_subplot(grid[1, :3])
            local_points = np.concatenate(
                (example.anchor_xyz, example.primitive_mean[None]), axis=0
            )
            projection = PCAProjection.fit(local_points)
            rotation = _quaternion_matrix(example.primitive_rotation)
            theta = np.linspace(0, 2 * np.pi, 160)
            disk_world = (
                example.primitive_mean[None]
                + 3.0
                * np.cos(theta)[:, None]
                * example.primitive_scales[0]
                * rotation[:, 0][None]
                + 3.0
                * np.sin(theta)[:, None]
                * example.primitive_scales[1]
                * rotation[:, 1][None]
            )
            disk_xy = projection.transform(disk_world)
            anchor_xy = projection.transform(example.anchor_xyz)
            center_xy = projection.transform(example.primitive_mean[None])[0]
            local_axis.add_patch(
                Polygon(
                    disk_xy,
                    closed=True,
                    facecolor=COLORS["primitive"],
                    edgecolor="#475467",
                    linewidth=0.9,
                    alpha=0.28,
                )
            )
            local_axis.scatter(
                [center_xy[0]],
                [center_xy[1]],
                marker="x",
                s=34,
                c="#475467",
                linewidths=1.0,
                label="rendering primitive center",
                zorder=4,
            )
            for index, point in enumerate(anchor_xy):
                color = example.observation_colors[index]
                local_axis.scatter(
                    [point[0]],
                    [point[1]],
                    marker="o",
                    s=36,
                    facecolors=color,
                    edgecolors="white",
                    linewidths=0.6,
                    zorder=5,
                )
                local_axis.plot(
                    [center_xy[0], point[0]],
                    [center_xy[1], point[1]],
                    color=color,
                    linewidth=0.8,
                    alpha=0.68,
                    zorder=3,
                )
                local_axis.annotate(
                    f"track {int(example.track_ids[index])}",
                    point,
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=6.8,
                    color=color,
                )
            bounds = np.concatenate((disk_xy, anchor_xy), axis=0)
            low, high = robust_limits(bounds, percent=0.0, padding=0.12)
            local_axis.set_xlim(low[0], high[0])
            local_axis.set_ylim(low[1], high[1])
            local_axis.set_aspect("equal", adjustable="datalim")
            clean_image_axis(local_axis)
            local_axis.set_title(
                "(b)  One Gaussian support, multiple triangulated anchors",
                loc="left",
                pad=4,
            )

            identity_axis = figure.add_subplot(grid[1, 3])
            identity_axis.set_xlim(0, 1)
            identity_axis.set_ylim(0, 1)
            identity_axis.axis("off")
            identity_axis.text(
                0.5,
                0.87,
                "(c)  Identity is not inherited",
                ha="center",
                va="center",
                fontsize=8.3,
                fontweight="semibold",
                color=COLORS["ink"],
            )
            identity_axis.text(
                0.5,
                0.68,
                f"Gaussian #{example.source_id}",
                ha="center",
                va="center",
                fontsize=8.0,
                color=COLORS["primitive"],
                bbox={"facecolor": "#EEF1F5", "edgecolor": COLORS["line"], "boxstyle": "round,pad=0.35"},
            )
            identity_axis.add_patch(
                FancyArrowPatch(
                    (0.5, 0.58),
                    (0.5, 0.40),
                    arrowstyle="-|>",
                    mutation_scale=9,
                    linewidth=0.8,
                    color=COLORS["muted"],
                )
            )
            identity_axis.text(
                0.5,
                0.27,
                f"{len(example.anchor_xyz)} distinct\ntrack-anchor identities",
                ha="center",
                va="center",
                fontsize=8.0,
                color=COLORS["anchor"],
                bbox={"facecolor": "#FFF4ED", "edgecolor": "#FDBA8C", "boxstyle": "round,pad=0.4"},
            )
            identity_axis.text(
                0.5,
                0.04,
                "Gaussian ID ≠ track ID ≠ anchor ID",
                ha="center",
                va="bottom",
                fontsize=7.1,
                color=COLORS["ink"],
                fontweight="semibold",
            )
            paths = save_figure(
                figure, self.output_dir, "figure_02_primitive_not_landmark"
            )
        return {
            "schema": FIGURE_SCHEMA,
            "version": FIGURE_VERSION,
            "figure": "rendering_primitive_not_localization_landmark",
            "source_primitive_id": example.source_id,
            "track_ids": [int(value) for value in example.track_ids],
            "observation_images": list(example.observation_names),
            "outputs": paths,
        }

    def build_topology_distillation(self) -> dict[str, Any]:
        prior_xyz, prior_colors = self._shop_prior_sample()
        a0 = _torch(self.shop_a0_state)
        a1 = _torch(self.shop_a1_map)
        tracks = _torch(self.shop_track_payload)
        a0_xyz = torch.as_tensor(a0["landmark_xyz"]).numpy()
        a1_xyz = torch.as_tensor(a1["anchor_xyz"]).numpy()
        anchor_type = torch.as_tensor(a1["anchor_type"]).numpy()
        projection = PCAProjection.fit(prior_xyz)
        prior_xy = projection.transform(prior_xyz)
        a0_xy = projection.transform(a0_xyz)
        a1_xy = projection.transform(a1_xyz)
        low, high = robust_limits(prior_xy, percent=0.6, padding=0.035)
        primitive_count = int(_torch(
            self.artifacts.shop_root / "function_graph/raster_provenance.pt"
        )["primitive_count"])
        track_count = int(tracks["diagnostics"]["track_count"])
        triangulated_count = int(
            tracks["diagnostics"]["geometry_teacher_triangulated_track_count"]
        )
        core_count = int(np.count_nonzero(anchor_type == 1))
        reserve_count = int(np.count_nonzero(anchor_type == 0))

        with publication_style():
            figure = plt.figure(figsize=(7.2, 2.65))
            grid = figure.add_gridspec(
                2,
                4,
                height_ratios=(4.0, 1.05),
                left=0.025,
                right=0.99,
                top=0.9,
                bottom=0.12,
                wspace=0.11,
                hspace=0.12,
            )
            axes = [figure.add_subplot(grid[0, index]) for index in range(4)]
            sample_rows = np.linspace(0, len(prior_xy) - 1, min(32000, len(prior_xy))).astype(int)
            _scatter_map(
                axes[0], prior_xy[sample_rows], prior_colors[sample_rows], size=0.45, alpha=0.27
            )
            axes[0].set_title(
                f"(a) RGB Gaussian prior\n{primitive_count:,} rendering splats",
                loc="left",
                fontsize=7.45,
            )
            _scatter_map(
                axes[1], a0_xy, COLORS["candidate"], size=0.65, alpha=0.20
            )
            axes[1].set_title(
                f"(b) KCS/GWFF evidence\n{len(a0_xy):,} candidates",
                loc="left",
                fontsize=7.45,
            )
            triangulated = torch.as_tensor(tracks["track_geometry"]["triangulated_xyz"])
            valid = torch.as_tensor(tracks["track_geometry"]["triangulated"]).bool()
            track_xy = projection.transform(triangulated[valid].numpy())
            _scatter_map(axes[2], track_xy, COLORS["track"], size=1.0, alpha=0.46)
            axes[2].set_title(
                f"(c) Track-first evidence\n{track_count:,} tracks\n"
                f"{triangulated_count:,} triangulated",
                loc="left",
                fontsize=7.45,
            )
            colors = np.where(anchor_type == 1, COLORS["core"], COLORS["reserve"])
            _scatter_map(axes[3], a1_xy, colors, size=1.55, alpha=0.78)
            axes[3].set_title(
                f"(d) Distilled topology\n{len(a1_xy):,} anchors",
                loc="left",
                fontsize=7.45,
            )
            for index, axis in enumerate(axes):
                axis.set_xlim(low[0], high[0])
                axis.set_ylim(low[1], high[1])
                if index < 3:
                    figure.add_artist(
                        ConnectionPatch(
                            xyA=(1.01, 0.5),
                            coordsA=axis.transAxes,
                            xyB=(-0.03, 0.5),
                            coordsB=axes[index + 1].transAxes,
                            arrowstyle="-|>",
                            mutation_scale=9,
                            linewidth=0.8,
                            color=COLORS["ink"],
                        )
                    )

            flow = figure.add_subplot(grid[1, :])
            flow.axis("off")
            values = np.asarray([primitive_count, len(a0_xy), track_count, len(a1_xy)])
            x = np.arange(4)
            width = np.log10(values) / np.log10(values.max())
            for index, (position, fraction) in enumerate(zip(x, width)):
                flow.plot(
                    [position - 0.35 * fraction, position + 0.35 * fraction],
                    [0.62, 0.62],
                    color=(COLORS["primitive"], COLORS["candidate"], COLORS["track"], COLORS["core"])[index],
                    linewidth=5.4,
                    solid_capstyle="round",
                )
            flow.set_xlim(-0.55, 3.55)
            flow.set_ylim(0, 1)
            flow.text(
                3.0,
                0.05,
                f"Track core {core_count:,}   |   Gaussian reserve {reserve_count:,}   |   "
                f"85.1% fewer than the 48K scaffold",
                ha="center",
                va="bottom",
                fontsize=7.2,
                color=COLORS["ink"],
            )
            figure.legend(
                handles=[
                    Line2D([0], [0], marker="o", linestyle="", color=COLORS["core"], label="Track core"),
                    Line2D([0], [0], marker="o", linestyle="", color=COLORS["reserve"], label="Gaussian-supported reserve"),
                ],
                loc="upper right",
                bbox_to_anchor=(0.99, 0.995),
                ncol=2,
                handletextpad=0.35,
                columnspacing=1.0,
            )
            paths = save_figure(
                figure, self.output_dir, "figure_03_topology_distillation"
            )
        return {
            "schema": FIGURE_SCHEMA,
            "version": FIGURE_VERSION,
            "figure": "topology_distillation",
            "counts": {
                "rgb_gaussian_primitives": primitive_count,
                "kcs_gwff_candidates": int(len(a0_xy)),
                "feature_tracks": track_count,
                "triangulated_tracks": triangulated_count,
                "final_anchors": int(len(a1_xy)),
                "track_core": core_count,
                "gaussian_reserve": reserve_count,
            },
            "outputs": paths,
        }

    def build_method_overview(self) -> dict[str, Any]:
        prior_xyz, prior_colors = self._shop_prior_sample()
        a0 = _torch(self.shop_a0_state)
        a1 = _torch(self.shop_a1_map)
        example = self._track_primitive_example()
        matches = self._match_bundle()
        projection = PCAProjection.fit(prior_xyz)
        prior_xy = projection.transform(prior_xyz)
        a0_xy = projection.transform(torch.as_tensor(a0["landmark_xyz"]).numpy())
        a1_xyz = torch.as_tensor(a1["anchor_xyz"]).numpy()
        a1_xy = projection.transform(a1_xyz)
        anchor_type = torch.as_tensor(a1["anchor_type"]).numpy()
        low, high = robust_limits(prior_xy, percent=0.8, padding=0.025)

        with publication_style():
            figure = plt.figure(figsize=(7.2, 4.12))
            grid = figure.add_gridspec(
                2,
                4,
                left=0.025,
                right=0.99,
                top=0.89,
                bottom=0.06,
                hspace=0.38,
                wspace=0.14,
            )
            axes = [figure.add_subplot(grid[0, index]) for index in range(4)]
            axes += [figure.add_subplot(grid[1, index]) for index in range(3)]
            deployment_axis = figure.add_subplot(grid[1, 3])
            axes.append(deployment_axis)

            sample_rows = np.linspace(0, len(prior_xy) - 1, min(26000, len(prior_xy))).astype(int)
            _scatter_map(axes[0], prior_xy[sample_rows], prior_colors[sample_rows], size=0.42, alpha=0.30)
            axes[0].set_xlim(low[0], high[0])
            axes[0].set_ylim(low[1], high[1])

            strip = np.concatenate(example.observation_crops[:3], axis=1)
            axes[1].imshow(strip)
            crop_width = example.observation_crops[0].shape[1]
            for index in range(3):
                axes[1].scatter(
                    [index * crop_width + crop_width / 2],
                    [strip.shape[0] / 2],
                    s=30,
                    facecolors="none",
                    edgecolors=example.observation_colors[index],
                    linewidths=1.0,
                )
            clean_image_axis(axes[1])

            local_projection = PCAProjection.fit(
                np.concatenate((example.anchor_xyz, example.primitive_mean[None]), axis=0)
            )
            local_xy = local_projection.transform(example.anchor_xyz)
            axes[2].scatter(
                local_xy[:, 0],
                local_xy[:, 1],
                c=example.observation_colors,
                s=25,
                edgecolors="white",
                linewidths=0.5,
            )
            center = local_projection.transform(example.primitive_mean[None])[0]
            for point, color in zip(local_xy, example.observation_colors):
                axes[2].plot(
                    [center[0], point[0]], [center[1], point[1]], color=color, linewidth=0.75
                )
            axes[2].scatter([center[0]], [center[1]], c=COLORS["primitive"], marker="x", s=28)
            local_bounds = np.concatenate((local_xy, center[None]), axis=0)
            local_low, local_high = robust_limits(
                local_bounds, percent=0.0, padding=0.15
            )
            axes[2].set_xlim(local_low[0], local_high[0])
            axes[2].set_ylim(local_low[1], local_high[1])
            clean_image_axis(axes[2])
            axes[2].set_aspect("equal", adjustable="box")

            _scatter_map(axes[3], a0_xy, COLORS["candidate"], size=0.55, alpha=0.18)
            axes[3].set_xlim(low[0], high[0])
            axes[3].set_ylim(low[1], high[1])

            final_colors = np.where(anchor_type == 1, COLORS["core"], COLORS["reserve"])
            _scatter_map(axes[4], a1_xy, final_colors, size=1.25, alpha=0.72)
            axes[4].set_xlim(low[0], high[0])
            axes[4].set_ylim(low[1], high[1])

            # The reconstruction panel visualizes current-map outcomes used offline.
            reconstruction = matches["A1"]
            axes[5].imshow(reconstruction.image)
            clean = reconstruction.errors_px <= 2.0
            false_inliers = reconstruction.inlier_mask & ~clean
            clean_rows = np.flatnonzero(clean)
            if len(clean_rows) > 160:
                clean_rows = clean_rows[np.argsort(reconstruction.scores[clean_rows])[-160:]]
            axes[5].scatter(
                reconstruction.keypoints[clean_rows, 0],
                reconstruction.keypoints[clean_rows, 1],
                c=COLORS["clean"],
                s=5,
                alpha=0.78,
                linewidths=0,
            )
            axes[5].scatter(
                reconstruction.keypoints[false_inliers, 0],
                reconstruction.keypoints[false_inliers, 1],
                facecolors="none",
                edgecolors=COLORS["false"],
                s=10,
                linewidths=0.5,
                alpha=0.62,
            )
            clean_image_axis(axes[5])

            deployment_axis.imshow(reconstruction.image)
            inlier_rows = np.flatnonzero(reconstruction.inlier_mask)
            deployment_axis.scatter(
                reconstruction.keypoints[inlier_rows, 0],
                reconstruction.keypoints[inlier_rows, 1],
                facecolors="none",
                edgecolors=COLORS["inlier"],
                s=9,
                linewidths=0.45,
                alpha=0.64,
            )
            clean_image_axis(deployment_axis)

            titles = (
                ("1  Gaussian scaffold", "1.20M frozen RGB splats"),
                ("2  Feature tracks", "real mapping observations"),
                ("3  Triangulated anchors", "independent PnP geometry"),
                ("4  Candidate universe", "48K localization evidence"),
                ("5  Compact topology", f"{len(a1_xy):,} single-descriptor anchors"),
                ("6  Self-localization", "keep · swap · miss · attractor"),
                ("", ""),
                ("7  Sparse deployment", "SuperPoint → top-1 → one PnP/RANSAC"),
            )
            # Axis 6 is intentionally a compact feedback annotation rather than a new stage.
            axes[6].axis("off")
            axes[6].text(
                0.5,
                0.62,
                "current map",
                ha="center",
                va="center",
                fontsize=8.0,
                color=COLORS["a1"],
                fontweight="semibold",
            )
            axes[6].add_patch(
                FancyArrowPatch(
                    (0.25, 0.48),
                    (0.75, 0.48),
                    connectionstyle="arc3,rad=-0.55",
                    arrowstyle="-|>",
                    mutation_scale=9,
                    color=COLORS["a1"],
                    linewidth=1.0,
                )
            )
            axes[6].add_patch(
                FancyArrowPatch(
                    (0.75, 0.43),
                    (0.25, 0.43),
                    connectionstyle="arc3,rad=-0.55",
                    arrowstyle="-|>",
                    mutation_scale=9,
                    color=COLORS["candidate"],
                    linewidth=1.0,
                )
            )
            axes[6].text(
                0.5,
                0.18,
                "matching outcomes update\ndescriptor reconstruction",
                ha="center",
                va="center",
                fontsize=7.0,
                color=COLORS["muted"],
            )
            for index, axis in enumerate(axes):
                title, subtitle = titles[index]
                if title:
                    axis.set_title(title, loc="left", pad=3, fontsize=7.65)
                    axis.text(
                        0,
                        -0.075,
                        subtitle,
                        transform=axis.transAxes,
                        ha="left",
                        va="top",
                        fontsize=6.65,
                        color=COLORS["muted"],
                    )

            figure.text(
                0.025,
                0.965,
                "OFFLINE · rendering-to-localization map reconstruction",
                ha="left",
                va="top",
                fontsize=8.4,
                fontweight="semibold",
                color=COLORS["ink"],
            )
            figure.text(
                0.025,
                0.505,
                "OFFLINE · topology and descriptor reconstruction",
                ha="left",
                va="bottom",
                fontsize=6.8,
                fontweight="semibold",
                color=COLORS["muted"],
            )
            figure.text(
                0.75,
                0.505,
                "ONLINE · one-shot sparse deployment",
                ha="left",
                va="bottom",
                fontsize=6.8,
                fontweight="semibold",
                color=COLORS["a1"],
            )
            connections = ((0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7))
            for left_index, right_index in connections:
                left_axis, right_axis = axes[left_index], axes[right_index]
                start, end = (1.015, 0.5), (-0.04, 0.5)
                figure.add_artist(
                    ConnectionPatch(
                        xyA=start,
                        coordsA=left_axis.transAxes,
                        xyB=end,
                        coordsB=right_axis.transAxes,
                        arrowstyle="-|>",
                        mutation_scale=8.5,
                        linewidth=0.75,
                        color=COLORS["ink"],
                    )
                )
            paths = save_figure(figure, self.output_dir, "figure_01_method_overview")
        return {
            "schema": FIGURE_SCHEMA,
            "version": FIGURE_VERSION,
            "figure": "method_overview",
            "stages": [
                "frozen Gaussian scaffold",
                "real-image feature tracks",
                "robustly triangulated localization anchors",
                "candidate evidence universe",
                "compact topology distillation",
                "self-localization-guided descriptor reconstruction",
                "one-shot sparse deployment",
            ],
            "counts": {
                "prior_primitives": 1202378,
                "wide_candidates": int(len(a0_xy)),
                "final_anchors": int(len(a1_xy)),
            },
            "qualitative_query": matches["A1"].image_name,
            "outputs": paths,
        }

    def _render_prior_view(self, profile: str, image_name: str) -> Path:
        cache = self.output_dir / "cache/prior_renders"
        cache.mkdir(parents=True, exist_ok=True)
        output = cache / f"OldHospital_{profile}_{image_name.replace('/', '__')}"
        output = output.with_suffix(".png")
        if output.is_file():
            return output
        model_root = (
            self.artifacts.prior_experiment_root
            / "priors/OldHospital"
            / profile
            / "stdloc_model"
        )
        manifest = _json(model_root / "rgb_prior_manifest.json")
        prior_type = str(manifest.get("gaussian_type", manifest.get("type", "3dgs")))
        degree = int(manifest.get("sh_degree", 3))
        ply = model_root / "point_cloud/iteration_30000/point_cloud.ply"
        dataset = ColmapDataset(self.artifacts.old_hospital_eval_dataset, images="images")
        camera_record = dataset.camera(image_name)
        camera = load_camera(dataset, camera_record, uid=0, data_device="cpu")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        model = (
            GaussianModel2D(degree, device=self.device)
            if prior_type == "2dgs"
            else GaussianModel3D(degree, device=self.device)
        )
        model.load_ply(ply, loc_feature_dim=1)
        white = bool(manifest.get("white_background", False))
        background = torch.full(
            (3,), 1.0 if white else 0.0, device=self.device, dtype=torch.float32
        )
        with torch.inference_mode():
            rendered = render_gsplat(
                camera,
                model,
                background,
                rgb_only=True,
                longest_edge=max(camera.image_width, camera.image_height),
            )["render"]
        array = (
            rendered.detach()
            .clamp(0, 1)
            .mul(255)
            .round()
            .byte()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        Image.fromarray(array).save(output)
        del rendered, model, camera
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return output

    def build_prior_flexibility(self) -> dict[str, Any]:
        profiles = (
            ("vanilla_3dgs", "Vanilla 3DGS"),
            ("vanilla_2dgs", "Vanilla 2DGS"),
            ("anysplat_ff", "AnySplat · feed-forward"),
        )
        query_name = "seq8/frame00061.png"
        audit_path = (
            self.artifacts.prior_experiment_root
            / "prior_robustness_complete_with_anysplat_strict_audit.json"
        )
        audit = _json(audit_path)
        records = {
            row["profile"]: row
            for row in audit["records"]
            if row.get("scene") == "OldHospital"
            and row.get("profile") in {profile for profile, _ in profiles}
            and row.get("complete") is True
        }
        if set(records) != {profile for profile, _ in profiles}:
            raise RuntimeError("strict prior-flexibility audit is incomplete")

        renders: dict[str, Path] = {}
        maps: dict[str, dict[str, Any]] = {}
        xyz_all = []
        for profile, _ in profiles:
            renders[profile] = self._render_prior_view(profile, query_name)
            root = (
                self.artifacts.prior_experiment_root
                / "lafgs_strict_v2"
                / profile
                / "OldHospital"
            )
            maps[profile] = _torch(
                root / "self_localization_reconstruction/anchor_map_step_0175.pt"
            )
            xyz_all.append(torch.as_tensor(maps[profile]["anchor_xyz"]).numpy())
        projection = PCAProjection.fit(np.concatenate(xyz_all, axis=0))
        projected = {
            profile: projection.transform(torch.as_tensor(maps[profile]["anchor_xyz"]).numpy())
            for profile, _ in profiles
        }
        low, high = robust_limits(np.concatenate(list(projected.values()), axis=0), percent=0.5, padding=0.04)

        with publication_style():
            figure = plt.figure(figsize=(7.2, 4.0))
            grid = figure.add_gridspec(
                2,
                3,
                height_ratios=(1.14, 1.0),
                left=0.025,
                right=0.99,
                top=0.91,
                bottom=0.085,
                wspace=0.065,
                hspace=0.18,
            )
            top_axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
            map_axes = [figure.add_subplot(grid[1, index]) for index in range(3)]
            report: dict[str, Any] = {}
            for index, (profile, label) in enumerate(profiles):
                record = records[profile]
                top_axes[index].imshow(Image.open(renders[profile]).convert("RGB"))
                top_axes[index].set_title(
                    f"({chr(97 + index)})  {label}",
                    loc="left",
                    fontsize=8.25,
                    pad=4,
                )
                primitive_count = int(record["prior"]["primitive_count"])
                prior_time = float(record["prior"]["total_prior_seconds"])
                quality = record["heldout_rgb_quality"]
                top_axes[index].text(
                    0.02,
                    0.035,
                    f"{primitive_count / 1e6:.2f}M splats  |  "
                    f"PSNR {quality['psnr_db_mean']:.1f} dB  |  prior {prior_time / 60:.1f} min",
                    transform=top_axes[index].transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=6.65,
                    color="white",
                    bbox={"facecolor": "#111827", "edgecolor": "none", "alpha": 0.78, "pad": 2.2},
                )
                clean_image_axis(top_axes[index])

                anchor_type = torch.as_tensor(maps[profile]["anchor_type"]).numpy()
                colors = np.where(anchor_type == 1, COLORS["core"], COLORS["reserve"])
                _scatter_map(
                    map_axes[index], projected[profile], colors, size=1.25, alpha=0.75
                )
                map_axes[index].set_xlim(low[0], high[0])
                map_axes[index].set_ylim(low[1], high[1])
                a0 = record["A0_bootstrap"]["median_te_cm"]["mean"]
                a1 = record["A1_reconstructed"]["median_te_cm"]["mean"]
                raw = record["A1_reconstructed"]["raw_gt_precision_2px_percent"]["mean"]
                inlier = record["A1_reconstructed"]["inlier_gt_precision_2px_percent"]["mean"]
                count = int(record["topology_distillation"]["final_anchor_count"])
                map_axes[index].set_title(
                    f"A1 topology · {count:,} anchors\n"
                    f"median TE {a0:.1f} → {a1:.1f} cm\n"
                    f"raw / inlier P@2  {raw:.1f}% / {inlier:.1f}%",
                    loc="left",
                    fontsize=6.85,
                    pad=3,
                )
                report[profile] = {
                    "prior_primitives": primitive_count,
                    "prior_seconds": prior_time,
                    "psnr_db_mean": quality["psnr_db_mean"],
                    "A0_median_te_cm": a0,
                    "A1_median_te_cm": a1,
                    "A1_raw_gt_precision_2px_percent": raw,
                    "A1_inlier_gt_precision_2px_percent": inlier,
                    "final_anchor_count": count,
                    "render": str(renders[profile]),
                }
            figure.text(
                0.025,
                0.965,
                "Same scene · same mapping images, tracks, LaFGS schedule and one-shot sparse deployment",
                ha="left",
                va="top",
                fontsize=8.2,
                color=COLORS["ink"],
                fontweight="semibold",
            )
            figure.legend(
                handles=[
                    Line2D([0], [0], marker="o", linestyle="", color=COLORS["core"], label="Track core"),
                    Line2D([0], [0], marker="o", linestyle="", color=COLORS["reserve"], label="Gaussian reserve"),
                ],
                loc="lower right",
                bbox_to_anchor=(0.985, 0.005),
                ncol=2,
                handletextpad=0.3,
                columnspacing=0.9,
            )
            paths = save_figure(figure, self.output_dir, "figure_05_prior_flexibility")
        return {
            "schema": FIGURE_SCHEMA,
            "version": FIGURE_VERSION,
            "figure": "prior_flexibility",
            "scene": "OldHospital",
            "render_view": query_name,
            "protocol": "only the frozen RGB Gaussian prior changes",
            "profiles": report,
            "audit": source_record(audit_path),
            "outputs": paths,
        }

    def build_all(self) -> dict[str, Any]:
        # Match diagnostics and the track example are cached and reused across figures.
        figures = {
            "figure_04": self.build_match_comparison(),
            "figure_02": self.build_primitive_identity(),
            "figure_03": self.build_topology_distillation(),
            "figure_01": self.build_method_overview(),
            "figure_05": self.build_prior_flexibility(),
        }
        manifest = {
            "schema": "lafgs_publication_figure_set",
            "version": 1,
            "seed": self.seed,
            "device": str(self.device),
            "style": {
                "vector_outputs": ["PDF", "SVG"],
                "raster_output": "PNG at 360 dpi",
                "color_semantics": COLORS,
                "font": "DejaVu Sans",
            },
            "sources": {
                "ShopFacade_prior": source_record(self.shop_prior_ply),
                "ShopFacade_track_payload": source_record(self.shop_track_payload),
                "ShopFacade_A0": source_record(self.shop_a0_state),
                "ShopFacade_A1": source_record(self.shop_a1_map),
                "mainline_config": source_record(self.artifacts.config),
            },
            "figures": figures,
        }
        path = self.output_dir / "figure_manifest.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest
