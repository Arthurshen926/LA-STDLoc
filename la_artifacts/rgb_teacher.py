import json
import math
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class RgbTeacherSpec:
    scene: str
    source_path: str
    backend: str = "wildgaussians"
    checkpoint: str = ""
    output_root: str = ""
    nerfbaselines_bin: str = "nerfbaselines"
    nerfbaselines_backend: str = "conda"
    train_command: list = field(default_factory=list)
    render_command_template: list = field(default_factory=list)
    fallback_backend: str = "mip-splatting"
    status: str = "planned"
    metrics: dict = field(default_factory=dict)

    def validate(self):
        backend = str(self.backend).lower()
        if backend == "wildgaussians":
            if shutil.which(self.nerfbaselines_bin) is None:
                return False, f"nerfbaselines executable not found: {self.nerfbaselines_bin}"
            if self.checkpoint and not os.path.exists(self.checkpoint):
                return False, f"WildGaussians checkpoint not found: {self.checkpoint}"
        elif backend in {"mip-splatting", "inrepo"}:
            if self.checkpoint and not os.path.exists(self.checkpoint):
                return False, f"RGB teacher checkpoint not found: {self.checkpoint}"
        else:
            return False, f"Unsupported RGB teacher backend: {self.backend}"
        return True, "ok"


@dataclass
class RgbTeacherManifest:
    version: int
    teachers: list

    @classmethod
    def single(cls, spec):
        return cls(version=1, teachers=[spec])

    @classmethod
    def load(cls, path):
        with open(path) as f:
            payload = json.load(f)
        teachers = [RgbTeacherSpec(**item) for item in payload.get("teachers", [])]
        return cls(version=int(payload.get("version", 1)), teachers=teachers)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": int(self.version),
            "teachers": [asdict(item) for item in self.teachers],
        }
        with path.open("w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)


def wildgaussians_train_command(
    source_path,
    output_root,
    scene,
    output_path=None,
    nerfbaselines_bin="nerfbaselines",
    nerfbaselines_backend="conda",
    train_steps=0,
    logger="tensorboard",
    save_iters=None,
    eval_few_iters=None,
    eval_all_iters=None,
    config_sets=None,
    disable_output_artifact=False,
):
    command = [
        nerfbaselines_bin,
        "train",
        "--method",
        "wild-gaussians",
        "--data",
        os.path.abspath(source_path),
        "--output",
        os.path.abspath(output_path) if output_path else os.path.abspath(os.path.join(output_root, scene)),
    ]
    if nerfbaselines_backend:
        command.extend(["--backend", nerfbaselines_backend])
    if logger:
        command.extend(["--logger", logger])
    train_steps = int(train_steps or 0)
    if train_steps > 0:
        save_iters = str(save_iters if save_iters is not None else train_steps)
        eval_few_iters = str(eval_few_iters if eval_few_iters is not None else train_steps)
        eval_all_iters = str(eval_all_iters if eval_all_iters is not None else 999999)
        command.extend(
            [
                "--set",
                f"iterations={train_steps}",
                "--save-iters",
                save_iters,
                "--eval-few-iters",
                eval_few_iters,
                "--eval-all-iters",
                eval_all_iters,
            ]
        )
    for item in config_sets or []:
        item = str(item).strip()
        if item:
            command.extend(["--set", item])
    if disable_output_artifact:
        command.append("--disable-output-artifact")
    return command


def normalize_render_resolution(resolution):
    if resolution is None:
        return ""
    resolution = str(resolution).strip().lower()
    if not resolution:
        return ""
    parts = resolution.split("x")
    if len(parts) != 2:
        raise ValueError(f"Render resolution must use WIDTHxHEIGHT format, got: {resolution}")
    try:
        width, height = (int(parts[0]), int(parts[1]))
    except ValueError as exc:
        raise ValueError(f"Render resolution must use integer WIDTHxHEIGHT format, got: {resolution}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"Render resolution must be positive, got: {resolution}")
    return f"{width}x{height}"


def wildgaussians_render_command_template(
    checkpoint,
    nerfbaselines_bin="nerfbaselines",
    nerfbaselines_backend="conda",
    output_names="color",
    resolution="",
):
    resolution = normalize_render_resolution(resolution)
    command = [
        nerfbaselines_bin,
        "render-trajectory",
        "--checkpoint",
        os.path.abspath(checkpoint) if checkpoint else "{checkpoint}",
        "--trajectory",
        "{trajectory_json}",
        "--output",
        "{output_path}",
        "--output-names",
        str(output_names),
    ]
    if resolution:
        command.extend(["--resolution", resolution])
    if nerfbaselines_backend:
        command.extend(["--backend", nerfbaselines_backend])
    return command


def _record_value(record, name):
    if isinstance(record, dict):
        return record[name]
    return getattr(record, name)


def _record_meta(record):
    if isinstance(record, dict):
        return record.get("meta", {}) or {}
    return getattr(record, "meta", {}) or {}


def _parse_scalar_config_value(value):
    value = str(value).strip()
    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def read_wildgaussians_config(checkpoint):
    checkpoint = Path(checkpoint)
    config_path = checkpoint / "config.yaml" if checkpoint.is_dir() else checkpoint.parent / "config.yaml"
    if not config_path.exists():
        return {}
    config = {}
    for line in config_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        config[key.strip()] = _parse_scalar_config_value(value)
    return config


def _records_have_appearance_metadata(records):
    records = list(records or [])
    return bool(records) and all(_record_appearance_pairs(record) for record in records)


def resolve_wildgaussians_appearance_mode(mode, checkpoint, records=None):
    mode = str(mode or "auto").strip().lower()
    if mode in {"none", "record"}:
        return mode
    if mode != "auto":
        raise ValueError(f"Unknown WildGaussians appearance mode: {mode}")
    config = read_wildgaussians_config(checkpoint) if checkpoint else {}
    appearance_enabled = config.get("appearance_enabled")
    if appearance_enabled is False:
        return "none"
    if appearance_enabled is True and _records_have_appearance_metadata(records):
        return "record"
    return "none"


def fov_to_pinhole_intrinsics(width, height, fovx, fovy):
    width = int(width)
    height = int(height)
    fx = 0.5 * float(width) / math.tan(0.5 * float(fovx))
    fy = 0.5 * float(height) / math.tan(0.5 * float(fovy))
    return [float(fx), float(fy), float(width) * 0.5, float(height) * 0.5]


def _record_appearance_pairs(record):
    meta = _record_meta(record)
    indices = meta.get("wildgaussians_appearance_train_indices")
    weights = meta.get("wildgaussians_appearance_weights")
    if indices is None and meta.get("wildgaussians_embedding_train_index") is not None:
        indices = [meta.get("wildgaussians_embedding_train_index")]
        weights = [1.0]
    if indices is None:
        return []
    if weights is None:
        weights = [1.0] * len(indices)
    if len(indices) != len(weights):
        raise ValueError("WildGaussians appearance indices and weights must have the same length.")
    pairs = []
    for index, weight in zip(indices, weights):
        weight = float(weight)
        if weight > 0:
            pairs.append((int(index), weight))
    return pairs


def _trajectory_appearances_and_weights(records, appearance_mode):
    if appearance_mode in ("", None, "none"):
        return [], [[] for _ in records]
    if appearance_mode != "record":
        raise ValueError(f"Unknown appearance_mode: {appearance_mode}")
    per_record_pairs = [_record_appearance_pairs(record) for record in records]
    if not any(per_record_pairs):
        return [], [[] for _ in records]
    if not all(per_record_pairs):
        raise ValueError("All records must provide WildGaussians appearance metadata when appearance_mode='record'.")
    train_indices = sorted({index for pairs in per_record_pairs for index, _ in pairs})
    index_to_column = {index: column for column, index in enumerate(train_indices)}
    appearances = [{"embedding_train_index": int(index)} for index in train_indices]
    frame_weights = []
    for pairs in per_record_pairs:
        weights = [0.0] * len(train_indices)
        for index, weight in pairs:
            weights[index_to_column[index]] += float(weight)
        total = sum(weights)
        if total <= 0:
            raise ValueError("WildGaussians appearance weights must sum to a positive value.")
        frame_weights.append([float(weight / total) for weight in weights])
    return appearances, frame_weights


def nerfbaselines_trajectory_from_records(records, fps=1, image_scale=1.0, appearance_mode="none"):
    records = list(records)
    if not records:
        raise ValueError("At least one pseudo-query record is required.")
    image_scale = float(1.0 if image_scale is None else image_scale)
    if image_scale <= 0:
        raise ValueError("image_scale must be positive.")
    sizes = {
        (int(_record_value(record, "width")), int(_record_value(record, "height")))
        for record in records
    }
    if len(sizes) != 1:
        raise ValueError(f"NerfBaselines trajectory requires one image size, got: {sorted(sizes)}")
    width, height = next(iter(sizes))
    render_width = max(1, int(round(float(width) * image_scale)))
    render_height = max(1, int(round(float(height) * image_scale)))
    appearances, frame_weights = _trajectory_appearances_and_weights(records, appearance_mode)
    frames = []
    for idx, record in enumerate(records):
        pose_w2c = np.asarray(_record_value(record, "pose_w2c"), dtype=np.float32).reshape(4, 4)
        pose_c2w = np.linalg.inv(pose_w2c)[:3, :4].astype(np.float32)
        frames.append(
            {
                "pose": pose_c2w.reshape(-1).tolist(),
                "intrinsics": fov_to_pinhole_intrinsics(
                    render_width,
                    render_height,
                    _record_value(record, "fovx"),
                    _record_value(record, "fovy"),
                ),
                "appearance_weights": frame_weights[idx],
            }
        )
    return {
        "format": "nerfbaselines-v1",
        "camera_model": "pinhole",
        "image_size": [int(render_width), int(render_height)],
        "fps": int(fps),
        "frames": frames,
        "appearances": appearances,
    }


def save_nerfbaselines_trajectory(records, path, fps=1, image_scale=1.0, appearance_mode="none"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = nerfbaselines_trajectory_from_records(
        records,
        fps=fps,
        image_scale=image_scale,
        appearance_mode=appearance_mode,
    )
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path
