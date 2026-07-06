import csv
import json
import math
import os
from dataclasses import dataclass


class ArtifactThresholds:
    def __init__(
        self,
        severe_psnr=13.5,
        severe_ssim=0.42,
        severe_residual=0.18,
        mild_psnr=15.5,
        mild_ssim=0.56,
        mild_residual=0.10,
        mild_alpha_cov=0.85,
        mild_abs_bias=0.04,
    ):
        self.severe_psnr = float(severe_psnr)
        self.severe_ssim = float(severe_ssim)
        self.severe_residual = float(severe_residual)
        self.mild_psnr = float(mild_psnr)
        self.mild_ssim = float(mild_ssim)
        self.mild_residual = float(mild_residual)
        self.mild_alpha_cov = float(mild_alpha_cov)
        self.mild_abs_bias = float(mild_abs_bias)

    @classmethod
    def from_args(cls, args):
        return cls(
            severe_psnr=args.severe_psnr,
            severe_ssim=args.severe_ssim,
            severe_residual=args.severe_residual,
            mild_psnr=args.mild_psnr,
            mild_ssim=args.mild_ssim,
            mild_residual=args.mild_residual,
            mild_alpha_cov=args.mild_alpha_cov,
            mild_abs_bias=args.mild_abs_bias,
        )

    def to_dict(self):
        return {
            "severe_psnr": self.severe_psnr,
            "severe_ssim": self.severe_ssim,
            "severe_residual": self.severe_residual,
            "mild_psnr": self.mild_psnr,
            "mild_ssim": self.mild_ssim,
            "mild_residual": self.mild_residual,
            "mild_alpha_cov": self.mild_alpha_cov,
            "mild_abs_bias": self.mild_abs_bias,
        }


@dataclass
class ArtifactWeightLookup:
    weights_by_name: dict
    severity_by_name: dict
    default_weight: float = 1.0

    def weight_for_name(self, image_name):
        return float(self.weights_by_name.get(normalize_image_name(image_name), self.default_weight))

    def weight_for_camera(self, camera):
        return self.weight_for_name(getattr(camera, "image_name", ""))

    def severity_for_name(self, image_name):
        return self.severity_by_name.get(normalize_image_name(image_name), "none")

    def summary(self):
        counts = {}
        for severity in self.severity_by_name.values():
            counts[severity] = counts.get(severity, 0) + 1
        return dict(sorted(counts.items()))


@dataclass
class ArtifactRegionWeightLookup:
    maps_by_name: dict
    severity_by_name: dict
    default_weight: float = 1.0

    def __post_init__(self):
        self._cache = {}

    def path_for_name(self, image_name):
        return self.maps_by_name.get(normalize_image_name(image_name))

    def path_for_camera(self, camera):
        return self.path_for_name(getattr(camera, "image_name", ""))

    def map_for_name(self, image_name, device=None, dtype=None):
        path = self.path_for_name(image_name)
        if not path:
            return None
        weight_map = self._cache.get(path)
        if weight_map is None:
            weight_map = load_region_weight_map(path)
            self._cache[path] = weight_map.cpu()
        if device is not None or dtype is not None:
            weight_map = weight_map.to(device=device, dtype=dtype)
        return weight_map

    def map_for_camera(self, camera, device=None, dtype=None):
        return self.map_for_name(getattr(camera, "image_name", ""), device=device, dtype=dtype)

    def sample_weights_for_name(self, image_name, uv, image_size, device=None, dtype=None):
        weight_map = self.map_for_name(image_name, device=device or getattr(uv, "device", None), dtype=dtype)
        return sample_region_weight_map(
            weight_map,
            uv,
            image_size=image_size,
            default_weight=self.default_weight,
        )

    def sample_weights_for_camera(self, camera, uv, image_size, device=None, dtype=None):
        return self.sample_weights_for_name(
            getattr(camera, "image_name", ""),
            uv,
            image_size=image_size,
            device=device,
            dtype=dtype,
        )

    def severity_for_name(self, image_name):
        return self.severity_by_name.get(normalize_image_name(image_name), "none")

    def summary(self):
        counts = {}
        for severity in self.severity_by_name.values():
            counts[severity] = counts.get(severity, 0) + 1
        return dict(sorted(counts.items()))


def normalize_image_name(name):
    return str(name).replace("\\", "/").lstrip("./")


def _as_output_hw(input_hw, output_size):
    if isinstance(output_size, (tuple, list)):
        if len(output_size) != 2:
            raise ValueError("output_size tuple must be (height, width)")
        return max(1, int(output_size[0])), max(1, int(output_size[1]))
    output_size = int(output_size)
    if output_size <= 0:
        return tuple(int(value) for value in input_hw)
    height, width = int(input_hw[0]), int(input_hw[1])
    longest = max(height, width, 1)
    scale = float(output_size) / float(longest)
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def _squeeze_weight_map(weight_map):
    while getattr(weight_map, "dim", lambda: 0)() > 2:
        weight_map = weight_map.squeeze(0)
    if weight_map.dim() != 2:
        raise ValueError(f"Region weight map must be 2D after squeeze, got shape {tuple(weight_map.shape)}")
    return weight_map


def sample_region_weight_map(weight_map, uv, image_size, default_weight=1.0):
    import torch
    import torch.nn.functional as F

    uv = torch.as_tensor(uv)
    if uv.numel() == 0:
        return uv.new_empty((0,), dtype=torch.float32)
    if weight_map is None:
        return torch.full(
            (uv.shape[0],),
            float(default_weight),
            device=uv.device,
            dtype=uv.dtype if uv.is_floating_point() else torch.float32,
        )
    weight_map = _squeeze_weight_map(torch.as_tensor(weight_map))
    dtype = uv.dtype if uv.is_floating_point() else torch.float32
    weight_map = weight_map.to(device=uv.device, dtype=dtype)
    uv = uv.to(device=weight_map.device, dtype=dtype).reshape(-1, 2)
    height, width = int(image_size[0]), int(image_size[1])
    if width > 1:
        x = uv[:, 0] / float(width - 1) * 2.0 - 1.0
    else:
        x = torch.zeros_like(uv[:, 0])
    if height > 1:
        y = uv[:, 1] / float(height - 1) * 2.0 - 1.0
    else:
        y = torch.zeros_like(uv[:, 1])
    grid = torch.stack([x, y], dim=-1).reshape(1, -1, 1, 2)
    sampled = F.grid_sample(
        weight_map.reshape(1, 1, weight_map.shape[-2], weight_map.shape[-1]),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.reshape(-1).clamp(0.0, float(default_weight))


def combine_artifact_confidence(local_weights, image_weight=1.0, mode="product"):
    import torch

    local = torch.as_tensor(local_weights)
    local = local.clamp(0.0, 1.0)
    image = local.new_tensor(float(image_weight)).clamp(0.0, 1.0)
    mode = str(mode).strip().lower()
    if mode == "product":
        return local * image
    if mode == "min":
        return torch.minimum(local, image.expand_as(local))
    if mode == "none":
        return local
    raise ValueError(f"Unsupported artifact confidence combine mode: {mode}")


def local_artifact_weight_map(
    rendered,
    target,
    alpha=None,
    output_size=64,
    default_weight=1.0,
    min_weight=0.25,
    power=1.0,
    residual_start=0.10,
    residual_stop=0.40,
    alpha_threshold=0.05,
):
    import torch
    import torch.nn.functional as F

    rendered = torch.as_tensor(rendered).detach()
    target = torch.as_tensor(target, device=rendered.device, dtype=rendered.dtype).detach()
    if rendered.shape[-2:] != target.shape[-2:]:
        target = F.interpolate(
            target.reshape(1, *target.shape[-3:]),
            size=rendered.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0]
    residual = (rendered - target).abs()
    if residual.dim() == 3:
        residual = residual.mean(dim=0)
    residual_penalty = (residual - float(residual_start)) / max(float(residual_stop) - float(residual_start), 1e-6)
    penalties = [residual_penalty.clamp(0.0, 1.0)]
    if alpha is not None:
        alpha = _squeeze_weight_map(torch.as_tensor(alpha, device=rendered.device, dtype=rendered.dtype).detach())
        if alpha.shape[-2:] != rendered.shape[-2:]:
            alpha = F.interpolate(
                alpha.reshape(1, 1, alpha.shape[-2], alpha.shape[-1]),
                size=rendered.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        alpha_penalty = (float(alpha_threshold) - alpha) / max(float(alpha_threshold), 1e-6)
        penalties.append(alpha_penalty.clamp(0.0, 1.0))
    if len(penalties) == 1:
        penalty = penalties[0]
    else:
        stacked = torch.stack(penalties, dim=0)
        penalty = torch.sqrt((stacked * stacked).mean(dim=0))
    if float(power) != 1.0:
        penalty = penalty.clamp(0.0, 1.0) ** max(float(power), 0.0)
    default_weight = float(default_weight)
    min_weight = max(0.0, min(float(min_weight), default_weight))
    weight_map = default_weight - (default_weight - min_weight) * penalty.clamp(0.0, 1.0)
    output_hw = _as_output_hw(rendered.shape[-2:], output_size)
    if tuple(weight_map.shape[-2:]) != tuple(output_hw):
        weight_map = F.interpolate(
            weight_map.reshape(1, 1, weight_map.shape[-2], weight_map.shape[-1]),
            size=output_hw,
            mode="area",
        )[0, 0]
    return weight_map.clamp(min_weight, default_weight)


def load_region_weight_map(path):
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("weight", "weights", "weight_map", "region_weight"):
            if key in payload:
                return _squeeze_weight_map(torch.as_tensor(payload[key], dtype=torch.float32)).clamp(0.0, 1.0)
        raise ValueError(f"Region weight map payload has no weight tensor: {path}")
    return _squeeze_weight_map(torch.as_tensor(payload, dtype=torch.float32)).clamp(0.0, 1.0)


def comma_set(value, lower=True):
    if value is None:
        return set()
    if isinstance(value, (set, list, tuple)):
        items = value
    else:
        items = str(value).split(",")
    out = set()
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        out.add(text.lower() if lower else text)
    return out


def metric_value(row, key, default=float("nan")):
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def classify_artifact_severity(metrics, thresholds=None):
    thresholds = thresholds or ArtifactThresholds()
    psnr_value = metric_value(metrics, "psnr_mean_matched", metric_value(metrics, "psnr"))
    ssim_value = metric_value(metrics, "ssim")
    residual_value = metric_value(metrics, "residual_frac_025")
    alpha_cov = metric_value(metrics, "alpha_cov_05")
    mean_abs_bias = metric_value(metrics, "mean_abs_bias")

    severe = (
        psnr_value <= thresholds.severe_psnr
        or ssim_value <= thresholds.severe_ssim
        or residual_value >= thresholds.severe_residual
    )
    if severe:
        return "severe"

    mild = (
        psnr_value <= thresholds.mild_psnr
        or ssim_value <= thresholds.mild_ssim
        or residual_value >= thresholds.mild_residual
        or alpha_cov <= thresholds.mild_alpha_cov
        or mean_abs_bias >= thresholds.mild_abs_bias
    )
    return "mild" if mild else "none"


def row_severity(row, thresholds=None):
    severity = str(row.get("gate_severity", row.get("severity", ""))).strip().lower()
    if severity in {"none", "mild", "severe"}:
        return severity
    return classify_artifact_severity(row, thresholds=thresholds)


def metric_quality_weight(
    metrics,
    thresholds=None,
    default_weight=1.0,
    mild_weight=0.65,
    severe_weight=0.25,
):
    severity = row_severity(metrics, thresholds=thresholds)
    if severity == "severe":
        return max(0.0, min(float(severe_weight), float(default_weight)))
    if severity == "mild":
        return max(0.0, min(float(mild_weight), float(default_weight)))
    return float(default_weight)


def _bounded_penalty(value, start, stop, higher_is_worse):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value) or start == stop:
        return 0.0
    if higher_is_worse:
        raw = (value - start) / (stop - start)
    else:
        raw = (start - value) / (start - stop)
    return max(0.0, min(1.0, float(raw)))


def _extended_stop(mild_value, severe_value, higher_is_worse):
    gap = abs(float(mild_value) - float(severe_value))
    if gap <= 0.0:
        return float(severe_value)
    if higher_is_worse:
        return float(severe_value) + gap
    return float(severe_value) - gap


def _rms(values):
    values = [float(value) for value in values]
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def continuous_quality_weight(
    metrics,
    thresholds=None,
    default_weight=1.0,
    min_weight=0.70,
    power=1.0,
):
    thresholds = thresholds or ArtifactThresholds()
    default_weight = float(default_weight)
    min_weight = max(0.0, min(float(min_weight), default_weight))
    power = max(0.0, float(power))

    psnr_value = metric_value(metrics, "psnr_mean_matched", metric_value(metrics, "psnr"))
    ssim_value = metric_value(metrics, "ssim")
    residual_value = metric_value(metrics, "residual_frac_025")
    alpha_cov = metric_value(metrics, "alpha_cov_05")
    mean_abs_bias = metric_value(metrics, "mean_abs_bias")

    psnr_stop = _extended_stop(thresholds.mild_psnr, thresholds.severe_psnr, higher_is_worse=False)
    ssim_stop = _extended_stop(thresholds.mild_ssim, thresholds.severe_ssim, higher_is_worse=False)
    residual_stop = _extended_stop(thresholds.mild_residual, thresholds.severe_residual, higher_is_worse=True)
    bias_stop = thresholds.mild_abs_bias * 3.0

    penalties = [
        _bounded_penalty(psnr_value, thresholds.mild_psnr, psnr_stop, higher_is_worse=False),
        _bounded_penalty(ssim_value, thresholds.mild_ssim, ssim_stop, higher_is_worse=False),
        _bounded_penalty(residual_value, thresholds.mild_residual, residual_stop, higher_is_worse=True),
        _bounded_penalty(alpha_cov, thresholds.mild_alpha_cov, 0.0, higher_is_worse=False),
        _bounded_penalty(mean_abs_bias, thresholds.mild_abs_bias, bias_stop, higher_is_worse=True),
    ]
    penalty = _rms(penalties)
    if penalty <= 0.0:
        return default_weight
    if power != 1.0:
        penalty = penalty**power
    return max(min_weight, min(default_weight, default_weight - (default_weight - min_weight) * penalty))


def iter_artifact_rows(path):
    suffix = os.path.splitext(os.fspath(path))[1].lower()
    if suffix == ".json":
        with open(path) as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            payload = payload.get("items", payload.get("rows", []))
        if not isinstance(payload, list):
            raise ValueError(f"Artifact JSON must contain a list of rows: {path}")
        for row in payload:
            if isinstance(row, dict):
                yield row
        return

    with open(path, newline="") as f:
        yield from csv.DictReader(f)


def _row_matches(row, scene_name=None, splits=None, severities=None, thresholds=None):
    row_scene = str(row.get("scene", "")).strip()
    if scene_name and row_scene and row_scene != scene_name:
        return False
    allowed_splits = comma_set(splits)
    row_split = str(row.get("split", row.get("dataset_split", ""))).strip().lower()
    if allowed_splits and row_split not in allowed_splits:
        return False
    allowed_severities = comma_set(severities)
    if allowed_severities and row_severity(row, thresholds=thresholds) not in allowed_severities:
        return False
    return True


def load_artifact_filter_names(path, scene_name=None, severities="mild,severe", splits="heldout_query_sample"):
    if not path:
        return set()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact filter file not found: {path}")
    names = set()
    for row in iter_artifact_rows(path):
        if not _row_matches(row, scene_name=scene_name, splits=splits, severities=severities):
            continue
        image_name = row.get("image_name", row.get("name", ""))
        if image_name:
            names.add(normalize_image_name(image_name))
    return names


def filter_cameras_by_artifacts(cameras, artifact_names):
    artifact_names = {normalize_image_name(name) for name in artifact_names}
    filtered = []
    removed = []
    for camera in cameras:
        image_name = normalize_image_name(getattr(camera, "image_name", ""))
        if image_name in artifact_names:
            removed.append(image_name)
        else:
            filtered.append(camera)
    if cameras and not filtered:
        raise ValueError("Artifact filter removed all query cameras.")
    return filtered, removed


def candidate_rows(rows, severities=None, thresholds=None):
    thresholds = thresholds or ArtifactThresholds()
    allowed = comma_set(severities or {"mild", "severe"})
    candidates = []
    for row in rows:
        severity = row_severity(row, thresholds=thresholds)
        if severity not in allowed:
            continue
        out = dict(row)
        out["image_name"] = normalize_image_name(out.get("image_name", ""))
        out["gate_severity"] = severity
        candidates.append(out)
    return candidates


def load_artifact_weight_lookup(
    path,
    scene_name=None,
    splits="heldout_query_sample",
    severities="mild,severe",
    thresholds=None,
    default_weight=1.0,
    mild_weight=0.65,
    severe_weight=0.25,
    mode="severity",
    continuous_min_weight=0.70,
    continuous_power=1.0,
):
    if not path:
        return ArtifactWeightLookup({}, {}, default_weight=float(default_weight))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact weight file not found: {path}")
    thresholds = thresholds or ArtifactThresholds()
    weights = {}
    severities_by_name = {}
    for row in iter_artifact_rows(path):
        if not _row_matches(row, scene_name=scene_name, splits=splits, severities=severities, thresholds=thresholds):
            continue
        image_name = row.get("image_name", row.get("name", ""))
        if not image_name:
            continue
        name = normalize_image_name(image_name)
        severity = row_severity(row, thresholds=thresholds)
        if str(mode).strip().lower() == "continuous":
            weight = continuous_quality_weight(
                row,
                thresholds=thresholds,
                default_weight=default_weight,
                min_weight=continuous_min_weight,
                power=continuous_power,
            )
        else:
            weight = metric_quality_weight(
                row,
                thresholds=thresholds,
                default_weight=default_weight,
                mild_weight=mild_weight,
                severe_weight=severe_weight,
            )
        if name not in weights or weight < weights[name]:
            weights[name] = weight
            severities_by_name[name] = severity
    return ArtifactWeightLookup(weights, severities_by_name, default_weight=float(default_weight))


def _region_map_path_from_row(row):
    for key in ("region_weight_path", "map_path", "artifact_map_path", "weight_map_path"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def _resolve_manifest_path(manifest_path, map_path, root=None):
    if os.path.isabs(map_path):
        return map_path
    base = root if root else os.path.dirname(os.path.abspath(manifest_path))
    return os.path.abspath(os.path.join(os.fspath(base), map_path))


def load_artifact_region_weight_lookup(
    path,
    scene_name=None,
    splits="heldout_query_sample",
    severities="mild,severe",
    thresholds=None,
    default_weight=1.0,
    root=None,
):
    if not path:
        return ArtifactRegionWeightLookup({}, {}, default_weight=float(default_weight))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact region weight manifest not found: {path}")
    thresholds = thresholds or ArtifactThresholds()
    maps = {}
    severities_by_name = {}
    severity_rank = {"none": 0, "mild": 1, "severe": 2}
    for row in iter_artifact_rows(path):
        if not _row_matches(row, scene_name=scene_name, splits=splits, severities=severities, thresholds=thresholds):
            continue
        image_name = row.get("image_name", row.get("name", ""))
        map_path = _region_map_path_from_row(row)
        if not image_name or not map_path:
            continue
        row_root = str(row.get("region_weight_root", "")).strip()
        resolved_path = _resolve_manifest_path(path, map_path, root=row_root or root)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Artifact region weight map not found: {resolved_path}")
        name = normalize_image_name(image_name)
        severity = row_severity(row, thresholds=thresholds)
        previous = severity_rank.get(severities_by_name.get(name, "none"), 0)
        if name not in maps or severity_rank.get(severity, 0) >= previous:
            maps[name] = resolved_path
            severities_by_name[name] = severity
    return ArtifactRegionWeightLookup(maps, severities_by_name, default_weight=float(default_weight))


def weighted_mean(values, weights=None):
    values = [float(value) for value in values]
    if not values or any(not math.isfinite(value) for value in values):
        return float("inf")
    if weights is None:
        return float(sum(values) / len(values))
    weights = [max(0.0, float(weight)) for weight in weights[: len(values)]]
    if len(weights) != len(values) or sum(weights) <= 0.0:
        return float(sum(values) / len(values))
    return float(sum(value * weight for value, weight in zip(values, weights)) / sum(weights))
