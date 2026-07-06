import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch

from localization_training.episode_sampler import sample_interpolated_novel_view, sample_spatial_offset_novel_view
from la_artifacts.quality_gate import TeacherCacheGate, TeacherCacheGateConfig, summarize_gate_decisions

PseudoQueryCamera = SimpleNamespace


def normalize_image_name(name):
    return str(name).replace("\\", "/").lstrip("./")


def _pose_to_list(pose):
    return torch.as_tensor(pose, dtype=torch.float32).reshape(4, 4).cpu().tolist()


def _image_to_tensor(path, device="cpu"):
    import numpy as np
    from PIL import Image

    image = Image.open(path).convert("RGB")
    tensor = torch.from_numpy(np.array(image)).to(dtype=torch.float32) / 255.0
    return tensor.permute(2, 0, 1).to(device)


@dataclass
class PseudoQueryRecord:
    query_id: str
    scene: str
    source: str
    image_name: str
    image_path: str
    pose_w2c: list
    fovx: float
    fovy: float
    width: int
    height: int
    accepted: bool = True
    reason: str = "ok"
    artifact_score: float = 0.0
    repair_action: str = "none"
    nearest_train_image: str = ""
    synthetic_alpha: float = 0.0
    teacher_cache_key: str = ""
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_camera(cls, camera, scene, image_root=None, source="train_rgb", train_index=None):
        image_name = normalize_image_name(getattr(camera, "image_name", ""))
        image_path = ""
        if image_root:
            image_path = os.path.join(os.fspath(image_root), image_name)
        pose_w2c = getattr(camera, "world_view_transform").transpose(0, 1)
        meta = {}
        if train_index is not None:
            train_index = int(train_index)
            meta.update(
                {
                    "wildgaussians_embedding_train_index": train_index,
                    "wildgaussians_appearance_train_indices": [train_index],
                    "wildgaussians_appearance_weights": [1.0],
                }
            )
        return cls(
            query_id=f"{source}:{image_name}",
            scene=str(scene),
            source=str(source),
            image_name=image_name,
            image_path=image_path,
            pose_w2c=_pose_to_list(pose_w2c),
            fovx=float(camera.FoVx),
            fovy=float(camera.FoVy),
            width=int(getattr(camera, "image_width", 0)),
            height=int(getattr(camera, "image_height", 0)),
            teacher_cache_key=f"{source}:{image_name}",
            meta=meta,
        )

    def to_camera(self, device="cpu"):
        image = _image_to_tensor(self.image_path, device=device) if self.image_path else None
        pose = torch.as_tensor(self.pose_w2c, dtype=torch.float32)
        camera = SimpleNamespace(
            uid=self.query_id,
            colmap_id=self.query_id,
            image_name=self.image_name,
            original_image=image,
            image_width=int(self.width),
            image_height=int(self.height),
            FoVx=float(self.fovx),
            FoVy=float(self.fovy),
            world_view_transform=pose.transpose(0, 1),
            gt_alpha_mask=None,
            pseudo_query_record=self,
        )
        camera.camera_center = camera.world_view_transform.inverse()[3, :3]
        return camera


@dataclass
class PseudoQueryManifest:
    version: int
    records: list

    @classmethod
    def load(cls, path):
        text = Path(path).read_text()
        stripped = text.lstrip()
        records = []
        if stripped.startswith("{"):
            first_line = stripped.splitlines()[0] if stripped.splitlines() else stripped
            try:
                payload = json.loads(stripped)
                if isinstance(payload, dict) and "records" in payload:
                    for row in payload.get("records", []):
                        records.append(PseudoQueryRecord(**row))
                    return cls(version=int(payload.get("version", 1)), records=records)
                if isinstance(payload, dict) and "query_id" in payload:
                    records.append(PseudoQueryRecord(**payload))
                    return cls(version=1, records=records)
            except json.JSONDecodeError:
                json.loads(first_line)
        for line in text.splitlines():
            line = line.strip()
            if line:
                records.append(PseudoQueryRecord(**json.loads(line)))
        return cls(version=1, records=records)

    def save_jsonl(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for record in self.records:
                f.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    def accepted(self, sources=None):
        allowed = {str(item) for item in sources} if sources else None
        rows = [row for row in self.records if row.accepted and (allowed is None or row.source in allowed)]
        return PseudoQueryManifest(version=self.version, records=rows)

    def filter_by_teacher_cache(
        self,
        cache,
        max_sparse_te=None,
        max_dense_te=None,
        allowed_stages=None,
        require_not_failed=True,
    ):
        rows = []
        for row in self.records:
            key = row.teacher_cache_key or row.query_id
            item = cache.get(key) if cache is not None else None
            if PseudoTeacherCache.item_is_usable(
                item,
                max_sparse_te=max_sparse_te,
                max_dense_te=max_dense_te,
                allowed_stages=allowed_stages,
                require_not_failed=require_not_failed,
            ):
                rows.append(row)
        return PseudoQueryManifest(version=self.version, records=rows)

    def gate_by_teacher_cache(
        self,
        cache,
        max_sparse_te=100.0,
        max_dense_te=100.0,
        allowed_stages=("teacher_ok",),
        require_not_failed=True,
    ):
        gate = TeacherCacheGate(
            TeacherCacheGateConfig(
                max_sparse_te=max_sparse_te,
                max_dense_te=max_dense_te,
                allowed_stages=tuple(allowed_stages or ()),
                require_not_failed=bool(require_not_failed),
            )
        )
        rows = []
        decisions = []
        source_counts = {}
        for row in self.records:
            key = row.teacher_cache_key or row.query_id
            decision = gate.evaluate(cache.get(key) if cache is not None else None)
            decisions.append(decision)
            source_counts.setdefault(row.source, {"accepted": 0, "rejected": 0})
            if decision.accepted:
                rows.append(row)
                source_counts[row.source]["accepted"] += 1
            else:
                source_counts[row.source]["rejected"] += 1
        summary = summarize_gate_decisions(decisions)
        summary["source_counts"] = dict(sorted(source_counts.items()))
        summary["gate"] = {
            "max_sparse_te_cm": max_sparse_te,
            "max_dense_te_cm": max_dense_te,
            "allowed_stages": list(allowed_stages or []),
            "require_not_failed": bool(require_not_failed),
        }
        return PseudoQueryManifest(version=self.version, records=rows), summary

    def source_counts(self):
        counts = {}
        for row in self.records:
            key = f"{row.source}:{'accepted' if row.accepted else 'rejected'}"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))


class PseudoTeacherCache:
    def __init__(self, items=None):
        self.items = items or {}

    @classmethod
    def load(cls, path):
        if not path:
            return cls()
        try:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
            if isinstance(payload, dict) and "items" in payload:
                return cls(payload["items"])
            return cls(payload if isinstance(payload, dict) else {})
        except Exception:
            with open(path) as f:
                payload = json.load(f)
            return cls(payload.get("items", payload))

    def get(self, key, default=None):
        return self.items.get(key, default)

    @staticmethod
    def _float_or_none(value):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def item_is_usable(
        item,
        max_sparse_te=None,
        max_dense_te=None,
        allowed_stages=None,
        require_not_failed=True,
    ):
        if not item:
            return False
        if require_not_failed and bool(item.get("failed", False)):
            return False
        if allowed_stages:
            stage = str(item.get("failure_stage", ""))
            if stage not in {str(value) for value in allowed_stages}:
                return False
        sparse_te = PseudoTeacherCache._float_or_none(item.get("te"))
        dense_te = PseudoTeacherCache._float_or_none(item.get("dense_te"))
        if max_sparse_te is not None:
            if sparse_te is None or sparse_te > float(max_sparse_te):
                return False
        if max_dense_te is not None:
            if dense_te is None or dense_te > float(max_dense_te):
                return False
        return True

    def error_distribution(self):
        translations = []
        rotations = []
        for item in self.items.values():
            if item.get("te") is not None:
                translations.append(float(item["te"]) / 100.0)
            if item.get("ae") is not None:
                rotations.append(float(item["ae"]))
        return {
            "translation": torch.tensor(translations or [0.0], dtype=torch.float32),
            "rotation_deg": torch.tensor(rotations or [0.0], dtype=torch.float32),
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"items": self.items}, path)


class PseudoQuerySampler:
    def __init__(
        self,
        records,
        real_weight=2.0,
        synthetic_weight=1.0,
        seed=0,
        sampling_mode="record_proportional",
    ):
        self.real = [row for row in records if row.accepted and row.source == "train_rgb"]
        self.synthetic = [row for row in records if row.accepted and row.source == "synthetic_rgb"]
        self.real_weight = max(0.0, float(real_weight))
        self.synthetic_weight = max(0.0, float(synthetic_weight))
        self.sampling_mode = str(sampling_mode or "record_proportional").strip().lower()
        if self.sampling_mode not in {"source_balanced", "record_proportional"}:
            raise ValueError(f"Unknown pseudo-query sampling mode: {sampling_mode}")
        self.rng = random.Random(int(seed))

    def sample_record(self):
        if not self.real and not self.synthetic:
            return None
        if not self.synthetic:
            return self.rng.choice(self.real)
        if not self.real:
            return self.rng.choice(self.synthetic)
        if self.sampling_mode == "record_proportional":
            weighted = []
            weighted.extend((row, self.real_weight) for row in self.real if self.real_weight > 0)
            weighted.extend((row, self.synthetic_weight) for row in self.synthetic if self.synthetic_weight > 0)
            if not weighted:
                weighted = [(row, 1.0) for row in self.real]
            rows, weights = zip(*weighted)
            return self.rng.choices(rows, weights=weights, k=1)[0]
        total = self.real_weight + self.synthetic_weight
        use_synth = total > 0 and self.rng.random() < self.synthetic_weight / total
        return self.rng.choice(self.synthetic if use_synth else self.real)


def _normalize_weights(weights):
    values = [float(weight) for weight in weights]
    total = sum(values)
    if total <= 0:
        return [1.0 / len(values) for _ in values] if values else []
    return [float(weight / total) for weight in values]


def apply_wildgaussians_appearance_strategy(records, strategy="blend"):
    strategy = str(strategy or "blend").strip().lower()
    if strategy not in {"blend", "nearest", "none", "endpoint_a", "endpoint_b"}:
        raise ValueError(f"Unknown WildGaussians appearance strategy: {strategy}")
    for record in records:
        meta = getattr(record, "meta", {}) or {}
        indices = list(meta.get("wildgaussians_appearance_train_indices") or [])
        weights = list(meta.get("wildgaussians_appearance_weights") or [])
        if strategy == "none":
            meta.pop("wildgaussians_embedding_train_index", None)
            meta.pop("wildgaussians_appearance_train_indices", None)
            meta.pop("wildgaussians_appearance_weights", None)
            meta["wildgaussians_appearance_strategy"] = "none"
            record.meta = meta
            continue
        if not indices:
            meta["wildgaussians_appearance_strategy"] = strategy
            record.meta = meta
            continue
        if not weights:
            weights = [1.0] * len(indices)
        if len(indices) != len(weights):
            raise ValueError("WildGaussians appearance indices and weights must have the same length.")
        if strategy == "blend":
            selected_indices = [int(index) for index in indices]
            selected_weights = _normalize_weights(weights)
        elif strategy == "nearest":
            best = max(range(len(indices)), key=lambda idx: float(weights[idx]))
            selected_indices = [int(indices[best])]
            selected_weights = [1.0]
        elif strategy == "endpoint_a":
            selected_indices = [int(indices[0])]
            selected_weights = [1.0]
        else:
            selected_indices = [int(indices[-1])]
            selected_weights = [1.0]
        meta["wildgaussians_appearance_train_indices"] = selected_indices
        meta["wildgaussians_appearance_weights"] = selected_weights
        meta["wildgaussians_embedding_train_index"] = selected_indices[0] if len(selected_indices) == 1 else None
        meta["wildgaussians_appearance_strategy"] = strategy
        record.meta = meta
    return records


def synthetic_records_from_cameras(
    cameras,
    scene,
    count,
    image_root,
    seed=0,
    alpha_min=0.35,
    alpha_max=0.65,
    pose_sampler="adjacent_interpolate",
    spatial_min_offset_ratio=1.0,
    spatial_max_offset_ratio=3.0,
    spatial_yaw_deg=20.0,
    spatial_height_offset_ratio=0.15,
):
    rng = random.Random(int(seed))
    records = []
    cameras = list(cameras)
    pose_sampler = str(pose_sampler or "adjacent_interpolate").strip().lower()
    for idx in range(int(count)):
        generator = torch.Generator().manual_seed(rng.randint(0, 2**31 - 1))
        if pose_sampler in {"adjacent", "adjacent_interpolate", "interpolate"}:
            view = sample_interpolated_novel_view(
                cameras,
                generator=generator,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
            )
        elif pose_sampler in {"spatial", "spatial_offset"}:
            view = sample_spatial_offset_novel_view(
                cameras,
                generator=generator,
                min_offset_ratio=spatial_min_offset_ratio,
                max_offset_ratio=spatial_max_offset_ratio,
                yaw_deg=spatial_yaw_deg,
                height_offset_ratio=spatial_height_offset_ratio,
            )
        else:
            raise ValueError(f"Unknown synthetic pose sampler: {pose_sampler}")
        name = f"synthetic/{idx:06d}.png"
        pose_w2c = view.world_view_transform.transpose(0, 1)
        alpha = float(view.alpha)
        appearance_indices = [int(view.train_index_a)] if int(view.train_index_a) >= 0 else []
        appearance_weights = [1.0] if appearance_indices else []
        if int(view.train_index_b) >= 0 and int(view.train_index_b) not in appearance_indices:
            appearance_indices.append(int(view.train_index_b))
            appearance_weights = [float(1.0 - alpha), float(alpha)]
        records.append(
            PseudoQueryRecord(
                query_id=f"synthetic_rgb:{name}",
                scene=str(scene),
                source="synthetic_rgb",
                image_name=name,
                image_path=os.path.join(os.fspath(image_root), name),
                pose_w2c=_pose_to_list(pose_w2c),
                fovx=float(view.FoVx),
                fovy=float(view.FoVy),
                width=int(getattr(cameras[0], "image_width", 0)) if cameras else 0,
                height=int(getattr(cameras[0], "image_height", 0)) if cameras else 0,
                accepted=False,
                reason="not_rendered",
                nearest_train_image=str(view.image_name),
                synthetic_alpha=alpha,
                teacher_cache_key=f"synthetic_rgb:{name}",
                meta={
                    "synthetic_pose_sampler": str(view.sampler_mode),
                    "coverage": float(view.coverage),
                    "difficulty": float(view.difficulty),
                    "anchor_train_index": int(view.anchor_index),
                    "nearest_train_distance": float(view.nearest_train_distance),
                    "nearest_train_angle_deg": float(view.nearest_train_angle_deg),
                    "spatial_offset_distance": float(view.spatial_offset_distance),
                    "wildgaussians_appearance_train_indices": appearance_indices,
                    "wildgaussians_appearance_weights": appearance_weights,
                },
            )
        )
    return records
