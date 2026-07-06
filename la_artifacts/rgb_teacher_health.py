import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TensorHealth:
    name: str
    shape: list
    finite_fraction: float
    nan_count: int
    inf_count: int


@dataclass
class CheckpointHealth:
    checkpoint: str
    state_path: str
    ok: bool
    reason: str
    tensors: list = field(default_factory=list)
    bad_tensors: list = field(default_factory=list)

    def to_dict(self):
        payload = asdict(self)
        payload["tensors"] = [asdict(item) if hasattr(item, "__dataclass_fields__") else item for item in self.tensors]
        return payload

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


DEFAULT_MIN_FINITE = {
    "xyz": 0.999,
    "features_dc": 0.999,
    "features_rest": 0.999,
    "scales": 0.999,
    "rotations": 0.999,
    "embeddings": 0.999,
    "appearance_embeddings": 0.999,
    "appearance_mlp": 0.999,
    "opacities": 0.30,
}


def _latest_state_path(checkpoint):
    checkpoint = Path(checkpoint)
    if checkpoint.is_file():
        return checkpoint
    candidates = []
    for path in checkpoint.glob("chkpnt-*.pth"):
        try:
            step = int(path.stem.split("-", 1)[1])
        except (IndexError, ValueError):
            step = -1
        candidates.append((step, path))
    if candidates:
        return sorted(candidates)[-1][1]
    fallback = checkpoint / "chkpnt.pth"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No WildGaussians state checkpoint found under {checkpoint}")


def _threshold_for(name, thresholds):
    if name in thresholds:
        return thresholds[name]
    for prefix, value in thresholds.items():
        if name.startswith(prefix + "."):
            return value
    return None


def check_wildgaussians_checkpoint(checkpoint, min_finite=None):
    import torch

    thresholds = dict(DEFAULT_MIN_FINITE)
    if min_finite:
        thresholds.update(min_finite)

    state_path = _latest_state_path(checkpoint)
    state = torch.load(state_path, map_location="cpu")
    if not isinstance(state, dict):
        return CheckpointHealth(
            checkpoint=str(checkpoint),
            state_path=str(state_path),
            ok=False,
            reason=f"Unsupported checkpoint payload type: {type(state).__name__}",
        )

    tensors = []
    bad = []
    for name, value in state.items():
        if not torch.is_tensor(value) or not value.is_floating_point():
            continue
        threshold = _threshold_for(str(name), thresholds)
        if threshold is None:
            continue
        finite = torch.isfinite(value)
        finite_fraction = float(finite.float().mean().item()) if value.numel() else 1.0
        item = TensorHealth(
            name=str(name),
            shape=list(value.shape),
            finite_fraction=finite_fraction,
            nan_count=int(torch.isnan(value).sum().item()),
            inf_count=int(torch.isinf(value).sum().item()),
        )
        tensors.append(item)
        if finite_fraction < float(threshold):
            bad.append(item)

    if bad:
        names = ", ".join(f"{item.name}:{item.finite_fraction:.3f}" for item in bad[:8])
        return CheckpointHealth(
            checkpoint=str(checkpoint),
            state_path=str(state_path),
            ok=False,
            reason=f"Non-finite critical WildGaussians tensors: {names}",
            tensors=tensors,
            bad_tensors=[asdict(item) for item in bad],
        )
    return CheckpointHealth(
        checkpoint=str(checkpoint),
        state_path=str(state_path),
        ok=True,
        reason="ok",
        tensors=tensors,
        bad_tensors=[],
    )
