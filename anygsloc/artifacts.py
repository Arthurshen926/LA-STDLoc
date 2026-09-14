"""Minimal deployment map, distinct from a research map and its observation CSR."""

import hashlib
from pathlib import Path
import torch

SCHEMA = "anygsloc_deployment_map_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(state):
    if (
        set(state)
        != {"schema", "anchor_ids", "anchor_xyz", "anchor_features", "provenance"}
        or state["schema"] != SCHEMA
    ):
        raise ValueError("not a minimal deployment map")
    ids, xyz, features = (
        state[k] for k in ["anchor_ids", "anchor_xyz", "anchor_features"]
    )
    n = ids.numel()
    if (
        ids.dtype != torch.int64
        or ids.shape != (n,)
        or not n
        or ids.unique().numel() != n
    ):
        raise ValueError("invalid anchor identities")
    if (
        xyz.shape != (n, 3)
        or features.shape != (n, 256)
        or xyz.dtype != torch.float32
        or features.dtype != torch.float32
    ):
        raise ValueError("deployment map row alignment/dtype mismatch")
    if (
        not torch.isfinite(xyz).all()
        or not torch.isfinite(features).all()
        or (features.norm(dim=1) == 0).any()
    ):
        raise ValueError("invalid geometry or descriptor")
    return state


def validate_identity(metric, state, source):
    if (
        metric.get("schema") != "lafgs_shared_metric_state"
        or metric.get("version") != 1
        or metric.get("protocol")
        not in {"v6_identity_shared_metric", "rendered_track_map_bound_identity"}
        or metric.get("step") != 0
        or metric.get("metric_config")
        != dict(descriptor_dim=256, rank=1, max_residual_norm=0.0)
        or metric.get("map_sha256") != sha256(source)
        or not torch.equal(metric["landmark_indices"], state["anchor_ids"])
    ):
        raise ValueError("expected an aligned, map-bound frozen identity metric")
    parameters = metric.get("metric_state_dict", {})
    shapes = {"down.weight": (1, 256), "down.bias": (1,), "up.weight": (256, 1)}
    if set(parameters) != set(shapes) or any(
        tuple(parameters[k].shape) != shape or bool(torch.count_nonzero(parameters[k]))
        for k, shape in shapes.items()
    ):
        raise ValueError("identity parameters must be exactly zero")


def export_map(source, metric_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    state = torch.load(source, map_location="cpu", weights_only=False)
    metric = torch.load(metric_path, map_location="cpu", weights_only=False)
    validate_identity(metric, state, source)
    for key in ("anchor_extra_prototype_features", "anchor_extra_prototype_owner_rows"):
        if key in state and state[key].numel():
            raise ValueError(
                "multiple-prototype research maps are not a frozen release map"
            )
    if state.get("schema") != "lafgs_materialized_anchor_map":
        raise ValueError("unsupported source map")
    if (
        state.get("provenance", {}).get("uses_test_queries") is not False
        or state.get("provenance", {}).get("uses_source_mapping_rgb") is not False
    ):
        raise ValueError("export requires mapping-only rendered observations")
    if metric["metric_config"]["max_residual_norm"] != 0 or not torch.equal(
        metric["landmark_indices"], state["anchor_ids"]
    ):
        raise ValueError("only aligned identity metric is supported")
    if metric.get("map_sha256") != sha256(source):
        raise ValueError("metric/map hash mismatch")
    if state.get("photometric_canonicalization_contract") or metric.get(
        "photometric_canonicalization_contract"
    ):
        raise ValueError("photometric adaptation is outside release scope")
    tensors = metric.get("metric_state_dict", {})
    if any(not torch.isfinite(v).all() for v in tensors.values()):
        raise ValueError("nonfinite identity state")
    minimal = dict(
        schema=SCHEMA,
        **{
            k: state[k].detach().cpu().clone()
            for k in ["anchor_ids", "anchor_xyz", "anchor_features"]
        },
        provenance=dict(
            source_map_sha256=sha256(source),
            source_metric_sha256=sha256(metric_path),
            source=str(Path(source).resolve()),
            training=False,
        ),
    )
    validate(minimal)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(minimal, output)
    return dict(
        anchor_count=len(minimal["anchor_ids"]),
        file_bytes=output.stat().st_size,
        tensor_bytes=sum(
            minimal[k].numel() * minimal[k].element_size()
            for k in ["anchor_ids", "anchor_xyz", "anchor_features"]
        ),
        sha256=sha256(output),
    )
