"""Explicit experimental input contracts; production source-free contracts stay intact."""

from __future__ import annotations

import torch

from anygsloc.mapping.contracts import ASSOCIATION_GRAPH_SCHEMA
from anygsloc.mapping.observation_provider import ObservationProvider

MODES = ("pure_gs", "raw_only", "hybrid", "real_gs_assisted")
SPARSE = ("native_keypoints", "native_descriptors", "native_scores")
GEOMETRY = ("native_K", "pose_w2c", "native_input_hw")
AUXILIARY = (
    "native_depth",
    "native_alpha",
    "native_valid_mask",
    "native_depth_at_keypoints",
    "native_alpha_at_keypoints",
    "native_valid_keypoint_mask",
    "v2_valid",
)


def source_contract(observations):
    if "input_mode" not in observations.payload:
        return {"uses_source_mapping_rgb": False, "uses_test_queries": False}
    p = observations.payload
    mode = p.get("input_mode")
    if mode not in MODES or p.get("uses_test_queries") is not False:
        raise ValueError("invalid mapping-only experimental input")
    expected = mode != "pure_gs"
    if p.get("uses_source_mapping_rgb") is not expected:
        raise ValueError("source RGB declaration differs from input mode")
    scope = {
        "uses_source_mapping_rgb": expected,
        "uses_test_queries": False,
        "input_mode": mode,
        "experimental_extension": True,
    }
    if mode == "real_gs_assisted":
        if p.get("uses_render_descriptors") is not False:
            raise ValueError("assisted input must contain only real-image descriptors")
        scope["uses_render_descriptors"] = False
    return scope


def require_association(association, observations):
    expected = source_contract(observations)
    if association.get("schema") != ASSOCIATION_GRAPH_SCHEMA:
        raise ValueError("wrong experimental association schema")
    for k, v in expected.items():
        if association.get(k) != v:
            raise ValueError("association input contract differs: " + k)


def camera_identity(record):
    return (
        tuple(torch.as_tensor(record["pose_w2c"]).double().reshape(-1).tolist())
        + tuple(torch.as_tensor(record["native_K"]).double().reshape(-1).tolist())
        + tuple(torch.as_tensor(record["native_input_hw"]).reshape(-1).tolist())
    )


def validate_payload(payload):
    names = list(payload["queries"])
    whitelist = payload.get("mapping_names")
    if names != whitelist or len(set(names)) != len(names):
        raise ValueError("cache must exactly cover ordered mapping whitelist")
    if set(names) & set(payload.get("test_names", ())):
        raise ValueError("test camera entered mapping cache")
    identities = [camera_identity(r) for r in payload["queries"].values()]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate physical camera geometry")
    mode = payload["input_mode"]
    for name, r in payload["queries"].items():
        n = len(r["native_keypoints"])
        if n > 2048:
            raise ValueError("per-camera total exceeds 2048")
        src = torch.as_tensor(r["row_source"]).long()
        if src.shape != (n,) or bool(((src < 0) | (src > 1)).any()):
            raise ValueError("invalid row provenance")
        if mode == "hybrid" and any(int((src == k).sum()) > 1024 for k in (0, 1)):
            raise ValueError("hybrid exceeds source quota")
        if mode == "raw_only" and bool((src != 0).any()):
            raise ValueError("raw-only contains render rows")
        if mode == "pure_gs" and bool((src != 1).any()):
            raise ValueError("pure GS contains real rows")
        if mode == "raw_only" and any(k in r for k in AUXILIARY):
            raise ValueError("raw-only contains Gaussian evidence")
    return payload


class MatrixObservationProvider(ObservationProvider):
    def __init__(self, payload, **kwargs):
        validate_payload(payload)
        kind = "gaussian_render" if payload["input_mode"] == "pure_gs" else "real_rgb"
        super().__init__(payload, source_kind=kind, **kwargs)
        source_contract(self)


def classic_rgb_records(payload):
    """Deliberately drop every auxiliary field before classic construction."""
    validate_payload(payload)
    return [
        {k: torch.as_tensor(r[k]).clone() for k in SPARSE + GEOMETRY}
        for r in payload["queries"].values()
    ]
