"""Independent all-real appearance contract; legacy modes remain unchanged."""

from __future__ import annotations

import torch

from anygsloc.mapping.observation_provider import ObservationProvider
from anygsloc.pipeline.observations import (
    AUXILIARY,
    GEOMETRY,
)
from anygsloc.pipeline.observations import (
    validate_payload as validate_raw_payload,
)

MODE = "real_gs_assisted"
ASSOCIATION_GRAPH_SCHEMA = "cambridge_gs_assisted_real_association_v1"
ANCHOR_CANDIDATE_SCHEMA = "cambridge_gs_assisted_real_candidates_v1"
SCIENCE_FIELDS = (
    "anchor_ids",
    "anchor_xyz",
    "anchor_features",
    "anchor_position_covariance",
    "anchor_matchability",
    "v6_mapping_query_names",
    "v6_mapping_query_bins",
    "projective_anchor_observations",
)


def source_contract(observations):
    p = observations.payload
    if (
        p.get("input_mode") != MODE
        or p.get("uses_source_mapping_rgb") is not True
        or p.get("uses_test_queries") is not False
        or p.get("uses_render_descriptors") is not False
    ):
        raise ValueError("invalid all-real Gaussian-assisted provenance")
    return dict(
        input_mode=MODE,
        uses_source_mapping_rgb=True,
        uses_render_descriptors=False,
        uses_test_queries=False,
        experimental_extension=True,
    )


def require_association(association, observations):
    if association.get("schema") != ASSOCIATION_GRAPH_SCHEMA:
        raise ValueError("wrong assisted association schema")
    for key, value in source_contract(observations).items():
        if association.get(key) != value:
            raise ValueError("assisted association provenance differs: " + key)


def validate_raw(raw):
    if (
        raw.get("input_mode") != "raw_only"
        or raw.get("uses_source_mapping_rgb") is not True
        or raw.get("uses_test_queries") is not False
    ):
        raise ValueError("base must be accepted mapping-only raw input")
    validate_raw_payload(raw)
    for r in raw["queries"].values():
        n = len(r["native_keypoints"])
        if not torch.equal(torch.as_tensor(r["source_row"]).long(), torch.arange(n)):
            raise ValueError("base raw rows must retain original local identity")


def attach_auxiliary(raw, aux, protocol_sha):
    """Never slice sparse rows. Certificate masks only the proposal alpha."""
    validate_raw(raw)
    if (
        aux.get("mapping_names") != raw["mapping_names"]
        or aux.get("uses_test_queries") is not False
    ):
        raise ValueError("auxiliary registry or test boundary differs")
    if list(aux.get("queries", {})) != raw["mapping_names"]:
        raise ValueError("auxiliary observation order differs")
    records = {}
    for name, original in raw["queries"].items():
        a = aux["queries"][name]
        for key in GEOMETRY:
            if not torch.equal(torch.as_tensor(a[key]), torch.as_tensor(original[key])):
                raise ValueError("auxiliary physical camera differs: " + key)
        if not torch.equal(
            torch.as_tensor(a["native_keypoints"]), original["native_keypoints"]
        ):
            raise ValueError("auxiliary sampling coordinates differ")
        n = len(original["native_keypoints"])
        depth = torch.as_tensor(a["depth"])
        alpha = torch.as_tensor(a["alpha"])
        valid = torch.as_tensor(a["certificate_valid"])
        if (
            depth.shape != (n,)
            or alpha.shape != (n,)
            or valid.shape != (n,)
            or valid.dtype != torch.bool
        ):
            raise ValueError("auxiliary shape/certificate dtype differs")
        if not depth.is_floating_point() or not alpha.is_floating_point():
            raise ValueError("auxiliary depth/alpha dtype differs")
        record = dict(original)
        record["native_depth_at_keypoints"] = depth.clone()
        record["native_alpha_at_keypoints"] = torch.where(
            valid, alpha, torch.zeros_like(alpha)
        )
        records[name] = record
    return dict(
        schema="cambridge_gs_assisted_real_observations_v1",
        input_mode=MODE,
        uses_source_mapping_rgb=True,
        uses_render_descriptors=False,
        uses_test_queries=False,
        mapping_names=list(raw["mapping_names"]),
        test_names=list(raw["test_names"]),
        queries=records,
        protocol_sha256=protocol_sha,
    )


class AssistedObservationProvider(ObservationProvider):
    def __init__(self, payload):
        source_contract(type("Payload", (), {"payload": payload})())
        names = payload["mapping_names"]
        if list(payload["queries"]) != names or set(names) & set(payload["test_names"]):
            raise ValueError("assisted mapping registry differs")
        # Validate all original row and physical-camera rules through a stripped raw view.
        original = dict(payload, input_mode="raw_only")
        original["queries"] = {
            name: {key: value for key, value in record.items() if key not in AUXILIARY}
            for name, record in payload["queries"].items()
        }
        validate_raw(original)
        for record in payload["queries"].values():
            if any(
                k in record
                for k in ("v2_valid", "native_valid_mask", "native_valid_keypoint_mask")
            ):
                raise ValueError(
                    "hard validity filtering is forbidden for raw observations"
                )
        super().__init__(payload, source_kind="real_rgb")
