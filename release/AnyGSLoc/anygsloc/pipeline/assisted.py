"""Append-only Full construction; unchanged native selector consumes truthful evidence."""

from __future__ import annotations

import torch

from anygsloc.mapping.projective_completion import build_projective_completion
from anygsloc.pipeline.observations import (
    MatrixObservationProvider,
    require_association,
)
from anygsloc.pipeline.selector_state import validate_csr

from .assisted_contracts import (
    MODE,
    SCIENCE_FIELDS,
    AssistedObservationProvider,
    attach_auxiliary,
    validate_raw,
)

EMPTY_COMPLETION = (
    "no unused render-valid observations for completion",
    "depth proposals produced no descriptor-consistent component",
    "association graph contains no ray-triangulated Anchor",
)
COMPLETION_PARAMETERS = dict(
    voxel_size_m=0.05,
    alpha_minimum=0.05,
    minimum_similarity=0.7,
    minimum_margin=0.01,
    robust_reweighting=True,
    maximum_epipolar_error_px=2.0,
    minimum_observations=3,
    minimum_camera_families=2,
    maximum_rows_per_view=256,
    safety_maximum_components=100000,
)


def assert_equal(left, right, label):
    if isinstance(left, torch.Tensor):
        if (
            not isinstance(right, torch.Tensor)
            or left.dtype != right.dtype
            or left.shape != right.shape
            or not torch.equal(left, right)
        ):
            raise ValueError("scientific tensor changed: " + label)
    elif isinstance(left, dict):
        if set(left) != set(right):
            raise ValueError("scientific keys changed: " + label)
        for key in left:
            assert_equal(left[key], right[key], label + "." + key)
    elif left != right:
        raise ValueError("scientific value changed: " + label)


def assert_base_prefix(base, combined):
    n = len(base["anchor_xyz"])
    for key in SCIENCE_FIELDS:
        if key == "projective_anchor_observations":
            old = base[key]
            new = combined[key]
            edges = int(old["observation_offsets"][-1])
            for col in ("observation_offsets", "query_indices", "keypoint_indices"):
                amount = n + 1 if col == "observation_offsets" else edges
                assert_equal(old[col], new[col][:amount], key + "." + col)
            if old["keypoint_index_semantics"] != new["keypoint_index_semantics"]:
                raise ValueError("CSR row meaning changed")
        elif key in ("v6_mapping_query_names", "v6_mapping_query_bins"):
            assert_equal(base[key], combined[key], key)
        else:
            assert_equal(base[key], combined[key][:n], key)


def append_candidate(base, candidate, raw, association, protocol_sha):
    records = list(raw["queries"].values())
    n = len(base["anchor_xyz"])
    validate_csr(base["projective_anchor_observations"], n, records)
    if not torch.equal(base["anchor_ids"], torch.arange(n)):
        raise ValueError("base Full row identity is not contiguous")
    if base["v6_mapping_query_names"] != raw["mapping_names"]:
        raise ValueError("base mapping names differ")
    result = dict(base)
    if candidate is not None:
        c = candidate
        m = len(c["anchor_xyz"])
        if (
            c.get("input_mode") != MODE
            or c.get("uses_source_mapping_rgb") is not True
            or c.get("uses_render_descriptors") is not False
        ):
            raise ValueError("completion provenance differs")
        if (
            c.get("uses_test_queries") is not False
            or c["query_names"] != raw["mapping_names"]
        ):
            raise ValueError("completion mapping registry differs")
        if (
            c["contract"]["final_xyz_source"] != "fixed_camera_robust_ray_triangulation"
            or c["contract"]["gaussian_depth_role"] != "proposal_neighborhood_only"
        ):
            raise ValueError("completion geometry source differs")
        assert_equal(base["v6_mapping_query_bins"], c["query_bins"], "camera bins")
        off, qs, rows = validate_csr(c["projective_anchor_observations"], m, records)
        used = set(
            zip(
                association["tracks"]["query_index"].tolist(),
                association["tracks"]["keypoint_index"].tolist(),
            )
        )
        if any(pair in used for pair in zip(qs.tolist(), rows.tolist())):
            raise ValueError("completion reused an occupied raw observation")
        x, f, cov, iq, gq = (
            c[key]
            for key in (
                "anchor_xyz",
                "anchor_features",
                "anchor_position_covariance",
                "identity_reliability",
                "geometry_reliability",
            )
        )
        if (
            x.shape != (m, 3)
            or f.shape != (m, base["anchor_features"].shape[1])
            or cov.shape != (m, 3, 3)
            or iq.shape != (m,)
            or gq.shape != (m,)
        ):
            raise ValueError("completion tensors misaligned")
        if not all(bool(torch.isfinite(t).all()) for t in (x, f, cov, iq, gq)):
            raise ValueError("nonfinite completion evidence")
        if any(bool(((q < 0) | (q > 1)).any()) for q in (iq, gq)):
            raise ValueError("completion quality out of range")
        if not torch.allclose(cov, cov.transpose(-1, -2), atol=1e-6, rtol=1e-5) or bool(
            (torch.linalg.eigvalsh(cov) < -1e-6).any()
        ):
            raise ValueError("completion covariance invalid")
        result.update(
            anchor_ids=torch.arange(n + m),
            anchor_xyz=torch.cat((base["anchor_xyz"], x)),
            anchor_features=torch.cat((base["anchor_features"], f)),
            anchor_position_covariance=torch.cat(
                (base["anchor_position_covariance"], cov)
            ),
            anchor_matchability=torch.cat(
                (base["anchor_matchability"], (iq * gq).clamp(0, 1))
            ),
        )
        old = base["projective_anchor_observations"]
        result["projective_anchor_observations"] = {
            "observation_offsets": torch.cat(
                (
                    old["observation_offsets"],
                    off[1:] + int(old["observation_offsets"][-1]),
                )
            ),
            "query_indices": torch.cat((old["query_indices"], qs)),
            "keypoint_indices": torch.cat((old["keypoint_indices"], rows)),
            "keypoint_index_semantics": "original_unfiltered_observation_cache_row",
        }
    result["schema"] = "gs_assisted_real_selector_state_v1"
    result["provenance"] = dict(
        input_mode=MODE,
        uses_source_mapping_rgb=True,
        uses_render_descriptors=False,
        uses_test_queries=False,
        protocol_sha256=protocol_sha,
        gaussian_depth_used_for_final_xyz=False,
        raw_base_count=n,
        completion_count=len(result["anchor_xyz"]) - n,
        full_is_append_only=True,
        compact_is_append_only=False,
    )
    assert_base_prefix(base, result)
    validate_csr(
        result["projective_anchor_observations"], len(result["anchor_xyz"]), records
    )
    if candidate is None:
        for key in SCIENCE_FIELDS:
            assert_equal(base[key], result[key], "fallback." + key)
    return result


def construct(
    raw, auxiliary, raw_association, raw_state, raw_full, protocol_sha, device
):
    validate_raw(raw)
    require_association(raw_association, MatrixObservationProvider(raw))
    if raw_state["provenance"].get("input_mode") != "raw_only":
        raise ValueError("base selector state is not raw-only")
    for key in ("anchor_xyz", "anchor_features"):
        assert_equal(raw_state[key], raw_full[key], "raw Full." + key)
    assisted = attach_auxiliary(raw, auxiliary, protocol_sha)
    provider = AssistedObservationProvider(assisted)
    used = set(
        zip(
            raw_association["tracks"]["query_index"].tolist(),
            raw_association["tracks"]["keypoint_index"].tolist(),
        )
    )
    budget = {"raw_rows": 0, "eligible_unused_rows": 0, "capped_proposal_rows": 0}
    for qi in range(len(provider)):
        view = provider.build_view(qi)
        depth, alpha = view.keypoint_depth, view.keypoint_alpha
        valid = (
            torch.isfinite(depth)
            & (depth > 0)
            & torch.isfinite(alpha)
            & (alpha >= 0.05)
        )
        count = sum(
            (qi, row) not in used
            for row in torch.nonzero(valid, as_tuple=False).reshape(-1).tolist()
        )
        budget["raw_rows"] += len(view.keypoints)
        budget["eligible_unused_rows"] += count
        budget["capped_proposal_rows"] += min(
            count, COMPLETION_PARAMETERS["maximum_rows_per_view"]
        )
    candidate = None
    reason = None
    try:
        candidate = build_projective_completion(
            provider, raw_association, device=device, **COMPLETION_PARAMETERS
        )
    except ValueError as error:
        if str(error) not in EMPTY_COMPLETION:
            raise
        reason = str(error)
    state = append_candidate(raw_state, candidate, raw, raw_association, protocol_sha)
    return state, {
        "completion": candidate,
        "completion_reason": reason,
        "base_frozen": True,
        "v2_deletes_raw_rows": False,
        "actual_budget": budget,
        "unused_definition": "all rows not in original raw association tracks",
    }
