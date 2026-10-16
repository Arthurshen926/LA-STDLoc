import torch

from anygsloc.pipeline.observations import GEOMETRY, MODES, SPARSE, validate_payload


def assert_unique_track_cameras(csr):
    offsets = torch.as_tensor(csr["observation_offsets"]).tolist()
    queries = torch.as_tensor(csr["query_indices"])
    for a, b in zip(offsets[:-1], offsets[1:]):
        if len(torch.unique(queries[a:b])) != b - a:
            raise ValueError("track contains repeated physical camera")


def source_contract(data):
    mode = data.get("input_mode")
    if mode not in MODES or data.get("uses_test_queries") is not False:
        raise ValueError("invalid input mode or test-input provenance")
    if data.get("uses_source_mapping_rgb") is not (mode != "pure_gs"):
        raise ValueError("source RGB provenance disagrees with mode")
    return mode


def validate_csr(csr, count, records):
    offsets = torch.as_tensor(csr["observation_offsets"]).long()
    query = torch.as_tensor(csr["query_indices"]).long()
    local = torch.as_tensor(csr["keypoint_indices"]).long()
    if offsets.ndim != 1 or len(offsets) != count + 1 or int(offsets[0]) != 0:
        raise ValueError("invalid CSR offsets")
    if query.ndim != 1 or local.shape != query.shape or int(offsets[-1]) != len(query):
        raise ValueError("unaligned CSR columns")
    if bool((offsets[1:] <= offsets[:-1]).any()):
        raise ValueError("empty or decreasing CSR track")
    if (
        bool((query < 0).any())
        or bool((query >= len(records)).any())
        or bool((local < 0).any())
    ):
        raise ValueError("CSR index out of bounds")
    for qi in torch.unique(query).tolist():
        rows = local[query == qi]
        if int(rows.max()) >= len(records[qi]["native_keypoints"]):
            raise ValueError("CSR row out of bounds")
    assert_unique_track_cameras(csr)
    return offsets, query, local


def adapt_selector_state(evidence, original, deployed, *, lineage=None):
    """Restore CSR to the actual pre-V2 mixed cache; never use source_row as a global row."""
    mode = source_contract(original)
    validate_payload(original)
    names = list(original["mapping_names"])
    if set(names) & set(original.get("test_names", ())):
        raise ValueError("test input in mapping registry")
    filtered = evidence["filtered_observations"]
    if source_contract(filtered) != mode or list(filtered["queries"]) != names:
        raise ValueError("filtered mapping scope differs")
    if evidence.get("v2_enabled") is not original.get(
        "filter_enabled", mode != "raw_only"
    ):
        raise ValueError("V2 availability differs from input mode")
    if (
        mode == "raw_only"
        and evidence.get("completion_reason") != "disabled_without_Gaussian"
    ):
        raise ValueError("raw-only Gaussian completion is not disabled")
    source_rows = []
    filtered_records = []
    for name in names:
        src, dst = original["queries"][name], filtered["queries"][name]
        n = len(src["native_keypoints"])
        rows = (
            torch.arange(n)
            if mode == "raw_only" or not original.get("filter_enabled", True)
            else torch.nonzero(src["v2_valid"], as_tuple=False).reshape(-1)
        )
        for k in SPARSE + ("row_source", "source_row"):
            if not torch.equal(dst[k], src[k][rows]):
                raise ValueError("filtered/source row mismatch: " + name + " " + k)
        for k in GEOMETRY:
            if not torch.equal(torch.as_tensor(src[k]), torch.as_tensor(dst[k])):
                raise ValueError("filtered/source camera mismatch")
        pairs = torch.stack((src["row_source"].long(), src["source_row"].long()), dim=1)
        if len(torch.unique(pairs, dim=0)) != n:
            raise ValueError("duplicate per-source observation identity")
        source_rows.append(rows)
        filtered_records.append(dst)
    parts = evidence["parts"]
    if not parts or (mode == "raw_only" and len(parts) != 1):
        raise ValueError("invalid candidate parts")
    xyz, feats, cov, quality, query_cols, point_cols = [], [], [], [], [], []
    offsets = [0]
    bins = torch.as_tensor(parts[0]["query_bins"]).long()
    if bins.shape != (len(names),):
        raise ValueError("mapping bins misaligned")
    for part in parts:
        if source_contract(part) != mode or list(part["query_names"]) != names:
            raise ValueError("candidate input provenance or registry differs")
        if not torch.equal(torch.as_tensor(part["query_bins"]).long(), bins):
            raise ValueError("candidate bins differ")
        x, f = part["anchor_xyz"], part["anchor_features"]
        c = part["anchor_position_covariance"]
        iq, gq = part["identity_reliability"], part["geometry_reliability"]
        n = len(x)
        if x.shape != (n, 3) or f.ndim != 2 or len(f) != n or c.shape != (n, 3, 3):
            raise ValueError("candidate tensors misaligned")
        if iq.shape != (n,) or gq.shape != (n,):
            raise ValueError("quality tensors misaligned")
        if any(not bool(torch.isfinite(v).all()) for v in (x, f, c, iq, gq)):
            raise ValueError("nonfinite authentic candidate evidence")
        if bool(((iq < 0) | (iq > 1) | (gq < 0) | (gq > 1)).any()):
            raise ValueError("quality outside unit interval")
        off, qs, local = validate_csr(
            part["projective_anchor_observations"], n, filtered_records
        )
        remapped = torch.empty_like(local)
        for qi in torch.unique(qs).tolist():
            mask = qs == qi
            remapped[mask] = source_rows[qi][local[mask]]
        base = offsets[-1]
        offsets.extend((off[1:] + base).tolist())
        query_cols.append(qs)
        point_cols.append(remapped)
        xyz.append(x)
        feats.append(f)
        cov.append(c)
        quality.append((iq * gq).clamp(0, 1))
    state = {
        "schema": "mainline_input_selector_state_v1",
        "anchor_ids": torch.arange(sum(len(x) for x in xyz)),
        "anchor_xyz": torch.cat(xyz),
        "anchor_features": torch.cat(feats),
        "anchor_position_covariance": torch.cat(cov),
        "anchor_matchability": torch.cat(quality),
        "v6_mapping_query_names": names,
        "v6_mapping_query_bins": bins,
        "projective_anchor_observations": {
            "observation_offsets": torch.tensor(offsets, dtype=torch.long),
            "query_indices": torch.cat(query_cols),
            "keypoint_indices": torch.cat(point_cols),
            "keypoint_index_semantics": "original_unfiltered_observation_cache_row",
        },
        "provenance": {
            **dict(lineage or {}),
            "input_mode": mode,
            "uses_source_mapping_rgb": mode != "pure_gs",
            "uses_test_queries": False,
            "mapping_source": mode,
            "gaussian_depth_used_for_final_xyz": False,
            "adapter": "authentic_quality_covariance_and_source_CSR_no_numerical_change",
        },
    }
    for key in ("anchor_xyz", "anchor_features"):
        if not torch.equal(state[key], deployed[key]):
            raise ValueError("adapter differs from deployed Full: " + key)
    validate_csr(
        state["projective_anchor_observations"],
        len(state["anchor_ids"]),
        [original["queries"][n] for n in names],
    )
    return state
