#!/usr/bin/env python3
"""Attribute the fixed-row rendered-vs-real descriptor gap on mapping evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from evidence.tracks import LeaveOneQueryOutProjectiveAnchorDescriptorBank
from map_learning.trainer import track_descriptor_payload_for_loo
from scripts.materialize_mapping_rgb_descriptors import TOPOLOGY_FIELDS


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _summary(value: torch.Tensor) -> dict:
    value = torch.as_tensor(value).float().reshape(-1)
    value = value[torch.isfinite(value)]
    if not value.numel():
        return {key: None for key in ("mean", "p10", "median", "p90")}
    return {
        "mean": float(value.mean()),
        "p10": float(torch.quantile(value, 0.1)),
        "median": float(value.median()),
        "p90": float(torch.quantile(value, 0.9)),
    }


def _binned(factor: torch.Tensor, outcome: torch.Tensor, cosine: torch.Tensor, bins: int = 10) -> list[dict]:
    factor = torch.as_tensor(factor).float()
    finite = torch.isfinite(factor) & torch.isfinite(outcome) & torch.isfinite(cosine)
    factor, outcome, cosine = factor[finite], outcome[finite], cosine[finite]
    edges = torch.unique(torch.quantile(factor, torch.linspace(0, 1, bins + 1)))
    result = []
    for index in range(max(0, edges.numel() - 1)):
        selected = (factor >= edges[index]) & (
            factor <= edges[index + 1] if index + 2 == edges.numel() else factor < edges[index + 1]
        )
        if not bool(selected.any()):
            continue
        result.append({
            "minimum": float(edges[index]), "maximum": float(edges[index + 1]),
            "count": int(selected.sum()),
            "paired_cosine_mean": float(cosine[selected].mean()),
            "rgb_correct_score_gain_mean": float(outcome[selected].mean()),
            "benefited_fraction": float((outcome[selected] > 0).float().mean()),
        })
    return result


def _augment_rgb_cache(rgb: dict, rendered: dict) -> dict:
    queries = {}
    for name, record in rgb["queries"].items():
        source = rendered["queries"][name]
        queries[name] = {**source, **record}
    # Projective LOO replay currently validates the Gaussian support fields via
    # GaussianRenderObservationProvider.  This in-memory audit adapter retains
    # those rendered alpha/depth fields while replacing only descriptors; it is
    # never persisted as a source-image-free artifact.
    return {
        **rendered,
        **rgb,
        "uses_source_mapping_rgb": False,
        "descriptor_source_override_audit": "source_mapping_rgb_fixed_rows",
        "queries": queries,
    }


@torch.inference_mode()
def run(args) -> dict:
    torch.set_num_threads(int(args.cpu_threads))
    state = torch.load(args.render_map, map_location="cpu", weights_only=False)
    rgb_state = torch.load(args.rgb_map, map_location="cpu", weights_only=False)
    rendered = torch.load(args.render_cache, map_location="cpu", weights_only=False)
    rgb = _augment_rgb_cache(
        torch.load(args.rgb_cache, map_location="cpu", weights_only=False), rendered
    )
    payload = torch.load(args.track_payload, map_location="cpu", weights_only=False)
    if rendered.get("uses_test_queries") is not False or rgb.get("uses_test_queries") is not False:
        raise ValueError("descriptor-gap attribution must use mapping-only caches")
    for field in TOPOLOGY_FIELDS:
        if field in state and not torch.equal(torch.as_tensor(state[field]), torch.as_tensor(rgb_state[field])):
            raise ValueError(f"render/RGB maps differ in frozen topology field {field}")
    if not (
        list(rendered["queries"])
        == list(rgb["queries"])
        == list(payload["query_names"])
    ):
        raise ValueError("mapping query schedule differs")
    names = list(payload["query_names"])
    observations = state["projective_anchor_observations"]
    offsets = torch.as_tensor(observations["observation_offsets"]).long()
    query_indices = torch.as_tensor(observations["query_indices"]).long()
    keypoint_indices = torch.as_tensor(observations["keypoint_indices"]).long()
    anchor_rows = torch.repeat_interleave(torch.arange(offsets.numel() - 1), offsets[1:] - offsets[:-1])
    edge_count = int(anchor_rows.numel())
    values = {
        name: torch.empty(edge_count, dtype=torch.float32)
        for name in (
            "paired_cosine", "rgb_self_consistency", "render_self_consistency",
            "alpha", "depth_discontinuity", "border_distance", "grid_phase_distance",
            "view_angle_cosine", "correct_score_gain",
        )
    }
    render_map_raw = torch.as_tensor(state["anchor_features"]).float()
    rgb_map_raw = torch.as_tensor(rgb_state["anchor_features"]).float()
    render_map = F.normalize(render_map_raw, dim=1)
    rgb_map = F.normalize(rgb_map_raw, dim=1)
    bins = torch.as_tensor(payload["query_bins"]).long()[query_indices]
    for query in torch.unique(query_indices, sorted=True).tolist():
        selected = torch.nonzero(query_indices == int(query), as_tuple=False).reshape(-1)
        keypoints = keypoint_indices[selected]
        rows = anchor_rows[selected]
        render_record = rendered["queries"][names[int(query)]]
        rgb_record = rgb["queries"][names[int(query)]]
        render_desc = F.normalize(torch.as_tensor(render_record["native_descriptors"])[keypoints].float(), dim=1)
        rgb_desc = F.normalize(torch.as_tensor(rgb_record["native_descriptors"])[keypoints].float(), dim=1)
        values["paired_cosine"][selected] = (render_desc * rgb_desc).sum(1)
        values["render_self_consistency"][selected] = (render_desc * render_map[rows]).sum(1)
        values["rgb_self_consistency"][selected] = (rgb_desc * rgb_map[rows]).sum(1)
        values["correct_score_gain"][selected] = values["rgb_self_consistency"][selected] - values["render_self_consistency"][selected]
        alpha = torch.as_tensor(render_record["native_alpha_at_keypoints"])[keypoints].float()
        values["alpha"][selected] = alpha
        xy = torch.as_tensor(render_record["native_keypoints"])[keypoints].float()
        height, width = torch.as_tensor(render_record["native_input_hw"]).long().tolist()
        border = torch.minimum(torch.minimum(xy[:, 0], width - 1 - xy[:, 0]), torch.minimum(xy[:, 1], height - 1 - xy[:, 1]))
        values["border_distance"][selected] = border / float(min(height, width))
        phase = torch.remainder(xy + 0.5, 8.0)
        values["grid_phase_distance"][selected] = torch.minimum(phase, 8.0 - phase).norm(dim=1) / (4.0 * 2**0.5)
        depth = torch.as_tensor(render_record["native_rendered_depth"]).float().squeeze()
        local_max = F.max_pool2d(depth[None, None], 3, 1, 1)[0, 0]
        local_min = -F.max_pool2d(-depth[None, None], 3, 1, 1)[0, 0]
        pixel = xy.round().long(); pixel[:, 0].clamp_(0, width - 1); pixel[:, 1].clamp_(0, height - 1)
        center = depth[pixel[:, 1], pixel[:, 0]].abs().clamp_min(1e-3)
        values["depth_discontinuity"][selected] = (local_max[pixel[:, 1], pixel[:, 0]] - local_min[pixel[:, 1], pixel[:, 0]]).abs() / center
        pose = torch.as_tensor(render_record["pose_w2c"]).float()
        center_world = -(pose[:3, :3].T @ pose[:3, 3])
        optical = pose[:3, :3].T[:, 2]
        direction = F.normalize(torch.as_tensor(state["anchor_xyz"])[rows].float() - center_world, dim=1)
        values["view_angle_cosine"][selected] = direction @ optical
    track = torch.as_tensor(state["track_cluster_ids"]).long()[anchor_rows] >= 0

    # Strict query-local LOO retrieval proxy.  It remains a descriptor audit,
    # not a pose metric; no official test query is opened.
    device = torch.device(args.device)
    loo_payload = track_descriptor_payload_for_loo(payload)
    g_replay = LeaveOneQueryOutProjectiveAnchorDescriptorBank(
        state=state, payload=loo_payload, query_cache=rendered,
        reference_features=render_map_raw, trim_fraction=0.2,
    )
    r_replay = LeaveOneQueryOutProjectiveAnchorDescriptorBank(
        state=rgb_state, payload=loo_payload, query_cache=rgb,
        reference_features=rgb_map_raw, trim_fraction=0.2,
    )
    torch.set_num_threads(1)
    g_base, r_base = render_map.to(device), rgb_map.to(device)
    g_bank, r_bank = g_base.clone(), r_base.clone()
    previous = torch.empty(0, dtype=torch.long, device=device)
    query_records = []
    selected_queries = list(range(0, len(names), int(args.query_stride)))
    for position, query in enumerate(selected_queries):
        if position % 10 == 0:
            print(
                json.dumps(
                    {
                        "loo_query_progress": position,
                        "loo_query_total": len(selected_queries),
                        "global_query_index": query,
                    }
                ),
                flush=True,
            )
        if previous.numel():
            g_bank[previous] = g_base[previous]
            r_bank[previous] = r_base[previous]
        g_rows, g_features = g_replay.query_update(query)
        r_rows, r_features = r_replay.query_update(query)
        if not torch.equal(g_rows, r_rows):
            raise ValueError("render/RGB LOO affected Anchor rows differ")
        previous = g_rows.to(device)
        if previous.numel():
            g_bank[previous] = F.normalize(g_features.float(), dim=1).to(device)
            r_bank[previous] = F.normalize(r_features.float(), dim=1).to(device)
        edges = torch.nonzero(query_indices == query, as_tuple=False).reshape(-1)
        if edges.numel() == 0:
            continue
        if edges.numel() > args.maximum_query_observations:
            take = torch.linspace(0, edges.numel() - 1, args.maximum_query_observations).round().long()
            edges = edges[take]
        keypoints = keypoint_indices[edges]
        correct = anchor_rows[edges].to(device)
        gd = F.normalize(torch.as_tensor(rendered["queries"][names[query]]["native_descriptors"])[keypoints].float(), dim=1).to(device)
        rd = F.normalize(torch.as_tensor(rgb["queries"][names[query]]["native_descriptors"])[keypoints].float(), dim=1).to(device)
        g_score = gd @ g_bank.T; r_score = rd @ r_bank.T
        g_correct = g_score[torch.arange(gd.shape[0], device=device), correct]
        r_correct = r_score[torch.arange(rd.shape[0], device=device), correct]
        g_rank = 1 + (g_score > g_correct[:, None]).sum(1)
        r_rank = 1 + (r_score > r_correct[:, None]).sum(1)
        factors = {key: float(value[edges].mean()) for key, value in values.items() if key not in ("correct_score_gain",)}
        query_records.append({
            "query_index": query, "image_name": names[query], "sampled_observation_count": int(edges.numel()),
            "render_recall_at_1": float((g_rank == 1).float().mean()),
            "rgb_recall_at_1": float((r_rank == 1).float().mean()),
            "rgb_repaired_proxy": bool((r_rank == 1).float().mean() > (g_rank == 1).float().mean()),
            "rgb_harmed_proxy": bool((r_rank == 1).float().mean() < (g_rank == 1).float().mean()),
            "factors": factors,
        })
    repaired = torch.tensor([row["rgb_repaired_proxy"] for row in query_records])
    harmed = torch.tensor([row["rgb_harmed_proxy"] for row in query_records])
    factor_prediction = {}
    def selected_mean(value: torch.Tensor, selected: torch.Tensor) -> float | None:
        chosen = value[selected]
        chosen = chosen[torch.isfinite(chosen)]
        return float(chosen.mean()) if chosen.numel() else None
    for key in ("paired_cosine", "rgb_self_consistency", "alpha", "depth_discontinuity", "border_distance", "grid_phase_distance", "view_angle_cosine"):
        value = torch.tensor([row["factors"][key] for row in query_records])
        factor_prediction[key] = {
            "repaired_mean": selected_mean(value, repaired),
            "harmed_mean": selected_mean(value, harmed),
            "neutral_mean": selected_mean(value, ~(repaired | harmed)),
        }
    records_path = args.output.with_suffix(".records.pt")
    records_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "lafgs_render_real_descriptor_gap_records", "version": 1,
        "anchor_rows": anchor_rows, "query_indices": query_indices,
        "keypoint_indices": keypoint_indices, "view_bins": bins,
        "is_track": track, "factors": values, "query_records": query_records,
    }, records_path)
    report = {
        "schema": "lafgs_render_real_descriptor_gap_attribution", "version": 1,
        "uses_test_queries": False, "audit_only": True, "trains_feature_or_matcher": False,
        "same_anchor_rows_identity_xyz_selection": True,
        "observation_count": edge_count, "anchor_count": int(offsets.numel() - 1),
        "track_observation_count": int(track.sum()), "completion_observation_count": int((~track).sum()),
        "factor_summary": {key: _summary(value) for key, value in values.items()},
        "factor_bins": {key: _binned(value, values["correct_score_gain"], values["paired_cosine"]) for key, value in values.items() if key != "correct_score_gain"},
        "categorical": {
            "track": {"count": int(track.sum()), "paired_cosine": _summary(values["paired_cosine"][track]), "gain": _summary(values["correct_score_gain"][track])},
            "completion": {"count": int((~track).sum()), "paired_cosine": _summary(values["paired_cosine"][~track]), "gain": _summary(values["correct_score_gain"][~track])},
            "view_bins": {str(int(view_bin)): {"count": int((bins == view_bin).sum()), "paired_cosine": _summary(values["paired_cosine"][bins == view_bin]), "gain": _summary(values["correct_score_gain"][bins == view_bin])} for view_bin in torch.unique(bins)},
        },
        "mapping_retrieval_proxy": {
            "strict_loo": True, "self_inclusion_warning": False,
            "maximum_observations_per_query": int(args.maximum_query_observations),
            "query_stride": int(args.query_stride),
            "sampled_query_count": len(query_records),
            "repaired_query_count": int(repaired.sum()), "harmed_query_count": int(harmed.sum()),
            "neutral_query_count": int((~(repaired | harmed)).sum()),
            "factor_prediction": factor_prediction,
        },
        "records": str(records_path.resolve()), "records_sha256": sha256_file(records_path),
        "inputs": {"render_map": str(args.render_map), "rgb_map": str(args.rgb_map), "render_cache": str(args.render_cache), "rgb_cache": str(args.rgb_cache), "track_payload": str(args.track_payload)},
        "input_sha256": {"render_map": sha256_file(args.render_map), "rgb_map": sha256_file(args.rgb_map), "render_cache": sha256_file(args.render_cache), "rgb_cache": sha256_file(args.rgb_cache), "track_payload": sha256_file(args.track_payload)},
    }
    _atomic_json(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-map", type=Path, required=True)
    parser.add_argument("--rgb-map", type=Path, required=True)
    parser.add_argument("--render-cache", type=Path, required=True)
    parser.add_argument("--rgb-cache", type=Path, required=True)
    parser.add_argument("--track-payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--maximum-query-observations", type=int, default=256)
    parser.add_argument("--query-stride", type=int, default=1)
    args = parser.parse_args()
    if int(args.query_stride) < 1:
        parser.error("query-stride must be positive")
    for field in ("render_map", "rgb_map", "render_cache", "rgb_cache", "track_payload", "output"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    report = run(args)
    print(json.dumps({"factor_summary": report["factor_summary"], "mapping_retrieval_proxy": report["mapping_retrieval_proxy"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
