#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

from localization_training.micro_anchors import (
    compute_track_coverage_gain,
    compute_track_functional_statistics,
    select_micro_anchor_set,
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gather_candidate_csr(offsets, values, selected):
    selected = torch.as_tensor(selected, dtype=torch.long)
    offsets = torch.as_tensor(offsets, dtype=torch.long)
    values = torch.as_tensor(values)
    chunks = [
        values[int(offsets[row]) : int(offsets[row + 1])]
        for row in selected.tolist()
    ]
    lengths = torch.as_tensor(
        [chunk.shape[0] for chunk in chunks], dtype=torch.long
    )
    output_offsets = torch.cat(
        (torch.zeros(1, dtype=torch.long), lengths.cumsum(dim=0))
    )
    output_values = (
        torch.cat(chunks)
        if chunks
        else values.new_empty((0,) + values.shape[1:])
    )
    return output_offsets, output_values


def _materialize_selection(
    candidate_state,
    selected_rows,
    *,
    frozen_count,
    config,
    provenance,
):
    base_count = int(candidate_state["base_anchor_count"])
    selected_rows = torch.as_tensor(selected_rows, dtype=torch.long)
    rows = torch.cat((torch.arange(frozen_count), selected_rows))
    output = {
        "version": 3,
        "schema": "lafgs_materialized_anchor_map",
        "anchor_ids": torch.arange(rows.numel()),
        "source_primitive_ids": candidate_state["source_primitive_ids"][rows],
        "track_cluster_ids": candidate_state["track_cluster_ids"][rows],
        "anchor_xyz": candidate_state["anchor_xyz"][rows],
        "anchor_features": candidate_state["anchor_features"][rows],
        "anchor_type": candidate_state["anchor_type"][rows],
        "base_anchor_count": base_count,
        "canonical_anchor_count": int(frozen_count),
        "requested_micro_anchor_budget": int(
            frozen_count - base_count + selected_rows.numel()
        ),
        "requested_extension_budget": int(config["requested_budget"]),
        "selected_extension_count": int(selected_rows.numel()),
        "micro_anchor_count": int(
            frozen_count - base_count + selected_rows.numel()
        ),
        "config": config,
        "provenance": provenance,
    }
    if "full_prior_quality" in candidate_state:
        candidate_start = int(frozen_count)
        candidate_indices = selected_rows - candidate_start
        output["full_prior_quality"] = {
            key: torch.as_tensor(value)[candidate_indices].clone()
            for key, value in candidate_state["full_prior_quality"].items()
        }
        prefix = "full_prior_source_group"
        offset_key = f"{prefix}_offsets"
        id_key = f"{prefix}_primitive_ids"
        if offset_key in candidate_state and id_key in candidate_state:
            offsets, primitive_ids = _gather_candidate_csr(
                candidate_state[offset_key],
                candidate_state[id_key],
                candidate_indices,
            )
            output[offset_key] = offsets
            output[id_key] = primitive_ids
            for suffix in ("responsibilities", "costs"):
                key = f"{prefix}_{suffix}"
                if key in candidate_state:
                    _, values = _gather_candidate_csr(
                        candidate_state[offset_key],
                        candidate_state[key],
                        candidate_indices,
                    )
                    output[key] = values
    return output


def _align_visibility_to_prefix(visibility, state, prefix_count):
    base_count = int(state["base_anchor_count"])
    if prefix_count == base_count:
        return visibility
    base_sources = torch.as_tensor(
        state["source_primitive_ids"][:base_count], dtype=torch.long
    )
    prefix_sources = torch.as_tensor(
        state["source_primitive_ids"][:prefix_count], dtype=torch.long
    )
    source_to_row = {
        int(source): row for row, source in enumerate(base_sources.tolist())
    }
    lookup = torch.as_tensor(
        [source_to_row[int(source)] for source in prefix_sources.tolist()],
        dtype=torch.long,
    )
    aligned = {}
    for name, value in visibility.items():
        mask = torch.as_tensor(value, dtype=torch.bool).reshape(-1)
        if mask.numel() == prefix_count:
            aligned[name] = mask
        elif mask.numel() == base_count:
            aligned[name] = mask[lookup]
        else:
            raise ValueError(
                f"visibility rows for {name} do not align with the map"
            )
    return aligned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-map", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--visibility-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--profiles",
        default="unique_gap,query_saturated,sequence_tail",
    )
    parser.add_argument("--budgets", default="1500,2000")
    parser.add_argument("--false-attractor-penalty", type=float, default=0.25)
    args = parser.parse_args()

    candidate_path = Path(args.candidate_map).resolve()
    payload_path = Path(args.track_payload).resolve()
    query_path = Path(args.query_cache).resolve()
    visibility_path = Path(args.visibility_cache).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = [
        value.strip() for value in args.profiles.split(",") if value.strip()
    ]
    budgets = sorted(
        {int(value) for value in args.budgets.split(",") if value.strip()}
    )

    state = torch.load(candidate_path, map_location="cpu", weights_only=False)
    if state.get("schema") != "lafgs_materialized_anchor_map":
        raise ValueError("candidate map must be a materialized anchor map")
    base_count = int(state["base_anchor_count"])
    frozen_count = int(state.get("canonical_anchor_count", base_count))
    candidate_rows = torch.arange(
        frozen_count, state["anchor_ids"].numel(), dtype=torch.long
    )
    candidate_tracks = state["track_cluster_ids"][candidate_rows].long()
    if bool((candidate_tracks < 0).any()):
        raise ValueError("candidate micro-anchor rows require track IDs")
    if torch.unique(candidate_tracks).numel() != candidate_tracks.numel():
        raise ValueError("marginal selection requires one row per track")

    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    print(f"Loading query cache: {query_path}", flush=True)
    query_payload = torch.load(
        query_path, map_location="cpu", weights_only=False
    )
    query_cache = query_payload.get("queries", query_payload)
    visibility_payload = torch.load(
        visibility_path, map_location="cpu", weights_only=False
    )
    visibility = visibility_payload.get("visibility", visibility_payload)
    visibility = _align_visibility_to_prefix(
        visibility, state, frozen_count
    )
    track_count = int(
        payload["track_geometry"]["triangulated_xyz"].shape[0]
    )
    candidate_mask = torch.zeros(track_count, dtype=torch.bool)
    candidate_mask[candidate_tracks] = True
    base_xyz = state["anchor_xyz"][:frozen_count].float()
    base_features = state["anchor_features"][:frozen_count].float()

    coverage = compute_track_coverage_gain(
        payload=payload,
        query_cache=query_cache,
        base_xyz=base_xyz,
        visibility_cache=visibility,
        candidate_track_mask=candidate_mask,
    )
    functional = compute_track_functional_statistics(
        payload=payload,
        query_cache=query_cache,
        base_xyz=base_xyz,
        base_features=base_features,
        track_indices=candidate_tracks,
        track_features=state["anchor_features"][candidate_rows],
        visibility_cache=visibility,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    tracks = payload["tracks"]
    observations_by_track = defaultdict(list)
    gap_mask = coverage["coverage_gap_observation_mask"]
    for observation, track in enumerate(tracks["track_index"].tolist()):
        if bool(gap_mask[observation]):
            observations_by_track[int(track)].append(observation)
    candidate_gaps = [
        observations_by_track[int(track)]
        for track in candidate_tracks.tolist()
    ]
    query_names = payload["query_names"]
    sequence_to_index = {}
    query_sequences = []
    for name in query_names:
        sequence = str(name).split("/", 1)[0]
        if sequence not in sequence_to_index:
            sequence_to_index[sequence] = len(sequence_to_index)
        query_sequences.append(sequence_to_index[sequence])
    false_rates = functional["false_attractor_opportunity_rate"][
        candidate_tracks
    ]

    provenance = {
        "candidate_map_path": str(candidate_path),
        "candidate_map_sha256": _sha256(candidate_path),
        "track_payload_path": str(payload_path),
        "track_payload_sha256": _sha256(payload_path),
        "query_cache_path": str(query_path),
        "query_cache_signature": query_payload.get("signature"),
        "visibility_cache_path": str(visibility_path),
        "visibility_cache_sha256": _sha256(visibility_path),
        "visibility_cache_signature": visibility_payload.get("signature"),
        "statistics_split": "all_895_mapping_train",
        "base_rows_frozen": True,
        "frozen_prefix_count": int(frozen_count),
        "candidate_descriptor_geometry_frozen": True,
    }
    summary = {
        "candidate_count": int(candidate_rows.numel()),
        "candidate_gap_observation_count": int(
            coverage["coverage_gain"][candidate_tracks].sum()
        ),
        "candidate_functional_gap_count": int(
            functional["functional_gap"][candidate_tracks].sum()
        ),
        "candidate_false_attractor_count": int(
            functional["false_attractor_incoming_count"][
                candidate_tracks
            ].sum()
        ),
        "candidate_opportunity_count": int(
            functional["candidate_opportunity_count"][
                candidate_tracks
            ].sum()
        ),
        "profiles": {},
    }
    for profile in profiles:
        summary["profiles"][profile] = {}
        for budget in budgets:
            budget_selection, selection_diagnostics = (
                select_micro_anchor_set(
                    candidate_gap_observations=candidate_gaps,
                    observation_query_indices=tracks["query_index"],
                    query_sequence_indices=torch.as_tensor(query_sequences),
                    budget=budget,
                    profile=profile,
                    false_attractor_rates=false_rates,
                    false_attractor_penalty=args.false_attractor_penalty,
                )
            )
            selected_rows = candidate_rows[budget_selection]
            config = {
                "method": "micro_anchor_v3_marginal_coverage",
                "profile": profile,
                "requested_budget": int(budget),
                "false_attractor_penalty": float(
                    args.false_attractor_penalty
                ),
                "base_rows_frozen": True,
                "candidate_descriptor_geometry_frozen": True,
            }
            output = _materialize_selection(
                state,
                selected_rows,
                frozen_count=frozen_count,
                config=config,
                provenance=provenance,
            )
            profile_dir = output_dir / profile
            profile_dir.mkdir(parents=True, exist_ok=True)
            output_path = profile_dir / f"micro_anchor_{budget:04d}.pt"
            torch.save(output, output_path)
            selected_tracks = candidate_tracks[budget_selection]
            selected_track_mask = torch.zeros(track_count, dtype=torch.bool)
            selected_track_mask[selected_tracks] = True
            selected_gap_mask = gap_mask & selected_track_mask[
                tracks["track_index"].long()
            ]
            covered_gap_by_sequence = defaultdict(int)
            for observation in torch.nonzero(
                selected_gap_mask, as_tuple=False
            ).reshape(-1).tolist():
                query = int(tracks["query_index"][observation])
                sequence = str(query_names[query]).split("/", 1)[0]
                covered_gap_by_sequence[sequence] += 1
            diagnostics = {
                **selection_diagnostics,
                "state": str(output_path),
                "requested_budget": int(budget),
                "selected_count": int(budget_selection.numel()),
                "coverage_gain_sum": int(
                    coverage["coverage_gain"][selected_tracks].sum()
                ),
                "functional_gap_sum": int(
                    functional["functional_gap"][selected_tracks].sum()
                ),
                "false_attractor_count": int(
                    functional["false_attractor_incoming_count"][
                        selected_tracks
                    ].sum()
                ),
                "candidate_opportunity_count": int(
                    functional["candidate_opportunity_count"][
                        selected_tracks
                    ].sum()
                ),
                "covered_gap_by_sequence": dict(
                    sorted(covered_gap_by_sequence.items())
                ),
                "selected_track_ids": selected_tracks.tolist(),
            }
            summary["profiles"][profile][str(budget)] = diagnostics
            print(json.dumps({profile: {str(budget): diagnostics}}), flush=True)
    (output_dir / "marginal_selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
