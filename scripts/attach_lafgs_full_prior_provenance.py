#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _gather(offsets, values, rows):
    offsets = torch.as_tensor(offsets, dtype=torch.long)
    values = torch.as_tensor(values)
    chunks = [
        values[int(offsets[row]) : int(offsets[row + 1])]
        for row in rows.tolist()
    ]
    lengths = torch.as_tensor(
        [chunk.shape[0] for chunk in chunks], dtype=torch.long
    )
    output_offsets = torch.cat(
        (torch.zeros(1, dtype=torch.long), lengths.cumsum(dim=0))
    )
    return output_offsets, torch.cat(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--candidate-state", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    state_path = Path(args.state).resolve()
    candidate_path = Path(args.candidate_state).resolve()
    output_path = Path(args.output).resolve()
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    candidates = torch.load(
        candidate_path, map_location="cpu", weights_only=False
    )
    frozen_count = int(state["canonical_anchor_count"])
    candidate_frozen_count = int(candidates["canonical_anchor_count"])
    if frozen_count != candidate_frozen_count:
        raise ValueError("canonical prefixes do not align")
    selected_tracks = torch.as_tensor(
        state["track_cluster_ids"][frozen_count:], dtype=torch.long
    )
    candidate_tracks = torch.as_tensor(
        candidates["track_cluster_ids"][candidate_frozen_count:],
        dtype=torch.long,
    )
    track_to_row = {
        int(track): row for row, track in enumerate(candidate_tracks.tolist())
    }
    try:
        rows = torch.as_tensor(
            [track_to_row[int(track)] for track in selected_tracks.tolist()],
            dtype=torch.long,
        )
    except KeyError as error:
        raise ValueError(
            f"selected track {int(error.args[0])} is absent from candidates"
        ) from error
    prefix = "full_prior_source_group"
    offsets, primitive_ids = _gather(
        candidates[f"{prefix}_offsets"],
        candidates[f"{prefix}_primitive_ids"],
        rows,
    )
    output = dict(state)
    output[f"{prefix}_offsets"] = offsets
    output[f"{prefix}_primitive_ids"] = primitive_ids
    for suffix in ("responsibilities", "costs"):
        key = f"{prefix}_{suffix}"
        _, values = _gather(candidates[f"{prefix}_offsets"], candidates[key], rows)
        output[key] = values
    provenance = dict(output.get("provenance", {}))
    provenance.update(
        {
            "full_prior_csr_candidate_state": str(candidate_path),
            "full_prior_csr_attached_to": str(state_path),
        }
    )
    output["provenance"] = provenance
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(
        {
            "output": str(output_path),
            "selected_anchor_count": int(rows.numel()),
            "source_group_entry_count": int(primitive_ids.numel()),
        }
    )


if __name__ == "__main__":
    main()
