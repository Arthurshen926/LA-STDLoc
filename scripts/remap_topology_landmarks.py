#!/usr/bin/env python
import argparse
import json
import os
import pickle

import torch

from stdloc import remap_sampled_indices_from_source_index


def _load_sampled_idx(path):
    with open(path, "rb") as f:
        return torch.as_tensor(pickle.load(f), dtype=torch.long).reshape(-1)


def _write_sampled_idx(path, sampled_idx):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(sampled_idx.detach().cpu().tolist(), f)


def _scores_from_state(state, source):
    source = str(source)
    if source == "none":
        return None
    if source == "label_quality":
        repeat = torch.as_tensor(state.get("loc_repeatability_ema", 0.0), dtype=torch.float32)
        info = torch.as_tensor(state.get("loc_information_ema", 0.0), dtype=torch.float32)
        outlier = torch.as_tensor(state.get("loc_outlier_ema", 0.0), dtype=torch.float32)
        return repeat + info - outlier
    if source not in state:
        raise ValueError(f"Unknown score source: {source}")
    return torch.as_tensor(state[source], dtype=torch.float32).reshape(-1)


def remap_topology_landmarks(
    source_sampled_idx,
    topology_loc_state,
    fill_missing=False,
    fill_score_source="label_quality",
    remap_score_source="label_quality",
):
    source_idx = _load_sampled_idx(source_sampled_idx)
    state = torch.load(topology_loc_state, map_location="cpu")
    if "loc_source_index" not in state:
        raise ValueError(
            "topology_loc_state does not contain loc_source_index; "
            "rerun topology with source-index tracking enabled."
        )
    remap_scores = _scores_from_state(state, remap_score_source)
    fill_scores = _scores_from_state(state, fill_score_source) if fill_missing else None
    remapped, missing = remap_sampled_indices_from_source_index(
        source_idx,
        state["loc_source_index"],
        return_missing=True,
        fill_missing=fill_missing,
        fill_scores=fill_scores,
        remap_scores=remap_scores,
    )
    return {
        "sampled_idx": remapped,
        "missing": missing,
        "source_count": int(source_idx.numel()),
        "remapped_count": int(remapped.numel()),
        "missing_count": int(missing.numel()),
        "point_count": int(torch.as_tensor(state["loc_source_index"]).numel()),
    }


def main():
    parser = argparse.ArgumentParser(description="Remap sparse sampled_idx after topology prune/split.")
    parser.add_argument("--source_sampled_idx", required=True)
    parser.add_argument("--topology_loc_state", required=True)
    parser.add_argument("--output_sampled_idx", required=True)
    parser.add_argument("--summary_output", default=None)
    parser.add_argument("--fill_missing", action="store_true", default=False)
    parser.add_argument("--fill_score_source", default="label_quality")
    parser.add_argument("--remap_score_source", default="label_quality")
    args = parser.parse_args()

    result = remap_topology_landmarks(
        args.source_sampled_idx,
        args.topology_loc_state,
        fill_missing=args.fill_missing,
        fill_score_source=args.fill_score_source,
        remap_score_source=args.remap_score_source,
    )
    _write_sampled_idx(args.output_sampled_idx, result["sampled_idx"])
    summary = {
        "source_sampled_idx": args.source_sampled_idx,
        "topology_loc_state": args.topology_loc_state,
        "output_sampled_idx": args.output_sampled_idx,
        "source_count": result["source_count"],
        "remapped_count": result["remapped_count"],
        "missing_count": result["missing_count"],
        "point_count": result["point_count"],
        "fill_missing": bool(args.fill_missing),
        "fill_score_source": args.fill_score_source,
        "remap_score_source": args.remap_score_source,
        "missing_preview": result["missing"][:20].detach().cpu().tolist(),
    }
    text = json.dumps(summary, indent=2)
    if args.summary_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_output)), exist_ok=True)
        with open(args.summary_output, "w") as f:
            f.write(text)
            f.write("\n")
    print(text)


if __name__ == "__main__":
    main()
