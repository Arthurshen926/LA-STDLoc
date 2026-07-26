#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema") != "lafgs_materialized_anchor_map":
        raise ValueError(f"Unsupported anchor map: {path}")
    return state


def _append_rows(base, extension, extension_rows, *, label, provenance):
    base_count = int(base["anchor_ids"].numel())
    rows = torch.as_tensor(extension_rows, dtype=torch.long)
    output = {
        "version": 2,
        "schema": "lafgs_materialized_anchor_map",
        "anchor_ids": torch.arange(base_count + rows.numel()),
        "source_primitive_ids": torch.cat(
            (base["source_primitive_ids"], extension["source_primitive_ids"][rows])
        ),
        "track_cluster_ids": torch.cat(
            (base["track_cluster_ids"], extension["track_cluster_ids"][rows])
        ),
        "anchor_xyz": torch.cat(
            (base["anchor_xyz"], extension["anchor_xyz"][rows])
        ),
        "anchor_features": torch.cat(
            (base["anchor_features"], extension["anchor_features"][rows])
        ),
        "anchor_type": torch.cat(
            (base["anchor_type"], extension["anchor_type"][rows])
        ),
        "base_anchor_count": int(base["base_anchor_count"]),
        "requested_micro_anchor_budget": int(
            base["micro_anchor_count"] + rows.numel()
        ),
        "micro_anchor_count": int(base["micro_anchor_count"] + rows.numel()),
        "config": {
            "method": "micro_anchor_v2_conservative_additive_audit",
            "label": label,
            "canonical_v1_rows_frozen": True,
            "extension_row_count": int(rows.numel()),
        },
        "provenance": provenance,
    }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-v1", required=True)
    parser.add_argument("--m2-full", required=True)
    parser.add_argument("--m3-full", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    v1_path = Path(args.canonical_v1).resolve()
    m2_path = Path(args.m2_full).resolve()
    m3_path = Path(args.m3_full).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    v1, m2, m3 = _load(v1_path), _load(m2_path), _load(m3_path)
    v1_base = int(v1["base_anchor_count"])
    m2_base = int(m2["base_anchor_count"])
    m3_base = int(m3["base_anchor_count"])
    v1_tracks = set(v1["track_cluster_ids"][v1_base:].tolist())

    m2_offsets = m2["track_cluster_member_offsets"]
    m2_new_rows = []
    m2_new_tracks = []
    for row in range(m2_base, m2["anchor_ids"].numel()):
        begin, end = int(m2_offsets[row]), int(m2_offsets[row + 1])
        members = set(m2["track_cluster_member_ids"][begin:end].tolist())
        new_members = members - v1_tracks
        if new_members:
            m2_new_rows.append(row)
            m2_new_tracks.extend(sorted(new_members))
    identity_rows = torch.nonzero(
        m3["anchor_type"] == 2, as_tuple=False
    ).reshape(-1).tolist()
    provenance = {
        "canonical_v1_path": str(v1_path),
        "canonical_v1_sha256": _sha256(v1_path),
        "m2_full_path": str(m2_path),
        "m2_full_sha256": _sha256(m2_path),
        "m3_full_path": str(m3_path),
        "m3_full_sha256": _sha256(m3_path),
    }
    specifications = {
        "v1_plus_m2_new_candidates": (m2, m2_new_rows),
        "v1_plus_identity_only": (m3, identity_rows),
        "v1_plus_m2_new_candidates_identity": (
            m3,
            [
                row
                for row in range(m3_base, m3["anchor_ids"].numel())
                if (
                    int(m3["anchor_type"][row]) == 2
                    or int(m3["track_cluster_ids"][row])
                    in set(m2_new_tracks)
                )
            ],
        ),
    }
    summary = {}
    for label, (extension, rows) in specifications.items():
        output = _append_rows(
            v1,
            extension,
            rows,
            label=label,
            provenance=provenance,
        )
        output_path = output_dir / f"{label}.pt"
        torch.save(output, output_path)
        summary[label] = {
            "path": str(output_path),
            "extension_row_count": len(rows),
            "total_anchor_count": int(output["anchor_ids"].numel()),
            "micro_anchor_count": int(output["micro_anchor_count"]),
        }
    summary["m2_new_track_count"] = len(set(m2_new_tracks))
    summary["m2_new_tracks"] = sorted(set(m2_new_tracks))
    (output_dir / "additive_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
