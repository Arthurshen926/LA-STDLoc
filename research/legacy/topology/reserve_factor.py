#!/usr/bin/env python3
"""Materialize causal Track/Gaussian coverage and pose-reserve factors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

FACTOR_ORDER = (
    "core",
    "track_coverage",
    "all_coverage",
    "track_pose",
    "track_only_final",
    "full",
)


def factor_universe_ids(provenance: dict, factor: str) -> torch.Tensor:
    """Return one nested causal factor from recorded adaptive selections."""
    if factor not in FACTOR_ORDER:
        raise ValueError(f"unknown reserve factor: {factor}")
    keys = ["track_core_universe_ids"]
    if factor in {"track_coverage", "all_coverage", "track_pose", "track_only_final", "full"}:
        keys.append("coverage_track_universe_ids")
    if factor in {"all_coverage", "track_pose", "full"}:
        keys.append("coverage_gaussian_universe_ids")
    if factor in {"track_pose", "track_only_final", "full"}:
        keys.append("pose_track_universe_ids")
    if factor == "full":
        keys.append("pose_gaussian_universe_ids")
    values = [torch.as_tensor(provenance[key]).long().reshape(-1) for key in keys]
    ordered = []
    seen = set()
    for value in torch.cat(values).tolist():
        if int(value) not in seen:
            seen.add(int(value))
            ordered.append(int(value))
    return torch.as_tensor(ordered, dtype=torch.long)


def remap_teacher_to_factor(source_teacher: dict, factor_state: dict) -> dict:
    """Filter and reindex a complete-positive teacher to an exact map subset."""
    source_count = int(source_teacher["anchor_count"])
    source_rows = torch.as_tensor(factor_state["factor_source_rows"]).long()
    if source_rows.numel() and (
        int(source_rows.min()) < 0 or int(source_rows.max()) >= source_count
    ):
        raise ValueError("factor source rows exceed teacher anchor range")
    old_to_new = torch.full((source_count,), -1, dtype=torch.long)
    old_to_new[source_rows] = torch.arange(source_rows.numel())

    def remap_csr(offsets: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        offsets = torch.as_tensor(offsets).long()
        indices = torch.as_tensor(indices).long()
        counts = offsets[1:] - offsets[:-1]
        rows = torch.repeat_interleave(torch.arange(counts.numel()), counts)
        mapped = old_to_new[indices]
        keep = mapped >= 0
        retained_counts = torch.bincount(rows[keep], minlength=counts.numel())
        retained_offsets = torch.cat(
            [torch.zeros(1, dtype=torch.long), retained_counts.cumsum(0)]
        )
        return retained_offsets, mapped[keep]

    records = []
    positive_rows = 0
    strong_pairs = 0
    ambiguous_pairs = 0
    for source_record in source_teacher["records"]:
        record = dict(source_record)
        positive_offsets, positive_indices = remap_csr(
            record["positive_offsets"], record["positive_indices"]
        )
        ambiguous_offsets, ambiguous_indices = remap_csr(
            record["ambiguous_offsets"], record["ambiguous_indices"]
        )
        record.update(
            {
                "positive_offsets": positive_offsets,
                "positive_indices": positive_indices,
                "ambiguous_offsets": ambiguous_offsets,
                "ambiguous_indices": ambiguous_indices,
            }
        )
        records.append(record)
        positive_rows += int(((positive_offsets[1:] - positive_offsets[:-1]) > 0).sum())
        strong_pairs += int(positive_indices.numel())
        ambiguous_pairs += int(ambiguous_indices.numel())
    teacher = dict(source_teacher)
    teacher.update(
        {
            "anchor_count": int(source_rows.numel()),
            "records": records,
            "diagnostics": {
                "query_count": len(records),
                "positive_rows": positive_rows,
                "strong_pair_count": strong_pairs,
                "ambiguous_pair_count": ambiguous_pairs,
                "source_exact_track_positive_count": source_teacher.get(
                    "diagnostics", {}
                ).get("exact_track_positive_count"),
            },
            "factor_teacher_remap": {
                "source_anchor_count": source_count,
                "target_anchor_count": int(source_rows.numel()),
                "exact_subset_reindex": True,
            },
        }
    )
    return teacher


def materialize_factor(
    *,
    source: dict,
    canonical: dict,
    payload: dict,
    universe_ids: torch.Tensor,
    source_path: Path,
    payload_path: Path,
    factor: str,
) -> dict:
    """Select a factor without resetting learned descriptor/geometry state."""
    metadata = source["track_centric_reconstruction"]
    track_count = int(torch.as_tensor(payload["track_geometry"]["triangulated"]).numel())
    universe_ids = torch.as_tensor(universe_ids).long().reshape(-1)
    tracks = universe_ids[universe_ids < track_count]
    bases = universe_ids[universe_ids >= track_count] - track_count
    source_count = int(torch.as_tensor(source["anchor_ids"]).numel())
    source_type = torch.as_tensor(source["anchor_type"]).long().reshape(-1)
    source_track_ids = torch.as_tensor(source["track_cluster_ids"]).long().reshape(-1)
    track_rows = torch.nonzero(source_type != 0, as_tuple=False).reshape(-1)
    base_rows = torch.nonzero(source_type == 0, as_tuple=False).reshape(-1)
    source_bases = torch.as_tensor(metadata["base_canonical_rows"]).long().reshape(-1)
    if source_bases.numel() != base_rows.numel():
        raise ValueError("source base rows do not align with base_canonical_rows")

    def select_rows(
        requested: torch.Tensor,
        identities: torch.Tensor,
        rows: torch.Tensor,
        semantic: str,
    ) -> torch.Tensor:
        lookup = {int(identity): int(row) for identity, row in zip(identities.tolist(), rows.tolist())}
        if len(lookup) != int(identities.numel()):
            raise ValueError(f"source {semantic} identities are not unique")
        missing = [int(value) for value in requested.tolist() if int(value) not in lookup]
        if missing:
            raise ValueError(f"factor contains {semantic} identities absent from source map: {missing[:3]}")
        return torch.as_tensor([lookup[int(value)] for value in requested.tolist()], dtype=torch.long)

    selected_rows = torch.cat(
        [
            select_rows(tracks, source_track_ids[track_rows], track_rows, "track"),
            select_rows(bases, source_bases, base_rows, "Gaussian"),
        ]
    )
    output = dict(source)
    for key, value in source.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == source_count:
            output[key] = value[selected_rows].clone()
    output["anchor_ids"] = torch.arange(selected_rows.numel(), dtype=torch.long)
    output["factor_source_rows"] = selected_rows
    output.update(
        {
            "base_anchor_count": int(bases.numel()),
            "canonical_anchor_count": int(selected_rows.numel()),
            "micro_anchor_count": int(tracks.numel()),
            "requested_micro_anchor_budget": int(tracks.numel()),
        }
    )
    rebuilt = dict(metadata)
    rebuilt.update(output["track_centric_reconstruction"])
    rebuilt.update(
        {
            "budget": int(selected_rows.numel()),
            "track_anchor_count": int(tracks.numel()),
            "base_reserve_count": int(bases.numel()),
            "track_indices": tracks,
            "base_canonical_rows": bases,
            "quality_tier": f"adaptive_reserve_factor_{factor}",
            "reserve_factor": factor,
            "factor_track_count": int(tracks.numel()),
            "factor_gaussian_count": int(bases.numel()),
            "uses_test_queries": False,
        }
    )
    output["track_centric_reconstruction"] = rebuilt
    output["provenance"] = {
        **source.get("provenance", {}),
        "reserve_factor": {
            "factor": factor,
            "source_anchor_count": source_count,
            "source_map": str(source_path.resolve()),
            "preserves_learned_anchor_state": True,
            "uses_test_queries": False,
        },
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-map", type=Path, required=True)
    parser.add_argument("--canonical-map", type=Path, required=True)
    parser.add_argument("--track-payload", type=Path, required=True)
    parser.add_argument("--selection-provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--factors", default=",".join(FACTOR_ORDER))
    args = parser.parse_args()

    source_path = args.source_map.resolve()
    canonical_path = args.canonical_map.resolve()
    payload_path = args.track_payload.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    canonical = torch.load(canonical_path, map_location="cpu", weights_only=False)
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    provenance = torch.load(
        args.selection_provenance, map_location="cpu", weights_only=False
    )
    factors = [value for value in args.factors.split(",") if value]
    report = {}
    for factor in factors:
        universe = factor_universe_ids(provenance, factor)
        state = materialize_factor(
            source=source,
            canonical=canonical,
            payload=payload,
            universe_ids=universe,
            source_path=source_path,
            payload_path=payload_path,
            factor=factor,
        )
        path = output / f"reserve_factor_{factor}.pt"
        torch.save(state, path)
        report[factor] = {
            "map": str(path),
            "anchor_count": int(torch.as_tensor(state["anchor_ids"]).numel()),
            "track_count": int(
                state["track_centric_reconstruction"]["factor_track_count"]
            ),
            "gaussian_count": int(
                state["track_centric_reconstruction"]["factor_gaussian_count"]
            ),
        }
    payload_report = {
        "schema": "lafgs_reserve_causal_factor_maps",
        "version": 1,
        "changes_default_mainline": False,
        "source_map": str(source_path),
        "factors": report,
    }
    (output / "reserve_factor_maps.json").write_text(
        json.dumps(payload_report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload_report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
