#!/usr/bin/env python3
"""Retire harmful-only anchors while preserving successful P3P basis coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from localization_training.artifact_contract import sha256_file
from localization_training.basin_distillation import GOOD_SET, HARMFUL_SET, NEAR_MISS_SET


def select_basis_safe_retirements(
    teacher: dict,
    anchor_count: int,
    *,
    maximum_retirements: int,
    minimum_bases_per_query: int,
    harmful_weight: float,
    blame_weight: float,
    good_weight: float,
) -> tuple[torch.Tensor, dict]:
    good_count = torch.zeros(anchor_count)
    harmful_count = torch.zeros(anchor_count)
    blame = torch.zeros(anchor_count)
    query_good_sets: list[torch.Tensor] = []
    for record in teacher["records"]:
        anchors = torch.as_tensor(record["set_anchor_indices"]).long()
        types = torch.as_tensor(record["set_types"]).long()
        correct = torch.as_tensor(record["correct_basin"]).bool()
        good_mask = correct & ((types == GOOD_SET) | (types == NEAR_MISS_SET))
        good_sets = anchors[good_mask]
        query_good_sets.append(good_sets)
        if good_sets.numel():
            good_count.index_add_(
                0,
                good_sets.reshape(-1),
                torch.ones(good_sets.numel()),
            )
        harmful_sets = anchors[types == HARMFUL_SET]
        if harmful_sets.numel():
            harmful_count.index_add_(
                0,
                harmful_sets.reshape(-1),
                torch.ones(harmful_sets.numel()),
            )
        harmful_anchors = torch.as_tensor(
            record["blame_harmful_anchors"]
        ).long()
        blame_weights = torch.as_tensor(record["blame_weights"]).float()
        if harmful_anchors.numel():
            blame.index_add_(0, harmful_anchors, blame_weights)
    score = (
        float(harmful_weight) * harmful_count
        + float(blame_weight) * blame
        - float(good_weight) * good_count
    )
    # A retirement is justified only by observed harmful evidence and must
    # never remove an anchor participating in a known successful basis.
    eligible = (score > 0) & ((harmful_count > 0) | (blame > 0)) & (good_count == 0)
    order = torch.nonzero(eligible, as_tuple=False).reshape(-1)
    order = order[torch.argsort(score[order], descending=True, stable=True)]
    active = torch.ones(anchor_count, dtype=torch.bool)
    initial_basis = torch.as_tensor(
        [int(sets.shape[0]) for sets in query_good_sets], dtype=torch.long
    )
    required_basis = torch.minimum(
        initial_basis,
        torch.full_like(initial_basis, int(minimum_bases_per_query)),
    )
    retired = []
    for anchor in order.tolist():
        if len(retired) >= int(maximum_retirements):
            break
        active[anchor] = False
        safe = True
        for query_index, sets in enumerate(query_good_sets):
            if int(required_basis[query_index]) == 0:
                continue
            surviving = active[sets].all(dim=1).sum()
            if int(surviving) < int(required_basis[query_index]):
                safe = False
                break
        if safe:
            retired.append(anchor)
        else:
            active[anchor] = True
    report = {
        "eligible_harmful_only_count": int(eligible.sum()),
        "retired_count": len(retired),
        "good_participating_anchor_count": int((good_count > 0).sum()),
        "harmful_participating_anchor_count": int((harmful_count > 0).sum()),
        "blamed_anchor_count": int((blame > 0).sum()),
        "minimum_initial_good_bases": int(initial_basis.min()) if initial_basis.numel() else 0,
        "minimum_required_good_bases": int(required_basis.min()) if required_basis.numel() else 0,
        "retired_good_incidence": float(good_count[~active].sum()),
        "retired_harmful_incidence": float(harmful_count[~active].sum()),
        "retired_blame": float(blame[~active].sum()),
    }
    return torch.nonzero(active, as_tuple=False).reshape(-1), report


def materialize_subset(state: dict, selected: torch.Tensor, source_path: Path) -> dict:
    count = int(torch.as_tensor(state["anchor_xyz"]).shape[0])
    selected = torch.as_tensor(selected).long()
    output = dict(state)
    for key, value in state.items():
        if torch.is_tensor(value) and value.ndim and int(value.shape[0]) == count:
            output[key] = value[selected]
    reconstruction = dict(state.get("track_centric_reconstruction", {}))
    old_track_count = int(reconstruction.get("track_anchor_count", 0))
    track_rows = selected[selected < old_track_count]
    base_rows = selected[selected >= old_track_count] - old_track_count
    if "track_indices" in reconstruction:
        reconstruction["track_indices"] = torch.as_tensor(
            reconstruction["track_indices"]
        ).long()[track_rows]
    if "base_canonical_rows" in reconstruction:
        reconstruction["base_canonical_rows"] = torch.as_tensor(
            reconstruction["base_canonical_rows"]
        ).long()[base_rows]
    reconstruction.update(
        {
            "schema": "lafgs_basin_aware_compact_map",
            "parent_map": str(source_path),
            "track_anchor_count": int(track_rows.numel()),
            "base_reserve_count": int(base_rows.numel()),
            "budget": int(selected.numel()),
            "parent_anchor_rows": selected,
        }
    )
    output["track_centric_reconstruction"] = reconstruction
    output["base_anchor_count"] = int(base_rows.numel())
    output["canonical_anchor_count"] = int(selected.numel())
    output["micro_anchor_count"] = int(track_rows.numel())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--basin-teacher", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--maximum-retirements", type=int, default=1024)
    parser.add_argument("--minimum-bases-per-query", type=int, default=2)
    parser.add_argument("--harmful-weight", type=float, default=1.0)
    parser.add_argument("--blame-weight", type=float, default=1.0)
    parser.add_argument("--good-weight", type=float, default=2.0)
    args = parser.parse_args()
    map_path = Path(args.map).resolve()
    teacher_path = Path(args.basin_teacher).resolve()
    state = torch.load(map_path, map_location="cpu", weights_only=False)
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=False)
    anchor_count = int(torch.as_tensor(state["anchor_xyz"]).shape[0])
    if int(teacher["anchor_count"]) != anchor_count:
        raise ValueError("basin teacher does not align with map")
    selected, report = select_basis_safe_retirements(
        teacher,
        anchor_count,
        maximum_retirements=args.maximum_retirements,
        minimum_bases_per_query=args.minimum_bases_per_query,
        harmful_weight=args.harmful_weight,
        blame_weight=args.blame_weight,
        good_weight=args.good_weight,
    )
    output = materialize_subset(state, selected, map_path)
    output["basis_aware_selection"] = {
        **report,
        "config": vars(args),
        "source_map_sha256": sha256_file(map_path),
        "basin_teacher_sha256": sha256_file(teacher_path),
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    output_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": "lafgs_basin_aware_compact_map",
                "source_anchor_count": anchor_count,
                "output_anchor_count": int(selected.numel()),
                **output["basis_aware_selection"],
            },
            indent=2,
        )
        + "\n"
    )
    print(output_path)


if __name__ == "__main__":
    main()
