#!/usr/bin/env python3
"""Transfer a fixed query-row selection onto a new exact top-K graph."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topk-outcomes", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    topk = torch.load(
        args.topk_outcomes, map_location="cpu", weights_only=False
    )
    selection = torch.load(
        args.selection, map_location="cpu", weights_only=False
    )
    if list(topk["query_names"]) != list(selection["query_names"]):
        raise ValueError("selection-transfer query registries differ")
    records = []
    for topk_record, selected_record in zip(
        topk["records"], selection["records"]
    ):
        rows = torch.as_tensor(topk_record["query_rows"]).long()
        if not torch.equal(
            rows, torch.as_tensor(selected_record["query_rows"]).long()
        ):
            raise ValueError("selection-transfer query rows differ")
        records.append(
            {
                **topk_record,
                "selected_row_mask": torch.as_tensor(
                    selected_record["selected_row_mask"]
                ).bool(),
            }
        )
    payload = {
        **topk,
        "version": max(int(topk.get("version", 1)), 4),
        "records": records,
        "method": "fixed_query_row_selection_transfer",
        "selection_transfer": {
            "source": str(Path(args.selection).resolve()),
            "semantic": (
                "safety gate only; candidate identities and scores come "
                "from the target top-K graph"
            ),
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)


if __name__ == "__main__":
    main()
