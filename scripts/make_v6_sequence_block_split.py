#!/usr/bin/env python3
"""Create a SHA-bound whole-sequence train/validation split for V6 feedback."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

import torch

from common.hashing import sha256_file
from common.v6_contracts import (
    DESCRIPTOR_SPLIT_SCHEMA,
    FEEDBACK_SCHEMA,
    ordered_query_registry_sha256,
    require_schema,
)


def build_sequence_block_split(
    feedback: dict,
    *,
    source_feedback_sha256: str,
    modulus: int | None = None,
    training_remainders: list[int] | None = None,
    validation_sequences: list[str] | None = None,
) -> dict:
    require_schema(feedback, FEEDBACK_SCHEMA, label="feedback")
    explicit_validation = sorted(
        {str(value).lower() for value in (validation_sequences or [])}
    )
    use_modulus = modulus is not None or bool(training_remainders)
    if bool(explicit_validation) == bool(use_modulus):
        raise ValueError("choose exactly one sequence split rule")
    remainders = []
    if use_modulus:
        if modulus is None or int(modulus) < 2:
            raise ValueError("sequence split modulus must be at least two")
        remainders = sorted({int(value) for value in (training_remainders or [])})
        if not remainders or remainders[0] < 0 or remainders[-1] >= int(modulus):
            raise ValueError("sequence split remainders are invalid")
    names = [str(name) for name in feedback.get("query_names", ())]
    if not names or len(names) != len(feedback.get("records", ())):
        raise ValueError("feedback query registry is empty or misaligned")
    sequence_number = []
    sequence_name = []
    for name in names:
        group = Path(name).parts[0]
        match = re.fullmatch(r"seq(\d+)", group, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"query {name!r} has no canonical seqN group")
        sequence_name.append(group)
        sequence_number.append(int(match.group(1)))
    if explicit_validation:
        available = {value.lower() for value in sequence_name}
        missing = sorted(set(explicit_validation) - available)
        if missing:
            raise ValueError(f"validation sequences are absent: {missing}")
        training = [
            index
            for index, value in enumerate(sequence_name)
            if value.lower() not in explicit_validation
        ]
        rule = {
            "sequence_pattern": "seqN",
            "validation_sequences": explicit_validation,
        }
    else:
        training = [
            index
            for index, number in enumerate(sequence_number)
            if number % int(modulus) in remainders
        ]
        rule = {
            "sequence_pattern": "seqN",
            "modulus": int(modulus),
            "training_remainders": remainders,
        }
    training_set = set(training)
    validation = [
        index for index in range(len(names)) if index not in training_set
    ]
    if not training or not validation:
        raise ValueError("sequence split must contain train and validation queries")
    layer_names = ("L1", "L2", "L3", "L4")

    def summarize(indices: list[int]) -> dict:
        records = [feedback["records"][index] for index in indices]
        return {
            "query_count": len(indices),
            "sequences": sorted({sequence_name[index] for index in indices}),
            "failure_layer_counts": {
                layer: sum(
                    int(
                        layer
                        in record.get(
                            "failure_layers", (record.get("failure_layer"),)
                        )
                    )
                    for record in records
                )
                for layer in layer_names
            },
        }

    return {
        "schema": DESCRIPTOR_SPLIT_SCHEMA,
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "source_feedback_sha256": str(source_feedback_sha256),
        "query_names_sha256": ordered_query_registry_sha256(names),
        "rule": rule,
        "training_query_indices": training,
        "validation_query_indices": validation,
        "training_summary": summarize(training),
        "validation_summary": summarize(validation),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--expected-feedback-sha256", required=True)
    parser.add_argument("--modulus", type=int)
    parser.add_argument(
        "--training-remainder", type=int, action="append"
    )
    parser.add_argument("--validation-sequence", action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    feedback_path = args.feedback.resolve()
    actual_sha = sha256_file(feedback_path)
    if actual_sha != args.expected_feedback_sha256:
        raise ValueError("feedback SHA differs")
    payload = torch.load(feedback_path, map_location="cpu", weights_only=False)
    feedback = payload.get("feedback", payload)
    split = build_sequence_block_split(
        feedback,
        source_feedback_sha256=actual_sha,
        modulus=args.modulus,
        training_remainders=args.training_remainder,
        validation_sequences=args.validation_sequence,
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({**split, "training_query_indices": "omitted", "validation_query_indices": "omitted"}, indent=2))


if __name__ == "__main__":
    main()
