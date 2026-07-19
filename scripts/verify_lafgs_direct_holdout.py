#!/usr/bin/env python3

"""Verify that map, detector, and direct validation use one camera split."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from localization_training.episode_sampler import split_support_query_cameras


def canonical_name(value):
    return str(value).replace("\\", "/")


def names_sha256(names):
    names = sorted(canonical_name(name) for name in names)
    return hashlib.sha256(("\n".join(names) + "\n").encode("utf-8")).hexdigest()


def read_training_names(source_path):
    dataset_train = Path(source_path) / "dataset_train.txt"
    if not dataset_train.is_file():
        raise FileNotFoundError(f"Missing Cambridge training list: {dataset_train}")
    names = []
    for line in dataset_train.read_text().splitlines():
        fields = line.strip().split()
        if fields and fields[0].startswith("seq"):
            names.append(canonical_name(fields[0]))
    if not names:
        raise ValueError(f"No camera names parsed from {dataset_train}")
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate camera names in {dataset_train}")
    return sorted(names)


def expected_direct_holdout(names, *, validation_ratio, split_mode, split_seed):
    if float(validation_ratio) <= 0.0:
        return list(names), []
    return split_support_query_cameras(
        list(names),
        query_ratio=float(validation_ratio),
        seed=int(split_seed) + 1,
        mode=str(split_mode),
    )


def verify(map_state_path, detector_summary_path, source_path):
    map_state = torch.load(map_state_path, map_location="cpu", weights_only=False)
    map_config = map_state.get("config", {})
    detector_summary = json.loads(Path(detector_summary_path).read_text())
    detector_config = detector_summary.get("config", {})

    validation_ratio = float(map_config.get("validation_ratio", 0.0))
    split_mode = str(map_config.get("split_mode", ""))
    split_seed = int(map_config.get("split_seed", -1))
    if validation_ratio <= 0.0:
        raise ValueError("Direct-holdout verification requires validation_ratio > 0")
    if detector_config.get("validation_ratio") != validation_ratio:
        raise ValueError("Detector validation ratio differs from map state")
    if str(detector_config.get("split_mode")) != split_mode:
        raise ValueError("Detector split mode differs from map state")
    if int(detector_config.get("split_seed", -1)) != split_seed:
        raise ValueError("Detector split seed differs from map state")
    if detector_config.get("camera_order") != "image_name_lexicographic":
        raise ValueError("Detector did not use canonical lexical camera order")

    all_names = read_training_names(source_path)
    expected_train, expected_validation = expected_direct_holdout(
        all_names,
        validation_ratio=validation_ratio,
        split_mode=split_mode,
        split_seed=split_seed,
    )
    expected = {
        "input_camera_names_sha256": names_sha256(all_names),
        "train_camera_names_sha256": names_sha256(expected_train),
        "validation_camera_names_sha256": names_sha256(expected_validation),
        "input_camera_count": len(all_names),
        "train_camera_count": len(expected_train),
        "validation_camera_count": len(expected_validation),
    }
    expected_detector = {
        "input_camera_names_sha256": expected["input_camera_names_sha256"],
        "query_camera_names_sha256": expected["train_camera_names_sha256"],
        "validation_camera_names_sha256": expected[
            "validation_camera_names_sha256"
        ],
    }
    mismatches = {
        key: {
            "expected": expected_value,
            "actual": detector_config.get(key),
        }
        for key, expected_value in expected_detector.items()
        if detector_config.get(key) != expected_value
    }
    for key in (
        "train_camera_names_sha256",
        "validation_camera_names_sha256",
        "input_camera_names_sha256",
    ):
        recorded = map_config.get(key)
        if recorded is not None and recorded != expected[key]:
            mismatches[f"map_{key}"] = {
                "expected": expected[key],
                "actual": recorded,
            }
    if int(map_config.get("train_camera_count", -1)) != expected[
        "train_camera_count"
    ]:
        mismatches["map_train_camera_count"] = {
            "expected": expected["train_camera_count"],
            "actual": map_config.get("train_camera_count"),
        }
    if int(map_config.get("validation_camera_count", -1)) != expected[
        "validation_camera_count"
    ]:
        mismatches["map_validation_camera_count"] = {
            "expected": expected["validation_camera_count"],
            "actual": map_config.get("validation_camera_count"),
        }
    if mismatches:
        raise ValueError(
            "Map/detector direct holdout mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )

    return {
        "version": 1,
        "verified": True,
        "camera_order": "image_name_lexicographic",
        "map_state": str(Path(map_state_path).resolve()),
        "detector_summary": str(Path(detector_summary_path).resolve()),
        "source_path": str(Path(source_path).resolve()),
        "map_hashes_recorded": all(
            key in map_config
            for key in (
                "train_camera_names_sha256",
                "validation_camera_names_sha256",
                "input_camera_names_sha256",
            )
        ),
        "split": {
            "validation_ratio": validation_ratio,
            "split_mode": split_mode,
            "split_seed": split_seed,
            **expected,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-state", required=True, type=Path)
    parser.add_argument("--detector-summary", required=True, type=Path)
    parser.add_argument("--source-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = verify(args.map_state, args.detector_summary, args.source_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
