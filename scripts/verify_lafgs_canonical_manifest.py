#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

import torch


def _sha256(path, chunk_bytes=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            value = handle.read(chunk_bytes)
            if not value:
                break
            digest.update(value)
    return digest.hexdigest()


def _sampled_sha256(path, chunk_bytes=4 * 1024 * 1024):
    size = path.stat().st_size
    offsets = (0, max((size - chunk_bytes) // 2, 0), max(size - chunk_bytes, 0))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            digest.update(str(offset).encode("ascii"))
            digest.update(handle.read(chunk_bytes))
    digest.update(str(size).encode("ascii"))
    return digest.hexdigest()


def _verify_artifact(name, artifact):
    path = Path(artifact["path"]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name}: missing artifact {path}")
    if "size_bytes" in artifact and path.stat().st_size != int(
        artifact["size_bytes"]
    ):
        raise ValueError(f"{name}: artifact size changed")
    if "sha256" in artifact:
        actual = _sha256(path)
        if actual != artifact["sha256"]:
            raise ValueError(f"{name}: SHA-256 mismatch")
    sampled_key = "sampled_sha256_3x4MiB_plus_size"
    if sampled_key in artifact:
        actual = _sampled_sha256(path)
        if actual != artifact[sampled_key]:
            raise ValueError(f"{name}: sampled SHA-256 mismatch")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Verify immutable LaFGS canonical states and protocol"
    )
    parser.add_argument(
        "--manifest",
        default=(
            "configs/locaware/"
            "lafgs_v2_oldhospital_canonical_48k_a4.json"
        ),
    )
    args = parser.parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    expected_count = int(manifest["training_protocol"]["landmark_count"])
    verified = []
    for prior_name in ("rgb_only_2dgs", "dirty_rgb_3dgs"):
        prior = manifest[prior_name]
        for artifact_name in (
            "gaussian_ply",
            "lafgs_state",
            "visibility",
            "query_cache",
        ):
            path = _verify_artifact(
                f"{prior_name}.{artifact_name}", prior[artifact_name]
            )
            verified.append(str(path))
        state = torch.load(
            prior["lafgs_state"]["path"],
            map_location="cpu",
            weights_only=False,
        )
        count = int(torch.as_tensor(state["landmark_indices"]).numel())
        if count != expected_count:
            raise ValueError(
                f"{prior_name}: expected {expected_count} landmarks, got {count}"
            )
        summary_path = _verify_artifact(
            f"{prior_name}.evaluation",
            {
                "path": prior["evaluation"]["summary_path"],
                "sha256": prior["evaluation"]["summary_sha256"],
            },
        )
        summary = json.loads(summary_path.read_text())
        if int(summary["evaluation_camera_count"]) != int(
            manifest["evaluation_protocol"]["images"]
        ):
            raise ValueError(f"{prior_name}: evaluation image count changed")
        verified.append(str(summary_path))
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "verified_artifact_count": len(verified),
                "landmark_count": expected_count,
                "mapping_images": manifest["training_protocol"][
                    "mapping_images"
                ],
                "evaluation_images": manifest["evaluation_protocol"]["images"],
                "status": "verified",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
