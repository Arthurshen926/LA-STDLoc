#!/usr/bin/env python3
"""Run official AnySplat inference without post optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anysplat-repo", type=Path, required=True)
    parser.add_argument("--windows-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="lhjiang/anysplat")
    parser.add_argument("--revision", default="d2e8c343672646041ad4ea518184968f94362f01")
    parser.add_argument("--maximum-windows", type=int, default=0)
    args = parser.parse_args()
    repo = args.anysplat_repo.resolve()
    sys.path.insert(0, str(repo))
    from src.model.model.anysplat import AnySplat
    from src.utils.image import process_image

    manifest = json.loads(args.windows_manifest.read_text())
    windows = manifest["windows"]
    if args.maximum_windows > 0:
        windows = windows[: args.maximum_windows]
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    load_started = time.perf_counter()
    model = AnySplat.from_pretrained(args.model_id, revision=args.revision)
    model = model.to(device).eval()
    model.requires_grad_(False)
    model_load_seconds = time.perf_counter() - load_started
    run_records = []
    total_inference_seconds = 0.0
    for window in windows:
        window_id = str(window["window_id"])
        output = args.output / f"{window_id}.pt"
        if output.is_file():
            payload = torch.load(output, map_location="cpu", weights_only=False)
            run_records.append(payload["run_record"])
            continue
        image_paths = [
            Path(manifest["dataset"]) / "images" / name
            for name in window["image_names"]
        ]
        images = torch.stack([process_image(path) for path in image_paths])
        images = images.unsqueeze(0).to(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with torch.inference_mode():
            gaussians, predicted = model.inference((images + 1.0) * 0.5)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        total_inference_seconds += elapsed
        run_record = {
            "window_id": window_id,
            "image_names": list(window["image_names"]),
            "view_count": len(image_paths),
            "primitive_count": int(gaussians.means.shape[1]),
            "inference_seconds": elapsed,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
        payload = {
            "schema": "lafgs_anysplat_raw_feedforward_window",
            "version": 1,
            "model_id": args.model_id,
            "model_revision": args.revision,
            "post_optimization_used": False,
            "run_record": run_record,
            "means": gaussians.means[0].float().cpu(),
            "covariances": gaussians.covariances[0].float().cpu(),
            "f_dc": gaussians.harmonics[0, :, :, 0].float().cpu(),
            "opacity_probability": gaussians.opacities[0].float().cpu(),
            "predicted_c2w": predicted["extrinsic"][0].float().cpu(),
            "predicted_intrinsics_normalized": predicted["intrinsic"][0].float().cpu(),
        }
        torch.save(payload, output)
        run_records.append(run_record)
        del images, gaussians, predicted, payload
        torch.cuda.empty_cache()
    summary = {
        "schema": "lafgs_anysplat_feedforward_summary",
        "version": 1,
        "model_id": args.model_id,
        "model_revision": args.revision,
        "post_optimization_used": False,
        "model_load_seconds": model_load_seconds,
        "inference_seconds": total_inference_seconds,
        "windows": run_records,
    }
    (args.output / "feedforward_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

