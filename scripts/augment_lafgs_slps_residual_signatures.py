#!/usr/bin/env python3
"""Append leave-query-out residual signatures to an exact SLPS corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from localization_training.slps_residual_signatures import (
    RESIDUAL_SIGNATURE_FEATURE_NAMES,
    normalized_bias_target,
    residual_signature_features,
    residual_statistics_contribution,
    signed_reprojection_residual,
    subtract_residual_statistics,
)
from localization_training.slps_selector import (
    SLPS_BIAS_AWARE_FEATURE_NAMES,
    SLPS_FEATURE_NAMES,
)
from scripts.train_lafgs_pose_sufficient_selector import _records_by_name


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--residual-signatures", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    corpus_path = Path(args.corpus).resolve()
    signature_path = Path(args.residual_signatures).resolve()
    output = Path(args.output).resolve()
    corpus = torch.load(corpus_path, map_location="cpu", weights_only=False)
    signature = torch.load(
        signature_path, map_location="cpu", weights_only=False
    )
    if corpus.get("schema") != "lafgs_slps_set_outcomes":
        raise ValueError("unsupported SLPS corpus")
    if signature.get("schema") != "lafgs_slps_residual_signatures":
        raise ValueError("unsupported residual signature state")
    if list(corpus.get("feature_names", ())) != list(SLPS_FEATURE_NAMES):
        raise ValueError("residual augmentation requires the legacy SLPS features")
    if (
        int(corpus["anchor_count"]) != int(signature["anchor_count"])
        or corpus["anchor_ids_sha256"] != signature["anchor_ids_sha256"]
        or dict(corpus["candidate_graph_contract"])
        != dict(signature["candidate_graph_contract"])
    ):
        raise ValueError("residual signatures and SLPS corpus do not align")

    map_path = Path(signature["source"]["map"])
    cache_path = Path(signature["source"]["query_cache"])
    topk_path = Path(signature["source"]["topk_outcomes"])
    state = torch.load(map_path, map_location="cpu", weights_only=False)
    cache_payload = torch.load(
        cache_path, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    topk = torch.load(topk_path, map_location="cpu", weights_only=False)
    topk_by_name = _records_by_name(topk)
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    full_statistics = {
        name: torch.as_tensor(value).float()
        for name, value in signature["statistics"].items()
    }
    config = dict(signature["config"])
    signature_rows = []
    target_weight_sum = 0.0
    support_removed_sum = 0.0
    for query_index, query in enumerate(corpus["queries"]):
        name = query["query_name"]
        record = topk_by_name[name]
        cached = cache[name]
        rows = torch.as_tensor(query["query_rows"]).long()
        if not torch.equal(rows, torch.as_tensor(record["query_rows"]).long()):
            raise ValueError(f"residual row contract differs for {name}")
        top1 = torch.as_tensor(record["topk_anchor_indices"]).long()[:, 0]
        selector_keypoints = torch.as_tensor(
            cached["native_keypoints"]
        ).float()[rows]
        observed = selector_keypoints + float(
            cached.get("pixel_center_offset", 0.5)
        )
        residual, valid = signed_reprojection_residual(
            xyz[top1],
            observed,
            cached["native_K"],
            cached["pose_w2c"],
        )
        contribution = residual_statistics_contribution(
            anchor_indices=top1,
            keypoints=selector_keypoints,
            image_hw=cached["native_input_hw"],
            signed_residual=residual,
            valid=valid,
            anchor_count=int(signature["anchor_count"]),
            grid_size=int(config["grid_size"]),
            clip_px=float(config["clip_px"]),
            strict_px=float(config["strict_px"]),
        )
        history = subtract_residual_statistics(
            full_statistics, contribution
        )
        residual_features = residual_signature_features(
            history,
            anchor_indices=top1,
            keypoints=selector_keypoints,
            image_hw=cached["native_input_hw"],
            grid_size=int(config["grid_size"]),
            clip_px=float(config["clip_px"]),
            anchor_prior=float(config["anchor_prior"]),
            cell_prior=float(config["cell_prior"]),
            rate_prior=float(config["rate_prior"]),
        )
        if residual_features.shape != (
            len(query["features"]),
            len(RESIDUAL_SIGNATURE_FEATURE_NAMES),
        ):
            raise AssertionError("residual signature feature shape differs")
        query["features"] = torch.cat(
            (query["features"].float(), residual_features), dim=1
        ).to(torch.float16)
        target, target_weight = normalized_bias_target(
            residual, valid, clip_px=float(config["clip_px"])
        )
        query["signed_residual_target"] = target.to(torch.float16)
        query["signed_residual_weight"] = target_weight.to(torch.float16)
        signature_rows.append(residual_features)
        target_weight_sum += float(target_weight.sum())
        support_removed_sum += float(contribution["soft_weight"].sum())
        if (query_index + 1) % 50 == 0:
            print(
                json.dumps(
                    {
                        "completed_queries": query_index + 1,
                        "total_queries": len(corpus["queries"]),
                    }
                ),
                flush=True,
            )

    feature_rows = torch.cat(
        [query["features"].float() for query in corpus["queries"]], dim=0
    )
    corpus.update(
        {
            "version": max(int(corpus.get("version", 1)), 3),
            "feature_names": list(SLPS_BIAS_AWARE_FEATURE_NAMES),
            "feature_mean": feature_rows.mean(dim=0),
            "feature_scale": feature_rows.std(
                dim=0, unbiased=False
            ).clamp_min(1e-4),
            "residual_signature_state": signature,
            "residual_signature_augmentation": {
                "source_corpus": str(corpus_path),
                "source_corpus_sha256": _sha256_file(corpus_path),
                "residual_signatures": str(signature_path),
                "residual_signatures_sha256": _sha256_file(signature_path),
                "leave_query_out": True,
                "target_weight_sum": target_weight_sum,
                "removed_soft_support_sum": support_removed_sum,
            },
        }
    )
    summary = dict(corpus.get("summary", {}))
    stacked = torch.cat(signature_rows, dim=0)
    summary.update(
        {
            "residual_signature_feature_count": len(
                RESIDUAL_SIGNATURE_FEATURE_NAMES
            ),
            "residual_signature_abs_mean": float(stacked.abs().mean()),
            "residual_signature_nonzero_rate": float(
                (stacked.abs().sum(dim=1) > 0).float().mean()
            ),
            "residual_signature_leave_query_out": True,
        }
    )
    corpus["summary"] = summary
    _atomic_torch(output, corpus)
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
