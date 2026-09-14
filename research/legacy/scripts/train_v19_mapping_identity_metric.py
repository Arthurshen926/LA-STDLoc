#!/usr/bin/env python3
"""Pretrain a bounded shared metric from frozen mapping Track identities."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from map_learning.v9_metric_controller import (
    metric_artifact,
    train_v9_shared_metric,
    transform_map_anchor_features,
)


def _save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _balanced_sample(
    rows: torch.Tensor,
    row_families: torch.Tensor,
    maximum: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    families = torch.unique(row_families[rows], sorted=True)
    quota = max(int(maximum) // max(int(families.numel()), 1), 1)
    selected = []
    for family in families.tolist():
        local = rows[row_families[rows] == int(family)]
        order = torch.randperm(local.numel(), generator=generator)
        selected.append(local[order[:quota]])
    output = torch.cat(selected)
    if output.numel() > int(maximum):
        output = output[torch.randperm(output.numel(), generator=generator)[: int(maximum)]]
    return output.sort().values


def _gather_descriptors(
    rows: torch.Tensor,
    observation_queries: torch.Tensor,
    observation_keypoints: torch.Tensor,
    descriptors: list[torch.Tensor],
) -> torch.Tensor:
    queries = observation_queries[rows]
    keypoints = observation_keypoints[rows]
    output = torch.empty((rows.numel(), descriptors[0].shape[1]))
    for query in torch.unique(queries, sorted=True).tolist():
        local = torch.nonzero(queries == int(query), as_tuple=False).reshape(-1)
        output[local] = descriptors[query][keypoints[local]]
    return output


@torch.inference_mode()
def _topk(
    query: torch.Tensor,
    anchors: torch.Tensor,
    *,
    k: int,
    device: str,
    query_chunk: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.device(device)
    bank = F.normalize(torch.as_tensor(anchors).float(), dim=1).to(target)
    scores = []
    rows = []
    for start in range(0, query.shape[0], int(query_chunk)):
        local = F.normalize(query[start : start + int(query_chunk)].to(target), dim=1)
        value, index = torch.topk(local @ bank.T, int(k), dim=1)
        scores.append(value.cpu())
        rows.append(index.cpu())
    return torch.cat(scores), torch.cat(rows)


def _strongest_negative(
    candidates: torch.Tensor,
    candidate_scores: torch.Tensor,
    positive: torch.Tensor,
    equivalence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    different = equivalence[candidates] != equivalence[positive, None]
    if not bool(different.any(1).all()):
        raise RuntimeError("mapping identity retrieval did not expose a negative")
    column = candidate_scores.masked_fill(~different, -torch.inf).argmax(1)
    rows = torch.arange(candidates.shape[0])
    return candidates[rows, column], candidate_scores[rows, column]


def _metrics(
    candidates: torch.Tensor,
    positive: torch.Tensor,
    equivalence: torch.Tensor,
) -> dict:
    correct = equivalence[candidates[:, 0]] == equivalence[positive]
    return {
        "row_count": int(positive.numel()),
        "top1_correct_count": int(correct.sum()),
        "top1_accuracy": float(correct.float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-map", type=Path, required=True)
    parser.add_argument("--mapping-provenance", type=Path, required=True)
    parser.add_argument("--mapping-feature-cache", type=Path, required=True)
    parser.add_argument("--teacher-validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-training-rows", type=int, default=20000)
    parser.add_argument("--maximum-calibration-rows", type=int, default=4000)
    parser.add_argument("--maximum-validation-rows", type=int, default=4000)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--maximum-residual-norm", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1920260830)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    state = torch.load(args.anchor_map, map_location="cpu", weights_only=False)
    provenance = torch.load(
        args.mapping_provenance, map_location="cpu", weights_only=False
    )
    cache = torch.load(
        args.mapping_feature_cache, map_location="cpu", weights_only=False
    )
    validation = torch.load(
        args.teacher_validation, map_location="cpu", weights_only=False
    )
    names = list(state["v6_mapping_query_names"])
    records = cache.get("queries", cache)
    if not (
        validation.get("selection_uses_validation") is False
        and validation.get("uses_test_queries") is False
        and provenance.get("uses_test_queries") is False
        and cache.get("uses_test_queries") is False
        and cache.get("uses_source_mapping_rgb") is False
        and names == list(records)
    ):
        raise ValueError("V19 mapping identity metric input contract differs")
    observations = state["projective_anchor_observations"]
    offsets = torch.as_tensor(observations["observation_offsets"]).long()
    observation_queries = torch.as_tensor(observations["query_indices"]).long()
    observation_keypoints = torch.as_tensor(observations["keypoint_indices"]).long()
    observation_anchor = torch.repeat_interleave(
        torch.arange(offsets.numel() - 1), offsets[1:] - offsets[:-1]
    )
    provenance_rows = torch.as_tensor(provenance["observation_rows"]).long()
    valid = torch.zeros(observation_queries.numel(), dtype=torch.bool)
    valid[provenance_rows] = torch.as_tensor(provenance["observation_valid"]).bool()
    mapping_families = torch.as_tensor(provenance["mapping_view_family_ids"]).long()
    row_families = mapping_families[observation_queries]
    roles = {int(key): value for key, value in validation["family_roles"].items()}
    row_roles = [roles[int(family)] for family in row_families.tolist()]
    role_masks = {
        role: valid & torch.tensor([value == role for value in row_roles])
        for role in ("track_bank", "threshold_calibration", "independent_validation")
    }
    train_rows = _balanced_sample(
        torch.nonzero(role_masks["track_bank"], as_tuple=False).reshape(-1),
        row_families,
        args.maximum_training_rows,
        args.seed,
    )
    calibration_rows = _balanced_sample(
        torch.nonzero(role_masks["threshold_calibration"], as_tuple=False).reshape(-1),
        row_families,
        args.maximum_calibration_rows,
        args.seed + 1,
    )
    heldout_rows = _balanced_sample(
        torch.nonzero(role_masks["independent_validation"], as_tuple=False).reshape(-1),
        row_families,
        args.maximum_validation_rows,
        args.seed + 2,
    )
    mapping_descriptors = [
        torch.as_tensor(records[name]["native_descriptors"]).float() for name in names
    ]
    all_rows = torch.cat((train_rows, calibration_rows, heldout_rows))
    all_descriptors = _gather_descriptors(
        all_rows, observation_queries, observation_keypoints, mapping_descriptors
    )
    train_count = train_rows.numel()
    calibration_count = calibration_rows.numel()
    train_descriptors = all_descriptors[:train_count]
    calibration_descriptors = all_descriptors[
        train_count : train_count + calibration_count
    ]
    heldout_descriptors = all_descriptors[train_count + calibration_count :]
    anchors = torch.as_tensor(state["anchor_features"]).float()
    equivalence = torch.as_tensor(
        state.get("fine_identity_ids", torch.arange(anchors.shape[0]))
    ).long()
    train_positive = observation_anchor[train_rows]
    calibration_positive = observation_anchor[calibration_rows]
    heldout_positive = observation_anchor[heldout_rows]
    train_scores, train_candidates = _topk(
        train_descriptors, anchors, k=8, device=args.device
    )
    train_negative, train_negative_score = _strongest_negative(
        train_candidates, train_scores, train_positive, equivalence
    )
    positive_score = (F.normalize(train_descriptors, dim=1) * F.normalize(anchors[train_positive], dim=1)).sum(1)
    initially_correct = equivalence[train_candidates[:, 0]] == equivalence[train_positive]
    family_counts = Counter(row_families[train_rows].tolist())
    weights = torch.tensor(
        [1.0 / family_counts[int(family)] for family in row_families[train_rows].tolist()]
    )
    metric, training = train_v9_shared_metric(
        anchor_features=anchors,
        query_descriptors=train_descriptors,
        positive_anchor_rows=train_positive,
        negative_anchor_rows=train_negative,
        sample_weights=weights,
        clean_query_descriptors=train_descriptors[initially_correct],
        clean_positive_anchor_rows=train_positive[initially_correct],
        clean_negative_anchor_rows=train_negative[initially_correct],
        clean_initial_margin=(positive_score - train_negative_score)[initially_correct],
        clean_sample_weights=weights[initially_correct],
        rank=args.rank,
        maximum_residual_norm=args.maximum_residual_norm,
        steps=args.steps,
        clean_protection_weight=2.0,
        seed=args.seed,
        device=args.device,
    )
    transformed_anchors = transform_map_anchor_features(
        metric, anchors, device=args.device
    )
    metric = metric.to(args.device).eval()
    evaluations = {}
    for name, descriptors, positive in (
        ("calibration", calibration_descriptors, calibration_positive),
        ("independent_validation", heldout_descriptors, heldout_positive),
    ):
        raw_scores, raw_candidates = _topk(
            descriptors, anchors, k=1, device=args.device
        )
        del raw_scores
        transformed_query = []
        with torch.inference_mode():
            for start in range(0, descriptors.shape[0], 1024):
                value, _ = metric(descriptors[start : start + 1024].to(args.device))
                transformed_query.append(value.cpu())
        metric_scores, metric_candidates = _topk(
            torch.cat(transformed_query), transformed_anchors, k=1, device=args.device
        )
        del metric_scores
        raw = _metrics(raw_candidates, positive, equivalence)
        adapted = _metrics(metric_candidates, positive, equivalence)
        raw_correct = equivalence[raw_candidates[:, 0]] == equivalence[positive]
        metric_correct = equivalence[metric_candidates[:, 0]] == equivalence[positive]
        evaluations[name] = {
            "raw": raw,
            "metric": adapted,
            "wrong_to_truth_count": int((~raw_correct & metric_correct).sum()),
            "truth_to_wrong_count": int((raw_correct & ~metric_correct).sum()),
            "net_correct_gain": int(metric_correct.sum() - raw_correct.sum()),
        }
    args.output_dir.mkdir(parents=True)
    metric_state = metric_artifact(
        metric,
        anchor_ids=state["anchor_ids"],
        map_path=str(args.anchor_map.resolve()),
        map_sha256=sha256_file(args.anchor_map),
        training_report=training,
    )
    metric_state["protocol"] = "v19_mapping_track_identity_pretraining"
    metric_path = args.output_dir / "shared_metric.pt"
    _save(metric_state, metric_path)
    report = {
        "schema": "lafgs_v19_mapping_identity_metric_report",
        "version": 1,
        "uses_test_queries": False,
        "loo_used": False,
        "feedback_evidence_used": False,
        "mapping_family_split": "teacher_track_bank_calibration_validation",
        "training_family_count": len(set(row_families[train_rows].tolist())),
        "training": training,
        "evaluations": evaluations,
        "status": "PROPOSAL_REQUIRES_EXACT_POSE_CONTROL",
        "outputs": {
            "metric": str(metric_path.resolve()),
            "metric_sha256": sha256_file(metric_path),
        },
        "inputs": {
            "anchor_map": str(args.anchor_map.resolve()),
            "anchor_map_sha256": sha256_file(args.anchor_map),
            "mapping_provenance": str(args.mapping_provenance.resolve()),
            "mapping_provenance_sha256": sha256_file(args.mapping_provenance),
            "mapping_feature_cache": str(args.mapping_feature_cache.resolve()),
            "mapping_feature_cache_sha256": sha256_file(args.mapping_feature_cache),
            "teacher_validation": str(args.teacher_validation.resolve()),
            "teacher_validation_sha256": sha256_file(args.teacher_validation),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
