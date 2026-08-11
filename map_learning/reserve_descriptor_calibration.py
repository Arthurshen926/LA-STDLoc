#!/usr/bin/env python3
"""Cross-fitted calibration of repeatedly harmful Gaussian-reserve descriptors."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from localization.localizer import load_shared_metric
from localization.matcher import global_cosine_top2
from map_learning.trainer import _build_training_records
from topology.deployment_revision import collect_deployment_statistics
from topology.reserve_geometry_refinement import temporal_threeway_split


def _append_evidence(storage: dict, anchor: int, value: dict) -> None:
    storage[int(anchor)].append(value)


@torch.inference_mode()
def collect_descriptor_evidence(
    *,
    state: dict,
    metric_state_path: str | Path,
    graph: dict,
    payload: dict,
    teacher: dict,
    query_cache: dict,
    query_indices: list[int],
    device: torch.device,
    max_positives: int = 8,
) -> dict:
    """Collect legal positives and current false top-1 assignments by anchor."""
    records, _ = _build_training_records(
        graph, payload, state, teacher, int(max_positives)
    )
    names = list(graph["query_names"])
    cache = query_cache.get("queries", query_cache)
    anchor_ids = torch.as_tensor(state["anchor_ids"]).long()
    anchor_type = torch.as_tensor(state["anchor_type"]).long()
    anchor_type_device = anchor_type.to(device)
    bank = F.normalize(
        torch.as_tensor(state["anchor_features"], device=device).float(), dim=1
    )
    bank_cpu = bank.cpu()
    metric = load_shared_metric(metric_state_path, anchor_ids=anchor_ids, device=device)
    positives: dict[int, list[dict]] = defaultdict(list)
    negatives: dict[int, list[dict]] = defaultdict(list)
    row_count = 0
    for query_index in query_indices:
        record = records[int(query_index)]
        rows = record["cache_rows"]
        if not rows.numel():
            continue
        raw = torch.as_tensor(
            cache[names[int(query_index)]]["native_descriptors"],
            device=device,
        ).float()[rows.to(device)]
        descriptors, _ = metric(raw)
        matches = global_cosine_top2(descriptors, bank)
        top1 = matches.anchor_indices[:, 0]
        top2_score = matches.scores[:, 1]
        positive = record["positives"].to(device)
        ignored = record["ignored_anchors"].to(device)
        matchable = record["matchable"].to(device)
        top1_is_positive = ((positive == top1[:, None]) & (positive >= 0)).any(dim=1)
        top1_is_ignored = ((ignored == top1[:, None]) & (ignored >= 0)).any(dim=1)
        false = (
            matchable
            & ~top1_is_positive
            & ~top1_is_ignored
            & (anchor_type_device[top1] == 0)
        )
        descriptors = descriptors.detach().cpu()
        top1 = top1.cpu()
        top2_score = top2_score.cpu()
        top1_score = matches.scores[:, 0].cpu()
        positive = positive.cpu()
        matchable = matchable.cpu()
        false = false.cpu()
        for local in torch.nonzero(false, as_tuple=False).reshape(-1).tolist():
            anchor = int(top1[local])
            _append_evidence(
                negatives,
                anchor,
                {
                    "query": int(query_index),
                    "descriptor": descriptors[local],
                    "competitor_score": float(top2_score[local]),
                    "old_score": float(top1_score[local]),
                },
            )
        for local in torch.nonzero(matchable, as_tuple=False).reshape(-1).tolist():
            legal = torch.unique(positive[local][positive[local] >= 0])
            for anchor_tensor in legal:
                anchor = int(anchor_tensor)
                if int(anchor_type[anchor]) != 0:
                    continue
                if int(top1[local]) == anchor:
                    competitor = float(top2_score[local])
                else:
                    competitor = float(matches.scores[local, 0])
                _append_evidence(
                    positives,
                    anchor,
                    {
                        "query": int(query_index),
                        "descriptor": descriptors[local],
                        "competitor_score": competitor,
                        "old_score": float(descriptors[local] @ bank_cpu[anchor]),
                    },
                )
        row_count += int(rows.numel())
    return {
        "positives": dict(positives),
        "negatives": dict(negatives),
        "query_count": len(query_indices),
        "row_count": row_count,
    }


def _distinct_queries(values: list[dict]) -> int:
    return len({int(value["query"]) for value in values})


def optimize_bounded_descriptor(
    initial: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    maximum_residual_norm: float,
    margin: float,
    temperature: float,
    trust_weight: float,
    steps: int,
    learning_rate: float,
) -> tuple[torch.Tensor, dict]:
    """Fit one final map descriptor without changing the query-side metric."""
    initial = F.normalize(torch.as_tensor(initial).float(), dim=0)
    positive = F.normalize(torch.as_tensor(positive).float(), dim=1)
    negative = F.normalize(torch.as_tensor(negative).float(), dim=1)
    residual = torch.zeros_like(initial, requires_grad=True)
    optimizer = torch.optim.Adam([residual], lr=float(learning_rate))
    history = []
    for step in range(int(steps)):
        bounded = residual * torch.clamp(
            float(maximum_residual_norm) / torch.linalg.norm(residual).clamp_min(1e-8),
            max=1.0,
        )
        descriptor = F.normalize(initial + bounded, dim=0)
        positive_score = positive @ descriptor
        negative_score = negative @ descriptor
        ranking = F.softplus(
            (
                torch.quantile(negative_score, 0.9)
                - torch.quantile(positive_score, 0.1)
                + float(margin)
            )
            / float(temperature)
        ) * float(temperature)
        attraction = 1.0 - positive_score.mean()
        trust = bounded.square().sum()
        loss = ranking + 0.1 * attraction + float(trust_weight) * trust
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in {0, int(steps) - 1}:
            history.append(float(loss.detach()))
    with torch.no_grad():
        bounded = residual * torch.clamp(
            float(maximum_residual_norm) / torch.linalg.norm(residual).clamp_min(1e-8),
            max=1.0,
        )
        descriptor = F.normalize(initial + bounded, dim=0)
    return descriptor, {
        "residual_norm": float(torch.linalg.norm(bounded)),
        "loss_initial": history[0],
        "loss_final": history[-1],
    }


def validate_descriptor(
    initial: torch.Tensor,
    candidate: torch.Tensor,
    positive: list[dict],
    negative: list[dict],
    *,
    maximum_positive_score_drop: float,
) -> dict:
    positive_descriptor = torch.stack([value["descriptor"] for value in positive])
    negative_descriptor = torch.stack([value["descriptor"] for value in negative])
    positive_competitor = torch.as_tensor(
        [value["competitor_score"] for value in positive]
    )
    negative_competitor = torch.as_tensor(
        [value["competitor_score"] for value in negative]
    )
    old_positive = positive_descriptor @ initial
    new_positive = positive_descriptor @ candidate
    old_negative = negative_descriptor @ initial
    new_negative = negative_descriptor @ candidate
    old_positive_wins = int((old_positive >= positive_competitor).sum())
    new_positive_wins = int((new_positive >= positive_competitor).sum())
    old_false_wins = int((old_negative >= negative_competitor).sum())
    new_false_wins = int((new_negative >= negative_competitor).sum())
    accepted = bool(
        new_positive_wins >= old_positive_wins
        and float(new_positive.mean())
        >= float(old_positive.mean()) - float(maximum_positive_score_drop)
        and new_false_wins < old_false_wins
    )
    return {
        "positive_count": len(positive),
        "negative_count": len(negative),
        "old_positive_wins": old_positive_wins,
        "new_positive_wins": new_positive_wins,
        "old_false_wins": old_false_wins,
        "new_false_wins": new_false_wins,
        "old_positive_score_mean": float(old_positive.mean()),
        "new_positive_score_mean": float(new_positive.mean()),
        "old_negative_score_mean": float(old_negative.mean()),
        "new_negative_score_mean": float(new_negative.mean()),
        "accepted": accepted,
    }


def calibrate_reserve_descriptors(
    *,
    state: dict,
    fit_evidence: dict,
    validation_evidence: dict,
    minimum_fit_positive_views: int,
    minimum_fit_negative_views: int,
    minimum_validation_positive_views: int,
    minimum_validation_negative_views: int,
    maximum_residual_norm: float,
    maximum_positive_score_drop: float,
    margin: float,
    temperature: float,
    trust_weight: float,
    steps: int,
    learning_rate: float,
) -> tuple[dict, dict]:
    output = dict(state)
    features = F.normalize(torch.as_tensor(state["anchor_features"]).float(), dim=1)
    initial_features = features.clone()
    diagnostics = []
    candidates = sorted(
        set(fit_evidence["negatives"]) & set(validation_evidence["negatives"])
    )
    for anchor in candidates:
        fit_positive = fit_evidence["positives"].get(anchor, [])
        fit_negative = fit_evidence["negatives"].get(anchor, [])
        validation_positive = validation_evidence["positives"].get(anchor, [])
        validation_negative = validation_evidence["negatives"].get(anchor, [])
        if (
            _distinct_queries(fit_positive) < int(minimum_fit_positive_views)
            or _distinct_queries(fit_negative) < int(minimum_fit_negative_views)
            or _distinct_queries(validation_positive)
            < int(minimum_validation_positive_views)
            or _distinct_queries(validation_negative)
            < int(minimum_validation_negative_views)
        ):
            continue
        candidate, fit_report = optimize_bounded_descriptor(
            features[anchor],
            torch.stack([value["descriptor"] for value in fit_positive]),
            torch.stack([value["descriptor"] for value in fit_negative]),
            maximum_residual_norm=maximum_residual_norm,
            margin=margin,
            temperature=temperature,
            trust_weight=trust_weight,
            steps=steps,
            learning_rate=learning_rate,
        )
        validation = validate_descriptor(
            features[anchor],
            candidate,
            validation_positive,
            validation_negative,
            maximum_positive_score_drop=maximum_positive_score_drop,
        )
        diagnostics.append(
            {
                "anchor_row": int(anchor),
                "fit_positive_views": _distinct_queries(fit_positive),
                "fit_negative_views": _distinct_queries(fit_negative),
                "validation_positive_views": _distinct_queries(validation_positive),
                "validation_negative_views": _distinct_queries(validation_negative),
                **fit_report,
                **validation,
            }
        )
        if validation["accepted"]:
            features[anchor] = candidate
    accepted = [value["anchor_row"] for value in diagnostics if value["accepted"]]
    output["anchor_features"] = features
    output["track_centric_reconstruction"] = {
        **state["track_centric_reconstruction"],
        "reserve_descriptor_calibration": {
            "candidate_anchor_count": len(diagnostics),
            "accepted_anchor_count": len(accepted),
            "maximum_residual_norm": float(maximum_residual_norm),
            "uses_test_queries": False,
        },
    }
    output["provenance"] = {
        **state.get("provenance", {}),
        "reserve_descriptor_calibration": {
            "policy": "cross_fitted_harmful_top1_with_positive_protection",
            "uses_test_queries": False,
        },
    }
    realized = torch.linalg.norm(features - initial_features, dim=1)
    return output, {
        "candidate_anchor_count": len(diagnostics),
        "accepted_anchor_rows": accepted,
        "accepted_anchor_count": len(accepted),
        "residual_norm_max": float(realized.max()),
        "diagnostics": diagnostics,
    }


def _gate(before: dict, after: dict) -> dict:
    source, revised = before["summary"], after["summary"]
    checks = {
        "median_non_degraded": revised["median_te_cm"] <= 1.01 * source["median_te_cm"],
        "mean_non_degraded": revised["mean_te_cm"] <= 1.01 * source["mean_te_cm"],
        "p90_non_degraded": revised["p90_te_cm"] <= 1.01 * source["p90_te_cm"],
        "cvar95_non_degraded": revised["cvar95_te_cm"] <= 1.01 * source["cvar95_te_cm"],
        "catastrophic_non_degraded": revised["catastrophic_100cm_count"]
        <= source["catastrophic_100cm_count"],
        "raw_precision_non_degraded": revised["raw_gt_precision_percent"] + 0.01
        >= source["raw_gt_precision_percent"],
    }
    checks["meaningful_improvement"] = bool(
        revised["mean_te_cm"] <= 0.995 * source["mean_te_cm"]
        or revised["p90_te_cm"] <= 0.995 * source["p90_te_cm"]
        or revised["cvar95_te_cm"] <= 0.995 * source["cvar95_te_cm"]
        or revised["raw_gt_precision_percent"]
        >= source["raw_gt_precision_percent"] + 0.01
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--function-graph", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--complete-positive-teacher", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crossfit-blocks", type=int, default=9)
    parser.add_argument("--minimum-fit-positive-views", type=int, default=3)
    parser.add_argument("--minimum-fit-negative-views", type=int, default=2)
    parser.add_argument("--minimum-validation-positive-views", type=int, default=2)
    parser.add_argument("--minimum-validation-negative-views", type=int, default=1)
    parser.add_argument("--maximum-residual-norm", type=float, default=0.025)
    parser.add_argument("--maximum-positive-score-drop", type=float, default=0.002)
    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument("--temperature", type=float, default=0.04)
    parser.add_argument("--trust-weight", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--ransac-reprojection-px", type=float, required=True)
    parser.add_argument("--clean-reprojection-px", type=float, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.manual_seed(int(args.seed))
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    graph = torch.load(args.function_graph, map_location="cpu", weights_only=False)
    payload = torch.load(args.track_payload, map_location="cpu", weights_only=False)
    query_cache = torch.load(args.query_cache, map_location="cpu", weights_only=False)
    teacher = torch.load(
        args.complete_positive_teacher, map_location="cpu", weights_only=False
    )
    fit, validation, gate_queries, split = temporal_threeway_split(
        list(graph["query_names"]), args.crossfit_blocks
    )
    common_evidence = {
        "state": state,
        "metric_state_path": args.metric_state,
        "graph": graph,
        "payload": payload,
        "teacher": teacher,
        "query_cache": query_cache,
        "device": torch.device(args.device),
    }
    fit_evidence = collect_descriptor_evidence(query_indices=fit, **common_evidence)
    validation_evidence = collect_descriptor_evidence(
        query_indices=validation, **common_evidence
    )
    revised, calibration = calibrate_reserve_descriptors(
        state=state,
        fit_evidence=fit_evidence,
        validation_evidence=validation_evidence,
        minimum_fit_positive_views=args.minimum_fit_positive_views,
        minimum_fit_negative_views=args.minimum_fit_negative_views,
        minimum_validation_positive_views=args.minimum_validation_positive_views,
        minimum_validation_negative_views=args.minimum_validation_negative_views,
        maximum_residual_norm=args.maximum_residual_norm,
        maximum_positive_score_drop=args.maximum_positive_score_drop,
        margin=args.margin,
        temperature=args.temperature,
        trust_weight=args.trust_weight,
        steps=args.steps,
        learning_rate=args.learning_rate,
    )
    map_path = output / "calibrated_anchor_map.pt"
    metric_path = output / "calibrated_metric_state.pt"
    metric_state = torch.load(args.metric_state, map_location="cpu", weights_only=False)
    metric_state = {**metric_state, "map_path": str(map_path)}
    torch.save(revised, map_path)
    torch.save(metric_state, metric_path)
    parameters = state["track_centric_reconstruction"]["calibration"]["parameters"]
    common_gate = {
        "query_cache": query_cache,
        "teacher": teacher,
        "device": torch.device(args.device),
        "ransac_reprojection_px": args.ransac_reprojection_px,
        "clean_reprojection_px": args.clean_reprojection_px,
        "task_translation_m": float(parameters["task_translation_m"]),
        "task_rotation_deg": float(parameters["task_rotation_deg"]),
        "seed": args.seed,
        "query_indices": gate_queries,
    }
    before = collect_deployment_statistics(
        state=state,
        metric_state_path=args.metric_state,
        progress_label="descriptor_gate_before",
        **common_gate,
    )
    after = collect_deployment_statistics(
        state=revised,
        metric_state_path=metric_path,
        progress_label="descriptor_gate_after",
        **common_gate,
    )
    checks = _gate(before, after)
    accepted = bool(calibration["accepted_anchor_count"]) and all(checks.values())
    report = {
        "schema": "lafgs_crossfit_reserve_descriptor_calibration",
        "version": 1,
        "uses_test_queries": False,
        "source_map": str(Path(args.map).resolve()),
        "calibrated_map": str(map_path),
        "calibrated_metric_state": str(metric_path),
        "split": split,
        "fit_evidence": {
            "query_count": fit_evidence["query_count"],
            "row_count": fit_evidence["row_count"],
        },
        "validation_evidence": {
            "query_count": validation_evidence["query_count"],
            "row_count": validation_evidence["row_count"],
        },
        "calibration": calibration,
        "gate_before": before["summary"],
        "gate_after": after["summary"],
        "gate": checks,
        "accepted": accepted,
    }
    (output / "reserve_descriptor_calibration_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "accepted": accepted,
                "calibration": {
                    key: value
                    for key, value in calibration.items()
                    if key != "diagnostics"
                },
                "gate_before": before["summary"],
                "gate_after": after["summary"],
                "gate": checks,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
