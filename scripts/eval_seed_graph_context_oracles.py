#!/usr/bin/env python
"""Evaluate support-outcome and seed-graph assignment context oracles."""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localization_training.seed_graph_context import (
    apply_bounded_context,
    assignment_metrics,
    build_landmark_outcomes,
    build_positive_pmi_graph,
    camera_center_and_direction,
    confusion_candidate_scores,
    graph_candidate_scores,
)
from scripts.eval_discrete_decision_oracles import (
    CandidateSet,
    nearest_gt_targets,
    project_points,
    run_pose,
    select_candidates,
    summarize_pose_errors,
)
from utils.pose_utils import cal_pose_error


def parse_float_list(text):
    return [float(item) for item in str(text).split(",") if item.strip()]


def softmax_negative(value):
    value = np.asarray(value, dtype=np.float64)
    shifted = -(value - value.min())
    scale = max(float(np.median(np.abs(shifted))), 1e-6)
    probability = np.exp(np.clip(shifted / scale, -50.0, 0.0))
    return probability / probability.sum()


def jaccard(left, right):
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    union = np.union1d(left, right)
    if union.size == 0:
        return 1.0
    return float(np.intersect1d(left, right, assume_unique=False).size / union.size)


def binary_jaccard(left, right):
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    union = left | right
    return float((left & right).sum() / max(int(union.sum()), 1))


def candidate_auroc(labels, scores):
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float(
        (ranks[labels].sum() - positive * (positive + 1) / 2)
        / (positive * negative)
    )


def nearest_supports(
    pose_w2c,
    support_centers,
    support_directions,
    center_scale,
    count,
):
    center, direction = camera_center_and_direction(pose_w2c)
    center_distance = np.linalg.norm(support_centers - center[None], axis=1)
    cosine = np.clip(support_directions @ direction, -1.0, 1.0)
    angle = np.arccos(cosine)
    distance = center_distance / max(float(center_scale), 1e-6)
    distance += angle / (math.pi / 6.0)
    selected = np.argsort(distance, kind="stable")[: int(count)]
    return selected, softmax_negative(distance[selected]), center_distance, angle


def support_center_scale(centers):
    centers = np.asarray(centers, dtype=np.float64)
    nearest = []
    for index in range(centers.shape[0]):
        distance = np.linalg.norm(centers - centers[index], axis=1)
        distance[index] = np.inf
        nearest.append(float(distance.min()))
    positive = np.asarray(nearest)[np.asarray(nearest) > 1e-8]
    return float(np.median(positive)) if positive.size else 1.0


def high_confidence_seed_rows(
    logits,
    selected,
    landmark_ids,
    reliability,
    *,
    seed_count,
    require_reliability,
    eligible_rows=None,
):
    logits = np.asarray(logits, dtype=np.float64)
    selected = np.asarray(selected, dtype=np.int64)
    rows = np.arange(selected.size)
    selected_logits = logits[rows, selected]
    alternative = logits.copy()
    alternative[rows, selected] = -np.inf
    margin = selected_logits - alternative.max(axis=1)
    selected_landmarks = landmark_ids[rows, selected]
    quality = margin.copy()
    eligible = (
        np.ones(rows.size, dtype=bool)
        if eligible_rows is None
        else np.asarray(eligible_rows, dtype=bool).reshape(-1).copy()
    )
    if eligible.size != rows.size:
        raise ValueError("eligible seed rows do not align with logits")
    if require_reliability:
        selected_reliability = reliability[selected_landmarks]
        observed = reliability[reliability > 0.0]
        reliability_threshold = (
            float(np.quantile(observed, 0.75)) if observed.size else 1.0
        )
        eligible &= selected_reliability >= reliability_threshold
        quality += np.log(np.maximum(selected_reliability, 1e-6))
    order = np.argsort(-quality, kind="stable")
    chosen = []
    used = set()
    for row in order:
        landmark = int(selected_landmarks[row])
        if not eligible[row] or landmark in used:
            continue
        chosen.append(int(row))
        used.add(landmark)
        if len(chosen) >= int(seed_count):
            break
    return np.asarray(chosen, dtype=np.int64), margin


def aggregate_variant(records):
    positive = np.concatenate([row["positive"] for row in records], axis=0)
    baseline = np.concatenate([row["baseline"] for row in records])
    selected = np.concatenate([row["selected"] for row in records])
    matchable = np.concatenate([row["matchable"] for row in records])
    summary = assignment_metrics(positive, baseline, selected, matchable)
    by_sequence = {}
    for sequence in sorted({row["sequence"] for row in records}):
        subset = [row for row in records if row["sequence"] == sequence]
        by_sequence[sequence] = assignment_metrics(
            np.concatenate([row["positive"] for row in subset], axis=0),
            np.concatenate([row["baseline"] for row in subset]),
            np.concatenate([row["selected"] for row in subset]),
            np.concatenate([row["matchable"] for row in subset]),
        )
    summary["by_sequence"] = by_sequence
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump_dir", required=True)
    parser.add_argument("--query_cache", required=True)
    parser.add_argument("--visibility_cache", required=True)
    parser.add_argument("--support_graph", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--nearest_views", type=int, default=8)
    parser.add_argument("--minimum_cohits", type=int, default=2)
    parser.add_argument("--seed_count", type=int, default=16)
    parser.add_argument(
        "--protect_margin_quantile", type=float, default=0.75
    )
    parser.add_argument(
        "--protect_reliability_quantile", type=float, default=0.75
    )
    parser.add_argument("--confusion_weight", type=float, default=1.0)
    parser.add_argument(
        "--delta_values", default="0.01,0.02,0.05,0.10,0.20"
    )
    parser.add_argument("--correct_radius", type=float, default=2.0)
    parser.add_argument("--gate_gain", type=float, default=0.02)
    parser.add_argument("--gate_ratio", type=float, default=2.0)
    parser.add_argument("--gate_retention", type=float, default=0.99)
    parser.add_argument(
        "--run_passing_pose", action="store_true",
        help="Run PnP only for a passing deployable C4 configuration.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    manifest = json.loads((dump_dir / "manifest.json").read_text())
    with np.load(dump_dir / manifest["landmark_bank"]) as loaded:
        landmark_xyz = np.asarray(loaded["landmark_xyz"], dtype=np.float64)
    landmark_count = landmark_xyz.shape[0]

    cache = torch.load(args.query_cache, map_location="cpu")["queries"]
    visibility = torch.load(
        args.visibility_cache, map_location="cpu"
    )["visibility"]
    graph_blob = torch.load(args.support_graph, map_location="cpu")
    support_records = graph_blob["graph"]
    support_names = [record["name"] for record in support_records]
    if any(name not in cache or name not in visibility for name in support_names):
        raise ValueError("support graph, query cache, and visibility do not align")
    correct_incidence, false_incidence, correct_sets, false_sets = (
        build_landmark_outcomes(support_records, landmark_count)
    )
    pmi_graph = build_positive_pmi_graph(
        correct_incidence, minimum_cohits=args.minimum_cohits
    )
    correct_dense = correct_incidence.toarray().astype(np.float32)
    false_dense = false_incidence.toarray().astype(np.float32)
    outcome_dense = correct_dense - false_dense
    support_visibility = np.stack(
        [np.asarray(visibility[name], dtype=bool) for name in support_names]
    )
    support_centers = []
    support_directions = []
    for name in support_names:
        center, direction = camera_center_and_direction(
            cache[name]["pose_w2c"]
        )
        support_centers.append(center)
        support_directions.append(direction)
    support_centers = np.asarray(support_centers)
    support_directions = np.asarray(support_directions)
    center_scale = support_center_scale(support_centers)
    correct_count = np.asarray(correct_incidence.sum(axis=0)).reshape(-1)
    false_count = np.asarray(false_incidence.sum(axis=0)).reshape(-1)
    observed_count = correct_count + false_count
    reliability = np.zeros_like(observed_count, dtype=np.float64)
    observed = observed_count > 0
    reliability[observed] = (correct_count[observed] + 1.0) / (
        observed_count[observed] + 2.0
    )

    deltas = parse_float_list(args.delta_values)
    methods = ("C1_gt_nearest_visibility", "C2_gt_nearest_outcome",
               "C3_gt_clean_seed_pmi", "C4_predicted_seed_pmi")
    records = {
        (method, delta): [] for method in methods for delta in deltas
    }
    baseline_records = []
    context_labels = defaultdict(list)
    context_scores = defaultdict(list)
    support_audit = defaultdict(list)
    pose_payload = []
    retrieval_records = []
    rng = np.random.default_rng(args.seed)

    for query_index, query_file in enumerate(manifest["query_files"]):
        with np.load(dump_dir / query_file, allow_pickle=False) as loaded:
            query = {key: np.asarray(loaded[key]) for key in loaded.files}
        required = (
            "assignment_topk_landmark_idx",
            "assignment_candidate_logits",
            "assignment_selected_position",
        )
        missing = [key for key in required if key not in query]
        if missing:
            raise ValueError(
                f"{query_file}: dump predates candidate-logit schema: {missing}"
            )
        image_name = str(query["image_name"].item())
        sequence = Path(image_name).parts[0]
        candidate = np.asarray(
            query["assignment_topk_landmark_idx"], dtype=np.int64
        )
        logits = np.asarray(
            query["assignment_candidate_logits"], dtype=np.float64
        )
        baseline = np.asarray(
            query["assignment_selected_position"], dtype=np.int64
        )
        keypoint_xy = np.asarray(query["keypoint_xy"], dtype=np.float64) + 0.5
        K = np.asarray(query["K"], dtype=np.float64)
        gt_pose = np.asarray(query["gt_pose_w2c"], dtype=np.float64)
        visible = np.asarray(query["render_visible_bank"], dtype=bool)
        projected, _, projection_valid = project_points(
            landmark_xyz, K, gt_pose
        )
        width = int(query["width"])
        height = int(query["height"])
        valid = (
            visible
            & projection_valid
            & (projected[:, 0] >= 0.0)
            & (projected[:, 0] < width)
            & (projected[:, 1] >= 0.0)
            & (projected[:, 1] < height)
        )
        distance = np.linalg.norm(
            keypoint_xy[:, None] - projected[candidate], axis=2
        )
        positive = valid[candidate] & (
            distance <= float(args.correct_radius)
        )
        _, nearest_distance = nearest_gt_targets(
            keypoint_xy, projected, valid, float(args.correct_radius)
        )
        matchable = np.isfinite(nearest_distance) & (
            nearest_distance <= float(args.correct_radius)
        )
        rows = np.arange(candidate.shape[0])
        baseline_correct = positive[rows, baseline]

        nearest, nearest_weight, center_distance, angle = nearest_supports(
            gt_pose,
            support_centers,
            support_directions,
            center_scale,
            args.nearest_views,
        )
        c1 = np.tensordot(
            nearest_weight, support_visibility[nearest][:, candidate], axes=1
        )
        c2 = np.tensordot(
            nearest_weight, outcome_dense[nearest][:, candidate], axes=1
        )

        gt_seed_rows, margin = high_confidence_seed_rows(
            logits,
            baseline,
            candidate,
            reliability,
            seed_count=args.seed_count,
            require_reliability=False,
            eligible_rows=baseline_correct,
        )
        gt_seed_ids = candidate[gt_seed_rows, baseline[gt_seed_rows]]
        gt_seed_weights = np.maximum(margin[gt_seed_rows], 1e-6)
        c3_positive = graph_candidate_scores(
            pmi_graph, candidate, gt_seed_ids, gt_seed_weights
        )
        c3_confusion = confusion_candidate_scores(
            correct_incidence,
            false_incidence,
            candidate,
            gt_seed_ids,
            gt_seed_weights,
        )
        c3 = c3_positive - float(args.confusion_weight) * c3_confusion

        selected_landmarks = candidate[rows, baseline]
        selected_reliability = reliability[selected_landmarks]
        observed_reliability = reliability[reliability > 0.0]
        reliability_threshold = (
            float(
                np.quantile(
                    observed_reliability,
                    float(args.protect_reliability_quantile),
                )
            )
            if observed_reliability.size
            else 1.0
        )
        margin_threshold = float(
            np.quantile(margin, float(args.protect_margin_quantile))
        )
        predicted_core = (
            (margin >= margin_threshold)
            & (selected_reliability >= reliability_threshold)
        )
        predicted_seed_rows, _ = high_confidence_seed_rows(
            logits,
            baseline,
            candidate,
            reliability,
            seed_count=args.seed_count,
            require_reliability=True,
            eligible_rows=predicted_core,
        )
        predicted_seed_ids = candidate[
            predicted_seed_rows, baseline[predicted_seed_rows]
        ]
        predicted_seed_weights = np.maximum(
            margin[predicted_seed_rows], 1e-6
        )
        c4_positive = graph_candidate_scores(
            pmi_graph,
            candidate,
            predicted_seed_ids,
            predicted_seed_weights,
        )
        c4_confusion = confusion_candidate_scores(
            correct_incidence,
            false_incidence,
            candidate,
            predicted_seed_ids,
            predicted_seed_weights,
        )
        c4 = c4_positive - float(args.confusion_weight) * c4_confusion
        predicted_protected = predicted_core
        gt_protected = baseline_correct & (margin >= margin_threshold)

        baseline_records.append(
            {
                "positive": positive,
                "baseline": baseline,
                "selected": baseline,
                "matchable": matchable,
                "sequence": sequence,
            }
        )
        contexts = {
            "C1_gt_nearest_visibility": (c1, predicted_protected),
            "C2_gt_nearest_outcome": (c2, predicted_protected),
            "C3_gt_clean_seed_pmi": (c3, gt_protected),
            "C4_predicted_seed_pmi": (c4, predicted_protected),
        }
        for method, (context, protected) in contexts.items():
            context_labels[method].append(positive)
            context_scores[method].append(context)
            for delta in deltas:
                selected, adjusted = apply_bounded_context(
                    logits,
                    context,
                    delta_max=delta,
                    protected_rows=protected,
                )
                records[(method, delta)].append(
                    {
                        "positive": positive,
                        "baseline": baseline,
                        "selected": selected,
                        "matchable": matchable,
                        "sequence": sequence,
                    }
                )

        retrieval_candidate = np.asarray(
            query["topk_landmark_idx"], dtype=np.int64
        )
        retrieval_distance = np.linalg.norm(
            keypoint_xy[:, None] - projected[retrieval_candidate], axis=2
        )
        retrieval_positive = valid[retrieval_candidate] & (
            retrieval_distance <= float(args.correct_radius)
        )
        retrieval_records.append(
            {
                "sequence": sequence,
                "matchable": matchable,
                "positive": retrieval_positive,
            }
        )

        query_positive_ids = np.unique(candidate[positive])
        query_correct_ids = np.unique(
            candidate[rows[baseline_correct], baseline[baseline_correct]]
        )
        query_false_ids = np.unique(
            candidate[rows[~baseline_correct], baseline[~baseline_correct]]
        )
        random_support = rng.choice(
            len(support_names),
            size=min(args.nearest_views, len(support_names)),
            replace=False,
        )
        for label, support_ids in (
            ("nearest", nearest),
            ("random", random_support),
        ):
            support_audit[f"{label}_center_distance_m"].extend(
                center_distance[support_ids].tolist()
            )
            support_audit[f"{label}_direction_degrees"].extend(
                np.degrees(angle[support_ids]).tolist()
            )
            support_audit[f"{label}_visible_jaccard"].extend(
                [
                    binary_jaccard(visible, support_visibility[index])
                    for index in support_ids
                ]
            )
            support_audit[f"{label}_positive_jaccard"].extend(
                [jaccard(query_positive_ids, correct_sets[index])
                 for index in support_ids]
            )
            support_audit[f"{label}_correct_hit_jaccard"].extend(
                [jaccard(query_correct_ids, correct_sets[index])
                 for index in support_ids]
            )
            support_audit[f"{label}_false_attractor_jaccard"].extend(
                [jaccard(query_false_ids, false_sets[index])
                 for index in support_ids]
            )
        pose_payload.append(
            {
                "query": query,
                "candidate": candidate,
                "logits": logits,
                "context": c4,
                "protected": predicted_protected,
                "keypoint_xy": keypoint_xy,
                "landmark_xyz": landmark_xyz,
                "K": K,
                "gt_pose": gt_pose,
                "query_index": query_index,
            }
        )

    baseline_summary = aggregate_variant(baseline_records)
    matrix = {}
    passing = []
    for method in methods:
        method_rows = {}
        labels = np.concatenate(context_labels[method], axis=0)
        scores = np.concatenate(context_scores[method], axis=0)
        auroc = candidate_auroc(labels, scores)
        for delta in deltas:
            summary = aggregate_variant(records[(method, delta)])
            summary["candidate_level_auroc"] = auroc
            gain = (
                summary["conditional_recall_at_1_given_matchable"]
                - baseline_summary[
                    "conditional_recall_at_1_given_matchable"
                ]
            )
            gate = (
                gain >= float(args.gate_gain)
                and summary["beneficial_harmful_ratio"]
                > float(args.gate_ratio)
                and summary["clean_top1_retention"]
                >= float(args.gate_retention)
            )
            summary["conditional_gain"] = float(gain)
            summary["passes_gate"] = bool(gate)
            method_rows[str(delta)] = summary
            if gate:
                passing.append((method, delta, summary))
        matrix[method] = method_rows

    deployable = [
        item for item in passing if item[0] == "C4_predicted_seed_pmi"
    ]
    deployable.sort(
        key=lambda item: (
            item[2]["conditional_recall_at_1_given_matchable"],
            -item[2]["harmful_swaps"],
        ),
        reverse=True,
    )
    pose_summary = {}
    if args.run_passing_pose and deployable:
        _, best_delta, _ = deployable[0]
        base_ae, base_te, context_ae, context_te = [], [], [], []
        for payload in pose_payload:
            query = payload["query"]
            rows = np.arange(payload["candidate"].shape[0])
            baseline_selected = payload["logits"].argmax(axis=1)
            context_selected, adjusted = apply_bounded_context(
                payload["logits"],
                payload["context"],
                delta_max=best_delta,
                protected_rows=payload["protected"],
            )
            candidates = {}
            for name, selected, scores in (
                ("baseline", baseline_selected, payload["logits"]),
                ("context", context_selected, adjusted),
            ):
                raw = select_candidates(
                    rows,
                    payload["candidate"][rows, selected],
                    scores[rows, selected],
                    threshold=float(query["candidate_threshold"].item()),
                    max_matches_per_keypoint=int(
                        query["max_matches_per_keypoint"].item()
                    ),
                    max_matches_per_landmark=int(
                        query["max_matches_per_landmark"].item()
                    ),
                    min_match_count=int(
                        query["min_candidate_matches"].item()
                    ),
                    refill_trigger_count=int(
                        query["candidate_refill_trigger_count"].item()
                    ),
                )
                pose, _ = run_pose(
                    raw,
                    payload["keypoint_xy"],
                    payload["landmark_xyz"],
                    payload["K"],
                    query,
                    args.seed + payload["query_index"],
                )
                ae, te = cal_pose_error(pose, payload["gt_pose"])
                if name == "baseline":
                    base_ae.append(ae)
                    base_te.append(te)
                else:
                    context_ae.append(ae)
                    context_te.append(te)
        pose_summary = {
            "method": "C4_predicted_seed_pmi",
            "delta": best_delta,
            "baseline": summarize_pose_errors(base_ae, base_te),
            "context": summarize_pose_errors(context_ae, context_te),
            "paired_wins": int(
                (np.asarray(context_te) < np.asarray(base_te)).sum()
            ),
            "paired_losses": int(
                (np.asarray(context_te) > np.asarray(base_te)).sum()
            ),
        }

    retrieval_summary = {}
    for topk in (1, 4, 8, 16, 32):
        topk_rows = {}
        for sequence in ("all", *sorted({
            row["sequence"] for row in retrieval_records
        })):
            subset = (
                retrieval_records
                if sequence == "all"
                else [
                    row for row in retrieval_records
                    if row["sequence"] == sequence
                ]
            )
            positive = np.concatenate(
                [row["positive"][:, :topk].any(axis=1) for row in subset]
            )
            matchable = np.concatenate(
                [row["matchable"] for row in subset]
            )
            topk_rows[sequence] = {
                "all_rows": float(positive.mean()),
                "given_matchable": float(
                    positive[matchable].mean()
                    if matchable.any()
                    else 0.0
                ),
                "matchable_rows": int(matchable.sum()),
            }
        retrieval_summary[str(topk)] = topk_rows

    result = {
        "protocol": {
            **vars(args),
            "query_count": len(manifest["query_files"]),
            "support_count": len(support_names),
            "landmark_count": landmark_count,
            "support_center_scale_m": center_scale,
            "pmi_edges": int(pmi_graph.nnz),
            "support_correct_hits_mean": float(
                correct_incidence.getnnz(axis=1).mean()
            ),
            "support_false_attractors_mean": float(
                false_incidence.getnnz(axis=1).mean()
            ),
        },
        "C0_legacy_assign": baseline_summary,
        "retrieval_positive_in_topk": retrieval_summary,
        "matrix": matrix,
        "passing": [
            {"method": method, "delta": delta, **summary}
            for method, delta, summary in passing
        ],
        "support_retrieval_audit": {
            key: {
                "mean": float(np.mean(value)),
                "median": float(np.median(value)),
            }
            for key, value in support_audit.items()
        },
        "pose": pose_summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
    nearest_gt_targets,
