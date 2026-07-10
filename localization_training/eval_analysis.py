import math
import random

import numpy as np


def _query_keyed(results):
    if all("image_name" in item for item in results):
        return {item["image_name"]: item for item in results}
    return {str(idx): item for idx, item in enumerate(results)}


def _recall_mask(item, te_cm, ae_deg):
    return float(item["sparse_TE"]) <= float(te_cm) and float(item["sparse_AE"]) <= float(ae_deg)


def _sparse_inliers(item):
    sparse = item.get("sparse", {})
    if isinstance(sparse, dict) and "inliers" in sparse:
        return float(sparse["inliers"])
    if "inliers" in item:
        return float(item["inliers"])
    return 0.0


def _sparse_optional_metric(item, name):
    sparse = item.get("sparse", {})
    if isinstance(sparse, dict) and name in sparse:
        try:
            return float(sparse[name])
        except (TypeError, ValueError):
            return None
    if name in item:
        try:
            return float(item[name])
        except (TypeError, ValueError):
            return None
    return None


def _numeric_sparse_diagnostic_names(*items):
    names = set()
    for item in items:
        sparse = item.get("sparse", {})
        for source in (item, sparse if isinstance(sparse, dict) else {}):
            if not isinstance(source, dict):
                continue
            for key, value in source.items():
                if not str(key).startswith("sparse_diag_"):
                    continue
                try:
                    float(value)
                except (TypeError, ValueError):
                    continue
                names.add(str(key))
    return names


def _sequence_name(image_name):
    image_name = str(image_name)
    if "/" in image_name:
        return image_name.split("/", 1)[0]
    return ""


def sparse_metric_summary(results):
    total = max(1, len(results))
    ae = np.asarray([float(item["sparse_AE"]) for item in results], dtype=np.float64)
    te = np.asarray([float(item["sparse_TE"]) for item in results], dtype=np.float64)
    inliers = np.asarray([_sparse_inliers(item) for item in results], dtype=np.float64)
    return {
        "query_count": int(len(results)),
        "median_ae": float(np.median(ae)) if ae.size else 0.0,
        "median_te": float(np.median(te)) if te.size else 0.0,
        "recall_5cm_5deg": float(((ae <= 5.0) & (te <= 5.0)).sum() / total),
        "recall_2cm_2deg": float(((ae <= 2.0) & (te <= 2.0)).sum() / total),
        "avg_inliers": float(inliers.mean()) if inliers.size else 0.0,
    }


def paired_sparse_stage_rows(baseline_results, candidate_results):
    baseline_by_key = _query_keyed(baseline_results)
    candidate_by_key = _query_keyed(candidate_results)
    rows = []
    for key in sorted(set(baseline_by_key) & set(candidate_by_key)):
        base = baseline_by_key[key]
        cand = candidate_by_key[key]
        base_te = float(base["sparse_TE"])
        cand_te = float(cand["sparse_TE"])
        base_ae = float(base["sparse_AE"])
        cand_ae = float(cand["sparse_AE"])
        base_inliers = _sparse_inliers(base)
        cand_inliers = _sparse_inliers(cand)
        base_matches = _sparse_optional_metric(base, "matches")
        cand_matches = _sparse_optional_metric(cand, "matches")
        base_keypoints = _sparse_optional_metric(base, "detected_keypoints")
        cand_keypoints = _sparse_optional_metric(cand, "detected_keypoints")
        image_name = str(base.get("image_name", key))
        row = (
            {
                "image_name": image_name,
                "sequence": _sequence_name(image_name),
                "baseline_te": base_te,
                "candidate_te": cand_te,
                "delta_te": cand_te - base_te,
                "baseline_ae": base_ae,
                "candidate_ae": cand_ae,
                "delta_ae": cand_ae - base_ae,
                "baseline_inliers": base_inliers,
                "candidate_inliers": cand_inliers,
                "delta_inliers": cand_inliers - base_inliers,
                "baseline_matches": base_matches,
                "candidate_matches": cand_matches,
                "delta_matches": cand_matches - base_matches
                if base_matches is not None and cand_matches is not None
                else None,
                "baseline_keypoints": base_keypoints,
                "candidate_keypoints": cand_keypoints,
                "delta_keypoints": cand_keypoints - base_keypoints
                if base_keypoints is not None and cand_keypoints is not None
                else None,
            }
        )
        for diag_key in sorted(_numeric_sparse_diagnostic_names(base, cand)):
            base_value = _sparse_optional_metric(base, diag_key)
            cand_value = _sparse_optional_metric(cand, diag_key)
            row[f"baseline_{diag_key}"] = base_value
            row[f"candidate_{diag_key}"] = cand_value
            row[f"delta_{diag_key}"] = (
                cand_value - base_value if base_value is not None and cand_value is not None else None
            )
        rows.append(row)
    return rows


def _mean(values):
    values = [value for value in values if value is not None]
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()) if values.size else 0.0


def _median(values):
    values = [value for value in values if value is not None]
    values = np.asarray(values, dtype=np.float64)
    return float(np.median(values)) if values.size else 0.0


def _top_rows(rows, key, top_k, reverse=True):
    public_keys = (
        "image_name",
        "sequence",
        "baseline_te",
        "candidate_te",
        "delta_te",
        "baseline_ae",
        "candidate_ae",
        "delta_ae",
        "baseline_inliers",
        "candidate_inliers",
        "delta_inliers",
        "baseline_matches",
        "candidate_matches",
        "delta_matches",
        "baseline_keypoints",
        "candidate_keypoints",
        "delta_keypoints",
    )
    return [
        {public_key: row.get(public_key) for public_key in public_keys if public_key in row}
        for row in sorted(rows, key=lambda row: float(row.get(key, 0.0)), reverse=reverse)[
            : max(0, int(top_k))
        ]
    ]


def _pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return 0.0
    xs = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    ys = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    if float(xs.std()) == 0.0 or float(ys.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(xs, ys)[0, 1])


def _paired_sparse_diagnostic_summary(rows):
    keys = sorted(
        key[len("baseline_") :]
        for row in rows
        for key in row
        if key.startswith("baseline_sparse_diag_")
    )
    summaries = {}
    for key in keys:
        paired_rows = [
            row
            for row in rows
            if row.get(f"baseline_{key}") is not None and row.get(f"candidate_{key}") is not None
        ]
        if not paired_rows:
            continue
        deltas = [row[f"delta_{key}"] for row in paired_rows]
        degraded = [row[f"delta_{key}"] for row in paired_rows if float(row["delta_te"]) > 0.0]
        improved = [row[f"delta_{key}"] for row in paired_rows if float(row["delta_te"]) < 0.0]
        summaries[key] = {
            "count": int(len(paired_rows)),
            "baseline_mean": _mean([row[f"baseline_{key}"] for row in paired_rows]),
            "candidate_mean": _mean([row[f"candidate_{key}"] for row in paired_rows]),
            "delta_mean": _mean(deltas),
            "baseline_median": _median([row[f"baseline_{key}"] for row in paired_rows]),
            "candidate_median": _median([row[f"candidate_{key}"] for row in paired_rows]),
            "delta_median": _median(deltas),
            "pose_degraded_delta_mean": _mean(degraded),
            "pose_improved_delta_mean": _mean(improved),
            "translation_delta_pearson": _pearson(deltas, [row["delta_te"] for row in paired_rows]),
        }
    return summaries


def paired_sparse_stage_summary(
    baseline_results,
    candidate_results,
    te_ok_cm=5.0,
    ae_ok_deg=5.0,
    inlier_drop_threshold=50,
    top_k=10,
):
    rows = paired_sparse_stage_rows(baseline_results, candidate_results)
    paired = paired_sparse_summary(baseline_results, candidate_results, bootstrap_samples=0)
    te_ok_cm = float(te_ok_cm)
    ae_ok_deg = float(ae_ok_deg)
    inlier_drop_threshold = float(inlier_drop_threshold)

    def ok(prefix, row):
        return float(row[f"{prefix}_te"]) <= te_ok_cm and float(row[f"{prefix}_ae"]) <= ae_ok_deg

    recall_gain = 0
    recall_loss = 0
    both_ok = 0
    both_bad = 0
    inlier_drop = 0
    pose_degraded_and_inlier_drop = 0
    pose_improved_despite_inlier_drop = 0
    for row in rows:
        base_ok = ok("baseline", row)
        cand_ok = ok("candidate", row)
        recall_gain += int((not base_ok) and cand_ok)
        recall_loss += int(base_ok and (not cand_ok))
        both_ok += int(base_ok and cand_ok)
        both_bad += int((not base_ok) and (not cand_ok))
        dropped = float(row["delta_inliers"]) <= -inlier_drop_threshold
        inlier_drop += int(dropped)
        pose_degraded_and_inlier_drop += int(dropped and float(row["delta_te"]) > 0.0)
        pose_improved_despite_inlier_drop += int(dropped and float(row["delta_te"]) < 0.0)

    sequence_groups = []
    for sequence in sorted({row["sequence"] for row in rows}):
        group = [row for row in rows if row["sequence"] == sequence]
        sequence_groups.append(
            {
                "sequence": sequence,
                "count": int(len(group)),
                "baseline_median_te": _median([row["baseline_te"] for row in group]),
                "candidate_median_te": _median([row["candidate_te"] for row in group]),
                "delta_median_te": _median([row["delta_te"] for row in group]),
                "delta_mean_te": _mean([row["delta_te"] for row in group]),
                "delta_mean_inliers": _mean([row["delta_inliers"] for row in group]),
                "candidate_avg_inliers": _mean([row["candidate_inliers"] for row in group]),
                "recall_loss_count": int(sum(ok("baseline", row) and not ok("candidate", row) for row in group)),
                "recall_gain_count": int(sum((not ok("baseline", row)) and ok("candidate", row) for row in group)),
            }
        )

    return {
        "query_count": int(len(rows)),
        "baseline": sparse_metric_summary([_query_keyed(baseline_results)[row["image_name"]] for row in rows])
        if rows
        else sparse_metric_summary([]),
        "candidate": sparse_metric_summary([_query_keyed(candidate_results)[row["image_name"]] for row in rows])
        if rows
        else sparse_metric_summary([]),
        "paired": paired,
        "delta_mean_te": _mean([row["delta_te"] for row in rows]),
        "delta_median_te": _median([row["delta_te"] for row in rows]),
        "delta_mean_ae": _mean([row["delta_ae"] for row in rows]),
        "delta_median_ae": _median([row["delta_ae"] for row in rows]),
        "delta_mean_inliers": _mean([row["delta_inliers"] for row in rows]),
        "delta_median_inliers": _median([row["delta_inliers"] for row in rows]),
        "delta_mean_matches": _mean([row["delta_matches"] for row in rows]),
        "delta_mean_keypoints": _mean([row["delta_keypoints"] for row in rows]),
        "diagnostics": _paired_sparse_diagnostic_summary(rows),
        "recall_5cm_gain_count": int(recall_gain),
        "recall_5cm_loss_count": int(recall_loss),
        "both_ok_count": int(both_ok),
        "both_bad_count": int(both_bad),
        "inlier_drop_count": int(inlier_drop),
        "pose_degraded_and_inlier_drop_count": int(pose_degraded_and_inlier_drop),
        "pose_improved_despite_inlier_drop_count": int(pose_improved_despite_inlier_drop),
        "sequence_groups": sequence_groups,
        "top_te_degraded": _top_rows(rows, "delta_te", top_k, reverse=True),
        "top_te_improved": _top_rows(rows, "delta_te", top_k, reverse=False),
        "top_inlier_drop": _top_rows(rows, "delta_inliers", top_k, reverse=False),
        "worst_candidate_te": _top_rows(rows, "candidate_te", top_k, reverse=True),
    }


def _bootstrap_mean_ci(values, samples=1000, seed=0, alpha=0.05):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return [0.0, 0.0]
    if values.size == 1 or samples <= 0:
        value = float(values.mean())
        return [value, value]
    rng = random.Random(seed)
    means = []
    n = int(values.size)
    for _ in range(int(samples)):
        idx = [rng.randrange(n) for _ in range(n)]
        means.append(float(values[idx].mean()))
    means.sort()
    lo = means[max(0, int(math.floor((alpha / 2.0) * len(means))) - 1)]
    hi = means[min(len(means) - 1, int(math.ceil((1.0 - alpha / 2.0) * len(means))) - 1)]
    return [float(lo), float(hi)]


def paired_sparse_summary(baseline_results, la_results, bootstrap_samples=1000, seed=0):
    baseline_by_key = _query_keyed(baseline_results)
    la_by_key = _query_keyed(la_results)
    keys = sorted(set(baseline_by_key) & set(la_by_key))
    translation_delta = []
    rotation_delta = []
    gain_5cm = 0
    loss_5cm = 0
    gain_2cm = 0
    loss_2cm = 0
    for key in keys:
        base = baseline_by_key[key]
        la = la_by_key[key]
        translation_delta.append(float(la["sparse_TE"]) - float(base["sparse_TE"]))
        rotation_delta.append(float(la["sparse_AE"]) - float(base["sparse_AE"]))
        base_5 = _recall_mask(base, 5.0, 5.0)
        la_5 = _recall_mask(la, 5.0, 5.0)
        base_2 = _recall_mask(base, 2.0, 2.0)
        la_2 = _recall_mask(la, 2.0, 2.0)
        gain_5cm += int((not base_5) and la_5)
        loss_5cm += int(base_5 and (not la_5))
        gain_2cm += int((not base_2) and la_2)
        loss_2cm += int(base_2 and (not la_2))

    te = np.asarray(translation_delta, dtype=np.float64)
    ae = np.asarray(rotation_delta, dtype=np.float64)
    if te.size == 0:
        return {
            "query_count": 0,
            "translation_delta_mean": 0.0,
            "translation_delta_median": 0.0,
            "rotation_delta_mean": 0.0,
            "rotation_delta_median": 0.0,
            "improved_translation_fraction": 0.0,
            "degraded_translation_fraction": 0.0,
            "recall_5cm_gain_count": 0,
            "recall_5cm_loss_count": 0,
            "recall_2cm_gain_count": 0,
            "recall_2cm_loss_count": 0,
            "translation_delta_bootstrap_ci95": [0.0, 0.0],
            "rotation_delta_bootstrap_ci95": [0.0, 0.0],
        }
    return {
        "query_count": int(te.size),
        "translation_delta_mean": float(te.mean()),
        "translation_delta_median": float(np.median(te)),
        "rotation_delta_mean": float(ae.mean()),
        "rotation_delta_median": float(np.median(ae)),
        "improved_translation_fraction": float((te < 0).mean()),
        "degraded_translation_fraction": float((te > 0).mean()),
        "recall_5cm_gain_count": int(gain_5cm),
        "recall_5cm_loss_count": int(loss_5cm),
        "recall_2cm_gain_count": int(gain_2cm),
        "recall_2cm_loss_count": int(loss_2cm),
        "translation_delta_bootstrap_ci95": _bootstrap_mean_ci(te, bootstrap_samples, seed),
        "rotation_delta_bootstrap_ci95": _bootstrap_mean_ci(ae, bootstrap_samples, seed + 1),
    }


def threshold_curve(results, thresholds):
    total = max(1, len(results))
    curve = []
    for threshold in thresholds:
        metrics = sparse_metric_summary(results)
        curve.append(
            {
                "threshold_px": float(threshold),
                "median_ae": metrics["median_ae"],
                "median_te": metrics["median_te"],
                "recall_5cm_5deg": metrics["recall_5cm_5deg"],
                "recall_2cm_2deg": metrics["recall_2cm_2deg"],
            }
        )
    return curve


def _normalized_auc(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.size == 0:
        return 0.0
    if xs.size == 1 or float(xs[-1] - xs[0]) == 0.0:
        return float(ys.mean())
    return float(np.trapz(ys, xs) / (xs[-1] - xs[0]))


def solver_threshold_sweep_summary(baseline_runs, la_runs, bootstrap_samples=1000, seed=0):
    thresholds = sorted({float(k) for k in baseline_runs} & {float(k) for k in la_runs})
    curve = []
    for offset, threshold in enumerate(thresholds):
        base = baseline_runs[threshold] if threshold in baseline_runs else baseline_runs[int(threshold)]
        la = la_runs[threshold] if threshold in la_runs else la_runs[int(threshold)]
        base_metrics = sparse_metric_summary(base)
        la_metrics = sparse_metric_summary(la)
        curve.append(
            {
                "threshold_px": float(threshold),
                "baseline": base_metrics,
                "la": la_metrics,
                "delta": {
                    key: float(la_metrics[key] - base_metrics[key])
                    for key in (
                        "median_ae",
                        "median_te",
                        "recall_5cm_5deg",
                        "recall_2cm_2deg",
                        "avg_inliers",
                    )
                },
                "paired": paired_sparse_summary(
                    base,
                    la,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + offset,
                ),
            }
        )

    auc = {}
    auc_delta = {}
    for key in ("median_ae", "median_te", "recall_5cm_5deg", "recall_2cm_2deg", "avg_inliers"):
        base_values = [item["baseline"][key] for item in curve]
        la_values = [item["la"][key] for item in curve]
        auc[f"baseline_{key}"] = _normalized_auc(thresholds, base_values)
        auc[f"la_{key}"] = _normalized_auc(thresholds, la_values)
        auc_delta[key] = auc[f"la_{key}"] - auc[f"baseline_{key}"]

    return {
        "thresholds": thresholds,
        "curve": curve,
        "auc": auc,
        "auc_delta": auc_delta,
    }
