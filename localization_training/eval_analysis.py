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
