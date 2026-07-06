import math


def safe_float(value, default=None):
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def median_or_default(values, default):
    values = sorted(float(value) for value in values if safe_float(value) is not None)
    if not values:
        return float(default)
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return float(0.5 * (values[mid - 1] + values[mid]))


def teacher_item_final_te(item):
    if not item:
        return None
    sparse_te = safe_float(item.get("te"))
    dense_te = safe_float(item.get("dense_te"))
    values = [value for value in (sparse_te, dense_te) if value is not None]
    return min(values) if values else None


def teacher_item_support_fraction(item):
    if not item:
        return None
    sparse_mask = item.get("sparse_valid_mask")
    if isinstance(sparse_mask, dict):
        for key in ("support_frac", "valid_frac", "largest_component_frac"):
            value = safe_float(sparse_mask.get(key))
            if value is not None:
                return max(0.0, min(1.0, value))
    value = safe_float(item.get("sparse_valid_mask_valid_frac"))
    if value is not None:
        return max(0.0, min(1.0, value))
    return None


def pseudo_teacher_cache_reliability_stats(cache):
    items = getattr(cache, "items", None)
    if items is None and isinstance(cache, dict):
        items = cache
    items = items or {}
    grouped = {"__global__": {"te": [], "inliers": []}}
    for item in items.values():
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "") or "__unknown__")
        grouped.setdefault(source, {"te": [], "inliers": []})
        final_te = teacher_item_final_te(item)
        if final_te is not None:
            grouped["__global__"]["te"].append(final_te)
            grouped[source]["te"].append(final_te)
        inliers = safe_float(item.get("inliers"))
        if inliers is not None and inliers > 0:
            grouped["__global__"]["inliers"].append(inliers)
            grouped[source]["inliers"].append(inliers)

    stats = {}
    global_te = median_or_default(grouped["__global__"]["te"], 1.0)
    global_inliers = median_or_default(grouped["__global__"]["inliers"], 1.0)
    stats["__global__"] = {
        "median_final_te": max(global_te, 1e-6),
        "median_inliers": max(global_inliers, 1.0),
    }
    for source, values in grouped.items():
        if source == "__global__":
            continue
        stats[source] = {
            "median_final_te": max(median_or_default(values["te"], global_te), 1e-6),
            "median_inliers": max(median_or_default(values["inliers"], global_inliers), 1.0),
        }
    return stats


def pseudo_query_stage_weight(stage, args):
    stage = str(stage or "unknown")
    if stage == "teacher_ok":
        return float(args.pseudo_query_reliability_teacher_ok_weight)
    if stage == "dense_improves_sparse":
        return float(args.pseudo_query_reliability_dense_improves_weight)
    if stage == "mixed_or_uncertain":
        return float(args.pseudo_query_reliability_mixed_weight)
    if stage == "dense_rescues_sparse":
        return float(args.pseudo_query_reliability_dense_rescues_weight)
    if stage == "sparse_failure":
        return float(args.pseudo_query_reliability_sparse_failure_weight)
    if stage == "dense_regression_after_good_sparse":
        return float(args.pseudo_query_reliability_dense_regression_weight)
    return float(args.pseudo_query_reliability_unknown_weight)


def pseudo_query_reliability_decision(record, item, stats, args):
    mode = str(getattr(args, "pseudo_query_reliability_mode", "none") or "none").lower()
    if mode == "none" or not item:
        return {
            "enabled": False,
            "reason": "disabled" if mode == "none" else "missing_cache_item",
            "weight": 1.0,
            "stage_weight": 1.0,
            "error_weight": 1.0,
            "inlier_weight": 1.0,
            "support_weight": 1.0,
            "update_memory": True,
            "update_stats": True,
        }
    source = str(getattr(record, "source", "") or item.get("source", ""))
    source_stats = (stats or {}).get(source) or (stats or {}).get("__global__", {})
    global_stats = (stats or {}).get("__global__", {})
    median_te = float(source_stats.get("median_final_te", global_stats.get("median_final_te", 1.0)))
    median_inliers = float(source_stats.get("median_inliers", global_stats.get("median_inliers", 1.0)))
    stage = str(item.get("failure_stage", "unknown"))
    stage_weight = max(0.0, min(1.0, pseudo_query_stage_weight(stage, args)))

    final_te = teacher_item_final_te(item)
    error_weight = 1.0
    error_scale = max(float(getattr(args, "pseudo_query_reliability_error_scale", 2.0)), 1e-6)
    if final_te is not None and median_te > 0:
        ratio = float(final_te) / max(median_te * error_scale, 1e-6)
        error_weight = 1.0 / (1.0 + max(0.0, ratio - 1.0))

    inliers = safe_float(item.get("inliers"))
    inlier_weight = 1.0
    if inliers is not None and median_inliers > 0:
        ratio = max(0.0, min(1.0, float(inliers) / median_inliers))
        power = max(0.0, float(getattr(args, "pseudo_query_reliability_inlier_power", 0.5)))
        inlier_weight = ratio ** power if power > 0 else 1.0

    support_weight = 1.0
    support_fraction = teacher_item_support_fraction(item)
    if source == "synthetic_rgb" and support_fraction is not None:
        support_weight = max(0.0, min(1.0, support_fraction))

    raw = stage_weight * error_weight * inlier_weight * support_weight
    floor = max(0.0, min(1.0, float(getattr(args, "pseudo_query_reliability_min_weight", 0.0))))
    if source == "train_rgb":
        floor = max(floor, max(0.0, min(1.0, float(getattr(args, "pseudo_query_reliability_real_min_weight", 0.0)))))
    elif source == "synthetic_rgb":
        floor = max(
            floor,
            max(0.0, min(1.0, float(getattr(args, "pseudo_query_reliability_synthetic_min_weight", 0.0)))),
        )
    weight = max(floor, min(1.0, raw))
    memory_threshold = max(0.0, min(1.0, float(getattr(args, "pseudo_query_reliability_memory_min_weight", 0.75))))
    stats_threshold_raw = getattr(args, "pseudo_query_reliability_stats_min_weight", None)
    if stats_threshold_raw is None:
        stats_threshold_raw = memory_threshold
    stats_threshold = max(
        0.0,
        min(
            1.0,
            float(stats_threshold_raw),
        ),
    )
    update_memory = weight >= memory_threshold
    update_stats = weight >= stats_threshold
    return {
        "enabled": True,
        "reason": "ok",
        "weight": float(weight),
        "stage": stage,
        "stage_weight": float(stage_weight),
        "error_weight": float(error_weight),
        "inlier_weight": float(inlier_weight),
        "support_weight": float(support_weight),
        "final_te": None if final_te is None else float(final_te),
        "median_final_te": float(median_te),
        "inliers": None if inliers is None else float(inliers),
        "median_inliers": float(median_inliers),
        "update_memory": bool(update_memory),
        "update_stats": bool(update_stats),
    }


__all__ = [
    "pseudo_query_reliability_decision",
    "pseudo_teacher_cache_reliability_stats",
    "pseudo_query_stage_weight",
    "teacher_item_final_te",
    "teacher_item_support_fraction",
]
