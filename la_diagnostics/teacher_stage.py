import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


def load_results(path: Path) -> List[Mapping[str, object]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of per-query results in {path}.")
    return data


def load_sparse_diagnostics(path: Optional[Path]) -> Dict[str, Mapping[str, object]]:
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("images", data if isinstance(data, list) else [])
    if not isinstance(rows, list):
        return {}
    return {str(row.get("image_name", "")): row for row in rows if row.get("image_name")}


def selected_image_names_from_sample_flow(path: Path, groups: Sequence[str], limit: int = 0) -> List[str]:
    wanted = set(groups)
    names = []
    seen = set()
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if wanted and row.get("group") not in wanted:
                continue
            name = row.get("image_name", "")
            if not name or name in seen:
                continue
            names.append(name)
            seen.add(name)
            if limit and len(names) >= int(limit):
                break
    return names


def build_teacher_stage_records(
    results: Sequence[Mapping[str, object]],
    sparse_diagnostics: Optional[Mapping[str, Mapping[str, object]]] = None,
    sparse_bad_te: float = 20.0,
    good_te: float = 5.0,
    dense_worse_margin: float = 5.0,
    sparse_correct_rate_bad: float = 0.05,
    min_sparse_inliers: int = 16,
) -> List[Dict[str, object]]:
    sparse_diagnostics = sparse_diagnostics or {}
    records = []
    for row in results:
        image_name = str(row.get("image_name", ""))
        if not image_name:
            continue
        diag = sparse_diagnostics.get(image_name, {})
        record = {
            "image_name": image_name,
            "sparse_te": _float(row.get("sparse_TE")),
            "sparse_ae": _float(row.get("sparse_AE")),
            "dense_te": _float(row.get("dense_TE")),
            "dense_ae": _float(row.get("dense_AE")),
            "sparse_inliers": _nested_inliers(row.get("sparse")),
            "dense_inliers": _dense_final_inliers(row),
            "matches": _float(diag.get("matches")),
            "correct_matches": _float(diag.get("correct_matches")),
            "diagnostic_inliers": _float(diag.get("inliers")),
        }
        matches = record["matches"]
        correct = record["correct_matches"]
        record["sparse_correct_rate"] = (
            float(correct) / float(matches) if matches is not None and matches > 0 and correct is not None else None
        )
        record["dense_delta_te"] = (
            record["dense_te"] - record["sparse_te"]
            if record["dense_te"] is not None and record["sparse_te"] is not None
            else None
        )
        record["failure_stage"] = classify_teacher_stage(
            sparse_te=record["sparse_te"],
            dense_te=record["dense_te"],
            sparse_inliers=record["sparse_inliers"],
            sparse_correct_rate=record["sparse_correct_rate"],
            sparse_bad_te=sparse_bad_te,
            good_te=good_te,
            dense_worse_margin=dense_worse_margin,
            sparse_correct_rate_bad=sparse_correct_rate_bad,
            min_sparse_inliers=min_sparse_inliers,
        )
        records.append(record)
    return records


def classify_teacher_stage(
    sparse_te: Optional[float],
    dense_te: Optional[float],
    sparse_inliers: Optional[float] = None,
    sparse_correct_rate: Optional[float] = None,
    sparse_bad_te: float = 20.0,
    good_te: float = 5.0,
    dense_worse_margin: float = 5.0,
    sparse_correct_rate_bad: float = 0.05,
    min_sparse_inliers: int = 16,
) -> str:
    sparse_bad = False
    if sparse_te is None:
        sparse_bad = True
    elif sparse_te >= float(sparse_bad_te):
        sparse_bad = True
    if sparse_inliers is not None and sparse_inliers < int(min_sparse_inliers):
        sparse_bad = True
    if sparse_correct_rate is not None and sparse_correct_rate <= float(sparse_correct_rate_bad):
        sparse_bad = True

    if sparse_bad:
        if dense_te is not None and sparse_te is not None and sparse_te - dense_te >= float(dense_worse_margin):
            return "dense_rescues_sparse"
        return "sparse_failure"

    if dense_te is None:
        return "sparse_only_no_dense"
    dense_delta = dense_te - sparse_te
    if sparse_te <= float(good_te) and dense_te <= float(good_te):
        return "teacher_ok"
    if dense_delta >= float(dense_worse_margin):
        return "dense_regression_after_good_sparse"
    if dense_delta <= -float(dense_worse_margin):
        return "dense_improves_sparse"
    return "mixed_or_uncertain"


def summarize_stage_records(records: Iterable[Mapping[str, object]]) -> Dict[str, object]:
    records = list(records)
    counts: Dict[str, int] = {}
    for row in records:
        stage = str(row.get("failure_stage", "unknown"))
        counts[stage] = counts.get(stage, 0) + 1
    return {
        "count": len(records),
        "failure_stage_counts": counts,
        "median_sparse_te": _median([row.get("sparse_te") for row in records]),
        "median_dense_te": _median([row.get("dense_te") for row in records]),
        "median_dense_delta_te": _median([row.get("dense_delta_te") for row in records]),
    }


def write_stage_csv(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    fields = [
        "image_name",
        "failure_stage",
        "sparse_te",
        "sparse_ae",
        "dense_te",
        "dense_ae",
        "dense_delta_te",
        "sparse_inliers",
        "dense_inliers",
        "matches",
        "correct_matches",
        "sparse_correct_rate",
        "diagnostic_inliers",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _nested_inliers(value: object) -> Optional[float]:
    if isinstance(value, Mapping):
        return _float(value.get("inliers"))
    return None


def _dense_final_inliers(row: Mapping[str, object]) -> Optional[float]:
    dense = row.get("dense")
    if isinstance(dense, list) and dense:
        return _nested_inliers(dense[-1])
    return _nested_inliers(row.get("sparse"))


def _float(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _median(values: Iterable[object]) -> Optional[float]:
    nums = sorted(_float(value) for value in values if _float(value) is not None)
    if not nums:
        return None
    mid = len(nums) // 2
    if len(nums) % 2:
        return nums[mid]
    return 0.5 * (nums[mid - 1] + nums[mid])


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.9g}"
    return value
