#!/usr/bin/env python
"""Write a provenance-aware validation ledger for LaFGS bootstrap studies.

The capacity/support-view matrix is meaningful only if its sparse deployment
contract is identical.  This helper therefore records both localization
metrics and the per-run protocol manifest, rather than silently merging legacy
artifacts with the current locked full-resolution native protocol.
"""

import argparse
import json
from pathlib import Path


FORMAL_PROTOCOL_ID = "lafgs_v2_fullres_native_uncapped_v1"


def _read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def _resolve_summary(run_root):
    run_root = Path(run_root)
    ref = run_root / "validation" / "results" / "bootstrap_validation.results_path"
    if not ref.is_file():
        raise FileNotFoundError(f"bootstrap validation reference is missing: {ref}")
    result_root = Path(ref.read_text().strip())
    summary = result_root / "results_summary.json"
    if not summary.is_file():
        raise FileNotFoundError(f"bootstrap validation summary is missing: {summary}")
    return summary


def _protocol(run_root):
    run_root = Path(run_root)
    manifests = sorted(run_root.glob("protocol_manifest*.json"))
    if not manifests:
        raise FileNotFoundError(f"protocol manifest is missing: {run_root}")
    return manifests[0], _read_json(manifests[0])


def _metric(payload, section, key):
    return payload.get(section, {}).get(key)


def _entry(label, run_root):
    run_root = Path(run_root).resolve()
    protocol_path, protocol = _protocol(run_root)
    summary_path = _resolve_summary(run_root)
    summary = _read_json(summary_path)
    bootstrap = protocol.get("bootstrap", {})
    formal = protocol.get("formal_protocol", {})
    native_matcher = protocol.get("native_matcher", {})
    diagnostics = summary.get("sparse_diagnostics", {})
    return {
        "label": label,
        "run_root": str(run_root),
        "protocol_manifest": str(protocol_path),
        "results_summary": str(summary_path),
        "formal_protocol_status": (
            "locked"
            if formal.get("id") == FORMAL_PROTOCOL_ID
            else "legacy_or_incomplete"
        ),
        "formal_protocol_id": formal.get("id"),
        "landmark_budget": bootstrap.get("landmark_budget"),
        "support_views": bootstrap.get("support_views"),
        "support_view_sampling": bootstrap.get("support_view_sampling"),
        "longest_edge": bootstrap.get("longest_edge"),
        "frontend": native_matcher.get("frontend"),
        "cosine_threshold": native_matcher.get("cosine_threshold"),
        "max_matches_per_landmark": native_matcher.get("max_matches_per_landmark"),
        "median_te_cm": _metric(summary, "sparse", "median_te"),
        "median_ae_deg": _metric(summary, "sparse", "median_ae"),
        "recall_5cm": _metric(summary, "sparse", "recall_5cm_5d"),
        "avg_inliers": _metric(summary, "sparse", "avg_inliers"),
        "raw_gt_precision_2px": diagnostics.get(
            "sparse_diag_all_gt_precision_2px_mean"
        ),
        "inlier_gt_precision_2px": diagnostics.get(
            "sparse_diag_inlier_gt_precision_2px_mean"
        ),
        "translation_pose_info_logdet": diagnostics.get(
            "sparse_diag_inlier_pose_info_translation_logdet_mean"
        ),
        "total_runtime_ms": diagnostics.get("sparse_diag_runtime_total_ms_mean"),
        "ransac_actual_hypotheses": diagnostics.get(
            "sparse_diag_ransac_actual_hypotheses_mean"
        ),
    }


def _format(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize validation-only LaFGS bootstrap capacity studies."
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        metavar="LABEL=RUN_ROOT",
        help="Repeat once for each U0--U4 bootstrap artifact.",
    )
    args = parser.parse_args()

    entries = []
    for raw in args.entry:
        if "=" not in raw:
            raise ValueError(f"entry must be LABEL=RUN_ROOT, got {raw!r}")
        label, path = raw.split("=", 1)
        if not label or not path:
            raise ValueError(f"entry must be LABEL=RUN_ROOT, got {raw!r}")
        entries.append(_entry(label, path))

    payload = {
        "schema_version": 1,
        "purpose": "validation_only_lafgs_bootstrap_capacity_sampling_matrix",
        "test_metrics_used": False,
        "formal_protocol_id": FORMAL_PROTOCOL_ID,
        "entries": entries,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# LaFGS Bootstrap Capacity/Sampling Matrix",
        "",
        "Validation only. Rows without the locked contract are retained as historical context, not as formal U0--U4 evidence.",
        "",
        "| ID | Contract | Bank | Support views | Sampling | TE (cm) | R@5cm | Raw P@2 | Inlier P@2 | Pose-info | Total ms | RANSAC hypotheses |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in entries:
        lines.append(
            "| {label} | {contract} | {budget} | {views} | {sampling} | {te} | {r5} | {raw} | {inlier} | {pose_info} | {runtime} | {hypotheses} |".format(
                label=item["label"],
                contract=item["formal_protocol_status"],
                budget=_format(item["landmark_budget"]),
                views=_format(item["support_views"]),
                sampling=_format(item["support_view_sampling"]),
                te=_format(item["median_te_cm"]),
                r5=_format(item["recall_5cm"]),
                raw=_format(item["raw_gt_precision_2px"]),
                inlier=_format(item["inlier_gt_precision_2px"]),
                pose_info=_format(item["translation_pose_info_logdet"]),
                runtime=_format(item["total_runtime_ms"]),
                hypotheses=_format(item["ransac_actual_hypotheses"]),
            )
        )
    output_markdown = Path(args.output_markdown)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text("\n".join(lines) + "\n")
    print(output_json)
    print(output_markdown)


if __name__ == "__main__":
    main()
