"""Aggregate all twenty accepted full-query native baseline evaluations.

Writes one new report. Old scores and telemetry are never promoted to a formal
benchmark. Primary values are equal-scene averages of each scene's statistic.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.native_baseline_repair.query_plan import (
    ROOT, SCENES, METHODS, PROTOCOL_NAME, ref, verify, validate_contract, write_new)
from experiments.native_baseline_repair import query_results as results_module
from experiments.native_baseline_repair.query_results import (
    errors, metrics, verify_identity, query_order, verify_terminal)

RESULT_VERIFIER_SHA = "e8af12e3970a5920d5dbf7ed6e17718e26d452e4945efa5301cff28bb2242861"
STAT_KEYS = ("median_te_cm", "median_re_deg", "r5_percent", "r15_percent", "p90_te_cm")


def summarize(cells):
    """Require all five predefined scenes; no favorable scene or failed-query omission."""
    if len(cells) != len(SCENES) or {x["scene"] for x in cells} != set(SCENES):
        raise ValueError("exact five-scene coverage required")
    if len({(x["method"], x["mapping_semantics"]) for x in cells}) != 1:
        raise ValueError("mixed method or mapping semantic condition")
    result = {}
    for stage in ("sparse", "dense"):
        stats = [x["metrics"][stage] for x in cells]
        result[stage] = {
            "equal_scene": {k: float(np.mean([s[k] for s in stats])) for k in STAT_KEYS},
            "pooled": metrics(
                [v for x in cells for v in x["errors"][stage]["te_cm"]],
                [v for x in cells for v in x["errors"][stage]["re_deg"]]),
            "query_count": sum(s["query_count"] for s in stats),
            "gt_1m_count": sum(s["gt_1m_count"] for s in stats),
        }
    sizes = [x["deployment_bytes"] for x in cells]
    result["storage"] = {
        "scope": "necessary scene-specific native inference files; actual serialized bytes",
        "total_bytes": sum(sizes), "mean_bytes": float(np.mean(sizes)),
        "total_decimal_MB": sum(sizes) / 1e6, "mean_decimal_MB": float(np.mean(sizes)) / 1e6,
        "excluded": ["training checkpoints and optimizer", "training logs and images",
                     "source SfM", "global SuperPoint weights", "query RGB and masks",
                     "segmentation network"],
    }
    return result


def check_cell_bindings(v, c, path, verifier_ref):
    if path.name != c["case"]:
        raise ValueError("cell directory and contract identity differ")
    if verifier_ref["sha256"] != RESULT_VERIFIER_SHA or v.get("verifier") != verifier_ref:
        raise ValueError("actual imported metric verifier differs from approved code")
    paths = {
        "contract": path / "contract.json",
        "terminal_evidence": path / "terminal.json",
        "native_receipt": Path(c["receipt"]),
        "results": Path(c["native_output"]) / "results.json",
        "native_log": path / "launch/native.log",
        "launch": path / "launch/launch.json",
    }
    for key, expected in paths.items():
        if Path(v[key]["path"]) != expected:
            raise ValueError("cross-cell reference: " + key)


def check_receipt(receipt, c, query_count):
    for key, expected in dict(status="complete", method=c["method"], phase="evaluate",
            mapping_semantics=c["mapping_semantics"], query_semantics=True,
            seed=2026, warmup_queries=0, native_output=c["native_output"]).items():
        if receipt.get(key) != expected:
            raise ValueError("native receipt condition mismatch: " + key)
    if len(receipt.get("timings", [])) != query_count:
        raise ValueError("native timing/query count mismatch")


def actual_deployment_bytes(assets):
    actual = [ref(a["path"]) for a in assets.values()]
    if actual != list(assets.values()):
        raise ValueError("deployment asset path/hash/bytes mismatch")
    return sum(a["bytes"] for a in actual)


def load_cell(path, protocol, protocol_path):
    contract_path = path / "contract.json"
    c = json.loads(contract_path.read_text())
    validate_contract(c, protocol, protocol_path)
    verify(c["protocol"])
    vpath = path / "verification.json"
    v = json.loads(vpath.read_text())
    if v.get("complete") is not True or v.get("case") != c["case"]:
        raise ValueError("cell not accepted")
    check_cell_bindings(v, c, path, ref(results_module.__file__))
    for key in ("contract", "terminal_evidence", "native_receipt", "results",
                "native_log", "launch", "verifier"):
        verify(v[key])
    if v["contract"] != ref(contract_path):
        raise ValueError("verification detached from exact contract")
    receipt = json.loads(Path(v["native_receipt"]["path"]).read_text())
    check_receipt(receipt, c, len(c["query_names"]))
    verify_terminal(json.loads(Path(v["terminal_evidence"]["path"]).read_text()),
                    json.loads(Path(v["launch"]["path"]).read_text()),
                    ref(contract_path)["sha256"], c["case"])
    verify(c["query_snapshot"])
    snap = json.loads(Path(c["query_snapshot"]["path"]).read_text())
    rows = json.loads(Path(v["results"]["path"]).read_text())
    names = query_order(Path(v["native_log"]["path"]).read_text(errors="replace"))
    gt = verify_identity(names, rows, c["query_names"], snap)
    out = {"case": c["case"], "method": c["method"], "scene": c["scene"],
           "mapping_semantics": c["mapping_semantics"], "metrics": {}, "errors": {},
           "verification": ref(vpath), "contract": ref(contract_path),
           "results": v["results"], "query_names": names, "model_assets": c["model_assets"]}
    for stage in ("sparse", "dense"):
        poses = [x["sparse"]["pose_w2c"] if stage == "sparse"
                 else x["dense"][-1]["pose_w2c"] for x in rows]
        pairs = [errors(pose, gt[name]) for name, pose in zip(names, poses)]
        te = [p[0] for p in pairs]; re_deg = [p[1] for p in pairs]
        out["metrics"][stage] = metrics(te, re_deg)
        if out["metrics"][stage] != v["metrics"][stage]:
            raise ValueError("cell's recomputed metrics changed")
        out["errors"][stage] = {"te_cm": te, "re_deg": re_deg}
    out["deployment_bytes"] = actual_deployment_bytes(c["model_assets"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()
    qroot = args.root / "query_semantic_on"
    ppath = qroot / PROTOCOL_NAME
    p = json.loads(ppath.read_text())
    review_path = qroot / "independent_review.json"
    review = json.loads(review_path.read_text())
    # Require structured acceptance, not a PASS string in arbitrary free text.
    if (review.get("complete") is not True or review.get("verdict") != "PASS"
            or review.get("protocol") != ref(ppath)
            or review.get("protocol_sha256") != ref(ppath)["sha256"]
            or review.get("query_semantics") is not True or review.get("seed") != 2026
            or review.get("total_cell_query_evaluations") != 7672):
        raise ValueError("independent protocol review binding missing")
    for name, digest in p["code_sha256"].items():
        if ref(name)["sha256"] != digest:
            raise ValueError("frozen code drift: " + name)
    expected = {f"{m}_{s}_mapping_semantics_{z}_eval"
                for s in SCENES for m in METHODS for z in ("off", "on")}
    if set(p["cells"]) != expected:
        raise ValueError("protocol is not the frozen twenty-cell matrix")
    cells = [load_cell(qroot / "cells" / k, p, ppath) for k in sorted(expected)]
    groups = {}
    for method in METHODS:
        for semantic in ("off", "on"):
            selected = [c for c in cells if c["method"] == method
                        and c["mapping_semantics"] == semantic]
            groups[f"{method}_mapping_semantics_{semantic}"] = summarize(selected)
    report = {
        "complete": True, "protocol": ref(ppath), "independent_preflight": ref(review_path),
        "report_generator": ref(__file__), "cell_count": len(cells),
        "query_count_per_condition": sum(len(p["scenes"][s]["query_names"]) for s in SCENES),
        "query_semantics": True, "seed": 2026, "scenes": list(SCENES),
        "primary_aggregation": "five scenes equally weighted; mean of per-scene medians for TE/RE, mean per-scene recall and P90",
        "primary_native_output": "final dense refinement; sparse/coarse separately retained",
        "method_scope": {
            "STDLoc": "controlled external query RGB masking; native feature model, solver and refinement",
            "ULFLoc": "native query semantic-mask path; native feature model, solver and refinement"},
        "initialization": "user-selected native SfM initialization; test/mixed-track geometry retained and disclosed",
        "data_version": "four original scenes plus q-normalized, retrained StMarysChurch",
        "timing": "accuracy-only unwarmed concurrent telemetry, not a formal benchmark",
        "groups": groups, "cells": cells,
    }
    output = qroot / "report.json"
    write_new(output, report)
    print(json.dumps({"report": ref(output), "groups": groups}))


if __name__ == "__main__":
    main()
