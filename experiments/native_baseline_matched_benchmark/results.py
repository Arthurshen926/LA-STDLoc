"""Read-only aggregation requires exact90 finite successful jobs and fresh records."""
import argparse
import math
from pathlib import Path
from .common import *
from .plan import protocol, validate_round

def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] * (hi - pos) + values[hi] * (pos - lo) if hi != lo else values[lo]

def stats(rows):
    keys = sorted(set().union(*(row["stages"] for row in rows)))
    out = {}
    for key in keys:
        values = [row["stages"].get(key) for row in rows]
        if all(value is None for value in values):
            out[key] = None
            continue
        if not all(finite_ms(value) for value in values):
            raise ValueError("mixed unavailable/invalid timing boundary")
        out[key] = dict(count=len(values), mean=sum(values)/len(values),
                        p50=percentile(values,50), p90=percentile(values,90), p95=percentile(values,95))
    return out

def verify_terminal(evidence, contract_ref, case, round_id, result_ref, launch_ref):
    if not (evidence.get("terminal") is True and evidence.get("exit_code") == 0
            and evidence.get("evidence_source") == "WebCodex observe_jobs"
            and str(evidence.get("job_id", "")).startswith("wc_job_")
            and evidence.get("contract_sha256") == contract_ref["sha256"]
            and evidence.get("case") == case and evidence.get("round") == round_id):
        raise ValueError("finite Runner terminal evidence absent/mismatched")
    if evidence.get("result") != result_ref or evidence.get("launch") != launch_ref:
        raise ValueError("terminal result/launch canonical paths or bytes changed")
    verify(result_ref)
    verify(launch_ref)

def canonical_ref(path):
    path = Path(path)
    value = ref(path)
    if value["path"] != str(path):
        raise ValueError("result/launch cannot be redirected through a symlink")
    return value

def verify_contexts(value, c, contract_ref):
    if value.get("monitor_closed_clean") is not True:
        raise ValueError("monitor did not finish cleanly")
    expected = dict(case=c["case"], round=c["round"], contract=contract_ref,
                    visible_device_count=1, logical_device=0, actual_context_uuid=value["gpu_uuid"])
    if value.get("context_proofs") != dict(initialization=expected, completion=expected):
        raise ValueError("round actual CUDA context evidence changed")

def check_accuracy_evidence(value, c, terminal):
    """Checks evidence for every measured query, including failures; never filters timing."""
    cell = c["cell"]
    if value.get("accuracy_reference") != cell["accuracy_reference"]:
        raise ValueError("full-query accuracy reference changed")
    verify(cell["accuracy_reference"])
    reference = read(cell["accuracy_reference"]["path"])
    current = value["queries"]
    if type(value.get("process_peak_rss_kib")) is not int or value["process_peak_rss_kib"] < 0:
        raise ValueError("process-lifetime peak RSS invalid")
    pose_delta = []
    if cell["family"] == "any":
        by_name = {row["image_name"]: row for row in reference["queries"]}
        changed = 0
        failures = 0
        reference_failures = 0
        for row in current:
            evidence = row["localization"]
            old = by_name[row["name"]]
            changed += evidence["success"] != old["success"]
            failures += not evidence["success"]
            reference_failures += not old["success"]
            validate_pose(old["pose_w2c"])
            pose_delta.extend(abs(a-b) for x,y in zip(evidence["pose_w2c"],old["pose_w2c"]) for a,b in zip(x,y))
        return dict(measured_query_count=100, reported_failures=failures, reference_failures=reference_failures,
                    changed_success_count=changed, max_abs_pose_component_delta=max(pose_delta),
                    timing_queries_filtered=0)
    from .plan import native_contract
    verify(cell["accuracy_contract"])
    native = native_contract(c)
    canonical = dict(native_receipt=Path(native["receipt"]), native_results=Path(native["native_output"]) / "results.json")
    for key, path in canonical.items():
        if value.get(key) != canonical_ref(path) or terminal.get(key) != canonical_ref(path):
            raise ValueError("native output canonical path or bytes not terminal-bound")
    receipt = read(native["receipt"])
    for key, expected in dict(status="complete", method=cell["method"], phase="evaluate",
                              mapping_semantics=cell["mapping_semantics"], query_semantics=True,
                              seed=2026, warmup_queries=0, native_output=native["native_output"]).items():
        if receipt.get(key) != expected:
            raise ValueError("new native receipt differs: " + key)
    if len(receipt["timings"]) != 150:
        raise ValueError("native warmup+measured core count differs")
    rows = read(canonical["native_results"])
    if len(rows) != 100:
        raise ValueError("native measured result count differs")
    original = read(cell["accuracy_contract"]["path"])
    verify(cell["accuracy_results"])
    oldrows = read(cell["accuracy_results"]["path"])
    if len(oldrows) != len(original["query_names"]):
        raise ValueError("native full-query accuracy count differs")
    by_name = dict(zip(original["query_names"], oldrows))
    snapshot = read(original["query_snapshot"]["path"])
    gt = {q["name"]: q["pose_w2c"] for q in snapshot["queries"]}
    for index, (record, row) in enumerate(zip(current, rows)):
        evidence = native_evidence(row)
        if evidence != record["localization"] or row["gt_pose_w2c"] != gt[record["name"]]:
            raise ValueError("native per-query result/GT identity differs")
        old = native_evidence(by_name[record["name"]])
        for stage in ("sparse_pose_w2c", "dense_pose_w2c"):
            pose_delta.extend(abs(a-b) for x,y in zip(evidence[stage],old[stage]) for a,b in zip(x,y))
        timing = receipt["timings"][50+index]
        if record["stages"]["core_ms"] != timing["total_ms"]:
            raise ValueError("native per-query timing differs from receipt")
    return dict(measured_query_count=100, reported_failures=None,
                failure_note="native output has no explicit success flag; sparse/dense inlier evidence retained per query",
                max_abs_pose_component_delta=max(pose_delta), timing_queries_filtered=0)

def aggregate(root=PREPARATION):
    root = Path(root)
    p = protocol(root)
    assets = read(root / "assets.json")
    cells = {}
    jobs = set()
    for case, cell in assets["cells"].items():
        rounds = []
        records = []
        for round_id in ROUNDS:
            cp = root / "contracts" / case / ("round%d.json" % round_id)
            c = read(cp)
            validate_round(c, root)
            output = Path(c["output"])
            row = read(output / "round.json")
            validate_result(row, c["selection"], round_id, c["cell"]["family"])
            if row["contract"] != ref(cp) or row["case"] != case:
                raise ValueError("round records detached from contract")
            terminal = read(output / "terminal.json")
            verify_terminal(terminal, ref(cp), case, round_id, canonical_ref(output / "round.json"), canonical_ref(output / "launch.json"))
            verify_contexts(row, c, ref(cp))
            if terminal["job_id"] in jobs:
                raise ValueError("job reused across rounds")
            jobs.add(terminal["job_id"])
            launch = read(output / "launch.json")
            verify(row["lease"])
            if row["lease"] != launch["lease_ref"] or launch["contract_ref"] != ref(cp):
                raise ValueError("launch/lease identity changed")
            lease = read(row["lease"]["path"])
            if row["gpu_uuid"] != launch["gpu_uuid"] or row["gpu_uuid"] != lease["gpu_uuid"] or lease["contract_sha256"] != ref(cp)["sha256"]:
                raise ValueError("actual context evidence detached from released UUID")
            comparison = check_accuracy_evidence(row, c, terminal)
            rounds.append(dict(round=round_id, timing_ms=stats(row["queries"]), result=ref(output / "round.json"),
                               terminal=ref(output / "terminal.json"), accuracy_comparison=comparison))
            records.extend(row["queries"])
        cells[case] = dict(cell=cell, rounds=rounds, pooled_timing_ms=stats(records))
    if len(cells) != 30 or len(jobs) != 90:
        raise ValueError("requires30 cells and90 distinct completed cell-round jobs")
    groups = {}
    for value in cells.values():
        cell = value["cell"]
        key = cell["method"] + "_" + cell.get("variant", cell.get("mapping_semantics", ""))
        groups.setdefault(key, []).append(value["pooled_timing_ms"])
    equal = {}
    for key, group in groups.items():
        if len(group) != 5:
            raise ValueError("scene-equal summary needs all5 scenes")
        equal[key] = {stage: None if group[0][stage] is None else
                      dict(scene_count=5, mean_of_scene_means=sum(g[stage]["mean"] for g in group)/5)
                      for stage in group[0]}
    value = dict(complete=True, protocol=ref(root / "protocol.json"), assets=ref(root / "assets.json"),
                 cells=cells, scene_equal_ms=equal, warmup_included=False, accuracy_table_replaced=False,
                 warning=p["design"]["segmentation"])
    write_new(root / "report.json", value)
    return ref(root / "report.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PREPARATION)
    print(__import__("json").dumps(aggregate(parser.parse_args().root)))
