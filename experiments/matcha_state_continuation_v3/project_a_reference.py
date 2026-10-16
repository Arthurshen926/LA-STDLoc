"""Strict raw A1..8 -> eight light records + full8 projection; stdlib only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

LIGHT_FIELDS = {"iteration", "pool", "index", "name", "cadence_at_selection",
                "loss_components", "stacks", "rng_digest", "events", "phase",
                "cadence_after_step", "lr", "loss", "points", "active_sh_degree"}
LOSS_FIELDS = {"Ll1", "dssim", "lambda_dssim", "loss", "normal_loss", "dist_loss",
               "depth_prior_loss", "normal_prior_loss", "curv_prior_loss",
               "total_regularization_loss", "total_loss", "depth_order_prior_loss",
               "anisotropy_loss", "large_splat_loss", "coefficients"}
GROUPS = {"xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"}

def require(ok, message):
    if not ok:
        raise RuntimeError("A8-projection: " + message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def is_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

def validate_scalar(value):
    require(isinstance(value, dict), "missing scalar bit record")
    if set(value) == {"dtype", "bits"}:
        sizes = {"torch.float16": 4, "torch.bfloat16": 4, "torch.float32": 8, "torch.float64": 16}
        bits = value["bits"]
        require(value["dtype"] in sizes and isinstance(bits, str) and
                len(bits) == sizes[value["dtype"]] and
                all(c in "0123456789abcdef" for c in bits), "scalar byte encoding")
    else:
        require(set(value) == {"dtype", "hex"} and value["dtype"] in ("int", "float") and
                isinstance(value["hex"], str), "scalar hex encoding")
        require(math.isfinite(float.fromhex(value["hex"])), "nonfinite scalar")

def validate_record(record):
    require(isinstance(record, dict) and type(record.get("iteration")) is int and
            1 <= record["iteration"] <= 8, "unexpected iteration")
    if "exact_digest" in record:
        require(set(record) == {"iteration", "exact_digest"} and
                is_digest(record["exact_digest"]), "full record fields")
        return (record["iteration"], True)
    require(set(record) == LIGHT_FIELDS, "missing or extra light component")
    chart = record["iteration"] == 5
    require(record["pool"] == ("chart" if chart else "dense"), "frozen first-eight native view pool")
    require(record["cadence_at_selection"] == (999999 if record["iteration"] == 1 else 5) and
            record["cadence_after_step"] == 5 and
            record["phase"] == (None if record["iteration"] < 5 else "early"), "native cadence/phase")
    require(is_digest(record["rng_digest"]), "missing RNG byte digest")
    require(set(record["loss_components"]) == LOSS_FIELDS, "missing or extra loss component")
    coefficients = {"lambda_normal", "lambda_dist", "lambda_anisotropy"}
    coefficients |= ({"lambda_prior_depth", "lambda_prior_depth_derivative", "lambda_prior_normal",
                      "lambda_prior_curvature", "regularization_factor", "lambda_depth_order"}
                     if chart else {"lambda_dense_depth"})
    require(set(record["loss_components"]["coefficients"]) == coefficients,
            "first-eight coefficient coverage")
    losses = record["loss_components"]
    inactive = {"large_splat_loss"} if chart else {"depth_order_prior_loss", "anisotropy_loss", "large_splat_loss"}
    require(all(losses[k] is None for k in inactive), "unexpected first-eight active loss")
    for key in LOSS_FIELDS - inactive - {"coefficients"}:
        validate_scalar(losses[key])
    for value in losses["coefficients"].values():
        validate_scalar(value)
    require(set(record["lr"]) == GROUPS and all(type(v) in (int, float) and
            math.isfinite(v) for v in record["lr"].values()), "LR coverage")
    require(type(record["index"]) is int and record["index"] >= 0 and
            isinstance(record["name"], str) and bool(record["name"]), "view identity")
    require(type(record["points"]) is int and record["points"] > 0 and
            record["active_sh_degree"] == 0, "first-eight topology/SH")
    require(type(record["loss"]) in (int, float) and math.isfinite(record["loss"]), "total loss")
    require(set(record["stacks"]) == {"viewpoint_idx_stack", "dense_viewpoint_idx_stack"},
            "sampling-stack coverage")
    for value in record["stacks"].values():
        require(value is None or (type(value) is list and
                all(type(i) is int and i >= 0 for i in value) and
                len(value) == len(set(value))), "ordered stack descriptor")
    require(set(record["events"]) == {"evaluation", "save", "native_checkpoint"} and
            all(v == [] for v in record["events"].values()), "event coverage")
    return (record["iteration"], False)

def validate_projected_records(records):
    expected = {(i, False) for i in range(1, 9)} | {(8, True)}
    require(set(records) == expected, "projection requires exactly eight light records plus full8")
    require(list(records) == [(i, False) for i in range(1, 9)] + [(8, True)], "projected record order")
    for key, value in records.items():
        require(validate_record(value) == key, "record key mismatch")

def project_records(raw):
    expected_order = [(i, full) for i in range(1, 9) for full in (False, True)]
    keys = [validate_record(r) for r in raw]
    require(keys == expected_order, "raw A records missing, extra, reordered or duplicated")
    result = [r for r in raw if "exact_digest" not in r or r["iteration"] == 8]
    validate_projected_records({validate_record(r): r for r in result})
    return result

def validate_review_links(review, source, launch_path):
    evidence = review.get("evidence_sha256", {})
    for path in (source, launch_path):
        require(evidence.get(str(path)) == sha(path), "independent receipt does not bind " + str(path))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=False)
    launch_path = args.a_root / "launcher_result.json"
    review_path = args.a_root / "independent_calibration_review.json"
    launch = json.loads(launch_path.read_text())
    review = json.loads(review_path.read_text())
    require(launch["verdict"] == "N8_COMPLETE_NOT_EQUIVALENCE" and
            launch["native_horizon"] == 30000 and launch["terminal"] == 8, "A is not complete")
    require(review["schema"] == "independent_matcha_n8_calibration_completion_v1" and
            review["verdict"] == "PASS" and review["complete"] and
            review["completed_step"] == 8 and review["training_horizon"] == 30000 and
            review["gpu_equivalence_passed"] is False, "independent A receipt missing")
    source = args.a_root / "state/trace_from_1.jsonl"
    source_sha = sha(source)
    validate_review_links(review, source, launch_path)
    raw = [json.loads(line) for line in source.read_text().splitlines()]
    projected = project_records(raw)
    require(projected[-1]["exact_digest"] == review["step8_full_digest"], "review endpoint differs")
    target = args.output_dir / "reference_A_light1_8_full8.jsonl"
    with target.open("x") as f:
        for record in projected:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    require([json.loads(line) for line in target.read_text().splitlines()] == projected,
            "projection readback differs")
    require(sha(source) == source_sha, "raw A trace changed during projection")
    receipt = {"schema": "matcha_native_a_projection_v2",
               "raw_source": str(source), "raw_source_sha256": source_sha,
               "raw_record_count": 16, "raw_A_preserved": True,
               "projection_path": str(target), "projection_sha256": sha(target),
               "projected_record_count": 9, "light_steps": [1, 8], "full_steps": [8],
               "calibration_review_path": str(review_path), "calibration_review_sha256": sha(review_path),
               "launcher_result_sha256": sha(launch_path), "projector_sha256": sha(__file__),
               "claim": "Exact eight per-step trajectories plus full step8 only; A full1..7 retained as historical evidence."}
    path = args.output_dir / "projection_receipt.json"
    with path.open("x") as f:
        json.dump(receipt, f, sort_keys=True, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({"receipt": str(path), "sha256": sha(path), "projection_sha256": sha(target)}))

if __name__ == "__main__":
    main()
