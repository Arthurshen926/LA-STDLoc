"""CPU-only comparison of STD-native 32-step boundary traces; no GPU admission."""
import argparse
import json
import math
from pathlib import Path

from std_exact import atomic_json, sha


def compare(base):
    base = Path(base)
    runs = [base / name for name in ("reference", "probe_a", "probe_b")]
    config = [json.loads((p / "config.json").read_text()) for p in runs]
    identity = config[0]["identity"]
    if any(c["identity"] != identity or not c["probe_only"] for c in config):
        raise ValueError("diagnostic identity mismatch")
    record = json.loads((runs[0] / "checkpoint_10000/complete.json").read_text())
    if record.get("complete") is not True or record["iteration"] != 10000:
        raise ValueError("reference checkpoint incomplete")
    evidence = []
    for p in runs:
        done = json.loads((p / "stage_complete.json").read_text())
        if done.get("status") != "complete" or not done.get("probe_only"):
            raise ValueError("diagnostic stage incomplete")
        for name in ("config.json", "stage_complete.json", "trace.json"):
            path = p / name
            evidence.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)})
    for p in runs[1:]:
        path = p / "restore_verified.json"
        restore = json.loads(path.read_text())
        if (restore.get("complete") is not True or restore["iteration"] != 10000
                or restore["state_digest"] != record["state_digest"]):
            raise ValueError("restore state digest differs from native boundary")
        evidence.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)})
    traces = [json.loads((p / "trace.json").read_text()) for p in runs]
    traces = [[row for row in trace if 10001 <= row["iteration"] <= 10032] for trace in traces]
    if any([row["iteration"] for row in t] != list(range(10001, 10033)) for t in traces):
        raise ValueError("must compare exactly 32 original native steps")
    rows = []
    envelopes = {key: max(abs(float(a[key]) - float(b[key])) for a, b in zip(traces[1], traces[2]))
                 for key in ("loss", "l1", "feature_l1")}
    within_observed = True
    for ref, a, b in zip(*traces):
        exact = all(ref[k] == a[k] == b[k] for k in ("iteration", "camera", "rng_digest", "optimizer_lr"))
        metrics = {}
        for key in ("loss", "l1", "feature_l1"):
            values = [float(row[key]) for row in (ref, a, b)]
            if not all(math.isfinite(v) for v in values):
                raise ValueError("nonfinite diagnostic metric")
            native_aa = abs(values[1] - values[2])
            tolerance = envelopes[key]
            error = max(abs(values[0] - values[1]), abs(values[0] - values[2]))
            metrics[key] = {"native_repeat_aa_abs": native_aa, "reference_probe_abs": error,
                            "observed_native_aa_envelope_abs": tolerance, "within_observed_native_repeatability": error <= tolerance}
        ok = exact and all(m["within_observed_native_repeatability"] for m in metrics.values())
        within_observed &= ok
        rows.append({"iteration": ref["iteration"], "sampling_rng_lr_identical": exact,
                     "metrics": metrics, "within_observed_native_repeatability": ok})
    return {"schema": "std_native_boundary_comparison_v1", "complete": True,
            "within_observed_native_repeatability": within_observed, "independent_review_still_required": True,
            "protocol_sha256": identity["protocol_sha256"],
            "build_sha256": identity["build_sha256"],
            "mapping_semantics": identity["mapping_semantics"],
            "boundary": 10000, "native_steps": 32, "state_restore_byte_exact": True,
            "observed_native_aa_envelopes_abs": envelopes,
            "tolerance_rule": "Report only the observed max absolute A/A difference across these 32 native steps; no arbitrary epsilon, multiplier or imported renderer tolerance. Independent reviewer must judge adequacy, and request more native A/A observations if needed.",
            "important_limit": "Two restored native replays estimate kernel repeatability; continuous reference is the original process after checkpoint save. No claim of full-trajectory bitwise GPU reproducibility.",
            "rows": rows, "evidence": evidence}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.base)
    atomic_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "evidence")}, indent=2))
