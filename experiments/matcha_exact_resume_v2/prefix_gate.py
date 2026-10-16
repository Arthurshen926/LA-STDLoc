"""Fresh B8 exact A/A and measured-budget gate, before any step9."""
import json
import math
import os
from pathlib import Path

POLICY = {"native_horizon": 30000, "comparison_end": 8, "conditional_training_end": 15101,
          "save_steps": [8, 14999, 15101], "full_steps": [8, 14999, 15101],
          "historical_prefix_seconds": 2186.0, "historical_slowdown_factor": 1.10,
          "future_full_count": 2, "future_capture_publication_factor": 1.25,
          "tail_and_light_margin_seconds": 120.0, "admission_ceiling_seconds": 3380.0,
          "total_deadline_seconds": 3500.0}
def require(ok, message):
    if not ok:
        raise RuntimeError("B-prefix-gate: " + message)

def forecast(elapsed, capture, publication, policy):
    require(policy == POLICY, "unreviewed forecast policy")
    require(all(type(v) in (int, float) and math.isfinite(v) and v > 0
                for v in (elapsed, capture, publication)), "missing actual timing")
    require(elapsed >= capture + publication, "inconsistent measured elapsed")
    value = (elapsed + policy["historical_prefix_seconds"] * policy["historical_slowdown_factor"] +
             policy["future_full_count"] * policy["future_capture_publication_factor"] *
             (capture + publication) + policy["tail_and_light_margin_seconds"])
    return {"predicted_total_seconds": value, "elapsed_through_B8_seconds": elapsed,
            "measured_B8_capture_hash_seconds": capture,
            "measured_B8_publication_seconds": publication,
            "verdict": "PASS" if value < policy["admission_ceiling_seconds"] else "STOP_AT_8_BUDGET",
            "policy": policy}

def write_once(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.flush()
        os.fsync(f.fileno())

class PrefixGate:
    def __init__(self, reference, policy):
        from project_a_reference import validate_projected_records
        validate_projected_records(reference)
        require(policy == POLICY, "unregistered policy")
        self.reference, self.policy = reference, policy
        self.completed = []
        self.passed = False

    def observe(self, record, full=None):
        k = record["iteration"]
        require(k == len(self.completed) + 1 and k <= 8, "missing/repeated A/A step")
        require(record == self.reference[(k, False)], "first A/A trajectory difference")
        if k == 8:
            require(full == self.reference[(8, True)]["exact_digest"], "missing/different A/A endpoint")
        self.completed.append(k)

    def decide(self, elapsed, capture, publication, path, proof):
        require(self.completed == list(range(1, 9)), "incomplete A/A comparison")
        value = forecast(elapsed, capture, publication, self.policy)
        value.update(schema="matcha_B8_conditional_prefix_gate_v1",
                     comparison_verdict="PASS", full_state_steps=[8], light_steps=[1, 8],
                     zero_tolerance=True, proof=proof)
        write_once(path, value)
        # The durable gate record must exist before the next iteration can start.
        require(json.loads(Path(path).read_text()) == value, "gate receipt readback mismatch")
        self.passed = value["verdict"] == "PASS"
        return self.passed

    def before_step(self, k):
        require((k <= 8 and k == len(self.completed) + 1) or (k > 8 and self.passed),
                "step9+ cannot run before durable A/A plus measured-budget PASS")
