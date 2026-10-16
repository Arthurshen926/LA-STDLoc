"""Pre-registered unique-cut gate. Pure CPU contracts; no execution by itself."""
import hashlib
import json
from pathlib import Path

SCHEMA = "matcha14999_continue_v1"
def require(ok, message):
    if not ok:
        raise RuntimeError("continuation-gate: " + message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def gate_digest(value):
    body = {k: v for k, v in value.items() if k not in ("source_review_path", "source_review_sha256")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def validate_gate(value, manifest_sha, reference_sha, checkpoint_sha):
    require(set(value) == {"schema", "checkpoint_iteration", "comparison_start",
        "comparison_end", "training_end", "binding_sha256", "reference_trace_sha256",
        "checkpoint14999_sha256", "checkpoint15101_sha256", "source_review_sha256", "source_review_path",
        "budget"}, "schema fields")
    require(value["schema"] == SCHEMA, "schema/version")
    require((value["checkpoint_iteration"], value["comparison_start"],
             value["comparison_end"], value["training_end"]) == (14999, 15000, 15101, 30000),
            "unapproved cut or horizon")
    require(value["binding_sha256"] == manifest_sha and
            value["reference_trace_sha256"] == reference_sha and
            value["checkpoint14999_sha256"] == checkpoint_sha, "frozen evidence mismatch")
    for key in ("checkpoint15101_sha256", "source_review_sha256"):
        require(isinstance(value[key], str) and len(value[key]) == 64, "missing evidence " + key)
    require(Path(value["source_review_path"]).is_absolute(), "missing gate review path")
    budget = value["budget"]
    require(set(budget) == {"verdict", "based_on_iteration", "measured_prefix_seconds",
        "measured_checkpoint_bytes", "measured_restore_seconds", "predicted_total_seconds",
        "deadline_seconds", "physical_gpu", "logical_gpu", "uuid"}, "budget fields")
    require(budget["verdict"] == "PASS" and budget["based_on_iteration"] == 14999,
            "second-job budget must use actual14999 evidence")
    require(budget["deadline_seconds"] == 3500 and
            0 < budget["predicted_total_seconds"] < 3380, "insufficient deadline margin")
    require(all(budget[k] > 0 for k in ("measured_prefix_seconds", "measured_checkpoint_bytes",
                                       "measured_restore_seconds")), "unmeasured budget")
    require(budget["physical_gpu"] == 0 and budget["logical_gpu"] == 0 and
            isinstance(budget["uuid"], str) and budget["uuid"].startswith("GPU-"), "assigned GPU")
    return value

class ComparisonGate:
    def __init__(self, reference):
        self.reference = reference
        self.completed = []
        need = set(range(15000, 15102))
        require(need <= {k for k, full in reference if not full}, "missing comparison step")
        require((14999, True) in reference and (15101, True) in reference,
                "missing endpoint full-state evidence")
        self.restored = False
        self.passed = False

    def restored_boundary(self, digest):
        require(digest == self.reference[(14999, True)]["exact_digest"],
                "14999 restored full state differs")
        self.restored = True

    def observe(self, iteration, record, full_digest=None):
        if iteration > 15101:
            require(self.passed, "cannot enter15102 without comparison PASS")
            return
        require(self.restored, "restored14999 not verified")
        require(iteration == 15000 + len(self.completed), "missing/out-of-order comparison step")
        require(record == self.reference[(iteration, False)], "first differing trajectory step " + str(iteration))
        if iteration == 15101:
            require(full_digest is not None and
                    full_digest == self.reference[(15101, True)]["exact_digest"],
                    "15101 full terminal state differs or missing")
        self.completed.append(iteration)
        if iteration == 15101:
            require(self.completed == list(range(15000, 15102)), "incomplete comparison")
            self.passed = True

    def before_step(self, iteration):
        require(self.restored, "cannot take a step before restored14999 full-state verification")
        if iteration > 15101:
            require(self.passed, "automatic continuation gate is closed")
        else:
            require(iteration == 15000 + len(self.completed), "comparison step would be skipped or repeated")
