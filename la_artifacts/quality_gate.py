import math
from dataclasses import asdict, dataclass, field
from typing import Optional


def _float_or_none(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


@dataclass
class GateDecision:
    accepted: bool
    reason: str = "ok"
    failures: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class SyntheticQualityGateConfig:
    max_artifact_mean: float = 0.60
    max_artifact_p95: Optional[float] = None
    max_artifact_mild_frac: float = 0.85
    max_artifact_severe_frac: float = 0.58
    max_low_detail_mean: Optional[float] = 0.60
    reject_missing_metrics: bool = True


class SyntheticQualityGate:
    def __init__(self, config=None):
        self.config = config or SyntheticQualityGateConfig()

    def evaluate_summary(self, summary):
        summary = dict(summary or {})
        checks = {
            "artifact_score_mean": self.config.max_artifact_mean,
            "artifact_score_p95": self.config.max_artifact_p95,
            "artifact_mild_frac": self.config.max_artifact_mild_frac,
            "artifact_severe_frac": self.config.max_artifact_severe_frac,
            "low_detail_mean": self.config.max_low_detail_mean,
        }
        failures = {}
        metrics = {}
        for name, limit in checks.items():
            if limit is None:
                continue
            value = _float_or_none(summary.get(name))
            metrics[name] = value
            if value is None:
                if self.config.reject_missing_metrics:
                    failures[name] = {"value": None, "max": float(limit), "reason": "missing"}
                continue
            if value > float(limit):
                failures[name] = {"value": value, "max": float(limit), "reason": "above_max"}
        return GateDecision(
            accepted=not failures,
            reason="ok" if not failures else "synthetic_quality_rejected",
            failures=failures,
            metrics=metrics,
        )

    def apply_to_record(self, record, summary):
        decision = self.evaluate_summary(summary)
        record.artifact_score = float(decision.metrics.get("artifact_score_mean") or 0.0)
        record.accepted = bool(decision.accepted)
        record.reason = decision.reason
        meta = getattr(record, "meta", {}) or {}
        meta["artifact_summary"] = dict(summary or {})
        meta["synthetic_quality_gate"] = decision.to_dict()
        record.meta = meta
        return decision


@dataclass
class TeacherCacheGateConfig:
    max_sparse_te: Optional[float] = 100.0
    max_dense_te: Optional[float] = 100.0
    max_sparse_ae: Optional[float] = None
    max_dense_ae: Optional[float] = None
    allowed_stages: tuple = ("teacher_ok",)
    require_not_failed: bool = True
    reject_missing_metrics: bool = True


class TeacherCacheGate:
    def __init__(self, config=None):
        self.config = config or TeacherCacheGateConfig()

    def evaluate(self, item):
        if not item:
            return GateDecision(False, "missing_teacher_cache", {"cache": {"reason": "missing"}}, {})
        failures = {}
        metrics = {}
        if self.config.require_not_failed and bool(item.get("failed", False)):
            failures["failed"] = {"value": True, "reason": "teacher_failed"}
        allowed = {str(value) for value in (self.config.allowed_stages or []) if str(value)}
        stage = str(item.get("failure_stage", ""))
        metrics["failure_stage"] = stage
        if allowed and stage not in allowed:
            failures["failure_stage"] = {
                "value": stage,
                "allowed": sorted(allowed),
                "reason": "teacher_stage_rejected",
            }
        metric_checks = {
            "te": self.config.max_sparse_te,
            "dense_te": self.config.max_dense_te,
            "ae": self.config.max_sparse_ae,
            "dense_ae": self.config.max_dense_ae,
        }
        for name, limit in metric_checks.items():
            if limit is None:
                continue
            value = _float_or_none(item.get(name))
            metrics[name] = value
            if value is None:
                if self.config.reject_missing_metrics:
                    failures[name] = {"value": None, "max": float(limit), "reason": "missing"}
                continue
            if value > float(limit):
                failures[name] = {"value": value, "max": float(limit), "reason": "above_max"}
        if not failures:
            return GateDecision(True, "ok", {}, metrics)
        reason = "teacher_cache_rejected"
        if "cache" in failures:
            reason = "missing_teacher_cache"
        elif "failure_stage" in failures:
            reason = "teacher_stage_rejected"
        elif "failed" in failures:
            reason = "teacher_failed"
        return GateDecision(False, reason, failures, metrics)


def summarize_gate_decisions(decisions):
    accepted = 0
    rejected_reasons = {}
    for decision in decisions:
        if decision.accepted:
            accepted += 1
        else:
            rejected_reasons[decision.reason] = rejected_reasons.get(decision.reason, 0) + 1
    total = len(decisions)
    return {
        "count": total,
        "accepted_count": accepted,
        "rejected_count": total - accepted,
        "rejected_reasons": dict(sorted(rejected_reasons.items())),
    }
