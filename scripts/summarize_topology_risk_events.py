#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path


TOPOLOGY_PREFIX = "[Topology]"
KV_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>\S+)")


def _parse_value(value):
    if value == "True":
        return True
    if value == "False":
        return False
    if value.lower() == "nan":
        return float("nan")
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        return float(value)
    except ValueError:
        return value


def parse_topology_event(line):
    if TOPOLOGY_PREFIX not in line:
        return None
    event = {}
    for match in KV_RE.finditer(line):
        event[match.group("key")] = _parse_value(match.group("value"))
    return event if event else None


def parse_topology_events(text):
    events = []
    for line in str(text).splitlines():
        event = parse_topology_event(line)
        if event is not None:
            events.append(event)
    return events


def _finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _sum_metric_delta(events, key):
    total = 0
    for event in events:
        value = event.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            total += int(value)
        elif isinstance(value, float) and math.isfinite(value):
            total += int(value)
    return total


def _mean(values):
    finite = [float(value) for value in values if _finite_number(value)]
    if not finite:
        return None
    return sum(finite) / len(finite)


def summarize_events(events):
    events = list(events)
    accepted = [event for event in events if event.get("risk_accepted") is True]
    rejected = [event for event in events if event.get("risk_accepted") is False]
    reason_counts = {}
    for event in events:
        reason = event.get("risk_reason")
        if reason is None:
            continue
        reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1

    def metric_delta(group):
        return {
            "r5": _sum_metric_delta(group, "risk_r5_delta"),
            "r2": _sum_metric_delta(group, "risk_r2_delta"),
            "tail_fail": _sum_metric_delta(group, "risk_tail_fail_delta"),
        }

    return {
        "events": len(events),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "requested_split_total": sum(int(event.get("requested_split", 0) or 0) for event in events),
        "children_added_total": sum(int(event.get("children_added", 0) or 0) for event in events),
        "accepted_with_metric_count": sum(1 for event in accepted if "risk_metric_count" in event),
        "rejected_with_metric_count": sum(1 for event in rejected if "risk_metric_count" in event),
        "accepted_metric_missing": sum(1 for event in accepted if "risk_metric_count" not in event),
        "rejected_metric_missing": sum(1 for event in rejected if "risk_metric_count" not in event),
        "accepted_metric_delta": metric_delta(accepted),
        "rejected_metric_delta": metric_delta(rejected),
        "risk_delta_mean": _mean(event.get("risk_delta") for event in events),
        "accepted_risk_delta_mean": _mean(event.get("risk_delta") for event in accepted),
        "rejected_risk_delta_mean": _mean(event.get("risk_delta") for event in rejected),
        "reason_counts": reason_counts,
    }


def summarize_log(path):
    path = Path(path)
    text = path.read_text(errors="replace")
    events = parse_topology_events(text)
    return {
        "path": str(path),
        "summary": summarize_events(events),
        "events": events,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", help="Topology log files to summarize")
    parser.add_argument("--events", action="store_true", help="Include full event rows")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    rows = []
    for log_path in args.logs:
        row = summarize_log(log_path)
        if not args.events:
            row.pop("events", None)
        rows.append(row)

    payload = rows[0] if len(rows) == 1 else rows
    text = json.dumps(payload, indent=2, allow_nan=True)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
