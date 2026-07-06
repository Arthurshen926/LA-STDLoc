#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from la_artifacts.pseudo_query import PseudoQueryManifest, PseudoTeacherCache


def _comma_list(value):
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _record_key(record):
    return record.teacher_cache_key or record.query_id


def _read_previous_summary(path):
    path = Path(path) if path else None
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _limited(values, limit=16):
    return sorted(values)[: int(limit)]


def audit_teacher_cache(
    manifest_path,
    cache_path,
    summary_json="",
    sources=None,
    sample_limit=16,
):
    manifest_path = Path(manifest_path)
    cache_path = Path(cache_path)
    previous_summary = _read_previous_summary(summary_json)
    allowed_sources = {str(item) for item in (sources or []) if str(item)}

    manifest = PseudoQueryManifest.load(manifest_path)
    records = [
        record
        for record in manifest.records
        if record.accepted and (not allowed_sources or record.source in allowed_sources)
    ]
    manifest_keys = [_record_key(record) for record in records]
    manifest_key_set = set(manifest_keys)
    cache = PseudoTeacherCache.load(cache_path)
    cache_keys = set(cache.items)

    stage_counts = Counter(str(item.get("failure_stage", "unknown")) for item in cache.items.values())
    source_counts = Counter(str(record.source) for record in records)
    missing_keys = manifest_key_set - cache_keys
    extra_keys = cache_keys - manifest_key_set

    summary = {
        "manifest": os.path.abspath(os.fspath(manifest_path)),
        "output": os.path.abspath(os.fspath(cache_path)),
        "count": len(cache.items),
        "manifest_count": len(records),
        "sources": sorted(allowed_sources),
        "source_counts": dict(sorted(source_counts.items())),
        "stage_counts": dict(sorted(stage_counts.items())),
        "cache_coverage": {
            "manifest_records": len(records),
            "cache_records": len(cache.items),
            "cached_manifest_records": len(manifest_key_set & cache_keys),
            "missing_cache_count": len(missing_keys),
            "extra_cache_count": len(extra_keys),
            "missing_cache_keys": _limited(missing_keys, sample_limit),
            "extra_cache_keys": _limited(extra_keys, sample_limit),
        },
        "sparse_valid_mask": previous_summary.get("sparse_valid_mask", {}),
    }
    if summary_json:
        summary_path = Path(summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Audit pseudo teacher cache provenance and manifest coverage.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", "--cache", dest="cache_path", required=True)
    parser.add_argument("--summary_json", default="")
    parser.add_argument("--sources", default="")
    parser.add_argument("--sample_limit", type=int, default=16)
    args = parser.parse_args()

    summary = audit_teacher_cache(
        manifest_path=args.manifest,
        cache_path=args.cache_path,
        summary_json=args.summary_json,
        sources=_comma_list(args.sources),
        sample_limit=args.sample_limit,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
