#!/usr/bin/env python3
"""Execute the approved full LaFGS/STDLoc archive plan.

This executor is deliberately resumable and conservative.  It only mutates
rows explicitly marked ARCHIVE or DELETE_CANDIDATE in the generated TSV; KEEP
rows are never moved.  Archive moves happen before deletion, and every action
is recorded in a machine-readable receipt.

Usage:
    python3 ... --preflight
    python3 ... --execute
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


SQY = Path("/mnt/pool/sqy")
MANIFEST = Path("/root/STDLoc/docs/lafgs_archive_full_dry_run_20260909.tsv")
ARCHIVE_ROOT = SQY / "LaFGS_archive"
RECEIPT_MD = Path("/root/STDLoc/docs/lafgs_archive_execution_receipt_20260909.md")
RECEIPT_TSV = Path("/root/STDLoc/docs/lafgs_archive_execution_receipt_20260909.tsv")

PROTECTED_WHOLE_ROOTS = {
    SQY / "anygsloc_paper_experiments_20260902",
    SQY / "anygsloc_execution_queue_20260909",
    SQY / "Cambridge_stdloc",
    SQY / "lafgs_v24_cambridge_high_capacity_online_20260901",
    SQY / "stdloc_lafgs_v1_frozen_multiscene_20260731",
    SQY / "lafgs_assets",
    SQY / "lafgs_experiment_tools",
    SQY / "third_party_lafgs_priors",
}
NEVER_TOUCH_ROOTS = {
    SQY / "G4Splat_runs",
    SQY / "datasets",
    ARCHIVE_ROOT,
}
ACTION_SET = {"KEEP", "ARCHIVE", "DELETE_CANDIDATE"}


def lexists(path: Path) -> bool:
    return os.path.lexists(path)


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def lstat_size(path: Path) -> tuple[int, int]:
    st = path.lstat()
    return st.st_size, st.st_blocks * 512


def manifest_hash() -> str:
    digest = hashlib.sha256()
    with MANIFEST.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows() -> list[dict[str, str]]:
    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "source_root",
        "root_name",
        "scope_type",
        "relative_path",
        "source_path",
        "apparent_bytes",
        "allocated_bytes",
        "action",
        "archive_destination",
        "reason",
    }
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError("manifest has missing columns or no rows")
    return rows


def source_from_row(row: dict[str, str]) -> Path:
    root = Path(row["source_root"])
    relative = Path(row["relative_path"])
    if not root.is_absolute() or relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"unsafe source path fields: {row}")
    source = root if str(relative) == "." else root / relative
    if Path(row["source_path"]) != source:
        raise RuntimeError(f"manifest source mismatch: {row}")
    return source


def destination_from_row(row: dict[str, str]) -> Path | None:
    destination_text = row["archive_destination"]
    if not destination_text:
        return None
    destination = Path(destination_text)
    if not destination.is_absolute() or not is_under(destination, ARCHIVE_ROOT):
        raise RuntimeError(f"unsafe archive destination: {row}")
    return destination


def process_lines() -> list[str]:
    try:
        output = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    hits: list[str] = []
    needles = (
        "/mnt/pool/sqy/lafgs_",
        "/mnt/pool/sqy/stdloc_lafgs",
        "/mnt/pool/sqy/stdloc_matcha",
    )
    for line in output.splitlines():
        if not any(needle in line for needle in needles):
            continue
        if "execute_lafgs_archive_full_dry_run_20260909.py" in line:
            continue
        if "generate_lafgs_archive_full_dry_run_20260909.py" in line:
            continue
        hits.append(line.strip())
    return hits


def protected_violation(source: Path, action: str) -> str | None:
    for root in NEVER_TOUCH_ROOTS:
        if is_under(source, root):
            return f"source is under never-touch root {root}"
    for root in PROTECTED_WHOLE_ROOTS:
        if is_under(source, root) and action != "KEEP":
            return f"non-KEEP action targets protected root {root}"
    if is_under(source, ARCHIVE_ROOT):
        return "source is inside archive destination"
    return None


def validate_manifest(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[str], Counter, Counter]:
    errors: list[str] = []
    actionable: list[dict[str, str]] = []
    counts: Counter = Counter()
    allocated: Counter = Counter()
    seen_sources: set[Path] = set()

    for row in rows:
        action = row["action"]
        if action not in ACTION_SET:
            errors.append(f"unknown action {action}: {row}")
            continue
        try:
            source = source_from_row(row)
            destination = destination_from_row(row)
        except RuntimeError as exc:
            errors.append(str(exc))
            continue
        violation = protected_violation(source, action)
        if violation:
            errors.append(f"{violation}: {source}")
            continue
        if action == "ARCHIVE" and destination is None:
            errors.append(f"ARCHIVE row has no destination: {source}")
            continue
        if action != "ARCHIVE" and destination is not None:
            errors.append(f"non-ARCHIVE row has destination: {source}")
            continue
        if action != "KEEP":
            if source in seen_sources:
                errors.append(f"duplicate mutating source: {source}")
                continue
            seen_sources.add(source)
            actionable.append(row)
            counts[action] += 1
            allocated[action] += int(row["allocated_bytes"] or 0)
        else:
            counts["KEEP"] += 1

    return actionable, errors, counts, allocated


def compare_destination(source: Path, destination: Path, expected_apparent: int) -> bool:
    if not lexists(source):
        if destination.is_symlink():
            try:
                return destination.lstat().st_size == expected_apparent
            except OSError:
                return False
        try:
            return destination.is_file() and destination.stat().st_size == expected_apparent
        except OSError:
            return False
    if source.is_symlink() or destination.is_symlink():
        if not destination.is_symlink():
            return False
        return os.readlink(source) == os.readlink(destination)
    try:
        return destination.is_file() and destination.stat().st_size == expected_apparent
    except OSError:
        return False


def preflight_rows(rows: list[dict[str, str]], actionable: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
    completed: list[dict[str, str]] = []
    errors: list[str] = []
    for row in rows:
        if row["action"] != "KEEP":
            continue
        source = source_from_row(row)
        if protected_violation(source, "KEEP"):
            errors.append(f"KEEP row is under a never-touch source unexpectedly: {source}")
            continue
        # Active protected roots are allowed to grow/change while the queue is
        # running; they are never mutated by this executor.
        if any(is_under(source, root) for root in PROTECTED_WHOLE_ROOTS):
            continue
        if not lexists(source):
            errors.append(f"KEEP source missing: {source}")
            continue
        expected = int(row["apparent_bytes"] or 0)
        try:
            current, _ = lstat_size(source)
        except OSError as exc:
            errors.append(f"KEEP source stat failed: {source}: {exc}")
            continue
        if current != expected:
            errors.append(f"KEEP source size changed: {source} manifest={expected} current={current}")

    for row in actionable:
        source = source_from_row(row)
        destination = destination_from_row(row)
        action = row["action"]
        if action == "ARCHIVE" and destination is None:
            errors.append(f"missing archive destination: {source}")
            continue
        if not lexists(source):
            if action == "ARCHIVE" and destination is not None and lexists(destination):
                if not compare_destination(source, destination, int(row["apparent_bytes"] or 0)):
                    errors.append(f"existing archive destination mismatch: {destination}")
                    continue
                completed.append({**row, "result": "already_archived", "source": str(source), "destination": str(destination), "detail": "source absent and matching destination already exists"})
                continue
            if action == "DELETE_CANDIDATE":
                completed.append({**row, "result": "already_absent", "source": str(source), "destination": "", "detail": "source already absent"})
                continue
            errors.append(f"source missing: {source}")
            continue
        if action == "ARCHIVE" and lexists(destination):
            errors.append(f"archive destination already exists while source exists: {destination}")
            continue
        if row["scope_type"] == "tree":
            if not source.is_dir() or source.is_symlink():
                errors.append(f"tree row is not a real directory: {source}")
            continue
        try:
            current, _ = lstat_size(source)
        except OSError as exc:
            errors.append(f"source stat failed: {source}: {exc}")
            continue
        expected = int(row["apparent_bytes"] or 0)
        if current != expected:
            errors.append(f"source size changed: {source} manifest={expected} current={current}")
            continue
    return completed, errors


def move_one(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if lexists(destination):
        raise RuntimeError(f"destination already exists: {destination}")
    try:
        os.rename(source, destination)
        return "rename"
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
    # Cross-device fallback is not expected for /mnt/pool/sqy, but keep it
    # recoverable and type-preserving if the mount changes.
    temporary = destination.with_name(destination.name + ".partial")
    if lexists(temporary):
        raise RuntimeError(f"stale partial destination exists: {temporary}")
    if source.is_symlink():
        temporary.symlink_to(os.readlink(source))
        os.rename(temporary, destination)
        source.unlink()
        return "symlink-copy+unlink"
    shutil.copy2(source, temporary)
    if temporary.stat().st_size != source.stat().st_size:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"copy verification failed: {source}")
    os.rename(temporary, destination)
    source.unlink()
    return "copy+unlink"


def prune_empty_dirs(root: Path) -> int:
    if not root.is_dir() or root.is_symlink():
        return 0
    removed = 0
    for dirpath, dirnames, filenames in os.walk(root, topdown=False, followlinks=False):
        for dirname in list(dirnames):
            candidate = Path(dirpath) / dirname
            if candidate.is_symlink():
                continue
            try:
                if candidate.is_dir() and not any(candidate.iterdir()):
                    candidate.rmdir()
                    removed += 1
            except OSError:
                pass
        try:
            if not any(Path(dirpath).iterdir()):
                Path(dirpath).rmdir()
                removed += 1
        except OSError:
            pass
    return removed


def disk_free(path: Path) -> int:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return 0


def write_receipt(
    logs: list[dict[str, str]],
    summary: Counter,
    bytes_by_action: Counter,
    status: str,
    started: str,
    finished: str,
    pre_free: int,
    post_free: int,
    pruned: int,
    errors: list[str],
    manifest_digest: str,
) -> None:
    RECEIPT_TSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "root_name",
        "source_root",
        "relative_path",
        "action",
        "result",
        "source",
        "destination",
        "apparent_bytes",
        "allocated_bytes",
        "detail",
    ]
    with RECEIPT_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for log in logs:
            writer.writerow({field: log.get(field, "") for field in fields})

    def gib(value: int) -> str:
        return f"{value / 1024**3:.2f} GiB"

    lines = [
        "# LaFGS / STDLoc full archive execution receipt",
        "",
        f"Status: `{status}`",
        f"Started: `{started}`",
        f"Finished: `{finished}`",
        f"Manifest SHA-256: `{manifest_digest}`",
        "",
        "本次执行只处理 dry-run 中的 ARCHIVE 与 DELETE_CANDIDATE。KEEP、AnyGSLoc 当前目录、原始数据、G4Splat_runs 均未被移动或删除。",
        "",
        "| action/result | count | allocated from manifest |",
        "|---|---:|---:|",
    ]
    for key in ("ARCHIVE", "DELETE_CANDIDATE", "KEEP", "already_archived", "already_absent", "archived", "deleted", "kept"):
        if key in summary:
            lines.append(f"| `{key}` | {summary[key]} | {gib(bytes_by_action.get(key, 0))} |")
    lines += [
        "",
        f"Empty directories pruned: `{pruned}`.",
        f"Pool free bytes before: `{pre_free}` ({gib(pre_free)}); after: `{post_free}` ({gib(post_free)}).",
        f"Errors recorded: `{len(errors)}`.",
        "",
        f"Machine receipt: [`lafgs_archive_execution_receipt_20260909.tsv`](/root/STDLoc/docs/lafgs_archive_execution_receipt_20260909.tsv)",
    ]
    if errors:
        lines += ["", "## Errors", ""]
        lines.extend(f"- `{error}`" for error in errors[:200])
    RECEIPT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(execute: bool) -> int:
    if not MANIFEST.exists():
        print(f"manifest missing: {MANIFEST}", file=sys.stderr)
        return 2
    rows = read_rows()
    digest = manifest_hash()
    active_hits = process_lines()
    if active_hits:
        print("active old STDLoc/LaFGS processes detected; refusing to proceed:", file=sys.stderr)
        for hit in active_hits:
            print(hit, file=sys.stderr)
        return 3
    actionable, validation_errors, counts, planned_bytes = validate_manifest(rows)
    completed, preflight_errors = preflight_rows(rows, actionable)
    errors = validation_errors + preflight_errors
    print(f"manifest rows={len(rows)} mutating={len(actionable)} already_completed={len(completed)}")
    print("planned counts=" + ", ".join(f"{key}:{value}" for key, value in sorted(counts.items())))
    print("planned allocated=" + ", ".join(f"{key}:{value}" for key, value in sorted(planned_bytes.items())))
    if errors:
        print("preflight errors:", file=sys.stderr)
        for error in errors[:200]:
            print(error, file=sys.stderr)
        if len(errors) > 200:
            print(f"... {len(errors) - 200} more", file=sys.stderr)
        return 4
    if not execute:
        print("preflight OK; no files changed")
        return 0

    started = datetime.now().astimezone().isoformat(timespec="seconds")
    pre_free = disk_free(SQY)
    logs: list[dict[str, str]] = []
    bytes_by_action: Counter = Counter()
    summary: Counter = Counter()
    summary.update({"ARCHIVE": sum(1 for row in actionable if row["action"] == "ARCHIVE"), "DELETE_CANDIDATE": sum(1 for row in actionable if row["action"] == "DELETE_CANDIDATE"), "KEEP": sum(1 for row in rows if row["action"] == "KEEP")})
    bytes_by_action["ARCHIVE"] = sum(int(row["allocated_bytes"] or 0) for row in actionable if row["action"] == "ARCHIVE")
    bytes_by_action["DELETE_CANDIDATE"] = sum(int(row["allocated_bytes"] or 0) for row in actionable if row["action"] == "DELETE_CANDIDATE")

    for row in completed:
        result = row["result"]
        summary[result] += 1
        bytes_by_action[result] += int(row.get("allocated_bytes") or 0)
        logs.append(row)

    completed_keys = {
        (row.get("source", ""), row.get("action", ""))
        for row in completed
    }
    pending = [
        row
        for row in actionable
        if (str(source_from_row(row)), row["action"]) not in completed_keys
    ]

    try:
        ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
        # Move archive records first.  A move failure aborts before any delete.
        for index, row in enumerate(pending, 1):
            if row["action"] != "ARCHIVE":
                continue
            source = source_from_row(row)
            destination = destination_from_row(row)
            assert destination is not None
            method = move_one(source, destination)
            summary["archived"] += 1
            logs.append({
                **row,
                "result": "archived",
                "source": str(source),
                "destination": str(destination),
                "detail": method,
            })
            if index % 1000 == 0:
                print(f"archive progress: {index}/{len(actionable)} mutating rows")

        # Delete only after every ARCHIVE move succeeds.
        for index, row in enumerate(pending, 1):
            if row["action"] != "DELETE_CANDIDATE":
                continue
            source = source_from_row(row)
            if not lexists(source):
                continue
            if row["scope_type"] == "tree":
                if not source.is_dir() or source.is_symlink():
                    raise RuntimeError(f"tree deletion target is not a real directory: {source}")
                prune_empty_dirs(source)
                if not lexists(source):
                    summary["deleted"] += 1
                    logs.append({
                        **row,
                        "result": "deleted",
                        "source": str(source),
                        "destination": "",
                        "detail": row["reason"],
                    })
                    continue
                if any(source.iterdir()):
                    raise RuntimeError(f"tree deletion target is not empty: {source}")
                source.rmdir()
            else:
                source.unlink()
            summary["deleted"] += 1
            logs.append({
                **row,
                "result": "deleted",
                "source": str(source),
                "destination": "",
                "detail": row["reason"],
            })
            if index % 1000 == 0:
                print(f"delete progress: {index}/{len(actionable)} mutating rows")

        # Prune only source roots that had mutating rows; never walk a protected
        # root or a parent such as /mnt/pool/sqy itself.
        source_roots = sorted({Path(row["source_root"]) for row in actionable}, key=lambda p: len(p.parts), reverse=True)
        pruned = 0
        for source_root in source_roots:
            if any(is_under(source_root, root) or is_under(root, source_root) for root in NEVER_TOUCH_ROOTS):
                continue
            pruned += prune_empty_dirs(source_root)

        # Verify every mutation.  KEEP rows under protected roots are not
        # size-checked because the active queue may legitimately grow.
        verification_errors: list[str] = []
        for row in actionable:
            source = source_from_row(row)
            destination = destination_from_row(row)
            if row["action"] == "ARCHIVE":
                if lexists(source) or destination is None or not lexists(destination) or not compare_destination(source, destination, int(row["apparent_bytes"] or 0)):
                    verification_errors.append(f"archive verification failed: {source} -> {destination}")
            elif row["action"] == "DELETE_CANDIDATE" and lexists(source):
                verification_errors.append(f"delete verification failed: {source}")
        if verification_errors:
            raise RuntimeError("; ".join(verification_errors[:20]))
    except Exception as exc:  # noqa: BLE001 - write a receipt for resumable recovery
        errors.append(str(exc))
        finished = datetime.now().astimezone().isoformat(timespec="seconds")
        write_receipt(
            logs,
            summary,
            bytes_by_action,
            "ABORTED",
            started,
            finished,
            pre_free,
            disk_free(SQY),
            locals().get("pruned", 0),
            errors,
            digest,
        )
        print(f"execution aborted: {exc}", file=sys.stderr)
        return 5

    post_free = disk_free(SQY)
    finished = datetime.now().astimezone().isoformat(timespec="seconds")
    write_receipt(
        logs,
        summary,
        bytes_by_action,
        "COMPLETE",
        started,
        finished,
        pre_free,
        post_free,
        pruned,
        errors,
        digest,
    )
    print(f"execution complete: archived={summary['archived']} deleted={summary['deleted']} pruned={pruned}")
    print(f"receipt={RECEIPT_MD}")
    print(f"machine_receipt={RECEIPT_TSV}")
    print(f"pool_free_before={pre_free} pool_free_after={post_free}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    return run(execute=args.execute)


if __name__ == "__main__":
    raise SystemExit(main())
