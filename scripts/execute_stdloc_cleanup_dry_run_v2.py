#!/usr/bin/env python3
"""Execute the approved, read-only-generated STDLoc cleanup manifest safely.

The script is intentionally conservative:
* it never touches the AnyGSLoc paper root or G4Splat_runs;
* it validates every source path and apparent size before mutating anything;
* ARCHIVE files are moved under one versioned archive root;
* DELETE_CANDIDATE files are unlinked only after the archive pass succeeds;
* a machine-readable and Markdown receipt is written after verification.

Use ``--preflight`` first.  ``--execute`` is the only mode that mutates files.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


MANIFEST = Path("/root/STDLoc/docs/stdloc_cleanup_dry_run_20260909_v2.tsv")
ARCHIVE_ROOT = Path("/mnt/pool/sqy/stdloc_closed_loop_archive_20260909/stdloc_cleanup_v2")
RECEIPT_MD = Path("/root/STDLoc/docs/stdloc_cleanup_execution_receipt_20260909_v2.md")
RECEIPT_TSV = Path("/root/STDLoc/docs/stdloc_cleanup_execution_receipt_20260909_v2.tsv")

PROTECTED_ROOT_KEYS = {"anygsloc_paper"}
PROTECTED_PATHS = {
    Path("/mnt/pool/sqy/anygsloc_paper_experiments_20260902"),
    Path("/mnt/pool/sqy/anygsloc_execution_queue_20260909"),
    Path("/mnt/pool/sqy/G4Splat_runs"),
}


def parse_rows() -> list[dict[str, str]]:
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def path_for(row: dict[str, str]) -> tuple[Path, Path]:
    root = Path(row["root_path"])
    rel = Path(row["relative_path"])
    if rel.is_absolute() or ".." in rel.parts:
        raise RuntimeError(f"unsafe relative path in manifest: {row}")
    source = root / rel
    destination = ARCHIVE_ROOT / row["root"] / rel
    return source, destination


def ensure_under(path: Path, parent: Path) -> None:
    try:
        path.relative_to(parent)
    except ValueError as exc:
        raise RuntimeError(f"path escapes protected parent: {path} not under {parent}") from exc


def check_old_processes() -> list[str]:
    try:
        out = subprocess.check_output(["ps", "-eo", "pid=,cmd="], text=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    needles = (
        "/mnt/pool/sqy/stdloc_lafgs_",
        "/mnt/pool/sqy/lafgs_",
        "/mnt/pool/sqy/stdloc_lafgs",
    )
    hits = []
    for line in out.splitlines():
        if any(n in line for n in needles) and "execute_stdloc_cleanup_dry_run_v2.py" not in line:
            hits.append(line.strip())
    return hits


def preflight(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str], Counter, defaultdict]:
    actionable = []
    completed = []
    errors = []
    counts: Counter = Counter()
    bytes_by_action = defaultdict(lambda: [0, 0])

    for row in rows:
        root_key = row["root"]
        root = Path(row["root_path"])
        source, destination = path_for(row)
        if root_key in PROTECTED_ROOT_KEYS:
            counts["PROTECTED_SKIP"] += 1
            continue
        if root in PROTECTED_PATHS:
            errors.append(f"manifest unexpectedly targets protected root: {root}")
            continue
        if not root.exists():
            errors.append(f"root missing: {root}")
            continue
        ensure_under(source, root)
        ensure_under(destination, ARCHIVE_ROOT)
        if source.is_symlink():
            # The manifest contains two visibility aliases.  They are safe to
            # unlink as DELETE_CANDIDATE entries, but must never be copied as
            # archived symlinks because moving them would change relative-link
            # resolution.  Reject any other symlink action.
            try:
                resolved = source.resolve(strict=False)
                ensure_under(resolved, root)
            except (OSError, RuntimeError) as exc:
                errors.append(f"unsafe symlink source: {source}: {exc}")
                continue
            if row["action"] != "DELETE_CANDIDATE":
                errors.append(f"refusing non-delete symlink source: {source}")
                continue
            # A broken link is still a safe DELETE_CANDIDATE: unlink only the
            # link itself.  The target is handled by its own manifest row.
            counts[row["action"]] += 1
            bytes_by_action[row["action"]][0] += int(row["apparent_bytes"])
            bytes_by_action[row["action"]][1] += int(row["allocated_bytes"])
            actionable.append(row)
            continue
        expected = int(row["apparent_bytes"])
        if not source.exists():
            if row["action"] == "ARCHIVE" and destination.exists():
                dst_size = destination.stat().st_size
                if dst_size != expected:
                    errors.append(f"archive destination size mismatch: {destination} manifest={expected} current={dst_size}")
                    continue
                completed.append({
                    "root": row["root"], "root_path": row["root_path"], "relative_path": row["relative_path"],
                    "action": row["action"], "result": "already_archived", "source": str(source), "destination": str(destination),
                    "bytes": str(dst_size), "allocated_bytes": str(destination.stat().st_blocks * 512), "detail": "detected during resumable preflight",
                })
                counts[row["action"]] += 1
                bytes_by_action[row["action"]][0] += expected
                bytes_by_action[row["action"]][1] += int(row["allocated_bytes"])
                continue
            if row["action"] == "DELETE_CANDIDATE":
                completed.append({
                    "root": row["root"], "root_path": row["root_path"], "relative_path": row["relative_path"],
                    "action": row["action"], "result": "already_deleted", "source": str(source), "destination": "",
                    "bytes": str(expected), "allocated_bytes": str(int(row["allocated_bytes"])), "detail": "detected during resumable preflight",
                })
                counts[row["action"]] += 1
                bytes_by_action[row["action"]][0] += expected
                bytes_by_action[row["action"]][1] += int(row["allocated_bytes"])
                continue
            errors.append(f"source missing: {source}")
            continue
        st = source.stat()
        if st.st_size != expected:
            errors.append(f"size changed: {source} manifest={expected} current={st.st_size}")
            continue
        action = row["action"]
        if action not in {"KEEP", "ARCHIVE", "DELETE_CANDIDATE"}:
            errors.append(f"unknown action {action}: {source}")
            continue
        if action == "ARCHIVE" and destination.exists():
            errors.append(f"archive destination already exists: {destination}")
            continue
        actionable.append(row)
        counts[action] += 1
        bytes_by_action[action][0] += st.st_size
        bytes_by_action[action][1] += st.st_blocks * 512

    return actionable, completed, errors, counts, bytes_by_action


def move_archive(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(source, destination)
        return "rename"
    except OSError as exc:
        if getattr(exc, "errno", None) != 18:  # EXDEV
            raise
        tmp = destination.with_name(destination.name + ".partial")
        if tmp.exists():
            tmp.unlink()
        shutil.copy2(source, tmp)
        if tmp.stat().st_size != source.stat().st_size:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"copy size verification failed: {source}")
        os.replace(tmp, destination)
        source.unlink()
        return "copy+unlink"


def prune_empty_dirs(root: Path) -> int:
    removed = 0
    if not root.exists():
        return 0
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        if dirnames or filenames:
            continue
        path = Path(dirpath)
        try:
            path.rmdir()
            removed += 1
        except OSError:
            pass
    return removed


def write_receipt(rows: list[dict[str, str]], action_log: list[dict[str, str]], counts: Counter, bytes_by_action, pruned: int, started: str, finished: str) -> None:
    RECEIPT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with RECEIPT_TSV.open("w", encoding="utf-8", newline="") as f:
        fields = ["root", "root_path", "relative_path", "action", "result", "source", "destination", "bytes", "allocated_bytes", "detail"]
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(action_log)

    def gib(n: int) -> str:
        return f"{n / 1024**3:.2f} GiB"

    lines = [
        "# STDLoc / LaFGS cleanup execution receipt v2",
        "",
        f"Started: `{started}`",
        f"Finished: `{finished}`",
        "",
        "本次只处理 v2 dry-run manifest 中的旧 STDLoc/LaFGS 根目录。`anygsloc_paper_experiments_20260902`、当前 AnyGSLoc 队列和 `G4Splat_runs` 均跳过。",
        "",
        "| action | files | apparent | allocated before |",
        "|---|---:|---:|---:|",
    ]
    for action in ("KEEP", "ARCHIVE", "DELETE_CANDIDATE", "PROTECTED_SKIP"):
        if action == "PROTECTED_SKIP":
            lines.append(f"| `{action}` | {counts.get(action, 0)} | — | — |")
        else:
            b = bytes_by_action.get(action, [0, 0])
            lines.append(f"| `{action}` | {counts.get(action, 0)} | {gib(b[0])} | {gib(b[1])} |")
    lines += [
        "",
        f"Empty directories pruned under selected old roots: `{pruned}`.",
        "",
        f"Machine receipt: [`stdloc_cleanup_execution_receipt_20260909_v2.tsv`](/root/STDLoc/docs/stdloc_cleanup_execution_receipt_20260909_v2.tsv)",
    ]
    RECEIPT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if not MANIFEST.exists():
        print(f"manifest missing: {MANIFEST}", file=sys.stderr)
        return 2
    rows = parse_rows()
    hits = check_old_processes()
    if hits:
        print("active old STDLoc/LaFGS processes detected; refusing to proceed:", file=sys.stderr)
        for hit in hits:
            print(hit, file=sys.stderr)
        return 3
    actionable, completed, errors, counts, bytes_by_action = preflight(rows)
    print(f"manifest rows={len(rows)} actionable={len(actionable)} already_completed={len(completed)}")
    print("counts=" + ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())))
    for action in ("KEEP", "ARCHIVE", "DELETE_CANDIDATE"):
        b = bytes_by_action.get(action, [0, 0])
        print(f"{action}: apparent={b[0]} allocated={b[1]}")
    if errors:
        print("preflight errors:", file=sys.stderr)
        for error in errors[:100]:
            print(error, file=sys.stderr)
        if len(errors) > 100:
            print(f"... {len(errors) - 100} more", file=sys.stderr)
        return 4
    if args.preflight:
        print("preflight OK; no files changed")
        return 0

    started = datetime.now().astimezone().isoformat(timespec="seconds")
    action_log = list(completed)
    # Archive pass first.  If a move fails, no DELETE_CANDIDATE file is touched.
    for row in actionable:
        source, destination = path_for(row)
        action = row["action"]
        if action == "ARCHIVE":
            method = move_archive(source, destination)
            st = destination.stat()
            action_log.append({
                "root": row["root"], "root_path": row["root_path"], "relative_path": row["relative_path"],
                "action": action, "result": "archived", "source": str(source), "destination": str(destination),
                "bytes": str(st.st_size), "allocated_bytes": str(st.st_blocks * 512), "detail": method,
            })

    # Delete pass only after every archive move has completed.
    for row in actionable:
        if row["action"] != "DELETE_CANDIDATE":
            continue
        source, destination = path_for(row)
        st = source.lstat() if source.is_symlink() else source.stat()
        source.unlink()
        action_log.append({
            "root": row["root"], "root_path": row["root_path"], "relative_path": row["relative_path"],
            "action": row["action"], "result": "deleted", "source": str(source), "destination": "",
            "bytes": str(st.st_size), "allocated_bytes": str(st.st_blocks * 512), "detail": row["reason"],
        })

    # KEEP files are logged only after checking they survived unchanged.
    for row in actionable:
        if row["action"] != "KEEP":
            continue
        source, _ = path_for(row)
        st = source.stat()
        action_log.append({
            "root": row["root"], "root_path": row["root_path"], "relative_path": row["relative_path"],
            "action": row["action"], "result": "kept", "source": str(source), "destination": "",
            "bytes": str(st.st_size), "allocated_bytes": str(st.st_blocks * 512), "detail": row["reason"],
        })

    pruned = 0
    for root in sorted({Path(row["root_path"]) for row in actionable}):
        pruned += prune_empty_dirs(root)

    # Verify archived files and deletions before receipt is written.
    verification_errors = []
    for row in actionable:
        source, destination = path_for(row)
        if row["action"] == "ARCHIVE":
            if source.exists() or not destination.exists() or destination.stat().st_size != int(row["apparent_bytes"]):
                verification_errors.append(f"archive verification failed: {source} -> {destination}")
        elif row["action"] == "DELETE_CANDIDATE":
            if source.exists() or source.is_symlink():
                verification_errors.append(f"delete verification failed: {source}")
        elif row["action"] == "KEEP":
            if not source.exists() or source.stat().st_size != int(row["apparent_bytes"]):
                verification_errors.append(f"keep verification failed: {source}")
    if verification_errors:
        print("post-action verification errors:", file=sys.stderr)
        for error in verification_errors[:100]:
            print(error, file=sys.stderr)
        return 5

    finished = datetime.now().astimezone().isoformat(timespec="seconds")
    # Include skipped protected rows in the receipt count but never touch them.
    counts["PROTECTED_SKIP"] = sum(1 for row in rows if row["root"] in PROTECTED_ROOT_KEYS)
    write_receipt(rows, action_log, counts, bytes_by_action, pruned, started, finished)
    print(f"execution complete; receipt={RECEIPT_MD}")
    print(f"machine receipt={RECEIPT_TSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
