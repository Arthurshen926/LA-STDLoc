#!/usr/bin/env python3
"""Archive the render-only experiment's small, human-auditable records.

Large tensors, caches, images, and rendered outputs intentionally remain outside
Git.  JSON query results are deterministically gzip-compressed; all other JSON
and status text files are copied byte-for-byte.  The manifest retains the
original path and content digest and makes the archive independently verifiable.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "lafgs_rendered_track_record_archive"
VERSION = 1
DEFAULT_OUTPUT = Path("docs/evidence/rendered_track_runs")

SOURCE_ROOTS = (
    ("v10_render_probe", Path("/mnt/pool/sqy/lafgs_rendered_rgb_only_probe_20260813")),
    (
        "v10_mapping_audits",
        Path("/mnt/pool/sqy/lafgs_rendered_rgb_only_trainonly_20260813"),
    ),
    (
        "v10_fullchain",
        Path("/mnt/pool/sqy/lafgs_render_track_only_fullchain_20260814"),
    ),
    (
        "v11_appearance",
        Path("/mnt/pool/sqy/lafgs_render_track_only_appearance_v11_20260814"),
    ),
    (
        "v12_support",
        Path("/mnt/pool/sqy/lafgs_render_track_only_support_v12_20260814"),
    ),
    (
        "v13_component_repair",
        Path("/mnt/pool/sqy/lafgs_render_track_only_support_v13_20260814"),
    ),
    (
        "v14_fullmap",
        Path("/mnt/pool/sqy/lafgs_render_track_only_fullmap_v14_20260815"),
    ),
    (
        "r1_artifact_stability",
        Path("/mnt/pool/sqy/lafgs_render_track_only_artifact_r1_20260815"),
    ),
    (
        "v14_method_enhancements",
        Path("/mnt/pool/sqy/lafgs_render_track_only_conditional_gwff_20260815"),
    ),
    (
        "full_reference_history_replay",
        Path("/mnt/pool/sqy/lafgs_render_track_full_reference_history_replay_20260816"),
    ),
    (
        "full_reference_v14_v15",
        Path("/mnt/pool/sqy/lafgs_render_track_full_reference_v5_20260815"),
    ),
)

# Invalid launch/preflight products are summarized in the checked-in result
# documents but are not themselves scientific records.  Cross-fit directories
# named blocked_00/01/02 are valid held blocks and are deliberately retained.
EXCLUDED_PATH_MARKERS = (
    "preflight_invalid",
    "launch_subshell_invalid",
    "preopt_invalid",
    "execution_invalid",
    "invalid_invocation",
    "_invalid_",
    "/smoke",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _eligible(source: Path, root: Path) -> bool:
    if source.suffix not in {".json", ".md", ".txt"}:
        return False
    relative = "/" + source.relative_to(root).as_posix().lower()
    return not any(marker in relative for marker in EXCLUDED_PATH_MARKERS)


def _gzip_deterministic(payload: bytes) -> bytes:
    return gzip.compress(payload, compresslevel=9, mtime=0)


def _is_json(payload: bytes) -> bool:
    try:
        json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return True


def _archive(
    output: Path, *, source_roots: tuple[tuple[str, Path], ...] = SOURCE_ROOTS
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"archive output already exists: {output}")
    missing = [str(root) for _, root in source_roots if not root.is_dir()]
    if missing:
        raise FileNotFoundError(f"missing source roots: {missing}")

    output_parent = output.parent.resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output_parent))
    records: list[dict[str, Any]] = []
    try:
        for stage, root in source_roots:
            sources = sorted(path for path in root.rglob("*") if path.is_file())
            for source in sources:
                if not _eligible(source, root):
                    continue
                relative = source.relative_to(root)
                raw = source.read_bytes()
                is_query_result = source.name == "results.json"
                archived_relative = Path(stage) / relative
                if is_query_result:
                    if not _is_json(raw):
                        raise ValueError(f"query result is not valid JSON: {source}")
                    archived_relative = archived_relative.with_suffix(".json.gz")
                    archived = _gzip_deterministic(raw)
                    encoding = "gzip_mtime_0"
                    content_type = "application/json"
                else:
                    archived = raw
                    encoding = "identity"
                    if source.suffix == ".json" and _is_json(raw):
                        content_type = "application/json"
                    elif source.suffix == ".json":
                        # Some historical invocation files are terminal
                        # transcripts despite their `.json` suffix.  Preserve
                        # their bytes without advertising them as JSON.
                        archived_relative = archived_relative.with_name(
                            archived_relative.name + ".log"
                        )
                        content_type = "text/plain"
                    else:
                        content_type = "text/plain"
                destination = temp_root / archived_relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archived)
                records.append(
                    {
                        "stage": stage,
                        "source_path": str(source),
                        "source_size_bytes": len(raw),
                        "source_sha256": _sha256_bytes(raw),
                        "archive_path": archived_relative.as_posix(),
                        "archive_size_bytes": len(archived),
                        "archive_sha256": _sha256_bytes(archived),
                        "encoding": encoding,
                        "content_type": content_type,
                    }
                )

        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "version": VERSION,
            "scope": (
                "render_only_stairs_shopfacade_history"
                if source_roots == SOURCE_ROOTS
                else "explicit_render_only_experiment_roots"
            ),
            "large_artifacts_in_git": False,
            "included_extensions": [".json", ".md", ".txt"],
            "query_results_encoding": "deterministic_gzip_mtime_0",
            "excluded_path_markers": list(EXCLUDED_PATH_MARKERS),
            "source_roots": [
                {"stage": stage, "path": str(root)} for stage, root in source_roots
            ],
            "record_count": len(records),
            "source_size_bytes": sum(item["source_size_bytes"] for item in records),
            "archive_size_bytes": sum(item["archive_size_bytes"] for item in records),
            "records": records,
        }
        manifest_path = temp_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _verify(temp_root, check_sources=True)
        os.replace(temp_root, output.resolve())
        return manifest
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def _verify(output: Path, *, check_sources: bool) -> dict[str, Any]:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("version") != VERSION:
        raise ValueError("unexpected archive manifest schema/version")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != manifest.get("record_count"):
        raise ValueError("archive record count mismatch")
    seen: set[str] = set()
    source_bytes = 0
    archive_bytes = 0
    for record in records:
        archive_relative = record["archive_path"]
        if archive_relative in seen:
            raise ValueError(f"duplicate archive path: {archive_relative}")
        seen.add(archive_relative)
        archived_path = output / archive_relative
        archived = archived_path.read_bytes()
        if len(archived) != record["archive_size_bytes"]:
            raise ValueError(f"archive size mismatch: {archived_path}")
        if _sha256_bytes(archived) != record["archive_sha256"]:
            raise ValueError(f"archive digest mismatch: {archived_path}")
        if record["encoding"] == "gzip_mtime_0":
            raw = gzip.decompress(archived)
        elif record["encoding"] == "identity":
            raw = archived
        else:
            raise ValueError(f"unknown encoding: {record['encoding']}")
        if len(raw) != record["source_size_bytes"]:
            raise ValueError(f"source size mismatch: {archive_relative}")
        if _sha256_bytes(raw) != record["source_sha256"]:
            raise ValueError(f"source digest mismatch: {archive_relative}")
        if record["content_type"] == "application/json":
            if not _is_json(raw):
                raise ValueError(f"invalid archived JSON: {archive_relative}")
        elif record["content_type"] != "text/plain":
            raise ValueError(f"unknown content type: {record['content_type']}")
        if check_sources:
            source = Path(record["source_path"])
            if _sha256_file(source) != record["source_sha256"]:
                raise ValueError(f"live source drift: {source}")
        source_bytes += len(raw)
        archive_bytes += len(archived)
    if source_bytes != manifest.get("source_size_bytes"):
        raise ValueError("aggregate source size mismatch")
    if archive_bytes != manifest.get("archive_size_bytes"):
        raise ValueError("aggregate archive size mismatch")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--check-sources", action="store_true")
    parser.add_argument(
        "--source-root",
        action="append",
        default=[],
        metavar="STAGE=PATH",
        help="Archive only these explicit roots instead of the historical defaults.",
    )
    return parser.parse_args()


def _explicit_source_roots(values: list[str]) -> tuple[tuple[str, Path], ...]:
    roots = []
    seen = set()
    for value in values:
        stage, separator, raw_path = value.partition("=")
        if not separator or not stage or not raw_path:
            raise ValueError("--source-root must use STAGE=PATH")
        if stage in seen or Path(stage).name != stage or stage in {".", ".."}:
            raise ValueError(f"invalid or duplicate archive stage: {stage}")
        seen.add(stage)
        roots.append((stage, Path(raw_path).resolve()))
    return tuple(roots)


def main() -> int:
    args = _parse_args()
    source_roots = _explicit_source_roots(args.source_root)
    if args.verify and source_roots:
        raise ValueError("--source-root cannot be combined with --verify")
    manifest = (
        _verify(args.output.resolve(), check_sources=args.check_sources)
        if args.verify
        else _archive(
            args.output.resolve(),
            source_roots=source_roots if source_roots else SOURCE_ROOTS,
        )
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "record_count": manifest["record_count"],
                "source_size_bytes": manifest["source_size_bytes"],
                "archive_size_bytes": manifest["archive_size_bytes"],
                "verified": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
