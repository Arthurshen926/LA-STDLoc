#!/usr/bin/env python3
"""Fail-closed workspace and stage manifests for the P7 pair full chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.hashing import sha256_file


SCHEMA = "lafgs_pair_policy_fullchain_workspace"


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def preflight_workspace(
    *, root: Path, allowed_inputs: list[Path], report_path: Path
) -> dict:
    root = root.expanduser().resolve()
    report_path = report_path.expanduser().resolve()
    allowed = {path.expanduser().resolve() for path in allowed_inputs}
    if any(not path.is_file() for path in allowed):
        missing = next(path for path in allowed if not path.is_file())
        raise FileNotFoundError(missing)
    root.mkdir(parents=True, exist_ok=True)
    present = {
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != report_path
    }
    unexpected = sorted(present - allowed)
    outside = sorted(path for path in allowed if not _inside(path, root))
    if outside:
        raise ValueError(
            f"Allowed preflight input is outside output root: {outside[0]}"
        )
    if unexpected:
        raise RuntimeError(
            "Pair full-chain output is not empty/contract-only: "
            + ", ".join(str(path) for path in unexpected[:5])
        )
    report = {
        "schema": SCHEMA,
        "version": 1,
        "kind": "empty_output_preflight",
        "uses_test_queries": False,
        "valid": True,
        "root": str(root),
        "allowed_inputs": {
            str(path.relative_to(root)): sha256_file(path) for path in sorted(allowed)
        },
        "scientific_artifact_count": 0,
        "resume_policy": "quarantine_failed_root_and_restart_from_empty",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _named_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, found {value!r}")
        name, path = value.split("=", 1)
        if not name or name in result:
            raise ValueError(f"Artifact name is empty or duplicated: {name!r}")
        result[name] = Path(path).expanduser().resolve()
    return result


def _named_values(values: list[str], *, label: str) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME={label}, found {value!r}")
        name, resolved = value.split("=", 1)
        if not name or name in result:
            raise ValueError(f"Input name is empty or duplicated: {name!r}")
        result[name] = resolved
    return result


def _validate_semantic_input(name: str, path: Path) -> None:
    """Reject a hash-locked decision artifact whose decision is not a Go."""
    if name != "mechanism_gate":
        return
    payload = json.loads(path.read_text())
    if (
        payload.get("schema") != "lafgs_pair_policy_mechanism_gate"
        or payload.get("version") != 2
        or payload.get("uses_test_queries") is not False
        or payload.get("valid") is not True
        or payload.get("mechanism_gate_passed") is not True
        or payload.get("advance_to_full_pipeline_pose") is not True
        or payload.get("decision") != "GO_TO_PIPELINE"
        or not payload.get("gates")
        or not all(payload["gates"].values())
    ):
        raise ValueError("Locked mechanism gate does not authorize fullchain")


def lock_inputs(
    *,
    root: Path,
    inputs: dict[str, Path],
    expected_sha256: dict[str, str],
    parent: Path,
    report_path: Path,
) -> dict:
    root = root.expanduser().resolve()
    parent = parent.expanduser().resolve()
    report_path = report_path.expanduser().resolve()
    if set(inputs) != set(expected_sha256):
        raise ValueError("Every locked input requires exactly one expected SHA-256")
    if not parent.is_file() or json.loads(parent.read_text()).get("valid") is not True:
        raise ValueError("Locked inputs require a valid fresh-workspace preflight")
    records = {}
    for name, path in sorted(inputs.items()):
        path = path.expanduser().resolve()
        expected = expected_sha256[name].strip().lower()
        if len(expected) != 64 or any(
            character not in "0123456789abcdef" for character in expected
        ):
            raise ValueError(f"Invalid expected SHA-256 for {name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Locked input SHA-256 differs for {name}")
        _validate_semantic_input(name, path)
        records[name] = {
            "path": str(path),
            "expected_sha256": expected,
            "sha256": actual,
            "size_bytes": path.stat().st_size,
            "inside_output_root": _inside(path, root),
        }
    report = {
        "schema": SCHEMA,
        "version": 1,
        "kind": "locked_external_inputs",
        "uses_test_queries": False,
        "valid": True,
        "root": str(root),
        "inputs": records,
        "parents": [{"path": str(parent), "sha256": sha256_file(parent)}],
        "silent_resume_authorized": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def write_stage_manifest(
    *,
    root: Path,
    stage: str,
    artifacts: dict[str, Path],
    parents: list[Path],
    report_path: Path,
) -> dict:
    root = root.expanduser().resolve()
    report_path = report_path.expanduser().resolve()
    if not stage:
        raise ValueError("Stage name must be non-empty")
    if not artifacts:
        raise ValueError("A stage manifest requires at least one artifact")
    records = {}
    for name, path in sorted(artifacts.items()):
        if not path.is_file():
            raise FileNotFoundError(path)
        if not _inside(path, root):
            raise ValueError(f"Scientific artifact escapes output root: {path}")
        records[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    parent_records = []
    for path in parents:
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text())
        if payload.get("valid") is not True:
            raise ValueError(f"Parent manifest is not valid: {path}")
        parent_records.append({"path": str(path), "sha256": sha256_file(path)})
    report = {
        "schema": SCHEMA,
        "version": 1,
        "kind": "stage_sha256_manifest",
        "stage": stage,
        "uses_test_queries": False,
        "valid": True,
        "root": str(root),
        "artifacts": records,
        "parents": parent_records,
        "silent_resume_authorized": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def verify_stage_manifest(path: Path) -> dict:
    path = path.expanduser().resolve()
    report = json.loads(path.read_text())
    if (
        report.get("schema") != SCHEMA
        or report.get("version") != 1
        or report.get("kind") not in {"stage_sha256_manifest", "locked_external_inputs"}
        or report.get("uses_test_queries") is not False
        or report.get("valid") is not True
        or report.get("silent_resume_authorized") is not False
    ):
        raise ValueError("Unsupported or invalid pair full-chain stage manifest")
    records = (
        report.get("inputs", {})
        if report.get("kind") == "locked_external_inputs"
        else report.get("artifacts", {})
    )
    for record in records.values():
        artifact = Path(record["path"]).resolve()
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(record["size_bytes"])
            or sha256_file(artifact) != record["sha256"]
        ):
            raise ValueError(f"Stage artifact differs from manifest: {artifact}")
    for record in report.get("parents", []):
        parent = Path(record["path"]).resolve()
        if not parent.is_file() or sha256_file(parent) != record["sha256"]:
            raise ValueError(f"Parent manifest differs: {parent}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--root", type=Path, required=True)
    preflight.add_argument("--allow-input", type=Path, action="append", default=[])
    preflight.add_argument("--output", type=Path, required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--root", type=Path, required=True)
    manifest.add_argument("--stage", required=True)
    manifest.add_argument("--artifact", action="append", default=[])
    manifest.add_argument("--parent-manifest", type=Path, action="append", default=[])
    manifest.add_argument("--output", type=Path, required=True)
    lock = subparsers.add_parser("lock-inputs")
    lock.add_argument("--root", type=Path, required=True)
    lock.add_argument("--input", action="append", default=[])
    lock.add_argument("--expected-sha256", action="append", default=[])
    lock.add_argument("--parent-manifest", type=Path, required=True)
    lock.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight_workspace(
            root=args.root,
            allowed_inputs=args.allow_input,
            report_path=args.output,
        )
    elif args.command == "manifest":
        result = write_stage_manifest(
            root=args.root,
            stage=args.stage,
            artifacts=_named_paths(args.artifact),
            parents=args.parent_manifest,
            report_path=args.output,
        )
    elif args.command == "lock-inputs":
        result = lock_inputs(
            root=args.root,
            inputs=_named_paths(args.input),
            expected_sha256=_named_values(args.expected_sha256, label="SHA256"),
            parent=args.parent_manifest,
            report_path=args.output,
        )
    else:
        result = verify_stage_manifest(args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
