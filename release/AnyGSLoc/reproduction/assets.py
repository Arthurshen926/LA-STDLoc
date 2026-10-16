"""Inspect required assets, verify downloaded files, or materialize local bindings."""

import argparse
import json
import shutil
from pathlib import Path

from anygsloc.artifacts import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["check", "link", "copy"])
    p.add_argument("--root", type=Path, default=Path("assets"))
    p.add_argument(
        "--bindings",
        type=Path,
        help="Local JSON mapping portable paths to existing source files",
    )
    p.add_argument(
        "--kind",
        action="append",
        help="Restrict by asset kind; repeat to select multiple kinds",
    )
    p.add_argument(
        "--hash",
        action="store_true",
        help="Read complete files and verify SHA-256 where recorded",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Required for copying; otherwise show byte estimate",
    )
    args = p.parse_args()
    catalog = json.loads((Path(__file__).parent / "assets.json").read_text())["assets"]
    selected = [a for a in catalog if not args.kind or a["kind"] in args.kind]
    if args.action != "check" and not args.bindings:
        p.error("--bindings is required")
    bindings = json.loads(args.bindings.read_text())["assets"] if args.bindings else {}
    missing = []
    changed = []
    done = 0
    for item in sorted(selected, key=lambda v: (len(Path(v["path"]).parts), v["path"])):
        target = args.root / item["path"]
        if args.action != "check" and not target.exists():
            source = Path(bindings[item["path"]])
            if not source.exists():
                missing.append(item["path"])
                continue
            if args.action == "copy" and not args.execute:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if args.action == "link":
                target.symlink_to(source.resolve(), target_is_directory=source.is_dir())
            elif source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
        if not target.exists():
            missing.append(item["path"])
            continue
        if "bytes" in item and target.stat().st_size != item["bytes"]:
            changed.append(item["path"])
        if args.hash and "sha256" in item and sha256(target) != item["sha256"]:
            changed.append(item["path"])
        done += 1
    if args.action == "copy" and not args.execute:
        print(
            json.dumps(
                {
                    "files": len(selected),
                    "listed_file_bytes": sum(a.get("bytes", 0) for a in selected),
                    "directory_contents_additional": True,
                    "executed": False,
                },
                indent=2,
            )
        )
        return
    print(
        json.dumps(
            {"checked": done, "missing": missing, "changed": sorted(set(changed))},
            indent=2,
        )
    )
    if missing or changed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
