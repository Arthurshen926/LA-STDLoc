"""One-shot mechanical moves after extracting the frozen dependency closure."""

import hashlib
import json
from pathlib import Path
import shutil

from extract_frozen_core import ROOT, apply_files, rewrite_source


def main():
    legacy = ROOT / "research/legacy"
    legacy.mkdir(exist_ok=False)
    moved = []
    for name in [
        "common",
        "data",
        "evaluation",
        "evidence",
        "features",
        "localization",
        "map_learning",
        "priors",
        "topology",
        "visualization",
        "scripts",
        "configs",
        "tests",
        "paper_baseline",
        "benchmarks",
        "third_party_patches",
        "paper",
        "docs",
    ]:
        shutil.move(str(ROOT / name), str(legacy / name))
        moved.append(name)
    (ROOT / "docs").mkdir()
    shutil.move(str(legacy / "docs/release"), str(ROOT / "docs/release"))
    shutil.move(str(ROOT / "tests_mainline_staging"), str(ROOT / "tests"))
    online = ROOT / "anygsloc/online"
    online.mkdir()
    for name in ["runtime", "matching", "pose"]:
        shutil.move(str(ROOT / f"anygsloc/{name}.py"), str(online / f"{name}.py"))
    shutil.move(
        str(ROOT / "anygsloc/mapping.py"), str(ROOT / "anygsloc/mapping/compact.py")
    )
    files = {}
    for p in (ROOT / "anygsloc").rglob("*.py"):
        text = p.read_text()
        # Extractor has already rewritten its frozen kernels. Rewrite the
        # previous release wrapper's legacy imports explicitly after moving.
        for old, new in {
            "from features.superpoint": "from anygsloc.features.superpoint",
            "from topology.sufficiency_selector": "from anygsloc.mapping.selector",
            "from topology.adaptive_distillation": "from anygsloc.mapping.pose_evidence",
            "from topology.dynamic_reserve": "from anygsloc.mapping.observability",
            "from scripts.materialize_purity_candidates": "from anygsloc.mapping.subset",
            "from anygsloc.runtime": "from anygsloc.online.runtime",
            "from anygsloc.matching": "from anygsloc.online.matching",
            "from anygsloc.pose": "from anygsloc.online.pose",
        }.items():
            text = text.replace(old, new)
        if text != p.read_text():
            files[str(p.relative_to(ROOT))] = text
    test = (legacy / "tests/test_release_runtime.py").read_text()
    for old, new in [
        ("anygsloc.runtime", "anygsloc.online.runtime"),
        ("anygsloc.matching", "anygsloc.online.matching"),
        ("anygsloc.pose", "anygsloc.online.pose"),
    ]:
        test = test.replace(old, new)
    files["tests/test_release_runtime.py"] = test
    files["anygsloc/online/__init__.py"] = (
        '"""Frozen sparse online localization; no mapping imports."""\n'
    )
    files["anygsloc/mapping/__init__.py"] = (
        '"""Frozen offline map construction and compact selection."""\n'
    )
    files["research/repository_moves.json"] = (
        json.dumps({"moved_roots": moved, "legacy_root": "research/legacy"}, indent=2)
        + "\n"
    )
    apply_files(files)


if __name__ == "__main__":
    main()
