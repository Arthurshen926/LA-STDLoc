"""Code identity for auditable mapping-pose evaluation artifacts."""

from __future__ import annotations

from pathlib import Path
import subprocess

from common.hashing import sha256_file


MAPPING_POSE_ENTRYPOINTS = (
    "scripts/evaluate_mapping_cache.py",
    "scripts/compare_mapping_pose_gate.py",
)


def mapping_pose_evaluation_code_identity(*, require_clean: bool = True) -> dict:
    """Bind a pose replay and its decision gate to one clean Git revision."""
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    clean = not status.strip()
    if require_clean and not clean:
        raise RuntimeError("mapping-pose evaluation requires a clean Git worktree")
    return {
        "schema": "lafgs_mapping_pose_evaluation_code",
        "version": 1,
        "repository": str(repository),
        "git_commit": commit,
        "git_worktree_clean": clean,
        "entrypoints": {
            relative: sha256_file(repository / relative)
            for relative in MAPPING_POSE_ENTRYPOINTS
        },
    }
