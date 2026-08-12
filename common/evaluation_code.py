"""Code identity for auditable mapping-pose evaluation artifacts."""

from __future__ import annotations

from pathlib import Path
import subprocess

from common.hashing import sha256_file


MAPPING_POSE_ENTRYPOINTS = (
    "map_learning/equal_energy_descriptor_factor.py",
    "scripts/materialize_equal_energy_descriptor_factor.py",
    "scripts/evaluate_mapping_cache.py",
    "scripts/compare_mapping_pose_gate.py",
)
FRONTEND_DESCRIPTOR_ENTRYPOINTS = (
    "map_learning/frontend_upper_bound.py",
    "scripts/audit_frontend_upper_bound.py",
    "scripts/compare_frontend_descriptor_arm_b.py",
)
FRONTEND_DETECTOR_ENTRYPOINTS = (
    "map_learning/frontend_upper_bound.py",
    "scripts/audit_frontend_upper_bound.py",
    "scripts/compare_frontend_detector_arm_a.py",
)


def _evaluation_code_identity(
    *,
    schema: str,
    entrypoints: tuple[str, ...],
    require_clean: bool,
    label: str,
) -> dict:
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
        raise RuntimeError(f"{label} requires a clean Git worktree")
    return {
        "schema": schema,
        "version": 1,
        "repository": str(repository),
        "git_commit": commit,
        "git_worktree_clean": clean,
        "entrypoints": {
            relative: sha256_file(repository / relative) for relative in entrypoints
        },
    }


def mapping_pose_evaluation_code_identity(*, require_clean: bool = True) -> dict:
    """Bind a pose replay and its decision gate to one clean Git revision."""
    return _evaluation_code_identity(
        schema="lafgs_mapping_pose_evaluation_code",
        entrypoints=MAPPING_POSE_ENTRYPOINTS,
        require_clean=require_clean,
        label="mapping-pose evaluation",
    )


def frontend_descriptor_evaluation_code_identity(*, require_clean: bool = True) -> dict:
    """Bind a descriptor audit and its gate to one clean Git revision."""
    return _evaluation_code_identity(
        schema="lafgs_frontend_descriptor_evaluation_code",
        entrypoints=FRONTEND_DESCRIPTOR_ENTRYPOINTS,
        require_clean=require_clean,
        label="frontend-descriptor evaluation",
    )


def frontend_detector_evaluation_code_identity(*, require_clean: bool = True) -> dict:
    """Bind a detector audit and its decision gate to one clean revision."""
    return _evaluation_code_identity(
        schema="lafgs_frontend_detector_evaluation_code",
        entrypoints=FRONTEND_DETECTOR_ENTRYPOINTS,
        require_clean=require_clean,
        label="frontend-detector evaluation",
    )
