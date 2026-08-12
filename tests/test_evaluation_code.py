from pathlib import Path

from common.evaluation_code import (
    frontend_descriptor_evaluation_code_identity,
    mapping_pose_evaluation_code_identity,
)
from common.hashing import sha256_file


def test_mapping_pose_code_identity_binds_commit_and_entrypoint_bytes() -> None:
    identity = mapping_pose_evaluation_code_identity(require_clean=False)
    repository = Path(identity["repository"])

    assert identity["schema"] == "lafgs_mapping_pose_evaluation_code"
    assert len(identity["git_commit"]) == 40
    assert isinstance(identity["git_worktree_clean"], bool)
    assert identity["entrypoints"] == {
        relative: sha256_file(repository / relative)
        for relative in (
            "scripts/evaluate_mapping_cache.py",
            "scripts/compare_mapping_pose_gate.py",
        )
    }


def test_frontend_descriptor_code_identity_binds_audit_and_gate_bytes() -> None:
    identity = frontend_descriptor_evaluation_code_identity(require_clean=False)
    repository = Path(identity["repository"])
    assert identity["schema"] == "lafgs_frontend_descriptor_evaluation_code"
    assert identity["entrypoints"] == {
        relative: sha256_file(repository / relative)
        for relative in (
            "map_learning/frontend_upper_bound.py",
            "scripts/audit_frontend_upper_bound.py",
            "scripts/compare_frontend_descriptor_arm_b.py",
        )
    }
