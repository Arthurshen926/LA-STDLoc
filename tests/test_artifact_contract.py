import json

import pytest
import torch

from localization_training.artifact_contract import (
    anchor_registry,
    build_contract,
    query_registry,
    verify_contract,
)


def test_query_and_anchor_registry_are_identity_sensitive():
    queries = {
        "queries": {
            "b.png": {
                "native_descriptors": torch.ones(2, 3),
                "native_keypoints": torch.ones(2, 2),
                "native_input_hw": (10, 20),
            },
            "a.png": {
                "native_descriptors": torch.ones(1, 3),
                "native_keypoints": torch.ones(1, 2),
                "native_input_hw": (10, 20),
            },
        }
    }
    first = query_registry(queries)
    queries["queries"] = dict(reversed(list(queries["queries"].items())))
    second = query_registry(queries)
    assert first["ordered_query_sha256"] != second["ordered_query_sha256"]

    state = {
        "anchor_ids": torch.tensor([4, 8]),
        "source_primitive_ids": torch.tensor([1, 1]),
        "track_cluster_ids": torch.tensor([-1, 7]),
        "anchor_type": torch.tensor([0, 1]),
    }
    original = anchor_registry(state)
    state["anchor_ids"] = state["anchor_ids"].flip(0)
    assert original["registry_sha256"] != anchor_registry(state)[
        "registry_sha256"
    ]


def test_contract_fails_fast_when_parent_changes(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact.pt"
    parent = tmp_path / "parent.pt"
    artifact.write_bytes(b"artifact")
    parent.write_bytes(b"parent")
    monkeypatch.setattr(
        "localization_training.artifact_contract.git_worktree_state",
        lambda _: {
            "commit": "deadbeef",
            "status_porcelain": "",
            "diff_sha256": "0" * 64,
        },
    )
    contract = build_contract(
        artifact=artifact,
        kind="fixture",
        run_type="full_chain_rebuild",
        repo_root=tmp_path,
        parents={"parent": parent},
        resolved_config={"seed": 2026},
    )
    verify_contract(json.loads(json.dumps(contract)))
    parent.write_bytes(b"changed")
    with pytest.raises(ValueError, match="parent hash mismatch"):
        verify_contract(contract)
