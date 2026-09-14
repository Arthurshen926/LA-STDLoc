from pathlib import Path

import pytest

from common.hashing import sha256_file
from scripts.evaluate import _input_artifact_contract


@pytest.mark.parametrize("role", ["metric_state", "context_state"])
def test_evaluation_artifact_contract_binds_exact_inputs(
    tmp_path: Path, role: str
) -> None:
    map_path = tmp_path / "map.pt"
    descriptor_path = tmp_path / "descriptor.pt"
    map_path.write_bytes(b"map")
    descriptor_path.write_bytes(b"descriptor")
    contract = _input_artifact_contract(
        map_path, descriptor_path, descriptor_role=role
    )
    assert contract["map"] == {
        "path": str(map_path.resolve()),
        "sha256": sha256_file(map_path),
    }
    assert contract["descriptor_state"] == {
        "role": role,
        "path": str(descriptor_path.resolve()),
        "sha256": sha256_file(descriptor_path),
    }

