from pathlib import Path

import pytest
import torch

from common.hashing import sha256_file
from scripts.materialize_rendered_track_identity_metric import materialize


def _map(path: Path) -> Path:
    torch.save(
        {
            "anchor_ids": torch.tensor([17, 23], dtype=torch.long),
            "anchor_features": torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
            ),
        },
        path,
    )
    return path


def test_identity_metric_is_zero_residual_and_map_bound(tmp_path: Path):
    map_path = _map(tmp_path / "map.pt")
    output = tmp_path / "identity.pt"
    report = materialize(
        map_path=map_path,
        expected_map_sha256=sha256_file(map_path),
        output_path=output,
    )
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert payload["map_path"] == str(map_path.resolve())
    assert payload["map_sha256"] == sha256_file(map_path)
    assert payload["landmark_indices"].tolist() == [0, 1]
    assert payload["metric_config"] == {
        "descriptor_dim": 3,
        "rank": 1,
        "max_residual_norm": 0.0,
    }
    assert all(
        not torch.count_nonzero(value)
        for value in payload["metric_state_dict"].values()
    )
    assert report["output_sha256"] == sha256_file(output)


def test_identity_metric_rejects_wrong_sha_and_existing_output(tmp_path: Path):
    map_path = _map(tmp_path / "map.pt")
    with pytest.raises(ValueError, match="map SHA differs"):
        materialize(
            map_path=map_path,
            expected_map_sha256="0" * 64,
            output_path=tmp_path / "identity.pt",
        )
    output = tmp_path / "identity.pt"
    output.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        materialize(
            map_path=map_path,
            expected_map_sha256=sha256_file(map_path),
            output_path=output,
        )
    assert output.read_bytes() == b"keep"
