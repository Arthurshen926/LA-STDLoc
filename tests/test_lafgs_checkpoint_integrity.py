from pathlib import Path

from train_lafgs_map import _checkpoint_integrity


def test_checkpoint_integrity_reports_missing_intermediate_checkpoint(tmp_path):
    for step in (0, 500, 1500):
        (Path(tmp_path) / f"{step}_lafgs_map_state.pt").touch()

    report = _checkpoint_integrity(tmp_path, [0, 500, 1000, 1500, 1500])

    assert report == {
        "requested_steps": [0, 500, 1000, 1500],
        "saved_steps": [0, 500, 1500],
        "missing_steps": [1000],
        "complete": False,
    }
