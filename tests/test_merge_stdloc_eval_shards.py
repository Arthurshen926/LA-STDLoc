import json

import pytest

from scripts.merge_stdloc_eval_shards import main


def _row(name, te):
    return {
        "image_name": name,
        "sparse_TE": te,
        "sparse_AE": 1.0,
        "sparse": {"sparse_diag_x": te},
    }


def test_merge_script_rejects_duplicate_camera_names(tmp_path, monkeypatch):
    directories = []
    for index in range(2):
        directory = tmp_path / str(index)
        directory.mkdir()
        (directory / "results.json").write_text(json.dumps([_row("same", index)]))
        directories.append(directory)
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps(["same"]))
    monkeypatch.setattr(
        "sys.argv",
        [
            "merge",
            "--results",
            *(str(path) for path in directories),
            "--expected-lists",
            str(expected),
            "--output",
            str(tmp_path / "summary.json"),
        ],
    )
    with pytest.raises(ValueError, match="duplicate image names"):
        main()
