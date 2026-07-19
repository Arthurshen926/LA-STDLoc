import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from localization_training.episode_sampler import split_support_query_cameras


def load_verifier():
    path = Path(__file__).parents[1] / "scripts" / "verify_lafgs_direct_holdout.py"
    spec = importlib.util.spec_from_file_location("lafgs_holdout_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(names):
    return hashlib.sha256(("\n".join(sorted(names)) + "\n").encode()).hexdigest()


def make_fixture(tmp_path):
    source = tmp_path / "scene"
    source.mkdir()
    names = [f"seq1/frame{i:05d}.png" for i in range(12)]
    (source / "dataset_train.txt").write_text(
        "Visual Landmark Dataset V1\nImageFile, Camera Position [X Y Z W P Q R]\n"
        + "\n".join(f"{name} 0 0 0" for name in names)
        + "\n"
    )
    train, validation = split_support_query_cameras(
        names, query_ratio=0.25, seed=2027, mode="temporal_block"
    )
    state_path = tmp_path / "state.pt"
    torch.save(
        {
            "config": {
                "validation_ratio": 0.25,
                "split_mode": "temporal_block",
                "split_seed": 2026,
                "train_camera_count": len(train),
                "validation_camera_count": len(validation),
                "input_camera_names_sha256": digest(names),
                "train_camera_names_sha256": digest(train),
                "validation_camera_names_sha256": digest(validation),
            }
        },
        state_path,
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "config": {
                    "validation_ratio": 0.25,
                    "split_mode": "temporal_block",
                    "split_seed": 2026,
                    "camera_order": "image_name_lexicographic",
                    "input_camera_names_sha256": digest(names),
                    "query_camera_names_sha256": digest(train),
                    "validation_camera_names_sha256": digest(validation),
                }
            }
        )
    )
    return source, state_path, summary_path


def test_verifies_identical_direct_holdout(tmp_path):
    verifier = load_verifier()
    source, state_path, summary_path = make_fixture(tmp_path)

    report = verifier.verify(state_path, summary_path, source)

    assert report["verified"] is True
    assert report["split"]["train_camera_count"] == 9
    assert report["split"]["validation_camera_count"] == 3


def test_rejects_detector_camera_set_mismatch(tmp_path):
    verifier = load_verifier()
    source, state_path, summary_path = make_fixture(tmp_path)
    summary = json.loads(summary_path.read_text())
    summary["config"]["query_camera_names_sha256"] = "wrong"
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="direct holdout mismatch"):
        verifier.verify(state_path, summary_path, source)
