import argparse
import json
from pathlib import Path

import pytest

from scripts.run_anygsloc_scene import command_plan, resolve_prior


def _args(tmp_path: Path) -> argparse.Namespace:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    ply = tmp_path / "point_cloud.ply"
    ply.write_bytes(b"ply\n")
    return argparse.Namespace(
        scene="scene",
        dataset=dataset,
        images="processed",
        prior_manifest=None,
        gaussian_ply=ply,
        gaussian_type="2dgs",
        sh_degree=3,
        white_background=False,
        output=tmp_path / "output",
        config=Path("configs/anygsloc_mainline.yaml").resolve(),
        seed=2026,
        audit_shards=2,
        cpu_threads=4,
        triangulation_workers=2,
    )


def test_scene_plan_has_only_mapping_base_and_test_evaluation(tmp_path):
    args = _args(tmp_path)
    prior = resolve_prior(args)
    plan = command_plan(args, prior)
    assert [stage["name"] for stage in plan] == [
        "observations",
        "v2_audit_000",
        "v2_audit_001",
        "projective_map",
        "base_evaluation",
    ]
    flattened = " ".join(part for stage in plan for part in stage["command"])
    assert "feedback" not in flattened
    assert "training" not in flattened
    assert "--render-only" in flattened
    assert "--split test" in flattened


def test_normalized_manifest_controls_renderer_contract(tmp_path):
    args = _args(tmp_path)
    digest = __import__("hashlib").sha256(args.gaussian_ply.read_bytes()).hexdigest()
    manifest = tmp_path / "rgb_prior_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "prior_kind": "rgb_only",
                "prior_training_used_feature_loss": False,
                "exported_ply": str(args.gaussian_ply),
                "exported_ply_sha256": digest,
                "gaussian_type": "3dgs",
                "sh_degree": 0,
            }
        )
    )
    args.prior_manifest = manifest
    args.gaussian_ply = None
    args.gaussian_type = None
    args.sh_degree = None
    prior = resolve_prior(args)
    assert prior["gaussian_type"] == "3dgs"
    assert prior["sh_degree"] == 0
    assert prior["white_background"] is False


def test_feature_trained_prior_fails_closed(tmp_path):
    args = _args(tmp_path)
    digest = __import__("hashlib").sha256(args.gaussian_ply.read_bytes()).hexdigest()
    manifest = tmp_path / "rgb_prior_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "prior_kind": "rgb_only",
                "prior_training_used_feature_loss": True,
                "exported_ply": str(args.gaussian_ply),
                "exported_ply_sha256": digest,
                "gaussian_type": "2dgs",
                "sh_degree": 3,
            }
        )
    )
    args.prior_manifest = manifest
    args.gaussian_ply = None
    with pytest.raises(ValueError, match="feature-trained"):
        resolve_prior(args)
