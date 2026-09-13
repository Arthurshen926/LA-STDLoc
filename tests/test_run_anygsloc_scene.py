import argparse
import json
from pathlib import Path

import pytest

from scripts.run_anygsloc_scene import command_plan, resolve_prior, run_stage


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
        mapping_raw_keypoints=2048,
        mapping_retained_keypoints=2048,
        mapping_render_supersampling_factor=1,
        joint_track_purification=False,
        purification_minimum_weight=0.05,
        purification_maximum_adjusted_fraction_per_track=0.20,
        cycle_core_coverage_reserve=False,
        reserve_spatial_voxel_size_m=0.50,
        reserve_image_grid_y=6,
        reserve_image_grid_x=8,
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
    assert "--maximum-valid-rows-per-query" not in flattened
    assert "--joint-track-purification" not in flattened


def test_scene_plan_supports_isolated_weak_prior_mapping_factors(tmp_path):
    args = _args(tmp_path)
    args.mapping_raw_keypoints = 3072
    args.joint_track_purification = True
    args.cycle_core_coverage_reserve = True
    plan = command_plan(args, resolve_prior(args))
    observation = plan[0]["command"]
    mapping = next(stage for stage in plan if stage["name"] == "projective_map")[
        "command"
    ]
    assert observation[observation.index("--keypoints") + 1] == "3072"
    assert mapping[mapping.index("--maximum-valid-rows-per-query") + 1] == "2048"
    assert "--joint-track-purification" in mapping
    assert "--cycle-core-coverage-reserve" in mapping


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


def test_unattested_stage_cannot_be_reused(tmp_path):
    output=tmp_path/'result.txt';output.write_text('old')
    with pytest.raises(RuntimeError,match='unattested'):
        run_stage({'name':'synthetic','output':str(output),'command':['unused']},root=tmp_path,env={})


def test_stage_reuse_binds_inputs_and_output_content(tmp_path):
    import sys
    source=tmp_path/'input.txt';source.write_text('one')
    output=tmp_path/'result.txt'
    stage={'name':'synthetic','output':str(output),'command':[sys.executable,'-c',
           'import pathlib,sys;pathlib.Path(sys.argv[2]).write_text(pathlib.Path(sys.argv[1]).read_text())',str(source),'--output',str(output)]}
    stage['command'][2]='import pathlib,sys;pathlib.Path(sys.argv[3]).write_text(pathlib.Path(sys.argv[1]).read_text())'
    first=run_stage(stage,root=tmp_path,env={})
    assert first['status']=='executed'
    assert run_stage(stage,root=tmp_path,env={})['status']=='reused'
    source.write_text('two')
    with pytest.raises(RuntimeError,match='stale'): run_stage(stage,root=tmp_path,env={})
    source.write_text('one');output.write_text('corrupt')
    with pytest.raises(RuntimeError,match='artifacts changed'): run_stage(stage,root=tmp_path,env={})
