import json
from argparse import Namespace


def test_write_training_args_snapshot_records_full_locaware_args(tmp_path):
    from train_locaware import write_training_args_snapshot

    dataset = Namespace(
        model_path=str(tmp_path),
        source_path="/data/ShopFacade",
        gaussian_type="2dgs",
        render_items=["RGB", "Feature Map"],
    )
    opt = Namespace(iterations=40000, loc_feature_lr=0.001)
    args = Namespace(
        model_path=str(tmp_path),
        lafgs_diff_pnp_weight=0.0001,
        loc_interval=2,
        save_iterations=[31000, 32000, 33000],
        quiet=False,
    )

    path = write_training_args_snapshot(
        dataset,
        opt,
        args,
        argv=["train_lafgs.py", "--lafgs_diff_pnp_weight", "0.0001"],
    )

    payload = json.loads(path.read_text())
    assert payload["argv"] == ["train_lafgs.py", "--lafgs_diff_pnp_weight", "0.0001"]
    assert payload["dataset"]["gaussian_type"] == "2dgs"
    assert payload["opt"]["iterations"] == 40000
    assert payload["args"]["lafgs_diff_pnp_weight"] == 0.0001
    assert payload["args"]["loc_interval"] == 2
    assert payload["args"]["save_iterations"] == [31000, 32000, 33000]


def test_append_unique_iteration_keeps_checkpoint_args_deduplicated():
    from train_locaware import append_unique_iteration

    values = [31000, 32000, 33000]
    append_unique_iteration(values, 33000)
    append_unique_iteration(values, 34000)

    assert values == [31000, 32000, 33000, 34000]


def test_locaware_full_checkpoint_save_mode_splits_map_and_training_state():
    from train_locaware import should_save_locaware_full_checkpoint

    args = Namespace(
        save_iterations=[10000, 15000],
        loc_full_checkpoint_mode="none",
        loc_full_checkpoint_iterations=[],
        iterations=30000,
    )
    assert not should_save_locaware_full_checkpoint(args, 10000)

    args.loc_full_checkpoint_mode = "save_iterations"
    assert should_save_locaware_full_checkpoint(args, 10000)
    assert not should_save_locaware_full_checkpoint(args, 12000)

    args.loc_full_checkpoint_mode = "final"
    assert not should_save_locaware_full_checkpoint(args, 15000)
    assert should_save_locaware_full_checkpoint(args, 30000)

    args.loc_full_checkpoint_mode = "explicit"
    args.loc_full_checkpoint_iterations = [15000]
    assert not should_save_locaware_full_checkpoint(args, 10000)
    assert should_save_locaware_full_checkpoint(args, 15000)


def test_sfm_from_zero_raw_xyz_geometry_grad_can_be_explicitly_disabled():
    from train_lafgs import _explicit_lafgs_overrides, build_parser, lafgs_defaults

    parser = build_parser()
    default_argv = ["--lafgs_stage_schedule", "sfm_from_zero"]
    default_args = lafgs_defaults(
        parser.parse_args(default_argv),
        explicit_overrides=_explicit_lafgs_overrides(default_argv),
    )
    assert default_args.allow_raw_xyz_geometry_grad is True

    disabled_argv = ["--lafgs_stage_schedule", "sfm_from_zero", "--disallow_raw_xyz_geometry_grad"]
    disabled_args = lafgs_defaults(
        parser.parse_args(disabled_argv),
        explicit_overrides=_explicit_lafgs_overrides(disabled_argv),
    )
    assert disabled_args.allow_raw_xyz_geometry_grad is False
