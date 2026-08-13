from pathlib import Path

import pytest

from scripts import run_pipeline


def _arguments(tmp_path: Path) -> list[str]:
    return [
        "--dataset",
        str(tmp_path / "dataset"),
        "--prior",
        str(tmp_path / "prior"),
        "--output",
        str(tmp_path / "run"),
        "--gaussian-type",
        "2dgs",
    ]


def test_public_pipeline_defaults_to_no_test_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = {}

    def capture(args, *, experimental_factors):
        observed["evaluate"] = args.evaluate
        observed["factors"] = experimental_factors
        return {}

    monkeypatch.setattr(run_pipeline, "run", capture)
    run_pipeline.main(_arguments(tmp_path))
    assert observed["evaluate"] is False
    assert observed["factors"] == {
        "joint_keypoints": None,
        "mapping_keypoints": None,
        "surface_supported_tracks": False,
    }
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "factor",
    [
        ["--keypoints", "1024"],
        ["--mapping-keypoints", "1024"],
        ["--surface-supported-tracks"],
    ],
)
def test_experimental_factor_and_test_evaluation_fail_before_any_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factor: list[str],
) -> None:
    monkeypatch.setattr(
        run_pipeline,
        "run",
        lambda *_args, **_kwargs: pytest.fail("pipeline run must not start"),
    )
    with pytest.raises(SystemExit) as error:
        run_pipeline.main([*_arguments(tmp_path), *factor, "--evaluate"])
    assert error.value.code == 2
    assert not (tmp_path / "run").exists()


def test_pipeline_rejects_stale_or_partial_root_before_any_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    stale = output / "partial.pt"
    stale.write_bytes(b"partial")
    monkeypatch.setattr(
        run_pipeline,
        "run",
        lambda *_args, **_kwargs: pytest.fail("pipeline run must not start"),
    )
    with pytest.raises(SystemExit) as error:
        run_pipeline.main(_arguments(tmp_path))
    assert error.value.code == 2
    assert stale.read_bytes() == b"partial"


def test_joint_and_mapping_density_fail_before_config_materialization(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit) as error:
        run_pipeline.main(
            [
                *_arguments(tmp_path),
                "--keypoints",
                "1024",
                "--mapping-keypoints",
                "1024",
            ]
        )
    assert error.value.code == 2
    assert not (tmp_path / "run").exists()
