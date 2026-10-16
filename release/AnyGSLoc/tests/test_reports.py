import pytest

from anygsloc.pipeline.io import complete, write_json
from reproduction.report import quality_report, timing_report


def test_psnr_reports_use_scene_equal_means_and_reject_missing_views(tmp_path):
    config = {"scenes": {"a": {"mapping_count": 2}, "b": {"mapping_count": 1}}}
    for scene, values in {"a": [20.0, 40.0], "b": [60.0]}.items():
        path = tmp_path / scene / "psnr"
        write_json(
            path / "records.json",
            [
                {"image_name": str(i), "psnr_db": value}
                for i, value in enumerate(values)
            ],
        )
        complete(path, ["records.json"])
    report = quality_report(tmp_path, config, ["a", "b"])
    assert report["PSNR_dB"] == 45.0
    assert report["images"] == 3
    config["scenes"]["b"]["mapping_count"] = 2
    with pytest.raises(ValueError, match="incomplete mapping views"):
        quality_report(tmp_path, config, ["a", "b"])


def test_empty_timing_campaign_cannot_be_reported_as_complete(tmp_path):
    with pytest.raises(ValueError, match="no completed timing rounds"):
        timing_report(tmp_path)
