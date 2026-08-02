from scripts.evaluate_offtheshelf_gaussian_prior import _summary


def test_prior_quality_summary_reports_distribution():
    rows = [
        {"psnr_db": 20.0, "ssim": 0.7, "lpips": 0.3},
        {"psnr_db": 30.0, "ssim": 0.9, "lpips": 0.1},
    ]
    report = _summary(rows)
    assert report["query_count"] == 2
    assert report["psnr_db"]["mean"] == 25.0
    assert report["ssim"]["median"] == 0.8
    assert report["lpips"]["mean"] == 0.2
