import torch

from scripts.verify_geometry_teacher_statistics import verify_statistics


def test_verify_track_statistics_rejects_identity_mismatch(tmp_path):
    path = tmp_path / "statistics.pt"
    torch.save(
        {
            "diagnostics": {
                "geometry_teacher_identity_mode": "map_top1",
            }
        },
        path,
    )
    try:
        verify_statistics(path, "track_first")
    except ValueError as error:
        assert "identity mismatch" in str(error)
    else:
        raise AssertionError("Mislabeled teacher statistics were accepted")


def test_verify_track_statistics_requires_positive_counts(tmp_path):
    path = tmp_path / "statistics.pt"
    torch.save(
        {
            "diagnostics": {
                "geometry_teacher_identity_mode": "track_first",
                "track_count": 12,
                "geometry_teacher_triangulated_track_count": 8,
                "geometry_teacher_high_confidence_track_count": 4,
                "geometry_teacher_assigned_landmark_count": 3,
            }
        },
        path,
    )
    summary = verify_statistics(path, "track_first")
    assert summary["track_count"] == 12
