import numpy as np

from evaluation.evaluator import _gt_reprojection_diagnostics
from evaluation.metrics import summarize_pose_errors


def test_gt_reprojection_diagnostics_counts_invalid_depth_as_incorrect():
    intrinsic = np.asarray(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
    )
    points_3d = np.asarray(
        [[0.0, 0.0, 1.0], [0.01, 0.0, 1.0], [0.0, 0.0, -1.0]]
    )
    points_2d = np.asarray([[50.0, 40.0], [54.0, 40.0], [50.0, 40.0]])
    result = _gt_reprojection_diagnostics(
        points_2d,
        points_3d,
        np.eye(4),
        intrinsic,
        np.asarray([0, 2]),
    )
    assert result == {
        "raw_count": 3,
        "raw_correct_2px": 1,
        "raw_correct_4px": 2,
        "inlier_count": 2,
        "inlier_correct_2px": 1,
        "inlier_correct_4px": 1,
    }


def test_pose_summary_reports_rotation_p90():
    summary = summarize_pose_errors([0.0, 1.0, 10.0], [1.0, 2.0, 3.0])
    assert summary["p90_ae_deg"] == np.percentile([0.0, 1.0, 10.0], 90)


def test_correspondence_telemetry_preserves_pose_results(tmp_path):
    from types import SimpleNamespace as NS
    import torch
    from evaluation.evaluator import evaluate_dataset
    camera = NS(image_name='query', fov_x=1., fov_y=1., pose_w2c=np.eye(4))
    result = NS(
        pose=NS(pose_w2c=np.eye(4), inliers=np.array([0]), diagnostics={'iterations': 1}),
        sparse_features=NS(keypoints=torch.tensor([[-.5,-.5],[9.5,-.5]])),
        matches=NS(keypoint_indices=torch.tensor([0,1]), anchor_indices=torch.tensor([0,1]), scores=torch.ones(2)),
        intrinsic=np.eye(3), match_diagnostics={'duplicate_anchor_count': 0, 'top1_match_count': 2, 'retained_match_count': 2},
        runtime_ms={k:1. for k in ['frontend_ms','matching_ms','ransac_ms','total_ms']})
    dataset = NS(load_image=lambda _: None, valid_mask=lambda _: None)
    calls = []
    def localize(*args, **kwargs):
        calls.append(kwargs)
        return result
    localizer = NS(localize=localize, anchor_xyz=torch.tensor([[0.,0.,1.],[0.,0.,1.]]))
    plain = evaluate_dataset(dataset=dataset, localizer=localizer, cameras=[camera])
    audited = evaluate_dataset(dataset=dataset, localizer=localizer, cameras=[camera], output=tmp_path, audit_correspondences=True)
    assert plain == audited
    assert calls[0] == calls[1]  # No GT or new decisions reach the localizer.
    with np.load(tmp_path/'correspondence_audit.npz') as data:
        assert data['anchor_rows'].tolist() == [0,1]
        assert data['gt_reprojection_px'].tolist() == [0.,10.]
        assert data['inlier'].tolist() == [True,False]
