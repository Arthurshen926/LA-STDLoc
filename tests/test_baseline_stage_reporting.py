import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('suite_reporting',
    Path(__file__).resolve().parents[1]/'tools/report_cambridge_suite_v2.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_valid_sparse_pose_survives_failed_refinement_fallback():
    pose=np.eye(4).tolist()
    row=dict(sparse=dict(inliers=10,pose_w2c=pose),dense=[dict(inliers=0,pose_w2c=pose)])
    assert module.stage_success(row,'coarse')
    assert module.stage_success(row,'fine')


def test_invalid_sparse_pose_not_promoted_without_new_inliers():
    pose=np.eye(4).tolist()
    row=dict(sparse=dict(inliers=0,pose_w2c=pose),dense=[dict(inliers=0,pose_w2c=pose)])
    assert not module.stage_success(row,'fine')


def test_failed_solver_identity_does_not_inherit_unrelated_pose_success():
    pose=np.eye(4);pose[0,3]=5
    row=dict(sparse=dict(inliers=10,pose_w2c=pose.tolist()),
             dense=[dict(inliers=0,pose_w2c=np.eye(4).tolist())])
    assert not module.stage_success(row,'fine')
