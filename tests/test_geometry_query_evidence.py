import numpy as np
import pytest
from scripts.analyze_geometry_query_evidence import analyze


def fixture():
    geometry = {'input': {'map_sha256': 'map'}, 'anchor_count': 3, 'rows': [
        {'anchor_row': 0, 'status': 'internally_consistent', 'full_observation_count': 12},
        {'anchor_row': 1, 'status': 'unstable', 'full_observation_count': 12}]}
    summary = {'input_map_sha256': 'map', 'query_count': 1}
    audit = {'schema': np.array('anygsloc_postsolve_gt_diagnostic_v1'), 'uses_test_gt': True,
             'query_names': np.array(['query']), 'offsets': np.array([0, 3]),
             'anchor_rows': np.array([0, 1, 2]), 'gt_reprojection_px': np.array([1., 10., np.inf]),
             'inlier': np.array([True, True, False])}
    return geometry, summary, audit


def test_groups_do_not_treat_unmeasured_rows_as_unknown_or_correct():
    r = analyze(*fixture())
    assert r['groups']['internally_consistent']['gt_within_4px_percent'] == 100
    assert r['groups']['unstable']['gt_within_4px_percent'] == 0
    assert r['groups']['unknown']['matches'] == 0
    assert r['groups']['not_sampled']['matches'] == 1
    assert not r['deployment_eligible']


def test_wrong_map_cannot_be_joined():
    g, s, a = fixture()
    s['input_map_sha256'] = 'other'
    with pytest.raises(ValueError): analyze(g, s, a)
