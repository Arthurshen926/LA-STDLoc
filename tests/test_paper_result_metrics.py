import numpy as np
import pytest
from scripts.build_anygsloc_paper_tables import numbers


def test_thresholds_tail_and_throughput_are_not_conflated():
    rows=[{'translation_error_cm':1.,'rotation_error_deg':.5,'total_ms':20.},
          {'translation_error_cm':100.,'rotation_error_deg':3.,'total_ms':200.}]
    out=numbers(rows)
    assert out['R1_1']==0
    assert out['R2_2']==50
    assert out['cat100']==1
    assert out['te']==50.5
    assert out['ms']==110
    assert np.isclose(out['hz'],1000/110)
    assert out['deadline100']==50


def test_nonfinite_pose_errors_fail_closed():
    with pytest.raises(ValueError,match='nonfinite'):
        numbers([{'translation_error_cm':float('nan'),'rotation_error_deg':0.,'total_ms':20.}])
