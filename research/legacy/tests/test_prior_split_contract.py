import pytest
from priors.i3dgs_adapter import _read_mapping_names


def test_prior_training_rejects_overlap(tmp_path):
    (tmp_path/'dataset_train.txt').write_text('seq1/a.png 0 0 0 1 0 0 0\n')
    (tmp_path/'dataset_test.txt').write_text('seq1/a.png 0 0 0 1 0 0 0\n')
    with pytest.raises(ValueError,match='overlap'): _read_mapping_names(tmp_path)


def test_disjoint_split_is_unchanged(tmp_path):
    (tmp_path/'dataset_train.txt').write_text('seq1/a.png 0 0 0 1 0 0 0\n')
    (tmp_path/'dataset_test.txt').write_text('seq2/a.png 0 0 0 1 0 0 0\n')
    assert _read_mapping_names(tmp_path) == ['seq1/a.png']
