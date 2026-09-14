import pytest
from scripts.prepare_rectified_prior_replay import safe_flat_registry


def test_registry_rejects_ambiguous_or_escaping_names():
    assert safe_flat_registry(['seq1/a.png']) == {'seq1__a.png': 'seq1/a.png'}
    for names in [['seq1/a.png', 'seq1__a.png'], ['../a.png'], ['/a.png']]:
        with pytest.raises(ValueError): safe_flat_registry(names)
