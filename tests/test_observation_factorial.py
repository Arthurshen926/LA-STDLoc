import importlib.util
from pathlib import Path

import pytest
import torch


spec = importlib.util.spec_from_file_location(
    'observation_factorial', Path(__file__).resolve().parents[1] / 'tools/observation_factorial.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_representative_is_mapping_name_order_not_csr_order():
    state = dict(v6_mapping_query_names=['z', 'a'], projective_anchor_observations=dict(
        observation_offsets=torch.tensor([0, 2]), query_indices=torch.tensor([0, 1]),
        keypoint_indices=torch.tensor([0, 0])))
    records = {'z': {'native_descriptors': torch.tensor([[1., 0.]])},
               'a': {'native_descriptors': torch.tensor([[0., 2.]])}}
    assert torch.equal(module.representative_features(state, records), torch.tensor([[0., 1.]]))
    state['projective_anchor_observations']['query_indices'] = torch.tensor([1, 0])
    assert torch.equal(module.representative_features(state, records), torch.tensor([[0., 1.]]))


def test_empty_anchor_fails_instead_of_inventing_descriptor():
    state = dict(v6_mapping_query_names=[], projective_anchor_observations=dict(
        observation_offsets=torch.tensor([0, 0]), query_indices=torch.tensor([]),
        keypoint_indices=torch.tensor([])))
    with pytest.raises(ValueError, match='without observations'):
        module.representative_features(state, {})
