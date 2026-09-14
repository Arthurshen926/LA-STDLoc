import pytest
import torch

from evidence.v8_fixed_descriptor_replay import prepare_replay, fuse_replay
from evidence.tracks import fuse_projective_anchor_observations


def test_replay_preserves_base_and_completion_weights():
    state = {'v6_mapping_query_names': ['a'], 'v6_mapping_query_bins': torch.tensor([0]),
             'provenance': {}, 'anchor_ids': torch.tensor([0, 1]),
             'anchor_candidate_kind': ['v2_projective_track', 'depth_proposed_projective_completion'],
             'projective_anchor_observations': {
                 'keypoint_index_semantics': 'original_unfiltered_observation_cache_row',
                 'query_indices': torch.tensor([0, 0, 0]), 'keypoint_indices': torch.tensor([2, 0, 1]),
                 'observation_offsets': torch.tensor([0, 2, 3])}}
    cache = {'queries': {'a': {'native_keypoints': torch.zeros(3, 2),
                              'native_scores': torch.tensor([.8, .5, .4])}}}
    graph = {'query_names': ['a'], 'query_bins': torch.tensor([0]),
             'source_keypoint_indices_by_query': [torch.tensor([2, 0])],
             'tracks': {'query_index': torch.tensor([0, 0]), 'keypoint_index': torch.tensor([0, 1]),
                        'confidence': torch.tensor([.25, .5])}}
    replay = prepare_replay(state, cache, graph)
    assert torch.equal(replay['source_index'], torch.tensor([2, 0, 1]))
    assert torch.equal(replay['weights'], torch.tensor([.1, .4, .25]))
    desc = torch.nn.functional.normalize(torch.tensor([[1., 0.], [0., 1.], [1., 1.]]), dim=1)
    expected = fuse_projective_anchor_observations(desc[[2, 0]], torch.tensor([0, 0]),
        detector_weight=torch.tensor([.1, .4]), trim_fraction=.2)
    result = fuse_replay([desc], replay)
    assert torch.equal(result[0], expected)
    assert torch.equal(result[1], desc[1])
    graph['source_keypoint_indices_by_query'] = [torch.tensor([2, 1])]
    with pytest.raises(ValueError, match='base confidence missing'):
        prepare_replay(state, cache, graph)


def test_source_rgb_map_requires_explicit_opt_in(tmp_path):
    from localization.localizer import SparseLocalizer
    path = tmp_path/'diagnostic.pt'
    torch.save({'provenance': {'uses_source_mapping_rgb': True}}, path)
    with pytest.raises(ValueError, match='explicit diagnostic opt-in'):
        SparseLocalizer(path, device='cpu')


def test_cross_bin_reliability_survives_singleton_normalization():
    from evidence.v8_fixed_descriptor_replay import fuse_geometry_balanced
    desc=torch.tensor([[1.,0.],[0.,1.]])
    bins=torch.tensor([0,1]); weight=torch.ones(2); quality=torch.tensor([1.,.1])
    old=fuse_projective_anchor_observations(desc,bins,detector_weight=quality)
    new=fuse_geometry_balanced(desc,bins,weight,quality)
    assert torch.allclose(old,torch.nn.functional.normalize(torch.ones(2),dim=0))
    assert torch.allclose(new,torch.nn.functional.normalize(quality,dim=0))
    assert torch.equal(fuse_geometry_balanced(desc,bins,weight,torch.ones(2)),old)
