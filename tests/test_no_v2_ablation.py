import sys
import pytest
import torch
from scripts import materialize_v8_v2_projective_map as materializer


@pytest.mark.parametrize('disabled,expected_rows',[(False,1),(True,3)])
def test_no_v2_is_explicit_and_default_still_filters(tmp_path,monkeypatch,disabled,expected_rows):
    record={'native_keypoints':torch.tensor([[1.,1.],[2.,2.],[3.,3.]]),
            'native_descriptors':torch.eye(3),'native_scores':torch.ones(3),
            'native_K':torch.eye(3),'pose_w2c':torch.eye(4),'native_input_hw':torch.tensor([8,8]),
            'native_valid_mask':torch.ones(8,8,dtype=torch.bool),'native_alpha':torch.ones(8,8),
            'native_depth':torch.ones(8,8)}
    cache={'schema':'render_observation_cache_v2','version':2,'uses_source_mapping_rgb':False,
           'uses_test_queries':False,'queries':{'a':record}}
    path=tmp_path/'cache.pt';torch.save(cache,path)
    audit=tmp_path/'audit.pt';torch.save({},audit)
    args=['materializer','--observation-cache',str(path),'--v2-audit-shards',str(audit),
          '--output-dir',str(tmp_path/'output'),'--device','cpu']
    if disabled:args+=['--disable-v2-row-filter']
    monkeypatch.setattr(sys,'argv',args)
    monkeypatch.setattr(materializer,'_load_v2_rows',lambda *args,**kwargs:[torch.tensor([True,False,False])])
    monkeypatch.setattr(materializer,'build_projective_association_graph',lambda *args,**kwargs:{})
    def inspect(provider,graph,**kwargs):
        assert len(provider.build_view(0).keypoints)==expected_rows
        assert graph['input_sha256'].get('v2_row_filter_disabled',False)==disabled
        assert len(graph['source_keypoint_indices_by_query'][0])==expected_rows
        if disabled:
            assert graph['v2_preassociation_filter']['v2_row_filter_applied'] is False
            assert graph['v2_preassociation_filter']['v2_valid_row_count_before_budget']==1
            assert graph['v2_preassociation_filter']['row_budget_selection']=='all_cached_detector_rows'
            assert provider.payload['v2_preassociation_filter']['v2_row_filter_applied'] is False
        raise RuntimeError('probe complete')
    monkeypatch.setattr(materializer,'reconstruct_projective_anchors',inspect)
    with pytest.raises(RuntimeError,match='probe complete'):materializer.main()
