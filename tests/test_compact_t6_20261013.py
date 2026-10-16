import copy, tempfile, unittest
import io, pickle
from pathlib import Path
import numpy as np
import torch
from experiments import compact_t6_20261013 as D


def fixture():
    n=8;q=3
    xyz=torch.stack((torch.linspace(-1,1,n),torch.zeros(n),torch.full((n,),5.)),1)
    f=torch.nn.functional.normalize(torch.arange(n*4,dtype=torch.float32).reshape(n,4)+1,dim=1)
    part=dict(query_names=['a','b','c'],query_bins=torch.arange(q),uses_test_queries=False,uses_source_mapping_rgb=False,
       anchor_xyz=xyz,anchor_features=f,anchor_position_covariance=torch.eye(3)[None].repeat(n,1,1)*.01,
       identity_reliability=torch.linspace(.9,.8,n),geometry_reliability=torch.full((n,),.95),
       projective_anchor_observations=dict(observation_offsets=torch.arange(0,(n+1)*q,q),
           query_indices=torch.arange(q).repeat(n),keypoint_indices=torch.arange(n).repeat_interleave(q)))
    cons=dict(v2_enabled=False,mapping_only=True,parts=[part])
    K=torch.tensor([[100.,0,100],[0,100.,100],[0,0,1.]])
    cache={k:dict(native_scores=torch.linspace(.1,1,n),native_K=K,pose_w2c=torch.eye(4),native_keypoints=torch.stack((torch.linspace(80,120,n),torch.full((n,),100.)),1),native_input_hw=(200,200)) for k in part['query_names']}
    return cons,cache

class CompactT6Tests(unittest.TestCase):
    def test_cpu_mask_unpickle_preserves_values_dtype_shape(self):
        source={'q':[torch.tensor([[1,0]],dtype=torch.uint8),torch.tensor([[False,True]]),torch.tensor([[.25,.75]],dtype=torch.float32)]}
        restored=D.CPUUnpickler(io.BytesIO(pickle.dumps(source))).load()
        for x,y in zip(source['q'],restored['q']):
            self.assertEqual(x.dtype,y.dtype);self.assertEqual(x.shape,y.shape);self.assertTrue(torch.equal(x,y));self.assertEqual(y.device.type,'cpu')
    def test_original_unfiltered_rows_preserved(self):
        cons,cache=fixture();state=D.off_state(cons,['a','b','c'])
        self.assertTrue(torch.equal(state['projective_anchor_observations']['keypoint_indices'],cons['parts'][0]['projective_anchor_observations']['keypoint_indices']))
        self.assertTrue(torch.equal(state['anchor_matchability'],(cons['parts'][0]['identity_reliability']*cons['parts'][0]['geometry_reliability']).clamp(0,1)))
    def test_two_parts_offsets_preserved(self):
        cons,_=fixture();cons['parts'].append(copy.deepcopy(cons['parts'][0]));state=D.off_state(cons,['a','b','c'])
        self.assertEqual(len(state['anchor_ids']),16);self.assertEqual(state['projective_anchor_observations']['observation_offsets'].tolist(),list(range(0,49,3)))
    def test_provenance_fail_closed(self):
        cons,_=fixture();cons['parts'][0]['uses_test_queries']=True
        with self.assertRaises(ValueError):D.off_state(cons,['a','b','c'])
    def test_selector_exported_descriptor_independence(self):
        cons,cache=fixture();state=D.off_state(cons,['a','b','c'])
        class Protected(dict):
            def __getitem__(self,k):
                if k=='anchor_features':raise AssertionError('forbidden descriptor read')
                return super().__getitem__(k)
        a,ta=D.select_compact_rows_v2(Protected(state),cache)
        state['anchor_features']=torch.randn_like(state['anchor_features'])
        b,tb=D.select_compact_rows_v2(Protected(state),cache)
        self.assertTrue(torch.equal(a,b));self.assertEqual(ta['final_selected_count'],tb['final_selected_count'])
    def test_export_shares_ids_and_xyz(self):
        cons,_=fixture();state=D.off_state(cons,['a','b','c']);state['anchor_features']=torch.nn.functional.normalize(torch.randn(8,256),dim=1)
        with tempfile.TemporaryDirectory() as td:
            rows=torch.tensor([1,3,6]);prov=dict(uses_test_queries=False,uses_source_mapping_rgb=False)
            a=D.save_subset(Path(td)/'a.pt',state,rows,prov)
            altered=dict(state,anchor_features=torch.roll(state['anchor_features'],1,1));b=D.save_subset(Path(td)/'b.pt',altered,rows,prov)
            av,bv=D.load(a['path']),D.load(b['path'])
            self.assertTrue(torch.equal(av['anchor_xyz'],bv['anchor_xyz']));self.assertFalse(torch.equal(av['anchor_features'],bv['anchor_features']))
    def test_invalid_source_row_rejected(self):
        cons,cache=fixture();state=D.off_state(cons,['a','b','c']);state['projective_anchor_observations']['keypoint_indices'][0]=99
        with self.assertRaises(ValueError):D.verify_state(state,state,cache)

if __name__=='__main__':
    torch.set_num_threads(4);torch.set_num_interop_threads(1);unittest.main()
