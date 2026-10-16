import copy,unittest
from experiments import compact_t6_gpu_admission_20261013 as E

class GateTests(unittest.TestCase):
    def setUp(self):
        self.cases=[dict(scene='s',mode='m',map={'sha256':'map'},names=['q'])]
        self.r=dict(complete=True,protocol_sha256='p',sealed_sha256='s',actual_cpu_threads=4,no_ground_truth_metrics=True,
            hardware=dict(index=0,uuid=E.D.GPU_UUIDS[0],name='NVIDIA GeForce RTX 3090',python='same',torch='same',cuda='same'),peak_reserved_bytes=1024,
            outputs=[dict(scene='s',mode='m',map_sha256='map',rows=[dict(name='q',success=True,inliers=[1,2],matched_keypoints_sha256='k',matched_anchors_sha256='a',pose=[[1.,0.],[0.,1.]])])])
    def test_accept_reference(self):E.validate_record(self.r,0,'p','s',self.cases);E.same_predictions(self.r,copy.deepcopy(self.r))
    def test_reject_population(self):
        self.r['outputs'][0]['rows'][0]['name']='wrong'
        with self.assertRaises(ValueError):E.validate_record(self.r,0,'p','s',self.cases)
    def test_reject_memory(self):
        self.r['peak_reserved_bytes']=20*1024**3
        with self.assertRaises(ValueError):E.validate_record(self.r,0,'p','s',self.cases)
    def test_reject_matches(self):
        other=copy.deepcopy(self.r);other['outputs'][0]['rows'][0]['matched_anchors_sha256']='different'
        with self.assertRaises(ValueError):E.same_predictions(self.r,other)
    def test_reject_pose(self):
        other=copy.deepcopy(self.r);other['outputs'][0]['rows'][0]['pose'][0][0]+=.01
        with self.assertRaises(ValueError):E.same_predictions(self.r,other)
if __name__=='__main__':unittest.main()
