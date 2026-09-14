"""Run pinned native baselines with explicit semantic policy and cost receipts.

Only protocol/measurement adapters: native matching, training losses, schedules,
landmark selectors and pose refinement are not replaced.
"""
import argparse
import builtins
import functools
import json
import os
from pathlib import Path
import pickle
import random
import resource
import shutil
import runpy
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--method', choices=['STDLoc','ULFLoc'], required=True)
    p.add_argument('--phase', choices=['train','evaluate'], required=True)
    p.add_argument('--mapping-semantics', choices=['on','off'], required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--warmup', type=int, default=0)
    p.add_argument('--run-tag', default='test')
    p.add_argument('native', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.receipt.exists():
        raise FileExistsError(a.receipt)
    sys.path.insert(0, str(a.repo))
    import numpy as np
    import torch
    import torch.nn.functional as F
    import poselib
    from anygsloc.data.colmap import read_intrinsics_binary, read_extrinsics_binary
    from anygsloc.data.geometric_validity import rectification_support
    torch.set_num_threads(4)
    intr = read_intrinsics_binary(str(a.source/'sparse/0/cameras.bin'))
    extr = read_extrinsics_binary(str(a.source/'sparse/0/images.bin'))
    by_name = {v.name: intr[v.camera_id] for v in extr.values()}
    def seed():
        random.seed(2026); np.random.seed(2026); torch.manual_seed(2026)
    seed()
    import utils.general_utils as general
    original_safe_state = general.safe_state
    def safe_state(*args, **kwargs):
        value = original_safe_state(*args, **kwargs)
        seed()
        return value
    general.safe_state = safe_state
    original_pose = poselib.estimate_absolute_pose
    def pose(p2, p3, camera, options, bundle, *args, **kwargs):
        return original_pose(p2,p3,camera,dict(options,seed=2026),bundle,*args,**kwargs)
    poselib.estimate_absolute_pose = pose
    build_stages=[]
    if a.phase=='train':
        import importlib
        targets=([('train_detector','training_detector')] if a.method=='STDLoc' else
                 [('utils.keypoints_sample','keypoints_vote'),('utils.gsfeature_fusion','feature_fusion')])
        def instrument(fn,label):
            @functools.wraps(fn)
            def measured(*args,**kwargs):
                torch.cuda.synchronize();begin=time.perf_counter()
                value=fn(*args,**kwargs)
                torch.cuda.synchronize()
                build_stages.append(dict(stage=label,wall_seconds=time.perf_counter()-begin))
                return value
            return measured
        for module_name,function_name in targets:
            module=importlib.import_module(module_name)
            setattr(module,function_name,instrument(getattr(module,function_name),function_name))
    original_load = pickle.load
    @functools.lru_cache(maxsize=32)
    def geometry(name, height, width):
        mask = rectification_support(by_name[name])
        if mask.shape != (height,width):
            mask=F.interpolate(mask.float()[None,None],size=(height,width),mode='nearest')[0,0].bool()
        return mask
    class TrainingMasks(dict):
        __hash__ = object.__hash__
        def __init__(self, raw):
            super().__init__((name,None) for name in by_name)
            self.raw = raw
        @functools.lru_cache(maxsize=32)
        def __getitem__(self, name):
            cam=by_name[name]; scale=min(1.,640/max(cam.width,cam.height))
            h,w=int(cam.height*scale),int(cam.width*scale)
            geom=geometry(name,h,w)
            if self.raw is None:
                one=torch.ones_like(geom)
                return one,one,geom
            layers=[]
            for value in self.raw[name][:2]:
                value=torch.as_tensor(value).float().squeeze()
                layers.append(F.interpolate(value[None,None],size=(h,w),mode='nearest')[0,0].bool())
            return layers[0],layers[1],geom
    def load(stream, *args, **kwargs):
        if str(getattr(stream,'name','')).endswith('masks.pkl'):
            if a.phase != 'train':
                raise RuntimeError('query semantic file must not be read')
            raw=original_load(stream,*args,**kwargs) if a.mapping_semantics=='on' else None
            return TrainingMasks(raw)
        return original_load(stream,*args,**kwargs)
    pickle.load=load
    original_exists=os.path.exists
    if a.phase=='evaluate':
        os.path.exists=lambda path: False if str(path).endswith('masks.pkl') else original_exists(path)
        from scene.cameras import Camera
        original_init=Camera.__init__
        def camera_init(self,*args,**kwargs):
            original_init(self,*args,**kwargs)
            if getattr(self,'original_image',None) is not None:
                h,w=self.original_image.shape[-2:]
                self.original_image*=geometry(self.image_name,h,w).to(self.original_image.device)[None]
        Camera.__init__=camera_init
    # Capture both native stage outputs without altering their return values.
    timings=[]
    original_build=builtins.__build_class__
    def build(func,name,*args,**kwargs):
        cls=original_build(func,name,*args,**kwargs)
        if name in ('STDLoc','ULFLoc') and hasattr(cls,'localize'):
            localize=cls.localize
            coarse_name='loc_sparse' if name=='STDLoc' else 'loc_coarse'
            coarse=getattr(cls,coarse_name)
            def coarse_call(self,*aa,**kk):
                result=coarse(self,*aa,**kk);torch.cuda.synchronize()
                self._v2_coarse_ms=(time.perf_counter()-self._v2_started)*1000
                return result
            setattr(cls,coarse_name,coarse_call)
            def localize_call(self,*aa,**kk):
                if not getattr(self,'_v2_warmed',False):
                    self._v2_warmed=True
                    for _ in range(a.warmup):
                        self._v2_started=time.perf_counter()
                        localize(self,*aa,**kk)
                torch.cuda.synchronize();self._v2_started=time.perf_counter()
                result=localize(self,*aa,**kk);torch.cuda.synchronize()
                total=(time.perf_counter()-self._v2_started)*1000
                timings.append(dict(coarse_prefix_ms=self._v2_coarse_ms,total_ms=total,
                    fine_increment_ms=total-self._v2_coarse_ms,
                    peak_torch_gpu_bytes=torch.cuda.max_memory_allocated(),
                    native_stage_timing=result.get('timing_ms')))
                return result
            cls.localize=localize_call
        return cls
    builtins.__build_class__=build
    entry='train.py' if a.phase=='train' else ('stdloc.py' if a.method=='STDLoc' else 'ulfloc.py')
    native=a.native[1:] if a.native[:1]==['--'] else a.native
    if a.phase=='evaluate' and a.method=='ULFLoc' and a.run_tag!='test':
        import yaml
        config=Path(native[native.index('--cfg')+1])
        model=Path(native[native.index('-m')+1])
        settings=yaml.safe_load(config.read_text())
        target=model/a.run_tag
        target.mkdir(parents=True,exist_ok=False)
        for key in ['landmark_file_name','kpts_feature_file_name']:
            filename=settings['sample'][key]
            shutil.copy2(model/settings['log_name']/filename,target/filename)
        settings['log_name']=a.run_tag
        config=target/'config.yaml';config.write_text(yaml.safe_dump(settings))
        native[native.index('--cfg')+1]=str(config)
    sys.argv=[str(a.repo/entry)]+native
    os.chdir(a.repo)
    started=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    status='failed'
    try:
        namespace=runpy.run_path(str(a.repo/entry),run_name='__main__')
        status='complete'
    finally:
        a.receipt.parent.mkdir(parents=True,exist_ok=True)
        a.receipt.write_text(json.dumps(dict(status=status,method=a.method,phase=a.phase,
            mapping_semantics=a.mapping_semantics,query_semantics=False,seed=2026,
            wall_seconds=time.perf_counter()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_torch_gpu_bytes=max([torch.cuda.max_memory_allocated()]+[t['peak_torch_gpu_bytes'] for t in timings]),timings=timings,
            native_output=namespace.get('output_path') if status=='complete' else None,
            warmup_queries=a.warmup,
            timing_protocol='synchronized_unwarmed_telemetry_not_isolated_benchmark',
            coarse_timing_boundary='actual_native_pipeline_prefix_including_shared_precomputation'),indent=2))
        value=json.loads(a.receipt.read_text())
        value['build_stages']=build_stages
        value['build_cost_scope']=('joint_gaussian_and_feature_training_plus_detector' if a.method=='STDLoc'
                                   else 'gaussian_reconstruction_plus_sampling_and_feature_fusion')
        value['segmentation_generation_cost']='not_measured_here_precomputed_input; never_assume_zero'
        a.receipt.write_text(json.dumps(value,indent=2))


if __name__=='__main__': main()
