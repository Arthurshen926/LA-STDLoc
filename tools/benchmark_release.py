"""Batch-one, 50-warmup, three-round deployment benchmark on a fixed map."""
import argparse
import json
from pathlib import Path
import resource
import time
import torch
from anygsloc.data.datasets import ColmapDataset
from anygsloc.data.calibration import resolve_scene_calibration
from anygsloc.data.geometric_validity import SameKRectificationSupport
from anygsloc.online.runtime import Localizer
from anygsloc.evaluation.metrics import pose_error
from anygsloc.evaluation.runner import summarize
from anygsloc.artifacts import sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True)
    p.add_argument('--map',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    begin=time.perf_counter()
    data=ColmapDataset(a.dataset,load_masks=False)
    support=SameKRectificationSupport(data)
    cal=resolve_scene_calibration(data.split('mapping'))
    engine=Localizer(a.map,keypoint_count=cal['keypoint_count'],reprojection_error_px=cal['reprojection_error_px'],
                     mask_policy='geometric_validity',device='cuda:0',seed=2026)
    torch.cuda.synchronize();init=time.perf_counter()-begin
    cameras=data.split('test')
    if not cameras:raise ValueError('empty test split')
    for i in range(50):
        cam=cameras[i%len(cameras)]
        engine.localize(data.load_image(cam),fov_x=cam.fov_x,fov_y=cam.fov_y,valid_mask=support(cam))
    reports=[]
    for repeat in range(3):
        rows=[];torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
        for cam in cameras:
            frame_started=time.perf_counter()
            result=engine.localize(data.load_image(cam),fov_x=cam.fov_x,fov_y=cam.fov_y,valid_mask=support(cam))
            frame_ms=(time.perf_counter()-frame_started)*1000
            re,te=pose_error(result['pose'].pose_w2c,cam.pose_w2c) if result['success'] else (None,None)
            rows.append(dict(image_name=cam.image_name,success=result['success'],rotation_error_deg=re,
                translation_error_cm=te,runtime_ms=result['runtime_ms'],frame_io_geometry_pose_ms=frame_ms))
        report=summarize(rows,time.perf_counter()-begin)
        report.update(timing_protocol='50_warmup_batch1_three_rounds',
                      peak_torch_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_torch_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                      process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (a.output/f'round{repeat}.json').write_text(json.dumps(dict(summary=report,queries=rows),indent=2))
        reports.append(report)
    (a.output/'summary.json').write_text(json.dumps(dict(complete=True,map_sha256=sha256(a.map),
        map_bytes=Path(a.map).stat().st_size,initialization_seconds=init,rounds=reports,
        isolation='requires queue idle prerequisite; CPU scheduling not pinned',
        frame_wall_includes='RGB decode, geometry validity construction, transfers and pose'),indent=2))


if __name__=='__main__':main()
