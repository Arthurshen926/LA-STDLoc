"""Explicit solver-budget evaluation and warmed complete-localizer timing."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import time

import torch

from anygsloc.artifacts import sha256
from anygsloc.data.datasets import ColmapDataset
from anygsloc.data.calibration import resolve_scene_calibration
from anygsloc.data.geometric_validity import SameKRectificationSupport
from anygsloc.evaluation.metrics import pose_error
from anygsloc.evaluation.runner import summarize
import anygsloc.online.runtime as runtime
from supplement_experiments import BUDGETS, read, write


def gpu_snapshot():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu',
                             '--format=csv,noheader'], capture_output=True, text=True)
    return result.stdout.strip()


def evaluate(dataset_path, map_path, output, budget='A', warmup=0, rounds=1):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    dataset = ColmapDataset(dataset_path, load_masks=False)
    cameras = dataset.split('test')
    support = SameKRectificationSupport(dataset)
    calibration = resolve_scene_calibration(dataset.split('mapping'))
    probability, minimum, maximum = BUDGETS[budget]
    real_solve = runtime.solve_absolute_pose
    def solve(p2, p3, K, **options):
        return real_solve(p2, p3, K, **dict(options, confidence=probability,
                         min_iterations=minimum, max_iterations=maximum))
    runtime.solve_absolute_pose = solve
    engine = runtime.Localizer(map_path, keypoint_count=calibration['keypoint_count'],
        reprojection_error_px=calibration['reprojection_error_px'], mask_policy='geometric_validity', seed=2026)
    engine.config.update(confidence=probability, minimum_iterations=minimum, maximum_iterations=maximum)
    manifest = dict(map=str(map_path), map_sha256=sha256(map_path), dataset=str(dataset_path),
        queries=[c.image_name for c in cameras], config=engine.config, calibration=calibration,
        seed=2026, budget=budget, warmup=warmup, rounds=rounds, gpu_start=gpu_snapshot(),
        torch_version=torch.__version__, cuda_device=torch.cuda.get_device_name(0),
        physical_device_mapping=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
        image_domain='rectified', rectification_contract='opencv_same_k', query_semantics=False)
    write(output/'manifest.json', manifest)
    reports = []
    try:
        for repeat in range(rounds):
            for i in range(warmup):
                camera = cameras[i % len(cameras)]
                engine.localize(dataset.load_image(camera), fov_x=camera.fov_x,
                                fov_y=camera.fov_y, valid_mask=support(camera))
            records = []
            start = time.perf_counter()
            for i, camera in enumerate(cameras):
                result = engine.localize(dataset.load_image(camera), fov_x=camera.fov_x,
                                        fov_y=camera.fov_y, valid_mask=support(camera))
                re, te = pose_error(result['pose'].pose_w2c, camera.pose_w2c) if result['success'] else (None, None)
                info = {k: v for k, v in result['pose'].diagnostics.items() if k != 'inliers'}
                records.append(dict(image_name=camera.image_name, success=result['success'],
                    translation_error_cm=te, rotation_error_deg=re, runtime_ms=result['runtime_ms'],
                    inlier_count=len(result['pose'].inliers), pose_w2c=result['pose'].pose_w2c.tolist(),
                    solver_diagnostics=info))
                if i % 50 == 0:
                    write(output/'progress.json', dict(round=repeat, completed=i+1, total=len(cameras)))
                    print('evaluate', output.name, repeat, i+1, '/', len(cameras), flush=True)
            report = summarize(records, time.perf_counter()-start)
            report['timing_protocol'] = ('warmed_complete_localizer_50_calls_three_rounds_hardware_snapshot_attached'
                if warmup == 50 and rounds == 3 else 'accuracy_run_unwarmed_telemetry')
            reports.append(report)
            write(output/f'round{repeat}/results.json', records)
            write(output/f'round{repeat}/summary.json', report)
            if repeat == 0:
                write(output/'results.json', records)
        write(output/'summary.json', dict(reports[0], complete=True, rounds=reports,
            gpu_end=gpu_snapshot(), map_bytes=map_path.stat().st_size,
            strict_host_isolation_claimed=False))
    finally:
        runtime.solve_absolute_pose = real_solve


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--map', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--budget', choices=list(BUDGETS), default='A')
    p.add_argument('--warmup', type=int, default=0)
    p.add_argument('--rounds', type=int, default=1)
    a = p.parse_args()
    evaluate(a.dataset, a.map, a.output, a.budget, a.warmup, a.rounds)
