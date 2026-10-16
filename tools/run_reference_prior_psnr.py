"""Run reference PSNR shards as GPUs become idle, then verify and package results."""
from pathlib import Path
import csv
import io
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path('/root/miniconda3/envs/g4splat/bin/python')
SCRIPT = ROOT / 'tools/evaluate_reference_prior_psnr.py'
OUT = ROOT / 'artifacts/prior_psnr_reference_20261008'


def idle_gpus():
    text = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
        text=True)
    # Check both utilization and memory: an idle moment in another job is not a free GPU.
    return [row[0].strip() for row in csv.reader(io.StringIO(text))
            if int(row[1]) < 100 and int(row[2]) == 0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['PATH'] = str(PYTHON.parent) + os.pathsep + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = str(PYTHON.parent.parent / 'lib') + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    subprocess.run([str(PYTHON), str(SCRIPT), 'prepare'], cwd=ROOT, env=env, check=True)
    pending, running = [0, 1, 2], {}
    try:
        while pending or running:
            for gpu, (index, process, stream) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                del running[gpu]
                if code:
                    raise RuntimeError(f'Reference PSNR shard {index} failed ({code}); see worker log.')
                print(f'Shard {index} complete on GPU {gpu}', flush=True)
            for gpu in idle_gpus():
                if not pending:
                    break
                if gpu in running:
                    continue
                index = pending.pop(0)
                local = dict(env, CUDA_VISIBLE_DEVICES=gpu)
                stream = (OUT / f'worker_{index}.log').open('a', buffering=1)
                process = subprocess.Popen(
                    [str(PYTHON), str(SCRIPT), 'worker', '--worker', str(index), '--workers', '3'],
                    cwd=ROOT, env=local, stdout=stream, stderr=subprocess.STDOUT)
                running[gpu] = (index, process, stream)
                print(f'Started shard {index} on idle GPU {gpu}', flush=True)
            if pending or running:
                time.sleep(10)
    finally:
        for _, process, stream in running.values():
            if process.poll() is None:
                process.terminate()
                process.wait()
            stream.close()
    subprocess.run([str(PYTHON), str(SCRIPT), 'summarize'], cwd=ROOT, env=env, check=True)
    while not (available := idle_gpus()):
        time.sleep(10)
    local = dict(env, CUDA_VISIBLE_DEVICES=available[0])
    subprocess.run([str(PYTHON), str(SCRIPT), 'verify'], cwd=ROOT, env=local, check=True)
    subprocess.run([str(PYTHON), str(SCRIPT), 'report'], cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
