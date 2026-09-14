"""Wait for an idle accelerator host before benchmark jobs; never kill jobs."""
import subprocess
import time


while True:
    result=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if not result.strip():break
    print('Benchmark waiting for other GPU processes to finish',flush=True)
    time.sleep(10)
