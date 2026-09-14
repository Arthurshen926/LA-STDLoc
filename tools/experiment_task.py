"""Queue task utilities: measured commands and read-only asset accounting."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--asset',type=Path,action='append',default=[])
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args()
    if a.receipt.exists(): raise FileExistsError(a.receipt)
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    if a.asset:
        rows=[]
        for asset in a.asset:
            if not asset.exists():raise FileNotFoundError(asset)
            files=[asset] if asset.is_file() else [x for x in asset.rglob('*') if x.is_file()]
            rows.append(dict(path=str(asset),bytes=sum(x.stat().st_size for x in files),files=len(files),
                             scope='supplied_asset_not_automatically_minimal_deployment'))
        a.receipt.write_text(json.dumps(dict(status='complete',assets=rows),indent=2));return
    import psutil
    cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    if not cmd:raise ValueError('empty command')
    begin=time.perf_counter();child=subprocess.Popen(cmd)
    peak=0
    while child.poll() is None:
        try:
            parent=psutil.Process(child.pid)
            rss=0
            for proc in [parent]+parent.children(recursive=True):
                try:rss+=proc.memory_info().rss
                except psutil.Error:pass
            peak=max(peak,rss)
        except psutil.Error:pass
        time.sleep(1)
    a.receipt.write_text(json.dumps(dict(status='complete' if child.returncode==0 else 'failed',
        returncode=child.returncode,command=cmd,wall_seconds=time.perf_counter()-begin,
        peak_process_tree_rss_bytes=peak,rss_sampling_seconds=1,
        gpu_memory='see native receipts; process-tree GPU peak not inferred'),indent=2))
    sys.exit(child.returncode)


if __name__=='__main__':main()
