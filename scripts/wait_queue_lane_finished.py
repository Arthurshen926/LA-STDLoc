"""Wait without occupying CUDA until the preceding lane has released its jobs."""
import argparse
import json
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--state',type=Path,required=True);a=p.parse_args()
    while True:
        state=json.loads(a.state.read_text()) if a.state.exists() else {}
        if state.get('queue_finished') and not any(v.get('status')=='running' for v in state.get('jobs',{}).values()):
            print('preceding lane finished; GPU jobs released',flush=True);return
        time.sleep(10)


if __name__=='__main__':main()
