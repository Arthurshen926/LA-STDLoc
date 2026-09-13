"""Collect an already running map without interrupting or duplicating it."""
import argparse
import json
from pathlib import Path
import time
from common.hashing import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True)
    p.add_argument('--identity',required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();started=time.time()
    while time.time()-started<24*3600:
        stat=Path(f'/proc/{a.pid}/stat')
        try:
            fields=stat.read_text().rsplit(')',1)[1].split()
            alive=fields[19]==a.identity and fields[0]!='Z'
        except FileNotFoundError:alive=False
        if not alive:break
        print('waiting for existing map PID',a.pid,flush=True);time.sleep(30)
    else:raise TimeoutError('existing mapping process exceeded 24 hours')
    r=json.loads(a.report.read_text())
    for key in ['map','metric','association_graph']:
        out=r['output']
        if sha256_file(Path(out[key]))!=out[key+'_sha256']:raise ValueError('map bundle hash mismatch')
    print('existing map complete; bundle verified',flush=True)


if __name__=='__main__':main()
