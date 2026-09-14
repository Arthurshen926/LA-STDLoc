"""Run an unchanged upstream Python entrypoint under a PyTorch allocator budget."""
import argparse
import runpy
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--fraction',type=float,required=True)
    p.add_argument('script');p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
    if not 0<a.fraction<=1:raise ValueError('invalid memory fraction')
    import torch
    torch.cuda.set_per_process_memory_fraction(a.fraction,0)
    from pathlib import Path
    sys.path.insert(0,str(Path(a.script).resolve().parent))
    sys.argv=[a.script,*a.arguments]
    runpy.run_path(a.script,run_name='__main__')


if __name__=='__main__':main()
