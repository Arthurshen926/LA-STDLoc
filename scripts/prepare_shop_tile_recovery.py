"""Retry ShopFacade mapping without reducing resolution or pruning primitives."""
import json
import shutil
from pathlib import Path
from common.hashing import sha256_file


def main():
    old=Path('/mnt/pool/sqy/anygsloc_three_gpu_20260910')
    root=Path('/mnt/pool/sqy/anygsloc_shop_tile_recovery_20260910')
    snap=root/'code'
    shutil.copytree(old/'code',snap,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    target=snap/'priors/rendering.py';target.chmod(0o644)
    shutil.copyfile(Path(__file__).resolve().parents[1]/'priors/rendering.py',target)
    jobs=[j for j in json.loads((old/'gpu2/jobs.json').read_text()) if j['id'].startswith('ShopFacade_vanilla_2dgs_') and not j['id'].endswith('_reconstruct')]
    ids={j['id'] for j in jobs}
    for j in jobs:
        j['deps']=[d for d in j['deps'] if d in ids]
        encoded=json.dumps(j).replace(str(old/'code'),str(snap)).replace(str(old/'runs'),str(root/'runs'))
        j.clear();j.update(json.loads(encoded))
    lane=root/'gpu2';lane.mkdir()
    (lane/'jobs.json').write_text(json.dumps(jobs,indent=2))
    (root/'code_manifest.json').write_text(json.dumps({str(p.relative_to(snap)):sha256_file(p) for p in snap.rglob('*') if p.is_file()},indent=2))
    for p in snap.rglob('*'):
        if p.is_file():p.chmod(0o444)
    print([j['id'] for j in jobs])


if __name__=='__main__':main()
