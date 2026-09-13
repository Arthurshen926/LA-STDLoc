"""Fresh, provenance-preserving recovery of failed i3DGS export descendants."""
import json
import shutil
from pathlib import Path
from common.hashing import sha256_file


def main():
    old = Path('/mnt/pool/sqy/anygsloc_three_gpu_20260910')
    root = Path('/mnt/pool/sqy/anygsloc_invalid_primitive_recovery_20260910')
    root.mkdir(exist_ok=False)
    snap = root / 'code'
    shutil.copytree(old/'code', snap, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    target = snap/'priors/i3dgs_adapter.py'
    target.chmod(0o644)
    shutil.copyfile(Path(__file__).resolve().parents[1]/'priors/i3dgs_adapter.py', target)
    for gpu, scene in [(0, 'GreatCourt'), (1, 'OldHospital')]:
        jobs = json.loads((old/f'gpu{gpu}/jobs.json').read_text())
        selected = [j for j in jobs if j['id'].startswith(scene+'_i3dgs') and not j['id'].endswith('_reconstruct')]
        ids = {j['id'] for j in selected}
        for j in selected:
            j['deps'] = [d for d in j['deps'] if d in ids]
            # Reuse reconstruction inputs, but never overwrite previous outputs.
            encoded = json.dumps(j).replace(str(old/'code'), str(snap))
            encoded = encoded.replace(str(old/'runs'), str(root/'runs'))
            for part in ['world.ply', 'export.json', 'ft1000']:
                encoded = encoded.replace(str(old/f'priors/{scene}/i3dgs'/part), str(root/f'priors/{scene}/i3dgs'/part))
            j.clear(); j.update(json.loads(encoded))
        lane = root/f'gpu{gpu}'; lane.mkdir()
        (lane/'jobs.json').write_text(json.dumps(selected, indent=2))
        print(gpu, len(selected), [j['id'] for j in selected], flush=True)
    manifest = {str(p.relative_to(snap)):sha256_file(p) for p in snap.rglob('*') if p.is_file()}
    (root/'code_manifest.json').write_text(json.dumps(manifest, indent=2))
    for p in snap.rglob('*'):
        if p.is_file(): p.chmod(0o444)


if __name__ == '__main__': main()
