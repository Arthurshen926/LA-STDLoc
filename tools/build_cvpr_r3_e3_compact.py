"""Preserve the accepted R2 compact-map figure in the R3 figure suite."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
PREV = ROOT / 'paper/experiments_cvpr_r2_20261003'
OUT = ROOT / 'paper/experiments_cvpr_r3_20261003'
STEM = 'E3_compact_map'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ['pdf', 'svg', 'png', 'caption.txt']:
        shutil.copy2(PREV / f'{STEM}.{ext}', OUT / f'{STEM}.{ext}')
    old = PREV / f'{STEM}.provenance.json'
    p = json.loads(old.read_text())
    p['revision'] = 3
    p['unchanged_from_revision_2'] = True
    p['previous_figure_provenance'] = {'path': str(old), 'sha256': sha(old)}
    for ext in ['pdf', 'svg', 'png']:
        entry = p['outputs'][STEM][ext]
        entry['path'] = str(OUT / f'{STEM}.{ext}')
        assert entry['sha256'] == sha(entry['path'])
    p['source_script'] = {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)}
    (OUT / f'{STEM}.provenance.json').write_text(json.dumps(p, indent=2))
    print('E3 preserved byte-for-byte from R2; R3 provenance updated.')


if __name__ == '__main__':
    main()
