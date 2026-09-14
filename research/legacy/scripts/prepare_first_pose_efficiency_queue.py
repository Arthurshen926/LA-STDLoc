"""Register fixed-map, single-solve Cambridge experiments; never select on GT online."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1]
    snapshot = root / 'code'
    snapshot.mkdir()
    for directory in ['common', 'data', 'evidence', 'features', 'localization',
                      'evaluation', 'priors', 'map_learning', 'topology', 'scripts', 'configs']:
        shutil.copytree(source / directory, snapshot / directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()
    hashes = {str(p.relative_to(snapshot)): digest(p)
              for p in snapshot.rglob('*') if p.is_file()}
    variants = {
        'base': [],
        'core80': ['--match-retention-fraction', '0.8'],
        'core65': ['--match-retention-fraction', '0.65'],
        'core80_prosac': ['--match-retention-fraction', '0.8',
                          '--confidence-core-progressive-sampling'],
        'core80_diverse': ['--match-retention-fraction', '0.8',
                           '--minimum-sufficient-confidence-core'],
    }
    maps = Path('/mnt/pool/sqy/lafgs_v24_cambridge_high_capacity_online_20260901')
    python = '/root/miniconda3/envs/g4splat/bin/python'
    jobs, inputs = [], {}
    for scene in ['StMarysChurch', 'OldHospital', 'GreatCourt', 'KingsCollege', 'ShopFacade']:
        folder = maps / scene / 'projective_map'
        for name in ['projective_anchor_map.pt', 'identity_metric.pt']:
            inputs[str(folder / name)] = digest(folder / name)
        for variant, options in variants.items():
            output = root / 'runs' / scene / variant
            command = [python, '-u', '-m', 'scripts.evaluate', '--dataset',
                       '/mnt/pool/sqy/Cambridge_stdloc/' + scene, '--images', 'processed',
                       '--map', str(folder / 'projective_anchor_map.pt'),
                       '--metric-state', str(folder / 'identity_metric.pt'),
                       '--output', str(output), '--config',
                       str(snapshot / 'configs/anygsloc_mainline.yaml'),
                       '--split', 'test', '--seed', '2026', '--device', 'cuda:0',
                       '--deployment-mode', *options]
            jobs.append(dict(id=scene + '_' + variant, command=command,
                             cwd=str(snapshot), deps=[],
                             outputs=[str(output / n) for n in ['summary.json', 'results.json']],
                             env=dict(PYTHONPATH=str(snapshot), CUDA_VISIBLE_DEVICES='0',
                                      LD_LIBRARY_PATH='/root/miniconda3/envs/g4splat/lib')))
    protocol = dict(scope='fixed-map first-pose only; no second solve or offline feedback',
                    seed=2026, variants=variants, input_sha256=inputs,
                    code_sha256=hashes, test_calibrated=True,
                    timing='shared CPU, reserved GPU0; localize excludes IO; report warm10 separately',
                    selection='Report all cells, paired medians/recall/catastrophes and mean/P90 latency; no automatic promotion')
    for name, value in [('jobs.json', jobs), ('protocol.json', protocol)]:
        (root / name).write_text(json.dumps(value, indent=2))
    for path in snapshot.rglob('*'):
        if path.is_file():
            path.chmod(0o444)
    print(root, len(jobs), 'registered evaluations')


if __name__ == '__main__':
    main()
