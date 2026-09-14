"""Repair an attested legacy mapping-only prior input, without resampling RGB.

    python -m scripts.prepare_rectified_prior_replay --dataset ... \
      --legacy-input ... --pixel-audit ... --output ...

The original input/prior is never modified. Reuse its mapping-only PINHOLE
camera/pose/SfM initialization, but link ONCE-rectified processed images. This
prepares a controlled Gaussian retraining experiment, not a new localization
method and not a claim that previously trained Gaussian geometry is repaired.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from common.hashing import sha256_file
from data.colmap import read_extrinsics_binary, read_intrinsics_binary


def safe_flat_registry(names):
    result = {}
    for name in names:
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('relative image names required')
        flat = name.replace('/', '__')
        if flat in result: raise ValueError('flattened names collide')
        result[flat] = name
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['dataset', 'legacy-input', 'pixel-audit', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    audit = json.loads(args.pixel_audit.read_text())
    if Path(audit['dataset']).resolve() != args.dataset.resolve() or audit['uses_test_queries'] or len(audit['rows']) < 12:
        raise ValueError('same-dataset mapping-only pixel replay required')
    if not all(r['processed_vs_raw_clahe_undistort']['bit_exact'] and r['prior_vs_undistort_processed']['bit_exact'] for r in audit['rows']):
        raise ValueError('extra rectification not attested')
    original = read_extrinsics_binary(str(args.dataset / 'sparse/0/images.bin'))
    original_cameras = read_intrinsics_binary(str(args.dataset / 'sparse/0/cameras.bin'))
    legacy = read_extrinsics_binary(str(args.legacy_input / 'sparse/0/images.bin'))
    legacy_cameras = read_intrinsics_binary(str(args.legacy_input / 'sparse/0/cameras.bin'))
    test = {s.split()[0] for s in (args.dataset / 'dataset_test.txt').read_text().splitlines() if s and not s.startswith('#')}
    mapping = {r.name: r for r in original.values() if r.name not in test}
    registry = safe_flat_registry(mapping)
    if set(registry) != {r.name for r in legacy.values()}: raise ValueError('mapping-only image registries differ')
    for r in audit['rows']:
        name = r['image']
        if name not in mapping: raise ValueError('audit is not mapping-only')
        if sha256_file(args.dataset / 'processed' / name) != r['input_sha256'][1]: raise ValueError('processed pixels changed since audit')
        if sha256_file(args.legacy_input / 'images' / name.replace('/', '__')) != r['input_sha256'][2]: raise ValueError('legacy prior pixels differ from audit')
    records = []
    for r in legacy.values():
        name = registry[r.name]; old = mapping[name]
        c, oc = legacy_cameras[r.camera_id], original_cameras[old.camera_id]
        if c.model != 'PINHOLE' or oc.model != 'SIMPLE_RADIAL': raise ValueError('explicit rectified camera contract required')
        if not np.allclose(c.params, [oc.params[0],oc.params[0],oc.params[1],oc.params[2]], atol=1e-7, rtol=0): raise ValueError('intrinsics mismatch')
        if (c.width,c.height) != (oc.width,oc.height): raise ValueError('image geometry differs')
        if not np.allclose(r.qvec,old.qvec,atol=1e-10,rtol=0) or not np.allclose(r.tvec,old.tvec,atol=1e-10,rtol=0): raise ValueError('pose mismatch')
        path = args.dataset / 'processed' / name
        records.append({'name': r.name, 'source': str(path.resolve()), 'sha256': sha256_file(path)})
    args.output.mkdir(parents=True)
    (args.output / 'images').mkdir()
    for r in records: (args.output / 'images' / r['name']).symlink_to(r['source'])
    shutil.copytree(args.legacy_input / 'sparse', args.output / 'sparse', symlinks=False)
    manifest = {'schema': 'anygsloc_rectified_prior_input_replay_v1', 'mapping_rgb_only': True,
                'uses_test_queries': False, 'dataset': str(args.dataset.resolve()),
                'legacy_input': str(args.legacy_input.resolve()), 'pixel_audit_sha256': sha256_file(args.pixel_audit),
                'image_policy': 'link already CLAHE-and-once-undistorted processed RGB; NO second resampling',
                'camera_policy': 'reuse exact legacy PINHOLE model, known mapping poses, and mapping-only SfM initialization',
                'training_executed': False, 'prior_repaired': False, 'images': records,
                'sparse_sha256': {p.name: sha256_file(p) for p in (args.output / 'sparse/0').iterdir() if p.is_file()},
                'implementation_sha256': sha256_file(Path(__file__))}
    (args.output / 'rectification_replay_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k != 'images'}), flush=True)
    print('Mapping image count', len(records), flush=True)


if __name__ == '__main__': main()
