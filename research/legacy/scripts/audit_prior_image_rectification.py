"""Read-only pixel replay of legacy preprocessing, using mapping images only."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from common.hashing import sha256_file
from data.colmap import read_extrinsics_binary, read_intrinsics_binary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--prior-images', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=12)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cameras = read_intrinsics_binary(str(args.dataset / 'sparse/0/cameras.bin'))
    images = read_extrinsics_binary(str(args.dataset / 'sparse/0/images.bin'))
    tests = {x.split()[0] for x in (args.dataset / 'dataset_test.txt').read_text().splitlines() if x and not x.startswith('#')}
    mapping = sorted([r for r in images.values() if r.name not in tests], key=lambda x: x.name)
    selected = [mapping[i] for i in np.linspace(0, len(mapping)-1, min(args.count, len(mapping)), dtype=int)]
    clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8))
    rows = []
    for image in selected:
        c = cameras[image.camera_id]
        if c.model != 'SIMPLE_RADIAL':
            raise ValueError('audit currently expects SIMPLE_RADIAL')
        f, cx, cy, k = c.params
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.]])
        paths = [args.dataset / image.name, args.dataset / 'processed' / image.name, args.prior_images / image.name]
        raw, processed, prior = [cv2.imread(str(x)) for x in paths]
        if any(x is None for x in [raw, processed, prior]):
            raise FileNotFoundError(paths)
        equalized = cv2.merge([clahe.apply(channel) for channel in cv2.split(raw)])
        # Match the original ULF preprocessing's FP32 camera parameters exactly.
        once = cv2.undistort(equalized, K.astype(np.float32), np.array([k, 0, 0, 0], np.float32))
        twice = cv2.undistort(processed, K, np.array([k, 0, 0, 0.]), None, K)
        mapx, mapy = cv2.initUndistortRectifyMap(K, np.array([k, 0, 0, 0.]), None, K, (c.width, c.height), cv2.CV_32FC1)
        yy, xx = np.indices(mapx.shape)
        displacement = np.hypot(mapx-xx, mapy-yy)
        def diff(a, b):
            return {'bit_exact': bool(np.array_equal(a, b)), 'mean_abs_u8': float(np.abs(a.astype(float)-b).mean()), 'different_pixel_fraction': float(np.any(a != b, axis=2).mean())}
        rows.append({'image': image.name, 'radial_k': float(k), 'input_sha256': [sha256_file(x) for x in paths],
                     'processed_vs_raw_clahe_undistort': diff(processed, once),
                     'prior_vs_undistort_processed': diff(prior, twice), 'prior_vs_processed': diff(prior, processed),
                     'extra_undistort_displacement_p50_p90_p99_px': np.quantile(displacement, [.5, .9, .99]).tolist()})
        print(json.dumps(rows[-1]), flush=True)
    report = {'schema': 'anygsloc_legacy_rectification_pixel_replay_v1', 'uses_source_mapping_rgb': True,
              'uses_test_queries': False, 'audit_only': True, 'dataset': str(args.dataset),
              'prior_images': str(args.prior_images), 'mapping_images_checked': len(rows), 'rows': rows,
              'implementation_sha256': sha256_file(Path(__file__))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
