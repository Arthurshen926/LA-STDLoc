"""CPU-only, sequential preparation for the corrected five-scene prior matrix.

Does not launch GPU work, modify old inputs, or imply experiments completed.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    import cv2
    cv2.setNumThreads(1)
    from priors.i3dgs_adapter import prepare_cambridge_mapping as prepare_i3
    from priors.long_lrm_adapter import prepare_cambridge_mapping as prepare_lrm
    args.output.mkdir(parents=True)
    ledger = {'schema': 'corrected_cambridge_cpu_preparation_v1',
              'gpu_jobs_launched': False, 'gpu_assignment_required': True,
              'complete': False, 'stages': [], 'started_unix': time.time()}
    def save():
        tmp = args.output / 'preparation.tmp.json'
        tmp.write_text(json.dumps(ledger, indent=2))
        tmp.replace(args.output / 'preparation.json')
    save()
    for scene in ['ShopFacade', 'OldHospital', 'KingsCollege', 'GreatCourt', 'StMarysChurch']:
        for variant in ['i3dgs_input_d1', 'i3dgs_input_d2', 'long_lrm_input_pose32']:
            if shutil.disk_usage(args.output).free < 250 * 1024**3:
                raise RuntimeError('preserving 250 GiB shared-disk reserve; preparation stopped')
            row = {'scene': scene, 'variant': variant, 'status': 'running', 'started_unix': time.time()}
            ledger['stages'].append(row); save()
            try:
                dataset = Path('/mnt/pool/sqy/Cambridge_stdloc') / scene
                output = args.output / scene / variant
                options = dict(images='processed', input_image_domain='rectified')
                if variant.startswith('i3dgs'):
                    result = prepare_i3(dataset, output, downscale=1 if variant.endswith('d1') else 2, **options)
                else:
                    result = prepare_lrm(dataset, output, **options)
                row.update(status='complete', output=str(output), images=len(result['records']),
                           seconds=time.time()-row['started_unix'])
                print(scene, variant, row['images'], 'COMPLETE', flush=True)
            except Exception as exc:
                row.update(status='failed', error=str(exc)); save(); raise
            save()
    ledger['complete'] = True; save()


if __name__ == '__main__':
    main()
