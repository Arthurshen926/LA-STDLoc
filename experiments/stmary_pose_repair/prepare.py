"""Make an isolated q/t-only repair; never edit the original scene or kernels."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
import struct
import numpy as np
from anygsloc.artifacts import sha256
from anygsloc.data.colmap import read_extrinsics_binary, read_extrinsics_text, qvec2rotmat
from anygsloc.data.datasets import ColmapDataset
from experiments.cambridge_input_matrix.prior_provenance import audit_prepared_camera_registry

SOURCE = Path('/mnt/pool/sqy/Cambridge_stdloc/StMarysChurch')
INPUT = Path('/mnt/pool/sqy/anygsloc_registered_matrix_20260909/vanilla_inputs/StMarysChurch')
JOBS = Path('/mnt/pool/sqy/anygsloc_three_gpu_20260910/gpu1/jobs.json')
ROOT = Path('/mnt/pool/sqy/anygsloc_stmary_pose_repair_20261011_v1')

def normal_pose(q, center):
    q = np.asarray(q, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    if q.shape != (4,) or center.shape != (3,) or not np.isfinite(q).all() or not np.isfinite(center).all():
        raise ValueError('invalid quaternion or NVM camera center')
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm <= np.finfo(np.float64).tiny:
        raise ValueError('zero or invalid quaternion norm')
    unit = q / norm
    rotation = qvec2rotmat(unit)
    return unit, -rotation @ center

def nvm_cameras(path):
    with Path(path).open() as f:
        if f.readline().strip() != 'NVM_V3':
            raise ValueError('unexpected NVM header')
        line = f.readline()
        while not line.strip():
            line = f.readline()
        count = int(line)
        rows = {}
        for i in range(count):
            a = f.readline().split()
            name = str(Path(a[0]).with_suffix('.png'))
            if name in rows:
                raise ValueError('duplicate NVM camera')
            rows[name] = {'id': i, 'raw_name': a[0], 'q': np.array(a[2:6], float),
                          'C': np.array(a[6:9], float)}
        return rows

def write_images_binary(path, images):
    with Path(path).open('xb') as f:
        f.write(struct.pack('<Q', len(images)))
        for row in images.values():
            f.write(struct.pack('<idddddddi', row.id, *row.qvec, *row.tvec, row.camera_id))
            f.write(row.name.encode('utf8') + b'\0')
            f.write(struct.pack('<Q', len(row.point3D_ids)))
            for xy, pid in zip(row.xys, row.point3D_ids):
                f.write(struct.pack('<ddq', *xy, int(pid)))

def repaired_rows(original, cameras):
    if {x.name for x in original.values()} != set(cameras):
        raise ValueError('NVM/source camera registry differs')
    result = {}
    for key, row in original.items():
        source = cameras[row.name]
        if row.id != source['id'] or not np.array_equal(row.qvec, source['q']):
            raise ValueError('NVM/source raw quaternion identity differs')
        q, t = normal_pose(source['q'], source['C'])
        if not np.allclose(t, row.tvec, rtol=0, atol=1e-10):
            raise ValueError('original translation is not normalized-NVM convention')
        result[key] = row._replace(qvec=q, tvec=t)
    return result

def rewrite_prepared_text(source, destination, manifest, cameras):
    reverse = {r['prepared_name']: r['source_name'] for r in manifest['images']}
    lines = source.read_text().splitlines(keepends=True)
    out = []
    seen = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith('#'):
            out.append(line); i += 1; continue
        fields = line.split()
        name = fields[9]
        source_camera = cameras[reverse[name]]
        if int(fields[0]) != source_camera['id']:
            raise ValueError('prepared source image ID differs')
        q, t = normal_pose(source_camera['q'], source_camera['C'])
        fields[1:8] = [format(float(x), '.17g') for x in np.r_[q, t]]
        out.append(' '.join(fields) + '\n')
        if i + 1 >= len(lines):
            raise ValueError('prepared missing observation line')
        out.append(lines[i+1]); i += 2; seen.append(name)
    if len(seen) != len(reverse) or set(seen) != set(reverse):
        raise ValueError('prepared mapping whitelist differs')
    with destination.open('x') as f:
        f.writelines(out)

def json_new(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False); f.write('\n')

def prepare(root):
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    cameras = nvm_cameras(SOURCE/'reconstruction.nvm')
    original = read_extrinsics_binary(str(SOURCE/'sparse/0/images.bin'))
    repaired = repaired_rows(original, cameras)
    manifest = json.loads((INPUT/'input_manifest.json').read_text())
    if manifest['test_used'] is not False or manifest['retained_points'] != 193428:
        raise ValueError('unexpected mapping-only initialization')
    scene = root/'dataset/StMarysChurch'
    prepared = root/'vanilla_input/StMarysChurch'
    (scene/'sparse/0').mkdir(parents=True)
    (prepared/'sparse/0').mkdir(parents=True)
    # Shared large RGB collections are only read. All sparse and generated outputs are private.
    (scene/'processed').symlink_to(SOURCE/'processed', target_is_directory=True)
    (scene/'reconstruction.nvm').symlink_to(SOURCE/'reconstruction.nvm')
    (prepared/'images').symlink_to(INPUT/'images', target_is_directory=True)
    for name in ('dataset_train.txt', 'dataset_test.txt'):
        shutil.copyfile(SOURCE/name, scene/name)
    for name in ('cameras.bin', 'points3D.bin', 'points3D.ply'):
        shutil.copyfile(SOURCE/'sparse/0'/name, scene/'sparse/0'/name)
    write_images_binary(scene/'sparse/0/images.bin', repaired)
    for name in ('cameras.txt', 'points3D.txt', 'points3D.ply'):
        shutil.copyfile(INPUT/'sparse/0'/name, prepared/'sparse/0'/name)
    rewrite_prepared_text(INPUT/'sparse/0/images.txt', prepared/'sparse/0/images.txt', manifest, cameras)
    manifest = copy.deepcopy(manifest)
    manifest['pose_repair'] = {'rule': 'q_norm=q/||q||; t=-R(q_norm)*original_NVM_C',
        'quaternion_order': 'wxyz', 'quaternion_sign_preserved': True,
        'source_nvm': str(SOURCE/'reconstruction.nvm'), 'source_nvm_sha256': sha256(SOURCE/'reconstruction.nvm'),
        'source_input_manifest': str(INPUT/'input_manifest.json'),
        'source_input_manifest_sha256': sha256(INPUT/'input_manifest.json'),
        'original_points_retained_without_retriangulation': True,
        'point_approval': 'requires independent NVM XYZ/original mapping-only track review before training'}
    manifest['repaired_sparse_sha256'] = {p.name:sha256(p) for p in (prepared/'sparse/0').iterdir()}
    json_new(prepared/'input_manifest.json', manifest)
    job = copy.deepcopy(next(x for x in json.loads(JOBS.read_text()) if x['id']=='StMarysChurch_vanilla_3dgs_reconstruct'))
    job['id'] = 'StMarysChurch_pose_repaired_vanilla_3dgs_reconstruct'
    argv = job['command']
    argv[argv.index('--source_path')+1] = str(prepared)
    argv[argv.index('--model_path')+1] = str(root/'prior/vanilla_3dgs/native')
    job['outputs'] = [str(root/'prior/vanilla_3dgs/native/point_cloud/iteration_30000/point_cloud.ply')]
    job['source_job_file'] = str(JOBS)
    job['source_job_sha256'] = sha256(JOBS)
    job['effective_training_seed'] = 0
    job['launch_status'] = 'not_started_pending_cpu_review_and_parent_release'
    job['resume_policy'] = 'no resume from old checkpoints; ordinary native checkpoints omit RNG and camera stack; do not silently resume after timeout'
    json_new(root/'training_job.json', job)
    return validate(root)

def validate(root):
    from scipy.spatial.transform import Rotation
    from anygsloc.data.colmap import read_intrinsics_text
    root = Path(root)
    scene, prepared = root/'dataset/StMarysChurch', root/'vanilla_input/StMarysChurch'
    cameras = nvm_cameras(SOURCE/'reconstruction.nvm')
    original = read_extrinsics_binary(str(SOURCE/'sparse/0/images.bin'))
    new = read_extrinsics_binary(str(scene/'sparse/0/images.bin'))
    old_prepared = read_extrinsics_text(str(INPUT/'sparse/0/images.txt'))
    new_prepared = read_extrinsics_text(str(prepared/'sparse/0/images.txt'))
    if list(new) != list(original) or list(new_prepared) != list(old_prepared):
        raise ValueError('image order or IDs changed')
    max_orth = max_det = max_center = max_reference = max_t_delta = 0.
    for key, row in new.items():
        old, nvm = original[key], cameras[row.name]
        q, t = normal_pose(nvm['q'], nvm['C'])
        if not np.array_equal(q, row.qvec) or not np.array_equal(t, row.tvec):
            raise ValueError('repaired binary differs from normalized source')
        for field in ('id', 'camera_id', 'name', 'xys', 'point3D_ids'):
            if not np.array_equal(getattr(row, field), getattr(old, field)):
                raise ValueError('changed non-pose image field: '+field)
        if np.dot(row.qvec, nvm['q']) <= 0:
            raise ValueError('quaternion sign changed')
        R = qvec2rotmat(row.qvec)
        max_orth = max(max_orth, float(np.max(np.abs(R.T@R-np.eye(3)))))
        max_det = max(max_det, abs(float(np.linalg.det(R))-1))
        max_center = max(max_center, float(np.max(np.abs(-R.T@row.tvec-nvm['C']))))
        ref = Rotation.from_quat(nvm['q'][[1,2,3,0]]).as_matrix()
        max_reference = max(max_reference, float(np.max(np.abs(R-ref))))
        max_t_delta = max(max_t_delta, float(np.max(np.abs(t-old.tvec))))
    if max(max_orth, max_det, max_reference) > 1e-12 or max_center > 1e-10:
        raise ValueError('SO3/center/reference failure')
    unchanged = {}
    for name in ('dataset_train.txt', 'dataset_test.txt', 'sparse/0/cameras.bin', 'sparse/0/points3D.bin', 'sparse/0/points3D.ply'):
        if sha256(scene/name) != sha256(SOURCE/name):
            raise ValueError('source invariant changed: '+name)
        unchanged[str(scene/name)] = sha256(scene/name)
    for name in ('cameras.txt', 'points3D.txt', 'points3D.ply'):
        if sha256(prepared/'sparse/0'/name) != sha256(INPUT/'sparse/0'/name):
            raise ValueError('prepared invariant changed: '+name)
        unchanged[str(prepared/'sparse/0'/name)] = sha256(prepared/'sparse/0'/name)
    for key, row in new_prepared.items():
        old = old_prepared[key]
        for field in ('id', 'camera_id', 'name', 'xys', 'point3D_ids'):
            if not np.array_equal(getattr(row, field), getattr(old, field)):
                raise ValueError('prepared non-pose field changed')
    if (scene/'processed').resolve() != (SOURCE/'processed').resolve() or (prepared/'images').resolve() != (INPUT/'images').resolve():
        raise ValueError('RGB source changed')
    before = ColmapDataset(SOURCE, load_masks=False)
    dataset = ColmapDataset(scene, load_masks=False)
    for split in ('all', 'mapping', 'test'):
        if [c.image_name for c in before.split(split)] != [c.image_name for c in dataset.split(split)]:
            raise ValueError('split registry changed')
    manifest = json.loads((prepared/'input_manifest.json').read_text())
    entries = {r['source_name']:(r['prepared_name'],r['sha256']) for r in manifest['images']}
    audit = audit_prepared_camera_registry(new_prepared, entries,
        {c.image_name:c for c in dataset.split('mapping')},
        read_intrinsics_text(str(prepared/'sparse/0/cameras.txt')))
    loader_orth = max(float(np.max(np.abs(c.pose_w2c[:3,:3].T@c.pose_w2c[:3,:3]-np.eye(3)))) for c in dataset.cameras)
    if loader_orth > 5e-7:
        raise ValueError('actual dataset loader non-SO3')
    if torch_cuda_initialized():
        raise ValueError('CPU preflight initialized CUDA')
    report = {'schema':'stmary_pose_repair_cpu_v1','complete':True,'root':str(root),
        'camera_count':len(new),'mapping_count':len(dataset.split('mapping')),'test_count':len(dataset.split('test')),
        'raw_q_norm_min':float(min(np.linalg.norm(v['q']) for v in cameras.values())),
        'raw_q_norm_max':float(max(np.linalg.norm(v['q']) for v in cameras.values())),
        'max_SO3_orthogonality_error':max_orth,'max_determinant_error':max_det,
        'max_camera_center_roundtrip_error':max_center,'max_scipy_rotation_difference':max_reference,
        'max_t_difference_from_original':max_t_delta,'actual_float32_loader_max_orthogonality_error':loader_orth,
        'prepared_camera_audit':audit,'nonpose_sparse_split_and_point_invariants_sha256':unchanged,
        'RGB_policy':'same resolved source files, no rewrite or resampling',
        'calibration_policy':'original SIMPLE_RADIAL and prepared rectified PINHOLE unchanged bytewise',
        'geometry_domain':'same RGB/K/distortion/rectification contract and valid masks; only q/t corrected',
        'training_initial_points':193428,'cuda_initialized':False,
        'projection_and_original_point_review':'independent reviewer pending; not a training release',
        'source_nvm_sha256':sha256(SOURCE/'reconstruction.nvm'),
        'original_images_sha256':sha256(SOURCE/'sparse/0/images.bin'),
        'repaired_images_sha256':sha256(scene/'sparse/0/images.bin'),
        'prepared_images_sha256':sha256(prepared/'sparse/0/images.txt'),
        'repair_code_sha256':sha256(Path(__file__)),
        'training_job_sha256':sha256(root/'training_job.json')}
    out = root/'cpu_validation.json'
    if not out.exists(): json_new(out, report)
    else:
        previous = json.loads(out.read_text())
        if previous != report: raise ValueError('validation report changed')
    print(json.dumps(report, indent=2)); return report

def torch_cuda_initialized():
    import torch
    return torch.cuda.is_initialized()

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage',choices=['prepare','validate']); p.add_argument('--root',type=Path,default=ROOT)
    a = p.parse_args(); (prepare if a.stage=='prepare' else validate)(a.root)
