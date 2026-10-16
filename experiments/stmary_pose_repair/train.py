"""Runner-owned one-shot exact vanilla training; no detached execution or partial resume."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from anygsloc.artifacts import sha256
from anygsloc.data.colmap import read_extrinsics_text, read_intrinsics_text, qvec2rotmat
from experiments.cambridge_input_matrix.prior_provenance import namespace_values
from .prepare import ROOT, JOBS, json_new


def read(p): return json.loads(Path(p).read_text())

def bound(p):
    p=Path(p)
    return {'path':str(p.resolve()),'sha256':sha256(p),'bytes':p.stat().st_size}

def validate_release(root):
    root=Path(root)
    cpu=read(root/'cpu_validation.json'); review=read(root/'independent_review.json')
    if cpu.get('complete') is not True or review.get('complete') is not True or review.get('verdict')!='PASS':
        raise ValueError('CPU and independent review release required')
    for p,h in review['reviewed_paths_sha256'].items():
        if sha256(p)!=h:raise ValueError('reviewed input changed: '+p)
    job=read(root/'training_job.json')
    original=next(x for x in read(JOBS) if x['id']=='StMarysChurch_vanilla_3dgs_reconstruct')
    expected=copy.deepcopy(original['command'])
    expected[expected.index('--source_path')+1]=str(root/'vanilla_input/StMarysChurch')
    expected[expected.index('--model_path')+1]=str(root/'prior/vanilla_3dgs/native')
    if (job['command']!=expected or job['env']!=original['env'] or job['cwd']!=original['cwd'] or
        job['effective_training_seed']!=0 or sha256(JOBS)!=job['source_job_sha256']):
        raise ValueError('exact recipe changed')
    if '--start_checkpoint' in job['command']:raise ValueError('resume not permitted')
    if Path(expected[expected.index('--model_path')+1]).resolve().is_relative_to(root.resolve()) is False:
        raise ValueError('output resolves outside repair root')
    previous=read(Path(__file__).parents[1]/'cambridge_input_matrix/validation/prior_StMarysChurch.json')
    script=Path(job['command'][2])
    for p in (script,script.parent/'scene/dataset_readers.py'):
        if sha256(p)!=previous['evidence_sha256'][str(p)]:raise ValueError('frozen trainer implementation changed')
    general=script.parent/'utils/general_utils.py'
    seed_source=general.read_text()
    if not all(x in seed_source for x in ('random.seed(0)','np.random.seed(0)','torch.manual_seed(0)')):
        raise ValueError('seed0 initialization source changed')
    return job

def launch(root,gpu):
    root=Path(root); job=validate_release(root)
    native=root/'prior/vanilla_3dgs/native'
    if native.exists() or (root/'training_launch.json').exists():
        raise FileExistsError('existing training attempt; never silently restart')
    script=Path(job['command'][2]); env=os.environ.copy();env.update(job['env']);env['CUDA_VISIBLE_DEVICES']=str(gpu)
    record={'schema':'stmary_exact_training_v1','command':job['command'],'cwd':job['cwd'],
        'environment':{k:env[k] for k in job['env']},'seed':0,
        'training_job_sha256':sha256(root/'training_job.json'),
        'independent_review_sha256':sha256(root/'independent_review.json'),
        'started_unix':time.time(),'run_mode':'one Runner-owned invocation, no detach and no resume',
        'recipe_changes':['source_path','model_path','assigned physical GPU only'],
        'source_files':{str(p):sha256(p) for p in (script,script.parent/'scene/dataset_readers.py',script.parent/'utils/general_utils.py')}}
    json_new(root/'training_launch.json',record)
    print(json.dumps(record),flush=True)
    with (root/'training.log').open('x') as log:
        proc=subprocess.run(job['command'],cwd=job['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT)
    outcome={'returncode':proc.returncode,'wall_seconds':time.time()-record['started_unix'],'ended_unix':time.time()}
    json_new(root/'training_outcome.json',outcome)
    print(json.dumps(outcome),flush=True)
    if proc.returncode!=0:raise RuntimeError('training failed; inspect training.log, do not resume as equivalent')
    accept(root)

def accept(root):
    root=Path(root);job=validate_release(root);native=root/'prior/vanilla_3dgs/native'
    outcome=read(root/'training_outcome.json');launch=read(root/'training_launch.json')
    if outcome['returncode']!=0 or launch['command']!=job['command'] or launch['seed']!=0:
        raise ValueError('training completion/identity mismatch')
    for p,h in launch['source_files'].items():
        if sha256(p)!=h:raise ValueError('trainer source drifted while running')
    cfg=namespace_values(native/'cfg_args')
    if Path(cfg['source_path']).resolve()!=(root/'vanilla_input/StMarysChurch').resolve() or Path(cfg['model_path']).resolve()!=native.resolve():
        raise ValueError('training used different source/output')
    for k,v in {'images':'images','resolution':1,'data_device':'cpu','sh_degree':3,'white_background':False,'depths':''}.items():
        if cfg[k]!=v:raise ValueError('training cfg drift: '+k)
    trained=read(native/'cameras.json')
    prepared=read_extrinsics_text(str(root/'vanilla_input/StMarysChurch/sparse/0/images.txt'))
    intrinsics=read_intrinsics_text(str(root/'vanilla_input/StMarysChurch/sparse/0/cameras.txt'))
    by_name={r.name:r for r in prepared.values()}
    if len(trained)!=len(by_name) or {r['img_name'] for r in trained}!=set(by_name):
        raise ValueError('trained camera registry differs from mapping-only input')
    max_pose=max_K=0.
    for row in trained:
        expected=by_name[row['img_name']]; R=qvec2rotmat(expected.qvec)
        expected_pose=np.eye(4);expected_pose[:3,:3]=R;expected_pose[:3,3]=expected.tvec
        actual_c2w=np.eye(4);actual_c2w[:3,:3]=row['rotation'];actual_c2w[:3,3]=row['position']
        max_pose=max(max_pose,float(np.max(np.abs(np.linalg.inv(actual_c2w)-expected_pose))))
        camera=intrinsics[expected.camera_id];fx,fy=camera.params[:2]
        max_K=max(max_K,abs(row['fx']-fx),abs(row['fy']-fy))
        if (row['width'],row['height'])!=(camera.width,camera.height):raise ValueError('trained image shape differs')
    if max_pose>1e-10 or max_K>1e-8:raise ValueError('trained camera pose or K mismatch')
    ply=native/'point_cloud/iteration_30000/point_cloud.ply'
    from plyfile import PlyData
    vertices=PlyData.read(str(ply),mmap=True)['vertex']
    required=['x','y','z','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
    if len(vertices)==0 or not all(np.isfinite(vertices[k]).all() for k in required):
        raise ValueError('trained PLY empty/nonfinite')
    report={'schema':'stmary_training_acceptance_v1','complete':True,'native_path':str(native),'ply':bound(ply),
        'cfg_args_sha256':sha256(native/'cfg_args'),'cameras_json_sha256':sha256(native/'cameras.json'),
        'training_job_sha256':sha256(root/'training_job.json'),'training_launch_sha256':sha256(root/'training_launch.json'),
        'training_outcome_sha256':sha256(root/'training_outcome.json'),'seed':0,'iterations':30000,
        'recipe_matches_original_except_source_model_gpu':True,'training_camera_count':len(trained),
        'training_camera_max_pose_error':max_pose,'training_camera_max_focal_error':max_K,
        'gaussian_count':len(vertices),'source_hashes':launch['source_files'],
        'cpu_validation_sha256':sha256(root/'cpu_validation.json'),'independent_review_sha256':sha256(root/'independent_review.json')}
    json_new(root/'training_acceptance.json',report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['validate_release','launch','accept']);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--gpu',type=int)
    a=p.parse_args()
    if a.stage=='launch':
        if a.gpu not in (0,1,2):p.error('--gpu must identify one assigned card')
        launch(a.root,a.gpu)
    elif a.stage=='accept':accept(a.root)
    else:print(json.dumps(validate_release(a.root),indent=2))
