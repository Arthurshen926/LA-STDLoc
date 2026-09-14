"""Re-render missing mapping caches from existing RGB priors, never reconstruct priors."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=False)
    src=Path(__file__).resolve().parents[1];code=a.root/'code'
    for d in ['common','data','evidence','features','localization','evaluation','priors','map_learning','topology','scripts','configs']:
        shutil.copytree(src/d,code/d,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    hashes={str(p.relative_to(code)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code.rglob('*') if p.is_file()}
    for gpu,scene in enumerate(['GreatCourt','StMarysChurch']):
        lane=a.root/f'gpu{gpu}';lane.mkdir()
        ply=Path('/mnt/pool/sqy/stdloc_lafgs_v1_frozen_multiscene_20260731')/scene/'prior/rgb_matcha_2dgs/point_cloud/iteration_30000/point_cloud.ply'
        assert ply.is_file()
        out=a.root/scene/'observations'
        cmd=['/root/miniconda3/envs/g4splat/bin/python','-u','-m','scripts.probe_rendered_rgb_track_map',
             '--dataset',f'/mnt/pool/sqy/Cambridge_stdloc/{scene}','--images','processed',
             '--gaussian-ply',str(ply),'--gaussian-type','2dgs','--sh-degree','3',
             '--output-dir',str(out),'--render-only','--deduplicate-camera-geometry',
             '--keypoints','2048','--nms-radius','4','--detection-threshold','0.0',
             '--render-valid-alpha-minimum','0.05','--render-valid-neighborhood-radius','4']
        jobs=[dict(id=scene+'_recover_observations',command=cmd,cwd=str(code),deps=[],
                   outputs=[str(out/'rendered_rgb_feature_cache.pt')],
                   env=dict(PYTHONPATH=str(code),CUDA_VISIBLE_DEVICES=str(gpu),
                            LD_LIBRARY_PATH='/root/miniconda3/envs/g4splat/lib',
                            PATH='/root/miniconda3/envs/g4splat/bin:/usr/local/cuda/bin:/usr/bin:/bin'))]
        (lane/'jobs.json').write_text(json.dumps(jobs,indent=2))
    (a.root/'protocol.json').write_text(json.dumps(dict(code_sha256=hashes,
        scope='Re-extract mapping observations only; existing MAtCHA priors remain frozen',
        compatibility='New cache is NOT assumed row-compatible with archived tracks. Audit mapping names, CSR bounds, descriptor reconstruction and reprojection before reuse; otherwise rebuild downstream reference and variants.',
        overwrite_original=False),indent=2))
    for p in code.rglob('*'):
        if p.is_file():p.chmod(0o444)
    print('two recovery lanes ready')


if __name__=='__main__':main()
