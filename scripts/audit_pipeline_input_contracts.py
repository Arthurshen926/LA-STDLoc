"""Read-only calibration and exported-prior parity audit; no pose tuning."""
import argparse
import json
from pathlib import Path
import numpy as np
from plyfile import PlyData
from common.hashing import sha256_file
from data.colmap import read_intrinsics_binary, read_intrinsics_text
from data.datasets import validate_centered_camera


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiments',type=Path,required=True)
    p.add_argument('--prior-manifests',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    datasets={}
    for path in sorted(args.experiments.rglob('anygsloc_scene_manifest.json')):
        root=Path(json.loads(path.read_text())['dataset'])
        if str(root) in datasets:continue
        camera_path=root/'sparse/0/cameras.bin'
        if camera_path.exists(): cameras=read_intrinsics_binary(str(camera_path))
        else:
            camera_path=root/'sparse/0/cameras.txt';cameras=read_intrinsics_text(str(camera_path))
        failures=[]
        for c in cameras.values():
            try:validate_centered_camera(c)
            except ValueError as e:failures.append({'id':int(c.id),'error':str(e)})
        datasets[str(root)]={'camera_count':len(cameras),'calibration_sha256':sha256_file(camera_path),'failures':failures}
    priors=[]
    for path in args.prior_manifests:
        manifest=json.loads(path.read_text());source_path=Path(manifest['source_ply']);export_path=Path(manifest['exported_ply'])
        source=PlyData.read(source_path)['vertex'].data;exported=PlyData.read(export_path)['vertex'].data
        names=manifest['retained_properties']
        mask=np.ones(len(source),dtype=bool)
        for name in names:
            if name not in {'nx','ny','nz'}:mask&=np.isfinite(source[name])
        errors=[]
        if len(exported)!=int(mask.sum()):errors.append('finite row count mismatch')
        elif any(not np.array_equal(np.asarray(source[name][mask],dtype=np.float32),exported[name]) for name in names):errors.append('retained source fields changed')
        rotation=np.stack([exported[f'rot_{i}'] for i in range(4)],axis=1)
        scale_names=[k for k in names if k.startswith('scale_')]
        logs=np.stack([exported[k] for k in scale_names],axis=1)
        priors.append({'manifest':str(path),'manifest_sha256':sha256_file(path),'source_sha256':sha256_file(source_path),
                       'export_sha_matches':sha256_file(export_path)==manifest['exported_ply_sha256'],
                       'source_count':len(source),'export_count':len(exported),'retained_values_bit_exact':not errors,'errors':errors,
                       'zero_quaternion_count':int((np.linalg.norm(rotation.astype(np.float64),axis=1)<=1e-12).sum()),
                       'scale_exp_overflow_count':int((logs>np.log(np.finfo(np.float32).max)).any(1).sum())})
        print(json.dumps(priors[-1]),flush=True)
    report={'schema':'anygsloc_pipeline_input_contract_audit_v1','audit_only':True,'uses_test_images':False,
            'uses_test_gt_for_decisions':False,'calibration_datasets':datasets,'prior_export_audits':priors}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
    print('Calibration datasets',len(datasets),'failures',sum(len(v['failures']) for v in datasets.values()),flush=True)


if __name__=='__main__':main()
