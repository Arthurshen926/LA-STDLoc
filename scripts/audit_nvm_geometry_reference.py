"""Independent real-track corroboration diagnostic, never a deployable map.

NVM identities are associated through multiple observed mapping keypoints,
not by nearest 3D point or query GT projection. These automatic associations
remain a selected proxy, not manually verified physical identity or truth.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
import torch
from common.hashing import sha256_file
from data.colmap import qvec2rotmat, read_intrinsics_binary, read_extrinsics_binary
from evidence.geometry_crossfit import crossfit_geometry


def read_reference(path, names, K, T, dataset):
    registry = {n: i for i, n in enumerate(names)}
    intr = read_intrinsics_binary(dataset/'sparse/0/cameras.bin')
    extr = {v.name: v for v in read_extrinsics_binary(dataset/'sparse/0/images.bin').values()}
    per_view = [[] for _ in names]
    with path.open() as f:
        lines = (x.strip() for x in f if x.strip())
        if next(lines) != 'NVM_V3': raise ValueError('unsupported NVM header')
        cameras = [next(lines).split() for _ in range(int(next(lines)))]
        indices = []
        for c in cameras:
            name = c[0]
            # Cambridge conversion keeps the stem but changes JPEG to PNG.
            # The alias is accepted only with matching pose and calibration.
            if name not in registry and Path(name).suffix.lower() in ['.jpg','.jpeg']:
                name=str(Path(name).with_suffix('.png'))
            if name not in registry:
                indices.append(-1)
                continue
            i = registry[name]
            R, center = qvec2rotmat(np.array(c[2:6], float)), np.array(c[6:9], float)
            if not np.allclose(R, T[i,:3,:3], atol=2e-6) or not np.allclose(-R@center, T[i,:3,3], atol=2e-5):
                raise ValueError('NVM/cache camera pose mismatch')
            camera = intr[extr[name].camera_id]
            if camera.model != 'SIMPLE_RADIAL': raise ValueError('unsupported processed camera')
            if not np.allclose(camera.params, [float(c[1]), K[i,0,2], K[i,1,2], -float(c[9])], atol=2e-4):
                raise ValueError('NVM/COLMAP intrinsics mismatch')
            if not np.allclose(K[i,0,0],float(c[1]),atol=2e-4): raise ValueError('resolution mismatch')
            indices.append(i)
        n = int(next(lines))
        xyz = np.empty((n,3))
        for p in range(n):
            a = next(lines).split()
            xyz[p] = np.array(a[:3],float)
            count = int(a[6])
            if len(a) != 7+count*4: raise ValueError('invalid NVM track')
            for j in range(count):
                row = a[7+4*j:11+4*j]
                c = int(row[0]); i = indices[c]
                if i >= 0: per_view[i].append((p, float(row[2]), float(row[3])))
    converted, residuals = [], []
    for i, rows in enumerate(per_view):
        a = np.asarray(rows, float).reshape(-1,3)
        ids, uv = a[:,0].astype(int), a[:,1:]+K[i,:2,2]
        if not len(ids):
            converted.append((ids,uv))
            continue
        distortion = intr[extr[names[i]].camera_id].params[3]
        uv = cv2.undistortPoints(uv.reshape(-1,1,2), K[i], np.array([distortion,0.,0.,0.]), P=K[i]).reshape(-1,2)
        cam = xyz[ids]@T[i,:3,:3].T + T[i,:3,3]
        h = cam@K[i].T
        residuals.extend(np.linalg.norm(h[:,:2]/h[:,2:]-uv,axis=1).tolist())
        converted.append((ids,uv))
    if not residuals: raise ValueError('no NVM mapping camera observations resolved')
    return xyz, converted, np.percentile(residuals,[50,90,99]).tolist()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['map','cache','dataset','output']: p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    torch.set_num_threads(4)
    state=torch.load(args.map,map_location='cpu',weights_only=False)
    cache=torch.load(args.cache,map_location='cpu',weights_only=False)
    cache_sha=sha256_file(args.cache)
    if state['provenance']['v8_observation_cache_sha256']!=cache_sha: raise ValueError('map/cache mismatch')
    if state['provenance'].get('uses_source_mapping_rgb') is not False: raise ValueError('render map required')
    names=state['v6_mapping_query_names']
    if names!=list(cache['queries']): raise ValueError('registry mismatch')
    records=list(cache['queries'].values())
    K=np.stack([np.asarray(r['native_K'],float) for r in records])
    T=np.stack([np.asarray(r['pose_w2c'],float) for r in records])
    path=args.dataset/'reconstruction.nvm'
    original, views, calibration=read_reference(path,names,K,T,args.dataset)
    print('Reference mapping-observation residual P50/P90/P99:',calibration,flush=True)
    if calibration[0]>2. or calibration[1]>5.: raise ValueError('reference calibration not adequate')
    candidates=[]; point_tracks={}
    for i,(ids,uv) in enumerate(views):
        if len(ids)<2:
            count=len(records[i]['native_keypoints'])
            candidates.append((np.full(count,-1),np.full(count,np.inf),np.full(count,np.inf)))
            continue
        distance,near=cKDTree(uv).query(np.asarray(records[i]['native_keypoints'])+.5,k=2)
        candidates.append((ids[near[:,0]],distance[:,0],distance[:,1]))
        for pid,pixel in zip(ids,uv): point_tracks.setdefault(int(pid),[]).append((i,pixel))
    csr=state['projective_anchor_observations']
    if csr.get('keypoint_index_semantics')!='original_unfiltered_observation_cache_row': raise ValueError('row semantics')
    offsets,q,k=[np.asarray(csr[x]) for x in ['observation_offsets','query_indices','keypoint_indices']]
    reference_fit={}; panels={}
    for radius in [2.,4.]:
        rows=[]; rejected=Counter()
        for a in range(len(state['anchor_ids'])):
            qs,ks=q[offsets[a]:offsets[a+1]],k[offsets[a]:offsets[a+1]]
            hits=[int(candidates[i][0][j]) for i,j in zip(qs,ks)
                  if candidates[i][1][j]<=radius and candidates[i][2][j]-candidates[i][1][j]>=.5]
            counts=Counter(hits).most_common(2)
            if not counts or counts[0][1]<3 or counts[0][1]<.6*len(qs) or (len(counts)>1 and counts[0][1]-counts[1][1]<2):
                rejected['not_multiview_corroborated']+=1; continue
            pid,votes=counts[0]
            if pid not in reference_fit:
                track=point_tracks[pid]
                qi=np.array([v[0] for v in track]); uv=np.array([v[1] for v in track])
                if len(np.unique(qi))!=len(qi):
                    reference_fit[pid]=None; continue
                audit=crossfit_geometry(uv,K[qi],T[qi],original[pid])
                if audit['status']!='internally_consistent': reference_fit[pid]=None
                else:
                    def residual(x):
                        cam=np.einsum('nij,j->ni',T[qi,:3,:3],x)+T[qi,:3,3]
                        h=np.einsum('nij,nj->ni',K[qi],cam)
                        return (h[:,:2]/np.maximum(h[:,2:],1e-8)-uv).ravel()
                    fit=least_squares(residual,original[pid],loss='huber',f_scale=2.,max_nfev=40)
                    reference_fit[pid]=(fit.x,audit) if fit.success and np.isfinite(fit.x).all() else None
            ref=reference_fit[pid]
            if ref is None: rejected['reference_not_independently_consistent']+=1; continue
            delta=np.asarray(state['anchor_xyz'][a])-ref[0]
            rows.append({'anchor_row':a,'reference_point_id':pid,'support_votes':votes,
                         'track_length':len(qs),'reference_xyz_mapping_refit':ref[0].tolist(),
                         'xyz_delta_m':delta.tolist(),'xyz_error_cm':float(np.linalg.norm(delta)*100)})
        errors=[r['xyz_error_cm'] for r in rows]
        panels[str(radius)]={'count':len(rows),'coverage_percent':100*len(rows)/len(state['anchor_ids']),
                           'xyz_error_cm_p50_p90':np.percentile(errors,[50,90]).tolist() if errors else None,
                           'rejected':dict(rejected),'rows':rows}
        print('Panel',radius,{k:v for k,v in panels[str(radius)].items() if k!='rows'},flush=True)
    report={'schema':'anygsloc_nvm_real_track_geometry_diagnostic','deployment_eligible':False,
            'uses_external_real_track_reference':True,'uses_test_observations':False,
            'reference_initialized_from_all_view_sfm':True,'physical_identity_manually_verified':False,
            'selection_bias':'Only spatially agreeing multiview features; not representative of all Anchors',
            'reference_role':'mapping-only refit of independent real-image tracks, not absolute ground truth',
            'map_sha256':sha256_file(args.map),'cache_sha256':cache_sha,'nvm_sha256':sha256_file(path),
            'implementation_sha256':sha256_file(Path(__file__)),
            'reference_calibration_reprojection_px_p50_p90_p99':calibration,'panels':panels}
    with args.output.open('x') as f: json.dump(report,f,indent=2,allow_nan=False)


if __name__=='__main__':main()
