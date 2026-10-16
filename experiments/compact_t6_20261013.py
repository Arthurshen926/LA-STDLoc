"""Approved Compact 2x2: frozen adaptive selector; no reconstruction or tuning."""
from __future__ import annotations
import argparse, contextlib, copy, fcntl, gc, hashlib, io, json, os, pickle, resource
from pathlib import Path
import subprocess, sys, time
import numpy as np
import torch
from anygsloc.artifacts import SCHEMA, sha256, validate
from anygsloc.mapping.compact_v2 import select_compact_rows_v2
from experiments.cambridge_mainline_ablations import t6 as T, t7 as C

ROOT = Path('/mnt/pool/sqy/anygsloc_compact_t6_20261013_v1')
SOURCE = Path('/mnt/pool/sqy/anygsloc_mainline_ablations_20261011_v2')
MODES = ('filter_on_fused','filter_on_single','filter_off_fused','filter_off_single')
NEW = MODES[1:]
GPU_UUIDS = ['GPU-55a12bad-1a5e-1538-5097-3b422bdb7ef1','GPU-1f21bf0e-0ae0-aad5-c30e-653ee1e13775','GPU-6ddf6453-abc9-a5b8-0ef2-2d9da5c04a27']
DESIGN = dict(schema='compact_t6_current_mainline_20261013_v1', scenes=list(C.SCENES), modes=list(MODES),
 new_scene_queries=15, new_queries=5754, reused_scene_queries=5, seed=2026,
 input='frozen vanilla3D; repaired StMary v2; complete unmasked SP2048 mapping cache',
 filter='existing V2 admission before association; existing completion physical checks unchanged',
 fusion='same filter uses identical rows/XYZ; only existing Full exported descriptors differ',
 selector='unchanged select_compact_rows_v2; natural adaptive N; no target budget; mapping only',
 query='same frozen evaluator, explicit_valid_mask, rectified processed RGB, scene-calibrated PoseLib',
 metrics='all 1918 queries in all four cells; no failures/extreme estimates excluded; scene equal',
 execution='max three physical RTX3090 workers, one per GPU, logical cuda:0, four CPU threads; parity gate',
 maximum_job_seconds=3400, formal_timing=False, test_status='benchmark-informed development, not blind',
 result_commitment='Compact main and completed Full supplement regardless of direction; no policy retuning')


def read(p): return C.read(p)
def write(p,v): return C.write_new(p,v)
def ref(p): return C.evidence(p)
def check(r): return C.unchanged(r)
def load(p): return C.load(p)
def digest(x):
    a=np.ascontiguousarray(torch.as_tensor(x).cpu().numpy())
    return hashlib.sha256(a.tobytes()).hexdigest()
class CPUUnpickler(pickle.Unpickler):
    """Trusted frozen mask archive only; storage location changes, values do not."""
    def find_class(self,module,name):
        if (module,name)==('torch.storage','_load_from_bytes'):
            return lambda b:torch.load(io.BytesIO(b),map_location='cpu',weights_only=False)
        return super().find_class(module,name)
def mask_record(mask):
    if mask is None:raise ValueError('missing semantic query mask')
    return dict(shape=list(mask.shape),dtype=str(mask.dtype),sha256=digest(mask))
def code_hashes():
    p=list((C.REPO/'anygsloc').rglob('*.py'))+[Path(__file__).resolve(), C.REPO/'tests/test_compact_t6_20261013.py']
    p += [C.REPO/'experiments/cambridge_mainline_ablations'/n for n in ('t6.py','t7.py')]
    p += list((C.REPO/'experiments/cambridge_input_matrix').glob('*.py'))
    return {str(x.relative_to(C.REPO)):sha256(x) for x in sorted(set(p))}
def protocol(root):
    p=read(Path(root)/'protocol.json')
    if p['design'] != DESIGN or p['code_sha256'] != code_hashes():raise ValueError('frozen protocol/code changed')
    for r in p['parent_protocols'].values():check(r)
    return p

def cpu_only():
    if torch.cuda.is_initialized() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU stage must hide all CUDA devices')

def resources():
    mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemTotal:','MemAvailable:'))}
    free=os.statvfs('/mnt/pool/sqy');disk=free.f_bavail*free.f_frsize
    return dict(mem, free_disk_bytes=disk, cpu_affinity=len(os.sched_getaffinity(0)),load=list(os.getloadavg()))

@contextlib.contextmanager
def slot(root,kind,index,minimum_ram):
    root=Path(root);d=root/'locks';d.mkdir(exist_ok=True)
    with (d/f'{kind}_{index}.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        r=resources()
        if r['MemAvailable']<minimum_ram or r['free_disk_bytes']<8*1024**3:raise RuntimeError('resource headroom gate')
        yield r
        fcntl.flock(f,fcntl.LOCK_UN)

def require_review(root,stage):
    r=read(Path(root)/f'independent_{stage}_review.json')
    if r.get('verdict')!='PASS' or r.get('protocol_sha256')!=sha256(Path(root)/'protocol.json'):
        raise ValueError('independent review missing/stale: '+stage)
    return r


def prepare(root):
    cpu_only();root=Path(root)
    if root.exists():raise FileExistsError(root)
    a=T.protocol(SOURCE/'T6');b=C.protocol(SOURCE/'T7');sources={}
    for scene in C.SCENES:
        m=read(SOURCE/'T6/maps'/scene/'manifest.json')
        sources[scene]=dict(cache=a['sources'][scene]['cache'], state=a['sources'][scene]['state'],
            full=a['sources'][scene]['full'], construction=m['construction'], full_variants=m['maps'],
            native=b['reused'][scene]['native_compact'], trace=b['sources'][scene]['trace'],
            snapshot=ref(SOURCE/'T6/query_inputs'/(scene+'.json')))
    write(root/'protocol.json',dict(design=DESIGN,scenes=a['scenes'],sources=sources,
        code_sha256=code_hashes(),parent_protocols={x:ref(SOURCE/x/'protocol.json') for x in ('T6','T7')},
        resource_snapshot=resources(),gpu_uuids=GPU_UUIDS))
    print('PREPARED',root,sha256(root/'protocol.json'),flush=True)


def off_state(construction,names):
    if construction.get('v2_enabled') is not False or construction.get('mapping_only') is not True:raise ValueError('wrong off construction')
    parts=construction['parts'];off=[0];qs=[];ks=[];xyz=[];feat=[];quality=[];cov=[]
    bins=torch.as_tensor(parts[0]['query_bins']).long()
    for p in parts:
        if list(p['query_names'])!=names or p.get('uses_test_queries') is not False or p.get('uses_source_mapping_rgb') is not False:raise ValueError('part provenance')
        if not torch.equal(torch.as_tensor(p['query_bins']).long(),bins):raise ValueError('part bins')
        c=p['projective_anchor_observations'];o=torch.as_tensor(c['observation_offsets']).long()
        if o[0]!=0 or len(o)!=len(p['anchor_xyz'])+1 or int(o[-1])!=len(c['query_indices']):raise ValueError('part CSR')
        off.extend((o[1:]+off[-1]).tolist());qs.append(torch.as_tensor(c['query_indices']).long());ks.append(torch.as_tensor(c['keypoint_indices']).long())
        xyz.append(p['anchor_xyz']);feat.append(p['anchor_features']);cov.append(p['anchor_position_covariance'])
        quality.append((p['identity_reliability']*p['geometry_reliability']).clamp(0,1))
    state=dict(anchor_ids=torch.arange(sum(len(x) for x in xyz)),anchor_xyz=torch.cat(xyz),anchor_features=torch.cat(feat),
        anchor_matchability=torch.cat(quality),anchor_position_covariance=torch.cat(cov),
        v6_mapping_query_names=names,v6_mapping_query_bins=bins,
        projective_anchor_observations=dict(observation_offsets=torch.tensor(off),query_indices=torch.cat(qs),keypoint_indices=torch.cat(ks)),
        provenance=dict(uses_test_queries=False,uses_source_mapping_rgb=False,adapter='off_original_cache_rows_no_remap'))
    C.validate_csr(state)
    if not torch.isfinite(state['anchor_position_covariance']).all():raise ValueError('nonfinite covariance')
    return state


def verify_state(state,full,cache):
    C.validate_csr(state)
    for k in ('anchor_xyz','anchor_features'):
        if not torch.equal(state[k],full[k]):raise ValueError('adapter changed '+k)
    names=state['v6_mapping_query_names'];csr=state['projective_anchor_observations']
    q=torch.as_tensor(csr['query_indices']);k=torch.as_tensor(csr['keypoint_indices'])
    for i,n in enumerate(names):
        if bool((k[q==i]>=len(cache[n]['native_keypoints'])).any()):raise ValueError('source cache row out of range')
    return {k:digest(state[k]) for k in ('anchor_xyz','anchor_features','anchor_matchability','anchor_position_covariance')}


def preflight(root,scene):
    cpu_only();root=Path(root);p=protocol(root);require_review(root,'code');s=p['sources'][scene]
    with slot(root,'preflight',scene,32*1024**3) as budget:
        start=time.monotonic()
        for k in ('cache','state','full','construction','trace','snapshot'):check(s[k])
        for r in s['full_variants'].values():check(r)
        check(s['native']['map']);check(s['native']['map']['rows'])
        for r in s['native']['files'].values():check(r)
        names=p['scenes'][scene]['mapping_names'];data=T.validate_cache(load(s['cache']['path']),names)
        state=load(s['state']['path']);full=validate(load(s['full']['path']));native=validate(load(s['native']['map']['path']))
        onrows=np.load(s['native']['map']['rows']['path']);C.check_subset(full,native,onrows)
        trace=load(s['trace']['path'])
        if not np.array_equal(np.sort(np.asarray(trace['final_source_ids'])),np.sort(onrows)):raise ValueError('native trace rows differ')
        on=verify_state(state,full,data['queries'])
        on_single=validate(load(s['full_variants']['filter_on_single']['path']))
        if not torch.equal(on_single['anchor_xyz'],full['anchor_xyz']):raise ValueError('on single XYZ changed')
        del state,full,on_single,native,trace;gc.collect()
        state=off_state(load(s['construction']['path']),names)
        off_full=validate(load(s['full_variants']['filter_off_fused']['path']))
        off=verify_state(state,off_full,data['queries'])
        off_single=validate(load(s['full_variants']['filter_off_single']['path']))
        if not torch.equal(off_single['anchor_xyz'],off_full['anchor_xyz']):raise ValueError('off single XYZ changed')
        from anygsloc.config import FROZEN
        from anygsloc.data.datasets import ColmapDataset
        from anygsloc.data.calibration import resolve_scene_calibration
        dataset=ColmapDataset(p['scenes'][scene]['dataset'],load_masks=False)
        with Path(p['scenes'][scene]['mask_path']).open('rb') as handle:
            dataset._masks=CPUUnpickler(handle).load()
        masks={c.image_name:mask_record(dataset.valid_mask(c)) for c in dataset.split('test')}
        snap=C.snapshot(root,scene,p['scenes'][scene]);C.same_snapshot(read(root/'query_inputs'/(scene+'.json')),read(s['snapshot']['path']))
        manifest=read(Path(s['native']['evaluation_folder'])/'manifest.json')
        C.manifest_check(s['native']['evaluation_folder'],s['native']['map'],p['scenes'][scene])
        calibration=resolve_scene_calibration(dataset.split('mapping'))
        if manifest['calibration']!=calibration or manifest['config']!=FROZEN.resolved(keypoint_count=calibration['keypoint_count'],reprojection_error_px=calibration['reprojection_error_px'],mask_policy='explicit_valid_mask'):raise ValueError('baseline query config/calibration drift')
        metrics=C.independent_metrics(read(Path(s['native']['evaluation_folder'])/'results.json'),p['scenes'][scene]['test_names'])
        receipt=dict(complete=True,scene=scene,protocol_sha256=sha256(root/'protocol.json'),query_inputs=ref(root/'query_inputs'/(scene+'.json')),
          on=on,off=off,native_count=len(onrows),off_count=len(state['anchor_xyz']),reused_metrics=metrics,
          baseline_config=manifest['config'],baseline_calibration=calibration,semantic_masks=masks,resources=budget,elapsed_seconds=time.monotonic()-start,
          maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,cuda_initialized=torch.cuda.is_initialized())
        write(root/'preflight'/(scene+'.json'),receipt)
        print('PREFLIGHT_PASS',scene,json.dumps({'off_count':receipt['off_count'],'native_count':len(onrows),'seconds':receipt['elapsed_seconds']}),flush=True)


def saved_preflight(root,scene):
    root=Path(root);a=read(root/'preflight'/(scene+'.json'))
    if not a['complete'] or a['protocol_sha256']!=sha256(root/'protocol.json'):raise ValueError('preflight missing/stale')
    check(a['query_inputs']);return a


def save_subset(path,full,rows,prov):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    rows=torch.as_tensor(rows).long();state=dict(schema=SCHEMA,anchor_ids=torch.arange(len(rows)),
        anchor_xyz=full['anchor_xyz'][rows].clone(),anchor_features=full['anchor_features'][rows].clone(),provenance=prov)
    validate(state);C.check_subset(full,state,rows.numpy());torch.save(state,path)
    return dict(ref(path),count=len(rows))


def select(root,scene,slot_id):
    cpu_only();root=Path(root);p=protocol(root);require_review(root,'code');a=saved_preflight(root,scene);s=p['sources'][scene]
    if slot_id not in (0,1,2):raise ValueError('CPU slot outside admitted three')
    with slot(root,'cpu',slot_id,48*1024**3) as budget:
        start=time.monotonic()
        for k in ('cache','construction'):check(s[k])
        data=T.validate_cache(load(s['cache']['path']),p['scenes'][scene]['mapping_names'])
        state=off_state(load(s['construction']['path']),data['mapping_names'])
        for k,h in a['off'].items():
            if digest(state[k])!=h:raise ValueError('state changed since preflight')
        query_cache={n:data['queries'][n] for n in state['v6_mapping_query_names']}
        # Native selector never reads the exported descriptor. Forbid that read at runtime.
        class SelectorInputs(dict):
            def __getitem__(self,k):
                if k=='anchor_features':raise AssertionError('selector unexpectedly consumes exported descriptors')
                return super().__getitem__(k)
        print('SELECT_START',scene,len(state['anchor_xyz']),flush=True)
        rows,trace=select_compact_rows_v2(SelectorInputs({k:v for k,v in state.items() if k!='anchor_features'}),query_cache)
        d=root/'selection'/scene;d.mkdir(parents=True,exist_ok=False)
        torch.save(trace,d/'trace.pt');np.save(d/'source_rows.npy',rows.numpy())
        rowref=ref(d/'source_rows.npy');onrows=np.load(s['native']['map']['rows']['path'])
        del data,query_cache,state;gc.collect();maps={}
        for mode in NEW:
            src=s['full_variants'][mode];check(src);full=validate(load(src['path']))
            selected=onrows if mode=='filter_on_single' else rows
            maps[mode]=save_subset(root/'maps'/scene/mode/'deployment_map.pt',full,selected,
                dict(experiment=DESIGN['schema'],configuration=mode,uses_test_queries=False,uses_source_mapping_rgb=False,
                  protocol_sha256=sha256(root/'protocol.json'),source_full_sha256=src['sha256'],
                  selected_rows_sha256=(s['native']['map']['rows']['sha256'] if mode=='filter_on_single' else rowref['sha256'])))
            maps[mode]['rows']=s['native']['map']['rows'] if mode=='filter_on_single' else rowref
            del full;gc.collect()
        write(d/'summary.json',dict(complete=True,scene=scene,protocol_sha256=sha256(root/'protocol.json'),
            natural_on_N=len(onrows),natural_off_N=len(rows),trace=ref(d/'trace.pt'),rows=rowref,maps=maps,
            elapsed_seconds=time.monotonic()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            resource_start=budget,cuda_initialized=torch.cuda.is_initialized(),descriptor_access_forbidden=True))
        print('SELECT_COMPLETE',scene,len(rows),time.monotonic()-start,flush=True)


def seal(root):
    cpu_only();root=Path(root);p=protocol(root);refs={};maps={}
    for scene in C.SCENES:
        saved_preflight(root,scene);s=p['sources'][scene];a=read(root/'selection'/scene/'summary.json')
        if not a['complete'] or a['protocol_sha256']!=sha256(root/'protocol.json'):raise ValueError('selection incomplete')
        maps[scene]=dict(a['maps'],filter_on_fused=s['native']['map'])
        for f in ('filter_on','filter_off'):
            r0,r1=(maps[scene][f+'_'+fusion] for fusion in ('fused','single'))
            check(r0);check(r1);check(r0['rows']);check(r1['rows'])
            if r0['rows']['sha256']!=r1['rows']['sha256']:raise ValueError('fusion changed selected rows')
            x,y=validate(load(r0['path'])),validate(load(r1['path']))
            if not torch.equal(x['anchor_xyz'],y['anchor_xyz']):raise ValueError('fusion changed selected XYZ')
        refs[scene]=ref(root/'selection'/scene/'summary.json')
    # Pilot uses one existing all-on map for baseline parity and the largest new map.
    candidates=[(r['count'],scene,mode) for scene,mm in maps.items() for mode,r in mm.items() if mode in NEW]
    _,scene,mode=max(candidates);old_scene=max(C.SCENES,key=lambda s:maps[s]['filter_on_fused']['count'])
    cases=[]
    from anygsloc.data.datasets import ColmapDataset
    from PIL import Image
    for s,m in [(old_scene,'filter_on_fused'),(scene,mode)]:
        ds=ColmapDataset(p['scenes'][s]['dataset'],load_masks=False);cams=ds.split('test');names=[c.image_name for c in cams]
        sizes={c.image_name:Image.open(c.image_path).size for c in cams}
        largest=max(names,key=lambda n:(sizes[n][0]*sizes[n][1],-names.index(n)))
        selected=names[:7];selected=(selected+[largest]) if largest not in selected else names[:8]
        cases.append(dict(scene=s,mode=m,names=selected,map=maps[s][m]))
    write(root/'sealed.json',dict(complete=True,protocol_sha256=sha256(root/'protocol.json'),selection_refs=refs,maps=maps,
        pilot_cases=cases,pilot_predictions=3*sum(len(c['names']) for c in cases)))
    print('SEALED',json.dumps(cases),flush=True)


def sealed(root):
    root=Path(root);p=protocol(root);s=read(root/'sealed.json')
    if not s['complete'] or s['protocol_sha256']!=sha256(root/'protocol.json'):raise ValueError('unsealed')
    for r in s['selection_refs'].values():check(r)
    return p,s


def gpu_guard(gpu):
    if gpu not in (0,1,2):raise ValueError('GPU outside approved set')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise ValueError('physical mapping differs')
    fields=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid,name','--format=csv,noheader,nounits'],text=True).strip().split(', ')
    if fields!=[GPU_UUIDS[gpu],'NVIDIA GeForce RTX 3090']:raise ValueError('GPU identity changed')
    procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    if any(l.split(',')[0].strip()==GPU_UUIDS[gpu] and int(l.split(',')[1])!=os.getpid() for l in procs):raise RuntimeError('foreign compute on assigned GPU')
    return dict(index=gpu,uuid=GPU_UUIDS[gpu],name=fields[1],python=sys.version,torch=torch.__version__,cuda=torch.version.cuda)


def pilot(root,gpu):
    root=Path(root);p,s=sealed(root);require_review(root,'maps')
    from anygsloc.data.datasets import ColmapDataset
    from anygsloc.online.runtime import Localizer
    with slot(root,'gpu',gpu,24*1024**3) as budget:
        hardware=gpu_guard(gpu);outputs=[]
        for case in s['pilot_cases']:
            scene=case['scene'];a=saved_preflight(root,scene);check(case['map'])
            before=C.snapshot(root,scene,p['scenes'][scene]);ds=ColmapDataset(p['scenes'][scene]['dataset'],load_masks=True)
            cams={c.image_name:c for c in ds.split('test')};cal=a['baseline_calibration']
            engine=Localizer(case['map']['path'],keypoint_count=cal['keypoint_count'],reprojection_error_px=cal['reprojection_error_px'],mask_policy='explicit_valid_mask',device='cuda:0',seed=2026)
            if {k:v for k,v in engine.config.items() if k!='actual_cpu_threads'}!=a['baseline_config']:raise ValueError('pilot config differs')
            rows=[]
            for name in case['names']:
                c=cams[name];mask=ds.valid_mask(c)
                if mask_record(mask)!=a['semantic_masks'][name]:raise ValueError('CPU/native semantic mask mismatch')
                r=engine.localize(ds.load_image(c),fov_x=c.fov_x,fov_y=c.fov_y,valid_mask=mask)
                rows.append(dict(name=name,success=bool(r['success']),pose=r['pose'].pose_w2c.tolist(),inliers=r['pose'].inliers.tolist(),
                   matched_keypoints_sha256=digest(r['matches'].keypoint_indices),matched_anchors_sha256=digest(r['matches'].anchor_indices)))
            if before!=C.snapshot(root,scene,p['scenes'][scene]):raise ValueError('pilot query changed')
            if case['mode']=='filter_on_fused':
                old={r['image_name']:r for r in read(Path(p['sources'][scene]['native']['evaluation_folder'])/'results.json')}
                for r in rows:
                    o=old[r['name']]
                    if r['success']!=o['success'] or len(r['inliers'])!=o['inlier_count'] or not np.allclose(r['pose'],o['pose_w2c'],atol=1e-7,rtol=1e-6):raise ValueError('production baseline prediction parity failed')
            outputs.append(dict(scene=scene,mode=case['mode'],map_sha256=case['map']['sha256'],rows=rows))
            del engine;gc.collect();torch.cuda.empty_cache()
        write(root/'pilot'/f'gpu{gpu}.json',dict(complete=True,protocol_sha256=sha256(root/'protocol.json'),sealed_sha256=sha256(root/'sealed.json'),hardware=hardware,outputs=outputs,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(0),actual_cpu_threads=torch.get_num_threads(),no_ground_truth_metrics=True))
        print('PILOT_COMPLETE',gpu,flush=True)


def pilot_check(root):
    cpu_only();root=Path(root);p,s=sealed(root);records=[read(root/'pilot'/f'gpu{i}.json') for i in range(3)]
    for r in records:
        if not r['complete'] or r['protocol_sha256']!=sha256(root/'protocol.json') or r['sealed_sha256']!=sha256(root/'sealed.json') or r['actual_cpu_threads']!=4:raise ValueError('pilot identity')
        if r['peak_reserved_bytes'] >= 20*1024**3:raise ValueError('pilot exceeds per-GPU memory admission')
    for r in records[1:]:
        for a,b in zip(records[0]['outputs'],r['outputs']):
            if (a['scene'],a['mode'],a['map_sha256'])!=(b['scene'],b['mode'],b['map_sha256']) or len(a['rows'])!=len(b['rows']):raise ValueError('pilot case mismatch')
            for x,y in zip(a['rows'],b['rows']):
                for k in ('name','success','inliers','matched_keypoints_sha256','matched_anchors_sha256'):
                    if x[k]!=y[k]:raise ValueError('cross-GPU prediction mismatch '+k)
                if not np.allclose(x['pose'],y['pose'],atol=1e-7,rtol=1e-6):raise ValueError('cross-GPU pose mismatch')
    write(root/'pilot_pass.json',dict(complete=True,verdict='PASS',protocol_sha256=sha256(root/'protocol.json'),sealed_sha256=sha256(root/'sealed.json'),
       records=[ref(root/'pilot'/f'gpu{i}.json') for i in range(3)],maximum_concurrency=3,predictions_only=True,no_gt_selection=True))
    print('THREE_GPU_PARITY_PASS',flush=True)


def evaluate(root,scene,mode,gpu):
    root=Path(root);p,s=sealed(root);require_review(root,'maps');gate=read(root/'pilot_pass.json')
    if gate['verdict']!='PASS' or gate['protocol_sha256']!=sha256(root/'protocol.json') or gate['sealed_sha256']!=sha256(root/'sealed.json'):raise ValueError('parity gate')
    for r in gate['records']:check(r)
    if mode not in NEW or scene not in C.SCENES:raise ValueError('outside finite cells')
    from anygsloc.evaluation.runner import main as run
    with slot(root,'gpu',gpu,24*1024**3) as budget:
        hardware=gpu_guard(gpu);m=s['maps'][scene][mode];check(m);a=saved_preflight(root,scene)
        from anygsloc.data.datasets import ColmapDataset
        dataset=ColmapDataset(p['scenes'][scene]['dataset'],load_masks=True)
        for camera in dataset.split('test'):
            if mask_record(dataset.valid_mask(camera))!=a['semantic_masks'][camera.image_name]:raise ValueError('CPU/native semantic mask mismatch')
        del dataset;gc.collect();torch.cuda.empty_cache()
        before=C.snapshot(root,scene,p['scenes'][scene]);folder=root/'evaluations'/scene/mode
        started=time.monotonic()
        run(['--dataset',p['scenes'][scene]['dataset'],'--map',m['path'],'--output',str(folder),'--images','processed',
             '--image-domain','rectified','--mask-policy','explicit_valid_mask','--rectification-contract','opencv_same_k','--seed','2026','--device','cuda:0'])
        if C.snapshot(root,scene,p['scenes'][scene])!=before:raise ValueError('query input changed')
        reference=read(Path(p['sources'][scene]['native']['evaluation_folder'])/'manifest.json')
        C.manifest_check(folder,m,p['scenes'][scene],reference);check(m);protocol(root)
        rows=read(folder/'results.json');metrics=C.independent_metrics(rows,p['scenes'][scene]['test_names'])
        write(folder/'attestation.json',dict(complete=True,scene=scene,mode=mode,protocol_sha256=sha256(root/'protocol.json'),
           map=m,query_snapshot=ref(root/'query_inputs'/(scene+'.json')),hardware=hardware,resources=budget,
           elapsed_seconds=time.monotonic()-started,formal_timing=False,actual_cpu_threads=torch.get_num_threads(),
           files={n:ref(folder/n) for n in ('results.json','summary.json','manifest.json')},metrics=metrics))
        print('EVALUATION_COMPLETE',scene,mode,json.dumps(metrics),flush=True)


def report(root):
    cpu_only();root=Path(root);p,s=sealed(root);out={}
    for mode in MODES:
        per={}
        for scene in C.SCENES:
            m=s['maps'][scene][mode];check(m)
            if mode=='filter_on_fused':
                cell=p['sources'][scene]['native'];folder=Path(cell['evaluation_folder'])
                for r in cell['files'].values():check(r)
            else:
                folder=root/'evaluations'/scene/mode;a=read(folder/'attestation.json')
                if not a['complete'] or a['protocol_sha256']!=sha256(root/'protocol.json') or a['map']['sha256']!=m['sha256']:raise ValueError('evaluation receipt')
                for r in a['files'].values():check(r)
            C.manifest_check(folder,m,p['scenes'][scene]);rows=read(folder/'results.json')
            v=C.independent_metrics(rows,p['scenes'][scene]['test_names'])
            te=np.asarray([r['translation_error_cm'] if r['success'] else np.inf for r in rows]);v.update(P90_TE_cm=float(np.percentile(te,90)),catastrophic_100cm_count=int(np.sum(te>=100)),points=m['count'],bytes=m['bytes'])
            per[scene]=v
        keys=('median_TE_cm','median_RE_deg','mean_TE_cm','P90_TE_cm','P95_TE_cm','R5_pct','R15_pct')
        macro={k:float(np.mean([v[k] for v in per.values()])) for k in keys}
        macro.update(total_points=sum(v['points'] for v in per.values()),total_bytes=sum(v['bytes'] for v in per.values()),queries=sum(v['query_count'] for v in per.values()),failures=sum(v['failures'] for v in per.values()))
        out[mode]=dict(per_scene=per,macro=macro)
    write(root/'report.json',dict(complete=True,design=DESIGN,protocol_sha256=sha256(root/'protocol.json'),rows=out,
       full_supplement=ref(SOURCE/'T6/report.json'),interpretation='Report all outcomes, including contrary effects; no test-result policy selection'))
    print('COMPACT_T6_COMPLETE',json.dumps({k:v['macro'] for k,v in out.items()}),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=('prepare','preflight','select','seal','pilot','pilot-check','evaluate','report'))
    ap.add_argument('--root',type=Path,default=ROOT);ap.add_argument('--scene',choices=C.SCENES);ap.add_argument('--mode',choices=MODES);ap.add_argument('--gpu',type=int);ap.add_argument('--slot',type=int,default=0)
    a=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    if a.stage in ('prepare','seal','report'):globals()[a.stage](a.root)
    elif a.stage=='preflight':preflight(a.root,a.scene)
    elif a.stage=='select':select(a.root,a.scene,a.slot)
    elif a.stage=='pilot':pilot(a.root,a.gpu)
    elif a.stage=='pilot-check':pilot_check(a.root)
    elif a.stage=='evaluate':evaluate(a.root,a.scene,a.mode,a.gpu)
if __name__=='__main__':main()
