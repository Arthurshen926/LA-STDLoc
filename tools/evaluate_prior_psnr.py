"""Full-resolution PSNR of the frozen E4 Gaussian priors on common test views.

No training, pose estimation, masking, exposure fitting, or source mutation.
Usage: prepare; worker --worker N --workers 3; summarize.
"""
from pathlib import Path
import argparse, csv, gc, hashlib, io, json, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image

# Match the environment used by the archived figure renderers (Ninja / CUDA JIT).
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'artifacts/prior_psnr_20261008'
SOURCE=ROOT/'paper/experiments_cvpr_r5_20261003/E4_gaussian_prior.provenance.json'
SCENES=['StMarysChurch','OldHospital','GreatCourt','ShopFacade','KingsCollege']
KEYS=['matcha','vanilla_2dgs','vanilla_3dgs','i3dgs_raw','i3dgs_1k']
LABELS=['MAtCha','2DGS','3DGS','i3DGS raw','i3DGS post-optimized']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def write_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)

def load_json(path):return json.loads(Path(path).read_text())

def prepare():
    from anygsloc.data.datasets import ColmapDataset
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():
        print('Frozen manifest exists; use the existing protocol.',flush=True);return
    source=load_json(SOURCE)
    protocol={
        'evaluation_views':'Every registered image in the official dataset_test.txt split, sorted lexically; no localization-success or appearance filtering',
        'ground_truth':'Cambridge_stdloc/<scene>/processed RGB and the frozen ColmapDataset calibration / GT world-to-camera pose',
        'resolution':[1920,1080], 'resize_or_crop':False,
        'pixel_domain':'Decoded 8-bit RGB / 255.0 (stored display RGB); no linear-light conversion',
        'prediction':'Full-resolution float32 RGB render clamped to [0,1] before metric computation; no 8-bit quantization',
        'pixel_population':'All H x W x 3 RGB samples; no masks, border exclusion, alpha thresholds or method-dependent subsets',
        'mse':'Mean over all RGB samples; difference, square and reduction in float64',
        'PSNR_dB':'-10*log10(MSE), maximum signal value 1',
        'scene_mean':'Arithmetic mean of per-image PSNR values in dB',
        'five_scene_macro':'Equal-weight arithmetic mean of the five scene means',
        'pooled_mean':'Also report per-image mean over all 1,918 images; do not confuse with the scene-equal macro',
        'renderer':'gsplat 1.5.3, native 2DGS or antialiased 3DGS, original SH degree and all original primitives',
        'render_parameters':{'near_plane':.01,'far_plane':10000.,'eps2d':.3,'radius_clip':0.,'background':[0,0,0],'packed':False,'tile_size':16},
        'OOM_retry':'2DGS may retry tile_size=32 without changing primitives, cameras, resolution or pixels; each retry is recorded',
        'no_exposure_or_color_fitting':True,'no_pose_optimization':True,'no_prior_updates':True,
    }
    scenes={}
    for scene in SCENES:
        root=Path('/mnt/pool/sqy/Cambridge_stdloc')/scene
        ds=ColmapDataset(root,load_masks=False);cams=ds.split('test')
        assert not set(c.image_name for c in cams)&set(c.image_name for c in ds.split('mapping'))
        def record(c):
            data=c.image_path.read_bytes()
            with Image.open(io.BytesIO(data)) as image:
                assert image.size==(c.width,c.height)==(1920,1080)
                assert image.mode=='RGB',(scene,c.image_name,image.mode)
            K=[[c.width/(2*math.tan(c.fov_x/2)),0,c.width/2],[0,c.height/(2*math.tan(c.fov_y/2)),c.height/2],[0,0,1]]
            return {'name':c.image_name,'path':str(c.image_path),'sha256':hashlib.sha256(data).hexdigest(),
                    'width':c.width,'height':c.height,'K':K,'pose_w2c':c.pose_w2c.tolist()}
        with ThreadPoolExecutor(max_workers=8) as pool:views=list(pool.map(record,cams))
        priors={}
        for key in KEYS:
            v=source['scenes'][scene]['priors'][key]
            actual=sha(v['ply']);assert actual==v['ply_sha256'],(scene,key)
            priors[key]={k:v[k] for k in ['ply','ply_sha256','gaussian_type','sh_degree']}
        scenes[scene]={'views':views,'priors':priors,'n_views':len(views),
            'calibration_sources':{str(root/'sparse/0'/name):sha(root/'sparse/0'/name) for name in ['cameras.bin','images.bin']},
            'test_split_source':str(root/'dataset_test.txt'),'test_split_sha256':sha(root/'dataset_test.txt')}
        print('Prepared',scene,len(views),'test views',flush=True)
    assert sum(v['n_views'] for v in scenes.values())==1918
    write_json(OUT/'manifest.json',{'protocol':protocol,'source':str(SOURCE),'source_sha256':sha(SOURCE),'scenes':scenes,
                                  'scene_order':SCENES,'prior_order':KEYS,'prior_labels':dict(zip(KEYS,LABELS))})

class Renderer:
    def __init__(self,source):
        import torch
        from anygsloc.priors.models import FrozenGaussianModel
        assert sha(source['ply'])==source['ply_sha256']
        self.model=FrozenGaussianModel(source['sh_degree'],device='cuda')
        self.model.load_ply(source['ply'],loc_feature_dim=0)
        m=self.model;self.two=m.get_scaling.shape[1]==2
        scales=m.get_scaling
        if self.two:scales=torch.cat([scales,torch.ones_like(scales[:,:1])],1)
        self.params=dict(means=m.get_xyz,quats=m.get_rotation,scales=scales,opacities=m.get_opacity[:,0],
                         colors=m.get_features,sh_degree=m.active_sh_degree,packed=False,
                         near_plane=.01,far_plane=10000.,eps2d=.3,radius_clip=0.,
                         backgrounds=torch.zeros((1,3),device='cuda'),render_mode='RGB')
        for k in ['means','quats','scales','opacities','colors']:
            assert torch.isfinite(self.params[k]).all().item(),('nonfinite prior',k,source['ply'])
        assert self.two==(source['gaussian_type']=='2dgs')
        self.n_primitives=len(m.get_xyz)

    def render(self,view):
        import torch
        from gsplat import rasterization,rasterization_2dgs
        args=dict(self.params,viewmats=torch.tensor(view['pose_w2c'],dtype=torch.float32,device='cuda')[None],
                  Ks=torch.tensor(view['K'],dtype=torch.float32,device='cuda')[None],width=view['width'],height=view['height'])
        tile=16
        if self.two:
            try:result=rasterization_2dgs(**args,tile_size=16)
            except torch.cuda.OutOfMemoryError:
                # Leave the failed-call traceback before retrying.
                result=None
            if result is None:
                torch.cuda.empty_cache();tile=32
                result=rasterization_2dgs(**args,tile_size=32)
        else:result=rasterization(**args,rasterize_mode='antialiased',tile_size=16)
        pred=result[0][0].clamp(0,1)
        assert tuple(pred.shape)==(view['height'],view['width'],3)
        assert torch.isfinite(pred).all().item()
        del result,args
        return pred,tile

def read_rgb(view):
    data=Path(view['path']).read_bytes()
    assert hashlib.sha256(data).hexdigest()==view['sha256']
    return np.asarray(Image.open(io.BytesIO(data)).convert('RGB')).copy()

def evaluate_job(scene,key,manifest,limit=0):
    import torch,gsplat
    job=OUT/'jobs'/f'{scene}__{key}';job.mkdir(parents=True,exist_ok=True)
    if limit:job=OUT/'smoke'/f'{scene}__{key}';job.mkdir(parents=True,exist_ok=True)
    binding=sha(OUT/'manifest.json');source=manifest['scenes'][scene]['priors'][key]
    views=manifest['scenes'][scene]['views'][:limit or None]
    done={}
    if (job/'records.jsonl').exists():
        for ln in (job/'records.jsonl').read_text().splitlines():
            r=json.loads(ln);assert r['manifest_sha256']==binding;assert r['name'] not in done;done[r['name']]=r
    if len(done)==len(views) and (job/'summary.json').exists():
        print('Already complete',scene,key,len(views),flush=True);return
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    renderer=Renderer(source)
    print('Loaded',scene,key,renderer.n_primitives,'primitives',flush=True)
    pending=[(i,v) for i,v in enumerate(views) if v['name'] not in done]
    with torch.inference_mode(),ThreadPoolExecutor(max_workers=2) as pool,(job/'records.jsonl').open('a',buffering=1) as stream:
        next_rgb=pool.submit(read_rgb,pending[0][1]) if pending else None
        for position,(index,view) in enumerate(pending):
            gt_np=next_rgb.result()
            if position+1<len(pending):next_rgb=pool.submit(read_rgb,pending[position+1][1])
            t=time.monotonic();pred,tile=renderer.render(view)
            gt=torch.from_numpy(gt_np).to(device='cuda',dtype=torch.float64)/255.
            error=(pred.double()-gt).square();mse=error.mean().item()
            assert mse>0 and math.isfinite(mse)
            psnr=-10*math.log10(mse)
            row={'index':index,'name':view['name'],'mse':mse,'psnr_db':psnr,'pixels':view['width']*view['height'],
                 'tile_size':tile,'seconds':time.monotonic()-t,'GT_sha256':view['sha256'],'manifest_sha256':binding}
            if index==0:
                native=pred.cpu().numpy();cpu_mse=float(np.mean((native.astype(np.float64)-gt_np.astype(np.float64)/255.)**2,dtype=np.float64))
                assert abs(cpu_mse-mse)<1e-12
                row['independent_CPU_mse']=cpu_mse
                Image.fromarray(np.uint8(np.round(native*255))).save(job/'first_view_render.png')
                Image.fromarray(gt_np).save(job/'first_view_GT.png')
            stream.write(json.dumps(row,allow_nan=False)+'\n');done[view['name']]=row
            del pred,gt,error
            if position%25==0 or position+1==len(pending):
                state={'scene':scene,'prior':key,'complete':len(done),'total':len(views),'elapsed_seconds':time.monotonic()-start,'running_mean_PSNR_dB':float(np.mean([r['psnr_db'] for r in done.values()]))}
                write_json(job/'progress.json',state);print(json.dumps(state),flush=True)
    rows=[done[v['name']] for v in views]
    summary={'scene':scene,'prior':key,'n_views':len(rows),'mean_psnr_db':float(np.mean([r['psnr_db'] for r in rows])),
             'std_psnr_db':float(np.std([r['psnr_db'] for r in rows])),
             'median_psnr_db':float(np.median([r['psnr_db'] for r in rows])),
             'mean_mse':float(np.mean([r['mse'] for r in rows])),
             'tile32_views':[r['name'] for r in rows if r['tile_size']==32],
             'n_primitives':renderer.n_primitives,'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(),
             'runtime_seconds_this_invocation':time.monotonic()-start,
             'manifest_sha256':binding,'records_sha256':sha(job/'records.jsonl'),
             'script_sha256':sha(__file__),'torch_version':torch.__version__,'gsplat_version':gsplat.__version__,
             'GPU':torch.cuda.get_device_name(0),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')}
    write_json(job/'summary.json',summary)
    del renderer;gc.collect();torch.cuda.empty_cache()
    print('COMPLETE',scene,key,summary['mean_psnr_db'],flush=True)

def worker(index,count,scene=None,key=None,limit=0):
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(2)
    manifest=load_json(OUT/'manifest.json')
    if scene and key:evaluate_job(scene,key,manifest,limit);return
    jobs=[(s,k) for s in SCENES for k in KEYS]
    for i,(s,k) in enumerate(jobs):
        if i%count==index:evaluate_job(s,k,manifest)

def summarize():
    manifest=load_json(OUT/'manifest.json');binding=sha(OUT/'manifest.json')
    results={};all_rows=[];checks=[]
    for scene in SCENES:
        views=manifest['scenes'][scene]['views'];results[scene]={}
        for key in KEYS:
            job=OUT/'jobs'/f'{scene}__{key}';summary=load_json(job/'summary.json')
            rows=[json.loads(x) for x in (job/'records.jsonl').read_text().splitlines()]
            assert summary['manifest_sha256']==binding and summary['records_sha256']==sha(job/'records.jsonl')
            assert [r['name'] for r in sorted(rows,key=lambda r:r['index'])]==[v['name'] for v in views]
            assert len(rows)==summary['n_views']==len(views)
            for r,v in zip(sorted(rows,key=lambda r:r['index']),views):
                assert r['GT_sha256']==v['sha256'] and r['manifest_sha256']==binding
                assert abs(r['psnr_db']+10*math.log10(r['mse']))<1e-10
                all_rows.append({'scene':scene,'prior':key,**r})
            assert abs(np.mean([r['psnr_db'] for r in rows])-summary['mean_psnr_db'])<1e-10
            assert sha(manifest['scenes'][scene]['priors'][key]['ply'])==manifest['scenes'][scene]['priors'][key]['ply_sha256']
            results[scene][key]=summary
            checks.append({'scene':scene,'prior':key,'expected_views_complete':True,'same_GT_and_cameras':True,'metric_recomputed':True,'prior_hash_unchanged':True})
    macro={key:float(np.mean([results[s][key]['mean_psnr_db'] for s in SCENES])) for key in KEYS}
    pooled={key:float(np.mean([r['psnr_db'] for r in all_rows if r['prior']==key])) for key in KEYS}
    result={'protocol':manifest['protocol'],'scene_order':SCENES,'prior_order':KEYS,'prior_labels':dict(zip(KEYS,LABELS)),
            'scene_means':{s:{k:results[s][k]['mean_psnr_db'] for k in KEYS} for s in SCENES},
            'scene_view_counts':{s:len(manifest['scenes'][s]['views']) for s in SCENES},
            'five_scene_macro_psnr_db':macro,'pooled_image_mean_psnr_db':pooled,
            'n_test_images':1918,'n_rendered_comparisons':len(all_rows),'manifest_sha256':binding,'job_summaries':results}
    write_json(OUT/'summary.json',result)
    with (OUT/'scene_psnr.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['scene','n_test_views']+LABELS)
        for s in SCENES:w.writerow([s,len(manifest['scenes'][s]['views'])]+[results[s][k]['mean_psnr_db'] for k in KEYS])
        w.writerow(['Five-scene equal-weight mean',1918]+[macro[k] for k in KEYS])
        w.writerow(['All-test-image mean',1918]+[pooled[k] for k in KEYS])
    columns=['scene','prior','index','name','mse','psnr_db','pixels','tile_size','seconds','GT_sha256','manifest_sha256']
    with (OUT/'per_image_psnr.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(all_rows)
    write_json(OUT/'verification.json',{'status':'passed','rendered_comparisons':len(all_rows),'checks':checks,'all_25_first_view_CPU_GPU_MSE_checks_passed':True})
    print(json.dumps(result['scene_means'],indent=2));print('Five-scene macro',macro)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','worker','summarize'])
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=3)
    p.add_argument('--scene',choices=SCENES);p.add_argument('--prior',choices=KEYS);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    if a.phase=='prepare':prepare()
    elif a.phase=='worker':worker(a.worker,a.workers,a.scene,a.prior,a.limit)
    else:summarize()
